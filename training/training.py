"""Module pour la logique d'entraînement."""

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Tuple, Optional, Any
from dataclasses import dataclass
import time
import pandas as pd
from accelerate import Accelerator

from sklearn.inspection import permutation_importance
from .metrique import MetricsCalculator, MetricsAccumulator
from .Loss import BCE
from .checkpoint import CheckpointManager
from .config import TrainingConfig, Config

@dataclass
class TrainingState:
    """État de l'entraînement."""
    epoch: int = 0
    global_step: int = 0
    best_val_mcc: float = -1.0
    best_val_f1: float = 0.0
    best_val_loss: float = float('inf')
    patience_counter: int = 0
    history: Dict[str, Any] = None

    def __post_init__(self):
        if self.history is None:
            self.history = {
                # Losses
                'train_loss': [],
                'val_loss': [],
                'iteration_losses': [],
                
                # Métriques de base
                'train_accuracy': [],
                'val_accuracy': [],
                'train_balanced_accuracy': [],
                'val_balanced_accuracy': [],
                'train_precision': [],
                'val_precision': [],
                'train_recall': [],
                'val_recall': [],
                'train_specificity': [],
                'val_specificity': [],
                
                # F-scores
                'train_f1': [],
                'val_f1': [],
                'train_f2': [],
                'val_f2': [],
                
                # IoU
                'train_iou': [],
                'val_iou': [],
                
                # Métriques pour déséquilibre (CRUCIALES!)
                'train_mcc': [],
                'val_mcc': [],
                'train_cohen_kappa': [],
                'val_cohen_kappa': [],
                
                # Métriques par classe
                'train_class_0_precision': [],
                'val_class_0_precision': [],
                'train_class_0_recall': [],
                'val_class_0_recall': [],
                'train_class_1_precision': [],
                'val_class_1_precision': [],
                'train_class_1_recall': [],
                'val_class_1_recall': [],
                
                # Support (utile pour analyser le déséquilibre)
                'train_support_class_0': [],
                'val_support_class_0': [],
                'train_support_class_1': [],
                'val_support_class_1': [],
                
                # Métriques probabilistes
                'train_auroc': [],
                'val_auroc': [],
                'train_auprc': [],
                'val_auprc': [],
                'train_brier_score': [],
                'val_brier_score': [],
                
                # Matrice de confusion (pour debug/analyse)
                'train_tp': [],
                'val_tp': [],
                'train_fp': [],
                'val_fp': [],
                'train_tn': [],
                'val_tn': [],
                'train_fn': [],
                'val_fn': [],
            }

        # Exposer des attributs pratiques (références aux listes de l'historique)
        # Losses
        self.train_losses = self.history['train_loss']
        self.val_losses = self.history['val_loss']
        self.iteration_losses = self.history['iteration_losses']
        
        # Métriques de base
        self.train_accuracy = self.history['train_accuracy']
        self.val_accuracy = self.history['val_accuracy']
        self.train_balanced_accuracy = self.history['train_balanced_accuracy']
        self.val_balanced_accuracy = self.history['val_balanced_accuracy']
        self.train_precision = self.history['train_precision']
        self.val_precision = self.history['val_precision']
        self.train_recall = self.history['train_recall']
        self.val_recall = self.history['val_recall']
        self.train_specificity = self.history['train_specificity']
        self.val_specificity = self.history['val_specificity']
        
        # F-scores
        self.train_f1 = self.history['train_f1']
        self.val_f1 = self.history['val_f1']
        self.train_f2 = self.history['train_f2']
        self.val_f2 = self.history['val_f2']
        
        # IoU
        self.train_iou = self.history['train_iou']
        self.val_iou = self.history['val_iou']
        
        # Métriques pour déséquilibre
        self.train_mcc = self.history['train_mcc']
        self.val_mcc = self.history['val_mcc']
        self.train_cohen_kappa = self.history['train_cohen_kappa']
        self.val_cohen_kappa = self.history['val_cohen_kappa']
        
        # Métriques par classe
        self.train_class_0_precision = self.history['train_class_0_precision']
        self.val_class_0_precision = self.history['val_class_0_precision']
        self.train_class_0_recall = self.history['train_class_0_recall']
        self.val_class_0_recall = self.history['val_class_0_recall']
        self.train_class_1_precision = self.history['train_class_1_precision']
        self.val_class_1_precision = self.history['val_class_1_precision']
        self.train_class_1_recall = self.history['train_class_1_recall']
        self.val_class_1_recall = self.history['val_class_1_recall']
        
        # Support
        self.train_support_class_0 = self.history['train_support_class_0']
        self.val_support_class_0 = self.history['val_support_class_0']
        self.train_support_class_1 = self.history['train_support_class_1']
        self.val_support_class_1 = self.history['val_support_class_1']
        
        # Métriques probabilistes
        self.train_auroc = self.history['train_auroc']
        self.val_auroc = self.history['val_auroc']
        self.train_auprc = self.history['train_auprc']
        self.val_auprc = self.history['val_auprc']
        self.train_brier_score = self.history['train_brier_score']
        self.val_brier_score = self.history['val_brier_score']
        
        # Matrice de confusion
        self.train_tp = self.history['train_tp']
        self.val_tp = self.history['val_tp']
        self.train_fp = self.history['train_fp']
        self.val_fp = self.history['val_fp']
        self.train_tn = self.history['train_tn']
        self.val_tn = self.history['val_tn']
        self.train_fn = self.history['train_fn']
        self.val_fn = self.history['val_fn']

    def get_history(self) -> Dict[str, Any]:
        """Retourne une copie de l'historique d'entraînement."""
        # Retourner une shallow copy pour éviter modifications externes non voulues
        return {k: list(v) if isinstance(v, list) else v for k, v in self.history.items()}

class Trainer:
    """Gestionnaire d'entraînement pour le modèle de visibilité."""

    def __init__(self, model: nn.Module, train_loader : torch.utils.data.DataLoader, val_loader: torch.utils.data.DataLoader, config: Config):
        """
        Args:
            model: Modèle à entraîner
            train_loader: DataLoader d'entraînement
            val_loader: DataLoader de validation
            config: Configuration (objet Config)
        """

        self.device = config.device
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config

        # Déplacer le modèle sur le device
        self.model = self.model.to(self.device)

        # Calculateur de métriques
        self.metrics_calculator = MetricsCalculator()

        # Setup des composants d'entraînement
        self.criterion = self._setup_criterion()
        self.optimizer = self._setup_optimizer()
        self.scheduler = self._setup_scheduler()

        # TensorBoard
        self.writer = SummaryWriter(config.paths.experiment_name)

        # Checkpoint manager
        self.checkpoint_manager = CheckpointManager(
            model_folder=config.paths.model_folder,
            model_basename=config.paths.model_basename,
            keep_last_n=config.training.keep_last_n_checkpoints
        )

        # État de l'entraînement
        self.state = TrainingState()

        # Compute feature statistics (mean/std) from the training loader for later use at inference
        try:
            self.feature_stats = self._compute_feature_stats()
            print(f"   • Feature stats computed: mean shape {np.array(self.feature_stats['mean']).shape}, std shape {np.array(self.feature_stats['std']).shape}")
        except Exception as e:
            print(f"     Could not compute feature stats: {e}")
            self.feature_stats = None

        # Charger un checkpoint si spécifié
        if config.training.preload:
            self._load_pretrained(config.training.preload)

        print(f"\nTrainer initialisé:")
        print(f"   • Device: {self.device}")

    def _setup_criterion(self) -> nn.Module:
        """Configure la fonction de perte."""
        pos_weight = getattr(self.config.training, 'pos_weight', None)
        pw_tensor = None

        if pos_weight is not None:
            try:
                pw_tensor = torch.tensor(pos_weight, device=self.device, dtype=torch.float32)
            except Exception:
                pass

        return BCE(
            pos_weight=pw_tensor,
            reduction=getattr(self.config.training, 'reduction', 'mean'),
            label_smoothing=getattr(self.config.training, 'label_smoothing', 0.0),
        )

    def _setup_optimizer(self) -> torch.optim.Optimizer:
        """Configure l'optimiseur."""
        return AdamW(
            self.model.parameters(),
            lr=self.config.training.learning_rate,
            betas=self.config.training.optimizer_betas,
            eps=self.config.training.optimizer_eps,
            weight_decay=self.config.training.weight_decay
        )

    def _setup_scheduler(self) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
        """Configure le scheduler."""
        steps_per_epoch = len(self.train_loader)
        total_steps = steps_per_epoch * self.config.training.num_epochs

        return OneCycleLR(
            self.optimizer,
            max_lr=self.config.training.learning_rate,
            total_steps=total_steps,
            pct_start=self.config.training.scheduler_pct_start,
            div_factor=self.config.training.scheduler_div_factor,
            final_div_factor=self.config.training.scheduler_final_div_factor
        )


    def _load_pretrained(self, checkpoint_path: str):
        """Charge un checkpoint pré-entraîné."""
        print(f"\n Chargement du checkpoint: {checkpoint_path}")

        # Option 1: Si checkpoint_path est juste un nom ('best', '01', etc.)
        checkpoint = self.checkpoint_manager.load_checkpoint(
            checkpoint_name=checkpoint_path,
            device=str(self.device)  # Convertir en string
        )

        # Option 2: Si checkpoint_path est un chemin complet
        # checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])

        if 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if 'scheduler_state_dict' in checkpoint and self.scheduler:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        self.state.epoch = checkpoint.get('epoch', 0)
        self.state.global_step = checkpoint.get('global_step', 0)
        self.state.best_val_f1 = checkpoint.get('best_val_f1', 0.0)

        print(f" Checkpoint chargé (epoch {self.state.epoch}, best F1: {self.state.best_val_f1:.4f})")

    def train_epoch(self) -> Dict[str, float]:
        """Effectue une epoch d'entraînement."""
        self.model.train()
        
        total_loss = 0.0
        metrics_accum = MetricsAccumulator(threshold=self.config.training.classification_threshold)
        
        log_every = max(1, len(self.train_loader) // 10)
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.state.epoch + 1}", leave=False)
        
        for batch_idx, batch in enumerate(pbar):
            if len(batch) == 4:
                spectra, auxiliary, labels, ids = batch
            else:
                spectra, auxiliary, labels = batch
            
            spectra = spectra.to(self.device)
            auxiliary = auxiliary.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            predictions = self.model(spectra, auxiliary)
            loss = self.criterion(predictions, labels)
            
            # Backward pass
            loss.backward()
            
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.training.max_grad_norm
            )
            
            # Optimizer step
            self.optimizer.step()
            
            # Scheduler step
            if self.scheduler:
                self.scheduler.step()
            
            # Métriques
            batch_loss = loss.item()
            total_loss += batch_loss
            
            self.state.iteration_losses.append(batch_loss)
            
            with torch.no_grad():
                class_idx = min(1, predictions.shape[1] - 1)
                preds_cls = predictions[:, class_idx]
                labels_cls = labels[:, class_idx]
                
                # Accumuler les métriques
                metrics_accum.update(preds_cls, labels_cls, store_for_probabilistic=True)
                
                # Obtenir les métriques du batch actuel pour affichage
                current_metrics = self.metrics_calculator.compute_metrics(
                    preds_cls, labels_cls
                )
            
            # TensorBoard
            if (batch_idx + 1) % log_every == 0:
                self.writer.add_scalar('Train/Loss_iter', batch_loss, self.state.global_step)
                self.writer.add_scalar('Train/LR', self._get_lr(), self.state.global_step)
                self.writer.add_scalar('Train/MCC_iter', current_metrics.mcc, self.state.global_step)
                self.writer.add_scalar('Train/BalancedAcc_iter', current_metrics.balanced_accuracy, self.state.global_step)
            
            self.state.global_step += 1
            
            # Mise à jour progress bar
            pbar.set_postfix({
                'loss': f'{batch_loss:.4f}',
                'mcc': f'{current_metrics.mcc:.4f}',
                'bal_acc': f'{current_metrics.balanced_accuracy:.4f}',
                'f1': f'{current_metrics.f1:.4f}',
                'f2': f'{current_metrics.f2:.4f}',
                'lr': f'{self._get_lr():.2e}'
            })
        
        # Calcul final sur TOUTES les métriques accumulées
        num_batches = len(self.train_loader)
        avg_loss = total_loss / num_batches
        
        # Compute avec métriques probabilistes
        epoch_metrics = metrics_accum.compute(compute_probabilistic=True)
        
        # 1. G-Mean: √(Recall × Specificity)
        g_mean = np.sqrt(epoch_metrics.recall * epoch_metrics.specificity)
        
        # 2. Min Class Recall
        min_class_recall = min(epoch_metrics.class_0_recall, epoch_metrics.class_1_recall)
        
        # 3. Class Balance Gap
        class_balance_gap = abs(epoch_metrics.class_0_recall - epoch_metrics.class_1_recall)
        
        # 4. Stability Score (moyenne de MCC et Cohen's Kappa)
        stability_score = (epoch_metrics.mcc + epoch_metrics.cohen_kappa) / 2.0
        
        # 5. Production Score (équilibre precision/recall/specificity)
        production_score = (
            0.3 * epoch_metrics.precision +
            0.4 * epoch_metrics.recall +
            0.3 * epoch_metrics.specificity
        )
        
        # 6. F-Harmonic (moyenne harmonique de F1 et F2)
        if epoch_metrics.f1 > 0 and epoch_metrics.f2 > 0:
            f_harmonic = 2 * (epoch_metrics.f1 * epoch_metrics.f2) / (epoch_metrics.f1 + epoch_metrics.f2)
        else:
            f_harmonic = 0.0
        
        # 7. Composite Score (combinaison pondérée de 5 métriques clés)
        composite_score = (
            0.25 * epoch_metrics.mcc +                    # Robustesse générale
            0.20 * epoch_metrics.balanced_accuracy +      # Équilibre des classes
            0.20 * epoch_metrics.f1 +                     # Compromis precision/recall
            0.20 * g_mean +                               # Équilibre recall/specificity
            0.15 * epoch_metrics.cohen_kappa              # Accord au-delà du hasard
        )
        
        # 8. Probabilistic Score (si AUROC disponible)
        probabilistic_score = None
        if epoch_metrics.auroc is not None and epoch_metrics.auroc > 0:
            probabilistic_score = (
                0.50 * epoch_metrics.auroc +
                0.30 * epoch_metrics.auprc +
                0.20 * (1.0 - epoch_metrics.brier_score)  # Brier score inversé (plus bas = mieux)
            )

        return {
            'loss': avg_loss,
            'accuracy': epoch_metrics.accuracy,
            'balanced_accuracy': epoch_metrics.balanced_accuracy,
            'precision': epoch_metrics.precision,
            'recall': epoch_metrics.recall,
            'specificity': epoch_metrics.specificity,
            'f1': epoch_metrics.f1,
            'f2': epoch_metrics.f2,
            'iou': epoch_metrics.iou,
            'mcc': epoch_metrics.mcc,
            'cohen_kappa': epoch_metrics.cohen_kappa,
            'class_0_precision': epoch_metrics.class_0_precision,
            'class_0_recall': epoch_metrics.class_0_recall,
            'class_1_precision': epoch_metrics.class_1_precision,
            'class_1_recall': epoch_metrics.class_1_recall,
            'support_class_0': epoch_metrics.support_class_0,
            'support_class_1': epoch_metrics.support_class_1,
            'auroc': epoch_metrics.auroc if epoch_metrics.auroc is not None else 0.0,
            'auprc': epoch_metrics.auprc if epoch_metrics.auprc is not None else 0.0,
            'brier_score': epoch_metrics.brier_score if epoch_metrics.brier_score is not None else 0.0,
            'tp': epoch_metrics.tp,
            'tn': epoch_metrics.tn,
            'fp': epoch_metrics.fp,
            'fn': epoch_metrics.fn,
            # Nouvelles métriques composites
            'composite_score': composite_score,
            'g_mean': g_mean,
            'min_class_recall': min_class_recall,
            'class_balance_gap': class_balance_gap,
            'stability_score': stability_score,
            'production_score': production_score,
            'f_harmonic': f_harmonic,
            'probabilistic_score': probabilistic_score
        }


    def validate(self) -> Dict[str, float]:
        """Évalue le modèle sur le set de validation."""
        self.model.eval()
        
        total_loss = 0.0
        metrics_accum = MetricsAccumulator(threshold=self.config.training.classification_threshold)
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation", leave=False):
                if len(batch) == 4:
                    spectra, auxiliary, labels, ids = batch
                else:
                    spectra, auxiliary, labels = batch
                
                spectra = spectra.to(self.device)
                auxiliary = auxiliary.to(self.device)
                labels = labels.to(self.device)
                
                predictions = self.model(spectra, auxiliary)
                loss = self.criterion(predictions, labels)
                total_loss += loss.item()
                
                class_idx = min(1, predictions.shape[1] - 1)
                preds_cls = predictions[:, class_idx]
                labels_cls = labels[:, class_idx]
                
                metrics_accum.update(preds_cls, labels_cls, store_for_probabilistic=True)
        
        num_batches = len(self.val_loader)
        avg_loss = total_loss / num_batches
        epoch_metrics = metrics_accum.compute(compute_probabilistic=True)
        
        g_mean = np.sqrt(epoch_metrics.recall * epoch_metrics.specificity)
        min_class_recall = min(epoch_metrics.class_0_recall, epoch_metrics.class_1_recall)
        class_balance_gap = abs(epoch_metrics.class_0_recall - epoch_metrics.class_1_recall)
        stability_score = (epoch_metrics.mcc + epoch_metrics.cohen_kappa) / 2.0
        production_score = (
            0.3 * epoch_metrics.precision +
            0.4 * epoch_metrics.recall +
            0.3 * epoch_metrics.specificity
        )
        if epoch_metrics.f1 > 0 and epoch_metrics.f2 > 0:
            f_harmonic = 2 * (epoch_metrics.f1 * epoch_metrics.f2) / (epoch_metrics.f1 + epoch_metrics.f2)
        else:
            f_harmonic = 0.0
        composite_score = (
            0.25 * epoch_metrics.mcc +                    # Robustesse générale
            0.20 * epoch_metrics.balanced_accuracy +      # Équilibre des classes
            0.20 * epoch_metrics.f1 +                     # Compromis precision/recall
            0.20 * g_mean +                               # Équilibre recall/specificity
            0.15 * epoch_metrics.cohen_kappa              # Accord au-delà du hasard
        )
        
        # 8. Probabilistic Score (si AUROC disponible)
        probabilistic_score = None
        if epoch_metrics.auroc is not None and epoch_metrics.auroc > 0:
            probabilistic_score = (
                0.50 * epoch_metrics.auroc +
                0.30 * epoch_metrics.auprc +
                0.20 * (1.0 - epoch_metrics.brier_score)  # Brier score inversé (plus bas = mieux)
            )
        

        return {
            'loss': avg_loss,
            'accuracy': epoch_metrics.accuracy,
            'balanced_accuracy': epoch_metrics.balanced_accuracy,
            'precision': epoch_metrics.precision,
            'recall': epoch_metrics.recall,
            'specificity': epoch_metrics.specificity,
            'f1': epoch_metrics.f1,
            'f2': epoch_metrics.f2,
            'iou': epoch_metrics.iou,
            'mcc': epoch_metrics.mcc,
            'cohen_kappa': epoch_metrics.cohen_kappa,
            'class_0_precision': epoch_metrics.class_0_precision,
            'class_0_recall': epoch_metrics.class_0_recall,
            'class_1_precision': epoch_metrics.class_1_precision,
            'class_1_recall': epoch_metrics.class_1_recall,
            'support_class_0': epoch_metrics.support_class_0,
            'support_class_1': epoch_metrics.support_class_1,
            'auroc': epoch_metrics.auroc if epoch_metrics.auroc is not None else 0.0,
            'auprc': epoch_metrics.auprc if epoch_metrics.auprc is not None else 0.0,
            'brier_score': epoch_metrics.brier_score if epoch_metrics.brier_score is not None else 0.0,
            'tp': epoch_metrics.tp,
            'tn': epoch_metrics.tn,
            'fp': epoch_metrics.fp,
            'fn': epoch_metrics.fn,
            # Nouvelles métriques composites
            'composite_score': composite_score,
            'g_mean': g_mean,
            'min_class_recall': min_class_recall,
            'class_balance_gap': class_balance_gap,
            'stability_score': stability_score,
            'production_score': production_score,
            'f_harmonic': f_harmonic,
            'probabilistic_score': probabilistic_score
        }

    def _compute_feature_stats(self):
        """Calcule la moyenne et l'écart-type des features sur l'ensemble d'entraînement.
        
        Retourne un dictionnaire {'mean': [...], 'std': [...]} utilisable directement par le script
        de prédiction pour normaliser les features à l'inférence.
        """

        sum_ = None
        sumsq_ = None
        total_count = 0
        
        with torch.no_grad():
            for features, labels, masks in tqdm(self.train_loader, desc="Computing feature stats", leave=False):
                f = features.detach().cpu().numpy()
                m = masks.detach().cpu().numpy()
                
                batch_size = f.shape[0]
                for i in range(batch_size):
                    valid = m[i]
                    if not valid.any():
                        continue
                    vals = f[i, valid, :].astype(np.float64)
                    if sum_ is None:
                        sum_ = vals.sum(axis=0)
                        sumsq_ = (vals ** 2).sum(axis=0)
                    else:
                        sum_ += vals.sum(axis=0)
                        sumsq_ += (vals ** 2).sum(axis=0)
                    total_count += vals.shape[0]
        
        if total_count == 0:
            raise RuntimeError('No valid points found in train_loader to compute feature stats')
        
        mean = sum_ / total_count
        var = (sumsq_ / total_count) - (mean ** 2)
        std = np.sqrt(np.maximum(var, 1e-8))
        
        return {'mean': mean.tolist(), 'std': std.tolist()}

    def train(self) -> Dict:
        """
        Boucle d'entraînement complète avec métriques composites avancées.

        Returns:
            Dict contenant les meilleurs résultats et statistiques d'entraînement
        """
        start_time = time.time()
        num_epochs = self.config.training.num_epochs
        patience = self.config.training.patience
        min_delta = self.config.training.min_delta
        save_every = self.config.training.save_every_n_epochs

        best_metrics = {
            'mcc': -1.0,
            'composite_score': -float('inf'),
            'g_mean': 0.0,
            'stability_score': -1.0,
            'production_score': 0.0,
            'auroc': 0.0,
            'f_harmonic': 0.0
        }

        # Historiques pour les nouvelles métriques
        train_composite_scores = []
        val_composite_scores = []
        train_g_means = []
        val_g_means = []
        val_min_class_recalls = []
        val_class_balance_gaps = []
        val_stability_scores = []
        val_production_scores = []
        val_f_harmonic = []

        print(f"\n{'='*90}")
        print(f"🚀 DÉBUT DE L'ENTRAÎNEMENT")
        print(f"{'='*90}")
        print(f"  Epochs:              {num_epochs}")
        print(f"  Patience:            {patience}")
        print(f"  Min Delta:           {min_delta}")
        print(f"  Device:              {self.device}")
        print(f"  Batch Size:          {self.config.training.batch_size}")
        print(f"  Learning Rate:       {self.config.training.learning_rate}")
        print(f"  Weight Decay:        {self.config.training.weight_decay}")
        print(f"  Optimizer:           {self.optimizer.__class__.__name__}")
        print(f"  Scheduler:           {self.scheduler.__class__.__name__ if self.scheduler else 'None'}")
        print(f"  Criterion:           {self.criterion.__class__.__name__}")
        print(f"  Train Batches:       {len(self.train_loader)}")
        print(f"  Val Batches:         {len(self.val_loader)}")
        print(f"{'='*90}\n")

        try:
            for epoch in range(num_epochs):
                self.state.epoch = epoch
                epoch_start = time.time()

                train_metrics = self.train_epoch()
                
                val_metrics = self.validate()

                self.state.train_losses.append(train_metrics['loss'])
                self.state.val_losses.append(val_metrics['loss'])
                
                self.state.train_accuracy.append(train_metrics['accuracy'])
                self.state.val_accuracy.append(val_metrics['accuracy'])
                
                self.state.train_balanced_accuracy.append(train_metrics['balanced_accuracy'])
                self.state.val_balanced_accuracy.append(val_metrics['balanced_accuracy'])
                
                self.state.train_precision.append(train_metrics['precision'])
                self.state.val_precision.append(val_metrics['precision'])
                
                self.state.train_recall.append(train_metrics['recall'])
                self.state.val_recall.append(val_metrics['recall'])
                
                self.state.train_specificity.append(train_metrics['specificity'])
                self.state.val_specificity.append(val_metrics['specificity'])
                
                self.state.train_f1.append(train_metrics['f1'])
                self.state.val_f1.append(val_metrics['f1'])
                
                self.state.train_f2.append(train_metrics['f2'])
                self.state.val_f2.append(val_metrics['f2'])
                
                self.state.train_iou.append(train_metrics['iou'])
                self.state.val_iou.append(val_metrics['iou'])
                
                self.state.train_mcc.append(train_metrics['mcc'])
                self.state.val_mcc.append(val_metrics['mcc'])
                
                self.state.train_cohen_kappa.append(train_metrics['cohen_kappa'])
                self.state.val_cohen_kappa.append(val_metrics['cohen_kappa'])
                
                # Métriques par classe
                self.state.train_class_0_precision.append(train_metrics['class_0_precision'])
                self.state.val_class_0_precision.append(val_metrics['class_0_precision'])
                
                self.state.train_class_0_recall.append(train_metrics['class_0_recall'])
                self.state.val_class_0_recall.append(val_metrics['class_0_recall'])
                
                self.state.train_class_1_precision.append(train_metrics['class_1_precision'])
                self.state.val_class_1_precision.append(val_metrics['class_1_precision'])
                
                self.state.train_class_1_recall.append(train_metrics['class_1_recall'])
                self.state.val_class_1_recall.append(val_metrics['class_1_recall'])
                
                # Support
                self.state.train_support_class_0.append(train_metrics['support_class_0'])
                self.state.val_support_class_0.append(val_metrics['support_class_0'])
                
                self.state.train_support_class_1.append(train_metrics['support_class_1'])
                self.state.val_support_class_1.append(val_metrics['support_class_1'])
                
                # Métriques probabilistes
                self.state.train_auroc.append(train_metrics['auroc'] if train_metrics['auroc'] is not None else 0.0)
                self.state.val_auroc.append(val_metrics['auroc'] if val_metrics['auroc'] is not None else 0.0)
                
                self.state.train_auprc.append(train_metrics['auprc'] if train_metrics['auprc'] is not None else 0.0)
                self.state.val_auprc.append(val_metrics['auprc'] if val_metrics['auprc'] is not None else 0.0)
                
                self.state.train_brier_score.append(train_metrics['brier_score'] if train_metrics['brier_score'] is not None else 0.0)
                self.state.val_brier_score.append(val_metrics['brier_score'] if val_metrics['brier_score'] is not None else 0.0)

                train_composite_scores.append(train_metrics['composite_score'])
                val_composite_scores.append(val_metrics['composite_score'])
                
                train_g_means.append(train_metrics['g_mean'])
                val_g_means.append(val_metrics['g_mean'])
                
                val_min_class_recalls.append(val_metrics['min_class_recall'])
                val_class_balance_gaps.append(val_metrics['class_balance_gap'])
                val_stability_scores.append(val_metrics['stability_score'])
                val_production_scores.append(val_metrics['production_score'])
                val_f_harmonic.append(val_metrics['f_harmonic'])

                self.writer.add_scalars('Loss', {
                    'train': train_metrics['loss'],
                    'val': val_metrics['loss']
                }, epoch)

                self.writer.add_scalars('MCC', {
                    'train': train_metrics['mcc'],
                    'val': val_metrics['mcc']
                }, epoch)

                self.writer.add_scalars('Balanced_Accuracy', {
                    'train': train_metrics['balanced_accuracy'],
                    'val': val_metrics['balanced_accuracy']
                }, epoch)

                self.writer.add_scalars('F1_Score', {
                    'train': train_metrics['f1'],
                    'val': val_metrics['f1']
                }, epoch)

                self.writer.add_scalars('F2_Score', {
                    'train': train_metrics['f2'],
                    'val': val_metrics['f2']
                }, epoch)

                self.writer.add_scalars('Precision', {
                    'train': train_metrics['precision'],
                    'val': val_metrics['precision']
                }, epoch)

                self.writer.add_scalars('Recall', {
                    'train': train_metrics['recall'],
                    'val': val_metrics['recall']
                }, epoch)

                self.writer.add_scalars('Specificity', {
                    'train': train_metrics['specificity'],
                    'val': val_metrics['specificity']
                }, epoch)

                self.writer.add_scalars('Cohen_Kappa', {
                    'train': train_metrics['cohen_kappa'],
                    'val': val_metrics['cohen_kappa']
                }, epoch)

                self.writer.add_scalars('IoU', {
                    'train': train_metrics['iou'],
                    'val': val_metrics['iou']
                }, epoch)

                # Métriques par classe
                self.writer.add_scalars('Class_0_Precision', {
                    'train': train_metrics['class_0_precision'],
                    'val': val_metrics['class_0_precision']
                }, epoch)

                self.writer.add_scalars('Class_1_Precision', {
                    'train': train_metrics['class_1_precision'],
                    'val': val_metrics['class_1_precision']
                }, epoch)

                self.writer.add_scalars('Class_0_Recall', {
                    'train': train_metrics['class_0_recall'],
                    'val': val_metrics['class_0_recall']
                }, epoch)

                self.writer.add_scalars('Class_1_Recall', {
                    'train': train_metrics['class_1_recall'],
                    'val': val_metrics['class_1_recall']
                }, epoch)

                # Métriques probabilistes
                if train_metrics['auroc'] is not None and val_metrics['auroc'] is not None:
                    self.writer.add_scalars('AUROC', {
                        'train': train_metrics['auroc'],
                        'val': val_metrics['auroc']
                    }, epoch)

                if train_metrics['auprc'] is not None and val_metrics['auprc'] is not None:
                    self.writer.add_scalars('AUPRC', {
                        'train': train_metrics['auprc'],
                        'val': val_metrics['auprc']
                    }, epoch)

                if train_metrics['brier_score'] is not None and val_metrics['brier_score'] is not None:
                    self.writer.add_scalars('Brier_Score', {
                        'train': train_metrics['brier_score'],
                        'val': val_metrics['brier_score']
                    }, epoch)

                self.writer.add_scalars('Composite_Score', {
                    'train': train_metrics['composite_score'],
                    'val': val_metrics['composite_score']
                }, epoch)
                
                self.writer.add_scalars('G_Mean', {
                    'train': train_metrics['g_mean'],
                    'val': val_metrics['g_mean']
                }, epoch)
                
                self.writer.add_scalar('Val/Min_Class_Recall', val_metrics['min_class_recall'], epoch)
                self.writer.add_scalar('Val/Class_Balance_Gap', val_metrics['class_balance_gap'], epoch)
                self.writer.add_scalar('Val/Stability_Score', val_metrics['stability_score'], epoch)
                self.writer.add_scalar('Val/Production_Score', val_metrics['production_score'], epoch)
                self.writer.add_scalar('Val/F_Harmonic', val_metrics['f_harmonic'], epoch)
                
                if val_metrics.get('probabilistic_score') is not None:
                    self.writer.add_scalar('Val/Probabilistic_Score', val_metrics['probabilistic_score'], epoch)

                # Learning rate
                self.writer.add_scalar('Learning_Rate', self._get_lr(), epoch)

                # ================================================================
                # AFFICHAGE CONSOLE
                # ================================================================
                epoch_time = time.time() - epoch_start
                
                print(f"\n{'='*90}")
                print(f" Epoch {epoch + 1}/{num_epochs} - Temps: {epoch_time:.2f}s")
                print(f"{'='*90}")
                
                print(f"\n MÉTRIQUES DE BASE:")
                print(f"  Loss:              Train: {train_metrics['loss']:.4f} | Val: {val_metrics['loss']:.4f}")
                print(f"  Accuracy:          Train: {train_metrics['accuracy']:.4f} | Val: {val_metrics['accuracy']:.4f}")
                print(f"  Balanced Acc:      Train: {train_metrics['balanced_accuracy']:.4f} | Val: {val_metrics['balanced_accuracy']:.4f}")
                print(f"  MCC:               Train: {train_metrics['mcc']:.4f} | Val: {val_metrics['mcc']:.4f}")
                print(f"  Cohen's Kappa:     Train: {train_metrics['cohen_kappa']:.4f} | Val: {val_metrics['cohen_kappa']:.4f}")
                
                print(f"\n F-SCORES & IoU:")
                print(f"  F1:                Train: {train_metrics['f1']:.4f} | Val: {val_metrics['f1']:.4f}")
                print(f"  F2:                Train: {train_metrics['f2']:.4f} | Val: {val_metrics['f2']:.4f}")
                print(f"  IoU:               Train: {train_metrics['iou']:.4f} | Val: {val_metrics['iou']:.4f}")
                
                print(f"\n PRECISION, RECALL, SPECIFICITY:")
                print(f"  Precision:         Train: {train_metrics['precision']:.4f} | Val: {val_metrics['precision']:.4f}")
                print(f"  Recall:            Train: {train_metrics['recall']:.4f} | Val: {val_metrics['recall']:.4f}")
                print(f"  Specificity:       Train: {train_metrics['specificity']:.4f} | Val: {val_metrics['specificity']:.4f}")
                
                print(f"\n MÉTRIQUES COMPOSITES:")
                print(f"  Composite Score:   Train: {train_metrics['composite_score']:.4f} | Val: {val_metrics['composite_score']:.4f}")
                print(f"  G-Mean:            Train: {train_metrics['g_mean']:.4f} | Val: {val_metrics['g_mean']:.4f}")
                print(f"  Stability Score:   Train: {train_metrics['stability_score']:.4f} | Val: {val_metrics['stability_score']:.4f}")
                print(f"  Production Score:  Train: {train_metrics['production_score']:.4f} | Val: {val_metrics['production_score']:.4f}")
                print(f"  F-Harmonic:        Train: {train_metrics['f_harmonic']:.4f} | Val: {val_metrics['f_harmonic']:.4f}")
                
                print(f"\n  ÉQUILIBRE ENTRE CLASSES:")
                print(f"  Min Class Recall:         {val_metrics['min_class_recall']:.4f}")
                print(f"  Class Balance Gap:        {val_metrics['class_balance_gap']:.4f} (↓ = mieux)")
                print(f"  Classe 0 - Precision:     {val_metrics['class_0_precision']:.4f}")
                print(f"  Classe 0 - Recall:        {val_metrics['class_0_recall']:.4f}")
                print(f"  Classe 1 - Precision:     {val_metrics['class_1_precision']:.4f}")
                print(f"  Classe 1 - Recall:        {val_metrics['class_1_recall']:.4f}")
                
                if val_metrics.get('auroc') is not None:
                    print(f"\n MÉTRIQUES PROBABILISTES:")
                    print(f"  AUROC:             Train: {train_metrics.get('auroc', 0.0):.4f} | Val: {val_metrics['auroc']:.4f}")
                    print(f"  AUPRC:             Train: {train_metrics.get('auprc', 0.0):.4f} | Val: {val_metrics['auprc']:.4f}")
                    print(f"  Brier Score:       Train: {train_metrics.get('brier_score', 0.0):.4f} | Val: {val_metrics['brier_score']:.4f}")
                    if val_metrics.get('probabilistic_score') is not None:
                        print(f"  Probabilistic Sc:         {val_metrics['probabilistic_score']:.4f}")
                
                print(f"\n SUPPORT:")
                print(f"  Classe 0:          Train: {train_metrics['support_class_0']} | Val: {val_metrics['support_class_0']}")
                print(f"  Classe 1:          Train: {train_metrics['support_class_1']} | Val: {val_metrics['support_class_1']}")
                
                print(f"\n HYPERPARAMÈTRES:")
                print(f"  Learning Rate:     {self._get_lr():.2e}")
                print(f"  Patience Counter:  {self.state.patience_counter}/{patience}")

                composite_improvement = val_metrics['composite_score'] - best_metrics['composite_score']
                gmean_improvement = val_metrics['g_mean'] - best_metrics['g_mean']
                stability_improvement = val_metrics['stability_score'] - best_metrics['stability_score']
                production_improvement = val_metrics['production_score'] - best_metrics['production_score']
                mcc_improvement = val_metrics['mcc'] - best_metrics['mcc']
                f_harmonic_improvement = val_metrics['f_harmonic'] - best_metrics['f_harmonic']

                primary_improvement = composite_improvement
                primary_metric_name = 'Composite Score'
                primary_metric_value = val_metrics['composite_score']
                primary_best_value = best_metrics['composite_score']
                
                # 🎯 OPTION B: MCC pur (votre approche actuelle)
                # primary_improvement = mcc_improvement
                # primary_metric_name = 'MCC'
                # primary_metric_value = val_metrics['mcc']
                # primary_best_value = best_metrics['mcc']
                
                # 🎯 OPTION C: G-Mean (excellent pour classes déséquilibrées)
                # primary_improvement = gmean_improvement
                # primary_metric_name = 'G-Mean'
                # primary_metric_value = val_metrics['g_mean']
                # primary_best_value = best_metrics['g_mean']
                
                # 🎯 OPTION D: Stability Score (robuste au bruit)
                # primary_improvement = stability_improvement
                # primary_metric_name = 'Stability Score'
                # primary_metric_value = val_metrics['stability_score']
                # primary_best_value = best_metrics['stability_score']
                
                # 🎯 OPTION E: Production Score (orienté déploiement)
                # primary_improvement = production_improvement
                # primary_metric_name = 'Production Score'
                # primary_metric_value = val_metrics['production_score']
                # primary_best_value = best_metrics['production_score']


                if primary_improvement > min_delta:
                    print(f"\n{'='*90}")
                    print(f" AMÉLIORATION DÉTECTÉE!")
                    print(f"{'='*90}")
                    print(f"  Métrique principale:      {primary_metric_name}")
                    print(f"  Amélioration:             +{primary_improvement:.4f}")
                    print(f"  Valeur actuelle:          {primary_metric_value:.4f}")
                    print(f"  Meilleure valeur précédente: {primary_best_value:.4f}")
                    
                    # Mettre à jour tous les best_metrics
                    best_metrics['composite_score'] = val_metrics['composite_score']
                    best_metrics['mcc'] = val_metrics['mcc']
                    best_metrics['g_mean'] = val_metrics['g_mean']
                    best_metrics['stability_score'] = val_metrics['stability_score']
                    best_metrics['production_score'] = val_metrics['production_score']
                    best_metrics['f_harmonic'] = val_metrics['f_harmonic']
                    if val_metrics.get('auroc') is not None:
                        best_metrics['auroc'] = val_metrics['auroc']
                    
                    # Pour compatibilité avec l'ancien code
                    self.state.best_val_mcc = val_metrics['mcc']
                    self.state.best_val_f1 = val_metrics['f1']
                    self.state.patience_counter = 0

                    # Afficher toutes les améliorations
                    print(f"\n   Détail des améliorations:")
                    print(f"     Composite Score:      +{composite_improvement:.4f}")
                    print(f"     MCC:                  +{mcc_improvement:.4f}")
                    print(f"     G-Mean:               +{gmean_improvement:.4f}")
                    print(f"     Stability Score:      +{stability_improvement:.4f}")
                    print(f"     Production Score:     +{production_improvement:.4f}")
                    print(f"     F-Harmonic:           +{f_harmonic_improvement:.4f}")
                    
                    # Conditions supplémentaires (optionnel - décommenter si besoin)
                    # Exemple: Exiger un minimum de recall sur la classe minoritaire
                    min_recall_threshold = 0.50  # Ajustez selon vos besoins
                    min_class_recall_ok = val_metrics['min_class_recall'] >= min_recall_threshold
                    
                    if min_class_recall_ok:
                        print(f"     ✓ Min Class Recall:   {val_metrics['min_class_recall']:.4f} >= {min_recall_threshold}")
                    else:
                        print(f"     ⚠ Min Class Recall:   {val_metrics['min_class_recall']:.4f} < {min_recall_threshold} (seuil non atteint)")

                    # Sauvegarder le meilleur modèle
                    try:
                        self.checkpoint_manager.save_checkpoint(
                            model=self.model,
                            optimizer=self.optimizer,
                            scheduler=self.scheduler,
                            epoch=epoch,
                            global_step=self.state.global_step,
                            best_val_mcc=self.state.best_val_mcc,
                            best_val_f1=self.state.best_val_f1,
                            history={
                                **self.state.get_history(),
                                # Nouvelles métriques
                                'train_composite_scores': train_composite_scores,
                                'val_composite_scores': val_composite_scores,
                                'train_g_means': train_g_means,
                                'val_g_means': val_g_means,
                                'val_min_class_recalls': val_min_class_recalls,
                                'val_class_balance_gaps': val_class_balance_gaps,
                                'val_stability_scores': val_stability_scores,
                                'val_production_scores': val_production_scores,
                                'val_f_harmonic': val_f_harmonic,
                                'best_metrics': best_metrics
                            },
                            feature_stats=self.feature_stats,
                            config=self.config.to_dict(),
                            is_best=True
                        )
                        print(f"\n💾 Meilleur modèle sauvegardé!")
                        print(f"{'='*90}\n")
                    except Exception as e:
                        print(f"\n❌ Erreur lors de la sauvegarde du meilleur modèle: {e}")
                        print(f"{'='*90}\n")
                    
                else:
                    self.state.patience_counter += 1
                    print(f"\n{'='*90}")
                    print(f" PATIENCE: {self.state.patience_counter}/{patience}")
                    print(f"{'='*90}")
                    print(f"  Métrique principale:      {primary_metric_name}")
                    print(f"  Valeur actuelle:          {primary_metric_value:.4f}")
                    print(f"  Meilleure valeur:         {primary_best_value:.4f}")
                    print(f"  Amélioration:             {primary_improvement:+.4f}")
                    print(f"  Amélioration requise:     >{min_delta:.4f}")
                    print(f"\n   État des autres métriques:")
                    print(f"     Composite Score:      {val_metrics['composite_score']:.4f} (best: {best_metrics['composite_score']:.4f})")
                    print(f"     MCC:                  {val_metrics['mcc']:.4f} (best: {best_metrics['mcc']:.4f})")
                    print(f"     G-Mean:               {val_metrics['g_mean']:.4f} (best: {best_metrics['g_mean']:.4f})")
                    print(f"     Stability Score:      {val_metrics['stability_score']:.4f} (best: {best_metrics['stability_score']:.4f})")
                    print(f"     Production Score:     {val_metrics['production_score']:.4f} (best: {best_metrics['production_score']:.4f})")
                    print(f"{'='*90}\n")


                if epoch == 0 and primary_improvement <= min_delta:
                    try:
                        self.checkpoint_manager.save_checkpoint(
                            model=self.model,
                            optimizer=self.optimizer,
                            scheduler=self.scheduler,
                            epoch=epoch,
                            global_step=self.state.global_step,
                            best_val_mcc=self.state.best_val_mcc,
                            best_val_f1=self.state.best_val_f1,
                            history={
                                **self.state.get_history(),
                                'train_composite_scores': train_composite_scores,
                                'val_composite_scores': val_composite_scores,
                                'train_g_means': train_g_means,
                                'val_g_means': val_g_means,
                                'val_min_class_recalls': val_min_class_recalls,
                                'val_class_balance_gaps': val_class_balance_gaps,
                                'val_stability_scores': val_stability_scores,
                                'val_production_scores': val_production_scores,
                                'val_f_harmonic': val_f_harmonic,
                                'best_metrics': best_metrics
                            },
                            feature_stats=self.feature_stats,
                            config=self.config.to_dict(),
                            is_best=False
                        )
                        print(f" Checkpoint initial epoch {epoch + 1} sauvegardé\n")
                    except Exception as e:
                        print(f"  Erreur sauvegarde checkpoint initial: {e}\n")

                if (epoch + 1) % save_every == 0:
                    try:
                        self.checkpoint_manager.save_checkpoint(
                            model=self.model,
                            optimizer=self.optimizer,
                            scheduler=self.scheduler,
                            epoch=epoch,
                            global_step=self.state.global_step,
                            best_val_mcc=self.state.best_val_mcc,
                            best_val_f1=self.state.best_val_f1,
                            history={
                                **self.state.get_history(),
                                'train_composite_scores': train_composite_scores,
                                'val_composite_scores': val_composite_scores,
                                'train_g_means': train_g_means,
                                'val_g_means': val_g_means,
                                'val_min_class_recalls': val_min_class_recalls,
                                'val_class_balance_gaps': val_class_balance_gaps,
                                'val_stability_scores': val_stability_scores,
                                'val_production_scores': val_production_scores,
                                'val_f_harmonic': val_f_harmonic,
                                'best_metrics': best_metrics
                            },
                            feature_stats=self.feature_stats,
                            config=self.config.to_dict(),
                            is_best=False
                        )
                        print(f"💾 Checkpoint périodique epoch {epoch + 1} sauvegardé\n")
                    except Exception as e:
                        print(f"⚠️  Erreur sauvegarde checkpoint périodique: {e}\n")

                # ================================================================
                # EARLY STOPPING
                # ================================================================
                if self.state.patience_counter >= patience:
                    print(f"\n{'='*90}")
                    print(f"  EARLY STOPPING DÉCLENCHÉ")
                    print(f"{'='*90}")
                    print(f"  Patience épuisée:         {patience} epochs")
                    print(f"  Meilleur epoch:           {epoch + 1 - patience}")
                    print(f"  Meilleur {primary_metric_name}: {primary_best_value:.4f}")
                    print(f"\n   Meilleures valeurs:")
                    print(f"     Composite Score:      {best_metrics['composite_score']:.4f}")
                    print(f"     MCC:                  {best_metrics['mcc']:.4f}")
                    print(f"     G-Mean:               {best_metrics['g_mean']:.4f}")
                    print(f"     Stability Score:      {best_metrics['stability_score']:.4f}")
                    print(f"     Production Score:     {best_metrics['production_score']:.4f}")
                    print(f"     F-Harmonic:           {best_metrics['f_harmonic']:.4f}")
                    if best_metrics['auroc'] > 0:
                        print(f"     AUROC:                {best_metrics['auroc']:.4f}")
                    print(f"{'='*90}\n")
                    break

            # ================================================================
            # FIN DE L'ENTRAÎNEMENT - RÉSUMÉ FINAL
            # ================================================================
            elapsed_time = time.time() - start_time
            
            print(f"\n{'='*90}")
            print(f" ENTRAÎNEMENT TERMINÉ")
            print(f"{'='*90}")
            print(f"  Durée totale:             {self._format_time(elapsed_time)}")
            print(f"  Epochs effectués:         {epoch + 1}/{num_epochs}")
            print(f"  Steps totaux:             {self.state.global_step}")
            print(f"  Early stopping:           {'Oui' if self.state.patience_counter >= patience else 'Non'}")
            
            # Trouver l'epoch avec le meilleur score
            best_epoch_idx = np.argmax(val_composite_scores)
            
            print(f"\n{'='*90}")
            print(f" MEILLEURS RÉSULTATS (Epoch {best_epoch_idx + 1})")
            print(f"{'='*90}")
            
            print(f"\n   Métriques composites:")
            print(f"     Composite Score:      {best_metrics['composite_score']:.4f}")
            print(f"     G-Mean:               {best_metrics['g_mean']:.4f}")
            print(f"     Stability Score:      {best_metrics['stability_score']:.4f}")
            print(f"     Production Score:     {best_metrics['production_score']:.4f}")
            print(f"     F-Harmonic:           {best_metrics['f_harmonic']:.4f}")
            
            print(f"\n   Métriques de base:")
            print(f"     MCC:                  {best_metrics['mcc']:.4f}")
            print(f"     F1:                   {self.state.val_f1[best_epoch_idx]:.4f}")
            print(f"     F2:                   {self.state.val_f2[best_epoch_idx]:.4f}")
            print(f"     Balanced Accuracy:    {self.state.val_balanced_accuracy[best_epoch_idx]:.4f}")
            print(f"     Cohen's Kappa:        {self.state.val_cohen_kappa[best_epoch_idx]:.4f}")
            print(f"     IoU:                  {self.state.val_iou[best_epoch_idx]:.4f}")
            
            print(f"\n    Équilibre des classes:")
            print(f"     Min Class Recall:     {val_min_class_recalls[best_epoch_idx]:.4f}")
            print(f"     Class Balance Gap:    {val_class_balance_gaps[best_epoch_idx]:.4f}")
            print(f"     Classe 0 Precision:   {self.state.val_class_0_precision[best_epoch_idx]:.4f}")
            print(f"     Classe 0 Recall:      {self.state.val_class_0_recall[best_epoch_idx]:.4f}")
            print(f"     Classe 1 Precision:   {self.state.val_class_1_precision[best_epoch_idx]:.4f}")
            print(f"     Classe 1 Recall:      {self.state.val_class_1_recall[best_epoch_idx]:.4f}")
            
            if best_metrics['auroc'] > 0:
                print(f"\n   Métriques probabilistes:")
                print(f"     AUROC:                {best_metrics['auroc']:.4f}")
                print(f"     AUPRC:                {self.state.val_auprc[best_epoch_idx]:.4f}")
                print(f"     Brier Score:          {self.state.val_brier_score[best_epoch_idx]:.4f}")
            
            print(f"\n   Support:")
            print(f"     Classe 0:             {self.state.val_support_class_0[best_epoch_idx]}")
            print(f"     Classe 1:             {self.state.val_support_class_1[best_epoch_idx]}")
            
            print(f"\n{'='*90}\n")

            # Fermeture du TensorBoard writer
            self.writer.close()

            return {
                'best_metrics': best_metrics,
                'best_epoch': best_epoch_idx + 1,
                'final_epoch': epoch + 1,
                'training_time': elapsed_time,
                'early_stopped': self.state.patience_counter >= patience,
                'history': {
                    **self.state.get_history(),
                    'train_composite_scores': train_composite_scores,
                    'val_composite_scores': val_composite_scores,
                    'train_g_means': train_g_means,
                    'val_g_means': val_g_means,
                    'val_min_class_recalls': val_min_class_recalls,
                    'val_class_balance_gaps': val_class_balance_gaps,
                    'val_stability_scores': val_stability_scores,
                    'val_production_scores': val_production_scores,
                    'val_f_harmonic': val_f_harmonic
                }
            }

        except KeyboardInterrupt:
            print("\n\n  Entraînement interrompu par l'utilisateur")
            print(f"   Epoch actuel: {epoch + 1}/{num_epochs}")
            print(f"   Steps effectués: {self.state.global_step}")
            self.writer.close()
            raise
            
        except Exception as e:
            print(f"\n\n ERREUR PENDANT L'ENTRAÎNEMENT")
            print(f"   Epoch: {epoch + 1}/{num_epochs}")
            print(f"   Erreur: {str(e)}")
            import traceback
            print(f"\n   Traceback:")
            traceback.print_exc()
            self.writer.close()
            raise

    def _get_lr(self) -> float:
        """Récupère le learning rate actuel."""
        return self.optimizer.param_groups[0]['lr']

    def _format_time(self, seconds: float) -> str:
        """Formate une durée en secondes."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)

        if hours > 0:
            return f"{hours}h {minutes}min {secs}s"
        elif minutes > 0:
            return f"{minutes}min {secs}s"
        else:
            return f"{secs}s"