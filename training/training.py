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
from .Loss import BinaryClassificationLoss
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
            print(f"   ⚠️  Could not compute feature stats: {e}")
            self.feature_stats = None

        # Charger un checkpoint si spécifié
        if config.training.preload:
            self._load_pretrained(config.training.preload)

        print(f"\n✅ Trainer initialisé:")
        print(f"   • Device: {self.device}")
        print(f"   • Loss: {config.training.loss_type}")

    def _setup_criterion(self) -> nn.Module:
        """Configure la fonction de perte."""
        # Utiliser BCEWithLogitsLoss compatible avec la sortie du CNN
        # Si des poids de classes sont fournis dans la config, les appliquer
        if getattr(self.config.training, 'pos_weight', None) is not None:
            pw = self.config.training.pos_weight
            try:
                pw_tensor = torch.tensor(pw, device=self.device, dtype=torch.float32)
            except Exception:
                pw_tensor = None
            print(f"   📊 Utilisation de BCEWithLogitsLoss avec pos_weight={pw}")
            if pw_tensor is not None:
                return nn.BCEWithLogitsLoss(pos_weight=pw_tensor)
        # Par défaut
        return nn.BCEWithLogitsLoss()

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
        print(f"\n📥 Chargement du checkpoint: {checkpoint_path}")

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

        print(f"✅ Checkpoint chargé (epoch {self.state.epoch}, best F1: {self.state.best_val_f1:.4f})")

    def train_epoch(self) -> Dict[str, float]:
        """Entraîne le modèle pour une epoch."""
        
        self.model.train()
        
        total_loss = 0.0
        # Utiliser MetricsAccumulator pour accumuler correctement les métriques
        metrics_accum = MetricsAccumulator(threshold=self.config.training.classification_threshold)
        
        log_every = self.config.training.log_every_n_steps
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.state.epoch + 1}")

        for batch_idx, batch in enumerate(pbar):

            # Unpack batch (compatibilité avec collate_fn de dataset)
            if len(batch) == 4:
                spectra, auxiliary, labels, ids = batch
            else:
                spectra, auxiliary, labels = batch

            # Déplacer les données sur le device
            spectra = spectra.to(self.device)
            auxiliary = auxiliary.to(self.device)
            labels = labels.to(self.device)

            # Forward pass
            predictions = self.model(spectra, auxiliary)  # (B, num_classes)
            loss = self.criterion(predictions, labels)
            
            self.optimizer.zero_grad()
           
            
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
                # Pour compatibilité avec les métriques (qui attendent un seul signal binaire),
                # on évalue la métrique principale sur la 2ème classe (index 1) si disponible.
                class_idx = min(1, predictions.shape[1] - 1)
                preds_cls = predictions[:, class_idx]
                labels_cls = labels[:, class_idx]

                # Accumuler les métriques (store_for_probabilistic=True)
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
            
            # Mise à jour progress bar avec métriques clés
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
        
        return {
            'loss': avg_loss,
            
            # Métriques de base
            'accuracy': epoch_metrics.accuracy,
            'balanced_accuracy': epoch_metrics.balanced_accuracy,
            'precision': epoch_metrics.precision,
            'recall': epoch_metrics.recall,
            'specificity': epoch_metrics.specificity,
            
            # F-scores
            'f1': epoch_metrics.f1,
            'f2': epoch_metrics.f2,
            
            # IoU
            'iou': epoch_metrics.iou,
            
            # Métriques pour déséquilibre (CRUCIALES!)
            'mcc': epoch_metrics.mcc,
            'cohen_kappa': epoch_metrics.cohen_kappa,
            
            # Métriques par classe
            'class_0_precision': epoch_metrics.class_0_precision,
            'class_0_recall': epoch_metrics.class_0_recall,
            'class_1_precision': epoch_metrics.class_1_precision,
            'class_1_recall': epoch_metrics.class_1_recall,
            
            # Support
            'support_class_0': epoch_metrics.support_class_0,
            'support_class_1': epoch_metrics.support_class_1,
            
            # Métriques probabilistes
            'auroc': epoch_metrics.auroc,
            'auprc': epoch_metrics.auprc,
            'brier_score': epoch_metrics.brier_score,
            
            # Matrice de confusion (utile pour debug)
            'tp': epoch_metrics.tp,
            'fp': epoch_metrics.fp,
            'tn': epoch_metrics.tn,
            'fn': epoch_metrics.fn
        }


    def validate(self) -> Dict[str, float]:
        """Évalue le modèle sur le set de validation."""
        self.model.eval()
        
        total_loss = 0.0
        # Utiliser MetricsAccumulator pour accumuler correctement les métriques
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

                predictions = self.model(spectra, auxiliary)  # (B, num_classes)

                loss = self.criterion(predictions, labels)
                total_loss += loss.item()

                # Accumuler les métriques pour la classe cible (index 1 par défaut)
                class_idx = min(1, predictions.shape[1] - 1)
                preds_cls = predictions[:, class_idx]
                labels_cls = labels[:, class_idx]

                metrics_accum.update(preds_cls, labels_cls, store_for_probabilistic=True)
        
        num_batches = len(self.val_loader)
        avg_loss = total_loss / num_batches
        
        # Compute avec métriques probabilistes
        epoch_metrics = metrics_accum.compute(compute_probabilistic=True)
        
        return {
            'loss': avg_loss,
            
            # Métriques de base
            'accuracy': epoch_metrics.accuracy,
            'balanced_accuracy': epoch_metrics.balanced_accuracy,
            'precision': epoch_metrics.precision,
            'recall': epoch_metrics.recall,
            'specificity': epoch_metrics.specificity,
            
            # F-scores
            'f1': epoch_metrics.f1,
            'f2': epoch_metrics.f2,
            
            # IoU
            'iou': epoch_metrics.iou,
            
            # Métriques pour déséquilibre (CRUCIALES!)
            'mcc': epoch_metrics.mcc,
            'cohen_kappa': epoch_metrics.cohen_kappa,
            
            # Métriques par classe
            'class_0_precision': epoch_metrics.class_0_precision,
            'class_0_recall': epoch_metrics.class_0_recall,
            'class_1_precision': epoch_metrics.class_1_precision,
            'class_1_recall': epoch_metrics.class_1_recall,
            
            # Support
            'support_class_0': epoch_metrics.support_class_0,
            'support_class_1': epoch_metrics.support_class_1,
            
            # Métriques probabilistes
            'auroc': epoch_metrics.auroc,
            'auprc': epoch_metrics.auprc,
            'brier_score': epoch_metrics.brier_score,
            
            # Matrice de confusion (utile pour debug)
            'tp': epoch_metrics.tp,
            'fp': epoch_metrics.fp,
            'tn': epoch_metrics.tn,
            'fn': epoch_metrics.fn
        }


    def _compute_feature_stats(self):
        """Calcule la moyenne et l'écart-type des features sur l'ensemble d'entraînement.
        
        Retourne un dictionnaire {'mean': [...], 'std': [...]} utilisable directement par le script
        de prédiction pour normaliser les features à l'inférence.
        """
        # Accumulate sums in double precision for numeric stability
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
        """Boucle d'entraînement complète."""

        start_time = time.time()

        num_epochs = self.config.training.num_epochs
        patience = self.config.training.patience
        min_delta = self.config.training.min_delta
        save_every = self.config.training.save_every_n_epochs

        try:
            for epoch in range(num_epochs):
                self.state.epoch = epoch

                # Entraînement
                train_metrics = self.train_epoch()

                # Validation
                val_metrics = self.validate()

                # Historique - TOUTES les métriques
                self.state.train_losses.append(train_metrics['loss'])
                self.state.val_losses.append(val_metrics['loss'])
                
                # Métriques de base
                self.state.train_accuracy.append(train_metrics['accuracy'])
                self.state.val_accuracy.append(val_metrics['accuracy'])
                self.state.train_balanced_accuracy.append(train_metrics['balanced_accuracy'])
                self.state.val_balanced_accuracy.append(val_metrics['balanced_accuracy'])
                
                # F-scores
                self.state.train_f1.append(train_metrics['f1'])
                self.state.val_f1.append(val_metrics['f1'])
                self.state.train_f2.append(train_metrics['f2'])
                self.state.val_f2.append(val_metrics['f2'])
                
                # IoU
                self.state.train_iou.append(train_metrics['iou'])
                self.state.val_iou.append(val_metrics['iou'])
                
                # Métriques pour déséquilibre (CRUCIALES!)
                self.state.train_mcc.append(train_metrics['mcc'])
                self.state.val_mcc.append(val_metrics['mcc'])
                self.state.train_cohen_kappa.append(train_metrics['cohen_kappa'])
                self.state.val_cohen_kappa.append(val_metrics['cohen_kappa'])
                
                # Recall/Precision/Specificity
                self.state.train_recall.append(train_metrics['recall'])
                self.state.val_recall.append(val_metrics['recall'])
                self.state.train_precision.append(train_metrics['precision'])
                self.state.val_precision.append(val_metrics['precision'])
                self.state.train_specificity.append(train_metrics['specificity'])
                self.state.val_specificity.append(val_metrics['specificity'])
                
                # Par classe
                self.state.train_class_0_precision.append(train_metrics['class_0_precision'])
                self.state.val_class_0_precision.append(val_metrics['class_0_precision'])
                self.state.train_class_0_recall.append(train_metrics['class_0_recall'])
                self.state.val_class_0_recall.append(val_metrics['class_0_recall'])
                self.state.train_class_1_precision.append(train_metrics['class_1_precision'])
                self.state.val_class_1_precision.append(val_metrics['class_1_precision'])
                self.state.train_class_1_recall.append(train_metrics['class_1_recall'])
                self.state.val_class_1_recall.append(val_metrics['class_1_recall'])
                
                # Métriques probabilistes (peuvent être None)
                self.state.train_auroc.append(train_metrics.get('auroc'))
                self.state.val_auroc.append(val_metrics.get('auroc'))
                self.state.train_auprc.append(train_metrics.get('auprc'))
                self.state.val_auprc.append(val_metrics.get('auprc'))
                self.state.train_brier_score.append(train_metrics.get('brier_score'))
                self.state.val_brier_score.append(val_metrics.get('brier_score'))

                # TensorBoard - Métriques principales
                self.writer.add_scalars('Loss', {
                    'train': train_metrics['loss'],
                    'val': val_metrics['loss']
                }, epoch)

                # MCC - LA métrique la plus importante pour le déséquilibre
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

                self.writer.add_scalars('IoU', {
                    'train': train_metrics['iou'],
                    'val': val_metrics['iou']
                }, epoch)

                self.writer.add_scalars('Cohen_Kappa', {
                    'train': train_metrics['cohen_kappa'],
                    'val': val_metrics['cohen_kappa']
                }, epoch)

                # Performance par classe
                self.writer.add_scalars('Class_0_Recall', {
                    'train': train_metrics['class_0_recall'],
                    'val': val_metrics['class_0_recall']
                }, epoch)

                self.writer.add_scalars('Class_1_Recall', {
                    'train': train_metrics['class_1_recall'],
                    'val': val_metrics['class_1_recall']
                }, epoch)

                # Métriques probabilistes
                if train_metrics.get('auroc') is not None:
                    self.writer.add_scalars('AUROC', {
                        'train': train_metrics['auroc'],
                        'val': val_metrics['auroc']
                    }, epoch)
                    
                    self.writer.add_scalars('AUPRC', {
                        'train': train_metrics['auprc'],
                        'val': val_metrics['auprc']
                    }, epoch)

                # Affichage enrichi
                print(f"\n{'='*80}")
                print(f"📊 Epoch {epoch + 1}/{num_epochs}")
                print(f"{'='*80}")
                print(f"Loss:")
                print(f"  Train: {train_metrics['loss']:.4f} | Val: {val_metrics['loss']:.4f}")
                print(f"\n🎯 Métriques clés pour déséquilibre:")
                print(f"  MCC (Matthews Corr.):")
                print(f"    Train: {train_metrics['mcc']:.4f} | Val: {val_metrics['mcc']:.4f}")
                print(f"  Balanced Accuracy:")
                print(f"    Train: {train_metrics['balanced_accuracy']:.4f} | Val: {val_metrics['balanced_accuracy']:.4f}")
                print(f"  Cohen's Kappa:")
                print(f"    Train: {train_metrics['cohen_kappa']:.4f} | Val: {val_metrics['cohen_kappa']:.4f}")
                print(f"\n📈 F-Scores:")
                print(f"  F1: Train: {train_metrics['f1']:.4f} | Val: {val_metrics['f1']:.4f}")
                print(f"  F2: Train: {train_metrics['f2']:.4f} | Val: {val_metrics['f2']:.4f}")
                print(f"\n🔍 Performance par classe:")
                print(f"  Classe 0 (caché):   Recall Val: {val_metrics['class_0_recall']:.4f}")
                print(f"  Classe 1 (visible): Recall Val: {val_metrics['class_1_recall']:.4f}")

                # ============================================================
                # EARLY STOPPING BASÉ SUR MCC (CHANGEMENT PRINCIPAL!)
                # ============================================================
                improvement = val_metrics['mcc'] - self.state.best_val_mcc

                if improvement > min_delta:
                    print(f"\n✨ Amélioration MCC: +{improvement:.4f}")
                    self.state.best_val_mcc = val_metrics['mcc']
                    self.state.best_val_f1 = val_metrics['f1']  # Garder aussi pour info
                    self.state.patience_counter = 0

                    # Sauvegarder le meilleur modèle (basé sur MCC)
                    self.checkpoint_manager.save_checkpoint(
                        model=self.model,
                        optimizer=self.optimizer,
                        scheduler=self.scheduler,
                        epoch=epoch,
                        global_step=self.state.global_step,
                        best_val_mcc=self.state.best_val_mcc,
                        best_val_f1=self.state.best_val_f1,
                        history=self.state.get_history(),
                        feature_stats=self.feature_stats,
                        config=self.config.to_dict(),
                        is_best=True
                    )
                    print(f"💾 Meilleur modèle sauvegardé (MCC: {self.state.best_val_mcc:.4f}, F1: {self.state.best_val_f1:.4f})")
                else:
                    self.state.patience_counter += 1
                    print(f"\n⏳ Patience: {self.state.patience_counter}/{patience}")
                    print(f"   Current MCC: {val_metrics['mcc']:.4f} | Best MCC: {self.state.best_val_mcc:.4f}")

                # Sauvegarde checkpoint initial (epoch 0)
                if epoch == 0 and improvement <= min_delta:
                    try:
                        self.checkpoint_manager.save_checkpoint(
                            model=self.model,
                            optimizer=self.optimizer,
                            scheduler=self.scheduler,
                            epoch=epoch,
                            global_step=self.state.global_step,
                            best_val_mcc=self.state.best_val_mcc,
                            best_val_f1=self.state.best_val_f1,
                            history=self.state.get_history(),
                            feature_stats=self.feature_stats,
                            config=self.config.to_dict(),
                            is_best=False
                        )
                        print(f"💾 Checkpoint initial epoch {epoch + 1} sauvegardé")
                    except Exception as e:
                        print(f"⚠️  Erreur sauvegarde checkpoint initial: {e}")

                # Sauvegarde périodique
                if (epoch + 1) % save_every == 0:
                    self.checkpoint_manager.save_checkpoint(
                        model=self.model,
                        optimizer=self.optimizer,
                        scheduler=self.scheduler,
                        epoch=epoch,
                        global_step=self.state.global_step,
                        best_val_mcc=self.state.best_val_mcc,
                        best_val_f1=self.state.best_val_f1,
                        history=self.state.get_history(),
                        feature_stats=self.feature_stats,
                        config=self.config.to_dict(),
                        is_best=False
                    )
                    print(f"💾 Checkpoint epoch {epoch + 1} sauvegardé")

                # Arrêt si patience dépassée
                if self.state.patience_counter >= patience:
                    print(f"\n⚠️  Early stopping déclenché (patience: {patience})")
                    print(f"   Meilleur MCC: {self.state.best_val_mcc:.4f} (epoch {epoch + 1 - patience})")
                    break

            elapsed_time = time.time() - start_time
            print(f"\n{'='*80}")
            print(f"✅ Entraînement terminé en {self._format_time(elapsed_time)}")
            print(f"{'='*80}")
            print(f"\n🏆 Meilleurs résultats:")
            print(f"   MCC:              {self.state.best_val_mcc:.4f}")
            print(f"   F1:               {self.state.best_val_f1:.4f}")
            print(f"   Balanced Acc:     {self.state.val_balanced_accuracy[np.argmax(self.state.val_mcc)]:.4f}")
            print(f"   Cohen's Kappa:    {self.state.val_cohen_kappa[np.argmax(self.state.val_mcc)]:.4f}")
            print(f"{'='*80}\n")

        except KeyboardInterrupt:
            print("\n\n⚠️  Entraînement interrompu")
            print(f"   Meilleur MCC jusqu'ici: {self.state.best_val_mcc:.4f}")

        finally:
            self.writer.close()

        return self.state.get_history()

    
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