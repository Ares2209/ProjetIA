"""Module pour la logique d'entraînement."""

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
from typing import Dict, Optional, Any
from dataclasses import dataclass, field
import time

from .metrique import MetricsCalculator, MetricsAccumulator
from .Loss import BCE
from .checkpoint import CheckpointManager
from .config import Config

# Seuil pour considérer que le MCC est "proche" du meilleur
# Si le MCC actuel est dans cette marge, on vérifie aussi le composite score
MCC_CLOSE_THRESHOLD = 0.02


@dataclass
class TrainingState:
    """État de l'entraînement."""
    epoch: int = 0
    global_step: int = 0
    best_val_mcc: float = -1.0
    best_val_f1: float = 0.0
    best_val_loss: float = float('inf')
    patience_counter: int = 0
    history: Dict[str, Any] = field(default_factory=lambda: {
        'train_loss': [], 'val_loss': [], 'iteration_losses': [],
        'train_accuracy': [], 'val_accuracy': [],
        'train_balanced_accuracy': [], 'val_balanced_accuracy': [],
        'train_precision': [], 'val_precision': [],
        'train_recall': [], 'val_recall': [],
        'train_specificity': [], 'val_specificity': [],
        'train_f1': [], 'val_f1': [],
        'train_f2': [], 'val_f2': [],
        'train_iou': [], 'val_iou': [],
        'train_mcc': [], 'val_mcc': [],
        'train_cohen_kappa': [], 'val_cohen_kappa': [],
        'train_class_0_precision': [], 'val_class_0_precision': [],
        'train_class_0_recall': [], 'val_class_0_recall': [],
        'train_class_1_precision': [], 'val_class_1_precision': [],
        'train_class_1_recall': [], 'val_class_1_recall': [],
        'train_support_class_0': [], 'val_support_class_0': [],
        'train_support_class_1': [], 'val_support_class_1': [],
        'train_auroc': [], 'val_auroc': [],
        'train_auprc': [], 'val_auprc': [],
        'train_brier_score': [], 'val_brier_score': [],
        'train_tp': [], 'val_tp': [],
        'train_fp': [], 'val_fp': [],
        'train_tn': [], 'val_tn': [],
        'train_fn': [], 'val_fn': [],
        # Métriques composites
        'train_composite_score': [], 'val_composite_score': [],
        'train_g_mean': [], 'val_g_mean': [],
        'val_min_class_recall': [],
        'val_class_balance_gap': [],
        'val_stability_score': [],
        'val_production_score': [],
        'val_f_harmonic': [],
    })

    def __getattr__(self, name: str):
        """Accès transparent aux listes de l'historique via attributs."""
        # Évite la récursion infinie lors de l'init
        if name == 'history':
            raise AttributeError(name)
        history = object.__getattribute__(self, 'history')
        if name in history:
            return history[name]
        raise AttributeError(f"'TrainingState' has no attribute '{name}'")

    def append_epoch(self, prefix: str, metrics: Dict[str, float]) -> None:
        """Ajoute les métriques d'une epoch à l'historique pour un préfixe donné (train/val)."""
        mapping = {
            'loss': f'{prefix}_loss',
            'accuracy': f'{prefix}_accuracy',
            'balanced_accuracy': f'{prefix}_balanced_accuracy',
            'precision': f'{prefix}_precision',
            'recall': f'{prefix}_recall',
            'specificity': f'{prefix}_specificity',
            'f1': f'{prefix}_f1',
            'f2': f'{prefix}_f2',
            'iou': f'{prefix}_iou',
            'mcc': f'{prefix}_mcc',
            'cohen_kappa': f'{prefix}_cohen_kappa',
            'class_0_precision': f'{prefix}_class_0_precision',
            'class_0_recall': f'{prefix}_class_0_recall',
            'class_1_precision': f'{prefix}_class_1_precision',
            'class_1_recall': f'{prefix}_class_1_recall',
            'support_class_0': f'{prefix}_support_class_0',
            'support_class_1': f'{prefix}_support_class_1',
            'auroc': f'{prefix}_auroc',
            'auprc': f'{prefix}_auprc',
            'brier_score': f'{prefix}_brier_score',
            'tp': f'{prefix}_tp',
            'fp': f'{prefix}_fp',
            'tn': f'{prefix}_tn',
            'fn': f'{prefix}_fn',
            'composite_score': f'{prefix}_composite_score',
            'g_mean': f'{prefix}_g_mean',
        }
        for src_key, hist_key in mapping.items():
            if src_key in metrics and hist_key in self.history:
                self.history[hist_key].append(metrics[src_key])

        # Métriques val-only
        if prefix == 'val':
            for key in ('min_class_recall', 'class_balance_gap', 'stability_score',
                        'production_score', 'f_harmonic'):
                if key in metrics:
                    self.history[f'val_{key}'].append(metrics[key])

    def get_history(self) -> Dict[str, Any]:
        """Retourne une shallow copy de l'historique."""
        return {k: list(v) if isinstance(v, list) else v for k, v in self.history.items()}


class Trainer:
    """Gestionnaire d'entraînement pour le modèle de visibilité."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        config: Config,
    ):
        self.device = config.device
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config

        self.metrics_calculator = MetricsCalculator()
        self.criterion = self._setup_criterion()
        self.optimizer = self._setup_optimizer()
        self.scheduler = self._setup_scheduler()
        self.writer = SummaryWriter(config.paths.experiment_name)
        self.checkpoint_manager = CheckpointManager(
            model_folder=config.paths.model_folder,
            model_basename=config.paths.model_basename,
            keep_last_n=config.training.keep_last_n_checkpoints,
        )
        self.state = TrainingState()

        try:
            self.feature_stats = self._compute_feature_stats()
            print(f"   • Feature stats computed: mean shape {np.array(self.feature_stats['mean']).shape}")
        except Exception as e:
            print(f"     Could not compute feature stats: {e}")
            self.feature_stats = None

        if config.training.preload:
            self._load_pretrained(config.training.preload)

        print(f"\nTrainer initialisé:")
        print(f"   • Device: {self.device}")

    # =========================================================================
    # SETUP
    # =========================================================================

    def _setup_criterion(self) -> nn.Module:
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
        return AdamW(
            self.model.parameters(),
            lr=self.config.training.learning_rate,
            betas=self.config.training.optimizer_betas,
            eps=self.config.training.optimizer_eps,
            weight_decay=self.config.training.weight_decay,
        )

    def _setup_scheduler(self) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
        total_steps = len(self.train_loader) * self.config.training.num_epochs
        return OneCycleLR(
            self.optimizer,
            max_lr=self.config.training.learning_rate,
            total_steps=total_steps,
            pct_start=self.config.training.scheduler_pct_start,
            div_factor=self.config.training.scheduler_div_factor,
            final_div_factor=self.config.training.scheduler_final_div_factor,
        )

    def _load_pretrained(self, checkpoint_path: str):
        print(f"\n Chargement du checkpoint: {checkpoint_path}")
        checkpoint = self.checkpoint_manager.load_checkpoint(
            checkpoint_name=checkpoint_path,
            device=str(self.device),
        )
        self.model.load_state_dict(checkpoint['model_state_dict'])
        if 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint and self.scheduler:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.state.epoch = checkpoint.get('epoch', 0)
        self.state.global_step = checkpoint.get('global_step', 0)
        self.state.best_val_f1 = checkpoint.get('best_val_f1', 0.0)
        print(f" Checkpoint chargé (epoch {self.state.epoch}, best F1: {self.state.best_val_f1:.4f})")

    # =========================================================================
    # MÉTRIQUES — méthodes factorisées
    # =========================================================================

    def _compute_composite_metrics(self, epoch_metrics) -> Dict[str, float]:
        """Calcule toutes les métriques composites à partir d'un objet epoch_metrics.
        Factorisé pour éviter la duplication entre train_epoch() et validate().
        """
        g_mean = np.sqrt(max(epoch_metrics.recall * epoch_metrics.specificity, 0.0))
        min_class_recall = min(epoch_metrics.class_0_recall, epoch_metrics.class_1_recall)
        class_balance_gap = abs(epoch_metrics.class_0_recall - epoch_metrics.class_1_recall)
        stability_score = (epoch_metrics.mcc + epoch_metrics.cohen_kappa) / 2.0
        production_score = (
            0.3 * epoch_metrics.precision
            + 0.4 * epoch_metrics.recall
            + 0.3 * epoch_metrics.specificity
        )
        f_harmonic = (
            2 * (epoch_metrics.f1 * epoch_metrics.f2) / (epoch_metrics.f1 + epoch_metrics.f2)
            if epoch_metrics.f1 > 0 and epoch_metrics.f2 > 0
            else 0.0
        )
        composite_score = (
            0.25 * epoch_metrics.mcc
            + 0.20 * epoch_metrics.balanced_accuracy
            + 0.20 * epoch_metrics.f1
            + 0.20 * g_mean
            + 0.15 * epoch_metrics.cohen_kappa
        )
        probabilistic_score = 0.0
        if epoch_metrics.auroc is not None and epoch_metrics.auroc > 0:
            probabilistic_score = (
                0.50 * epoch_metrics.auroc
                + 0.30 * epoch_metrics.auprc
                + 0.20 * (1.0 - epoch_metrics.brier_score)
            )

        return {
            'g_mean': g_mean,
            'min_class_recall': min_class_recall,
            'class_balance_gap': class_balance_gap,
            'stability_score': stability_score,
            'production_score': production_score,
            'f_harmonic': f_harmonic,
            'composite_score': composite_score,
            'probabilistic_score': probabilistic_score,
        }

    def _build_epoch_metrics(self, avg_loss: float, epoch_metrics, composite: Dict) -> Dict[str, float]:
        """Construit le dictionnaire complet de métriques pour une epoch.
        Factorisé pour éviter la duplication entre train_epoch() et validate().
        """
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
            **composite,
        }

    # =========================================================================
    # TRAIN / VALIDATE
    # =========================================================================

    def _unpack_batch(self, batch):
        """Dépaquète un batch selon son format (avec ou sans ids)."""
        if len(batch) == 4:
            spectra, auxiliary, labels, _ = batch
        else:
            spectra, auxiliary, labels = batch
        return (
            spectra.to(self.device),
            auxiliary.to(self.device),
            labels.to(self.device),
        )

    def train_epoch(self) -> Dict[str, float]:
        """Effectue une epoch d'entraînement."""
        self.model.train()
        total_loss = 0.0
        metrics_accum = MetricsAccumulator(threshold=self.config.training.classification_threshold)
        log_every = max(1, len(self.train_loader) // 10)

        pbar = tqdm(self.train_loader, desc=f"Epoch {self.state.epoch + 1}", leave=False)
        for batch_idx, batch in enumerate(pbar):
            spectra, auxiliary, labels = self._unpack_batch(batch)

            self.optimizer.zero_grad()
            predictions = self.model(spectra, auxiliary)
            loss = self.criterion(predictions, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.training.max_grad_norm)
            self.optimizer.step()
            if self.scheduler:
                self.scheduler.step()

            batch_loss = loss.item()
            total_loss += batch_loss
            self.state.history['iteration_losses'].append(batch_loss)

            with torch.no_grad():
                class_idx = min(1, predictions.shape[1] - 1)
                preds_cls = predictions[:, class_idx]
                labels_cls = labels[:, class_idx]
                metrics_accum.update(preds_cls, labels_cls, store_for_probabilistic=True)
                current_metrics = self.metrics_calculator.compute_metrics(preds_cls, labels_cls)

            if (batch_idx + 1) % log_every == 0:
                self.writer.add_scalar('Train/Loss_iter', batch_loss, self.state.global_step)
                self.writer.add_scalar('Train/LR', self._get_lr(), self.state.global_step)
                self.writer.add_scalar('Train/MCC_iter', current_metrics.mcc, self.state.global_step)
                self.writer.add_scalar('Train/BalancedAcc_iter', current_metrics.balanced_accuracy, self.state.global_step)

            self.state.global_step += 1
            pbar.set_postfix({
                'loss': f'{batch_loss:.4f}',
                'mcc': f'{current_metrics.mcc:.4f}',
                'bal_acc': f'{current_metrics.balanced_accuracy:.4f}',
                'f1': f'{current_metrics.f1:.4f}',
                'lr': f'{self._get_lr():.2e}',
            })

        avg_loss = total_loss / len(self.train_loader)
        epoch_metrics = metrics_accum.compute(compute_probabilistic=True)
        composite = self._compute_composite_metrics(epoch_metrics)
        return self._build_epoch_metrics(avg_loss, epoch_metrics, composite)

    def validate(self) -> Dict[str, float]:
        """Évalue le modèle sur le set de validation."""
        self.model.eval()
        total_loss = 0.0
        metrics_accum = MetricsAccumulator(threshold=self.config.training.classification_threshold)

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation", leave=False):
                spectra, auxiliary, labels = self._unpack_batch(batch)
                predictions = self.model(spectra, auxiliary)
                loss = self.criterion(predictions, labels)
                total_loss += loss.item()
                class_idx = min(1, predictions.shape[1] - 1)
                metrics_accum.update(
                    predictions[:, class_idx],
                    labels[:, class_idx],
                    store_for_probabilistic=True,
                )

        avg_loss = total_loss / len(self.val_loader)
        epoch_metrics = metrics_accum.compute(compute_probabilistic=True)
        composite = self._compute_composite_metrics(epoch_metrics)
        return self._build_epoch_metrics(avg_loss, epoch_metrics, composite)

    # =========================================================================
    # FEATURE STATS
    # =========================================================================

    def _compute_feature_stats(self) -> Dict[str, list]:
        """Calcule mean/std des features sur le train set via l'algorithme de Chan.

        Avantages par rapport à la méthode sum/sumsq :
          - Stable numériquement : pas de soustraction de deux grands nombres
            (évite la perte de précision sur les grands datasets)
          - Accumulation directe en torch.float64 sur CPU → pas de conversion
            numpy par batch, une seule à la fin via .tolist()

        Algorithme de Chan (réduction parallèle en un seul passage) :
            new_n    = n_a + n_b
            delta    = mean_b - mean_a
            new_mean = (n_a * mean_a + n_b * mean_b) / new_n
            new_M2   = M2_a + M2_b + delta² * n_a * n_b / new_n
            std      = sqrt(new_M2 / new_n)          ← variance de population
        """
        total_n: int = 0
        mean: Optional[torch.Tensor] = None
        M2:   Optional[torch.Tensor] = None   # somme des carrés des écarts (Welford)

        with torch.no_grad():
            for batch in tqdm(self.train_loader, desc="Computing feature stats", leave=False):
                if len(batch) == 4:
                    spectra, auxiliary, _, _ = batch
                else:
                    spectra, auxiliary, _ = batch

                # Tout en float64 sur CPU pour la précision numérique
                spectra_flat = spectra.reshape(spectra.shape[0], -1).cpu().double()  # (B, C*L)
                features     = torch.cat([spectra_flat, auxiliary.cpu().double()], dim=1)  # (B, D)

                b_n    = features.shape[0]
                b_mean = features.mean(dim=0)                        # (D,)
                b_M2   = ((features - b_mean) ** 2).sum(dim=0)      # (D,)

                if mean is None:
                    total_n, mean, M2 = b_n, b_mean, b_M2
                else:
                    # Fusion des deux groupes (algorithme de Chan)
                    new_n   = total_n + b_n
                    delta   = b_mean - mean
                    mean    = (total_n * mean + b_n * b_mean) / new_n
                    M2      = M2 + b_M2 + delta ** 2 * (total_n * b_n / new_n)
                    total_n = new_n

        if total_n == 0:
            raise RuntimeError("No valid samples found in train_loader to compute feature stats")

        std = torch.sqrt(torch.clamp(M2 / total_n, min=1e-8))
        return {'mean': mean.tolist(), 'std': std.tolist()}

    # =========================================================================
    # CHECKPOINT HELPER
    # =========================================================================

    def _build_checkpoint_history(self) -> Dict:
        """Construit le dictionnaire d'historique complet pour les checkpoints."""
        return self.state.get_history()

    def _save_checkpoint(self, is_best: bool):
        """Sauvegarde un checkpoint."""
        self.checkpoint_manager.save_checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            epoch=self.state.epoch,
            global_step=self.state.global_step,
            best_val_mcc=self.state.best_val_mcc,
            best_val_f1=self.state.best_val_f1,
            history=self._build_checkpoint_history(),
            feature_stats=self.feature_stats,
            config=self.config.to_dict(),
            is_best=is_best,
        )

    # =========================================================================
    # BOUCLE D'ENTRAÎNEMENT PRINCIPALE
    # =========================================================================

    def train(self) -> Dict:
        """Boucle d'entraînement complète."""
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
            'f_harmonic': 0.0,
        }

        print(f"\n{'='*90}")
        print(f"🚀 DÉBUT DE L'ENTRAÎNEMENT")
        print(f"{'='*90}")
        print(f"  Epochs:              {num_epochs}")
        print(f"  Patience:            {patience}  |  Min Delta: {min_delta}")
        print(f"  MCC Close Threshold: {MCC_CLOSE_THRESHOLD} (composite score vérifié dans cette marge)")
        print(f"  Device:              {self.device}")
        print(f"  Batch Size:          {self.config.training.batch_size}")
        print(f"  Learning Rate:       {self.config.training.learning_rate}")
        print(f"  Weight Decay:        {self.config.training.weight_decay}")
        print(f"  Optimizer:           {self.optimizer.__class__.__name__}")
        print(f"  Scheduler:           {self.scheduler.__class__.__name__ if self.scheduler else 'None'}")
        print(f"  Criterion:           {self.criterion.__class__.__name__}")
        print(f"  Train Batches:       {len(self.train_loader)}  |  Val Batches: {len(self.val_loader)}")
        print(f"{'='*90}\n")

        epoch = -1  # Protège les blocs except si la boucle ne démarre pas
        try:
            for epoch in range(num_epochs):
                self.state.epoch = epoch
                epoch_start = time.time()

                train_metrics = self.train_epoch()
                val_metrics = self.validate()

                # ── Historique ──────────────────────────────────────────────
                self.state.append_epoch('train', train_metrics)
                self.state.append_epoch('val', val_metrics)

                # ── TensorBoard ─────────────────────────────────────────────
                self._log_tensorboard(epoch, train_metrics, val_metrics)

                # ── Affichage console ────────────────────────────────────────
                self._log_console(epoch, num_epochs, time.time() - epoch_start,
                                  train_metrics, val_metrics, best_metrics, patience)

                # ── Critère d'amélioration ───────────────────────────────────
                mcc_improvement = val_metrics['mcc'] - best_metrics['mcc']
                composite_improvement = val_metrics['composite_score'] - best_metrics['composite_score']

                # Amélioration si :
                #   (a) MCC progresse clairement au-delà de min_delta, OU
                #   (b) MCC est très proche du meilleur ET le composite score progresse
                mcc_close = abs(mcc_improvement) <= MCC_CLOSE_THRESHOLD
                is_improved = (
                    mcc_improvement > min_delta
                    or (mcc_close and composite_improvement > min_delta)
                )
                improvement_reason = (
                    "MCC" if mcc_improvement > min_delta
                    else "Composite Score (MCC proche du meilleur)" if is_improved
                    else None
                )

                if is_improved:
                    self._on_improvement(val_metrics, train_metrics, best_metrics,
                                         mcc_improvement, composite_improvement,
                                         improvement_reason)
                else:
                    self.state.patience_counter += 1
                    self._log_no_improvement(val_metrics, best_metrics, mcc_improvement,
                                              composite_improvement, min_delta, patience)

                # ── Sauvegarde epoch 0 si pas d'amélioration ────────────────
                if epoch == 0 and not is_improved:
                    try:
                        self._save_checkpoint(is_best=False)
                        print(f" Checkpoint initial epoch 1 sauvegardé\n")
                    except Exception as e:
                        print(f"  Erreur sauvegarde checkpoint initial: {e}\n")

                # ── Sauvegarde périodique ────────────────────────────────────
                if (epoch + 1) % save_every == 0:
                    try:
                        self._save_checkpoint(is_best=False)
                        print(f"💾 Checkpoint périodique epoch {epoch + 1} sauvegardé\n")
                    except Exception as e:
                        print(f"⚠️  Erreur sauvegarde checkpoint périodique: {e}\n")

                # ── Early stopping ───────────────────────────────────────────
                if self.state.patience_counter >= patience:
                    self._log_early_stopping(epoch, patience, best_metrics)
                    break

            # ── Résumé final ─────────────────────────────────────────────────
            elapsed_time = time.time() - start_time
            self._log_final_summary(epoch, num_epochs, elapsed_time, patience, best_metrics)
            self.writer.close()

            return {
                'best_metrics': best_metrics,
                'best_epoch': int(np.argmax(self.state.history['val_composite_score'])) + 1,
                'final_epoch': epoch + 1,
                'training_time': elapsed_time,
                'early_stopped': self.state.patience_counter >= patience,
                'history': self.state.get_history(),
            }

        except KeyboardInterrupt:
            print(f"\n\n  Entraînement interrompu — epoch {epoch + 1}, step {self.state.global_step}")
            self.writer.close()
            raise

        except Exception as e:
            import traceback
            print(f"\n\n ERREUR PENDANT L'ENTRAÎNEMENT — epoch {epoch + 1}")
            print(f"   Erreur: {e}")
            traceback.print_exc()
            self.writer.close()
            raise

    # =========================================================================
    # CALLBACKS D'AMÉLIORATION
    # =========================================================================

    def _on_improvement(self, val_metrics, train_metrics, best_metrics,
                        mcc_improvement, composite_improvement, reason):
        """Appelé quand une amélioration est détectée."""
        best_metrics.update({
            'composite_score': val_metrics['composite_score'],
            'mcc': val_metrics['mcc'],
            'g_mean': val_metrics['g_mean'],
            'stability_score': val_metrics['stability_score'],
            'production_score': val_metrics['production_score'],
            'f_harmonic': val_metrics['f_harmonic'],
        })
        if val_metrics['auroc'] > 0:
            best_metrics['auroc'] = val_metrics['auroc']

        self.state.best_val_mcc = val_metrics['mcc']
        self.state.best_val_f1 = val_metrics['f1']
        self.state.patience_counter = 0

        print(f"\n{'='*90}")
        print(f" AMÉLIORATION DÉTECTÉE! ({reason})")
        print(f"{'='*90}")
        print(f"  MCC:              {val_metrics['mcc']:.4f}  (Δ {mcc_improvement:+.4f})")
        print(f"  Composite Score:  {val_metrics['composite_score']:.4f}  (Δ {composite_improvement:+.4f})")
        print(f"  G-Mean:           {val_metrics['g_mean']:.4f}")
        print(f"  Stability Score:  {val_metrics['stability_score']:.4f}")
        print(f"  Production Score: {val_metrics['production_score']:.4f}")
        print(f"  F-Harmonic:       {val_metrics['f_harmonic']:.4f}")

        min_recall_threshold = 0.50
        ok = val_metrics['min_class_recall'] >= min_recall_threshold
        symbol = "✓" if ok else "⚠"
        print(f"  {symbol} Min Class Recall: {val_metrics['min_class_recall']:.4f} "
              f"({'≥' if ok else '<'} {min_recall_threshold})")

        try:
            self._save_checkpoint(is_best=True)
            print(f"\n💾 Meilleur modèle sauvegardé!")
            print(f"{'='*90}\n")
        except Exception as e:
            print(f"\n❌ Erreur lors de la sauvegarde: {e}")
            print(f"{'='*90}\n")

    # =========================================================================
    # LOGGING
    # =========================================================================

    def _log_tensorboard(self, epoch: int, train: Dict, val: Dict):
        paired = [
            ('Loss', 'loss'), ('MCC', 'mcc'), ('Balanced_Accuracy', 'balanced_accuracy'),
            ('F1_Score', 'f1'), ('F2_Score', 'f2'), ('Precision', 'precision'),
            ('Recall', 'recall'), ('Specificity', 'specificity'),
            ('Cohen_Kappa', 'cohen_kappa'), ('IoU', 'iou'),
            ('Class_0_Precision', 'class_0_precision'), ('Class_1_Precision', 'class_1_precision'),
            ('Class_0_Recall', 'class_0_recall'), ('Class_1_Recall', 'class_1_recall'),
            ('AUROC', 'auroc'), ('AUPRC', 'auprc'), ('Brier_Score', 'brier_score'),
            ('Composite_Score', 'composite_score'), ('G_Mean', 'g_mean'),
        ]
        for tag, key in paired:
            self.writer.add_scalars(tag, {'train': train[key], 'val': val[key]}, epoch)

        val_only = [
            'min_class_recall', 'class_balance_gap', 'stability_score',
            'production_score', 'f_harmonic', 'probabilistic_score',
        ]
        for key in val_only:
            v = val.get(key, 0.0)
            if v:
                self.writer.add_scalar(f'Val/{key.title()}', v, epoch)

        self.writer.add_scalar('Learning_Rate', self._get_lr(), epoch)

    def _log_console(self, epoch, num_epochs, epoch_time, train, val, best, patience):
        print(f"\n{'='*90}")
        print(f" Epoch {epoch + 1}/{num_epochs} — {epoch_time:.2f}s")
        print(f"{'='*90}")

        def row(label, key):
            print(f"  {label:<22} Train: {train[key]:.4f} | Val: {val[key]:.4f}")

        print(f"\n MÉTRIQUES DE BASE:")
        for lbl, key in [("Loss", "loss"), ("Accuracy", "accuracy"),
                          ("Balanced Acc", "balanced_accuracy"), ("MCC", "mcc"),
                          ("Cohen's Kappa", "cohen_kappa")]:
            row(lbl, key)

        print(f"\n F-SCORES & IoU / PRECISION-RECALL:")
        for lbl, key in [("F1", "f1"), ("F2", "f2"), ("IoU", "iou"),
                          ("Precision", "precision"), ("Recall", "recall"),
                          ("Specificity", "specificity")]:
            row(lbl, key)

        print(f"\n MÉTRIQUES COMPOSITES:")
        for lbl, key in [("Composite Score", "composite_score"), ("G-Mean", "g_mean"),
                          ("Stability Score", "stability_score"),
                          ("Production Score", "production_score"), ("F-Harmonic", "f_harmonic")]:
            row(lbl, key)

        print(f"\n ÉQUILIBRE ENTRE CLASSES (Val):")
        print(f"  Min Class Recall:      {val['min_class_recall']:.4f}")
        print(f"  Class Balance Gap:     {val['class_balance_gap']:.4f} (↓ = mieux)")
        print(f"  Classe 0 — Precision:  {val['class_0_precision']:.4f}  Recall: {val['class_0_recall']:.4f}")
        print(f"  Classe 1 — Precision:  {val['class_1_precision']:.4f}  Recall: {val['class_1_recall']:.4f}")

        if val['auroc'] > 0:
            print(f"\n MÉTRIQUES PROBABILISTES:")
            for lbl, key in [("AUROC", "auroc"), ("AUPRC", "auprc"), ("Brier Score", "brier_score")]:
                row(lbl, key)
            if val['probabilistic_score'] > 0:
                print(f"  {'Probabilistic Sc':<22}       Val: {val['probabilistic_score']:.4f}")

        print(f"\n SUPPORT / HYPERPARAMÈTRES:")
        print(f"  Classe 0:  Train {train['support_class_0']} | Val {val['support_class_0']}")
        print(f"  Classe 1:  Train {train['support_class_1']} | Val {val['support_class_1']}")
        print(f"  LR: {self._get_lr():.2e}  |  Patience: {self.state.patience_counter}/{patience}")

    def _log_no_improvement(self, val, best, mcc_imp, comp_imp, min_delta, patience):
        print(f"\n{'='*90}")
        print(f" PATIENCE: {self.state.patience_counter}/{patience}")
        print(f"{'='*90}")
        print(f"  MCC:              {val['mcc']:.4f} (best: {best['mcc']:.4f}, Δ {mcc_imp:+.4f})")
        print(f"  Composite Score:  {val['composite_score']:.4f} (best: {best['composite_score']:.4f}, Δ {comp_imp:+.4f})")
        print(f"  Amélioration requise: > {min_delta:.4f} sur MCC ou Composite (si MCC dans ±{MCC_CLOSE_THRESHOLD})")
        print(f"  G-Mean:           {val['g_mean']:.4f} (best: {best['g_mean']:.4f})")
        print(f"  Stability Score:  {val['stability_score']:.4f} (best: {best['stability_score']:.4f})")
        print(f"  Production Score: {val['production_score']:.4f} (best: {best['production_score']:.4f})")
        print(f"{'='*90}\n")

    def _log_early_stopping(self, epoch, patience, best):
        print(f"\n{'='*90}")
        print(f"  EARLY STOPPING — patience épuisée ({patience} epochs)")
        print(f"  Meilleur epoch estimé: {epoch + 1 - patience}")
        print(f"  MCC: {best['mcc']:.4f}  |  Composite: {best['composite_score']:.4f}")
        print(f"  G-Mean: {best['g_mean']:.4f}  |  F-Harmonic: {best['f_harmonic']:.4f}")
        if best['auroc'] > 0:
            print(f"  AUROC: {best['auroc']:.4f}")
        print(f"{'='*90}\n")

    def _log_final_summary(self, epoch, num_epochs, elapsed, patience, best):
        hist = self.state.history
        best_idx = int(np.argmax(hist['val_composite_score'])) if hist['val_composite_score'] else 0

        print(f"\n{'='*90}")
        print(f" ENTRAÎNEMENT TERMINÉ — {self._format_time(elapsed)}")
        print(f"  Epochs: {epoch + 1}/{num_epochs}  |  Steps: {self.state.global_step}")
        print(f"  Early stopping: {'Oui' if self.state.patience_counter >= patience else 'Non'}")
        print(f"\n MEILLEURS RÉSULTATS (Epoch {best_idx + 1}):")
        print(f"  Composite: {best['composite_score']:.4f}  |  MCC: {best['mcc']:.4f}")
        print(f"  G-Mean: {best['g_mean']:.4f}  |  F-Harmonic: {best['f_harmonic']:.4f}")
        print(f"  Stability: {best['stability_score']:.4f}  |  Production: {best['production_score']:.4f}")
        if hist.get('val_f1'):
            print(f"  F1: {hist['val_f1'][best_idx]:.4f}  |  Balanced Acc: {hist['val_balanced_accuracy'][best_idx]:.4f}")
        if best['auroc'] > 0:
            print(f"  AUROC: {best['auroc']:.4f}")
        print(f"{'='*90}\n")

    def _get_lr(self) -> float:
        return self.optimizer.param_groups[0]['lr']

    def _format_time(self, seconds: float) -> str:
        h, rem = divmod(int(seconds), 3600)
        m, s = divmod(rem, 60)
        return f"{h}h {m}min {s}s" if h else (f"{m}min {s}s" if m else f"{s}s")