#!/usr/bin/env python3
"""Script d'entraînement principal.

Usage: python main.py [--config PATH]
"""
import argparse
import warnings
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from training.LightGBM import LightGBM
from training.config import Config, get_config_object
from training.training import Trainer
from training.TrainingGBM import LightGBMTrainer
from training.utils import set_seed
from models.dataset import ExoplanetDataset, collate_fn
from sklearn.model_selection import train_test_split
from models.CNN import CNN
from models.ResNetCNN import ResNet1D, resnet18_1d, resnet34_1d, resnet8_1d
from torch.utils.data import DataLoader


# ─────────────────────────────────────────────────────────────────────────────
# UTILITAIRES
# ─────────────────────────────────────────────────────────────────────────────

class NumpyEncoder(json.JSONEncoder):
    """Encodeur JSON pour les types NumPy."""
    def default(self, obj):
        if isinstance(obj, np.integer):  return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray):  return obj.tolist()
        return super().default(obj)


# ─────────────────────────────────────────────────────────────────────────────
# DONNÉES
# ─────────────────────────────────────────────────────────────────────────────

def _resolve_path(configured: Path, fallback: Path, label: str) -> Path:
    """Retourne le path configuré s'il existe, sinon le fallback avec un avertissement."""
    if configured.exists():
        return configured
    warnings.warn(
        f"[build_dataloaders] '{label}' introuvable : {configured}\n"
        f"  → Fallback vers : {fallback}",
        UserWarning, stacklevel=3,
    )
    if not fallback.exists():
        raise FileNotFoundError(f"Fallback introuvable : {fallback}")
    return fallback


def build_dataloaders(config) -> tuple[DataLoader, DataLoader, dict]:
    """Construit les DataLoaders train/val avec split stratifié."""
    data_cfg      = config.data
    fallback_root = Path('Défi-IA-2026') / 'DATA' / 'defi-ia-cnes'

    spectra_path  = _resolve_path(Path(data_cfg.spectra_train_path),  fallback_root / 'spectra.npy',   'spectra')
    aux_path      = _resolve_path(Path(data_cfg.auxiliary_train_path), fallback_root / 'auxiliary.csv', 'auxiliary')
    targets_path  = _resolve_path(Path(data_cfg.targets_train_path),   fallback_root / 'targets.csv',   'targets')

    print(f"  spectra   : {spectra_path}")
    print(f"  auxiliary : {aux_path}")
    print(f"  targets   : {targets_path}")

    spectra_all    = np.load(spectra_path)
    aux_df_all     = pd.read_csv(aux_path)
    targets_df_all = pd.read_csv(targets_path)

    strat_labels = targets_df_all['eau'].astype(str) + "_" + targets_df_all['nuage'].astype(str)
    indices      = list(range(len(spectra_all)))
    train_idx, val_idx = train_test_split(
        indices,
        test_size    = 1 - data_cfg.train_ratio,
        random_state = 42,
        stratify     = strat_labels,
    )

    spectra_train = spectra_all[train_idx]
    spectra_val   = spectra_all[val_idx]
    aux_train     = aux_df_all.iloc[train_idx].reset_index(drop=True)
    aux_val       = aux_df_all.iloc[val_idx].reset_index(drop=True)
    targets_train = targets_df_all.iloc[train_idx].reset_index(drop=True)
    targets_val   = targets_df_all.iloc[val_idx].reset_index(drop=True)

    print("\n=== Distribution TRAIN ===")
    print(targets_train['eau'].value_counts(normalize=True).to_string())
    print(targets_train['nuage'].value_counts(normalize=True).to_string())
    print("\n=== Distribution VAL ===")
    print(targets_val['eau'].value_counts(normalize=True).to_string())
    print(targets_val['nuage'].value_counts(normalize=True).to_string())

    train_dataset = ExoplanetDataset(
        spectra              = spectra_train,
        auxiliary_df         = aux_train,
        targets_df           = targets_train,
        is_train             = True,
        augmentation_factor  = data_cfg.augmentation_factor,
        shift_range          = data_cfg.shift_range,
        scale_range          = data_cfg.scale_range,
        noise_std            = data_cfg.noise_std,
        flip_prob            = data_cfg.flip_prob,
        channel_dropout_prob = data_cfg.channel_dropout_prob,
    )
    val_dataset = ExoplanetDataset(
        spectra             = spectra_val,
        auxiliary_df        = aux_val,
        targets_df          = targets_val,
        is_train            = False,
        augmentation_factor = 0,
        aux_mean            = train_dataset.aux_mean,
        aux_std             = train_dataset.aux_std,
    )

    loader_kwargs = dict(
        batch_size  = config.training.batch_size,
        num_workers = data_cfg.num_workers,
        pin_memory  = data_cfg.pin_memory,
        collate_fn  = collate_fn,
    )
    train_loader = DataLoader(train_dataset, shuffle=True,  **loader_kwargs)
    val_loader   = DataLoader(val_dataset,   shuffle=False, **loader_kwargs)

    dataset_info = {
        'spectrum_length': train_dataset.spectra.shape[1],
        'auxiliary_dim':   train_dataset.aux_features.shape[1],
    }
    del spectra_all
    return train_loader, val_loader, dataset_info


def build_model(cfg, spectrum_length: int, auxiliary_dim: int):
    """Instancie le modèle selon la config."""
    arch = cfg.model.architecture

    # Paramètres communs à tous les modèles PyTorch
    common_torch = dict(
        spectrum_length = spectrum_length,
        auxiliary_dim   = auxiliary_dim,
        num_classes     = cfg.training.num_classes,
        input_channels  = 3,
        dropout         = cfg.model.dropout,
    )

    if arch == "CNN":
        print("  Architecture : CNN")
        return CNN(
            **common_torch,
            conv_channels  = cfg.model.conv_channels,
            kernel_sizes   = cfg.model.kernel_sizes,
            pool_sizes     = cfg.model.pool_sizes,
            fc_dims        = cfg.model.fc_dims,
            use_batch_norm = cfg.model.use_batch_norm,
        )

    elif arch == "ResNet8":
        print("  Architecture : ResNet-8")
        return resnet8_1d(**common_torch)

    elif arch == "ResNet18":
        print("  Architecture : ResNet-18")
        return resnet18_1d(**common_torch)

    elif arch == "ResNet34":
        print("  Architecture : ResNet-34 (⚠ risque surapprentissage)")
        return resnet34_1d(**common_torch)

    elif arch == "LightGBM":
        # ── LightGBM : pas de dropout, pas de batch_norm, paramètres dédiés ──
        print("  Architecture : LightGBM")
        return LightGBM(
            spectrum_length        = spectrum_length,
            auxiliary_dim          = auxiliary_dim,
            num_classes            = cfg.training.num_classes,
            input_channels         = 3,
            # Hyperparamètres boosting (lus depuis cfg.model s'ils existent)
            n_estimators           = getattr(cfg.model, 'n_estimators',    500),
            learning_rate          = getattr(cfg.model, 'lgbm_lr',         0.05),
            num_leaves             = getattr(cfg.model, 'num_leaves',       63),
            max_depth              = getattr(cfg.model, 'max_depth',        -1),
            min_child_samples      = getattr(cfg.model, 'min_child_samples', 20),
            subsample              = getattr(cfg.model, 'subsample',        0.8),
            colsample_bytree       = getattr(cfg.model, 'colsample_bytree', 0.8),
            reg_alpha              = getattr(cfg.model, 'reg_alpha',        0.1),
            reg_lambda             = getattr(cfg.model, 'reg_lambda',       0.1),
            # Feature engineering
            use_pca                = getattr(cfg.model, 'use_pca',               False),
            pca_components         = getattr(cfg.model, 'pca_components',        50),
            use_statistical_features = getattr(cfg.model, 'use_statistical_features', True),
            use_diff_features      = getattr(cfg.model, 'use_diff_features',     True),
            random_state           = cfg.data.random_seed,
        )

    else:
        raise ValueError(
            f"Architecture inconnue : '{arch}'. "
            f"Valeurs valides : CNN | ResNet8 | ResNet18 | ResNet34 | LightGBM"
        )


def _select_trainer(model, train_loader, val_loader, cfg):
    """
    Retourne le bon Trainer selon le type de modèle.

    - LightGBM → LightGBMTrainer  (pas de backprop, pas de scheduler)
    - Tout modèle PyTorch nn.Module → Trainer classique
    """
    if isinstance(model, LightGBM):
        print("  Trainer : LightGBMTrainer")
        return LightGBMTrainer(model, train_loader, val_loader, cfg)
    else:
        print("  Trainer : Trainer (PyTorch)")
        return Trainer(model, train_loader, val_loader, cfg)


# ─────────────────────────────────────────────────────────────────────────────
# GRAPHIQUES — helper commun
# ─────────────────────────────────────────────────────────────────────────────

def _plot_pair(ax, epochs, train_vals, val_vals, title: str, ylabel: str,
               ylim=None, train_color='#3498db', val_color='#e74c3c',
               ref_lines: list = None):
    """Trace une courbe train/val sur un axe matplotlib."""
    if train_vals:
        ax.plot(epochs, train_vals, label='Train',      linewidth=2,
                marker='o', markersize=5, color=train_color)
    ax.plot(epochs, val_vals, label='Validation', linewidth=2,
            marker='s', markersize=5, color=val_color)
    if ref_lines:
        for y, color, ls, lbl in ref_lines:
            ax.axhline(y=y, color=color, linestyle=ls, alpha=0.6, label=lbl)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    if ylim:
        ax.set_ylim(ylim)


def _get(history: dict, key: str) -> list:
    """Retourne history[key] ou une liste vide si absent."""
    return history.get(key, [])


# ─────────────────────────────────────────────────────────────────────────────
# GRAPHIQUES
# ─────────────────────────────────────────────────────────────────────────────

def plot_training_results(history: dict, save_dir: str = 'results'):
    """Crée la figure agrégée et les graphiques individuels."""
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    # LightGBM n'a qu'une seule "epoch" et pas de train_loss
    # → on utilise val_loss comme référence de longueur si train_loss absent
    ref_series = _get(history, 'train_loss') or _get(history, 'val_loss')
    if not ref_series:
        print("  Aucune donnée d'historique à tracer.")
        return

    epochs = range(1, len(ref_series) + 1)

    fig, axes = plt.subplots(4, 3, figsize=(26, 22))
    axs = axes.flatten()

    # 1 — Loss par itération (absent pour LightGBM)
    ax = axs[0]
    iters = _get(history, 'iteration_losses')
    if iters:
        ax.plot(iters, linewidth=1, alpha=0.7, color='#3498db')
        ax.set_title('Loss Train (par itération)', fontsize=13, fontweight='bold')
        ax.set_xlabel('Itération', fontsize=11)
        ax.set_ylabel('Loss', fontsize=11)
        ax.grid(True, alpha=0.3)
    else:
        ax.set_visible(False)

    # 2 — Loss par epoch
    _plot_pair(axs[1], epochs,
               _get(history, 'train_loss'), _get(history, 'val_loss'),
               'Loss', 'Loss')

    # 3 — Accuracy
    if _get(history, 'val_accuracy'):
        _plot_pair(axs[2], epochs,
                   _get(history, 'train_accuracy'), _get(history, 'val_accuracy'),
                   'Accuracy', 'Accuracy', ylim=[0, 1])

    # 4 — Precision
    if _get(history, 'val_precision'):
        _plot_pair(axs[3], epochs,
                   _get(history, 'train_precision'), _get(history, 'val_precision'),
                   'Precision', 'Precision', ylim=[0, 1])

    # 5 — Recall
    if _get(history, 'val_recall'):
        _plot_pair(axs[4], epochs,
                   _get(history, 'train_recall'), _get(history, 'val_recall'),
                   'Recall', 'Recall', ylim=[0, 1])

    # 6 — F1
    if _get(history, 'val_f1'):
        _plot_pair(axs[5], epochs,
                   _get(history, 'train_f1'), _get(history, 'val_f1'),
                   'F1 Score', 'F1', ylim=[0, 1])

    # 7 — Balanced Accuracy
    if _get(history, 'val_balanced_accuracy'):
        _plot_pair(axs[6], epochs,
                   _get(history, 'train_balanced_accuracy'), _get(history, 'val_balanced_accuracy'),
                   'Balanced Accuracy', 'Bal. Acc', ylim=[0, 1])

    # 8 — Specificity
    if _get(history, 'val_specificity'):
        _plot_pair(axs[7], epochs,
                   _get(history, 'train_specificity'), _get(history, 'val_specificity'),
                   'Specificity', 'Specificity', ylim=[0, 1])

    # 9 — AUROC
    if _get(history, 'val_auroc'):
        _plot_pair(axs[8], epochs,
                   _get(history, 'train_auroc'), _get(history, 'val_auroc'),
                   'AUROC', 'AUROC', ylim=[0, 1])

    # 10 — MCC
    if _get(history, 'val_mcc'):
        _plot_pair(axs[9], epochs,
                   _get(history, 'train_mcc'), _get(history, 'val_mcc'),
                   'Matthews Correlation Coefficient', 'MCC', ylim=[-1, 1],
                   ref_lines=[(0, 'gray', '--', 'Aléatoire (0)')])

    # 11 — Composite Score
    if _get(history, 'val_composite_score'):
        _plot_pair(axs[10], epochs,
                   _get(history, 'train_composite_score'), _get(history, 'val_composite_score'),
                   'Composite Score', 'Score')

    # 12 — Bar chart métriques finales
    ax = axs[11]
    metric_keys = [
        ('accuracy', 'Acc'), ('precision', 'Prec'), ('recall', 'Rec'),
        ('f1', 'F1'), ('specificity', 'Spec'), ('balanced_accuracy', 'BalAcc'), ('mcc', 'MCC'),
    ]
    names, train_vals, val_vals = [], [], []
    for key, label in metric_keys:
        vk = f'val_{key}'
        if _get(history, vk):
            names.append(label)
            tk = f'train_{key}'
            train_vals.append(_get(history, tk)[-1] if _get(history, tk) else None)
            val_vals.append(history[vk][-1])
    if names:
        x, w = np.arange(len(names)), 0.35
        if any(v is not None for v in train_vals):
            b1 = ax.bar(x - w/2,
                        [v if v is not None else 0 for v in train_vals],
                        w, label='Train', color='#3498db', edgecolor='black')
            for bar, v in zip(b1, train_vals):
                if v is not None:
                    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                            f'{v:.3f}', ha='center', va='bottom', fontsize=8)
        b2 = ax.bar(x + w/2 if any(v is not None for v in train_vals) else x,
                    val_vals, w, label='Validation', color='#e74c3c', edgecolor='black')
        ax.set_xticks(x)
        ax.set_xticklabels(names, fontsize=10, rotation=15)
        ax.set_title('Métriques Finales (Val)', fontsize=13, fontweight='bold')
        ax.set_ylim([-1, 1.1])
        ax.axhline(0, color='black', linewidth=0.8, alpha=0.5)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        for bar in b2:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., h, f'{h:.3f}',
                    ha='center', va='bottom' if h >= 0 else 'top', fontsize=8)

    plt.tight_layout()
    out = f'{save_dir}/training_history.png'
    plt.savefig(out, dpi=100, bbox_inches='tight')
    plt.close()
    print(f"  Figure agrégée : {out}")

    _create_individual_plots(history, epochs, save_dir)


def _create_individual_plots(history: dict, epochs, save_dir: str):
    """Graphiques individuels détaillés."""

    def _save(filename: str):
        path = f'{save_dir}/{filename}'
        plt.tight_layout()
        plt.savefig(path, dpi=100, bbox_inches='tight')
        plt.close()

    # Precision & Recall combinés
    if _get(history, 'val_precision') and _get(history, 'val_recall'):
        fig, ax = plt.subplots(figsize=(10, 6))
        series = [
            (_get(history, 'train_precision'), 'Train Precision', '#9b59b6', '-'),
            (_get(history, 'val_precision'),   'Val Precision',   '#e67e22', '-'),
            (_get(history, 'train_recall'),    'Train Recall',    '#1abc9c', '--'),
            (_get(history, 'val_recall'),      'Val Recall',      '#e74c3c', '--'),
        ]
        for vals, lbl, color, ls in series:
            if vals:
                ax.plot(epochs, vals, label=lbl, linewidth=2, linestyle=ls, color=color)
        ax.set(title='Precision & Recall', xlabel='Epoch', ylabel='Score', ylim=[0, 1])
        ax.legend(); ax.grid(True, alpha=0.3)
        _save('precision_recall.png')

    # Loss détaillée
    train_loss = _get(history, 'train_loss')
    val_loss   = _get(history, 'val_loss')
    if val_loss:
        fig, ax = plt.subplots(figsize=(10, 6))
        if train_loss:
            ax.plot(epochs, train_loss, label='Train', linewidth=2, marker='o', color='#3498db')
            ax.axhline(min(train_loss), color='#3498db', linestyle=':', alpha=0.5,
                       label=f'Min Train: {min(train_loss):.4f}')
        ax.plot(epochs, val_loss, label='Val', linewidth=2, marker='s', color='#e74c3c')
        ax.axhline(min(val_loss), color='#e74c3c', linestyle=':', alpha=0.5,
                   label=f'Min Val: {min(val_loss):.4f}')
        ax.set(title='Évolution de la Loss', xlabel='Epoch', ylabel='Loss')
        ax.legend(); ax.grid(True, alpha=0.3)
        _save('loss_detailed.png')

    # Heatmap corrélation métriques
    avail = [(k, k.upper() if k == 'mcc' else k.capitalize())
             for k in ['accuracy', 'precision', 'recall', 'f1', 'mcc', 'balanced_accuracy']
             if _get(history, f'val_{k}')]
    if len(avail) >= 2:
        keys, labels = zip(*avail)
        corr = np.corrcoef(np.array([history[f'val_{k}'] for k in keys]))
        fig, ax = plt.subplots(figsize=(9, 7))
        im = ax.imshow(corr, cmap='RdYlGn', vmin=-1, vmax=1)
        plt.colorbar(im, ax=ax, label='Corrélation')
        ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=45)
        ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels)
        for i in range(len(labels)):
            for j in range(len(labels)):
                ax.text(j, i, f'{corr[i,j]:.2f}', ha='center', va='center', fontweight='bold')
        ax.set_title('Corrélation entre Métriques (Validation)', fontsize=13, fontweight='bold')
        _save('metrics_correlation.png')

    # MCC détaillé
    if _get(history, 'val_mcc'):
        val_mcc = [x if x is not None else np.nan for x in history['val_mcc']]
        fig, ax = plt.subplots(figsize=(10, 6))
        if _get(history, 'train_mcc'):
            ax.plot(epochs, [x if x is not None else np.nan for x in history['train_mcc']],
                    label='Train MCC', linewidth=2, marker='o', color='#2980b9')
        ax.plot(epochs, val_mcc, label='Val MCC', linewidth=2, marker='s', color='#c0392b')
        for y, color, ls, lbl in [(0, 'gray', '--', 'Aléatoire'), (1, 'green', ':', 'Parfait')]:
            ax.axhline(y, color=color, linestyle=ls, alpha=0.5, label=lbl)
        if not all(np.isnan(val_mcc)):
            bi = int(np.nanargmax(val_mcc))
            ax.annotate(f'Best Val\n{val_mcc[bi]:.4f}',
                        xy=(list(epochs)[bi], val_mcc[bi]), xytext=(10, -30),
                        textcoords='offset points', fontsize=10, fontweight='bold',
                        arrowprops=dict(arrowstyle='->', color='black'),
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
        ax.set(title='Matthews Correlation Coefficient', xlabel='Epoch',
               ylabel='MCC', ylim=[-1, 1])
        ax.legend(); ax.grid(True, alpha=0.3)
        _save('mcc.png')

    # Confusion matrix (absent pour LightGBM history)
    if all(_get(history, k) for k in ['val_tp', 'val_fp', 'val_tn', 'val_fn']):
        fig, ax = plt.subplots(figsize=(10, 6))
        for key, lbl, color, marker in [
            ('val_tp', 'True Positives',  '#27ae60', 'o'),
            ('val_fp', 'False Positives', '#e67e22', 's'),
            ('val_tn', 'True Negatives',  '#3498db', '^'),
            ('val_fn', 'False Negatives', '#e74c3c', 'v'),
        ]:
            ax.plot(epochs, history[key], label=lbl, linewidth=2, marker=marker)
        ax.set(title='Éléments Matrice de Confusion (Val)', xlabel='Epoch', ylabel='Count')
        ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
        _save('confusion_matrix_elements.png')

    print(f"  Graphiques individuels : {save_dir}/")


# ─────────────────────────────────────────────────────────────────────────────
# SAUVEGARDE CSV + RAPPORT
# ─────────────────────────────────────────────────────────────────────────────

def save_training_results(history: dict, save_dir: str = 'results'):
    """Sauvegarde l'historique complet en CSV."""
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    # Référence de longueur : train_loss si dispo, sinon val_loss (cas LightGBM)
    ref_key = 'train_loss' if _get(history, 'train_loss') else 'val_loss'
    n_epochs = len(_get(history, ref_key))
    if n_epochs == 0:
        print("  Historique vide — pas de CSV à écrire.")
        return

    df_data = {'epoch': range(1, n_epochs + 1)}
    for key, vals in history.items():
        if key == 'iteration_losses':
            continue
        if isinstance(vals, list) and len(vals) == n_epochs:
            df_data[key] = vals

    pd.DataFrame(df_data).to_csv(f'{save_dir}/training_history_epochs.csv', index=False)
    print(f"  Historique epochs : {save_dir}/training_history_epochs.csv")

    if history.get('iteration_losses'):
        pd.DataFrame({
            'iteration': range(len(history['iteration_losses'])),
            'loss':      history['iteration_losses'],
        }).to_csv(f'{save_dir}/training_history_iterations.csv', index=False)
        print(f"  Historique itérations : {save_dir}/training_history_iterations.csv")

    _create_summary_report(history, save_dir, n_epochs)


def _create_summary_report(history: dict, save_dir: str, n_epochs: int):
    """Rapport texte de l'entraînement."""
    METRICS = ['accuracy', 'precision', 'recall', 'f1', 'specificity',
               'balanced_accuracy', 'mcc', 'auroc', 'auprc', 'cohen_kappa']

    def _write_epoch_metrics(f, idx: int):
        for m in METRICS:
            vk, tk = f'val_{m}', f'train_{m}'
            if vk in history and idx < len(history[vk]):
                if _get(history, tk):
                    f.write(f"  Train {m:20s} {history[tk][idx]:.6f}\n")
                f.write(f"  Val   {m:20s} {history[vk][idx]:.6f}\n")

    report_path = Path(save_dir) / 'training_report.txt'

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\nRAPPORT D'ENTRAÎNEMENT\n" + "="*80 + "\n\n")
        f.write(f"Epochs : {n_epochs}\n")
        if history.get('iteration_losses'):
            total = len(history['iteration_losses'])
            f.write(f"Itérations totales : {total:,}  ({total // n_epochs:,} / epoch)\n")

        # Meilleure epoch selon val_loss (si disponible)
        if _get(history, 'val_loss'):
            best = int(np.argmin(history['val_loss']))
            f.write(f"\n{'─'*80}\nMEILLEURE EPOCH (val_loss) : {best + 1}\n{'─'*80}\n")
            if _get(history, 'train_loss'):
                f.write(f"  Train Loss : {history['train_loss'][best]:.6f}\n")
            f.write(f"  Val   Loss : {history['val_loss'][best]:.6f}\n")
            _write_epoch_metrics(f, best)

        # Dernière epoch
        f.write(f"\n{'─'*80}\nDERNIÈRE EPOCH : {n_epochs}\n{'─'*80}\n")
        if _get(history, 'train_loss'):
            f.write(f"  Train Loss : {history['train_loss'][-1]:.6f}\n")
        if _get(history, 'val_loss'):
            f.write(f"  Val   Loss : {history['val_loss'][-1]:.6f}\n")
        _write_epoch_metrics(f, n_epochs - 1)

        if n_epochs > 1 and _get(history, 'train_loss'):
            f.write(f"\n{'─'*80}\nAMÉLIORATION (epoch 1 → {n_epochs})\n{'─'*80}\n")
            dl = history['train_loss'][0] - history['train_loss'][-1]
            f.write(f"  Δ Loss train : {dl:+.6f} ({dl/history['train_loss'][0]*100:+.2f}%)\n")
            if _get(history, 'val_mcc'):
                dm = history['val_mcc'][-1] - history['val_mcc'][0]
                f.write(f"  Δ MCC val    : {dm:+.6f}\n")

        f.write("\n" + "="*80 + "\n")

    print(f"  Rapport : {report_path}")


# ─────────────────────────────────────────────────────────────────────────────
# RÉSUMÉ CONSOLE
# ─────────────────────────────────────────────────────────────────────────────

def print_training_summary(result: dict):
    """Affiche un résumé concis post-entraînement."""
    history      = result.get('history', {})
    best_metrics = result.get('best_metrics', {})
    best_epoch   = result.get('best_epoch', 1)
    final_epoch  = result.get('final_epoch', 1)
    t            = result.get('training_time', 0)
    idx          = best_epoch - 1

    print(f"\n{'='*90}")
    print(f" RÉSUMÉ POST-ENTRAÎNEMENT")
    print(f"{'='*90}")
    print(f"  Epochs : {final_epoch}  |  Meilleure : {best_epoch}  |  "
          f"Durée : {t/3600:.2f}h ({t/60:.1f}min)  |  "
          f"Early stopping : {'Oui' if result.get('early_stopped') else 'Non'}")

    classic = {
        'Accuracy': 'val_accuracy', 'Balanced Acc': 'val_balanced_accuracy',
        'Precision': 'val_precision', 'Recall': 'val_recall',
        'Specificity': 'val_specificity', 'F1': 'val_f1', 'F2': 'val_f2',
        'IoU': 'val_iou', "Cohen κ": 'val_cohen_kappa', 'MCC': 'val_mcc',
    }
    print(f"\n  Métriques Val — Epoch {best_epoch}:")
    for lbl, key in classic.items():
        vals = _get(history, key)
        if vals and idx < len(vals):
            print(f"    {lbl:16s} {vals[idx]:.4f}")

    if all(_get(history, k) and idx < len(_get(history, k))
           for k in ['val_tp', 'val_tn', 'val_fp', 'val_fn']):
        tp = int(history['val_tp'][idx]); tn = int(history['val_tn'][idx])
        fp = int(history['val_fp'][idx]); fn = int(history['val_fn'][idx])
        print(f"\n  Confusion (Val) :  TP {tp:5d} | FP {fp:5d}")
        print(f"                     FN {fn:5d} | TN {tn:5d}")

    if best_metrics.get('auroc', 0) > 0:
        print(f"\n  Probabilistes :  AUROC {best_metrics['auroc']:.4f}", end='')
        auprc = _get(history, 'val_auprc')
        if auprc and idx < len(auprc):
            print(f"  |  AUPRC {auprc[idx]:.4f}", end='')
        brier = _get(history, 'val_brier_score')
        if brier and idx < len(brier):
            print(f"  |  Brier {brier[idx]:.4f}", end='')
        print()

    print(f"{'='*90}\n")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Entraînement du modèle exoplanète")
    parser.add_argument('--config', type=str, default=None, help="Chemin vers la config YAML/JSON")
    args = parser.parse_args()

    cfg = Config.load(args.config) if args.config else get_config_object()
    set_seed(cfg.data.random_seed)
    cfg.print_summary()

    print("\n── Chargement des données ──")
    train_loader, val_loader, dataset_info = build_dataloaders(cfg)

    print("\n── Construction du modèle ──")
    model = build_model(cfg, dataset_info['spectrum_length'], dataset_info['auxiliary_dim'])

    print("\n── Initialisation du Trainer ──")
    trainer = _select_trainer(model, train_loader, val_loader, cfg)  # ← routage automatique

    result = trainer.train()

    print_training_summary(result)

    out_dir = Path(cfg.results_folder)
    out_dir.mkdir(parents=True, exist_ok=True)
    history = result.get('history', {})

    print("\n── Sauvegarde des résultats ──")
    with open(out_dir / 'training_history.json', 'w') as f:
        json.dump(result, f, indent=2, cls=NumpyEncoder)
    print(f"  JSON : {out_dir}/training_history.json")

    save_training_results(history, str(out_dir))
    plot_training_results(history, str(out_dir))

    print(f"\n Tout est sauvegardé dans : {out_dir}\n")


if __name__ == '__main__':
    main()