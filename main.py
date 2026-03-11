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

from training.config import Config, get_config_object
from training.training import Trainer
from training.utils import set_seed
from models.dataset import ExoplanetDataset, collate_fn
from sklearn.model_selection import train_test_split
from models.CNN import CNN
from models.ResNetCNN import ResNet1D
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
    """Construit les DataLoaders train/val avec split stratifié.

    Returns:
        (train_loader, val_loader, dataset_info) où dataset_info contient
        spectrum_length et auxiliary_dim pour construire le modèle sans
        garder le dataset entier en mémoire dans main().
    """
    data_cfg     = config.data
    fallback_root = Path('Défi-IA-2026') / 'DATA' / 'defi-ia-cnes'

    spectra_path  = _resolve_path(Path(data_cfg.spectra_train_path),   fallback_root / 'spectra.npy',    'spectra')
    aux_path      = _resolve_path(Path(data_cfg.auxiliary_train_path),  fallback_root / 'auxiliary.csv',  'auxiliary')
    targets_path  = _resolve_path(Path(data_cfg.targets_train_path),    fallback_root / 'targets.csv',    'targets')

    print(f"  spectra   : {spectra_path}")
    print(f"  auxiliary : {aux_path}")
    print(f"  targets   : {targets_path}")

    spectra_all    = np.load(spectra_path)
    aux_df_all     = pd.read_csv(aux_path)
    targets_df_all = pd.read_csv(targets_path)

    # Split stratifié sur la combinaison (eau, nuage)
    strat_labels = targets_df_all['eau'].astype(str) + "_" + targets_df_all['nuage'].astype(str)
    indices      = list(range(len(spectra_all)))
    train_idx, val_idx = train_test_split(
        indices,
        test_size    = 1 - data_cfg.train_ratio,
        random_state = 42,
        stratify     = strat_labels,
    )

    spectra_train   = spectra_all[train_idx]
    spectra_val     = spectra_all[val_idx]
    aux_train       = aux_df_all.iloc[train_idx].reset_index(drop=True)
    aux_val         = aux_df_all.iloc[val_idx].reset_index(drop=True)
    targets_train   = targets_df_all.iloc[train_idx].reset_index(drop=True)
    targets_val     = targets_df_all.iloc[val_idx].reset_index(drop=True)

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
        spectra      = spectra_val,
        auxiliary_df = aux_val,
        targets_df   = targets_val,
        is_train     = False,
        augmentation_factor = 0,
        aux_mean     = train_dataset.aux_mean,   # pas de leakage
        aux_std      = train_dataset.aux_std,
    )

    loader_kwargs = dict(
        batch_size  = config.training.batch_size,
        num_workers = data_cfg.num_workers,
        pin_memory  = data_cfg.pin_memory,
        collate_fn  = collate_fn,
    )
    train_loader = DataLoader(train_dataset, shuffle=True,  **loader_kwargs)
    val_loader   = DataLoader(val_dataset,   shuffle=False, **loader_kwargs)

    # Extraire les dimensions ici pour éviter de garder le dataset en RAM dans main()
    dataset_info = {
        'spectrum_length': train_dataset.spectra.shape[1],
        'auxiliary_dim':   train_dataset.aux_features.shape[1],
    }
    # Libérer les références numpy brutes — le dataset garde ce dont il a besoin
    del spectra_all

    return train_loader, val_loader, dataset_info


# ─────────────────────────────────────────────────────────────────────────────
# MODÈLE
# ─────────────────────────────────────────────────────────────────────────────

def build_model(cfg, spectrum_length: int, auxiliary_dim: int):
    """Instancie le modèle selon la config."""
    arch = cfg.model.architecture
    common = dict(
        spectrum_length = spectrum_length,
        auxiliary_dim   = auxiliary_dim,
        num_classes     = cfg.training.num_classes,
        input_channels  = 3,
        dropout         = cfg.model.dropout,
    )
    if arch == "CNN":
        print("  Architecture : CNN")
        return CNN(
            **common,
            conv_channels  = cfg.model.conv_channels,
            kernel_sizes   = cfg.model.kernel_sizes,
            pool_sizes      = cfg.model.pool_sizes,
            fc_dims        = cfg.model.fc_dims,
            use_batch_norm = cfg.model.use_batch_norm,
        )
    else:
        print("  Architecture : ResNet1D")
        return ResNet1D(
            **common,
            block_type  = 'basic',
            num_blocks  = [2, 2, 2, 2],
            base_channels = 64,
        )


# ─────────────────────────────────────────────────────────────────────────────
# GRAPHIQUES — helper commun
# ─────────────────────────────────────────────────────────────────────────────

def _plot_pair(ax, epochs, train_vals, val_vals, title: str, ylabel: str,
               ylim=None, train_color='#3498db', val_color='#e74c3c',
               ref_lines: list = None):
    """Trace une courbe train/val sur un axe matplotlib.
    Factorisé pour éviter les ~20 blocs identiques dans plot_training_results.
    """
    ax.plot(epochs, train_vals, label='Train',      linewidth=2, marker='o', markersize=5, color=train_color)
    ax.plot(epochs, val_vals,   label='Validation', linewidth=2, marker='s', markersize=5, color=val_color)
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
    """Crée la figure agrégée (4×3) et les graphiques individuels."""
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    epochs = range(1, len(history['train_loss']) + 1)

    fig, axes = plt.subplots(4, 3, figsize=(26, 22))
    axs = axes.flatten()

    # 1 — Loss par itération
    ax = axs[0]
    iters = _get(history, 'iteration_losses')
    if iters:
        ax.plot(iters, linewidth=1, alpha=0.7, color='#3498db')
        ax.set_title('Loss Train (par itération)', fontsize=13, fontweight='bold')
        ax.set_xlabel('Itération', fontsize=11)
        ax.set_ylabel('Loss', fontsize=11)
        ax.grid(True, alpha=0.3)

    # 2 — Loss par epoch
    _plot_pair(axs[1], epochs, history['train_loss'], history['val_loss'], 'Loss', 'Loss')

    # 3 — Accuracy
    if _get(history, 'train_accuracy'):
        _plot_pair(axs[2], epochs, history['train_accuracy'], history['val_accuracy'],
                   'Accuracy', 'Accuracy', ylim=[0, 1])

    # 4 — Precision
    if _get(history, 'train_precision'):
        _plot_pair(axs[3], epochs, history['train_precision'], history['val_precision'],
                   'Precision', 'Precision', ylim=[0, 1])

    # 5 — Recall
    if _get(history, 'train_recall'):
        _plot_pair(axs[4], epochs, history['train_recall'], history['val_recall'],
                   'Recall', 'Recall', ylim=[0, 1])

    # 6 — F1
    if _get(history, 'train_f1'):
        _plot_pair(axs[5], epochs, history['train_f1'], history['val_f1'],
                   'F1 Score', 'F1', ylim=[0, 1])

    # 7 — Balanced Accuracy
    if _get(history, 'train_balanced_accuracy'):
        _plot_pair(axs[6], epochs, history['train_balanced_accuracy'], history['val_balanced_accuracy'],
                   'Balanced Accuracy', 'Bal. Acc', ylim=[0, 1])

    # 8 — Specificity
    if _get(history, 'train_specificity'):
        _plot_pair(axs[7], epochs, history['train_specificity'], history['val_specificity'],
                   'Specificity', 'Specificity', ylim=[0, 1])

    # 9 — AUROC
    if _get(history, 'train_auroc'):
        _plot_pair(axs[8], epochs, history['train_auroc'], history['val_auroc'],
                   'AUROC', 'AUROC', ylim=[0, 1])

    # 10 — MCC
    if _get(history, 'train_mcc'):
        _plot_pair(axs[9], epochs, history['train_mcc'], history['val_mcc'],
                   'Matthews Correlation Coefficient', 'MCC', ylim=[-1, 1],
                   ref_lines=[(0, 'gray', '--', 'Aléatoire (0)')])

    # 11 — Composite Score
    if _get(history, 'train_composite_score'):
        _plot_pair(axs[10], epochs, history['train_composite_score'], history['val_composite_score'],
                   'Composite Score', 'Score')

    # 12 — Bar chart métriques finales
    ax = axs[11]
    metric_keys = [
        ('accuracy', 'Acc'), ('precision', 'Prec'), ('recall', 'Rec'),
        ('f1', 'F1'), ('specificity', 'Spec'), ('balanced_accuracy', 'BalAcc'), ('mcc', 'MCC'),
    ]
    names, train_vals, val_vals = [], [], []
    for key, label in metric_keys:
        if _get(history, f'val_{key}'):
            names.append(label)
            train_vals.append(_get(history, f'train_{key}')[-1] if _get(history, f'train_{key}') else 0)
            val_vals.append(history[f'val_{key}'][-1])
    if names:
        x, w = np.arange(len(names)), 0.35
        b1 = ax.bar(x - w/2, train_vals, w, label='Train',      color='#3498db', edgecolor='black')
        b2 = ax.bar(x + w/2, val_vals,   w, label='Validation', color='#e74c3c', edgecolor='black')
        ax.set_xticks(x)
        ax.set_xticklabels(names, fontsize=10, rotation=15)
        ax.set_title('Métriques Finales (Train vs Val)', fontsize=13, fontweight='bold')
        ax.set_ylim([-1, 1.1])
        ax.axhline(0, color='black', linewidth=0.8, alpha=0.5)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        for bars in [b1, b2]:
            for bar in bars:
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
    if _get(history, 'train_precision') and _get(history, 'train_recall'):
        fig, ax = plt.subplots(figsize=(10, 6))
        for vals, lbl, color, ls in [
            (history['train_precision'], 'Train Precision', '#9b59b6', '-'),
            (history['val_precision'],   'Val Precision',   '#e67e22', '-'),
            (history['train_recall'],    'Train Recall',    '#1abc9c', '--'),
            (history['val_recall'],      'Val Recall',      '#e74c3c', '--'),
        ]:
            ax.plot(epochs, vals, label=lbl, linewidth=2, linestyle=ls)
        ax.set(title='Precision & Recall', xlabel='Epoch', ylabel='Score', ylim=[0, 1])
        ax.legend(); ax.grid(True, alpha=0.3)
        _save('precision_recall.png')

    # Loss détaillée avec min
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, history['train_loss'], label='Train', linewidth=2, marker='o', color='#3498db')
    ax.plot(epochs, history['val_loss'],   label='Val',   linewidth=2, marker='s', color='#e74c3c')
    for vals, color in [(history['train_loss'], '#3498db'), (history['val_loss'], '#e74c3c')]:
        m = min(vals)
        ax.axhline(m, color=color, linestyle=':', alpha=0.5, label=f'Min: {m:.4f}')
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

    # MCC détaillé avec annotation du meilleur
    if _get(history, 'train_mcc'):
        val_mcc = [x if x is not None else np.nan for x in history['val_mcc']]
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(epochs, [x if x is not None else np.nan for x in history['train_mcc']],
                label='Train MCC', linewidth=2, marker='o', color='#2980b9')
        ax.plot(epochs, val_mcc, label='Val MCC', linewidth=2, marker='s', color='#c0392b')
        for y, color, ls, lbl in [(0, 'gray', '--', 'Aléatoire'), (1, 'green', ':', 'Parfait'), (-1, 'red', ':', 'Inversé')]:
            ax.axhline(y, color=color, linestyle=ls, alpha=0.5, label=lbl)
        if not all(np.isnan(val_mcc)):
            bi = int(np.nanargmax(val_mcc))
            ax.annotate(f'Best Val\n{val_mcc[bi]:.4f}',
                        xy=(list(epochs)[bi], val_mcc[bi]), xytext=(10, -30),
                        textcoords='offset points', fontsize=10, fontweight='bold',
                        arrowprops=dict(arrowstyle='->', color='black'),
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
        ax.set(title='Matthews Correlation Coefficient', xlabel='Epoch', ylabel='MCC', ylim=[-1, 1])
        ax.legend(); ax.grid(True, alpha=0.3)
        _save('mcc.png')

    # Confusion matrix elements (val)
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
    """Sauvegarde l'historique complet en CSV (toutes les clés list de l'history)."""
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    n_epochs = len(history['train_loss'])

    # On exporte toutes les clés qui ont exactement n_epochs valeurs (= métriques par epoch)
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

    _create_summary_report(history, save_dir)


def _create_summary_report(history: dict, save_dir: str):
    """Rapport texte de l'entraînement."""
    METRICS = ['accuracy', 'precision', 'recall', 'f1', 'specificity',
               'balanced_accuracy', 'mcc', 'auroc', 'auprc', 'cohen_kappa']

    def _write_epoch_metrics(f, idx: int):
        for m in METRICS:
            vk, tk = f'val_{m}', f'train_{m}'
            if vk in history and idx < len(history[vk]):
                if tk in history:
                    f.write(f"  Train {m:20s} {history[tk][idx]:.6f}\n")
                f.write(f"  Val   {m:20s} {history[vk][idx]:.6f}\n")

    report_path = Path(save_dir) / 'training_report.txt'
    n = len(history['train_loss'])

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\nRAPPORT D'ENTRAÎNEMENT\n" + "="*80 + "\n\n")
        f.write(f"Epochs : {n}\n")
        if history.get('iteration_losses'):
            total = len(history['iteration_losses'])
            f.write(f"Itérations totales : {total:,}  ({total // n:,} / epoch)\n")

        # Meilleure epoch selon val_loss
        if history.get('val_loss'):
            best = int(np.argmin(history['val_loss']))
            f.write(f"\n{'─'*80}\nMEILLEURE EPOCH (val_loss) : {best + 1}\n{'─'*80}\n")
            f.write(f"  Train Loss : {history['train_loss'][best]:.6f}\n")
            f.write(f"  Val   Loss : {history['val_loss'][best]:.6f}\n")
            _write_epoch_metrics(f, best)

        # Dernière epoch
        f.write(f"\n{'─'*80}\nDERNIÈRE EPOCH : {n}\n{'─'*80}\n")
        f.write(f"  Train Loss : {history['train_loss'][-1]:.6f}\n")
        f.write(f"  Val   Loss : {history['val_loss'][-1]:.6f}\n")
        _write_epoch_metrics(f, n - 1)

        # Amélioration globale
        if n > 1:
            f.write(f"\n{'─'*80}\nAMÉLIORATION (epoch 1 → {n})\n{'─'*80}\n")
            dl = history['train_loss'][0] - history['train_loss'][-1]
            f.write(f"  Δ Loss train : {dl:+.6f} ({dl/history['train_loss'][0]*100:+.2f}%)\n")
            if history.get('train_mcc'):
                dm = history['val_mcc'][-1] - history['val_mcc'][0]
                f.write(f"  Δ MCC val    : {dm:+.6f}\n")

        f.write("\n" + "="*80 + "\n")

    print(f"  Rapport : {report_path}")


# ─────────────────────────────────────────────────────────────────────────────
# RÉSUMÉ CONSOLE (post-entraînement)
# ─────────────────────────────────────────────────────────────────────────────

def print_training_summary(result: dict):
    """Affiche un résumé concis — sans redoubler ce que le Trainer a déjà affiché."""
    history      = result.get('history', {})
    best_metrics = result.get('best_metrics', {})
    best_epoch   = result.get('best_epoch', 0)
    final_epoch  = result.get('final_epoch', 0)
    t            = result.get('training_time', 0)
    idx          = best_epoch - 1

    print(f"\n{'='*90}")
    print(f" RÉSUMÉ POST-ENTRAÎNEMENT")
    print(f"{'='*90}")
    print(f"  Epochs : {final_epoch}  |  Meilleure : {best_epoch}  |  "
          f"Durée : {t/3600:.2f}h ({t/60:.1f}min)  |  "
          f"Early stopping : {'Oui' if result.get('early_stopped') else 'Non'}")

    # Métriques classiques à la meilleure epoch
    classic = {
        'Accuracy': 'val_accuracy', 'Balanced Acc': 'val_balanced_accuracy',
        'Precision': 'val_precision', 'Recall': 'val_recall',
        'Specificity': 'val_specificity', 'F1': 'val_f1', 'F2': 'val_f2',
        'IoU': 'val_iou', "Cohen κ": 'val_cohen_kappa', 'MCC': 'val_mcc',
    }
    print(f"\n  Métriques Val — Epoch {best_epoch}:")
    for lbl, key in classic.items():
        if key in history and idx < len(history[key]):
            print(f"    {lbl:16s} {history[key][idx]:.4f}")

    # Matrice de confusion
    if all(k in history and idx < len(history[k]) for k in ['val_tp', 'val_tn', 'val_fp', 'val_fn']):
        tp, tn = int(history['val_tp'][idx]), int(history['val_tn'][idx])
        fp, fn = int(history['val_fp'][idx]), int(history['val_fn'][idx])
        print(f"\n  Confusion (Val) :  TP {tp:5d} | FP {fp:5d}")
        print(f"                     FN {fn:5d} | TN {tn:5d}")

    # Métriques probabilistes
    if best_metrics.get('auroc', 0) > 0:
        print(f"\n  Probabilistes :  AUROC {best_metrics['auroc']:.4f}", end='')
        if 'val_auprc' in history and idx < len(history['val_auprc']):
            print(f"  |  AUPRC {history['val_auprc'][idx]:.4f}", end='')
        if 'val_brier_score' in history and idx < len(history['val_brier_score']):
            print(f"  |  Brier {history['val_brier_score'][idx]:.4f}", end='')
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
    trainer = Trainer(model, train_loader, val_loader, cfg)

    result  = trainer.train()

    # Résumé concis (le Trainer a déjà affiché le détail complet)
    print_training_summary(result)

    # Sauvegarde
    out_dir  = Path(cfg.results_folder)
    out_dir.mkdir(parents=True, exist_ok=True)
    history  = result.get('history', {})

    print("\n── Sauvegarde des résultats ──")
    with open(out_dir / 'training_history.json', 'w') as f:
        json.dump(result, f, indent=2, cls=NumpyEncoder)
    print(f"  JSON : {out_dir}/training_history.json")

    save_training_results(history, str(out_dir))
    plot_training_results(history, str(out_dir))

    print(f"\n Tout est sauvegardé dans : {out_dir}\n")


if __name__ == '__main__':
    main()