"""Génération des graphiques d'entraînement.

Module dédié à la production des figures à partir d'un historique
d'entraînement (loss, métriques, matrice de confusion, etc.).
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS PRIVÉS
# ─────────────────────────────────────────────────────────────────────────────

def _get(history: dict, key: str) -> list:
    """Retourne history[key] ou une liste vide si absent."""
    return history.get(key, [])


def _plot_pair(ax, epochs, train_vals, val_vals, title: str, ylabel: str,
               ylim=None, train_color='#3498db', val_color='#e74c3c',
               ref_lines: list = None):
    """Trace une courbe train/val sur un axe matplotlib."""
    if train_vals:
        ax.plot(epochs, train_vals, label='Train', linewidth=2,
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


# ─────────────────────────────────────────────────────────────────────────────
# API PUBLIQUE
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
