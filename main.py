#!/usr/bin/env python3
"""Script d'entraînement principal.

Usage: python main.py [--spectra PATH] [--aux PATH] [--targets PATH]
"""
import argparse
from pathlib import Path
import json
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

from training.config import Config, get_config_object
from training.training import Trainer
from models.dataset import ExoplanetDataset, collate_fn
from models.CNN import CNN


class NumpyEncoder(json.JSONEncoder):
    """Encodeur JSON personnalisé pour gérer les types NumPy."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def build_dataloaders(config: Config):
    """Construit les dataloaders pour l'entraînement et la validation."""
    data_cfg = config.data
    spectra_path = Path(data_cfg.spectra_train_path)    
    aux_path = Path(data_cfg.auxiliary_train_path)  
    targets_path = Path(data_cfg.targets_train_path)

    # If default paths don't exist, try the provided dataset folder
    if not spectra_path.exists():
        alt = Path('Défi-IA-2026') / 'DATA' / 'defi-ia-cnes' / 'spectra.npy'
        if alt.exists():
            spectra_path = alt
    if not aux_path.exists():
        alt = Path('Défi-IA-2026') / 'DATA' / 'defi-ia-cnes' / 'auxiliary.csv'
        if alt.exists():
            aux_path = alt
    if not targets_path.exists():
        alt = Path('Défi-IA-2026') / 'DATA' / 'defi-ia-cnes' / 'targets.csv'
        if alt.exists():
            targets_path = alt

    print(f"Using spectra: {spectra_path}")
    print(f"Using auxiliary: {aux_path}")
    print(f"Using targets: {targets_path}")

    dataset = ExoplanetDataset(
        str(spectra_path), 
        str(aux_path), 
        str(targets_path), 
        is_train=True
    )

    # Split train/val
    total = len(dataset)
    train_len = int(total * data_cfg.train_ratio)
    val_len = total - train_len

    train_ds, val_ds = random_split(dataset, [train_len, val_len])

    train_loader = DataLoader(
        train_ds,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=data_cfg.num_workers,
        pin_memory=data_cfg.pin_memory,
        collate_fn=collate_fn
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=data_cfg.num_workers,
        pin_memory=data_cfg.pin_memory,
        collate_fn=collate_fn
    )

    return train_loader, val_loader, dataset


def plot_training_results(history: dict, save_dir: str = 'results'):
    """Crée tous les graphiques d'entraînement avec métriques enrichies."""
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    # Figure principale élargie
    fig = plt.figure(figsize=(24, 16))

    epochs = range(1, len(history['train_loss']) + 1)

    # 1. Loss par itération (train)
    ax1 = plt.subplot(3, 3, 1)
    if 'iteration_losses' in history and len(history['iteration_losses']) > 0:
        ax1.plot(history['iteration_losses'], linewidth=1, alpha=0.7, color='#3498db')
        ax1.set_xlabel('Itération', fontsize=12)
        ax1.set_ylabel('Loss', fontsize=12)
        ax1.set_title('Loss Train (par itération)', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)

    # 2. Loss par epoch
    ax2 = plt.subplot(3, 3, 2)
    ax2.plot(epochs, history['train_loss'], label='Train', linewidth=2, 
            marker='o', markersize=6, color='#3498db')
    ax2.plot(epochs, history['val_loss'], label='Validation', linewidth=2, 
            marker='s', markersize=6, color='#e74c3c')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Loss', fontsize=12)
    ax2.set_title('Loss (par epoch)', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)

    # 3. Accuracy
    ax3 = plt.subplot(3, 3, 3)
    if 'train_accuracy' in history:
        ax3.plot(epochs, history['train_accuracy'], label='Train', linewidth=2, 
                marker='o', markersize=6, color='#2ecc71')
        ax3.plot(epochs, history['val_accuracy'], label='Validation', linewidth=2, 
                marker='s', markersize=6, color='#f39c12')
        ax3.set_xlabel('Epoch', fontsize=12)
        ax3.set_ylabel('Accuracy', fontsize=12)
        ax3.set_title('Accuracy', fontsize=14, fontweight='bold')
        ax3.legend(fontsize=11)
        ax3.grid(True, alpha=0.3)

    # 4. Precision & Recall
    ax4 = plt.subplot(3, 3, 4)
    if 'train_precision' in history:
        ax4.plot(epochs, history['train_precision'], label='Train Precision', linewidth=2, 
                marker='o', markersize=6, color='#9b59b6')
        ax4.plot(epochs, history['val_precision'], label='Val Precision', linewidth=2, 
                marker='s', markersize=6, color='#e67e22')
        ax4.plot(epochs, history['train_recall'], label='Train Recall', linewidth=2, 
                marker='^', markersize=6, color='#1abc9c', linestyle='--')
        ax4.plot(epochs, history['val_recall'], label='Val Recall', linewidth=2, 
                marker='v', markersize=6, color='#e74c3c', linestyle='--')
        ax4.set_xlabel('Epoch', fontsize=12)
        ax4.set_ylabel('Score', fontsize=12)
        ax4.set_title('Precision & Recall', fontsize=14, fontweight='bold')
        ax4.legend(fontsize=10)
        ax4.grid(True, alpha=0.3)

    # 5. F1 Score
    ax5 = plt.subplot(3, 3, 5)
    if 'train_f1' in history:
        ax5.plot(epochs, history['train_f1'], label='Train F1', linewidth=2, 
                marker='o', markersize=6, color='#2ecc71')
        ax5.plot(epochs, history['val_f1'], label='Val F1', linewidth=2, 
                marker='s', markersize=6, color='#f39c12')
        ax5.set_xlabel('Epoch', fontsize=12)
        ax5.set_ylabel('F1 Score', fontsize=12)
        ax5.set_title('F1 Score', fontsize=14, fontweight='bold')
        ax5.legend(fontsize=11)
        ax5.grid(True, alpha=0.3)

    # 6. Learning Rate (si disponible)
    ax6 = plt.subplot(3, 3, 6)
    if 'learning_rates' in history and len(history['learning_rates']) > 0:
        ax6.plot(epochs, history['learning_rates'], linewidth=2, 
                marker='o', markersize=6, color='#e74c3c')
        ax6.set_xlabel('Epoch', fontsize=12)
        ax6.set_ylabel('Learning Rate', fontsize=12)
        ax6.set_title('Learning Rate', fontsize=14, fontweight='bold')
        ax6.set_yscale('log')
        ax6.grid(True, alpha=0.3)

    # 7. Confusion Matrix Evolution (si disponible)
    ax7 = plt.subplot(3, 3, 7)
    if 'train_true_positives' in history:
        ax7.plot(epochs, history['train_true_positives'], label='Train TP', linewidth=2, marker='o')
        ax7.plot(epochs, history['train_false_positives'], label='Train FP', linewidth=2, marker='s')
        ax7.plot(epochs, history['train_true_negatives'], label='Train TN', linewidth=2, marker='^')
        ax7.plot(epochs, history['train_false_negatives'], label='Train FN', linewidth=2, marker='v')
        ax7.set_xlabel('Epoch', fontsize=12)
        ax7.set_ylabel('Count', fontsize=12)
        ax7.set_title('Confusion Matrix Elements (Train)', fontsize=14, fontweight='bold')
        ax7.legend(fontsize=9)
        ax7.grid(True, alpha=0.3)

    # 8. AUC (si disponible)
    ax8 = plt.subplot(3, 3, 8)
    if 'train_auc' in history:
        train_auc = [x if x is not None else np.nan for x in history['train_auc']]
        val_auc = [x if x is not None else np.nan for x in history['val_auc']]
        ax8.plot(epochs, train_auc, label='Train AUC', linewidth=2, 
                marker='o', markersize=6, color='#3498db')
        ax8.plot(epochs, val_auc, label='Val AUC', linewidth=2, 
                marker='s', markersize=6, color='#e74c3c')
        ax8.set_xlabel('Epoch', fontsize=12)
        ax8.set_ylabel('AUC', fontsize=12)
        ax8.set_title('Area Under Curve', fontsize=14, fontweight='bold')
        ax8.legend(fontsize=11)
        ax8.grid(True, alpha=0.3)

    # 9. Métriques finales en barres
    ax9 = plt.subplot(3, 3, 9)
    metrics_to_show = {}
    
    if 'val_accuracy' in history:
        metrics_to_show['Accuracy'] = history['val_accuracy'][-1]
    if 'val_precision' in history:
        metrics_to_show['Precision'] = history['val_precision'][-1]
    if 'val_recall' in history:
        metrics_to_show['Recall'] = history['val_recall'][-1]
    if 'val_f1' in history:
        metrics_to_show['F1'] = history['val_f1'][-1]
    
    if metrics_to_show:
        colors = ['#2ecc71', '#9b59b6', '#e74c3c', '#f39c12']
        bars = ax9.bar(range(len(metrics_to_show)), list(metrics_to_show.values()), 
                       color=colors[:len(metrics_to_show)], edgecolor='black', linewidth=1.5)
        ax9.set_xticks(range(len(metrics_to_show)))
        ax9.set_xticklabels(list(metrics_to_show.keys()), fontsize=11)
        ax9.set_ylabel('Valeur', fontsize=12)
        ax9.set_title('Métriques Finales (Validation)', fontsize=14, fontweight='bold')
        ax9.grid(True, alpha=0.3, axis='y')
        ax9.set_ylim([0, 1])
        
        for bar in bars:
            height = bar.get_height()
            ax9.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}', ha='center', va='bottom', 
                    fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig(f'{save_dir}/training_history.png', dpi=100, bbox_inches='tight')
    print(f"📊 Graphiques sauvegardés dans '{save_dir}/training_history.png'")
    plt.close()


def save_training_results(history: dict, save_dir: str = 'results'):
    """Sauvegarde tous les résultats d'entraînement."""
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    # Historique epochs
    df_data = {
        'epoch': range(1, len(history['train_loss']) + 1),
        'train_loss': history['train_loss'],
        'val_loss': history['val_loss'],
    }
    
    # Ajouter les métriques disponibles
    optional_metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc']
    for metric in optional_metrics:
        train_key = f'train_{metric}'
        val_key = f'val_{metric}'
        if train_key in history:
            df_data[train_key] = history[train_key]
        if val_key in history:
            df_data[val_key] = history[val_key]
    
    if 'learning_rates' in history:
        df_data['learning_rate'] = history['learning_rates']
    
    df_epochs = pd.DataFrame(df_data)
    df_epochs.to_csv(f'{save_dir}/training_history_epochs.csv', index=False)
    print(f"💾 Historique (epochs) sauvegardé dans '{save_dir}/training_history_epochs.csv'")

    # Historique itérations
    if 'iteration_losses' in history and len(history['iteration_losses']) > 0:
        df_iters = pd.DataFrame({
            'iteration': range(len(history['iteration_losses'])),
            'train_loss': history['iteration_losses']
        })
        df_iters.to_csv(f'{save_dir}/training_history_iterations.csv', index=False)
        print(f"💾 Historique (itérations) sauvegardé dans '{save_dir}/training_history_iterations.csv'")


def print_training_summary(history: dict):
    """Affiche un résumé détaillé de l'entraînement."""
    print("\n" + "="*80)
    print("📝 RÉSUMÉ DE L'ENTRAÎNEMENT")
    print("="*80)

    num_epochs = len(history['train_loss'])
    print(f"\nNombre d'epochs: {num_epochs}")

    if 'iteration_losses' in history:
        total_iterations = len(history['iteration_losses'])
        print(f"Nombre total d'itérations: {total_iterations:,}")
        if num_epochs > 0:
            print(f"Itérations par epoch: {total_iterations // num_epochs:,}")

    # Meilleure epoch
    if 'val_loss' in history:
        best_epoch = np.argmin(history['val_loss']) + 1
        print(f"\n🏆 Meilleure epoch (selon val_loss): {best_epoch}")
        print(f"   • Val Loss: {history['val_loss'][best_epoch - 1]:.4f}")
        
        if 'val_accuracy' in history:
            print(f"   • Val Accuracy: {history['val_accuracy'][best_epoch - 1]:.4f}")
        if 'val_f1' in history:
            print(f"   • Val F1: {history['val_f1'][best_epoch - 1]:.4f}")

    # Dernière epoch
    print(f"\n📈 Dernière epoch ({num_epochs}):")
    print(f"   • Train Loss: {history['train_loss'][-1]:.4f}")
    print(f"   • Val Loss:   {history['val_loss'][-1]:.4f}")
    
    if 'train_accuracy' in history:
        print(f"   • Train Accuracy: {history['train_accuracy'][-1]:.4f}")
        print(f"   • Val Accuracy:   {history['val_accuracy'][-1]:.4f}")
    
    if 'train_f1' in history:
        print(f"   • Train F1: {history['train_f1'][-1]:.4f}")
        print(f"   • Val F1:   {history['val_f1'][-1]:.4f}")

    # Amélioration
    if num_epochs > 1:
        loss_improvement = history['train_loss'][0] - history['train_loss'][-1]
        loss_improvement_pct = (loss_improvement / history['train_loss'][0]) * 100

        print(f"\n📊 Amélioration:")
        print(f"   • Réduction loss: {loss_improvement:.4f} ({loss_improvement_pct:.2f}%)")
        
        if 'train_accuracy' in history:
            acc_improvement = history['train_accuracy'][-1] - history['train_accuracy'][0]
            print(f"   • Gain accuracy: {acc_improvement:+.4f}")

    print("="*80 + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=None)
    args = parser.parse_args()

    if args.config:
        cfg = Config.load(args.config)
    else:
        cfg = get_config_object()

    cfg.print_summary()

    train_loader, val_loader, dataset = build_dataloaders(cfg)

    # Construire le modèle
    spectrum_length = dataset.spectra.shape[1]
    auxiliary_dim = dataset.aux_features.shape[1]

    model = CNN(
        spectrum_length=spectrum_length,
        auxiliary_dim=auxiliary_dim,
        num_classes=cfg.training.num_classes,
        conv_channels=cfg.model.conv_channels,
        kernel_sizes=cfg.model.kernel_sizes,
        pool_sizes=cfg.model.pool_sizes,
        fc_dims=cfg.model.fc_dims,
        dropout=cfg.model.dropout,
        use_batch_norm=cfg.model.use_batch_norm,
        input_channels=3
    )

    trainer = Trainer(model, train_loader, val_loader, cfg)

    print("\n🚀 Début de l'entraînement...\n")
    history = trainer.train()

    # Affichage et sauvegarde des résultats
    print_training_summary(history)
    
    # Sauvegarde finale de l'historique JSON
    out_json = Path(cfg.results_folder) / 'training_history.json'
    out_json.parent.mkdir(parents=True, exist_ok=True)
    
    with open(out_json, 'w') as f:
        json.dump(history, f, indent=2, cls=NumpyEncoder)
    
    print(f"💾 Training history (JSON) saved to {out_json}")
    
    # Sauvegarde CSV et graphiques
    save_training_results(history, cfg.results_folder)
    plot_training_results(history, cfg.results_folder)


if __name__ == '__main__':
    main()
