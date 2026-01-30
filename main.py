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

    # Figure principale élargie - 4x3 grille
    fig = plt.figure(figsize=(28, 20))

    epochs = range(1, len(history['train_loss']) + 1)

    # 1. Loss par itération (train)
    ax1 = plt.subplot(4, 3, 1)
    if 'iteration_losses' in history and len(history['iteration_losses']) > 0:
        ax1.plot(history['iteration_losses'], linewidth=1, alpha=0.7, color='#3498db')
        ax1.set_xlabel('Itération', fontsize=12)
        ax1.set_ylabel('Loss', fontsize=12)
        ax1.set_title('Loss Train (par itération)', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)

    # 2. Loss par epoch
    ax2 = plt.subplot(4, 3, 2)
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
    ax3 = plt.subplot(4, 3, 3)
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
        ax3.set_ylim([0, 1])

    # 4. Precision (séparé)
    ax4 = plt.subplot(4, 3, 4)
    if 'train_precision' in history:
        ax4.plot(epochs, history['train_precision'], label='Train Precision', linewidth=2, 
                marker='o', markersize=6, color='#9b59b6')
        ax4.plot(epochs, history['val_precision'], label='Val Precision', linewidth=2, 
                marker='s', markersize=6, color='#e67e22')
        ax4.set_xlabel('Epoch', fontsize=12)
        ax4.set_ylabel('Precision', fontsize=12)
        ax4.set_title('Precision', fontsize=14, fontweight='bold')
        ax4.legend(fontsize=11)
        ax4.grid(True, alpha=0.3)
        ax4.set_ylim([0, 1])

    # 5. Recall (séparé)
    ax5 = plt.subplot(4, 3, 5)
    if 'train_recall' in history:
        ax5.plot(epochs, history['train_recall'], label='Train Recall', linewidth=2, 
                marker='o', markersize=6, color='#1abc9c')
        ax5.plot(epochs, history['val_recall'], label='Val Recall', linewidth=2, 
                marker='s', markersize=6, color='#e74c3c')
        ax5.set_xlabel('Epoch', fontsize=12)
        ax5.set_ylabel('Recall', fontsize=12)
        ax5.set_title('Recall', fontsize=14, fontweight='bold')
        ax5.legend(fontsize=11)
        ax5.grid(True, alpha=0.3)
        ax5.set_ylim([0, 1])

    # 6. F1 Score
    ax6 = plt.subplot(4, 3, 6)
    if 'train_f1' in history:
        ax6.plot(epochs, history['train_f1'], label='Train F1', linewidth=2, 
                marker='o', markersize=6, color='#2ecc71')
        ax6.plot(epochs, history['val_f1'], label='Val F1', linewidth=2, 
                marker='s', markersize=6, color='#f39c12')
        ax6.set_xlabel('Epoch', fontsize=12)
        ax6.set_ylabel('F1 Score', fontsize=12)
        ax6.set_title('F1 Score', fontsize=14, fontweight='bold')
        ax6.legend(fontsize=11)
        ax6.grid(True, alpha=0.3)
        ax6.set_ylim([0, 1])

    # 7. Learning Rate
    ax7 = plt.subplot(4, 3, 7)
    if 'learning_rates' in history and len(history['learning_rates']) > 0:
        ax7.plot(epochs, history['learning_rates'], linewidth=2, 
                marker='o', markersize=6, color='#e74c3c')
        ax7.set_xlabel('Epoch', fontsize=12)
        ax7.set_ylabel('Learning Rate', fontsize=12)
        ax7.set_title('Learning Rate', fontsize=14, fontweight='bold')
        ax7.set_yscale('log')
        ax7.grid(True, alpha=0.3)

    # 8. AUC (si disponible)
    ax8 = plt.subplot(4, 3, 8)
    if 'train_auc' in history:
        train_auc = [x if x is not None else np.nan for x in history['train_auc']]
        val_auc = [x if x is not None else np.nan for x in history['val_auc']]
        ax8.plot(epochs, train_auc, label='Train AUC', linewidth=2, 
                marker='o', markersize=6, color='#3498db')
        ax8.plot(epochs, val_auc, label='Val AUC', linewidth=2, 
                marker='s', markersize=6, color='#e74c3c')
        ax8.set_xlabel('Epoch', fontsize=12)
        ax8.set_ylabel('AUC', fontsize=12)
        ax8.set_title('Area Under Curve (ROC)', fontsize=14, fontweight='bold')
        ax8.legend(fontsize=11)
        ax8.grid(True, alpha=0.3)
        ax8.set_ylim([0, 1])

    # 9. Specificity (si disponible)
    ax9 = plt.subplot(4, 3, 9)
    if 'train_specificity' in history:
        ax9.plot(epochs, history['train_specificity'], label='Train Specificity', linewidth=2, 
                marker='o', markersize=6, color='#8e44ad')
        ax9.plot(epochs, history['val_specificity'], label='Val Specificity', linewidth=2, 
                marker='s', markersize=6, color='#c0392b')
        ax9.set_xlabel('Epoch', fontsize=12)
        ax9.set_ylabel('Specificity', fontsize=12)
        ax9.set_title('Specificity (True Negative Rate)', fontsize=14, fontweight='bold')
        ax9.legend(fontsize=11)
        ax9.grid(True, alpha=0.3)
        ax9.set_ylim([0, 1])

    # 10. Confusion Matrix Elements (Train)
    ax10 = plt.subplot(4, 3, 10)
    if 'train_true_positives' in history:
        ax10.plot(epochs, history['train_true_positives'], label='True Positives', 
                 linewidth=2, marker='o', color='#27ae60')
        ax10.plot(epochs, history['train_false_positives'], label='False Positives', 
                 linewidth=2, marker='s', color='#e67e22')
        ax10.plot(epochs, history['train_true_negatives'], label='True Negatives', 
                 linewidth=2, marker='^', color='#3498db')
        ax10.plot(epochs, history['train_false_negatives'], label='False Negatives', 
                 linewidth=2, marker='v', color='#e74c3c')
        ax10.set_xlabel('Epoch', fontsize=12)
        ax10.set_ylabel('Count', fontsize=12)
        ax10.set_title('Confusion Matrix Elements (Train)', fontsize=14, fontweight='bold')
        ax10.legend(fontsize=9, loc='best')
        ax10.grid(True, alpha=0.3)

    # 11. Balanced Accuracy (si disponible)
    ax11 = plt.subplot(4, 3, 11)
    if 'train_balanced_accuracy' in history:
        ax11.plot(epochs, history['train_balanced_accuracy'], label='Train Balanced Acc', 
                 linewidth=2, marker='o', markersize=6, color='#16a085')
        ax11.plot(epochs, history['val_balanced_accuracy'], label='Val Balanced Acc', 
                 linewidth=2, marker='s', markersize=6, color='#d35400')
        ax11.set_xlabel('Epoch', fontsize=12)
        ax11.set_ylabel('Balanced Accuracy', fontsize=12)
        ax11.set_title('Balanced Accuracy', fontsize=14, fontweight='bold')
        ax11.legend(fontsize=11)
        ax11.grid(True, alpha=0.3)
        ax11.set_ylim([0, 1])

    # 12. Métriques finales en barres (comparaison Train vs Val)
    ax12 = plt.subplot(4, 3, 12)
    metrics_names = []
    train_values = []
    val_values = []

    metric_keys = [
        ('accuracy', 'Accuracy'),
        ('precision', 'Precision'),
        ('recall', 'Recall'),
        ('f1', 'F1'),
        ('specificity', 'Specificity'),
        ('balanced_accuracy', 'Bal. Acc')
    ]

    for key, name in metric_keys:
        if f'val_{key}' in history:
            metrics_names.append(name)
            train_values.append(history[f'train_{key}'][-1] if f'train_{key}' in history else 0)
            val_values.append(history[f'val_{key}'][-1])

    if metrics_names:
        x = np.arange(len(metrics_names))
        width = 0.35

        bars1 = ax12.bar(x - width/2, train_values, width, label='Train', 
                        color='#3498db', edgecolor='black', linewidth=1.5)
        bars2 = ax12.bar(x + width/2, val_values, width, label='Validation', 
                        color='#e74c3c', edgecolor='black', linewidth=1.5)

        ax12.set_xticks(x)
        ax12.set_xticklabels(metrics_names, fontsize=10, rotation=15)
        ax12.set_ylabel('Valeur', fontsize=12)
        ax12.set_title('Comparaison Métriques Finales', fontsize=14, fontweight='bold')
        ax12.legend(fontsize=11)
        ax12.grid(True, alpha=0.3, axis='y')
        ax12.set_ylim([0, 1.1])

        # Ajouter les valeurs sur les barres
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax12.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.3f}', ha='center', va='bottom', 
                        fontsize=8, fontweight='bold')

    plt.tight_layout()
    plt.savefig(f'{save_dir}/training_history.png', dpi=100, bbox_inches='tight')
    print(f" Graphiques sauvegardés dans '{save_dir}/training_history.png'")
    plt.close()

    # Créer des graphiques individuels supplémentaires
    create_individual_plots(history, epochs, save_dir)

def create_individual_plots(history: dict, epochs, save_dir: str):
    """Crée des graphiques individuels pour chaque métrique principale."""
    
    # 1. Graphique Precision vs Recall
    if 'train_precision' in history and 'train_recall' in history:
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, history['train_precision'], label='Train Precision', 
                linewidth=2, marker='o', color='#9b59b6')
        plt.plot(epochs, history['val_precision'], label='Val Precision', 
                linewidth=2, marker='s', color='#e67e22')
        plt.plot(epochs, history['train_recall'], label='Train Recall', 
                linewidth=2, marker='^', color='#1abc9c', linestyle='--')
        plt.plot(epochs, history['val_recall'], label='Val Recall', 
                linewidth=2, marker='v', color='#e74c3c', linestyle='--')
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Score', fontsize=12)
        plt.title('Precision et Recall', fontsize=14, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.ylim([0, 1])
        plt.tight_layout()
        plt.savefig(f'{save_dir}/precision_recall.png', dpi=100, bbox_inches='tight')
        plt.close()

    # 2. Graphique Loss détaillé
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history['train_loss'], label='Train Loss', 
            linewidth=2, marker='o', color='#3498db')
    plt.plot(epochs, history['val_loss'], label='Val Loss', 
            linewidth=2, marker='s', color='#e74c3c')
    
    # Ajouter min/max
    min_train = min(history['train_loss'])
    min_val = min(history['val_loss'])
    plt.axhline(y=min_train, color='#3498db', linestyle=':', alpha=0.5, 
                label=f'Min Train: {min_train:.4f}')
    plt.axhline(y=min_val, color='#e74c3c', linestyle=':', alpha=0.5, 
                label=f'Min Val: {min_val:.4f}')
    
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Evolution de la Loss', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/loss_detailed.png', dpi=100, bbox_inches='tight')
    plt.close()

    # 3. Heatmap de corrélation des métriques (dernière epoch)
    if 'val_accuracy' in history:
        plt.figure(figsize=(10, 8))
        
        metric_keys = ['accuracy', 'precision', 'recall', 'f1']
        available_metrics = []
        metric_names = []
        
        for key in metric_keys:
            if f'val_{key}' in history:
                available_metrics.append(key)
                metric_names.append(key.capitalize())
        
        if len(available_metrics) >= 2:
            # Créer matrice de valeurs pour toutes les epochs
            data_matrix = []
            for key in available_metrics:
                data_matrix.append(history[f'val_{key}'])
            
            data_matrix = np.array(data_matrix)
            correlation = np.corrcoef(data_matrix)
            
            im = plt.imshow(correlation, cmap='RdYlGn', vmin=-1, vmax=1, aspect='auto')
            plt.colorbar(im, label='Corrélation')
            
            plt.xticks(range(len(metric_names)), metric_names, rotation=45)
            plt.yticks(range(len(metric_names)), metric_names)
            
            # Ajouter les valeurs
            for i in range(len(metric_names)):
                for j in range(len(metric_names)):
                    text = plt.text(j, i, f'{correlation[i, j]:.2f}',
                                  ha="center", va="center", color="black", fontweight='bold')
            
            plt.title('Corrélation entre Métriques (Validation)', fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.savefig(f'{save_dir}/metrics_correlation.png', dpi=100, bbox_inches='tight')
            plt.close()

    # 4. Evolution du rapport Precision/Recall
    if 'train_precision' in history and 'train_recall' in history:
        plt.figure(figsize=(10, 6))
        
        train_ratio = [p/r if r > 0 else 0 for p, r in 
                      zip(history['train_precision'], history['train_recall'])]
        val_ratio = [p/r if r > 0 else 0 for p, r in 
                    zip(history['val_precision'], history['val_recall'])]
        
        plt.plot(epochs, train_ratio, label='Train Precision/Recall', 
                linewidth=2, marker='o', color='#9b59b6')
        plt.plot(epochs, val_ratio, label='Val Precision/Recall', 
                linewidth=2, marker='s', color='#e67e22')
        plt.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='Équilibre')
        
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Ratio Precision/Recall', fontsize=12)
        plt.title('Ratio Precision/Recall', fontsize=14, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{save_dir}/precision_recall_ratio.png', dpi=100, bbox_inches='tight')
        plt.close()

    print(f"📊 Graphiques individuels sauvegardés dans '{save_dir}/'")

def save_training_results(history: dict, save_dir: str = 'results'):
    """Sauvegarde tous les résultats d'entraînement."""
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    # Historique epochs
    df_data = {
        'epoch': range(1, len(history['train_loss']) + 1),
        'train_loss': history['train_loss'],
        'val_loss': history['val_loss'],
    }

    # Ajouter toutes les métriques disponibles
    optional_metrics = [
        'accuracy', 'precision', 'recall', 'f1', 'auc',
        'specificity', 'balanced_accuracy', 
        'true_positives', 'false_positives', 
        'true_negatives', 'false_negatives'
    ]
    
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
    print(f" Historique (epochs) sauvegardé dans '{save_dir}/training_history_epochs.csv'")

    # Historique itérations
    if 'iteration_losses' in history and len(history['iteration_losses']) > 0:
        df_iters = pd.DataFrame({
            'iteration': range(len(history['iteration_losses'])),
            'train_loss': history['iteration_losses']
        })
        df_iters.to_csv(f'{save_dir}/training_history_iterations.csv', index=False)
        print(f" Historique (itérations) sauvegardé dans '{save_dir}/training_history_iterations.csv'")

    # Créer un rapport de synthèse
    create_summary_report(history, save_dir)

def create_summary_report(history: dict, save_dir: str):
    """Crée un rapport texte détaillé de l'entraînement."""
    report_path = Path(save_dir) / 'training_report.txt'
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("RAPPORT D'ENTRAÎNEMENT DÉTAILLÉ\n")
        f.write("="*80 + "\n\n")
        
        num_epochs = len(history['train_loss'])
        f.write(f"Nombre d'epochs: {num_epochs}\n")
        
        if 'iteration_losses' in history:
            total_iterations = len(history['iteration_losses'])
            f.write(f"Nombre total d'itérations: {total_iterations:,}\n")
            if num_epochs > 0:
                f.write(f"Itérations par epoch: {total_iterations // num_epochs:,}\n")
        
        f.write("\n" + "-"*80 + "\n")
        f.write("MEILLEURE PERFORMANCE (selon val_loss)\n")
        f.write("-"*80 + "\n")
        
        if 'val_loss' in history:
            best_epoch = np.argmin(history['val_loss']) + 1
            f.write(f"\nÉpoch: {best_epoch}\n")
            f.write(f"Train Loss: {history['train_loss'][best_epoch - 1]:.6f}\n")
            f.write(f"Val Loss:   {history['val_loss'][best_epoch - 1]:.6f}\n\n")
            
            metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc', 'specificity', 'balanced_accuracy']
            for metric in metrics:
                train_key = f'train_{metric}'
                val_key = f'val_{metric}'
                if val_key in history:
                    train_val = history[train_key][best_epoch - 1] if train_key in history else None
                    val_val = history[val_key][best_epoch - 1]
                    
                    if train_val is not None:
                        f.write(f"Train {metric.capitalize()}: {train_val:.6f}\n")
                    f.write(f"Val {metric.capitalize()}:   {val_val:.6f}\n")
        
        f.write("\n" + "-"*80 + "\n")
        f.write("PERFORMANCE FINALE\n")
        f.write("-"*80 + "\n")
        
        f.write(f"\nÉpoch: {num_epochs}\n")
        f.write(f"Train Loss: {history['train_loss'][-1]:.6f}\n")
        f.write(f"Val Loss:   {history['val_loss'][-1]:.6f}\n\n")
        
        for metric in metrics:
            train_key = f'train_{metric}'
            val_key = f'val_{metric}'
            if val_key in history:
                train_val = history[train_key][-1] if train_key in history else None
                val_val = history[val_key][-1]
                
                if train_val is not None:
                    f.write(f"Train {metric.capitalize()}: {train_val:.6f}\n")
                f.write(f"Val {metric.capitalize()}:   {val_val:.6f}\n")
        
        if num_epochs > 1:
            f.write("\n" + "-"*80 + "\n")
            f.write("AMÉLIORATION\n")
            f.write("-"*80 + "\n\n")
            
            loss_improvement = history['train_loss'][0] - history['train_loss'][-1]
            loss_improvement_pct = (loss_improvement / history['train_loss'][0]) * 100
            f.write(f"Réduction loss: {loss_improvement:.6f} ({loss_improvement_pct:.2f}%)\n")
            
            if 'train_accuracy' in history:
                acc_improvement = history['train_accuracy'][-1] - history['train_accuracy'][0]
                acc_improvement_pct = (acc_improvement / history['train_accuracy'][0]) * 100
                f.write(f"Gain accuracy: {acc_improvement:+.6f} ({acc_improvement_pct:+.2f}%)\n")
        
        f.write("\n" + "="*80 + "\n")
    
    print(f" Rapport détaillé sauvegardé dans '{report_path}'")

def print_training_summary(result: dict):
    """
    Affiche un résumé détaillé de l'entraînement.
    
    Args:
        result: Dictionnaire retourné par trainer.train() contenant:
                - best_metrics: dict des meilleures métriques
                - best_epoch: numéro de la meilleure epoch
                - final_epoch: numéro de la dernière epoch
                - training_time: temps total d'entraînement
                - early_stopped: bool si early stopping activé
                - history: dict contenant tous les historiques
    """
    print("\n" + "="*90)
    print(" RÉSUMÉ DE L'ENTRAÎNEMENT")
    print("="*90)
    
    # Extraire les données
    history = result.get('history', {})
    best_metrics = result.get('best_metrics', {})
    best_epoch = result.get('best_epoch', 0)
    final_epoch = result.get('final_epoch', 0)
    training_time = result.get('training_time', 0)
    early_stopped = result.get('early_stopped', False)
    
    # Informations générales
    print(f"\n📊 Informations générales:")
    print(f"   Epochs effectuées:        {final_epoch}")
    print(f"   Meilleure epoch:          {best_epoch}")
    print(f"   Temps d'entraînement:     {training_time/3600:.2f}h ({training_time/60:.1f}min)")
    print(f"   Early stopping:           {'✅ Oui' if early_stopped else '❌ Non'}")
    
    if 'iteration_losses' in history:
        total_iterations = len(history['iteration_losses'])
        print(f"   Total d'itérations:       {total_iterations:,}")
        if final_epoch > 0:
            print(f"   Itérations par epoch:     {total_iterations // final_epoch:,}")
    
    # Meilleurs résultats (selon la métrique composite ou MCC)
    print(f"\n🏆 MEILLEURS RÉSULTATS (Epoch {best_epoch}):")
    print(f"{'='*90}")
    
    # Métriques principales
    print(f"\n  📈 Métriques Principales:")
    if 'composite_score' in best_metrics:
        print(f"     Composite Score:      {best_metrics['composite_score']:.4f} ⭐")
    if 'mcc' in best_metrics:
        print(f"     MCC:                  {best_metrics['mcc']:.4f}")
    if 'g_mean' in best_metrics:
        print(f"     G-Mean:               {best_metrics['g_mean']:.4f}")
    if 'stability_score' in best_metrics:
        print(f"     Stability Score:      {best_metrics['stability_score']:.4f}")
    if 'production_score' in best_metrics:
        print(f"     Production Score:     {best_metrics['production_score']:.4f}")
    if 'f_harmonic' in best_metrics:
        print(f"     F-Harmonic:           {best_metrics['f_harmonic']:.4f}")
    
    # Métriques classiques de validation
    if 'val_accuracy' in history and len(history['val_accuracy']) >= best_epoch:
        print(f"\n  📊 Métriques Classiques:")
        idx = best_epoch - 1
        
        metrics_map = {
            'Accuracy': 'val_accuracy',
            'Balanced Acc': 'val_balanced_accuracy',
            'Precision': 'val_precision',
            'Recall': 'val_recall',
            'Specificity': 'val_specificity',
            'F1-Score': 'val_f1',
            'F2-Score': 'val_f2',
            'IoU': 'val_iou',
            "Cohen's Kappa": 'val_cohen_kappa'
        }
        
        for label, key in metrics_map.items():
            if key in history and idx < len(history[key]):
                print(f"     {label:18s} {history[key][idx]:.4f}")
    
    # Métriques probabilistes
    if 'auroc' in best_metrics and best_metrics['auroc'] > 0:
        print(f"\n  🎲 Métriques Probabilistes:")
        print(f"     AUROC:                {best_metrics['auroc']:.4f}")
        if 'val_auprc' in history and len(history['val_auprc']) >= best_epoch:
            print(f"     AUPRC:                {history['val_auprc'][best_epoch-1]:.4f}")
        if 'val_brier_score' in history and len(history['val_brier_score']) >= best_epoch:
            print(f"     Brier Score:          {history['val_brier_score'][best_epoch-1]:.4f}")
    
    # Métriques par classe
    if 'val_class_0_recall' in history and len(history['val_class_0_recall']) >= best_epoch:
        print(f"\n  🎯 Métriques par Classe:")
        idx = best_epoch - 1
        
        if 'val_class_0_precision' in history:
            print(f"     Classe 0 - Precision: {history['val_class_0_precision'][idx]:.4f}")
        print(f"     Classe 0 - Recall:    {history['val_class_0_recall'][idx]:.4f}")
        
        if 'val_class_1_precision' in history:
            print(f"     Classe 1 - Precision: {history['val_class_1_precision'][idx]:.4f}")
        if 'val_class_1_recall' in history:
            print(f"     Classe 1 - Recall:    {history['val_class_1_recall'][idx]:.4f}")
        
        # Min class recall et balance gap
        if 'val_min_class_recalls' in history and len(history['val_min_class_recalls']) >= best_epoch:
            print(f"     Min Class Recall:     {history['val_min_class_recalls'][idx]:.4f}")
        if 'val_class_balance_gaps' in history and len(history['val_class_balance_gaps']) >= best_epoch:
            print(f"     Class Balance Gap:    {history['val_class_balance_gaps'][idx]:.4f}")
    
    # Support
    if 'val_support_class_0' in history and len(history['val_support_class_0']) >= best_epoch:
        print(f"\n  📊 Support:")
        idx = best_epoch - 1
        print(f"     Classe 0:             {int(history['val_support_class_0'][idx])}")
        if 'val_support_class_1' in history:
            print(f"     Classe 1:             {int(history['val_support_class_1'][idx])}")
    
    # Matrice de confusion
    if all(k in history for k in ['val_tp', 'val_tn', 'val_fp', 'val_fn']):
        if len(history['val_tp']) >= best_epoch:
            print(f"\n  📋 Matrice de Confusion:")
            idx = best_epoch - 1
            tp = int(history['val_tp'][idx])
            tn = int(history['val_tn'][idx])
            fp = int(history['val_fp'][idx])
            fn = int(history['val_fn'][idx])
            
            print(f"     TP: {tp:5d}  |  FP: {fp:5d}")
            print(f"     FN: {fn:5d}  |  TN: {tn:5d}")
    
    # Évolution des pertes
    if 'train_loss' in history and 'val_loss' in history:
        if len(history['train_loss']) > 0 and len(history['val_loss']) > 0:
            print(f"\n📉 ÉVOLUTION DES PERTES:")
            print(f"{'='*90}")
            
            print(f"\n  Première epoch:")
            print(f"     Train Loss:           {history['train_loss'][0]:.4f}")
            print(f"     Val Loss:             {history['val_loss'][0]:.4f}")
            
            print(f"\n  Dernière epoch:")
            print(f"     Train Loss:           {history['train_loss'][-1]:.4f}")
            print(f"     Val Loss:             {history['val_loss'][-1]:.4f}")
            
            print(f"\n  Meilleure epoch ({best_epoch}):")
            idx = best_epoch - 1
            if idx < len(history['train_loss']):
                print(f"     Train Loss:           {history['train_loss'][idx]:.4f}")
            if idx < len(history['val_loss']):
                print(f"     Val Loss:             {history['val_loss'][idx]:.4f}")
            
            # Amélioration
            if len(history['train_loss']) > 1:
                loss_improvement = history['train_loss'][0] - history['train_loss'][-1]
                loss_improvement_pct = (loss_improvement / history['train_loss'][0]) * 100
                
                print(f"\n  💡 Amélioration:")
                print(f"     Réduction loss:       {loss_improvement:.4f} ({loss_improvement_pct:.2f}%)")
    
    # Évolution des métriques composites
    if 'val_composite_scores' in history and len(history['val_composite_scores']) > 0:
        print(f"\n📊 ÉVOLUTION DES SCORES COMPOSITES:")
        print(f"{'='*90}")
        
        composite_scores = history['val_composite_scores']
        print(f"\n  Composite Score:")
        print(f"     Premier:              {composite_scores[0]:.4f}")
        print(f"     Meilleur:             {max(composite_scores):.4f}")
        print(f"     Dernier:              {composite_scores[-1]:.4f}")
        
        if len(composite_scores) > 1:
            improvement = composite_scores[-1] - composite_scores[0]
            print(f"     Amélioration totale:  {improvement:+.4f}")
    
    if 'val_g_means' in history and len(history['val_g_means']) > 0:
        g_means = history['val_g_means']
        print(f"\n  G-Mean:")
        print(f"     Premier:              {g_means[0]:.4f}")
        print(f"     Meilleur:             {max(g_means):.4f}")
        print(f"     Dernier:              {g_means[-1]:.4f}")
    
    print(f"\n{'='*90}\n")


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
    result = trainer.train()  # ⚠️ Changement: 'result' au lieu de 'history'

    # Affichage et sauvegarde des résultats
    print_training_summary(result)  # ⚠️ On passe 'result'

    # Extraction de l'historique pour la sauvegarde
    history = result.get('history', {})

    # Sauvegarde finale de l'historique JSON
    out_json = Path(cfg.results_folder) / 'training_history.json'
    out_json.parent.mkdir(parents=True, exist_ok=True)

    # Sauvegarder tout le résultat (pas juste l'history)
    with open(out_json, 'w') as f:
        json.dump(result, f, indent=2, cls=NumpyEncoder)

    # Sauvegarde CSV et graphiques (utilise l'history extrait)
    save_training_results(history, cfg.results_folder)
    plot_training_results(history, cfg.results_folder)
    
    print(f"\n✅ Résultats sauvegardés dans: {cfg.results_folder}")
    print(f"   - training_history.json")
    print(f"   - training_results.csv")
    print(f"   - Graphiques PNG\n")


if __name__ == '__main__':
    main()
