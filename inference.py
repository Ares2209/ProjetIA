"""Script d'inférence pour le projet Exoplanet.

Charge un checkpoint et évalue sur un ensemble de données complet.
Génère une matrice de confusion et des métriques de performance.

Usage:
    python inference.py --checkpoint checkpoints/exoplanet_modelbest.pth

Exemple avec targets pour évaluation:
    python inference.py \
      --checkpoint checkpoints/exoplanet_modelbest.pth \
      --spectra Défi-IA-2026/DATA/defi-ia-cnes/spectra.npy \
      --auxiliary Défi-IA-2026/DATA/defi-ia-cnes/auxiliary.csv \
      --targets Défi-IA-2026/DATA/defi-ia-cnes/targets.csv \
      --batch-size 64 \
      --output results/

Exemple sans targets (prédictions seulement):
    python inference.py \
      --checkpoint checkpoints/exoplanet_modelbest.pth \
      --spectra Défi-IA-2026/DATA/defi-ia-cnes/spectra_test.npy \
      --auxiliary Défi-IA-2026/DATA/defi-ia-cnes/auxiliary_test.csv \
      --output results/predictions.csv
"""
import argparse
import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

from models.CNN import CNN
from models.ResNetCNN import ResNet1D
from models.dataset import ExoplanetDataset, collate_fn
from training.config import get_config_object


def remove_module_prefix(state_dict: dict) -> dict:
    """Retire le préfixe 'module.' si présent (sauvegarde DataParallel)."""
    new_state = {}
    for k, v in state_dict.items():
        new_key = k[len('module.'):] if k.startswith('module.') else k
        new_state[new_key] = v
    return new_state


def detect_model_class(state_dict: dict) -> str:
    """Detecte la famille de modèle à partir des clés du state dict."""
    keys = list(state_dict.keys())
    if any(k.startswith('stem.') or 'stage' in k for k in keys):
        return 'resnet'
    return 'cnn'


def load_checkpoint(checkpoint_path: str, device: str = 'cpu') -> dict:
    """Charge un checkpoint PyTorch de manière compatible."""
    try:
        import torch.serialization
        try:
            from numpy._core.multiarray import scalar as np_scalar
        except ImportError:
            from numpy.core.multiarray import scalar as np_scalar
        torch.serialization.add_safe_globals([np_scalar])
        checkpoint = torch.load(checkpoint_path, map_location=device)
    except Exception:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    return checkpoint


def plot_confusion_matrices(
    cm_eau: np.ndarray,
    cm_nuage: np.ndarray,
    output_dir: Path,
    title_suffix: str = ""
):
    """Affiche et sauvegarde les matrices de confusion."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    sns.heatmap(
        cm_eau,
        annot=True, fmt='d', cmap='Blues',
        xticklabels=['Absent', 'Présent'],
        yticklabels=['Absent', 'Présent'],
        ax=axes[0],
        cbar_kws={'label': 'Nombre d\'échantillons'}
    )
    axes[0].set_title(f'Matrice de Confusion - EAU{title_suffix}')
    axes[0].set_ylabel('Vérité Terrain')
    axes[0].set_xlabel('Prédiction')

    sns.heatmap(
        cm_nuage,
        annot=True, fmt='d', cmap='Oranges',
        xticklabels=['Absent', 'Présent'],
        yticklabels=['Absent', 'Présent'],
        ax=axes[1],
        cbar_kws={'label': 'Nombre d\'échantillons'}
    )
    axes[1].set_title(f'Matrice de Confusion - NUAGES{title_suffix}')
    axes[1].set_ylabel('Vérité Terrain')
    axes[1].set_xlabel('Prédiction')

    plt.tight_layout()
    output_path = output_dir / 'confusion_matrices.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n[SAVE] Matrices de confusion sauvegardées : {output_path}")
    plt.show()


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, label_name: str) -> Dict:
    """Calcule les métriques pour une tâche binaire."""
    cm = confusion_matrix(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='binary')

    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

    precision   = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall      = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return {
        'label':            label_name,
        'confusion_matrix': cm.tolist(),
        'accuracy':         float(accuracy),
        'f1_score':         float(f1),
        'precision':        float(precision),
        'recall':           float(recall),
        'specificity':      float(specificity),
        'true_positive':    int(tp),
        'true_negative':    int(tn),
        'false_positive':   int(fp),
        'false_negative':   int(fn),
    }


def run_inference(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    threshold: float = 0.5,
    has_targets: bool = True
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """Exécute l'inférence sur un dataset complet."""
    model.eval()

    all_probs   = []
    all_ids     = []
    all_targets = [] if has_targets else None

    with torch.no_grad():
        for batch_data in dataloader:
            if has_targets:
                spectra, auxiliary, targets, ids = batch_data
                all_targets.append(targets.cpu().numpy())
            else:
                spectra, auxiliary, ids = batch_data

            spectra   = spectra.to(device)
            auxiliary = auxiliary.to(device)

            logits = model(spectra, auxiliary)          # (B, 2)
            probs  = torch.sigmoid(logits).cpu().numpy()

            all_probs.append(probs)
            all_ids.append(ids.cpu().numpy())

    probabilities = np.vstack(all_probs)                # (N, 2)
    predictions   = (probabilities >= threshold).astype(int)
    ids           = np.concatenate(all_ids)
    true_labels   = np.vstack(all_targets) if has_targets else None

    return probabilities, predictions, true_labels, ids


def save_predictions(
    ids: np.ndarray,
    probabilities: np.ndarray,
    predictions: np.ndarray,
    output_dir: Path,
    true_labels: Optional[np.ndarray] = None
):
    """Sauvegarde les prédictions dans inference_res.csv."""
    df = pd.DataFrame({
        'id':         ids,
        'prob_eau':   probabilities[:, 0],
        'prob_nuage': probabilities[:, 1],
        'pred_eau':   predictions[:, 0],
        'pred_nuage': predictions[:, 1],
    })

    if true_labels is not None:
        df['true_eau']      = true_labels[:, 0]
        df['true_nuage']    = true_labels[:, 1]
        df['correct_eau']   = (df['pred_eau']   == df['true_eau']).astype(int)
        df['correct_nuage'] = (df['pred_nuage'] == df['true_nuage']).astype(int)
        df['both_correct']  = (
            (df['correct_eau'] == 1) & (df['correct_nuage'] == 1)
        ).astype(int)

    output_path = output_dir / 'inference_res.csv'
    df.to_csv(output_path, index=False)
    print(f"[SAVE] Predictions sauvegardées : {output_path}")
    return df


def main():
    parser = argparse.ArgumentParser(description='Inference script with confusion matrix')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Chemin vers le checkpoint .pth')
    parser.add_argument('--spectra', type=str,
                        default='Défi-IA-2026/DATA/defi-ia-cnes/spectra.npy')
    parser.add_argument('--auxiliary', type=str,
                        default='Défi-IA-2026/DATA/defi-ia-cnes/auxiliary.csv')
    parser.add_argument('--targets', type=str, default=None,
                        help='Chemin vers targets.csv (optionnel)')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--output', type=str, default='results/')
    parser.add_argument('--no-plot', action='store_true')
    args = parser.parse_args()

    device     = torch.device(
        args.device
        if torch.cuda.is_available() and args.device.startswith('cuda')
        else 'cpu'
    )
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print('\n' + '='*60)
    print(' SCRIPT D\'INFERENCE - EXOPLANET')
    print('='*60)
    print(f" Checkpoint : {args.checkpoint}")
    print(f" Spectra    : {args.spectra}")
    print(f" Auxiliary  : {args.auxiliary}")
    print(f" Targets    : {args.targets if args.targets else 'Non fourni (mode prédiction)'}")
    print(f" Device     : {device}")
    print(f" Batch size : {args.batch_size}")
    print(f" Seuil      : {args.threshold}")
    print(f" Sortie     : {output_dir}")
    print('='*60)

    # ── Vérifications ────────────────────────────────────────────────────────
    if not Path(args.checkpoint).exists():
        raise FileNotFoundError(f"Checkpoint introuvable: {args.checkpoint}")
    if not Path(args.spectra).exists():
        raise FileNotFoundError(f"Fichier spectra introuvable: {args.spectra}")
    if not Path(args.auxiliary).exists():
        raise FileNotFoundError(f"Fichier auxiliary introuvable: {args.auxiliary}")

    has_targets = args.targets is not None and Path(args.targets).exists()

    # ── Chargement des données brutes ─────────────────────────────────────────
    print("\n[LOAD] Chargement des données...")
    spectra_np = np.load(args.spectra)                          # (N, 52, 3)
    aux_df     = pd.read_csv(args.auxiliary)
    targets_df = pd.read_csv(args.targets) if has_targets else None

    # ── Construction du dataset ───────────────────────────────────────────────
    # is_train=False  → pas d'augmentation, normalisation per-sample
    # aux_mean/std=None → stats calculées sur ce dataset (pas de train dispo)
    dataset = ExoplanetDataset(
        spectra              = spectra_np,
        auxiliary_df         = aux_df,
        targets_df           = targets_df,
        is_train             = False,
        augmentation_factor  = 0,
        # Pas de stats externes : per-sample pour les spectres,
        # et normalisation auxiliaire autonome (inférence standalone)
        aux_mean             = None,
        aux_std              = None,
    )
    print(f"   [OK] {len(dataset)} exemples chargés")

    dataloader = DataLoader(
        dataset,
        batch_size  = args.batch_size,
        shuffle     = False,
        collate_fn  = collate_fn,
        num_workers = 0,
    )

    # ── Chargement du checkpoint ──────────────────────────────────────────────
    print("\n[LOAD] Chargement du checkpoint...")
    checkpoint  = load_checkpoint(args.checkpoint, str(device))
    state_dict  = checkpoint.get('model_state_dict', checkpoint)
    state_dict  = remove_module_prefix(state_dict)
    model_type  = detect_model_class(state_dict)
    print(f"   - Type de modèle détecté : {model_type.upper()}")

    # Inférer les dimensions depuis un batch
    sample_batch = next(iter(dataloader))
    spectrum, auxiliary = sample_batch[0], sample_batch[1]
    _, channels, length = spectrum.shape
    auxiliary_dim       = auxiliary.shape[1]

    # ── Instanciation du modèle ───────────────────────────────────────────────
    cfg = get_config_object()

    if model_type.lower() == 'resnet':
        model = ResNet1D(
            spectrum_length = length,
            auxiliary_dim   = auxiliary_dim,
            num_classes     = 2,
            input_channels  = channels,
            block_type      = 'basic',
            num_blocks      = [2, 2, 2, 2],
            base_channels   = 64,
            dropout         = cfg.model.dropout,
        )
    else:
        model = CNN(
            spectrum_length = length,
            auxiliary_dim   = auxiliary_dim,
            num_classes     = 2,
            input_channels  = channels,
        )

    try:
        model.load_state_dict(state_dict, strict=True)
        print("   [OK] Poids chargés (strict)")
    except RuntimeError:
        model.load_state_dict(state_dict, strict=False)
        print("   [WARNING] Poids chargés (non-strict)")

    model = model.to(device)

    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   - Paramètres totaux       : {total_params:,}")
    print(f"   - Paramètres entraînables : {trainable_params:,}")

    # ── Inférence ─────────────────────────────────────────────────────────────
    print("\n[INFERENCE] Inférence en cours...")
    probabilities, predictions, true_labels, ids = run_inference(
        model       = model,
        dataloader  = dataloader,
        device      = device,
        threshold   = args.threshold,
        has_targets = has_targets,
    )
    print(f"   [OK] Inférence terminée sur {len(ids)} exemples")

    df_results = save_predictions(ids, probabilities, predictions, output_dir, true_labels)

    # ── Évaluation (si targets disponibles) ───────────────────────────────────
    if has_targets and true_labels is not None:
        print("\n" + "="*60)
        print(" EVALUATION DES PERFORMANCES")
        print("="*60)

        metrics_eau   = compute_metrics(true_labels[:, 0], predictions[:, 0], 'EAU')
        metrics_nuage = compute_metrics(true_labels[:, 1], predictions[:, 1], 'NUAGES')

        print("\n[EAU]")
        print(f"   Accuracy    : {metrics_eau['accuracy']:.4f}")
        print(f"   F1-Score    : {metrics_eau['f1_score']:.4f}")
        print(f"   Precision   : {metrics_eau['precision']:.4f}")
        print(f"   Recall      : {metrics_eau['recall']:.4f}")
        print(f"   Specificity : {metrics_eau['specificity']:.4f}")

        print("\n[NUAGES]")
        print(f"   Accuracy    : {metrics_nuage['accuracy']:.4f}")
        print(f"   F1-Score    : {metrics_nuage['f1_score']:.4f}")
        print(f"   Precision   : {metrics_nuage['precision']:.4f}")
        print(f"   Recall      : {metrics_nuage['recall']:.4f}")
        print(f"   Specificity : {metrics_nuage['specificity']:.4f}")

        both_correct        = np.all(predictions == true_labels, axis=1)
        exact_match_accuracy = both_correct.mean()
        print(f"\n[GLOBAL] Exactitude globale (2 labels corrects) : {exact_match_accuracy:.4f}")

        metrics = {
            'threshold':             args.threshold,
            'total_samples':         int(len(ids)),
            'exact_match_accuracy':  float(exact_match_accuracy),
            'eau':                   metrics_eau,
            'nuages':                metrics_nuage,
        }
        metrics_path = output_dir / 'metrics.json'
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        print(f"\n[SAVE] Métriques sauvegardées : {metrics_path}")

        if not args.no_plot:
            cm_eau   = np.array(metrics_eau['confusion_matrix'])
            cm_nuage = np.array(metrics_nuage['confusion_matrix'])
            plot_confusion_matrices(cm_eau, cm_nuage, output_dir)

        print("\n" + "="*60)
        print(" EVALUATION TERMINEE")
        print("="*60)

    else:
        print("\n" + "="*60)
        print(" Mode prédiction uniquement (pas de targets fournis)")
        print("="*60)

    print(f"\n[INFO] Tous les résultats sont dans : {output_dir}")
    print(f"[INFO] Fichier principal : {output_dir / 'inference_res.csv'}")


if __name__ == '__main__':
    main()
