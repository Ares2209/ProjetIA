"""Script d'inférence pour le projet Exoplanet.

Charge un checkpoint et évalue sur un ensemble de données complet.
Génère une matrice de confusion et des métriques de performance.

Usage:
    python inference.py --checkpoint models/ResNetCNN/checkpoints/resnet_modelbest.pth

Exemple avec targets pour évaluation:
    python inference.py \
      --checkpoint models/ResNetCNN/checkpoints/resnet_modelbest.pth \
      --spectra Défi-IA-2026/DATA/defi-ia-cnes/spectra.npy \
      --auxiliary Défi-IA-2026/DATA/defi-ia-cnes/auxiliary.csv \
      --targets Défi-IA-2026/DATA/defi-ia-cnes/targets.csv \
      --batch-size 64 \
      --output models/ResNetCNN/results/

Exemple sans targets (prédictions seulement):
    python inference.py \
      --checkpoint models/ResNetCNN/checkpoints/resnet_modelbest.pth \
      --spectra Défi-IA-2026/DATA/defi-ia-cnes/spectra_test.npy \
      --auxiliary Défi-IA-2026/DATA/defi-ia-cnes/auxiliary_test.csv \
      --output models/ResNetCNN/results/predictions.csv
"""
import argparse
import json
import pickle
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score
import matplotlib.pyplot as plt
try:
    import seaborn as sns
except ImportError:
    sns = None

from models.CNN import CNN
from models.LightGBM.LightGBM import LABEL_NAMES, _as_frame, _build_tabular_features
from models.ResNetCNN import ResNet1D, ensemble_resnet_1d, resnet8_1d, resnet18_1d, resnet34_1d
from models.dataset import ExoplanetDataset, collate_fn
from training.config import Config, get_config_object


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
    if any(k.startswith('resnet8.') or k.startswith('resnet18.') for k in keys):
        return 'ensemble_resnet'
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

    if sns is not None:
        sns.heatmap(
            cm_eau,
            annot=True, fmt='d', cmap='Blues',
            xticklabels=['Absent', 'Présent'],
            yticklabels=['Absent', 'Présent'],
            ax=axes[0],
            cbar_kws={'label': 'Nombre d\'échantillons'}
        )
    else:
        axes[0].imshow(cm_eau, cmap='Blues')
        for i in range(cm_eau.shape[0]):
            for j in range(cm_eau.shape[1]):
                axes[0].text(j, i, str(cm_eau[i, j]), ha='center', va='center')
        axes[0].set_xticks([0, 1], ['Absent', 'Présent'])
        axes[0].set_yticks([0, 1], ['Absent', 'Présent'])
    axes[0].set_title(f'Matrice de Confusion - EAU{title_suffix}')
    axes[0].set_ylabel('Vérité Terrain')
    axes[0].set_xlabel('Prédiction')

    if sns is not None:
        sns.heatmap(
            cm_nuage,
            annot=True, fmt='d', cmap='Oranges',
            xticklabels=['Absent', 'Présent'],
            yticklabels=['Absent', 'Présent'],
            ax=axes[1],
            cbar_kws={'label': 'Nombre d\'échantillons'}
        )
    else:
        axes[1].imshow(cm_nuage, cmap='Oranges')
        for i in range(cm_nuage.shape[0]):
            for j in range(cm_nuage.shape[1]):
                axes[1].text(j, i, str(cm_nuage[i, j]), ha='center', va='center')
        axes[1].set_xticks([0, 1], ['Absent', 'Présent'])
        axes[1].set_yticks([0, 1], ['Absent', 'Présent'])
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


def run_inference_tta(
    model: torch.nn.Module,
    raw_spectra: np.ndarray,
    aux_df: pd.DataFrame,
    targets_df: Optional[pd.DataFrame],
    base_dataset: ExoplanetDataset,
    device: torch.device,
    n_aug: int = 10,
    batch_size: int = 64,
    threshold: float = 0.5,
    weight_orig: float = 0.3,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], np.ndarray]:
    """Test-Time Augmentation : 1 pass original + N pass augmentées, moyenne pondérée.

    Args:
        base_dataset : dataset non-augmenté (déjà construit avec stats train)
                       — utilisé pour récupérer aux_mean, aux_std, spectra_mean, spectra_std
                       et pour la passe sans augmentation.
        n_aug        : nombre de passes augmentées (10-20 typique).
        weight_orig  : poids de la passe originale dans la moyenne finale.
                       proba_finale = weight_orig * proba_orig + (1-weight_orig) * mean(proba_aug).

    Returns:
        (probabilities, predictions, true_labels_or_None, ids)
    """
    has_targets = targets_df is not None

    # 1) Passe originale (sans augmentation)
    base_loader = DataLoader(
        base_dataset, batch_size=batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=0,
    )
    probs_orig, _, true_labels, ids = run_inference(
        model, base_loader, device, threshold=threshold, has_targets=has_targets,
    )

    if n_aug <= 0:
        return probs_orig, (probs_orig >= threshold).astype(int), true_labels, ids

    # 2) Dataset augmenté (réutilise les stats du base_dataset → cohérence)
    aug_dataset = ExoplanetDataset(
        spectra              = raw_spectra,
        auxiliary_df         = aux_df,
        targets_df           = targets_df,
        is_train             = True,
        augmentation_factor  = n_aug,
        aux_mean             = base_dataset.aux_mean,
        aux_std              = base_dataset.aux_std,
        spectra_mean         = base_dataset.spectra_mean,
        spectra_std          = base_dataset.spectra_std,
    )

    # On accède uniquement aux versions augmentées (idx >= N)
    N = aug_dataset.original_size
    aug_indices = list(range(N, N * (1 + n_aug)))
    aug_loader = DataLoader(
        torch.utils.data.Subset(aug_dataset, aug_indices),
        batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=0,
    )

    probs_aug, _, _, _ = run_inference(
        model, aug_loader, device, threshold=threshold, has_targets=has_targets,
    )
    # Layout : (n_aug × N, 2) où block k = pass augmentée k. Moyenne par sample.
    probs_aug = probs_aug.reshape(n_aug, N, 2).mean(axis=0)

    # 3) Moyenne pondérée
    probabilities = weight_orig * probs_orig + (1.0 - weight_orig) * probs_aug
    predictions   = (probabilities >= threshold).astype(int)
    return probabilities, predictions, true_labels, ids


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
        'pred_eau':   predictions[:, 0],
        'pred_nuage': predictions[:, 1],
    })

    if true_labels is not None:
        df.insert(1, 'prob_eau',   probabilities[:, 0])
        df.insert(2, 'prob_nuage', probabilities[:, 1])
        df['true_eau']      = true_labels[:, 0]
        df['true_nuage']    = true_labels[:, 1]
        df['correct_eau']   = (df['pred_eau']   == df['true_eau']).astype(int)
        df['correct_nuage'] = (df['pred_nuage'] == df['true_nuage']).astype(int)
        df['both_correct']  = (
            (df['correct_eau'] == 1) & (df['correct_nuage'] == 1)
        ).astype(int)

    output_path = output_dir / f"inference_res_{datetime.now():%Y-%m-%d}.csv"
    df.to_csv(output_path, index=False)
    print(f"[SAVE] Predictions sauvegardées : {output_path}")
    return df


def _config_from_checkpoint(checkpoint: dict) -> Config:
    cfg_dict = checkpoint.get('config')
    if cfg_dict:
        return Config.from_dict(cfg_dict)
    return get_config_object()


def _build_torch_model(model_type: str, checkpoint: dict, channels: int,
                       length: int, auxiliary_dim: int):
    cfg = _config_from_checkpoint(checkpoint)
    model_cfg = cfg.model
    architecture = getattr(model_cfg, 'architecture', '')
    dropout = getattr(model_cfg, 'dropout', 0.3)

    if model_type == 'ensemble_resnet' or architecture == '2ResNet':
        return ensemble_resnet_1d(
            spectrum_length=length,
            auxiliary_dim=auxiliary_dim,
            num_classes=2,
            input_channels=channels,
            dropout=dropout,
        )

    if model_type == 'resnet':
        if architecture == 'ResNet8':
            return resnet8_1d(length, auxiliary_dim, 2, channels, dropout=dropout)
        if architecture == 'ResNet34':
            return resnet34_1d(length, auxiliary_dim, 2, channels, dropout=dropout)
        return resnet18_1d(length, auxiliary_dim, 2, channels, dropout=dropout)

    return CNN(
        spectrum_length=length,
        auxiliary_dim=auxiliary_dim,
        num_classes=2,
        input_channels=channels,
        conv_channels=getattr(model_cfg, 'conv_channels', [32, 64, 128, 256]),
        kernel_sizes=getattr(model_cfg, 'kernel_sizes', [7, 5, 3, 3]),
        pool_sizes=getattr(model_cfg, 'pool_sizes', [2, 2, 2, 2]),
        fc_dims=getattr(model_cfg, 'fc_dims', [256, 128]),
        dropout=dropout,
        use_batch_norm=getattr(model_cfg, 'use_batch_norm', True),
    )


def load_lightgbm_artifact(checkpoint_path: str) -> dict:
    with open(checkpoint_path, 'rb') as f:
        return pickle.load(f)


def build_lightgbm_dataset(
    artifact: dict,
    spectra_np: np.ndarray,
    aux_df: pd.DataFrame,
    targets_df: Optional[pd.DataFrame],
) -> ExoplanetDataset:
    """Reconstruit le dataset d'inférence en réutilisant les stats du train sauvegardées."""
    aux_mean     = artifact.get('aux_mean')
    aux_std      = artifact.get('aux_std')
    spectra_mean = artifact.get('spectra_mean')
    spectra_std  = artifact.get('spectra_std')

    if any(s is None for s in (aux_mean, aux_std, spectra_mean, spectra_std)):
        print("   [WARNING] Stats train absentes du checkpoint LightGBM : "
              "recalcul sur le split d'inférence (résultats potentiellement dégradés).")

    return ExoplanetDataset(
        spectra              = spectra_np,
        auxiliary_df         = aux_df,
        targets_df           = targets_df,
        is_train             = False,
        augmentation_factor  = 0,
        aux_mean             = aux_mean,
        aux_std              = aux_std,
        spectra_mean         = spectra_mean,
        spectra_std          = spectra_std,
    )


def run_lightgbm_inference(
    artifact: dict,
    dataset: ExoplanetDataset,
    threshold: float,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], np.ndarray]:
    cfg = Config.from_dict(artifact['config'])
    features, labels, ids, feature_names = _build_tabular_features(dataset, cfg)

    pca = artifact.get('pca')
    if pca is not None:
        features = pca.transform(features)
        feature_names = artifact['feature_names']

    if feature_names != artifact['feature_names']:
        raise RuntimeError("Features d'inférence incompatibles avec le checkpoint LightGBM.")

    frame = _as_frame(features, artifact['feature_names'])
    probabilities = np.zeros((features.shape[0], len(LABEL_NAMES)), dtype=np.float32)
    for idx, label_name in enumerate(LABEL_NAMES):
        probabilities[:, idx] = artifact['models'][label_name].predict_proba(frame)[:, 1]

    predictions = (probabilities >= threshold).astype(int)
    return probabilities, predictions, labels, ids


def main():
    parser = argparse.ArgumentParser(description='Inference script with confusion matrix')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Chemin vers le checkpoint .pth ou .pkl')
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
    parser.add_argument('--tta', type=int, default=0,
                        help="N passes Test-Time Augmentation (0 = désactivé)")
    parser.add_argument('--tta-weight-orig', type=float, default=0.3,
                        help="Poids de la passe originale dans la moyenne TTA")
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

    is_lightgbm = Path(args.checkpoint).suffix == '.pkl'

    # ── Chargement du checkpoint + inférence ─────────────────────────────────
    print("\n[LOAD] Chargement du checkpoint...")
    if is_lightgbm:
        print("   - Type de modèle détecté : LIGHTGBM")
        artifact = load_lightgbm_artifact(args.checkpoint)
        dataset = build_lightgbm_dataset(artifact, spectra_np, aux_df, targets_df)
        print(f"   [OK] {len(dataset)} exemples chargés")

        print("\n[INFERENCE] Inférence LightGBM en cours...")
        probabilities, predictions, true_labels, ids = run_lightgbm_inference(
            artifact,
            dataset,
            args.threshold,
        )
    else:
        dataset = ExoplanetDataset(
            spectra              = spectra_np,
            auxiliary_df         = aux_df,
            targets_df           = targets_df,
            is_train             = False,
            augmentation_factor  = 0,
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

        model = _build_torch_model(model_type, checkpoint, channels, length, auxiliary_dim)
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

        # ── Inférence PyTorch (avec ou sans TTA) ──────────────────────────────
        if args.tta and args.tta > 0:
            print(f"\n[INFERENCE] TTA activé : 1 pass original + {args.tta} pass augmentées "
                  f"(weight_orig={args.tta_weight_orig})")
            probabilities, predictions, true_labels, ids = run_inference_tta(
                model        = model,
                raw_spectra  = spectra_np,
                aux_df       = aux_df,
                targets_df   = targets_df,
                base_dataset = dataset,
                device       = device,
                n_aug        = args.tta,
                batch_size   = args.batch_size,
                threshold    = args.threshold,
                weight_orig  = args.tta_weight_orig,
            )
        else:
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
