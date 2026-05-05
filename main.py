#!/usr/bin/env python3
"""Script d'entraînement principal.

Usage: python main.py [--config PATH]
"""
import argparse
import warnings
from pathlib import Path
import json
import numpy as np
import pandas as pd

from training.config import Config
from training.training import Trainer
from training.utils import set_seed
from training.plot import plot_training_results
from models.dataset import ExoplanetDataset, collate_fn
from sklearn.model_selection import train_test_split
from models.CNN import CNN
from models.ResNetCNN import ResNet1D, resnet18_1d, resnet34_1d, resnet8_1d, ensemble_resnet_1d
from torch.utils.data import DataLoader

ARCH_TO_CONFIG = {
    'CNN':       'models.CNN.config',
    'ResNetCNN': 'models.ResNetCNN.config',
    'XGBoost':   'models.XGBoost.config',
}


def load_config_for_arch(arch: str) -> Config:
    """Charge la config presets du modèle depuis models/<arch>/config.py."""
    import importlib
    if arch not in ARCH_TO_CONFIG:
        raise ValueError(f"--arch invalide : '{arch}'. Choix : {list(ARCH_TO_CONFIG)}")
    return importlib.import_module(ARCH_TO_CONFIG[arch]).get_config()


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
        'input_channels':  train_dataset.spectra.shape[2],
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
        input_channels  = cfg.model.input_channels,
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
    elif arch == "2ResNet":
        print("  Architecture : Ensemble SE-ResNet8 + ResNet18")
        return ensemble_resnet_1d(**common_torch)

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
    print("  Trainer : Trainer (PyTorch)")
    return Trainer(model, train_loader, val_loader, cfg)


# ─────────────────────────────────────────────────────────────────────────────
# HELPER COMMUN
# ─────────────────────────────────────────────────────────────────────────────

def _get(history: dict, key: str) -> list:
    """Retourne history[key] ou une liste vide si absent."""
    return history.get(key, [])


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
    parser.add_argument('--config', type=str, default=None, help="Chemin vers la config JSON")
    parser.add_argument('--arch', type=str, default='ResNetCNN',
                        choices=list(ARCH_TO_CONFIG.keys()),
                        help="Modèle à entraîner — charge models/<arch>/config.py")
    args = parser.parse_args()

    cfg = Config.load(args.config) if args.config else load_config_for_arch(args.arch)
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