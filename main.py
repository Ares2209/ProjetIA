#!/usr/bin/env python3
"""Script d'entraînement principal.

Usage: python main.py [--arch CNN|ResNetCNN|XGBoost] [--config PATH]
"""
import argparse
import importlib
import json
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from training.config import Config
from training.training import Trainer
from training.utils import (
    NumpyEncoder,
    log_training_summary,
    save_training_results,
    set_seed,
    setup_logging,
)
from training.plot import plot_training_results
from models.dataset import ExoplanetDataset, collate_fn, seed_worker
from models.CNN import CNN
from models.ResNetCNN import ResNet1D, resnet18_1d, resnet34_1d, resnet8_1d, ensemble_resnet_1d


logger = logging.getLogger(__name__)


ARCH_TO_CONFIG = {
    'CNN':       'models.CNN.config',
    'ResNet8':   'models.ResNetCNN.config',
    'ResNet18':  'models.ResNetCNN.config',
    'ResNet34':  'models.ResNetCNN.config',
    'ResNetCNN': 'models.ResNetCNN.config',
    'XGBoost':   'models.XGBoost.config',
    'LightGBM':  'models.LightGBM.config',
}


def load_config_for_arch(arch: str) -> Config:
    """Charge la config presets du modèle depuis models/<arch>/config.py."""
    if arch not in ARCH_TO_CONFIG:
        raise ValueError(f"--arch invalide : '{arch}'. Choix : {list(ARCH_TO_CONFIG)}")
    return importlib.import_module(ARCH_TO_CONFIG[arch]).get_config()


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


def load_raw_data(config):
    """Charge spectres / auxiliaires / targets depuis le disque (paths config + fallback)."""
    data_cfg      = config.data
    fallback_root = Path('Défi-IA-2026') / 'DATA' / 'defi-ia-cnes'

    spectra_path  = _resolve_path(Path(data_cfg.spectra_train_path),  fallback_root / 'spectra.npy',   'spectra')
    aux_path      = _resolve_path(Path(data_cfg.auxiliary_train_path), fallback_root / 'auxiliary.csv', 'auxiliary')
    targets_path  = _resolve_path(Path(data_cfg.targets_train_path),   fallback_root / 'targets.csv',   'targets')

    logger.info("spectra   : %s", spectra_path)
    logger.info("auxiliary : %s", aux_path)
    logger.info("targets   : %s", targets_path)

    return {
        'spectra':    np.load(spectra_path),
        'auxiliary':  pd.read_csv(aux_path),
        'targets':    pd.read_csv(targets_path),
    }


def build_dataloaders_from_indices(data, train_idx, val_idx, config):
    """Construit les DataLoaders train/val à partir d'indices déjà séparés.

    Réutilisable pour split simple ET k-fold (chaque fold appelle ça avec ses indices).
    """
    data_cfg = config.data
    spectra_all    = data['spectra']
    aux_df_all     = data['auxiliary']
    targets_df_all = data['targets']

    spectra_train = spectra_all[train_idx]
    spectra_val   = spectra_all[val_idx]
    aux_train     = aux_df_all.iloc[train_idx].reset_index(drop=True)
    aux_val       = aux_df_all.iloc[val_idx].reset_index(drop=True)
    targets_train = targets_df_all.iloc[train_idx].reset_index(drop=True)
    targets_val   = targets_df_all.iloc[val_idx].reset_index(drop=True)

    logger.info("Train: %d  |  Val: %d", len(train_idx), len(val_idx))

    train_dataset = ExoplanetDataset(
        spectra              = spectra_train,
        auxiliary_df         = aux_train,
        targets_df           = targets_train,
        is_train             = True,
        augmentation_factor  = data_cfg.augmentation_factor,
        shift_range          = data_cfg.shift_range,
        scale_range          = data_cfg.scale_range,
        noise_std            = data_cfg.noise_std,
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
        spectra_mean        = train_dataset.spectra_mean,
        spectra_std         = train_dataset.spectra_std,
    )

    loader_kwargs = dict(
        batch_size     = config.training.batch_size,
        num_workers    = data_cfg.num_workers,
        pin_memory     = data_cfg.pin_memory,
        collate_fn     = collate_fn,
        worker_init_fn = seed_worker,
    )
    train_loader = DataLoader(train_dataset, shuffle=True,  **loader_kwargs)
    val_loader   = DataLoader(val_dataset,   shuffle=False, **loader_kwargs)

    # Stats issues du dataset normalisé (1, 52, 5) — shape garantie par _normalize_spectra
    _, n_lambda, n_channels = train_dataset.spectra_mean.shape
    dataset_info = {
        'spectrum_length': n_lambda,
        'auxiliary_dim':   train_dataset.aux_features.shape[1],
        'input_channels':  n_channels,
        'train_dataset':   train_dataset,
        'val_dataset':     val_dataset,
    }
    return train_loader, val_loader, dataset_info


def build_dataloaders(config):
    """Construit les DataLoaders train/val avec split stratifié simple."""
    data = load_raw_data(config)

    strat_labels = (
        data['targets']['eau'].astype(str) + "_" + data['targets']['nuage'].astype(str)
    )
    train_idx, val_idx = train_test_split(
        list(range(len(data['spectra']))),
        test_size    = 1 - config.data.train_ratio,
        random_state = config.data.random_seed,
        stratify     = strat_labels,
    )
    return build_dataloaders_from_indices(data, train_idx, val_idx, config)


def build_model(cfg, spectrum_length: int, auxiliary_dim: int):
    """Instancie le modèle selon la config."""
    arch = cfg.model.architecture

    common_torch = dict(
        spectrum_length = spectrum_length,
        auxiliary_dim   = auxiliary_dim,
        num_classes     = cfg.training.num_classes,
        input_channels  = cfg.model.input_channels,
        dropout         = cfg.model.dropout,
    )

    if arch == "CNN":
        logger.info("Architecture : CNN")
        return CNN(
            **common_torch,
            conv_channels  = cfg.model.conv_channels,
            kernel_sizes   = cfg.model.kernel_sizes,
            pool_sizes     = cfg.model.pool_sizes,
            fc_dims        = cfg.model.fc_dims,
            use_batch_norm = cfg.model.use_batch_norm,
        )

    elif arch == "ResNet8":
        logger.info("Architecture : ResNet-8")
        return resnet8_1d(**common_torch)

    elif arch == "ResNet18":
        logger.info("Architecture : ResNet-18")
        return resnet18_1d(**common_torch)

    elif arch == "ResNet34":
        logger.warning("Architecture : ResNet-34 (risque surapprentissage)")
        return resnet34_1d(**common_torch)

    elif arch == "2ResNet":
        logger.info("Architecture : Ensemble SE-ResNet8 + ResNet18")
        return ensemble_resnet_1d(**common_torch)

    else:
        raise ValueError(
            f"Architecture inconnue : '{arch}'. "
            f"Valeurs valides : CNN | ResNet8 | ResNet18 | ResNet34 | 2ResNet"
        )


def _select_trainer(model, train_loader, val_loader, cfg):
    """Retourne le Trainer PyTorch."""
    logger.info("Trainer : Trainer (PyTorch)")
    return Trainer(model, train_loader, val_loader, cfg)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Entraînement du modèle exoplanète")
    parser.add_argument('--config', type=str, default=None, help="Chemin vers la config JSON")
    parser.add_argument('--arch', type=str, default='ResNetCNN',
                        choices=list(ARCH_TO_CONFIG.keys()),
                        help="Modèle à entraîner — charge models/<arch>/config.py")
    parser.add_argument('--log-file', type=str, default=None,
                        help="Duplique les logs dans ce fichier")
    parser.add_argument('--log-level', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'])
    parser.add_argument('--cv', type=int, default=0,
                        help="K folds de cross-validation stratifiée (0 = single split)")
    parser.add_argument('--seeds', type=int, nargs='+', default=None,
                        help="Seeds pour ensembling (ex: --seeds 42 7 123). Par défaut : "
                             "[cfg.data.random_seed].")
    args = parser.parse_args()

    setup_logging(level=getattr(logging, args.log_level), log_file=args.log_file)

    cfg = Config.load(args.config) if args.config else load_config_for_arch(args.arch)
    set_seed(cfg.data.random_seed)
    cfg.print_summary()

    # ── MODE CROSS-VALIDATION ──────────────────────────────────────────────
    if args.cv and args.cv > 1:
        from training.cv import cross_validate
        seeds = tuple(args.seeds) if args.seeds else (cfg.data.random_seed,)
        logger.info("── Cross-validation %d folds × %d seeds ──", args.cv, len(seeds))
        cv_result = cross_validate(cfg, n_splits=args.cv, seeds=seeds)
        logger.info("CV terminé. MCC OOF mean=%.4f (eau=%.4f, nuage=%.4f)",
                    cv_result['aggregate']['mcc_mean'],
                    cv_result['aggregate']['mcc_eau'],
                    cv_result['aggregate']['mcc_nuage'])
        return

    logger.info("── Chargement des données ──")
    train_loader, val_loader, dataset_info = build_dataloaders(cfg)

    logger.info("── Construction du modèle ──")
    model = build_model(cfg, dataset_info['spectrum_length'], dataset_info['auxiliary_dim'])

    logger.info("── Initialisation du Trainer ──")
    trainer = _select_trainer(model, train_loader, val_loader, cfg)

    result = trainer.train()

    log_training_summary(result)

    out_dir = Path(cfg.results_folder)
    out_dir.mkdir(parents=True, exist_ok=True)
    history = result.get('history', {})

    logger.info("── Sauvegarde des résultats ──")
    with open(out_dir / 'training_history.json', 'w') as f:
        json.dump(result, f, indent=2, cls=NumpyEncoder)
    logger.info("JSON : %s/training_history.json", out_dir)

    save_training_results(history, str(out_dir))
    plot_training_results(history, str(out_dir))

    logger.info("Tout est sauvegardé dans : %s", out_dir)


if __name__ == '__main__':
    main()
