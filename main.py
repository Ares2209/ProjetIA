#!/usr/bin/env python3
"""Script d'entraînement principal.

Usage: python main.py [--arch CNN|ResNetCNN|LightGBM|XGBoost] [--config PATH]
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
from models.LightGBM import LightGBMTrainer
from models.ResNetCNN import ResNet1D, resnet18_1d, resnet34_1d, resnet8_1d, ensemble_resnet_1d


logger = logging.getLogger(__name__)


ARCH_TO_CONFIG = {
    'CNN':       'models.CNN.config',
    'ResNetCNN': 'models.ResNetCNN.config',
    'LightGBM':  'models.LightGBM.config',
    'XGBoost':   'models.XGBoost.config',
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


def build_dataloaders(config) -> tuple[DataLoader, DataLoader, dict]:
    """Construit les DataLoaders train/val avec split stratifié."""
    data_cfg      = config.data
    fallback_root = Path('Défi-IA-2026') / 'DATA' / 'defi-ia-cnes'

    spectra_path  = _resolve_path(Path(data_cfg.spectra_train_path),  fallback_root / 'spectra.npy',   'spectra')
    aux_path      = _resolve_path(Path(data_cfg.auxiliary_train_path), fallback_root / 'auxiliary.csv', 'auxiliary')
    targets_path  = _resolve_path(Path(data_cfg.targets_train_path),   fallback_root / 'targets.csv',   'targets')

    logger.info("spectra   : %s", spectra_path)
    logger.info("auxiliary : %s", aux_path)
    logger.info("targets   : %s", targets_path)

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

    logger.info("Distribution TRAIN — eau :\n%s",
                targets_train['eau'].value_counts(normalize=True).to_string())
    logger.info("Distribution TRAIN — nuage :\n%s",
                targets_train['nuage'].value_counts(normalize=True).to_string())
    logger.info("Distribution VAL — eau :\n%s",
                targets_val['eau'].value_counts(normalize=True).to_string())
    logger.info("Distribution VAL — nuage :\n%s",
                targets_val['nuage'].value_counts(normalize=True).to_string())

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
    )

    loader_kwargs = dict(
        batch_size     = config.training.batch_size,
        num_workers    = data_cfg.num_workers,
        pin_memory     = data_cfg.pin_memory,
        collate_fn     = collate_fn,
        worker_init_fn = seed_worker,   # seed numpy/random par worker
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

    elif arch == "LightGBM":
        logger.info("Architecture : LightGBM")
        return None

    elif arch == "XGBoost":
        raise NotImplementedError(
            "XGBoost n'est pas encore implémenté dans ce dépôt et le package "
            "'xgboost' n'est pas installé. Utilise --arch LightGBM pour le "
            "baseline gradient boosting disponible."
        )

    else:
        raise ValueError(
            f"Architecture inconnue : '{arch}'. "
            f"Valeurs valides : CNN | ResNet8 | ResNet18 | ResNet34 | 2ResNet | LightGBM"
        )


def _select_trainer(model, train_loader, val_loader, cfg):
    """Retourne le Trainer PyTorch."""
    if cfg.model.architecture == "LightGBM":
        logger.info("Trainer : LightGBMTrainer")
        return LightGBMTrainer(train_loader, val_loader, cfg)

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
    args = parser.parse_args()

    setup_logging(level=getattr(logging, args.log_level), log_file=args.log_file)

    cfg = Config.load(args.config) if args.config else load_config_for_arch(args.arch)
    set_seed(cfg.data.random_seed)
    cfg.print_summary()

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
