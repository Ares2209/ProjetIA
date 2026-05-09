#!/usr/bin/env python3
"""Compare multiple ML/DL models on exoplanet classification.

Trains multiple models with the same configuration and compares performance.
Generates comparison report with metrics, training time, and memory usage.

Usage:
    python compare_models.py --models CNN ResNet18 XGBoost --epochs 50
"""
import argparse
import json
import time
import psutil
from pathlib import Path
from typing import Dict, List, Any
import pandas as pd
from dataclasses import dataclass
import importlib

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, matthews_corrcoef

from training.config import Config
from training.training import Trainer
from training.utils import set_seed, setup_logging
from models.dataset import ExoplanetDataset, collate_fn
from main import build_dataloaders, build_model, _select_trainer, load_config_for_arch

# Import models
from models.XGBoost import XGBoostModel
from models.LightGBM import LightGBMModel


def build_model_extended(cfg, spectrum_length: int, auxiliary_dim: int):
    """Build model instance according to architecture type."""
    arch = cfg.model.architecture

    if arch in ['CNN', 'ResNet8', 'ResNet18', 'ResNet34', '2ResNet']:
        return build_model(cfg, spectrum_length, auxiliary_dim)

    elif arch == 'XGBoost':
        return XGBoostModel(
            spectrum_length=spectrum_length,
            auxiliary_dim=auxiliary_dim,
            num_classes=cfg.training.num_classes,
            use_gpu=getattr(cfg.model, 'use_gpu', False),
            random_state=getattr(cfg.model, 'random_state', 42)
        )

    elif arch == 'LightGBM':
        return LightGBMModel(
            spectrum_length=spectrum_length,
            auxiliary_dim=auxiliary_dim,
            num_classes=cfg.training.num_classes,
            use_gpu=getattr(cfg.model, 'use_gpu', False),
            random_state=getattr(cfg.model, 'random_state', 42)
        )

    else:
        raise ValueError(f"Unknown architecture: '{arch}'")


# Import models
from models.XGBoost import XGBoostModel
from models.LightGBM import LightGBMModel

# Architecture to config mapping
ARCH_TO_CONFIG = {
    'CNN':       'models.CNN.config',
    'ResNet8':   'models.ResNetCNN.config',
    'ResNet18':  'models.ResNetCNN.config',
    'ResNet34':  'models.ResNetCNN.config',
    '2ResNet':   'models.ResNetCNN.config',
    'XGBoost':   'models.XGBoost.config',
    'LightGBM':  'models.LightGBM.config',
}


@dataclass
class ModelResult:
    """Result of a trained model."""
    name: str
    best_f1: float
    best_mcc: float
    training_time: float
    peak_memory_mb: float
    final_checkpoint: str
    history: Dict[str, List[float]]


def get_memory_usage() -> float:
    """Get current process memory usage in MB."""
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024


def train_single_model(
    model_name: str,
    cfg: Config,
    train_loader: DataLoader,
    val_loader: DataLoader,
    dataset_info: dict,
    epochs: int = 50
) -> ModelResult:
    """Train a single model and return results."""
    print(f"\n{'='*60}")
    print(f"Training {model_name}")
    print(f"{'='*60}")

    cfg.model.architecture = model_name
    cfg.training.num_epochs = epochs
    model = build_model_extended(cfg, dataset_info['spectrum_length'], dataset_info['auxiliary_dim'])

    start_memory = get_memory_usage()
    peak_memory = start_memory
    start_time = time.time()

    if model_name in ['XGBoost', 'LightGBM']:
        train_dataset = train_loader.dataset
        val_dataset = val_loader.dataset

        spectrum_train = train_dataset.spectra
        auxiliary_train = train_dataset.aux_features
        targets_train = train_dataset.targets[['eau', 'nuage']].values.astype(np.float32)

        spectrum_val = val_dataset.spectra
        auxiliary_val = val_dataset.aux_features
        targets_val = val_dataset.targets[['eau', 'nuage']].values.astype(np.float32)

        model.fit(
            spectrum_train,
            auxiliary_train,
            targets_train,
            spectrum_val=spectrum_val,
            auxiliary_val=auxiliary_val,
            targets_val=targets_val,
        )

        training_time = time.time() - start_time
        peak_memory = max(peak_memory, get_memory_usage())

        val_probas = model.predict_proba(spectrum_val, auxiliary_val)
        val_preds = (val_probas >= cfg.training.classification_threshold).astype(int)
        best_f1 = f1_score(targets_val, val_preds, average='macro')
        best_mcc = matthews_corrcoef(targets_val.ravel(), val_preds.ravel())
        history = {'val_f1': [best_f1], 'val_mcc': [best_mcc]}

        checkpoint_path = Path(cfg.paths.model_folder) / cfg.paths.model_basename
        model.save(str(checkpoint_path))
        best_checkpoint = str(checkpoint_path)

    else:
        trainer = _select_trainer(model, train_loader, val_loader, cfg)
        trainer.train()
        training_time = time.time() - start_time
        peak_memory = max(peak_memory, get_memory_usage())

        history = trainer.state.history
        best_f1 = max(history['val_f1'])
        best_mcc = max(history['val_mcc'])

        checkpoint_dir = Path(cfg.paths.model_folder)
        best_checkpoint = str(checkpoint_dir / f"{cfg.paths.model_basename}_best.pth")

    return ModelResult(
        name=model_name,
        best_f1=best_f1,
        best_mcc=best_mcc,
        training_time=training_time,
        peak_memory_mb=peak_memory - start_memory,
        final_checkpoint=best_checkpoint,
        history=history
    )


def main():
    parser = argparse.ArgumentParser(description="Compare multiple models")
    parser.add_argument('--models', nargs='+', default=['CNN', 'ResNet18', 'XGBoost'],
                        help='Models to train')
    parser.add_argument('--epochs', type=int, default=50, help='Epochs per model')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--output', type=str, default='results/model_comparison.json',
                        help='Output JSON file')
    args = parser.parse_args()

    setup_logging()
    set_seed(args.seed)

    cfg_data = load_config_for_arch('CNN')
    train_loader, val_loader, dataset_info = build_dataloaders(cfg_data)

    results = []
    for model_name in args.models:
        try:
            cfg_model = load_config_for_arch(model_name)
            cfg_model.data = cfg_data.data
            result = train_single_model(model_name, cfg_model, train_loader, val_loader,
                                       dataset_info, args.epochs)
            results.append(result)
        except Exception as e:
            print(f"Error training {model_name}: {e}")
            continue

    comparison = {
        'models': [r.name for r in results],
        'best_f1': [r.best_f1 for r in results],
        'best_mcc': [r.best_mcc for r in results],
        'training_time_sec': [r.training_time for r in results],
        'peak_memory_mb': [r.peak_memory_mb for r in results],
        'checkpoints': [r.final_checkpoint for r in results],
        'config': {'epochs': args.epochs, 'seed': args.seed,
                   'batch_size': cfg_data.training.batch_size,
                   'lr': cfg_data.training.learning_rate}
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(comparison, f, indent=2)

    print(f"\n{'='*80}")
    print("Model Comparison Results")
    print(f"{'='*80}")
    print(f"{'Model':<15} {'F1':<8} {'MCC':<8} {'Time(s)':<10} {'Memory(MB)':<12}")
    print("-" * 80)
    for r in results:
        print(f"{r.name:<15} {r.best_f1:.4f}  {r.best_mcc:.4f}  {r.training_time:.1f}      {r.peak_memory_mb:.1f}")
    print(f"\nResults saved to: {output_path}")

    if results:
        best_idx = max(range(len(results)), key=lambda i: results[i].best_mcc)
        best_model = results[best_idx]
        print(f"\nBest model: {best_model.name} (MCC: {best_model.best_mcc:.4f})")



if __name__ == '__main__':
    main()