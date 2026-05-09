#!/usr/bin/env python3
"""CNN hyperparameter optimization using Optuna.

Finds optimal learning rate and other CNN parameters via Bayesian optimization.
Generates optimized checkpoint for inference.

Usage:
    python optimize_cnn.py --trials 50 --epochs 30
"""
import argparse
import json
import optuna
from pathlib import Path
from typing import Dict, Any

import torch
from torch.utils.data import DataLoader

from training.config import Config
from training.training import Trainer
from training.utils import set_seed, setup_logging
from models.dataset import ExoplanetDataset, collate_fn
from main import build_dataloaders, build_model, _select_trainer


def objective(trial: optuna.Trial, cfg: Config, train_loader: DataLoader, val_loader: DataLoader, dataset_info: dict) -> float:
    """Objective function for Optuna - maximize MCC."""

    lr = trial.suggest_float('learning_rate', 1e-5, 1e-1, log=True)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
    dropout = trial.suggest_float('dropout', 0.1, 0.5)
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)

    cfg.training.learning_rate = lr
    cfg.training.batch_size = batch_size
    cfg.model.dropout = dropout
    cfg.training.weight_decay = weight_decay
    cfg.training.num_epochs = 30

    cfg.data.train_ratio = 0.8
    train_loader, val_loader, _ = build_dataloaders(cfg)

    model = build_model(cfg, dataset_info['spectrum_length'], dataset_info['auxiliary_dim'])
    trainer = _select_trainer(model, train_loader, val_loader, cfg)
    trainer.train()

    best_mcc = max(trainer.state.history['val_mcc'])
    trial.set_user_attr('best_f1', max(trainer.state.history['val_f1']))
    trial.set_user_attr('final_loss', trainer.state.history['val_loss'][-1])

    return best_mcc


def main():
    parser = argparse.ArgumentParser(description="Optimize CNN with Optuna")
    parser.add_argument('--trials', type=int, default=50, help='Number of trials')
    parser.add_argument('--epochs', type=int, default=30, help='Epochs per trial')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--study-name', type=str, default='cnn_optimization', help='Optuna study name')
    parser.add_argument('--output-dir', type=str, default='results/optuna_cnn', help='Output directory')

    args = parser.parse_args()

    setup_logging()
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    cfg = Config()
    cfg.model.architecture = 'CNN'

    train_loader, val_loader, dataset_info = build_dataloaders(cfg)

    study = optuna.create_study(
        direction='maximize',
        study_name=args.study_name,
        sampler=optuna.samplers.TPESampler(seed=args.seed),
        pruner=optuna.pruners.MedianPruner()
    )

    print(f"Starting optimization with {args.trials} trials...")
    study.optimize(
        lambda trial: objective(trial, cfg, train_loader, val_loader, dataset_info),
        n_trials=args.trials,
        timeout=3600
    )

    print("\nOptimal Parameters:")
    best_params = study.best_params
    print(json.dumps(best_params, indent=2))
    print(f"\nBest MCC: {study.best_value:.4f}")

    study_path = output_dir / 'optuna_study.pkl'
    with open(study_path, 'wb') as f:
        import pickle
        pickle.dump(study, f)

    params_path = output_dir / 'best_params.json'
    with open(params_path, 'w') as f:
        json.dump({
            'best_params': best_params,
            'best_mcc': study.best_value,
            'study_name': args.study_name,
            'n_trials': args.trials
        }, f, indent=2)

    print("\nTraining final model with optimal parameters...")
    cfg.training.learning_rate = best_params['learning_rate']
    cfg.training.batch_size = best_params['batch_size']
    cfg.model.dropout = best_params['dropout']
    cfg.training.weight_decay = best_params['weight_decay']
    cfg.training.num_epochs = 100

    train_loader, val_loader, _ = build_dataloaders(cfg)
    model = build_model(cfg, dataset_info['spectrum_length'], dataset_info['auxiliary_dim'])
    trainer = _select_trainer(model, train_loader, val_loader, cfg)
    trainer.train()

    final_checkpoint = output_dir / 'cnn_optimized.pth'
    torch.save({
        'model_state_dict': trainer.model.state_dict(),
        'config': cfg.__dict__,
        'best_mcc': study.best_value,
        'optuna_params': best_params
    }, final_checkpoint)

    print(f"Optimized model saved to: {final_checkpoint}")
    print(f"Results saved to: {output_dir}")



if __name__ == '__main__':
    main()