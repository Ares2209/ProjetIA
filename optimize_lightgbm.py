#!/usr/bin/env python3
"""LightGBM hyperparameter optimization using Optuna.

Optimise les hyperparamètres LightGBM via TPE en maximisant le composite_score
(val) renvoyé par LightGBMTrainer. Sauvegarde l'étude Optuna et best_params.json.

Usage:
    python optimize_lightgbm.py --trials 50
    python optimize_lightgbm.py --trials 100 --timeout 7200
"""
import argparse
import json
import pickle
from pathlib import Path

import optuna

from main import build_dataloaders, build_model, _select_trainer
from models.LightGBM.config import get_config as get_lightgbm_config
from training.utils import set_seed, setup_logging


LIGHTGBM_HP_KEYS = (
    "learning_rate",
    "num_leaves",
    "max_depth",
    "min_child_samples",
    "subsample",
    "colsample_bytree",
    "reg_alpha",
    "reg_lambda",
)


def _suggest_hyperparams(trial: optuna.Trial) -> dict:
    return {
        "learning_rate":     trial.suggest_float("learning_rate", 1e-3, 3e-1, log=True),
        "num_leaves":        trial.suggest_int("num_leaves", 15, 255),
        "max_depth":         trial.suggest_int("max_depth", -1, 16),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        "subsample":         trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha":         trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda":        trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
    }


def objective(
    trial: optuna.Trial,
    train_loader,
    val_loader,
    base_output_dir: Path,
) -> float:
    cfg = get_lightgbm_config()
    params = _suggest_hyperparams(trial)

    cfg.training.learning_rate = params["learning_rate"]
    for key in ("num_leaves", "max_depth", "min_child_samples",
                "subsample", "colsample_bytree", "reg_alpha", "reg_lambda"):
        setattr(cfg.model, key, params[key])

    trial_dir = base_output_dir / f"trial_{trial.number:04d}"
    cfg.paths.model_folder = str(trial_dir / "checkpoints")
    cfg.results_folder = str(trial_dir / "results")

    model = build_model(cfg, spectrum_length=0, auxiliary_dim=0)
    trainer = _select_trainer(model, train_loader, val_loader, cfg)
    result = trainer.train()

    best = result["best_metrics"]
    trial.set_user_attr("mcc", float(best["mcc"]))
    trial.set_user_attr("mcc_at_best_threshold", float(best["mcc_at_best_threshold"]))
    trial.set_user_attr("auroc", float(best["auroc"]))
    trial.set_user_attr("training_time", float(result["training_time"]))

    return float(best["composite_score"])


def main():
    parser = argparse.ArgumentParser(description="Optimize LightGBM with Optuna")
    parser.add_argument("--trials", type=int, default=50, help="Number of trials")
    parser.add_argument("--timeout", type=int, default=3600,
                        help="Time limit in seconds (default: 1h)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--study-name", type=str, default="lightgbm_optimization",
                        help="Optuna study name")
    parser.add_argument("--output-dir", type=str, default="results/optuna_lightgbm",
                        help="Output directory")
    args = parser.parse_args()

    setup_logging()
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = get_lightgbm_config()
    train_loader, val_loader, _ = build_dataloaders(cfg)

    study = optuna.create_study(
        direction="maximize",
        study_name=args.study_name,
        sampler=optuna.samplers.TPESampler(seed=args.seed),
    )

    print(f"Starting optimization with {args.trials} trials "
          f"(timeout={args.timeout}s)...")
    study.optimize(
        lambda trial: objective(trial, train_loader, val_loader, output_dir),
        n_trials=args.trials,
        timeout=args.timeout,
    )

    print("\nOptimal Parameters:")
    best_params = study.best_params
    print(json.dumps(best_params, indent=2))
    print(f"\nBest composite_score: {study.best_value:.4f}")

    study_path = output_dir / "optuna_study.pkl"
    with open(study_path, "wb") as f:
        pickle.dump(study, f)

    best_trial = study.best_trial
    params_path = output_dir / "best_params.json"
    with open(params_path, "w") as f:
        json.dump(
            {
                "best_params":       best_params,
                "best_composite":    study.best_value,
                "best_trial_number": best_trial.number,
                "best_trial_metrics": dict(best_trial.user_attrs),
                "study_name":        args.study_name,
                "n_trials":          len(study.trials),
                "seed":              args.seed,
            },
            f,
            indent=2,
        )

    print(f"\nStudy saved to: {study_path}")
    print(f"Best params saved to: {params_path}")


if __name__ == "__main__":
    main()
