#!/usr/bin/env python3
"""LightGBM hyperparameter optimization using Optuna.

Optimise via TPE :
  - 8 hyperparamètres LightGBM (learning_rate, num_leaves, max_depth,
    min_child_samples, subsample, colsample_bytree, reg_alpha, reg_lambda)
  - 4 toggles de feature engineering (use_pca, pca_components,
    use_statistical_features, use_diff_features)

Le composite_score est moyenné sur --seeds-per-trial splits stratifiés
différents (graine de data différente) pour réduire le bruit d'évaluation.

Sauvegarde l'étude Optuna et best_params.json dans --output-dir.

Usage:
    python optimize_lightgbm.py --trials 50
    python optimize_lightgbm.py --trials 100 --seeds-per-trial 5 --timeout 7200
"""
import argparse
import json
import pickle
from pathlib import Path
from statistics import mean, pstdev

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

FEATURE_FLAG_KEYS = (
    "use_pca",
    "use_statistical_features",
    "use_diff_features",
)


def _suggest_hyperparams(trial: optuna.Trial) -> dict:
    params = {
        "learning_rate":     trial.suggest_float("learning_rate", 1e-3, 3e-1, log=True),
        "num_leaves":        trial.suggest_int("num_leaves", 15, 255),
        "max_depth":         trial.suggest_int("max_depth", -1, 16),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        "subsample":         trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha":         trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda":        trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "use_pca":           trial.suggest_categorical("use_pca", [False, True]),
        "use_statistical_features": trial.suggest_categorical(
            "use_statistical_features", [False, True]
        ),
        "use_diff_features": trial.suggest_categorical("use_diff_features", [False, True]),
    }
    if params["use_pca"]:
        params["pca_components"] = trial.suggest_int("pca_components", 20, 150)
    return params


def _apply_params_to_cfg(cfg, params: dict) -> None:
    cfg.training.learning_rate = params["learning_rate"]
    for key in ("num_leaves", "max_depth", "min_child_samples",
                "subsample", "colsample_bytree", "reg_alpha", "reg_lambda"):
        setattr(cfg.model, key, params[key])
    for key in FEATURE_FLAG_KEYS:
        setattr(cfg.model, key, params[key])
    if params["use_pca"]:
        cfg.model.pca_components = params["pca_components"]


def _train_one_seed(params: dict, data_seed: int, seed_dir: Path) -> dict:
    """Entraîne LightGBM avec un split stratifié défini par data_seed."""
    cfg = get_lightgbm_config()
    _apply_params_to_cfg(cfg, params)
    cfg.data.random_seed = data_seed
    cfg.paths.model_folder = str(seed_dir / "checkpoints")
    cfg.results_folder = str(seed_dir / "results")

    train_loader, val_loader, _ = build_dataloaders(cfg)
    model = build_model(cfg, spectrum_length=0, auxiliary_dim=0)
    trainer = _select_trainer(model, train_loader, val_loader, cfg)
    result = trainer.train()
    return result["best_metrics"]


def objective(
    trial: optuna.Trial,
    base_output_dir: Path,
    base_seed: int,
    seeds_per_trial: int,
) -> float:
    params = _suggest_hyperparams(trial)
    trial_dir = base_output_dir / f"trial_{trial.number:04d}"

    composites, mccs, aurocs = [], [], []
    for i in range(seeds_per_trial):
        data_seed = base_seed + 1000 * i
        best = _train_one_seed(params, data_seed, trial_dir / f"seed_{i}")
        composites.append(float(best["composite_score"]))
        mccs.append(float(best["mcc"]))
        aurocs.append(float(best["auroc"]))

    composite_mean = mean(composites)
    composite_std = pstdev(composites) if len(composites) > 1 else 0.0

    trial.set_user_attr("composite_mean", composite_mean)
    trial.set_user_attr("composite_std",  composite_std)
    trial.set_user_attr("composite_scores", composites)
    trial.set_user_attr("mcc_mean",  mean(mccs))
    trial.set_user_attr("auroc_mean", mean(aurocs))
    trial.set_user_attr("seeds_per_trial", seeds_per_trial)

    print(f"\n>> Trial {trial.number:04d} | composite "
          f"{composite_mean:.4f} ± {composite_std:.4f} "
          f"(seeds={seeds_per_trial})")
    return composite_mean


def main():
    parser = argparse.ArgumentParser(description="Optimize LightGBM with Optuna")
    parser.add_argument("--trials", type=int, default=50, help="Number of trials")
    parser.add_argument("--timeout", type=int, default=3600,
                        help="Time limit in seconds (default: 1h)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--seeds-per-trial", type=int, default=3,
                        help="Splits stratifiés à moyenner par trial (default: 3)")
    parser.add_argument("--study-name", type=str, default="lightgbm_optimization",
                        help="Optuna study name")
    parser.add_argument("--output-dir", type=str, default="results/optuna_lightgbm",
                        help="Output directory")
    args = parser.parse_args()

    setup_logging()
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    study = optuna.create_study(
        direction="maximize",
        study_name=args.study_name,
        sampler=optuna.samplers.TPESampler(seed=args.seed),
    )

    print(f"Starting optimization with {args.trials} trials "
          f"({args.seeds_per_trial} seeds/trial, timeout={args.timeout}s)...")
    study.optimize(
        lambda trial: objective(
            trial,
            base_output_dir=output_dir,
            base_seed=args.seed,
            seeds_per_trial=args.seeds_per_trial,
        ),
        n_trials=args.trials,
        timeout=args.timeout,
    )

    print("\nOptimal Parameters:")
    best_params = study.best_params
    print(json.dumps(best_params, indent=2))
    print(f"\nBest composite_score (mean over seeds): {study.best_value:.4f}")

    study_path = output_dir / "optuna_study.pkl"
    with open(study_path, "wb") as f:
        pickle.dump(study, f)

    best_trial = study.best_trial
    params_path = output_dir / "best_params.json"
    with open(params_path, "w") as f:
        json.dump(
            {
                "best_params":        best_params,
                "best_composite":     study.best_value,
                "best_trial_number":  best_trial.number,
                "best_trial_metrics": dict(best_trial.user_attrs),
                "study_name":         args.study_name,
                "n_trials":           len(study.trials),
                "seeds_per_trial":    args.seeds_per_trial,
                "seed":               args.seed,
            },
            f,
            indent=2,
        )

    print(f"\nStudy saved to: {study_path}")
    print(f"Best params saved to: {params_path}")


if __name__ == "__main__":
    main()
