#!/usr/bin/env python3
"""LightGBM hyperparameter optimization using Optuna.

Optimise par défaut le **MCC moyen (eau + nuage) / 2** sur la validation, en
moyennant sur `--seeds-per-trial` splits stratifiés pour réduire le bruit.

Pour chaque trial, on enregistre dans `user_attrs` :
  - MCC val par label (seuil 0.5 et seuil optimal)
  - MCC train par label
  - gap train-val par label (diagnostic d'overfitting)
  - composite_score nuage et AUROC (compat ascendante)

À la fin, le script produit :
  - `best_params.json` : meilleur trial selon l'objectif
  - `analysis.json`    : meilleurs trials par critère (mean / eau / nuage /
                         min-label / overfit le plus faible) + agrégats
  - `trials.csv`       : DataFrame complet des trials pour exploration
  - `optuna_study.pkl` : étude Optuna sérialisée

Usage:
    python optimize_lightgbm.py --trials 50
    python optimize_lightgbm.py --trials 100 --seeds-per-trial 5 --timeout 7200
    python optimize_lightgbm.py --trials 50 --objective val_mcc_min_mean
"""
import argparse
import json
import pickle
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, List

import numpy as np
import optuna
from sklearn.metrics import matthews_corrcoef

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

LABEL_NAMES = ("eau", "nuage")

OBJECTIVE_CHOICES = (
    "val_mcc_mean_mean",       # MCC moyen (eau+nuage)/2 — défaut
    "val_mcc_mean_best_mean",  # idem mais avec seuils optimaux
    "val_mcc_min_mean",        # MCC de la pire tête (force à équilibrer)
    "val_mcc_eau_mean",        # uniquement eau
    "val_mcc_nuage_mean",      # uniquement nuage
    "composite_score_mean",    # ancien score (nuage)
)


def _suggest_hyperparams(trial: optuna.Trial) -> dict:
    params = {
        "learning_rate":     trial.suggest_float("learning_rate", 1e-3, 3e-1, log=True),
        "num_leaves":        trial.suggest_int("num_leaves", 15, 255),
        "max_depth":         trial.suggest_int("max_depth", -1, 16),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 200),
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


def _per_label_metrics(probs: np.ndarray, labels: np.ndarray) -> dict:
    """MCC par label à seuil 0.5 et seuil optimal (recherche dense)."""
    out: Dict[str, float] = {}
    for i, name in enumerate(LABEL_NAMES):
        y_true = labels[:, i].astype(int)
        p = probs[:, i]
        out[f"mcc_{name}"] = float(matthews_corrcoef(y_true, (p >= 0.5).astype(int)))
        best_t, best_mcc = 0.5, -1.0
        for t in np.linspace(0.2, 0.8, 61):
            m = matthews_corrcoef(y_true, (p >= float(t)).astype(int))
            if m > best_mcc:
                best_t, best_mcc = float(t), float(m)
        out[f"mcc_{name}_best"] = float(best_mcc)
        out[f"best_threshold_{name}"] = best_t
    out["mcc_mean"]      = (out["mcc_eau"] + out["mcc_nuage"]) / 2
    out["mcc_mean_best"] = (out["mcc_eau_best"] + out["mcc_nuage_best"]) / 2
    out["mcc_min"]       = min(out["mcc_eau"], out["mcc_nuage"])
    return out


def _train_one_seed(params: dict, data_seed: int, seed_dir: Path) -> dict:
    """Entraîne LightGBM avec un split stratifié défini par data_seed.

    Renvoie un dict riche avec métriques val ET train par label, gaps,
    et score composite (compat ascendante).
    """
    cfg = get_lightgbm_config()
    _apply_params_to_cfg(cfg, params)
    cfg.data.random_seed = data_seed
    cfg.paths.model_folder = str(seed_dir / "checkpoints")
    cfg.results_folder = str(seed_dir / "results")

    train_loader, val_loader, _ = build_dataloaders(cfg)
    model = build_model(cfg, spectrum_length=0, auxiliary_dim=0)
    trainer = _select_trainer(model, train_loader, val_loader, cfg)
    result = trainer.train()

    val_probs   = np.asarray(result["val_probs"],   dtype=np.float32)
    val_labels  = np.asarray(result["val_labels"],  dtype=np.float32)
    train_probs = np.asarray(result["train_probs"], dtype=np.float32)
    train_labels = np.asarray(result["train_labels"], dtype=np.float32)

    val_m   = _per_label_metrics(val_probs,   val_labels)
    train_m = _per_label_metrics(train_probs, train_labels)

    bm = result.get("best_metrics", {})
    return {
        "val_mcc_eau":         val_m["mcc_eau"],
        "val_mcc_nuage":       val_m["mcc_nuage"],
        "val_mcc_eau_best":    val_m["mcc_eau_best"],
        "val_mcc_nuage_best":  val_m["mcc_nuage_best"],
        "val_mcc_mean":        val_m["mcc_mean"],
        "val_mcc_mean_best":   val_m["mcc_mean_best"],
        "val_mcc_min":         val_m["mcc_min"],
        "best_threshold_eau":  val_m["best_threshold_eau"],
        "best_threshold_nuage": val_m["best_threshold_nuage"],
        "train_mcc_eau":       train_m["mcc_eau"],
        "train_mcc_nuage":     train_m["mcc_nuage"],
        "gap_eau":             train_m["mcc_eau"]   - val_m["mcc_eau"],
        "gap_nuage":           train_m["mcc_nuage"] - val_m["mcc_nuage"],
        "composite_score":     float(bm.get("composite_score", 0.0)),
        "auroc":               float(bm.get("auroc", 0.0)),
    }


def _aggregate_runs(runs: List[dict]) -> Dict[str, float]:
    """Calcule mean/std de chaque clé numérique sur la liste de runs."""
    agg: Dict[str, float] = {}
    if not runs:
        return agg
    for k in runs[0].keys():
        vals = [r[k] for r in runs]
        agg[f"{k}_mean"] = float(mean(vals))
        agg[f"{k}_std"]  = float(pstdev(vals)) if len(vals) > 1 else 0.0
    return agg


def objective(
    trial: optuna.Trial,
    base_output_dir: Path,
    base_seed: int,
    seeds_per_trial: int,
    objective_metric: str,
) -> float:
    params = _suggest_hyperparams(trial)
    trial_dir = base_output_dir / f"trial_{trial.number:04d}"

    runs: List[dict] = []
    for i in range(seeds_per_trial):
        data_seed = base_seed + 1000 * i
        runs.append(_train_one_seed(params, data_seed, trial_dir / f"seed_{i}"))

    agg = _aggregate_runs(runs)
    for k, v in agg.items():
        trial.set_user_attr(k, v)
    trial.set_user_attr("seeds_per_trial", seeds_per_trial)

    obj_val = agg.get(objective_metric)
    if obj_val is None:
        raise optuna.TrialPruned(f"Objective '{objective_metric}' absent du run.")

    print(
        f"\n>> Trial {trial.number:04d} | "
        f"obj({objective_metric})={obj_val:.4f}  "
        f"| eau val={agg['val_mcc_eau_mean']:.3f} (gap {agg['gap_eau_mean']:+.3f}) "
        f"| nuage val={agg['val_mcc_nuage_mean']:.3f} (gap {agg['gap_nuage_mean']:+.3f})"
    )
    return obj_val


def _best_by(study: optuna.Study, key: str) -> optuna.trial.FrozenTrial | None:
    finished = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    finished = [t for t in finished if key in t.user_attrs]
    if not finished:
        return None
    return max(finished, key=lambda t: t.user_attrs[key])


def _trial_to_dict(trial: optuna.trial.FrozenTrial | None) -> dict:
    if trial is None:
        return {}
    return {
        "number":  trial.number,
        "value":   trial.value,
        "params":  trial.params,
        "metrics": dict(trial.user_attrs),
    }


def _analyze_study(study: optuna.Study, output_dir: Path, objective_metric: str) -> dict:
    finished = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not finished:
        print("Aucun trial complété, pas d'analyse possible.")
        return {}

    bests = {
        "balanced_mean":  _best_by(study, "val_mcc_mean_mean"),
        "eau":            _best_by(study, "val_mcc_eau_mean"),
        "nuage":          _best_by(study, "val_mcc_nuage_mean"),
        "worst_label_min": _best_by(study, "val_mcc_min_mean"),
        "composite":      _best_by(study, "composite_score_mean"),
    }

    gaps_eau   = [t.user_attrs.get("gap_eau_mean",   0.0) for t in finished]
    gaps_nuage = [t.user_attrs.get("gap_nuage_mean", 0.0) for t in finished]
    eaus       = [t.user_attrs.get("val_mcc_eau_mean",   0.0) for t in finished]
    nuages     = [t.user_attrs.get("val_mcc_nuage_mean", 0.0) for t in finished]

    print("\n" + "=" * 90)
    print("ANALYSE PAR LABEL")
    print("=" * 90)
    labels_for_print = [
        ("Best balanced (mean)",      bests["balanced_mean"]),
        ("Best eau seul",             bests["eau"]),
        ("Best nuage seul",           bests["nuage"]),
        ("Best worst-label (min)",    bests["worst_label_min"]),
    ]
    for tag, trial in labels_for_print:
        if trial is None:
            continue
        ua = trial.user_attrs
        print(f"\n{tag} — trial #{trial.number}")
        print(f"  val MCC eau    : {ua.get('val_mcc_eau_mean', 0):.4f} "
              f"(gap {ua.get('gap_eau_mean', 0):+.4f}, train {ua.get('train_mcc_eau_mean', 0):.4f})")
        print(f"  val MCC nuage  : {ua.get('val_mcc_nuage_mean', 0):.4f} "
              f"(gap {ua.get('gap_nuage_mean', 0):+.4f}, train {ua.get('train_mcc_nuage_mean', 0):.4f})")
        print(f"  val MCC mean   : {ua.get('val_mcc_mean_mean', 0):.4f}")
        kp = {k: trial.params[k] for k in ('learning_rate', 'num_leaves', 'max_depth',
                                          'min_child_samples', 'reg_alpha', 'reg_lambda')
              if k in trial.params}
        print(f"  params clés    : {kp}")

    print("\n" + "-" * 90)
    print(f"Stats sur {len(finished)} trials complétés :")
    print(f"  val MCC eau    : moy {mean(eaus):.4f} ± {pstdev(eaus):.4f}  "
          f"(min {min(eaus):.4f}, max {max(eaus):.4f})")
    print(f"  val MCC nuage  : moy {mean(nuages):.4f} ± {pstdev(nuages):.4f}  "
          f"(min {min(nuages):.4f}, max {max(nuages):.4f})")
    print(f"  gap train-val eau   : moy {mean(gaps_eau):+.4f} ± {pstdev(gaps_eau):.4f}")
    print(f"  gap train-val nuage : moy {mean(gaps_nuage):+.4f} ± {pstdev(gaps_nuage):.4f}")
    print("=" * 90)

    # Importance par paramètre — sur chaque cible (eau / nuage / mean)
    importances: Dict[str, Dict[str, float]] = {}
    for target in ("val_mcc_eau_mean", "val_mcc_nuage_mean", "val_mcc_mean_mean"):
        try:
            imp = optuna.importance.get_param_importances(
                study, target=lambda t, target=target: t.user_attrs.get(target, 0.0),
            )
            importances[target] = {k: float(v) for k, v in imp.items()}
        except Exception as ex:
            importances[target] = {"error": str(ex)}

    if importances.get("val_mcc_eau_mean") and "error" not in importances["val_mcc_eau_mean"]:
        print("\nImportance des hyperparamètres sur MCC eau (top 5) :")
        for k, v in list(importances["val_mcc_eau_mean"].items())[:5]:
            print(f"  {k:<30} {v:.4f}")
        print("\nImportance des hyperparamètres sur MCC nuage (top 5) :")
        for k, v in list(importances["val_mcc_nuage_mean"].items())[:5]:
            print(f"  {k:<30} {v:.4f}")

    analysis = {
        "objective_metric": objective_metric,
        "n_trials_complete": len(finished),
        "best_trials": {k: _trial_to_dict(v) for k, v in bests.items()},
        "aggregate": {
            "val_mcc_eau":   {"mean": float(mean(eaus)),   "std": float(pstdev(eaus)),
                              "min": float(min(eaus)),     "max": float(max(eaus))},
            "val_mcc_nuage": {"mean": float(mean(nuages)), "std": float(pstdev(nuages)),
                              "min": float(min(nuages)),   "max": float(max(nuages))},
            "gap_eau":       {"mean": float(mean(gaps_eau)),   "std": float(pstdev(gaps_eau))},
            "gap_nuage":     {"mean": float(mean(gaps_nuage)), "std": float(pstdev(gaps_nuage))},
        },
        "param_importances": importances,
    }
    with open(output_dir / "analysis.json", "w") as f:
        json.dump(analysis, f, indent=2, default=str)

    # CSV des trials pour exploration
    try:
        df = study.trials_dataframe()
        df.to_csv(output_dir / "trials.csv", index=False)
    except Exception as ex:
        print(f"(warn) Impossible d'écrire trials.csv : {ex}")

    return analysis


def main():
    parser = argparse.ArgumentParser(description="Optimize LightGBM with Optuna")
    parser.add_argument("--trials", type=int, default=50, help="Number of trials")
    parser.add_argument("--timeout", type=int, default=3600,
                        help="Time limit in seconds (default: 1h)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--seeds-per-trial", type=int, default=3,
                        help="Splits stratifiés à moyenner par trial (default: 3)")
    parser.add_argument("--objective", type=str, default="val_mcc_mean_mean",
                        choices=OBJECTIVE_CHOICES,
                        help="Métrique à maximiser (défaut: val_mcc_mean_mean — "
                             "moyenne MCC eau+nuage)")
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
          f"({args.seeds_per_trial} seeds/trial, "
          f"objective={args.objective}, timeout={args.timeout}s)...")
    study.optimize(
        lambda trial: objective(
            trial,
            base_output_dir=output_dir,
            base_seed=args.seed,
            seeds_per_trial=args.seeds_per_trial,
            objective_metric=args.objective,
        ),
        n_trials=args.trials,
        timeout=args.timeout,
    )

    print("\nOptimal Parameters (selon objectif) :")
    best_params = study.best_params
    print(json.dumps(best_params, indent=2))
    print(f"\nBest {args.objective}: {study.best_value:.4f}")

    study_path = output_dir / "optuna_study.pkl"
    with open(study_path, "wb") as f:
        pickle.dump(study, f)

    best_trial = study.best_trial
    params_path = output_dir / "best_params.json"
    with open(params_path, "w") as f:
        json.dump(
            {
                "objective_metric":   args.objective,
                "best_params":        best_params,
                "best_value":         study.best_value,
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

    _analyze_study(study, output_dir, args.objective)

    print(f"\nStudy saved to:        {study_path}")
    print(f"Best params saved to:  {params_path}")
    print(f"Analysis saved to:     {output_dir / 'analysis.json'}")
    print(f"Trials CSV saved to:   {output_dir / 'trials.csv'}")


if __name__ == "__main__":
    main()
