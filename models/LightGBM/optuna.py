"""Optuna search for LightGBM with persistent SQLite storage.

Stocke tous les trials dans `models/LightGBM/optuna_search` (SQLite). L'étude
est reprenable : relancer la commande continue d'ajouter des trials à la même
DB. La recherche est consciente du surapprentissage : on log le gap train-val
par label et on peut le pénaliser dans l'objectif (`--gap-penalty`).

Usage:
    python -m models.LightGBM.optuna --trials 100
    python -m models.LightGBM.optuna --trials 50 --gap-penalty 0.5
    python -m models.LightGBM.optuna --trials 30 --seeds-per-trial 5 --timeout 7200

Outputs (dans `models/LightGBM/optuna_search/`):
  - optuna_search           : base SQLite (trials, résumable)
  - best_params.json        : meilleur trial selon l'objectif
  - trials.csv              : DataFrame complet pour exploration
  - analysis.json           : meilleurs par critère + agrégats
  
# Première recherche (50 trials, ~1h)
python -m models.LightGBM.optuna --trials 50

# Reprendre / élargir avec 30 trials de plus dans la même DB
python -m models.LightGBM.optuna --trials 30

# Sans pénalisation du gap (comportement de l'ancien script)
python -m models.LightGBM.optuna --trials 50 --gap-penalty 0

# Forcer une recherche plus robuste (5 seeds × trial)
python -m models.LightGBM.optuna --trials 30 --seeds-per-trial 5 --timeout 7200
  
"""

# Le fichier s'appelle `optuna.py` → quand il est exécuté directement, son
# dossier finit en sys.path[0] et shadow le package `optuna`. On nettoie avant
# tout import.
from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parents[1]
sys.path[:] = [p for p in sys.path if p and Path(p).resolve() != _HERE]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import argparse
import json
from statistics import mean, pstdev
from typing import Dict, List

import numpy as np
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from sklearn.metrics import matthews_corrcoef

from main import _select_trainer, build_dataloaders, build_model
from models.LightGBM.config import get_config as get_lightgbm_config
from training.utils import set_seed, setup_logging


LABEL_NAMES = ("eau", "nuage")

FEATURE_FLAG_KEYS = (
    "use_pca",
    "use_statistical_features",
    "use_diff_features",
)

OBJECTIVE_CHOICES = (
    "val_mcc_mean_mean",
    "val_mcc_mean_best_mean",
    "val_mcc_min_mean",
    "val_mcc_eau_mean",
    "val_mcc_nuage_mean",
)


def _suggest_hyperparams(trial: optuna.Trial) -> dict:
    """Espace de recherche centré sur la régularisation (anti-overfitting)."""
    params = {
        "learning_rate":     trial.suggest_float("learning_rate", 5e-3, 1.5e-1, log=True),
        "n_estimators":      trial.suggest_int("n_estimators", 200, 2000, step=100),
        "num_leaves":        trial.suggest_int("num_leaves", 15, 127),
        "max_depth":         trial.suggest_int("max_depth", 3, 12),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 150),
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
    cfg.model.n_estimators = params["n_estimators"]
    for key in ("num_leaves", "max_depth", "min_child_samples",
                "subsample", "colsample_bytree", "reg_alpha", "reg_lambda"):
        setattr(cfg.model, key, params[key])
    for key in FEATURE_FLAG_KEYS:
        setattr(cfg.model, key, params[key])
    if params["use_pca"]:
        cfg.model.pca_components = params["pca_components"]


def _per_label_metrics(probs: np.ndarray, labels: np.ndarray) -> dict:
    """MCC par label à seuil 0.5 et seuil optimal."""
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


def _train_one_seed(params: dict, data_seed: int, trial_dir: Path) -> dict:
    cfg = get_lightgbm_config()
    _apply_params_to_cfg(cfg, params)
    cfg.data.random_seed = data_seed
    cfg.paths.model_folder = str(trial_dir / "checkpoints")
    cfg.results_folder = str(trial_dir / "results")

    train_loader, val_loader, _ = build_dataloaders(cfg)
    model = build_model(cfg, spectrum_length=0, auxiliary_dim=0)
    trainer = _select_trainer(model, train_loader, val_loader, cfg)
    result = trainer.train()

    val_probs    = np.asarray(result["val_probs"],    dtype=np.float32)
    val_labels   = np.asarray(result["val_labels"],   dtype=np.float32)
    train_probs  = np.asarray(result["train_probs"],  dtype=np.float32)
    train_labels = np.asarray(result["train_labels"], dtype=np.float32)

    val_m   = _per_label_metrics(val_probs,   val_labels)
    train_m = _per_label_metrics(train_probs, train_labels)

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
    }


def _aggregate_runs(runs: List[dict]) -> Dict[str, float]:
    agg: Dict[str, float] = {}
    if not runs:
        return agg
    for k in runs[0].keys():
        vals = [r[k] for r in runs]
        agg[f"{k}_mean"] = float(mean(vals))
        agg[f"{k}_std"]  = float(pstdev(vals)) if len(vals) > 1 else 0.0
    return agg


def _objective(
    trial: optuna.Trial,
    base_output_dir: Path,
    base_seed: int,
    seeds_per_trial: int,
    objective_metric: str,
    gap_penalty: float,
) -> float:
    params = _suggest_hyperparams(trial)
    trial_dir = base_output_dir / f"trial_{trial.number:04d}"

    runs: List[dict] = []
    for i in range(seeds_per_trial):
        data_seed = base_seed + 1000 * i
        run = _train_one_seed(params, data_seed, trial_dir / f"seed_{i}")
        runs.append(run)

        # Pruning intermédiaire : si la perf moyenne courante est très mauvaise,
        # on coupe avant de faire les seeds suivants.
        running_mean = float(np.mean([r["val_mcc_mean"] for r in runs]))
        trial.report(running_mean, step=i)
        if trial.should_prune():
            raise optuna.TrialPruned()

    agg = _aggregate_runs(runs)
    for k, v in agg.items():
        trial.set_user_attr(k, v)
    trial.set_user_attr("seeds_per_trial", seeds_per_trial)
    trial.set_user_attr("gap_penalty", gap_penalty)

    base_value = agg.get(objective_metric)
    if base_value is None:
        raise optuna.TrialPruned(f"Objective '{objective_metric}' absent.")

    # Pénalisation du gap (uniquement positif : on punit le surapprentissage,
    # pas le sous-apprentissage).
    gap_mean = max(0.0, (agg.get("gap_eau_mean", 0.0) + agg.get("gap_nuage_mean", 0.0)) / 2)
    final_value = base_value - gap_penalty * gap_mean
    trial.set_user_attr("objective_raw", base_value)
    trial.set_user_attr("objective_final", final_value)

    print(
        f"\n>> Trial {trial.number:04d} | "
        f"obj={final_value:.4f} (raw={base_value:.4f}, gap_pen={gap_penalty * gap_mean:.4f}) "
        f"| eau val={agg['val_mcc_eau_mean']:.3f} (gap {agg['gap_eau_mean']:+.3f}) "
        f"| nuage val={agg['val_mcc_nuage_mean']:.3f} (gap {agg['gap_nuage_mean']:+.3f})"
    )
    return final_value


def _best_by(study: optuna.Study, key: str) -> optuna.trial.FrozenTrial | None:
    finished = [
        t for t in study.trials
        if t.state == optuna.trial.TrialState.COMPLETE and key in t.user_attrs
    ]
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


def _analyze_study(study: optuna.Study, output_dir: Path, objective_metric: str) -> None:
    finished = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not finished:
        print("Aucun trial complété, pas d'analyse possible.")
        return

    bests = {
        "balanced_mean":   _best_by(study, "val_mcc_mean_mean"),
        "eau":             _best_by(study, "val_mcc_eau_mean"),
        "nuage":           _best_by(study, "val_mcc_nuage_mean"),
        "worst_label_min": _best_by(study, "val_mcc_min_mean"),
        "lowest_gap":      min(
            (t for t in finished if "gap_eau_mean" in t.user_attrs),
            key=lambda t: (t.user_attrs.get("gap_eau_mean", 1.0)
                           + t.user_attrs.get("gap_nuage_mean", 1.0)) / 2,
            default=None,
        ),
    }

    eaus       = [t.user_attrs.get("val_mcc_eau_mean",   0.0) for t in finished]
    nuages     = [t.user_attrs.get("val_mcc_nuage_mean", 0.0) for t in finished]
    gaps_eau   = [t.user_attrs.get("gap_eau_mean",   0.0) for t in finished]
    gaps_nuage = [t.user_attrs.get("gap_nuage_mean", 0.0) for t in finished]

    print("\n" + "=" * 90)
    print("ANALYSE DES TRIALS")
    print("=" * 90)
    for tag, trial in bests.items():
        if trial is None:
            continue
        ua = trial.user_attrs
        print(f"\n[{tag}] trial #{trial.number}")
        print(f"  val MCC eau    : {ua.get('val_mcc_eau_mean', 0):.4f} "
              f"(gap {ua.get('gap_eau_mean', 0):+.4f})")
        print(f"  val MCC nuage  : {ua.get('val_mcc_nuage_mean', 0):.4f} "
              f"(gap {ua.get('gap_nuage_mean', 0):+.4f})")
        print(f"  val MCC mean   : {ua.get('val_mcc_mean_mean', 0):.4f}")

    print("\n" + "-" * 90)
    print(f"Stats sur {len(finished)} trials complétés :")
    print(f"  val MCC eau    : moy {mean(eaus):.4f} ± {pstdev(eaus):.4f}")
    print(f"  val MCC nuage  : moy {mean(nuages):.4f} ± {pstdev(nuages):.4f}")
    print(f"  gap train-val eau   : moy {mean(gaps_eau):+.4f} ± {pstdev(gaps_eau):.4f}")
    print(f"  gap train-val nuage : moy {mean(gaps_nuage):+.4f} ± {pstdev(gaps_nuage):.4f}")
    print("=" * 90)

    analysis = {
        "objective_metric": objective_metric,
        "n_trials_complete": len(finished),
        "best_trials": {k: _trial_to_dict(v) for k, v in bests.items()},
        "aggregate": {
            "val_mcc_eau":   {"mean": float(mean(eaus)),   "std": float(pstdev(eaus))},
            "val_mcc_nuage": {"mean": float(mean(nuages)), "std": float(pstdev(nuages))},
            "gap_eau":       {"mean": float(mean(gaps_eau)),   "std": float(pstdev(gaps_eau))},
            "gap_nuage":     {"mean": float(mean(gaps_nuage)), "std": float(pstdev(gaps_nuage))},
        },
    }
    with open(output_dir / "analysis.json", "w") as f:
        json.dump(analysis, f, indent=2, default=str)

    try:
        df = study.trials_dataframe()
        df.to_csv(output_dir / "trials.csv", index=False)
    except Exception as ex:
        print(f"(warn) Impossible d'écrire trials.csv : {ex}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Optuna search LightGBM (persistent)")
    parser.add_argument("--trials", type=int, default=50)
    parser.add_argument("--timeout", type=int, default=3600,
                        help="Limite en secondes (défaut: 1h)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seeds-per-trial", type=int, default=3,
                        help="Splits stratifiés moyennés par trial")
    parser.add_argument("--objective", type=str, default="val_mcc_mean_mean",
                        choices=OBJECTIVE_CHOICES)
    parser.add_argument("--gap-penalty", type=float, default=0.5,
                        help="Coefficient de pénalisation du gap train-val "
                             "(0 = ignore, 1 = forte pénalité). Défaut 0.5.")
    parser.add_argument("--study-name", type=str, default="lightgbm_search")
    parser.add_argument("--storage-path", type=str,
                        default=str(_HERE / "optuna_search"),
                        help="Chemin du fichier SQLite des trials")
    parser.add_argument("--output-dir", type=str,
                        default=str(_HERE / "optuna_search"),
                        help="Dossier pour best_params.json / trials.csv / analysis.json")
    args = parser.parse_args()

    setup_logging()
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    storage_path = Path(args.storage_path)
    storage_path.parent.mkdir(parents=True, exist_ok=True)

    storage_url = f"sqlite:///{storage_path.as_posix()}"
    study = optuna.create_study(
        study_name=args.study_name,
        storage=storage_url,
        load_if_exists=True,
        direction="maximize",
        sampler=TPESampler(seed=args.seed),
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=1),
    )

    n_done = len([t for t in study.trials
                  if t.state == optuna.trial.TrialState.COMPLETE])
    print(f"Étude « {args.study_name} »  |  storage: {storage_url}")
    print(f"Trials déjà complétés : {n_done}  |  Cibles ajoutées : {args.trials}")
    print(f"Objectif : {args.objective}  |  Gap penalty : {args.gap_penalty}")
    print(f"Seeds par trial : {args.seeds_per_trial}  |  Timeout : {args.timeout}s\n")

    study.optimize(
        lambda trial: _objective(
            trial,
            base_output_dir=output_dir,
            base_seed=args.seed,
            seeds_per_trial=args.seeds_per_trial,
            objective_metric=args.objective,
            gap_penalty=args.gap_penalty,
        ),
        n_trials=args.trials,
        timeout=args.timeout,
        gc_after_trial=True,
    )

    print("\nMeilleurs hyperparamètres :")
    print(json.dumps(study.best_params, indent=2))
    print(f"\nBest value ({args.objective} pénalisé) : {study.best_value:.4f}")

    best_trial = study.best_trial
    with open(output_dir / "best_params.json", "w") as f:
        json.dump(
            {
                "objective_metric":   args.objective,
                "gap_penalty":        args.gap_penalty,
                "best_params":        study.best_params,
                "best_value":         study.best_value,
                "best_trial_number":  best_trial.number,
                "best_trial_metrics": dict(best_trial.user_attrs),
                "study_name":         args.study_name,
                "storage":            storage_url,
                "n_trials_total":     len(study.trials),
                "seeds_per_trial":    args.seeds_per_trial,
                "seed":               args.seed,
            },
            f,
            indent=2,
        )

    _analyze_study(study, output_dir, args.objective)

    print(f"\nStorage SQLite   : {storage_path}")
    print(f"Best params      : {output_dir / 'best_params.json'}")
    print(f"Analyse          : {output_dir / 'analysis.json'}")
    print(f"Trials CSV       : {output_dir / 'trials.csv'}")


if __name__ == "__main__":
    main()
