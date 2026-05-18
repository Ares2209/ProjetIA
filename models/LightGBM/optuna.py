"""Optuna search for LightGBM with persistent SQLite storage.

Stocke tous les trials dans `models/LightGBM/optuna_search` (SQLite). L'étude
est reprenable : relancer la commande continue d'ajouter des trials à la même
DB. La recherche est consciente du surapprentissage : on log le gap train-val
par label et on peut le pénaliser dans l'objectif (`--gap-penalty`).

Usage:
    python -m models.LightGBM.optuna --trials 100
    python -m models.LightGBM.optuna --trials 50 --gap-penalty 0.5
    python -m models.LightGBM.optuna --trials 30 --seeds-per-trial 5 --timeout 7200

Outputs :
  - models/LightGBM/optuna_search          : base SQLite (trials, résumable)
  - models/LightGBM/results/optuna/best_params.json
  - models/LightGBM/results/optuna/trials.csv
  - models/LightGBM/results/optuna/analysis.json
  
# Première recherche (50 trials, ~1h) 50 trials, 3-fold CV, hold-out 15%
python -m models.LightGBM.optuna --trials 50

# Plus rigoureux (5 folds)
python models/LightGBM/optuna.py --trials 30 --cv-folds 5

# Si tu veux juste itérer rapidement sans le test final
python models/LightGBM/optuna.py --trials 20 --skip-final-test

python -m models.LightGBM.optuna --trials 60 --study-name lightgbm_search_v2 --storage-path models/LightGBM/optuna_search_v2

  
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
import contextlib
import copy
import json
import logging
import tempfile
from statistics import mean, pstdev
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from sklearn.metrics import matthews_corrcoef
from sklearn.model_selection import StratifiedKFold, train_test_split

from main import (
    _select_trainer,
    build_dataloaders_from_indices,
    build_model,
    load_raw_data,
)
from models.LightGBM.config import get_config as get_lightgbm_config
from training.utils import set_seed, setup_logging


logger = logging.getLogger(__name__)


@contextlib.contextmanager
def _silence_loggers(names: Tuple[str, ...], level: int = logging.WARNING) -> Iterator[None]:
    """Élève temporairement le seuil de certains loggers (utile pendant les trials)."""
    targets = [logging.getLogger(name) for name in names]
    saved = [lg.level for lg in targets]
    try:
        for lg in targets:
            lg.setLevel(level)
        yield
    finally:
        for lg, lvl in zip(targets, saved):
            lg.setLevel(lvl)


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
        "augmentation_factor": trial.suggest_categorical(
            "augmentation_factor", [0, 2, 4]
        ),
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
    # Les trials antérieurs à l'ajout de `augmentation_factor` dans la search
    # space tournaient avec la valeur forcée 0 → on retombe dessus si la clé
    # manque (typiquement quand `study.best_params` pointe sur un vieux trial).
    cfg.data.augmentation_factor = params.get("augmentation_factor", 0)


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


def _slice_raw_data(data: dict, idx: np.ndarray) -> dict:
    """Sous-ensemble du dict de données par indices."""
    return {
        "spectra":   data["spectra"][idx],
        "auxiliary": data["auxiliary"].iloc[idx].reset_index(drop=True),
        "targets":   data["targets"].iloc[idx].reset_index(drop=True),
    }


def _train_on_split(
    params: dict,
    raw_data: dict,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    cfg_base,
    work_dir: Optional[Path] = None,
) -> dict:
    """Entraîne un LightGBM sur les indices fournis et retourne les métriques.

    Si `work_dir` est None, un dossier temporaire auto-nettoyé est utilisé →
    aucun artefact ne reste sur disque (utile pour les trials Optuna).
    """
    cfg = copy.deepcopy(cfg_base)
    _apply_params_to_cfg(cfg, params)

    tmp_ctx = tempfile.TemporaryDirectory(prefix="lgbm_optuna_") if work_dir is None else None
    if tmp_ctx is not None:
        work_dir = Path(tmp_ctx.name)
    cfg.paths.model_folder = str(work_dir / "checkpoints")
    cfg.results_folder = str(work_dir / "results")

    try:
        train_loader, val_loader, _ = build_dataloaders_from_indices(
            raw_data, list(train_idx), list(val_idx), cfg,
        )
        model = build_model(cfg, spectrum_length=0, auxiliary_dim=0)
        trainer = _select_trainer(model, train_loader, val_loader, cfg)
        result = trainer.train()

        val_probs    = np.asarray(result["val_probs"],    dtype=np.float32)
        val_labels   = np.asarray(result["val_labels"],   dtype=np.float32)
        train_probs  = np.asarray(result["train_probs"],  dtype=np.float32)
        train_labels = np.asarray(result["train_labels"], dtype=np.float32)
    finally:
        if tmp_ctx is not None:
            tmp_ctx.cleanup()

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
        "_val_probs":          val_probs,
        "_val_labels":         val_labels,
    }


def _stratify_pool(targets_df) -> np.ndarray:
    """Étiquette combinée eau_nuage pour StratifiedKFold (4 classes)."""
    return (targets_df["eau"].astype(str) + "_" + targets_df["nuage"].astype(str)).values


def _get_or_create_test_split(
    raw_data: dict, test_ratio: float, seed: int, cache_path: Path,
) -> Tuple[np.ndarray, np.ndarray]:
    """Réserve un hold-out test stable et persistant. Optuna ne le voit JAMAIS."""
    if cache_path.exists():
        loaded = np.load(cache_path)
        return loaded["pool_idx"], loaded["test_idx"]
    n = len(raw_data["spectra"])
    strat = _stratify_pool(raw_data["targets"])
    pool_idx, test_idx = train_test_split(
        np.arange(n), test_size=test_ratio, stratify=strat, random_state=seed,
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache_path, pool_idx=pool_idx, test_idx=test_idx)
    return pool_idx, test_idx


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
    cfg_base,
    pool_data: dict,
    folds: List[Tuple[np.ndarray, np.ndarray]],
    objective_metric: str,
    gap_penalty: float,
) -> float:
    params = _suggest_hyperparams(trial)

    runs: List[dict] = []
    # Pendant un trial on tait la log détaillée de l'entraîneur LightGBM :
    # 50 trials × 3 folds × ~5 lignes par run sinon, illisible.
    with _silence_loggers(("models.LightGBM.LightGBM",)):
        for i, (train_idx, val_idx) in enumerate(folds):
            # work_dir=None → artefacts dans un tmpdir auto-nettoyé.
            # Seules les métriques (renvoyées) finissent dans la DB Optuna.
            run = _train_on_split(
                params, pool_data, train_idx, val_idx, cfg_base, work_dir=None,
            )
            runs.append(run)

            # Pruning intermédiaire : si la perf moyenne courante est très
            # mauvaise, on coupe avant de faire les folds suivants.
            running_mean = float(np.mean([r["val_mcc_mean"] for r in runs]))
            trial.report(running_mean, step=i)
            if trial.should_prune():
                logger.info("Trial #%04d pruned après %d/%d folds (running mean %.4f)",
                            trial.number, i + 1, len(folds), running_mean)
                raise optuna.TrialPruned()

    # On exclut les clés "_*" (probs/labels bruts) de l'agrégation
    runs_for_agg = [{k: v for k, v in r.items() if not k.startswith("_")} for r in runs]
    agg = _aggregate_runs(runs_for_agg)
    for k, v in agg.items():
        trial.set_user_attr(k, v)
    trial.set_user_attr("cv_folds", len(folds))
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

    logger.info(
        "Trial #%04d | folds=%d aug=%d pca=%s | obj=%.4f (raw=%.4f, pen=%.4f) "
        "| eau %.3f (gap %+.3f) | nuage %.3f (gap %+.3f)",
        trial.number, len(folds),
        params.get("augmentation_factor", 0),
        params.get("pca_components", "off") if params.get("use_pca") else "off",
        final_value, base_value, gap_penalty * gap_mean,
        agg["val_mcc_eau_mean"], agg["gap_eau_mean"],
        agg["val_mcc_nuage_mean"], agg["gap_nuage_mean"],
    )
    return final_value


def _retrain_and_eval_test(
    best_params: dict,
    raw_data: dict,
    pool_idx: np.ndarray,
    test_idx: np.ndarray,
    cfg_base,
    work_dir: Path,
) -> dict:
    """Réentraîne avec les meilleurs params sur tout le pool, évalue sur le hold-out test.

    Le test n'a JAMAIS été vu par Optuna : c'est le rapport honnête final.
    """
    logger.info("=" * 90)
    logger.info("RÉ-ENTRAÎNEMENT FINAL SUR POOL + ÉVALUATION SUR HOLD-OUT TEST")
    logger.info("=" * 90)
    run = _train_on_split(
        best_params, raw_data, pool_idx, test_idx, cfg_base, work_dir=work_dir,
    )
    val_probs = run.pop("_val_probs", None)
    val_labels = run.pop("_val_labels", None)

    logger.info("Test hold-out (n=%d) :", len(test_idx))
    logger.info("  MCC eau   %.4f  (opt %.4f @ t=%.2f)  gap %+.4f",
                run["val_mcc_eau"], run["val_mcc_eau_best"],
                run["best_threshold_eau"], run["gap_eau"])
    logger.info("  MCC nuage %.4f  (opt %.4f @ t=%.2f)  gap %+.4f",
                run["val_mcc_nuage"], run["val_mcc_nuage_best"],
                run["best_threshold_nuage"], run["gap_nuage"])
    logger.info("  MCC mean  %.4f", run["val_mcc_mean"])
    logger.info("=" * 90)

    report = {k: v for k, v in run.items() if not k.startswith("_")}
    if val_probs is not None and val_labels is not None:
        report["n_test_samples"] = int(len(test_idx))
    return report


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
        logger.warning("Aucun trial complété, pas d'analyse possible.")
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

    logger.info("=" * 90)
    logger.info("ANALYSE DES TRIALS")
    logger.info("=" * 90)
    for tag, trial in bests.items():
        if trial is None:
            continue
        ua = trial.user_attrs
        logger.info(
            "[%-15s] #%d | eau %.4f (gap %+.4f) | nuage %.4f (gap %+.4f) | mean %.4f",
            tag, trial.number,
            ua.get("val_mcc_eau_mean", 0), ua.get("gap_eau_mean", 0),
            ua.get("val_mcc_nuage_mean", 0), ua.get("gap_nuage_mean", 0),
            ua.get("val_mcc_mean_mean", 0),
        )

    logger.info("-" * 90)
    logger.info("Stats sur %d trials complétés :", len(finished))
    logger.info("  val MCC eau    : moy %.4f ± %.4f", mean(eaus), pstdev(eaus))
    logger.info("  val MCC nuage  : moy %.4f ± %.4f", mean(nuages), pstdev(nuages))
    logger.info("  gap train-val eau   : moy %+.4f ± %.4f", mean(gaps_eau), pstdev(gaps_eau))
    logger.info("  gap train-val nuage : moy %+.4f ± %.4f", mean(gaps_nuage), pstdev(gaps_nuage))
    logger.info("=" * 90)

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
        logger.warning("Impossible d'écrire trials.csv : %s", ex)


def main() -> None:
    parser = argparse.ArgumentParser(description="Optuna search LightGBM (persistent)")
    parser.add_argument("--trials", type=int, default=50)
    parser.add_argument("--timeout", type=int, default=36000,
                        help="Limite en secondes (défaut: 1h)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cv-folds", type=int, default=3,
                        help="Nombre de folds stratifiés par trial (CV K-fold)")
    parser.add_argument("--test-ratio", type=float, default=0.15,
                        help="Fraction des données réservée en hold-out test "
                             "(jamais vue par Optuna). Défaut 15 %%.")
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
                        default=str(_HERE / "results" / "optuna"),
                        help="Dossier pour best_params.json / trials.csv / analysis.json "
                             "(distinct du fichier SQLite)")
    parser.add_argument("--skip-final-test", action="store_true",
                        help="Ne pas réentraîner sur le pool ni évaluer sur le hold-out test à la fin.")
    args = parser.parse_args()

    setup_logging()
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    storage_path = Path(args.storage_path)
    storage_path.parent.mkdir(parents=True, exist_ok=True)
    if storage_path.is_dir():
        raise SystemExit(
            f"[ERREUR] {storage_path} existe et est un dossier. SQLite a besoin "
            f"d'un fichier à ce chemin. Supprime-le ou utilise --storage-path."
        )

    storage_url = f"sqlite:///{storage_path.as_posix()}"
    study = optuna.create_study(
        study_name=args.study_name,
        storage=storage_url,
        load_if_exists=True,
        direction="maximize",
        sampler=TPESampler(seed=args.seed),
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=1),
    )

    # ── Données + hold-out test (stable, mis en cache à côté de la DB) ────
    cfg_base = get_lightgbm_config()
    raw_data = load_raw_data(cfg_base)
    test_cache = output_dir / "test_split.npz"
    pool_idx, test_idx = _get_or_create_test_split(
        raw_data, args.test_ratio, args.seed, test_cache,
    )
    pool_data = _slice_raw_data(raw_data, pool_idx)
    strat = _stratify_pool(pool_data["targets"])
    skf = StratifiedKFold(n_splits=args.cv_folds, shuffle=True, random_state=args.seed)
    folds = [
        (np.asarray(tr, dtype=np.int64), np.asarray(va, dtype=np.int64))
        for tr, va in skf.split(np.zeros(len(pool_data["spectra"])), strat)
    ]

    n_done = len([t for t in study.trials
                  if t.state == optuna.trial.TrialState.COMPLETE])
    logger.info("=" * 90)
    logger.info("OPTUNA LIGHTGBM | étude « %s »", args.study_name)
    logger.info("=" * 90)
    logger.info("Storage   : %s", storage_url)
    logger.info("Trials    : %d déjà complétés  →  +%d à ajouter (timeout %ds)",
                n_done, args.trials, args.timeout)
    logger.info("Objectif  : %s  |  gap penalty : %.2f", args.objective, args.gap_penalty)
    logger.info("CV        : %d folds sur pool=%d  (hold-out test=%d)",
                args.cv_folds, len(pool_idx), len(test_idx))
    logger.info("=" * 90)
    # Optuna lui-même log à INFO chaque fin de trial ("Trial X finished..."),
    # ce qui doublonne notre ligne. On le met à WARNING.
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    study.optimize(
        lambda trial: _objective(
            trial,
            cfg_base=cfg_base,
            pool_data=pool_data,
            folds=folds,
            objective_metric=args.objective,
            gap_penalty=args.gap_penalty,
        ),
        n_trials=args.trials,
        timeout=args.timeout,
        gc_after_trial=True,
    )

    logger.info("=" * 90)
    logger.info("MEILLEUR TRIAL")
    logger.info("=" * 90)
    logger.info("Best value (%s pénalisé) : %.4f  |  trial #%d",
                args.objective, study.best_value, study.best_trial.number)
    for k, v in study.best_params.items():
        if isinstance(v, float):
            logger.info("  %-22s : %.6g", k, v)
        else:
            logger.info("  %-22s : %s", k, v)

    best_trial = study.best_trial
    best_params_payload = {
        "objective_metric":   args.objective,
        "gap_penalty":        args.gap_penalty,
        "best_params":        study.best_params,
        "best_value":         study.best_value,
        "best_trial_number":  best_trial.number,
        "best_trial_metrics": dict(best_trial.user_attrs),
        "study_name":         args.study_name,
        "storage":            storage_url,
        "n_trials_total":     len(study.trials),
        "cv_folds":           args.cv_folds,
        "test_ratio":         args.test_ratio,
        "n_pool":             int(len(pool_idx)),
        "n_test":             int(len(test_idx)),
        "seed":               args.seed,
    }

    # ── Évaluation finale sur le hold-out test (jamais vu par Optuna) ────
    if not args.skip_final_test:
        test_report = _retrain_and_eval_test(
            best_params=study.best_params,
            raw_data=raw_data,
            pool_idx=pool_idx,
            test_idx=test_idx,
            cfg_base=cfg_base,
            work_dir=output_dir / "final_holdout",
        )
        best_params_payload["holdout_test_metrics"] = test_report

    with open(output_dir / "best_params.json", "w") as f:
        json.dump(best_params_payload, f, indent=2)

    _analyze_study(study, output_dir, args.objective)

    logger.info("-" * 90)
    logger.info("Sorties :")
    logger.info("  storage SQLite   : %s", storage_path)
    logger.info("  best params JSON : %s", output_dir / "best_params.json")
    logger.info("  analyse JSON     : %s", output_dir / "analysis.json")
    logger.info("  trials CSV       : %s", output_dir / "trials.csv")
    logger.info("  test split cache : %s", test_cache)


if __name__ == "__main__":
    main()
