"""Force une comparaison équitable de `augmentation_factor`.

Le TPE de l'étude courante a éliminé `aug=2` et `aug=4` aux startup trials,
quand les autres hyperparams n'étaient pas encore optimisés. On clone donc le
best trial actuel et on relance `n_repeats` essais par valeur d'augmentation,
en ne faisant varier QUE `augmentation_factor`. Si aug>0 reste mauvais → la
conclusion de TPE est confirmée. Sinon → il faut continuer la recherche
autour des nouveaux gagnants.

Usage:
    python -m models.LightGBM.optuna_fair_aug --repeats 3
"""
from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parents[1]
sys.path[:] = [p for p in sys.path if p and Path(p).resolve() != _HERE]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import argparse
import logging

import numpy as np
import optuna
from sklearn.model_selection import StratifiedKFold

from main import load_raw_data
from models.LightGBM.config import get_config as get_lightgbm_config
from models.LightGBM.optuna import (
    _get_or_create_test_split,
    _objective,
    _slice_raw_data,
    _stratify_pool,
)
from training.utils import set_seed, setup_logging


logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Comparaison équitable de augmentation_factor")
    parser.add_argument("--repeats", type=int, default=3,
                        help="Nombre de répétitions par valeur d'augmentation (défaut 3)")
    parser.add_argument("--aug-values", type=int, nargs="+", default=[0, 2, 4],
                        help="Valeurs à tester (défaut 0 2 4)")
    parser.add_argument("--cv-folds", type=int, default=3)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--gap-penalty", type=float, default=0.5)
    parser.add_argument("--objective", type=str, default="val_mcc_mean_mean")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--study-name", type=str, default="lightgbm_search")
    parser.add_argument("--storage-path", type=str,
                        default=str(_HERE / "optuna_search"))
    parser.add_argument("--output-dir", type=str,
                        default=str(_HERE / "results" / "optuna"))
    args = parser.parse_args()

    setup_logging()
    set_seed(args.seed)

    storage_url = f"sqlite:///{Path(args.storage_path).as_posix()}"
    study = optuna.load_study(study_name=args.study_name, storage=storage_url)

    best = dict(study.best_params)
    logger.info("=" * 90)
    logger.info("COMPARAISON ÉQUITABLE | base = best trial #%d (value=%.4f)",
                study.best_trial.number, study.best_value)
    logger.info("=" * 90)
    for k, v in best.items():
        logger.info("  %-22s : %s", k, v)
    logger.info("Valeurs aug testées : %s  | %d répétitions chacune",
                args.aug_values, args.repeats)

    # Construit les mêmes données / folds que dans optuna.py
    cfg_base = get_lightgbm_config()
    raw_data = load_raw_data(cfg_base)
    output_dir = Path(args.output_dir)
    test_cache = output_dir / "test_split.npz"
    pool_idx, _ = _get_or_create_test_split(
        raw_data, args.test_ratio, args.seed, test_cache,
    )
    pool_data = _slice_raw_data(raw_data, pool_idx)
    strat = _stratify_pool(pool_data["targets"])
    skf = StratifiedKFold(n_splits=args.cv_folds, shuffle=True, random_state=args.seed)
    folds = [
        (np.asarray(tr, dtype=np.int64), np.asarray(va, dtype=np.int64))
        for tr, va in skf.split(np.zeros(len(pool_data["spectra"])), strat)
    ]

    # On désactive le pruner pour ces trials : on veut comparer des trials
    # complets, pas que TPE/MedianPruner les tue tôt parce qu'ils sont plus lents.
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    n_enqueued = 0
    for af in args.aug_values:
        forced = dict(best)
        forced["augmentation_factor"] = af
        for _ in range(args.repeats):
            study.enqueue_trial(forced, skip_if_exists=False)
            n_enqueued += 1
    logger.info("Trials enqueued : %d", n_enqueued)

    # Optimize ne fera que les trials enqueued (pas de nouveaux samples).
    study.optimize(
        lambda trial: _objective(
            trial, cfg_base=cfg_base, pool_data=pool_data, folds=folds,
            objective_metric=args.objective, gap_penalty=args.gap_penalty,
        ),
        n_trials=n_enqueued,
    )

    # Synthèse
    from collections import defaultdict
    from statistics import mean, pstdev

    bucket = defaultdict(list)
    for t in study.trials[-n_enqueued:]:
        if t.state.name == "COMPLETE" and t.value is not None:
            af = t.params.get("augmentation_factor", 0)
            bucket[af].append(t.value)

    logger.info("=" * 90)
    logger.info("RÉSULTATS (n=%d répétitions par valeur)", args.repeats)
    logger.info("=" * 90)
    logger.info(f"{'aug':>4} | {'n':>3} | {'mean':>8} | {'std':>7} | {'min':>8} | {'max':>8}")
    for af in sorted(bucket.keys()):
        vals = bucket[af]
        m = mean(vals); s = pstdev(vals) if len(vals) > 1 else 0.0
        logger.info(f"{af:>4} | {len(vals):>3} | {m:>8.4f} | {s:>7.4f} | "
                    f"{min(vals):>8.4f} | {max(vals):>8.4f}")
    logger.info("=" * 90)


if __name__ == "__main__":
    main()
