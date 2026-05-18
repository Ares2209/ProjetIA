"""LightGBM baseline for exoplanet multi-label classification.

The PyTorch models keep the spectral signal as a 1D tensor. LightGBM needs a
tabular matrix, so this module reuses the same ExoplanetDataset preprocessing
and converts each sample to engineered numeric features.
"""

from __future__ import annotations

import json
import logging
import pickle
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from training.metrique import MetricsCalculator
from training.utils import NumpyEncoder


logger = logging.getLogger(__name__)


LABEL_NAMES = ("eau", "nuage")
PRIMARY_LABEL_INDEX = 1  # Matches the current PyTorch Trainer metric focus.
CHANNEL_NAMES = ("lambda_or_mu", "signal", "uncertainty", "snr", "rel_unc")


def _binary_log_loss(labels: np.ndarray, probs: np.ndarray) -> float:
    probs = np.clip(probs, 1e-7, 1.0 - 1e-7)
    labels = labels.astype(np.float64)
    loss = -(labels * np.log(probs) + (1.0 - labels) * np.log(1.0 - probs))
    return float(np.mean(loss))


def _safe_feature_names(n_channels: int) -> Tuple[str, ...]:
    if n_channels <= len(CHANNEL_NAMES):
        return CHANNEL_NAMES[:n_channels]
    extra = tuple(f"ch{i}" for i in range(len(CHANNEL_NAMES), n_channels))
    return CHANNEL_NAMES + extra


def _build_tabular_features(dataset, cfg) -> Tuple[np.ndarray, np.ndarray | None, np.ndarray, List[str]]:
    """Build a tabular matrix by iterating __getitem__ (applies augmentation when active)."""
    # Seed RNG so augmented samples are reproducible across runs.
    seed = int(cfg.data.random_seed)
    np.random.seed(seed)
    random.seed(seed)

    has_targets = dataset.targets is not None
    n_total = len(dataset)

    spectra_buf: List[np.ndarray] = []
    aux_buf: List[np.ndarray] = []
    labels_buf: List[np.ndarray] = [] if has_targets else []
    ids_buf: List[int] = []

    for i in range(n_total):
        item = dataset[i]
        spec = item["spectrum"]
        aux = item["auxiliary"]
        spectra_buf.append(spec.numpy() if hasattr(spec, "numpy") else np.asarray(spec))
        aux_buf.append(aux.numpy() if hasattr(aux, "numpy") else np.asarray(aux))
        ids_buf.append(int(item["id"]))
        if has_targets:
            tgt = item["target"]
            labels_buf.append(tgt.numpy() if hasattr(tgt, "numpy") else np.asarray(tgt))

    spectra = np.stack(spectra_buf).astype(np.float32)
    auxiliary = np.stack(aux_buf).astype(np.float32)
    labels = np.stack(labels_buf).astype(np.int64) if has_targets else None
    ids = np.asarray(ids_buf)

    n_samples, n_positions, n_channels = spectra.shape
    channel_names = _safe_feature_names(n_channels)

    feature_blocks = [spectra.reshape(n_samples, n_positions * n_channels)]
    feature_names = [
        f"spec_{pos:02d}_{channel_names[ch]}"
        for pos in range(n_positions)
        for ch in range(n_channels)
    ]

    if cfg.model.use_statistical_features:
        stats = {
            "mean": spectra.mean(axis=1),
            "std": spectra.std(axis=1),
            "min": spectra.min(axis=1),
            "max": spectra.max(axis=1),
            "median": np.median(spectra, axis=1),
            "q25": np.quantile(spectra, 0.25, axis=1),
            "q75": np.quantile(spectra, 0.75, axis=1),
            "range": spectra.max(axis=1) - spectra.min(axis=1),
        }
        for stat_name, values in stats.items():
            feature_blocks.append(values.astype(np.float32))
            feature_names.extend(f"{stat_name}_{name}" for name in channel_names)

    if cfg.model.use_diff_features:
        diffs = np.diff(spectra, axis=1)
        diff_stats = {
            "diff_mean": diffs.mean(axis=1),
            "diff_std": diffs.std(axis=1),
            "diff_min": diffs.min(axis=1),
            "diff_max": diffs.max(axis=1),
        }
        for stat_name, values in diff_stats.items():
            feature_blocks.append(values.astype(np.float32))
            feature_names.extend(f"{stat_name}_{name}" for name in channel_names)

    feature_blocks.append(auxiliary)
    feature_names.extend(f"aux_{i:02d}" for i in range(auxiliary.shape[1]))

    features = np.concatenate(feature_blocks, axis=1).astype(np.float32)
    return features, labels, ids, feature_names


def _fit_optional_pca(
    x_train: np.ndarray,
    x_val: np.ndarray,
    feature_names: List[str],
    cfg,
) -> Tuple[np.ndarray, np.ndarray, List[str], PCA | None]:
    if not cfg.model.use_pca:
        return x_train, x_val, feature_names, None

    n_components = min(cfg.model.pca_components, x_train.shape[0], x_train.shape[1])
    pca = PCA(n_components=n_components, random_state=cfg.data.random_seed)
    x_train_pca = pca.fit_transform(x_train)
    x_val_pca = pca.transform(x_val)
    pca_names = [f"pca_{i:02d}" for i in range(n_components)]
    return x_train_pca, x_val_pca, pca_names, pca


def _as_frame(features: np.ndarray, feature_names: List[str]) -> pd.DataFrame:
    return pd.DataFrame(features, columns=feature_names)


def _metrics_from_probs(
    labels: np.ndarray,
    probs: np.ndarray,
    threshold: float,
) -> Dict[str, float]:
    preds = probs >= threshold
    labels_bool = labels.astype(bool)

    tp = int(np.logical_and(preds, labels_bool).sum())
    fp = int(np.logical_and(preds, ~labels_bool).sum())
    tn = int(np.logical_and(~preds, ~labels_bool).sum())
    fn = int(np.logical_and(~preds, labels_bool).sum())

    raw = MetricsCalculator.compute_metrics_from_confusion_matrix(
        tp=tp,
        fp=fp,
        tn=tn,
        fn=fn,
        probs=probs,
        labels=labels,
    ).to_dict()

    g_mean = float(np.sqrt(max(raw["recall"] * raw["specificity"], 0.0)))
    min_class_recall = min(raw["class_0_recall"], raw["class_1_recall"])
    class_balance_gap = abs(raw["class_0_recall"] - raw["class_1_recall"])
    stability_score = (raw["mcc"] + raw["cohen_kappa"]) / 2.0
    production_score = (
        0.3 * raw["precision"]
        + 0.4 * raw["recall"]
        + 0.3 * raw["specificity"]
    )
    f_harmonic = (
        2 * (raw["f1"] * raw["f2"]) / (raw["f1"] + raw["f2"])
        if raw["f1"] > 0 and raw["f2"] > 0
        else 0.0
    )
    composite_score = (
        0.25 * raw["mcc"]
        + 0.20 * raw["balanced_accuracy"]
        + 0.20 * raw["f1"]
        + 0.20 * g_mean
        + 0.15 * raw["cohen_kappa"]
    )
    auroc = raw.get("auroc", 0.0) or 0.0
    auprc = raw.get("auprc", 0.0) or 0.0
    brier_score = raw.get("brier_score", 0.0) or 0.0
    probabilistic_score = (
        0.50 * auroc + 0.30 * auprc + 0.20 * (1.0 - brier_score)
        if auroc > 0
        else 0.0
    )

    raw.update({
        "g_mean": g_mean,
        "min_class_recall": min_class_recall,
        "class_balance_gap": class_balance_gap,
        "stability_score": stability_score,
        "production_score": production_score,
        "f_harmonic": f_harmonic,
        "composite_score": composite_score,
        "probabilistic_score": probabilistic_score,
    })
    raw.setdefault("auroc", 0.0)
    raw.setdefault("auprc", 0.0)
    raw.setdefault("brier_score", 0.0)
    return raw


def _find_optimal_threshold(labels: np.ndarray, probs: np.ndarray) -> Tuple[float, float]:
    best_threshold = 0.5
    best_mcc = -1.0
    for threshold in np.linspace(0.1, 0.9, 81):
        metrics = _metrics_from_probs(labels, probs, float(threshold))
        if metrics["mcc"] > best_mcc:
            best_mcc = metrics["mcc"]
            best_threshold = float(threshold)
    return best_threshold, float(best_mcc)


HISTORY_METRIC_KEYS: Tuple[str, ...] = (
    "loss", "accuracy", "balanced_accuracy", "precision", "recall",
    "specificity", "f1", "f2", "iou", "mcc", "cohen_kappa",
    "class_0_precision", "class_0_recall", "class_1_precision",
    "class_1_recall", "support_class_0", "support_class_1",
    "auroc", "auprc", "brier_score", "tp", "fp", "tn", "fn",
    "composite_score", "g_mean",
)

VAL_ONLY_KEYS: Tuple[str, ...] = (
    "min_class_recall", "class_balance_gap", "stability_score",
    "production_score", "f_harmonic",
)


def _checkpoint_iterations(max_iter: int, n_checkpoints: int = 30) -> List[int]:
    """Return up to n_checkpoints evenly-spaced iteration indices in [1, max_iter]."""
    if max_iter <= 1:
        return [max(1, max_iter)]
    n = min(n_checkpoints, max_iter)
    iters = np.linspace(1, max_iter, n).round().astype(int).tolist()
    # Deduplicate while keeping order.
    seen: List[int] = []
    for it in iters:
        if not seen or it != seen[-1]:
            seen.append(int(it))
    return seen


def _build_iteration_history(
    models: Dict[str, "lgb.LGBMClassifier"],
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    x_val: pd.DataFrame,
    y_val: np.ndarray,
    threshold: float,
    primary: int,
    n_checkpoints: int = 30,
) -> Tuple[Dict[str, List[float]], List[int]]:
    """Snapshot metrics across boosting iterations to build an epoch-like history."""
    label_names = list(models.keys())
    max_iter = max(
        (m.best_iteration_ or m.n_estimators_) for m in models.values()
    )
    iters = _checkpoint_iterations(max_iter, n_checkpoints)

    history: Dict[str, List[float]] = {"iteration_losses": []}
    for key in HISTORY_METRIC_KEYS:
        history[f"train_{key}"] = []
        history[f"val_{key}"] = []
    for key in VAL_ONLY_KEYS:
        history[f"val_{key}"] = []

    # Per-iteration val logloss (averaged across labels) from evals_result_.
    val_loglosses_per_label: List[np.ndarray] = []
    for label_name in label_names:
        evals = models[label_name].evals_result_ or {}
        val_eval = evals.get("val") or evals.get("valid_0") or {}
        series = val_eval.get("binary_logloss")
        if series is not None:
            val_loglosses_per_label.append(np.asarray(series, dtype=np.float64))
    if val_loglosses_per_label:
        min_len = min(arr.shape[0] for arr in val_loglosses_per_label)
        stacked = np.stack([arr[:min_len] for arr in val_loglosses_per_label])
        history["iteration_losses"] = stacked.mean(axis=0).tolist()

    n_labels = y_train.shape[1]
    for it in iters:
        train_probs = np.zeros_like(y_train, dtype=np.float32)
        val_probs = np.zeros_like(y_val, dtype=np.float32)
        for idx, label_name in enumerate(label_names):
            model = models[label_name]
            max_for_model = model.best_iteration_ or model.n_estimators_
            n_used = int(min(it, max_for_model))
            train_probs[:, idx] = model.predict_proba(x_train, num_iteration=n_used)[:, 1]
            val_probs[:, idx] = model.predict_proba(x_val, num_iteration=n_used)[:, 1]

        train_m = _metrics_from_probs(y_train[:, primary], train_probs[:, primary], threshold)
        val_m = _metrics_from_probs(y_val[:, primary], val_probs[:, primary], threshold)
        train_m["loss"] = float(np.mean([
            _binary_log_loss(y_train[:, i], train_probs[:, i]) for i in range(n_labels)
        ]))
        val_m["loss"] = float(np.mean([
            _binary_log_loss(y_val[:, i], val_probs[:, i]) for i in range(n_labels)
        ]))

        for key in HISTORY_METRIC_KEYS:
            history[f"train_{key}"].append(float(train_m[key]))
            history[f"val_{key}"].append(float(val_m[key]))
        for key in VAL_ONLY_KEYS:
            history[f"val_{key}"].append(float(val_m[key]))

    return history, iters


class LightGBMTrainer:
    """Trainer with the same public API as training.training.Trainer."""

    def __init__(self, train_loader, val_loader, config):
        self.train_dataset = train_loader.dataset
        self.val_dataset = val_loader.dataset
        self.config = config
        self.models: Dict[str, lgb.LGBMClassifier] = {}
        self.pca: PCA | None = None
        self.feature_names: List[str] = []

    def _make_model(self) -> lgb.LGBMClassifier:
        return lgb.LGBMClassifier(
            objective="binary",
            n_estimators=self.config.model.n_estimators,
            learning_rate=self.config.training.learning_rate,
            num_leaves=self.config.model.num_leaves,
            max_depth=self.config.model.max_depth,
            min_child_samples=self.config.model.min_child_samples,
            subsample=self.config.model.subsample,
            colsample_bytree=self.config.model.colsample_bytree,
            reg_alpha=self.config.model.reg_alpha,
            reg_lambda=self.config.model.reg_lambda,
            class_weight=self.config.model.class_weight,
            random_state=self.config.data.random_seed,
            n_jobs=-1,
            verbosity=-1,
        )

    def _save_artifacts(
        self,
        y_train: np.ndarray,
        y_val: np.ndarray,
        train_probs: np.ndarray,
        val_probs: np.ndarray,
        val_ids: np.ndarray,
        history: Dict[str, List[float]],
        best_metrics: Dict[str, float],
    ) -> None:
        model_dir = Path(self.config.paths.model_folder)
        result_dir = Path(self.config.results_folder)
        model_dir.mkdir(parents=True, exist_ok=True)
        result_dir.mkdir(parents=True, exist_ok=True)

        artifact = {
            "models": self.models,
            "pca": self.pca,
            "feature_names": self.feature_names,
            "config": self.config.to_dict(),
            "best_metrics": best_metrics,
            "aux_mean": self.train_dataset.aux_mean,
            "aux_std": self.train_dataset.aux_std,
            "spectra_mean": self.train_dataset.spectra_mean,
            "spectra_std": self.train_dataset.spectra_std,
        }
        with open(model_dir / f"{self.config.paths.model_basename}best.pkl", "wb") as f:
            pickle.dump(artifact, f)

        importances = pd.DataFrame({"feature": self.feature_names})
        for label_name, model in self.models.items():
            importances[f"importance_{label_name}"] = model.feature_importances_
        importances.to_csv(result_dir / "feature_importances.csv", index=False)

        val_pred = (val_probs >= self.config.training.classification_threshold).astype(int)
        pred_df = pd.DataFrame({
            "id": val_ids,
            "prob_eau": val_probs[:, 0],
            "prob_nuage": val_probs[:, 1],
            "pred_eau": val_pred[:, 0],
            "pred_nuage": val_pred[:, 1],
            "true_eau": y_val[:, 0],
            "true_nuage": y_val[:, 1],
        })
        pred_df.to_csv(result_dir / "validation_predictions.csv", index=False)

        per_label = {}
        for idx, label_name in enumerate(LABEL_NAMES):
            threshold, mcc_at_t = _find_optimal_threshold(y_val[:, idx], val_probs[:, idx])
            per_label[label_name] = {
                "train": _metrics_from_probs(
                    y_train[:, idx],
                    train_probs[:, idx],
                    self.config.training.classification_threshold,
                ),
                "val": _metrics_from_probs(
                    y_val[:, idx],
                    val_probs[:, idx],
                    self.config.training.classification_threshold,
                ),
                "best_threshold": threshold,
                "mcc_at_best_threshold": mcc_at_t,
            }
        with open(result_dir / "per_label_metrics.json", "w") as f:
            json.dump(per_label, f, indent=2, cls=NumpyEncoder)

        with open(result_dir / "lightgbm_history.json", "w") as f:
            json.dump({"history": history, "best_metrics": best_metrics}, f, indent=2, cls=NumpyEncoder)

    def train(self) -> Dict[str, Any]:
        start = time.time()
        x_train, y_train, _, feature_names = _build_tabular_features(self.train_dataset, self.config)
        x_val, y_val, val_ids, val_feature_names = _build_tabular_features(self.val_dataset, self.config)
        if feature_names != val_feature_names:
            raise RuntimeError("Train/val feature names do not match.")

        x_train, x_val, feature_names, self.pca = _fit_optional_pca(
            x_train, x_val, feature_names, self.config
        )
        self.feature_names = feature_names

        logger.info(
            "LightGBM | train %s | val %s | %d features | n_estim=%d  lr=%.4g  "
            "patience=%d  aug=%d  pca=%s",
            tuple(x_train.shape), tuple(x_val.shape), len(feature_names),
            self.config.model.n_estimators, self.config.training.learning_rate,
            self.config.training.patience, self.config.data.augmentation_factor,
            f"{self.pca.n_components_}" if self.pca is not None else "off",
        )

        x_train_frame = _as_frame(x_train, feature_names)
        x_val_frame = _as_frame(x_val, feature_names)
        train_probs = np.zeros_like(y_train, dtype=np.float32)
        val_probs = np.zeros_like(y_val, dtype=np.float32)

        for idx, label_name in enumerate(LABEL_NAMES):
            t0 = time.time()
            model = self._make_model()
            model.fit(
                x_train_frame,
                y_train[:, idx],
                eval_set=[(x_val_frame, y_val[:, idx])],
                eval_names=["val"],
                eval_metric="binary_logloss",
                callbacks=[
                    lgb.early_stopping(
                        stopping_rounds=self.config.training.patience,
                        verbose=False,
                    ),
                ],
            )
            self.models[label_name] = model
            train_probs[:, idx] = model.predict_proba(x_train_frame)[:, 1]
            val_probs[:, idx] = model.predict_proba(x_val_frame)[:, 1]
            best_iter = model.best_iteration_ or model.n_estimators_
            val_evals = (model.evals_result_ or {}).get("val", {})
            ll_series = val_evals.get("binary_logloss") or []
            final_ll = ll_series[best_iter - 1] if 0 < best_iter <= len(ll_series) else float("nan")
            logger.info(
                "  → %-6s | best_iter=%4d/%d  val_logloss=%.4f  (%.1fs)",
                label_name, best_iter, self.config.model.n_estimators,
                final_ll, time.time() - t0,
            )

        primary = PRIMARY_LABEL_INDEX
        threshold = self.config.training.classification_threshold
        train_metrics = _metrics_from_probs(
            y_train[:, primary], train_probs[:, primary], threshold,
        )
        val_metrics = _metrics_from_probs(
            y_val[:, primary], val_probs[:, primary], threshold,
        )

        train_metrics["loss"] = float(np.mean([
            _binary_log_loss(y_train[:, idx], train_probs[:, idx])
            for idx in range(y_train.shape[1])
        ]))
        val_metrics["loss"] = float(np.mean([
            _binary_log_loss(y_val[:, idx], val_probs[:, idx])
            for idx in range(y_val.shape[1])
        ]))

        best_t, mcc_at_best_t = _find_optimal_threshold(
            y_val[:, primary], val_probs[:, primary],
        )
        val_metrics["best_threshold"] = best_t
        val_metrics["mcc_at_best_threshold"] = mcc_at_best_t

        val_metrics_per_label: Dict[str, Dict[str, float]] = {}
        for idx, label_name in enumerate(LABEL_NAMES):
            m = _metrics_from_probs(y_val[:, idx], val_probs[:, idx], threshold)
            bt, mcc_bt = _find_optimal_threshold(y_val[:, idx], val_probs[:, idx])
            m["best_threshold"] = bt
            m["mcc_at_best_threshold"] = mcc_bt
            val_metrics_per_label[label_name] = m

        history, checkpoint_iters = _build_iteration_history(
            self.models,
            x_train_frame, y_train,
            x_val_frame, y_val,
            threshold=threshold,
            primary=primary,
            n_checkpoints=30,
        )
        history["checkpoint_iterations"] = [int(it) for it in checkpoint_iters]
        history["val_best_threshold"] = [float(best_t)]
        history["val_mcc_at_best_threshold"] = [float(mcc_at_best_t)]
        best_metrics = {
            "mcc": val_metrics["mcc"],
            "mcc_at_best_threshold": val_metrics["mcc_at_best_threshold"],
            "best_threshold": val_metrics["best_threshold"],
            "composite_score": val_metrics["composite_score"],
            "g_mean": val_metrics["g_mean"],
            "stability_score": val_metrics["stability_score"],
            "production_score": val_metrics["production_score"],
            "auroc": val_metrics["auroc"],
            "f_harmonic": val_metrics["f_harmonic"],
        }

        self._save_artifacts(
            y_train=y_train,
            y_val=y_val,
            train_probs=train_probs,
            val_probs=val_probs,
            val_ids=val_ids,
            history=history,
            best_metrics=best_metrics,
        )

        elapsed = time.time() - start
        # Log par label (le résumé principal côté primary est géré par
        # training.utils.log_training_summary, appelé après train()).
        mcc_means: List[float] = []
        for label_name, m in val_metrics_per_label.items():
            mcc_means.append(m["mcc"])
            logger.info(
                "Val %-6s | MCC %.4f (opt %.4f @ t=%.2f)  F1 %.4f  AUROC %.4f",
                label_name, m["mcc"], m["mcc_at_best_threshold"],
                m["best_threshold"], m["f1"], m["auroc"],
            )
        logger.info(
            "Val mean MCC (eau, nuage) : %.4f  |  durée %.1fs",
            float(np.mean(mcc_means)), elapsed,
        )

        n_checkpoints_built = len(history.get("val_mcc", []))
        final_epoch = max(1, n_checkpoints_built)
        return {
            "best_metrics": best_metrics,
            "best_epoch": final_epoch,
            "final_epoch": final_epoch,
            "training_time": elapsed,
            "early_stopped": False,
            "history": history,
            "best_threshold": best_t,
            "val_probs": val_probs,
            "val_labels": y_val,
            "val_ids": val_ids,
            "train_probs": train_probs,
            "train_labels": y_train,
        }
