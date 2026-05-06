"""K-fold cross-validation stratifié avec out-of-fold predictions et ensembling.

Pourquoi ?
  Avec un single split 80/20 (≈600 val), la variance de MCC est de l'ordre
  de ±0.01 — la majorité des "améliorations" inférieures à ce bruit sont
  illusoires. Le k-fold stratifié donne :
    - une estimation MCC robuste (moyenne sur K validations indépendantes)
    - K modèles → moyenne de leurs probas sur le test set (ensemble bag)
    - des out-of-fold predictions sur 100 % du train (utile pour stacking)

Usage :
  cross_validate(cfg, n_splits=5, seeds=[42])  → entraîne K modèles, écrit
  les checkpoints dans cfg.paths.model_folder/fold_{k}/, sauvegarde les OOF
  dans cfg.results_folder/oof_predictions.npz.
"""

from __future__ import annotations

import copy
import logging
from pathlib import Path
from typing import List, Sequence

import numpy as np
from sklearn.metrics import matthews_corrcoef
from sklearn.model_selection import StratifiedKFold


logger = logging.getLogger(__name__)


def _stratify_labels(targets_df) -> np.ndarray:
    """Combine eau+nuage en une étiquette unique pour stratification (4 classes)."""
    return (targets_df['eau'].astype(str) + "_" + targets_df['nuage'].astype(str)).values


def _evaluate_oof(probs: np.ndarray, labels: np.ndarray) -> dict:
    """Métriques sur les OOF predictions (toutes classes confondues, deux têtes eau/nuage).

    probs shape  : (N, 2) — sigmoïde des logits (eau, nuage)
    labels shape : (N, 2)
    """
    metrics = {}
    for i, name in enumerate(['eau', 'nuage']):
        y_true = labels[:, i].astype(int)
        # On cherche le seuil optimal en MCC (équilibre FP/FN)
        best_t, best_mcc = 0.5, -1.0
        for t in np.linspace(0.2, 0.8, 61):
            y_pred = (probs[:, i] >= t).astype(int)
            mcc = matthews_corrcoef(y_true, y_pred)
            if mcc > best_mcc:
                best_t, best_mcc = float(t), float(mcc)
        metrics[f'mcc_{name}'] = best_mcc
        metrics[f'best_threshold_{name}'] = best_t
        # Au seuil 0.5 par défaut aussi
        metrics[f'mcc_{name}_at_0.5'] = matthews_corrcoef(
            y_true, (probs[:, i] >= 0.5).astype(int)
        )
    metrics['mcc_mean'] = (metrics['mcc_eau'] + metrics['mcc_nuage']) / 2
    return metrics


def cross_validate(
    cfg,
    n_splits: int = 5,
    seeds: Sequence[int] = (42,),
    fold_indices: Sequence[int] | None = None,
) -> dict:
    """Lance un k-fold CV stratifié avec optionnellement plusieurs seeds par fold.

    Args:
        cfg          : Config principale (paths, training, data…)
        n_splits     : nombre de folds (défaut 5)
        seeds        : seeds pour réinitialisation modèle/loaders. Si len(seeds) > 1,
                       chaque fold est entraîné `len(seeds)` fois → ensemble seed×fold.
        fold_indices : sous-ensemble de folds à entraîner (ex: [0, 2] pour reprendre
                       un run interrompu). None → tous.

    Returns:
        dict avec :
            'oof_probs'       (N, 2)
            'oof_labels'      (N, 2)
            'fold_metrics'    list[dict]
            'aggregate'       dict des métriques OOF (mcc_eau, mcc_nuage, mcc_mean…)
    """
    # Imports locaux pour éviter les cycles à l'import du module
    from main import (
        build_dataloaders_from_indices,
        build_model,
        load_raw_data,
        _select_trainer,
    )
    from training.utils import set_seed

    data = load_raw_data(cfg)
    n_samples = len(data['spectra'])
    strat = _stratify_labels(data['targets'])

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=cfg.data.random_seed)
    folds = list(skf.split(np.zeros(n_samples), strat))

    if fold_indices is not None:
        folds_to_run = [(i, folds[i]) for i in fold_indices]
    else:
        folds_to_run = list(enumerate(folds))

    # Conteneurs OOF
    oof_probs  = np.zeros((n_samples, 2), dtype=np.float32)
    oof_labels = np.zeros((n_samples, 2), dtype=np.float32)
    oof_count  = np.zeros(n_samples, dtype=np.int32)   # >1 si plusieurs seeds

    fold_metrics = []

    base_model_folder    = cfg.paths.model_folder
    base_results_folder  = cfg.results_folder
    base_experiment_name = cfg.paths.experiment_name

    for fold_id, (train_idx, val_idx) in folds_to_run:
        for seed in seeds:
            tag = f"fold{fold_id}" + (f"_seed{seed}" if len(seeds) > 1 else "")
            logger.info("=" * 90)
            logger.info("CV fold %d/%d  |  seed %d  |  tag=%s", fold_id + 1, n_splits, seed, tag)
            logger.info("=" * 90)

            # Config dédiée à ce (fold, seed) — paths isolés
            fold_cfg = copy.deepcopy(cfg)
            fold_cfg.paths.model_folder    = f"{base_model_folder}/{tag}"
            fold_cfg.paths.experiment_name = f"{base_experiment_name}/{tag}"
            fold_cfg.results_folder        = f"{base_results_folder}/{tag}"
            Path(fold_cfg.paths.model_folder).mkdir(parents=True, exist_ok=True)
            Path(fold_cfg.results_folder).mkdir(parents=True, exist_ok=True)

            set_seed(seed)

            train_loader, val_loader, dataset_info = build_dataloaders_from_indices(
                data, train_idx, val_idx, fold_cfg,
            )
            model = build_model(
                fold_cfg,
                dataset_info['spectrum_length'],
                dataset_info['auxiliary_dim'],
            )
            trainer = _select_trainer(model, train_loader, val_loader, fold_cfg)
            result = trainer.train()

            # OOF predictions sur ce fold avec le best checkpoint
            best_path = Path(fold_cfg.paths.model_folder) / f"{fold_cfg.paths.model_basename}best.pth"
            fold_probs, fold_labels = _predict_dataset(
                model, val_loader, fold_cfg.device, best_path,
            )
            oof_probs[val_idx]  += fold_probs
            oof_labels[val_idx]  = fold_labels
            oof_count[val_idx]  += 1

            fm = {
                'fold': fold_id,
                'seed': seed,
                'best_epoch':    result.get('best_epoch'),
                'final_epoch':   result.get('final_epoch'),
                'training_time': result.get('training_time'),
            }
            fold_metrics.append(fm)
            logger.info("Fold %d seed %d terminé : best_epoch=%s, durée=%.1fmin",
                        fold_id, seed, fm['best_epoch'], (fm['training_time'] or 0) / 60)

    # Moyenne si plusieurs seeds par fold
    mask = oof_count > 0
    oof_probs[mask] /= oof_count[mask, None]

    aggregate = _evaluate_oof(oof_probs[mask], oof_labels[mask])
    logger.info("=" * 90)
    logger.info("OOF AGRÉGÉ (n=%d)", int(mask.sum()))
    for k, v in aggregate.items():
        logger.info("  %-25s %.4f", k, v)
    logger.info("=" * 90)

    # Sauvegarde
    out_path = Path(base_results_folder) / 'oof_predictions.npz'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, probs=oof_probs, labels=oof_labels, count=oof_count, mask=mask)
    logger.info("OOF sauvegardé : %s", out_path)

    return {
        'oof_probs':  oof_probs,
        'oof_labels': oof_labels,
        'oof_mask':   mask,
        'fold_metrics': fold_metrics,
        'aggregate':  aggregate,
    }


def _predict_dataset(model, loader, device, checkpoint_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Charge le best checkpoint puis prédit sur tout le loader. Renvoie (probs, labels)."""
    import torch

    if checkpoint_path.exists():
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        state = ckpt.get('model_state_dict', ckpt)
        # Retire le préfixe DataParallel si présent
        state = {k[7:] if k.startswith('module.') else k: v for k, v in state.items()}
        try:
            model.load_state_dict(state, strict=True)
        except RuntimeError:
            model.load_state_dict(state, strict=False)

    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            if len(batch) == 4:
                spectra, auxiliary, labels, _ = batch
            else:
                spectra, auxiliary, _ = batch
                labels = None
            spectra   = spectra.to(device)
            auxiliary = auxiliary.to(device)
            logits    = model(spectra, auxiliary)
            probs     = torch.sigmoid(logits).cpu().numpy()
            all_probs.append(probs)
            if labels is not None:
                all_labels.append(labels.numpy())

    probs  = np.concatenate(all_probs, axis=0)
    labels = np.concatenate(all_labels, axis=0) if all_labels else np.zeros_like(probs)
    return probs, labels
