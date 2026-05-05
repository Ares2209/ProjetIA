"""Utilitaires divers pour l'entraînement.

Inclut :
- Helpers reproductibilité / device / formatage
- Configuration du logging (`setup_logging`)
- Sauvegarde de l'historique d'entraînement (CSV + rapport texte)
- Affichage du résumé post-entraînement (`log_training_summary`)
"""

import json
import logging
import random
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import torch


logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# REPRODUCTIBILITÉ / DEVICE
# ─────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int = 42):
    """Fixe toutes les graines aléatoires pour reproductibilité."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device_info() -> Dict[str, Any]:
    """Retourne les informations sur le device disponible."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    info = {'device': device}

    if device == 'cuda':
        info['gpu_name'] = torch.cuda.get_device_name(0)
        info['cuda_version'] = torch.version.cuda
        info['total_memory_gb'] = torch.cuda.get_device_properties(0).total_memory / 1024**3
        info['num_gpus'] = torch.cuda.device_count()

    return info


def count_parameters(model: torch.nn.Module) -> Dict[str, int]:
    """Compte les paramètres du modèle."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return {'total': total, 'trainable': trainable, 'frozen': total - trainable}


def format_time(seconds: float) -> str:
    """Formate une durée en secondes en format lisible."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)

    if hours > 0:
        return f"{hours}h {minutes}min {secs}s"
    elif minutes > 0:
        return f"{minutes}min {secs}s"
    else:
        return f"{secs}s"


def get_lr(optimizer: torch.optim.Optimizer) -> float:
    """Récupère le learning rate actuel."""
    for param_group in optimizer.param_groups:
        return param_group['lr']
    return 0.0


# ─────────────────────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────────────────────

class _ColorFormatter(logging.Formatter):
    """Formatter avec couleurs ANSI selon le niveau."""

    COLORS = {
        logging.DEBUG:    '\033[37m',   # gris
        logging.INFO:     '\033[36m',   # cyan
        logging.WARNING:  '\033[33m',   # jaune
        logging.ERROR:    '\033[31m',   # rouge
        logging.CRITICAL: '\033[1;31m', # rouge gras
    }
    RESET = '\033[0m'

    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelno, '')
        levelname = f"{color}{record.levelname:<7}{self.RESET}" if color else record.levelname
        record.levelname = levelname
        return super().format(record)


def setup_logging(level: int = logging.INFO, log_file: Optional[str] = None,
                  use_colors: bool = True) -> None:
    """Configure le logger racine pour un rendu propre.

    Args:
        level: niveau global (INFO par défaut).
        log_file: si fourni, duplique la sortie dans ce fichier (sans couleurs).
        use_colors: applique les couleurs ANSI sur la sortie console.
    """
    fmt = '%(asctime)s | %(levelname)-7s | %(message)s'
    datefmt = '%H:%M:%S'

    console = logging.StreamHandler(sys.stdout)
    if use_colors and sys.stdout.isatty():
        console.setFormatter(_ColorFormatter(fmt, datefmt=datefmt))
    else:
        console.setFormatter(logging.Formatter(fmt, datefmt=datefmt))

    handlers = [console]
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, encoding='utf-8')
        fh.setFormatter(logging.Formatter(fmt, datefmt=datefmt))
        handlers.append(fh)

    root = logging.getLogger()
    root.handlers.clear()
    for h in handlers:
        root.addHandler(h)
    root.setLevel(level)


# ─────────────────────────────────────────────────────────────────────────────
# JSON ENCODER
# ─────────────────────────────────────────────────────────────────────────────

class NumpyEncoder(json.JSONEncoder):
    """Encodeur JSON pour les types NumPy."""

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS HISTORIQUE
# ─────────────────────────────────────────────────────────────────────────────

def _get(history: dict, key: str) -> list:
    """Retourne history[key] ou une liste vide si absent."""
    return history.get(key, [])


# ─────────────────────────────────────────────────────────────────────────────
# SAUVEGARDE CSV + RAPPORT
# ─────────────────────────────────────────────────────────────────────────────

def save_training_results(history: dict, save_dir: str = 'results') -> None:
    """Sauvegarde l'historique complet en CSV et écrit un rapport texte."""
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    # Référence de longueur : train_loss si dispo, sinon val_loss (cas LightGBM)
    ref_key = 'train_loss' if _get(history, 'train_loss') else 'val_loss'
    n_epochs = len(_get(history, ref_key))
    if n_epochs == 0:
        logger.warning("Historique vide — pas de CSV à écrire.")
        return

    df_data = {'epoch': range(1, n_epochs + 1)}
    for key, vals in history.items():
        if key == 'iteration_losses':
            continue
        if isinstance(vals, list) and len(vals) == n_epochs:
            df_data[key] = vals

    pd.DataFrame(df_data).to_csv(f'{save_dir}/training_history_epochs.csv', index=False)
    logger.info("Historique epochs : %s/training_history_epochs.csv", save_dir)

    if history.get('iteration_losses'):
        pd.DataFrame({
            'iteration': range(len(history['iteration_losses'])),
            'loss':      history['iteration_losses'],
        }).to_csv(f'{save_dir}/training_history_iterations.csv', index=False)
        logger.info("Historique itérations : %s/training_history_iterations.csv", save_dir)

    _create_summary_report(history, save_dir, n_epochs)


def _create_summary_report(history: dict, save_dir: str, n_epochs: int) -> None:
    """Rapport texte de l'entraînement."""
    METRICS = ['accuracy', 'precision', 'recall', 'f1', 'specificity',
               'balanced_accuracy', 'mcc', 'auroc', 'auprc', 'cohen_kappa']

    def _write_epoch_metrics(f, idx: int):
        for m in METRICS:
            vk, tk = f'val_{m}', f'train_{m}'
            if vk in history and idx < len(history[vk]):
                if _get(history, tk):
                    f.write(f"  Train {m:20s} {history[tk][idx]:.6f}\n")
                f.write(f"  Val   {m:20s} {history[vk][idx]:.6f}\n")

    report_path = Path(save_dir) / 'training_report.txt'

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\nRAPPORT D'ENTRAÎNEMENT\n" + "=" * 80 + "\n\n")
        f.write(f"Epochs : {n_epochs}\n")
        if history.get('iteration_losses'):
            total = len(history['iteration_losses'])
            f.write(f"Itérations totales : {total:,}  ({total // n_epochs:,} / epoch)\n")

        # Meilleure epoch selon val_loss (si disponible)
        if _get(history, 'val_loss'):
            best = int(np.argmin(history['val_loss']))
            f.write(f"\n{'─'*80}\nMEILLEURE EPOCH (val_loss) : {best + 1}\n{'─'*80}\n")
            if _get(history, 'train_loss'):
                f.write(f"  Train Loss : {history['train_loss'][best]:.6f}\n")
            f.write(f"  Val   Loss : {history['val_loss'][best]:.6f}\n")
            _write_epoch_metrics(f, best)

        # Dernière epoch
        f.write(f"\n{'─'*80}\nDERNIÈRE EPOCH : {n_epochs}\n{'─'*80}\n")
        if _get(history, 'train_loss'):
            f.write(f"  Train Loss : {history['train_loss'][-1]:.6f}\n")
        if _get(history, 'val_loss'):
            f.write(f"  Val   Loss : {history['val_loss'][-1]:.6f}\n")
        _write_epoch_metrics(f, n_epochs - 1)

        if n_epochs > 1 and _get(history, 'train_loss'):
            f.write(f"\n{'─'*80}\nAMÉLIORATION (epoch 1 → {n_epochs})\n{'─'*80}\n")
            dl = history['train_loss'][0] - history['train_loss'][-1]
            f.write(f"  Δ Loss train : {dl:+.6f} ({dl/history['train_loss'][0]*100:+.2f}%)\n")
            if _get(history, 'val_mcc'):
                dm = history['val_mcc'][-1] - history['val_mcc'][0]
                f.write(f"  Δ MCC val    : {dm:+.6f}\n")

        f.write("\n" + "=" * 80 + "\n")

    logger.info("Rapport : %s", report_path)


# ─────────────────────────────────────────────────────────────────────────────
# RÉSUMÉ POST-ENTRAÎNEMENT
# ─────────────────────────────────────────────────────────────────────────────

def log_training_summary(result: dict) -> None:
    """Affiche un résumé concis post-entraînement via le logger."""
    history      = result.get('history', {})
    best_metrics = result.get('best_metrics', {})
    best_epoch   = result.get('best_epoch', 1)
    final_epoch  = result.get('final_epoch', 1)
    t            = result.get('training_time', 0)
    idx          = best_epoch - 1

    logger.info("=" * 90)
    logger.info(" RÉSUMÉ POST-ENTRAÎNEMENT")
    logger.info("=" * 90)
    logger.info(
        "Epochs : %d  |  Meilleure : %d  |  Durée : %.2fh (%.1fmin)  |  Early stopping : %s",
        final_epoch, best_epoch, t / 3600, t / 60,
        'Oui' if result.get('early_stopped') else 'Non',
    )

    classic = {
        'Accuracy': 'val_accuracy', 'Balanced Acc': 'val_balanced_accuracy',
        'Precision': 'val_precision', 'Recall': 'val_recall',
        'Specificity': 'val_specificity', 'F1': 'val_f1', 'F2': 'val_f2',
        'IoU': 'val_iou', "Cohen κ": 'val_cohen_kappa', 'MCC': 'val_mcc',
    }
    logger.info("Métriques Val — Epoch %d :", best_epoch)
    for lbl, key in classic.items():
        vals = _get(history, key)
        if vals and idx < len(vals):
            logger.info("    %-16s %.4f", lbl, vals[idx])

    if all(_get(history, k) and idx < len(_get(history, k))
           for k in ['val_tp', 'val_tn', 'val_fp', 'val_fn']):
        tp = int(history['val_tp'][idx])
        tn = int(history['val_tn'][idx])
        fp = int(history['val_fp'][idx])
        fn = int(history['val_fn'][idx])
        logger.info("Confusion (Val) :  TP %5d | FP %5d", tp, fp)
        logger.info("                   FN %5d | TN %5d", fn, tn)

    if best_metrics.get('auroc', 0) > 0:
        parts = [f"AUROC {best_metrics['auroc']:.4f}"]
        auprc = _get(history, 'val_auprc')
        if auprc and idx < len(auprc):
            parts.append(f"AUPRC {auprc[idx]:.4f}")
        brier = _get(history, 'val_brier_score')
        if brier and idx < len(brier):
            parts.append(f"Brier {brier[idx]:.4f}")
        logger.info("Probabilistes :  %s", "  |  ".join(parts))

    logger.info("=" * 90)
