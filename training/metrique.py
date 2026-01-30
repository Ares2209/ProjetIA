"""Module pour le calcul des métriques de classification binaire enrichies."""

import torch
import numpy as np
from typing import Dict, Optional, List, Tuple
from dataclasses import dataclass
from sklearn.metrics import roc_auc_score, average_precision_score

@dataclass
class BinaryMetrics:
    """Conteneur pour les métriques binaires enrichies."""
    # Métriques de base
    accuracy: float
    balanced_accuracy: float
    precision: float
    recall: float
    specificity: float
    f1: float
    f2: float 
    iou: float
    
    # Métriques pour classes déséquilibrées
    mcc: float  # Matthews Correlation Coefficient
    cohen_kappa: float
    
    # Métriques probabilistes (optionnelles)
    auroc: Optional[float] = None  # Area Under ROC Curve
    auprc: Optional[float] = None  # Area Under Precision-Recall Curve
    brier_score: Optional[float] = None
    
    # Métriques par classe
    class_0_precision: float = 0.0
    class_0_recall: float = 0.0
    class_1_precision: float = 0.0
    class_1_recall: float = 0.0
    
    # Support (nombre d'échantillons par classe)
    support_class_0: int = 0
    support_class_1: int = 0
    
    # Matrice de confusion
    tp: int = 0
    fp: int = 0
    tn: int = 0
    fn: int = 0

    def to_dict(self) -> Dict[str, float]:
        """Convertit en dictionnaire."""
        result = {
            # Métriques de base
            'accuracy': self.accuracy,
            'balanced_accuracy': self.balanced_accuracy,
            'precision': self.precision,
            'recall': self.recall,
            'specificity': self.specificity,
            'f1': self.f1,
            'f2': self.f2,
            'iou': self.iou,
            
            # Métriques pour déséquilibre
            'mcc': self.mcc,
            'cohen_kappa': self.cohen_kappa,
            
            # Par classe
            'class_0_precision': self.class_0_precision,
            'class_0_recall': self.class_0_recall,
            'class_1_precision': self.class_1_precision,
            'class_1_recall': self.class_1_recall,
            
            # Support
            'support_class_0': self.support_class_0,
            'support_class_1': self.support_class_1,
            
            # Matrice de confusion
            'tp': self.tp,
            'fp': self.fp,
            'tn': self.tn,
            'fn': self.fn
        }
        
        # Ajouter métriques probabilistes si disponibles
        if self.auroc is not None:
            result['auroc'] = self.auroc
        if self.auprc is not None:
            result['auprc'] = self.auprc
        if self.brier_score is not None:
            result['brier_score'] = self.brier_score
            
        return result

    def __getitem__(self, key: str):
        """Permet l'accès par clé comme un dictionnaire."""
        return getattr(self, key)

    def __setitem__(self, key: str, value):
        """Permet la modification par clé comme un dictionnaire."""
        setattr(self, key, value)

    def keys(self):
        """Retourne les clés disponibles."""
        return self.to_dict().keys()

    def items(self):
        """Retourne les paires clé-valeur."""
        return self.to_dict().items()

    def values(self):
        """Retourne les valeurs."""
        return self.to_dict().values()


class MetricsCalculator:
    """Calculateur de métriques optimisé pour la classification binaire."""

    def __init__(self, threshold: float = 0.5, eps: float = 1e-8):
        """
        Args:
            threshold: Seuil de classification
            eps: Epsilon pour stabilité numérique
        """
        self.threshold = threshold
        self.eps = eps

    @torch.no_grad()
    def compute_metrics(
        self, 
        predictions: torch.Tensor, 
        labels: torch.Tensor, 
        masks: Optional[torch.Tensor] = None
    ) -> BinaryMetrics:
        """
        Calcule toutes les métriques de classification binaire (pour un batch).

        Args:
            predictions: Logits ou probabilités [batch, seq_len]
            labels: Labels binaires [batch, seq_len]
            masks: Masque optionnel [batch, seq_len]

        Returns:
            BinaryMetrics: Object contenant toutes les métriques
        """
        # Conversion en probabilités si nécessaire
        probs = torch.sigmoid(predictions)

        # Binarisation
        preds_binary = (probs > self.threshold)

        # Application du masque
        if masks is not None:
            mask_bool = masks.bool()
            preds_binary = preds_binary[mask_bool]
            labels = labels[mask_bool]
            probs = probs[mask_bool]
        else:
            preds_binary = preds_binary.flatten()
            labels = labels.flatten()
            probs = probs.flatten()

        # Conversion en bool pour les opérations logiques
        labels_bool = labels.bool()

        # Calcul des confusion matrix elements
        tp = torch.logical_and(preds_binary, labels_bool).sum()
        fp = torch.logical_and(preds_binary, ~labels_bool).sum()
        tn = torch.logical_and(~preds_binary, ~labels_bool).sum()
        fn = torch.logical_and(~preds_binary, labels_bool).sum()

        # Conversion en int puis calcul
        return self.compute_metrics_from_confusion_matrix(
            tp=tp.item(),
            fp=fp.item(),
            tn=tn.item(),
            fn=fn.item(),
            probs=probs.cpu().numpy(),
            labels=labels.cpu().numpy(),
            eps=self.eps
        )

    @staticmethod
    def compute_metrics_from_confusion_matrix(
        tp: int, 
        fp: int, 
        tn: int, 
        fn: int,
        probs: Optional[np.ndarray] = None,
        labels: Optional[np.ndarray] = None,
        eps: float = 1e-8
    ) -> BinaryMetrics:
        """
        Calcule les métriques à partir des valeurs de la matrice de confusion.
        IMPORTANT: À utiliser pour calculer les métriques finales à partir des TP/FP/TN/FN cumulés.

        Args:
            tp: True Positives (cumulés)
            fp: False Positives (cumulés)
            tn: True Negatives (cumulés)
            fn: False Negatives (cumulés)
            probs: Probabilités prédites (optionnel, pour métriques probabilistes)
            labels: Labels vrais (optionnel, pour métriques probabilistes)
            eps: Epsilon pour stabilité numérique

        Returns:
            BinaryMetrics: Métriques calculées sur les valeurs cumulées
        """
        tp_f = float(tp)
        fp_f = float(fp)
        tn_f = float(tn)
        fn_f = float(fn)

        # Support par classe
        support_class_0 = tn + fp  # Nombre de vrais négatifs
        support_class_1 = tp + fn  # Nombre de vrais positifs

        # Métriques de base
        accuracy = (tp_f + tn_f) / (tp_f + tn_f + fp_f + fn_f + eps)
        
        # Precision/Recall/Specificity
        precision = tp_f / (tp_f + fp_f + eps)
        recall = tp_f / (tp_f + fn_f + eps)  # Aussi appelé TPR (True Positive Rate)
        specificity = tn_f / (tn_f + fp_f + eps)  # TNR (True Negative Rate)
        
        # Balanced Accuracy: moyenne de recall et specificity
        balanced_accuracy = (recall + specificity) / 2.0
        
        # F-scores
        f1 = 2 * (precision * recall) / (precision + recall + eps)
        # F2 donne 2x plus de poids au recall qu'à la precision
        f2 = 5 * (precision * recall) / (4 * precision + recall + eps)
        
        # IoU (Intersection over Union)
        iou = tp_f / (tp_f + fp_f + fn_f + eps)
        
        # Matthews Correlation Coefficient (MCC)
        mcc_num = (tp_f * tn_f) - (fp_f * fn_f)
        mcc_den = np.sqrt((tp_f + fp_f) * (tp_f + fn_f) * (tn_f + fp_f) * (tn_f + fn_f))
        mcc = mcc_num / (mcc_den + eps)
        
        # Cohen's Kappa
        po = accuracy  # Observed agreement
        pe = ((tp_f + fp_f) * (tp_f + fn_f) + (tn_f + fn_f) * (tn_f + fp_f)) / ((tp_f + tn_f + fp_f + fn_f) ** 2 + eps)
        cohen_kappa = (po - pe) / (1 - pe + eps)
        
        # Métriques par classe
        # Classe 0 (non visible)
        class_0_precision = tn_f / (tn_f + fn_f + eps)
        class_0_recall = tn_f / (tn_f + fp_f + eps)
        
        # Classe 1 (visible) - c'est juste precision/recall globaux
        class_1_precision = precision
        class_1_recall = recall
        
        # Métriques probabilistes (si disponibles)
        auroc = None
        auprc = None
        brier_score = None
        
        if probs is not None and labels is not None and len(np.unique(labels)) > 1:
            try:
                # AUROC
                auroc = roc_auc_score(labels, probs)
                
                # AUPRC (Average Precision)
                auprc = average_precision_score(labels, probs)
                
                # Brier Score (plus bas = meilleur)
                brier_score = np.mean((probs - labels) ** 2)
                
            except Exception as e:
                print(f"⚠️  Erreur calcul métriques probabilistes: {e}")

        return BinaryMetrics(
            accuracy=accuracy,
            balanced_accuracy=balanced_accuracy,
            precision=precision,
            recall=recall,
            specificity=specificity,
            f1=f1,
            f2=f2,
            iou=iou,
            
            # Métriques pour déséquilibre
            mcc=mcc,
            cohen_kappa=cohen_kappa,
            
            # Métriques probabilistes
            auroc=auroc,
            auprc=auprc,
            brier_score=brier_score,
            
            # Par classe
            class_0_precision=class_0_precision,
            class_0_recall=class_0_recall,
            class_1_precision=class_1_precision,
            class_1_recall=class_1_recall,
            
            # Support
            support_class_0=support_class_0,
            support_class_1=support_class_1,
            
            # Matrice de confusion
            tp=tp,
            fp=fp,
            tn=tn,
            fn=fn
        )


class MetricsAccumulator:
    """Accumule les métriques sur plusieurs batches."""

    def __init__(self, threshold: float = 0.5):
        """
        Args:
            threshold: Seuil de classification
        """
        self.threshold = threshold
        self.reset()

    def reset(self):
        """Réinitialise l'accumulation."""
        self.tp = 0
        self.fp = 0
        self.tn = 0
        self.fn = 0
        self.count = 0
        
        # Pour les métriques probabilistes
        self.all_probs: List[np.ndarray] = []
        self.all_labels: List[np.ndarray] = []

    @torch.no_grad()
    def update(
        self, 
        predictions: torch.Tensor, 
        labels: torch.Tensor, 
        masks: Optional[torch.Tensor] = None,
        store_for_probabilistic: bool = False
    ):
        """
        Accumule les statistiques d'un batch.
        IMPORTANT: On accumule uniquement les TP/FP/TN/FN, pas les métriques!
        
        Args:
            predictions: Prédictions du modèle
            labels: Labels vrais
            masks: Masque optionnel
            store_for_probabilistic: Si True, stocke probs/labels pour AUROC/AUPRC/Brier
        """

        probs = torch.sigmoid(predictions)


        preds_binary = (probs > self.threshold)

        if masks is not None:
            mask_bool = masks.bool()
            preds_binary = preds_binary[mask_bool]
            labels_masked = labels[mask_bool]
            probs_masked = probs[mask_bool]
        else:
            preds_binary = preds_binary.flatten()
            labels_masked = labels.flatten()
            probs_masked = probs.flatten()

        labels_bool = labels_masked.bool()

        # CRUCIAL: On accumule les COMPTEURS en int, pas les métriques en float!
        self.tp += torch.logical_and(preds_binary, labels_bool).sum().item()
        self.fp += torch.logical_and(preds_binary, ~labels_bool).sum().item()
        self.tn += torch.logical_and(~preds_binary, ~labels_bool).sum().item()
        self.fn += torch.logical_and(~preds_binary, labels_bool).sum().item()
        self.count += 1
        
        # Stocker pour métriques probabilistes si demandé
        if store_for_probabilistic:
            self.all_probs.append(probs_masked.cpu().numpy())
            self.all_labels.append(labels_masked.cpu().numpy())

    def compute(self, eps: float = 1e-8, compute_probabilistic: bool = False) -> BinaryMetrics:
        """
        Calcule les métriques finales à partir des TP/FP/TN/FN cumulés.
        C'est ici qu'on calcule F1 et IoU sur les VRAIS totaux, pas sur des moyennes!
        
        Args:
            eps: Epsilon pour stabilité numérique
            compute_probabilistic: Si True, calcule AUROC/AUPRC/Brier (nécessite store_for_probabilistic=True dans update)
        """
        probs_all = None
        labels_all = None
        
        if compute_probabilistic and len(self.all_probs) > 0:
            probs_all = np.concatenate(self.all_probs)
            labels_all = np.concatenate(self.all_labels)
        
        return MetricsCalculator.compute_metrics_from_confusion_matrix(
            tp=self.tp,
            fp=self.fp,
            tn=self.tn,
            fn=self.fn,
            probs=probs_all,
            labels=labels_all,
            eps=eps
        )

    def get_confusion_matrix(self) -> Dict[str, int]:
        """Retourne les valeurs brutes de la matrice de confusion."""
        return {
            'tp': self.tp,
            'fp': self.fp,
            'tn': self.tn,
            'fn': self.fn
        }