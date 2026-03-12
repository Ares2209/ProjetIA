"""Module pour les fonctions de loss personnalisées."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class BCE(nn.Module):
    """
    Binary Cross Entropy Loss pour classification multi-label (Eau / Nuages).

    Supporte :
      - inputs 2D (batch, num_classes)          ← CNN, MLP
      - inputs 3D (batch, seq_len, num_classes) ← Transformer, RNN, avec masque optionnel

    Args:
        pos_weight:      Poids pour chaque classe positive (shape: [num_classes]).
                         Enregistré comme buffer → suit automatiquement .to() / .cuda().
        reduction:       'mean' | 'sum' | 'none'
        label_smoothing: Valeur dans [0, 1) — 0 = désactivé
        class_names:     Noms des classes (pour compute_class_losses)
    """

    def __init__(
        self,
        pos_weight: Optional[torch.Tensor] = None,
        reduction: str = 'mean',
        label_smoothing: float = 0.0,
        class_names: Optional[list] = None,
    ):
        super().__init__()

        if not 0.0 <= label_smoothing < 1.0:
            raise ValueError(f"label_smoothing doit être dans [0, 1), reçu : {label_smoothing}")
        if reduction not in ('mean', 'sum', 'none'):
            raise ValueError(f"reduction doit être 'mean', 'sum' ou 'none', reçu : {reduction}")

        if pos_weight is not None:
            self.register_buffer('pos_weight', pos_weight)
        else:
            self.pos_weight = None

        self.reduction      = reduction
        self.label_smoothing = label_smoothing
        self.class_names    = class_names or ['Eau', 'Nuages']

    def _smooth(self, labels: torch.Tensor) -> torch.Tensor:
        """Label smoothing symétrique : 1 → (1 − ε), 0 → ε/2."""
        if self.label_smoothing > 0.0:
            return labels * (1.0 - self.label_smoothing) + self.label_smoothing / 2.0
        return labels

    @staticmethod
    def _check_shapes(predictions: torch.Tensor, labels: torch.Tensor) -> None:
        if predictions.shape != labels.shape:
            raise ValueError(
                f"Shape mismatch : predictions={tuple(predictions.shape)}, "
                f"labels={tuple(labels.shape)}"
            )

    def forward(
        self,
        predictions: torch.Tensor,
        labels: torch.Tensor,
        masks: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            predictions: (batch, C) ou (batch, seq_len, C) — logits bruts
            labels:      même shape que predictions
            masks:       (batch,) ou (batch, seq_len) — 1=valide, 0=padding.
                         Ignoré (avec avertissement) si predictions est 2D.
        Returns:
            Scalaire si reduction != 'none', sinon tenseur de même shape que predictions.
        """
        self._check_shapes(predictions, labels)

        is_2d = predictions.dim() == 2

        if masks is not None and is_2d:
            raise ValueError(
                "Les masques (masks) ne sont supportés qu'avec des inputs 3D (batch, seq_len, C). "
                "Pour un input 2D (batch, C), passez masks=None."
            )

        if is_2d or masks is None:
            return F.binary_cross_entropy_with_logits(
                predictions,
                self._smooth(labels.float()),
                pos_weight=self.pos_weight,
                reduction=self.reduction,
            )

        valid_mask = masks.unsqueeze(-1).expand_as(predictions).bool()

        loss = F.binary_cross_entropy_with_logits(
            predictions[valid_mask],
            self._smooth(labels[valid_mask].float()),
            pos_weight=self.pos_weight,
            reduction='none',
        )

        if self.reduction == 'mean':
            return loss.mean()
        if self.reduction == 'sum':
            return loss.sum()
        # 'none' : reconstituer un tenseur de même shape avec 0 sur les positions masquées
        out = torch.zeros_like(predictions)
        out[valid_mask] = loss
        return out

    def compute_class_losses(
        self,
        predictions: torch.Tensor,
        labels: torch.Tensor,
        masks: Optional[torch.Tensor] = None,
    ) -> dict:
        """Retourne les losses par classe + loss totale (monitoring uniquement, pas le backward).

        Compatible 2D (batch, C) et 3D (batch, seq_len, C).
        """
        self._check_shapes(predictions, labels)
        is_2d = predictions.dim() == 2

        if is_2d and masks is not None:
            raise ValueError("masks n'est pas supporté avec des inputs 2D.")

        losses = {}

        for i, name in enumerate(self.class_names):
            if is_2d:
                preds_i = predictions[:, i]
                lbls_i  = labels[:, i]
            else:
                # Masque : (batch, seq_len) → bool
                if masks is not None:
                    valid = masks.bool()
                    preds_i = predictions[:, :, i][valid]
                    lbls_i  = labels[:, :, i][valid]
                else:
                    preds_i = predictions[:, :, i].reshape(-1)
                    lbls_i  = labels[:, :, i].reshape(-1)

            losses[f'loss_{name.lower()}'] = F.binary_cross_entropy_with_logits(
                preds_i,
                self._smooth(lbls_i.float()),
                reduction='mean',
            ).item()

        losses['loss_total'] = self.forward(predictions, labels, masks).item()
        return losses