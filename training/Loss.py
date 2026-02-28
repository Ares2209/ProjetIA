"""Module pour les fonctions de loss personnalisées."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class BCE(nn.Module):
    """
    Binary Cross Entropy Loss pour classification multi-label (Eau / Nuages).
    
    Args:
        pos_weight: Poids pour chaque classe positive (shape: [2])
        reduction: 'mean' | 'sum' | 'none'
        label_smoothing: Valeur dans [0, 1) — 0 = désactivé
        class_names: Noms des classes pour affichage
    """

    def __init__(
        self,
        pos_weight: Optional[torch.Tensor] = None,
        reduction: str = 'mean',
        label_smoothing: float = 0.0,
        class_names: list = None
    ):
        super().__init__()

        if not 0 <= label_smoothing < 1:
            raise ValueError("label_smoothing doit être dans [0, 1)")

        self.pos_weight = pos_weight
        self.reduction = reduction
        self.label_smoothing = label_smoothing
        self.class_names = class_names or ['Eau', 'Nuages']

    def _smooth(self, labels: torch.Tensor) -> torch.Tensor:
        """Applique le label smoothing: 1 → (1-ε), 0 → ε/2."""
        if self.label_smoothing > 0:
            return labels * (1 - self.label_smoothing) + self.label_smoothing / 2
        return labels

    def forward(
        self,
        predictions: torch.Tensor,
        labels: torch.Tensor,
        masks: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            predictions: (batch, 2) ou (batch, seq_len, 2) — logits bruts
            labels:      même shape que predictions
            masks:       (batch,) ou (batch, seq_len) — 1=valide, 0=padding
                        None si pas de padding
        """
        assert predictions.shape == labels.shape, (
            f"Shape mismatch: predictions={predictions.shape}, labels={labels.shape}"
        )

        # Cas sans masque ou CNN (batch, 2) → pas de padding à gérer
        if masks is None or predictions.dim() == 2:
            return F.binary_cross_entropy_with_logits(
                predictions,
                self._smooth(labels.float()),
                pos_weight=self.pos_weight,
                reduction=self.reduction
            )

        # Cas séquence (batch, seq_len, 2) avec masque
        valid_mask = masks.unsqueeze(-1).expand_as(predictions).bool()

        loss = F.binary_cross_entropy_with_logits(
            predictions[valid_mask],
            self._smooth(labels[valid_mask].float()),
            pos_weight=self.pos_weight,
            reduction='none'
        )

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            out = torch.zeros_like(predictions)
            out[valid_mask] = loss
            return out

    def compute_class_losses(
        self,
        predictions: torch.Tensor,
        labels: torch.Tensor,
        masks: Optional[torch.Tensor] = None
    ) -> dict:
        """Retourne les losses par classe + loss totale (pour monitoring)."""
        if masks is None:
            masks = torch.ones(predictions.shape[:2], device=predictions.device)

        valid_mask = masks.bool()
        losses = {}

        for i, name in enumerate(self.class_names):
            losses[f'loss_{name.lower()}'] = F.binary_cross_entropy_with_logits(
                predictions[:, :, i][valid_mask],
                self._smooth(labels[:, :, i][valid_mask].float()),
                reduction='mean'
            ).item()

        losses['loss_total'] = self.forward(predictions, labels, masks).item()
        return losses
