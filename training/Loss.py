"""Module pour les fonctions de loss personnalisées."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from tqdm import tqdm


class BinaryClassificationLoss(nn.Module):
    """
    Binary Cross Entropy Loss (BCE) pour classification multi-label.
    
    Conçue pour prédire simultanément la présence/absence de:
    - Eau (classe 0)
    - Nuages (classe 1)
    
    Formule: BCE = -[y*log(p) + (1-y)*log(1-p)] <= vu en cours, Focal loss c'est bien aussi mais si déséquilibre
    où:
        y = label vrai (0 ou 1)
        p = probabilité prédite (après sigmoid)
    
    Features:
        - Label smoothing pour régularisation
        - Pondération des classes pour déséquilibre
        - Gestion automatique du padding
    """
    
    def __init__(
        self, 
        pos_weight: Optional[torch.Tensor] = None,
        reduction: str = 'mean',
        label_smoothing: float = 0.0,
        class_names: list = None
    ):
        """
        Args:
            pos_weight: Poids pour chaque classe positive (shape: [2])
                       Utile si déséquilibre (ex: plus de spectres sans eau)
                       Valeur > 1 = pénalise plus les faux négatifs
            reduction: 'mean' | 'sum' | 'none'
            label_smoothing: Valeur entre 0 et 1 (ex: 0.1)
                            0 -> pas de smoothing
                            0.1 -> label 1 devient 0.95, label 0 devient 0.05
            class_names: ['eau', 'nuages'] pour affichage
        """
        super().__init__()
        
        self.pos_weight = pos_weight
        self.reduction = reduction
        self.label_smoothing = label_smoothing
        self.class_names = class_names or ['Eau', 'Nuages']
        
        # Vérifications
        if label_smoothing < 0 or label_smoothing >= 1:
            raise ValueError("label_smoothing doit être dans [0, 1)")
        
        self._print_config()
    
    def _print_config(self):
        """Affiche la configuration de la loss."""
        print(f"\n{'='*70}")
        print(f" BINARY CROSS ENTROPY LOSS CONFIGURÉE".center(70))
        print(f"{'='*70}")
        print(f"   • Tâche:                Multi-label (2 classes)")
        print(f"   • Classes:              {self.class_names[0]} | {self.class_names[1]}")
        print(f"   • Réduction:            {self.reduction}")
        
        if self.label_smoothing > 0:
            print(f"   • Label smoothing:      {self.label_smoothing:.3f}")
            print(f"     → Label 1 devient:    {1 - self.label_smoothing:.3f}")
            print(f"     → Label 0 devient:    {self.label_smoothing:.3f}")
        
        if self.pos_weight is not None:
            print(f"   • Pondération classes:")
            for i, name in enumerate(self.class_names):
                print(f"     → {name:8s}: {self.pos_weight[i].item():.3f}x")
        
        print(f"{'='*70}\n")
    
    def forward(
        self, 
        predictions: torch.Tensor, 
        labels: torch.Tensor, 
        masks: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Calcule la Binary Cross Entropy Loss.
        
        Args:
            predictions: (batch, seq_len, 2) - Logits bruts du modèle
                        [:, :, 0] = logits pour l'eau
                        [:, :, 1] = logits pour les nuages
            
            labels: (batch, seq_len, 2) - Labels binaires
                   [:, :, 0] = 1 si eau présente, 0 sinon
                   [:, :, 1] = 1 si nuages présents, 0 sinon
            
            masks: (batch, seq_len) - Masque de padding (optionnel)
                  1 = position valide, 0 = padding à ignorer
        
        Returns:
            loss: Scalaire (si reduction='mean') ou tenseur
        """
        # Vérifications des dimensions
        assert predictions.shape == labels.shape, \
            f"Shape mismatch: predictions {predictions.shape} vs labels {labels.shape}"
        
        batch_size, seq_len, num_classes = predictions.shape
        assert num_classes == 2, f"Attendu 2 classes, reçu {num_classes}"
        
        # Gestion du masque
        if masks is None:
            masks = torch.ones(batch_size, seq_len, device=predictions.device)
        
        # Expand mask pour les 2 classes: (batch, seq_len, 2)
        masks_expanded = masks.unsqueeze(-1).expand_as(predictions)
        
        # Filtrer les positions valides (non-padding)
        valid_mask = masks_expanded.bool()
        valid_predictions = predictions[valid_mask]  # (N,) où N = nombre de positions valides
        valid_labels = labels[valid_mask].float()    # (N,)
        
        # Label smoothing (régularisation)
        if self.label_smoothing > 0:
            # 1 → (1 - ε), 0 → ε
            valid_labels = valid_labels * (1 - self.label_smoothing) + \
                          self.label_smoothing / 2
        
        # Calcul de la Binary Cross Entropy
        # Utilise directement les logits (plus stable numériquement)
        loss = F.binary_cross_entropy_with_logits(
            valid_predictions,
            valid_labels,
            pos_weight=self.pos_weight,
            reduction='none'  # On gère la réduction nous-mêmes
        )
        
        # Réduction selon le paramètre
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            # Reconstruire la forme originale
            output = torch.zeros_like(predictions)
            output[valid_mask] = loss
            return output
    
    def compute_class_losses(
        self,
        predictions: torch.Tensor,
        labels: torch.Tensor,
        masks: Optional[torch.Tensor] = None
    ) -> dict:
        """
        Calcule les losses séparément pour chaque classe.
        Utile pour le monitoring pendant l'entraînement.
        
        Returns:
            dict: {
                'loss_eau': float,
                'loss_nuages': float,
                'loss_total': float
            }
        """
        if masks is None:
            masks = torch.ones(
                predictions.shape[0], 
                predictions.shape[1],
                device=predictions.device
            )
        
        losses = {}
        
        # Loss pour chaque classe
        for i, class_name in enumerate(self.class_names):
            # Extraire les prédictions et labels pour cette classe
            pred_i = predictions[:, :, i]
            label_i = labels[:, :, i]
            
            # Masquer les positions de padding
            valid_mask = masks.bool()
            valid_pred = pred_i[valid_mask]
            valid_label = label_i[valid_mask].float()
            
            # Label smoothing
            if self.label_smoothing > 0:
                valid_label = valid_label * (1 - self.label_smoothing) + \
                             self.label_smoothing / 2
            
            # BCE pour cette classe
            loss_i = F.binary_cross_entropy_with_logits(
                valid_pred,
                valid_label,
                reduction='mean'
            )
            
            losses[f'loss_{class_name.lower()}'] = loss_i.item()
        
        # Loss totale
        losses['loss_total'] = self.forward(predictions, labels, masks).item()
        
        return losses
