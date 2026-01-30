"""Architecture CNN pour la classification de spectres d'exoplanètes."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class CNN(nn.Module):
    """
    CNN 1D pour l'analyse de spectres d'exoplanètes avec données auxiliaires.

    Architecture:
        - Branche CNN pour les spectres (3 canaux: moyenne, incert. basse, incert. haute)
        - Branche MLP pour les données auxiliaires
        - Fusion et classification multi-label (eau, nuages)
    """

    def __init__(
        self,
        spectrum_length: int,
        auxiliary_dim: int = 5,
        num_classes: int = 2,
        input_channels: int = 1, 
        conv_channels: list = [32, 64, 128, 256],
        kernel_sizes: list = [7, 5, 3, 3],
        pool_sizes: list = [2, 2, 2, 2],
        fc_dims: list = [256, 128],
        dropout: float = 0.3,
        use_batch_norm: bool = True
    ):
        """
        Args:
            spectrum_length: Longueur du spectre d'entrée
            auxiliary_dim: Dimension des données auxiliaires (5 par défaut)
            num_classes: Nombre de classes de sortie (2: eau, nuages)
            input_channels: Nombre de canaux d'entrée (1 pour moyenne seule, 3 pour moyenne+incertitudes)
            conv_channels: Liste des canaux pour chaque couche conv
            kernel_sizes: Liste des tailles de kernel pour chaque couche
            pool_sizes: Liste des tailles de pooling
            fc_dims: Dimensions des couches fully connected
            dropout: Taux de dropout
            use_batch_norm: Utiliser BatchNorm ou non
        """
        super(CNN, self).__init__()

        self.spectrum_length = spectrum_length
        self.auxiliary_dim = auxiliary_dim
        self.num_classes = num_classes
        self.input_channels = input_channels
        self.use_batch_norm = use_batch_norm

        self.conv_layers = nn.ModuleList()
        in_channels = input_channels 

        for i, (out_channels, kernel_size, pool_size) in enumerate(
            zip(conv_channels, kernel_sizes, pool_sizes)
        ):
            layers = []

            # Convolution
            layers.append(nn.Conv1d(
                in_channels, 
                out_channels, 
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                bias=not use_batch_norm
            ))

            # Batch Normalization
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(out_channels))

            # Activation
            layers.append(nn.ReLU(inplace=True))

            # Pooling
            layers.append(nn.MaxPool1d(kernel_size=pool_size))

            # Dropout
            if dropout > 0:
                layers.append(nn.Dropout(dropout))

            self.conv_layers.append(nn.Sequential(*layers))
            in_channels = out_channels

        # Calculer la dimension après les convolutions
        self.conv_output_dim = self._calculate_conv_output_dim(
            spectrum_length, conv_channels, pool_sizes
        )

        # Global Average Pooling (optionnel, pour réduire encore)
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        self.auxiliary_mlp = nn.Sequential(
            nn.Linear(auxiliary_dim, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(32, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )

        fusion_dim = conv_channels[-1] + 64

        # Couches fully connected après fusion
        fc_layers = []
        in_dim = fusion_dim

        for fc_dim in fc_dims:
            fc_layers.extend([
                nn.Linear(in_dim, fc_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout)
            ])
            in_dim = fc_dim

        self.fc_layers = nn.Sequential(*fc_layers)

        # Couche de sortie (pas de sigmoid, sera appliqué dans la loss/inference)
        self.output_layer = nn.Linear(in_dim, num_classes)

        # Initialisation des poids
        self._initialize_weights()

    def _calculate_conv_output_dim(
        self, 
        input_length: int, 
        channels: list, 
        pool_sizes: list
    ) -> int:
        """Calcule la dimension de sortie après toutes les convolutions."""
        length = input_length
        for pool_size in pool_sizes:
            length = length // pool_size
        return channels[-1] * length

    def _initialize_weights(self):
        """Initialisation He pour ReLU."""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)

    def forward(self, spectrum: torch.Tensor, auxiliary: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            spectrum: (batch_size, input_channels, spectrum_length)
                     - input_channels=1: (B, 1, L) si moyenne seule
                     - input_channels=3: (B, 3, L) si moyenne + incertitudes
            auxiliary: (batch_size, auxiliary_dim)

        Returns:
            logits: (batch_size, num_classes) - logits bruts (pas de sigmoid)
        """
        x_spectrum = spectrum

        # Passer à travers les couches convolutionnelles
        for conv_layer in self.conv_layers:
            x_spectrum = conv_layer(x_spectrum)

        x_spectrum = self.global_pool(x_spectrum).squeeze(-1)
        x_auxiliary = self.auxiliary_mlp(auxiliary)
        x_fused = torch.cat([x_spectrum, x_auxiliary], dim=1)
        x = self.fc_layers(x_fused)
        logits = self.output_layer(x)

        return logits


class ResidualBlock1D(nn.Module):
    """Bloc résiduel pour CNN 1D (optionnel, pour architectures plus profondes)."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, 
                 stride: int = 1, dropout: float = 0.3):
        super(ResidualBlock1D, self).__init__()

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, 
                               stride=stride, padding=kernel_size // 2, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                               padding=kernel_size // 2, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)

        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, 
                         stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += self.shortcut(identity)
        out = self.relu(out)

        return out
