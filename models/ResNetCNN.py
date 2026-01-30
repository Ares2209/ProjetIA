"""Architecture ResNet 1D pour la classification de spectres d'exoplanètes."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


class ResidualBlock1D(nn.Module):
    """Bloc résiduel pour CNN 1D avec skip connection."""

    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        kernel_size: int = 3, 
        stride: int = 1, 
        dropout: float = 0.3,
        use_batch_norm: bool = True
    ):
        super(ResidualBlock1D, self).__init__()

        self.use_batch_norm = use_batch_norm

        # Première couche de convolution
        self.conv1 = nn.Conv1d(
            in_channels, 
            out_channels, 
            kernel_size, 
            stride=stride, 
            padding=kernel_size // 2, 
            bias=not use_batch_norm
        )
        if use_batch_norm:
            self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)

        # Deuxième couche de convolution
        self.conv2 = nn.Conv1d(
            out_channels, 
            out_channels, 
            kernel_size,
            padding=kernel_size // 2, 
            bias=not use_batch_norm
        )
        if use_batch_norm:
            self.bn2 = nn.BatchNorm1d(out_channels)

        # Skip connection (projection si dimensions différentes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            layers = [
                nn.Conv1d(
                    in_channels, 
                    out_channels, 
                    kernel_size=1, 
                    stride=stride, 
                    bias=not use_batch_norm
                )
            ]
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(out_channels))
            self.shortcut = nn.Sequential(*layers)

    def forward(self, x):
        identity = self.shortcut(x)

        out = self.conv1(x)
        if self.use_batch_norm:
            out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.conv2(out)
        if self.use_batch_norm:
            out = self.bn2(out)

        # Addition de la skip connection
        out += identity
        out = self.relu(out)

        return out


class BottleneckBlock1D(nn.Module):
    """Bloc bottleneck (1x1 -> 3x3 -> 1x1) pour ResNet plus profond."""

    expansion = 4

    def __init__(
        self, 
        in_channels: int, 
        bottleneck_channels: int, 
        stride: int = 1, 
        dropout: float = 0.3,
        use_batch_norm: bool = True
    ):
        super(BottleneckBlock1D, self).__init__()

        self.use_batch_norm = use_batch_norm
        out_channels = bottleneck_channels * self.expansion

        # 1x1 convolution (réduction)
        self.conv1 = nn.Conv1d(
            in_channels, 
            bottleneck_channels, 
            kernel_size=1, 
            bias=not use_batch_norm
        )
        if use_batch_norm:
            self.bn1 = nn.BatchNorm1d(bottleneck_channels)

        # 3x3 convolution
        self.conv2 = nn.Conv1d(
            bottleneck_channels, 
            bottleneck_channels, 
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=not use_batch_norm
        )
        if use_batch_norm:
            self.bn2 = nn.BatchNorm1d(bottleneck_channels)

        # 1x1 convolution (expansion)
        self.conv3 = nn.Conv1d(
            bottleneck_channels, 
            out_channels, 
            kernel_size=1, 
            bias=not use_batch_norm
        )
        if use_batch_norm:
            self.bn3 = nn.BatchNorm1d(out_channels)

        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)

        # Skip connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            layers = [
                nn.Conv1d(
                    in_channels, 
                    out_channels, 
                    kernel_size=1, 
                    stride=stride, 
                    bias=not use_batch_norm
                )
            ]
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(out_channels))
            self.shortcut = nn.Sequential(*layers)

    def forward(self, x):
        identity = self.shortcut(x)

        out = self.conv1(x)
        if self.use_batch_norm:
            out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        if self.use_batch_norm:
            out = self.bn2(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.conv3(out)
        if self.use_batch_norm:
            out = self.bn3(out)

        out += identity
        out = self.relu(out)

        return out


class ResNet1D(nn.Module):
    """
    ResNet 1D pour l'analyse de spectres d'exoplanètes avec données auxiliaires.

    Architecture inspirée de ResNet-18/34/50 adaptée aux signaux 1D.
    """

    def __init__(
        self,
        spectrum_length: int,
        auxiliary_dim: int = 5,
        num_classes: int = 2,
        input_channels: int = 1,
        block_type: str = 'basic',  # 'basic' ou 'bottleneck'
        num_blocks: List[int] = [2, 2, 2, 2],  # Nombre de blocs par stage
        base_channels: int = 64,
        dropout: float = 0.3,
        use_batch_norm: bool = True
    ):
        """
        Args:
            spectrum_length: Longueur du spectre d'entrée
            auxiliary_dim: Dimension des données auxiliaires
            num_classes: Nombre de classes de sortie
            input_channels: Nombre de canaux d'entrée (1 ou 3)
            block_type: Type de bloc ('basic' ou 'bottleneck')
            num_blocks: Liste du nombre de blocs par stage [stage1, stage2, stage3, stage4]
                       Par défaut [2,2,2,2] -> ResNet-18 style
            base_channels: Nombre de canaux de base (64 pour ResNet standard)
            dropout: Taux de dropout
            use_batch_norm: Utiliser BatchNorm
        """
        super(ResNet1D, self).__init__()

        self.spectrum_length = spectrum_length
        self.auxiliary_dim = auxiliary_dim
        self.num_classes = num_classes
        self.input_channels = input_channels
        self.use_batch_norm = use_batch_norm
        self.dropout = dropout

        # Sélection du type de bloc
        if block_type == 'basic':
            self.block = ResidualBlock1D
            self.expansion = 1
        elif block_type == 'bottleneck':
            self.block = BottleneckBlock1D
            self.expansion = BottleneckBlock1D.expansion
        else:
            raise ValueError(f"block_type doit être 'basic' ou 'bottleneck', reçu: {block_type}")

        # Couche initiale (stem)
        stem_layers = [
            nn.Conv1d(
                input_channels, 
                base_channels, 
                kernel_size=7, 
                stride=2, 
                padding=3, 
                bias=not use_batch_norm
            )
        ]
        if use_batch_norm:
            stem_layers.append(nn.BatchNorm1d(base_channels))
        stem_layers.extend([
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        ])
        self.stem = nn.Sequential(*stem_layers)

        # Stages de blocs résiduels
        self.in_channels = base_channels
        self.stage1 = self._make_stage(base_channels, num_blocks[0], stride=1)
        self.stage2 = self._make_stage(base_channels * 2, num_blocks[1], stride=2)
        self.stage3 = self._make_stage(base_channels * 4, num_blocks[2], stride=2)
        self.stage4 = self._make_stage(base_channels * 8, num_blocks[3], stride=2)

        # Global Average Pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        # MLP pour données auxiliaires
        self.auxiliary_mlp = nn.Sequential(
            nn.Linear(auxiliary_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )

        # Dimension après fusion
        spectrum_features = base_channels * 8 * self.expansion
        fusion_dim = spectrum_features + 128

        # Couches fully connected après fusion
        self.fc = nn.Sequential(
            nn.Linear(fusion_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )

        # Couche de sortie
        self.output_layer = nn.Linear(256, num_classes)

        # Initialisation des poids
        self._initialize_weights()

    def _make_stage(self, out_channels: int, num_blocks: int, stride: int):
        """Crée un stage avec plusieurs blocs résiduels."""
        layers = []

        # Premier bloc avec potentiel downsampling
        if self.block == ResidualBlock1D:
            layers.append(
                ResidualBlock1D(
                    self.in_channels, 
                    out_channels, 
                    stride=stride,
                    dropout=self.dropout,
                    use_batch_norm=self.use_batch_norm
                )
            )
        else:  # BottleneckBlock1D
            layers.append(
                BottleneckBlock1D(
                    self.in_channels, 
                    out_channels, 
                    stride=stride,
                    dropout=self.dropout,
                    use_batch_norm=self.use_batch_norm
                )
            )

        self.in_channels = out_channels * self.expansion

        # Blocs suivants
        for _ in range(1, num_blocks):
            if self.block == ResidualBlock1D:
                layers.append(
                    ResidualBlock1D(
                        self.in_channels, 
                        out_channels,
                        dropout=self.dropout,
                        use_batch_norm=self.use_batch_norm
                    )
                )
            else:  # BottleneckBlock1D
                layers.append(
                    BottleneckBlock1D(
                        self.in_channels, 
                        out_channels,
                        dropout=self.dropout,
                        use_batch_norm=self.use_batch_norm
                    )
                )

        return nn.Sequential(*layers)

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
            auxiliary: (batch_size, auxiliary_dim)

        Returns:
            logits: (batch_size, num_classes)
        """
        # Traitement du spectre
        x = self.stem(spectrum)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.global_pool(x).squeeze(-1)

        # Traitement des données auxiliaires
        aux = self.auxiliary_mlp(auxiliary)

        # Fusion
        x_fused = torch.cat([x, aux], dim=1)

        # Classification
        x = self.fc(x_fused)
        logits = self.output_layer(x)

        return logits


# Fonctions helper pour créer des variantes standard
def resnet18_1d(spectrum_length: int, auxiliary_dim: int = 5, num_classes: int = 2, 
                input_channels: int = 1, dropout: float = 0.3):
    """ResNet-18 1D (8 blocs résiduels)."""
    return ResNet1D(
        spectrum_length=spectrum_length,
        auxiliary_dim=auxiliary_dim,
        num_classes=num_classes,
        input_channels=input_channels,
        block_type='basic',
        num_blocks=[2, 2, 2, 2],
        base_channels=64,
        dropout=dropout
    )


def resnet34_1d(spectrum_length: int, auxiliary_dim: int = 5, num_classes: int = 2,
                input_channels: int = 1, dropout: float = 0.3):
    """ResNet-34 1D (16 blocs résiduels)."""
    return ResNet1D(
        spectrum_length=spectrum_length,
        auxiliary_dim=auxiliary_dim,
        num_classes=num_classes,
        input_channels=input_channels,
        block_type='basic',
        num_blocks=[3, 4, 6, 3],
        base_channels=64,
        dropout=dropout
    )


def resnet50_1d(spectrum_length: int, auxiliary_dim: int = 5, num_classes: int = 2,
                input_channels: int = 1, dropout: float = 0.3):
    """ResNet-50 1D avec blocs bottleneck."""
    return ResNet1D(
        spectrum_length=spectrum_length,
        auxiliary_dim=auxiliary_dim,
        num_classes=num_classes,
        input_channels=input_channels,
        block_type='bottleneck',
        num_blocks=[3, 4, 6, 3],
        base_channels=64,
        dropout=dropout
    )


# Exemple d'utilisation
if __name__ == "__main__":
    # Test du modèle
    batch_size = 16
    spectrum_length = 283
    auxiliary_dim = 5
    input_channels = 3  # moyenne + 2 incertitudes
    
    # Créer le modèle
    model = resnet18_1d(
        spectrum_length=spectrum_length,
        auxiliary_dim=auxiliary_dim,
        num_classes=2,
        input_channels=input_channels,
        dropout=0.3
    )
    
    # Données de test
    spectrum = torch.randn(batch_size, input_channels, spectrum_length)
    auxiliary = torch.randn(batch_size, auxiliary_dim)
    
    # Forward pass
    logits = model(spectrum, auxiliary)
    
    print(f"Input spectrum shape: {spectrum.shape}")
    print(f"Input auxiliary shape: {auxiliary.shape}")
    print(f"Output logits shape: {logits.shape}")
    print(f"\nNombre de paramètres: {sum(p.numel() for p in model.parameters()):,}")
