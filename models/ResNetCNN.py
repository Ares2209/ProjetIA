"""
ResNet 1D pour analyse spectrale binaire.
Adapté pour ~2400 échantillons avec features auxiliaires.
"""

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Bloc résiduel de base (BasicBlock)
# ---------------------------------------------------------------------------

class BasicBlock1D(nn.Module):
    """
    Bloc résiduel standard : deux couches Conv1D + BN + ReLU.
    Le shortcut est une projection 1x1 si les dimensions changent.
    """
    expansion = 1  # les blocs basic ne changent pas la largeur des canaux

    def __init__(self, in_channels: int, out_channels: int,
                 stride: int = 1, dropout: float = 0.0):
        super().__init__()

        self.conv1 = nn.Conv1d(in_channels, out_channels,
                               kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn1   = nn.BatchNorm1d(out_channels)
        self.relu  = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv1d(out_channels, out_channels,
                               kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn2   = nn.BatchNorm1d(out_channels)

        self.dropout = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()

        # Shortcut : projection uniquement si stride > 1 ou canaux différents
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))

        out = self.relu(out + identity)
        return out


# ---------------------------------------------------------------------------
# ResNet1D
# ---------------------------------------------------------------------------

class ResNet1D(nn.Module):
    """
    ResNet 1D pour classification spectrale avec features auxiliaires.

    Architecture :
        Stem  →  Couche1 → Couche2 → Couche3 → Couche4
              →  GlobalAvgPool  →  Concat(auxiliary)  →  Classifier

    Args:
        spectrum_length : longueur du spectre en entrée (ex. 283)
        input_channels  : nombre de canaux d'entrée (ex. 3 : mean + 2 incertitudes)
        auxiliary_dim   : dimension du vecteur auxiliaire (ex. 5)
        num_classes     : nombre de classes (2 pour binaire)
        num_blocks      : liste de 4 entiers → nombre de blocs par couche
        base_channels   : largeur de base (32 conseillé pour ~2400 samples)
        dropout         : taux de dropout dans les blocs et le classifier
    """

    def __init__(
        self,
        spectrum_length: int,
        input_channels:  int   = 1,
        auxiliary_dim:   int   = 5,
        num_classes:     int   = 2,
        num_blocks:      list  = (2, 2, 2, 2),   # ResNet-18 par défaut
        base_channels:   int   = 32,
        dropout:         float = 0.3,
    ):
        super().__init__()

        self.in_channels = base_channels  # état interne pour _make_layer

        # --- Stem : grande conv initiale pour capter les patterns larges ---
        self.stem = nn.Sequential(
            nn.Conv1d(input_channels, base_channels,
                      kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(base_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
        )

        # --- 4 couches résiduelles ---
        # Les canaux doublent à chaque couche ; stride=2 sous-échantillonne
        c = base_channels
        self.layer1 = self._make_layer(c,     num_blocks[0], stride=1, dropout=dropout)
        self.layer2 = self._make_layer(c * 2, num_blocks[1], stride=2, dropout=dropout)
        self.layer3 = self._make_layer(c * 4, num_blocks[2], stride=2, dropout=dropout)
        self.layer4 = self._make_layer(c * 8, num_blocks[3], stride=2, dropout=dropout)

        # --- Tête de classification ---
        feature_dim = c * 8 + auxiliary_dim   # après concat avec auxiliaire

        self.global_pool = nn.AdaptiveAvgPool1d(1)   # → (B, C, 1)
        self.classifier   = nn.Sequential(
            nn.Flatten(),                            # → (B, C)
            nn.Linear(feature_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(128, num_classes),
        )

        # --- Initialisation des poids ---
        self._init_weights()

    # -----------------------------------------------------------------------

    def _make_layer(self, out_channels: int, num_blocks: int,
                    stride: int, dropout: float) -> nn.Sequential:
        """Construit une couche de `num_blocks` BasicBlock1D."""
        layers = [BasicBlock1D(self.in_channels, out_channels,
                               stride=stride, dropout=dropout)]
        self.in_channels = out_channels
        for _ in range(1, num_blocks):
            layers.append(BasicBlock1D(out_channels, out_channels,
                                       stride=1, dropout=dropout))
        return nn.Sequential(*layers)

    def _init_weights(self):
        """He/Kaiming pour Conv, constant pour BN, Xavier pour Linear."""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias,   0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    # -----------------------------------------------------------------------

    def forward(self, spectrum: torch.Tensor,
                auxiliary: torch.Tensor) -> torch.Tensor:
        """
        Args:
            spectrum  : (B, input_channels, spectrum_length)
            auxiliary : (B, auxiliary_dim)
        Returns:
            logits    : (B, num_classes)
        """
        x = self.stem(spectrum)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.global_pool(x)   # (B, C, 1)
        x = x.squeeze(-1)         # (B, C)

        # Fusion des features auxiliaires après le pooling global
        x = torch.cat([x, auxiliary], dim=1)   # (B, C + auxiliary_dim)

        return self.classifier(x)


def resnet8_1d(spectrum_length: int, auxiliary_dim: int = 5,
                num_classes: int = 2, input_channels: int = 1,
                base_channels: int = 32, dropout: float = 0.3) -> ResNet1D:
    return ResNet1D(
        spectrum_length=spectrum_length,
        input_channels=input_channels,
        auxiliary_dim=auxiliary_dim,
        num_classes=num_classes,
        num_blocks=[1, 1, 1, 1],
        base_channels=base_channels,
        dropout=dropout,
    )


def resnet18_1d(spectrum_length: int, auxiliary_dim: int = 5,
                num_classes: int = 2, input_channels: int = 1,
                base_channels: int = 32, dropout: float = 0.3) -> ResNet1D:
    return ResNet1D(
        spectrum_length=spectrum_length,
        input_channels=input_channels,
        auxiliary_dim=auxiliary_dim,
        num_classes=num_classes,
        num_blocks=[2, 2, 2, 2],
        base_channels=base_channels,
        dropout=dropout,
    )


def resnet34_1d(spectrum_length: int, auxiliary_dim: int = 5,
                num_classes: int = 2, input_channels: int = 1,
                base_channels: int = 32, dropout: float = 0.3) -> ResNet1D:
    """ResNet-34 1D – 16 blocs résiduels."""
    return ResNet1D(
        spectrum_length=spectrum_length,
        input_channels=input_channels,
        auxiliary_dim=auxiliary_dim,
        num_classes=num_classes,
        num_blocks=[3, 4, 6, 3],
        base_channels=base_channels,
        dropout=dropout,
    )


