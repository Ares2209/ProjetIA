"""
ResNet 1D pour analyse spectrale binaire.
Adapté pour ~2400 échantillons avec features auxiliaires.
"""

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Squeeze-and-Excitation block 1D
# ---------------------------------------------------------------------------

class SEBlock1D(nn.Module):
    """
    Squeeze-and-Excitation block adapté aux signaux 1D.

    Recalibre chaque canal en apprenant un vecteur d'attention (B, C, 1)
    à partir d'un average-pooling global suivi d'un mini MLP.

    Args:
        channels  : nombre de canaux d'entrée
        reduction : facteur de compression du bottleneck (ex. 4 → C//4 neurones)
    """

    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        hidden = max(1, channels // reduction)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),   # (B, C, L) → (B, C, 1)
            nn.Flatten(),              # (B, C)
            nn.Linear(channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.se(x).unsqueeze(-1)   # (B, C, 1)
        return x * w                   # recalibrage canal par canal


# ---------------------------------------------------------------------------
# Bloc résiduel de base (BasicBlock) avec SE optionnel
# ---------------------------------------------------------------------------

class BasicBlock1D(nn.Module):
    """
    Bloc résiduel standard : deux couches Conv1D + BN + ReLU.
    Le shortcut est une projection 1x1 si les dimensions changent.
    Option SE : recalibre les canaux après la 2e conv, avant l'addition residuelle.
    """
    expansion = 1  # les blocs basic ne changent pas la largeur des canaux

    def __init__(self, in_channels: int, out_channels: int,
                 stride: int = 1, dropout: float = 0.0, use_se: bool = True):
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

        # SE block : après la 2e conv, avant l'addition residuelle
        self.se = SEBlock1D(out_channels) if use_se else nn.Identity()

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
        out = self.se(out)              # recalibrage SE avant l'addition

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
              →  GlobalAvgPool  →  Concat(auxiliary)
              →  Trunk partagé (FC 128)
              →  head_eau  (FC 128→1)
              →  head_nuage (FC 128→1)
              →  Concat → logits (B, 2)

    Args:
        spectrum_length : longueur du spectre en entrée (ex. 52)
        input_channels  : nombre de canaux d'entrée (ex. 5 : mean + unc + SNR + rel_unc)
        auxiliary_dim   : dimension du vecteur auxiliaire
        num_classes     : nombre de classes (2 : eau + nuages)
        num_blocks      : liste de 4 entiers → nombre de blocs par couche
        base_channels   : largeur de base (32 conseillé pour ~2400 samples)
        dropout         : taux de dropout dans les blocs et les têtes
        use_se          : active les Squeeze-and-Excitation blocks dans chaque BasicBlock
    """

    def __init__(
        self,
        spectrum_length: int,
        input_channels:  int   = 1,
        auxiliary_dim:   int   = 5,
        num_classes:     int   = 2,
        num_blocks:      list  = (2, 2, 2, 2),
        base_channels:   int   = 32,
        dropout:         float = 0.3,
        use_se:          bool  = True,
    ):
        super().__init__()

        self.in_channels = base_channels  # état interne pour _make_layer
        self.use_se      = use_se

        # --- Stem : conv initiale adaptée aux spectres courts (52 points) ---
        self.stem = nn.Sequential(
            nn.Conv1d(input_channels, base_channels,
                      kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm1d(base_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
        )

        # --- 4 couches résiduelles ---
        c = base_channels
        self.layer1 = self._make_layer(c,     num_blocks[0], stride=1, dropout=dropout)
        self.layer2 = self._make_layer(c * 2, num_blocks[1], stride=2, dropout=dropout)
        self.layer3 = self._make_layer(c * 4, num_blocks[2], stride=2, dropout=dropout)
        self.layer4 = self._make_layer(c * 8, num_blocks[3], stride=2, dropout=dropout)

        # --- Trunk partagé + têtes séparées par label ---
        feature_dim = c * 8 + auxiliary_dim   # après concat avec auxiliaire

        self.global_pool = nn.AdaptiveAvgPool1d(1)   # → (B, C, 1)

        # Trunk commun : extrait une représentation de 128 dims partagée
        self.trunk = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feature_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
        )

        # Tête dédiée à la détection de l'eau
        self.head_eau = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
        )

        # Tête dédiée à la détection des nuages
        self.head_nuage = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
        )

        # --- Initialisation des poids ---
        self._init_weights()

    # -----------------------------------------------------------------------

    def _make_layer(self, out_channels: int, num_blocks: int,
                    stride: int, dropout: float) -> nn.Sequential:
        """Construit une couche de `num_blocks` BasicBlock1D (avec SE si use_se)."""
        layers = [BasicBlock1D(self.in_channels, out_channels,
                               stride=stride, dropout=dropout, use_se=self.use_se)]
        self.in_channels = out_channels
        for _ in range(1, num_blocks):
            layers.append(BasicBlock1D(out_channels, out_channels,
                                       stride=1, dropout=dropout, use_se=self.use_se))
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
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    # -----------------------------------------------------------------------

    def forward(self, spectrum: torch.Tensor,
                auxiliary: torch.Tensor) -> torch.Tensor:
        """
        Args:
            spectrum  : (B, input_channels, spectrum_length)
            auxiliary : (B, auxiliary_dim)
        Returns:
            logits    : (B, 2)  — [logit_eau, logit_nuage]
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

        # Trunk partagé → représentation commune 128 dims
        shared = self.trunk(x)

        # Têtes indépendantes par label
        logit_eau   = self.head_eau(shared)    # (B, 1)
        logit_nuage = self.head_nuage(shared)  # (B, 1)

        return torch.cat([logit_eau, logit_nuage], dim=1)   # (B, 2)


def resnet8_test(spectrum_length: int, auxiliary_dim: int = 8,
                num_classes: int = 2, input_channels: int = 1,
                base_channels: int = 32, dropout: float = 0.3,
                use_se: bool = True) -> ResNet1D:
    return ResNet1D(
        spectrum_length=spectrum_length,
        input_channels=input_channels,
        auxiliary_dim=auxiliary_dim,
        num_classes=num_classes,
        num_blocks=[1, 1, 1, 1],
        base_channels=base_channels,
        dropout=dropout,
        use_se=use_se,
    )


def resnet8_1d(spectrum_length: int, auxiliary_dim: int = 17,
                num_classes: int = 2, input_channels: int = 1,
                base_channels: int = 32, dropout: float = 0.3,
                use_se: bool = True) -> ResNet1D:
    return ResNet1D(
        spectrum_length=spectrum_length,
        input_channels=input_channels,
        auxiliary_dim=auxiliary_dim,
        num_classes=num_classes,
        num_blocks=[1, 1, 1, 1],
        base_channels=base_channels,
        dropout=dropout,
        use_se=use_se,
    )


def resnet18_1d(spectrum_length: int, auxiliary_dim: int = 5,
                num_classes: int = 2, input_channels: int = 1,
                base_channels: int = 32, dropout: float = 0.3,
                use_se: bool = True) -> ResNet1D:
    return ResNet1D(
        spectrum_length=spectrum_length,
        input_channels=input_channels,
        auxiliary_dim=auxiliary_dim,
        num_classes=num_classes,
        num_blocks=[2, 2, 2, 2],
        base_channels=base_channels,
        dropout=dropout,
        use_se=use_se,
    )


def resnet34_1d(spectrum_length: int, auxiliary_dim: int = 5,
                num_classes: int = 2, input_channels: int = 1,
                base_channels: int = 32, dropout: float = 0.3,
                use_se: bool = True) -> ResNet1D:
    """ResNet-34 1D – 16 blocs résiduels."""
    return ResNet1D(
        spectrum_length=spectrum_length,
        input_channels=input_channels,
        auxiliary_dim=auxiliary_dim,
        num_classes=num_classes,
        num_blocks=[3, 4, 6, 3],
        base_channels=base_channels,
        dropout=dropout,
        use_se=use_se,
    )

class EnsembleResNet(nn.Module):
    """
    Ensemble de deux ResNet1D (ResNet-8 et ResNet-18) dont les logits
    sont fusionnés par une combinaison linéaire apprise.

    Architecture :
        ResNet8  →  logits_8  ──┐
                                 ├→ Linear(2*num_classes → num_classes) → logits
        ResNet18 →  logits_18 ──┘

    Args :
        spectrum_length : longueur du spectre (ex. 52)
        input_channels  : canaux d'entrée (ex. 3)
        auxiliary_dim   : dimension des features auxiliaires
        num_classes     : nombre de classes (2 par défaut)
        base_channels   : largeur de base partagée entre les deux sous-réseaux
        dropout         : taux de dropout commun
    """

    def __init__(
        self,
        spectrum_length: int,
        input_channels:  int   = 3,
        auxiliary_dim:   int   = 5,
        num_classes:     int   = 2,
        base_channels:   int   = 32,
        dropout:         float = 0.3,
    ):
        super().__init__()

        # SE-ResNet8 : attention par canal → diversité architecturale avec ResNet18
        self.resnet8 = resnet8_1d(
            spectrum_length=spectrum_length,
            auxiliary_dim=auxiliary_dim,
            num_classes=num_classes,
            input_channels=input_channels,
            base_channels=base_channels,
            dropout=dropout,
        )
        self.resnet18 = resnet18_1d(
            spectrum_length=spectrum_length,
            auxiliary_dim=auxiliary_dim,
            num_classes=num_classes,
            input_channels=input_channels,
            base_channels=base_channels,
            dropout=dropout,
        )

        # Fusion apprise : concaténation des deux sorties → projection
        self.fusion = nn.Linear(num_classes * 2, num_classes, bias=True)
        nn.init.constant_(self.fusion.weight, 0.5)   # poids initiaux équilibrés
        nn.init.zeros_(self.fusion.bias)

    def forward(self, spectrum: torch.Tensor,
                auxiliary: torch.Tensor) -> torch.Tensor:
        """
        Args :
            spectrum  : (B, input_channels, spectrum_length)
            auxiliary : (B, auxiliary_dim)
        Returns :
            logits    : (B, num_classes)
        """
        out8  = self.resnet8(spectrum, auxiliary)    # (B, num_classes)
        out18 = self.resnet18(spectrum, auxiliary)   # (B, num_classes)
        fused = torch.cat([out8, out18], dim=1)      # (B, 2*num_classes)
        return self.fusion(fused)                    # (B, num_classes)


def ensemble_resnet_1d(
    spectrum_length: int,
    auxiliary_dim:   int   = 5,
    num_classes:     int   = 2,
    input_channels:  int   = 3,
    base_channels:   int   = 32,
    dropout:         float = 0.3,
) -> EnsembleResNet:
    """Construit un EnsembleResNet (ResNet8 + ResNet18) avec fusion apprise."""
    return EnsembleResNet(
        spectrum_length=spectrum_length,
        input_channels=input_channels,
        auxiliary_dim=auxiliary_dim,
        num_classes=num_classes,
        base_channels=base_channels,
        dropout=dropout,
    )
