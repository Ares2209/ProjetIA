import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class ResNetSpectralCNN(nn.Module):
    """
    CNN avec blocs résiduels (plus profond et plus performant).
    """
    
    def __init__(
        self,
        spectrum_length: int,
        auxiliary_dim: int = 5,
        num_classes: int = 2,
        initial_channels: int = 64,
        num_blocks: list = [2, 2, 2, 2],
        dropout: float = 0.3
    ):
        super(ResNetSpectralCNN, self).__init__()
        
        self.spectrum_length = spectrum_length
        
        # Stem: Première convolution
        self.stem = nn.Sequential(
            nn.Conv1d(1, initial_channels, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(initial_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2)
        )
        
        # Blocs résiduels
        self.layer1 = self._make_layer(initial_channels, initial_channels, 
                                       num_blocks[0], dropout=dropout)
        self.layer2 = self._make_layer(initial_channels, initial_channels * 2, 
                                       num_blocks[1], stride=2, dropout=dropout)
        self.layer3 = self._make_layer(initial_channels * 2, initial_channels * 4, 
                                       num_blocks[2], stride=2, dropout=dropout)
        self.layer4 = self._make_layer(initial_channels * 4, initial_channels * 8, 
                                       num_blocks[3], stride=2, dropout=dropout)
        
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
        
        # Fusion et classification
        fusion_dim = initial_channels * 8 + 128
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )
    
    def _make_layer(self, in_channels, out_channels, num_blocks, 
                   stride=1, dropout=0.3):
        layers = []
        layers.append(ResidualBlock1D(in_channels, out_channels, stride=stride, 
                                      dropout=dropout))
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock1D(out_channels, out_channels, 
                                         dropout=dropout))
        return nn.Sequential(*layers)
    
    def forward(self, spectrum, auxiliary):
        # Branche spectre
        x = self.stem(spectrum)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.global_pool(x).squeeze(-1)
        
        # Branche auxiliaire
        aux = self.auxiliary_mlp(auxiliary)
        
        # Fusion
        fused = torch.cat([x, aux], dim=1)
        
        # Classification
        logits = self.classifier(fused)
        
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


