"""Module de configuration adapté pour CNN spectral."""

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, Dict, Any, List
import json
import torch

@dataclass
class ModelConfig:
    """Configuration du modèle CNN."""
    # Architecture choice: 'simple' or 'resnet'
    architecture: str = 'simple'
    
    # Pour SpectralCNN simple
    conv_channels: List[int] = field(default_factory=lambda: [32, 64, 128, 256])
    kernel_sizes: List[int] = field(default_factory=lambda: [7, 5, 3, 3])
    pool_sizes: List[int] = field(default_factory=lambda: [2, 2, 2, 2])
    fc_dims: List[int] = field(default_factory=lambda: [256, 128])
    
    # Pour ResNetSpectralCNN
    d_model: int = 64  # initial_channels pour ResNet
    num_layers: int = 2  # num_blocks par layer
    
    # Commun
    dropout: float = 0.3
    use_batch_norm: bool = True
    auxiliary_dim: int = 5
    
    def __post_init__(self):
        """Validation."""
        if self.architecture not in ['simple', 'resnet']:
            raise ValueError("architecture must be 'simple' or 'resnet'")
        if self.dropout < 0 or self.dropout > 1:
            raise ValueError("dropout must be between 0 and 1")

@dataclass
class TrainingConfig:
    """Configuration de l'entraînement."""
    num_classes: int = 2  # eau, nuages
    batch_size: int = 32
    num_epochs: int = 50
    learning_rate: float = 1e-3
    patience: int = 10
    preload: Optional[str] = None
    
    weight_decay: float = 1e-4
    optimizer_betas: tuple = (0.9, 0.999)
    optimizer_eps: float = 1e-8
    max_grad_norm: float = 1.0
    
    # Scheduler
    scheduler_type: str = 'cosine'  # 'cosine', 'step', 'plateau'
    scheduler_pct_start: float = 0.1
    min_lr: float = 1e-6
    
    # Early stopping
    min_delta: float = 0.001
    
    # Checkpointing
    save_every_n_epochs: int = 5
    keep_last_n_checkpoints: int = 3
    
    # Logging
    track_iteration_loss: bool = True
    log_every_n_steps: int = 10
    
    # Loss
    loss_type: str = 'bce'  # 'bce' or 'focal'
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0
    pos_weight: Optional[List[float]] = None  # [weight_eau, weight_nuages]
    
    classification_threshold: float = 0.5
    
    # Multi-GPU
    use_multi_gpu: bool = False
    gpu_ids: Optional[List[int]] = None
    distributed_backend: str = 'dp'
    
    def __post_init__(self):
        """Validation."""
        if self.batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        if self.num_epochs <= 0:
            raise ValueError("num_epochs must be > 0")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be > 0")

@dataclass
class DataConfig:
    """Configuration des données exoplanètes."""
    # Chemins des fichiers
    spectra_train_path: str = 'spectra.npy'
    auxiliary_train_path: str = 'auxiliary.csv'
    targets_train_path: str = 'targets_train.csv'
    
    spectra_test_path: str = 'spectra_test.npy'
    auxiliary_test_path: str = 'auxiliary_test.csv'
    
    # Split validation
    train_ratio: float = 0.8
    val_ratio: float = 0.2
    
    # DataLoader
    num_workers: int = 4
    pin_memory: bool = True
    
    # Normalisation
    normalize_spectra: str = 'minmax'  # 'minmax', 'zscore', or 'none'
    normalize_auxiliary: str = 'zscore'
    
    # Augmentation (optionnel)
    use_augmentation: bool = False
    noise_std: float = 0.01
    
    # Seed
    random_seed: int = 42
    
    def __post_init__(self):
        """Validation."""
        if self.train_ratio + self.val_ratio > 1.0:
            raise ValueError("train_ratio + val_ratio cannot exceed 1.0")

@dataclass
class PathsConfig:
    """Configuration des chemins."""
    model_folder: str = 'checkpoints'
    model_basename: str = 'exoplanet_model'
    experiment_name: str = 'runs/exoplanet_cnn_v1'
    
    def __post_init__(self):
        """Crée les dossiers si nécessaire."""
        Path(self.model_folder).mkdir(parents=True, exist_ok=True)
        Path(self.experiment_name).mkdir(parents=True, exist_ok=True)

@dataclass
class Config:
    """Configuration principale."""
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    results_folder: str = 'results'
    
    device: str = field(default_factory=lambda: 'cuda' if torch.cuda.is_available() else 'cpu')
    num_gpus: int = 0
    
    def __post_init__(self):
        """Validation globale et configuration device."""
        Path(self.results_folder).mkdir(parents=True, exist_ok=True)
        
        # Configuration multi-GPU
        if torch.cuda.is_available():
            self.device = 'cuda'
            self.num_gpus = torch.cuda.device_count()
            
            if self.num_gpus > 1 and self.training.use_multi_gpu:
                if self.training.gpu_ids is None:
                    self.training.gpu_ids = list(range(self.num_gpus))
                print(f" Multi-GPU: {self.num_gpus} GPUs")
                for i in self.training.gpu_ids:
                    print(f"   - GPU {i}: {torch.cuda.get_device_name(i)}")
            else:
                self.num_gpus = 1
                print(f" Single GPU: {torch.cuda.get_device_name(0)}")
        else:
            self.device = 'cpu'
            self.num_gpus = 0
            print(" CPU mode")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertit en dictionnaire."""
        result = {
            'model': asdict(self.model),
            'training': asdict(self.training),
            'data': asdict(self.data),
            'paths': asdict(self.paths),
            'results_folder': self.results_folder,
            'device': self.device
        }
        return result
    
    def save(self, path: str):
        """Sauvegarde en JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        print(f"Configuration sauvegardée: {path}")
    
    @classmethod
    def load(cls, path: str) -> 'Config':
        """Charge depuis JSON."""
        with open(path, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'Config':
        """Crée depuis un dictionnaire."""
        return cls(
            model=ModelConfig(**config_dict.get('model', {})),
            training=TrainingConfig(**config_dict.get('training', {})),
            data=DataConfig(**config_dict.get('data', {})),
            paths=PathsConfig(**config_dict.get('paths', {})),
            results_folder=config_dict.get('results_folder', 'results')
        )
    
    def print_summary(self):
        """Affiche un résumé."""
        print("\n" + "="*80)
        print("CONFIGURATION - EXOPLANET CLASSIFICATION")
        print("="*80)
        
        print("\n  MODÈLE:")
        print(f"  • Architecture: {self.model.architecture.upper()}")
        if self.model.architecture == 'simple':
            print(f"  • Conv channels: {self.model.conv_channels}")
            print(f"  • FC dims: {self.model.fc_dims}")
        else:
            print(f"  • Initial channels: {self.model.d_model}")
            print(f"  • Blocks per layer: {self.model.num_layers}")
        print(f"  • Dropout: {self.model.dropout}")
        
        print("\nENTRAÎNEMENT:")
        print(f"  • Epochs: {self.training.num_epochs}")
        print(f"  • Batch size: {self.training.batch_size}")
        print(f"  • Learning rate: {self.training.learning_rate:.2e}")
        print(f"  • Loss: {self.training.loss_type.upper()}")
        print(f"  • Device: {self.device} ({self.num_gpus} GPU(s))")
        
        print("\nDONNÉES:")
        print(f"  • Spectra train: {self.data.spectra_train_path}")
        print(f"  • Split: {self.data.train_ratio:.0%} train / {self.data.val_ratio:.0%} val")
        print(f"  • Workers: {self.data.num_workers}")
        
        print("\n" + "="*80 + "\n")


def get_config_object() -> Config:
    """Retourne une configuration par défaut."""
    return Config()


# Fonctions utilitaires (compatibilité)
def get_weights_file_path(config, epoch: str) -> str:
    """Construit le chemin vers un fichier de poids."""
    if isinstance(config, Config):
        model_folder = config.paths.model_folder
        model_basename = config.paths.model_basename
    else:
        model_folder = config['paths']['model_folder']
        model_basename = config['paths']['model_basename']
    
    model_filename = f"{model_basename}_{epoch}.pth"
    return str(Path(model_folder) / model_filename)


def latest_weights_file_path(config) -> Optional[str]:
    """Retourne le dernier fichier de poids."""
    import glob
    import os
    
    if isinstance(config, Config):
        model_folder = config.paths.model_folder
        model_basename = config.paths.model_basename
    else:
        model_folder = config['paths']['model_folder']
        model_basename = config['paths']['model_basename']
    
    if not Path(model_folder).exists():
        return None
    
    pattern = os.path.join(model_folder, f"{model_basename}_*.pth")
    weights_files = sorted(glob.glob(pattern), key=os.path.getmtime)
    
    return weights_files[-1] if weights_files else None
