"""Module de configuration avec validation et compatibilité."""

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, Dict, Any, List
import json
import glob
import os
import torch


@dataclass
class ModelConfig:
    """Configuration du modèle."""
    input_dim: int = 8
    d_model: int = 64
    num_layers: int = 4
    dropout: float = 0.1 
    def __post_init__(self):
        """Validation."""
        if self.d_model <= 0:
            raise ValueError("d_model must be > 0")
        if self.num_layers <= 0:
            raise ValueError("num_layers must be > 0")


@dataclass
class TrainingConfig:
    """Configuration de l'entraînement."""
    num_classes: int = 4
    batch_size: int = 4
    num_epochs: int = 10
    learning_rate: float = 5e-5
    patience: int = 5
    preload: Optional[str] = None

    weight_decay: float = 0.01

    optimizer_betas: tuple = (0.9, 0.999)
    optimizer_eps: float = 1e-8
    max_grad_norm: float = 1.0
    scheduler_pct_start: float = 0.9
    scheduler_div_factor: float = 1
    scheduler_final_div_factor: float = 1
    min_delta: float = 0.001
    save_every_n_epochs: int = 5
    keep_last_n_checkpoints: int = 3
    track_iteration_loss: bool = True
    log_every_n_steps: int = 10
    
    focal_alpha: float = 0.35
    focal_gamma: float = 2.0
    
    loss_type: str = 'Random'
    
    classification_threshold: float = 0.5
    # Multi-GPU settings
    use_multi_gpu: bool = True
    gpu_ids: list = field(default_factory=lambda: None)  # None = use all available
    distributed_backend: str = 'dp'  # 'dp' or 'ddp'
    
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
    """Configuration des données."""
    data_dir: str = 'Data'
    train_ratio: float = 0.7
    val_ratio: float = 0.3
    max_files: Optional[int] = 5000
    num_workers: int = 4
    file_loading_seed: int = 42
    
    def __post_init__(self):
        """Validation avec détection automatique."""
        # Essayer plusieurs chemins possibles
        possible_paths = [
            self.data_dir,
            os.path.join('NN', self.data_dir),
        ]
        
        found = False
        for path in possible_paths:
            full_path = Path(path)
            if full_path.exists():
                pattern = str(full_path / "ville_colored_*.txt")
                if glob.glob(pattern):
                    self.data_dir = str(full_path)
                    found = True
                    print(f"[OK] Donnees trouvees dans: {self.data_dir}")
                    break
        
        if not found:
            print(f"[WARN] Aucun fichier trouvé dans '{self.data_dir}'")
            print(f"   Chemins testes: {possible_paths}")


@dataclass
class PathsConfig:
    """Configuration des chemins."""
    model_folder: str = 'NN/checkpoints'
    model_basename: str = 'visibility_model_'
    experiment_name: str = 'NN/runs/visibility_transformer_v1'
    
    def __post_init__(self):
        """Crée les dossiers si nécessaire."""
        Path(self.model_folder).mkdir(parents=True, exist_ok=True)
        Path(self.experiment_name).mkdir(parents=True, exist_ok=True)


@dataclass
class Config:
    """Configuration principale regroupant toutes les sous-configurations."""
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    results_folder: str = 'results'
    
    # Device (calculé automatiquement)
    device: str = field(default_factory=lambda: 'cuda' if torch.cuda.is_available() else 'cpu')
    num_gpus: int = 0
    
    def __post_init__(self):
        """Validation globale et création des dossiers."""
        Path(self.results_folder).mkdir(parents=True, exist_ok=True)
        
        # Validation globale
        errors = self.validate()
        if errors:
            raise ValueError(f"Configuration invalide:\n" + "\n".join(f"  - {e}" for e in errors))

        # Automatic device and multi-GPU configuration
        if torch.cuda.is_available():
            # Prefer CUDA device
            self.device = 'cuda'
            self.num_gpus = torch.cuda.device_count()

            # Configure multi-GPU defaults
            if self.num_gpus > 1 and getattr(self.training, 'use_multi_gpu', False):
                if self.training.gpu_ids is None:
                    self.training.gpu_ids = list(range(self.num_gpus))
                print(f"🎮 Multi-GPU activé: {self.num_gpus} GPU détectés")
                print(f"   GPU utilisés: {self.training.gpu_ids}")
                for i in self.training.gpu_ids:
                    try:
                        print(f"   - GPU {i}: {torch.cuda.get_device_name(i)}")
                    except Exception:
                        pass
            else:
                # Single GPU
                self.num_gpus = 1
                try:
                    print(f"🎮 Single GPU: {torch.cuda.get_device_name(0)}")
                except Exception:
                    pass
        else:
            self.device = 'cpu'
            self.num_gpus = 0
            print("💻 CPU mode")
    
    def validate(self) -> List[str]:
        """Valide toute la configuration et retourne les erreurs."""
        errors = []
        
        # Validation déjà faite par les __post_init__ des sous-configs
        # On peut ajouter des validations croisées ici
        
        return errors
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertit en dictionnaire imbriqué (compatible ancien format)."""
        return {
            'model': asdict(self.model),
            'training': asdict(self.training),
            'data': asdict(self.data),
            'paths': asdict(self.paths),
            'results_folder': self.results_folder,
            'device': self.device
        }
    
    def save(self, path: str):
        """Sauvegarde la configuration en JSON."""
        path = Path(path)
        
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        print(f"✅ Configuration sauvegardée dans '{path}'")

    
    @classmethod
    def load(cls, path: str) -> 'Config':
        """Charge une configuration depuis JSON."""
        with open(path, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'Config':
        """Crée une configuration depuis un dictionnaire."""
        return cls(
            model=ModelConfig(**config_dict.get('model', {})),
            training=TrainingConfig(**config_dict.get('training', {})),
            data=DataConfig(**config_dict.get('data', {})),
            paths=PathsConfig(**config_dict.get('paths', {})),
            results_folder=config_dict.get('results_folder', 'results')
        )
    
    def print_summary(self):
        """Affiche un résumé de la configuration."""
        print("\n" + "="*80)
        print("📋 CONFIGURATION")
        print("="*80)
        
        print("\n🎯 ENTRAÎNEMENT:")
        print(f"  • Epochs: {self.training.num_epochs}")
        print(f"  • Batch size: {self.training.batch_size}")
        print(f"  • Learning rate: {self.training.learning_rate:.2e}")
        print(f"  • Loss: {self.training.loss_type.upper()}", end='')
        if self.training.loss_type == 'focal':
            print(f" (α={self.training.focal_alpha}, γ={self.training.focal_gamma})")
        else:
            print(f" (pos_weight={'ON' if self.training.use_pos_weight else 'OFF'})")
        print(f"  • Early stopping: patience={self.training.patience}, min_delta={self.training.min_delta}")
        print(f"  • Device: {self.device}")
        
        print("\n📊 DONNÉES:")
        print(f"  • Directory: {self.data.data_dir}")
        print(f"  • Split: {self.data.train_ratio:.0%} train / {self.data.val_ratio:.0%} val")
        if self.data.max_files:
            print(f"  • Max files: {self.data.max_files}")
        print(f"  • Workers: {self.data.num_workers}")
        
        print("\n" + "="*80 + "\n")
def get_config() -> Dict[str, Any]:
    """
    Retourne une configuration par défaut au format dictionnaire.

    COMPATIBILITÉ: Cette fonction maintient l'interface de l'ancien config.py
    pour ne pas casser le code existant.
    """
    config = Config()
    return config.to_dict()

def get_config_object() -> Config:
    """
    Retourne une configuration par défaut sous forme d'objet Config.

    NOUVEAU: Utilisez cette fonction pour profiter des avantages des dataclasses.
    """
    return Config()

def get_weights_file_path(config: Dict[str, Any], epoch: str) -> str:
    """
    Construit le chemin vers un fichier de poids (epoch ou 'best').

    Args:
        config: Configuration (dict ou Config object)
        epoch: Numéro d'epoch (ex: "01", "15") ou "best"

    Returns:
        Chemin vers le fichier de poids
    """
    # Support des deux formats (dict et Config object)
    if isinstance(config, Config):
        model_folder = config.paths.model_folder
        model_basename = config.paths.model_basename
    else:
        model_folder = config['paths']['model_folder']
        model_basename = config['paths']['model_basename']

    # Construire le nom du fichier: basename_epoch.pth
    model_filename = f"{model_basename}_{epoch}.pth"
    return str(Path(model_folder) / model_filename)

def latest_weights_file_path(config: Dict[str, Any]) -> Optional[str]:
    """
    Retourne le chemin vers le dernier fichier de poids trouvé.

    Args:
        config: Configuration (dict ou Config object)

    Returns:
        Chemin vers le dernier fichier ou None
    """
    # Support des deux formats
    if isinstance(config, Config):
        model_folder = config.paths.model_folder
        model_basename = config.paths.model_basename
    else:
        model_folder = config['paths']['model_folder']
        model_basename = config['paths']['model_basename']

    if not Path(model_folder).exists():
        return None

    # Pattern: basename_*.pth (ex: visibility_model_*.pth)
    pattern = os.path.join(model_folder, f"{model_basename}_*.pth")
    weights_files = sorted(glob.glob(pattern), key=os.path.getmtime)

    if len(weights_files) == 0:
        return None

    return weights_files[-1]

def get_best_weights_file_path(config: Dict[str, Any]) -> str:
    """Retourne le chemin vers le fichier 'best'."""
    return get_weights_file_path(config, 'best')

def create_directories(config: Dict[str, Any]):
    """
    Crée tous les dossiers nécessaires.

    Args:
        config: Configuration (dict ou Config object)
    """
    if isinstance(config, Config):
        Path(config.paths.model_folder).mkdir(parents=True, exist_ok=True)
        Path(config.results_folder).mkdir(parents=True, exist_ok=True)
        Path(config.paths.experiment_name).mkdir(parents=True, exist_ok=True)
    else:
        Path(config['paths']['model_folder']).mkdir(parents=True, exist_ok=True)
        Path(config.get('results_folder', 'NN/results')).mkdir(parents=True, exist_ok=True)
        Path(config['paths']['experiment_name']).mkdir(parents=True, exist_ok=True)

def validate_config(config: Dict[str, Any]) -> List[str]:
    """
    Valide une configuration et retourne les erreurs.

    Args:
        config: Configuration (dict ou Config object)

    Returns:
        Liste des erreurs (vide si valide)
    """
    if isinstance(config, Config):
        return config.validate()

    # Validation format dict (ancien code)
    errors = []

    try:
        if config['training']['batch_size'] <= 0:
            errors.append('training.batch_size must be > 0')
        if config['training']['num_epochs'] <= 0:
            errors.append('training.num_epochs must be > 0')
        if config['model']['d_model'] <= 0:
            errors.append('model.d_model must be > 0')
        if config['model']['num_heads'] <= 0:
            errors.append('model.num_heads must be > 0')
        if config['model']['d_model'] % config['model']['num_heads'] != 0:
            errors.append('model.d_model must be divisible by model.num_heads')
    except Exception as e:
        errors.append(f'Configuration unexpected format: {str(e)}')

    if config['data'].get('train_ratio', 0) + config['data'].get('val_ratio', 0) > 1.0:
        errors.append('data.train_ratio + data.val_ratio cannot exceed 1.0')

    if not Path(config['data']['data_dir']).exists():
        errors.append(f"Data directory '{config['data']['data_dir']}' does not exist")

    return errors