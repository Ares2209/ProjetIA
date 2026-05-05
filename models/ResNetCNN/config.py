"""Configuration spécifique au modèle ResNetCNN (ensemble SE-ResNet8 + ResNet18)."""

from training.config import (
    Config,
    ModelConfig,
    TrainingConfig,
    DataConfig,
    PathsConfig,
)

MODEL_DIR = "models/ResNetCNN"


def get_config() -> Config:
    """Retourne la configuration complète pour entraîner le ResNetCNN."""
    return Config(
        model=ModelConfig(
            architecture="2ResNet",
            spectrum_length=283,
            input_channels=5,
            auxiliary_dim=5,
            dropout=0.3,
            use_batch_norm=True,
        ),
        training=TrainingConfig(
            num_classes=2,
            batch_size=32,
            num_epochs=500,
            learning_rate=3e-4,
            patience=50,
            weight_decay=1e-4,
            label_smoothing=0.05,
            classification_threshold=0.5,
        ),
        data=DataConfig(),
        paths=PathsConfig(
            model_folder=f"{MODEL_DIR}/checkpoints",
            model_basename="resnet_model",
            experiment_name=f"{MODEL_DIR}/runs/resnet_v1",
        ),
        results_folder=f"{MODEL_DIR}/results",
    )
