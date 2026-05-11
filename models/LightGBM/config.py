"""Configuration spécifique au modèle LightGBM."""

from training.config import (
    Config,
    ModelConfig,
    TrainingConfig,
    DataConfig,
    PathsConfig,
)

MODEL_DIR = "models/LightGBM"


def get_config() -> Config:
    """Retourne la configuration complète pour entraîner LightGBM."""
    return Config(
        model=ModelConfig(
            architecture="LightGBM",
            spectrum_length=283,
            auxiliary_dim=5,
            use_gpu=False,  # LightGBM peut utiliser GPU mais commençons par CPU
            random_state=42,
        ),
        training=TrainingConfig(
            num_classes=2,
            batch_size=32,  # Non utilisé par LightGBM mais gardé pour compatibilité
            num_epochs=1,   # LightGBM gère ses propres itérations
            learning_rate=0.1,
            patience=20,    # early_stopping_rounds
            weight_decay=0.0,  # Non utilisé par LightGBM
            label_smoothing=0.0,  # Non utilisé par LightGBM
            classification_threshold=0.5,
        ),
        data=DataConfig(),
        paths=PathsConfig(
            model_folder=f"{MODEL_DIR}/checkpoints",
            model_basename="lightgbm_model",
            experiment_name=f"{MODEL_DIR}/runs/lightgbm_v1",
        ),
        results_folder=f"{MODEL_DIR}/results",
    )