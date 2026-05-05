"""Configuration spécifique au modèle XGBoost (à compléter)."""

from training.config import (
    Config,
    ModelConfig,
    TrainingConfig,
    DataConfig,
    PathsConfig,
)

MODEL_DIR = "models/XGBoost"


def get_config() -> Config:
    """Retourne la configuration complète pour entraîner XGBoost."""
    return Config(
        model=ModelConfig(
            architecture="XGBoost",
            spectrum_length=283,
            input_channels=5,
            auxiliary_dim=5,
            use_pca=False,
            pca_components=50,
            use_statistical_features=True,
            use_diff_features=False,
        ),
        training=TrainingConfig(
            num_classes=2,
            batch_size=32,
            num_epochs=500,
            learning_rate=3e-4,
        ),
        data=DataConfig(),
        paths=PathsConfig(
            model_folder=f"{MODEL_DIR}/checkpoints",
            model_basename="xgboost_model",
            experiment_name=f"{MODEL_DIR}/runs/xgboost_v1",
        ),
        results_folder=f"{MODEL_DIR}/results",
    )
