"""Configuration specific to the LightGBM model."""

from training.config import (
    Config,
    ModelConfig,
    TrainingConfig,
    DataConfig,
    PathsConfig,
)

MODEL_DIR = "models/LightGBM"


def get_config() -> Config:
    """Return the complete configuration for LightGBM training."""
    return Config(
        model=ModelConfig(
            architecture="LightGBM",
            input_channels=5,
            auxiliary_dim=21,
            use_pca=False,
            pca_components=50,
            use_statistical_features=True,
            use_diff_features=True,
            n_estimators=1000,
            num_leaves=31,
            max_depth=-1,
            min_child_samples=20,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_alpha=0.0,
            reg_lambda=1.0,
            class_weight="balanced",
        ),
        training=TrainingConfig(
            num_classes=2,
            batch_size=32,
            num_epochs=1,
            learning_rate=0.03,
            patience=50,
            classification_threshold=0.5,
        ),
        data=DataConfig(
            augmentation_factor=0,
            num_workers=0,
            pin_memory=False,
        ),
        paths=PathsConfig(
            model_folder=f"{MODEL_DIR}/checkpoints",
            model_basename="lightgbm_model",
            experiment_name=f"{MODEL_DIR}/runs/lightgbm_v1",
        ),
        results_folder=f"{MODEL_DIR}/results",
    )
