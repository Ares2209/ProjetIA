"""LightGBM pour la classification de spectres d'exoplanètes."""

import numpy as np
import lightgbm as lgb
from typing import Dict, Any, Optional, Tuple
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import joblib
import os

class LightGBMModel:
    """
    LightGBM pour l'analyse de spectres d'exoplanètes avec données auxiliaires.

    Combine les spectres et données auxiliaires en un seul vecteur de features
    pour l'entraînement avec LightGBM.
    """

    def __init__(
        self,
        spectrum_length: int,
        auxiliary_dim: int = 5,
        num_classes: int = 2,
        use_gpu: bool = False,
        random_state: int = 42
    ):
        """
        Args:
            spectrum_length: Longueur du spectre d'entrée
            auxiliary_dim: Dimension des données auxiliaires
            num_classes: Nombre de classes de sortie
            use_gpu: Utiliser GPU si disponible
            random_state: Seed pour reproductibilité
        """
        self.spectrum_length = spectrum_length
        self.auxiliary_dim = auxiliary_dim
        self.num_classes = num_classes
        self.use_gpu = use_gpu
        self.random_state = random_state

        # Feature dimension totale (spectre + auxiliaires)
        self.total_features = spectrum_length + auxiliary_dim

        # Scaler pour normalisation
        self.scaler = StandardScaler()

        # Modèles LightGBM (un par classe pour multi-label)
        self.models = {}
        self.best_params = {}

        # Configuration par défaut
        self.default_params = {
            'objective': 'binary',
            'metric': ['binary_logloss', 'auc'],
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.1,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'random_state': random_state,
            'n_estimators': 100,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'min_child_samples': 20,
            'min_child_weight': 0.001,
            'subsample_for_bin': 200000,
        }

        if use_gpu:
            self.default_params['device'] = 'gpu'

    def _prepare_features(self, spectrum: np.ndarray, auxiliary: np.ndarray) -> np.ndarray:
        """
        Combine spectres et données auxiliaires en un seul vecteur de features.

        Args:
            spectrum: (batch_size, spectrum_length) ou (batch_size, channels, spectrum_length)
            auxiliary: (batch_size, auxiliary_dim)

        Returns:
            features: (batch_size, total_features)
        """
        # Gérer différents formats de spectre
        if spectrum.ndim == 3:  # (B, C, L)
            # Utiliser seulement la moyenne (premier canal) ou aplatir
            spectrum = spectrum[:, 0, :]  # (B, L)

        # Concaténer spectres et auxiliaires
        features = np.concatenate([spectrum, auxiliary], axis=1)
        return features

    def fit(
        self,
        spectrum_train: np.ndarray,
        auxiliary_train: np.ndarray,
        targets_train: np.ndarray,
        spectrum_val: Optional[np.ndarray] = None,
        auxiliary_val: Optional[np.ndarray] = None,
        targets_val: Optional[np.ndarray] = None,
        params: Optional[Dict[str, Any]] = None
    ):
        """
        Entraîner le modèle LightGBM.

        Args:
            spectrum_train: Spectres d'entraînement
            auxiliary_train: Données auxiliaires d'entraînement
            targets_train: Labels d'entraînement (batch_size, num_classes)
            spectrum_val: Spectres de validation (optionnel)
            auxiliary_val: Données auxiliaires de validation (optionnel)
            targets_val: Labels de validation (optionnel)
            params: Paramètres LightGBM personnalisés
        """
        # Préparer les features
        X_train = self._prepare_features(spectrum_train, auxiliary_train)

        # Normalisation
        X_train = self.scaler.fit_transform(X_train)

        # Préparer validation si fournie
        eval_set = None
        eval_names = ['train']
        if spectrum_val is not None and auxiliary_val is not None and targets_val is not None:
            X_val = self._prepare_features(spectrum_val, auxiliary_val)
            X_val = self.scaler.transform(X_val)
            eval_set = [(X_train, targets_train[:, 0]), (X_val, targets_val[:, 0])]
            eval_names = ['train', 'val']
        else:
            eval_set = [(X_train, targets_train[:, 0])]

        # Paramètres
        model_params = self.default_params.copy()
        if params:
            model_params.update(params)

        # Entraîner un modèle par classe (multi-label)
        for class_idx in range(self.num_classes):
            print(f"Entraînement LightGBM pour la classe {class_idx}...")

            # Labels pour cette classe
            y_train_class = targets_train[:, class_idx]
            y_val_class = targets_val[:, class_idx] if targets_val is not None else None

            # Créer Dataset
            train_data = lgb.Dataset(X_train, label=y_train_class)

            if eval_set and len(eval_set) > 1:
                val_data = lgb.Dataset(X_val, label=y_val_class, reference=train_data)
                valid_sets = [train_data, val_data]
            else:
                valid_sets = [train_data]

            # Entraîner
            model = lgb.train(
                model_params,
                train_data,
                num_boost_round=model_params.get('n_estimators', 100),
                valid_sets=valid_sets,
                valid_names=eval_names,
                callbacks=[
                    lgb.early_stopping(stopping_rounds=20, verbose=False),
                    lgb.log_evaluation(period=0)  # Désactiver logs
                ]
            )

            self.models[class_idx] = model

        print("Modèles LightGBM entraînés")

    def predict_proba(self, spectrum: np.ndarray, auxiliary: np.ndarray) -> np.ndarray:
        """
        Prédire les probabilités.

        Args:
            spectrum: Spectres de test
            auxiliary: Données auxiliaires de test

        Returns:
            probas: (batch_size, num_classes) - probabilités
        """
        X = self._prepare_features(spectrum, auxiliary)
        X = self.scaler.transform(X)

        probas = []
        for class_idx in range(self.num_classes):
            proba = self.models[class_idx].predict(X, num_iteration=self.models[class_idx].best_iteration)
            probas.append(proba)

        return np.column_stack(probas)

    def predict(self, spectrum: np.ndarray, auxiliary: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """
        Prédire les classes avec seuil.

        Args:
            spectrum: Spectres de test
            auxiliary: Données auxiliaires de test
            threshold: Seuil de classification

        Returns:
            predictions: (batch_size, num_classes) - prédictions binaires
        """
        probas = self.predict_proba(spectrum, auxiliary)
        return (probas >= threshold).astype(int)

    def save(self, path: str):
        """
        Sauvegarder le modèle.

        Args:
            path: Chemin de sauvegarde (sans extension)
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)

        # Sauvegarder modèles
        for class_idx, model in self.models.items():
            model_path = f"{path}_class_{class_idx}.txt"
            model.save_model(model_path)

        # Sauvegarder scaler
        scaler_path = f"{path}_scaler.pkl"
        joblib.dump(self.scaler, scaler_path)

        # Sauvegarder paramètres
        params_path = f"{path}_params.json"
        import json
        with open(params_path, 'w') as f:
            json.dump({
                'spectrum_length': self.spectrum_length,
                'auxiliary_dim': self.auxiliary_dim,
                'num_classes': self.num_classes,
                'use_gpu': self.use_gpu,
                'random_state': self.random_state,
                'total_features': self.total_features
            }, f, indent=2)

        print(f"Modèle LightGBM sauvegardé dans {path}")

    @classmethod
    def load(cls, path: str) -> 'LightGBMModel':
        """
        Charger un modèle sauvegardé.

        Args:
            path: Chemin du modèle (sans extension)

        Returns:
            model: Modèle chargé
        """
        import json
        # Charger paramètres
        params_path = f"{path}_params.json"
        with open(params_path, 'r') as f:
            params = json.load(f)

        # Créer instance
        model = cls(
            spectrum_length=params['spectrum_length'],
            auxiliary_dim=params['auxiliary_dim'],
            num_classes=params['num_classes'],
            use_gpu=params['use_gpu'],
            random_state=params['random_state']
        )

        # Charger scaler
        scaler_path = f"{path}_scaler.pkl"
        model.scaler = joblib.load(scaler_path)

        # Charger modèles
        for class_idx in range(params['num_classes']):
            model_path = f"{path}_class_{class_idx}.txt"
            model.models[class_idx] = lgb.Booster(model_file=model_path)

        print(f"Modèle LightGBM chargé depuis {path}")
        return model

    def get_feature_importance(self, class_idx: int = 0, importance_type: str = 'gain') -> Dict[str, float]:
        """
        Obtenir l'importance des features pour une classe.

        Args:
            class_idx: Index de la classe
            importance_type: Type d'importance ('gain', 'split', etc.)

        Returns:
            importance: Dictionnaire feature -> importance
        """
        if class_idx not in self.models:
            raise ValueError(f"Classe {class_idx} non trouvée")

        importance = self.models[class_idx].feature_importance(importance_type=importance_type)

        # Créer noms de features
        feature_names = []
        for i in range(self.spectrum_length):
            feature_names.append(f'spectrum_{i}')
        for i in range(self.auxiliary_dim):
            feature_names.append(f'auxiliary_{i}')

        return dict(zip(feature_names, importance))

    def get_params(self) -> Dict[str, Any]:
        """Retourne les paramètres du modèle."""
        return {
            'spectrum_length': self.spectrum_length,
            'auxiliary_dim': self.auxiliary_dim,
            'num_classes': self.num_classes,
            'total_features': self.total_features,
            'use_gpu': self.use_gpu,
            'random_state': self.random_state
        }
