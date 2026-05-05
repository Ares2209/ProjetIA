import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd


class ExoplanetDataset(Dataset):
    def __init__(
        self,
        spectra,
        auxiliary_df,
        targets_df=None,
        is_train=True,
        augmentation_factor=0,
        shift_range=0.05,
        scale_range=0.1,
        noise_std=0.02,
        flip_prob=0,
        channel_dropout_prob=0.1,
        aux_mean=None,
        aux_std=None,
    ):
        self.is_train            = is_train
        self.augmentation_factor = augmentation_factor if is_train else 0

        self.shift_range          = shift_range
        self.scale_range          = scale_range
        self.noise_std            = noise_std
        self.flip_prob            = flip_prob
        self.channel_dropout_prob = channel_dropout_prob

        # Stats auxiliaires : calculées sur train, passées en val/test
        self.aux_features, self.aux_mean, self.aux_std = self._normalize_auxiliary(
            auxiliary_df, aux_mean, aux_std
        )

        # Per-sample → pas de paramètres à partager
        self.spectra, self.spectra_mean, self.spectra_std = self._normalize_spectra(spectra)

        self.targets = targets_df.reset_index(drop=True) if targets_df is not None else None

        self.original_size = len(self.spectra)
        self.total_size    = self.original_size * (1 + self.augmentation_factor)
        
    @staticmethod
    def _engineer_features(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # Rapport masse planète / masse étoile : hiérarchie gravitationnelle du système
        df['mass_ratio'] = df['planet_mass_kg'] / df['star_mass_kg']

        # Rapport demi-grand axe / rayon stellaire : géométrie du transit
        df['orbital_radius_ratio'] = df['semi_major_axis_m'] / df['star_radius_m']

        # Densité stellaire proxy (∝ M/R³) : structure interne de l'étoile
        df['star_density_proxy'] = df['star_mass_kg'] / df['star_radius_m'] ** 3

        # ── 2. Flux stellaire et température d'équilibre ─────────────────────────

        # Flux reçu par la planète (loi de Stefan-Boltzmann + géométrie en 1/r²)
        # F ∝ T_star⁴ × (R_star / a)²   [unités relatives]
        df['stellar_flux'] = (
            df['star_temperature'] ** 4
            * (df['star_radius_m'] / df['semi_major_axis_m']) ** 2
        )

        # Température d'équilibre (corps noir, albedo de Bond = 0.3)
        # T_eq = T_star × sqrt(R_star / 2a) × (1 - A)^(1/4)
        df['equilibrium_temp'] = (
            df['star_temperature']
            * np.sqrt(df['star_radius_m'] / (2 * df['semi_major_axis_m']))
            * (0.7 ** 0.25)
        )

        # ── 3. Indicateur zone habitable ─────────────────────────────────────────

        # Distance de la zone habitable : a pour lequel T_eq ≈ 255 K
        hz_distance = df['star_radius_m'] * (df['star_temperature'] / 255.0) ** 2
        # hz_ratio < 1 → trop proche (trop chaude)
        # hz_ratio ≈ 1 → zone habitable
        # hz_ratio > 1 → trop loin  (trop froide)
        df['hz_ratio'] = df['semi_major_axis_m'] / hz_distance

        # ── 4. Logs de toutes les colonnes à grande dynamique ────────────────────
        # Réduit les ordres de grandeur (1e25 → 1e30) à des valeurs dans [25, 30]
        # ce qui stabilise les activations du ResNet dès la première couche FC.
        log_targets = [
            'star_mass_kg',
            'star_radius_m',
            'star_temperature',
            'planet_mass_kg',
            'semi_major_axis_m',
            'stellar_flux',
            'equilibrium_temp',
            'star_density_proxy',
            'mass_ratio',
            'orbital_radius_ratio',
        ]
        for col in log_targets:
            if col in df.columns:
                df[f'log_{col}'] = np.log10(df[col].clip(lower=1e-30))

        return df

    def _normalize_auxiliary(self, auxiliary_df: pd.DataFrame,
                              aux_mean=None, aux_std=None):
        """Applique _engineer_features puis normalise (mean=0, std=1)."""
        df = ExoplanetDataset._engineer_features(auxiliary_df)
        features = df.values.astype(np.float32)

        if aux_mean is None:
            aux_mean = features.mean(axis=0)
            aux_std  = features.std(axis=0) + 1e-8

        normalized = (features - aux_mean) / aux_std
        return normalized, aux_mean, aux_std

    def _normalize_spectra(self, spectra, mean=None, std=None):
        spectra = spectra.astype(np.float32)  # (N, 52, 3)

        # Canaux dérivés calculés sur les données brutes (avant normalisation)
        mu  = spectra[:, :, 0:1]   # (N, 52, 1) — profondeur moyenne
        lo  = spectra[:, :, 1:2]   # (N, 52, 1) — borne basse de l'incertitude
        hi  = spectra[:, :, 2:3]   # (N, 52, 1) — borne haute de l'incertitude

        # Canal 4 : rapport signal/bruit (SNR) local — mesure la fiabilité de chaque point
        snr  = mu / (0.5 * np.abs(hi - lo) + 1e-8)        # (N, 52, 1)

        # Canal 5 : incertitude relative — normalise l'amplitude de l'incertitude
        rel_unc = np.abs(hi - lo) / (np.abs(mu) + 1e-8)   # (N, 52, 1)

        spectra = np.concatenate([spectra, snr, rel_unc], axis=2)  # (N, 52, 5)

        # Calcul par échantillon : mean/std sur les 52 pas de temps, par canal
        mean = spectra.mean(axis=1, keepdims=True)          # (N, 1, 5)
        std  = spectra.std(axis=1,  keepdims=True) + 1e-8   # (N, 1, 5)

        normalized = (spectra - mean) / std

        # On retourne None pour mean/std : inutiles en per-sample
        # (la val/test se normalisent elles-mêmes)
        return normalized, None, None

    def _augment_spectrum(self, spectrum: np.ndarray) -> np.ndarray:
        """
        Pipeline d'augmentation appliqué aléatoirement.
        Chaque transformation est indépendante et peut être désactivée
        via ses paramètres (mettre la prob/range à 0).

        Args:
            spectrum : (52, 5)  float32  — copie déjà faite par l'appelant
        Returns:
            spectrum augmenté : (52, 5)
        """

        if self.shift_range > 0:
            shift = np.random.uniform(-self.shift_range, self.shift_range,
                                      size=(1, spectrum.shape[1]))   # (1, 3)
            spectrum = spectrum + shift

        if self.scale_range > 0:
            scale = np.random.uniform(1 - self.scale_range, 1 + self.scale_range,
                                      size=(1, spectrum.shape[1]))   # (1, 3)
            spectrum = spectrum * scale

        if self.noise_std > 0:
            noise    = np.random.normal(0, self.noise_std, size=spectrum.shape)
            spectrum = spectrum + noise

        if self.flip_prob > 0 and np.random.rand() < self.flip_prob:
            spectrum = spectrum[::-1].copy()   # .copy() pour contiguïté mémoire

        if self.channel_dropout_prob > 0:
            for c in range(spectrum.shape[1]):
                if np.random.rand() < self.channel_dropout_prob:
                    spectrum[:, c] = 0.0

        return spectrum

    def __len__(self):
        return self.total_size

    def __getitem__(self, idx):
        original_idx = idx % self.original_size
        aug_version  = idx // self.original_size   # 0 = original, >0 = augmenté

        spectrum = self.spectra[original_idx].copy()   # (52, 3) — copie obligatoire
        aux_feat = self.aux_features[original_idx]

        if aug_version > 0 and self.is_train and self.augmentation_factor > 0:
            spectrum = self._augment_spectrum(spectrum)

        item = {
            'spectrum':  torch.FloatTensor(spectrum),
            'auxiliary': torch.FloatTensor(aux_feat),
            'id':        original_idx,
        }

        if self.targets is not None:
            eau   = self.targets.iloc[original_idx]['eau']
            nuage = self.targets.iloc[original_idx]['nuage']
            item['target'] = torch.FloatTensor([eau, nuage])

        return item
    
def collate_fn(batch):
    """Collation → spectres au format (B, C, 52) pour Conv1d (C=5 avec canaux dérivés)."""
    spectra   = torch.stack([item['spectrum']  for item in batch])   # (B, 52, C)
    spectra   = spectra.permute(0, 2, 1)                             # (B, C,  52)
    auxiliary = torch.stack([item['auxiliary'] for item in batch])   # (B, n_feat)
    ids       = torch.LongTensor([item['id']   for item in batch])

    if 'target' in batch[0]:
        targets = torch.stack([item['target'] for item in batch])    # (B, 2)
        return spectra, auxiliary, targets, ids

    return spectra, auxiliary, ids
