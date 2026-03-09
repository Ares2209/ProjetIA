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


    def _normalize_auxiliary(self, df, mean=None, std=None):

        df = df.copy()

        log_cols = [
            "star_mass_kg",
            "star_radius_m",
            "planet_mass_kg",
            "semi_major_axis_m"
        ]

        for col in log_cols:
            if col in df.columns:
                df[col] = np.log10(df[col])

        features = df.select_dtypes(include=[np.number]).values.astype(np.float32)

        if mean is None:
            mean = features.mean(axis=0)
            std  = features.std(axis=0) + 1e-8

        normalized = (features - mean) / std

        return normalized, mean, std

    def _normalize_spectra(self, spectra, mean=None, std=None):
        spectra = spectra.astype(np.float32)  # (N, 52, 3)

        # Calcul par échantillon : mean/std sur les 52 pas de temps, par canal
        mean = spectra.mean(axis=1, keepdims=True)          # (N, 1, 3)
        std  = spectra.std(axis=1,  keepdims=True) + 1e-8   # (N, 1, 3)

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
            spectrum : (52, 3)  float32  — copie déjà faite par l'appelant
        Returns:
            spectrum augmenté : (52, 3)
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
    """Collation → spectres au format (B, 3, 52) pour Conv1d."""
    spectra   = torch.stack([item['spectrum']  for item in batch])   # (B, 52, 3)
    spectra   = spectra.permute(0, 2, 1)                             # (B, 3,  52)
    auxiliary = torch.stack([item['auxiliary'] for item in batch])   # (B, n_feat)
    ids       = torch.LongTensor([item['id']   for item in batch])

    if 'target' in batch[0]:
        targets = torch.stack([item['target'] for item in batch])    # (B, 2)
        return spectra, auxiliary, targets, ids

    return spectra, auxiliary, ids
