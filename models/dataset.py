import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd

class ExoplanetDataset(Dataset):
    def __init__(self, spectra, auxiliary_df, targets_df=None, is_train=True,
                 augmentation_factor=0, shift_range=0.05, scale_range=0.1,
                 aux_mean=None, aux_std=None):
        """
        Dataset pour la classification des spectres d'exoplanètes.

        Args:
            spectra: numpy array (N, 52, 3) déjà chargé
            auxiliary_df: DataFrame des données auxiliaires
            targets_df: DataFrame des labels (None pour test)
            is_train: True si données d'entraînement, False si val/test
            augmentation_factor: Nombre de versions augmentées par échantillon (0 = pas d'augmentation)
            shift_range: Plage de décalage vertical (en fraction)
            scale_range: Plage de mise à l'échelle (en fraction)
            aux_mean: Moyenne pour normalisation auxiliaire (calculée sur train uniquement)
            aux_std: Std pour normalisation auxiliaire (calculée sur train uniquement)
        """
        self.is_train = is_train
        self.shift_range = shift_range
        self.scale_range = scale_range

        # Normaliser les données auxiliaires
        # Les stats sont passées en paramètre pour éviter le leakage
        self.aux_features, self.aux_mean, self.aux_std = self._normalize_auxiliary(
            auxiliary_df, aux_mean, aux_std
        )

        # Normaliser les spectres par échantillon (per-sample → pas de leakage)
        self.spectra = self._normalize_spectra(spectra)

        # Targets
        self.targets = targets_df.reset_index(drop=True) if targets_df is not None else None

        # Augmentation uniquement en train
        self.augmentation_factor = augmentation_factor if is_train else 0
        if not is_train and augmentation_factor > 0:
            print("⚠️  Augmentation désactivée en validation/test")

        self.original_size = len(self.spectra)
        self.total_size = self.original_size * (1 + self.augmentation_factor)

        if self.augmentation_factor > 0:
            print(f"📈 Data Augmentation activée:")
            print(f"   ↳ Dataset original: {self.original_size} exemples")
            print(f"   ↳ Dataset augmenté: {self.total_size} exemples")
            print(f"   ↳ Facteur: x{self.augmentation_factor + 1}")

    # ------------------------------------------------------------------ #
    #  Méthodes publiques Dataset                                          #
    # ------------------------------------------------------------------ #

    def __len__(self):
        return self.total_size

    def __getitem__(self, idx):
        original_idx = idx % self.original_size
        aug_version   = idx // self.original_size

        spectrum = self.spectra[original_idx].copy()   # (52, 3)
        aux_feat = self.aux_features[original_idx]

        if aug_version > 0 and self.augmentation_factor > 0:
            spectrum = self._augment_spectrum(spectrum, original_idx, aug_version)

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

    # ------------------------------------------------------------------ #
    #  Augmentation                                                        #
    # ------------------------------------------------------------------ #

    def _augment_spectrum(self, spectrum, original_idx, aug_version):
        seed = original_idx * 1000 + aug_version
        np.random.seed(seed)
        augmented = spectrum.copy()

        if np.random.rand() > 0.3:
            augmented = self._vertical_shift(augmented)
        if np.random.rand() > 0.3:
            augmented = self._scale_spectrum(augmented)
        if np.random.rand() > 0.5:
            augmented = self._wavelength_shift(augmented)
        if np.random.rand() > 0.5:
            augmented = self._augment_uncertainties(augmented)

        return augmented

    def _vertical_shift(self, spectrum):
        shift = np.random.uniform(-self.shift_range, self.shift_range)
        spectrum[:, 0] += shift
        return spectrum

    def _scale_spectrum(self, spectrum):
        scale = np.random.uniform(1 - self.scale_range, 1 + self.scale_range)
        spectrum[:, 0] *= scale
        return spectrum

    def _wavelength_shift(self, spectrum):
        shift_pixels = np.random.randint(-2, 3)
        if shift_pixels == 0:
            return spectrum
        shifted = np.roll(spectrum, shift_pixels, axis=0)
        if shift_pixels > 0:
            for i in range(shift_pixels):
                shifted[i] = spectrum[0]
        else:
            for i in range(abs(shift_pixels)):
                shifted[-(i + 1)] = spectrum[-1]
        return shifted

    def _augment_uncertainties(self, spectrum):
        scale = np.random.uniform(0.9, 1.1)
        spectrum[:, 1] *= scale
        spectrum[:, 2] *= scale
        return spectrum

    # ------------------------------------------------------------------ #
    #  Normalisation                                                       #
    # ------------------------------------------------------------------ #

    def _normalize_auxiliary(self, auxiliary_df, mean=None, std=None):
        """
        Normalisation Z-score des features auxiliaires.
        Si mean/std sont None (cas train), ils sont calculés ici.
        Sinon (cas val/test), les stats du train sont réutilisées.
        """
        aux_data = auxiliary_df.values.astype(np.float32)

        if mean is None or std is None:
            mean = np.mean(aux_data, axis=0)
            std  = np.std(aux_data,  axis=0) + 1e-8

        normalized = (aux_data - mean) / std
        return normalized.astype(np.float32), mean, std

    def _normalize_spectra(self, spectra):
        """
        Normalisation Min-Max par échantillon et par canal.
        ✅ Pas de leakage : chaque spectre est normalisé indépendamment.
        Input shape : (N, 52, 3)
        """
        spectra = spectra.astype(np.float32)
        normalized = np.zeros_like(spectra)

        for i in range(3):  # moyenne, incertitude basse, incertitude haute
            channel   = spectra[:, :, i]                          # (N, 52)
            min_vals  = channel.min(axis=1, keepdims=True)        # (N, 1)
            max_vals  = channel.max(axis=1, keepdims=True)        # (N, 1)
            normalized[:, :, i] = (channel - min_vals) / (max_vals - min_vals + 1e-8)

        return normalized


# ------------------------------------------------------------------ #
#  Collate                                                             #
# ------------------------------------------------------------------ #

def collate_fn(batch):
    """
    Collation → spectres au format (B, 3, 52) pour Conv1d.
    """
    spectra   = torch.stack([item['spectrum']  for item in batch])  # (B, 52, 3)
    spectra   = spectra.permute(0, 2, 1)                            # (B, 3,  52)
    auxiliary = torch.stack([item['auxiliary'] for item in batch])  # (B, n_feat)
    ids       = torch.LongTensor([item['id']   for item in batch])

    if 'target' in batch[0]:
        targets = torch.stack([item['target'] for item in batch])   # (B, 2)
        return spectra, auxiliary, targets, ids

    return spectra, auxiliary, ids
