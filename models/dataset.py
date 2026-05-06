"""Dataset Exoplanet — chargement spectres + features auxiliaires + augmentation cohérente.

Pipeline d'un échantillon (canaux après _normalize_spectra) :
    ch0 = mu       (profondeur moyenne)
    ch1 = lo       (borne basse de l'incertitude)
    ch2 = hi       (borne haute de l'incertitude)
    ch3 = SNR      (canal dérivé)
    ch4 = rel_unc  (canal dérivé)

Politique d'augmentation :
    - shift / scale partagés entre ch0/ch1/ch2 (préserve la cohérence physique)
    - noise gaussien per-pixel sur ch0/ch1/ch2
    - canaux dérivés (ch3, ch4) inchangés : ils décrivent la qualité de la
      mesure originale, pas une version perturbée
    - channel_dropout uniquement sur les canaux dérivés (métadonnées)
    - pas de flip horizontal (les bandes d'absorption sont à des longueurs
      d'onde fixes, le flip détruit l'information physique)
"""

import logging
import random
import warnings

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


logger = logging.getLogger(__name__)

# Index des canaux après _normalize_spectra
CH_MU, CH_LO, CH_HI, CH_SNR, CH_REL_UNC = 0, 1, 2, 3, 4
SIGNAL_CHANNELS = slice(0, 3)        # ch0, ch1, ch2
META_CHANNELS   = (CH_SNR, CH_REL_UNC)


class ExoplanetDataset(Dataset):
    def __init__(
        self,
        spectra,
        auxiliary_df,
        targets_df=None,
        is_train=True,
        augmentation_factor=0,
        shift_range=0.05,
        scale_range=0.10,
        noise_std=0.02,
        channel_dropout_prob=0.1,
        aux_mean=None,
        aux_std=None,
    ):
        self.is_train            = is_train
        self.augmentation_factor = augmentation_factor if is_train else 0

        self.shift_range          = shift_range
        self.scale_range          = scale_range
        self.noise_std            = noise_std
        self.channel_dropout_prob = channel_dropout_prob

        # Garde-fou : val/test doivent recevoir aux_mean/aux_std du train
        if not is_train and aux_mean is None:
            warnings.warn(
                "Dataset val/test sans aux_mean/aux_std du train : "
                "stats recalculées sur ce split (data leakage probable).",
                UserWarning, stacklevel=2,
            )

        self.aux_features, self.aux_mean, self.aux_std = self._normalize_auxiliary(
            auxiliary_df, aux_mean, aux_std
        )

        # spectres : (N, 52, 5) après ajout des canaux dérivés et normalisation
        self.spectra, self.spectra_mean, self.spectra_std = self._normalize_spectra(spectra)

        self.targets = targets_df.reset_index(drop=True) if targets_df is not None else None

        self.original_size = len(self.spectra)
        self.total_size    = self.original_size * (1 + self.augmentation_factor)

    @staticmethod
    def _engineer_features(df: pd.DataFrame) -> pd.DataFrame:
        """Enrichit le DataFrame auxiliaire avec des features physiquement motivées.

        Toutes les transformations sont déterministes (pas de fit) → peut être
        appliqué sur le DataFrame complet AVANT le split train/val sans leakage.
        """
        df = df.copy()

        # Ratios dimensionnels
        df['mass_ratio']           = df['planet_mass_kg']     / df['star_mass_kg']
        df['orbital_radius_ratio'] = df['semi_major_axis_m']  / df['star_radius_m']
        df['star_density_proxy']   = df['star_mass_kg']       / df['star_radius_m'] ** 3

        # Flux stellaire et température d'équilibre
        df['stellar_flux'] = (
            df['star_temperature'] ** 4
            * (df['star_radius_m'] / df['semi_major_axis_m']) ** 2
        )
        df['equilibrium_temp'] = (
            df['star_temperature']
            * np.sqrt(df['star_radius_m'] / (2 * df['semi_major_axis_m']))
            * (0.7 ** 0.25)   # albedo de Bond = 0.3
        )

        # Indicateur zone habitable
        hz_distance = df['star_radius_m'] * (df['star_temperature'] / 255.0) ** 2
        df['hz_ratio'] = df['semi_major_axis_m'] / hz_distance

        # Logs des colonnes à grande dynamique
        log_targets = [
            'star_mass_kg', 'star_radius_m', 'star_temperature',
            'planet_mass_kg', 'semi_major_axis_m',
            'stellar_flux', 'equilibrium_temp', 'star_density_proxy',
            'mass_ratio', 'orbital_radius_ratio',
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

    def _normalize_spectra(self, spectra):
        """Ajoute SNR + incertitude relative puis normalise per-sample.

        Entrée : (N, 52, 3) — mu, lo, hi
        Sortie : (N, 52, 5) — mu, lo, hi, SNR, rel_unc, normalisés
                              par échantillon et par canal
        """
        spectra = spectra.astype(np.float32)
        mu = spectra[:, :, 0:1]   # (N, 52, 1)
        lo = spectra[:, :, 1:2]
        hi = spectra[:, :, 2:3]

        snr     = mu / (0.5 * np.abs(hi - lo) + 1e-8)
        rel_unc = np.abs(hi - lo) / (np.abs(mu) + 1e-8)

        spectra = np.concatenate([spectra, snr, rel_unc], axis=2)   # (N, 52, 5)

        mean = spectra.mean(axis=1, keepdims=True)
        std  = spectra.std(axis=1, keepdims=True) + 1e-8
        normalized = (spectra - mean) / std

        # Per-sample : pas de stats à partager avec val/test
        return normalized, None, None

    def _augment_spectrum(self, spectrum: np.ndarray) -> np.ndarray:
        """Augmentation cohérente pour spectres (52, 5).

        - shift / scale appliqués UNIFORMÉMENT à (mu, lo, hi) → ratios préservés
        - bruit gaussien per-pixel sur (mu, lo, hi)
        - canaux dérivés (SNR, rel_unc) NON modifiés par shift/scale/noise :
          ils gardent leur sens de "métadonnée de qualité de la mesure brute"
        - channel_dropout : uniquement sur les canaux dérivés
        """
        # shift partagé entre mu/lo/hi (un scalaire pour les 3)
        if self.shift_range > 0:
            shift = float(np.random.uniform(-self.shift_range, self.shift_range))
            spectrum[:, SIGNAL_CHANNELS] += shift

        # scale partagé entre mu/lo/hi
        if self.scale_range > 0:
            scale = float(np.random.uniform(1 - self.scale_range, 1 + self.scale_range))
            spectrum[:, SIGNAL_CHANNELS] *= scale

        # bruit indépendant par pixel et par canal du signal brut
        if self.noise_std > 0:
            noise = np.random.normal(
                0, self.noise_std, size=(spectrum.shape[0], 3),
            ).astype(np.float32)
            spectrum[:, SIGNAL_CHANNELS] += noise

        # dropout des canaux dérivés (métadonnées) seulement
        if self.channel_dropout_prob > 0:
            for c in META_CHANNELS:
                if np.random.rand() < self.channel_dropout_prob:
                    spectrum[:, c] = 0.0

        return spectrum

    def __len__(self):
        return self.total_size

    def __getitem__(self, idx):
        original_idx = idx % self.original_size
        aug_version  = idx // self.original_size   # 0 = original, >0 = augmenté

        spectrum = self.spectra[original_idx].copy()   # (52, 5) — copie obligatoire
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


def seed_worker(_worker_id: int) -> None:
    """À passer en `worker_init_fn` du DataLoader.

    Sans ça, chaque worker hérite du même état numpy/random que le process
    parent → mêmes augmentations en parallèle. Cette fonction utilise la seed
    propre à chaque worker (générée par PyTorch) pour réinitialiser numpy et
    random, garantissant des augmentations diverses entre workers et stables
    entre epochs.
    """
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)


def collate_fn(batch):
    """Collation → spectres au format (B, C, 52) pour Conv1d (C=5)."""
    spectra   = torch.stack([item['spectrum']  for item in batch])   # (B, 52, C)
    spectra   = spectra.permute(0, 2, 1)                             # (B, C, 52)
    auxiliary = torch.stack([item['auxiliary'] for item in batch])   # (B, n_feat)
    ids       = torch.LongTensor([item['id']   for item in batch])

    if 'target' in batch[0]:
        targets = torch.stack([item['target'] for item in batch])    # (B, 2)
        return spectra, auxiliary, targets, ids

    return spectra, auxiliary, ids
