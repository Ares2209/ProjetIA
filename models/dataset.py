"""Dataset Exoplanet — chargement spectres + features auxiliaires + augmentation cohérente.

Pipeline de normalisation (sans data leakage) :
    1. Train calcule mean/std (auxiliaires + spectres avec canaux dérivés) en float64
    2. Val/test reçoivent ces statistiques en argument et les appliquent
    3. Si val/test n'ont pas reçu les stats du train → warning explicite

Pipeline d'un échantillon dans __getitem__ :
    1. raw  : (52, 3)   — copie du spectre brut stocké en float32
    2. raw' = augment_raw(raw)         (uniquement si aug_version > 0)
    3. spectrum = ajout SNR + rel_unc  → (52, 5)
    4. spectrum = (spectrum - μ_train) / σ_train      ← normalisation GLOBALE
    5. channel_dropout sur ch3/4       (uniquement si augmentation)

Politique d'augmentation (toutes invariantes par changement d'échelle entre canaux,
nécessaire car ch0 ~10⁰, ch1 ~10⁻³, ch2 ~10⁻⁷) :
    - shift_range : décalage de l'axe λ (en fraction de la longueur)
                    simule une erreur de calibration en longueur d'onde.
    - scale_range : facteur multiplicatif partagé entre mu/lo/hi
                    simule une dérive globale de calibration, préserve les ratios.
    - noise_std   : bruit gaussien RELATIF per-pixel  →  raw *= (1 + N(0, σ))
                    simule du bruit détecteur proportionnel au signal.
    - channel_dropout_prob : zéroïse aléatoirement les canaux DÉRIVÉS (SNR, rel_unc).
                    le canal mu n'est jamais effacé.
"""

import logging
import random
import warnings

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


logger = logging.getLogger(__name__)

# Index des canaux après _add_derived_channels
CH_MU, CH_LO, CH_HI, CH_SNR, CH_REL_UNC = 0, 1, 2, 3, 4
SIGNAL_CHANNELS = slice(0, 3)            # ch0, ch1, ch2 (signal brut)
META_CHANNELS   = (CH_SNR, CH_REL_UNC)   # canaux dérivés


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
        spectra_mean=None,
        spectra_std=None,
    ):
        self.is_train            = is_train
        self.augmentation_factor = augmentation_factor if is_train else 0

        self.shift_range          = shift_range
        self.scale_range          = scale_range
        self.noise_std            = noise_std
        self.channel_dropout_prob = channel_dropout_prob

        # Garde-fou : val/test doivent recevoir les stats du train
        if not is_train:
            missing = []
            if aux_mean is None or aux_std is None:
                missing.append('aux_mean/aux_std')
            if spectra_mean is None or spectra_std is None:
                missing.append('spectra_mean/spectra_std')
            if missing:
                warnings.warn(
                    f"Dataset val/test sans stats train ({', '.join(missing)}) : "
                    "les statistiques seront recalculées sur ce split (data leakage probable).",
                    UserWarning, stacklevel=2,
                )

        # Auxiliaires : engineer features puis normalise (mean/std en float64 → pas d'overflow)
        self.aux_features, self.aux_mean, self.aux_std = self._normalize_auxiliary(
            auxiliary_df, aux_mean, aux_std,
        )

        # Spectres : on garde le BRUT (52, 3) en mémoire, l'augmentation et le calcul
        # des canaux dérivés se font à __getitem__ pour préserver la cohérence physique.
        self.raw_spectra = np.ascontiguousarray(spectra, dtype=np.float32)   # (N, 52, 3)

        # Stats globales pour la normalisation des spectres (sur les 5 canaux)
        if spectra_mean is None or spectra_std is None:
            spectra_mean, spectra_std = self._compute_spectra_stats(self.raw_spectra)
        self.spectra_mean = spectra_mean.astype(np.float32)   # (1, 52, 5)
        self.spectra_std  = spectra_std.astype(np.float32)

        self.targets = targets_df.reset_index(drop=True) if targets_df is not None else None

        self.original_size = len(self.raw_spectra)
        self.total_size    = self.original_size * (1 + self.augmentation_factor)

    # ── FEATURES AUXILIAIRES ────────────────────────────────────────────────

    @staticmethod
    def _engineer_features(df: pd.DataFrame) -> pd.DataFrame:
        """Enrichit le DataFrame auxiliaire avec des features physiquement motivées.

        Toutes les transformations sont déterministes (pas de fit) → peut être
        appliqué sur le DataFrame complet AVANT le split train/val sans leakage.
        """
        df = df.copy()

        df['mass_ratio']           = df['planet_mass_kg']    / df['star_mass_kg']
        df['orbital_radius_ratio'] = df['semi_major_axis_m'] / df['star_radius_m']
        df['star_density_proxy']   = df['star_mass_kg']      / df['star_radius_m'] ** 3

        df['stellar_flux'] = (
            df['star_temperature'] ** 4
            * (df['star_radius_m'] / df['semi_major_axis_m']) ** 2
        )
        df['equilibrium_temp'] = (
            df['star_temperature']
            * np.sqrt(df['star_radius_m'] / (2 * df['semi_major_axis_m']))
            * (0.7 ** 0.25)   # albedo de Bond = 0.3
        )

        hz_distance = df['star_radius_m'] * (df['star_temperature'] / 255.0) ** 2
        df['hz_ratio'] = df['semi_major_axis_m'] / hz_distance

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

    @staticmethod
    def _normalize_auxiliary(auxiliary_df: pd.DataFrame, aux_mean=None, aux_std=None):
        """Engineer + z-score. Stats calculées en float64 (overflow possible
        sur des features comme star_density_proxy ~10²⁶ qui débordent float32 au carré)."""
        df = ExoplanetDataset._engineer_features(auxiliary_df)
        features = df.values.astype(np.float64)   # critique pour les features à grande dynamique

        if aux_mean is None or aux_std is None:
            aux_mean = features.mean(axis=0)
            aux_std  = features.std(axis=0) + 1e-8

        normalized = ((features - aux_mean) / aux_std).astype(np.float32)
        return normalized, aux_mean, aux_std

    # ── SPECTRES ────────────────────────────────────────────────────────────

    @staticmethod
    def _add_derived_channels(raw_3ch: np.ndarray) -> np.ndarray:
        """Concat SNR et incertitude relative. Accepte (52, 3) ou (N, 52, 3)."""
        # Indexation slice qui marche pour 2D et 3D
        mu = raw_3ch[..., 0:1]
        lo = raw_3ch[..., 1:2]
        hi = raw_3ch[..., 2:3]

        snr     = mu / (0.5 * np.abs(hi - lo) + 1e-8)
        rel_unc = np.abs(hi - lo) / (np.abs(mu) + 1e-8)

        return np.concatenate([raw_3ch, snr, rel_unc], axis=-1).astype(np.float32)

    @staticmethod
    def _compute_spectra_stats(raw_spectra: np.ndarray):
        """Statistiques globales (1, 52, 5) calculées sur l'ensemble train.

        Le calcul est fait en float64 pour éviter les imprécisions cumulatives,
        puis renvoyé en float32 (ré-casté côté appelant).
        """
        full = ExoplanetDataset._add_derived_channels(raw_spectra)        # (N, 52, 5)
        full_f64 = full.astype(np.float64)
        mean = full_f64.mean(axis=0, keepdims=True)                       # (1, 52, 5)
        std  = full_f64.std(axis=0,  keepdims=True) + 1e-8
        return mean, std

    # ── AUGMENTATION ─────────────────────────────────────────────────────────

    def _augment_raw(self, raw: np.ndarray) -> np.ndarray:
        """Augmentation MULTIPLICATIVE sur le signal brut (52, 3).

        Toutes les opérations sont invariantes par changement d'échelle entre
        canaux (mu ~10⁰, lo ~10⁻³, hi ~10⁻⁷) : indispensable pour que les
        mêmes hyperparamètres (`shift_range`, `noise_std`) aient un sens
        physique uniforme sur les 3 canaux.
        """
        n_lambda = raw.shape[0]

        # 1. Décalage de l'axe λ (par roll, bordures remplies par la valeur du bord)
        if self.shift_range > 0:
            max_shift = int(round(self.shift_range * n_lambda))
            if max_shift > 0:
                n = int(np.random.randint(-max_shift, max_shift + 1))
                if n != 0:
                    raw = np.roll(raw, n, axis=0)
                    if n > 0:
                        raw[:n] = raw[n:n + 1]
                    else:
                        raw[n:] = raw[n - 1:n]

        # 2. Scale multiplicatif partagé entre mu/lo/hi (préserve SNR)
        if self.scale_range > 0:
            scale = float(np.random.uniform(1 - self.scale_range, 1 + self.scale_range))
            raw = raw * scale

        # 3. Bruit gaussien RELATIF per-pixel per-canal (raw *= 1 + N(0, σ))
        if self.noise_std > 0:
            rel_noise = np.random.normal(0, self.noise_std, size=raw.shape).astype(np.float32)
            raw = raw * (1.0 + rel_noise)

        return raw

    # ── PYTORCH DATASET API ─────────────────────────────────────────────────

    def __len__(self):
        return self.total_size

    def __getitem__(self, idx):
        original_idx = idx % self.original_size
        aug_version  = idx // self.original_size   # 0 = original, >0 = augmenté

        raw = self.raw_spectra[original_idx].copy()   # (52, 3) — copie obligatoire

        is_augment_pass = (
            aug_version > 0 and self.is_train and self.augmentation_factor > 0
        )

        if is_augment_pass:
            raw = self._augment_raw(raw)

        # Calcul des canaux dérivés à partir du raw (potentiellement augmenté)
        # → SNR/rel_unc cohérents avec les mu/lo/hi visibles par le réseau
        spectrum = self._add_derived_channels(raw)                              # (52, 5)

        # Normalisation avec les stats GLOBALES du train (broadcast (1,52,5)→(52,5))
        spectrum = (spectrum - self.spectra_mean[0]) / self.spectra_std[0]

        # Channel dropout sur les canaux dérivés (post-normalisation)
        if is_augment_pass and self.channel_dropout_prob > 0:
            for c in META_CHANNELS:
                if np.random.rand() < self.channel_dropout_prob:
                    spectrum[:, c] = 0.0

        item = {
            'spectrum':  torch.from_numpy(spectrum),
            'auxiliary': torch.from_numpy(self.aux_features[original_idx]),
            'id':        original_idx,
        }

        if self.targets is not None:
            eau   = self.targets.iloc[original_idx]['eau']
            nuage = self.targets.iloc[original_idx]['nuage']
            item['target'] = torch.tensor([eau, nuage], dtype=torch.float32)

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
    auxiliary = torch.stack([item['auxiliary'] for item in batch])
    ids       = torch.LongTensor([item['id']   for item in batch])

    if 'target' in batch[0]:
        targets = torch.stack([item['target'] for item in batch])
        return spectra, auxiliary, targets, ids

    return spectra, auxiliary, ids
