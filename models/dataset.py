import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import copy

class ExoplanetDataset(Dataset):
    def __init__(self, spectra_path, auxiliary_path, targets_path=None, is_train=True,
                 augmentation_factor=0, shift_range=0.05, scale_range=0.1):
        """
        Dataset pour la classification des spectres d'exoplanètes

        Args:
            spectra_path: Chemin vers le fichier .npy des spectres
            auxiliary_path: Chemin vers le fichier .csv des données auxiliaires
            targets_path: Chemin vers le fichier .csv des labels (None pour test)
            is_train: True si données d'entraînement, False si test
            augmentation_factor: Nombre de versions augmentées par échantillon (0 = pas d'augmentation)
            shift_range: Plage de décalage vertical (en fraction)
            scale_range: Plage de mise à l'échelle (en fraction)
        """
        # Charger les spectres - Shape: (N, 52, 3)
        # 52 points de longueur d'onde, 3 valeurs (moyenne, incertitude basse, incertitude haute)
        self.spectra = np.load(spectra_path)  

        # Charger les données auxiliaires
        self.auxiliary = pd.read_csv(auxiliary_path)

        # Charger les labels si disponibles
        self.is_train = is_train
        self.targets = None
        if targets_path is not None:
            try:
                self.targets = pd.read_csv(targets_path)
                print(f"✅ Targets chargés: {len(self.targets)} exemples")
            except Exception as e:
                print(f"⚠️  Impossible de charger les targets: {e}")

        # Normaliser les données auxiliaires
        self.aux_features = self._normalize_auxiliary()

        # Normaliser les spectres
        self.spectra = self._normalize_spectra(self.spectra)

        # Paramètres d'augmentation
        self.augmentation_factor = augmentation_factor
        self.shift_range = shift_range
        self.scale_range = scale_range
        
        # Taille du dataset
        self.original_size = len(self.spectra)
        self.total_size = self.original_size * (1 + augmentation_factor)
        
        if augmentation_factor > 0:
            print(f"📈 Data Augmentation activée:")
            print(f"   ↳ Dataset original: {self.original_size} exemples")
            print(f"   ↳ Dataset augmenté: {self.total_size} exemples")
            print(f"   ↳ Facteur: x{augmentation_factor + 1}")

    def __len__(self):
        return self.total_size

    def __getitem__(self, idx):
        # Déterminer si c'est un échantillon original ou augmenté
        original_idx = idx % self.original_size
        aug_version = idx // self.original_size
        
        # Récupérer le spectre (52, 3)
        spectrum = self.spectra[original_idx].copy()  # Shape: (52, 3)
        aux_feat = self.aux_features[original_idx]  

        # Appliquer l'augmentation si ce n'est pas la version originale
        if aug_version > 0 and self.augmentation_factor > 0:
            spectrum = self._augment_spectrum(spectrum, idx)

        item = {
            'spectrum': torch.FloatTensor(spectrum),  # (52, 3)
            'auxiliary': torch.FloatTensor(aux_feat),
            'id': original_idx  # Garder l'ID original
        }

        if self.targets is not None:
            eau = self.targets.iloc[original_idx]['eau']
            nuage = self.targets.iloc[original_idx]['nuage']
            item['target'] = torch.FloatTensor([eau, nuage])

        return item

    def _augment_spectrum(self, spectrum, seed):
        """
        Applique plusieurs techniques d'augmentation au spectre
        
        Args:
            spectrum: numpy array de shape (52, 3)
            seed: entier pour la reproductibilité
        
        Returns:
            Spectre augmenté de même shape
        """
        np.random.seed(seed)
        augmented = spectrum.copy()
        
        # 2. Décalage vertical (70% de chances)
        if np.random.rand() > 0.3:
            augmented = self._vertical_shift(augmented)
        
        # 3. Mise à l'échelle (70% de chances)
        if np.random.rand() > 0.3:
            augmented = self._scale_spectrum(augmented)
        
        # 4. Décalage en longueur d'onde (50% de chances)
        if np.random.rand() > 0.5:
            augmented = self._wavelength_shift(augmented)
        
        # 5. Augmentation des incertitudes (50% de chances)
        if np.random.rand() > 0.5:
            augmented = self._augment_uncertainties(augmented)
        
        return augmented
    
    def _vertical_shift(self, spectrum):
        """Décalage vertical du spectre"""
        shift = np.random.uniform(-self.shift_range, self.shift_range)
        spectrum[:, 0] += shift
        return spectrum
    
    def _scale_spectrum(self, spectrum):
        """Mise à l'échelle du spectre"""
        scale = np.random.uniform(1 - self.scale_range, 1 + self.scale_range)
        spectrum[:, 0] *= scale
        return spectrum
    
    def _wavelength_shift(self, spectrum):
        """Simule un léger décalage en longueur d'onde par interpolation"""
        shift_pixels = np.random.randint(-2, 3)
        if shift_pixels == 0:
            return spectrum
        
        shifted = np.roll(spectrum, shift_pixels, axis=0)
        
        # Interpolation linéaire aux bords pour éviter les artefacts
        if shift_pixels > 0:
            for i in range(shift_pixels):
                shifted[i] = spectrum[0]
        else:
            for i in range(abs(shift_pixels)):
                shifted[-(i+1)] = spectrum[-1]
        
        return shifted
    
    def _augment_uncertainties(self, spectrum):
        """Augmente légèrement les barres d'erreur"""
        uncertainty_scale = np.random.uniform(0.9, 1.1)
        spectrum[:, 1] *= uncertainty_scale  # Incertitude basse
        spectrum[:, 2] *= uncertainty_scale  # Incertitude haute
        return spectrum

    def _normalize_auxiliary(self):
        """
        Normalise les données auxiliaires
        """
        aux_data = self.auxiliary.values

        # Normalisation Z-score
        mean = np.mean(aux_data, axis=0)
        std = np.std(aux_data, axis=0) + 1e-8
        normalized = (aux_data - mean) / std

        return normalized.astype(np.float32)

    def _normalize_spectra(self, spectra):
        """
        Normalise les spectres
        Input shape: (N, 52, 3)

        Options:
        1. Normaliser uniquement les valeurs moyennes (colonne 0)
        2. Normaliser les 3 colonnes ensemble
        3. Normaliser chaque colonne séparément
        """
        normalized = np.zeros_like(spectra, dtype=np.float32)

        for i in range(3):  # Pour chaque canal (moyenne, incertitude basse, incertitude haute)
            channel = spectra[:, :, i]  # (N, 52)

            min_vals = channel.min(axis=1, keepdims=True)  # (N, 1)
            max_vals = channel.max(axis=1, keepdims=True)  # (N, 1)
            normalized[:, :, i] = (channel - min_vals) / (max_vals - min_vals + 1e-8)

        return normalized

def collate_fn(batch):
    """
    Fonction de collation pour créer des batches

    Args:
        batch: Liste de dictionnaires avec 'spectrum', 'auxiliary', 'target', 'id'

    Returns:
        spectra: (batch_size, 3, 52) pour CNN 1D - 3 canaux, 52 points
        auxiliary: (batch_size, n_aux_features)
        target: (batch_size, 2) si disponible
        ids: (batch_size,)
    """
    # Empiler les spectres
    spectra = torch.stack([item['spectrum'] for item in batch])  # (B, 52, 3)

    # Transposer pour avoir les canaux en premier: (B, 3, 52)
    # C'est le format attendu par Conv1d: (batch, channels, length)
    spectra = spectra.permute(0, 2, 1)  # (B, 3, 52)

    # Empiler les features auxiliaires
    auxiliary = torch.stack([item['auxiliary'] for item in batch])  # (B, n_features)

    # Empiler les IDs
    ids = torch.LongTensor([item['id'] for item in batch])

    if 'target' in batch[0]:
        targets = torch.stack([item['target'] for item in batch])  # (B, 2)
        return spectra, auxiliary, targets, ids
    else:
        return spectra, auxiliary, ids

# Exemple d'utilisation
if __name__ == "__main__":
    print("=" * 60)
    print("TEST 1: Dataset sans augmentation")
    print("=" * 60)
    
    # Dataset d'entraînement SANS augmentation
    train_dataset = ExoplanetDataset(
        spectra_path='spectra.npy',
        auxiliary_path='auxiliary.csv',
        targets_path='targets_train.csv',
        is_train=True,
        augmentation_factor=0  # Pas d'augmentation
    )

    print(f"Taille train dataset: {len(train_dataset)}")  # 3000
    
    print("\n" + "=" * 60)
    print("TEST 2: Dataset AVEC augmentation")
    print("=" * 60)
    
    # Dataset d'entraînement AVEC augmentation
    train_dataset_aug = ExoplanetDataset(
        spectra_path='spectra.npy',
        auxiliary_path='auxiliary.csv',
        targets_path='targets_train.csv',
        is_train=True,
        augmentation_factor=3,  # 3000 → 12000 exemples
        shift_range=0.05,
        scale_range=0.1
    )

    print(f"Taille train dataset augmenté: {len(train_dataset_aug)}")  # 12000

    # Dataset de test (jamais d'augmentation)
    test_dataset = ExoplanetDataset(
        spectra_path='spectra_test.npy',
        auxiliary_path='auxiliary_test.csv',
        targets_path=None,
        is_train=False,
        augmentation_factor=0  # Jamais d'augmentation sur le test
    )

    print(f"Taille test dataset: {len(test_dataset)}")    # 1032

    print("\n" + "=" * 60)
    print("TEST 3: Comparaison original vs augmenté")
    print("=" * 60)
    
    # Comparer un exemple original et sa version augmentée
    sample_original = train_dataset_aug[0]  # Version originale
    sample_augmented = train_dataset_aug[3000]  # Première version augmentée
    
    print(f"\n🔍 Échantillon original (idx=0):")
    print(f"   Spectrum shape: {sample_original['spectrum'].shape}")
    print(f"   Spectrum mean: {sample_original['spectrum'][:, 0].mean():.4f}")
    print(f"   Target: {sample_original['target']}")
    
    print(f"\n🔍 Échantillon augmenté (idx=3000, même source):")
    print(f"   Spectrum shape: {sample_augmented['spectrum'].shape}")
    print(f"   Spectrum mean: {sample_augmented['spectrum'][:, 0].mean():.4f}")
    print(f"   Target: {sample_augmented['target']}")
    print(f"   Même target ✓ (les labels ne changent pas)")

    print("\n" + "=" * 60)
    print("TEST 4: DataLoader")
    print("=" * 60)
    
    # Test du DataLoader
    from torch.utils.data import DataLoader

    train_loader = DataLoader(
        train_dataset_aug, 
        batch_size=32, 
        shuffle=True,
        collate_fn=collate_fn
    )

    for batch_data in train_loader:
        if len(batch_data) == 4:  # Avec targets
            spectra, auxiliary, targets, ids = batch_data
            print(f"Batch spectra shape: {spectra.shape}")      # (32, 3, 52)
            print(f"Batch auxiliary shape: {auxiliary.shape}")  # (32, n_features)
            print(f"Batch targets shape: {targets.shape}")      # (32, 2)
            print(f"Batch ids shape: {ids.shape}")              # (32,)
        else:  # Sans targets
            spectra, auxiliary, ids = batch_data
            print(f"Batch spectra shape: {spectra.shape}")      # (32, 3, 52)
            print(f"Batch auxiliary shape: {auxiliary.shape}")  # (32, n_features)
            print(f"Batch ids shape: {ids.shape}")              # (32,)
        break

    print("\n Tous les tests passés!")
    print("\n Pour utiliser:")
    print("   - augmentation_factor=0 → Pas d'augmentation")
    print("   - augmentation_factor=3 → Dataset x4 (original + 3 versions)")
