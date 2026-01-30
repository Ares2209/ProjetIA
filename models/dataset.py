import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd

class ExoplanetDataset(Dataset):
    def __init__(self, spectra_path, auxiliary_path, targets_path=None, is_train=True):
        """
        Dataset pour la classification des spectres d'exoplanètes

        Args:
            spectra_path: Chemin vers le fichier .npy des spectres
            auxiliary_path: Chemin vers le fichier .csv des données auxiliaires
            targets_path: Chemin vers le fichier .csv des labels (None pour test)
            is_train: True si données d'entraînement, False si test
        """
        # Charger les spectres - Shape: (N, 52, 3)
        # 52 points de longueur d'onde, 3 valeurs (moyenne, incertitude basse, incertitude haute)
        self.spectra = np.load(spectra_path)  

        # Charger les données auxiliaires
        self.auxiliary = pd.read_csv(auxiliary_path)

        # Charger les labels si disponibles
        self.is_train = is_train
        self.targets = None
        if targets_path is not None:  # ← Correction: ne pas vérifier is_train
            try:
                self.targets = pd.read_csv(targets_path)
                print(f"✅ Targets chargés: {len(self.targets)} exemples")
            except Exception as e:
                print(f"⚠️  Impossible de charger les targets: {e}")

        # Normaliser les données auxiliaires
        self.aux_features = self._normalize_auxiliary()

        # Normaliser les spectres
        self.spectra = self._normalize_spectra(self.spectra)

    def __len__(self):
        return len(self.spectra)

    def __getitem__(self, idx):
        # Récupérer le spectre (52, 3)
        spectrum = self.spectra[idx]  # Shape: (52, 3)
        aux_feat = self.aux_features[idx]  

        item = {
            'spectrum': torch.FloatTensor(spectrum),  # (52, 3)
            'auxiliary': torch.FloatTensor(aux_feat),
            'id': idx
        }

        # ← Correction: utiliser 'target' au lieu de 'labels'
        # et vérifier seulement si self.targets existe
        if self.targets is not None:
            eau = self.targets.iloc[idx]['eau']
            nuage = self.targets.iloc[idx]['nuage']
            item['target'] = torch.FloatTensor([eau, nuage])  # ← Changé de 'labels' à 'target'

        return item

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

    # ← Correction: vérifier 'target' au lieu de 'labels'
    if 'target' in batch[0]:
        targets = torch.stack([item['target'] for item in batch])  # (B, 2)
        return spectra, auxiliary, targets, ids
    else:
        return spectra, auxiliary, ids

# Exemple d'utilisation
if __name__ == "__main__":
    # Dataset d'entraînement
    train_dataset = ExoplanetDataset(
        spectra_path='spectra.npy',
        auxiliary_path='auxiliary.csv',
        targets_path='targets_train.csv',
        is_train=True
    )

    # Dataset de test
    test_dataset = ExoplanetDataset(
        spectra_path='spectra_test.npy',
        auxiliary_path='auxiliary_test.csv',
        targets_path=None,
        is_train=False
    )

    print(f"Taille train dataset: {len(train_dataset)}")  # 3000
    print(f"Taille test dataset: {len(test_dataset)}")    # 1032

    # Test d'un exemple
    sample = train_dataset[0]
    print(f"Spectrum shape: {sample['spectrum'].shape}")      # (52, 3)
    print(f"Auxiliary shape: {sample['auxiliary'].shape}")    # (n_features,)
    if 'target' in sample:
        print(f"Target: {sample['target']}")                  # [eau, nuage]

    # Test du DataLoader
    from torch.utils.data import DataLoader

    train_loader = DataLoader(
        train_dataset, 
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
