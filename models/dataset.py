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
        # Charger les spectres
        self.spectra = np.load(spectra_path)  # Shape: (N, spectrum_length)
        
        # Charger les données auxiliaires
        self.auxiliary = pd.read_csv(auxiliary_path)
        
        # Charger les labels si disponibles
        self.is_train = is_train
        if is_train and targets_path is not None:
            self.targets = pd.read_csv(targets_path)
        else:
            self.targets = None
        
        # Normaliser les données auxiliaires
        self.aux_features = self._normalize_auxiliary()
        
        # Optionnel : normaliser les spectres
        self.spectra = self._normalize_spectra(self.spectra)
    
    def __len__(self):
        return len(self.spectra)
    
    def __getitem__(self, idx):
        # Récupérer le spectre
        spectrum = self.spectra[idx] 
        aux_feat = self.aux_features[idx]  
        item = {
            'spectrum': torch.FloatTensor(spectrum),
            'auxiliary': torch.FloatTensor(aux_feat),
            'id': idx
        }
        
        if self.is_train and self.targets is not None:
            eau = self.targets.iloc[idx]['eau']
            nuage = self.targets.iloc[idx]['nuage']
            item['labels'] = torch.FloatTensor([eau, nuage])
        
        return item
    
    def _normalize_auxiliary(self):
        """
        Normalise les données auxiliaires (masse étoile, rayon étoile, 
        température étoile, masse planète, demi-grand axe)
        """
        # Colonnes attendues (adapter selon vos noms de colonnes réels)
        # Par exemple: ['stellar_mass', 'stellar_radius', 'stellar_temp', 
        #               'planet_mass', 'semi_major_axis']
        
        aux_data = self.auxiliary.values
        
        # Normalisation Z-score
        mean = np.mean(aux_data, axis=0)
        std = np.std(aux_data, axis=0) + 1e-8
        normalized = (aux_data - mean) / std
        
        return normalized.astype(np.float32)
    
    def _normalize_spectra(self, spectra):
        """
        Normalise les spectres
        """
        # Option 1: Min-Max normalization par spectre
        min_vals = spectra.min(axis=1, keepdims=True)
        max_vals = spectra.max(axis=1, keepdims=True)
        normalized = (spectra - min_vals) / (max_vals - min_vals + 1e-8)
        
        # Option 2: Z-score normalization (décommenter si préféré)
        # mean = spectra.mean(axis=1, keepdims=True)
        # std = spectra.std(axis=1, keepdims=True) + 1e-8
        # normalized = (spectra - mean) / std
        
        return normalized.astype(np.float32)


def collate_fn(batch):
    """
    Fonction de collation pour créer des batches
    
    Args:
        batch: Liste de dictionnaires avec 'spectrum', 'auxiliary', 'labels', 'id'
    
    Returns:
        spectra: (batch_size, spectrum_length) ou (batch_size, 1, spectrum_length) pour CNN
        auxiliary: (batch_size, 5)
        labels: (batch_size, 2) si disponible
        ids: (batch_size,)
    """
    batch_size = len(batch)
    
    # Empiler les spectres
    spectra = torch.stack([item['spectrum'] for item in batch])  # (B, L)
    
    # Ajouter une dimension channel pour CNN 1D: (B, 1, L)
    spectra = spectra.unsqueeze(1)
    
    # Empiler les features auxiliaires
    auxiliary = torch.stack([item['auxiliary'] for item in batch])  # (B, 5)
    
    # Empiler les IDs
    ids = torch.LongTensor([item['id'] for item in batch])
    
    # Empiler les labels si disponibles
    if 'labels' in batch[0]:
        labels = torch.stack([item['labels'] for item in batch])  # (B, 2)
        return spectra, auxiliary, labels, ids
    else:
        return spectra, auxiliary, ids