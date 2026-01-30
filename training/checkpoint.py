
"""Module de gestion des checkpoints."""

import torch
from pathlib import Path
from typing import Optional, Dict, Any
import glob
import os
from datetime import datetime


class CheckpointManager:
    """Gestionnaire de checkpoints avec sauvegarde automatique."""
    
    def __init__(self, 
                 model_folder: str = 'checkpoints',
                 model_basename: str = 'visibility_model_',
                 keep_last_n: int = 3):
        """
        Args:
            model_folder: Dossier de sauvegarde des checkpoints
            model_basename: Préfixe des fichiers de checkpoint
            keep_last_n: Nombre de checkpoints récents à conserver
        """
        self.model_folder = Path(model_folder)
        self.model_basename = model_basename
        self.keep_last_n = keep_last_n
        
        # Créer le dossier si nécessaire
        self.model_folder.mkdir(parents=True, exist_ok=True)
        
        # Meilleur score pour déterminer si un checkpoint est "best"
        self.best_score = float('-inf')
        
        print(f" CheckpointManager initialisé:")
        print(f"   • Dossier: {self.model_folder}")
        print(f"   • Basename: {model_basename}")
        print(f"   • Conserver: {keep_last_n} derniers checkpoints")
    
    def save_checkpoint(self,
                       model: torch.nn.Module,
                       optimizer: torch.optim.Optimizer,
                       scheduler: Optional[Any],
                       epoch: int,
                       metrics: Optional[Dict[str, float]] = None,
                       history: Optional[Dict[str, list]] = None,
                       is_best: bool = False,
                       **extra_fields):
        """
        Sauvegarde un checkpoint.
        
        Args:
            model: Modèle à sauvegarder
            optimizer: Optimizer
            scheduler: Learning rate scheduler
            epoch: Numéro d'epoch
            metrics: Métriques de validation
            history: Historique d'entraînement
            is_best: Si True, sauvegarde aussi comme 'best'
        """
        checkpoint = {
            'epoch': epoch,
            # Support DataParallel/DistributedDataParallel
            'model_state_dict': (model.module.state_dict() if isinstance(model, (torch.nn.DataParallel, getattr(torch.nn, 'parallel', torch.nn).DistributedDataParallel)) else model.state_dict()),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'metrics': metrics or {},
            'history': history or {},
            'timestamp': datetime.now().isoformat()
        }

        # Intégrer champs supplémentaires passés (ex: global_step, best_val_f1)
        if extra_fields:
            checkpoint.update(extra_fields)
        
        # Sauvegarder le checkpoint de l'epoch
        epoch_path = self.model_folder / f"{self.model_basename}{epoch:02d}.pth"
        torch.save(checkpoint, epoch_path)
        
        # Si c'est le meilleur, sauvegarder aussi comme 'best'
        if is_best:
            best_path = self.model_folder / f"{self.model_basename}best.pth"
            torch.save(checkpoint, best_path)
            # Déduire le meilleur score si possible
            self.best_score = (metrics or {}).get('val_f1', extra_fields.get('best_val_f1', float('-inf')))
        
        # Nettoyer les anciens checkpoints
        self._cleanup_old_checkpoints()
    
    def _cleanup_old_checkpoints(self):
        """Supprime les anciens checkpoints pour économiser l'espace."""
        if self.keep_last_n <= 0:
            return

        # Trouver tous les checkpoints (sauf 'best')
        pattern = str(self.model_folder / f"{self.model_basename}*.pth")
        checkpoints = [
            f for f in glob.glob(pattern) 
            if 'best' not in os.path.basename(f)
        ]

        # Trier par date de modification
        checkpoints.sort(key=os.path.getmtime)

        # Supprimer les plus anciens si nécessaire
        if len(checkpoints) > self.keep_last_n:  # ← CORRECTION ICI
            to_delete = checkpoints[:-self.keep_last_n]
            
            for checkpoint_path in to_delete:
                try:
                    os.remove(checkpoint_path)
                except Exception as e:
                    print(f"   Erreur suppression {checkpoint_path}: {e}")

    
    def load_checkpoint(self, 
                       checkpoint_name: str = 'best',
                       device: str = 'cpu') -> Dict[str, Any]:
        """
        Charge un checkpoint.
        
        Args:
            checkpoint_name: Nom du checkpoint ('best', '01', '02', etc.)
            device: Device sur lequel charger
            
        Returns:
            Dictionnaire contenant le checkpoint
        """
        if checkpoint_name == 'latest':
            checkpoint_path = self.get_latest_checkpoint()
            if checkpoint_path is None:
                raise FileNotFoundError("Aucun checkpoint trouvé")
        else:
            checkpoint_path = self.model_folder / f"{self.model_basename}{checkpoint_name}.pth"
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint introuvable: {checkpoint_path}")
    
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        metrics = checkpoint.get('metrics', {}) or {}
        val_f1 = metrics.get('val_f1', None)
        
        return checkpoint
    
    def get_latest_checkpoint(self) -> Optional[Path]:
        """Retourne le chemin du dernier checkpoint (sauf 'best')."""
        pattern = str(self.model_folder / f"{self.model_basename}*.pth")
        checkpoints = [
            f for f in glob.glob(pattern)
            if 'best' not in os.path.basename(f)
        ]
        
        if not checkpoints:
            return None
        
        # Retourner le plus récent
        latest = max(checkpoints, key=os.path.getmtime)
        return Path(latest)
    
    def get_best_checkpoint_path(self) -> Path:
        """Retourne le chemin du meilleur checkpoint."""
        return self.model_folder / f"{self.model_basename}best.pth"
    
    def checkpoint_exists(self, checkpoint_name: str = 'best') -> bool:
        """Vérifie si un checkpoint existe."""
        checkpoint_path = self.model_folder / f"{self.model_basename}{checkpoint_name}.pth"
        return checkpoint_path.exists()


def load_model_from_checkpoint(model: torch.nn.Module,
                               checkpoint_path: str,
                               device: str = 'cpu',
                               load_optimizer: bool = False,
                               optimizer: Optional[torch.optim.Optimizer] = None,
                               scheduler: Optional[Any] = None) -> tuple:
    """
    Charge un modèle depuis un checkpoint (fonction utilitaire).
    
    Args:
        model: Modèle à charger
        checkpoint_path: Chemin vers le checkpoint
        device: Device
        load_optimizer: Si True, charge aussi l'optimizer
        optimizer: Optimizer (requis si load_optimizer=True)
        scheduler: Scheduler (optionnel)
        
    Returns:
        (model, history)
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    print(f"✓ Modèle chargé: {checkpoint_path}")
    print(f"  • Epoch: {checkpoint['epoch']}")
    metrics = checkpoint.get('metrics', {}) or {}
    val_f1 = metrics.get('val_f1', None)
    
    if load_optimizer and optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if scheduler is not None and checkpoint.get('scheduler_state_dict'):
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        print(f"  • Optimizer/Scheduler chargés")
    
    return model, checkpoint.get('history', None)
