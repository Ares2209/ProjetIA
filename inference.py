"""Script d'inférence pour le projet Exoplanet.

Charge un checkpoint (par défaut : `checkpoints/exoplanet_modelbest.pth`) et
prédit la sortie pour un exemple choisi dans les fichiers de données.

Usage:
    python inference.py --checkpoint checkpoints/exoplanet_modelbest.pth --index 0
    
ex : 
python inference.py   --checkpoint checkpoints/exoplanet_modelbest.pth   --spectra Défi-IA-2026/DATA/defi-ia-cnes/spectra.npy   --auxiliary Défi-IA-2026/DATA/defi-ia-cnes/auxiliary.csv   --targets Défi-IA-2026/DATA/defi-ia-cnes/targets.csv   --index 54   --verbose

Options:
    --checkpoint : chemin vers le fichier .pth
    --spectra    : chemin vers le fichier .npy de spectres
    --auxiliary  : chemin vers le fichier .csv auxiliaire
    --targets    : chemin vers le fichier .csv des targets (optionnel, pour comparaison)
    --index      : index de l'exemple à prédire (défaut 0)
    --device     : cpu ou cuda
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

# Importer les classes de modèles et le dataset
from models.CNN import CNN
from models.ResNetCNN import ResNet1D
from models.dataset import ExoplanetDataset, collate_fn

def remove_module_prefix(state_dict: dict) -> dict:
    """Retire le préfixe 'module.' si présent (sauvegarde DataParallel)."""
    new_state = {}
    for k, v in state_dict.items():
        new_key = k
        if k.startswith('module.'):
            new_key = k[len('module.'):]
        new_state[new_key] = v
    return new_state

def detect_model_class(state_dict: dict) -> str:
    """Detecte la famille de modèle à partir des clés du state dict."""
    keys = list(state_dict.keys())
    if any(k.startswith('stem.0.') or k.startswith('stem.0') for k in keys):
        return 'resnet'
    if any(k.startswith('conv_layers.0.0.') or k.startswith('conv_layers.0.0') for k in keys):
        return 'cnn'
    # fallback
    return 'cnn'

def load_checkpoint(checkpoint_path: str, device: str = 'cpu') -> dict:
    """Charge un checkpoint PyTorch de manière compatible."""
    try:
        # Tentative avec weights_only=True (mode sécurisé)
        import torch.serialization
        # Utiliser numpy._core au lieu de numpy.core (nouvelle API)
        try:
            from numpy._core.multiarray import scalar as np_scalar
        except ImportError:
            from numpy.core.multiarray import scalar as np_scalar

        torch.serialization.add_safe_globals([np_scalar])
        checkpoint = torch.load(checkpoint_path, map_location=device)
    except Exception as e:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    return checkpoint

def main():
    parser = argparse.ArgumentParser(description='Inference script')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/exoplanet_modelbest.pth')
    parser.add_argument('--spectra', type=str, default='Défi-IA-2026/DATA/defi-ia-cnes/spectra_test.npy')
    parser.add_argument('--auxiliary', type=str, default='Défi-IA-2026/DATA/defi-ia-cnes/auxiliary_test.csv')
    parser.add_argument('--targets', type=str, default=None, help='Fichier targets.csv pour comparaison (optionnel)')
    parser.add_argument('--index', type=int, default=0)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--save', type=str, default=None, help='Fichier JSON de sortie (optionnel)')
    parser.add_argument('--verbose', action='store_true', help='Affichage détaillé')

    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith('cuda') else 'cpu')

    print('\n' + '='*60)
    print('🔬 SCRIPT D\'INFÉRENCE - EXOPLANET')
    print('='*60)
    print(f"📁 Checkpoint: {args.checkpoint}")
    print(f"💻 Device: {device}")
    print(f"🎯 Index: {args.index}")
    print(f"📊 Seuil: {args.threshold}")
    print('='*60)

    # Vérifications des fichiers
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint introuvable: {ckpt_path}")

    if not Path(args.spectra).exists():
        raise FileNotFoundError(f"Fichier spectra introuvable: {args.spectra}")
    if not Path(args.auxiliary).exists():
        raise FileNotFoundError(f"Fichier auxiliary introuvable: {args.auxiliary}")

    # Vérifier si on a un fichier targets pour la comparaison
    has_targets = args.targets is not None and Path(args.targets).exists()

    # Charger les données (on utilisera ExoplanetDataset pour les mêmes normalisations)
    print("\n📥 Chargement des données...")
    dataset = ExoplanetDataset(
        spectra_path=args.spectra,
        auxiliary_path=args.auxiliary,
        targets_path=args.targets if has_targets else None,
        is_train=False
    )

    if args.index < 0 or args.index >= len(dataset):
        raise IndexError(f"Index hors limites: {args.index} (0..{len(dataset)-1})")

    sample = dataset[args.index]

    # Préparer un batch de taille 1
    spectrum = sample['spectrum'].unsqueeze(0)  # (1, 52, 3)
    # Transposer vers (batch, channels, length)
    spectrum = spectrum.permute(0, 2, 1).to(device)
    auxiliary = sample['auxiliary'].unsqueeze(0).to(device)

    # Récupérer les vraies valeurs si disponibles
    true_targets = None
    if has_targets and 'target' in sample:
        true_targets = sample['target'].cpu().numpy()  # [eau, nuage]

    checkpoint = load_checkpoint(str(ckpt_path), device=str(device))

    state_dict = checkpoint.get('model_state_dict', None)
    if state_dict is None:
        # Peut-être le checkpoint est le state_dict directement
        state_dict = checkpoint

    # Détecter type de modèle
    state_dict = remove_module_prefix(state_dict)
    model_type = detect_model_class(state_dict)
    print(f"   • Type de modèle détecté: {model_type.upper()}")

    # Inférer paramètres à partir des données
    _, channels, length = spectrum.shape  # (B, C, L)
    auxiliary_dim = auxiliary.shape[1]

    if model_type == 'ResNet':
        model = ResNet1D(
            spectrum_length=length,
            auxiliary_dim=auxiliary_dim,
            num_classes=2,
            augmentation_factor=10, 
            shift_range=0.05,
            scale_range=0.1
        )
    else:
        # CNN classique
        model = CNN(
            spectrum_length=length,
            auxiliary_dim=auxiliary_dim,
            num_classes=2,
            input_channels=channels
        )

    # Charger les poids
    try:
        model.load_state_dict(state_dict)
    except RuntimeError:
        # Tentative moins stricte: some keys may differ (ex: ancien modèle) → load partiel
        model.load_state_dict(state_dict, strict=False)

    model = model.to(device)
    model.eval()

    # Compter les paramètres
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    with torch.no_grad():
        logits = model(spectrum, auxiliary)  # (1, num_classes)
        probs = torch.sigmoid(logits).cpu().numpy().squeeze(0)

    preds = [1 if p >= args.threshold else 0 for p in probs]

    result = {
        'checkpoint': str(ckpt_path),
        'model_type': model_type,
        'index': args.index,
        'threshold': args.threshold,
        'logits': {
            'eau': float(logits[0, 0].cpu().item()),
            'nuage': float(logits[0, 1].cpu().item())
        },
        'probabilities': {
            'eau': float(probs[0]),
            'nuage': float(probs[1])
        },
        'predictions': {
            'eau': bool(preds[0]),
            'nuage': bool(preds[1])
        }
    }

    # Ajouter la comparaison avec la vérité terrain si disponible
    if true_targets is not None:
        true_eau = int(true_targets[0])
        true_nuage = int(true_targets[1])

        result['ground_truth'] = {
            'eau': bool(true_eau),
            'nuage': bool(true_nuage)
        }

        result['evaluation'] = {
            'eau': {
                'correct': bool(preds[0] == true_eau),
                'status': ' CORRECT' if preds[0] == true_eau else '❌ INCORRECT'
            },
            'nuage': {
                'correct': bool(preds[1] == true_nuage),
                'status': ' CORRECT' if preds[1] == true_nuage else '❌ INCORRECT'
            },
            'both_correct': bool(preds[0] == true_eau and preds[1] == true_nuage)
        }

    print('\n' + '='*60)
    print(' RÉSULTAT D\'INFÉRENCE')
    print('='*60)
    print(json.dumps(result, indent=2, ensure_ascii=False))

    # Affichage visuel amélioré
    print('\n' + '='*60)
    print(' PRÉDICTIONS DÉTAILLÉES')
    print('='*60)
    
    print(f"\n EAU:")
    print(f"   Logit:        {result['logits']['eau']:+.4f}")
    print(f"   Probabilité:  {result['probabilities']['eau']:.6f} ({result['probabilities']['eau']*100:.4f}%)")
    print(f"   Prédiction:   {' PRÉSENTE' if preds[0] else '❌ ABSENTE'}")
    
    print(f"\n  NUAGES:")
    print(f"   Logit:        {result['logits']['nuage']:+.4f}")
    print(f"   Probabilité:  {result['probabilities']['nuage']:.6f} ({result['probabilities']['nuage']*100:.4f}%)")
    print(f"   Prédiction:   {' PRÉSENTS' if preds[1] else '❌ ABSENTS'}")

    if true_targets is not None:
        print('\n' + '='*60)
        print(' COMPARAISON PRÉDICTIONS vs VÉRITÉ TERRAIN')
        print('='*60)

        print(f"\n EAU:")
        print(f"   Prédiction: {preds[0]} (probabilité: {probs[0]:.6f})")
        print(f"   Vérité:     {true_eau}")
        print(f"   Résultat:   {result['evaluation']['eau']['status']}")

        print(f"\n  NUAGES:")
        print(f"   Prédiction: {preds[1]} (probabilité: {probs[1]:.6f})")
        print(f"   Vérité:     {true_nuage}")
        print(f"   Résultat:   {result['evaluation']['nuage']['status']}")

        print(f"\n{'='*60}")
        if result['evaluation']['both_correct']:
            print(" SUCCÈS TOTAL : Les deux prédictions sont correctes!")
        else:
            correct_count = sum([result['evaluation']['eau']['correct'], 
                               result['evaluation']['nuage']['correct']])
            print(f" SUCCÈS PARTIEL : {correct_count}/2 prédictions correctes")
        print('='*60)
    else:
        print('\n' + '='*60)
        print('ℹ  Pas de vérité terrain disponible pour comparaison')
        print('='*60)

    if args.save:
        with open(args.save, 'w') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"\n Résultat sauvegardé dans {args.save}")

if __name__ == '__main__':
    main()