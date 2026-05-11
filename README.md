# ProjetIA

Il faudrait rajouter la prise en compte de utils.py dans l'entrainement avec la set_seed. C'est à dire que les points soient initialiser sur la seed et non de manière random que l'on puisse refaire des entrainements assez facilement. 

Comment fonctionne l'augmentation du dataset : 
https://www.datacamp.com/fr/tutorial/complete-guide-data-augmentation

# Les choses à faire : 

1 - Vérifier que le data leakage n'est plus présent, donc que les métriques d'évaluation ne sont pas supérieur à l'entrainement
2 - Modifier le LR pour trouver la meilleur valeur, le mieux c'est d'en prendre un grand et de déscendre, puis on compare. Modifier le main et faire une boucle sur différente valeur de LR et modifier la manière dont est nommé le fichier à chaque itérations
3 - Diminuer la quantité d'information sauvegardé à chaque fois.
4 - Tester les hyperparamètres du sheduler pour augmenter les perfos => en général, il faut une forme d' 1/exp, sinon ça peut être un escalier ou une gaussienne. Donc changer le OneCycleRl par autre chose
5 - Tester d'autres modèles même si le resnet fonctionne bien pour l'instant, il faudrait voir si il existe pas des modèles plus performant dans le traitement de Csv. 



Méthode importante : 

Mixed Precision Training
Le GradScaler importé mais non utilisé — Le code importe GradScaler et autocast pour la mixed-precision, mais ne les utilise jamais. C'est soit un oubli d'implémentation, soit un import mort.

Les historiques locaux dans train() — train_composite_scores, val_g_means, etc. sont des listes locales qui dupliquent ce qui pourrait être dans TrainingState.history. Il y a deux systèmes d'historique qui coexistent sans raison.


_compute_feature_stats inefficace — La méthode fait une passe complète sur le train loader rien que pour calculer des stats. On pourrait soit le calculer en parallèle du premier epoch d'entraînement, soit utiliser un algorithme de Welford pour éviter les instabilités numériques sur les grands datasets.
Problème 1 — Instabilité numérique de sumsq/n - mean²
L'ancienne approche calcule la variance comme E[X²] - E[X]². Quand les valeurs sont grandes (ex. features spectrales en milliers), les deux termes sont très proches et leur soustraction détruit des chiffres significatifs en float64. L'algorithme de Chan évite complètement cette soustraction en maintenant la somme des carrés des écarts (M2), qui reste numériquement petite.
Problème 2 — Conversion numpy par batch
L'ancienne version faisait .detach().cpu().numpy().astype(np.float64) à chaque batch, soit une allocation mémoire et une copie par itération. La nouvelle accumule directement en torch.float64 sur CPU et n'appelle .tolist() qu'une seule fois à la fin.
L'algorithme de Chan en résumé : au lieu d'accumuler des sommes brutes, on fusionne deux groupes (n_a, mean_a, M2_a) + (n_b, mean_b, M2_b) en une seule formule close, ce qui est mathématiquement équivalent à tout recalculer sur la totalité des données, sans jamais avoir à les charger en mémoire d'un coup.











Recommandation : ResNet1D déjà implémenté est le bon choix
Le models/ResNetCNN.py que vous avez est architecturalement bien adapté. Voici pourquoi et comment l'utiliser au mieux.

Pourquoi ResNet1D-CNN est optimal ici
Critère	Justification
Signal spectral 1D	Les Conv1D captent les raies d'absorption locales (eau ~1.4, 1.9, 2.7 µm)
3000 échantillons	Trop peu pour un Transformer ; les CNN généralisent mieux
52 points de spectre	Séquence courte — pas besoin d'un LSTM/GRU
Connexions résiduelles	Evitent la disparition du gradient avec des features subtiles
Fusion auxiliaire	Les paramètres physiques (T_eq, flux stellaire) sont prédictifs
Quelle variante choisir
Avec 52 points, le stem fait ÷4 (stride=2 + maxpool stride=2), puis les layers suivants font encore ÷2 chacun. Voici ce que ça donne :

Layer	Taille spatiale restante
Entrée	52
Après stem	13
Après layer2	7
Après layer3	4
Après layer4	2
→ resnet34_1d est trop profond pour 52 points (surcouche sur une dimension spatiale de 2).
→ resnet18_1d est le bon compromis pour 3000 échantillons.

Configuration recommandée

from models.ResNetCNN import resnet18_1d

model = resnet18_1d(
    spectrum_length=52,
    input_channels=2,    # transit_depth + incertitude (λ est constant → inutile)
    auxiliary_dim=17,    # avec _engineer_features() : 5 raw + 12 engineered
    num_classes=2,
    base_channels=32,
    dropout=0.3,
)
Pourquoi input_channels=2 ? La colonne wavelength spectra[:,:,0] est identique pour tous les échantillons — ce n'est pas une feature, c'est l'axe x. Seuls transit_depth et uncertainty portent l'information discriminante.

Pourquoi auxiliary_dim=17 ? Le models/dataset.py:45-114 a déjà une fonction _engineer_features() qui ajoute des features physiquement motivées (flux stellaire, température d'équilibre, zone habitable, logs) — il faut l'activer.

Loss et entraînement

# Multi-label binaire avec classes équilibrées → BCEWithLogitsLoss sans poids
criterion = nn.BCEWithLogitsLoss()

# Adam avec weight decay pour la régularisation
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

# Scheduler cosine pour éviter les oscillations en fin d'entraînement
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
Alternatives si les performances sont insuffisantes
SE-ResNet1D (Squeeze-and-Excitation) : ajouter une attention par canal pour pondérer les longueurs d'onde importantes — gain typique de 1-2% accuracy sur données spectrales
Weighted pooling par l'incertitude : remplacer le GlobalAvgPool par une moyenne pondérée par 1/σ² — exploite physiquement les barres d'erreur
Ensemble ResNet8 + ResNet18 : deux modèles légers en ensemble sont souvent plus robustes qu'un seul modèle profond sur peu de données
Le modèle actuel dans models/ResNetCNN.py est déjà le bon choix — l'essentiel est d'activer _engineer_features() dans le dataset et d'utiliser input_channels=2.


## Alternatives si les performances sont insuffisantes
SE-ResNet1D (Squeeze-and-Excitation) : ajouter une attention par canal pour pondérer les longueurs d'onde importantes — gain typique de 1-2% accuracy sur données spectrales
Weighted pooling par l'incertitude : remplacer le GlobalAvgPool par une moyenne pondérée par 1/σ² — exploite physiquement les barres d'erreur
Ensemble ResNet8 + ResNet18 : deux modèles légers en ensemble sont souvent plus robustes qu'un seul modèle profond sur peu de données
Le modèle actuel dans models/ResNetCNN.py est déjà le bon choix — l'essentiel est d'activer _engineer_features() dans le dataset et d'utiliser input_channels=2.









Diagnostic du système actuel
Architecture : EnsembleResNet = ResNet8 + ResNet18 fusionnés, 17 features auxiliaires ingéniérées, spectre (52, 3).

Problème critique identifié — l'augmentation est désactivée :

Dans training/config.py : augmentation_factor: int = 0

Dans models/dataset.py : l'augmentation n'est appliquée que si aug_version > 0, ce qui ne se produit jamais quand augmentation_factor=0. Aucune augmentation n'est effectuée.

Pistes d'amélioration par ordre d'impact
1. Activer l'augmentation (gain probable : +0.005 à +0.015 sur MCC)

augmentation_factor: int = 2   # triplement du train set
flip_prob: float = 0.3         # spectres retournés spectralement
noise_std: float = 0.015       # bruit physiquement réaliste
2. Downsampling trop agressif sur 52 points
Dans models/ResNetCNN.py:96-101, le stem applique stride=2 + MaxPool1d(stride=2) = facteur 4 avant même le premier bloc, laissant seulement ~13 points. Avec les 3 couches suivantes à stride=2, il ne reste que 1-2 points à la fin.

Suggestion : réduire le stem pour une séquence de 52 points :


self.stem = nn.Sequential(
    nn.Conv1d(input_channels, base_channels,
              kernel_size=5, stride=1, padding=2, bias=False),  # pas de stride
    nn.BatchNorm1d(base_channels),
    nn.ReLU(inplace=True),
    nn.MaxPool1d(kernel_size=3, stride=2, padding=1),  # ×2 seulement
)
3. Intégrer SEResNet dans l'ensemble
Le models/SEResNet1D.py est déjà implémenté mais non utilisé. L'attention par canal (SE) est particulièrement adaptée aux spectres de 52 points car les canaux mean/σ_low/σ_high ont des importances différentes selon la classe.

Remplacer l'EnsembleResNet par un ensemble SEResNet8 + ResNet18 (ou SEResNet8 + SEResNet18) dans main.py:171.

4. Ajouter des canaux dérivés (SNR)
Les 3 canaux (mean, low_unc, high_unc) contiennent implicitement le rapport signal/bruit. Ajouter dans _normalize_spectra ou collate_fn :

Canal 4 : mean / (0.5 * (high_unc - low_unc) + ε) → SNR local
Canal 5 : asymétrie d'incertitude (high_unc - low_unc) / (high_unc + low_unc + ε)
Cela passe input_channels de 3 à 5 sans modifier l'architecture.

5. Cross-validation avec ensemble de modèles
Avec 3000 samples, la variance entre runs est élevée (MCC actuel vs best = -0.0099). Entraîner 5 modèles sur des folds différents et moyenner les probabilités de sortie réduit cette variance.

6. Optimisation du seuil de décision par classe
"eau" et "nuage" sont deux problèmes binaires indépendants avec potentiellement des distributions déséquilibrées différentes. Utiliser un seuil optimal par classe (via MCC-sweep sur validation) plutôt qu'un seuil global 0.5.

7. Label smoothing dans la loss

# BCE avec label smoothing ε=0.05
target_smooth = target * (1 - ε) + ε / 2
Réduit la surconfiance et améliore la calibration, ce qui affecte positivement le G-Mean et la Stability Score.

Récapitulatif priorité/effort
Piste	Impact estimé	Effort
Activer augmentation_factor=2	Élevé	Trivial
Réduire downsampling stem	Moyen-élevé	Faible
Intégrer SEResNet dans ensemble	Moyen	Faible
Canaux SNR dérivés	Moyen	Faible
K-fold ensemble	Élevé	Moyen
Seuil par classe	Faible-moyen	Faible
Le point 1 est le correctif le plus immédiat car l'augmentation est configurée mais jamais exécutée.


















