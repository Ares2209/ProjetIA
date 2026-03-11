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