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
