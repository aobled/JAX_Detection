# 📉 Stratégie (Pour le Futur) : Pruning & Quantization

Ce document trace les lignes directrices et les concepts fondamentaux pour l'allègement des réseaux de neurones (Compression de modèle). Bien que l'architecture actuelle génère des modèles JAX purs très capables, leur déploiement sur des systèmes aux ressources limitées (embarqué, temps réel) nécessitera à terme ces optimisations.

---

## 1. Le Pruning (Élagage)

Le *Pruning* consiste à supprimer purement et simplement les connexions (les poids neuronaux) qui sont considérées comme non pertinentes pour la prédiction finale du modèle.

*   **Le Principe (Magnitude Pruning) :** Au lieu d'avoir une suppression "aléatoire", on scrute les matrices de poids du modèle. Tous les paramètres qui sont extrêmement proches de `0` sont forcés à la valeur exacte de `0`.
*   **Sparsity (Clairsemance) :** On obtient un réseau "clairsemé". Un modèle pruné de façon agressive peut voir 70% à 90% de ses connexions détruites tout en conservant d'excellentes capacités d'inférence.
*   **Le Piège Matériel :** Le Pruning classique ne réduit pas la taille du fichier du modèle statique ni son temps de calcul *à moins* que le compilateur d'inférence ou la puce (ex: NPU, Tensor Cores) soient capables de traiter des **Sparsity Matrices**. Dans ce cas précis, le processeur passe intentionnellement les `0` au lieu de réaliser l'opération multipliée par zéro, offrant des accélérations faramineuses (ex: x2 à x4 en temps réel).

## 2. La Quantization (Quantification)

Là où le Pruning enlève des poids, la *Quantization* cherche plutôt à réduire la complexité de comment ces poids sont calculés et mémorisés (la virgule).

*   **Le Principe :** Actuellement, nos modèles Flax s'appuient sur des calculs flottants complexes 32-bits ou 16-bits (`Float32`, `Float16`). La quantification a pour but de contraindre ces valeurs décimales dans une échelle de **nombres entiers 8 bits (`Int8`)** allant de -128 à 127.
*   **Fonctionnement :** On identifie les bornes Min et Max des poids d'une couche, et l'on "compresse" cette infinité de décimales vers l'échelle des entiers en conservant un unique "facteur d'échelle".
*   **Les Gains :** Le modèle divise magiquement son poids en RAM/Stockage par 4 (passant de `Float32` à `Int8`). Parallèlement, l'UGA/CPU additionne des entiers incroyablement plus vite qu'il ne traite l'arithmétique à virgule flottante, garantissant des gains de performance massifs lors du déploiement.

---

## 3. Comment les intégrer à l'architecture JAX ? Les deux écoles.

### A. L'école du PTQ (Post-Training Quantization) : *L'étape de fin de Pipeline*
C'est la méthode la plus conventionnelle. L'entraînement est terminé, le `Trainer` a sauvegardé le `best_model.pkl` en `Float32`. On instancie à la suite un nouveau module "d'export final".
*   **Comment ça marche ?** Un algorithme froid ouvre le modèle et arrondit statistiquement les poids vers du `Int8`.
*   **Pour & Contre :** Extrêmement simple à mettre en place "sans toucher à l'entraînement". En revanche, le modèle n'a jamais été prévu pour fonctionner ainsi, cela engendre souvent une légère détérioration de l'*Accuracy*, particulièrement sur de la Détection fine.

### B. L'école du QAT (Quantization-Aware Training) : *Modification du Trainer*
C'est l'approche "moderne" pour l'ultra-compétitivité. On ne le fait pas à la toute fin, mais à la fin de notre boucle d'Epochs.
*   **Comment ça marche ?** Durant les 5 ou 10 derniers Epochs appelés dans `trainer.py`, le réseau *"simule"* mathématiquement la future quantification (en arrondissant drastiquement ses prédictions JAX via des *Fake-Quantize Nodes*). Le réseau continue d'apprendre et utilise ses optimiseurs pour corriger le tir et s'adapter contre cet handicap.
*   **Pour & Contre :** Une fois exporté "pour de vrai", le modèle conserve des performances optimales (Accuracy identique). Complexe à mettre en œuvre (nécessite une modification de la stratégie du `Trainer` ou l'adoption de modules spécialisés dans la définition Flax).

---

## 4. Les Librairies recommandées pour l'écosystème actuel

Si l'on devait aborder ce chapitre sur notre architecture :
1.  **AQT (Accurate Quantized Training)** : Librairie poussée par Google pour injecter le QAT le plus proprement possible dans un environnement Flax/JAX JIT, idéalement intégrable dans notre couche `model_library.py`.
2.  **L'export JAX2TF (La voie Royale pour le PTQ)** : Au lieu d'essayer de quantifier dans le vacuum de JAX, la solution standard pour un déploiement reste d'utiliser la passerelle native `jax2tf` pour convertir notre Export PKL en `TensorFlow Lite`. L'écosystème TFLite intègre déjà des routines infaillibles appelées `$tflite_convert` pour élaguer et quantifier le modèle de manière industrielle (et ce, d'une seule ligne de code !).
