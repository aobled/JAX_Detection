# Stratégie de Chunking et Préparation des Données 📦

Dans notre architecture JAX/Flax, le processus d'entraînement a été drastiquement optimisé en adoptant une approche **"Data Engineering First"**. Nous séparons strictement la préparation des images (Resizing, Normalisation, Bundling) de l'entraînement lui-même (Loading, Forward, Backward).

## 1. La Philosophie : Génération vs Consommation

Historiquement, le DataLoader s'occupait de traiter les images (ouvrir le PNG, redimensionner, calculer les stats... puis convertir en tensors) _pendant_ le lancement du modèle. Cela créait plusieurs problèmes :
- Des Crash/Out-of-Memory sur de grands datasets.
- Des engorgements d'I/O (Input/Output).
- Un ralentissement massif du TPU/GPU qui se tournait les pouces en attendant les images.

### Le Modèle Actuel : Les `.npz` Chunks
Nous passons toutes les images par les scripts spécialisés (`fighterjet_*_dataset_tools.py`) qui opèrent **off-line**.
Ces scripts génèrent des **"Chunks" (blocs) de données Numpy (`.npz`)**.

Un Chunk rassemble (par exemple) 27,000 images sous forme matricielle pure, compressées en binaire, formatées exactement aux bonnes dimensions (ex: 128x128 ou 224x224), prêtes à être ingurgitées directement dans la VRAM (`tf.data.Dataset`).

Ensuite, `data_management.py` (appelé par `main.py`) n'a plus qu'une seule mission : **Lire séquentiellement ces Chunks** sans poser la moindre question.

## 2. Taxonomie et Répertoires

Désormais, tout est standardisé par rapport à la clé `task_type` de `dataset_configs.py`. 
Les fichiers générés sont stockés intelligemment dans `./data/chunks/` :

**Pour la Détection :**
```text
./data/chunks/detection/
    dataset_detection_train_chunk0.npz
    dataset_detection_val_chunk0.npz
```

**Pour la Classification :**
```text
./data/chunks/classification/
    dataset_classification_train_chunk0.npz
    dataset_classification_val_chunk0.npz
    dataset_classification_meanstd.npz
```

## 3. Le paradoxe du `meanstd.npz` (Pourquoi seulement en Classification ?)

Si tu inspectes les données produites ci-dessus, tu remarqueras l'absence du fichier `mean_std.npz` du côté "Détection". Est-ce un bug ? Non, c'est une excellente pratique empirique !

### 🎯 En Classification :
Les réseaux de classification (comme les CNN traditionnels) sont hautement sensibles au décalage de distribution ("Covariate Shift") :
- Les images sont **fortement** redimensionnées (perte d'attributs spatiaux primaires).
- La couleur/luminance est la feature prédominante.
- Placer toutes les features autour d'une espérance Mathématique de `0` (Mean=0) et une variance de `1` (Std=1) accélère exponentiellement la descente de gradient, en forçant le réseau à se concentrer sur les *anomalies* structurelles.
- Ce fichier `.npz` est donc vital. `dataset_classification_dataset_tools.py` le calcule en lisant incrémentalement l'intégralité des images d'entraînement, et le sauvegarde explicitement.

### 🎯 En Détection :
Les réseaux modernes de Détection Orientés Objets (YOLO-like) sont agencés différemment :
- Le réseau fait face à une matrice où *l'emplacement* relatif des pixels et leurs contrastes brusques (les arêtes/bordures) comptent plus que leur saturation absolue.
- Ils utilisent abondamment la Batch Normalization distribuée.
- Leur pipeline standard (depuis YOLOv1 à v8) favorise une conversion Min-Max simplissime : `$Image / 255.0$`. L'activation reste encadrable dans `[0.0, 1.0]`. 
- Gérer un décalage via Mean/Std global rajoute un risque de perturber la prédiction des ancres géométriques (BBox), sans apporter d'amélioration de convergence notable. 

C'est pour cela que seul notre pipeline de **Classification** utilise et réclame un `dataset_classification_meanstd.npz` !

## 4. Mode d'Emploi (Workflow d'Entraînement)

Pour lancer un entraînement depuis un dataset vierge :

1. **Générer les Chunks :**
   ```bash
   # Si tu entraînes ton classifieur
   python fighterjet_classification_dataset_tools.py
   
   # Si tu entraînes ton détecteur
   python fighterjet_detection_dataset_tools.py
   ```

2. **Lancer l'entraînement :**
   ```bash
   python main.py FIGHTERJET_CLASSIFICATION
   ```
*(Si vous avez oublié de générer les chunks, `data_management.py` lèvera une erreur rouge bloquante immédiatement en vous indiquant la marche à suivre).*
