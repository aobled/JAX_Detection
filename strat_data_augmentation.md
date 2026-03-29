# Stratégie d'Augmentation de Données (JAX Vision Framework)

L'architecture d'augmentation de données de ce projet repose sur une **séparation stricte des responsabilités (Separation of Concerns)** entre le traitement CPU (TensorFlow) et le traitement TPU/GPU (JAX).

Cette séparation est fondamentale pour garantir que les accélérateurs matériels calculent à 100% de leur capacité sans jamais attendre les transferts mémoire.

---

## 1. L'Augmentation Géométrique et Colorimétrique (Le "Cuisinier")

Ces augmentations impliquent de modifier physiquement l'image pixel par pixel (rotation, zoom, miroir). Ces opérations sont très gourmandes en calculs spatiaux classiques.

- **Acteur responsable :** `TensorFlow` (`tf.data` + `tf.keras.layers`)
- **Fichier de référence :** `data_management.py`
- **Exécution matérielle :** Multi-cœurs asynchrones sur le **CPU**.

### Pourquoi le CPU ?
Le CPU prépare les images en tâche de fond (multithreading) pendant que le TPU est occupé à entraîner le lot précédent. Ainsi, lorsque le TPU a terminé, son prochain lot ("batch") d'images est déjà prêt en mémoire RAM, découpé, tourné et traduit. Intégrer ces rotations à JAX obligerait le TPU à faire ces manipulations d'image au lieu de calculer des gradients, ce qui détruirait le rendement.

### Variables d'Augmentation disponibles
Dans votre configuration (`dataset_configs.py`), vous activez ces augmentations via le marqueur implicite du mode "train", mais le niveau d'intensité est dicté par la clé :
- `aggressive_augmentation: False` -> Mode modéré (Idéal CNNs). Applique un léger recadrage (zoom 10%), rotations faibles (12°), et variation légère de contraste.
- `aggressive_augmentation: True` -> Mode extrême (Idéal ViT). Simule une base de données 10x plus grande. Rotations fortes (30°), Zoom puissant (25%), translation (15%) et miroirs verticaux.

---

## 2. L'Augmentation Mathématique et Structurelle (Le "Mangeur")

Il s'agit des altérations mathématiques portants sur les étiquettes (labels) ou mélangeant l'algèbre de l'image.

- **Acteur responsable :** `JAX` (`pure functions` encapsulées via le Pattern Stratégie)
- **Fichier de référence :** `task_strategies.py`
- **Exécution matérielle :** Directement compilé (`@jax.jit`) sur le **TPU / GPU**.

### Pourquoi le TPU ?
Ces opérations consistent à additionner des matrices de pixels ou jouer avec des lois de probabilités sur les labels (Mixup). Le TPU est le maître incontesté de l'algèbre linéaire, il peut exécuter un Mixup sur 512 images en une nano-seconde, alors que le CPU ralentirait.

### Variables Mathématiques disponibles
La Stratégie (`ClassificationStrategy` par exemple) s'occupe de prétraiter ces données mathématiques juste avant l'inférence :
- `mixup_alpha` (ex: `0.05` ou `0.2`) : Superpose spectralement deux images d'avions et fusionne leurs labels (ex: l'image est à 80% un Rafale et 20% un Mirage). Formidable pour éviter le sur-apprentissage des CNNs.
- `label_smoothing` (ex: `0.1` ou `0.15`) : Écrase la certitude des "One-Hot Encodings" (100% Avion A -> 90% Avion A, 10% Inconnu). Oblige particulièrement les Vision Transformers (ViT) à ne jamais être trop sûrs d'eux sur leurs *Attention Maps*.
- **Cast dynamique :** Conversion mathématique ciblée (`int32` via Cross-Entropy pour classification vs `float32` coordonnés `x, y, w, h` pour la détection).

---

## Résumé du Workflow (Le "Pipeline")

1. L'utilisateur lance son entraînement avec une configuration donnée (ex: `FIGHTERJET_CLASSES`).
2. Pendant que le réseau calcule, **TensorFlow (`data_management.py`) déterre les images du disque dur, les tourne, les recolore, et les stocke dans un "buffer" RAM**. C'est le monde asynchrone CPU.
3. Pendant ce temps, dès qu'un buffer est prêt, **le TPU aspire le buffer**. Le Trainer JAX intercepte les images, appelle `TaskStrategy.preprocess_batch()` pour opérer un **Mixup algébrique ou lisser les probabilités**, puis exécute la fonction de perte (`compute_loss()`) à la vitesse de la lumière. Le tout est scellé sous le *Just-In-Time* compilateur de JAX.
