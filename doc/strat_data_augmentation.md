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

---

## 3. Analyse Comparative : Classification vs Détection

Bien que les deux tâches respectent l'architecture décrite ci-dessus (Augmentations sur CPU dans le pipeline `tf.data`), l'implémentation diffère radicalement pour s'adapter à la nature des données (les *Bounding Boxes*). 

**À noter : Les fichiers "Chunks" `.npz` stockés sur le disque ne contiennent QUE les données brutes (non augmentées). L'augmentation se fait 100% à la volée pendant le chargement.**

### A. Méthodologie d'Implémentation (`data_management.py`)

#### Mode Classification
La classification n'a pas à se soucier de déformer la cible (le label reste "Avion A" peu importe s'il est à l'envers ou étiré).
* **Outil utilisé :** `tf.keras.Sequential()` contenant les calques préfabriqués (ex: `layers.RandomRotation`, `layers.RandomZoom`). Très rapide et haut niveau.
* **Mise en place :** Le tenseur d'image traverse simplement le séquentiel. Le label (Int) n'est pas altéré.

#### Mode Détection
La détection est beaucoup plus complexe, car si l'image subit une translation de 10% vers la gauche, les coordonnées `cx, cy` de toutes les boîtes doivent être réduites de `0.10` avec précision.
* **Outils utilisés :** Opérations mathématiques pures via `tf.image` et des blocs conditionnels `tf.cond`.
* **Mise en place :** Chaque augmentation (Flip, Zoom, Translation) recalcule en parallèle les nouvelles valeurs dynamiques du tenseur `boxes (MAX_BOXES, 5)`. 
* *Fait notable :* La rotation n'est actuellement pas disponible en Détection, car calculer la nouvelle boite englobante orthogonale d'un box tourné nécessite une trigonométrie complexe qui élargit la boîte artificiellement à chaque fois.

### B. Contrôle Dynamique Unifié (`augmentation_params`)

Suite à la refactorisation de l'architecture, nous avons supprimé l'ancien booléen `aggressive_augmentation` (Classification) et les valeurs codées "en dur" (Détection).
Désormais, **les deux logiques utilisent un seul et même dictionnaire de configuration paramétrable** situé dans `dataset_configs.py` : `"augmentation_params"`.

* **Comment ?** Chaque dictionnaire injecte ses facteurs statistiques (`zoom_factor`, `translation_factor`, `contrast_factor`, etc.).
* **Avantage :** Keras (Classification) et les fonctions personnalisées (Détection) traduisent ces mêmes facteurs `float` en opérations géométriques, offrant un contrôle unifié et précis pour combattre le sur-apprentissage.

---

## 4. Cas d'Usage : Résolution des Biais Cognitifs 🧠

Le fait de pouvoir paramétrer finement cette agressivité nous permet de corriger des biais cognitifs (raccourcis que prend le réseau pour tricher). Deux problèmes majeurs ont été diagnostiqués et corrigés via ce dictionnaire d'augmentation :

### Problème 1 : Le "Scintillement" en Classification Vidéo
**Symptôme :** Dans une vidéo, si le cadre de l'avion se décale de quelques pixels ou zoome légèrement, le classificateur change brusquement d'avis sur le modèle de l'avion.
**Diagnostic :** Le réseau manque *d'invariance par translation*. Il a mémorisé par cœur où se situait le nez de l'avion sur la grille 128x128. Si le nez bouge, il panique.
**Solution appliquée :**
- `translation_factor` augmenté à `0.12`. Le réseau est forcé de s'entraîner sur des avions souvent désaxés ou coupés dans le cadre.
- `zoom_factor` augmenté à `0.15`. Le modèle apprend la silhouette indifféremment de sa taille d'occupation de l'image.

### Problème 2 : La cécité du "Low Pass" en Détection (Overfitting du fond)
**Symptôme :** Le détecteur V7 trouve tous les avions dans le ciel, mais perd tout avion volant à basse altitude au-dessus d'une ville ou des arbres.
**Diagnostic :** Une majorité d'images du dataset ont pour fond un ciel "gris uni/bleu". Le réseau a créé le raccourci cognitif : *"Silhouette sombre entourée d'un aplat uni = Avion"*. Face à la "complexité haute fréquence" d'une ville (des arêtes, du contraste), le réseau est perdu.
**Solution appliquée :**
- `brightness_delta` et `contrast_factor` bondissent à `0.30`. L'augmentation simule des ciels très nuageux gris foncés, ou au contraire une image "cramée" de blanc. Le réseau ne peut plus s'appuyer sur la luminance du fond gris clair pour s'assurer que c'est un ciel. Il *doit* trouver des caractéristiques géométriques d'ailes.
- `zoom_factor` poussé à `0.25` pour maîtriser l'architecture *Multi-échelle* V7 sur d'infimes cibles.
