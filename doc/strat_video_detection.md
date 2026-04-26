# Stratégie Vidéo : Découplage Détection et Classification "Haute Résolution"

Le script `bounding_boxes_with_classification_from_video_generation.py` est conçu autour d'une architecture découplée (Two-Stage Pipeline). L'objectif principal de ce design est de **maximiser la résolution des pixels fournis au classificateur**, même lorsque le détecteur travaille sur une résolution très compressée.

## 1. La Problématique
Si un pipeline unifié travaillait uniquement sur une image compressée à `224x224`, un avion qui fait "150 pixels" sur la vidéo 1080p d'origine ne ferait plus que `15 pixels` sur la matrice analysée.
Si l'on extrayait directement la boîte de cette matrice réduite pour l'envoyer à la classification, le modèle de classification (attendant du `128x128`) devrait *upscaler* artificiellement ces 15 pixels, résultant en une bouillie de gros pixels illisibles.

## 2. Le Pipeline Découplé

Pour pallier ce problème, le script utilise la vidéo Full-HD comme source de vérité (Raw Source) et ne se sert de la version `224x224` que comme un système de ciblage radar.

### Étape 1 : Ciblage (Détection basse résolution)
1. L'image d'origine (ex: `1920x1080`) est extraite de la vidéo. Sa résolution est stockée (`h_orig, w_orig`).
2. Une **copie basse résolution** (`224x224`) est générée.
3. Cette copie passe dans le modèle JAX de détection (`AircraftDetectorV7`).
4. Le modèle renvoie une grille de prédictions contenant des coordonnées **relatives** (pourcentages de l'image de 0.0 à 1.0).

### Étape 2 : Projection Géométrique
Les pourcentages fournis par le détecteur sont mathématiquement projetés sur la résolution d'origine :
```python
center_x = bx * w_orig  # bx (ex: 0.5) * 1920 = 960 pixels
center_y = by * h_orig  # by (ex: 0.5) * 1080 = 540 pixels
```
Les coordonnées finales de la boîte `(x1, y1, x2, y2)` sont ainsi ramenées dans le référentiel de l'image "Haute Définition".

### Étape 3 : Extraction Haute Fidélité (Crop HD)
C'est ici que la magie opère. Au lieu d'utiliser la matrice `224x224` du détecteur, le script découpe le rectangle directement dans le flux mémoire de l'image vidéo d'origine non-modifiée :
```python
crop = target_frame[y1:y2, x1:x2]
```
Ce `crop` contient 100% de la densité des pixels capturés par la caméra. 

### Étape 4 : Classification
Le `crop` HD est envoyé au modèle de classification.
Ce modèle s'attend à une matrice de `128x128`. L'image HD est donc redimensionnée à cette taille via une interpolation OpenCV (ex: `cv2.resize(..., (128, 128))`).
Contrairement à un *upscaling* baveux, il s'agit d'un **downscaling qualitatif**, préservant les arêtes, le contraste et les détails de l'avion, maximisant ainsi les probabilités de la couche Softmax de classification finale.

## 3. Le Lissage Temporel Centré (Sliding Window)
Pour sublimer ce processus, les boîtes projetées passent par un `CenteredTemporalSmoother` avant le découpage HD.
- La vidéo n'est pas traitée en "temps réel strict" mais avec un léger délai d'attente (buffer de `N` frames).
- Pour dessiner la frame `T`, le système agrège mathématiquement les positions de l'avion des frames `[T-N, T+N]`.
- La moyenne arithmétique de ces fenêtres neutralise les micro-sauts algorithmiques du modèle (flickering).
- En interpolant le futur et le passé, la boîte est toujours parfaitement **centrée** sur le centre de gravité de l'objet, même en déplacement très rapide, ce qui garantit un crop symétrique et optimal pour la classification.
