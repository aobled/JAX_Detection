# Stratégie de Détection : Segmentation Sémantique (U-Net)

Ce document décrit l'architecture de détection d'objets actuellement déployée dans le projet : **AircraftDetectorUNet**. Cette approche repose sur la segmentation sémantique plutôt que sur la prédiction mathématique de boîtes englobantes (Bounding Boxes).

## 1. Philosophie : Pixels vs Géométrie

### L'ancienne approche (YOLO-like / Anchor-Free / Grilles)
Historiquement, les modèles de détection (comme notre `AircraftDetectorV7_Advanced`) divisaient l'image en "grilles" (ex: 28x28 ou 14x14). Pour chaque case de la grille, le réseau tentait de résoudre une équation complexe :
- Est-ce qu'il y a un objet au centre de cette case ? (`confiance`)
- Si oui, quels sont ses décalages exacts par rapport à la case ? (`x`, `y`)
- Quelle est sa largeur et sa hauteur ? (`w`, `h`)

**Problèmes rencontrés :**
1. **Limitation de champ de vision (Receptive Field)** : Si un bombardier géant occupait toute l'image, le neurone responsable de la petite grille 28x28 au centre de l'avion n'avait pas assez de recul pour "voir" les bords de l'avion et était incapable de prédire une largeur `w` immense. 
2. **Le fléau du NMS (Non-Max Suppression)** : Quand plusieurs petits avions volaient en formation serrée, les prédictions se chevauchaient. L'algorithme NMS, censé nettoyer les doublons, supprimait souvent les chasseurs d'escorte par erreur, croyant qu'ils faisaient partie de la même boîte que le gros avion central.

### La nouvelle approche (Segmentation Sémantique / Heatmaps)
La segmentation sémantique jette toutes ces mathématiques par la fenêtre. Le réseau de neurones redevient un simple outil de "coloriage". Pour chaque pixel de l'image en entrée, le modèle doit répondre à une seule question :
- *"Ce pixel fait-il partie d'un avion ou du ciel ?"*

Le modèle `AircraftDetectorUNet` génère donc une **Carte de Chaleur (Heatmap) en 2D**, de la même taille que l'image originale (224x224). 
Un bombardier géant génère un énorme nuage blanc (des milliers de pixels allumés). Un petit chasseur génère une petite étoile (quelques dizaines de pixels allumés).

**Les avantages :**
1. **Agnostique de l'échelle (Scale-Free)** : Plus besoin d'inventer des grilles différentes (7x7, 14x14, 28x28). Le nuage thermique s'adapte naturellement à la taille réelle de l'objet.
2. **Adieu le NMS** : Deux avions en formation serrée créeront simplement deux taches blanches séparées par un mince filet noir (le ciel entre les deux). 

## 2. Le Modèle : U-Net (Encodeur / Décodeur)

Le modèle implémenté dans `model_library.py` s'appelle `AircraftDetectorUNet`. Il s'agit d'une architecture symétrique en forme de "U" très célèbre en imagerie médicale :

1. **L'Encodeur (Downsampling)** :
   - L'image entre en 224x224.
   - À travers une série de convolutions et de `Max Pooling`, la résolution diminue (112x112 -> 56x56 -> 28x28) tandis que le nombre de filtres augmente (32 -> 64 -> 128 -> 256).
   - Ce processus permet au modèle de comprendre le "Contexte Global" (ex: "Ah, c'est une forme de fuselage d'avion au milieu du ciel !").

2. **Le Décodeur (Upsampling)** :
   - À partir du goulot d'étranglement (28x28), on remonte progressivement en résolution (56x56 -> 112x112 -> 224x224) en utilisant des interpolations bilinéaires (`jax.image.resize`).
   - **Le Secret de l'U-Net (Skip Connections)** : À chaque étape de remontée, on fusionne (`jnp.concatenate`) la carte floue actuelle avec la carte haute-résolution correspondante provenant de l'Encodeur. Cela permet de récupérer la précision des bords (les ailes, la queue de l'avion).

3. **La Couche de Sortie** :
   - Une unique convolution 1x1 avec une activation `Sigmoid` crache un masque de probabilités (valeurs entre 0.0 et 1.0) de format `(224, 224, 1)`.

## 3. L'Entraînement et la Loss

Le grand changement réside dans la fonction d'erreur (Loss). Dans `loss_functions.py`, on utilise `compute_segmentation_loss`.
C'est une simple **Mean Squared Error (MSE)** pixel par pixel entre la prédiction et le masque cible. Le réseau apprend extrêmement vite car le signal d'erreur est direct pour chaque neurone.

L'augmentation de données dans `data_management.py` a également été simplifiée : si on fait pivoter l'image ou qu'on zoome dessus, on applique strictement la même fonction de pivot/zoom au Masque. Pas de trigonométrie requise.

## 4. L'Inférence par "Contouring"

Lors de l'utilisation en production (sur vidéo), le script d'inférence récupère la Heatmap 224x224.
Comment retrouver des `Bounding Boxes` pour l'interface utilisateur ? Grâce à OpenCV :

1. **Redimensionnement** : La heatmap 224x224 est étirée à la taille du flux vidéo (ex: 1920x1080).
2. **Binarisation** : Tout pixel ayant une probabilité supérieure au `DETECTION_CONF_THRESHOLD` (ex: 30%) devient purement blanc. Le reste devient noir.
3. **Contouring** : L'algorithme `cv2.findContours` isole chaque "île" de pixels blancs.
4. **Bounding Rect** : `cv2.boundingRect` trace automatiquement la boîte englobante mathématiquement parfaite autour de chaque île.

## 5. Configuration dans `dataset_configs.py`

Pour utiliser ce modèle, voici le paramétrage typique à appliquer :

```python
"training_params": {
    # Nom exact défini dans model_library.py
    "model_name": "aircraft_detector_unet",
    
    # La taille de la grille DOIT être égale à la taille de l'image (224)
    # car le modèle crache un masque en résolution 1:1
    "grid_size": 224,
    
    # Le batch size peut être de 16 ou 32 selon la VRAM (l'U-Net est léger)
    "batch_size": 16,
    ...
}
```

**Seuil d'Inférence** :
Dans tes scripts vidéo, le `DETECTION_CONF_THRESHOLD` contrôle désormais la tolérance d'allumage des pixels de la Heatmap. 
- S'il est trop bas (ex: 0.1), du "bruit" (nuages) créera de petites îles détectées par erreur.
- S'il est trop haut (ex: 0.8), les bords de l'avion ne s'allumeront pas, réduisant la taille de la Bounding Box finale. Un bon point de départ est **0.3 à 0.4**.
