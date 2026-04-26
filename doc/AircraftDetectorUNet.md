# Anatomie de AircraftDetectorUNet

Ce document plonge dans le code source de l'architecture **U-Net** (dans `model_library.py`) pour expliquer en détail ses dimensions, sa stratégie d'encodage/décodage, et l'utilisation spécifique des bibliothèques de l'écosystème JAX (`nn`, `jnp`, `jax`).

---

## 1. L'Écosystème JAX/Flax : `nn`, `jnp`, `jax`

Si tu regardes le code de l'U-Net, tu remarqueras qu'on jongle entre trois préfixes : `nn.`, `jnp.` et `jax.`. Chacun a un rôle très précis dans la construction du modèle :

- **`nn` (Flax Linen - `flax.linen`)** : C'est la bibliothèque de "Deep Learning". On l'utilise pour tout ce qui possède des **poids entraînables** (paramètres) ou des états internes (statistiques). 
  - *Exemples* : `nn.Conv` (convolutions avec poids), `nn.BatchNorm` (statistiques de moyenne/variance), `nn.silu` (fonctions d'activation natives de Flax).
- **`jnp` (JAX NumPy - `jax.numpy`)** : C'est le clone strict de NumPy, mais accéléré sur GPU/TPU. On l'utilise pour les **manipulations de tenseurs** pures qui ne nécessitent pas de paramètres d'apprentissage.
  - *Exemple* : `jnp.concatenate([u1, x3], axis=-1)` pour coller deux tableaux de pixels l'un contre l'autre.
- **`jax` (JAX Core - `jax`)** : Contient les fonctions mathématiques ou de transformation d'image pures.
  - *Exemple* : `jax.image.resize(...)` pour agrandir une image mathématiquement.

---

## 2. L'Architecture U-Net (Dimensions & Stratégie)

L'architecture est appelée "U-Net" car elle a une forme en "U" : on descend dans les abysses de la basse résolution pour comprendre "quoi" est dans l'image (Encodeur), puis on remonte à la haute résolution pour dire "où" c'est exactement au pixel près (Décodeur).

### A. L'Encodeur (La descente : 224x224 ➡️ 28x28)

L'objectif de l'encodeur est d'extraire la sémantique de l'image. Chaque bloc suit la même logique : *Deux convolutions (pour chercher des motifs) + un Max Pooling (pour réduire la taille de moitié).*

*   **Entrée :** `(Batch, 224, 224, Canaux)`
*   **Block 1 :** 
    *   2x `nn.Conv(32 filtres)` -> On cherche des bords, des lignes.
    *   `nn.max_pool` divise par 2.
    *   *Sortie `x1`* : `(Batch, 112, 112, 32)`
*   **Block 2 :** 
    *   2x `nn.Conv(64 filtres)` -> On cherche des formes simples (ailes, nez).
    *   `nn.max_pool` divise par 2.
    *   *Sortie `x2`* : `(Batch, 56, 56, 64)`
*   **Block 3 :** 
    *   2x `nn.Conv(128 filtres)` -> On cherche des concepts complexes (fuselage, réacteurs).
    *   `nn.max_pool` divise par 2.
    *   *Sortie `x3`* : `(Batch, 28, 28, 128)`

### B. Le Bottleneck (Le goulot d'étranglement : 28x28)

C'est le fond du "U". L'image n'est plus qu'une grille de 28x28 pixels, mais elle est extrêmement "épaisse" (256 canaux/filtres). À ce stade, le réseau "sait" qu'il y a un avion de chasse dans le quadrant supérieur droit, mais l'image est trop pixelisée pour dessiner un contour parfait.

*   **Bottleneck :** 
    *   2x `nn.Conv(256 filtres)`.
    *   *Sortie `b`* : `(Batch, 28, 28, 256)`

### C. Le Décodeur (La remontée : 28x28 ➡️ 224x224)

C'est ici que la magie de l'U-Net opère. Pour remonter en résolution, on utilise la combinaison de **l'Upsampling** (`jax.image.resize`) et des **Skip Connections** (`jnp.concatenate`).

1. **Upsampling (L'agrandissement mathématique)** : 
   Historiquement, les réseaux utilisaient des "Convolutions Transposées" (`ConvTranspose`) pour agrandir l'image. Le problème ? Ça créait des motifs en forme de damier (Checkerboard Artifacts) sur la chaleur générée.
   Dans notre modèle, on utilise **`jax.image.resize(..., method='bilinear')`**. C'est un simple zoom mathématique (comme sur Photoshop). L'avantage ? C'est ultra-lisse, rapide, et ça n'ajoute pas de paramètres inutiles au modèle !

2. **Skip Connections (La triche)** :
   Quand on zoome de 28x28 à 56x56, l'image devient floue. Le réseau a oublié les détails fins (comme les bouts d'ailes). L'astuce géniale de l'U-Net est de faire un "copier-coller" : on prend l'image `x3` (qui faisait 56x56 dans l'encodeur et qui contient encore les détails nets) et on la "colle" à côté de notre image zoomée floue avec `jnp.concatenate([u1, x3], axis=-1)`. Le réseau convolutionnel suivant va lire les deux images en même temps et utiliser les détails de `x3` pour rendre notre image zoomée super nette.

*   **Up 1 :** 
    *   Resize de `28x28` à `56x56`.
    *   Concaténation avec `x3`. L'épaisseur passe à `128 + 128 = 256` canaux.
    *   2x `nn.Conv(128)`.
    *   *Sortie `u1`* : `(Batch, 56, 56, 128)`
*   **Up 2 :** 
    *   Resize à `112x112`. Concaténation avec `x2`.
    *   2x `nn.Conv(64)`.
    *   *Sortie `u2`* : `(Batch, 112, 112, 64)`
*   **Up 3 :** 
    *   Resize à `224x224`. Concaténation avec `x1`.
    *   2x `nn.Conv(32)`.
    *   *Sortie `u3`* : `(Batch, 224, 224, 32)`

### D. La Sortie (1x1 Convolution)

Enfin, on a une belle image `224x224` avec 32 filtres remplis d'informations. On la passe dans une dernière convolution spéciale : `nn.Conv(1, (1, 1))`.
C'est un "aplatisseur" : il regarde les 32 canaux de chaque pixel et donne une seule note finale par pixel.
La fonction `nn.sigmoid` compresse cette note entre `0.0` et `1.0`.

*   **Sortie finale :** `(Batch, 224, 224, 1)` (Notre fameuse carte de chaleur prête pour OpenCV !).
