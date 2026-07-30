# Anatomie de ChessCnnAttentionPolicyValue

Ce document plonge dans le code source de l'architecture **CNN + bottleneck d'attention** (dans `model_library.py`) pour expliquer en détail ses dimensions, sa stratégie d'encodage/décodage, et pourquoi elle diffère structurellement de `AircraftDetectorUNet` (pas de U, pas de skip connections — un tout autre problème à résoudre).

Contexte : c'est le modèle du domaine échecs (Epic 9, `task_type="chess_policy_value"`), premier modèle du projet qui ne fait ni classification ni détection — il produit deux sorties simultanées (un coup à jouer + une estimation de qui gagne) à partir d'un plateau d'échecs.

---

## 1. Le problème posé, et pourquoi ce n'est pas un U-Net

`AircraftDetectorUNet` répond à "où est l'objet, précisément, au pixel près ?" — il a besoin de revenir à la résolution d'entrée (d'où le U : descendre puis remonter avec des skip connections pour ne pas perdre les détails fins).

Le problème des échecs est différent : la sortie n'est **pas une carte spatiale**, c'est (1) un coup parmi 4672 possibles et (2) un scalaire d'évaluation de la position. Il n'y a donc rien à "remonter en résolution" — le réseau doit plutôt **comprendre la position dans son ensemble** (menaces, contrôle de cases, structure de pions) puis **condenser** cette compréhension en une poignée de vecteurs avant de trancher. D'où le choix d'architecture : CNN (comprendre localement) → bottleneck d'attention (condenser globalement) → deux têtes denses (trancher).

---

## 2. L'entrée : 19 ou 29 plans de 8×8 selon la variante

L'entrée est une position d'échecs déjà encodée en plans binaires/continus par `chess_target_encoding.py::encode_position` (Story 9.1, source unique de cet encodage — jamais réimplémenté ici). Contrairement à `AircraftDetectorUNet` qui reçoit toujours une image classique (224×224 pixels), le modèle échecs n'a **jamais codé son nombre de canaux d'entrée en dur** (inféré par la première conv) — deux variantes coexistent, entraînées et comparées le 2026-07-29 (voir `deferred-work.md`, "test d'ablation historique") :

- **19 plans "position courante"** (toujours présents) : 6 plans pièces du joueur au trait + 6 plans pièces adverses (Roi/Dame/Tour/Fou/Cavalier/Pion, binaires) + 1 plan trait + 4 plans droits de roque + 1 plan répétition + 1 plan cases de destination des coups légaux.
- **10 plans "historique"** (optionnels, `include_history=True` par défaut dans `encode_position`) : les 5 derniers demi-coups, 2 plans chacun (case source + case destination), zéro-paddés en début de partie.

| Variante | Entrée | Config | Résultat mesuré (val PolicyAccuracy, 15 epochs, dataset Carlsen) |
|---|---|---|---|
| Avec historique | `(Batch, 8, 8, 29)` | `CHESS` | 24.43% |
| **Sans historique** (**défaut actuel de `chess_game.py --vs-model`**) | `(Batch, 8, 8, 19)` | `CHESS_NO_HISTORY` | 23.68% |

L'écart (0.75 point) est jugé réel mais modeste — retenu comme défaut malgré la légère perte, pour la simplicité (moins de plans à raisonner, pas de gestion d'historique côté consommateurs futurs). Les deux variantes restent disponibles et entraînables (`dataset_configs.py`), aucune n'a été supprimée.

Chaque case du plateau (8×8 = 64 cases) est donc un "pixel" avec 19 ou 29 canaux selon la variante — le CNN qui suit va le traiter exactement comme `AircraftDetectorUNet` traite un pixel d'image, sauf que la grille est minuscule (8×8 au lieu de 224×224).

---

## 3. Le tronc CNN : pas de Max Pooling, et c'est voulu

*   **Conv initiale :** `nn.Conv(D filtres, 3×3)` + `nn.BatchNorm` + `nn.silu`.
    *   *Sortie :* `(Batch, 8, 8, D)` (`D = token_dim`, 64 par défaut).
*   **2 blocs résiduels**, chacun : `SeparableConv(3×3)` → BN → silu → (branche résiduelle `Conv(1×1)` + BN) → `SeparableConv(3×3)` → BN → **addition** → silu.
    *   `SeparableConv` = convolution séparable (depthwise puis pointwise) : la depthwise applique un filtre 3×3 indépendant par canal (`feature_group_count=x.shape[-1]`), la pointwise (1×1) recombine les canaux ensuite — moins de paramètres qu'une conv 3×3 classique pour un résultat comparable, même brique que `SophisticatedCNN128Lite`.
    *   *Sortie après les 2 blocs :* toujours `(Batch, 8, 8, D)` — la résolution spatiale ne bouge jamais.

**Aucun `nn.max_pool` dans tout le tronc** — différence structurelle majeure avec `AircraftDetectorUNet`. Un plateau 8×8 est déjà minuscule : avec la conv initiale + les 2 blocs (5 convolutions 3×3 au total), le champ réceptif atteint déjà 11×11, ce qui couvre largement les 8×8 du plateau. Réduire encore la résolution détruirait l'identité des cases individuelles — or la tête policy (§5) a justement besoin de distinguer les 64 cases source des coups possibles. Pas de perte à rattraper via des skip connections : il n'y en a jamais eu.

---

## 4. Le bottleneck : un Perceiver, pas un pooling classique

C'est ici que l'architecture s'écarte le plus radicalement de `AircraftDetectorUNet`. Au lieu d'aplatir directement les 64 cases en un vecteur (`GlobalAveragePooling` classique, qui perdrait toute structure), le modèle utilise un mécanisme de **cross-attention façon Perceiver/TokenLearner** (AD-23) : un petit nombre de "questions apprises" viennent interroger les 64 cases pour en extraire l'information pertinente.

1. **Aplatissement spatial :**
   `(Batch, 8, 8, D)` → `(Batch, 64, D)` — les 64 cases deviennent 64 "tokens", chacun un vecteur de taille `D`.

2. **Requêtes apprises (`bottleneck_queries`) :**
   Un paramètre entraînable de forme `(K, D)` (`K = num_bottleneck_tokens`, **8 par défaut**) — indépendant du batch, dupliqué (`jnp.broadcast_to`) pour chaque exemple. Ce ne sont ni des cases du plateau ni une moyenne de quoi que ce soit : ce sont K vecteurs que le réseau apprend librement pendant l'entraînement, dont le rôle émerge ("qu'est-ce qui mérite d'être retenu de cette position ?").

3. **Cross-attention (`nn.MultiHeadDotProductAttention`, `inputs_q=queries, inputs_kv=board_tokens`) :**
   Chacune des K requêtes "regarde" les 64 cases et en retient un résumé pondéré. *Sortie :* `(Batch, K, D)` — la grille 8×8 a disparu, remplacée par K vecteurs de synthèse.

4. **Auto-attention (`inputs_q=tokens`, pas de `inputs_kv` séparé) :**
   Les K tokens se recroisent entre eux (sans biais géométrique — AD-23 différé, voir spine) pour affiner leurs résumés respectifs les uns par rapport aux autres. *Sortie :* toujours `(Batch, K, D)`.

5. **Pooling final (`jnp.mean(tokens, axis=1)`) :**
   Simple moyenne sur les K tokens → `(Batch, D)`. C'est le seul endroit où K disparaît de la forme des tenseurs — **conséquence directe et non-intuitive** : la taille des deux têtes (§5) ne dépend jamais de K, seulement de D (voir §6).

---

## 5. Les deux têtes

*   **Policy :** `nn.Dense(num_moves)` sur le vecteur `(Batch, D)` → `(Batch, 4672)`, logits bruts (pas de softmax ici — la cross-entropy de `loss_functions.py` l'attend en logits). Pas de masquage des coups illégaux (AD-22) : le réseau peut techniquement "voter" pour un coup illégal, filtré seulement à l'inférence (`chess_target_encoding.py::index_to_move`, jamais côté entraînement).
*   **Value :** `nn.Dense(1)` → `nn.tanh` → `jnp.squeeze` → `(Batch,)`, un scalaire dans `[-1, 1]` (AD-24) : qui gagne, du point de vue du joueur au trait.

```python
return {"policy": policy_logits, "value": value}
```

---

## 6. Où vivent vraiment les paramètres (et pourquoi c'est contre-intuitif)

Mesuré en session (2026-07-29) plutôt que supposé — à K=8/D=64, le modèle fait **382 017 paramètres**, dont **299 008 (78%) dans la seule tête policy**. Le CNN + le bottleneck d'attention réunis ne pèsent qu'environ 83 000 paramètres.

Conséquence directe de la formule `Dense(D → num_moves)` : la tête policy scale avec `D` (largeur des tokens), **jamais avec `K`** (nombre de tokens du bottleneck) — le pooling moyenne les K tokens en un seul vecteur *avant* la tête, donc K n'atteint jamais la couche qui coûte cher. Vérifié empiriquement :

| K | D | Total params | Dense policy |
|---|---|---|---|
| 8 | 64 | 382 017 | 299 008 |
| 16 | 64 | 382 529 | 299 008 (inchangé) |
| 32 | 64 | 383 553 | 299 008 (inchangé) |
| 8 | 128 | 874 049 | 598 016 |

Augmenter `K` est donc quasi gratuit ; augmenter `D` double quasiment le modèle. Un test réel à K=32 (15 epochs, dataset Carlsen, voir `deferred-work.md`) n'a d'ailleurs montré **aucun gain** de qualité de jeu par rapport à K=8 — confirmant que le goulot actuel n'est pas la capacité du bottleneck (voir `deferred-work.md`, chantier "modèle v2").

---

## 7. Résumé du flux complet

```
(B, 8, 8, 19 ou 29)                            ← position (+ historique optionnel, chess_target_encoding.py)
  → Conv 3×3 + BN + silu                       (B, 8, 8, D)
  → 2× bloc résiduel SeparableConv             (B, 8, 8, D)     aucun max_pool, résolution figée
  → reshape                                    (B, 64, D)       64 "tokens case"
  → cross-attention (K requêtes apprises)      (B, K, D)        Perceiver-style, K=8 par défaut
  → auto-attention entre les K tokens          (B, K, D)
  → mean pooling                               (B, D)           K disparaît ici
  → Dense(num_moves)                           (B, 4672)        policy (78% des paramètres)
  → Dense(1) + tanh                            (B,)             value ∈ [-1, 1]
```
