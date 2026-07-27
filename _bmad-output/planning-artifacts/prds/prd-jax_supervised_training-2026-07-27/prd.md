---
title: Généralisation du pipeline de training — preuve par un moteur d'échecs
status: final
created: 2026-07-27
updated: 2026-07-27
---

# PRD : Généralisation du pipeline de training — preuve par un moteur d'échecs

## 0. Document Purpose

Ce PRD traduit en exigences le brief produit déjà validé (`_bmad-output/planning-artifacts/briefs/brief-jax_supervised_training-2026-07-27/brief.md`, addendum inclus) : brancher un nouveau domaine — les échecs — sur le pipeline `jax_supervised_training`, comme test de généricité réelle de `Trainer`/`TaskStrategy` face à une sortie qui n'est ni une classification ni une détection. Projet solo, enjeu interne/epic technique — ce document sert de base directe à la session d'architecture (Winston) puis à la liste d'epics/stories, sans dupliquer le détail déjà capturé dans le brief et son addendum (état de l'art, design de label rejeté).

## 1. Vision

`jax_supervised_training` a déjà prouvé sa généricité sur plusieurs domaines à géométrie fixe (images ou séries 1D) mappés vers une classification ou une détection :

| Domaine | Type d'input | `task_type` | Modèle | Sortie |
|---|---|---|---|---|
| CIFAR10 | Images 32×32 RGB | `classification` | `sophisticated_cnn_32_plus` | classe |
| FIGHTERJET_CLASSIFICATION | Images 128×128 | `classification` | `sophisticated_cnn_128_lite` | classe (~30 avions) |
| FIGHTERJET_DETECTION | Images | `detection` | `aircraft_detector_unet` | masque/boîtes (UNet) |
| JAX_DETECTOR | Images | `detection_centernet` | `aircraft_detector_centernet` | heatmap + taille (CenterNet) |
| KEPLER | Séries 1D (courbes de lumière) | `kepler` | `kepler_1d_cnn` | classe |

Chaque nouveau domaine a jusqu'ici nécessité un nouveau `model_name`, une entrée `dataset_configs.py` dédiée, et parfois un nouveau `task_type`/`TaskStrategy` — jamais de modification de `Trainer` lui-même. Les échecs testent une rupture de nature différente : l'entrée est un état de jeu structuré (plateau + historique de coups + règles), et la sortie attendue est une paire **policy** (distribution sur les coups légaux) / **value** (évaluation de position) — le premier domaine du projet qui n'est ni une classe ni une carte spatiale.

Cette epic (temps 1) valide que le pipeline générique reste utilisable pour ce type de tâche, sans être complexifié inutilement pour l'absorber. Elle ne vise pas la qualité de jeu du modèle produit — celle-ci sera évaluée ultérieurement, dans un projet séparé, via un banc de test interactif existant (`./chess/chess_game.py`, basé sur `python-chess`).

## 2. Contexte d'usage

Usage strictement personnel, pas de portage ni de partage prévu. Un seul opérateur (Aymeric), pas d'interface produit à documenter — pas de parcours utilisateur au sens UX du terme.

**Jobs to be done :**
- *"En tant que porteur solo du pipeline, je veux vérifier que `Trainer`/`TaskStrategy` généralisent à un domaine structurellement différent (état de jeu → policy/value), sans devoir complexifier le cœur générique, pour savoir si le pattern actuel tient face à un cas encore plus exotique qu'une nouvelle géométrie d'image."*
- *"En tant que futur utilisateur de mon propre banc de test (`chess_game.py`), je veux qu'un modèle entraîné sur ce nouveau domaine soit un candidat plausible à y brancher plus tard — même si cette intégration n'est pas faite dans ce cycle."*

`[ASSUMPTION: JTBD reformulé à partir du memlog du brief (finalité "modèle jouable à terme", pas juste preuve jetable façon KEPLER) — à confirmer si la formulation ne correspond pas à l'intention.]`

## 3. Glossaire

- **`task_type`** — clé de configuration qui sélectionne la `TaskStrategy` utilisée par `Trainer` pour un domaine donné (ex. `classification`, `detection_centernet`, `kepler`). Le domaine échecs introduit un nouveau `task_type`, nom à trancher en architecture.
- **`TaskStrategy`** — pattern Strategy qui encapsule la logique spécifique à un domaine (construction du batch, loss, métriques d'évaluation) sans que `Trainer` en connaisse le détail. Les échecs nécessitent une `TaskStrategy` à **double tête** (policy + value, deux losses) — aucune `TaskStrategy` existante n'a ce besoin.
- **Policy (tête)** — sortie du modèle échecs : distribution de probabilité sur les coups légaux à jouer dans une position donnée.
- **Value (tête)** — sortie du modèle échecs : évaluation scalaire de la position, du point de vue du joueur au trait (+1 victoire probable, 0 nulle, -1 défaite probable).
- **Position** — un état de jeu à un instant donné (plateau + trait + droits de roque + historique récent), unité d'exemple du dataset (un demi-coup = une position).
- **Demi-coup (ply)** — un coup d'un seul joueur (Blancs ou Noirs). Une partie de N demi-coups produit N exemples de dataset.
- **Planes façon AlphaZero** — encodage de l'input sous forme de plans binaires/numériques empilés (pièces par type et couleur, trait, droits de roque, répétition, coups légaux), format d'entrée standard pour les modèles d'échecs par réseau de neurones.
- **Bottleneck de tokens** — étape de l'architecture envisagée où les features locales du CNN 8×8 sont réduites à un petit nombre de tokens, sur lesquels s'applique ensuite l'auto-attention.
- **`JAX_DETECTOR`** — pipeline de production avion actuel (CenterNet), dont le fonctionnement à l'identique est une contrainte dure de cette epic.
- **PGN** — format texte standard d'enregistrement des parties d'échecs (Portable Game Notation), source du dataset via pgnmentor.com.

## 4. Fonctionnalités

### 4.1 Dataset échecs (positions → labels policy/value)

**Description :** Construction d'un dataset d'entraînement à partir de parties PGN de grands joueurs (pgnmentor.com/files.html#players), rejouées coup par coup via `python-chess`. Chaque demi-coup produit un exemple (position → policy, value), sans filtrage par résultat sur la policy (design retenu après rejet de l'hypothèse initiale "coups du gagnant = True" — détail du raisonnement en `addendum.md` du brief).

#### FR-1 : Extraction du dataset depuis les archives PGN

Le pipeline construit un dataset de positions à partir d'archives PGN par joueur (pgnmentor.com), en rejouant chaque partie coup par coup via `python-chess`.

**Conséquences (vérifiables) :**
- Une partie PGN de N demi-coups produit N exemples de dataset (une position par demi-coup).
- Aucune détermination de gagnant n'est nécessaire pour extraire la policy — seul le résultat final de partie sert au label value.
- Aucun moteur d'échecs externe (Stockfish ou autre) n'est utilisé pour générer un label (voir NFR-2).
- Chaque archive pgnmentor traitée représente quelques dizaines de milliers de parties d'un même grand joueur (victoires/défaites/nulles confondues) — ordre de grandeur à prendre en compte pour calibrer la taille de batch et le nombre d'epochs en architecture.

#### FR-2 : Encodage de l'input

Chaque position est encodée en planes façon AlphaZero (pièces, trait, roques, répétition, coups légaux), enrichies de l'historique des 5 derniers coups des deux joueurs.

**Conséquences (vérifiables) :**
- L'encodage produit une représentation consommée directement par le modèle défini en FR-5, sans étape de transformation manuelle supplémentaire côté `TaskStrategy`.

#### FR-3 : Construction des labels policy et value

La policy head est entraînée à imiter le coup réellement joué à chaque position, Blancs et Noirs confondus, sans filtrage par résultat de partie. La value head porte le résultat final de la partie, du point de vue du joueur au trait à cette position (+1/0/-1, nulles conservées avec value = 0).

**Conséquences (vérifiables) :**
- Aucun exemple de policy n'est exclu du dataset au motif que le camp qui l'a joué a perdu la partie.
- Chaque exemple porte exactement un label policy (le coup joué) et un label value (résultat côté joueur au trait).

### 4.2 Intégration du domaine échecs dans le pipeline générique

**Description :** Le domaine échecs est branché sur `Trainer` via les mêmes points d'extension que les domaines précédents (nouveau `model_name`, nouvelle entrée `dataset_configs.py`, nouveau `task_type`), avec pour différence structurelle une `TaskStrategy` à double tête — jamais requise jusqu'ici. C'est le cœur du test de généricité de cette epic.

#### FR-4 : Nouvelle `TaskStrategy` à double tête (policy + value)

Une nouvelle `TaskStrategy` gère l'entraînement conjoint de la tête policy et de la tête value (deux losses), sur le modèle du pattern Strategy existant.

**Conséquences (vérifiables) :**
- La `TaskStrategy` échecs s'intègre à `Trainer` via l'interface Strategy existante, sans que `Trainer` ait besoin de connaître l'existence de deux têtes/deux losses.
- Les deux losses (policy, value) sont visibles séparément dans les logs d'entraînement.

**`[NOTE FOR PM]`** Conception de la double tête à trancher en session d'architecture (Winston) — voir §8 Open Questions, item 4.

#### FR-5 : Nouveau modèle échecs

Un nouveau modèle implémente l'architecture envisagée : CNN 8×8 (convolutions + blocs résiduels) → bottleneck de tokens → auto-attention entre tokens → tête policy + tête value.

**Conséquences (vérifiables) :**
- Le modèle est enregistré comme un `model_name` distinct, au même titre que les modèles existants (`aircraft_detector_centernet`, `kepler_1d_cnn`, etc.).
- Le modèle produit deux sorties distinctes (policy, value) consommables par la `TaskStrategy` de FR-4.

**`[NOTE FOR PM]`** Plusieurs choix de conception (construction des tokens, biais géométrique dans l'attention) restent ouverts — voir §8 Open Questions, items 1 et 3. Justification de l'attention (raisonnement relationnel entre pièces, pas couverture de receptive field) et précédents (Maia-3, AlphaVile, DeepMind searchless) détaillés dans l'addendum du brief.

#### FR-6 : Intégration sans modification structurelle de `Trainer`

Le domaine échecs est intégré sans modifier `Trainer` au-delà de ce que sa généricité actuelle permet déjà (nouveaux modèles, nouvelles losses, nouvelles méthodes d'évaluation acceptés ; changement structurel du cœur générique non accepté).

**Conséquences (vérifiables) :**
- Aucune modification du fichier `Trainer` lui-même n'est nécessaire pour absorber le double-loss policy/value — celui-ci est encapsulé entièrement dans la `TaskStrategy` de FR-4.
- Si une modification de `Trainer` s'avère nécessaire en cours de route, elle est documentée explicitement comme un écart au critère de succès de cette epic (voir §7 Success Metrics, SM-1), pas absorbée silencieusement. Le constat de nécessité et l'arbitrage reviennent à Aymeric, seul porteur du projet.

**Out of Scope :** L'encodage de l'espace de sortie policy (ex. espace fixe de coups à la AlphaZero, avec masquage des coups illégaux) impacte directement la faisabilité de ce FR — ce choix est tranché en architecture, pas ici (voir §8 Open Questions, item 2).

### 4.3 Non-régression `JAX_DETECTOR`

**Description :** `JAX_DETECTOR`, le pipeline de production avion actuel, doit continuer à fonctionner sans aucun impact de cette epic — même invariant de non-régression que celui posé par AD-20 pour `FIGHTERJET_DETECTION` lors de l'initiative JAX Single-Pass.

#### FR-7 : Story de validation de non-régression par exécution

Une story dédiée valide, par exécution réelle (pas par lecture de code), que `JAX_DETECTOR` produit un comportement identique avant et après l'intégration du domaine échecs.

**Conséquences (vérifiables) :**
- Une baseline (boxes/classes/scores sur un set fixe d'images de référence) est capturée avant l'intégration du domaine échecs.
- Un entraînement et/ou une inférence `JAX_DETECTOR` lancés après l'epic sont comparés à cette baseline par diff — mêmes images, mêmes métriques ; tout écart est un échec de la story, pas une simple observation qualitative.
- La méthode reprend celle des Stories 2.4/8.9 des epics précédentes (comparaison de baseline par exécution, pas lecture de code).

## 5. Non-Goals (Explicit)

- **Qualité de jeu du modèle** — non évaluée dans cette epic. Aucun objectif Elo, taux de coups corrects, ou benchmark de force de jeu n'est un critère de clôture.
- **Intégration dans `./chess/chess_game.py`** — ce fichier existant (plateau Tkinter, 2 joueurs humains, barre d'avantage optionnelle via Stockfish avec repli en évaluation matérielle) reste un banc de test **futur**. Brancher le modèle entraîné pour qu'il propose réellement un coup est un travail distinct, non traité dans cette epic.
- **Toute nouvelle dépendance à un moteur d'échecs externe** pour générer des labels d'entraînement — exclu par construction, y compris comme option de repli si la value head s'avère bruitée (voir §8 Open Questions).
- **Nouveau travail sur les domaines existants** (CIFAR10, FIGHTERJET_*, KEPLER) — cette epic ne les modifie pas, au-delà de ce qu'exige la non-régression `JAX_DETECTOR` (FR-7). `[ASSUMPTION: ce non-goal n'est pas énoncé explicitement dans le brief — inféré du périmètre "temps 1" de l'epic ; à confirmer.]`

## 6. Exigences transverses (NFR)

- **NFR-1 — Non-régression `JAX_DETECTOR` :** contrainte dure, non négociable. Détail et méthode de validation en FR-7.
- **NFR-2 — Aucune dépendance à un moteur d'échecs externe pour la génération de labels :** ni pour la policy, ni pour la value, y compris en repli. Détail en FR-1 et Non-Goals §5.
- **NFR-3 — Pas de complexification inutile du pipeline générique :** `Trainer`/`TaskStrategy` absorbent le domaine échecs par les mêmes points d'extension que les domaines précédents. Détail en FR-6.

## 7. Success Metrics

*Enjeu interne/epic technique — critères qualitatifs et binaires suffisent, pas de tableau de bord quantitatif.*

**Primary**
- **SM-1** : Le domaine échecs est branché via un nouveau modèle, un nouveau builder de dataset et une nouvelle `TaskStrategy`, sans modification de `Trainer` au-delà de ce que sa généricité actuelle permet déjà. Valide FR-4, FR-5, FR-6, NFR-3.
- **SM-2** : `JAX_DETECTOR` produit un résultat identique avant/après l'epic, vérifié par exécution réelle. Valide FR-7, NFR-1.

**Secondary**
- **SM-3** : Le dataset échecs (positions + labels policy/value) est généré sans dépendance à un moteur d'échecs externe. Valide FR-1, FR-2, FR-3, NFR-2.

**Counter-metrics (à ne pas optimiser)**
- **SM-C1** : La qualité de jeu du modèle (force, taux de coups "corrects", Elo estimé, etc.) n'est **pas** un critère de clôture de cette epic. Ne pas complexifier l'architecture ni prolonger l'entraînement pour l'optimiser dans ce cycle — son évaluation est reportée à un projet séparé. Contrebalance SM-1 : évite de sacrifier la simplicité du pipeline générique pour gagner en performance de jeu.

## 8. Open Questions

Points identifiés dans le brief comme à trancher en session d'architecture (Winston), pas dans ce PRD :

1. **Construction des tokens du bottleneck** — pooling par groupe de canaux vs. requêtes apprises type Perceiver/TokenLearner.
2. **Encodage de l'espace de sortie policy** — espace fixe de coups avec masquage des coups illégaux (à la AlphaZero) ou alternative. Impacte directement la faisabilité de FR-6 (« sans modification structurelle de `Trainer` »).
3. **Biais géométrique explicite dans l'attention** (à la Maia-3) ou attention "nue".
4. **Conception détaillée de la `TaskStrategy` à double tête** (FR-4) — aucune tâche existante n'a ce besoin ; test réel de la généricité du pattern `TaskStrategy` actuel.
5. **Risques dataset non résolus** (littérature Maia/DeepMind, détail en addendum du brief) : bruit des coups humains (blunders), mélange des cadences blitz/classique (à vérifier si pgnmentor distingue les cadences), déséquilibre de classes par ouverture, et prévisibilité non monotone selon le niveau du joueur (chez Maia, les joueurs de niveau intermédiaire sont les plus prévisibles — faibles et forts le sont moins, ce qui questionne un pooling naïf de parties de GM). À traiter en architecture ou dans la première story de validation du dataset.

## 9. Assumptions Index

- §2 — JTBD reformulé à partir du memlog du brief (finalité "modèle jouable à terme" via le futur banc de test `chess_game.py`), pas une citation directe d'Aymeric — à confirmer si la formulation dévie de l'intention.
- §5 — Non-Goal "pas de nouveau travail sur les domaines existants" n'est pas énoncé explicitement dans le brief — inféré du périmètre "temps 1" de l'epic ; à confirmer.
- Portée de ce PRD limitée au contenu du brief `brief-jax_supervised_training-2026-07-27` + addendum ; aucun élément supplémentaire n'a été apporté en brain dump lors de cette session — à signaler si un contexte non capturé existe (ex. contraintes de calcul/GPU, session d'architecture déjà tenue).
