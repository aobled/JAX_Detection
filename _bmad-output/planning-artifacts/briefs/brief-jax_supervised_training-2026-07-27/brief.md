---
title: "Brief produit — Généralisation du pipeline (validation par un domaine tiers : moteur d'échecs)"
status: validated
created: 2026-07-27
updated: 2026-07-27
---

# Brief produit : preuve de généralisation du pipeline via un moteur d'échecs

## Résumé exécutif

Le pipeline `jax_supervised_training` a déjà prouvé sa généricité sur plusieurs domaines à base d'images ou de séries 1D (CIFAR10, FIGHTERJET_CLASSIFICATION/DETECTION, JAX_DETECTOR, KEPLER), tous partageant une nature commune : une entrée à géométrie fixe mappée vers une sortie de classification ou de détection.

Les échecs testent une **rupture de nature différente** : l'entrée est un état de jeu structuré (plateau + historique de coups + règles), et la sortie attendue est une paire **policy** (distribution sur les coups légaux) / **value** (évaluation de position) — le premier domaine du projet qui n'est ni classification ni détection au sens classique.

**Objectif de cette epic (temps 1)** : valider que le pipeline reste utilisable pour ce type de tâche. De nouveaux modèles, de nouvelles fonctions de perte et de nouvelles méthodes d'évaluation sont attendus et acceptés — ce n'est pas un critère d'échec. Le critère de succès est que le pipeline générique (`Trainer`, `task_strategies`) **n'est pas complexifié inutilement** pour absorber ce nouveau domaine.

**Contrainte dure, non négociable** : `JAX_DETECTOR` (le pipeline de production avion actuel) doit continuer à fonctionner **sans aucun impact** de cette epic — le même invariant de non-régression que celui posé par AD-20 pour `FIGHTERJET_DETECTION` lors de l'initiative JAX Single-Pass.

La qualité de jeu du modèle résultant n'est pas un critère de clôture de cette epic ; elle sera évaluée ultérieurement, dans un projet séparé, via un petit programme interactif existant basé sur `python-chess`.

## Ce qui est déjà prouvé

| Domaine | Type d'input | `task_type` | Modèle | Sortie |
|---|---|---|---|---|
| CIFAR10 | Images 32×32 RGB | `classification` | `sophisticated_cnn_32_plus` | classe |
| FIGHTERJET_CLASSIFICATION | Images 128×128 | `classification` | `sophisticated_cnn_128_lite` | classe (~30 avions) |
| FIGHTERJET_DETECTION | Images | `detection` | `aircraft_detector_unet` | masque/boîtes (UNet) |
| JAX_DETECTOR | Images | `detection_centernet` | `aircraft_detector_centernet` | heatmap + taille (CenterNet) |
| KEPLER | Séries 1D (courbes de lumière) | `kepler` | `kepler_1d_cnn` | classe |

Chaque nouveau domaine a jusqu'ici nécessité : un nouveau `model_name`, une entrée `dataset_configs.py` dédiée, et pour KEPLER/JAX_DETECTOR un nouveau `task_type` avec sa propre `TaskStrategy`. Aucun n'a nécessité de modifier `Trainer` lui-même. C'est cette régularité que l'epic échecs doit confirmer ou infirmer.

**Échecs (à créer)** : `task_type` nouveau (nom à trancher en architecture), modèle à définir (voir ci-dessous), sortie = paire policy (distribution sur coups légaux) + value (évaluation position) — première sortie du projet qui n'est ni une classe ni une carte spatiale.

## Architecture envisagée

CNN 8×8 (convs + residuals) → bottleneck de tokens → self-attention entre tokens → policy head (distribution sur les coups) + value head (évaluation de position).

**Justification** : l'attention n'apporte pas de bénéfice de receptive field sur un plateau 8×8 — un empilement de 3-4 convs 3×3 le couvre déjà entièrement. Son intérêt réel est le raisonnement relationnel entre pièces spécifiques (clouages, fourchettes, attaques à la découverte). C'est la direction prise par **Maia-3** (CSSLab), le précédent le plus proche de ce design (imitation de coups humains, pas du meilleur coup) : parti d'un CNN résiduel pur (Maia/Maia-2), il a migré vers un transformer doté d'un *Geometric Attention Bias* dédié — un signal favorable à l'hypothèse CNN+tokens+attention. Détails et sources en annexe (`addendum.md`).

**Points ouverts, à trancher en session d'architecture (Winston), pas dans ce brief** :
- construction des tokens du bottleneck (pooling par groupe de canaux vs requêtes apprises type Perceiver/TokenLearner) ;
- encodage de l'espace de sortie policy (ex. AlphaZero utilise un espace fixe de coups avec masquage des coups illégaux) — ce choix impacte directement la faisabilité du "sans impact sur `Trainer`" posé en critère de succès ;
- biais géométrique explicite dans l'attention (à la Maia-3) ou attention "nue" ;
- conception d'une nouvelle `TaskStrategy` à double tête (policy + value, deux losses) — aucune tâche existante n'a ce besoin ; c'est en soi un test de la généricité réelle du pattern `TaskStrategy` actuel, pas un détail d'implémentation secondaire.

## Dataset & construction des labels

**Source** : parties PGN récupérées sur [pgnmentor.com/files.html#players](https://www.pgnmentor.com/files.html#players) — archives par grand joueur (quelques dizaines de milliers de parties), victoires/défaites/nulles confondues.

**Construction** : chaque partie est rejouée coup par coup via `python-chess`. À chaque demi-coup, on extrait un exemple :
- **Input** : plateau (planes façon AlphaZero — pièces, trait, roques, répétition, coups légaux) + historique des 5 derniers coups des deux joueurs.
- **Target policy** : le coup réellement joué à cet instant — imité tel quel, Blancs et Noirs confondus, sans filtrage par résultat de partie.
- **Target value** : résultat final de la partie, du point de vue du joueur au trait (+1/0/-1). Les nulles sont conservées (value = 0).

Aucun moteur externe (Stockfish ou autre) n'est utilisé pour générer un label — contrainte explicite du projet.

**Risques connus, documentés mais non résolus dans ce brief** (issus de la littérature — voir `addendum.md`) : bruit des coups humains (blunders ponctuels, inévitables même chez les GM) ; mélanger cadences blitz et classique est risqué, ces dernières reflétant des processus de décision différents — à vérifier si pgnmentor distingue les cadences ; déséquilibre de classes par ouverture. À traiter en architecture ou lors de la première story de validation du dataset.

## Hors scope explicite

- **Qualité de jeu du modèle** — non évaluée dans cette epic (voir Résumé exécutif).
- **Intégration dans `./chess/chess_game.py`** — ce fichier existant est un plateau Tkinter manuel (2 joueurs humains, aucune IA), avec une barre d'avantage optionnelle via Stockfish (repli gracieux en évaluation matérielle si Stockfish absent — Stockfish n'est volontairement pas ajouté à ce dépôt). Il servira de base de test future, mais brancher le modèle entraîné pour qu'il propose réellement un coup est un travail distinct, non traité ici.
- **Toute nouvelle dépendance à un moteur d'échecs externe** pour générer des labels — exclu par construction (voir Dataset).

## Critères de succès

- Un nouveau domaine (échecs) est branché au pipeline via un nouveau modèle, un nouveau builder de dataset, et une nouvelle `TaskStrategy` — sans modification de `Trainer` au-delà de ce que la généricité actuelle permet déjà.
- `JAX_DETECTOR` continue de fonctionner à l'identique — non-régression vérifiée par une story de validation explicite et par exécution (pas seulement par lecture de code), dans l'esprit des Stories 2.4/8.9 des epics précédentes.
- Le dataset échecs (positions + labels policy/value) est généré sans dépendance à un moteur d'échecs externe.
- La qualité de jeu du modèle n'est pas un critère de clôture — son évaluation est explicitement reportée à un projet séparé.
