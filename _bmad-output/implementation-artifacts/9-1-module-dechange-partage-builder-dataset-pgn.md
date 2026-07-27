---
baseline_commit: 1a85f2583c10770dcbeb84140aaac9fe842781f7
---

# Story 9.1: Module d'échange partagé + builder dataset PGN

Status: done

<!-- Note: Validation is optional. Run validate-create-story for quality check before dev-story. -->

## Story

As a mainteneur du pipeline,
I want un module `chess_target_encoding.py` définissant le schéma position/policy/value et un outil `dataset_builder/chess_pgn_dataset_tools.py` qui construit le dataset d'entraînement depuis des archives PGN,
so that le domaine échecs dispose d'une source unique de vérité pour le format d'échange (AD-18 hérité) avant que tout consommateur (modèle Story 9.2, `TaskStrategy` Story 9.3) ne soit développé — même principe que la Story 7.1 pour AD-18/heatmap+taille.

## Acceptance Criteria

1. **Given** des archives PGN pgnmentor (par joueur, plusieurs parties concaténées dans un même fichier) **When** `chess_pgn_dataset_tools.py` les traite via `python-chess` (`chess.pgn.read_game`, rejeu par `board.push(move)`) **Then** chaque partie de N demi-coups produit N exemples (position, policy target, value target) — aucune position n'est exclue, y compris les toutes premières du début de partie (FR1, AD-25).
2. **Given** une position à un demi-coup donné **When** elle est encodée par `chess_target_encoding.py::encode_position` **Then** elle produit des planes façon AlphaZero (pièces, trait, roques, répétition) + un plan coups légaux + l'historique des 5 derniers demi-coups (source unique documentée dans la docstring du module, contrat AD-18) (FR2).
3. **Given** une position avec moins de 5 demi-coups joués depuis le début de la partie **When** ses créneaux d'historique manquants sont encodés **Then** ils sont remplis à zéro (padding), jamais exclus du dataset ni remplacés par une répétition de la position de départ — pour que le modèle apprenne aussi les ouvertures classiques (FR2, précision Aymeric 2026-07-27).
4. **Given** un coup joué à une position **When** la policy target est construite **Then** c'est l'index entier du coup réellement joué dans l'espace de coups fixe (Blancs et Noirs confondus), sans filtrage par résultat de partie (FR3).
5. **Given** le résultat final d'une partie **When** la value target est construite pour une position donnée **Then** c'est un scalaire +1/0/-1 du point de vue du joueur au trait à cette position (nulles = 0), calculé et figé une seule fois par le producteur — jamais recalculé/re-dérivé côté consommateur (AD-25).
6. **Given** AD-25/NFR2 **When** le dataset est généré **Then** aucun moteur d'échecs externe (Stockfish ou autre) n'est utilisé à aucune étape — seul `python-chess` sert à la légalité des coups et au rejeu PGN.
7. **Given** AD-18/AD-22 **When** `chess_target_encoding.py` est créé **Then** il expose une constante nommée unique `NUM_MOVES` (taille de l'espace de sortie policy), destinée à être importée telle quelle par tout consommateur futur (Story 9.2 pour dimensionner la tête policy, Story 9.3) — jamais un littéral dupliqué ailleurs.
8. **Given** le pattern de chunking déjà établi par ce projet (`dataset_builder/jax_detector_dataset_tools.py`, `cifar10_classification_dataset_tools.py`) **When** `chess_pgn_dataset_tools.py` écrit le dataset sur disque **Then** il produit des chunks `.npz` compressés (plusieurs exemples/fichier) réutilisant les constantes de clé exposées par `chess_target_encoding.py` — jamais des noms de clés choisis localement dans l'outil.

## Tasks / Subtasks

- [x] Task 1: Créer `chess_target_encoding.py` (nouveau fichier racine, convention plate du projet — voir Dev Notes § Project Structure) (AC: 2, 3, 4, 5, 7)
  - [x] Constantes de clé `.npz` — `POSITION_KEY = "position"`, `POLICY_KEY = "policy"`, `VALUE_KEY = "value"` (miroir direct de `HEATMAP_KEY`/`SIZE_KEY` dans `detection_target_encoding.py`)
  - [x] `encode_position(board: chess.Board, move_history: list) -> np.ndarray` — planes façon AlphaZero (voir Dev Notes § Schéma d'encodage de position pour le détail exact des plans)
  - [x] `NUM_MOVES` constante entière (voir Dev Notes § Schéma d'encodage policy pour la valeur exacte et sa justification)
  - [x] `move_to_index(move: chess.Move, board: chess.Board) -> int` et `index_to_move(index: int, board: chess.Board) -> chess.Move` — encode/decode du coup dans l'espace fixe `NUM_MOVES` (voir Dev Notes)
  - [x] Docstring de tête citant explicitement les consommateurs prévus (Story 9.2 : modèle, dimensionne sa tête policy sur `NUM_MOVES` ; Story 9.3 : `TaskStrategy`/chargeur `data_management.py`) — même discipline que `detection_target_encoding.py` pour AD-18
- [x] Task 2: Créer `dataset_builder/chess_pgn_dataset_tools.py` (producteur, importe `chess_target_encoding.py`) (AC: 1, 3, 4, 5, 6, 8)
  - [x] Parcours d'une archive PGN multi-parties via `chess.pgn.read_game(pgn_handle)` en boucle jusqu'à `None`
  - [x] Pour chaque partie : rejeu coup par coup (`board.push(move)`), à chaque demi-coup extraire `(encode_position(board, historique), move_to_index(move, board), value_du_resultat)` **avant** de jouer le coup (le coup joué à cette position est la cible policy)
  - [x] Calcul de la value : résultat final de partie (`game.headers["Result"]`, `"1-0"`/`"0-1"`/`"1/2-1/2"`) converti en +1/0/-1 du point de vue du joueur au trait à la position courante (inverser le signe selon la couleur au trait, pas juste selon le vainqueur global) — calculé une fois par partie, appliqué à chaque position de cette partie (AC: 5)
  - [x] Écriture en chunks `.npz` compressés (`np.savez_compressed`, clés `POSITION_KEY`/`POLICY_KEY`/`VALUE_KEY` empilées sur l'axe N), même pattern que `jax_detector_dataset_tools.py`/`cifar10_classification_dataset_tools.py` — `chunk_size` en paramètre, pas de dimension batch dans `chess_target_encoding.py` lui-même (même distinction que `detection_target_encoding.py` § "Persistance par lot")
  - [x] Aucun import ni appel à un moteur d'échecs externe (`chess.engine`, Stockfish) — vérifié explicitement (test statique dédié) avant de considérer la tâche terminée (AC: 6)
- [x] Task 3: Script/test de validation autonome (pas de framework de test formel dans ce projet — voir Dev Notes § Testing Standards) (AC: 1, 2, 3, 4, 5, 7)
  - [x] Round-trip `move_to_index` / `index_to_move` sur un échantillon de coups incluant : coup normal, roque (court et long), prise en passant, promotion (dame et sous-promotion, Blancs **et** Noirs — la logique couleur-relative est le point le plus fragile de l'implémentation), et un balayage exhaustif de tous les coups légaux sur 40 demi-coups d'une partie semi-aléatoire + 8 demi-coups d'ouverture
  - [x] Vérifié qu'une position à 0 demi-coup (position de départ) produit bien des planes d'historique à zéro sur les 5 créneaux, et qu'un padding partiel (2 coups joués → 2 créneaux remplis, 3 à zéro) est correct (AC: 3)
  - [x] Vérifié sur une petite partie PGN de test (8 demi-coups) que le nombre d'exemples produits == nombre de demi-coups de la partie (AC: 1)
  - [x] Vérifié que la value change de signe selon le joueur au trait, sur 3 parties de test (victoire Blancs, victoire Noirs, nulle) couvrant les 3 résultats possibles (AC: 5)

## Dev Notes

### Portée exacte (ne pas dépasser)

- Cette story crée **uniquement** `chess_target_encoding.py` et `dataset_builder/chess_pgn_dataset_tools.py`. Elle ne touche **aucun** fichier existant (`dataset_configs.py`, `model_library.py`, `task_strategies.py`, `data_management.py`, `main.py` — zéro fichier UPDATE). L'entrée `CHESS` dans `dataset_configs.py`, le modèle et la `TaskStrategy` sont les Stories 9.2/9.3, hors scope ici.
- Pas de masquage des coups illégaux (AD-22) — `move_to_index`/`index_to_move` encodent/décodent un coup donné, ils n'ont pas besoin de connaître ni de filtrer l'ensemble des coups légaux d'une position pour cette story. Le plan "coups légaux" de `encode_position` (AC 2) sert uniquement de feature d'entrée au modèle, pas à un mécanisme de masquage de sortie.
- Aucune métadonnée de cadence/ECO/identité de partie n'est capturée ni persistée (AD-25) — le format d'exemple porte exactement `POSITION_KEY`/`POLICY_KEY`/`VALUE_KEY`, rien de plus.

### Schéma d'encodage de position (décision de cette story, sourcée — pas fixée par le PRD/spine)

Le PRD (FR2) et le spine (AD-22, Deferred) laissent le schéma exact à la story. Proposition, à ajuster si besoin en implémentation :

**Plans de position courante (19 plans 8×8) :**
- 6 plans pièces propres (Roi, Dame, Tour, Fou, Cavalier, Pion) — binaire, 1 si la pièce est présente sur la case
- 6 plans pièces adverses (mêmes 6 types)
- 1 plan trait — constante uniforme (tout à 1 si Blancs au trait, tout à 0 si Noirs)
- 4 plans droits de roque (roque court/long × Blancs/Noirs) — constante uniforme, lire directement `board.has_kingside_castling_rights(color)`/`has_queenside_castling_rights(color)`
- 1 plan répétition — `board.is_repetition(2)` (ou compteur normalisé), indicateur binaire suffisant pour cette epic
- 1 plan coups légaux — case de **destination** des coups légaux marquée à 1 (`board.legal_moves`, itérer `move.to_square`) ; convention simple retenue pour cette story, ajustable si le modèle (Story 9.2) montre qu'une autre représentation (par case de départ, ou 2 plans départ+arrivée) apporte plus de signal

Convention AlphaZero standard pour les plans pièces/trait/roques/répétition (Silver et al., *Mastering Chess and Shogi by Self-Play*, 2017/2018) — reprise ici sans le stacking historique de positions T=8 de l'article original (le PRD demande un historique de **coups**, pas de positions empilées, voir ci-dessous).

**Historique des coups (2 plans par créneau × 5 créneaux = 10 plans) :**
- Chaque créneau d'historique = 1 demi-coup, encodé en 2 plans binaires 8×8 : case de départ (1 plan, `move.from_square`) + case d'arrivée (1 plan, `move.to_square`)
- Fenêtre fixe des **5 derniers demi-coups** (pas 5 par joueur — confirmé par la formulation PRD "historique des 5 derniers coups des deux joueurs" = 5 demi-coups au total, alternant naturellement entre les deux joueurs)
- Créneau manquant (position à moins de 5 demi-coups joués) : les 2 plans de ce créneau sont entièrement à zéro (AC 3, décision Aymeric 2026-07-27) — jamais une répétition de la position de départ, jamais une exclusion de la position du dataset

**Total : 19 + 10 = 29 plans, tenseur `(8, 8, 29)`.** Décision nouvelle pour cette story (pas héritée d'une source externe) — si l'implémentation ou la Story 9.2 révèlent qu'un nombre de plans différent sert mieux le modèle, documenter le changement dans cette story plutôt que le laisser divergent entre `chess_target_encoding.py` et sa docstring.

### Schéma d'encodage policy (NUM_MOVES) — schéma d'action standard AlphaZero

Aymeric a confirmé l'option "espace de coups fixe façon AlphaZero" (session d'architecture 2026-07-27). Le schéma d'action publié par AlphaZero (Silver et al., 2018) et repris par de nombreuses implémentations publiques (ex. projets open-source "alpha-zero-general" et dérivés) encode un coup comme `(case_source, type_de_coup)` :

- 64 cases source possibles
- 73 types de coup par case source :
  - 56 = 8 directions (N, NE, E, SE, S, SW, W, NW) × 7 distances (1 à 7 cases) — couvre tous les coups de type "dame" (tour + fou), y compris les coups de roi normaux (distance 1) et le roque (représenté comme un déplacement de roi de 2 cases, une des 56 combinaisons)
  - 8 = coups de cavalier (les 8 déplacements en L)
  - 9 = sous-promotions (3 directions : tout droit, capture gauche, capture diagonale droite × 3 pièces de sous-promotion : cavalier, fou, tour) — une promotion en dame est un coup de pion normal (distance 1, une des 56 "dame"), pas une sous-promotion
- `NUM_MOVES = 64 * 73 = 4672`
- La prise en passant est un coup de pion diagonal normal du point de vue de cet encodage (case source/destination), donc couverte par les 56 types "dame" sans traitement spécial — `python-chess` expose déjà `board.is_en_passant(move)` si un traitement distinct s'avérait nécessaire en implémentation, mais l'encodage `(source, type)` n'en a pas besoin

`move_to_index(move, board)` : dériver `(case_source, type_de_coup)` depuis `move.from_square`/`move.to_square`/`move.promotion` (objet `chess.Move` de python-chess, déjà utilisé dans `chess/chess_game.py`), calculer `case_source * 73 + type_de_coup`. `index_to_move` fait l'inverse. Documenter la table exacte direction/distance/sous-promotion → index dans la docstring de ces deux fonctions (contrat AD-18 : c'est la source unique de vérité pour cette conversion, jamais réimplémentée côté Story 9.2/9.3).

### Calcul de la value (résultat de partie signé)

`game.headers["Result"]` (python-chess) vaut `"1-0"` (Blancs gagnent), `"0-1"` (Noirs gagnent), ou `"1/2-1/2"` (nulle). Pour une position donnée où `board.turn` indique le joueur au trait :
- Résultat "1-0" : value = +1 si Blancs au trait, -1 si Noirs au trait
- Résultat "0-1" : value = -1 si Blancs au trait, +1 si Noirs au trait
- Résultat "1/2-1/2" : value = 0 dans tous les cas

Ce signe est calculé **une fois par position** au moment de l'écriture de l'exemple (le producteur, AD-25) — jamais recalculé ou ré-inversé côté `data_management.py`/`TaskStrategy` (Story 9.3).

### Project Structure Notes

- `chess_target_encoding.py` — nouveau fichier **racine**, convention plate déjà en place (`detection_target_encoding.py`, `loss_functions.py`, `task_strategies.py` sont tous racine, pas de package imbriqué).
- `dataset_builder/chess_pgn_dataset_tools.py` — nouveau fichier dans `dataset_builder/`, même dossier que `jax_detector_dataset_tools.py`, `fighterjet_detection_dataset_tools.py`, `cifar10_classification_dataset_tools.py`, `kepler_dataset_tools.py` (vérifié : ce dossier existe déjà et contient exactement ce type d'outil).
- `python-chess` (import `chess`, `chess.pgn`) : déjà une dépendance du projet (v1.11.2, vérifié `pip show chess`), déjà utilisé dans `chess/chess_game.py` (`chess.Board()`, `board.push(move)`, `board.legal_moves`) — mais `chess.pgn` (lecture de fichiers PGN multi-parties) n'a encore aucun usage dans ce repo ; c'est une nouvelle surface d'API de la même librairie, pas une nouvelle dépendance.
- Aucun conflit détecté avec la structure existante.

### Testing Standards

Pas de suite de tests automatisée formelle dans ce projet (confirmé PRD historique Epic 1-3 : validation par script autonome/comparaison, pas de CI/CD). Pour cette story : un script de round-trip autonome (Task 3) suffit, dans l'esprit de `tests/test_detection_target_encoding.py` (Story 7.1) — pas de framework à introduire.

### References

- [Source: `_bmad-output/planning-artifacts/prds/prd-jax_supervised_training-2026-07-27/prd.md#FR-1,FR-2,FR-3`] — construction du dataset, encodage input, labels policy/value
- [Source: `_bmad-output/planning-artifacts/architecture/architecture-chess-2026-07-27/ARCHITECTURE-SPINE.md#AD-22,AD-25`] — espace policy fixe sans masquage, format d'exemple minimal, zéro moteur externe, propriétaire unique du signe value
- [Source: `_bmad-output/planning-artifacts/architecture/architecture-chess-2026-07-27/ARCHITECTURE-SPINE.md#AD-18`] — module d'échange partagé à source unique (hérité du parent 2026-07-15)
- [Source: `detection_target_encoding.py`] — modèle direct à mirroir : docstring de contrat, constantes de clé, séparation encode/decode single-example vs persistance chunk
- [Source: `dataset_builder/jax_detector_dataset_tools.py`, `dataset_builder/cifar10_classification_dataset_tools.py`] — pattern de chunking `.npz` établi (`np.savez_compressed`, `chunk_size`, noms de clés singuliers)
- [Source: `chess/chess_game.py`] — usage déjà établi de `python-chess` dans ce repo (`chess.Board`, `board.push`, `board.legal_moves`)
- [Source: `_bmad-output/planning-artifacts/briefs/brief-jax_supervised_training-2026-07-27/brief.md`, `addendum.md`] — contexte dataset (pgnmentor, design de label retenu après rejet de "coups du gagnant = True")
- [Source: `_bmad-output/planning-artifacts/epics.md` § Epic 9, Story 9.1] — story source, ACs
- [Source: `_bmad-output/implementation-artifacts/7-1-definition-du-schema-dechange-heatmap-taille-ad-18.md`] — précédent projet pour une story "auteur unique" prérequis AD-18 (même pattern qu'ici)

## Dev Agent Record

### Agent Model Used

Claude Sonnet 5

### Debug Log References

- `python3 tests/test_chess_target_encoding.py` — 9/9 tests passés (round-trip coups légaux position de départ, roque, prise en passant, promotion Blancs 4 pièces, promotion Blancs avec capture 12 coups, promotion **Noirs** 12 coups, balayage exhaustif 40 demi-coups partie semi-aléatoire, balayage exhaustif 8 demi-coups d'ouverture, shape+padding historique position de départ, padding partiel 2/5 créneaux, comptage exemples + alternance value sur partie de test 8 demi-coups)
- `python3 tests/test_chess_pgn_dataset_tools.py` — 2/2 tests passés (absence de dépendance moteur externe vérifiée statiquement ; construction dataset sur 3 parties de test PGN synthétique — 14 positions, 3 chunks 5/5/4, clés/shapes/values vérifiées pour les 3 résultats possibles 1-0/0-1/1-2-1-2)
- `python3 tests/test_detection_target_encoding.py` — 6/6 tests passés (non-régression : story ne touche aucun fichier existant, suite pré-existante repassée par précaution)
- Bug trouvé et corrigé pendant l'implémentation : FEN de test initiale pour `test_roundtrip_promotion_all_pieces` plaçait le roi noir sur la case de promotion (a8), bloquant tout coup de promotion (0 coups légaux au lieu de 4 attendus) — erreur de construction du test, pas un bug du module ; corrigée en déplaçant le roi noir en h8
- Après code review (2026-07-27, Blind Hunter + Edge Case Hunter + Acceptance Auditor) : 11 patchs appliqués (voir § Review Findings). `python3 tests/test_chess_target_encoding.py` — 13/13 passés (10 précédents + `test_contract_constants_pinned`, aucune régression). `python3 tests/test_chess_pgn_dataset_tools.py` — 2/2 passés, comportement identique (14 positions, 3 chunks 5/5/4) après ajout de la robustesse PGN malformé/progression/compteurs. `python3 tests/test_detection_target_encoding.py` — 6/6 repassés (non-régression confirmée à nouveau)

### Completion Notes List

- Implémenté `chess_target_encoding.py` (racine) : `encode_position` (planes (8,8,29) — 19 position courante façon AlphaZero + 10 historique 5 demi-coups × 2 plans départ/arrivée, padding à zéro vérifié pour les positions de début de partie) et `move_to_index`/`index_to_move` (espace d'action standard AlphaZero, `NUM_MOVES=4672` = 64 cases source × 73 types de coup — 56 coups "dame" + 8 cavalier + 9 sous-promotions). Décision d'implémentation non fixée par le PRD/spine, documentée dans la docstring de module : encodage à **cases absolues**, sans flip d'orientation selon la couleur au trait (plus simple pour cette première itération — un flip par couleur reste une piste d'amélioration future, pas un oubli).
- Implémenté `dataset_builder/chess_pgn_dataset_tools.py` (producteur) : `build_chess_dataset(pgn_paths, output_prefix, chunk_size)` parcourt une ou plusieurs archives PGN multi-parties (`chess.pgn.read_game`), rejoue chaque partie (`board.push`), extrait un exemple par demi-coup (policy = coup joué, non filtré par résultat — FR3), calcule la value signée une seule fois par position (`_value_for_mover`, jamais recalculée côté consommateur — AD-25), et écrit des chunks `.npz` compressés réutilisant les clés `POSITION_KEY`/`POLICY_KEY`/`VALUE_KEY` de `chess_target_encoding.py`. Ne dépend pas de `dataset_configs.py` (aucune entrée `CHESS` n'existe encore — Story 9.3) : paramètres explicites, pas de lecture de config.
- Décision d'implémentation non couverte par le PRD : un résultat de partie `"*"` (inconnu/inachevé, en-tête PGN non standard) est traité comme une nulle (value=0) — cas rare dans les archives pgnmentor (parties de tournois terminées), pas un cas testé explicitement mais couvert par le même branchement que `"1/2-1/2"`.
- Les 8 acceptance criteria sont satisfaits et vérifiés par test : AC1 (comptage exemples == demi-coups, y compris positions de début de partie), AC2 (planes façon AlphaZero + historique, shape vérifiée), AC3 (padding à zéro vérifié position de départ + padding partiel), AC4 (policy = index du coup joué, round-trip vérifié y compris balayage exhaustif de tous les coups légaux sur plusieurs positions), AC5 (value signée par position, calculée une fois par le producteur, vérifiée sur les 3 résultats de partie possibles), AC6 (absence de moteur externe, vérifiée statiquement), AC7 (`NUM_MOVES` exposé et importé par les tests, prêt pour la Story 9.2), AC8 (chunking `.npz` vérifié, tailles de chunk exactes 5/5/4 sur 14 exemples avec `chunk_size=5`).

### File List

- `chess_target_encoding.py` (nouveau)
- `dataset_builder/chess_pgn_dataset_tools.py` (nouveau)
- `tests/test_chess_target_encoding.py` (nouveau)
- `tests/test_chess_pgn_dataset_tools.py` (nouveau)

### Review Findings

Code review adversarial (2026-07-27, Blind Hunter + Edge Case Hunter + Acceptance Auditor en parallèle). Tests re-vérifiés (11/11 passés sur les 2 suites dédiées). 0 finding `high`. 1 finding `medium` de fond (bug de documentation, pas de logique) + plusieurs `medium`/`low` de robustesse. 4 findings écartés comme bruit, 1 différé.

**Patch (11) :**

- [x] [Review][Patch] `NUM_PLANES` documenté à 69 partout (docstring de module, commentaires, Dev Notes de cette story, Completion Notes, print de test) alors que la valeur réelle calculée est 29 (19 position + 10 historique = 29, pas 69) — bug de documentation pur, le code se comporte correctement (2 plans/créneau × 5 créneaux = 10, conforme au design décrit dans le détail des Dev Notes), mais toutes les synthèses en tête de section disent 69/50 par erreur de recopie [chess_target_encoding.py:14,34,81 ; dataset_builder/chess_pgn_dataset_tools.py:89,92 ; tests/test_chess_target_encoding.py:141 ; cette story, Dev Notes ligne 70/75, Completion Notes ligne 138]
- [x] [Review][Patch] `move_to_index` : `ZeroDivisionError` non gardée si `from_square == to_square` (ex. coup dégénéré/nul) [chess_target_encoding.py:~183]
- [x] [Review][Patch] `move_to_index` (branche sous-promotion) : ne garde pas `board.piece_at(from_sq) is None`, incohérent avec `index_to_move` qui le garde déjà [chess_target_encoding.py:~186]
- [x] [Review][Patch] `index_to_move` : ne valide pas `index` dans `[0, NUM_MOVES)` avant usage — lève `IndexError` non documenté au lieu du `ValueError` promis par la docstring [chess_target_encoding.py:~208]
- [x] [Review][Patch] Aucune gestion des parties PGN malformées : `game.errors` (signal natif de `python-chess`) jamais vérifié, et aucune isolation d'erreur par partie pendant le rejeu (`board.push`) — une seule partie corrompue dans une grosse archive interrompt tout l'import sans checkpoint [dataset_builder/chess_pgn_dataset_tools.py:~140,~190]
- [x] [Review][Patch] `_iter_game_examples` passe la liste `history` entière (croissante) à `encode_position` à chaque demi-coup au lieu de la pré-découper aux 5 derniers éléments — copie O(n) inutile à chaque appel, O(n²) sur la partie entière [dataset_builder/chess_pgn_dataset_tools.py:~76]
- [x] [Review][Patch] `test_dataset_example_count_matches_plies` réimplémente la logique de signe de la value en ligne au lieu d'importer et d'exercer `_value_for_mover` réel — la couverture réelle de cette fonction n'existe que dans l'autre fichier de test, risque de divergence silencieuse entre les deux copies [tests/test_chess_target_encoding.py:~150]
- [x] [Review][Patch] Commentaire affirmant "pas de risque de crash observé" sans aucun benchmark à l'appui dans ce diff — affirmation non vérifiée intégrée en commentaire [dataset_builder/chess_pgn_dataset_tools.py:~89]
- [x] [Review][Patch] Aucun retour de progression pendant le traitement d'une grosse archive — un seul print par chunk complété, impossible de distinguer "en cours" de "bloqué" sur un gros fichier unique [dataset_builder/chess_pgn_dataset_tools.py, `build_chess_dataset`]
- [x] [Review][Patch] Résultat PGN `"*"` (inconnu/inachevé) replié silencieusement sur nulle (value=0) sans compteur ni log — si l'hypothèse "rare dans pgnmentor" s'avère fausse pour une archive donnée, pollution silencieuse des cibles value [dataset_builder/chess_pgn_dataset_tools.py:~62, `_value_for_mover`]
- [x] [Review][Patch] Aucun test n'épingle `NUM_MOVES`/`NUM_PLANES` comme constantes littérales (seuls des asserts internes d'auto-cohérence existent) — un refactor de `_PIECE_TYPES`/`HISTORY_LENGTH` changerait silencieusement la forme du contrat sans faire échouer aucun test [tests/test_chess_target_encoding.py]

**Defer (1) :**

- [x] [Review][Defer] `board.is_repetition(2)` appelé à chaque demi-coup de chaque partie — coût potentiellement significatif à l'échelle d'une archive pgnmentor réelle (la documentation de `python-chess` avertit que cette fonction rejoue toute la partie, aucune table de transposition incrémentale). Aucune preuve de problème réel (spot-check du reviewer à ~130 demi-coups sans souci), mais non benchmarké à l'échelle réelle dans ce diff — deferred, nécessite un test à l'échelle d'une vraie archive pgnmentor, hors périmètre de cette session (exécution locale lourde à éviter sans confirmation explicite) [chess_target_encoding.py, `encode_position`]

**Écartés (dismiss, 4)** : test de non-dépendance à un moteur externe ne détecte que les imports littéraux `chess.engine`/`SimpleEngine`, pas une invocation indirecte (`subprocess`, etc.) — retenu tel quel, durcir indéfiniment une détection statique a un rendement décroissant pour un projet solo sans CI adverse ; `policies` stocké en `int32` plutôt qu'`int16` (`NUM_MOVES=4672` tiendrait sur 2 octets) — gain négligible en pratique, le tenseur position (8×8×29 float32 ≈ 7,4 Ko/exemple) domine largement les 4 octets de la policy, pas justifié ; boilerplate `sys.path.insert` copié 3× — recopie délibérée de la convention déjà établie ailleurs dans le projet, pas une régression ; import de `dataset_builder` comme package de namespace implicite (pas de `__init__.py`) — convention préexistante du projet (déjà le cas pour tous les autres outils de `dataset_builder/`), pas introduite par cette story.
