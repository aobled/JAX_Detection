# Schéma dataset spike "token + candidats légaux" — `chess_ai` → `jax_supervised_training`

**Statut : SPIKE / PROVISOIRE — n'est PAS une section du contrat stable
(`contract-chess-ai-training-interface.md` §2).** Même patron que
`docs/spike-chess-move-token-dataset-schema.md` (spike coup-token précédent) : ce fichier
existe pour permettre une session `jax_supervised_training` séparée de consommer le
dataset spike (CAP-2/CAP-3 de `SPEC.md`) sans lire tout le spec. **Peut changer de format
sans préavis** avant le verdict CAP-4/CAP-5/CAP-6 — ne pas fixer une entrée
`dataset_configs.py` "définitive" dessus avant le verdict.

Contexte complet, raisonnement, décisions : `SPEC.md` + `.memlog.md` de ce dossier.

## Fichier

Fichier unique (pas de chunks, contrairement aux 4 datasets du contrat stable — même choix
que le spike coup-token, volume par position bien plus petit ici : pas de plans 8x8xC) :

```
/home/aobled/Documents/data/chunks/chess_token_candidate_spike/chess_token_candidate_spike.npz
```

Généré via `chess_token_candidate_spike_dataset.py` (pattern mécaniquement porté de
`chess_move_token_spike_dataset.py`, déjà validé à 1,4M positions — mono+parallèle,
seedé, checkpoints, fusion), auto-jeu frais et seedé, professeur **Stockfish** profondeur
12 (comme `CHESS_SEARCH_TEACHER` depuis le 2026-08-07 — pas `chess_search.py`, abandonné
pour ce rôle : trop lent, ~84h à l'échelle mono-processus), mêmes paramètres par défaut que
`CHESS_SEARCH_TEACHER` (`depth=8`, `opening_plies=4`, `max_halfmoves=400`) pour rester
comparable aux baselines existantes.

**Run réel généré (2026-08-13)** : `n_games=3000`, `seed=42`, `n_workers=7` (défaut,
16 cœurs), ~1h00. **412 574 positions gardées** (1275 terminales ignorées, 3732 filtrées
par la politique `MAX_CANDIDATES=50` ci-dessous — 0,89% des positions non-terminales).
Validation croisée inattendue : `412574 + 3732 = 416306`, exactement le total de positions
non-terminales du 1er run (3000 parties, même seed) du spike coup-token — les deux scripts
partagent le même flux `generate_selfplay_positions`/RNG en amont malgré un encodage
totalement différent en aval, confirmation indépendante de l'absence de bug côté auto-jeu.
Vérifié exhaustivement (vectorisé, 412 574 lignes) : dtypes/shapes conformes, IDs token
dans `[0, 13)`, `global_flags` binaires, cohérence masque/padding/label à 100%, invariant
"exactement 1 roi par camp" à 100%.

## Vocabulaire de tokens (input, CAP-1)

13 valeurs statiques, mêmes 6 types de pièce et même ordre que
`chess_target_encoding.py::_PIECE_TYPES` (Roi, Dame, Tour, Fou, Cavalier, Pion) —
encodage **relatif** au joueur au trait (ami/ennemi), jamais couleur absolue :

| ID | Sémantique |
| --- | --- |
| 0 | case vide |
| 1-6 | pièce **amie** : Roi, Dame, Tour, Fou, Cavalier, Pion (dans cet ordre) |
| 7-12 | pièce **ennemie** : mêmes 6 types, même ordre |

## Clés `.npz`

| Clé | dtype | shape | Sémantique |
| --- | --- | --- | --- |
| `token_position` | `int32` | `(N, 64)` | Un ID de token (0-12 ci-dessus) par case. Index `i` de l'array = index de case `python-chess` natif (`chess.SQUARES`, 0=a1 … 63=h8) — **même numérotation que celle utilisée par `move_to_index`/`move_from_index`** (`from_square`/`to_square`), pour que l'embedding positionnel côté modèle et le décodage de coup restent alignés sur la même convention sans table de correspondance séparée. |
| `global_flags` | `float32` | `(N, 6)` | `[trait, roque_court_soi, roque_long_soi, roque_court_adv, roque_long_adv, répétition]` — même sémantique et même ordre que les plans 12-17 de `encode_position` actuel (`chess_target_encoding.py`), simplement extraits des plans spatiaux vers un vecteur global (ils étaient déjà des constantes uniformes par plan, jamais une information par case). |
| `candidate_moves` | `int32` | `(N, 50)` | Coups légaux de la position, encodés via `move_to_index` (espace `NUM_MOVES=4672` existant, AD-18 — aucun nouveau schéma). **Ordre = ordre naturel d'itération de `board.legal_moves`, jamais réordonné** (voir Politique de troncature ci-dessous — un ordre truqué laisserait deviner le label par sa position). Slots au-delà du nombre réel de coups légaux : `-1` (sentinel hors `[0, NUM_MOVES)`, sans ambiguïté même sans lire `candidate_mask`). |
| `candidate_mask` | `int8` | `(N, 50)` | `1` = slot valide (coup légal réel), `0` = slot de padding. Redondant avec le sentinel `-1` de `candidate_moves` par construction, fourni explicitement pour un masquage additif direct (`-inf` sur les slots à 0) avant softmax côté modèle, sans recalcul. |
| `candidate_label` | `int32` | `(N,)` | Index dans `[0, 50)` du coup choisi par le professeur (Stockfish profondeur 12) au sein de `candidate_moves` — toujours `< candidate_mask.sum()` par construction (voir Politique de troncature). |

Aucune clé `value` (positions d'auto-jeu, même raisonnement que `CHESS_SEARCH_TEACHER` et
`chess_decisive_teacher` — pas d'issue de partie complétée naturellement associée).

## Politique de troncature — `MAX_CANDIDATES=50`

Une position dont `len(list(board.legal_moves))` dépasse 50 (rare — le maximum théorique
de 218 exige une configuration de pièces irréaliste en jeu réel, ~20-40 coups légaux
typiques) est **filtrée, jamais tronquée** : elle n'est pas écrite dans le dataset. Une
troncature qui garderait artificiellement le coup professeur dans les 50 premiers slots
(ex. le placer en premier) introduirait un biais positionnel — le modèle pourrait
apprendre à privilégier un slot plutôt que le contenu du coup qu'il porte. **Fréquence
mesurée** (run 3000 parties, 2026-08-13, 417 581 positions non-terminales visitées) :
**0,89%** (3732 positions filtrées) — faible mais pas négligeable au sens strict, contrairement
à l'estimation initiale "proche de zéro" de ce document (corrigé ici après mesure, voir
`.memlog.md`). `n_legal` sur les positions gardées : médiane 26, moyenne 24,6, p99 48 —
cohérent avec les 20-40 coups typiques cités en intro, la coupe à 50 n'est donc pas
arbitraire mais elle mord réellement sur la queue de distribution.

## Vérification disponible côté `chess_ai`

Round-trip attendu (CAP-1) : reconstruire une position `chess.Board` depuis
`token_position` + `global_flags` doit produire un plateau identique à l'original — à
vérifier sur un échantillon de positions réelles avant tout run à pleine échelle, même
discipline que le round-trip `move_to_index`/`index_to_move` déjà en place.

## Ce qui reste à trancher côté `jax_supervised_training` (hors scope de ce companion)

- Dimension `D` de l'embedding de token (`nn.Embed(13, D)`) et de l'embedding positionnel
  (`nn.Embed(64, D)`).
- Dimension de l'embedding de coup candidat (`nn.Embed(NUM_MOVES, D_move)` ou équivalent)
  utilisé pour scorer chaque candidat contre la représentation de position.
- Nom exact de l'entrée `dataset_configs.py` (`CHESS_TOKEN_CANDIDATE` ou équivalent — open
  question de `SPEC.md`).
- Détail de la condition d'arrêt anticipé (métrique exacte, seuil, epoch de décision) —
  cadrée en principe dans `SPEC.md`/Constraints, à opérationnaliser dans l'épic séparée.
