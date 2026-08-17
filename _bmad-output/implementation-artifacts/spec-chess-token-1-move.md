---
title: 'CHESS_TOKEN_1_MOVE — modèle échecs à tête factorisée'
type: 'feature'
created: '2026-08-15'
status: 'done'
review_loop_iteration: 0
context: ['{project-root}/docs/contract-chess-ai-training-interface.md']
baseline_commit: '268a47a203876802902b0f02dceea50b85ec6508'
---

<frozen-after-approval reason="human-owned intent — do not modify unless human renegotiates">

## Intent

**Problem:** Le contrat `chess_ai`/`jax_supervised_training` (§3, `docs/contract-chess-ai-training-interface.md`, spec `spec-chess-token-1-move`) demande un nouveau modèle échecs `CHESS_TOKEN_1_MOVE` — tronc auto-attention pure sans CNN/sans bottleneck, 2 têtes indépendantes `from_square`(64)/`move_type`(73) — pour rendre l'illégalité de coup à nouveau un échec mesurable (les architectures précédentes ne peuvent structurellement jamais en produire) et donner à l'algo génétique `chess_ai` un espace de mutation significatif (`token_embed`/`pos_embed`). Aucune config/modèle/loss/stratégie de ce type n'existe encore ici.

**Approach:** Ajouter un domaine complet (loader, modèle, loss, stratégie, config) sur le patron déjà établi par `ChessTokenCandidateModel`/`CHESS_TOKEN`, en réutilisant le dataset `chess_token_candidate_spike.npz` existant tel quel (dérivation du label réel par `candidate_moves[i, candidate_label[i]]`, jamais de régénération), puis lancer un training réel jusqu'à obtention d'un checkpoint `.pkl`.

## Boundaries & Constraints

**Always:**
- Réutiliser `/home/aobled/Documents/data/chunks/chess_token_candidate_spike/chess_token_candidate_spike.npz` tel quel — aucune régénération, aucune dépendance à `python-chess`/`chess_ai`.
- `token_embed` = `nn.Embed(13,D)` nommé littéralement `"token_embed"`, `pos_embed` = `nn.Embed(64,D)` nommé `"pos_embed"` — CAP-6 côté `chess_ai` cible ces noms pour la mutation génétique, ne pas les remplacer par une autre forme de couche.
- Les 2 têtes (`from_square` 64 classes, `move_type` 73 classes) prédites indépendamment, sans conditionnement de l'une sur l'autre. **SUPERSEDED pour v2, voir `spec-chess-token-1-move-v2.md`** (renégociation actée avec Aymeric, 2026-08-16 : v1 éliminée — plateau `JointMoveAccuracy` à 5,10% — suspect principal = cette même contrainte ; v2 introduit un conditionnement léger `move_type` sur `from_square`, teacher-forcing en training uniquement, `from_square` prédit reste libre).
- Aucun étage bottleneck (pas de cross-/auto-attention vers K queries apprises).
- `decompose_move_index = divmod(index, 73)`, aucune validation de légalité côté `jax_supervised_training`.
- Style/conventions déjà en place (`ChessTokenCandidateModel`, `ChessLegalMovesStrategy`, commentaires denses type `dataset_configs.py`).

**Ask First:** Aucun — autonomie totale accordée explicitement par Aymeric pour cette tâche jusqu'au lancement du training et l'obtention d'un `.pkl`.

**Never:**
- Ne pas régénérer ni modifier le `.npz`.
- Ne pas ajouter de masquage de légalité côté modèle/loss (le contrat veut l'illégalité mesurable, pas absorbée).
- Ne pas toucher aux domaines échecs existants (`CHESS_SEARCH_TEACHER`/`CHESS_LEGAL_MOVES`/`CHESS_TOKEN`/`CHESS_MOVE_TOKEN`) ni à `trainer.py`.

## I/O & Edge-Case Matrix

| Scenario | Input / State | Expected Output / Behavior | Error Handling |
|----------|--------------|---------------------------|----------------|
| Forward pass | batch `(B,70)` int32 `[token_position(64)\|global_flags(6)]` | dict `{"from_square":(B,64), "move_type":(B,73)}` logits float32 | assert shape explicite si largeur ≠ 70 |
| Décomposition label | `move_index` réel `[0,4672)` | `(from_square,move_type) = divmod(move_index,73)` exact | N/A (arithmétique pure, pas de cas d'erreur) |
| Accuracy jointe | `from_square` ET `move_type` prédits corrects simultanément | métrique = 1.0 pour cet exemple, 0.0 sinon | aucune (pas de masquage à gérer, contrairement à `CHESS_TOKEN`) |
| Chargement dataset | fichier `.npz` absent | message d'erreur explicite + exit, même convention que `ChessTokenCandidateDataset` | N/A |

</frozen-after-approval>

## Code Map

- `data_management.py` -- nouveau loader `ChessTokenOneMoveDataset`, proche de `ChessTokenCandidateDataset` (:996) mais sans packer `candidate_moves`
- `model_library.py` -- nouvelle classe `ChessTokenOneMoveModel` (tronc copié de `ChessTokenCandidateModel` :1235-1391, bottleneck :1393-1411 non repris) + factory + enregistrement (~1513-1516, ~1605+)
- `loss_functions.py` -- nouvelle fonction `compute_chess_token_1_move_loss` (proche `compute_chess_policy_loss` :561) + métrique accuracy jointe
- `task_strategies.py` -- nouvelle classe `ChessTokenOneMoveStrategy` (proche `ChessLegalMovesStrategy` :539 / `ChessMoveTokenStrategy` :620)
- `dataset_configs.py` -- nouvelle entrée `"CHESS_TOKEN_1_MOVE"`
- `tests/test_chess_token_model.py` -- tests additionnels (shape, divmod, accuracy jointe)
- `main.py` -- vérifier si le dispatch générique existant couvre le nouveau `task_type` sans modification

## Tasks & Acceptance

**Execution:**
- [x] `data_management.py` -- `ChessTokenOneMoveDataset` -- charge les 5 clés, dérive `move_index` une fois via gather numpy, packe `(N,70)` uniquement
- [x] `model_library.py` -- `ChessTokenOneMoveModel` + factory + registration -- tronc allégé sans bottleneck, 2 têtes indépendantes
- [x] `loss_functions.py` -- `compute_chess_token_1_move_loss` + métrique accuracy jointe -- décomposition + CE double tête
- [x] `task_strategies.py` -- `ChessTokenOneMoveStrategy` -- délègue à la loss ci-dessus, rapporte accuracy jointe + par tête
- [x] `dataset_configs.py` -- entrée `CHESS_TOKEN_1_MOVE` -- `input_shape=(70,)`, mêmes `output_prefix`/volume que `CHESS_TOKEN`, hyperparamètres documentés "non tunés"
- [x] `tests/test_chess_token_model.py` -- tests shape/divmod/accuracy -- non-régression + validation de la nouvelle logique
- [x] Lancer `pytest` puis `python main.py CHESS_TOKEN_1_MOVE` en arrière-plan (smoke-test réel avant de laisser tourner)

**Acceptance Criteria:**
- Given la config `CHESS_TOKEN_1_MOVE` et le dataset réel, when `Trainer` lance quelques steps, then aucune erreur de forme/dtype et la loss varie de façon cohérente (pas de NaN)
- Given un batch construit à la main avec un `move_index` connu, when `compute_chess_token_1_move_loss` est appelé, then la décomposition correspond exactement à `divmod(index,73)`
- Given `from_square` et `move_type` tous deux corrects sur un exemple, when la métrique accuracy jointe est calculée, then elle vaut 1.0 pour cet exemple
- Given la suite de tests existante, when exécutée après cette implémentation, then elle passe sans régression sur les domaines échecs existants

## Design Notes

`num_classes` générique (plomberie `main.py`) ne porte pas de sens direct ici (2 têtes de tailles différentes 64/73, pas un seul nombre de classes) — décider d'une convention documentée (ex. ignoré par la factory, `from_square`/`move_type` fixes en dur comme constantes du modèle) plutôt que de forcer `num_classes` dans un rôle qui ne lui correspond pas. `token_dim`/`num_trunk_layers` du tronc "allégé" : pas de valeur imposée par le contrat, point de départ à choisir et documenter comme non tuné (même précédent que `CHESS_LEGAL_MOVES`/`CHESS_TOKEN` à leur création).

## Verification

**Commands:**
- `pytest tests/test_chess_token_model.py` -- expected: tous les tests passent (nouveaux + existants, aucune régression)
- `python main.py CHESS_TOKEN_1_MOVE` (arrière-plan, quelques steps puis training complet) -- expected: démarre sans erreur, loss loggée, checkpoint `.pkl` sauvegardé périodiquement

## Training Outcome (2026-08-15)

Training réel lancé (`python main.py CHESS_TOKEN_1_MOVE`, GPU, 50 epochs prévues) puis arrêté manuellement par Aymeric à l'epoch 40/50 : plateau net de `JointMoveAccuracy` depuis l'epoch ~34 (0.0500→0.0510, quasi stable sur 6 epochs, patience=8 proche de se déclencher de toute façon). Meilleur checkpoint conservé sur disque à `JointMoveAccuracy=0.0510` (5.10%) — très en-deçà des 28.00% val `PolicyAccuracy` de `CHESS_SEARCH_TEACHER` (comparaison contractuelle, §3 du contrat), mais une accuracy jointe sur 2 têtes 64×73 non masquées n'est pas strictement comparable terme à terme à un top-1 sur 4672 classes déjà filtré par construction — à interpréter avec cette réserve. Checkpoint final : `best_model_chess_token_1_move.pkl` (82 049 octets, 19 977 paramètres). CAP-2 (côté `jax_supervised_training`) atteint son critère de succès minimal ("le modèle s'entraîne et converge sans erreur ; accuracy jointe mesurée et documentée") — le verdict qualitatif complet (concluant/non concluant au sens du SPEC `chess_ai`) dépend aussi de CAP-3/CAP-4/CAP-5/CAP-6, hors scope de ce repo.
