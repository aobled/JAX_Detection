---
title: 'Modèle CHESS_TOKEN — tronc auto-attention + scoring candidats'
type: 'feature'
created: '2026-08-13'
status: 'done'
review_loop_iteration: 0
context: ['{project-root}/_bmad-output/specs/spec-chess-token-candidate-model/SPEC.md']
baseline_commit: 'c6c9d2970c1e8965f1ed9875f4a974228f18699f'
---

<frozen-after-approval reason="human-owned intent — do not modify unless human renegotiates">

## Intent

**Problem:** `chess_ai` a livré un dataset spike de candidats légaux tokenisés (`chess_token_candidate_spike.npz`), mais aucun modèle du repo ne sait le consommer — il faut remplacer le tronc CNN + tête `Dense(4672)` existants par un tronc auto-attention + une tête de scoring sur 50 candidats.

**Approach:** Nouvelle classe modèle + loss + config `dataset_configs.py` + `TaskStrategy` + dataset loader, en suivant strictement le patron déjà validé par `CHESS_MOVE_TOKEN` (Epic 11) pour la plomberie (`main.py`/`task_strategies.py`/`data_management.py`), avec une représentation de candidat décomposée (`from_square` + `move_type`) plutôt qu'une table indexée sur 4672.

## Boundaries & Constraints

**Always:**
- Bottleneck `K=8`/`token_dim=64` (défauts `model_library.py`), copié tel quel dans la nouvelle classe — convention du repo "copy the class, change the head" (`ChessCnnAttentionLegalMoves` duplique déjà `ChessCnnAttentionPolicyValue` ainsi), pas de mixin partagé.
- Embeddings token (`nn.Embed(13,64)`) + position (`nn.Embed(64,64)`) alimentent le tronc à `token_dim=64`, sans projection.
- Représentation d'un candidat **décomposée**, jamais via `nn.Embed(4672, D)` : `from_square = index // 73` réutilise l'embedding positionnel du tronc (poids partagés) ; `move_type = index % 73` via une nouvelle petite table `nn.Embed(73, 32)`.
- Loss = cross-entropy masquée (`candidate_mask=0` → `-1e9` avant softmax), label = `candidate_label ∈[0,50)`.
- Le modèle caste explicitement ses entrées entières en `int32` en tout début de `__call__` (pattern `ChessMoveTokenTransformer`, `model_library.py:1142-1147`) — `trainer.py:145` construit un dummy d'init `float32` en dur, jamais modifié.
- `main.py:228-235` (conditional `trainer_dtype`) doit inclure le nouveau `task_type`, sinon `trainer.py:313/430` caste silencieusement les tokens en `float16` (AD-29).
- `dataset_configs.py` : nouvelle entrée `CHESS_TOKEN`, `output_prefix` = chemin littéral vers `chess_token_candidate_spike.npz` (pas un glob chunké) — patron `CHESS_MOVE_TOKEN` (`dataset_configs.py:907-1102`).
- Aucune régression sur les chemins partagés existants (`CHESS_SEARCH_TEACHER`, `CHESS_MOVE_TOKEN`, `CHESS_LEGAL_MOVES`, `CHESS_NO_HISTORY`, `JAX_DETECTOR`, `trainer.py`) — additions pures, seuls les points d'enregistrement listés en Code Map sont touchés.
- Tests suivent le patron `tests/test_chess_move_token_model.py` (script simple, `assert` + `print("OK - ...")`, pas de framework).

**Ask First:** Lancer un entraînement réel (epochs/patience configurés) — un smoke test très court (quelques steps, forward+backward) est attendu pour valider CAP-1/CAP-2/CAP-3, mais pas un run complet. Toute modification d'un fichier partagé au-delà des points d'enregistrement listés (ex. toucher au bottleneck d'une classe existante, à `trainer.py`, à une `TaskStrategy` existante).

**Never:** Table `nn.Embed(NUM_MOVES=4672, -)` pour représenter un candidat. Comparaison stricte pass/fail contre le checkpoint 28,00% (CAP-4/CAP-5 = mesures de référence uniquement). Compatibilité de checkpoint avec `tournament_model_vs_model.py` (chess_ai) — hors scope explicite. Régénération/modification du dataset `.npz` existant.

## I/O & Edge-Case Matrix

| Scenario | Input / State | Expected Output / Behavior | Error Handling |
|----------|--------------|---------------------------|----------------|
| Batch normal | `candidate_mask` avec k<50 slots à 1 | loss/argmax ignorent totalement les slots à 0 | N/A |
| Padding | `candidate_moves[k:]==-1` | jamais décodé en `from_square`/`move_type` (mask=0 les exclut avant tout calcul de score utile) | N/A |
| Index extrême | move index=0 (`from_square=0,move_type=0`) et 4671 (`from_square=63,move_type=72`) | décodage reste dans `[0,64)`/`[0,73)` | assert dans le test unitaire |
| Dtype batch réel | `token_position`/`candidate_moves`/`candidate_label` entiers passés au `Trainer` | pas de cast `float16` destructeur (AD-29) | `task_type` ajouté à `main.py:235` |
| Dummy init `float32` | `model.init()` avec dummy `trainer.py:145` | le modèle caste lui-même en `int32` dès `__call__` (AD-33), pas de crash | N/A |

</frozen-after-approval>

## Code Map

- `model_library.py:1275-1286` (`MODELS` dict) -- ajouter `ChessTokenCandidateModel` + `create_chess_token_candidate_model`
- `model_library.py:837-960` (`ChessCnnAttentionPolicyValue` bottleneck) -- structure de référence à répliquer (copy-the-class)
- `model_library.py:1075-1225` (`ChessMoveTokenTransformer` + factory) -- référence directe la plus proche (cast `int32`, `compute_dtype`, lazy import `data_management`)
- `loss_functions.py` -- nouvelle `compute_chess_token_candidate_loss` (masked softmax cross-entropy), aucun précédent exact à réutiliser
- `task_strategies.py:620-686` (`ChessMoveTokenStrategy`) -- référence directe pour `ChessTokenStrategy`
- `data_management.py:866-993` (`ChessMoveTokenDataset`) + `data_management.py:1079-1090` (`get_datasets` dispatch) -- référence directe pour le nouveau loader + branche à ajouter
- `dataset_configs.py:907-1102` (`CHESS_MOVE_TOKEN`) -- référence directe pour la nouvelle entrée `CHESS_TOKEN`
- `main.py:117-149` (model kwargs), `main.py:152-223` (task_type dispatch), `main.py:228-235` (`trainer_dtype`) -- points d'enregistrement obligatoires
- `trainer.py:145` (dummy init `float32`), `trainer.py:313/430` (cast batch) -- lecture seule, zero-touch policy (AD-33/AD-29)
- `tests/test_chess_move_token_model.py` -- patron direct pour `tests/test_chess_token_model.py` (nouveau fichier)

## Tasks & Acceptance

**Execution:**
- [x] `model_library.py` -- ajouter `ChessTokenCandidateModel` (`nn.Module`) + `create_chess_token_candidate_model` + entrée `MODELS` -- CAP-1
- [x] `loss_functions.py` -- ajouter `compute_chess_token_candidate_loss` (masquage additif `-1e9`, cross-entropy sur `candidate_label`) -- CAP-2
- [x] `task_strategies.py` -- ajouter `ChessTokenStrategy` (`compute_loss` délègue à la loss ci-dessus, `compute_metrics` = top-1 argmax accuracy masquée) -- CAP-2/CAP-4
- [x] `data_management.py` -- ajouter `ChessTokenCandidateDataset` + branche `get_datasets()` -- CAP-3
- [x] `dataset_configs.py` -- ajouter entrée `CHESS_TOKEN` (task_type, model_name, input_shape, `output_prefix` littéral, hyperparamètres tpu/gpu, epochs/patience) -- CAP-3/CAP-6
- [x] `main.py` -- branche task_type dispatch + inclure `chess_token` dans le conditional `trainer_dtype` (int32) -- CAP-1/CAP-3
- [x] `tests/test_chess_token_model.py` -- shape/dtype, masquage, décomposition `from_square`/`move_type` aux bornes, registry, intégration fichier réel (skip si absent) -- CAP-1/CAP-2
- [x] Smoke test manuel (quelques steps, **pas** un run complet) -- valider forward/backward sans erreur (CAP-1) ; le run complet reste **Ask First**

**Acceptance Criteria:**
- [x] Given un batch réel du dataset, when le modèle fait un forward pass, then la sortie est `(B,50)` de logits. -- vérifié (test_output_shape_and_dtype + test_real_dataset_and_forward_pass_if_spike_available + smoke test réel : `outputs.shape=(16, 50)`).
- [x] Given `candidate_mask=0` sur un slot, when la loss est calculée, then ce slot n'influence ni la valeur ni le gradient. -- vérifié (test_masking_correctness_loss_and_argmax : valeur de loss identique à un calcul à la main restreint aux slots réels, malgré des logits énormes sur les slots masqués).
- [x] Given move index 0 et 4671, when la représentation candidate est calculée, then `from_square`/`move_type` restent dans leurs bornes sans erreur d'index. -- vérifié (test_from_square_move_type_decomposition_at_boundaries : (0,0) et (63,72), forward pass réel stable).
- [x] Given le dummy d'init `float32` (`trainer.py:145`), when `model.init()` est appelé, then aucune erreur de dtype. -- vérifié (test_init_with_float32_dummy_does_not_crash, dummy `(1,120)` float32 exact).
- [x] Given `CHESS_TOKEN` chargée via `get_dataset_config`, when un batch passe dans le `Trainer` réel, then aucune régression sur les tests existants (`CHESS_SEARCH_TEACHER`/`CHESS_MOVE_TOKEN`/`CHESS_LEGAL_MOVES`/`CHESS_NO_HISTORY`). -- vérifié : suite complète `tests/` (chess + non-chess, 20 fichiers) passe sans régression ; smoke test manuel (5 steps forward+backward réels, pas via `Trainer` - `Trainer` lui-même non exercé pour respecter Ask First sur tout run réel, mais la même séquence get_datasets/get_model/Strategy/apply_fn/grad qu'il orchestre est validée directement).

## Spec Change Log

## Design Notes

Décomposition d'un index `move_to_index` (`chess_ai/chess_target_encoding.py:243`, `index = from_square*73 + move_type`) sans table 4672 :

```python
from_square = move_idx // 73          # [0,64) -> réutilise l'embedding positionnel du tronc
move_type = move_idx % 73             # [0,73) -> nn.Embed(73, 32) dédiée
candidate_repr = jnp.concatenate([pos_embed_table[from_square], move_type_embed(move_type)], axis=-1)
```

Masquage avant softmax (aucun précédent exact dans `loss_functions.py`) :

```python
masked_logits = jnp.where(candidate_mask.astype(bool), logits, -1e9)
loss = optax.softmax_cross_entropy_with_integer_labels(masked_logits, candidate_label)
```

## Verification

**Commands:**
- `python tests/test_chess_token_model.py` -- expected: tous les tests passent (`OK - ...` imprimés, exit 0)
- `python -c "from dataset_configs import get_dataset_config; get_dataset_config('CHESS_TOKEN')"` -- expected: pas d'exception, `validate_config` passe
- Smoke test quelques steps (forward+backward), **pas** un run complet -- expected: aucune erreur ; tout run complet reste Ask First

## Suggested Review Order

**Décomposition d'un candidat (sans table 4672) — le cœur du design**

- Point d'entrée : nouveau modèle, tronc auto-attention + tête de scoring 50 candidats.
  [`model_library.py:1235`](../../model_library.py#L1235)

- `from_square`/`move_type` décomposés depuis l'index `move_to_index`, bornes garanties des deux côtés.
  [`model_library.py:1424`](../../model_library.py#L1424)

- Petite table dédiée `nn.Embed(73,32)`, jamais `nn.Embed(4672,-)`.
  [`model_library.py:1232`](../../model_library.py#L1232)

- `from_square` réutilise l'embedding positionnel du tronc (poids partagés), vérifié structurellement par `jaxpr`.
  [`tests/test_chess_token_model.py:300`](../../tests/test_chess_token_model.py#L300)

**Masquage — loss et métriques restreintes aux candidats réels**

- Cross-entropy masquée additive (-1e9 sur les slots invalides avant softmax).
  [`loss_functions.py:642`](../../loss_functions.py#L642)

- `compute_loss`/`compute_metrics` délèguent à la loss ci-dessus, precondition "au moins un candidat réel" documentée (non assertable sous `jax.jit`).
  [`task_strategies.py:688`](../../task_strategies.py#L688)

- Comportement de masquage vérifié à la main sur un batch synthétique + un batch réaliste 45/50 en padding sous gradient.
  [`tests/test_chess_token_model.py:145`](../../tests/test_chess_token_model.py#L145)

**Dataset — validation à l'entrée, jamais de corruption silencieuse**

- Nouveau loader : asserts de forme/plage sur les 5 clés (label pointe toujours sur un slot valide, flags binaires, tokens dans `[0,13)`).
  [`data_management.py:996`](../../data_management.py#L996)

- Branche de dispatch `task_type == "chess_token"` dans `get_datasets()`.
  [`data_management.py:1296`](../../data_management.py#L1296)

- Intégration bout en bout sur le vrai fichier `.npz` (412 574 lignes), skip si absent.
  [`tests/test_chess_token_model.py:409`](../../tests/test_chess_token_model.py#L409)

**Plomberie main.py — points d'enregistrement obligatoires**

- Dispatch du `task_type` vers `ChessTokenStrategy`.
  [`main.py:234`](../../main.py#L234)

- `trainer_dtype=int32` pour ce task_type (AD-29 : sinon cast `float16` destructeur des tokens).
  [`main.py:257`](../../main.py#L257)

- Forwarding de `num_trunk_layers` vers `model_kwargs` (comment corrigé pour rester honnête).
  [`main.py:146`](../../main.py#L146)

**Config — nouvelle entrée CHESS_TOKEN**

- Entrée `dataset_configs.py`, statut SPIKE/PROVISOIRE, K=8/token_dim=64 (décision "repartir à neuf", pas de comparaison stricte).
  [`dataset_configs.py:1072`](../../dataset_configs.py#L1072)

**Tests périphériques**

- Décomposition aux bornes exactes de l'espace de coups (index=0 et 4671).
  [`tests/test_chess_token_model.py:112`](../../tests/test_chess_token_model.py#L112)
