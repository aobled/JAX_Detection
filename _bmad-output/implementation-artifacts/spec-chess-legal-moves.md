---
title: 'Tâche chess_legal_moves : prédire les coups légaux'
type: 'feature'
created: '2026-08-02'
status: 'done'
review_loop_iteration: 0
context: []
baseline_commit: '2e44404393e68ba60066bc4d804821f0d2210e14'
---

<frozen-after-approval reason="human-owned intent — do not modify unless human renegotiates">

## Intent

**Problem:** Le dataset `chess_legal_moves` (149 chunks `.npz`, `position (N,8,8,29)` + `legal_mask (N,4672) int8`) est généré mais rien dans ce repo ne peut l'entraîner — c'est une tâche multi-label (plusieurs coups légaux à la fois), différente de `chess_policy_value` (single-label, un seul coup joué).

**Approach:** Dupliquer le pattern `chess_policy_value` de bout en bout pour une nouvelle tâche `chess_legal_moves` : nouveau modèle sans tête value (backbone identique, une seule sortie logits 4672), loss BCE sigmoid (pas softmax), métrique F1 sur la classe "légal" (l'accuracy brute est trompeuse ici — ~20-40 coups légaux sur 4672, "tout illégal" donnerait déjà ~99%). Test rapide de plomberie, pas un objectif de qualité.

## Boundaries & Constraints

**Always:** Dupliquer les classes existantes (`ChessPolicyValueStrategy`/`ChessPolicyValueDataset`/`ChessCnnAttentionPolicyValue`) plutôt que les généraliser par un flag — convention déjà en place (`CHESS_NO_HISTORY` est une copie, pas un flag). Le nouveau modèle retourne un tenseur unique `(B, 4672)`, jamais un dict (contrairement à policy+value).

**Ask First:** Lancer un vrai run d'entraînement multi-epochs (une machine a déjà planté sur un run lourd lancé sans confirmation dans ce projet) — implémenter et vérifier le câblage (shapes, un batch, gradients) suffit pour cette spec.

**Never:** Ne pas toucher à `ChessPolicyValueStrategy`/`ChessPolicyValueDataset`/`ChessCnnAttentionPolicyValue` (chemin `chess_policy_value` intouché). Ne pas ajouter de masquage des coups illégaux au policy existant (hors sujet).

## I/O & Edge-Case Matrix

| Scenario | Input / State | Expected Output / Behavior | Error Handling |
|----------|--------------|---------------------------|----------------|
| Batch nominal | `position (B,8,8,29)`, `legal_mask (B,4672)` int8 | `ChessCnnAttentionLegalMoves` retourne logits `(B,4672)` ; loss BCE finie ; F1 dans [0,1] | N/A |
| Chunks introuvables | `output_prefix` sans fichiers `_chunk*.npz` | Message d'erreur explicite (mirror `ChessPolicyValueDataset`) | `exit(1)` avec message clair |
| Position sans aucun coup légal (mat/pat, tous 0) | `legal_mask` tout à zéro | F1 défini sans division par zéro (précision/rappel à 0 si aucune prédiction positive, pas de NaN) | Garde explicite (ex. `jnp.where`) |

</frozen-after-approval>

## Code Map

- `model_library.py:963-1030` -- `ChessCnnAttentionPolicyValue`/factory/`MODELS` à dupliquer sans tête value
- `loss_functions.py:560-566` -- `compute_chess_policy_loss` à mirrorer en BCE sigmoid
- `task_strategies.py:469-537` -- `ChessPolicyValueStrategy` à dupliquer (F1 au lieu d'accuracy)
- `data_management.py:594-702,766-777` -- `ChessPolicyValueDataset`/`get_datasets()` à dupliquer
- `dataset_configs.py:585` (fin du dict) -- nouvelle entrée `CHESS_LEGAL_MOVES`
- `main.py:117-126,174-180` -- forwarding `model_kwargs` + dispatch `task_type`
- `tests/test_chess_model.py` -- convention de test (script autonome) à mirrorer pour le nouveau modèle

## Tasks & Acceptance

**Execution:**
- [x] `model_library.py` -- ajouter `ChessCnnAttentionLegalMoves` (copie backbone, sans tête value, sortie `logits (B,4672)`) + `create_chess_cnn_attention_legal_moves(num_classes, dropout_rate=0.1, **kwargs)` + entrées `MODELS`/`get_model_info` -- suit exactement le pattern `ChessCnnAttentionPolicyValue`
- [x] `loss_functions.py` -- ajouter `compute_chess_legal_moves_loss(logits, legal_mask)` = `optax.sigmoid_binary_cross_entropy(logits, legal_mask.astype(jnp.float32)).mean()`
- [x] `task_strategies.py` -- ajouter `ChessLegalMovesStrategy` : `preprocess_batch` cast `legal_mask` float32 ; `compute_metrics` = F1 (seuil 0.5 sur sigmoid(logits), garde division-par-zéro) ; `primary_metric_name="LegalMoveF1"` ; `optimization_mode="max"`
- [x] `data_management.py` -- ajouter `ChessLegalMovesDataset` (mirror tri/mélange/split de `ChessPolicyValueDataset`, retourne `(position, legal_mask)`) + branche `elif task_type == "chess_legal_moves"` dans `get_datasets()`
- [x] `dataset_configs.py` -- ajouter `"CHESS_LEGAL_MOVES"` : `task_type`, `num_classes=4672`, `num_channels=29`, `input_shape=(8,8,29)`, `model_name="chess_cnn_attention_legal_moves"`, `num_bottleneck_tokens=8`, `output_prefix` vers `chunks/chess_legal_moves/`, `val_split=0.1`, hyperparams gpu/tpu dérivés de l'ancienne config `CHESS` (mêmes ordres de grandeur, `decay_steps` réduit au prorata de `epochs=4`), `save_dir="./checkpoints_chess_legal_moves"`
- [x] `main.py` -- ajouter branche `elif task_type == "chess_legal_moves"` (instancie `ChessLegalMovesStrategy`)
- [x] `tests/test_chess_model.py`-like nouveau script `tests/test_chess_legal_moves_model.py` -- shapes/dtypes/gradients du nouveau modèle (mirror du test existant)

**Acceptance Criteria:**
- Given la config `CHESS_LEGAL_MOVES` chargée, when `get_datasets()` est appelé, then il retourne des `tf.data.Dataset` `(position, legal_mask)` sans erreur (chunks réels présents)
- Given un batch réel, when `ChessLegalMovesStrategy.compute_loss` est appelé, then la loss est un scalaire fini
- Given un `legal_mask` tout à zéro, when `compute_metrics` est appelé, then aucun NaN n'est produit

## Verification

**Commands:**
- `python3 tests/test_chess_legal_moves_model.py` -- expected: shapes/dtypes/gradients OK
- Dry-run minimal (1 batch réel via `get_datasets()` + forward pass) -- expected: loss finie, F1 calculée -- **pas de run d'entraînement multi-epochs sans confirmation explicite**

## Suggested Review Order

**Modèle (nouveau, sans tête value)**

- Point d'entrée : backbone identique à `ChessCnnAttentionPolicyValue`, mais une seule sortie `logits (B, 4672)` — pas de dict.
  [`model_library.py:979`](../../model_library.py#L979)
- Factory miroir : `num_classes` porte la taille de l'espace de coups, pas un nombre de classes.
  [`model_library.py:1066`](../../model_library.py#L1066)

**Loss & métrique (le choix le plus sensible de cette spec)**

- Sigmoid BCE, pas softmax : plusieurs coups légaux simultanés, décision binaire indépendante par coup.
  [`loss_functions.py:595`](../../loss_functions.py#L595)
- F1 (pas accuracy brute) sur la classe "légal" — accuracy serait trompeuse vu le déséquilibre ~0.5%.
  [`task_strategies.py:569`](../../task_strategies.py#L569)
- Helper factorisé après revue (évite la duplication compute_metrics/generate_reports).
  [`task_strategies.py:588`](../../task_strategies.py#L588)

**Chargement des données**

- Nouveau loader, mirror de `ChessPolicyValueDataset` — retourne `(position, legal_mask)`.
  [`data_management.py:707`](../../data_management.py#L707)
- Garde de forme ajoutée après revue (Edge Case Hunter) — erreur claire si un chunk ne correspond pas à la config.
  [`data_management.py:772`](../../data_management.py#L772)
- Dispatch dans `get_datasets()`.
  [`data_management.py:876`](../../data_management.py#L876)

**Câblage config & dispatch**

- Nouvelle entrée `CHESS_LEGAL_MOVES` — `epochs=4` volontairement petit (test de plomberie).
  [`dataset_configs.py:632`](../../dataset_configs.py#L632)
- Dispatch `task_type` dans `main.py`.
  [`main.py:181`](../../main.py#L181)

**Tests (script autonome, mirror de `tests/test_chess_model.py`)**

- Cas limite explicitement couvert par la spec : `legal_mask` tout à zéro, aucun NaN.
  [`test_chess_legal_moves_model.py:147`](../../tests/test_chess_legal_moves_model.py#L147)
- Cas nominal ajouté après revue : F1 borné dans [0,1] sur un taux de sparsité réaliste.
  [`test_chess_legal_moves_model.py:128`](../../tests/test_chess_legal_moves_model.py#L128)
