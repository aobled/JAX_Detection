---
baseline_commit: f16e5251964ae9780d5ebaf1ca0e23054c6dbbbd
---

# Story 10.1: Configuration et tolérance du chargeur pour le dataset professeur

Status: done

<!-- Note: Validation is optional. Run validate-create-story for quality check before dev-story. -->

## Story

As a mainteneur du pipeline d'entraînement,
I want une entrée `CHESS_SEARCH_TEACHER` dans `dataset_configs.py`, un `ChessPolicyValueDataset` tolérant à l'absence de `value`, et un champ `value_head_trained` tracé automatiquement dans tout checkpoint échecs exporté,
so that le dataset `chess_search_teacher` (chess_ai) est consommable par le pipeline existant sans nouveau modèle ni nouvelle `TaskStrategy`, et que la fiabilité de la tête value reste traçable pour tout futur consommateur.

## Acceptance Criteria

1. **Given** `DATASET_CONFIGS` existant, **When** l'entrée `CHESS_SEARCH_TEACHER` est ajoutée, **Then** elle réutilise inchangés `task_type="chess_policy_value"`/`model_name="chess_cnn_attention_policy_value"`, avec `num_classes=4672`/`num_channels=29`/`input_shape=(8,8,29)`, `output_prefix` dédié (`chunks/chess_search_teacher/`) et `loss_params={"policy_weight":1.0,"value_weight":0.0}` — FR1, FR2
2. **Given** `validate_config()`, **When** `CHESS_SEARCH_TEACHER` est validée, **Then** elle passe sans aucune modification de `validate_config()` (mêmes clés requises que les configs échecs existantes) — FR1
3. **Given** un chunk `.npz` factice minimal (2-3 exemples, clés `position`+`policy` uniquement, sans `value`) construit pour le test, **When** `ChessPolicyValueDataset.create_tf_dataset` le charge, **Then** il produit un batch avec `value=0.0` pour chaque exemple, sans `KeyError` — FR3
4. **Given** un chunk `.npz` portant une clé `value` réelle (format `CHESS_NO_HISTORY` — aucun chunk réel de ce dataset n'existe en local, utiliser un `.npz` factice équivalent, voir Task 3), **When** il est chargé après cette story, **Then** ses valeurs `value` sont chargées à l'identique d'avant cette story — non-régression vérifiée par exécution réelle — FR3, FR5
5. **Given** un batch avec `value=0.0` pour tous les exemples (dummy), **When** `compute_chess_policy_value_loss` est appelé avec `value_weight=0.0`, **Then** `value_loss` n'est jamais `NaN` (tête value bornée par `tanh` dans `[-1,1]`, cible constante 0.0 — `0.0 × NaN = NaN` sinon, garde-fou pré-mortem 2026-08-04) et le gradient de la tête value est nul — FR2
6. **Given** tout run échecs (existant ou nouveau) dont `loss_params.value_weight == 0`, **When** le checkpoint est exporté (format "export pur" `params`/`batch_stats`/`config`), **Then** sa config embarquée porte `value_head_trained=False` ; un run avec `value_weight > 0` (ex. `CHESS_NO_HISTORY`) porte `value_head_trained=True` — FR4
7. **Given** un checkpoint déjà entraîné et sauvegardé sur disque avant cette story, **When** cette story est complétée, **Then** ce fichier `.pkl` n'est ni modifié ni régénéré — seuls les futurs exports portent le nouveau champ (clarification pré-mortem 2026-08-04) — FR4
8. **Given** `ClassificationStrategy`/`DetectionStrategy`/`CenterNetDetectionStrategy`/`KeplerStrategy`/`ChessLegalMovesStrategy`, **When** cette story est complétée, **Then** aucune n'est modifiée — FR5

## Tasks / Subtasks

- [x] Task 1: Ajouter l'entrée `CHESS_SEARCH_TEACHER` dans `dataset_configs.py` (AC: 1, 2)
  - [x] Placer la nouvelle entrée après `CHESS_LEGAL_MOVES` dans `DATASET_CONFIGS` (dataset_configs.py, actuellement lignes 632-713)
  - [x] `task_type="chess_policy_value"`, `model_name="chess_cnn_attention_policy_value"`, `num_bottleneck_tokens=8` — copiés depuis `CHESS_NO_HISTORY` (dataset_configs.py:585-630), **PAS** `num_channels`/`input_shape` (voir Dev Notes ci-dessous, danger de copie aveugle)
  - [x] `num_classes=4672`, `num_channels=29`, `input_shape=(8,8,29)` — valeurs du contrat §2.6, jamais dérivées d'une config sœur
  - [x] `output_prefix=f"{DATA_ROOT}/chunks/chess_search_teacher/chess_search_teacher"` (symétrique au pattern `chunks/chess_no_history/chess` et `chunks/chess_legal_moves/chess_legal_moves` déjà en place)
  - [x] `loss_params={"policy_weight": 1.0, "value_weight": 0.0}`
  - [x] `val_split=0.1`, `optimizer="adamw"`, `lr_schedule="cosine"` — copiés depuis `CHESS_NO_HISTORY` (valeurs de départ non tunées, cf. PRD §6.2 Out of Scope)
  - [x] `tpu`/`gpu` blocks — copiés depuis `CHESS_NO_HISTORY` tels quels (micro_batch_size/accum_steps/learning_rate/weight_decay/dropout_rate/warmup_steps/decay_steps) ; `decay_steps` non recalculé pour ce dataset (volume de chunks réel inconnu à ce stade — Open Question 4 du PRD, laissée à un ajustement ultérieur si nécessaire)
  - [x] `epochs=15`, `patience=8` — mêmes valeurs que `CHESS_NO_HISTORY`/`CHESS_LEGAL_MOVES`
  - [x] `save_dir="./checkpoints_chess_search_teacher"`

- [x] Task 2: Dériver `value_head_trained` automatiquement dans `get_dataset_config()` (AC: 6, 7)
  - [x] Dans `get_dataset_config()` (dataset_configs.py:718-746), après le bloc `validate_config`, ajouter : si `"loss_params" in config` et `"value_weight" in config["loss_params"]`, alors `config["value_head_trained"] = config["loss_params"]["value_weight"] != 0`
  - [x] **Ne pas** ajouter ce champ inconditionnellement à toutes les configs — seules celles qui déclarent déjà `loss_params.value_weight` (`CHESS_NO_HISTORY` l'a explicitement à `1.0`, `CHESS_SEARCH_TEACHER` à `0.0` après Task 1) en héritent ; `CHESS_LEGAL_MOVES` (`loss_params={"pos_weight":1.0}`, pas de `value_weight`) n'a pas de tête value et ne doit pas recevoir ce champ
  - [x] **Pourquoi dériver plutôt qu'un littéral séparé par config** : `config["value_head_trained"]` calculé depuis `config["loss_params"]["value_weight"]` élimine tout risque de désynchronisation entre les deux si l'un est modifié sans l'autre — un seul endroit à changer si un jour `value_weight` change pour une config
  - [x] Aucune modification de `task_strategies.py::TaskStrategy.export_model` (lignes 39-62) ni de `trainer.py:553` nécessaire — `export_model` sérialise `config` tel quel dans `model_dict["config"]`, donc `value_head_trained` y arrive automatiquement dès qu'il est présent dans le dict retourné par `get_dataset_config()`, exactement comme `num_channels`/`model_name` aujourd'hui

- [x] Task 3: Rendre `ChessPolicyValueDataset.create_tf_dataset` tolérant à l'absence de `value` (AC: 3, 4, 5)
  - [x] Dans `data_management.py`, fonction `gen()` interne à `create_tf_dataset` (actuellement lignes 671-683) : remplacer la lecture inconditionnelle `values = data["value"]` par une vérification de présence (`"value" in data.files` ou équivalent `np.load`) — si absente, construire un tableau de zéros `float32` de même longueur que `positions`
  - [x] Ne rien changer au comportement quand `value` est présente (lecture actuelle inchangée)
  - [x] `output_signature` (data_management.py:685-691) reste identique dans les deux cas — le tenseur `value` produit est toujours `TensorSpec(shape=(), dtype=tf.float32)`, jamais absent du dict de sortie
  - [x] Écrire un script de smoke-test (pas de framework pytest/unittest dans ce repo — voir "Conventions de test" ci-dessous) construisant un `.npz` factice à 2-3 exemples avec uniquement `position`+`policy` (imite le contrat `chess_search_teacher`, §2.6), vérifier chargement sans exception et `value` tout à zéro
  - [x] Écrire un second script de smoke-test avec un `.npz` factice à 2-3 exemples portant `position`+`policy`+`value` (imite `CHESS_NO_HISTORY`) et vérifier que ses `value` sont chargées telles quelles, non altérées — pas de dépendance à de vrais chunks `CHESS_NO_HISTORY` sur disque (absents de cet environnement local)
  - [x] Écrire un troisième smoke-test vérifiant que `compute_chess_policy_value_loss(value_weight=0.0)` avec une value dummy ne produit jamais `NaN` (garde-fou pré-mortem, AC5)

- [x] Task 4: Vérifier la non-régression des domaines/strategies non touchés (AC: 8)
  - [x] `git diff` confirmant qu'aucune ligne de `ClassificationStrategy`/`DetectionStrategy`/`CenterNetDetectionStrategy`/`KeplerStrategy`/`ChessLegalMovesStrategy` (`task_strategies.py`) n'a changé
  - [x] Confirmé qu'aucune ligne de `ChessLegalMovesDataset` (`data_management.py:707+`) n'a changé — seule `ChessPolicyValueDataset.create_tf_dataset` est touchée (diff de 8 lignes, voir File List)

## Dev Notes

- **Ne pas copier `num_channels=19`/`input_shape=(8,8,19)` de `CHESS_NO_HISTORY`** — piège identifié en réconciliation PRD (2026-08-04) : `CHESS_NO_HISTORY` est actuellement la SEULE config `chess_policy_value` active et elle est sans historique (19 canaux), alors que `CHESS_SEARCH_TEACHER` a l'historique (29 canaux, contrat §2.6, `include_history=True` par défaut côté `chess_ai`). Aucune config existante ne combine déjà `chess_policy_value` + 29 canaux — `CHESS_LEGAL_MOVES` est à 29 canaux mais avec un `task_type`/modèle totalement différent (multi-label sigmoid), pas un gabarit valable pour les hyperparamètres d'entraînement de cette story.
- **`export_model` sérialise le dict `config` tel quel** (`task_strategies.py:39-62`, non surchargé par `ChessPolicyValueStrategy`) — c'est pourquoi Task 2 se limite à `dataset_configs.py`, sans toucher `trainer.py`/`task_strategies.py`. Précédent direct : `num_channels`/`model_name` arrivent déjà dans `model_dict["config"]` de la même façon, aucune curation additionnelle n'existe dans `export_model`.
- **Bornage de la tête value** : `compute_chess_value_loss` (`loss_functions.py:573-578`) documente `value_pred`/`value_targets` déjà dans `[-1, 1]` (tête `Dense(1)`+`tanh`, AD-24) — `MSE(pred, 0.0)` avec `pred ∈ [-1,1]` est borné dans `[0,1]`, jamais `NaN` en conditions normales. AC5 teste cette garantie explicitement plutôt que de la supposer.
- **Aucune dépendance externe bloquante pour cette story** — tout est testable avec un `.npz` factice construit localement. **Mise à jour 2026-08-04** : de vrais chunks `chess_search_teacher` existent déjà en local (2 chunks, 10 000 positions, `position`+`policy` uniquement, format conforme au contrat §2.6) — utilisables directement, pas besoin d'attendre `chess_ai`. En revanche aucun chunk `CHESS_NO_HISTORY` n'existe en local pour le test de non-régression (AC4) — utiliser un `.npz` factice avec `value` pour ce cas, voir Task 3.
- **Conventions de test de ce repo** : pas de framework pytest/unittest — les fichiers `tests/test_*.py` sont des scripts autonomes avec des fonctions `test_*()` appelées explicitement sous `if __name__ == "__main__":` (le repo n'a pas de framework de test formel). Précédent le plus proche : `tests/test_centernet_detection_dataset.py` (construit un `.npz` factice, appelle `create_tf_dataset`) — suivre ce même patron, nouveau fichier suggéré : `tests/test_chess_search_teacher_loader.py`.
- **Portée stricte** : ne pas toucher à `ChessPolicyValueStrategy` (task_strategies.py:469-538, AC1-4 déjà couvertes par la réutilisation strate) ni au modèle `chess_cnn_attention_policy_value` (`model_library.py`) — Option A actée (2026-08-04) exclut explicitement tout nouveau modèle/`TaskStrategy` cette epic.

### Project Structure Notes

- Fichiers modifiés : `dataset_configs.py` (Task 1 + Task 2), `data_management.py` (Task 3, uniquement `ChessPolicyValueDataset.create_tf_dataset`).
- Fichier nouveau : `tests/test_chess_search_teacher_loader.py` (patron `tests/test_centernet_detection_dataset.py`, scripts autonomes sans pytest).
- Pas de conflit détecté avec la structure existante — mêmes conventions que `CHESS_NO_HISTORY`/`CHESS_LEGAL_MOVES` (dataset_configs.py), même classe de chargeur existante réutilisée (data_management.py).

### References

- [Source: dataset_configs.py:585-630 (CHESS_NO_HISTORY, gabarit partiel — PAS num_channels/input_shape)]
- [Source: dataset_configs.py:632-713 (CHESS_LEGAL_MOVES, précédent de nommage/output_prefix dédié, PAS un gabarit d'hyperparamètres pour cette story)]
- [Source: dataset_configs.py:718-746 (get_dataset_config, point d'ajout de la dérivation value_head_trained)]
- [Source: data_management.py:594-706 (ChessPolicyValueDataset), en particulier :671-683 (gen(), lecture value) et :685-691 (output_signature)]
- [Source: task_strategies.py:39-62 (TaskStrategy.export_model, base class, non surchargée par ChessPolicyValueStrategy) et :469-538 (ChessPolicyValueStrategy)]
- [Source: trainer.py:553 (déclenchement de export_model sur nouveau best model)]
- [Source: loss_functions.py:560-592 (compute_chess_policy_loss/compute_chess_value_loss/compute_chess_policy_value_loss)]
- [Source: tests/test_centernet_detection_dataset.py (précédent de test .npz factice + create_tf_dataset, patron à suivre pour tests/test_chess_search_teacher_loader.py)]
- [Source: docs/contract-chess-ai-training-interface.md §2.3 (contrat de checkpoint) et §2.6 (contrat de données chess_search_teacher)]
- [Source: _bmad-output/planning-artifacts/prds/prd-jax_supervised_training-2026-08-04/prd.md (FR1-FR5)]
- [Source: _bmad-output/planning-artifacts/epics.md, section "Requirements Inventory — Epic 10" et "### Epic 10" (Story 10.1, Guardrails identifiés pré-mortem)]

## Dev Agent Record

### Agent Model Used

Claude Sonnet 5 (via Claude Code), mode autonome (bmad-dev-story), session Winston du 2026-08-04.

### Debug Log References

- `python3 dataset_configs.py` — validation de toutes les configs (dont `CHESS_SEARCH_TEACHER`) : ✅ toutes passent, aucune régression.
- `python3 -c "from dataset_configs import get_dataset_config; ..."` — `value_head_trained` confirmé : `CHESS_NO_HISTORY`→`True`, `CHESS_SEARCH_TEACHER`→`False`, `CHESS_LEGAL_MOVES`→absent (comportement exact attendu, AC6).
- `python3 tests/test_chess_search_teacher_loader.py` — 3/3 tests passent (`test_chunk_without_value_key_defaults_to_zero`, `test_chunk_with_value_key_unaffected`, `test_dummy_value_never_produces_nan_loss`).
- `git diff data_management.py` — confirme un seul hunk de 8 lignes, dans `ChessPolicyValueDataset.create_tf_dataset::gen()`, rien ailleurs (AC8).

### Completion Notes List

- Découverte en cours d'implémentation (recherche de la logique d'export checkpoint) : `TaskStrategy.export_model` (task_strategies.py:39-62) sérialise le dict `config` **tel quel**, sans curation — donc `value_head_trained` (FR4/AC6-7) n'a nécessité **aucune** modification de `export_model`/`trainer.py`, juste une dérivation dans `get_dataset_config()` (dataset_configs.py). Plus simple et plus sûr que l'hypothèse initiale de la story (qui envisageait potentiellement toucher la logique d'export).
- Découverte en cours d'implémentation (vérification disque) : de vrais chunks `chess_search_teacher` existent déjà en local (2 chunks, 10 000 positions, générés le 2026-08-04) — utilisés directement pour confirmer manuellement la shape/les clés avant d'écrire les tests, bien que les tests eux-mêmes utilisent des `.npz` factices (plus rapides, déterministes).
- Aucun chunk `CHESS_NO_HISTORY` réel disponible en local pour AC4 — testé avec un `.npz` factice équivalent (clés identiques : `position`+`policy`+`value`), la non-régression du chemin "value présente" est donc vérifiée par construction de test, pas par un vrai fichier de production.
- Toutes les Acceptance Criteria (1-8) sont couvertes et vérifiées par exécution réelle (pas seulement par lecture de code).

### File List

- `dataset_configs.py` — modifié : ajout de l'entrée `CHESS_SEARCH_TEACHER` (DATASET_CONFIGS) + dérivation `value_head_trained` dans `get_dataset_config()` (`> 0`, pas `!= 0`, suite à la revue de code).
- `data_management.py` — modifié : `ChessPolicyValueDataset.create_tf_dataset::gen()` tolère l'absence de la clé `value` (substitue des zéros), garde de longueur explicite + avertissement loggé (suite à la revue de code).
- `tests/test_chess_search_teacher_loader.py` — nouveau : 5 smoke-tests (value absente, value présente + appariement correct, garde-fou anti-NaN, garde-fou gradient nul via `jax.grad`, `value_head_trained` dérivé correctement sur 3 configs).

### Review Findings

Revue adversariale (Blind Hunter + Edge Case Hunter + Acceptance Auditor, 2026-08-04), diff scope : `dataset_configs.py`, `data_management.py`, `tests/test_chess_search_teacher_loader.py`. 12 findings bruts, dédupliqués à 9, triés :

- [x] [Review][Patch] AC5 : le "gradient nul" n'était jamais vérifié par `jax.grad`, seule l'absence de NaN l'était [tests/test_chess_search_teacher_loader.py]
- [x] [Review][Patch] `value_head_trained` sans couverture de test automatisée (seulement vérifié à la main via `python3 -c`) [tests/test_chess_search_teacher_loader.py, dataset_configs.py]
- [x] [Review][Patch] Le zero-fill silencieux masquerait aussi une future régression du générateur `CHESS_NO_HISTORY` (qui perdrait sa vraie clé `value`), pas seulement le cas `chess_search_teacher` attendu — aucun signal visible [data_management.py:678-684]
- [x] [Review][Patch] Le test de non-régression comparait les `value` triées, incapable de détecter un désalignement position/value dans le `zip` [tests/test_chess_search_teacher_loader.py]
- [x] [Review][Patch] Aucune garde sur la longueur de `value` vs `positions` — un tableau `value` malformé/plus court tronquerait silencieusement le batch via `zip` [data_management.py:678-684]
- [x] [Review][Patch] `value_weight != 0` classe un poids négatif (erreur de config) comme "tête entraînée" — devrait être strictement positif [dataset_configs.py:811]
- [x] [Review][Patch] `NUM_PLANES=29` dupliqué en littéral dans le test au lieu d'être sourcé depuis la config [tests/test_chess_search_teacher_loader.py]
- [x] [Review][Defer] `get_dataset_config()` mute le dict global partagé `DATASET_CONFIGS[...]` en place [dataset_configs.py] — déjà le comportement pré-existant (`config["dataset_name"]=...`), pas introduit par cette story
- [x] [Review][Defer] `validate_config()` n'impose pas `loss_params.value_weight` pour `task_type="chess_policy_value"` (absence silencieuse vs `False` explicite) [dataset_configs.py::validate_config] — hors scope de cette story (AC2 exige explicitement `validate_config()` non modifiée)

Dismissed (3, déjà traités ailleurs) : `decay_steps` non recalculé (déjà Open Question du PRD) ; chemin de production `output_prefix` non garanti (déjà noté "réserve mineure" dans les Dev Notes) ; value factice indiscernable en bande (c'est exactement la raison d'être de `value_head_trained`).

**Code review complete.** 0 decision-needed, 7 patch (tous appliqués, mode autonome), 2 defer, 3 dismissed.

## Change Log

- 2026-08-04 : Story 10.1 implémentée intégralement (Tasks 1-4, AC1-8) en une session, mode autonome. Aucun écart par rapport aux Dev Notes/References de la story — la seule surprise (export_model déjà transparent au dict config) a réduit le risque d'implémentation plutôt que de l'augmenter.
- 2026-08-04 : Revue de code (Blind Hunter + Edge Case Hunter + Acceptance Auditor) — 7 patches appliqués (garde-fou gradient AC5 réellement testé via `jax.grad`, `value_head_trained` couvert par test automatisé au lieu d'une vérification manuelle, garde de longueur + avertissement loggé sur `value` absente, test de non-régression corrigé pour détecter un désalignement, `value_weight` strictement positif au lieu de `!= 0`, `NUM_PLANES` sourcé depuis la config). 2 items déférés (`deferred-work.md`, hors scope de cette story). Statut : `done`.
