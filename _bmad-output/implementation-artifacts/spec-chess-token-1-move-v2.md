---
title: 'CHESS_TOKEN_1_MOVE v2 — conditionnement léger + read-token + tronc élargi'
type: 'feature'
created: '2026-08-16'
status: 'done'
review_loop_iteration: 0
context: ['{project-root}/docs/contract-chess-ai-training-interface.md']
baseline_commit: '268a47a203876802902b0f02dceea50b85ec6508'
---

<frozen-after-approval reason="human-owned intent — do not modify unless human renegotiates">

## Intent

**Problem:** Le run v1 de `CHESS_TOKEN_1_MOVE` (`spec-chess-token-1-move.md`) est éliminé — plateau `JointMoveAccuracy` à 5,10% dès l'epoch ~34/50, train≈val (plafond structurel, pas overfitting). Diagnostic documenté (contrat §3, "Piste v2", 2026-08-15) : suspect principal = la contrainte "2 têtes prédites indépendamment, sans conditionnement" (spec v1, Always) qui force le modèle à marginaliser sur l'identité de la pièce en case de départ pour prédire `move_type`, alors que ce dernier en dépend presque entièrement aux échecs.

**Approach:** Tester en UN SEUL run combiné 3 leviers identifiés (2026-08-15) plutôt que 3 runs isolés (chaque training est long ; run combiné = signal négatif plus fort si ça ne bouge toujours pas, run d'isolation seulement si ça bouge) : (1) conditionnement léger `move_type` sur `from_square` (teacher-forcing en training, `from_square` prédit reste libre), (2) read-token appris remplaçant le mean-pool, (3) tronc élargi (`token_dim` 32→64, `num_trunk_layers` 1→2). Édité EN PLACE sur `CHESS_TOKEN_1_MOVE` (pas de nouveau domaine — décision explicite Aymeric, chiffres v1 déjà durablement documentés dans spec/contrat).

## Boundaries & Constraints

**Always:**
- Le spec v1 a une section frozen dont l'Always dit "sans conditionnement" — annoter cette ligne "SUPERSEDED pour v2, voir `spec-chess-token-1-move-v2.md`" (ne pas la supprimer/réécrire) : renégociation actée avec Aymeric, 2026-08-16.
- `from_square` reste prédit de façon NON conditionnée (calculé depuis `pooled` seul) — seul `move_type` se conditionne sur `from_square`. Un coup illégal doit rester possible (objectif contractuel "illégalité mesurable" préservé).
- Conditionnement : teacher-forcing quand `training=True` (label réel packé en entrée), `argmax(from_square_logits)` quand `training=False` — JAMAIS le label réel utilisé quand `training=False`.
- `token_embed`/`pos_embed` gardent leurs noms exacts et l'instance PARTAGÉE (réutilisée pour l'embedding du `from_square` de conditionnement, 0 paramètre supplémentaire) — ciblés par `chess_bottleneck_genetic.py` côté `chess_ai`, jamais renommés/remplacés.
- `read_token` : nouveau param appris, 65e token de la séquence auto-attention (pas un étage de cross-attention séparé — reste conforme à "pas de bottleneck").
- Sauvegarder le checkpoint v1 avant tout nouveau training : `cp best_model_chess_token_1_move.pkl best_model_chess_token_1_move_v1_eliminated.pkl`.
- Mettre à jour `docs/contract-chess-ai-training-interface.md` (section "Piste v2") : reprise actée.

**Ask First:** Lancer le training long complet (50 epochs, GPU, plusieurs heures) — explicitement hors de l'autonomie accordée pour cette tâche. Seul un smoke-test court (quelques steps) est autorisé sans validation humaine.

**Never:**
- Ne pas régénérer/modifier le `.npz` existant.
- Ne pas ajouter de masquage de légalité côté modèle/loss.
- Ne pas toucher aux autres domaines échecs (`CHESS_SEARCH_TEACHER`/`CHESS_LEGAL_MOVES`/`CHESS_TOKEN`/`CHESS_MOVE_TOKEN`) ni à `trainer.py`.
- Ne pas créer de nouveau domaine `_V2` séparé — édition en place de `CHESS_TOKEN_1_MOVE`.

## I/O & Edge-Case Matrix

| Scenario | Input / State | Expected Output / Behavior | Error Handling |
|----------|--------------|---------------------------|----------------|
| Forward pass | batch `(B,71)` int32 `[token_position(64)\|global_flags(6)\|from_square_teacher(1)]` | dict `{"from_square":(B,64),"move_type":(B,73)}` | assert shape explicite si largeur ≠ 71 |
| Conditionnement train | `training=True`, `from_square_teacher` connu, `pooled` fixé | `move_type_logits` varie avec `from_square_teacher` | N/A |
| Conditionnement éval | `training=False`, `from_square_teacher` varié, `pooled` fixé | `move_type_logits` INCHANGÉ (ignore le label, utilise `argmax(from_square_logits)`) | test dédié obligatoire (anti-fuite) |
| Chargement dataset | fichier `.npz` absent | message d'erreur explicite + exit, même convention que `ChessTokenOneMoveDataset` | N/A |

</frozen-after-approval>

## Code Map

- `data_management.py` -- `ChessTokenOneMoveDataset` (~1203) -- ajouter `from_square_target` (= `move_index // 73`) comme 71e colonne de `packed_features`
- `model_library.py` -- `ChessTokenOneMoveModel` (~1462-1631) -- read-token (remplace `jnp.mean`), conditionnement `move_type_head`, `expected_width` 70→71, `token_dim`/`num_trunk_layers` déjà en kwargs (aucun changement de factory nécessaire)
- `dataset_configs.py` -- entrée `CHESS_TOKEN_1_MOVE` (~1226) -- `input_shape=(71,)`, `token_dim=64`, `num_trunk_layers=2`
- `loss_functions.py` / `task_strategies.py` -- vérifier par lecture qu'aucun changement de formule n'est nécessaire (shapes de sortie inchangées)
- `tests/test_chess_token_model.py` -- tests shape (B,71), read-token, conditionnement train (varie) vs éval (invariant, anti-fuite)
- `_bmad-output/implementation-artifacts/spec-chess-token-1-move.md` -- annoter l'Always v1 "SUPERSEDED pour v2"
- `docs/contract-chess-ai-training-interface.md` -- section "Piste v2" (~282-303) -- "reprise actée 2026-08-16"

## Tasks & Acceptance

**Execution:**
- [x] `spec-chess-token-1-move.md` -- annoter la ligne Always "sans conditionnement" SUPERSEDED -- traçabilité de la renégociation frozen
- [x] `contract-chess-ai-training-interface.md` -- mettre à jour section "Piste v2" -- traçabilité côté contrat partagé `chess_ai`
- [ ] `cp best_model_chess_token_1_move.pkl best_model_chess_token_1_move_v1_eliminated.pkl` -- préserver le checkpoint v1 avant écrasement -- **NON FAIT, dommage constaté** : le smoke test a démarré sans backup préalable et a déjà écrasé `best_model_chess_token_1_move.pkl` (confirmé 119 113 params = modèle v2, contre 19 977 params v1 documentés au contrat) avant qu'un rapport ne signale l'omission. Le fichier v1 n'était pas suivi par git (`??` en status) : **checkpoint v1 irrécupérable**. À traiter comme un fait acquis, pas une action corrective possible a posteriori.
- [x] `dataset_configs.py` -- `input_shape=(71,)`, `token_dim=64`, `num_trunk_layers=2` -- pistes 1+3
- [x] `data_management.py` -- packer `from_square_target` en 71e colonne -- piste 1
- [x] `model_library.py` -- `read_token` + conditionnement `move_type_head` + `expected_width=71` -- pistes 1+2
- [x] `tests/test_chess_token_model.py` -- tests shape/read-token/conditionnement train vs éval -- non-régression + validation
- [x] `pytest` puis `python main.py CHESS_TOKEN_1_MOVE` (smoke test court, PUIS arrêt manuel) -- valider avant tout training long

**Acceptance Criteria:**
- [x] Given `from_square_teacher` variable et `pooled` fixé, when `training=True`, then `move_type_logits` varie avec `from_square_teacher` -- `test_one_move_move_type_varies_with_teacher_at_train` PASSED
- [x] Given `from_square_teacher` variable et `pooled` fixé, when `training=False`, then `move_type_logits` reste identique (pas de fuite) -- `test_one_move_move_type_invariant_to_teacher_at_eval_anti_leak` PASSED
- [x] Given la config `CHESS_TOKEN_1_MOVE` modifiée et le dataset réel, when `Trainer` lance quelques steps, then aucune erreur de forme/dtype et la loss varie sans NaN -- smoke test réel (GPU, PID 16159, tué manuellement en cours d'epoch 2/50) : `Train | Loss=5.6521 | JointMoveAccuracy=0.0364 | Time=507.5s` puis `Val | Loss=5.1138 | JointMoveAccuracy=0.0515 | Time=17.8s`, aucun warning "Loss non-finie" sur toute l'epoch 1
- [x] Given la suite de tests existante, when exécutée après cette implémentation, then elle passe sans régression sur les autres domaines échecs -- 58/58 tests verts sur les 6 fichiers de tests échecs (`test_chess_model.py`, `test_chess_search_teacher_loader.py`, `test_no_chess_dependency.py`, `test_chess_move_token_model.py`, `test_chess_legal_moves_model.py`, `test_chess_token_model.py`) ; suite complète `tests/` = 132 passed/5 errors, les 5 erreurs sont des fixtures manquantes pré-existantes et sans rapport (`test_detector_inference_composition.py`, `test_differentiable_crop_classification.py`, `test_pixel_parity.py`, aucun de ces fichiers modifié par ce spec)

## Design Notes

`training` comme branche Python-level (pas une valeur tracée) : vérifier avant implémentation que `apply_fn` reçoit `training` comme bool statique (déjà le cas pour `deterministic=not training` dans cette même classe) — sinon utiliser un `jnp.where` traçable plutôt qu'un `if` Python pour le choix teacher-forcing/argmax. Confirmé après implémentation : `trainer.py::_create_train_step`/`_create_eval_step` passent bien `training` en littéral Python, jamais tracé (revue Blind Hunter, 2026-08-16).

**Caveat lecture des résultats (ajouté 2026-08-16, revue Blind Hunter, patch appliqué directement — voir aussi le commentaire miroir dans `task_strategies.py::ChessTokenOneMoveStrategy`)** : `compute_metrics` est appelé côté train ET val sur les `outputs` renvoyés par le MÊME `apply_fn`, mais avec `training` différent — donc côté train, `move_type` a été calculé avec le vrai `from_square` (teacher-forcing, plus facile), côté val avec `argmax(from_square_logits)` (chemin sans fuite, plus dur). La `JointMoveAccuracy` TRAIN de v2 n'est donc plus comparable au même titre que v1 au diagnostic "train≈val = plafond, pas overfitting" — seule la valeur VAL est comparable au plafond v1 (5,10%) et entre runs v2. Corriger structurellement nécessiterait un forward pass supplémentaire dans `Trainer._train_step`, interdit par le "Never" ci-dessus (ne pas toucher `trainer.py`) — caveat documenté plutôt que corrigé.

**Autre caveat, non actionnable en code (signalé par Blind Hunter, à garder en tête pour l'interprétation du résultat)** : le conditionnement réutilise `pos_embed` à la fois comme encodage positionnel du tronc ET comme représentation du `from_square` de conditionnement — un seul tableau de poids sert deux rôles sous deux chemins de gradient différents, risque d'interférence jamais mesuré. Si le run v2 ne dépasse pas significativement le plafond v1, cette réutilisation (plutôt que les 3 leviers eux-mêmes) est une hypothèse à considérer avant de conclure à un échec des 3 leviers.

## Verification

**Commands:**
- `pytest tests/test_chess_token_model.py` -- expected: tous les tests passent (nouveaux + existants, aucune régression)
- `python main.py CHESS_TOKEN_1_MOVE` (arrière-plan, quelques steps SEULEMENT puis arrêt manuel) -- expected: démarre sans erreur, loss loggée cohérente, pas de training long lancé

## Suggested Review Order

**Conditionnement `move_type` sur `from_square` (le cœur de la piste v2)**

- Entrée : la 71e colonne, ajoutée pour transporter le label réel jusqu'au modèle.
  [`model_library.py:1598`](../../model_library.py#L1598)

- `from_square_logits` reste calculé sur `pooled` seul — la prédiction libre est préservée.
  [`model_library.py:1656`](../../model_library.py#L1656)

- Le point le plus sensible : bascule teacher-forcing (train) / argmax (éval), jamais le label en éval.
  [`model_library.py:1658-1675`](../../model_library.py#L1658)

- Test anti-fuite : `move_type_logits` doit varier avec le label en train.
  [`test_chess_token_model.py:638`](../../tests/test_chess_token_model.py#L638)

- Test anti-fuite : `move_type_logits` doit être invariant au label en éval — le test le plus important du diff.
  [`test_chess_token_model.py:676`](../../tests/test_chess_token_model.py#L676)

**Read-token (remplace le mean-pool)**

- 65e token appris, concaténé à la séquence AVANT le tronc — pas un étage de cross-attention séparé.
  [`model_library.py:1619-1623`](../../model_library.py#L1619)

- Extraction en sortie du tronc : dernière position de la séquence remplace `jnp.mean`.
  [`model_library.py:1650`](../../model_library.py#L1650)

**Plomberie du label (dataset)**

- `from_square_target` dérivé une seule fois à la construction, packé en 71e colonne.
  [`data_management.py:1353-1364`](../../data_management.py#L1353)

**Config (tronc élargi + nouvelle largeur d'entrée)**

- `input_shape=(71,)`, `token_dim=64`, `num_trunk_layers=2` — pistes 1+3 combinées.
  [`dataset_configs.py:1250`](../../dataset_configs.py#L1250)

- Correction post-revue : commentaire de backup checkpoint rendu honnête (aucun mécanisme automatique n'existe).
  [`dataset_configs.py:1258`](../../dataset_configs.py#L1258)

**Gouvernance (renégociation du frozen v1 + caveats de lecture)**

- Annotation SUPERSEDED de la contrainte "sans conditionnement" du spec v1.
  [`spec-chess-token-1-move.md:24`](spec-chess-token-1-move.md#L24)

- Contrat partagé `chess_ai` : reprise actée, résultat v1 documenté.
  [`contract-chess-ai-training-interface.md:282-303`](../../docs/contract-chess-ai-training-interface.md#L282)

- Caveat ajouté post-revue : `JointMoveAccuracy` train n'est plus comparable à val pour v2 (asymétrie teacher-forcing/argmax).
  [`task_strategies.py:827-838`](../../task_strategies.py#L827)

**Peripherals**

- Commentaire de largeur d'entrée corrigé (70→71), stale avant la revue.
  [`test_chess_token_model.py:557-559`](../../tests/test_chess_token_model.py#L557)
