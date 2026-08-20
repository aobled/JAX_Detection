---
baseline_commit: 0d6ab1f78adc299922e1f4630fb3063bffa55de3
---

# Story 12.1: `build_kwargs_from_config` et migration de `model_kwargs`

Status: done

<!-- Note: Validation is optional. Run validate-create-story for quality check before dev-story. -->

## Story

As a mainteneur du pipeline d'entraînement,
I want une fonction générique d'introspection qui construit les kwargs d'une factory modèle depuis la config et des valeurs calculées,
so that `main.py` n'a plus besoin d'une branche `if "X" in config` par hyperparamètre de modèle, sans rien casser pour les modèles qui ne reçoivent ces hyperparamètres que via `**kwargs`.

## Acceptance Criteria

1. **Given** `build_kwargs_from_config(target, config, **overrides)` **When** appelée avec une clé d'`overrides` que la signature de `target` ne nomme PAS explicitement (`target` n'a qu'un `**kwargs` catch-all) **Then** cette clé n'est PAS forwardée — préserve la discipline stricte héritée d'`AD-3` (spine frère `architecture-compute-dtype-hardware-2026-08-17`).
2. **Given** la même fonction **When** appelée avec une clé de `config` que la signature de `target` ne nomme pas explicitement mais que `target` possède un `**kwargs` catch-all **Then** cette clé EST forwardée via `**kwargs` — préserve le comportement réel actuel des 5 factories `create_chess_*`/`create_aircraft_detector_unet`.
3. **Given** une clé présente à la fois dans `config` et dans `overrides` **When** les kwargs sont construits **Then** la valeur d'`overrides` gagne silencieusement — contrat documenté, pas une ambiguïté.
4. **Given** `main.py:135-188` (avant migration) **When** la construction de `model_kwargs` est migrée **Then** elle devient un unique appel à `build_kwargs_from_config` avec `compute_dtype`/`dropout_rate`/`num_classes` en `overrides` — plus aucune branche `if "X" in config: model_kwargs["X"] = ...`.
5. **Given** le print diagnostique `compute_dtype` (« injecté »/« non applicable ») **When** migré **Then** son booléen provient de la décision de forwarding réellement prise par le helper pour la clé `compute_dtype` (le helper expose quelles clés d'`overrides` ont été forwardées) — jamais une vérification `inspect.signature` dupliquée indépendamment dans `main.py`.
6. **Given** un nouveau module de test dédié (`tests/test_build_kwargs_from_config.py`) **When** il exerce `build_kwargs_from_config` en isolation (cible factice avec paramètres nommés + `**kwargs`, dict `config`, dict `overrides`) **Then** il couvre : override forwardé seulement si nommé explicitement ; clé `config` forwardée sans condition même via `**kwargs` ; `overrides` gagne sur une clé `config` de même nom ; une clé absente des deux n'est jamais comblée par une valeur par défaut choisie par le helper.
7. **Given** les 5 factories `create_chess_cnn_attention_policy_value`/`create_chess_cnn_attention_legal_moves`/`create_chess_move_token_transformer`/`create_chess_token_candidate_model`/`create_chess_token_one_move_model` **When** chacune est appelée via `build_kwargs_from_config` avec sa vraie entrée `dataset_configs.py` **Then** une instanciation réelle (`model.init()`, pas mockée) produit un modèle dont les formes de paramètres reflètent exactement l'hyperparamètre configuré (ex. nombre de tokens du bottleneck, `d_model`) — garde-fou critique trouvé en revue de spine (`review-web-verify.md`).
8. **Given** la suite de tests existante touchant `compute_dtype`/le forwarding de modèle (ex. `tests/test_compute_dtype_hardware.py`) **When** exécutée après cette story **Then** elle passe intégralement, sans modification.

## Tasks / Subtasks

- [x] Task 1 : Implémenter `build_kwargs_from_config` dans `model_library.py` (AC: #1, #2, #3)
  - [x] Signature `def build_kwargs_from_config(target, config, config_keys=(), **overrides) -> (dict, frozenset)` — **déviation documentée par rapport au texte de la Task, voir Dev Agent Record**
  - [x] Résoudre `inspect.signature(target).parameters` une seule fois ; détecter le `kind` de chaque paramètre (`VAR_KEYWORD` pour `**kwargs`)
  - [x] Canal `overrides` (strict) : pour chaque clé de `overrides`, ne forwarder que si un paramètre du même nom existe dans la signature avec un `kind` ≠ `VAR_KEYWORD`
  - [x] Canal `config_keys` (inconditionnel) : pour chaque nom de `config_keys` présent dans `config`, forwarder tel quel — **pas** "chaque clé de `config`" comme écrit initialement dans cette Task (voir déviation)
  - [x] `overrides` gagne sur `config`/`config_keys` pour une clé partagée
  - [x] Ne jamais injecter de valeur par défaut choisie par le helper lui-même pour une clé absente des deux sources — omise du dict retourné
  - [x] Paramètre requis absent des deux sources : aucune vérification ajoutée, le `TypeError` natif de l'appel `target(**kwargs)` en aval se propage tel quel
  - [x] Retourne `(kwargs, forwarded_override_keys)` — `forwarded_override_keys` réutilisé par le print diagnostique de Task 2
- [x] Task 2 : Migrer la construction de `model_kwargs` dans `main.py` (AC: #4, #5)
  - [x] Remplacé les 7 branches `if "X" in config` (`main.py:136-171` avant migration) par `MODEL_HYPERPARAM_CONFIG_KEYS` (constante module-level) + un appel à `build_kwargs_from_config`
  - [x] Remplacé le bloc d'injection `compute_dtype` (`main.py:172-188` avant migration) — `compute_dtype`/`dropout_rate`/`num_classes` passent en `overrides`
  - [x] Print diagnostique adapté : lit `"compute_dtype" in forwarded_overrides`, plus de vérification `inspect.signature` dupliquée dans `main.py` (import `inspect` retiré de `main.py`, devenu inutile)
  - [x] `model = get_model(model_name, **model_kwargs)` inchangé en aval
- [x] Task 3 : Tests dédiés d'isolation (AC: #6)
  - [x] Créé `tests/test_build_kwargs_from_config.py` (6 tests, tous verts)
  - [x] Cas 1 (override non nommé, jamais absorbé) — `test_override_forwarded_only_if_named_explicitly`
  - [x] Cas 2 (config_keys forwardé sans condition même via `**kw`) — `test_config_keys_forwarded_unconditionally_even_via_kwargs`
  - [x] Cas 3 (`overrides` gagne) — `test_overrides_wins_over_config_key_of_same_name`
  - [x] Cas 4 (clé absente jamais comblée par un défaut du helper) — `test_missing_key_never_filled_with_helper_chosen_default`
  - [x] Chaque cas asserte aussi `forwarded_override_keys` (valeur de retour secondaire)
- [x] Task 4 : Validation par instanciation réelle des 5 factories `**kwargs` (AC: #7)
  - [x] Créé `tests/test_model_kwargs_real_instantiation.py` — pour chacune des 5 factories, `model_kwargs` construit via `build_kwargs_from_config` à partir de sa vraie entrée `dataset_configs.py`, puis `target_factory(**model_kwargs)` + `model.init({"params": rng, "dropout": rng}, x, training=True)` (fixtures d'entrée basées sur le pattern des tests existants : `test_chess_model.py`/`test_chess_legal_moves_model.py`/`test_chess_move_token_model.py`/`test_chess_token_model.py`)
  - [x] Effet réel vérifié par comparaison de comptage total de paramètres (valeur configurée vs une valeur volontairement différente) — pas juste l'absence d'exception : `num_bottleneck_tokens` (policy_value, legal_moves), `num_layers` (move_token_transformer), `token_dim` (token_candidate_model), `num_trunk_layers` (token_one_move_model)
- [x] Task 5 : Non-régression (AC: #8)
  - [x] Suite complète `pytest tests/` (hors `fixtures/`) : 156 passed, 0 nouvel échec — 5 erreurs de collecte pytest préexistantes et sans rapport (`test_differentiable_crop_classification.py`/`test_pixel_parity.py`/`test_detector_inference_composition.py`, fichiers non modifiés par cette story, convention "arguments positionnels" du projet mal interprétée par la collecte `pytest` bare — pas une régression introduite ici, vérifié via `git status --porcelain` sur ces 3 fichiers avant cette story)

## Dev Notes

- **Ne pas toucher** `task_strategies.py`, `data_management.py`, `dataset_configs.py` dans cette story — le dispatch `Strategy`/`STRATEGIES` est Story 12.2, qui dépend de celle-ci.
- **Comportement actuel de `main.py:135-188` (avant migration) à préserver à l'identique**, sauf pour le mécanisme interne :
  ```python
  model_kwargs = {"num_classes": num_classes, "dropout_rate": dropout_rate}
  if "heatmap_prior" in config: model_kwargs["heatmap_prior"] = config["heatmap_prior"]
  if "num_bottleneck_tokens" in config: model_kwargs["num_bottleneck_tokens"] = config["num_bottleneck_tokens"]
  if "token_dim" in config: model_kwargs["token_dim"] = config["token_dim"]
  if "num_layers" in config: model_kwargs["num_layers"] = config["num_layers"]
  if "d_model" in config: model_kwargs["d_model"] = config["d_model"]
  if "num_heads" in config: model_kwargs["num_heads"] = config["num_heads"]
  if "num_trunk_layers" in config: model_kwargs["num_trunk_layers"] = config["num_trunk_layers"]
  target_factory = MODELS.get(model_name)
  compute_dtype_injected = target_factory is not None and "compute_dtype" in inspect.signature(target_factory).parameters
  if compute_dtype_injected:
      model_kwargs["compute_dtype"] = compute_dtype
  print(f"🔢 compute_dtype pour '{model_name}': {'injecte (' + compute_dtype.__name__ + ')' if compute_dtype_injected else 'non applicable (modele non adapte)'}")
  model = get_model(model_name, **model_kwargs)
  ```
  Cible après migration (forme indicative, pas prescriptive sur les noms internes) :
  ```python
  target_factory = MODELS.get(model_name)
  model_kwargs, forwarded_overrides = build_kwargs_from_config(
      target_factory, config, compute_dtype=compute_dtype, dropout_rate=dropout_rate, num_classes=num_classes
  )
  compute_dtype_injected = "compute_dtype" in forwarded_overrides
  print(f"🔢 compute_dtype pour '{model_name}': {'injecte (' + compute_dtype.__name__ + ')' if compute_dtype_injected else 'non applicable (modele non adapte)'}")
  model = get_model(model_name, **model_kwargs)
  ```
  `target_factory` peut être `None` si `model_name` est invalide (`MODELS.get()`, pas `MODELS[...]`, précédent déjà en place ligne 184) — `build_kwargs_from_config` doit gérer ce cas sans planter avant que `get_model()` ne lève son `ValueError` explicite habituel (comportement actuel à préserver).

- **Pourquoi deux canaux, pas un seul mécanisme strict** (le piège du premier jet d'AD-21, trouvé en Reviewer Gate) : 5 factories de `model_library.py` ne nomment PAS explicitement `num_bottleneck_tokens`/`token_dim`/`num_layers`/`d_model`/`num_heads`/`num_trunk_layers` dans leur propre signature — elles les reçoivent uniquement via `**kwargs`, retransmis fidèlement à leur `nn.Module` sous-jacent :
  - `create_chess_cnn_attention_policy_value(num_classes, dropout_rate=0.1, **kwargs)` → `model_library.py:925`
  - `create_chess_cnn_attention_legal_moves(num_classes, dropout_rate=0.1, **kwargs)` → `model_library.py:1028`
  - `create_chess_move_token_transformer(num_classes, dropout_rate=0.1, compute_dtype="float32", **kwargs)` → `model_library.py:1168` (note : ce `compute_dtype` string-ou-dtype est un cas historique préservé tel quel, pas touché par cette story)
  - `create_chess_token_candidate_model(num_classes, dropout_rate=0.1, **kwargs)` → `model_library.py:1423`
  - `create_chess_token_one_move_model(num_classes=None, dropout_rate=0.1, **kwargs)` → `model_library.py:1659`
  Ces modèles sont réellement entraînés (F1=0.8578 sur `CHESS_LEGAL_MOVES`, entre autres). Une discipline d'introspection stricte appliquée à *toutes* les clés (comme `AD-3` l'impose pour `compute_dtype` spécifiquement) casserait silencieusement leur forwarding. D'où : `overrides` (valeurs *calculées* par `main.py` : `compute_dtype`, `dropout_rate`, `num_classes`) reste strict (jamais `**kwargs`) ; les clés `config` (tout le reste, y compris `heatmap_prior` qui EST nommé explicitement dans `create_aircraft_detector_centernet`) sont forwardées sans condition.

- **`compute_dtype` reste protégé par `AD-1` hérité** (spine frère `architecture-compute-dtype-hardware-2026-08-17`) : aucune entrée de `dataset_configs.py` ne doit jamais déclarer une clé `compute_dtype` — ce n'est de toute façon pas un risque introduit par cette story (le canal `overrides` reste strict), mais à ne pas régresser en implémentation.

- **`resolve_compute_dtype(backend)`** (`model_library.py:1763-1776`) est le précédent direct pour l'emplacement et l'esprit de `build_kwargs_from_config` : extrait de `main.py` en fonction pure testable en isolation, avec sa propre suite de tests dédiée (`tests/test_compute_dtype_hardware.py`). Même discipline à suivre.

- **`get_model(model_name, **kwargs)`** (`model_library.py:1801-1815`) n'est PAS modifié par cette story — il continue de recevoir `**model_kwargs` en aval, inchangé.

### Project Structure Notes

- Fichiers touchés : `model_library.py` (ajout de `build_kwargs_from_config`, placée à proximité de `resolve_compute_dtype`/`get_model`), `main.py` (migration de la construction `model_kwargs`, lignes ~135-188).
- Nouveau fichier : `tests/test_build_kwargs_from_config.py` (suit la convention `tests/test_<module_ou_fonction>.py` déjà en place, ex. `tests/test_compute_dtype_hardware.py`, `tests/test_chess_model.py`).
- Aucun nouveau module/fichier de production — pas de dossier `library/`/`utils/` générique (convention projet déjà actée : Rule of Three avant centralisation, modules nommés par leur objet).
- Aucune nouvelle dépendance externe (`inspect` est stdlib).

### References

- [Source: `_bmad-output/planning-artifacts/architecture/architecture-jax_supervised_training-2026-07-15/ARCHITECTURE-SPINE.md#AD-21`] — règle complète, canaux `overrides`/`config`, contrainte héritée `AD-1`/`AD-3`.
- [Source: `_bmad-output/planning-artifacts/architecture/architecture-jax_supervised_training-2026-07-15/.memlog.md`] — historique complet de la décision (entrées à partir de « Resumed 2026-08-19 pour un nouvel amendement... »), y compris le finding critique du Reviewer Gate (canal `**kwargs`) et sa correction.
- [Source: `_bmad-output/planning-artifacts/architecture/architecture-compute-dtype-hardware-2026-08-17/ARCHITECTURE-SPINE.md#AD-1,AD-3`] — contrainte héritée, lue mais non modifiée.
- [Source: `_bmad-output/planning-artifacts/epics.md#Epic-12`] — Epic 12, Story 12.1, FR1/FR2/NFR1/NFR2.
- [Source: `main.py:71-189`] — code réel à migrer (fonction `main()`, section `CRÉATION DU MODÈLE`).
- [Source: `model_library.py:925,1028,1168,1423,1659,1749-1815`] — factories `**kwargs`-only, `MODELS`, `get_model`, `resolve_compute_dtype` (précédent d'emplacement/testabilité).

## Change Log

- 2026-08-19 : Story 12.1 implémentée intégralement (Tasks 1-5, AC1-8) en une session. Un écart réel par rapport au texte initial de la Task 1/Dev Notes trouvé pendant l'implémentation (mécanisme "forwarder tout `config` sans condition" aurait cassé les 5 factories `**kwargs`-only avec une config réelle) — corrigé par un paramètre `config_keys` explicite, voir Completion Notes.
- 2026-08-19 : `AD-21` (spine) et `epics.md#FR1` corrigés pour refléter le mécanisme `config_keys` réellement implémenté (Winston).
- 2026-08-19 : Revue de code (`/code-review high`, 7 agents parallèles) — 3 findings dans le périmètre de cette story corrigés (bug réel `create_kepler_1d_cnn` num_classes/dropout_rate jamais forwardés, bootstrap `sys.path` manquant dans `test_build_kwargs_from_config.py`, déduplication `MODEL_HYPERPARAM_CONFIG_KEYS` → `MODEL_FORWARDED_CONFIG_KEYS` dans `model_library.py`). 2 findings hors périmètre (gaps de couverture sur `tests/test_compute_dtype_hardware.py`, 3 bugs sur `tools/boxes_process_manual_tkinter.py`) signalés à Aymeric, non corrigés ici. Suite complète re-vérifiée après fixes : 157 passed, 0 régression. Statut : `done`.

## Dev Agent Record

### Agent Model Used

Claude Sonnet 5 (bmad-dev-story)

### Debug Log References

- `pytest tests/ --ignore=tests/fixtures` : 157 passed (post-revue), 5 erreurs de collecte préexistantes/sans rapport (voir Completion Notes)
- `pytest tests/test_compute_dtype_hardware.py tests/test_chess_model.py tests/test_chess_legal_moves_model.py tests/test_chess_move_token_model.py tests/test_chess_token_model.py tests/test_aircraft_detector_centernet.py tests/test_jax_detector_config.py tests/test_no_chess_dependency.py` : 81 passed, 0 échec
- `pytest tests/test_build_kwargs_from_config.py tests/test_model_kwargs_real_instantiation.py -v` : 12 passed (post-revue, +1 test kepler)
- `python tests/test_build_kwargs_from_config.py` / `python tests/test_model_kwargs_real_instantiation.py` (invocation directe, hors pytest) : tous verts

### Completion Notes List

- **Déviation par rapport au texte initial de la Task 1/Dev Notes, appliquée en cours d'implémentation** : le mécanisme décrit dans la story ("canal config forwardé sans condition, que `target` le nomme ou l'absorbe via `**kwargs`") aurait forwardé **tout** `config` (le dict `dataset_configs.py` complet — task_type, optimizer, tpu/gpu, loss_params, etc., une vingtaine de clés) vers toute factory ayant un `**kwargs`, dont 5 le retransmettent fidèlement à leur `nn.Module` qui n'a lui-même pas de `**kwargs` — crash `TypeError` immédiat et systématique (vérifié empiriquement avant d'écrire le code, avec la config réelle `CHESS_LEGAL_MOVES`). Corrigé en ajoutant un paramètre explicite `config_keys` à `build_kwargs_from_config`, sans vérification de signature, exactement comme les 7 anciennes branches `if "X" in config`. **`AD-21` (spine) et `epics.md#FR1` corrigés en conséquence** (au-delà du périmètre `dev-story`, fait par Winston juste après cette story).
- **Comportement observé, sans régression** : `create_aircraft_detector_centernet` ne nomme pas `num_classes` explicitement (seulement `**kwargs`, jamais retransmis à `AircraftDetectorCenterNet` — absorption déjà documentée dans `model_library.py` avant cette story). Avant migration, `num_classes` était donc toujours passé mais silencieusement ignoré ; après migration (canal `overrides` strict), `num_classes` n'est simplement plus construit dans `model_kwargs` pour ce modèle. Comportement runtime du modèle final identique dans les deux cas.
- **Revue de code (`/code-review high`, 7 agents parallèles) — 3 findings corrigés :**
  1. **[Bug réel, confirmé indépendamment par 2 agents]** `create_kepler_1d_cnn` ne nommait ni `num_classes` ni `dropout_rate` (seulement `compute_dtype` + `**kwargs`) — le canal `overrides` strict ne les forwardait donc jamais, masqué uniquement parce que les défauts de `Kepler1DConvNet` (`num_classes=2`, `dropout_rate=0.3`) coïncident avec `JAX_KEPLER`. Corrigé en nommant les deux paramètres explicitement dans la factory (`model_library.py`, même discipline qu'AD-3) + test de régression dédié (`test_kepler_1d_cnn_num_classes_and_dropout_rate_real_effect`, valeurs délibérément différentes des défauts).
  2. `tests/test_build_kwargs_from_config.py` n'avait pas le bootstrap `sys.path.insert` présent dans tous les autres fichiers de test du projet — échouait en invocation directe (`python tests/test_X.py`, convention documentée du projet), ne passait qu'via `pytest` (rootdir masque le problème). Corrigé.
  3. `MODEL_HYPERPARAM_CONFIG_KEYS` était une liste plate dupliquée entre `main.py` et `tests/test_model_kwargs_real_instantiation.py` (synchronisation manuelle, commentaire seul comme garde-fou), mélangeant les hyperparamètres de 5 modèles sans scoping. Remplacé par `MODEL_FORWARDED_CONFIG_KEYS` (dict `model_name -> tuple`), source unique dans `model_library.py` à côté de `MODELS`, importé par `main.py` et les tests — plus de duplication, scoping par modèle explicite.
- Les autres findings de la revue (tests `test_compute_dtype_hardware.py` — ordre recast/sigmoid non testé précisément, `training=True` non combiné à `jax.grad` — et le fichier `tools/boxes_process_manual_tkinter.py`) portent sur du code **hors périmètre de cette story** (commit `0d6ab1f` déjà en place / modifications non liées, présentes avant cette session) — signalés à Aymeric séparément, pas corrigés ici.
- Toutes les ACs (#1 à #8) vérifiées par test réel, pas par lecture de code — voir Debug Log References.
- `import inspect` retiré de `main.py` (devenu inutile après migration, l'introspection vit désormais uniquement dans `model_library.py::build_kwargs_from_config`).

### File List

- `model_library.py` (modifié) : ajout `import inspect`, ajout `build_kwargs_from_config`, ajout `MODEL_FORWARDED_CONFIG_KEYS` (dict par modèle), correction `create_kepler_1d_cnn` (nomme désormais `num_classes`/`dropout_rate` explicitement)
- `main.py` (modifié) : import `build_kwargs_from_config`/`MODEL_FORWARDED_CONFIG_KEYS`, migration de la construction `model_kwargs`, retrait `import inspect`
- `tests/test_build_kwargs_from_config.py` (nouveau) : tests d'isolation du helper
- `tests/test_model_kwargs_real_instantiation.py` (nouveau) : validation par instanciation réelle des 5 factories `**kwargs`-only
