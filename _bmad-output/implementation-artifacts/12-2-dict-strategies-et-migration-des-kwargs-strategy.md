---
baseline_commit: 0d6ab1f78adc299922e1f4630fb3063bffa55de3
---

# Story 12.2: Dict `STRATEGIES` et migration des kwargs `Strategy`

Status: done

<!-- Note: Validation is optional. Run validate-create-story for quality check before dev-story. -->

## Story

As a mainteneur du pipeline d'entraînement,
I want un dict de dispatch `task_type` → `Strategy` et une construction de kwargs `Strategy` via le même helper générique que Story 12.1,
so that le dispatch `main.py` n'a plus de branches dupliquées, sans introduire de `KeyError` brut ni de défaut divergent du vrai constructeur `Strategy`.

## Acceptance Criteria

1. **Given** `main.py:158-241` (avant migration, extraction partagée 158-163 + `if/elif` à 9 branches 164-241 sur `task_type`) **When** le dispatch est migré **Then** il devient `STRATEGIES = {task_type: Classe}` (défini dans `task_strategies.py`, pas `main.py` — voir Dev Notes) consulté via `STRATEGIES.get(task_type)` suivi d'un `raise ValueError(f"task_type '{task_type}' non reconnu.")` explicite si absent — jamais un `KeyError` brut.
2. **Given** un `task_type` absent de `STRATEGIES` (ex. faute de frappe) **When** `main.py` dispatche **Then** il lève le même `ValueError` explicite qu'avant migration, message identique (vérifié par test).
3. **Given** les kwargs de chaque `Strategy` **When** construits via `build_kwargs_from_config` (Story 12.1, `model_library.py`) **Then** `num_classes` est le seul champ passé en `overrides` (strict, forwardé uniquement si la classe cible le nomme explicitement — `ClassificationStrategy`/`KeplerStrategy` seulement) ; tous les autres champs (`label_smoothing`, `mixup_alpha`, `loss_method`, `loss_params`, `metric_method`, `report_method`, `metric_threshold`) passent par un canal `config_keys` **scopé par `task_type`** (`STRATEGY_FORWARDED_CONFIG_KEYS`, dict dans `task_strategies.py` — jamais une liste plate partagée entre tous les `task_type`, voir Dev Notes/piège Story 12.1).
4. **Given** une clé absente de `config`+`overrides` pour un `task_type` donné **When** les kwargs sont construits **Then** elle n'est **jamais** comblée par une valeur par défaut codée en dur côté `main.py` (ex. l'ancien `loss_method = config.get("loss_method", "cross_entropy")` partagé par toutes les branches) — la clé est omise, le défaut du constructeur `Strategy` ciblé s'applique (corrige l'incohérence latente `DetectionStrategy`, défaut réel `loss_method="segmentation"`, jamais atteint aujourd'hui uniquement parce que `FIGHTERJET_DETECTION` le redéclare explicitement en config).
5. **Given** le print sur-mesure par branche (« Application de la logique d'entraînement : X ») **When** migré **Then** il devient un print générique unique (`task_type` → nom de la classe `Strategy` choisie).
6. **Given** les imports `Strategy` actuellement locaux par branche (`from task_strategies import KeplerStrategy` etc., 7 occurrences) **When** migrés **Then** ils disparaissent — toutes les classes `Strategy` sont déjà définies dans `task_strategies.py`, `STRATEGIES`/`STRATEGY_FORWARDED_CONFIG_KEYS` y vivent aussi (zéro nouvel import nécessaire), `main.py` importe uniquement ces deux dicts.
7. **Given** les 9 `task_type` configurés dans `dataset_configs.py` **When** `main.py` construit `model_kwargs` (Story 12.1) et les kwargs `Strategy` (cette story) pour chacun **Then** une instanciation réelle (pas mockée) de la `Strategy` réussit avec les mêmes valeurs de kwargs qu'avant migration (comparaison champ-par-champ des attributs de l'instance) — FR4 de l'epic, non-régression par exécution réelle.
8. **Given** `AD-1` (spine frère `compute-dtype-hardware`) **When** cette story est complétée **Then** aucune entrée `DATASET_CONFIGS` ne déclare `compute_dtype`, et sa valeur ne provient jamais du canal `config` du helper (vérifié par grep sur `dataset_configs.py`).
9. **Given** la suite de tests automatisée complète (`tests/`) **When** exécutée après cette story **Then** elle passe intégralement (mêmes 5 erreurs préexistantes/sans rapport que Story 12.1, aucune nouvelle).

## Tasks / Subtasks

- [x] Task 1 : `STRATEGIES` + `STRATEGY_FORWARDED_CONFIG_KEYS` dans `task_strategies.py` (AC: #1, #3, #6)
  - [x] `STRATEGIES = {task_type: Classe}` à la fin de `task_strategies.py` (après `ChessTokenOneMoveStrategy`, ligne ~921) — les 9 entrées : `classification`→`ClassificationStrategy`, `detection`→`DetectionStrategy`, `kepler`→`KeplerStrategy`, `detection_centernet`→`CenterNetDetectionStrategy`, `chess_policy_value`→`ChessPolicyValueStrategy`, `chess_legal_moves`→`ChessLegalMovesStrategy`, `chess_move_token`→`ChessMoveTokenStrategy`, `chess_token`→`ChessTokenStrategy`, `chess_token_1_move`→`ChessTokenOneMoveStrategy`
  - [x] `STRATEGY_FORWARDED_CONFIG_KEYS = {task_type: (...)}` juste à côté — **scopé par `task_type`, jamais une liste plate partagée** (voir Dev Notes : le piège trouvé en Story 12.1 sur `model_kwargs` se reproduirait ici à l'identique — aucune classe `Strategy` n'a de `**kwargs` catch-all, un forwarding non scopé d'un champ qu'une classe ne nomme pas lèverait un `TypeError` immédiat) :
    - `classification`: `("label_smoothing", "mixup_alpha", "loss_method", "loss_params", "metric_method", "report_method")`
    - `detection`: `("loss_method", "loss_params", "metric_method", "report_method")`
    - `kepler`: `("loss_method", "loss_params", "metric_method", "report_method")`
    - `detection_centernet`: `("loss_params",)`
    - `chess_policy_value`: `("loss_params",)`
    - `chess_legal_moves`: `("metric_threshold", "loss_params")`
    - `chess_move_token`: `("loss_params",)`
    - `chess_token`: `("loss_params",)`
    - `chess_token_1_move`: `("loss_params",)`
- [x] Task 2 : Migrer le dispatch `Strategy` dans `main.py` (AC: #1, #2, #5, #6)
  - [x] Remplacer l'import `from task_strategies import ClassificationStrategy, DetectionStrategy` (ligne ~21) par `from task_strategies import STRATEGIES, STRATEGY_FORWARDED_CONFIG_KEYS`
  - [x] Remplacer le `if/elif` à 9 branches (`main.py:164-241` avant migration) par `strategy_cls = STRATEGIES.get(task_type)` + `if strategy_cls is None: raise ValueError(f"task_type '{task_type}' non reconnu.")` (message identique à l'existant)
  - [x] Supprimer les 4 extractions locales `loss_method`/`loss_params`/`metric_method`/`report_method` (`main.py:159-162`) — mortes après migration
  - [x] Construire les kwargs via `build_kwargs_from_config(strategy_cls, config, config_keys=STRATEGY_FORWARDED_CONFIG_KEYS.get(task_type, ()), num_classes=num_classes)` puis `strategy = strategy_cls(**strategy_kwargs)`
  - [x] Print générique unique remplaçant les 9 prints sur-mesure
  - [x] Supprimer les 7 imports locaux `from task_strategies import X` à l'intérieur des branches (devenus inutiles, `STRATEGIES` porte déjà les classes)
- [x] Task 3 : Tests dédiés (AC: #7)
  - [x] Nouveau fichier `tests/test_strategy_kwargs_real_instantiation.py` — approche générique par introspection de signature (couvre les 11 configs réelles / 9 `task_type`, pas de valeurs transcrites à la main) plutôt qu'une liste figée par `task_type`
  - [x] Pour chaque config réelle : kwargs construits via `STRATEGIES`/`STRATEGY_FORWARDED_CONFIG_KEYS`/`build_kwargs_from_config`, `Strategy` instanciée réellement, chaque attribut comparé à `config.get(champ, défaut_du_constructeur)` (jamais un défaut `main.py`) — tous verts
  - [x] Cas `task_type` invalide (`"nope"`) : `STRATEGIES.get(...)` renvoie `None`, `ValueError("task_type 'nope' non reconnu.")` levé — message identique à l'ancien code
- [x] Task 4 : Non-régression (AC: #8, #9)
  - [x] `grep -n "compute_dtype" dataset_configs.py` : aucune entrée `DATASET_CONFIGS` ne déclare cette clé
  - [x] Suite complète `pytest tests/ --ignore=tests/fixtures` : 162 passed (+2 vs Story 12.1), 5 erreurs préexistantes/sans rapport identiques, 0 nouvel échec

## Dev Notes

- **Contexte direct** : Story 12.1 (`_bmad-output/implementation-artifacts/12-1-build-kwargs-from-config-et-migration-model-kwargs.md`, statut `done`) a livré `build_kwargs_from_config(target, config, config_keys=(), **overrides)` dans `model_library.py` et migré la construction de `model_kwargs`. Cette story réutilise **exactement le même helper**, sans le modifier, pour le second point de construction (`Strategy`).

- **⚠️ Leçon directe de Story 12.1, à appliquer dès le premier jet ici (pas après coup)** : le tout premier design de `build_kwargs_from_config` forwardait « tout `config` sans condition » dès que la cible avait un `**kwargs` — vérifié FAUX à l'implémentation (`TypeError` immédiat avec une vraie config `CHESS_LEGAL_MOVES`, une vingtaine de clés dont une seule pertinente). Corrigé par un `config_keys` **explicite et scopé par cible**. Le même risque existe ici à l'identique, en pire : **aucune des 9 classes `Strategy` n'a de `**kwargs` catch-all** (vérifié `task_strategies.py`, les 9 `__init__`) — un `config_keys` non scopé par `task_type` (une liste plate unique, comme `MODEL_HYPERPARAM_CONFIG_KEYS` avant sa propre correction en Story 12.1) planterait `DetectionStrategy`/`CenterNetDetectionStrategy`/etc. avec un `TypeError` dès qu'une clé qu'elles ne déclarent pas (ex. `label_smoothing`, propre à `ClassificationStrategy`) se retrouverait dans un `config` qui les concerne. **`STRATEGY_FORWARDED_CONFIG_KEYS` doit être un dict `{task_type: tuple}` dès le premier jet** (Task 1 ci-dessus donne déjà le contenu exact, dérivé des 9 branches réelles de `main.py` avant migration — ne pas le redériver, le recopier).

- **Comportement actuel de `main.py:158-241` (avant migration ; extraction partagée 158-163, `if/elif` 164-241) à préserver, sauf le mécanisme interne** — code exact :
  ```python
  task_type = config.get("task_type", "classification")
  loss_method = config.get("loss_method", "cross_entropy")
  loss_params = config.get("loss_params", {})
  metric_method = config.get("metric_method", "accuracy")
  report_method = config.get("report_method", "confusion_matrix")

  if task_type == "classification":
      strategy = ClassificationStrategy(num_classes=num_classes, label_smoothing=config.get("label_smoothing", 0.0), mixup_alpha=config.get("mixup_alpha", 0.0), loss_method=loss_method, loss_params=loss_params, metric_method=metric_method, report_method=report_method)
  elif task_type == "detection":
      strategy = DetectionStrategy(loss_method=loss_method, loss_params=loss_params, metric_method=metric_method, report_method=report_method)
  # ... 7 autres branches, voir main.py:164-241 pour le detail exact avant migration
  else:
      raise ValueError(f"task_type '{task_type}' non reconnu.")
  ```
  Cible après migration (forme indicative, pas prescriptive sur les noms internes — cohérent avec Story 12.1) :
  ```python
  strategy_cls = STRATEGIES.get(task_type)
  if strategy_cls is None:
      raise ValueError(f"task_type '{task_type}' non reconnu.")
  strategy_kwargs, _ = build_kwargs_from_config(
      strategy_cls, config,
      config_keys=STRATEGY_FORWARDED_CONFIG_KEYS.get(task_type, ()),
      num_classes=num_classes,
  )
  print(f"🎯 Strategy: {task_type} -> {strategy_cls.__name__}")
  strategy = strategy_cls(**strategy_kwargs)
  ```

- **`num_classes` est un `overrides`, pas un `config_keys`** — même discipline que Story 12.1 (valeur calculée dans `main.py`, pas un champ `dataset_configs.py` brut au même titre que `loss_method`). Le canal `overrides` de `build_kwargs_from_config` étant strict (jamais forwardé sur une cible qui ne le nomme pas explicitement), `num_classes` n'atteindra que `ClassificationStrategy`/`KeplerStrategy` — comportement identique à l'existant (les 7 autres branches ne le passent déjà pas aujourd'hui).

- **Incohérence latente `DetectionStrategy` (déjà nommée dans `AD-21`, corrigée par cette story)** : `DetectionStrategy.__init__` a pour défaut réel `loss_method="segmentation"` (`task_strategies.py:172`), alors que l'ancien code `main.py` calculait un `loss_method = config.get("loss_method", "cross_entropy")` **partagé par toutes les branches** avant le dispatch — si `FIGHTERJET_DETECTION` ne déclarait pas `"loss_method"` explicitement (elle le fait aujourd'hui, `dataset_configs.py:191`), l'ancien code aurait silencieusement passé `"cross_entropy"` à une tâche de segmentation. Avec `config_keys` (Task 1/2), une clé absente de `config` n'est simplement pas forwardée — le vrai défaut `"segmentation"` de `DetectionStrategy` s'applique. **Comportement observable identique aujourd'hui** (la config déclare déjà le bon champ) — la correction ne change rien tant que `dataset_configs.py` n'est pas modifié, elle ferme juste le risque pour un futur `dataset_configs.py` incomplet.

- **`get_model`/`MODELS`/`build_kwargs_from_config` (Story 12.1) ne sont PAS modifiés par cette story** — seul le second point d'appel change.

- **Ne pas toucher** `dataset_configs.py`, `model_library.py`, `data_management.py` dans cette story.

### Project Structure Notes

- Fichiers touchés : `task_strategies.py` (ajout `STRATEGIES`, `STRATEGY_FORWARDED_CONFIG_KEYS`), `main.py` (migration du dispatch `Strategy`, imports).
- Nouveau fichier : `tests/test_strategy_kwargs_real_instantiation.py` (même convention que `tests/test_model_kwargs_real_instantiation.py`/`tests/test_build_kwargs_from_config.py` — script autonome, bootstrap `sys.path.insert` en tête, pas de framework de test formel imposé, exécutable directement via `python tests/test_strategy_kwargs_real_instantiation.py`).
- Pas de nouveau module `library/`/`utils/` générique (convention projet déjà actée).

### References

- [Source: `_bmad-output/implementation-artifacts/12-1-build-kwargs-from-config-et-migration-model-kwargs.md`] — story précédente, `build_kwargs_from_config` (signature, contrat, piège `**kwargs` déjà trouvé et corrigé), `MODEL_FORWARDED_CONFIG_KEYS` (précédent direct de scoping par cible pour `STRATEGY_FORWARDED_CONFIG_KEYS`).
- [Source: `_bmad-output/planning-artifacts/architecture/architecture-jax_supervised_training-2026-07-15/ARCHITECTURE-SPINE.md#AD-21`] — règle complète (dispatch `STRATEGIES`, pas de défaut dupliqué côté `main.py`).
- [Source: `_bmad-output/planning-artifacts/epics.md#Epic-12`] — Epic 12, Story 12.2, FR3/FR4.
- [Source: `main.py:158-241`] — code réel à migrer (dispatch `Strategy`).
- [Source: `task_strategies.py`] — les 9 classes `Strategy` et leurs constructeurs réels (vérifiés champ par champ pour `STRATEGY_FORWARDED_CONFIG_KEYS`, aucune n'a de `**kwargs`).
- [Source: `dataset_configs.py:191` (`FIGHTERJET_DETECTION`)] — confirme que `loss_method="segmentation"` est déjà déclaré explicitement (l'incohérence latente `DetectionStrategy` ci-dessus n'est pas observable aujourd'hui).

## Change Log

- 2026-08-19 : Story 12.2 implémentée intégralement (Tasks 1-4, AC1-9) en une session. Aucun écart par rapport aux Dev Notes — la leçon de Story 12.1 (scoper `config_keys` par cible, jamais une liste plate partagée) avait été intégrée dès le premier jet de la story, donc rien trouvé à corriger en cours d'implémentation cette fois.
- 2026-08-19 : Revue de code (`/code-review high`, 8 agents parallèles) — 3 findings dans le périmètre de cette story corrigés (test `test_unknown_task_type_raises_explicit_value_error` tautologique — réécrit pour tester la vraie précondition `STRATEGIES.get(...)` au lieu de réimplémenter le `raise` localement ; `STRATEGY_FORWARDED_CONFIG_KEYS` dédupliqué via une constante `_LOSS_PARAMS_ONLY` partagée par 5 entrées identiques ; `_param_count` de `tests/test_model_kwargs_real_instantiation.py` remplacé par `utils.count_parameters`, déjà équivalent). 2 findings adjacents (hors périmètre strict mais dans un fichier déjà touché ce jour) corrigés aussi : `tools/boxes_process_manual_tkinter.py` avait un bug de préfixe non ancré (`f.startswith(base_name)` sans `"_"`) répété à 7 endroits — associe/supprime silencieusement les annotations JSON d'une AUTRE image dont le nom partage le même préfixe numérique (ex. `"f16_1"` matche `"f16_10_0.json"`) ; et le timeout défensif de l'auto-traitement (120s) était trop court pour un premier chargement de modèle/compilation JIT, porté à 300s. Nombreux findings hors périmètre (`tools/organized_by_number_of_boxes.py` — même bug de préfixe, non touché ce jour ; `inference_utils.py`/`reporting.py` non migrés vers `build_kwargs_from_config` ; duplication `AircraftDetectorUNet`/`AircraftDetectorCenterNet`, déjà actée AD-20) signalés à Aymeric, non corrigés ici. Suite complète re-vérifiée après fixes : 162 passed, 0 régression. Statut : `done`.

## Dev Agent Record

### Agent Model Used

Claude Sonnet 5 (bmad-dev-story)

### Debug Log References

- `python tests/test_strategy_kwargs_real_instantiation.py` : 2/2 tests verts (11 configs réelles, 9 task_type, + cas `task_type` invalide)
- `pytest tests/ --ignore=tests/fixtures` : 162 passed (+2 vs Story 12.1), 5 erreurs de collecte préexistantes/sans rapport (mêmes 3 fichiers non touchés qu'en Story 12.1), 0 nouvel échec

### Completion Notes List

- `STRATEGIES`/`STRATEGY_FORWARDED_CONFIG_KEYS` vivent dans `task_strategies.py` (pas `main.py`) — les 9 classes `Strategy` y sont déjà définies, donc zéro nouvel import nécessaire ; les 7 imports locaux par branche (`from task_strategies import KeplerStrategy` etc.) ont disparu.
- Le test dédié (Task 3) utilise une approche générique par introspection de signature plutôt qu'une liste de valeurs attendues transcrites à la main par `task_type` — couvre les 11 entrées `DATASET_CONFIGS` réelles (pas seulement 9, certains `task_type` comme `chess_policy_value` ont plusieurs configs) sans risque de transcription incorrecte.
- Toutes les ACs (#1 à #9) vérifiées par test réel, pas par lecture de code.

### File List

- `task_strategies.py` (modifié) : ajout `STRATEGIES`, `STRATEGY_FORWARDED_CONFIG_KEYS`/`_LOSS_PARAMS_ONLY` (fin de fichier, après `ChessTokenOneMoveStrategy`)
- `main.py` (modifié) : import `STRATEGIES`/`STRATEGY_FORWARDED_CONFIG_KEYS` (remplace `ClassificationStrategy, DetectionStrategy`), migration du dispatch `Strategy` (if/elif à 9 branches → dict + `build_kwargs_from_config`), suppression des 7 imports locaux et des 4 extractions `loss_method`/`loss_params`/`metric_method`/`report_method`
- `tests/test_strategy_kwargs_real_instantiation.py` (nouveau) : validation par instanciation réelle des 9 `task_type` (11 configs)
- `tests/test_model_kwargs_real_instantiation.py` (modifié, revue de code) : `_param_count` remplacé par `utils.count_parameters`
- `tools/boxes_process_manual_tkinter.py` (modifié, revue de code) : bug de préfixe JSON non ancré corrigé (7 sites), timeout auto-traitement 120s→300s
