---
title: 'compute_dtype dérivé du matériel, injecté par introspection (CIFAR10 + FIGHTERJET_CLASSIFICATION)'
type: 'feature'
created: '2026-08-17'
status: 'done'
review_loop_iteration: 0
context: ['{project-root}/_bmad-output/specs/spec-compute-dtype-hardware/SPEC.md', '{project-root}/_bmad-output/planning-artifacts/architecture/architecture-compute-dtype-hardware-2026-08-17/ARCHITECTURE-SPINE.md']
baseline_commit: '6e5127a9e1aded4e15b918f49d97e06ad8c4e04b'
---

<frozen-after-approval reason="human-owned intent — do not modify unless human renegotiates">

## Intent

**Problem:** `compute_dtype` (précision mixte TPU) n'existe que sur `ChessMoveTokenTransformer`, via une string codée en dur par config (`dataset_configs.py:966`) plutôt que dérivée du matériel réel. Jamais généralisé, contrairement à l'objectif du projet.

**Approach:** `main.py` dérive `compute_dtype` une fois depuis `jax.default_backend()` (bfloat16 si TPU, float32 sinon) et l'injecte uniquement aux modèles dont la factory déclare explicitement un paramètre nommé `compute_dtype` (introspection, pas de registre). CIFAR10 (`SophisticatedCNN32Plus`) et FIGHTERJET_CLASSIFICATION (`SophisticatedCNN128Lite`) + leurs sous-modules partagés adoptent ce champ. `ChessMoveTokenTransformer` reste inchangé.

## Boundaries & Constraints

**Always:**
- `compute_dtype` s'applique uniquement à `nn.Conv`/`nn.Dense` (matmul lourd) — jamais `nn.BatchNorm`/`nn.LayerNorm`/`nn.Embed` (ratifie le précédent `ChessMoveTokenTransformer`, docstring `model_library.py:1107-1123`).
- Sous-modules partagés (`SeparableConv`/`SEBlock`/`SpatialAttention`) reçoivent `compute_dtype` explicitement à CHAQUE site d'appel, jamais hérité implicitement.
- `compute_dtype` est un champ `dataclass` statique (jamais `nn.Param`) — checkpoints CIFAR10/FIGHTERJET existants restent chargeables sans changement de forme.
- `create_chess_move_token_transformer` (`model_library.py:1206-1224`) reste INCHANGÉE — sa validation string interne (lignes 1218-1221) et ses 2 tests existants (`test_compute_dtype_factory_rejects_unknown_string`, `test_compute_dtype_bfloat16_params_and_output_stay_float32`) doivent rester verts sans modification. **RENÉGOCIÉ (Aymeric, 2026-08-17, voir Spec Change Log)** : l'introspection AD-3 cible aussi cette factory (paramètre nommé `compute_dtype`), qui recevrait désormais un `jnp.dtype` déjà résolu au lieu d'une string — `getattr(jnp, <dtype>, None)` lève `TypeError` sur un objet, pas `None`. Fix minimal appliqué : la factory accepte maintenant `str` (chemin existant, inchangé) OU `jnp.dtype` déjà résolu (nouveau chemin, main.py). Les 2 tests existants restent verts sans modification.
- Chaque modèle adopté a un test dédié prouvant que le dtype est RÉELLEMENT observé (params/sortie diffèrent bfloat16 vs float32) — pas seulement l'absence d'erreur à l'instanciation.
- L'injection dans `main.py` (remplace le bloc `if "compute_dtype" in config` existant, ligne ~154) vérifie la présence d'un paramètre **nommé** `compute_dtype` dans la signature de la factory cible (`inspect.signature`) avant d'ajouter la clé à `model_kwargs` — jamais juste `**kwargs`.
- Suite de tests complète (`pytest`) exécutée, zéro régression, en particulier sur `aircraft_detector_*`/les autres modèles échecs/CIFAR10/FIGHTERJET.

**Ask First:** Lancer un vrai entraînement (même court) via `python main.py CIFAR10` ou `FIGHTERJET_CLASSIFICATION` — seul `pytest` est autorisé sans validation humaine. Aymeric fera lui-même le test Colab TPU sur FIGHTERJET_CLASSIFICATION séparément.

**Never:**
- Toucher aux 9 autres modèles de `MODELS` (`aircraft_detector_*`, `chess_cnn_attention_*`, `chess_token_candidate_model`, `chess_token_one_move_model`, `kepler_1d_cnn`, `sophisticated_cnn_128_plus`).
- Toucher au cast de dtype de l'ENTRÉE (`main.py:31-45`, `dtype = jnp.float16` pour TPU/GPU) — mécanisme séparé.
- Activer bfloat16/float16 sur GPU — reste `float32` par défaut hors TPU.
- Ajouter un flag `compute_dtype` dans une config de dataset — seule exemption sanctionnée : ne pas déclarer le paramètre nommé.

## I/O & Edge-Case Matrix

| Scenario | Input / State | Expected Output / Behavior | Error Handling |
|----------|--------------|---------------------------|----------------|
| Injection TPU, modèle compatible | `backend=="tpu"`, factory déclare `compute_dtype` | `model_kwargs["compute_dtype"]=jnp.bfloat16` | N/A |
| Injection GPU/CPU | `backend!="tpu"` | `model_kwargs["compute_dtype"]=jnp.float32` (tout modèle compatible) | N/A |
| Modèle non adapté | factory sans paramètre nommé `compute_dtype` (9 modèles restants) | rien injecté, comportement inchangé | aucune erreur |
| Appel direct chess existant | `create_chess_move_token_transformer(compute_dtype="bad")` | `ValueError` explicite (inchangé) | test existant doit rester vert |

</frozen-after-approval>

## Code Map

- `main.py` (~28-45 détection backend, ~117-165 construction model_kwargs, ~1766 `get_model`) -- dérivation + résolution centralisée + injection par introspection
- `model_library.py` (18-83 sous-modules partagés, 212-333 `SophisticatedCNN128Lite`, 335-446 `SophisticatedCNN32Plus`, 1206-1224 `create_chess_move_token_transformer` NON touchée) -- ajout du champ `compute_dtype`
- `dataset_configs.py` (953-966) -- retrait du littéral `"compute_dtype": "bfloat16"` de `CHESS_MOVE_TOKEN`
- `tests/` -- nouveaux tests dtype réellement observé (CIFAR10/FIGHTERJET) + test de l'injection par introspection

## Tasks & Acceptance

**Execution:**
- [x] `main.py` -- résoudre `compute_dtype` une fois depuis `backend` (bfloat16 si tpu, float32 sinon), remplacer le bloc conditionnel par config par une injection par introspection de signature -- AD-1/AD-2/AD-3
- [x] `model_library.py::SeparableConv/SEBlock/SpatialAttention` -- ajouter `compute_dtype: Any = jnp.float32`, appliquer à leurs `nn.Conv`/`nn.Dense` -- AD-4/AD-5
- [x] `model_library.py::SophisticatedCNN32Plus` + `create_sophisticated_cnn_32_plus` -- champ + application Conv/Dense + threading explicite vers les sous-modules + factory acceptant `compute_dtype` -- AD-4/AD-5/AD-6
- [x] `model_library.py::SophisticatedCNN128Lite` + `create_sophisticated_cnn_128_lite` -- même traitement -- AD-4/AD-5/AD-6
- [x] `dataset_configs.py::CHESS_MOVE_TOKEN` -- retirer le littéral `compute_dtype` et son commentaire -- AD-1/AD-7
- [x] `tests/` -- test dtype réellement observé (CIFAR10 + FIGHTERJET, bfloat16 vs float32) + test injection introspection (skip/inject correct) -- AD-3
- [x] `pytest` (suite complète) -- zéro régression -- AD-7

**Acceptance Criteria:**
- Given un modèle CIFAR10 construit avec `compute_dtype=jnp.bfloat16`, when on inspecte l'activation en sortie d'une couche `Conv`/`Dense`, then son dtype est `bfloat16` (pas juste absence d'erreur)
- Given `backend != "tpu"`, when `main.py` construit `model_kwargs` pour CIFAR10/FIGHTERJET, then `compute_dtype` résolu vaut `float32`
- Given les 9 modèles non adaptés, when `main.py` les instancie, then aucun `compute_dtype` n'est injecté et aucune régression n'apparaît
- Given `create_chess_move_token_transformer(compute_dtype="bad")`, when appelée directement, then `ValueError` explicite (test existant inchangé et vert)
- Given la suite `pytest` complète, when exécutée après implémentation, then tous les tests passent (existants + nouveaux)

## Spec Change Log

- **2026-08-17, renégociation (pas un review loopback step-04)** : l'implémentation a révélé que l'introspection pure-par-nom (AD-3) cible aussi `create_chess_move_token_transformer` (paramètre nommé `compute_dtype` pré-existant), qui recevrait un `jnp.dtype` déjà résolu au lieu de la string attendue par sa validation interne — `getattr(jnp, jnp.bfloat16, None)` lève `TypeError: attribute name must be string` (vérifié empiriquement), cassant tout run réel de `CHESS_MOVE_TOKEN` (modèle clos négativement, Epic 11, risque pratique faible mais bug réel). Aymeric a choisi le fix minimal plutôt que documenter-et-laisser : `create_chess_move_token_transformer` accepte désormais `str` (chemin historique, comportement inchangé, test existant vert) OU `jnp.dtype` déjà résolu (nouveau chemin main.py). Aucune régression : `pytest tests/` toujours 137 passed / 5 erreurs pré-existantes sans rapport après le fix. **KEEP** : tout le reste de l'implémentation (main.py/model_library.py/dataset_configs.py/tests/test_compute_dtype_hardware.py) validé sans changement.

## Design Notes

`inspect.signature(factory_fn).parameters` doit vérifier la présence de la clé littérale `"compute_dtype"` — pas la simple présence d'un `**kwargs` catch-all (les factories `aircraft_detector_*` ont déjà ce piège : `**kwargs` accepté en façade, jamais transmis au constructeur).

## Verification

**Commands:**
- `pytest tests/` -- expected: tous les tests passent, y compris les 2 tests `ChessMoveTokenTransformer` existants et les nouveaux tests dtype-observé

## Suggested Review Order

**Dérivation matérielle + injection par introspection (le cœur du mécanisme, AD-1/AD-2/AD-3)**

- Point d'entrée : dérivation unique du dtype depuis le backend déjà détecté, extraite en fonction pure testable.
  [`main.py:53`](../../main.py#L53)
  [`model_library.py:1811`](../../model_library.py#L1811)

- Injection par introspection de signature — vérifie un paramètre NOMMÉ, jamais `**kwargs` ; `.get()` (pas `[...]`) pour préserver le `ValueError` explicite de `get_model` sur un nom invalide (fix post-revue).
  [`main.py:178`](../../main.py#L178)

**Adoption CIFAR10/FIGHTERJET (AD-4/AD-5/AD-6)**

- Sous-modules partagés : nouveau champ `compute_dtype`, threadé à chaque site d'appel.
  [`model_library.py:29`](../../model_library.py#L29)
  [`model_library.py:61`](../../model_library.py#L61)
  [`model_library.py:83`](../../model_library.py#L83)

- `SophisticatedCNN32Plus`/`SophisticatedCNN128Lite` : champ + recast explicite en `float32` avant retour (fix post-revue, protège la loss cross-entropy — même garde que le précédent `ChessMoveTokenTransformer`).
  [`model_library.py:374`](../../model_library.py#L374)
  [`model_library.py:471`](../../model_library.py#L471)
  [`model_library.py:251`](../../model_library.py#L251)
  [`model_library.py:353`](../../model_library.py#L353)

- Factories mises à jour pour accepter/transmettre `compute_dtype`.
  [`model_library.py:474`](../../model_library.py#L474)
  [`model_library.py:356`](../../model_library.py#L356)

**Précédent chess préservé + fix post-revue (AD-2/AD-7)**

- `create_chess_move_token_transformer` accepte désormais `str` (chemin historique inchangé) OU `jnp.dtype` déjà résolu (nouveau chemin `main.py`), avec validation explicite dans les deux cas — corrige une régression trouvée par la revue (introspection ciblait aussi cette factory).
  [`model_library.py:1237`](../../model_library.py#L1237)
  [`model_library.py:1254`](../../model_library.py#L1254)

**Nettoyage config (AD-1/AD-7)**

- Retrait du littéral `compute_dtype` codé en dur de `CHESS_MOVE_TOKEN`.
  [`dataset_configs.py:953`](../../dataset_configs.py#L953)

**Peripherals**

- Tests dtype réellement observé (via `capture_intermediates`, la sortie étant désormais toujours `float32`), sous-modules en isolation, factories, dérivation matérielle, non-régression chess, contrat d'introspection.
  [`tests/test_compute_dtype_hardware.py`](../../tests/test_compute_dtype_hardware.py)
