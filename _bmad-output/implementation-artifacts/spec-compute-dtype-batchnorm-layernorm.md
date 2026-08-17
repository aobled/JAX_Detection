---
title: 'compute_dtype : étendre à BatchNorm/LayerNorm (amendement AD-4)'
type: 'feature'
created: '2026-08-17'
status: 'done'
review_loop_iteration: 0
context: ['{project-root}/_bmad-output/specs/spec-compute-dtype-hardware/SPEC.md', '{project-root}/_bmad-output/planning-artifacts/architecture/architecture-compute-dtype-hardware-2026-08-17/ARCHITECTURE-SPINE.md']
baseline_commit: '1b536238d32dc6db66af6cf7433602e97228fe9f'
---

<frozen-after-approval reason="human-owned intent — do not modify unless human renegotiates">

## Intent

**Problem:** `nn.BatchNorm`/`nn.LayerNorm` (jamais appelés avec `dtype=`, ancienne AD-4) ressortent toujours en `float32`, réinitialisant la chaîne `bfloat16` à chaque bloc au lieu de tenir en continu — gain de vitesse limité sur `SophisticatedCNN32Plus`/`128Lite`/`128Plus`, déjà validés/livrés.

**Approach:** Amendement AD-4 déjà acté (vérifié sûr : `force_float32_reductions` protège la réduction interne indépendamment de `dtype=`) — appliquer `dtype=self.compute_dtype` à tous les appels `nn.BatchNorm`/`nn.LayerNorm` dans ces 3 classes déjà adaptées. Mécanique, aucune nouvelle décision.

## Boundaries & Constraints

**Always:**
- `dtype=self.compute_dtype` ajouté à CHAQUE appel `nn.BatchNorm`/`nn.LayerNorm` dans `SophisticatedCNN32Plus`, `SophisticatedCNN128Lite`, `SophisticatedCNN128Plus` — aucun site oublié.
- `nn.Embed` reste exclu (inchangé, aucun modèle concerné ici n'en a).
- `params`/`batch_stats` de `BatchNorm`/`LayerNorm` restent `float32` dans tous les cas (AD-6 inchangé — champ statique, réduction protégée).
- Tests mis à jour : les intermediates `BatchNorm_*`/`LayerNorm_*` doivent maintenant matcher `compute_dtype` (l'ancienne exception documentée dans les commentaires de test est inversée pour ces 3 modèles).
- `pytest` complet, zéro régression (baseline : 142 passed + 5 erreurs pré-existantes sans rapport).

**Ask First:** Aucun run réel — seul `pytest`. Aymeric relance FIGHTERJET_CLASSIFICATION sur Colab TPU lui-même pour comparer au baseline 0.9459.

**Never:**
- Toucher `Kepler1DConvNet` (pas de BatchNorm/LayerNorm), `SeparableConv`/`SEBlock`/`SpatialAttention` (vérifié : aucun BatchNorm/LayerNorm interne), `main.py`/`dataset_configs.py` (mécanisme d'injection inchangé), les modèles échecs/`aircraft_detector_*`.

## I/O & Edge-Case Matrix

| Scenario | Input / State | Expected Output / Behavior | Error Handling |
|----------|--------------|---------------------------|----------------|
| BatchNorm sous bfloat16 | `compute_dtype=jnp.bfloat16` | sortie `BatchNorm_N` = `bfloat16` (via `capture_intermediates`) | N/A |
| LayerNorm sous bfloat16 | idem | sortie `LayerNorm_0` = `bfloat16` | N/A |
| Stats/poids BatchNorm | idem | `batch_stats`/`params` restent `float32` | N/A |
| Sortie finale du modèle | idem | reste `float32` (recast déjà en place, inchangé) | N/A |

</frozen-after-approval>

## Code Map

- `model_library.py:100` (`SophisticatedCNN128Plus`) -- `dtype=self.compute_dtype` sur ~13 `nn.BatchNorm` + 1 `nn.LayerNorm`
- `model_library.py:240` (`SophisticatedCNN128Lite`) -- même traitement
- `model_library.py:376` (`SophisticatedCNN32Plus`) -- même traitement
- `tests/test_compute_dtype_hardware.py` -- étendre la vérification exhaustive aux intermediates `BatchNorm_*`/`LayerNorm_*`, inverser les commentaires qui documentaient l'ancienne exception

## Tasks & Acceptance

**Execution:**
- [x] `SophisticatedCNN128Plus` -- `dtype=self.compute_dtype` sur tous les BatchNorm/LayerNorm -- AD-4 amendé
- [x] `SophisticatedCNN128Lite` -- idem -- AD-4 amendé
- [x] `SophisticatedCNN32Plus` -- idem -- AD-4 amendé
- [x] `tests/test_compute_dtype_hardware.py` -- étendre les 3 tests "really_observed" pour vérifier BatchNorm/LayerNorm, documenter le comportement réel de SEBlock/SpatialAttention (mesuré, pas supposé) -- AD-3
- [x] `pytest tests/` -- zéro régression

**Acceptance Criteria:**
- Given les 3 modèles construits avec `compute_dtype=jnp.bfloat16`, when on inspecte `BatchNorm_N`/`LayerNorm_0` via `capture_intermediates`, then leur dtype est `bfloat16`
- Given les mêmes modèles, when on inspecte `batch_stats`/`params`, then ils restent `float32`
- Given la suite `pytest` complète, when exécutée après implémentation, then tous les tests passent (existants + renforcés)

## Verification

**Commands:**
- `pytest tests/` -- expected: tous les tests passent, y compris les tests `compute_dtype` renforcés

## Suggested Review Order

**Amendement AD-4 : BatchNorm/LayerNorm suivent désormais compute_dtype**

- `SophisticatedCNN128Plus` — 13 `nn.BatchNorm` + 1 `nn.LayerNorm` reçoivent `dtype=self.compute_dtype`.
  [`model_library.py:100`](../../model_library.py#L100)

- `SophisticatedCNN128Lite` — même traitement.
  [`model_library.py:241`](../../model_library.py#L241)

- `SophisticatedCNN32Plus` — même traitement.
  [`model_library.py:378`](../../model_library.py#L378)

**Fixes post-revue (Blind Hunter + Edge Case Hunter)**

- `_assert_all_matmul_layers_match_dtype` : vérification renforcée pour garantir que les couches BatchNorm/LayerNorm/SEBlock/SpatialAttention sont réellement capturées (pas juste "le dict n'est pas vide").
  [`tests/test_compute_dtype_hardware.py:100`](../../tests/test_compute_dtype_hardware.py#L100)

- Les 3 tests "really_observed" vérifient maintenant aussi `training=True` (chemin de réduction BatchNorm EN DIRECT, jamais testé avant, celui qui compte pendant l'entraînement réel).
  [`tests/test_compute_dtype_hardware.py:141`](../../tests/test_compute_dtype_hardware.py#L141)

**Documentation resynchronisée**

- `main.py` : commentaire/print mis à jour (mentionnaient encore "Conv/Dense" seulement).
  [`main.py:44`](../../main.py#L44)

- `ARCHITECTURE-SPINE.md` : AD-4 amendé, diagramme et section Deferred resynchronisés (128Plus/Kepler sortis de la liste des modèles non adaptés).
  [`../planning-artifacts/architecture/architecture-compute-dtype-hardware-2026-08-17/ARCHITECTURE-SPINE.md:55`](../planning-artifacts/architecture/architecture-compute-dtype-hardware-2026-08-17/ARCHITECTURE-SPINE.md#L55)
