---
title: 'compute_dtype : AircraftDetectorCenterNet + AircraftDetectorUNet (dernier groupe du rollout)'
type: 'feature'
created: '2026-08-17'
status: 'done'
review_loop_iteration: 0
context: ['{project-root}/_bmad-output/specs/spec-compute-dtype-hardware/SPEC.md', '{project-root}/_bmad-output/planning-artifacts/architecture/architecture-compute-dtype-hardware-2026-08-17/ARCHITECTURE-SPINE.md']
baseline_commit: '7c41e75a84843beb182029261466767d2e0598e7'
---

<frozen-after-approval reason="human-owned intent — do not modify unless human renegotiates">

## Intent

**Problem:** `AircraftDetectorCenterNet` (JAX_DETECTOR, actif) et `AircraftDetectorUNet` (FIGHTERJET_DETECTION, actif) sont les 2 seuls modèles non-échecs restant sans `compute_dtype`, après suppression de `centernet_lite` (mort).

**Approach:** Patron AD-1..AD-7 déjà établi, appliqué mécaniquement — `nn.Conv`/`nn.BatchNorm` (pas de `LayerNorm` ici) reçoivent `dtype=self.compute_dtype`. Nouveau cas non couvert par le précédent classification : sorties passant par `nn.sigmoid` (masque UNet, tête heatmap CenterNet) — recast `float32` AVANT la non-linéarité, pas juste avant le retour.

## Boundaries & Constraints

**Always:**
- `dtype=self.compute_dtype` sur CHAQUE `nn.Conv`/`nn.BatchNorm` des deux classes.
- `AircraftDetectorUNet` : `nn.Conv` final recasté `float32` AVANT `nn.sigmoid` (pas de logits réduits dans la non-linéarité).
- `AircraftDetectorCenterNet` : les DEUX têtes (`heatmap` recasté avant `sigmoid`, `size` recasté directement, pas d'activation) restent `float32` en sortie du `dict`.
- Factories mises à jour : `compute_dtype=jnp.float32` en paramètre nommé explicite (bug `**kwargs` existant contourné, PAS corrigé), `_validate_compute_dtype` appelé (helper partagé existant).
- Tests dédiés (patron `tests/test_compute_dtype_hardware.py` existant) : couches internes vérifiées via `capture_intermediates`, `training=True` ET `False`, sortie(s) toujours `float32`.
- `pytest` complet, zéro régression (baseline 142 passed + 5 erreurs pré-existantes sans rapport).

**Ask First:** Aucun run réel — seul `pytest`. Pas de promesse de validation Colab immédiate cette fois (contrairement aux cycles CIFAR10/FIGHTERJET_CLASSIFICATION).

**Never:**
- Corriger le bug `**kwargs` non transmis au-delà de l'ajout du paramètre `compute_dtype` explicite.
- Toucher `main.py`/`dataset_configs.py` (injection déjà générique), les modèles échecs, `sophisticated_cnn_*`/`kepler_1d_cnn` (déjà faits), `centernet_lite` (supprimé).

## I/O & Edge-Case Matrix

| Scenario | Input / State | Expected Output / Behavior | Error Handling |
|----------|--------------|---------------------------|----------------|
| Couche interne sous bfloat16 | `compute_dtype=jnp.bfloat16` | `Conv_N`/`BatchNorm_N` = `bfloat16` (`capture_intermediates`) | N/A |
| Masque UNet | idem | sortie (post-sigmoid) = `float32` | N/A |
| Têtes CenterNet | idem | `HEATMAP_KEY` (post-sigmoid) et `SIZE_KEY` = `float32` | N/A |
| Mode entraînement | `training=True` | même garanties (réduction BatchNorm en direct) | N/A |

</frozen-after-approval>

## Code Map

- `model_library.py:504` (`AircraftDetectorUNet` + `create_aircraft_detector_unet`) -- champ + Conv/BatchNorm + recast avant sigmoid + factory
- `model_library.py:599` (`AircraftDetectorCenterNet` + `create_aircraft_detector_centernet`) -- idem, 2 têtes
- `tests/test_compute_dtype_hardware.py` -- nouveaux tests, shape d'entrée à déterminer empiriquement (224×224 réel, config JAX_DETECTOR/FIGHTERJET_DETECTION — une shape réduite divisible par 8 acceptable pour les tests)

## Tasks & Acceptance

**Execution:**
- [x] `AircraftDetectorUNet` -- champ `compute_dtype`, Conv/BatchNorm, recast float32 avant sigmoid -- AD-4/AD-6
- [x] `create_aircraft_detector_unet` -- paramètre nommé + validation -- AD-3
- [x] `AircraftDetectorCenterNet` -- idem, 2 têtes -- AD-4/AD-6
- [x] `create_aircraft_detector_centernet` -- idem -- AD-3
- [x] `tests/test_compute_dtype_hardware.py` -- tests dtype réellement observé pour les 2 modèles -- AD-3
- [x] `pytest tests/` -- zéro régression

**Acceptance Criteria:**
- Given les 2 modèles avec `compute_dtype=jnp.bfloat16`, when on inspecte les couches internes, then elles sont `bfloat16`
- Given les mêmes modèles, when on inspecte la/les sortie(s) (masque UNet, dict CenterNet), then elles restent `float32`
- Given `main.py` inchangé, when il construit ces 2 modèles, then l'injection les cible automatiquement
- Given `pytest` complet, when exécuté après implémentation, then tous les tests passent

## Verification

**Commands:**
- `pytest tests/` -- expected: tous les tests passent

## Suggested Review Order

**Adoption AircraftDetectorUNet + AircraftDetectorCenterNet**

- `AircraftDetectorUNet` — Conv/BatchNorm reçoivent `dtype=self.compute_dtype`, masque recasté `float32` avant `nn.sigmoid`.
  [`model_library.py:504`](../../model_library.py#L504)

- `AircraftDetectorCenterNet` — même patron, 2 têtes (`heatmap` recastée avant `sigmoid`, `size` recastée directement).
  [`model_library.py:621`](../../model_library.py#L621)

**Fixes post-revue (Blind Hunter + Edge Case Hunter)**

- Nouveau test de gradient réel (`jax.value_and_grad` à travers `compute_segmentation_loss`/`compute_centernet_loss`) — les tests précédents ne vérifiaient que le forward pass, jamais le chemin où une perte de précision bfloat16 se manifesterait le plus (termes log/power près de la saturation). Utilise le `heatmap_prior` RÉEL de `JAX_DETECTOR` (0.0000268, jamais exercé avant), pas le défaut de test.
  [`tests/test_compute_dtype_hardware.py:478`](../../tests/test_compute_dtype_hardware.py#L478)

- Print de fin de test : chiffre codé en dur remplacé par `len(adapted)`.
  [`tests/test_compute_dtype_hardware.py:675`](../../tests/test_compute_dtype_hardware.py#L675)

**Peripherals**

- Tests dtype réellement observé (`training=True`/`False`, sortie(s) toujours `float32`).
  [`tests/test_compute_dtype_hardware.py:382`](../../tests/test_compute_dtype_hardware.py#L382)
  [`tests/test_compute_dtype_hardware.py:431`](../../tests/test_compute_dtype_hardware.py#L431)
