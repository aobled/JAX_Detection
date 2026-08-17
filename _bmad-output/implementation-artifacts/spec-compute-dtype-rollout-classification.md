---
title: 'compute_dtype : rollout à sophisticated_cnn_128_plus + kepler_1d_cnn'
type: 'feature'
created: '2026-08-17'
status: 'done'
review_loop_iteration: 0
context: ['{project-root}/_bmad-output/specs/spec-compute-dtype-hardware/SPEC.md', '{project-root}/_bmad-output/planning-artifacts/architecture/architecture-compute-dtype-hardware-2026-08-17/ARCHITECTURE-SPINE.md']
baseline_commit: 'b3e7ca7c739596ea0663a00428259ee086378444'
---

<frozen-after-approval reason="human-owned intent — do not modify unless human renegotiates">

## Intent

**Problem:** `compute_dtype` (précision mixte, validée en réel sur TPU v5 pour CIFAR10/FIGHTERJET_CLASSIFICATION, 0.9459 val accuracy) ne couvre encore que 2 des 12 modèles de `MODELS`. Groupe 1 (le plus simple/sûr) du rollout approuvé par Aymeric.

**Approach:** Appliquer le patron déjà établi (AD-1..AD-7, `ARCHITECTURE-SPINE.md`) à `SophisticatedCNN128Plus` (même famille/sous-modules que 32_plus/128_lite déjà faits) et `Kepler1DConvNet` (plus simple, pas de sous-module partagé). Aucune nouvelle décision d'architecture — mécanique, pas de conception.

## Boundaries & Constraints

**Always:**
- `compute_dtype` s'applique uniquement à `nn.Conv`/`nn.Dense` — jamais `nn.BatchNorm`/`nn.LayerNorm`/`nn.Embed` (AD-4, déjà tranché).
- `SophisticatedCNN128Plus` thread `compute_dtype=self.compute_dtype` explicitement à chaque instanciation de `SeparableConv`/`SEBlock`/`SpatialAttention` (AD-5) — ces sous-modules sont DÉJÀ compute_dtype-aware, ne pas les re-modifier.
- Sortie finale (dernier `nn.Dense`) recastée en `float32` avant retour dans LES DEUX modèles — `KeplerStrategy` (`task_strategies.py:375`) utilise `cross_entropy` par défaut comme `ClassificationStrategy`, même risque de précision réduite dans la loss que déjà traité sur CIFAR10/FIGHTERJET.
- `create_sophisticated_cnn_128_plus` (signature fixe, pas de `**kwargs`) et `create_kepler_1d_cnn` (`**kwargs` déjà transmis correctement, pas de bug ici) mis à jour pour accepter/transmettre `compute_dtype` comme paramètre nommé explicite.
- `compute_dtype` reste un champ `dataclass` statique — checkpoints existants (`best_model_kepler.pkl`, variante `128_plus`) restent chargeables.
- Chaque modèle a un test dédié prouvant le dtype RÉELLEMENT observé (via `capture_intermediates` sur une couche interne, la sortie étant recastée en float32) — même patron que `tests/test_compute_dtype_hardware.py`.
- `pytest` complet exécuté, zéro régression.

**Ask First:** Aucun run réel (`python main.py ...`) — seul `pytest` autorisé. Validation matérielle réelle faite par Aymeric séparément (comme pour CIFAR10/FIGHTERJET).

**Never:**
- Toucher `aircraft_detector_*` (groupes 2/3, différés — `deferred-work.md`) ni aux modèles échecs (exclus de tout ce rollout, session future séparée).
- Toucher au point d'injection central `main.py` (déjà générique par introspection, AD-3 — fonctionne automatiquement dès que ces 2 factories déclarent le paramètre nommé, aucune modification requise).
- Toucher `SeparableConv`/`SEBlock`/`SpatialAttention` (déjà compute_dtype-aware) au-delà de les invoquer avec le paramètre.
- Ajouter un flag de config par dataset pour `compute_dtype` (AD-1).

## I/O & Edge-Case Matrix

| Scenario | Input / State | Expected Output / Behavior | Error Handling |
|----------|--------------|---------------------------|----------------|
| Injection automatique | `main.py` appelle `get_model` pour `sophisticated_cnn_128_plus`/`kepler_1d_cnn` | `compute_dtype` injecté automatiquement (introspection déjà générique, AD-3) sans modifier `main.py` | N/A |
| Couche interne sous bfloat16 | `compute_dtype=jnp.bfloat16` | 1er `nn.Conv` observé en `bfloat16` via `capture_intermediates` | N/A |
| Sortie finale | n'importe quel `compute_dtype` | toujours `float32` (recast explicite) | N/A |
| Checkpoint existant | `best_model_kepler.pkl` chargé après ce changement | poids compatibles, aucune erreur de forme | N/A |

</frozen-after-approval>

## Code Map

- `model_library.py:100-223` (`SophisticatedCNN128Plus` + `create_sophisticated_cnn_128_plus`) -- champ + Conv/Dense + threading sous-modules + recast float32 + factory
- `model_library.py:1749-1793` (`Kepler1DConvNet` + `create_kepler_1d_cnn`) -- champ + Conv/Dense (pas de sous-module) + recast float32 + factory
- `tests/test_compute_dtype_hardware.py` -- extension avec les 2 nouveaux modèles (même patron que CIFAR10/FIGHTERJET)

## Tasks & Acceptance

**Execution:**
- [x] `model_library.py::SophisticatedCNN128Plus` -- champ `compute_dtype`, `dtype=self.compute_dtype` sur Conv/Dense, threading vers SeparableConv/SEBlock/SpatialAttention, recast float32 sortie -- AD-4/AD-5/AD-6
- [x] `model_library.py::create_sophisticated_cnn_128_plus` -- accepte/transmet `compute_dtype` -- AD-3
- [x] `model_library.py::Kepler1DConvNet` -- champ `compute_dtype`, `dtype=self.compute_dtype` sur les 4 `nn.Conv` + 2 `nn.Dense`, recast float32 sortie -- AD-4/AD-6
- [x] `model_library.py::create_kepler_1d_cnn` -- accepte/transmet `compute_dtype` -- AD-3
- [x] `tests/test_compute_dtype_hardware.py` -- tests dtype réellement observé pour les 2 modèles -- AD-3
- [x] `pytest tests/` -- zéro régression -- non-régression

**Acceptance Criteria:**
- Given `SophisticatedCNN128Plus`/`Kepler1DConvNet` construits avec `compute_dtype=jnp.bfloat16`, when on inspecte une couche `Conv` interne (`capture_intermediates`), then son dtype est `bfloat16`
- Given les mêmes modèles, when on inspecte la sortie finale, then elle est toujours `float32` quel que soit `compute_dtype`
- Given `main.py` inchangé, when il construit ces 2 modèles, then l'injection par introspection les cible automatiquement (aucune modification de `main.py` nécessaire)
- Given la suite `pytest` complète, when exécutée après implémentation, then tous les tests passent (existants + nouveaux)

## Verification

**Commands:**
- `pytest tests/` -- expected: tous les tests passent, y compris les tests `compute_dtype` existants (CIFAR10/FIGHTERJET/chess) et les nouveaux

## Suggested Review Order

**Adoption SophisticatedCNN128Plus + Kepler1DConvNet (le cœur du diff, AD-4/AD-5/AD-6)**

- `SophisticatedCNN128Plus` : champ `compute_dtype`, Conv/Dense uniquement, threading vers les sous-modules déjà adaptés, recast float32 en sortie.
  [`model_library.py:100`](../../model_library.py#L100)

- `Kepler1DConvNet` : même patron, plus simple (pas de sous-module, pas de BatchNorm).
  [`model_library.py:1765`](../../model_library.py#L1765)

- Factories mises à jour pour accepter/transmettre `compute_dtype` comme paramètre nommé explicite.
  [`model_library.py:234`](../../model_library.py#L234)
  [`model_library.py:1820`](../../model_library.py#L1820)

**Fix post-revue : validation manquante (trouvé par Blind Hunter)**

- Nouveau helper partagé `_validate_compute_dtype` — les 4 factories non-échecs (32Plus/128Lite/128Plus/Kepler) acceptaient un `compute_dtype` invalide (ex. `jnp.int32`) sans erreur avant ce fix ; source unique de validation plutôt que dupliquée par factory.
  [`model_library.py:1868`](../../model_library.py#L1868)

**Trouvaille réelle post-revue : BatchNorm remet le réseau en float32 entre les blocs**

- Découverte en renforçant les tests (voir ci-dessous) : `nn.BatchNorm` ressort toujours en `float32`, donc `compute_dtype` ne "tient" pas entre les blocs — pas un bug (déjà validé sur TPU réel), mais une vraie limite du gain de vitesse actuel. Documentée dans `deferred-work.md`, pas corrigée ici (toucherait les modèles déjà validés).
  [`tests/test_compute_dtype_hardware.py:75`](../../tests/test_compute_dtype_hardware.py#L75)

**Tests renforcés (toutes les couches matmul, pas seulement la première)**

- Nouveaux tests pour les 2 modèles + vérification exhaustive (`_assert_all_matmul_layers_match_dtype`) rétro-appliquée aux 2 modèles déjà adoptés (32Plus/128Lite) — un site d'appel oublié plus loin dans le réseau serait maintenant détecté, pas seulement sur `Conv_0`.
  [`tests/test_compute_dtype_hardware.py:131`](../../tests/test_compute_dtype_hardware.py#L131)
  [`tests/test_compute_dtype_hardware.py:164`](../../tests/test_compute_dtype_hardware.py#L164)
