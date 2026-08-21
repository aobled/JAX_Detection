---
title: 'Priorité explicite decay_steps + correction accum_steps'
type: 'bugfix'
created: '2026-08-21'
status: 'done'
review_loop_iteration: 1
context: []
baseline_commit: '91090ecce7499f6e30b9dc6c7bf6ec7a48782610'
---

<frozen-after-approval reason="human-owned intent — do not modify unless human renegotiates">

## Intent

**Problem:** L'auto-calcul de `decay_steps` (`trainer.py::create_train_state`) ne divise jamais par `accum_steps`, produisant une valeur trop grande dès qu'un dataset combine `accum_steps>1` et des chunks détectables sur disque — confirmé en live sur FIGHTERJET_CLASSIFICATION (decay_steps=6419 aujourd'hui au lieu de ~1610, accum_steps=4). Il écrase aussi silencieusement toute valeur choisie à la main dès que des chunks existent, sans distinguer volonté explicite et repli automatique.

**Approach:** Inverser la priorité (config explicite toujours prioritaire ; auto-calcul seulement si absente), corriger la formule (diviser par `accum_steps`, respecter `drop_remainder` selon `task_type`), lever une erreur explicite si ni l'un ni l'autre n'est disponible. Migrer ensuite les 4 configs à convention de chunk standard (FIGHTERJET_CLASSIFICATION, FIGHTERJET_DETECTION, JAX_DETECTOR, CIFAR10) vers l'auto-calcul, en retirant leur champ `decay_steps`/`warmup_steps` littéral (gardé en commentaire pour l'historique). Confirmé avec Aymeric : la migration de FIGHTERJET_CLASSIFICATION change réellement son comportement d'entraînement (6419→~1610) sur un modèle déjà validé à 0.9521 val — acceptée en connaissance de cause.

## Boundaries & Constraints

**Always:**
- `decay_steps`/`warmup_steps` explicites dans `backend_config` → utilisés tels quels, jamais recalculés ni écrasés.
- Absents → auto-calcul : `steps_per_epoch = N_train // micro_batch_size` si `task_type` (défaut `"classification"`) n'est pas dans `{"classification","kepler"}` (ces deux-là utilisent `ceil(N_train/micro_batch_size)`, cf. `data_management.py:224` `drop_remainder=False`) ; puis `steps_per_epoch_optimiseur = ceil(steps_per_epoch / accum_steps)` (pas floor — `train_epoch` applique une mise à jour finale pour les micro-batches restants en fin d'epoch même si le groupe `accum_steps` n'est pas complet, corrigé 2026-08-21 suite revue adversariale) ; `decay_steps = steps_per_epoch_optimiseur * decay_epochs`.
- Ni valeur explicite ni chunks détectables → `ValueError` explicite citant le dataset/backend, jamais de repli silencieux sur `6000`.
- Aucune branche de code spécifique à un dataset ou une famille de dataset (chess incluse) — le mécanisme reste générique, conditionné uniquement par présence de fichiers glob + `task_type` déjà connu de `get_datasets()`.
- Les 6 configs échecs (CHESS_NO_HISTORY, CHESS_LEGAL_MOVES, CHESS_SEARCH_TEACHER, CHESS_MOVE_TOKEN, CHESS_TOKEN, CHESS_TOKEN_1_MOVE) restent inchangées — gardent leurs `decay_steps`/`warmup_steps` littéraux tels quels.

**Ask First:** (résolu 2026-08-21, revue adversariale) — JAX_KEPLER n'a aucun `decay_steps` en dur et aucun chunk sur disque (`./data/chunks/kepler/` inexistant, dataset déjà non-fonctionnel sur cette machine). Décision Aymeric : garder la `ValueError` (cohérent avec le principe fail-loud de cette story), corriger uniquement le commentaire adjacent désormais faux ("6000 s'applique toujours").

**Never:**
- Ne pas étendre `_count_real_train_samples` pour comprendre les conventions de chunk échecs (chunk-level split ou fichier unique).
- Ne pas toucher à la génération de chunks côté chess_ai (hors périmètre de ce repo).
- Ne pas fusionner `_count_real_train_samples` avec `ChunkManager.get_chunk_statistics` dans cette story — mentionner en Design Notes seulement si le diff s'y prête naturellement, sans l'imposer.

## I/O & Edge-Case Matrix

| Scenario | Input / State | Expected Output / Behavior | Error Handling |
|----------|--------------|---------------------------|----------------|
| Config explicite présente | `backend_config` a `decay_steps`+`warmup_steps` | valeurs utilisées telles quelles, `_count_real_train_samples` non consulté pour le decay | N/A |
| Auto, accum_steps>1, drop_remainder=True | pas de `decay_steps`, `task_type="detection_centernet"`, chunks présents, `accum_steps=2` | `decay_steps = (N//batch//2)*decay_epochs` | N/A |
| Auto, task_type classification/kepler | pas de `decay_steps`, chunks présents | `steps_per_epoch = ceil(N/batch)` | N/A |
| Ni config ni chunks | pas de `decay_steps`, `output_prefix` ne matche aucun fichier | — | `ValueError` explicite nommant dataset/backend |
| Migration FIGHTERJET_CLASSIFICATION | champ supprimé, `decay_epochs=7` conservé, 4 chunks réels sur disque | `decay_steps ≈ 1610` (changement de comportement accepté) | N/A |

</frozen-after-approval>

## Code Map

- `trainer.py` -- `_count_real_train_samples` (comptage, inchangé) ; `create_train_state` (~lignes 158-183) -- logique de résolution `decay_steps`/`warmup_steps` à réécrire, idéalement extraite en helper testable isolément
- `dataset_configs.py` -- retirer les champs `decay_steps`/`warmup_steps` littéraux pour FIGHTERJET_CLASSIFICATION (tpu:118-128, gpu:131-139), FIGHTERJET_DETECTION (tpu:200-214, gpu:217-234), JAX_DETECTOR (tpu:506-536, gpu:554-562), CIFAR10 (blocs tpu/gpu ~367-385) + commentaire historique ; configs échecs non touchées
- `tests/test_decay_steps_calculation.py` -- nouveau, couvre la matrice I/O ci-dessus

## Tasks & Acceptance

**Execution:**
- [x] `trainer.py` -- extraite `_resolve_lr_schedule_steps` implémentant priorité config-explicite/auto + formule corrigée (`// accum_steps`) + `drop_remainder` par `task_type` + `ValueError` si aucune source -- rend la logique testable isolément et lisible dans `create_train_state`
- [x] `dataset_configs.py` -- retiré les champs `decay_steps` de 3 des 4 configs listées (FIGHTERJET_CLASSIFICATION, JAX_DETECTOR, CIFAR10), commentaire avec l'ancienne valeur et la date ajouté. **FIGHTERJET_DETECTION exclue de la migration** (déviation découverte pendant l'implémentation, voir Design Notes) -- laisse l'auto-calcul corrigé prendre le relais partout où c'est sûr
- [x] `tests/test_decay_steps_calculation.py` -- nouveau fichier (convention du projet : script autonome, pas pytest), teste priorité explicite/auto, division `accum_steps` en ceil (reproduit le cas réel 117457/128/4/7→1610), `drop_remainder` selon `task_type`, `ValueError` si aucune source/accum_steps invalide/résultat dégénéré, audit systématique de `DATASET_CONFIGS`

**Acceptance Criteria:**
- Given une config avec `decay_steps` explicite, when `create_train_state` s'exécute, then la valeur configurée est utilisée sans être recalculée, même si des chunks existent sur disque.
- Given FIGHTERJET_CLASSIFICATION après migration (champ retiré), when `create_train_state` s'exécute avec les 4 chunks réels actuels, then `decay_steps ≈ 1610`.
- Given un dataset sans `decay_steps` en config et sans chunk détectable, when `create_train_state` s'exécute, then une `ValueError` est levée citant le dataset/backend concerné.
- Given les 6 configs échecs, when la story est terminée, then aucune n'a été modifiée et aucune branche dédiée à leur format de chunk n'existe dans `trainer.py`.

## Spec Change Log

**2026-08-21, review_loop_iteration 1 (intent_gap, root cause inside frozen block) :**
- **Finding 1 (Blind Hunter + Edge Case Hunter, indépendamment) :** JAX_KEPLER n'a ni `decay_steps` explicite ni chunks sur disque — la nouvelle `ValueError` s'y déclencherait, et le commentaire adjacent affirmant le repli implicite sur `6000` devient faux. **Amendé :** section Ask First résolue (Aymeric : garder la `ValueError`, corriger le commentaire). **Évite :** un commentaire de config trompeur affirmant un comportement qui n'existe plus.
- **Finding 2 (Edge Case Hunter) :** la formule `steps_per_epoch // accum_steps` (floor) ne correspond pas au comportement réel de `train_epoch`, qui flush les micro-batches restants en fin d'epoch même hors groupe `accum_steps` complet — le vrai compte est `ceil(steps_per_epoch / accum_steps)`. **Amendé :** formule dans Boundaries & Constraints > Always, passée de floor à ceil (Aymeric confirmé). **Évite :** une sous-estimation systématique de `decay_steps` d'un step/epoch dès que `steps_per_epoch % accum_steps != 0` — précisément la classe d'imprécision que cette story visait à éliminer.
- **KEEP :** toute la logique de priorité config-explicite/auto-calcul, le `ValueError` fail-loud, l'exclusion de FIGHTERJET_DETECTION (chunks absents) et des configs échecs (aucune branche dédiée) restent inchangées et correctes — seule la formule interne et le commentaire JAX_KEPLER sont amendés.

## Design Notes

Extraire la résolution en fonction pure (pas de dépendance à `self`) simplifie le test unitaire : elle prend `backend_config`, `task_type`, `real_train_samples`, `micro_batch_size`, `accum_steps`, `decay_epochs` en entrée et retourne `(warmup_steps, decay_steps)`, sans toucher à `_count_real_train_samples` ni à l'appel `optax.warmup_cosine_decay_schedule`.

Mapping `drop_remainder` : `task_type in {"classification", "kepler"}` (défaut si absent) → `False` (`ceil`) ; tout le reste → `True` (`//`), reflet exact de `data_management.py:224` vs. les autres classes `Dataset`.

**Déviation découverte pendant l'implémentation :** l'approche approuvée listait FIGHTERJET_DETECTION parmi les 4 configs à migrer, sur la base d'une vérification incomplète au moment de la planification. Vérification effective sur disque : `chunks/detection/` ne contient aucun fichier `*_train_chunk*.npz` (contrairement aux 3 autres, tous confirmés avec des chunks réels). Migrer FIGHTERJET_DETECTION aurait donc fait lever la nouvelle `ValueError` au tout premier entraînement — régression directe de AD-20 (non-régression, l'ancien pipeline doit rester pleinement fonctionnel). Ses `decay_steps` (tpu=22000, gpu=105280) restent donc en dur, avec un commentaire expliquant pourquoi et notant qu'ils ne sont eux-mêmes pas corrigés pour `accum_steps` (gpu=2) — à remigrer si/quand le dataset est régénéré.

Vérification en conditions réelles (les 3 configs migrées + FIGHTERJET_DETECTION, script `_count_real_train_samples`/`_resolve_lr_schedule_steps` exécuté contre les chunks réels sur disque) :
- CIFAR10 : auto = 17595 = valeur historique exacte (accum_steps=1, aucun changement de comportement)
- JAX_DETECTOR : auto = 11850(tpu)/23700(gpu), vs. ancien 14880/22845 — écart dû au volume réel actuel de chunks (101173 échantillons) différent de celui mesuré en 2026-07-19, pas un artefact du fix ; comportement voulu de l'auto-calcul (se corrige sur la donnée réelle)
- FIGHTERJET_CLASSIFICATION : auto = 1610 (ceil, confirmé exactement, changement de comportement assumé, voir Intent)
- FIGHTERJET_DETECTION : reste 22000(tpu)/105280(gpu), confirmé inchangé (pas de recalcul déclenché)

## Verification

**Commands:**
- `python3 tests/test_decay_steps_calculation.py` -- toutes les assertions passent (convention du projet : scripts autonomes, pas pytest)
- `python3 -c "import dataset_configs; [dataset_configs.validate_config(k, dataset_configs.DATASET_CONFIGS[k]) for k in ['FIGHTERJET_CLASSIFICATION','FIGHTERJET_DETECTION','JAX_DETECTOR','CIFAR10']]"` -- pas d'exception, `True` pour chaque dataset
- Suite de tests existante (`tests/test_*.py`, exécution directe) -- pas de régression

**Exécuté (2026-08-21) :** les 3 commandes ci-dessus confirmées vertes, plus `validate_config` sur les 14 `DATASET_CONFIGS` (pas seulement les 4 concernés) et la suite complète des 26 fichiers `tests/test_*.py` (25 hors ce nouveau fichier) — 0 échec.

## Suggested Review Order

**Priorité config-explicite vs auto-calcul (le cœur du fix)**

- Point d'entrée : signature + logique complète de résolution, priorité explicite > auto-calcul, formule ceil corrigée, 3 gardes ValueError.
  [`trainer.py:57`](../../trainer.py#L57)

- Une valeur `decay_steps` explicite est retournée telle quelle, jamais recalculée.
  [`trainer.py:89`](../../trainer.py#L89)

- Ni valeur explicite ni chunks détectables → erreur explicite, plus de repli silencieux sur 6000.
  [`trainer.py:99`](../../trainer.py#L99)

- Formule corrigée en ceil (pas floor) : reflète le flush du groupe incomplet en fin d'epoch par `train_epoch`.
  [`trainer.py:115`](../../trainer.py#L115)

- Garde contre un résultat dégénéré (0 step/epoch) plutôt qu'un `decay_steps=0` silencieux.
  [`trainer.py:116`](../../trainer.py#L116)

- Point d'appel dans `create_train_state` : remplace l'ancien if/else qui écrasait silencieusement les valeurs en dur.
  [`trainer.py:245`](../../trainer.py#L245)

**Migration config vers l'auto-calcul (3 datasets sûrs)**

- FIGHTERJET_CLASSIFICATION tpu : champ retiré, changement de comportement assumé (6419→1610) documenté en commentaire.
  [`dataset_configs.py:127`](../../dataset_configs.py#L127)

- CIFAR10 gpu : migration neutre, l'auto-calcul reproduit exactement l'ancienne valeur (accum_steps=1).
  [`dataset_configs.py:397`](../../dataset_configs.py#L397)

- JAX_DETECTOR tpu : migration neutre, élimine le recalcul manuel répété à chaque changement de volume.
  [`dataset_configs.py:552`](../../dataset_configs.py#L552)

**Exclusion délibérée (déviation vs plan initial, trouvée en implémentation)**

- FIGHTERJET_DETECTION NON migré : aucun chunk sur disque aujourd'hui, migrer aurait violé AD-20 (non-régression).
  [`dataset_configs.py:220`](../../dataset_configs.py#L220)

**Résolution des 2 findings de revue adversariale (JAX_KEPLER)**

- Commentaire périmé corrigé : JAX_KEPLER n'a plus de repli implicite sur 6000, lève désormais une ValueError explicite (décision assumée).
  [`dataset_configs.py:316`](../../dataset_configs.py#L316)

**Tests (périphérique)**

- Audit systématique de `DATASET_CONFIGS` : aurait détecté le trou JAX_KEPLER avant la revue adversariale.
  [`test_decay_steps_calculation.py:214`](../../tests/test_decay_steps_calculation.py#L214)

- Cas minimal isolant le comportement ceil vs floor sans ambiguïté.
  [`test_decay_steps_calculation.py:89`](../../tests/test_decay_steps_calculation.py#L89)
