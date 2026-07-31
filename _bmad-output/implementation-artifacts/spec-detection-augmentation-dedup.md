---
title: 'Extraction de la logique d''augmentation partagée entre DetectionDataset et CenterNetDetectionDataset'
type: 'refactor'
created: '2026-07-31'
status: 'done'
review_loop_iteration: 0
context: []
baseline_commit: '097554272540e5df09cc8d4bf4ad50e961ee81b3'
---

<frozen-after-approval reason="human-owned intent — do not modify unless human renegotiates">

## Intent

**Problem:** `DetectionDataset` et `CenterNetDetectionDataset` (`data_management.py`) dupliquent ~120 lignes de logique d'augmentation géométrique (flip V/H, translation pad+crop, zoom, brightness/contrast). La duplication a dérivé : la version CenterNet porte des corrections (interpolation `nearest` pour heatmap/size vs `bilinear` pour l'image, rescale de la magnitude `size` par le facteur de zoom réellement appliqué après clamp) que la version masque n'a jamais eu besoin d'avoir — deux implémentations à faire évoluer en parallèle, risque d'oubli d'un correctif d'un côté.

**Approach:** Extraire les transformations géométriques communes (flip, translation, zoom) en fonctions internes partagées à `data_management.py`, paramétrées par mode de padding et méthode d'interpolation par tenseur. Chaque classe garde en local son propre appel et son propre post-traitement spécifique au format de payload (clip `[0,1]` du masque/heatmap, rescale de `size`).

## Boundaries & Constraints

**Always:**
- Comportement numériquement identique avant/après pour chaque classe, sous les mêmes seeds/paramètres — vérifié pixel-level (voir Verification), pas seulement par tests unitaires existants. Non-régression dure : `FIGHTERJET_DETECTION` (AD-20 de la spine architecture) et `JAX_DETECTOR`.
- Les fonctions extraites restent dans `data_management.py` (2 seuls consommateurs — pas de nouveau module).
- Dispatch `task_type` et séparation des deux classes inchangés (AD-17) — pas de fusion en une classe à branchement conditionnel.
- `CenterNetDetectionDataset` garde son rescale de `size` au zoom et son interpolation `nearest` pour heatmap/size — ne pas les perdre, ne pas les propager silencieusement au masque de `DetectionDataset`.

**Ask First:** si un comportement de zoom s'avère impossible à préserver à l'identique entre les deux classes sans dupliquer une branche interne (edge case non anticipé), HALT et demander avant de trancher.

**Never:** corriger au passage les bugs connus mais hors scope de ce refactor (`drop_remainder=True` sur le split val, heatmap potentiellement vide après un crop de zoom défavorable — items de backlog séparés) ; toucher `rotation_factor` (paramètre mort pour ces deux classes, hors scope) ; changer les valeurs par défaut de `augmentation_params` dans `dataset_configs.py` ; toucher `ChunkManager` ou `ChessPolicyValueDataset`.

</frozen-after-approval>

## Code Map

- `data_management.py:264-419` -- `DetectionDataset` (payload masque binaire) -- source de la duplication à refactorer
- `data_management.py:422-601` -- `CenterNetDetectionDataset` (payload heatmap+size) -- source de la duplication, version avec correctifs zoom/size à préserver
- `tests/test_centernet_detection_dataset.py` -- pattern existant de mock déterministe de `tf.random.uniform` (forcer `do_zoom=True` + `scale` fixe) -- à réutiliser pour la capture de baseline et la vérification pixel-level
- `dataset_configs.py:173` (`FIGHTERJET_DETECTION`), `:466` (`JAX_DETECTOR`) -- seuls domaines avec augmentation géométrique active sur ces deux classes (flip/zoom/translation/brightness/contrast tous non-nuls) -- source des paramètres réalistes à utiliser en test

## Tasks & Acceptance

**Execution:**
- [x] `tests/test_augmentation_dedup_regression.py` (nouveau) -- capturer, AVANT tout changement à `data_management.py`, une baseline pixel-level pour `DetectionDataset` ET `CenterNetDetectionDataset` (chunk synthétique fixe façon `test_centernet_detection_dataset.py`, `tf.random.uniform` mocké pour couvrir flip/translation/zoom), avec les paramètres réels de `FIGHTERJET_DETECTION`/`JAX_DETECTOR` -- garantit qu'on compare le vrai comportement de prod, pas un cas jouet
- [x] `data_management.py` -- extraire les fonctions internes partagées de flip/translation/zoom (paramétrées padding mode + interpolation method par tenseur) -- élimine la duplication sans changer le comportement
- [x] `data_management.py` -- rebrancher `DetectionDataset.create_tf_dataset` et `CenterNetDetectionDataset.create_tf_dataset` sur ces fonctions partagées, en gardant local le post-traitement spécifique (clip masque/heatmap, rescale size) -- confine le changement au strict transport de logique
- [x] `tests/test_augmentation_dedup_regression.py` -- comparer la baseline capturée à la sortie post-refactor (mêmes seeds/paramètres) -- preuve exécutable de non-régression, pas une lecture de code

**Acceptance Criteria:**
- Given la baseline pixel-level pré-refactor pour `DetectionDataset` et `CenterNetDetectionDataset` (params réels `FIGHTERJET_DETECTION`/`JAX_DETECTOR`, tirages aléatoires mockés identiques), when le code post-refactor tourne avec les mêmes entrées, then les tenseurs de sortie (image, masque/heatmap/size) sont identiques (exact pour les opérations déterministes, tolérance uniquement pour l'arrondi flottant déjà présent avant refactor).
- Given `tests/test_centernet_detection_dataset.py`, when exécuté post-refactor, then toutes les assertions existantes passent sans modification.
- Given le dispatch `task_type` dans `main.py`/`task_strategies.py`/`data_management.py`, when inspecté post-refactor, then les littéraux et la séparation des deux classes restent inchangés (AD-17).

## Spec Change Log

- **Découvert en implémentation :** `tf.image.random_brightness`/`random_contrast` ne consomment pas `tf.random.uniform` en interne (vérifié empiriquement) - le mock déterministe ne les couvre pas, rendant `brightness_delta`/`contrast_factor` non-reproductibles d'un run de process à l'autre. Amendement : `AUGMENTATION_PARAMS` du test fixe ces deux valeurs à `0.0` (au lieu des 0.15/0.30 réels) ; sans impact sur la couverture car ces deux blocs de code sont strictement identiques (non dupliqués avec dérive) des deux côtés, avant comme après refactor. La comparaison pixel-level porte sur flip/translation/zoom, seule zone de dérive réelle.
- **Découvert en implémentation :** `ds.shuffle(1000)` utilise son propre générateur aléatoire (pas `tf.random.uniform`), non mocké. Avec 2+ échantillons synthétiques dans le chunk `DetectionDataset`, l'ordre du batch pouvait varier d'un run à l'autre et produire un faux écart pixel-level (diagnostiqué par un repro isolé montrant 0 écart entre ancienne et nouvelle logique dans le même process). Amendement : chunk de test réduit à 1 seul échantillon (`batch_size=1`), comme le fait déjà `CenterNetDetectionDataset` - rien à réordonner.
- **Trouvé en revue adversariale (Blind Hunter + Edge Case Hunter, convergents) :** `_fake_uniform` renvoyait la médiane exacte de chaque intervalle `tf.random.uniform(minval, maxval)`, ce qui annule silencieusement `shift_x`/`shift_y` (médiane de `[-f,f]` = 0) et `scale` (médiane de `[1-f,1+f]` = 1.0) - translation et zoom devenaient des no-ops pixel-wise malgré `do_translate`/`do_zoom` forcés à `True`. Le test ne vérifiait donc jamais réellement `extra_pad_modes` (REFLECT vs CONSTANT), `extra_interp_methods` (nearest vs bilinear) ni `extra_rescale_on_zoom` (rescale de `size`) - exactement la partie à risque de ce refactor. **KEEP :** la structure capture/comparaison et le choix de forcer tous les booléens à `True` restent corrects, seul le calcul de la valeur continue était en cause. Amendement (patch) : `_fake_uniform` renvoie `minval + 0.8*(maxval-minval)` au lieu de la médiane. Baseline recapturée sur le code pré-refactor (via `git stash` ciblé sur `data_management.py`) et re-vérifiée bit-identique post-refactor - confirmé que `size` porte maintenant une magnitude rescalée non-triviale (`scale≈1.21`) et que le masque contient des valeurs d'interpolation bilinéaire intermédiaires (pas purement binaire), preuve que zoom/translation sont réellement exercés.
- **Trouvé en revue adversariale (Edge Case Hunter) :** `_apply_geometric_and_color_augmentation` ne validait pas que `extra_tensors`/`extra_pad_modes`/`extra_interp_methods`/`extra_rescale_on_zoom` ont la même longueur - un futur 3ᵉ appelant avec des listes mal alignées tomberait sur une troncature silencieuse de `zip()` plutôt qu'une erreur claire. Amendement (patch) : `ValueError` explicite si les longueurs divergent.
- **Trouvé en revue adversariale (Blind Hunter) :** paramètre `rotation_factor` mort inclus dans `AUGMENTATION_PARAMS` du test (aucune logique de rotation dans la fonction partagée) - filler trompeur suggérant une couverture inexistante. Amendement (patch) : retiré. Also PEP8 : 2 lignes vides manquantes avant la nouvelle fonction top-level (patch, corrigé).
- **Findings rejetés (bruit ou déjà couverts par une décision délibérée déjà actée dans ce spec) :** couverture manquante de brightness/contrast sous test (déjà expliqué et accepté ci-dessus, code inchangé) ; `assert` nu strippable sous `python -O` et absence d'intégration pytest/CI (ce projet n'a délibérément aucun CI, tous les tests existants suivent ce même style d'exécution manuelle) ; branches `tf.cond` "skip" jamais exercées (triviales, risque nul, pas la zone de dérive visée) ; `img_h`/`img_w` dérivés de `img` avant flip plutôt que par tenseur (comportement identique au code original pré-refactor, pas introduit par ce diff) ; extraction perçue comme un écart à AD-17 (déjà négocié explicitement dans les Boundaries & Constraints de ce spec, décision délibérée d'Aymeric).
- **Déféré, hors scope de ce refactor (confirmé applicable, non actionnable ici) :** `np.array_equal` exige une égalité bit-exacte, potentiellement fragile si ce test venait à tourner sur une autre machine/GPU/version TF - jugé non-bloquant car ce test sert une vérification manuelle ponctuelle (pas de CI dans ce projet), déjà utilisée avec succès dans cette session pour son objectif déclaré.

## Design Notes

Réutiliser exactement le pattern de `test_centernet_detection_dataset.py` (monkeypatch de `tf.random.uniform` retournant des constantes fixes selon `minval`/`maxval`) pour forcer un tirage déterministe côté `DetectionDataset` aussi — ce fichier n'a aujourd'hui aucun test dédié, contrairement à `CenterNetDetectionDataset`.

## Verification

**Commands:**
- `python3 tests/test_augmentation_dedup_regression.py` -- expected: baseline pré-refactor == sortie post-refactor pour les deux classes, sur les paramètres réels `FIGHTERJET_DETECTION`/`JAX_DETECTOR`
- `python3 tests/test_centernet_detection_dataset.py` -- expected: `Tous les tests sont passés.` (comportement CenterNet inchangé)

## Suggested Review Order

**Fonction partagée (le cœur du refactor)**

- Point d'entrée : signature + garde de cohérence des listes parallèles (patch post-revue) - c'est ici qu'un futur 3ᵉ appelant pourrait casser silencieusement pad_mode/interpolation/rescale.
  [`data_management.py:266`](../../data_management.py#L266)

- Garde de longueur `extra_pad_modes`/`extra_interp_methods`/`extra_rescale_on_zoom` vs `extra_tensors` - échoue explicitement plutôt qu'un `zip()` tronqué silencieusement.
  [`data_management.py:299`](../../data_management.py#L299)

**Rebranchement des deux classes (le risque de non-régression)**

- `DetectionDataset.augment_fn` - masque : `bilinear`/`CONSTANT`, pas de rescale - vérifier que ça correspond bien à l'ancien comportement (pas de dérive vers `nearest`).
  [`data_management.py:475`](../../data_management.py#L475)

- `CenterNetDetectionDataset.augment_fn` - heatmap/size : `nearest`/`CONSTANT`, rescale de `size` activé uniquement ici - le point précis où une inversion de flag serait la régression la plus coûteuse.
  [`data_management.py:560`](../../data_management.py#L560)

**Vérification pixel-level (preuve de non-régression)**

- `AUGMENTATION_PARAMS` + `_fake_uniform` - le mock a été corrigé en revue (médiane exacte → 80% de l'intervalle) pour que translation/zoom ne dégénèrent pas en no-op.
  [`test_augmentation_dedup_regression.py:59`](../../tests/test_augmentation_dedup_regression.py#L59)

- `_capture_or_compare` - logique capture-puis-compare : premier run = baseline pré-refactor, runs suivants = comparaison bit-exacte.
  [`test_augmentation_dedup_regression.py:84`](../../tests/test_augmentation_dedup_regression.py#L84)

- Test `DetectionDataset` (1 échantillon, pas de shuffle-order non-déterministe).
  [`test_augmentation_dedup_regression.py:130`](../../tests/test_augmentation_dedup_regression.py#L130)

- Test `CenterNetDetectionDataset` (réutilise le pattern de `test_centernet_detection_dataset.py`).
  [`test_augmentation_dedup_regression.py:192`](../../tests/test_augmentation_dedup_regression.py#L192)
