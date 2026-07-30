---
baseline_commit: 049da4cdc45da1f4e243261b159e9e9ef68debfe
---

# Story 9.4: Validation non-régression `JAX_DETECTOR` par exécution

Status: done (clôturée sans code-review — aucune modification de code de production, décision explicite d'Aymeric 2026-07-29)

<!-- Note: Validation is optional. Run validate-create-story for quality check before dev-story. -->

## Story

As a mainteneur du pipeline,
I want valider par exécution réelle que `JAX_DETECTOR` fonctionne à l'identique avant et après l'intégration du domaine échecs,
so that NFR1/AD-21 (contrainte dure) est prouvé, pas supposé.

## Acceptance Criteria

1. **Given** une baseline (boxes/classes/scores sur un set fixe d'images) capturée avant le début de cette epic **When** `JAX_DETECTOR` est ré-exécuté (entraînement et/ou inférence) après les Stories 9.1 à 9.3 **Then** les résultats sont comparés à la baseline par diff — tout écart documenté et justifié, jamais silencieux — FR7
   - **Précision (PRD FR-7, formulation plus stricte que ci-dessus) :** "tout écart est un échec de la story, pas une simple observation qualitative." Il y a une tension réelle entre cette phrase du PRD et le "documenté et justifié" de l'AC ci-dessus (source `epics.md`) — traiter le PRD comme prioritaire en cas de doute (c'est la source amont), mais si un écart minime et explicable apparaît (ex. flottant epsilon), ne pas trancher seul : présenter le résultat à Aymeric via `AskUserQuestion` plutôt que de décider unilatéralement que "c'est acceptable" ou "c'est un échec."
   - **Méthode imposée, pas au choix du dev agent :** PRD §4.3 FR-7 dit explicitement "reprend celle des Stories 2.4/8.9 des epics précédentes (comparaison de baseline par exécution, pas lecture de code)." Voir Dev Notes § Précédents pour le detail exact de ces deux méthodes.
2. **Given** `CenterNetDetectionStrategy`, `CenterNetDetectionDataset`, `aircraft_detector_centernet(_lite)` et leurs consommateurs actuels **When** cette story est complétée **Then** aucun n'a été modifié par cette epic — AD-21
3. **Given** les Stories 9.1 à 9.3 complétées **When** cette story conclut l'Epic 9 **Then** FR1 à FR6 sont confirmés couverts et le domaine échecs est entraînable de bout en bout via `Trainer`

## Tasks / Subtasks

- [x] Task 1: Confirmer par `git diff` que `CenterNetDetectionStrategy`/`CenterNetDetectionDataset`/`aircraft_detector_centernet(_lite)` sont structurellement intouchés, et que le checkpoint `best_model_jax_detector.pkl` (racine du repo) n'a pas changé (AC: 2)
  - [x] Subtask 1.1: `git diff HEAD -- task_strategies.py data_management.py model_library.py` — confirmer que toute insertion est **après** la fin de `CenterNetDetectionStrategy`/`CenterNetDetectionDataset`/`create_aircraft_detector_centernet(_lite)`, jamais à l'intérieur (voir Dev Notes § Preuve déjà rassemblée pour les numéros de ligne exacts déjà vérifiés en amont de cette story — à revérifier, pas à re-découvrir de zéro)
  - [x] Subtask 1.2: `git diff HEAD -- main.py dataset_configs.py trainer.py` — confirmer que les 3 lignes pré-existantes modifiées (`class_names = config.get(...)`, retrait de `"image_size"` de `required` dans `validate_config`, `self.num_channels = config.get("num_channels", ...)`) sont behavior-preserving pour la config `JAX_DETECTOR` (qui fournit `class_names`, `image_size`, et ne définit jamais `num_channels` — donc tombe sur exactement le même comportement qu'avant, voir Dev Notes)
  - [x] Subtask 1.3: `git status`/`git diff` sur `best_model_jax_detector.pkl` (fichier suivi par git, `git ls-files` le confirme) — zéro modification
  - [x] Subtask 1.4: Lister les consommateurs réels actuels de la chaîne `JAX_DETECTOR`/single-pass (`bounding_boxes_with_classification_from_video_generation.py`, `tools/bounding_boxes_with_classification_from_images_generation.py`, `tools/audit/audit_dataset_detection_jax.py`) et confirmer par `git diff` qu'aucun n'a été touché par cette epic
- [x] Task 2: Capturer une baseline `JAX_DETECTOR` (boxes/classes/scores) sur le set d'images fixe déjà établi par les Stories 7/8, **dans l'état du code d'avant l'epic 9** (AC: 1)
  - [x] Subtask 2.1: Utiliser `test_media/testvid01.png`, `testvid02.png`, `testvid03.png` (déjà le set de référence de `test_pixel_parity.py`/`test_single_pass_predict_fn.py`, Epic 8) — ne pas inventer un nouveau set d'images
  - [x] Subtask 2.2: **Avant d'exécuter quoi que ce soit** (`git stash` inclus), présenter le plan à Aymeric et obtenir confirmation explicite — contrainte du projet, pas optionnelle (voir Dev Notes § Prudence exécution locale)
  - [x] Subtask 2.3 (méthode adaptée, voir Debug Log — `git stash` remplacé par `git worktree` car l'Epic 9 est désormais committée) : `git worktree add ... 1a85f25` (le vrai état pré-epic, parent du commit `049da4c` qui contient toute l'Epic 9) — exécuté la chaîne d'inférence `JAX_DETECTOR`/single-pass réelle (`build_single_pass_predict_fn`) sur les 3 images, sauvegardé boxes/classes/scores en JSON, puis `git worktree remove` immédiatement après capture
  - [x] Subtask 2.4: Alternative de repli non nécessaire — `git worktree` a fonctionné du premier coup, présentée et confirmée par Aymeric avant exécution
- [x] Task 3: Ré-exécuter la même capture avec le code **post-epic 9** (état courant du repo) et comparer par diff exact à la baseline de Task 2 (AC: 1)
  - [x] Subtask 3.1: Mêmes 3 images, même checkpoint, même seuils de config (`JAX_DETECTOR` dans `dataset_configs.py`, non modifiée par l'epic)
  - [x] Subtask 3.2: **Diff bit-exact confirmé** (`diff` exit code 0, zéro différence) — 7/7/7 détections valides identiques sur les 3 images, mêmes boxes/classes/scores au 4e décimal près. Aucun écart à documenter/justifier.
- [x] Task 4: Ré-exécuter la suite de tests de régression `JAX_DETECTOR`/CenterNet déjà existante (héritée des Epics 7/8, non modifiée par cette epic) et confirmer qu'elle passe intégralement (AC: 2)
  - [x] Subtask 4.1: 7/9 scripts passent (`test_centernet_detection_strategy`, `test_aircraft_detector_centernet`, `test_jax_detector_config`, `test_peak_extraction_topk`, `test_pixel_parity`, `test_single_pass_predict_fn`, `test_detector_inference_composition`). 2 échouent (`test_centernet_detection_dataset.py`, `test_jax_detector_dataset_tools.py`) — voir Subtask 4.2.
  - [x] Subtask 4.2: Écart documenté, pas juste "0 exception" — les 2 échecs sont un `ImportError: cannot import name 'process_detection_dataset_v2'`. **Confirmé non lié à cette epic** : `git diff 1a85f25..049da4c -- dataset_builder/jax_detector_dataset_tools.py` est vide (zéro changement sur ce fichier dans toute l'Epic 9), et la fonction `process_detection_dataset_v2` n'existe ni avant ni après (la fonction réelle s'appelle `process_detector_dataset`) — défaut préexistant de la suite de tests, indépendant d'Epic 9. Discuté avec Aymeric (pas tranché unilatéralement, conforme à la nuance AC1) : ne bloque pas AC2 (aucun lien avec cette epic, prouvé par le diff vide), traité comme dette technique séparée (voir `deferred-work.md`).
- [x] Task 5: Ré-exécuter les consommateurs de production réels (script(s) et/ou outil(s) listés en Subtask 1.4) sur un cas réel, avec arbitrage explicite d'Aymeric sur la profondeur (AC: 2)
  - [x] Subtask 5.1: Option proposée et choisie par Aymeric : (a) légère — `tools/bounding_boxes_with_classification_from_images_generation.py` sur son vrai dossier `/home/aobled/Downloads/tmp_multi` (367 images). **Point additionnel signalé avant exécution** (non anticipé par la story) : ce script déplace physiquement les fichiers traités (`shutil.move` vers des sous-dossiers par classe) — pas une opération en lecture seule comme Task 2/3. Présenté explicitement à Aymeric avant lancement, confirmé ("impact très limité... go").
  - [x] Subtask 5.2: Exécuté avec succès — 367 images traitées, 682 avions détectés/classifiés, ~22s, zéro erreur. Chemin de production réel (chargement checkpoints + `build_single_pass_predict_fn` + classification + écriture JSON + organisation fichiers) exercé de bout en bout sans problème.
- [x] Task 6: Bilan de couverture FR1-FR6 + confirmation que le domaine échecs est entraînable de bout en bout via `Trainer` (AC: 3)
  - [x] Subtask 6.1: Bilan de couverture, résultat d'exécution réel par FR (pas une relecture du texte des stories) :
    - **FR-1** (extraction PGN) : `dataset_builder/chess_pgn_dataset_tools.py` (Story 9.1) exécuté pour de vrai sur l'archive Carlsen (pgnmentor) — 7484 parties → 691 779 positions, 139 chunks `.npz`, confirmé par `tests/test_chess_pgn_dataset_tools.py`.
    - **FR-2** (encodage input) : `chess_target_encoding.py::encode_position` (Story 9.1), validé par `tests/test_chess_target_encoding.py` (round-trip exhaustif : 20 coups légaux position de départ, roque, prise en passant, 4 promotions, partie semi-aléatoire 40 demi-coups, padding d'historique).
    - **FR-3** (labels policy/value) : `_iter_game_examples`/`_value_for_mover` (Story 9.1), testé sur 3 parties réelles à résultats distincts (1-0/0-1/1/2-1/2) — value alterne correctement, aucun filtrage par résultat.
    - **FR-4** (`TaskStrategy` double tête) : `ChessPolicyValueStrategy` (Story 9.3, `task_strategies.py`), 8 tests (`tests/test_chess_task_strategy.py`) **+ validé par un entraînement réel complet cette session** (15 epochs, `policy_loss`/`value_loss` décroissants, PolicyAccuracy validation 8.2%→24.43%, aucune divergence).
    - **FR-5** (modèle échecs) : `ChessCnnAttentionPolicyValue`/`create_chess_cnn_attention_policy_value` (Story 9.2, `model_library.py`), 5 tests (`tests/test_chess_model.py`, dont différentiabilité de bout en bout sur les 19 modules de paramètres) **+ convergence réelle confirmée par le même entraînement** (382 017 paramètres à K=8).
    - **FR-6** (zéro modification structurelle de `Trainer`) : un seul écart de ligne dans tout `trainer.py` sur toute l'epic (`self.num_channels`, Story 9.3, arbitrage explicite d'Aymeric) — confirmé par `git diff 1a85f25..049da4c -- trainer.py` (Task 1 de cette story). Tout le reste (nouveau modèle, nouvelles losses, nouvelle `TaskStrategy`, nouvelle classe de dataset) passe par les mécanismes génériques déjà existants.
    - **FR-7/AD-21 (non-régression, objet de cette story)** : Tasks 1, 2/3, 4, 5 de cette story — diff vide sur le code CenterNet, baseline `JAX_DETECTOR` byte-exacte avant/après, suite de régression (7/9, 2 échecs préexistants non liés), consommateur de production réel exécuté avec succès.
  - [x] Subtask 6.2: "Entraînable de bout en bout via `Trainer`" — largement dépassé la preuve minimale envisagée par la story (`test_trainer_create_train_state_for_chess` seul). Un **entraînement réel complet** (15 epochs, dataset Carlsen réel 691 779 positions, `python3 main.py CHESS`) a été exécuté cette session, à l'initiative d'Aymeric et avec sa confirmation explicite à chaque étape (voir `deferred-work.md`, section "1er entraînement réel CHESS...") : convergence propre (PolicyAccuracy validation 8.2%→24.43%, aucun crash, aucune divergence), export de checkpoint fonctionnel, rechargement et inférence réelle validés (`chess/chess_game.py --vs-model`). Preuve largement suffisante, tranchée par l'exécution elle-même plutôt que par une décision a priori.
  - [x] Subtask 6.3: Confirmé — `git diff 1a85f25..049da4c -- trainer.py` (Task 1) montre exactement une seule ligne modifiée dans tout le fichier sur toute l'Epic 9, l'écart déjà explicitement autorisé par Aymeric (Story 9.3, `self.num_channels`). Aucune autre modification structurelle de `Trainer`.

## Dev Notes

### Nature de cette story — aucune nouvelle logique métier

Comme les Stories 7.8, 8.9 et 2.4, cette story n'introduit aucun nouveau code de production — elle exécute, compare, et documente. Le seul artefact de code potentiellement créé est un script de capture de baseline JAX_DETECTOR (Task 2/3, jetable ou conservé selon convention — voir précédent `_bmad-output/implementation-artifacts/baseline/capture_baseline.py`). Toute tentation de "corriger" un écart trouvé pendant cette story doit être signalée à Aymeric, pas résolue silencieusement — cette story est un filet de sécurité, pas un chantier de correction.

### Précédents exacts (PRD FR-7 les cite nommément — méthode imposée, pas une option)

- **Story 8.9** (`_bmad-output/implementation-artifacts/8-9-validation-finale-fusion-de-boites-resolue-non-regression.md`) : story de clôture d'epic, même structure (validation par exécution + bilan de couverture FR). Réutilise directement des captures déjà faites par une story précédente plutôt que de recapturer quand c'est possible (Task 1 de 8.9 réutilise `archive/baseline_video_8_7.json`/`archive/migrated_video_8_7.json`). Confirme la non-régression du chemin d'entraînement (`AircraftDetectorUNet`/`DetectionStrategy`/`DetectionDataset`) **par lecture de code + `git diff`, sans ré-exécution**, sur décision utilisateur explicite quand le risque résiduel est jugé faible et le coût de ré-exécution élevé (dataset ~150 dossiers). **Différence importante avec cette story** : 8.9 validait un ancien pipeline volontairement préservé sans modification depuis plusieurs epics (risque quasi nul) ; ici, `JAX_DETECTOR` partage des fichiers (`task_strategies.py`, `data_management.py`, `main.py`, `dataset_configs.py`) directement modifiés par les Stories 9.1-9.3 pour y ajouter du code chess — le risque de collatéral est réel (cf. le bug whitespace trouvé en review de la Story 9.3 sur `KeplerStrategy`, dans le même fichier). **Une preuve par lecture de code seule ne suffit donc pas ici** — AD-21 l'exige explicitement ("Validé par exécution réelle... jamais par lecture de code seule").
- **Story 2.4** (`_bmad-output/implementation-artifacts/2-4-validation-post-purge.md`) : établit le pattern "rechargement de checkpoint + comparaison bit-exacte des prédictions = le test le plus proche d'un entraînement réel sans en payer le coût." Un entraînement complet local a été explicitement refusé par l'utilisateur (risque de crash mémoire, entraînement réel réservé à Colab/TPU) et reporté, avec le raisonnement écrit noir sur blanc dans la story (voir citation dans le fichier). **Directement applicable ici** pour Task 5/6 : privilégier une preuve d'inférence réelle + rechargement de checkpoint plutôt qu'un entraînement complet local, sauf si Aymeric demande explicitement le contraire.
- **Story 1.1/1.10** : établit le format JSON de baseline (`_bmad-output/implementation-artifacts/baseline/baseline_before.json`, `capture_baseline.py`) et le script de comparaison (`_bmad-output/implementation-artifacts/baseline/verify_after_migration.py`). Ce format ne couvre pas `JAX_DETECTOR` (capturé avant l'existence de CenterNet, Epic 7) — mais sa structure (dict par image, boxes + predictions arrondies) est un bon modèle à réutiliser pour le nouveau script de Task 2/3, par cohérence de convention plutôt que par obligation stricte.

### État git actuel — important pour Task 2/3

Au moment de la création de cette story, **aucun commit n'a encore été fait pour l'Epic 9** — `git status` confirme que tous les fichiers modifiés par les Stories 9.1-9.3 (`data_management.py`, `dataset_configs.py`, `loss_functions.py`, `main.py`, `model_library.py`, `requirements.txt`, `task_strategies.py`, `trainer.py`) sont encore à l'état "modified, not staged", et que les nouveaux fichiers (`chess_target_encoding.py`, `dataset_builder/chess_pgn_dataset_tools.py`, `tests/test_chess_*.py`) sont "untracked". `HEAD` (commit `1a85f25`) est donc **exactement** l'état pré-epic-9 — `git stash` (qui inclut les fichiers untracked seulement avec `-u`, mais Task 2 n'a besoin que des fichiers modifiés/trackés pour restaurer le comportement JAX_DETECTOR, pas de supprimer les nouveaux fichiers chess) donne un accès direct et fiable à cet état pour Task 2, sans avoir besoin de `git checkout`/`git worktree` (plus lourd). Vérifier que `git status` n'a pas changé entre la création de cette story et son exécution (Aymeric peut avoir committé entretemps) avant de s'appuyer sur cette hypothèse.

### Prudence exécution locale — contrainte du projet, pas une suggestion

Ce projet a déjà connu un crash machine lors d'un entraînement/traitement lourd lancé sans confirmation préalable (voir mémoire `feedback_caution_before_heavy_local_execution`), et un incident similaire plus tôt dans cette même epic (exécution non planifiée de `chess_pgn_dataset_tools.py` sur un vrai dossier PGN via "Run Python File" de VSCode, Story 9.1/9.2). **Toute action de cette story impliquant une exécution non triviale — `git stash` inclus, même s'il est réversible, tout entraînement complet ou partiel, tout traitement du dataset réel de 691 779 positions échecs — doit être présentée à Aymeric et confirmée avant exécution, jamais lancée silencieusement.** L'inférence seule sur 3 images fixes (Task 2/3) et le rechargement de checkpoint (Task 1) sont légers et ne nécessitent pas cette prudence renforcée — mais Task 5/6 (entraînement potentiel) si.

### Preuve déjà rassemblée (à revérifier, pas à re-découvrir de zéro)

Une analyse préliminaire a déjà confirmé, avant la création de cette story :
- `git diff HEAD -- task_strategies.py` : toute insertion (classe `ChessPolicyValueStrategy`) est **après** la fin de `KeplerStrategy` (dernière classe du fichier avant l'ajout) — zéro ligne touchée à l'intérieur de `CenterNetDetectionStrategy`.
- `git diff HEAD -- data_management.py` : `ChessPolicyValueDataset` est insérée juste après la fin de `CenterNetDetectionDataset` (ligne ~600) — zéro ligne touchée à l'intérieur.
- `git diff HEAD -- model_library.py` : `ChessCnnAttentionPolicyValue` est insérée après `create_aircraft_detector_centernet_lite` — zéro ligne touchée à l'intérieur des architectures CenterNet. `MODELS` dict : `aircraft_detector_centernet`/`_lite` restent présentes et inchangées, seule une nouvelle entrée `chess_cnn_attention_policy_value` est ajoutée.
- `git diff HEAD -- main.py` : 2 changements seulement — (1) `class_names = config["class_names"]` → `config.get("class_names")` (JAX_DETECTOR fournit toujours `class_names: ['aircraft']`, donc résultat identique), (2) une nouvelle branche `elif task_type == "chess_policy_value"` ajoutée après la branche `detection_centernet` existante (celle-ci non modifiée).
- `git diff HEAD -- dataset_configs.py` : `validate_config`'s `required` liste perd `"image_size"` (devient conditionnel, comme `class_names`) — la config `JAX_DETECTOR` fournit toujours `image_size: (224, 224)`, donc aucun changement de comportement pour elle. Nouvelle entrée `"CHESS"` ajoutée au dict, `"JAX_DETECTOR"` intouchée.
- `git diff HEAD -- trainer.py` : **un seul écart de ligne dans tout le fichier** (déjà l'écart explicitement autorisé par Aymeric en Story 9.3) — `self.num_channels = 1 if self.grayscale else 3` → `self.num_channels = config.get("num_channels", 1 if self.grayscale else 3)`. La config `JAX_DETECTOR` définit `"grayscale": True` et ne définit jamais `"num_channels"` → `config.get("num_channels", 1)` retourne `1`, comportement strictement identique à avant.
- `best_model_jax_detector.pkl` (racine, suivi par git via `git ls-files`) : absent de `git status`, donc non modifié.

Cette analyse est un point de départ solide pour Task 1, mais reste une lecture de code — **elle ne remplace pas Task 2/3 (exécution réelle)**, qu'AD-21 exige explicitement.

### AD-21 (texte exact, contrainte dure)

[Source: `_bmad-output/planning-artifacts/architecture/architecture-chess-2026-07-27/ARCHITECTURE-SPINE.md#AD-21`] — "`JAX_DETECTOR` et toute sa chaîne (entraînement et inférence) restent utilisables de bout en bout, **sans modification fonctionnelle**, pendant et après cette epic. Aucune story de cette epic ne modifie `CenterNetDetectionStrategy`, `CenterNetDetectionDataset`, `aircraft_detector_centernet(_lite)`, ou tout fichier `tools/` qui en dépend. Validé par exécution réelle — comparaison par diff à une baseline (boxes/classes/scores sur un set fixe d'images) capturée avant l'epic (PRD FR-7), jamais par lecture de code seule. Même précédent qu'AD-20 (parent), appliqué ici à `JAX_DETECTOR` qu'AD-20 ne couvre pas."

### Set d'images fixe déjà établi (ne pas en inventer un nouveau)

`test_media/testvid01.png`, `testvid02.png`, `testvid03.png` (1920×1080, avions réels) + leurs JSON de boîtes annotées (`testvid01_0.json`, etc.) sont déjà le set de référence utilisé par `tests/test_pixel_parity.py` et `tests/test_single_pass_predict_fn.py` (Epic 8). Les réutiliser directement pour la baseline de cette story plutôt que d'introduire un nouveau set — cohérence avec la convention déjà établie du projet.

### Project Structure Notes

- Aucune modification de code de production dans cette story (exécution/analyse/documentation uniquement, même convention que Stories 2.4/7.8/8.9).
- Script(s) de capture/comparaison éventuellement créés : suivre la convention `_bmad-output/implementation-artifacts/baseline/` ou un fichier dédié nommé explicitement pour cette story — à la discrétion du dev agent, mais documenté dans File List.
- Clôture naturelle de l'Epic 9 — dernière story avant une éventuelle rétrospective (`epic-9-retrospective: optional` dans `sprint-status.yaml`).

### Testing Standards

Exécution réelle + documentation, même esprit que les Stories 1.10/2.4/6.4/7.8/8.9 de ce projet — comparaison contre un comportement observé, pas contre une attente théorique. Les tests de régression JAX_DETECTOR existants (Task 4) sont des scripts autonomes (`python tests/test_xxx.py`), pas pytest/unittest — même convention que les tests chess des Stories 9.1-9.3.

### References

- [Source: `_bmad-output/planning-artifacts/epics.md:1279-1297`] — Story 9.4, ACs telles qu'écrites dans l'epic
- [Source: `_bmad-output/planning-artifacts/prds/prd-jax_supervised_training-2026-07-27/prd.md#4.3`, FR-7, NFR-1, SM-2] — formulation plus stricte que l'AC1 de l'epic (tout écart = échec), méthode imposée (Stories 2.4/8.9)
- [Source: `_bmad-output/planning-artifacts/architecture/architecture-chess-2026-07-27/ARCHITECTURE-SPINE.md#AD-21`] — contrainte exacte, "jamais par lecture de code seule"
- [Source: `_bmad-output/implementation-artifacts/8-9-validation-finale-fusion-de-boites-resolue-non-regression.md`] — précédent direct le plus proche (structure, arbitrage utilisateur sur la profondeur de ré-exécution)
- [Source: `_bmad-output/implementation-artifacts/2-4-validation-post-purge.md`] — précédent sur "rechargement checkpoint = preuve suffisante sans entraînement complet local", contrainte machine locale déjà vécue sur ce projet
- [Source: `_bmad-output/implementation-artifacts/baseline/capture_baseline.py`, `baseline_before.json`, `verify_after_migration.py`] — format/convention de capture de baseline déjà établi (Story 1.1/1.10), à réutiliser par cohérence pour le nouveau script de cette story
- [Source: `_bmad-output/implementation-artifacts/9-3-taskstrategy-echecs-loss-composite-et-integration-au-pipeline.md`] — story précédente, écart `trainer.py` autorisé cité en Task 1/6, `test_trainer_create_train_state_for_chess` cité en Task 6

## Dev Agent Record

### Agent Model Used

Claude Sonnet 5 (claude-sonnet-5)

### Debug Log References

- **Divergence importante par rapport aux hypothèses de la story, détectée et gérée avant d'agir** : la story supposait "aucun commit n'a encore été fait pour l'Epic 9" (`HEAD` = état pré-epic). Ce n'était plus vrai à l'exécution — Aymeric avait commité/pushé entre la création de la story et son développement (`049da4c "adding chess model"`, parent `1a85f25` = le vrai état pré-epic). `git stash` (méthode prévue par la story pour Task 2) n'aurait servi à rien dans ce nouveau contexte. Remplacé par `git worktree add ... 1a85f25` (checkout séparé du parent pré-epic, sans toucher au répertoire de travail courant) — présenté à Aymeric et confirmé avant exécution (conforme à la contrainte de prudence de la story).
- `git diff 1a85f25..049da4c` (au lieu de `git diff HEAD` comme littéralement écrit dans la story) utilisé pour toute l'analyse de Task 1 — équivalent fonctionnel exact vu que `HEAD` == `049da4c`.
- Script de capture `_bmad-output/implementation-artifacts/baseline/capture_jax_detector_baseline.py` conçu pour s'insérer dans `sys.path` depuis le `cwd` courant (pas un chemin figé) — permet de l'exécuter tel quel depuis le worktree pré-epic ET depuis le répertoire de travail post-epic sans dupliquer le script.
- 2 tests de régression JAX_DETECTOR préexistants cassés trouvés (`test_centernet_detection_dataset.py`, `test_jax_detector_dataset_tools.py`, `ImportError: process_detection_dataset_v2`) — vérifié par `git diff` que `dataset_builder/jax_detector_dataset_tools.py` a un diff **vide** entre `1a85f25` et `049da4c` : défaut préexistant, non lié à cette epic. Discuté avec Aymeric plutôt que tranché seul (nuance AC1) — ne bloque pas AC2, documenté séparément (`deferred-work.md`).
- Avant Task 5, lecture du code de `tools/bounding_boxes_with_classification_from_images_generation.py` a révélé un effet de bord non anticipé par la story : le script déplace physiquement (`shutil.move`) les images traitées vers des sous-dossiers par classe — pas une opération en lecture seule. Signalé explicitement à Aymeric avant exécution (pas supposé anodin), confirmé.

### Completion Notes List

- **Task 1** : `git diff 1a85f25..049da4c` confirme zéro ligne touchée à l'intérieur de `CenterNetDetectionStrategy`/`CenterNetDetectionDataset`/`aircraft_detector_centernet(_lite)` ; les 3 changements pré-existants (`main.py`, `dataset_configs.py`, `trainer.py`) sont tous comportement-préservants pour `JAX_DETECTOR` ; `best_model_jax_detector.pkl` inchangé ; les 3 consommateurs de production réels ont un diff vide.
- **Task 2/3** : baseline `JAX_DETECTOR` capturée avant (`git worktree` sur `1a85f25`) et après (répertoire de travail courant) sur `test_media/testvid01/02/03.png`, via le vrai chemin de production (`build_single_pass_predict_fn`). **Diff byte-exact** (`diff` exit code 0, zéro différence) — 7/7/7 détections valides identiques. Preuve la plus forte possible pour AD-21/FR-7, exécution réelle comme exigé (pas une lecture de code).
- **Task 4** : 7/9 tests de régression passent. 2 échecs (`ImportError: process_detection_dataset_v2`) confirmés préexistants et non liés à cette epic (diff vide sur le fichier concerné) — documentés, pas corrigés dans cette story (hors périmètre, dette technique séparée).
- **Task 5** : après arbitrage explicite d'Aymeric (et signalement de l'effet de bord de déplacement de fichiers), `tools/bounding_boxes_with_classification_from_images_generation.py` exécuté pour de vrai sur 367 images réelles (`/home/aobled/Downloads/tmp_multi`) — 682 avions détectés/classifiés, ~22s, zéro erreur.
- **Task 6** : FR-1 à FR-6 tous couverts par un résultat d'exécution réel cité précisément (pas une relecture des stories) ; FR-7/AD-21 couvert par cette story elle-même. "Entraînable de bout en bout via `Trainer`" largement dépassé : un entraînement réel complet (15 epochs, dataset Carlsen réel, PolicyAccuracy validation finale 24.43%) a été exécuté cette session, à l'initiative d'Aymeric — bien au-delà de la preuve minimale envisagée par la story. Seul écart structurel sur tout `Trainer` durant toute l'epic : la ligne `self.num_channels` déjà autorisée (Story 9.3), confirmée unique par `git diff`.
- **Épic 9 conclue avec succès** — les 4 stories (9.1-9.4) sont complètes, le pattern `Trainer`/`TaskStrategy` a prouvé sa généricité sur un domaine véritablement différent (policy+value), sans aucune régression sur `JAX_DETECTOR`, et validé par un entraînement réel complet en plus des tests unitaires/intégration.

### File List

Aucune modification de code de production dans cette story (exécution/analyse/documentation uniquement, conforme aux Dev Notes) :
- `_bmad-output/implementation-artifacts/9-4-validation-non-regression-jax-detector-par-execution.md` (cette story, complétée)
- `_bmad-output/implementation-artifacts/sprint-status.yaml` (statut mis à jour)
- `_bmad-output/implementation-artifacts/baseline/capture_jax_detector_baseline.py` (nouveau — script de capture réutilisable)
- `_bmad-output/implementation-artifacts/baseline/jax_detector_baseline_before_epic9.json` (nouveau — baseline pré-epic)
- `_bmad-output/implementation-artifacts/baseline/jax_detector_baseline_after_epic9.json` (nouveau — baseline post-epic, byte-identique à la précédente)
- `_bmad-output/implementation-artifacts/deferred-work.md` (note sur le défaut de tests préexistant, ajoutée séparément de cette story)
