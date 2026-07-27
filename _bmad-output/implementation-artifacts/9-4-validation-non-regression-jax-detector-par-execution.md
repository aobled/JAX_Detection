# Story 9.4: Validation non-régression `JAX_DETECTOR` par exécution

Status: ready-for-dev

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

- [ ] Task 1: Confirmer par `git diff` que `CenterNetDetectionStrategy`/`CenterNetDetectionDataset`/`aircraft_detector_centernet(_lite)` sont structurellement intouchés, et que le checkpoint `best_model_jax_detector.pkl` (racine du repo) n'a pas changé (AC: 2)
  - [ ] Subtask 1.1: `git diff HEAD -- task_strategies.py data_management.py model_library.py` — confirmer que toute insertion est **après** la fin de `CenterNetDetectionStrategy`/`CenterNetDetectionDataset`/`create_aircraft_detector_centernet(_lite)`, jamais à l'intérieur (voir Dev Notes § Preuve déjà rassemblée pour les numéros de ligne exacts déjà vérifiés en amont de cette story — à revérifier, pas à re-découvrir de zéro)
  - [ ] Subtask 1.2: `git diff HEAD -- main.py dataset_configs.py trainer.py` — confirmer que les 3 lignes pré-existantes modifiées (`class_names = config.get(...)`, retrait de `"image_size"` de `required` dans `validate_config`, `self.num_channels = config.get("num_channels", ...)`) sont behavior-preserving pour la config `JAX_DETECTOR` (qui fournit `class_names`, `image_size`, et ne définit jamais `num_channels` — donc tombe sur exactement le même comportement qu'avant, voir Dev Notes)
  - [ ] Subtask 1.3: `git status`/`git diff` sur `best_model_jax_detector.pkl` (fichier suivi par git, `git ls-files` le confirme) — zéro modification
  - [ ] Subtask 1.4: Lister les consommateurs réels actuels de la chaîne `JAX_DETECTOR`/single-pass (`bounding_boxes_with_classification_from_video_generation.py`, `tools/bounding_boxes_with_classification_from_images_generation.py`, `tools/audit/audit_dataset_detection_jax.py`) et confirmer par `git diff` qu'aucun n'a été touché par cette epic
- [ ] Task 2: Capturer une baseline `JAX_DETECTOR` (boxes/classes/scores) sur le set d'images fixe déjà établi par les Stories 7/8, **dans l'état du code d'avant l'epic 9** (AC: 1)
  - [ ] Subtask 2.1: Utiliser `test_media/testvid01.png`, `testvid02.png`, `testvid03.png` (déjà le set de référence de `test_pixel_parity.py`/`test_single_pass_predict_fn.py`, Epic 8) — ne pas inventer un nouveau set d'images
  - [ ] Subtask 2.2: **Avant d'exécuter quoi que ce soit** (`git stash` inclus), présenter le plan à Aymeric et obtenir confirmation explicite — contrainte du projet, pas optionnelle (voir Dev Notes § Prudence exécution locale)
  - [ ] Subtask 2.3: `git stash` (aucun commit n'existe encore pour l'Epic 9 — voir Dev Notes § État git actuel — donc `git stash` restaure exactement l'état pré-epic, y compris `chess_target_encoding.py` non importé) — exécuter la chaîne d'inférence `JAX_DETECTOR`/single-pass réelle (`build_single_pass_predict_fn`, même chemin que `bounding_boxes_with_classification_from_video_generation.py`) sur les 3 images, sauvegarder boxes/classes/scores en JSON (même format que `_bmad-output/implementation-artifacts/baseline/baseline_before.json`, Story 1.1 — même convention de projet, nouveau fichier dédié à cette story) — puis `git stash pop` immédiatement après capture, avant toute autre action
  - [ ] Subtask 2.4: Si `git stash` s'avère impraticable (ex. conflit), alternative de repli explicite à proposer à Aymeric plutôt qu'à décider seul : `git worktree` sur `HEAD`, ou dérivation par argument (côté raisonnement, pas décision unilatérale)
- [ ] Task 3: Ré-exécuter la même capture avec le code **post-epic 9** (état courant du repo) et comparer par diff exact à la baseline de Task 2 (AC: 1)
  - [ ] Subtask 3.1: Mêmes 3 images, même checkpoint, même seuils de config (`JAX_DETECTOR` dans `dataset_configs.py`, non modifiée par l'epic)
  - [ ] Subtask 3.2: Diff bit-exact (mêmes boxes, classes, scores) — tout écart documenté avec sa cause précise (pas une hypothèse), voir la nuance PRD/epics.md dans l'AC1 ci-dessus sur comment traiter un écart
- [ ] Task 4: Ré-exécuter la suite de tests de régression `JAX_DETECTOR`/CenterNet déjà existante (héritée des Epics 7/8, non modifiée par cette epic) et confirmer qu'elle passe intégralement (AC: 2)
  - [ ] Subtask 4.1: `python tests/test_centernet_detection_strategy.py`, `test_centernet_detection_dataset.py`, `test_aircraft_detector_centernet.py`, `test_jax_detector_config.py`, `test_jax_detector_dataset_tools.py`, `test_peak_extraction_topk.py`, `test_pixel_parity.py`, `test_single_pass_predict_fn.py`, `test_detector_inference_composition.py` (scripts autonomes, pas pytest — même convention que les tests échecs des Stories 9.1-9.3, exécuter directement via `python`)
  - [ ] Subtask 4.2: Documenter tout écart de comportement par rapport aux résultats attendus/précédents de ces tests (pas seulement "0 exception")
- [ ] Task 5: Ré-exécuter les consommateurs de production réels (script(s) et/ou outil(s) listés en Subtask 1.4) sur un cas réel, avec arbitrage explicite d'Aymeric sur la profondeur (AC: 2)
  - [ ] Subtask 5.1: Proposer à Aymeric le choix de profondeur — même arbitrage que Stories 2.4/8.9 (voir Dev Notes § Précédents) : (a) inférence seule sur `test_media/`/une vidéo courte existante (léger, suffisant si Task 2-4 sont concluantes), (b) entraînement `JAX_DETECTOR` complet ou partiel (lourd, seulement si Aymeric le juge nécessaire) — ne pas décider seul, ne pas lancer d'entraînement complet sans confirmation explicite (voir Dev Notes § Prudence exécution locale)
  - [ ] Subtask 5.2: Exécuter l'option choisie et documenter le résultat
- [ ] Task 6: Bilan de couverture FR1-FR6 + confirmation que le domaine échecs est entraînable de bout en bout via `Trainer` (AC: 3)
  - [ ] Subtask 6.1: Pour chaque FR1-FR6 (PRD), citer la story et le résultat d'exécution réel qui la prouve — pas relire le texte des stories (même discipline que Story 8.9 Task 6)
  - [ ] Subtask 6.2: "Entraînable de bout en bout via `Trainer`" — `test_trainer_create_train_state_for_chess` (Story 9.3) prouve déjà `Trainer(...).create_train_state()` sur la config `CHESS` réelle (num_channels=29, 382 017 paramètres). Décider avec Aymeric si cette preuve est suffisante, ou si un entraînement réel court (quelques steps, pas une convergence complète) sur les 139 chunks réels (`/home/aobled/Documents/data/chunks/chess/chess_chunk*.npz`, déplacés par Aymeric lui-même après cette session — vérifier leur présence au chemin attendu avant de s'appuyer dessus) doit être exécuté — même prudence que Task 5 (ne pas lancer sans confirmation)
  - [ ] Subtask 6.3: Confirmer explicitement que le pattern `Trainer`/`TaskStrategy` n'a nécessité aucune modification structurelle, à l'exception du seul écart déjà autorisé et documenté (Story 9.3, `trainer.py::self.num_channels`) — citer le diff exact (déjà vérifié Task 1, une seule ligne)

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

### Debug Log References

### Completion Notes List

### File List
