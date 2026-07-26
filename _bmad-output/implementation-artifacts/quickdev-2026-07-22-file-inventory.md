# Inventaire des fichiers `.py` — jax_supervised_training

**Origine** : rétro `quickdev-2026-07-22` (`quickdev-retro-jax-detector-full-frame-2026-07-22.md`), item d'action #1 — "inventorier les ~56 fichiers `.py` (hors `archive/`), valider leur utilité un par un, archiver les scripts de diagnostic ponctuels obsolètes".

**Portée** : 55 fichiers `.py` (56 initialement — voir Historique pour le détail des suppressions/déplacements/ajouts/restaurations), hors `archive/`, `.venv/`, `__pycache__/`, `.claude/`, `_bmad/` (tooling BMAD lui-même).

**Usage** : document vivant, à reprendre au fil des sessions avec l'agent pertinent (Amelia pour l'implémentation, Winston pour les questions d'architecture/frontières de module, Claude en direct sinon). Chaque ligne porte une colonne **Statut** — mettre à jour en place au fil des décisions, pas de réécriture de l'historique. Valeurs possibles : `à trier` (défaut), `garder`, `candidat archivage`, `candidat fusion`, `archivé le AAAA-MM-JJ`.

## Racine (13)

| Fichier | Fonction | Statut |
|---|---|---|
| `bounding_boxes_with_classification_from_video_generation.py` | Driver de production : pipeline vidéo temps réel (JAX Single-Pass), annotation frame par frame, rendu heatmap synthétique | garder |
| `checkpoint_manager.py` | Sauvegarde/chargement des checkpoints, reprise d'entraînement | garder |
| `data_management.py` | Chargeurs de données (`ChunkManager`, `DetectionDataset`, `CenterNetDetectionDataset`), gestion CPU/GPU/TPU pour TensorFlow | garder |
| `dataset_configs.py` | Configuration centralisée de tous les datasets/modèles entraînables (`DATASET_CONFIGS`) | garder |
| `detection_target_encoding.py` | Encodage/décodage des cibles heatmap+taille (CenterNet), source unique de vérité (AD-18) | garder |
| `inference_utils.py` | Module partagé de toutes les fonctions d'inférence JAX/Flax — le plus gros fichier, source unique (AD-1) | garder |
| `loss_functions.py` | Fonctions de perte (grid/YOLO historique, segmentation, CenterNet focal+taille) | à trier |
| `main.py` | Point d'entrée entraînement, orchestration `Trainer`/`TaskStrategy`/`model_library` | garder |
| `model_library.py` | Toutes les architectures de modèles (classification, UNet détection, `AircraftDetectorCenterNet`) | garder |
| `reporting.py` | Rapports/visualisations post-entraînement (courbes, matrices de confusion) | garder |
| `task_strategies.py` | Stratégies d'entraînement par tâche (pattern Strategy : Classification, Detection, CenterNetDetection) | garder |
| `trainer.py` | Classe `Trainer`, boucle d'entraînement générique | garder |
| `utils.py` | Utilitaires JAX purs jittables (mixup, label smoothing, comptage params) | garder |

## `tools/` (10)

| Fichier | Fonction | Statut |
|---|---|---|
| `bounding_boxes_with_classification_from_images_generation.py` | Annotation d'images fixes (2 backends JAX_DETECTOR/FIGHTERJET_DETECTION) | garder |
| `boxes_process_manual_tkinter.py` | Outil GUI Tkinter de correction manuelle de boîtes — refactorisé le 2026-07-24 (1284→1084 lignes), délègue maintenant à `boxes_manual_prediction_assistant.py`/`boxes_manual_json_store.py` | garder |
| `boxes_manual_json_store.py` | Persistance JSON (validation/cohérence/sauvegarde), extrait de `boxes_process_manual_tkinter.py` le 2026-07-24 | garder |
| `boxes_manual_prediction_assistant.py` | Assistant d'inférence JAX (`PredictionAssistant`), extrait de `boxes_process_manual_tkinter.py` le 2026-07-24 | garder |
| `build_fixed_aircraft_dataset.py` | Construction d'un dataset filtré/rééquilibré depuis les JSON | à trier |
| `duplicate_files_find.py` | Exporte en CSV les images en double (même nom) trouvées dans un dossier | garder |
| `duplicate_image_detection_and_normalization.py` | Détection de doublons par hash perceptuel + normalisation JSON — **restauré le 2026-07-24** (supprimé par erreur le 2026-07-22, en réalité utile et non remplaçable par les outils existants) | garder |
| `inspect_pickle.py` | Debug : inspecte les clés/structure d'un checkpoint `.pkl` | à trier |
| `rename_category_in_json_files.py` | Renomme une catégorie dans tous les JSON d'un répertoire | à trier |
| `reporting_dataset_pandas.py` | Analyses pandas du dataset (tailles de boîtes, comptages) — étendu récemment | garder |

## `tools/audit/` (4) — regroupés par Aymeric le 2026-07-22

Cassés par le déplacement (voir Historique) — **laissés tels quels, Aymeric juge ces audits sans utilité réelle désormais** (2026-07-22). Candidats naturels pour un archivage complet plus tard plutôt qu'une correction.

| Fichier | Fonction | Statut |
|---|---|---|
| `audit_dataset_classification.py` | Audit précision du classifieur `FIGHTERJET_CLASSIFICATION` | candidat archivage (cassé, non prioritaire) |
| `audit_dataset_detection_jax.py` | Audit IoU `JAX_DETECTOR` (Single-Pass) — indépendant du suivant | candidat archivage (cassé, non prioritaire) |
| `audit_dataset_detection.py` | Audit IoU ancien pipeline `FIGHTERJET_DETECTION` (AD-20 protégé) | candidat archivage (cassé, non prioritaire) |
| `audit_dataset_results_pandas.py` | Lecture/synthèse texte d'un CSV d'audit déjà généré | candidat archivage (cherche son CSV au mauvais endroit, non prioritaire) |

## `tools/convert/` (6) — regroupés par Aymeric le 2026-07-22

| Fichier | Fonction | Statut |
|---|---|---|
| `convert_aeroscan_aircraft_to_json.py` | Convertisseur dataset externe AeroScan (YOLO) → JSON interne — rejoint le groupe le 2026-07-24 | à trier |
| `convert_HRPlanesv2_aircraft_to_json.py` | Convertisseur dataset externe HRPlanesv2 (YOLO) → JSON interne | à trier |
| `convert_Military_Aircraft_Detection_Dataset.py` | Convertisseur Military Aircraft Detection (CSV) → organisation par classe | à trier |
| `convert_Military_Aircraft_Detection_Dataset_Yolo.py` | Variante YOLO du convertisseur précédent | à trier |
| `convert_military_aircraft_to_json.py` | Convertisseur Military Aircraft (autre source/variante) → JSON interne | à trier |
| `convert_yolo8_to_json.py` | Convertisseur Air Military Vehicle Dataset (YOLO8) → JSON interne | à trier |

## `tests/` (14)

| Fichier | Fonction | Statut |
|---|---|---|
| `diagnose_single_full_size_aircraft.py` | Diagnostic ponctuel (2026-07-19) ancien/nouveau pipeline sur avions plein-cadre — investigation, pas un test de non-régression | candidat archivage (problème résolu depuis) |
| `test_aircraft_detector_centernet.py` | Test du modèle `AircraftDetectorCenterNet` (Story 7.2) | garder |
| `test_centernet_detection_dataset.py` | Test `CenterNetDetectionDataset` (Story 7.5) | garder |
| `test_centernet_detection_strategy.py` | Test `CenterNetDetectionStrategy` (Story 7.6) | garder |
| `test_centernet_loss.py` | Test des pertes CenterNet (Story 7.3) | garder |
| `test_detection_target_encoding.py` | Test round-trip encode/decode heatmap+taille (Story 7.1) | garder |
| `test_detector_inference_composition.py` | Test composition resize+détecteur (Story 8.2) | garder |
| `test_differentiable_crop_classification.py` | Test crop différentiable + classification (Story 8.5) | garder |
| `test_jax_detector_config.py` | Test cohérence config `JAX_DETECTOR` (Story 7.7) | garder |
| `test_jax_detector_dataset_tools.py` | Test génération dataset heatmap+taille (Story 7.4) | garder |
| `test_peak_extraction_topk.py` | Test extraction pics+Top-K JAX-natif (Story 8.3) | garder |
| `test_pixel_parity.py` | Test parité pixel JAX vs PIL/cv2 (Story 8.1) | garder |
| `test_rescale_boxes.py` | Test inverse exact RESCALE vs RESIZE (Story 8.4) | garder |
| `test_single_pass_predict_fn.py` | Test bout-en-bout `build_single_pass_predict_fn` (Story 8.6) | garder |

## `dataset_builder/` (5)

| Fichier | Fonction | Statut |
|---|---|---|
| `cifar10_classification_dataset_tools.py` | Téléchargement/préparation CIFAR10 en chunks | garder |
| `fighterjet_classification_dataset_tools.py` | Préparation dataset classification (crops depuis annotations) | garder |
| `fighterjet_detection_dataset_tools.py` | Préparation dataset détection ancien format (masques), AD-20 protégé | garder |
| `jax_detector_dataset_tools.py` | Préparation dataset détection nouveau format (heatmap+taille CenterNet) | garder |
| `kepler_dataset_tools.py` | Préparation dataset Kepler (exoplanètes) — sans lien avec l'aviation ; **correction 2026-07-24** : était listé à tort sous `tools/` dans l'inventaire initial, il vivait déjà ici | candidat archivage |

## `_bmad-output/implementation-artifacts/baseline/` (3)

| Fichier | Fonction | Statut |
|---|---|---|
| `capture_baseline.py` | Capture la baseline de non-régression avant/après refactor Epic 1 (Story 1.1) | candidat archivage (Epic 1 clos) |
| `verify_after_migration.py` | Diff de non-régression après migration complète (Story 1.10) | candidat archivage (Epic 1 clos) |
| `verify_inference_utils.py` | Vérification isolée de `inference_utils.py` contre la baseline (Story 1.2) | candidat archivage (Epic 1 clos) |

## Notes

- **6 convertisseurs `convert_*_to_json.py`** — scripts d'ingestion one-shot par source de dataset externe. À garder seulement si de nouvelles sources externes sont encore attendues ; sinon, candidats archivage en bloc.
- Aucun statut n'est définitif ici — `à trier` et `candidat *` sont des propositions à valider, pas des décisions prises.

## Historique

- 2026-07-22 : premier inventaire (Claude), 56 fichiers recensés et décrits.
- 2026-07-22 : Aymeric supprime `duplicate_image_detection_and_normalization.py` et `find_and_deduplicate.py` (redondants avec `duplicate_files_find.py`, jugés inutiles). `duplicate_files_find.py` conservé — son rôle réel est d'exporter en CSV la liste des doublons, pas de les supprimer/déplacer automatiquement comme les deux autres. Plus de candidat fusion sur ce groupe : le tri est fait. 56 → 54 fichiers.
- 2026-07-22 : Aymeric supprime `detect_non_free_images.py` (fait par Claude, `git rm`, fichier suivi par git donc récupérable si besoin). Aymeric regroupe manuellement les convertisseurs et les scripts d'audit dans deux nouveaux sous-dossiers, `tools/audit/` et `tools/convert/`, pour ne pas les perdre sans les garder à la racine de `tools/`. `convert_aeroscan_aircraft_to_json.py` n'a pas suivi le mouvement (reste à la racine de `tools/`) — à vérifier si volontaire. 54 → 53 fichiers.
  - **Vérifié (pas juste supposé)** : le déplacement casse `audit_dataset_classification.py`, `audit_dataset_detection.py` et `audit_dataset_detection_jax.py`. Les trois calculent la racine du repo avec `os.path.dirname(os.path.dirname(os.path.abspath(__file__)))` (2 niveaux) — correct depuis `tools/X.py`, mais résout maintenant vers `tools/` au lieu de la racine depuis `tools/audit/X.py` (3 niveaux réels). Confirmé en exécutant `audit_dataset_classification.py` : `ModuleNotFoundError: No module named 'dataset_configs'`, échec immédiat à l'import, avant tout calcul. Les deux autres partagent l'exacte même ligne, cassés pour la même raison (non ré-exécutés individuellement pour épargner un calcul détecteur/checkpoint inutile, mais la cause est identique et vérifiable par lecture).
  - `audit_dataset_results_pandas.py` ne plante pas (aucun import projet, aucun `sys.path`) mais résout son CSV par défaut (`audit_results.csv`) via le même calcul `script_dir` — cherche maintenant dans `tools/audit/` au lieu de `tools/`, silencieusement absent plutôt qu'une erreur explicite.
  - `tools/convert/*.py` non affectés — aucun de ces 5 scripts n'importe quoi que ce soit du projet (uniquement stdlib + PIL), le déplacement est sans risque pour eux.
  - **Décision d'Aymeric (2026-07-22)** : ne pas corriger — ces 4 audits n'ont plus d'utilité réelle. Restent cassés/mal résolus intentionnellement, candidats à un archivage complet plutôt qu'à une réparation. Le correctif serait trivial si un jour l'un d'eux redevient utile (3 niveaux de `dirname()` au lieu de 2, même ligne dans les 4 fichiers).
- 2026-07-24 : refactor de `boxes_process_manual_tkinter.py` (session dédiée, voir `refactor-boxes-process-manual-tkinter.md`) — 2 nouveaux fichiers ajoutés à l'inventaire (`boxes_manual_json_store.py`, `boxes_manual_prediction_assistant.py`). Aymeric supprime `move_excess_to_detection.py` (plus d'utilité) et déplace `convert_aeroscan_aircraft_to_json.py` vers `tools/convert/` (rejoint enfin le groupe, résout la remarque "non déplacé, à vérifier si volontaire" du 2026-07-22). **Correction au passage** : `kepler_dataset_tools.py` était listé par erreur sous `tools/` dans l'inventaire initial — vérifié via `git log`, il vivait déjà dans `dataset_builder/` avant même le début de cet inventaire, jamais déplacé depuis. Corrigé ici, pas un nouveau mouvement. 53 → 54 fichiers.
- 2026-07-24 : **retour en arrière** sur la suppression du 2026-07-22 de `duplicate_image_detection_and_normalization.py` — Aymeric : "j'ai été trop confiant en le supprimant, c'est utile et ne peut être remplacé par les programmes existants" (contrairement à `find_and_deduplicate.py`, dont la suppression reste valide). Fichier restauré avec `git checkout a639c12^ -- tools/duplicate_image_detection_and_normalization.py` (`a639c12` = commit qui l'avait supprimé ; le fichier n'était déjà plus modifié dans l'arbre de travail, la suppression était déjà committée). Restauré dans l'index (`git status` : `A`), pas encore commité — à Aymeric de décider. 54 → 55 fichiers.
