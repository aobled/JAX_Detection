---
stepsCompleted: [1, 2, 3]
inputDocuments:
  - _bmad-output/planning-artifacts/prds/prd-JAX_Detection-2026-07-12/prd.md
  - _bmad-output/planning-artifacts/architecture/architecture-JAX_Detection-2026-07-12/ARCHITECTURE-SPINE.md
  - docs/dead-code-and-duplication-audit.md
  - docs/architecture.md
  - docs/source-tree-analysis.md
  - _bmad-output/planning-artifacts/prds/prd-JAX_Detection-2026-07-14/prd.md
  - _bmad-output/specs/spec-jax-single-pass/SPEC.md
  - _bmad-output/planning-artifacts/architecture/architecture-jax_supervised_training-2026-07-15/ARCHITECTURE-SPINE.md
  - _bmad-output/planning-artifacts/prds/prd-jax_supervised_training-2026-07-27/prd.md
  - _bmad-output/planning-artifacts/architecture/architecture-chess-2026-07-27/ARCHITECTURE-SPINE.md
---

# Refactor jax_supervised_training — Epic Breakdown

## Overview

This document provides the complete epic and story breakdown for the jax_supervised_training refactor (nettoyage code mort et duplications), decomposing the requirements from the PRD and the Architecture Spine into implementable stories. No UX design contract applies — this is an internal code refactor with no user-facing interface changes.

## Requirements Inventory

### Functional Requirements

FR1: Un module partagé unique (`inference_utils.py`) contient les implémentations canoniques de `load_jax_model`, `load_detection_model`, `predict_crop`, `predict_crops_batch`, `build_predict_fn`, `build_clf_predict_fn`, `get_iou`, `non_max_suppression`, `_preprocess_crop_to_hwc`, `decode_segmentation_and_detect`, `decode_segmentation_and_detect_batch` — 11 fonctions au total.
FR2: Les 5 fichiers du scope initial (`bounding_boxes_with_classification_from_video_generation.py`, `tools/bounding_boxes_with_classification_from_images_generation.py`, `heatmap_generation.py`, `heatmap_contouring.py`, `tools/audit_dataset_classification.py`) importent depuis ce module au lieu de redéfinir localement. Voir FR10 pour 2 fichiers supplémentaires découverts en phase Architecture.
FR3: Pour chaque fonction ayant des comportements divergents entre fichiers, un comportement canonique est choisi explicitement avant fusion (pas de fusion mécanique aveugle). Quand la divergence reflète un vrai compromis (ex. temps réel vs précision), les deux comportements peuvent coexister comme fonctions distinctes plutôt que d'être forcés en un seul.
FR4: `dataset_configs.py` ne conserve que `FIGHTERJET_CLASSIFICATION`, `FIGHTERJET_DETECTION`, `JAX_KEPLER`. Suppression de `FIGHTERJET_VIT`, `FIGHTERJET_HYBRID_VIT`, `FIGHTERJET_LETTERBOX`, `FIGHTERJET_DETECTION_SOPHISTICATED`.
FR5: `model_library.py` ne conserve que les architectures référencées par les 3 configs restantes : `sophisticated_cnn_128_plus`, `aircraft_detector_unet`, `aircraft_detector_miniunet`, `kepler_1d_cnn`. Les 18 autres architectures de `MODELS` sont supprimées.
FR6: Avant suppression d'une architecture, vérifier qu'aucun `.pkl` actuellement versionné (`best_model.pkl`, `best_model_detection.pkl`) n'en dépend.
FR7: `train_detection.py` est supprimé (vestige confirmé de la fusion historique jax_supervised_training/JAX_Classification).
FR8: `tokens/` est supprimé intégralement (expérience token-based model vs UNet abandonnée, jamais raccordée au pipeline actif).
FR9: `bounding_boxes_with_classification_from_benchmark.py` est supprimé intégralement (et non fusionné dans le module partagé). Sa logique de décodage de détection (`decode_grid_and_detect`) est spécifique aux architectures grid-based supprimées par FR5.
FR10: `tools/audit_dataset_detection.py` et `tools/boxes_process_manual_tkinter.py` — des consommateurs invisibles au scope initial, qui importent aujourd'hui des fonctions directement depuis l'espace de noms de `bounding_boxes_with_classification_from_video_generation.py` — sont repointés vers `inference_utils.py` dans ce même cycle.

### NonFunctional Requirements

NFR1 — Généricité : le pipeline doit rester utilisable pour une tâche non liée à la détection/classification d'avions sans duplication de code (validé par `JAX_KEPLER`).
NFR2 — Modularité : le pattern Strategy/Factory/Dependency Injection (`task_strategies.py`, `model_library.get_model()`, `Trainer`) n'est pas cassé par le refactor.
NFR3 — Performance : aucune régression de performance d'entraînement ou d'inférence introduite par la mutualisation du code, mesurée par temps/epoch (entraînement) et latence/image (inférence) captés avant/après refactor.
NFR4 — Réversibilité : toute suppression (modèles, `tokens/`, `train_detection.py`) reste récupérable via l'historique git.

### Additional Requirements

- **AD-1 (FR1, FR2, FR10)** : `inference_utils.py` est la source unique des 11 fonctions d'inférence — aucune redéfinition locale n'est autorisée, y compris sous un autre nom faisant le même travail. `load_classification_model` n'est pas migré comme 12ᵉ fonction : sa composition (`load_jax_model` + `build_predict_fn`) devient explicite au site d'appel.
- **AD-2 (FR1, FR3, NFR3) `[ADOPTED]`** : `predict_crop` et `predict_crops_batch` sont deux implémentations pleinement indépendantes (pas de délégation interne). `_CLF_BATCH_SIZE = 32` devient une constante privée de `inference_utils.py`.
- **AD-3 (FR1, FR3, NFR4) `[ADOPTED]`** : `load_detection_model` conserve le fallback de résolution de chemin à 3 niveaux **et** la ré-initialisation des `batch_stats` manquants. `DETECTION_IMAGE_SIZE = (224, 224)` devient une constante privée.
- **AD-4 (FR1, FR5, FR6)** : tout fallback `model_name` par défaut dans `inference_utils.py` pointe vers `aircraft_detector_unet` (vivant), jamais vers un modèle supprimé par FR5. FR6 reste un gate bloquant à ré-exécuter avant la suppression effective, pas une simple note informative.
- **AD-5 (FR2, FR5, FR9)** : `bounding_boxes_with_classification_from_benchmark.py` est supprimé intégralement, pas migré/réconcilié vers `inference_utils.py`. Confirmé : aucun autre fichier du repo n'importe depuis lui.
- **AD-6 (FR1, FR3, NFR3) `[ADOPTED]`** : `decode_segmentation_and_detect` et `decode_segmentation_and_detect_batch` restent deux fonctions distinctes et indépendantes, chacune ratifiant son implémentation actuelle sans changement de comportement. Le pipeline vidéo (`_batch`) est prioritaire : aucun changement futur ne doit dégrader son débit temps réel au nom d'une unification.
- **AD-7 (FR1, FR2, FR10)** : Séquencement — une story dédiée ("Story 0") crée `inference_utils.py` avec les 11 fonctions canonisées et leurs corps définitifs, **avant** que toute story de migration de fichier consommateur ne démarre. Les stories suivantes ne font qu'importer ; aucune n'édite `inference_utils.py`.
- **AD-8 (FR2, FR10)** : Périmètre étendu aux consommateurs invisibles — `tools/audit_dataset_detection.py` et `tools/boxes_process_manual_tkinter.py` sont ajoutés à la liste des consommateurs de `inference_utils.py`, imports repointés dans le même cycle (pas de shim de compatibilité).
- **Convention format des boxes** : `[x1, y1, x2, y2, score]`, format liste (pas numpy vectorisé) — seul survivant après AD-5.
- **Convention checkpoints** : tout chargement de checkpoint applique le fallback de chemin 3 niveaux + ré-init des `batch_stats` manquants (AD-3) — jamais un chargement "nu" qui échoue silencieusement.
- **Convention performance** : en cas d'arbitrage entre le pipeline vidéo temps réel et un autre consommateur, le pipeline vidéo est prioritaire (AD-6, NFR3).
- **Stack** : aucune nouvelle dépendance externe introduite — `inference_utils.py` réutilise exclusivement JAX/Flax/NumPy/OpenCV déjà présents dans le projet.
- **Non-Goal rappelé (PRD)** : pas de suite de tests automatisée ni de CI/CD dans ce cycle — la méthode de validation est la comparaison baseline/diff manuelle (Goal 1 du PRD), pas une suite de tests. Ceci renforce l'importance d'AD-7 (auteur unique de `inference_utils.py`).

### UX Design Requirements

N/A — aucun contrat UX applicable. Ce refactor est un chantier interne sans changement d'interface utilisateur (pas de script CLI, GUI ou API publique modifié dans son comportement observable).

### FR Coverage Map

FR1:  Epic 1 - Module partagé inference_utils.py (11 fonctions canoniques)
FR2:  Epic 1 - Migration des 5 fichiers consommateurs du scope initial
FR3:  Epic 1 - Réconciliation des comportements divergents (double API si compromis réel)
FR4:  Epic 2 - Purge dataset_configs.py (7→3 configs)
FR5:  Epic 2 - Purge model_library.py (22→4 architectures)
FR6:  Epic 2 - Gate de vérification .pkl avant suppression
FR7:  Epic 3 - Suppression train_detection.py
FR8:  Epic 3 - Suppression tokens/
FR9:  Epic 3 - Suppression bounding_boxes_with_classification_from_benchmark.py
FR10: Epic 1 - Migration des 2 consommateurs invisibles (audit_dataset_detection.py, boxes_process_manual_tkinter.py)

## Epic List

### Epic 1: Mutualisation du code d'inférence

Le codebase dispose d'un point unique de vérité pour la logique d'inférence transverse — plus aucune fonction critique (chargement de modèle, prétraitement, prédiction, décodage, NMS/IoU) n'est redéfinie-et-divergente dans plusieurs fichiers. Une baseline de non-régression (boxes/classes/scores) est capturée avant et diffée après, sur les 7 fichiers concernés.
**FRs covered:** FR1, FR2, FR3, FR10
**Note d'implémentation :** contient une "Story 0" obligatoire (AD-7) qui crée `inference_utils.py` avec ses 11 fonctions canoniques *avant* toute story de migration des 7 fichiers consommateurs — celles-ci ne font qu'importer, aucune n'édite le module partagé.

### Epic 2: Purge des configs et modèles morts

`dataset_configs.py` et `model_library.py` ne contiennent plus que les configs/architectures réellement utilisées par le pipeline (3 configs, 4 architectures), sans risque de casser le chargement des checkpoints existants ni un entraînement complet.
**FRs covered:** FR4, FR5, FR6
**Note d'implémentation :** FR6 (vérification `.pkl`) est un gate bloquant à exécuter avant la suppression effective des architectures (FR5) — pas une story indépendante après coup.

### Epic 3: Suppression du code orphelin

Les vestiges confirmés (fusion historique, expérimentation abandonnée, script de benchmark devenu obsolète après AD-5) sont retirés du dépôt actif, sans perte d'information (récupérables via git).
**FRs covered:** FR7, FR8, FR9

## Epic 1: Mutualisation du code d'inférence

Le codebase dispose d'un point unique de vérité pour la logique d'inférence transverse — plus aucune fonction critique (chargement de modèle, prétraitement, prédiction, décodage, NMS/IoU) n'est redéfinie-et-divergente dans plusieurs fichiers. Une baseline de non-régression (boxes/classes/scores) est capturée avant et diffée après, sur les 7 fichiers concernés. **FRs covered:** FR1, FR2, FR3, FR10

### Story 1.1: Capture de la baseline de non-régression

As a mainteneur du pipeline d'inférence,
I want une baseline capturée (boxes/classes/scores) sur un petit set d'images fixes, pour les 7 fichiers concernés par la duplication,
So that je peux prouver par diff, pas par observation, l'absence de régression après le refactor (Goal 1 PRD).

**Acceptance Criteria:**

**Given** les 7 fichiers concernés (`bounding_boxes_with_classification_from_video_generation.py`, `tools/bounding_boxes_with_classification_from_images_generation.py`, `heatmap_generation.py`, `heatmap_contouring.py`, `tools/audit_dataset_classification.py`, `tools/audit_dataset_detection.py`, `tools/boxes_process_manual_tkinter.py`) dans leur état actuel (avant toute modification)
**When** j'exécute chacun sur un jeu d'images fixe représentatif (couvrant au moins un cas classification et un cas détection)
**Then** les sorties (boxes, classes, scores) sont capturées et sauvegardées dans un format comparable (JSON/CSV) associé à un identifiant de version baseline
**And** le jeu d'images utilisé est documenté (chemin, nombre d'images, critère de sélection) pour être réutilisé à l'identique en Story 1.10

**Given** `tools/boxes_process_manual_tkinter.py` a un usage interactif (GUI Tkinter)
**When** la capture automatique n'est pas possible pour ce fichier
**Then** une méthode de capture alternative (appel direct aux fonctions internes exposées, ou capture manuelle documentée) est utilisée et le choix est noté dans la baseline

### Story 1.2: Création de inference_utils.py (Story 0, AD-7)

As a mainteneur du pipeline d'inférence,
I want un module partagé inference_utils.py contenant les 11 fonctions canoniques d'inférence avec leurs comportements arbitrés,
So that toute la logique d'inférence dupliquée peut ensuite être remplacée par un import unique, sans ambiguïté sur quel comportement fait foi.

**Acceptance Criteria:**

**Given** le besoin de mutualiser les fonctions d'inférence (FR1, AD-1)
**When** `inference_utils.py` est créé
**Then** il contient exactement les 11 fonctions suivantes avec le comportement canonique arbitré : `load_jax_model`, `load_detection_model`, `predict_crop`, `predict_crops_batch`, `build_predict_fn`, `build_clf_predict_fn`, `get_iou`, `non_max_suppression`, `_preprocess_crop_to_hwc`, `decode_segmentation_and_detect`, `decode_segmentation_and_detect_batch`
**And** aucune 12ᵉ fonction (`load_classification_model`) n'est ajoutée — la composition `load_jax_model` + `build_predict_fn` remplace ce besoin au site d'appel

**Given** `predict_crop` et `predict_crops_batch` (AD-2)
**When** elles sont implémentées
**Then** elles sont deux implémentations pleinement indépendantes (`predict_crop` n'appelle pas `predict_crops_batch` en interne)
**And** `_CLF_BATCH_SIZE = 32` est définie comme constante privée du module

**Given** `load_detection_model` (AD-3)
**When** elle est implémentée
**Then** elle conserve le fallback de résolution de chemin à 3 niveaux (CWD → parent du CWD → parent du fichier appelant) et la ré-initialisation des `batch_stats` manquants
**And** `DETECTION_IMAGE_SIZE = (224, 224)` est définie comme constante privée du module

**Given** tout fallback `model_name` par défaut (AD-4)
**When** il est déclenché sur un checkpoint sans métadonnées `config`
**Then** il pointe vers `aircraft_detector_unet`, jamais vers un modèle supprimé par FR5

**Given** `decode_segmentation_and_detect` et `decode_segmentation_and_detect_batch` (AD-6)
**When** elles sont implémentées
**Then** elles restent deux fonctions distinctes et indépendantes, chacune ratifiant son implémentation actuelle sans changement de comportement

**Given** la convention de format des boxes
**When** une fonction retourne des boxes
**Then** le format est `[x1, y1, x2, y2, score]` en liste Python (pas de tableau numpy vectorisé)

**Given** AD-7 (auteur unique)
**When** cette story est complétée
**Then** aucune story suivante ne modifie `inference_utils.py` pour y ajouter/changer une fonction — seules des imports sont faits par les stories 1.3 à 1.9

**Given** FR6 (gate de sécurité, référencé par AD-4)
**When** le fallback `model_name` par défaut est fixé
**Then** il a été vérifié au préalable que `best_model.pkl` (`sophisticated_cnn_128_plus`) et `best_model_detection.pkl` (`aircraft_detector_unet`) ne dépendent d'aucune architecture supprimée

### Story 1.3: Migration bounding_boxes_with_classification_from_video_generation.py

As a mainteneur du pipeline vidéo,
I want que bounding_boxes_with_classification_from_video_generation.py importe ses fonctions d'inférence depuis inference_utils.py au lieu de les redéfinir localement,
So that le pipeline vidéo temps réel utilise la même logique canonique que les autres consommateurs, sans duplication ni divergence.

**Acceptance Criteria:**

**Given** la Story 1.2 complétée
**When** le fichier est migré
**Then** il importe `load_jax_model`, `load_detection_model`, `predict_crops_batch`, `build_predict_fn`/`build_clf_predict_fn`, `get_iou`, `non_max_suppression`, `decode_segmentation_and_detect_batch` depuis `inference_utils.py`
**And** les définitions locales de ces fonctions sont supprimées du fichier

**Given** `predict_crops_batch` (AD-2, nom de paramètre `predict_fn` ratifié depuis ce fichier)
**When** le fichier est migré
**Then** l'appel à `predict_crops_batch` utilise la signature canonique sans changement d'usage côté appelant

**Given** le pipeline vidéo prioritaire en performance (AD-6, NFR3)
**When** la migration est effectuée
**Then** aucune dégradation du débit temps réel n'est introduite (comparaison avant/après sur un extrait vidéo)

**Given** la baseline capturée en Story 1.1 pour ce fichier
**When** le fichier migré est exécuté sur le même jeu d'images/vidéo
**Then** les boxes/classes/scores produits sont identiques à la baseline

### Story 1.4: Migration tools/bounding_boxes_with_classification_from_images_generation.py

As a mainteneur des outils d'inférence sur images,
I want que ce script importe ses fonctions d'inférence depuis inference_utils.py,
So that le traitement d'image statique utilise la même logique canonique que le pipeline vidéo.

**Acceptance Criteria:**

**Given** la Story 1.2 complétée
**When** le fichier est migré
**Then** il importe `predict_crop` (ratifié depuis ce fichier, ligne ~128, AD-2), `load_jax_model`, `load_detection_model`, `get_iou`, `non_max_suppression`, `decode_segmentation_and_detect` depuis `inference_utils.py`
**And** les définitions locales sont supprimées

**Given** `predict_crop` (AD-2, signature `(crop_img, model, variables, mean, std, config)`)
**When** le fichier est migré
**Then** aucun appel n'utilise l'ancienne signature `(crop_img, predict_fn, mean, std, config)` de `video_generation.py`, confirmée morte et non migrée

**Given** la baseline Story 1.1
**When** le script migré est exécuté sur le même jeu d'images
**Then** les résultats sont identiques à la baseline

### Story 1.5: Migration heatmap_generation.py

As a mainteneur de la génération de heatmaps,
I want que heatmap_generation.py importe load_detection_model depuis inference_utils.py,
So that le chargement de modèle bénéficie du fallback robuste (AD-3) au lieu de l'échec silencieux actuel.

**Acceptance Criteria:**

**Given** la Story 1.2 complétée
**When** `heatmap_generation.py` est migré
**Then** il importe `load_detection_model` depuis `inference_utils.py` et supprime sa propre définition locale (qui n'a aujourd'hui aucun fallback ni ré-init de `batch_stats`)

**Given** AD-3 (robustesse uniformisée)
**When** un checkpoint sans `batch_stats` sauvegardés est chargé via ce fichier migré
**Then** la ré-initialisation de structure s'applique au lieu d'un comportement silencieusement dégradé

**Given** la baseline Story 1.1
**When** le script migré est exécuté sur le même jeu d'images
**Then** les heatmaps produites sont identiques à la baseline (hors correction explicite du bug de robustesse AD-3, à documenter si déclenché)

### Story 1.6: Migration heatmap_contouring.py

As a mainteneur de la génération de heatmaps avec contours,
I want que heatmap_contouring.py importe load_detection_model depuis inference_utils.py,
So that le chargement de modèle bénéficie du fallback robuste (AD-3) au lieu de l'échec silencieux actuel.

**Acceptance Criteria:**

**Given** la Story 1.2 complétée
**When** `heatmap_contouring.py` est migré
**Then** il importe `load_detection_model` depuis `inference_utils.py` et supprime sa propre définition locale

**Given** AD-3
**When** un checkpoint sans `batch_stats` sauvegardés est chargé via ce fichier migré
**Then** la ré-initialisation de structure s'applique

**Given** la baseline Story 1.1
**When** le script migré est exécuté sur le même jeu d'images
**Then** les résultats sont identiques à la baseline (hors correction explicite du bug AD-3, à documenter si déclenché)

### Story 1.7: Migration tools/audit_dataset_classification.py

As a mainteneur des outils d'audit de dataset,
I want que tools/audit_dataset_classification.py compose load_jax_model + build_predict_fn au lieu de sa fonction locale load_classification_model,
So that le chargement de modèle et la construction du predict_fn utilisent la logique canonique partagée (AD-1).

**Acceptance Criteria:**

**Given** la Story 1.2 complétée
**When** `tools/audit_dataset_classification.py` est migré
**Then** sa fonction locale `load_classification_model` est supprimée et remplacée par l'appel explicite : `model, variables, mean, std = load_jax_model(...)` puis `predict_fn = build_predict_fn(model, variables)`
**And** `_preprocess_crop_to_hwc` est importée depuis `inference_utils.py` au lieu d'être redéfinie localement

**Given** la baseline Story 1.1
**When** le script migré est exécuté sur le même jeu de données d'audit
**Then** les résultats d'audit sont identiques à la baseline

### Story 1.8: Migration tools/audit_dataset_detection.py

As a mainteneur des outils d'audit de détection,
I want que tools/audit_dataset_detection.py importe ses fonctions d'inférence depuis inference_utils.py au lieu de l'espace de noms de bounding_boxes_with_classification_from_video_generation.py,
So that ce consommateur invisible ne dépende plus d'un fichier qui va lui-même être migré, évitant une casse silencieuse (FR10, AD-8).

**Acceptance Criteria:**

**Given** la Story 1.2 complétée
**When** `tools/audit_dataset_detection.py` est migré
**Then** ses imports actuels depuis `bounding_boxes_with_classification_from_video_generation.py` (`load_detection_model`, `get_iou`, `decode_segmentation_and_detect_batch`, `load_jax_model`, `predict_crops_batch` selon usage réel) sont repointés vers `inference_utils.py`
**And** aucun shim de compatibilité n'est laissé en place

**Given** la baseline Story 1.1
**When** le script migré est exécuté
**Then** les résultats d'audit sont identiques à la baseline

### Story 1.9: Migration tools/boxes_process_manual_tkinter.py

As a utilisateur de l'éditeur manuel de bounding boxes,
I want que tools/boxes_process_manual_tkinter.py importe ses fonctions d'inférence depuis inference_utils.py au lieu de l'espace de noms de bounding_boxes_with_classification_from_video_generation.py,
So that l'outil d'annotation manuelle continue de fonctionner correctement même après la migration du fichier vidéo dont il dépend actuellement (FR10, AD-8).

**Acceptance Criteria:**

**Given** la Story 1.2 complétée
**When** `tools/boxes_process_manual_tkinter.py` est migré
**Then** ses imports actuels depuis `bounding_boxes_with_classification_from_video_generation.py` sont repointés vers `inference_utils.py`
**And** le comportement de l'interface Tkinter (édition manuelle de boxes) reste inchangé

**Given** la baseline Story 1.1 (capture alternative documentée pour ce fichier interactif)
**When** l'outil migré est utilisé sur le même scénario de test
**Then** le comportement observé est identique à la baseline

### Story 1.10: Diff de non-régression (après migration complète)

As a mainteneur du pipeline d'inférence,
I want comparer les sorties post-migration des 7 fichiers concernés à la baseline capturée en Story 1.1,
So that la non-régression fonctionnelle de l'Epic 1 est prouvée par comparaison, pas par observation (Goal 1 PRD).

**Acceptance Criteria:**

**Given** la baseline capturée en Story 1.1 et les 7 fichiers migrés (Stories 1.3 à 1.9)
**When** chaque fichier est ré-exécuté sur le même jeu d'images/scénario que la baseline
**Then** les sorties (boxes, classes, scores) sont diffées automatiquement contre la baseline
**And** tout écart est soit nul, soit explicitement documenté et justifié (ex. correction de bug AD-3)

**Given** les Goals 2 et 3 du PRD (entraînement complet, régénération datasets)
**When** cette story est complétée
**Then** `main.py` est exécuté sans régression sur `FIGHTERJET_CLASSIFICATION` et `FIGHTERJET_DETECTION`

## Epic 2: Purge des configs et modèles morts

`dataset_configs.py` et `model_library.py` ne contiennent plus que les configs/architectures réellement utilisées par le pipeline (3 configs, 4 architectures), sans risque de casser le chargement des checkpoints existants ni un entraînement complet. **FRs covered:** FR4, FR5, FR6

### Story 2.1: Gate FR6 — vérification des dépendances .pkl

As a mainteneur du pipeline d'entraînement,
I want vérifier explicitement qu'aucun .pkl actuellement versionné (best_model.pkl, best_model_detection.pkl) ne dépend d'une architecture qui sera supprimée,
So that la suppression des architectures mortes (FR5) ne casse pas le chargement des checkpoints existants.

**Acceptance Criteria:**

**Given** `best_model.pkl` et `best_model_detection.pkl` versionnés dans le dépôt
**When** leur métadonnée de config/architecture est inspectée
**Then** `best_model.pkl` est confirmé lié à `sophisticated_cnn_128_plus` et `best_model_detection.pkl` à `aircraft_detector_unet` (architectures vivantes après FR5, cf. AD-4)
**And** si l'un des deux dépendait d'une architecture candidate à suppression, cette architecture est retirée de la liste de suppression de FR5 et documentée comme exception

**Given** AD-4 (gate bloquant, pas une note informative)
**When** cette story est complétée avec succès
**Then** elle constitue la condition préalable explicite au démarrage de la Story 2.3

### Story 2.2: Purge dataset_configs.py

As a mainteneur du fichier de configuration,
I want ne conserver que FIGHTERJET_CLASSIFICATION, FIGHTERJET_DETECTION, JAX_KEPLER dans dataset_configs.py,
So that le fichier de config ne référence plus de scénarios d'entraînement abandonnés.

**Acceptance Criteria:**

**Given** `dataset_configs.py` avec 7 configs actuelles
**When** le fichier est purgé
**Then** `FIGHTERJET_VIT`, `FIGHTERJET_HYBRID_VIT`, `FIGHTERJET_LETTERBOX`, `FIGHTERJET_DETECTION_SOPHISTICATED` sont supprimées
**And** `FIGHTERJET_CLASSIFICATION`, `FIGHTERJET_DETECTION`, `JAX_KEPLER` restent inchangées

**Given** qu'aucun autre fichier du repo ne référence les 4 configs supprimées par leur nom littéral
**When** la purge est effectuée
**Then** aucune régression d'import n'est introduite (vérification par recherche textuelle dans le repo avant suppression)

### Story 2.3: Purge model_library.py

As a mainteneur de la factory de modèles,
I want ne conserver que les 4 architectures référencées par les 3 configs restantes,
So that model_library.py ne porte plus 18 architectures mortes (~68% du fichier).

**Acceptance Criteria:**

**Given** la Story 2.1 complétée (gate FR6 passé) et la Story 2.2 complétée
**When** `model_library.py` est purgé
**Then** seules `sophisticated_cnn_128_plus`, `aircraft_detector_unet`, `aircraft_detector_miniunet`, `kepler_1d_cnn` restent dans `MODELS`
**And** les 18 autres architectures (`aircraft_detector`, `v2`-`v7_advanced`, `aircraft_detector_unet_token`, `sophisticated_cnn`, `sophisticated_cnn_droped_out`, `sophisticated_cnn_128_ultimate`, `sophisticated_cnn_optimized`, `resnet_light`, `tiny_vit_plus`, `tiny_vit_plus_ultimate`, `hybrid_tiny_vit`, `tiny_vit_plus_balanced`, `aircraft_detector_sophisticated_unet`) sont supprimées

**Given** AD-4 (fallback `model_name`)
**When** `get_model()` est appelé sans `model_name` explicite
**Then** le fallback pointe vers `aircraft_detector_unet`

### Story 2.4: Validation post-purge

As a mainteneur du pipeline,
I want confirmer qu'un entraînement complet et la régénération des datasets fonctionnent toujours après la purge,
So that la suppression des configs/modèles morts n'a introduit aucune régression fonctionnelle (Goals 2, 3 PRD).

**Acceptance Criteria:**

**Given** les Stories 2.2 et 2.3 complétées
**When** `main.py` est exécuté sur `FIGHTERJET_CLASSIFICATION` puis `FIGHTERJET_DETECTION`
**Then** l'entraînement tourne sans erreur jusqu'à son terme

**Given** `fighterjet_classification_dataset_tools.py` et `fighterjet_detection_dataset_tools.py`
**When** la régénération des datasets `.npz` est exécutée
**Then** elle fonctionne toujours sans erreur

**Given** `best_model.pkl` et `best_model_detection.pkl`
**When** ils sont rechargés via le pipeline post-purge
**Then** le chargement réussit sans erreur (confirmation finale de la Story 2.1)

## Epic 3: Suppression du code orphelin

Les vestiges confirmés (fusion historique, expérimentation abandonnée, script de benchmark devenu obsolète après AD-5) sont retirés du dépôt actif, sans perte d'information (récupérables via git). **FRs covered:** FR7, FR8, FR9

### Story 3.1: Suppression de train_detection.py

As a mainteneur du pipeline,
I want supprimer train_detection.py, vestige confirmé de la fusion historique jax_supervised_training/JAX_Classification,
So that le point d'entrée du pipeline reste unique et non ambigu (main.py).

**Acceptance Criteria:**

**Given** `train_detection.py` contourne le pattern Strategy actuel et pointe vers `aircraft_detector_v3` (architecture supprimée par FR5)
**When** le fichier est supprimé
**Then** aucun autre fichier du repo ne l'importe (vérifié avant suppression)
**And** le fichier reste récupérable via l'historique git (NFR4)

### Story 3.2: Suppression de tokens/

As a mainteneur du repo,
I want supprimer intégralement le dossier tokens/, expérience token-based model vs UNet abandonnée et jamais raccordée au pipeline actif,
So that j'élimine un dossier au nom trompeur mélangeant des sujets sans rapport.

**Acceptance Criteria:**

**Given** `tokens/` contient `read_token.py`/`write_token.py`, `genetic_algorithm.py`/`genetic_algorithm2.py`, `loss_detection_only_directory.py`/`loss_with_classification_images_directory.py`
**When** le dossier est supprimé intégralement
**Then** aucun autre fichier du repo n'importe depuis `tokens/` (vérifié avant suppression)
**And** le contenu reste récupérable via l'historique git (NFR4)

### Story 3.3: Suppression de bounding_boxes_with_classification_from_benchmark.py

As a mainteneur du code d'inférence,
I want supprimer intégralement bounding_boxes_with_classification_from_benchmark.py sans le fusionner dans inference_utils.py,
So that j'élimine un vestige dont la logique de décodage (decode_grid_and_detect) est spécifique aux architectures grid-based déjà supprimées par FR5.

**Acceptance Criteria:**

**Given** AD-5 (confirmé : aucun autre fichier du repo n'importe depuis lui)
**When** le fichier est supprimé
**Then** il n'est pas migré vers `inference_utils.py` (contrairement aux 7 fichiers de l'Epic 1)
**And** le fichier reste récupérable via l'historique git (NFR4)

**Given** ce fichier faisait initialement partie du scope de duplication détecté (`dead-code-and-duplication-audit.md`)
**When** la suppression est effectuée
**Then** le périmètre des 7 fichiers de l'Epic 1 (Story 1.1) l'exclut bien, cohérent avec le scope révisé du PRD

## Addendum — déviations approuvées post-implémentation

Deux déviations mineures aux AC telles qu'écrites ci-dessus, découvertes et approuvées pendant l'implémentation (voir les fichiers story correspondants dans `_bmad-output/implementation-artifacts/` pour le détail complet) :

- **Story 2.2** dit que les 3 configs survivantes "restent inchangées". En pratique, `FIGHTERJET_DETECTION.output_prefix` a été corrigé (`f"{DATA_ROOT}/chunks/detection"` → `f"{DATA_ROOT}/chunks/detection/dataset_detection"`) — un mismatch pré-existant avec le layout réel des chunks (local et Colab/Drive), révélé par la suppression de `FIGHTERJET_DETECTION_SOPHISTICATED` (dont l'`output_prefix` masquait le problème). Corrigé avec l'accord explicite de l'utilisateur. Voir `2-2-purge-dataset-configs-py.md`.
- **Story 1.3** ne liste pas `DETECTION_IMAGE_SIZE` dans les imports attendus de `bounding_boxes_with_classification_from_video_generation.py`. En pratique, cette constante est importée (au lieu d'être redéfinie localement) pour respecter AD-1 ("aucune redéfinition locale, même sous un autre nom"), plutôt que de dupliquer la valeur `(224, 224)`. Voir `1-3-migration-bounding-boxes-with-classification-from-video-generation-py.md`.

## Epic 4: Nettoyage technique post-refactor

Trois corrections techniques ponctuelles issues du backlog de la rétrospective Epic 1-3 (2026-07-12), traitées avec le même niveau de rigueur (baseline/vérification, dev notes) mais sans cérémonie PRD/Architecture complète — aucune n'introduit de nouvelle décision produit ou d'architecture.
**FRs covered:** aucune (hors scope FR1-FR10 du PRD refactor initial ; items de maintenance identifiés post-cycle)

### Story 4.1: Suppression du code Letterbox résiduel

As a mainteneur du repo,
I want supprimer le code Letterbox devenu obsolète/cassé et corriger la documentation trompeuse restante,
so that le repo ne contienne plus de script non-fonctionnel ni de docstring qui décrit un comportement différent du code réel.

**Acceptance Criteria:**

**Given** `generate_letterbox_dataset.py` importe `process_dataset_letterbox` et `balance_and_split_dataset` depuis `fighterjet_classification_dataset_tools.py`
**When** on vérifie l'existence de ces deux fonctions dans tout le repo
**Then** elles n'existent nulle part — ce script est déjà non-fonctionnel (lèverait une `ImportError` dès sa première ligne exécutée)
**And** le script est supprimé intégralement (récupérable via l'historique git)

**Given** `fighterjet_detection_dataset_tools.py` (docstring de la fonction de traitement détection) affirme "2. Applique Letterbox"
**When** le code réel juste en dessous effectue un "STRETCHED RESIZING (au lieu de Letterbox)"
**Then** le docstring est corrigé pour refléter fidèlement le comportement réel du code

### Story 4.2: Correction des warnings Colab (absl/CUDA)

As a utilisateur lançant l'entraînement sur Colab,
I want ne plus voir de warnings alarmants liés à CUDA/absl au démarrage,
so that je puisse distinguer un vrai problème d'un bruit de log cosmétique.

**Acceptance Criteria:**

**Given** `data_management.py` importe `tensorflow` uniquement pour le pipeline `tf.data` (chargement CPU, aucun calcul GPU requis côté TF — JAX gère seul le calcul GPU pour l'entraînement)
**When** TensorFlow est importé
**Then** TF est explicitement configuré pour ne pas rechercher de GPU (`tf.config.set_visible_devices([], 'GPU')`), rendant ce choix intentionnel plutôt qu'un échec d'initialisation CUDA affiché comme warning ("Could not find cuda drivers", "failed call to cuInit")

**Given** le warning `absl::InitializeLog()` apparaît avant toute sortie applicative
**When** on analyse son origine
**Then** il est documenté comme non-actionnable côté code applicatif — le message indique par construction des logs émis *avant* l'initialisation du logging (déjà en amont de `TF_CPP_MIN_LOG_LEVEL`, déjà positionné, qui ne le supprime pas) ; aucun correctif spéculatif n'est ajouté pour un warning inerte dont la cause est interne à TensorFlow

**Given** cette correction touche l'affichage de logs dans un environnement (Colab) que je ne peux pas reproduire localement
**When** le correctif est appliqué
**Then** il est explicitement marqué comme "à valider par l'utilisateur sur Colab", pas vérifié en local

### Story 4.3: Introduction de requirements.txt (résolution structurelle de l'incident cv2)

As a mainteneur du pipeline,
I want un fichier de dépendances déclaratif installable en une seule commande,
so that un runtime Colab frais (ou tout nouvel environnement) n'échoue plus sur un module manquant comme cv2, découvert un par un au fil des crashs.

**Acceptance Criteria:**

**Given** aucun fichier de dépendances n'existe actuellement (gap déjà documenté dans `docs/source-tree-analysis.md` § "Absence de packaging")
**When** `requirements.txt` est créé
**Then** il liste les dépendances tierces réellement importées dans le repo, vérifiées par recherche exhaustive (pas supposées) : `jax`, `flax`, `optax`, `opencv-python-headless`, `numpy`, `scipy`, `pandas`, `Pillow`, `matplotlib`, `tqdm`, `psutil`, `tensorflow`, `ultralytics`, `imagehash`
**And** `opencv-python-headless` (pas `opencv-python`) est utilisée — cohérent avec un environnement Colab sans display

**Given** cette story est la plus proche d'une décision d'architecture (introduction d'un mécanisme de gestion de dépendances)
**When** elle est prête à être implémentée
**Then** elle est soumise à l'agent architecte (Winston, `bmad-agent-architect`) pour avis avant merge — demande explicite de l'utilisateur

### Story 4.4: Archivage du code PyTorch/YOLO résiduel (pré-JAX)

As a mainteneur du repo,
I want déplacer (pas supprimer) le code lié à l'ancienne approche YOLO/PyTorch vers un dossier `archive/` documenté,
so that ce code historique reste traçable et récupérable sans polluer le pipeline actif ni la liste de dépendances par défaut.

**Acceptance Criteria:**

**Given** `tools/bounding_boxes_from_images_generation.py`, `tools/bounding_boxes_from_images_generation_main.py` (bootstrap d'annotation via YOLOv8n générique, PyTorch/Ultralytics) et `tools/yolov8n.pt` (checkpoint associé) ne sont importés par aucun autre fichier du repo (vérifié)
**And** l'utilisateur confirme qu'ils ont servi à un besoin réel (pré-génération de bounding boxes avant l'existence d'un modèle propriétaire) mais ne sont plus utilisés depuis que `aircraft_detector_unet` (JAX) existe
**When** ils sont déplacés vers `archive/` via `git mv` (historique préservé, pas une suppression)
**Then** `archive/README.md` documente leur contexte d'origine, pourquoi ils sont archivés (pas supprimés), et comment les réactiver

**Given** `tools/YOLOv8-n.py` (reproduction expérimentale de YOLOv8n en Flax/JAX, jamais finalisée, jamais importée non plus)
**When** il est déplacé vers `archive/`
**Then** `archive/README.md` documente que l'objectif était de reproduire en JAX un équivalent du modèle `yolov8n.pt`, jamais entraîné ni intégré à `model_library.py`

**Given** `docs/source-tree-analysis.md` référence l'ancien emplacement (`tools/`) de ces 4 fichiers
**When** l'archivage est effectué
**Then** la doc est mise à jour pour refléter le nouvel emplacement (`archive/`)

**Given** `ultralytics` (dépendance PyTorch de ces scripts) ne devient plus nécessaire par défaut une fois ces fichiers archivés
**When** `requirements.txt` (Story 4.3) est finalisé
**Then** `ultralytics` n'y figure ni en dépendance par défaut ni en note "à la demande" pour le pipeline principal — uniquement documentée dans `archive/README.md` pour la réactivation de ce cas d'usage spécifique

## Epic 5: Dataset CIFAR-10 pour boucle de test rapide

Le pipeline dispose d'un second jeu de données de classification, standard et léger (CIFAR-10, ~180 Mo), permettant d'itérer et de valider des changements de pipeline en quelques minutes sans dépendre du dataset avion complet (~10 Go) ni de Colab/TPU. Sert aussi de première preuve concrète de généricité du pipeline sur un cas image standard (`JAX_KEPLER` prouvait déjà la généricité côté 1D, pas côté image). Investigation préalable (Winston, `bmad-agent-architect`) : `data_management.py`, `loss_functions.py`, `reporting.py` et `model_library.py` (Global Average Pooling, pas de dimension spatiale figée) sont déjà génériques. Un vrai point de couplage a été découvert en creusant plus loin, non détecté à l'investigation initiale : `task_strategies.py` nomme les fichiers `.pkl` (export + training-state resumable) par `task_type` codé en dur, pas par config — sans conséquence tant qu'il n'existait qu'une seule config par `task_type`, mais CIFAR10 introduit la 2ᵉ config `classification` et ferait collision avec `FIGHTERJET_CLASSIFICATION` sur le même fichier de training-state. Traité comme un gate (Story 5.0) avant toute génération de données CIFAR-10.
**FRs covered:** aucune (hors scope FR1-FR10 du PRD refactor initial ; nouvelle capacité identifiée post-cycle, cf. `ARCHITECTURE-SPINE.md` § Deferred "Généralisation complète du pipeline")

### Story 5.0: Nommage des checkpoints dérivé de la config (gate)

As a mainteneur du pipeline,
I want que le nom des fichiers `.pkl` (export final et training-state resumable) soit dérivé du nom de la config utilisée plutôt que du `task_type`,
So that deux configs partageant le même `task_type` (ex. `FIGHTERJET_CLASSIFICATION` et `CIFAR10`, toutes deux `classification`) n'écrasent/ne lisent pas le même fichier de checkpoint.

**Acceptance Criteria:**

**Given** `dataset_configs.py::get_dataset_config(dataset_name)` retourne aujourd'hui un dict sans le nom de la config qu'il contient
**When** la fonction est modifiée
**Then** elle injecte `config["dataset_name"] = dataset_name` dans le dict retourné, avant validation

**Given** les 3 implémentations actuelles de `_get_export_path`/`get_training_state_path` (`ClassificationStrategy`, `DetectionStrategy`, `KeplerStrategy` dans `task_strategies.py`) — dont certaines ignorent totalement `config["checkpoint_path"]` (`DetectionStrategy._get_export_path`) et dont aucune ne lit un équivalent pour le training-state
**When** elles sont uniformisées
**Then** chacune retourne `config.get("checkpoint_path")` (export) ou `config.get("training_state_path")` (training-state) si fourni explicitement, sinon un nom par défaut dérivé de `best_model_{config["dataset_name"].lower()}.pkl` / `best_model_training_state_{config["dataset_name"].lower()}.pkl`

**Given** les 3 configs existantes (`FIGHTERJET_CLASSIFICATION`, `FIGHTERJET_DETECTION`, `JAX_KEPLER`) et les fichiers `.pkl` déjà versionnés/produits sous leurs noms actuels (`best_model.pkl`, `best_model_detection.pkl`, `best_model_training_state_classification.pkl`)
**When** la Story 5.0 est complétée
**Then** `checkpoint_path` reste explicite et inchangé pour les 3 configs (déjà le cas) **et** un `training_state_path` explicite est ajouté à chacune des 3, verrouillant le nom actuellement codé en dur (`best_model_training_state_classification.pkl`, `best_model_training_state_detection.pkl`, `best_model_training_state_kepler.pkl`) — aucun renommage silencieux d'un fichier déjà en usage

**Addendum (découvert pendant la validation locale, Story 5.2)** : le run CIFAR10 a silencieusement écrasé `sophisticated_cnn_128_plus.png` (courbes d'entraînement fighterjet, committé dans `2628ef6`) — même classe de bug, non détectée par le grep initial car dans `trainer.py` (`TrainingVisualizer`) et non dans `task_strategies.py`. `trainer.py:448` sauvegardait sous `f"{self.model_name}.png"` (nom d'**architecture**, partagée par CIFAR10 et FIGHTERJET_CLASSIFICATION), pas sous un nom dérivé de la config. Corrigé sur le même principe : `f"{self.config['dataset_name'].lower()}.png"`. Contrairement aux `.pkl`, aucune préservation de nom n'a été faite pour les configs existantes (fichier régénéré à chaque epoch, pas un artefact critique/versionné à préserver) — `sophisticated_cnn_128_plus.png` reste tel que restauré depuis `2628ef6` mais deviendra orphelin (un futur training `FIGHTERJET_CLASSIFICATION` produira désormais `fighterjet_classification.png`). Fichier CIFAR10 initialement écrasé restauré depuis git puis sa version CIFAR10 sauvegardée séparément (scratchpad) avant correctif.

**Addendum 2 (post-validation, retour utilisateur)** : après le premier run local réussi (`sophisticated_cnn_128_plus`, ~79,5% accuracy val en 10 epochs — **correctif** : une version antérieure de cet addendum citait par erreur 0.9448, qui est en réalité le résultat `FIGHTERJET_CLASSIFICATION` bfloat16/256×2 de `BENCHMARK-TPU-PERFORMANCE.md`, sans rapport avec CIFAR10), l'utilisateur a fait remarquer que réutiliser `sophisticated_cnn_128_plus` tel quel pour du 32×32 était surdimensionné (3 max-pools 32→16→8→4, pic à 512 canaux — taillé pour des silhouettes d'avion 128×128, pas pour CIFAR-10). Une nouvelle architecture `sophisticated_cnn_32_plus` a été ajoutée à `model_library.py` (même famille de blocs : SeparableConv, résiduelles, SE, Spatial Attention, tête GAP), avec 2 max-pools au lieu de 3 (32→16→8, pas de réduction jusqu'à 4×4) et des canaux à peu près divisés par 2 à chaque étage (pic à 256 au lieu de 512) — **318 691 paramètres vs 1 257 041** (~4× moins), vérifié par instanciation directe (forward pass train/eval, shapes de sortie correctes). `CIFAR10.model_name` mis à jour vers `sophisticated_cnn_32_plus`. Les paramètres d'augmentation de données de `CIFAR10` étaient déjà minimaux (seul `flip_h=True` actif, tout le reste à 0.0) — pas de changement nécessaire sur ce point, confirmé à l'utilisateur plutôt que modifié à l'aveugle.

**Addendum 3 (runs Colab successifs `sophisticated_cnn_32_plus`, tuning epochs/LR/dropout)** : premier run Colab (`archive/training_cifar10_log_bfloat_128x1.txt` — malgré son nom, le log confirme `float16`, pas bfloat16 ; `epochs=10`, `decay_steps=2000` hérité du réglage précédent) a plafonné à **73,30%** (val accuracy), stagnation nette dès l'epoch 5 (val loss figé 0.7601-0.7611 de l'epoch 6 à 10). Diagnostiqué comme un bug de LR schedule, pas une limite de capacité : avec 391 steps/epoch, `decay_steps=2000` fait tomber le LR à `end_value=1e-6` au step ~2000 (epoch ~5,1/10), gelant l'apprentissage pour le reste du run. Corrigé : `epochs` 10→30, `patience` 3→5, `decay_steps` 2000→11700 (≈ steps/epoch × epochs, couvre tout l'entraînement). Deuxième run (`archive/training_cifar10_log_float16_128x1.txt`) : **79,49%** (meilleur, epoch 10), mais overfitting net ensuite — train accuracy 86,97%→93,22% (epoch 10→15) pendant que le val loss remonte 0.6325→0.7442 au lieu de continuer à baisser ; early stopping déclenché correctement à l'epoch 15 (patience=5 depuis le meilleur). Diagnostiqué comme absence totale de régularisation (`dropout_rate=0.0` dans les deux runs ci-dessus), pas comme un manque de capacité (le train accuracy grimpant librement au-delà de 93% indique une capacité largement suffisante). Corrigé : `dropout_rate` 0.0→0.3 (`gpu` et `tpu`), aligné sur la valeur déjà utilisée par `JAX_KEPLER`. Résultat de ce 3ᵉ run non encore capturé au moment de la revue de code — à mettre à jour si un futur cycle reprend ce sujet.

### Story 5.1: Génération des chunks CIFAR-10 au format .npz

As a mainteneur du pipeline,
I want un script dédié (`cifar10_classification_dataset_tools.py`) qui télécharge le CIFAR-10 officiel (pickle, cs.toronto.edu) et génère les chunks `.npz` (`image`/`label`) et le fichier `meanstd.npz` dans le format exact attendu par `ChunkManager`,
So that le dataset soit consommable par le pipeline existant sans aucune modification de `data_management.py`.

**Acceptance Criteria:**

**Given** CIFAR-10 n'est disponible ni via un dossier d'images ni via un chargeur déjà présent dans le repo
**When** le script `cifar10_classification_dataset_tools.py` s'exécute
**Then** il télécharge `cifar-10-python.tar.gz` depuis `https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz` (source officielle) s'il n'est pas déjà présent localement, l'extrait, et parse les 5 batches `data_batch_1..5` (train, 50 000 images) et `test_batch` (val, 10 000 images) via `pickle`
**And** aucune fonction de `fighterjet_classification_dataset_tools.py` n'est réutilisée pour le chargement/décodage (conçues pour un dossier `.png`/`.jpg`, format non applicable ici)

**Given** le layout natif des batches CIFAR-10 (chaque image stockée en 3072 octets : 1024 rouge, 1024 vert, 1024 bleu, aplati ligne par ligne — pas HWC)
**When** les images sont décodées
**Then** chaque ligne de 3072 octets est reshape en `(3, 32, 32)` puis transposée en `(32, 32, 3)` (HWC) avant normalisation `/255.0` en float32

**Given** le contrat `.npz` attendu par `ChunkManager` (`data_management.py`)
**When** les chunks sont écrits
**Then** le split `train` produit `{output_prefix}_train_chunk0.npz` (clés `image` float32 `(N, 32, 32, 3)`, `label` int32 `(N,)`) et le split `test` produit `{output_prefix}_val_chunk0.npz` selon le même format

**Given** `ChunkManager` exige un fichier `{output_prefix}_meanstd.npz` (clés `mean`/`std`) au chargement
**When** le script termine la génération
**Then** ce fichier est calculé sur le split `train` et sauvegardé au bon emplacement, sans quoi `get_datasets()` lèverait un `FileNotFoundError`

### Story 5.2: Entrée CIFAR10 dans dataset_configs.py et validation locale

As a mainteneur du pipeline,
I want une nouvelle entrée `CIFAR10` dans `dataset_configs.py` réutilisant `ClassificationStrategy` sans modification,
So that j'obtienne une boucle d'entraînement/test complète, exécutable localement en quelques minutes, pour valider rapidement tout changement de pipeline.

**Acceptance Criteria:**

**Given** les Stories 5.0 (nommage checkpoints) et 5.1 (chunks + `meanstd.npz`) complétées
**When** l'entrée `CIFAR10` est ajoutée à `dataset_configs.py`
**Then** elle définit `num_classes=10`, les 10 `class_names` standards CIFAR-10, `image_size=(32, 32)`, `grayscale=False`, `model_name="sophisticated_cnn_128_plus"` (réutilisé tel quel — architecture à Global Average Pooling, aucune modification requise) ⚠️ **superseded par l'Addendum 2 ci-dessous** : `model_name` a été changé pour `sophisticated_cnn_32_plus` (nouvelle architecture, `model_library.py` modifié) suite au retour utilisateur post-validation — cette puce d'AC ne reflète plus le code livré, `loss_method="cross_entropy"` (dataset équilibré par construction, contrairement à `FIGHTERJET_CLASSIFICATION` qui utilise `focal_loss`)
**And** aucun `checkpoint_path`/`training_state_path` explicite n'est requis — le nommage dérivé de `dataset_name` (Story 5.0) s'applique par défaut (`best_model_cifar10.pkl`, `best_model_training_state_cifar10.pkl`)
**And** aucune modification n'est faite à `model_library.py`, `loss_functions.py`, `reporting.py` ou `data_management.py`

**Given** l'objectif explicite d'une boucle de test locale rapide (contrairement à `FIGHTERJET_CLASSIFICATION`/`FIGHTERJET_DETECTION` dont l'entraînement complet doit rester réservé à Colab/TPU pour éviter un crash mémoire local)
**When** `main.py CIFAR10` est exécuté localement (CPU/GPU) sur un nombre réduit d'epochs
**Then** l'entraînement se déroule sans erreur jusqu'à son terme, confirmant par l'exécution (pas seulement par lecture de code) la généricité du pipeline sur un second cas d'usage image
**And** les fichiers `.pkl` produits (`best_model_cifar10.pkl`, `best_model_training_state_cifar10.pkl`) sont distincts de ceux de `FIGHTERJET_CLASSIFICATION`, confirmant par l'exécution que la Story 5.0 élimine bien la collision

## Requirements Inventory — Renommage jax_supervised_training

Source : `_bmad-output/planning-artifacts/prds/prd-JAX_Detection-2026-07-14/prd.md` (status: final). Initiative distincte du refactor Epic 1-3 — pas de document Architecture dédié (renommage mécanique, pas de nouvelle décision technique), cohérent avec le PRD lui-même.

### Functional Requirements

FR1: Le dossier local du projet et le dépôt distant `aobled/JAX_Detection` sont renommés en `jax_supervised_training` ; le remote git local est mis à jour vers la nouvelle URL.
FR2: La variable d'environnement `JAX_DETECTION_DATA_ROOT` (seul site de lecture : `dataset_configs.py::DATA_ROOT`) est renommée en `JAX_SUPERVISED_TRAINING_DATA_ROOT`, sans compatibilité double-nom. Tout notebook Colab actif définissant cette variable est mis à jour dans le même geste.
FR3: Le dossier Google Drive `MyDrive/JAX_Detection/` est renommé en `MyDrive/jax_supervised_training/`. Tout notebook Colab actif référençant l'ancien chemin est mis à jour dans le même geste. `tools/process rclone GDrive and run collab.txt` (pense-bête manuel) est explicitement hors scope, laissé à la charge de l'utilisateur.
FR4: Toute référence textuelle à "JAX_Detection"/"JAX_DETECTION" dans le code activement importé (6 fichiers), `docs/` (6 fichiers) et les artefacts BMAD **vivants** (`sprint-status.yaml`, `epics.md`) est mise à jour. Les artefacts BMAD historiques datés (dossiers PRD/architecture snapshotés du 2026-07-12, rétrospectives passées, `docs/project-scan-report.json`) restent explicitement non touchés.

### NonFunctional Requirements

NFR1 — Non-régression fonctionnelle : un entraînement lancé après renommage (`FIGHTERJET_CLASSIFICATION` ou `CIFAR10`, sur Colab) produit un comportement identique à avant renommage — seuls les noms/chemins changent. Validé par exécution réelle, pas par lecture de code.
NFR2 — Pas d'échec silencieux : si un point de lecture de l'ancien nom (variable d'environnement, chemin Drive) est oublié, l'échec doit être détectable rapidement (erreur explicite), jamais un fallback silencieux vers un chemin local inexistant sur Colab.

### Additional Requirements (Architecture)

Aucun — pas de document Architecture pour cette initiative (confirmé avec l'utilisateur).

### UX Design Requirements

N/A — aucune interface utilisateur, pas de parcours à documenter (projet solo, un seul opérateur).

## Epic List

### Epic 6 : Renommage du projet en jax_supervised_training

Le projet porte un nom cohérent sur toutes ses surfaces actives (dossier local, dépôt GitHub, variable d'environnement, dossier Google Drive, code, documentation) — reflétant la généricité déjà prouvée du pipeline, sans laisser de référence résiduelle à l'ancien nom dans un chemin actif.
**FRs covered:** FR1, FR2, FR3, FR4

### FR Coverage Map

FR1: Epic 6 — Renommage dossier local + dépôt GitHub
FR2: Epic 6 — Renommage variable d'environnement `JAX_DETECTION_DATA_ROOT`
FR3: Epic 6 — Renommage dossier Google Drive
FR4: Epic 6 — Mise à jour références textuelles (code, docs, artefacts vivants)

## Epic 6: Renommage du projet en jax_supervised_training

Le projet porte un nom cohérent sur toutes ses surfaces actives (dossier local, dépôt GitHub, variable d'environnement, dossier Google Drive, code, documentation) — reflétant la généricité déjà prouvée du pipeline, sans laisser de référence résiduelle à l'ancien nom dans un chemin actif. **FRs covered:** FR1, FR2, FR3, FR4

### Story 6.1: Renommage du dossier local et du dépôt GitHub

As a mainteneur du projet,
I want renommer le dossier local et le dépôt GitHub en `jax_supervised_training`,
So that l'identité du projet reflète sa portée réelle avant toute autre étape de renommage.

**Acceptance Criteria:**

**Given** le dossier local `JAX_Detection` et le dépôt distant `aobled/JAX_Detection`
**When** le dossier est renommé et `gh repo rename` (ou équivalent) est exécuté
**Then** le dossier local répond au nouveau nom et le remote local (`git remote -v`) pointe vers la nouvelle URL
**And** l'ancienne URL GitHub redirige encore vers le nouveau dépôt (comportement natif, vérifié)

### Story 6.2: Renommage de la variable d'environnement et du dossier Google Drive

As a mainteneur du projet,
I want renommer `JAX_DETECTION_DATA_ROOT` en `JAX_SUPERVISED_TRAINING_DATA_ROOT` et le dossier Drive `MyDrive/JAX_Detection/` en `MyDrive/jax_supervised_training/`,
So that le pipeline de données fonctionne sous le nouveau nom sans mécanisme de compatibilité.

**Acceptance Criteria:**

**Given** `dataset_configs.py::DATA_ROOT` (seul site de lecture de la variable) et le dossier Drive utilisé par les notebooks Colab
**When** la variable et le dossier sont renommés
**Then** `dataset_configs.py` ne référence plus `JAX_DETECTION_DATA_ROOT`, sans fallback vers l'ancien nom
**And** chaque notebook Colab actif est mis à jour vers la nouvelle variable et le nouveau chemin Drive, dans le même geste — pas en différé (checklist explicite avant/après, ce mode d'échec s'étant déjà produit une fois sur ce projet)
**And** `tools/process rclone GDrive and run collab.txt` n'est pas touché — laissé à la charge de l'utilisateur (hors scope confirmé)

### Story 6.3: Mise à jour des références textuelles — code, documentation, artefacts vivants

As a mainteneur du projet,
I want mettre à jour toute référence textuelle à "JAX_Detection" dans le code actif, `docs/` et les artefacts BMAD vivants,
So that aucune trace de l'ancien nom ne subsiste dans une source consultée en continu.

**Acceptance Criteria:**

**Given** les 6 fichiers de code identifiés (`dataset_configs.py`, `inference_utils.py`, `reporting.py`, `bounding_boxes_with_classification_from_video_generation.py`, `tools/bounding_boxes_with_classification_from_images_generation.py`, `tools/kepler_dataset_tools.py`) et les 6 fichiers `docs/`
**When** ils sont mis à jour
**Then** aucun ne contient plus "JAX_Detection"/"JAX_DETECTION"
**And** `sprint-status.yaml` (champ `project:`) et `epics.md` sont mis à jour — traités comme artefacts vivants, pas historiques
**And** les artefacts BMAD historiques datés (dossiers PRD/architecture du 2026-07-12, rétrospectives passées, `docs/project-scan-report.json`) restent explicitement non touchés

### Story 6.4: Validation finale par exécution réelle

As a mainteneur du projet,
I want lancer un entraînement Colab complet après renommage,
So that la non-régression fonctionnelle (NFR1) et l'absence d'échec silencieux (NFR2) soient prouvées par l'exécution, pas par la lecture de code.

**Acceptance Criteria:**

**Given** les Stories 6.1 à 6.3 complétées
**When** `main.py` est exécuté sur Colab (`FIGHTERJET_CLASSIFICATION` ou `CIFAR10`) sous le nouveau nom
**Then** l'entraînement démarre et se déroule sans erreur de chemin, avec un comportement identique à avant renommage
**And** si un point de lecture de l'ancien nom avait été oublié, l'échec se serait manifesté immédiatement et explicitement (pas un fallback silencieux) — confirmé a posteriori par l'absence d'un tel incident

## Requirements Inventory — JAX Single-Pass

Source : `_bmad-output/specs/spec-jax-single-pass/SPEC.md` (kernel Capabilities/Constraints/Non-goals, produit par `bmad-spec` — pas de PRD classique pour cette initiative) et `_bmad-output/planning-artifacts/architecture/architecture-jax_supervised_training-2026-07-15/ARCHITECTURE-SPINE.md` (AD-9 à AD-20, hérite AD-1 à AD-8 de la spine parente du refactor Epic 1-3). Initiative distincte des Epics 1-6 — remplace le pipeline UNet+python/cv2+classification par un détecteur par point central (`JAX_DETECTOR`, nouvel entraînement) et une composition d'inférence JAX-native (`build_single_pass_predict_fn`) réutilisant `FIGHTERJET_CLASSIFICATION` figée.

### Functional Requirements

FR1 (CAP-1) : Un nouveau modèle de détection par point central (`JAX_DETECTOR`, heatmap de centres + régression de taille) peut être entraîné sur des chunks `.npz` classiques à 224×224 — jamais sur un dataset full-HD (ratio ~41× prohibitif) — remplaçant la segmentation UNet comme méthode de détection.
FR2 (CAP-2) : Une unique fonction JIT-compilée (`build_single_pass_predict_fn`) produit, à partir d'une image 1920×1080 grayscale, jusqu'à 20 détections (boîte+classe+scores), sans `cv2.findContours` ni boucle de recadrage python sur le chemin critique.
FR3 (CAP-3) : Deux avions proches/en contact, qui fusionnent aujourd'hui en une seule détection sous segmentation+`findContours`, sont prédits comme deux instances indépendantes sous la nouvelle tête par point central.
FR4 (CAP-4) : `FIGHTERJET_CLASSIFICATION` est chargée figée dans le nouveau graphe d'inférence et réutilisée sans réentraînement.

### NonFunctional Requirements

NFR1 : `FIGHTERJET_CLASSIFICATION` ne doit jamais être réentraînée par ce chantier — chargement figé uniquement.
NFR2 : Aucun dataset d'entraînement ne peut nécessiter des images full-HD ; `JAX_DETECTOR` s'entraîne uniquement à sa résolution de config.
NFR3 — Non-régression / rollback : l'ancien pipeline complet (`FIGHTERJET_DETECTION`, `AircraftDetectorUNet`, `DetectionStrategy`, `DetectionDataset`, `decode_segmentation_and_detect(_batch)`, `fighterjet_detection_dataset_tools.py`, et leurs consommateurs incl. `tools/audit_dataset_detection.py` et `tools/boxes_process_manual_tkinter.py`) reste pleinement fonctionnel, sans modification, pendant toute l'epic et après — exigence explicite de l'utilisateur, filet de sécurité en cas d'échec de l'initiative.
NFR4 : La composition d'inférence doit être zéro-python/cv2 sur son chemin critique (pas de `cv2.findContours`, pas de boucle de recadrage python).
NFR5 : La sortie de `build_single_pass_predict_fn` est toujours une structure à 20 slots fixes (`boxes`/`classes`/`class_scores`/`detection_scores`/`valid_mask`), slots invalides à zéro, `valid_mask` seule autorité pour distinguer un slot réel d'un slot vide.

### Additional Requirements (Architecture)

- **AD-9** : Tête de détection par point central (anchor-free) — heatmap de centres + régression de taille, extraction JAX-native (`jax.lax.reduce_window` + `jax.lax.top_k`), jamais `cv2.findContours`.
- **AD-10** : Backbone + Feature Pyramid Network reporté — encodeur dans la même famille que l'UNet actuel tant qu'aucune insuffisance mesurée ne le justifie.
- **AD-11** : Crop différentiable par `jax.scipy.ndimage.map_coordinates` (grid-sample bilinéaire), `vmap` sur 20 slots fixes — jamais un sous-réseau appris ni une découpe cv2.
- **AD-12** : Résolution canonique 1920×1080, double branche interne (`RESIZE` déterministe pour la détection, pleine résolution conservée pour le `CROP`) ; `RESIZE` dérive toujours sa taille cible de la clé `image_size` de la config `JAX_DETECTOR`, jamais un littéral codé en dur.
- **AD-13** : `RESCALE` symétrique entre décodage et recadrage ; le stride/résolution de sortie du détecteur est une valeur unique nommée, partagée par l'extraction de pics et `RESCALE`.
- **AD-14** : Entraînement toujours modulaire — le graphe unifié n'existe qu'à l'inférence ; `JAX_DETECTOR` s'entraîne seul, `FIGHTERJET_CLASSIFICATION` n'est jamais réentraînée.
- **AD-15** : Sortie à 20 slots fixes, slots invalides remplis à zéro ; le seuil de score de détection dérivant `valid_mask` vit dans la config `JAX_DETECTOR` (jamais une constante privée dupliquée).
- **AD-16** : `JAX_DETECTOR` est une entrée `DATASET_CONFIGS` standard ; la composition JAX Single-Pass n'en est pas une — elle vit comme fonction `build_single_pass_predict_fn` dans `inference_utils.py` (AD-1 hérité), lisant les deux configs via `get_dataset_config()` sans les modifier.
- **AD-17** : Nouveau format de tâche = nouvelle classe dédiée dans `task_strategies.py`/`data_management.py` (jamais une branche conditionnelle) ; le littéral `task_type` est défini une seule fois et référencé identiquement aux trois points de dispatch (`task_strategies.py`, `data_management.py`, **et `main.py`**, `main.py:107-143`).
- **AD-18** : Le schéma d'échange `.npz` heatmap+taille (noms de clés, formes, unités, formule du rayon gaussien) est défini par un module/paire de fonctions partagées, importées par `fighterjet_detection_dataset_tools_v2.py` (producteur) et la nouvelle classe `data_management.py` (consommateur) — jamais réimplémenté indépendamment de chaque côté.
- **AD-19** : Grid-based/YOLO à ancres reporté, pas abandonné — signal d'entraînement sur le chevauchement quasi inexistant (~1.36% des images ont 2+ boîtes) ; ne pas reconsidérer sans une stratégie de données synthétiques dédiée.
- **AD-20** : Non-régression explicite — aucune story de cette initiative ne supprime, ne renomme, ni ne modifie le comportement de l'ancien pipeline (`FIGHTERJET_DETECTION` et ses consommateurs).
- **Stack** : aucune nouvelle dépendance externe introduite — `jax.scipy.ndimage.map_coordinates`, `jax.lax.reduce_window`, `jax.lax.top_k` font partie de JAX déjà utilisé par le projet.

### UX Design Requirements

N/A — aucun contrat UX applicable. Aucun changement d'interface : les outils GUI existants (`tools/boxes_process_manual_tkinter.py`) restent sur l'ancien pipeline (AD-20), non modifiés par cette initiative.

### FR Coverage Map

FR1: Epic 7 — Entraînement JAX_DETECTOR
FR2: Epic 8 — Composition d'inférence JAX Single-Pass
FR3: Epic 8 — Composition d'inférence JAX Single-Pass (fusion de boîtes résolue structurellement)
FR4: Epic 8 — Composition d'inférence JAX Single-Pass (classification figée réutilisée)

## Epic List

### Epic 7: Entraînement de JAX_DETECTOR

Un nouveau modèle de détection par point central (heatmap de centres + régression de taille) est entraîné et produit des prédictions exploitables sur un jeu de validation — sans jamais nécessiter de dataset full-HD. Standalone : livre un détecteur entraîné et évaluable indépendamment, même avant que la composition d'inférence unifiée n'existe.
**FRs covered:** FR1
**Note d'implémentation :** nouvelle classe modèle, nouvelle stratégie dédiée (AD-9/AD-17), nouvelles fonctions de perte, nouvelle entrée `JAX_DETECTOR` (`dataset_configs.py`), nouvelle classe de chargeur (AD-17/AD-18), `fighterjet_detection_dataset_tools_v2.py` (AD-18), dispatch `task_type` ajouté à `main.py` (AD-17). Séquencement probable type "Story 0" (schéma `.npz` heatmap+taille défini avant les stories consommatrices), sur le même principe que l'Epic 1 (AD-7 hérité).

### Epic 8: Composition d'inférence JAX Single-Pass

Une unique fonction JIT-compilée transforme une image brute en détections classées, sans python/cv2 intermédiaire — remplace l'orchestration manuelle dans les scripts de génération vidéo/image. Standalone : consomme Epic 7 (détecteur entraîné) + `FIGHTERJET_CLASSIFICATION` existante (figée), sans que l'ancien pipeline `FIGHTERJET_DETECTION` ne soit jamais requis de fonctionner différemment (AD-20).
**FRs covered:** FR2, FR3, FR4
**Note d'implémentation :** `build_single_pass_predict_fn` dans `inference_utils.py` (AD-1 hérité, AD-16) ; dépend d'un `JAX_DETECTOR` entraîné (Epic 7) ; migration des scripts consommateurs ; non-régression garantie sur l'ancien pipeline (AD-20).

## Epic 7: Entraînement de JAX_DETECTOR

Un nouveau modèle de détection par point central (heatmap de centres + régression de taille) est entraîné et produit des prédictions exploitables sur un jeu de validation — sans jamais nécessiter de dataset full-HD. Standalone : livre un détecteur entraîné et évaluable indépendamment, même avant que la composition d'inférence unifiée n'existe. **FRs covered:** FR1

### Story 7.1: Définition du schéma d'échange heatmap+taille (AD-18)

As a mainteneur du pipeline de détection,
I want un module/paire de fonctions partagées qui encodent des boîtes brutes en cibles heatmap+taille et les décodent en retour,
So that le producteur (`fighterjet_detection_dataset_tools_v2.py`) et le consommateur (nouvelle classe `data_management.py`) ne réimplémentent jamais indépendamment le même format, évitant une lecture croisée silencieusement incompatible (AD-18).

**Acceptance Criteria:**

**Given** `raw_boxes` au format existant (`data["annotation"]["bbox"]`, x/y/w/h, déjà confirmé comme source de vérité)
**When** la fonction d'encodage est appelée sur un ensemble de boîtes pour une image à la résolution de config
**Then** elle retourne un heatmap de centres (gaussien, rayon dérivé de la taille de boîte) et une carte de régression de taille, avec un format de clés de sortie fixé et documenté (noms, shapes, unités)

**Given** les cibles heatmap+taille produites par la fonction d'encodage
**When** la fonction de décodage est appelée sur ces mêmes cibles (round-trip)
**Then** les boîtes récupérées correspondent aux boîtes d'origine à une tolérance near-exacte (validation du round-trip, pas juste de la forme)

**Given** AD-18 (source unique)
**When** `fighterjet_detection_dataset_tools_v2.py` (Story 7.4) et la nouvelle classe de `data_management.py` (Story 7.5) sont implémentées
**Then** les deux importent ces fonctions partagées ; aucune des deux ne réimplémente indépendamment le format des clés `.npz`

**Given** cette story est un prérequis bloquant (même principe qu'AD-7 hérité / Story 1.2)
**When** elle est complétée
**Then** aucune story suivante de l'Epic 7 ne redéfinit le schéma d'échange — seules 7.4 et 7.5 l'importent

### Story 7.2: Nouvelle classe modèle JAX_DETECTOR (AD-9, AD-10)

As a mainteneur de la factory de modèles,
I want une nouvelle classe de modèle encoder-decoder avec une tête heatmap de centres + régression de taille,
So that `JAX_DETECTOR` soit instanciable via `model_library.get_model()` comme n'importe quel autre modèle du pipeline.

**Acceptance Criteria:**

**Given** `AircraftDetectorUNet` actuel comme référence de famille d'architecture (encoder-decoder simple)
**When** la nouvelle classe est implémentée
**Then** elle reste dans la même famille (pas de backbone+FPN, AD-10) et remplace la sortie de segmentation par deux têtes : un heatmap de centres et une carte de régression de taille (2 canaux, largeur/hauteur)

**Given** le registre `MODELS` de `model_library.py`
**When** la classe est ajoutée
**Then** elle est enregistrée sous le nom `aircraft_detector_centernet` (cohérent avec `aircraft_detector_unet`, reflète la tête par point central style CenterNet, AD-9) et instanciable via `get_model()`

**Given** un forward pass sur une image à la résolution de config (batch factice)
**When** le modèle est appelé en mode train et en mode eval
**Then** les shapes de sortie (heatmap, taille) sont correctes et cohérentes avec le schéma défini en Story 7.1

### Story 7.3: Nouvelles fonctions de perte (heatmap focal loss + régression de taille)

As a mainteneur du pipeline d'entraînement,
I want des fonctions de perte dédiées à la sortie heatmap+taille,
So that `JAX_DETECTOR` puisse être entraîné avec un signal de gradient adapté à un heatmap creux (déséquilibre positif/négatif).

**Acceptance Criteria:**

**Given** un heatmap de centres cible (gaussien, creux) et un heatmap prédit
**When** la fonction de perte heatmap est calculée
**Then** elle utilise une formulation focal loss adaptée aux heatmaps creux, plutôt qu'une cross-entropy simple dominée par le fond

**Given** une carte de régression de taille cible et prédite, valide uniquement aux positions de centre réel
**When** la perte de régression de taille est calculée
**Then** elle n'est calculée qu'aux positions où un centre réel existe (masquée ailleurs)

**Given** `loss_functions.py`
**When** les deux fonctions sont ajoutées
**Then** aucune fonction existante n'est réutilisée ni réintroduite — nouvelles fonctions dédiées. En particulier `compute_grid_loss`/`compute_grid_loss_multilevel` (`loss_functions.py`, toujours présentes et importées par `task_strategies.py` — **vérifié le 2026-07-16, contrairement à une affirmation antérieure erronée les disant supprimées à l'Epic 3** ; probablement du code mort en pratique car aucune config active de `dataset_configs.py` n'utilise un `loss_method` grid-based, mais non supprimées) restent dédiées à l'ancienne approche grid-based et ne sont ni réutilisées ni modifiées par cette story

### Story 7.4: fighterjet_detection_dataset_tools_v2.py — génération des chunks depuis raw_boxes

As a mainteneur de la préparation de données,
I want un nouveau script qui génère des chunks `.npz` heatmap+taille depuis les mêmes `raw_boxes` que l'outil actuel,
So that `JAX_DETECTOR` dispose d'un dataset d'entraînement, sans jamais toucher `fighterjet_detection_dataset_tools.py` (AD-20, non-régression).

**Acceptance Criteria:**

**Given** le schéma d'échange défini en Story 7.1
**When** `fighterjet_detection_dataset_tools_v2.py` encode les `raw_boxes` d'une image
**Then** il produit des chunks `.npz` au format défini en 7.1, à la résolution de config `JAX_DETECTOR` (pas full-HD, NFR2)

**Given** `fighterjet_detection_dataset_tools.py` (l'outil actuel, approche masque)
**When** le nouveau script est créé
**Then** il n'est ni modifié ni supprimé — fichier séparé, coexistence complète (AD-20)

**Given** le pattern déjà en place dans l'outil actuel (résolution source quelconque → résolution de config stockée, coordonnées rescalées proportionnellement)
**When** le nouveau script traite des images sources de résolution variable
**Then** il applique le même principe de rescale proportionnel avant encodage

### Story 7.5: Nouvelle classe de chargeur dédiée (data_management.py, AD-17/AD-18)

As a mainteneur du pipeline de données,
I want une nouvelle classe de chargeur dédiée au format heatmap+taille,
So that `JAX_DETECTOR` charge ses chunks sans jamais modifier `DetectionDataset` (qui reste dédiée au format masque existant, AD-20).

**Acceptance Criteria:**

**Given** `DetectionDataset` actuelle (attend une clé `'masks'` explicite, vérifiée ligne ~300)
**When** la nouvelle classe est créée
**Then** `DetectionDataset` n'est ni modifiée ni étendue par une branche conditionnelle — nouvelle classe séparée (pattern `ChunkManager`/`DetectionDataset` hérité)

**Given** le schéma d'échange défini en Story 7.1
**When** la nouvelle classe charge un chunk produit par la Story 7.4
**Then** elle utilise les fonctions de décodage partagées, sans réimplémenter sa propre lecture des clés

**Given** le dispatch `task_type` existant (`data_management.py:429-463`)
**When** la nouvelle classe est intégrée
**Then** un nouveau `task_type` dédié (défini en Story 7.7) la sélectionne, sans casser le dispatch existant

### Story 7.6: Nouvelle stratégie dédiée (task_strategies.py, AD-9/AD-17)

As a mainteneur du pipeline d'entraînement,
I want une nouvelle `TaskStrategy` dédiée au format heatmap+taille,
So that `JAX_DETECTOR` s'entraîne via le pattern Strategy/Factory/DI existant, sans modifier `DetectionStrategy` (AD-20).

**Acceptance Criteria:**

**Given** `DetectionStrategy` actuelle (dédiée à l'approche segmentation)
**When** la nouvelle stratégie est créée
**Then** `DetectionStrategy` n'est ni modifiée ni étendue — nouvelle classe séparée implémentant l'interface `TaskStrategy`

**Given** les fonctions de perte de la Story 7.3
**When** la nouvelle stratégie calcule la perte d'entraînement
**Then** elle compose la perte heatmap + la perte de régression de taille

**Given** le nommage de checkpoint dérivé de `dataset_name` (pattern Story 5.0, déjà en place)
**When** la nouvelle stratégie exporte un modèle
**Then** elle suit le même pattern `_get_export_path`/`get_training_state_path` dérivé de `config["dataset_name"]`, pas de `task_type` codé en dur

### Story 7.7: Entrée JAX_DETECTOR (dataset_configs.py) + dispatch task_type (main.py, AD-17)

As a mainteneur du pipeline de configuration,
I want une nouvelle entrée `JAX_DETECTOR` dans `dataset_configs.py` et son dispatch dans `main.py`,
So that l'entraînement soit lançable via `main.py JAX_DETECTOR`, cohérent avec les configs existantes.

**Acceptance Criteria:**

**Given** `DATASET_CONFIGS` et `validate_config` (exige un `model_name` unique)
**When** `JAX_DETECTOR` est ajoutée
**Then** elle définit `model_name` (Story 7.2), `image_size`, `num_classes`, et un `task_type` dédié dont le littéral exact est utilisé identiquement dans `task_strategies.py` (7.6), `data_management.py` (7.5) et `main.py`

**Given** `main.py:107-143` (dispatch `if/elif` existant : classification, detection, kepler)
**When** le nouveau `task_type` est ajouté
**Then** une branche supplémentaire instancie la stratégie de la Story 7.6, sans modifier le comportement des branches existantes

**Given** AD-20 (non-régression)
**When** cette story est complétée
**Then** `FIGHTERJET_DETECTION` reste inchangée dans `dataset_configs.py`, aucun renommage ni suppression

### Story 7.8: Entraînement complet + validation (preuve d'exécution, FR1/CAP-1)

As a mainteneur du pipeline de détection,
I want lancer un entraînement complet de `JAX_DETECTOR` et valider ses prédictions sur un jeu de validation,
So that FR1 soit prouvé par exécution réelle — le critère de succès de CAP-1.

**Acceptance Criteria:**

**Given** les Stories 7.1 à 7.7 complétées
**When** `main.py JAX_DETECTOR` est exécuté
**Then** l'entraînement se déroule sans erreur jusqu'à son terme, produisant un checkpoint exporté

**Given** le checkpoint produit
**When** des prédictions sont faites sur un jeu de validation tenu à l'écart
**Then** les heatmaps de centres et les tailles de boîte prédites sont exploitables (comparaison qualitative a minima)

**Given** NFR2 (pas de dataset full-HD)
**When** l'entraînement est vérifié
**Then** aucune étape n'a nécessité de charger un dataset à résolution full-HD — uniquement les chunks à résolution de config (Story 7.4)

**Given** l'initialisation de l'encodeur (question ouverte de `SPEC.md`, non tranchée)
**When** cette story est lancée
**Then** l'initialisation aléatoire est utilisée par défaut — le transfert learning depuis l'UNet reste une question ouverte séparée, non traitée ici

## Epic 8: Composition d'inférence JAX Single-Pass

Une unique fonction JIT-compilée transforme une image brute en détections classées, sans python/cv2 intermédiaire — remplace l'orchestration manuelle dans les scripts de génération vidéo/image. Standalone : consomme Epic 7 (détecteur entraîné) + `FIGHTERJET_CLASSIFICATION` existante (figée), sans que l'ancien pipeline `FIGHTERJET_DETECTION` ne soit jamais requis de fonctionner différemment (AD-20). **FRs covered:** FR2, FR3, FR4

### Story 8.1: Validation de parité pixel (RESIZE + CROP)

As a mainteneur du pipeline d'inférence,
I want un script de test autonome comparant numériquement le `RESIZE` JAX (1920×1080→224×224) à PIL/LANCZOS, et le `CROP` JAX (`map_coordinates`) à `cv2.resize`, sur des images réelles,
So that les deux risques de régression silencieuse de `SPEC.md` soient écartés avant d'investir dans le code de composition.

**Acceptance Criteria:**

**Given** une image réelle de résolution 1920×1080
**When** elle est redimensionnée en 224×224 via la méthode JAX choisie et via PIL/LANCZOS (méthode utilisée pour préparer les chunks d'entraînement `JAX_DETECTOR`, `fighterjet_detection_dataset_tools.py:104`)
**Then** l'écart numérique pixel par pixel est mesuré et documenté (pas une comparaison visuelle)

**Given** une boîte connue sur une image réelle
**When** elle est recadrée et redimensionnée en 128×128 via `map_coordinates` et via `cv2.resize` (méthode utilisée pour entraîner `FIGHTERJET_CLASSIFICATION`)
**Then** l'écart numérique est mesuré, y compris la convention d'alignement pixel (bord vs centre)

**Given** une boîte prédite dont le centre est proche du bord du cadre (ex. x≈1919 ou y≈1079), une partie de la boîte tombant hors de l'image source
**When** le crop est effectué via `map_coordinates` sur cette boîte
**Then** le comportement hors-limites (clamp vs extrapolation) est explicitement mesuré et documenté, pas seulement le cas nominal centré dans l'image (trouvé en party mode, Code Review Crew — Boundary/Grumbal)

**Given** les deux Open Questions correspondantes de `SPEC.md`
**When** cette story est complétée
**Then** `SPEC.md` est mis à jour pour refléter le résultat mesuré, pas seulement affirmé

**Given** un écart significatif serait détecté
**When** cette story est complétée
**Then** la méthode JAX est ajustée jusqu'à obtenir une parité acceptable, avant que les Stories 8.2/8.5 ne s'appuient dessus

### Story 8.2: RESIZE + appel du détecteur figé (branche détection)

As a mainteneur du pipeline d'inférence,
I want une étape `RESIZE` déterministe (dérivée de la config `JAX_DETECTOR`) suivie d'un appel au détecteur figé,
So that l'image canonique produise un heatmap+taille en repère résolution détecteur.

**Acceptance Criteria:**

**Given** une image d'entrée canonique 1920×1080 grayscale
**When** `RESIZE` est appliqué
**Then** la taille cible est dérivée de la clé `image_size` de la config `JAX_DETECTOR` (jamais un littéral codé en dur, AD-12), avec la méthode validée en Story 8.1

**Given** le checkpoint `JAX_DETECTOR` (Epic 7)
**When** il est chargé pour la composition
**Then** il suit le fallback de chemin à 3 niveaux + ré-init `batch_stats` (AD-3 hérité), poids en lecture seule

**Given** l'image redimensionnée
**When** le détecteur figé est appelé
**Then** il produit un heatmap de centres + une carte de régression de taille au format défini en Story 7.1

### Story 8.3: Extraction de pics + Top-K

As a mainteneur du pipeline d'inférence,
I want extraire les boîtes candidates depuis le heatmap+taille via une extraction de pics JAX-native,
So that le décodage soit 100% JAX, sans `cv2.findContours` (AD-9).

**Acceptance Criteria:**

**Given** le heatmap de centres produit en Story 8.2
**When** l'extraction de pics est appliquée
**Then** elle utilise `jax.lax.reduce_window` (max-pool peak-NMS), jamais `cv2.findContours`/morphologie python

**Given** les pics extraits et leurs scores
**When** la sélection Top-K est appliquée
**Then** `jax.lax.top_k` retient au maximum 20 candidats (repère résolution détecteur), avec leurs tailles associées

**Given** une image contenant plus de 20 détections réelles au-dessus du seuil de confiance (formation dense)
**When** la sélection Top-K est appliquée
**Then** les 20 détections de plus haute confiance sont conservées et les autres sont écartées sans erreur — plafond silencieux assumé par conception, cohérent avec la limite déjà existante du pipeline `FIGHTERJET_DETECTION` actuel ("jusqu'à 20 avions"), pas une régression (décidé en party mode, Code Review Crew — Boundary/Grumbal/Dana)

**Given** AD-13 (source unique de stride)
**When** cette story et la Story 8.4 sont implémentées
**Then** les deux utilisent la même valeur nommée de stride/résolution de sortie du détecteur, définie une seule fois

### Story 8.4: RESCALE (repère détecteur → repère image d'origine)

As a mainteneur du pipeline d'inférence,
I want une étape `RESCALE` déterministe et symétrique du `RESIZE` d'entrée,
So that les coordonnées de boîte soient ramenées dans le repère de l'image source avant le crop (AD-13).

**Acceptance Criteria:**

**Given** les boîtes candidates au repère résolution détecteur (Story 8.3)
**When** `RESCALE` est appliqué
**Then** les coordonnées sont converties vers le repère image d'origine (1920×1080), avec la même valeur de stride nommée que la Story 8.3

**Given** le seuil de score de détection dérivant `valid_mask`
**When** `RESCALE` (ou l'étape immédiatement suivante) applique ce seuil
**Then** il est lu depuis la config `JAX_DETECTOR`, jamais une constante privée dupliquée dans `inference_utils.py` (AD-15)

**Given** les 20 slots (valides et invalides)
**When** `RESCALE` est complété
**Then** les slots invalides restent identifiables jusqu'à cette étape

### Story 8.5: CROP différentiable + appel classification figée

As a mainteneur du pipeline d'inférence,
I want recadrer chaque détection via `map_coordinates` puis appeler la classification figée sur le batch de crops,
So that chaque détection soit classée sans boucle de recadrage python ni cv2 (AD-11).

**Acceptance Criteria:**

**Given** les boîtes au repère image d'origine (Story 8.4) et l'image source pleine résolution
**When** le crop est appliqué
**Then** il utilise `jax.scipy.ndimage.map_coordinates` (ordre 1, bilinéaire), `vmap` sur les 20 slots fixes, avec la méthode validée en Story 8.1 (y compris le comportement en bord de cadre)

**Given** le checkpoint `FIGHTERJET_CLASSIFICATION`
**When** il est chargé pour la composition
**Then** il suit le même principe de chargement figé, lecture seule, que la Story 8.2 (aucun réentraînement, AD-14/NFR1)

**Given** le batch de crops 20×128×128
**When** la classification figée est appelée
**Then** elle retourne classes + class_scores pour les 20 slots (toujours remplis)

### Story 8.6: Assemblage final — build_single_pass_predict_fn

As a mainteneur du pipeline d'inférence,
I want assembler les étapes 8.2 à 8.5 en une unique fonction JIT-compilée dans `inference_utils.py`,
So that le graphe complet soit appelable en un seul point d'entrée, cohérent avec `build_predict_fn`/`build_clf_predict_fn` (AD-1 hérité, AD-16).

**Acceptance Criteria:**

**Given** les Stories 8.2 à 8.5 complétées
**When** `build_single_pass_predict_fn` est implémentée dans `inference_utils.py`
**Then** elle compose RESIZE→détecteur→pics/Top-K→RESCALE→CROP→classification en un seul callable, JIT-compilable de bout en bout

**Given** les configs `JAX_DETECTOR` et `FIGHTERJET_CLASSIFICATION`
**When** `build_single_pass_predict_fn` est construite
**Then** elle les lit via `get_dataset_config()` sans les modifier (AD-16) — ce n'est pas une entrée `DATASET_CONFIGS`

**Given** le contrat de sortie fixé par AD-15
**When** `build_single_pass_predict_fn` retourne son résultat
**Then** il s'agit exactement de `{boxes, classes, class_scores, detection_scores, valid_mask}`, 20 slots fixes, slots invalides à zéro

**Given** `decode_segmentation_and_detect(_batch)` et `non_max_suppression` (ancien pipeline)
**When** `build_single_pass_predict_fn` est créée
**Then** ces fonctions ne sont ni modifiées ni appelées par ce nouveau chemin (AD-20)

### Story 8.7: Migration bounding_boxes_with_classification_from_video_generation.py

As a mainteneur du pipeline vidéo,
I want que ce script utilise `build_single_pass_predict_fn` au lieu de l'orchestration manuelle actuelle,
So that le pipeline vidéo bénéficie de la composition unifiée, sans régression de débit temps réel (AD-6 hérité).

**Acceptance Criteria:**

**Given** la Story 8.6 complétée
**When** le script est migré
**Then** il appelle `build_single_pass_predict_fn` au lieu de l'enchaînement `decode_segmentation_and_detect_batch` + crop manuel + `predict_crops_batch`

**Given** une baseline de sortie capturée avant migration (boxes/classes/scores sur un jeu de vidéos fixe)
**When** le script migré est exécuté sur le même jeu
**Then** les résultats sont comparés à la baseline — tout écart documenté et justifié (ex. fusion de boîtes corrigée, FR3), pas silencieux

**Given** le débit temps réel prioritaire (AD-6 hérité)
**When** la migration est effectuée
**Then** aucune dégradation n'est introduite (comparaison avant/après sur un extrait vidéo)

**Given** AD-20
**When** ce script est migré
**Then** l'ancien chemin (`decode_segmentation_and_detect_batch`, etc.) reste disponible dans `inference_utils.py` et fonctionnel

### Story 8.8: Migration tools/bounding_boxes_with_classification_from_images_generation.py

As a mainteneur des outils d'inférence sur images,
I want que ce script utilise `build_single_pass_predict_fn`,
So that le traitement d'image statique bénéficie de la même composition unifiée.

**Acceptance Criteria:**

**Given** la Story 8.6 complétée
**When** le script est migré
**Then** il appelle `build_single_pass_predict_fn` au lieu de son orchestration actuelle

**Given** une baseline de sortie capturée avant migration
**When** le script migré est exécuté sur le même jeu d'images
**Then** les résultats sont comparés à la baseline, tout écart documenté et justifié

### Story 8.9: Validation finale — fusion de boîtes résolue + non-régression

As a mainteneur du pipeline,
I want valider que la fusion de boîtes est résolue sur un cas de formation serrée connu, et que l'ancien pipeline reste pleinement fonctionnel,
So that FR3 et AD-20/NFR3 soient prouvés par exécution réelle.

**Acceptance Criteria:**

**Given** un cas de formation serrée déjà identifié (ex. vidéo du 14 juillet, `notes-jax-single-pass.md`) où l'ancien pipeline fusionne deux avions en une seule détection
**When** ce cas est traité par `build_single_pass_predict_fn` (via les Stories 8.7/8.8)
**Then** deux détections distinctes sont produites structurellement (précision réelle sur le chevauchement encore limitée par le manque de données, AD-19/CAP-3)

**Given** `FIGHTERJET_DETECTION`, `AircraftDetectorUNet`, `DetectionStrategy`, `DetectionDataset` et leurs consommateurs (`tools/audit_dataset_detection.py`, `tools/boxes_process_manual_tkinter.py`)
**When** cette story est complétée
**Then** chacun est ré-exécuté et confirmé fonctionnel sans modification (AD-20)

**Given** les Stories 8.1 à 8.8 complétées
**When** cette story conclut l'Epic 8
**Then** FR2, FR3, FR4 sont tous confirmés couverts et prouvés par exécution

## Requirements Inventory — Epic Échecs (preuve de généralisation policy/value)

Source : `_bmad-output/planning-artifacts/prds/prd-jax_supervised_training-2026-07-27/prd.md` (status: final, issu du brief `brief-jax_supervised_training-2026-07-27`) et `_bmad-output/planning-artifacts/architecture/architecture-chess-2026-07-27/ARCHITECTURE-SPINE.md` (AD-21 à AD-25, hérite AD-3/AD-14/AD-17/AD-18 de la spine JAX Single-Pass 2026-07-15 comme contraintes read-only). Initiative distincte des Epics 1-8 — brancher un domaine échecs (sortie policy+value, ni classification ni détection) sur `Trainer`/`TaskStrategy`, comme test de généricité réelle du pattern, en parallèle/backlog du travail avion en cours, sans aucun impact sur `JAX_DETECTOR`.

### Functional Requirements

FR1 : Le pipeline construit un dataset de positions à partir d'archives PGN par joueur (pgnmentor.com/files.html#players), en rejouant chaque partie coup par coup via `python-chess` — une partie de N demi-coups produit N exemples.
FR2 : Chaque position est encodée en planes façon AlphaZero (pièces, trait, roques, répétition, coups légaux), enrichies de l'historique des 5 derniers coups des deux joueurs. Fenêtre fixe à 5 créneaux : une position de début de partie (moins de 5 demi-coups joués) n'est pas exclue du dataset — ses créneaux d'historique manquants sont remplis à zéro (padding), pour que le modèle apprenne aussi les ouvertures classiques.
FR3 : La policy head est entraînée à imiter le coup réellement joué, Blancs et Noirs confondus, sans filtrage par résultat de partie. La value head porte le résultat final de la partie du point de vue du joueur au trait (+1/0/-1, nulles = 0).
FR4 : Une nouvelle `TaskStrategy` gère l'entraînement conjoint de la tête policy et de la tête value (deux losses), sur le modèle du pattern Strategy existant.
FR5 : Un nouveau modèle implémente l'architecture envisagée : CNN 8×8 (convolutions + blocs résiduels) → bottleneck de tokens appris (Perceiver/TokenLearner-style) → auto-attention entre tokens → tête policy + tête value.
FR6 : Le domaine échecs est intégré sans modifier `Trainer` au-delà de ce que sa généricité actuelle permet déjà (nouveaux modèles/loss/évaluation acceptés ; changement structurel du cœur générique non accepté sans être documenté comme écart).
FR7 : Une story dédiée valide, par exécution réelle (pas par lecture de code), que `JAX_DETECTOR` produit un comportement identique avant et après l'intégration du domaine échecs.

### NonFunctional Requirements

NFR1 — Non-régression `JAX_DETECTOR` : contrainte dure, non négociable. `JAX_DETECTOR` continue de fonctionner sans aucun impact de cette epic. Validé par exécution réelle (FR7), pas par lecture de code.
NFR2 — Aucune dépendance à un moteur d'échecs externe pour la génération de labels : ni pour la policy, ni pour la value, y compris en repli en cas de value head bruitée.
NFR3 — Pas de complexification inutile du pipeline générique : `Trainer`/`TaskStrategy` absorbent le domaine échecs par les mêmes points d'extension que les domaines précédents.

### Additional Requirements (Architecture)

- **AD-21** : Non-régression — `JAX_DETECTOR` et toute sa chaîne restent utilisables de bout en bout sans modification fonctionnelle, pendant et après cette epic. Validé par diff à une baseline capturée avant l'epic. Précédent direct d'AD-20 (parent), qui protège l'ancien pipeline segmentation mais pas `JAX_DETECTOR`.
- **AD-22** : Policy à espace de sortie fixe façon AlphaZero, sans masquage des coups illégaux cette epic (repoussé au futur projet d'intégration `chess_game.py`). Taille de l'espace à source unique : constante `NUM_MOVES` dans `chess_target_encoding.py`, reprise par `num_classes` de l'entrée `CHESS` (`dataset_configs.py`) — réutilise la plomberie `model_kwargs` existante (`main.py:110`), aucun nouveau kwarg.
- **AD-23** : Bottleneck par tokens à requêtes apprises (Perceiver/TokenLearner-style), pas par pooling par groupe de canaux. Auto-attention standard entre les tokens (pas de biais géométrique cette epic, différé).
- **AD-24** : `TaskStrategy` échecs — loss composite zero-touch sur `Trainer` (`policy_weight*policy_loss + value_weight*value_loss`, exactement sur le modèle de `compute_centernet_loss`). Détail policy/value visible uniquement en fin d'entraînement via `generate_reports()`, jamais dans le log epoch-par-epoch. `primary_metric_name = PolicyAccuracy` (top-1), gate la sauvegarde du best model. Value head = sortie scalaire (`Dense(1)`+`tanh`), loss de régression (MSE).
- **AD-25** : Format d'exemple dataset minimal (position + policy target + value target uniquement, aucune métadonnée cadence/ECO). Aucune dépendance à un moteur d'échecs externe (NFR2). Le producteur (`dataset_builder/chess_pgn_dataset_tools.py`) calcule et fige le signe de la value une seule fois ; le consommateur ne le re-dérive jamais.
- **Hérité (read-only)** : AD-3 (fallback checkpoint 3 niveaux), AD-14 (entraînement toujours modulaire), AD-17 (nouveau `task_type="chess_policy_value"` = nouvelle classe `TaskStrategy` + nouvelle classe chargeur dédiées, référencé identiquement aux 2 points de dispatch réels `main.py`/`data_management.py`), AD-18 (module d'échange partagé unique `chess_target_encoding.py`, miroir de `detection_target_encoding.py`).
- **Naming** : `task_type="chess_policy_value"`, `model_name="chess_cnn_attention_policy_value"` (registry `model_library.MODELS`), entrée `dataset_configs.py`=`CHESS`.
- **Stack** : aucune nouvelle dépendance externe — `flax.linen` 0.10.7 (déjà présent, fournit `nn.MultiHeadDotProductAttention`) et `chess` (python-chess) 1.11.2 (déjà présent, déjà utilisé par `chess/chess_game.py`).

### UX Design Requirements

N/A — aucun contrat UX applicable. Projet solo, un seul opérateur, aucune interface modifiée (l'intégration dans `chess/chess_game.py` est un Non-Goal explicite de cette epic).

### FR Coverage Map

FR1: Epic 9 — Extraction du dataset depuis les archives PGN
FR2: Epic 9 — Encodage de l'input (planes AlphaZero + historique)
FR3: Epic 9 — Construction des labels policy et value
FR4: Epic 9 — Nouvelle `TaskStrategy` à double tête
FR5: Epic 9 — Nouveau modèle échecs (CNN + bottleneck tokens + attention)
FR6: Epic 9 — Intégration sans modification structurelle de `Trainer`
FR7: Epic 9 — Validation de non-régression `JAX_DETECTOR` par exécution

## Epic List

### Epic 9: Généralisation du pipeline via un domaine échecs (policy+value)

Un nouveau domaine (échecs) est branché sur `Trainer`/`TaskStrategy` — nouveau modèle, nouveau dataset, nouvelle `TaskStrategy` à double tête — sans modification structurelle de `Trainer` et sans aucun impact sur `JAX_DETECTOR`. Standalone : preuve de généricité du pattern, ne dépend d'aucun epic futur.
**FRs covered:** FR1, FR2, FR3, FR4, FR5, FR6, FR7
**Note d'implémentation :** séquence de stories dictée par le Structural Seed du spine — (1) module d'échange partagé `chess_target_encoding.py` + builder dataset `dataset_builder/chess_pgn_dataset_tools.py` (FR1-3, AD-18/AD-25, story "auteur unique" sur le même principe que la Story 0 des Epics 1/7) ; (2) nouveau modèle `chess_cnn_attention_policy_value` (FR5, AD-22/AD-23) ; (3) `ChessPolicyValueStrategy` + loss composite (FR4, FR6, AD-24) ; (4) validation de non-régression `JAX_DETECTOR` par exécution, story de clôture (FR7, AD-21).

## Epic 9: Généralisation du pipeline via un domaine échecs (policy+value)

Un nouveau domaine (échecs) est branché sur `Trainer`/`TaskStrategy` — nouveau modèle, nouveau dataset, nouvelle `TaskStrategy` à double tête — sans modification structurelle de `Trainer` et sans aucun impact sur `JAX_DETECTOR`. Standalone : preuve de généricité du pattern, ne dépend d'aucun epic futur. **FRs covered:** FR1, FR2, FR3, FR4, FR5, FR6, FR7

### Story 9.1: Module d'échange partagé + builder dataset PGN

As a mainteneur du pipeline,
I want un module `chess_target_encoding.py` définissant le schéma position/policy/value et un outil `dataset_builder/chess_pgn_dataset_tools.py` qui construit le dataset depuis des archives PGN,
So that le domaine échecs dispose d'une source unique de vérité pour le format d'échange (AD-18) avant que tout consommateur ne soit développé.

**Acceptance Criteria:**

**Given** des archives PGN pgnmentor (par joueur)
**When** `chess_pgn_dataset_tools.py` les traite via `python-chess`
**Then** chaque partie de N demi-coups produit N exemples (position, policy target, value target), y compris les positions de début de partie (aucun filtrage par ancienneté d'historique) — FR1

**Given** une position à un demi-coup donné
**When** elle est encodée par `chess_target_encoding.py`
**Then** elle produit des planes façon AlphaZero + historique des 5 derniers coups des deux joueurs — FR2

**Given** une position de début de partie avec moins de 5 demi-coups joués
**When** ses créneaux d'historique manquants sont encodés
**Then** ils sont remplis à zéro (padding), jamais exclus ni remplacés par une répétition de la position de départ — FR2 (précision Aymeric 2026-07-27)

**Given** un coup joué à une position
**When** la policy target est construite
**Then** c'est l'index du coup réellement joué (Blancs et Noirs confondus), sans filtrage par résultat de partie — FR3

**Given** le résultat final d'une partie
**When** la value target est construite pour une position donnée
**Then** c'est +1/0/-1 du point de vue du joueur au trait à cette position, nulles = 0, calculé et figé une seule fois par le producteur — AD-25

**Given** AD-25/NFR2
**When** le dataset est généré
**Then** aucun moteur d'échecs externe (Stockfish ou autre) n'est utilisé — seul `python-chess` sert à la légalité/au rejeu

**Given** AD-18/AD-22
**When** `chess_target_encoding.py` est créé
**Then** il expose une constante nommée unique `NUM_MOVES` (taille de l'espace policy), importée par tout consommateur futur — jamais un littéral dupliqué

### Story 9.2: Nouveau modèle échecs

As a mainteneur du pipeline de modèles,
I want un nouveau modèle `chess_cnn_attention_policy_value` enregistré dans `model_library.py`,
So that le domaine échecs dispose d'une architecture CNN+bottleneck de tokens appris+attention+têtes policy/value, dimensionnée par le schéma d'échange de la Story 9.1.

**Acceptance Criteria:**

**Given** `chess_target_encoding.py` (Story 9.1) et sa constante `NUM_MOVES`
**When** le modèle `chess_cnn_attention_policy_value` est implémenté
**Then** sa tête policy produit un vecteur de taille `NUM_MOVES` (source unique, pas un littéral dupliqué) — AD-22

**Given** l'architecture envisagée
**When** le modèle est construit
**Then** il enchaîne CNN 8×8 (convolutions + blocs résiduels) → bottleneck de K tokens par requêtes apprises (cross-attention, Perceiver/TokenLearner-style) → auto-attention standard entre les tokens (sans biais géométrique) → tête policy + tête value — FR5, AD-23

**Given** la tête value
**When** elle est implémentée
**Then** elle produit un scalaire (`Dense(1)`+`tanh`, borne [-1, 1]) — AD-24

**Given** `model_library.get_model()`
**When** `chess_cnn_attention_policy_value` est appelé avec les `model_kwargs` génériques (`num_classes`, `dropout_rate`)
**Then** il s'instancie sans modification de `get_model()` ni de `main.py` au-delà du dispatch déjà prévu

### Story 9.3: `TaskStrategy` échecs, loss composite et intégration au pipeline

As a mainteneur du pipeline d'entraînement,
I want une nouvelle `ChessPolicyValueStrategy` et une entrée `CHESS`/`task_type="chess_policy_value"` dispatchée dans `main.py`/`data_management.py`,
So that le domaine échecs s'entraîne de bout en bout via `Trainer`, sans modification structurelle de celui-ci.

**Acceptance Criteria:**

**Given** le modèle (Story 9.2) et le dataset (Story 9.1)
**When** `compute_loss` est appelé
**Then** il retourne une loss unique `policy_weight * policy_loss + value_weight * value_loss` (poids dans `loss_params`, exactement sur le modèle de `compute_centernet_loss`) — AD-24, FR4

**Given** `trainer.py`
**When** cette story est complétée
**Then** aucune ligne de `trainer.py` n'est modifiée — FR6

**Given** `generate_reports()`
**When** l'entraînement se termine
**Then** le détail `policy_loss`/`value_loss` est affiché séparément, en réutilisant `self.loss_params` et les mêmes sous-fonctions de `loss_functions.py` (pas de duplication) — AD-24

**Given** `primary_metric_name`/`optimization_mode`
**When** le meilleur modèle est sauvegardé
**Then** c'est la policy accuracy (top-1) qui gate la sauvegarde, `optimization_mode="max"` — AD-24

**Given** `task_type="chess_policy_value"` et l'entrée `CHESS` (`dataset_configs.py`)
**When** `main.py` et `data_management.py` sont mis à jour
**Then** le dispatch suit exactement le pattern des 2 points de dispatch réels existants (AD-17 hérité), sans branche conditionnelle sur une classe existante

**Given** AD-17
**When** cette story est complétée
**Then** aucune modification n'est faite à `ClassificationStrategy`/`DetectionStrategy`/`CenterNetDetectionStrategy`/`KeplerStrategy` ni à leurs classes de chargeur associées

### Story 9.4: Validation non-régression `JAX_DETECTOR` par exécution

As a mainteneur du pipeline,
I want valider par exécution réelle que `JAX_DETECTOR` fonctionne à l'identique avant et après l'intégration du domaine échecs,
So that NFR1/AD-21 (contrainte dure) est prouvé, pas supposé.

**Acceptance Criteria:**

**Given** une baseline (boxes/classes/scores sur un set fixe d'images) capturée avant le début de cette epic
**When** `JAX_DETECTOR` est ré-exécuté (entraînement et/ou inférence) après les Stories 9.1 à 9.3
**Then** les résultats sont comparés à la baseline par diff — tout écart documenté et justifié, jamais silencieux — FR7

**Given** `CenterNetDetectionStrategy`, `CenterNetDetectionDataset`, `aircraft_detector_centernet(_lite)` et leurs consommateurs actuels
**When** cette story est complétée
**Then** aucun n'a été modifié par cette epic — AD-21

**Given** les Stories 9.1 à 9.3 complétées
**When** cette story conclut l'Epic 9
**Then** FR1 à FR6 sont confirmés couverts et le domaine échecs est entraînable de bout en bout via `Trainer`
