# Tableau récapitulatif — `dataset_configs.py`

Généré le 2026-09-04 à partir de la lecture complète de `dataset_configs.py` (11 configs).
Objectif : identifier les paramètres réellement communs à toutes les configs (candidats à un socle générique)
vs les paramètres propres à un domaine ou à une seule config (code spécifique nécessaire).

Légende : `—` = clé absente de la config (pas juste vide). `Nb` = nombre de configs (sur 11) où la clé est présente.

Colonnes (dans l'ordre du fichier) :
`FJ_CLS`=FIGHTERJET_CLASSIFICATION · `FJ_DET`=FIGHTERJET_DETECTION · `KEPLER`=JAX_KEPLER · `CIFAR10`=CIFAR10 ·
`JAX_DET`=JAX_DETECTOR · `CH_NOHIST`=CHESS_NO_HISTORY · `CH_LEGAL`=CHESS_LEGAL_MOVES · `CH_TEACH`=CHESS_SEARCH_TEACHER ·
`CH_MVTOK`=CHESS_MOVE_TOKEN · `CH_TOKEN`=CHESS_TOKEN · `CH_TOK1MV`=CHESS_TOKEN_1_MOVE

---

## A. Identité / domaine

| Paramètre | FJ_CLS | FJ_DET | KEPLER | CIFAR10 | JAX_DET | CH_NOHIST | CH_LEGAL | CH_TEACH | CH_MVTOK | CH_TOKEN | CH_TOK1MV | Nb |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `task_type` | — *(défaut "classification")* | `detection` | `kepler` | — *(défaut "classification")* | `detection_centernet` | `chess_policy_value` | `chess_legal_moves` | `chess_policy_value` | `chess_move_token` | `chess_token` | `chess_token_1_move` | 9 |

---

## B. Données

| Paramètre | FJ_CLS | FJ_DET | KEPLER | CIFAR10 | JAX_DET | CH_NOHIST | CH_LEGAL | CH_TEACH | CH_MVTOK | CH_TOKEN | CH_TOK1MV | Nb |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `num_classes` **(requis)** | 32 | 1 | 2 | 10 | 1 | 4672 | 4672 *(taille sortie, pas des classes)* | 4672 | 4672 | 50 *(=MAX_CANDIDATES, détourné)* | -1 *(sentinelle, non utilisée)* | 11 |
| `class_names` | 32 classes avions (a10…v22) | `['aircraft']` | `['no_exoplanet','exoplanet']` | 10 classes CIFAR (airplane…truck) | `['aircraft']` | — | — | — | — | — | — | 5 |
| `num_channels` | — | — | — | — | — | 19 | 29 | 29 | — | — | — | 3 |
| `output_prefix` | `.../chunks/classification/dataset_classification` | `.../chunks/detection/dataset_detection` | `./data/chunks/kepler/dataset_kepler` | `.../chunks/cifar10/dataset_cifar10` | `.../chunks/jax_detector/jax_detector_targets` | `.../chunks/chess_no_history/chess` | `.../chunks/chess_legal_moves/chess_legal_moves` | `.../chunks/chess_search_teacher/chess_search_teacher` | `.../chess_move_token_spike/....npz` *(fichier littéral)* | `.../chess_token_candidate_spike/....npz` *(fichier littéral)* | même fichier que CH_TOKEN | 11 |
| `data_dir` *(source brute pré-chunking)* | `/home/aobled/Downloads/_balanced_dataset_split` | — | — | — | — | — | — | — | — | — | — | 1 |
| `chunk_size` | 30000 | — | — | — | 11000 | — | — | — | — | — | — | 2 |
| `image_size` | (128,128) | (224,224) | (3197,1) | (32,32) | (224,224) | — | — | — | — | — | — | 5 |
| `grayscale` | True | True | True | False | True | — | — | — | — | — | — | 5 |
| `input_shape` **(requis)** | (128,128,1) | (224,224,1) | (3197,1) | (32,32,3) | (224,224,1) | (8,8,19) | (8,8,29) | (8,8,29) | (1,) *(dummy init)* | (120,) *(dummy init)* | (71,) *(dummy init)* | 11 |
| `max_boxes` | — | 20 | — | — | 20 | — | — | — | — | — | — | 2 |
| `detection_score_threshold` | — | — | — | — | 0.17 | — | — | — | — | — | — | 1 |
| `nms_iou_threshold` | — | — | — | — | 0.5 | — | — | — | — | — | — | 1 |
| `zoom_augment_probability` | — | — | — | — | 0.0 | — | — | — | — | — | — | 1 |
| `val_split` | — | — | — | — | — | 0.1 | 0.1 | 0.1 | 0.1 | 0.1 | 0.1 | 6 |
| `mean` / `std` | None / None | — | — | None / None | — | — | — | — | — | — | — | 2 |
| `mean_std_path` | oui | — | — | oui | — | — | — | — | — | — | — | 2 |
| `metric_threshold` | — | — | — | — | — | — | 0.3 | — | — | — | — | 1 |

---

## C. Augmentation (`augmentation_params.*`)

| Paramètre | FJ_CLS | FJ_DET | KEPLER | CIFAR10 | JAX_DET | CH_* (6 configs) | Nb |
|---|---|---|---|---|---|---|---|
| `augmentation_params` (bloc) | oui | oui | oui | oui | oui | — (absent partout) | 5 |
| `flip_h` | True | True | False | True | True | — | 5 |
| `flip_v` | False | True | False | False | True | — | 5 |
| `rotation_factor` | 0.12 | 0.0 | 0.0 | 0.0 | 0.0 | — | 5 |
| `zoom_factor` | 0.10 | 0.35 | 0.0 | 0.0 | 0.35 | — | 5 |
| `translation_factor` | 0.06 | 0.25 | 0.0 | 0.1 | 0.25 | — | 5 |
| `brightness_delta` | 0.10 | 0.15 | 0.0 | 0.0 | 0.15 | — | 5 |
| `contrast_factor` | 0.10 | 0.30 | 0.0 | 0.0 | 0.30 | — | 5 |
| `pixelation_factor` | 4.0 | — | — | — | — | — | 1 |

→ `augmentation_params` est un bloc **cohérent à 100%** entre les 5 configs qui l'utilisent (mêmes 7 sous-clés, sauf `pixelation_factor` propre à FJ_CLS). C'est un bon candidat "commun" — mais uniquement pour le sous-groupe image/1D, jamais pour les échecs (pas d'augmentation spatiale sur un plateau).

---

## D. Modèle

| Paramètre | FJ_CLS | FJ_DET | KEPLER | CIFAR10 | JAX_DET | CH_NOHIST | CH_LEGAL | CH_TEACH | CH_MVTOK | CH_TOKEN | CH_TOK1MV | Nb |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `model_name` **(requis)** | sophisticated_cnn_128_lite | aircraft_detector_unet | kepler_1d_cnn | sophisticated_cnn_32_plus | aircraft_detector_centernet | chess_cnn_attention_policy_value | chess_cnn_attention_legal_moves | chess_cnn_attention_policy_value | chess_move_token_transformer | chess_token_candidate_model | chess_token_one_move_model | 11 |
| `grid_size` | — | 224 | — | — | — | — | — | — | — | — | — | 1 |
| `heatmap_prior` | — | — | — | — | 0.0000268 | — | — | — | — | — | — | 1 |
| `num_bottleneck_tokens` | — | — | — | — | — | 8 | 8 | 16 | — | — | — | 3 |
| `token_dim` | — | — | — | — | — | — | — | 196 | — | 128 | 64 | 3 |
| `num_layers` | — | — | — | — | — | — | — | — | 6 | — | — | 1 |
| `d_model` | — | — | — | — | — | — | — | — | 192 | — | — | 1 |
| `num_heads` | — | — | — | — | — | — | — | — | 4 | — | 4 | 2 |
| `num_trunk_layers` | — | — | — | — | — | — | — | — | — | — | 2 | 1 |

---

## E. Loss

| Paramètre | FJ_CLS | FJ_DET | KEPLER | CIFAR10 | JAX_DET | CH_NOHIST | CH_LEGAL | CH_TEACH | CH_MVTOK | CH_TOKEN | CH_TOK1MV | Nb |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `loss_method` | `focal_loss` | `segmentation` | — | `cross_entropy` | — *(1 seule loss, pas de dispatch)* | — | — | — | — | — | — | 3 |
| `loss_params` (bloc, présent partout même vide) | `{gamma}` | `{bce_weight, dice_weight, false_positive_penalty}` | `{}` *(implicite, absent en fait)* | `{}` | `{heatmap_weight, size_weight, alpha, beta}` | `{policy_weight, value_weight}` | `{pos_weight}` | `{policy_weight, value_weight, label_smoothing}` | `{label_smoothing}` | `{label_smoothing}` | `{from_square_weight, move_type_weight, label_smoothing}` | 10 |
| ↳ `.gamma` | 2.0 | — | — | — | — | — | — | — | — | — | — | 1 |
| ↳ `.bce_weight` | — | 0.3 | — | — | — | — | — | — | — | — | — | 1 |
| ↳ `.dice_weight` | — | 0.7 | — | — | — | — | — | — | — | — | — | 1 |
| ↳ `.false_positive_penalty` | — | 2.0 | — | — | — | — | — | — | — | — | — | 1 |
| ↳ `.policy_weight` | — | — | — | — | — | 1.0 | — | 1.0 | — | — | — | 2 |
| ↳ `.value_weight` | — | — | — | — | — | 1.0 | — | 0.0 | — | — | — | 2 |
| ↳ `.pos_weight` | — | — | — | — | — | — | 1.0 | — | — | — | — | 1 |
| ↳ `.heatmap_weight` | — | — | — | — | 1.0 | — | — | — | — | — | — | 1 |
| ↳ `.size_weight` | — | — | — | — | 0.1 | — | — | — | — | — | — | 1 |
| ↳ `.alpha` | — | — | — | — | 2.0 | — | — | — | — | — | — | 1 |
| ↳ `.beta` | — | — | — | — | 4.0 | — | — | — | — | — | — | 1 |
| ↳ `.label_smoothing` | — | — | — | — | — | — | — | 0.2 | 0.2 | 0.2 | 0.0 | 4 |
| ↳ `.from_square_weight` | — | — | — | — | — | — | — | — | — | — | 1.0 | 1 |
| ↳ `.move_type_weight` | — | — | — | — | — | — | — | — | — | — | 1.0 | 1 |
| `label_smoothing` **(top-level, hors loss_params)** | 0.15 | — | — | — | — | — | — | — | — | — | — | 1 |
| `mixup_alpha` **(top-level, hors loss_params)** | 0.05 | — | — | — | — | — | — | — | — | — | — | 1 |

⚠️ **Incohérence repérée** : `label_smoothing` existe à **3 emplacements différents** selon la config — top-level (FJ_CLS), niché dans `tpu`/`gpu` (KEPLER, valeur 0.0), niché dans `loss_params` (CH_TEACH/CH_MVTOK/CH_TOKEN/CH_TOK1MV). Même chose pour `mixup_alpha` (top-level FJ_CLS vs niché tpu/gpu KEPLER). Aucun code ne semble lire un emplacement unique — à vérifier avant toute unification.

---

## F. Hyperparamètres TPU (`tpu.*`)

| Paramètre | FJ_CLS | FJ_DET | KEPLER | CIFAR10 | JAX_DET | CH_NOHIST | CH_LEGAL | CH_TEACH | CH_MVTOK | CH_TOKEN | CH_TOK1MV | Nb |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `tpu` (bloc) **(requis)** | oui | oui | oui | oui | oui | oui | oui | oui | oui | oui | oui | 11 |
| ↳ `micro_batch_size` **(requis)** | 128 | 128 | 128 | 128 | 128 | 128 | 128 | 128 | 256 | 256 | 256 | 11 |
| ↳ `accum_steps` **(requis)** | 4 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 11 |
| ↳ `learning_rate` | 8e-3 | 4e-4 | 1e-4 | 1e-3 | 4e-4 | 4e-4 | 4e-4 | 4e-4 | 3e-4 | 3e-4 | 3e-4 | 11 |
| ↳ `weight_decay` | 5e-5 | 5e-5 | 1e-4 | 5e-5 | 5e-5 | 5e-5 | 5e-5 | 5e-5 | 5e-5 | 5e-5 | 5e-5 | 11 |
| ↳ `dropout_rate` | 0.0 | 0.0 | 0.3 | 0.3 | 0.05 | 0.1 | 0.1 | 0.35 | 0.3 | 0.3 | 0.1 | 11 |
| ↳ `warmup_steps` | — *(auto-calc ratio)* | 400 | — *(auto-calc)* | 200 | — *(auto-calc)* | 200 | 200 | 200 | 200 | 200 | 200 | 8 |
| ↳ `decay_steps` | — *(auto-calc)* | 22000 | — *(auto-calc)* | — *(auto-calc)* | — *(auto-calc)* | 10000 | 76700 | 51950 | 122425 | 218700 | 218700 | 7 |
| ↳ `label_smoothing` *(niché, hors norme)* | — | — | 0.0 | — | — | — | — | — | — | — | — | 1 |
| ↳ `mixup_alpha` *(niché, hors norme)* | — | — | 0.0 | — | — | — | — | — | — | — | — | 1 |

## G. Hyperparamètres GPU (`gpu.*`) — mêmes sous-clés

| Paramètre | FJ_CLS | FJ_DET | KEPLER | CIFAR10 | JAX_DET | CH_NOHIST | CH_LEGAL | CH_TEACH | CH_MVTOK | CH_TOKEN | CH_TOK1MV | Nb |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `gpu` (bloc) **(requis)** | oui | oui | oui | oui | oui | oui | oui | oui | oui | oui | oui | 11 |
| ↳ `micro_batch_size` | 128 | 16 | 32 | 128 | 64 | 256 | 256 | 256 | 64 | 256 | 256 | 11 |
| ↳ `accum_steps` | 4 | 2 | 4 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 11 |
| ↳ `learning_rate` | 8e-4 | 2e-4 | 1e-4 | 1e-3 | 2e-4 | 8e-4 | 8e-4 | 8e-4 | 3e-4 | 3e-4 | 3e-4 | 11 |
| ↳ `weight_decay` | 5e-5 | 5e-5 | 1e-4 | 5e-5 | 5e-5 | 5e-5 | 5e-5 | 5e-5 | 5e-5 | 5e-5 | 5e-5 | 11 |
| ↳ `dropout_rate` | 0.0 | 0.05 | 0.3 | 0.3 | 0.1 | 0.1 | 0.1 | 0.35 | 0.3 | 0.3 | 0.1 | 11 |
| ↳ `warmup_steps` | — | 400 | — | 200 | — | 200 | 200 | 200 | 200 | 200 | 200 | 8 |
| ↳ `decay_steps` | — | 105280 | — | — | — | 36700 | 38300 | 25975 | 489725 | 218700 | 218700 | 7 |
| ↳ `label_smoothing` *(niché)* | — | — | 0.0 | — | — | — | — | — | — | — | — | 1 |
| ↳ `mixup_alpha` *(niché)* | — | — | 0.0 | — | — | — | — | — | — | — | — | 1 |

→ `micro_batch_size`, `accum_steps`, `learning_rate`, `weight_decay`, `dropout_rate` sont **universels dans les 2 blocs, à 100% des configs** (11/11) : ce sont les seuls hyperparamètres réellement communs à tout le fichier avec `optimizer`/`lr_schedule`/`epochs`/`patience`/`input_shape`/`num_classes`/`model_name`/`output_prefix`/`loss_params`.
→ `warmup_steps`/`decay_steps` sont un **cas à part** : présents en dur (8/7 sur 11) *seulement* là où l'auto-calcul (`trainer.py::_resolve_lr_schedule_steps`) est soit désactivé (échecs — glob de fichiers qui ne matche jamais), soit délibérément écarté par un choix empirique validé (CIFAR10.warmup_steps=200, FJ_DET). Absents (auto-calculés) sur FJ_CLS, KEPLER, JAX_DET.

---

## H. Entraînement

| Paramètre | FJ_CLS | FJ_DET | KEPLER | CIFAR10 | JAX_DET | CH_NOHIST | CH_LEGAL | CH_TEACH | CH_MVTOK | CH_TOKEN | CH_TOK1MV | Nb |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `optimizer` | adamw | adamw | adamw | adamw | adamw | adamw | adamw | adamw | adamw | adamw | adamw | 11 |
| `lr_schedule` | cosine | cosine | cosine | cosine | cosine | cosine | cosine | cosine | cosine | cosine | cosine | 11 |
| `epochs` | 40 | 8 | 30 | 45 | 25 | 15 | 15 | 25 | 25 | 50 | 50 | 11 |
| `patience` | 5 | 5 | 5 | 8 | 8 | 8 | 8 | 8 | 8 | 8 | 8 | 11 |
| `decay_epochs` | — *(auto 70%)* | — | — | 45 *(explicite, retour arrière)* | — | — | — | — | — | — | — | 1 |

→ `optimizer` et `lr_schedule` valent **littéralement la même chose (`adamw`/`cosine`) sur les 11 configs, sans exception** — ce sont les meilleurs candidats à basculer en défaut codé plutôt qu'en clé de config répétée 11 fois.

---

## I. Évaluation

| Paramètre | FJ_CLS | FJ_DET | KEPLER | CIFAR10 | JAX_DET | CH_NOHIST | CH_LEGAL | CH_TEACH | CH_MVTOK | CH_TOKEN | CH_TOK1MV | Nb |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `metric_method` | accuracy | segmentation_iou | accuracy | accuracy | — | — | — | — | — | — | — | 4 |
| `report_method` | confusion_matrix | segmentation_heatmap | lightcurves | confusion_matrix | — | — | — | — | — | — | — | 4 |
| `eval_batch_size` | 128 | 16 | — | 128 | 16 | 64 | 64 | 64 | 64 | — | — | 8 |
| `eval_use_subset` | True | — | False | False | — | — | — | — | — | — | — | 3 |
| `eval_max_subset` | 100000 | — | — | — | — | — | — | — | — | — | — | 1 |
| `vis_freq` | — | 5 | — | — | 5 | — | — | — | — | — | — | 2 |

---

## J. Sauvegarde

| Paramètre | FJ_CLS | FJ_DET | KEPLER | CIFAR10 | JAX_DET | CH_NOHIST | CH_LEGAL | CH_TEACH | CH_MVTOK | CH_TOKEN | CH_TOK1MV | Nb |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `checkpoint_path` | best_model.pkl | best_model_detection.pkl | best_model_kepler.pkl | — *(dérivé du nom)* | — *(dérivé)* | — *(dérivé)* | — *(dérivé)* | — *(dérivé)* | — *(dérivé)* | — *(dérivé)* | — *(dérivé)* | 3 |
| `training_state_path` | best_model_training_state_classification.pkl | best_model_training_state_detection.pkl | best_model_training_state_kepler.pkl | — | — | — | — | — | — | — | — | 3 |
| `confusion_matrix_path` | confusion_matrix.png | — | confusion_matrix_kepler.png | confusion_matrix_cifar10.png | — | — | — | — | — | — | — | 3 |
| `save_dir` | — | ./checkpoints_detection | — | — | ./checkpoints_jax_detector | ./checkpoints_chess_no_history | ./checkpoints_chess_legal_moves | ./checkpoints_chess_search_teacher | ./checkpoints_chess_move_token | ./checkpoints_chess_token | ./checkpoints_chess_token_1_move | 8 |

---

## Synthèse pour le ménage

### 1. Paramètres réellement communs aux 11 configs (socle générique fiable)
`num_classes`, `input_shape`, `model_name`, `output_prefix`, `loss_params` (le bloc, pas son contenu),
`optimizer` (toujours `"adamw"`), `lr_schedule` (toujours `"cosine"`), `epochs`, `patience`,
`tpu`/`gpu` (blocs), et dans ces blocs : `micro_batch_size`, `accum_steps`, `learning_rate`, `weight_decay`, `dropout_rate`.

→ `optimizer`/`lr_schedule` n'ont **jamais varié une seule fois sur 11 configs** : candidats évidents à un défaut, pas une clé à répéter.

### 2. Paramètres partagés par sous-groupe de domaine (pas par tout le fichier)
- **Domaine "image/1D" (5 configs : FJ_CLS, FJ_DET, KEPLER, CIFAR10, JAX_DET)** : `class_names`, `image_size`, `grayscale`, `augmentation_params` (bloc cohérent à 7 sous-clés). Absent à 100% des 6 configs échecs.
- **Domaine "échecs" (6 configs : tous les CH_*)** : `val_split` (toujours 0.1, sans exception). Absent des 5 configs image/kepler.
- **Domaine "classification/segmentation avec dispatch" (FJ_CLS, FJ_DET, KEPLER, CIFAR10)** : `metric_method`/`report_method`/`loss_method`. Absents de JAX_DET (CenterNet, une seule stratégie) et de tous les CH_* (une seule stratégie par task_type).

### 3. Paramètres uniques à une seule config (code spécifique nécessaire)
`data_dir`, `grid_size`, `heatmap_prior`, `detection_score_threshold`, `nms_iou_threshold`, `zoom_augment_probability`,
`metric_threshold`, `pixelation_factor`, `eval_max_subset`, `num_layers`, `d_model`, `num_trunk_layers`, `decay_epochs` (explicite).
Ce sont les meilleurs signaux que ces domaines ne se généraliseront pas facilement — chacun porte une mécanique de modèle/loss propre.

### 4. Incohérences structurelles repérées (à trancher avant toute unification)
- **`label_smoothing`/`mixup_alpha` à 3 emplacements différents** selon la config : top-level (FJ_CLS), niché `tpu`/`gpu` (KEPLER), niché `loss_params` (4 configs échecs). Aucune règle unique de lecture visible dans ce fichier seul — à vérifier côté `trainer.py`/`task_strategies.py` avant de choisir un seul emplacement.
- **`checkpoint_path`/`training_state_path` vs `save_dir`** : deux mécanismes de nommage coexistent (chemin explicite pour 3 anciennes configs vs dérivation automatique depuis `dataset_name` pour les 8 autres, Story 5.0). Les 3 anciennes n'ont jamais été migrées vers la dérivation auto.
- **`warmup_steps`/`decay_steps` en dur vs auto-calculés** : la moitié des configs s'appuie sur l'auto-calcul de `trainer.py`, l'autre moitié a une valeur figée — pour les 6 configs échecs, ce n'est pas un choix mais une limitation connue (`_count_real_train_samples` ne matche jamais les fichiers échecs à flux unique, cf. commentaires 2026-08-07/08-11).
- **`num_classes` détourné 2 fois** : `CHESS_TOKEN` (=50, en fait `MAX_CANDIDATES`) et `CHESS_TOKEN_1_MOVE` (=-1, sentinelle inutilisée) — la clé "requise" par `validate_config` ne porte plus un vrai nombre de classes dans ces 2 cas.
