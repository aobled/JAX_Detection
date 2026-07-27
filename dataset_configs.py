"""
Configuration centralisée pour tous les datasets
Permet de gérer facilement plusieurs datasets avec leurs paramètres spécifiques
"""

import os

import numpy as np

# Racine des datasets chunkés (.npz). Local par défaut ; sur Colab, définir
# JAX_SUPERVISED_TRAINING_DATA_ROOT (ex: /content/drive/MyDrive/jax_supervised_training/data)
# AVANT d'importer ce module.
DATA_ROOT = os.environ.get("JAX_SUPERVISED_TRAINING_DATA_ROOT", "/home/aobled/Documents/data")


def validate_config(config_name, config):
    """Valide la cohérence d'une configuration de dataset"""
    errors = []
    
    # Vérifier que num_classes correspond à len(class_names)
    if "num_classes" in config and "class_names" in config:
        if len(config["class_names"]) != config["num_classes"]:
            errors.append(f"num_classes ({config['num_classes']}) != len(class_names) ({len(config['class_names'])})")
    
    # Vérifier que image_size est un tuple de 2 entiers
    if "image_size" in config:
        if not isinstance(config["image_size"], tuple) or len(config["image_size"]) != 2:
            errors.append(f"image_size doit être un tuple (H, W)")
    
    # Vérifier que les paramètres requis sont présents
    required = ["num_classes", "image_size", "model_name"]
    for key in required:
        if key not in config:
            errors.append(f"Paramètre requis manquant: {key}")

    # Vérifier que les blocs backend tpu/gpu sont présents et numériquement valides
    # (main.py/trainer.py/print_config y accèdent tous par config[backend] sans garde -
    # mieux vaut échouer ici, avec un message clair, que par un KeyError profond)
    for backend_key in ("tpu", "gpu"):
        if backend_key not in config:
            errors.append(f"Bloc backend manquant: {backend_key}")
            continue
        backend_cfg = config[backend_key]
        micro_batch_size = backend_cfg.get("micro_batch_size")
        if not isinstance(micro_batch_size, int) or micro_batch_size <= 0:
            errors.append(f"{backend_key}.micro_batch_size doit être un entier positif (reçu {micro_batch_size!r})")
        accum_steps = backend_cfg.get("accum_steps")
        if not isinstance(accum_steps, int) or accum_steps <= 0:
            errors.append(f"{backend_key}.accum_steps doit être un entier positif (reçu {accum_steps!r})")

    if errors:
        print(f"❌ Erreurs de configuration pour {config_name}:")
        for error in errors:
            print(f"  - {error}")
        return False
    
    return True


DATASET_CONFIGS = {
    "FIGHTERJET_CLASSIFICATION": {
        # === CONFIG OPTIMALE CNN 128×128 GRAYSCALE STRETCHED ===
        # === Données ===
        "num_classes": 32,
        "class_names": ['a10', 'a4', 'a400m', 'alphajet', 'b1b', 'b2', 'b52', 'c130', 'c17', 'f117', 'f14', 'f15', 'f16', 'f18', 'f22', 'f35', 'f4', 'flanker', 'gripen', 'harrier', 'hawk', 'hawkeye', 'mig29', 'mirage2000', 'mustang', 'rafale', 'spitfire', 'sr71', 'su57', 'tornado', 'typhoon', 'v22'],
        "data_dir": "/home/aobled/Downloads/_balanced_dataset_split",
        "output_prefix": f"{DATA_ROOT}/chunks/classification/dataset_classification",
        "chunk_size": 30000,
        "image_size": (128, 128),
        "grayscale": True,  # ✅ GRAYSCALE (3× plus rapide, même accuracy)
        "augmentation_params": {
            "flip_h": True,
            "flip_v": False,              # False par défaut, mais on tente à True
            "rotation_factor": 0.12,      # On tente 0.15, 0.10 avant
            "zoom_factor": 0.10,          # Adouci (était 0.15)
            "translation_factor": 0.06,   # Divisé par 2 (0.12 -> 0.06) : Assez pour garder l'invariance anti-scintillement sans détruire la cible
            "brightness_delta": 0.10,
            "contrast_factor": 0.10,
            "pixelation_factor": 4.0      # Simule un upscale depuis un tout petit crop
        },
        
        "mean": None,
        "std": None,
        "mean_std_path": f"{DATA_ROOT}/chunks/classification/dataset_classification_meanstd.npz",
        
        # === Modèle ===
        "model_name": "sophisticated_cnn_128_lite",  # ✅ VALIDE 2026-07-26 (Winston/Aymeric): variante allegee de sophisticated_cnn_128_plus (Bloc 1 96->64, Bloc 4 pic 512->384 canaux), 0.83M params vs 1.26M pour Plus (-34%), pour la vitesse d'inference. 0.9451 val obtenu (archive/training_classification_log_128x4_ligthv2.txt) vs 0.9521 pour Plus a schedule egal (-0.7pt pour -34% params) - premier essai (0.9322, training_classification_log_128x4_ligth.txt) fausse par un decay_steps etale sur 40 epochs au lieu de 7, corrige depuis (voir "decay_epochs" ci-dessous). Retour arriere: remettre "sophisticated_cnn_128_plus". ATTENTION: checkpoint_path/training_state_path ci-dessous sont derives de dataset_name, pas de model_name (trainer.py:511-512) - sauvegarder best_model.pkl/best_model_training_state_classification.pkl avant de lancer un run si vous voulez garder un point de comparaison.
        "loss_method": "focal_loss",
        "loss_params": {"gamma": 2.0},


        
        # === Hyperparamètres TPU ===
        "tpu": {
            "micro_batch_size": 128,   # ✅ Optimal testé
            "accum_steps": 4,          # ✅ Batch effectif: 512 (sweet spot)
            "learning_rate": 8e-3,     # 🔥 NOUVEAU: LR augmenté pour apprentissage plus agressif
            "weight_decay": 5e-5,      # ✅ Optimal trouvé
            "dropout_rate": 0.0,       # 🔥 NOUVEAU: Pas de dropout pour exploiter le potentiel
            # warmup/decay_steps nichés ici (2026-07-18, migration structurelle) - meme
            # micro_batch_size que gpu (128) donc memes valeurs, comportement inchange.
            "warmup_steps": 1200,
            "decay_steps": 6000,
        },

        # === Hyperparamètres GPU ===
        "gpu": {
            "micro_batch_size": 128,   # ✅ Testé, fonctionne
            "accum_steps": 4,          # Batch effectif: 512
            "learning_rate": 8e-4,     # 🔥 NOUVEAU: LR/10 pour GPU (augmenté aussi)
            "weight_decay": 5e-5,
            "dropout_rate": 0.0,       # 🔥 NOUVEAU: Pas de dropout pour GPU aussi
            "warmup_steps": 1200,
            "decay_steps": 6000,
        },

        # === Entraînement ===
        "optimizer": "adamw",
        "lr_schedule": "cosine",
        "epochs": 40,              # 40
        "patience": 5,
        "decay_epochs": 7,  # 🧪 2026-07-26: schedule cosinus decay sur 7 epochs (~6000 steps, comme l'ancien decay_steps fige) au lieu des 40 epochs par defaut - restaure la longue queue de fine-tuning a bas LR qui a produit 0.9458/0.9521 val avant le passage au calcul auto (trainer.py, 2026-07-21). Voir deferred-work.md pour le run sophisticated_cnn_128_lite (0.9322) qui a motive ce changement.
        "label_smoothing": 0.15,  # ✅ Validé 2026-07-14 (après fix task_strategies.py:110-118, ex-mort par if/elif) : combiné à mixup, 0.9521 val vs 0.9458 référence identique sans smoothing (+0.63pt, archive/training_classification_log_128x4_mixup_smoothing.txt)
        "mixup_alpha": 0.05,        # ✅ OPTIMAL: Mixup doux (meilleur compromis trouvé) - confirmé combinable avec label_smoothing (voir ci-dessus)
        
        # === Évaluation ===
        "metric_method": "accuracy",
        "report_method": "confusion_matrix",
        "eval_batch_size": 128,

        "eval_use_subset": True,
        "eval_max_subset": 100000,  
        
        # === Sauvegarde ===
        "checkpoint_path": "best_model.pkl",
        "training_state_path": "best_model_training_state_classification.pkl",
        "confusion_matrix_path": "confusion_matrix.png",
    },

    "FIGHTERJET_DETECTION": {
        # === CONFIG DETECTION D'AVIONS ===
        "task_type": "detection",
        
        # === Données ===
        "num_classes": 1,  # Single Class Object Detection
        "class_names": ['aircraft'],
        "output_prefix": f"{DATA_ROOT}/chunks/detection/dataset_detection",
        "image_size": (224, 224),
        "grayscale": True,
        "max_boxes": 20,  # 🔥 Images avec plus de 20 boxes seront ignorées (évite padding excessif et faux négatifs)
        
        # === Augmentation de Données ===
        "augmentation_params": {
            "flip_h": True,
            "flip_v": True,
            "rotation_factor": 0.0,
            "zoom_factor": 0.35,          # ±25% (était 20%) : Pour apprendre le multi-échelles
            "translation_factor": 0.25,   # Shifting ±15% (était 10%) : Désaxer les cibles
            "brightness_delta": 0.15,     # Beaucoup plus agressif (casser la dominante ciel gris clair)
            "contrast_factor": 0.30       # Très agressif (simule contre-jour et nuages sombres)
        },
        
        # === Modèle ---
        "model_name": "aircraft_detector_unet",
        "grid_size": 224,      # Segmentation sémantique (output size = input size)
        "loss_method": "segmentation",
        "loss_params": {
            "bce_weight": 0.3,
            "dice_weight": 0.7,
            "false_positive_penalty": 2.0
        },

        
        # === Hyperparamètres TPU ===
        "tpu": {
            "micro_batch_size": 128,    # Batch ok à 65, test avec 128
            "accum_steps": 1,          # Accumulation pour simuler 64
            "learning_rate": 4e-4,     # LR légèrement remonté
            "weight_decay": 5e-5,
            "dropout_rate": 0.0,
            # (2026-07-18, migration structurelle) valeur preexistante deplacee telle
            # quelle, PAS recalibree - decay_steps=22000 represente ~13.4 epochs reelles
            # (210570 images // 128 = 1645 steps/epoch) plutot que les 8 epochs configures
            # (config.epochs) ; ecart deja present avant cette session, laisse tel quel
            # (fonctionne en pratique, hors perimetre de la demande utilisateur qui portait
            # sur le decalage TPU/GPU, corrige cote gpu ci-dessous).
            "warmup_steps": 400,
            "decay_steps": 22000,
        },

        # === Hyperparamètres GPU ===
        "gpu": {
            "micro_batch_size": 16,
            "accum_steps": 2,
            "learning_rate": 2e-4,
            "weight_decay": 5e-5,
            "dropout_rate": 0.05,
            # decay_steps recalcule (2026-07-18) - le decalage preexistant signale lors de
            # la migration structurelle est corrige ici, a la demande de l'utilisateur.
            # micro_batch_size=16 volontairement INCHANGE (config de production, checkpoint
            # deja entraine avec cette valeur - contrairement a JAX_DETECTOR, encore
            # experimental, pas de raison de le remonter ici sans le demander explicitement).
            # 210 570 images train (verifie sur disque, chunks/detection/, cette session)
            # // 16 = 13160 steps/epoch reels x 8 epochs (config actuelle, inchangee) = 105280.
            # warmup_steps garde a 400 en valeur brute (meme convention que JAX_DETECTOR :
            # warmup_steps constant entre backends, seul decay_steps varie avec steps/epoch).
            "warmup_steps": 400,
            "decay_steps": 105280,
        },

        # === Entraînement ===
        "optimizer": "adamw",
        "lr_schedule": "cosine",
        "epochs": 8,
        "patience": 5,
        
        # === Évaluation/Visualization ===
        "metric_method": "segmentation_iou",
        "report_method": "segmentation_heatmap",
        "eval_batch_size": 16,

        "vis_freq": 5,
        
        # === Sauvegarde ===
        "checkpoint_path": "best_model_detection.pkl",
        "training_state_path": "best_model_training_state_detection.pkl",
        "save_dir": "./checkpoints_detection",
    },
    
    "JAX_KEPLER": {
        # === CONFIG CLASSIFICATION EXOPLANETES (1D) ===
        "task_type": "kepler",
        
        # === Données ===
        "num_classes": 2,
        "class_names": ['no_exoplanet', 'exoplanet'],
        "output_prefix": "./data/chunks/kepler/dataset_kepler",
        # (longueur, 1) : format (H,W) attendu par la validation et ChunkManager → tenseurs (L, 1, C)
        "image_size": (3197, 1),
        "grayscale": True,       # Force data_management à ne pas chercher de RGB
        
        # === Augmentation de Données ===
        # Pas d'augmentation spatiale (flip_h n'a pas de sens physique direct ici)
        "augmentation_params": {
            "flip_h": False,
            "flip_v": False,
            "rotation_factor": 0.0,
            "zoom_factor": 0.0,
            "translation_factor": 0.0,
            "brightness_delta": 0.0,
            "contrast_factor": 0.0
        },
        
        # === Modèle ===
        "model_name": "kepler_1d_cnn",
        "loss_method": "cross_entropy",

        
        # === Hyperparamètres TPU ===
        "tpu": {
            "micro_batch_size": 128,
            "accum_steps": 1,
            "learning_rate": 1e-4,
            "weight_decay": 1e-4,
            "dropout_rate": 0.3,
            "label_smoothing": 0.0,
            "mixup_alpha": 0.0,
            # (2026-07-18, migration structurelle) valeur preexistante deplacee telle
            # quelle. Pas de decay_steps historique (defaut Trainer 6000 s'applique
            # toujours via backend_config.get(), inchange).
            "warmup_steps": 500,
        },

        # === Hyperparamètres GPU ===
        "gpu": {
            "micro_batch_size": 32,
            "accum_steps": 4,
            "learning_rate": 1e-4,
            "weight_decay": 1e-4,
            "dropout_rate": 0.3,
            "label_smoothing": 0.0,
            "mixup_alpha": 0.0,
            "warmup_steps": 500,
        },

        # === Entraînement ===
        "optimizer": "adamw",
        "lr_schedule": "cosine",
        "epochs": 30,
        "patience": 5,

        "eval_use_subset": False, # On évalue sur tout
        "metric_method": "accuracy",
        "report_method": "lightcurves",
        
        # === Sauvegarde ===
        "checkpoint_path": "best_model_kepler.pkl",
        "training_state_path": "best_model_training_state_kepler.pkl",
        "confusion_matrix_path": "confusion_matrix_kepler.png",
    },

    "CIFAR10": {
        # === CONFIG BOUCLE DE TEST RAPIDE, tunée le 2026-07-14 (0.8110 -> 0.8582 val accuracy) ===
        # Historique complet des runs et de la démarche d'isolation : voir dev notes de la session
        # (git log sur ce fichier) et deferred-work.md. Résumé : augmentation (translation_factor)
        # + patience/epochs plus généreux aident nettement (0.8110 -> 0.8582, Run A) ; batch=256 +
        # LR scalée + mixup_alpha (v3, 0.8570) et + label_smoothing (v4, 0.8534) n'apportent RIEN
        # de plus que Run A une fois isolés proprement - abandonnés, pas portés sur FIGHTERJET_CLASSIFICATION.
        # === Données ===
        "num_classes": 10,
        "class_names": ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'],
        "output_prefix": f"{DATA_ROOT}/chunks/cifar10/dataset_cifar10",
        "image_size": (32, 32),
        "grayscale": False,
        "augmentation_params": {
            "flip_h": True,
            "flip_v": False,
            "rotation_factor": 0.0,
            "zoom_factor": 0.0,
            "translation_factor": 0.1,  # random-crop-like shift (~3px/32px) ; annule l'Addendum 2 Epic 5 ("pas de changement nécessaire") - le surapprentissage persiste malgré le fix dropout de l'Addendum 3 (meilleur epoch 19 : train 93.45% vs val 81.10%, archive/training_cifar10_log_GPU_128x1.txt)
            "brightness_delta": 0.0,
            "contrast_factor": 0.0
        },

        "mean": None,
        "std": None,
        "mean_std_path": f"{DATA_ROOT}/chunks/cifar10/dataset_cifar10_meanstd.npz",

        # === Modèle ===
        "model_name": "sophisticated_cnn_32_plus",  # Variante réduite pour 32×32 (128_plus était surdimensionné/trop de pooling pour cette taille)
        "loss_method": "cross_entropy",  # Dataset équilibré par construction (contrairement à FIGHTERJET_CLASSIFICATION)
        "loss_params": {},

        # === Hyperparamètres GPU ===
        "gpu": {
            "micro_batch_size": 128,  # batch=256+LR scalée (testé v3/v4) n'apportait rien de plus que epochs seul - non retenu
            "accum_steps": 1,
            "learning_rate": 1e-3,
            "weight_decay": 5e-5,
            "dropout_rate": 0.3,
            # (2026-07-18, migration structurelle) meme micro_batch_size que tpu (128)
            # donc memes valeurs, comportement inchange.
            "warmup_steps": 200,
            "decay_steps": 17595,  # 391 (steps/epoch à micro_batch_size=128) × 45 (epochs) - recalculer si l'un des deux change (cf. bug Epic 5 Addendum 3)
        },

        # === Hyperparamètres TPU ===
        "tpu": {
            "micro_batch_size": 128,
            "accum_steps": 1,
            "learning_rate": 1e-3,
            "weight_decay": 5e-5,
            "dropout_rate": 0.3,
            "warmup_steps": 200,
            "decay_steps": 17595,
        },

        # === Entraînement ===
        "optimizer": "adamw",
        "lr_schedule": "cosine",
        "epochs": 45,          # 30 -> 45 (2026-07-14) : le levier qui compte vraiment - seul changement qui bat la référence v2 (0.8416 -> 0.8582)
        "patience": 8,  # 5 -> 8 : nécessaire pour laisser les 45 epochs se dérouler sans early-stop prématuré
        # PAS de mixup_alpha/label_smoothing : testés (v3 avec mixup=0.05 -> 0.8570, v4 +smoothing=0.15 -> 0.8534),
        # tous les deux sous la référence epochs-seul ci-dessus (0.8582) une fois isolés proprement du changement de batch/LR.
        # Le bug de code qui les rendait mutuellement exclusifs reste corrigé (task_strategies.py:110-118),
        # mais aucun des deux n'a de valeur démontrée sur ce dataset - rien à porter sur FIGHTERJET_CLASSIFICATION.

        # === Évaluation ===
        "metric_method": "accuracy",
        "report_method": "confusion_matrix",
        "eval_batch_size": 128,
        "eval_use_subset": False,  # 10 000 images val, taille déjà raisonnable

        # === Sauvegarde ===
        # Pas de checkpoint_path/training_state_path explicite : nommage dérivé de dataset_name (Story 5.0)
        "confusion_matrix_path": "confusion_matrix_cifar10.png",
    },

    "JAX_DETECTOR": {
        # === CONFIG DETECTION D'AVIONS - CenterNet (heatmap+taille, AD-9/AD-17) ===
        # Coexiste avec FIGHTERJET_DETECTION (approche masque, AD-20 non-régression) - ne la
        # modifie pas, output_prefix pointe vers un répertoire distinct (chunks/jax_detector/,
        # jamais chunks/detection/).
        "task_type": "detection_centernet",

        # === Données ===
        "num_classes": 1,  # Detection mono-classe (Story 7.1)
        "class_names": ['aircraft'],
        # Le nom de base doit correspondre exactement au préfixe codé en dur dans
        # dataset_builder/jax_detector_dataset_tools.py (Story 7.4) pour que le glob de
        # CenterNetDetectionDataset (Story 7.5) trouve les chunks.
        "output_prefix": f"{DATA_ROOT}/chunks/jax_detector/jax_detector_targets",
        # Pic memoire pendant la generation ~ chunk_size x 784 Ko x 2 (liste Python + tableau
        # numpy empile simultanement, dataset_builder/jax_detector_dataset_tools.py::_save_chunk_v2) -
        # 3000 = ~4.5 Go, valide sur la machine locale (30 Go RAM + 2 Go swap). Recalculer avant
        # d'augmenter sur un environnement avec plus de RAM (ex. Colab).
        "chunk_size": 12800,
        "image_size": (224, 224),
        "grayscale": True,
        "max_boxes": 20,
        # Seuil de score pour valid_mask a l'inference (Story 8.3, AD-15 : seuil en config,
        # jamais une constante privee dupliquee). Valeur de depart alignee sur le
        # conf_threshold=0.3 deja utilise par decode_segmentation_and_detect
        # (inference_utils.py) pour le pipeline actuel - pas encore tunee pour ce format.
        # Abaisse a 0.2 le 2026-07-19 (retour utilisateur : avions petits/lointains sur fond
        # de ville non confirmes malgre un vrai pic de score visible dans le heatmap
        # exploratoire bas-gauche) - le diagnostic threshold_sensitivity (archive/) avait
        # deja verifie zero faux positif sur les images annotees jusqu'a 0.1.
        # Abaisse a 0.1 le 2026-07-22 : audit_dataset_detection_jax.py (v10, dilatation +
        # contexte global) a montre que le score de confiance BRUT (avant seuil) est
        # correct geometriquement dans 81% des cas sur les boites plein-cadre (70-100%
        # de l'image), mais que seulement 9.6% depassaient le seuil de 0.2 - un probleme
        # de calibration de confiance, pas de capacite geometrique. Puisque 0.1 est deja
        # verifie sans faux positif (voir ci-dessus), pas de raison de garder la marge a 0.2.
        "detection_score_threshold": 0.1,

        # Seuil IoU du NMS JAX-natif a l'inference (2026-07-22, voir deferred-work.md,
        # AD-15 : seuil en config, jamais une constante privee dupliquee - meme discipline
        # que detection_score_threshold ci-dessus). Un candidat est supprime si son IoU
        # avec un candidat mieux score depasse ce seuil. Necessaire car le rayon gaussien
        # de la cible d'entrainement (_gaussian_radius, detection_target_encoding.py,
        # Story 7.1) grandit avec la taille de la boite : un grand objet produit un
        # plateau large de score eleve (pas un pic pointu), et plusieurs pics voisins
        # peuvent depasser detection_score_threshold simultanement pour le meme objet.
        # Valide sur audit complet (120 102 images, 2026-07-22) : ratio predictions/vraies
        # boites 1.353->1.119, sans effet de bord notable sur IoU moyen/GOOD.
        "nms_iou_threshold": 0.5,

        # Augmentation "zoom plein-cadre" a la GENERATION des chunks (dataset_builder/
        # jax_detector_dataset_tools.py::process_detection_dataset_v2, distincte de
        # augmentation_params ci-dessous qui agit au chargement/entrainement sur les
        # tenseurs deja encodes, CenterNetDetectionDataset, Story 7.5). 0.0 = defaut,
        # aucun changement de sortie tant que non explicitement releve (diagnostic
        # tests/diagnose_single_full_size_aircraft.py, 2026-07-19).
        "zoom_augment_probability": 0.0,

        # === Augmentation de Données ===
        # Copiée telle quelle de FIGHTERJET_DETECTION - point de départ raisonnable, pas
        # encore tunée pour ce nouveau format (même statut que les hyperparamètres de perte).
        "augmentation_params": {
            "flip_h": True,
            "flip_v": True,
            "rotation_factor": 0.0,
            "zoom_factor": 0.35,
            "translation_factor": 0.25,
            "brightness_delta": 0.15,
            "contrast_factor": 0.30
        },

        # === Modèle ===
        "model_name": "aircraft_detector_centernet",  # Testé le 2026-07-26 : variante "lite" (SeparableConv, -36% params, 1.36M) entrainee et auditee - impact localisation limite mais gain de vitesse/taille reel quasi nul en inference (mesure utilisateur), donc pas de raison de payer la baisse de confiance heatmap (HeatmapActivation val 0.2022->0.1583). Retour a "aircraft_detector_centernet" (2.13M params) comme modele actif. Checkpoint lite conserve en reference (archive/best_model_jax_detector.pkl = non-lite v10, root = lite tant que non re-entraine). ATTENTION: nommage checkpoint derive de dataset_name (JAX_DETECTOR), pas de model_name (Story 5.0, voir "Sauvegarde" plus bas).

        # Proportion reelle de pixels positifs (gt_heatmap==1.0) mesuree sur les 211 266
        # images du train set (2026-07-17, execution reelle Story 7.8) : 283 753 pixels
        # positifs / 10 600 482 816 pixels totaux = 1.34 objet/image en moyenne sur une
        # grille 224x224. Initialise le biais de la tete heatmap (AircraftDetectorCenterNet,
        # model_library.py) pour eviter le collapse observe avec le biais par defaut
        # (sigmoid(0)=0.5 partout - voir addendum post-hoc Story 7.2, 2026-07-17).
        "heatmap_prior": 0.0000268,

        # Pas de loss_method/metric_method/report_method : CenterNetDetectionStrategy
        # (Story 7.6) n'a qu'une seule méthode de perte et une seule métrique, aucun
        # dispatch interne - inclure ces clés suggérerait un dispatch qui n'existe pas.
        "loss_params": {
            "heatmap_weight": 1.0,
            "size_weight": 0.1,
            "alpha": 2.0,
            "beta": 4.0
        },
        # Pas de metric_threshold : HeatmapActivation (addendum post-hoc 2026-07-18,
        # CenterNetDetectionStrategy.compute_metrics) est une moyenne continue, plus de
        # seuil dur - metric_threshold n'existe plus dans __init__.

        # === Hyperparamètres TPU ===
        "tpu": {
            "micro_batch_size": 128,
            "accum_steps": 1,
            # batch effectif = 128 (micro_batch_size x accum_steps) - retour au meme batch
            # effectif que le tout premier run (v1, jamais teste avec le correctif
            # heatmap_prior). learning_rate reste a 4e-4 : c'etait deja la valeur calibree
            # pour ce batch effectif (valeur d'origine FIGHTERJET_DETECTION/JAX_DETECTOR,
            # Story 7.7) - pas besoin de la remonter puisqu'on ne l'augmente pas au-dela de
            # 128, contrairement a 256x1/128x2 (batch effectif 256) qui auraient justifie
            # un scaling proportionnel (~8e-4).
            "learning_rate": 4e-4,
            "weight_decay": 5e-5,
            # 0.0 -> 0.1 (2026-07-18) : premier signe reel de divergence train/val a partir
            # de l'epoch 3 (v4, detection/train seul, cf. archive/training_jax_detector_v4.txt)
            # - train continue de monter, val plafonne. Seul point de dropout du modele
            # (bottleneck, AircraftDetectorCenterNet), valeur moderee pour ne pas freiner
            # davantage un apprentissage deja lent sur un signal heatmap eparse.
            "dropout_rate": 0.1,
            "warmup_steps": 400,
            # 992 steps/epoch (mesure reellement, archive/training_jax_detector_rv7.txt,
            # 2026-07-19) x 15 epochs. Recalcule suite a zoom_augment_probability=0.3
            # (2026-07-19) : l'ancienne valeur (11415 = 761 steps/epoch x 15, mesuree AVANT
            # cette augmentation, run v4) etait restee figee apres l'activation de
            # l'augmentation, qui ajoute ~30% d'echantillons/epoch (992/761 = 1.304, coherent
            # avec le +30% attendu) - le schedule cosine atteignait donc son plancher
            # (LR~1e-6) vers l'epoch 12/15 au lieu de l'epoch 15, gelant l'apprentissage sur
            # les ~3 dernieres epochs du run rv7 (plateau observe dans le log a partir de
            # l'epoch 10). A recalculer de nouveau si le volume de donnees ou le nombre
            # d'epochs changent encore.
            "decay_steps": 14880,
        },

        # === Hyperparamètres GPU (2026-07-18, cible T4 Colab) ===
        # Recalibre sur FIGHTERJET_DETECTION.gpu (dataset_configs.py:163-170) - meme
        # architecture (UNet, profondeur/canaux identiques a CenterNet, Story 7.2 AC1),
        # meme resolution 224x224, deja valide en pratique sur GPU local 6 Go a
        # micro_batch_size=16. Le T4 Colab a ~16 Go VRAM (~2.7x le GPU local) - estimation
        # raisonnable ~x4 (16->64), PAS verifiee sur un vrai T4 : redescendre a 32 en cas
        # d'OOM. accum_steps=1 (pas 2/4) : les runs a mise a jour frequente (128x1, TPU)
        # se sont mieux comportes cette session que ceux a batch effectif accumule
        # (128x2) - pas de raison de recompliquer ici. learning_rate=2e-4 inchange :
        # ratio 4e-4(TPU,128)/2e-4(GPU,64) = 2, proportionnel au ratio de batch (128/64=2),
        # cohalent avec la calibration TPU. dropout_rate aligne sur le correctif TPU
        # (2026-07-18, divergence train/val observee - propriete du modele/tache, pas du
        # materiel). warmup/decay_steps niches ici (plus top-level partage, migration
        # structurelle 2026-07-18) : 97531 // 64 = 1523 steps/epoch reels (drop_remainder=True)
        # x 15 epochs = 22845 - meme duree relative (15 epochs) que le TPU (761x15=11415),
        # calcul distinct car steps/epoch differe (batch 64 vs 128).
        "gpu": {
            "micro_batch_size": 64,
            "accum_steps": 1,
            "learning_rate": 2e-4,
            "weight_decay": 5e-5,
            "dropout_rate": 0.1,
            "warmup_steps": 400,
            "decay_steps": 22845,
        },

        # === Entraînement ===
        # Structure copiée de FIGHTERJET_DETECTION comme point de départ - valeurs à
        # réajuster empiriquement une fois l'entraînement réel lancé (Story 7.8).
        "optimizer": "adamw",
        "lr_schedule": "cosine",
        # 8 -> 15 epochs, patience 5 -> 8 (2026-07-18) : meme levier que CIFAR10
        # (dataset_configs.py, "necessaire pour laisser les epochs se derouler sans
        # early-stop premature") - le run v4 (detection/train seul) a plafonne en val a
        # partir de l'epoch 3 puis arrete a l'epoch 8 (patience=5 epuisee) alors que
        # train continuait de progresser ; plus de marge pour voir si ca se degage.
        "epochs": 15,
        "patience": 8,
        # warmup_steps/decay_steps niches sous tpu/gpu (2026-07-18, migration structurelle -
        # ce sont des comptes de STEPS, dependants du steps/epoch donc du micro_batch_size,
        # contrairement a epochs/patience qui restent partages ici).

        # === Évaluation/Visualization ===
        "eval_batch_size": 16,
        "vis_freq": 5,

        # === Sauvegarde ===
        # Pas de checkpoint_path/training_state_path explicite : nommage dérivé de
        # dataset_name (Story 5.0, CenterNetDetectionStrategy._get_export_path, Story 7.6)
        # -> best_model_jax_detector.pkl / best_model_training_state_jax_detector.pkl
        "save_dir": "./checkpoints_jax_detector",
    }
}



def get_dataset_config(dataset_name):
    """
    Récupère la configuration d'un dataset
    
    Args:
        dataset_name: Nom du dataset (ex: "FIGHTERJET_9CLASSES")
    
    Returns:
        dict: Configuration du dataset
    
    Raises:
        ValueError: Si le dataset n'existe pas
    """
    if dataset_name not in DATASET_CONFIGS:
        available = list(DATASET_CONFIGS.keys())
        raise ValueError(f"Dataset '{dataset_name}' inconnu. Datasets disponibles: {available}")
    
    config = DATASET_CONFIGS[dataset_name]
    config["dataset_name"] = dataset_name

    # Par défaut, classification si non spécifié
    if "task_type" not in config:
        config["task_type"] = "classification"
    
    # Valider la configuration
    if not validate_config(dataset_name, config):
        raise ValueError(f"Configuration invalide pour {dataset_name}")
    
    return config


def list_available_datasets():
    """Retourne la liste des datasets disponibles"""
    return list(DATASET_CONFIGS.keys())


def print_config(dataset_name):
    """Affiche la configuration d'un dataset"""
    config = get_dataset_config(dataset_name)
    
    print(f"\n📊 CONFIGURATION: {dataset_name}")
    print("=" * 60)
    print(f"Classes: {config['num_classes']}")
    print(f"Noms: {config['class_names']}")
    print(f"Image size: {config['image_size']}")
    print(f"Modèle: {config['model_name']}")
    print(f"Epochs: {config['epochs']}")
    print(f"Patience: {config['patience']}")
    print(f"\nTPU: batch={config['tpu']['micro_batch_size']}×{config['tpu']['accum_steps']}, lr={config['tpu']['learning_rate']}, wd={config['tpu']['weight_decay']}")
    print(f"GPU: batch={config['gpu']['micro_batch_size']}×{config['gpu']['accum_steps']}, lr={config['gpu']['learning_rate']}, wd={config['gpu']['weight_decay']}")
    print("=" * 60)


if __name__ == "__main__":
    # Test de validation
    print("🧪 TEST DES CONFIGURATIONS")
    print("=" * 60)
    
    for dataset_name in list_available_datasets():
        print(f"\n✅ Validation de {dataset_name}...")
        try:
            config = get_dataset_config(dataset_name)
            print(f"   ✓ {config['num_classes']} classes: {config['class_names']}")
        except Exception as e:
            print(f"   ❌ Erreur: {e}")
    
    print("\n🎯 Affichage d'une config complète:")
    print_config("FIGHTERJET_CLASSIFICATION")
