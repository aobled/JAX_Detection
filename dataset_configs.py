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
    
    # Vérifier que image_size est un tuple de 2 entiers - usage restreint désormais au
    # retraitement d'image réel (data_management.py/dataset_builder/, resize) depuis que
    # "input_shape" (2026-07-30) a repris le rôle de forme d'entrée modèle (voir plus bas).
    if "image_size" in config:
        if not isinstance(config["image_size"], tuple) or len(config["image_size"]) != 2:
            errors.append(f"image_size doit être un tuple (H, W)")

    # Vérifier que input_shape est un tuple d'au moins 1 entier positif - forme
    # complète (hors batch) que Trainer.create_train_state() utilise pour construire son
    # entrée factice (2026-07-30, remplace l'ancien couple image_size+num_channels/
    # grayscale + branche if/elif par task_type dans trainer.py - source unique, 100%
    # générique, jamais un domaine qui détourne une autre clé comme JAX_KEPLER le faisait
    # avec image_size). (H, W, C) pour une image, (longueur, canaux) pour une séquence 1D
    # avec canaux (Kepler), (8, 8, C) pour les échecs, (longueur,) pour une séquence
    # d'identifiants sans dimension canal (chess_move_token, Epic 11 - un décodeur
    # transformer n'a besoin que de (B, L), pas de dimension canal) - la forme exacte
    # est spécifique à chaque domaine, cette fonction ne valide que la structure
    # générique (tuple, entiers positifs). Borne basse relâchée de 2 à 1 (2026-08-10,
    # Epic 11) : changement strictement monotone (accepte plus qu'avant), aucune config
    # existante n'a de input_shape de longueur 1 aujourd'hui - comportement de
    # validation inchangé pour tous les domaines déjà en place (AD-21 hérité).
    if "input_shape" in config:
        shape = config["input_shape"]
        if not isinstance(shape, tuple) or len(shape) < 1 or not all(isinstance(d, int) and d > 0 for d in shape):
            errors.append("input_shape doit être un tuple d'au moins 1 entier positif, ex. (H, W, C) ou (longueur,)")

    # Vérifier que les paramètres requis sont présents. "image_size" retiré de cette
    # liste (2026-07-27, décision Aymeric, Story 9.3) puis remplacé par "input_shape"
    # (2026-07-30, obligatoire pour tous les domaines - Trainer le lit sans garde).
    required = ["num_classes", "model_name", "input_shape"]
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
        "input_shape": (128, 128, 1),  # forme d'entrée modèle (Trainer, 2026-07-30) - dérivée de image_size+grayscale
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
        "input_shape": (224, 224, 1),  # forme d'entrée modèle (Trainer, 2026-07-30) - dérivée de image_size+grayscale
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
        # input_shape (Trainer, 2026-07-30) : ancien détournement de image_size en (longueur,
        # 1) pour une séquence 1D remplacé par une forme d'entrée explicite - Trainer ne
        # devine plus la famille de forme depuis task_type=="kepler", il lit juste ce tuple.
        "input_shape": (3197, 1),

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
        "input_shape": (32, 32, 3),  # forme d'entrée modèle (Trainer, 2026-07-30) - dérivée de image_size+grayscale
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
        "input_shape": (224, 224, 1),  # forme d'entrée modèle (Trainer, 2026-07-30) - dérivée de image_size+grayscale
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
    },

    "CHESS_NO_HISTORY": {
        # === Seule config echecs restante (Epic 9, policy+value) ===
        # Etait a l'origine une variante d'ablation de "CHESS" (avec historique) - test
        # "l'historique sert-il a quelque chose ?" (2026-07-29, voir deferred-work.md "test
        # ablation historique"). "CHESS" et "CHESS_NAKAMURA_NO_HISTORY" retires le 2026-08-02
        # (nettoyage post-tournoi Carlsen/Nakamura, plus aucun interet - voir deferred-work.md
        # "tournoi model1 (Carlsen) vs model2 (Nakamura)") : celle-ci devient la config de
        # reference du domaine echecs. num_channels=19 (pas 29) : sans les 10 plans
        # d'historique (encode_position cote chess_ai avec include_history=False).
        "task_type": "chess_policy_value",
        "num_classes": 4672,
        "num_channels": 19,  # 19, pas 29 - pas d'historique
        "input_shape": (8, 8, 19),  # forme d'entrée modèle (Trainer, 2026-07-30)
        "model_name": "chess_cnn_attention_policy_value",
        "num_bottleneck_tokens": 8,
        "output_prefix": f"{DATA_ROOT}/chunks/chess_no_history/chess",
        "val_split": 0.1,
        "loss_params": {
            "policy_weight": 1.0,
            "value_weight": 1.0,
        },
        "tpu": {
            "micro_batch_size": 128,
            "accum_steps": 1,
            "learning_rate": 4e-4,
            "weight_decay": 5e-5,
            "dropout_rate": 0.1,
            "warmup_steps": 200,
            "decay_steps": 10000,
        },
        "gpu": {
            "micro_batch_size": 256,
            "accum_steps": 1,
            "learning_rate": 8e-4,
            "weight_decay": 5e-5,
            "dropout_rate": 0.1,
            "warmup_steps": 200,
            "decay_steps": 36700,
        },
        "optimizer": "adamw",
        "lr_schedule": "cosine",
        "epochs": 15,
        "patience": 8,
        "eval_batch_size": 64,
        "save_dir": "./checkpoints_chess_no_history",
    },

    "CHESS_LEGAL_MOVES": {
        # === Domaine échecs - tâche multi-label "coups légaux" (test de plomberie) ===
        # Prédit l'ENSEMBLE des coups légaux d'une position (plusieurs "1" simultanés,
        # espace de 4672 coups), pas le seul coup joué (chess_policy_value ci-dessus) -
        # sigmoid BCE (ChessLegalMovesStrategy), pas softmax cross-entropy. Dataset déjà
        # généré côté chess_ai (149 chunks, position (8,8,29) + legal_mask (4672,) int8).
        "task_type": "chess_legal_moves",
        "num_classes": 4672,  # taille de la sortie - PAS un nombre de classes (idem CHESS_NO_HISTORY)
        "num_channels": 29,  # avec historique (contrat .npz chess_legal_moves, cote chess_ai)
        "input_shape": (8, 8, 29),
        "model_name": "chess_cnn_attention_legal_moves",
        "num_bottleneck_tokens": 8,
        "output_prefix": f"{DATA_ROOT}/chunks/chess_legal_moves/chess_legal_moves",
        "val_split": 0.1,
        # metric_threshold : seuil sigmoid pour compute_metrics/generate_reports
        # (ChessLegalMovesStrategy) - 0.3 retenu suite au balayage de seuil sur le
        # 1er run (2026-08-02, 4 epochs) : F1=0.6030 a 0.3 contre 0.5285 au defaut
        # 0.5 (comptage agrege sur les 67840 positions val, pas une moyenne de F1
        # par batch).
        "metric_threshold": 0.3,
        # loss_params.pos_weight : ponderation du terme positif dans la BCE
        # (compute_chess_legal_moves_loss) - TESTE le 2026-08-02 (3e run, 15 epochs,
        # pos_weight=2.0) contre le 2e run (15 epochs, pos_weight=1.0 implicite,
        # F1=0.8578, precision=0.8364/rappel=0.8753) : resultat NEGATIF net
        # (F1=0.8467, precision=0.7642/rappel=0.9145 - le rappel monte comme prevu
        # mais la precision recule plus fort). Confirme que le desequilibre de
        # classe n'etait deja plus le facteur limitant a ce stade (corrige par les
        # epochs, pas par la loss) - remis a 1.0 (BCE non ponderee, le meilleur
        # resultat mesure a ce jour) en consequence. Voir deferred-work.md pour le
        # detail des 3 runs.
        "loss_params": {
            "pos_weight": 1.0,
        },

        # === Hyperparamètres GPU/TPU ===
        # Repris des memes ordres de grandeur que l'ancienne config CHESS (num_channels=29,
        # retiree le 2026-08-02) - jamais tune empiriquement pour cette tache au-dela du
        # nombre d'epochs. decay_steps recalcule pour epochs=15 (pas 4, 1er run de
        # plomberie) a partir des comptages REELS mesures lors de ce 1er run (135 chunks
        # train / 14 chunks val, 67840 positions val sur 14 chunks -> ~4846 positions/chunk
        # en moyenne -> ~654 200 positions train) : ~2555 steps/epoch a batch=256 (gpu) x 15
        # epochs ~= 38300 ; ~5111 steps/epoch a batch=128 (tpu) x 15 epochs ~= 76700.
        "tpu": {
            "micro_batch_size": 128,
            "accum_steps": 1,
            "learning_rate": 4e-4,
            "weight_decay": 5e-5,
            "dropout_rate": 0.1,
            "warmup_steps": 200,
            "decay_steps": 76700,
        },
        "gpu": {
            "micro_batch_size": 256,
            "accum_steps": 1,
            "learning_rate": 8e-4,
            "weight_decay": 5e-5,
            "dropout_rate": 0.1,
            "warmup_steps": 200,
            "decay_steps": 38300,
        },

        # === Entraînement ===
        # epochs=15 (aligne sur CHESS_NO_HISTORY) : le 1er run (4 epochs, test de
        # plomberie) montrait encore un gain net epoch 3->4 (+2.4pt F1 au seuil 0.5,
        # pas de plateau clair) - 15 epochs donne une vraie chance de convergence
        # sans changer plusieurs variables a la fois (pas de pos_weight ajoute ici,
        # voir deferred-work.md - un seul facteur teste isolement). patience=8
        # (idem CHESS_NO_HISTORY) : arrete plus tot si le plateau se confirme,
        # pas d'obligation d'aller jusqu'a 15 si LegalMoveF1 stagne avant.
        "optimizer": "adamw",
        "lr_schedule": "cosine",
        "epochs": 15,
        "patience": 8,

        # === Évaluation ===
        "eval_batch_size": 64,

        # === Sauvegarde ===
        # save_dir derive de dataset_name (TaskStrategy._get_export_path) ->
        # best_model_chess_legal_moves.pkl / best_model_training_state_chess_legal_moves.pkl
        "save_dir": "./checkpoints_chess_legal_moves",
    },

    "CHESS_SEARCH_TEACHER": {
        # === Domaine échecs - distillation depuis la recherche classique (Epic 10) ===
        # Reutilise a l'identique le task_type/modele policy+value de CHESS_NO_HISTORY
        # (Option A, 2026-08-04 - decision Aymeric/Winston, voir
        # _bmad-output/planning-artifacts/prds/prd-jax_supervised_training-2026-08-04/prd.md) :
        # dataset chess_ai chess_search_teacher, labellise par chess_search.py (professeur
        # alpha-beta, jamais appele a l'inference - principe AlphaZero). Aucune cle "value"
        # dans les .npz (positions d'auto-jeu, pas d'issue de partie completee) - la tete
        # value existante est neutralisee par ponderation (value_weight=0.0 ci-dessous),
        # pas supprimee. ATTENTION : num_channels=29 (avec historique, contrat #2.6), PAS
        # 19 comme CHESS_NO_HISTORY - ce dataset a l'historique, contrairement a
        # CHESS_NO_HISTORY qui n'en a pas. Aucune config existante ne combinait deja
        # chess_policy_value + 29 canaux avant celle-ci.
        "task_type": "chess_policy_value",
        "num_classes": 4672,
        "num_channels": 29,  # avec historique (contrat #2.6 chess_ai) - pas un gabarit CHESS_NO_HISTORY
        "input_shape": (8, 8, 29),
        "model_name": "chess_cnn_attention_policy_value",
        # 16 au lieu de 8 : test de capacite (recherche technique 2026-08-06, voir
        # _bmad-output/planning-artifacts/research/technical-chesscnnattention-bottleneck-capacity-chess-search-teacher-research-2026-08-06.md).
        # Leviers moins cher a tester que token_dim (cout lineaire en N vs quadratique
        # en hidden dim) - protocole : observer le Train Accuracy, pas seulement Val,
        # pour distinguer sous-capacite de plafond intrinseque a la tache/au professeur.
        # Resultat 2026-08-06 : Train Accuracy inchange (31.18% -> 31.05%) malgre 8->16 -
        # ce levier precis ne semble pas la cause. num_bottleneck_tokens reste a 16
        # (pas de raison de revenir a 8), on teste maintenant token_dim (levier plus
        # significatif d'apres la comparaison Leela BT4, cf. doc de recherche).
        #
        # token_dim 128 -> 256 (2026-08-07) : justifie par un gap train/val quasi nul
        # (~0.1pt) observe en cours de route sur le run depth=12 (vs 4.4-7.2pt sur
        # dim=128/depth=8) - le volume de donnees x2 semblait absorber la capacite
        # sans surapprendre. Resultat final (2026-08-08) : Val 26.16%->27.63% (+1.47pt
        # seulement) pour 2,52x plus de params (875K->2,2M) et un gap qui explose
        # (1.33pt->11.64pt final, 32.49% en Simpson) - mauvais rapport cout/benefice,
        # voir _bmad-output/implementation-artifacts/chess-search-teacher-strategy.md
        # (doc dedie, options pour combler ce gap).
        #
        # token_dim 256 -> 192 (2026-08-08, decision Aymeric/Winston, Option 5 de la
        # strategie ci-dessus) : point intermediaire jamais teste, entre le gap sain
        # de 128 et le meilleur Val de 256 - recherche d'un compromis avant d'investir
        # dans les leviers de regularisation (weight_decay/dropout/label_smoothing,
        # options 1/2/3 du meme document). 192 % num_heads(4) == 0, contrainte
        # respectee. Checkpoints best_model_chess_search_teacher-token_dim_{128,256}.pkl
        # conserves cote chess_ai avant ce changement (Aymeric, 2026-08-08).
        "num_bottleneck_tokens": 16,
        "token_dim": 196,
        "output_prefix": f"{DATA_ROOT}/chunks/chess_search_teacher/chess_search_teacher",
        "val_split": 0.1,
        # value_weight=0.0 : neutralise le terme value de la loss composite existante
        # (compute_chess_policy_value_loss, AD-24) sans la modifier - aucune cible value
        # reelle dans ce dataset. value_head_trained derive automatiquement de cette valeur
        # par get_dataset_config() ci-dessous (voir cette fonction), jamais un litteral
        # duplique ici.
        # label_smoothing=0.2 (2026-08-08, Option 3 de chess-search-teacher-strategy.md,
        # decision Aymeric/Winston) : 0.1 (defaut standard de la litterature) initialement
        # propose, revu a 0.2 avant le premier run - le dropout venait de montrer que meme
        # un changement substantiel (0.25->0.35) ne deplace le Val que de ~0.2pt ici, donc
        # 0.1 risquait de se noyer dans ce meme bruit sans trancher clairement "ca marche"
        # vs "ca ne marche pas". 0.2 reste dans la plage couramment testee (pas un pari
        # extreme). compute_chess_policy_loss (loss_functions.py) relaie ce parametre via
        # **self.loss_params (ChessPolicyValueStrategy, task_strategies.py) - aucun autre
        # code touche a ce mecanisme de plomberie deja en place pour
        # policy_weight/value_weight. Ne touche jamais value_loss (MSE, label smoothing
        # sans objet). Teste : label_smoothing=0.0 (absent du dict, comme avant) reste
        # bit-a-bit identique a l'ancienne formule (verifie par test).
        "loss_params": {
            "policy_weight": 1.0,
            "value_weight": 0.0,
            "label_smoothing": 0.2,
        },
        # Hyperparametres copies de CHESS_NO_HISTORY tels quels (point de depart non tune
        # pour ce nouveau dataset, meme statut que CHESS_LEGAL_MOVES a sa creation).
        #
        # CORRECTIF 2026-08-07 (diagnostic Winston) : decay_steps ci-dessous EST bien la
        # valeur active, pas un repli mort comme un commentaire precedent l'affirmait a
        # tort. _count_real_train_samples (trainer.py:45) cherche des fichiers
        # "{output_prefix}_train_chunk*.npz", mais chess_ai ne produit qu'un seul flux
        # "{output_prefix}_chunk*.npz" sans split fichier (ChessPolicyValueDataset,
        # data_management.py:602-613 - split fait au chargement par fraction de chunks).
        # Le glob ne matche donc jamais rien pour AUCUNE config echecs
        # (CHESS_SEARCH_TEACHER/CHESS_NO_HISTORY/CHESS_LEGAL_MOVES, meme convention a
        # fichier unique partout) - recalcul automatique jamais fonctionnel sur ce
        # domaine, decay_steps code en dur toujours reellement utilise en pratique.
        # Valeurs ci-dessous recalculees a la main pour epochs=25 et le volume reel du
        # dataset depth=12 (1 402 252 positions, voir note plus bas) : train ~=
        # 1 402 252*0.9 ~= 1 262 027 -> gpu (batch=256) 4929 steps/epoch x 25 = 123225 ;
        # tpu (batch=128) 9859 steps/epoch x 25 = 246475. Fix generique de
        # _count_real_train_samples pour le domaine echecs differe (deferred-work.md).
        #
        # RECALCUL 2026-08-12 (output_prefix bascule sur chess_decisive_teacher,
        # dataset filtre aux seules parties a issue decisive) : volume reel mesure
        # directement dans les .npz (29 chunks, meme split chunk-level 27
        # train/2 val, seed=42 - PAS une fraction 90/10 de position, cf. le split par
        # chunk ci-dessus) = 266 107 positions train (20 000 val, 286 107 total) - un
        # log reel (chess_ai/chess_decisive_teacher.txt) confirme deja 1039 steps/epoch
        # a batch=256, coherent avec ce recalcul. Le run 2026-08-11/12 avait tourne
        # avec le repli code en dur ci-dessus (123225, calcule pour l'ancien volume
        # depth=12 4.75x plus gros) - "⚠️ decay_steps automatique indisponible... repli
        # sur la config : 123225" dans le log - le schedule cosine n'a donc parcouru
        # qu'environ 25975/123225 ~= 21% de sa decroissance prevue sur les 25 epochs
        # reellement jouees, LR restee proche de sa valeur haute tout du long (a
        # corriger avant tout autre changement, memoire feedback_verify_numbers_against_logs).
        # gpu (batch=256) 1039 steps/epoch x 25 = 25975 ; tpu (batch=128) 2078
        # steps/epoch x 25 = 51950 (jamais encore mesure en reel, extrapole du meme
        # volume train).
        #
        # ATTENTION volume/profondeur du professeur, 2026-08-07 (decision chess_ai,
        # revue de code) : dataset regenere a depth=12 (etait depth=8) - depth=8 etait
        # quasi-redondant avec le coup deja joue par l'auto-jeu (meme profondeur, meme
        # position), videant le professeur de tout signal reel ; depth=12 garantit un
        # professeur distinct. Volume : 10 000 parties, 141 fichiers, 1 402 252
        # positions (~2x le volume depth=8 du 2026-08-05/06/07, qui donnait ~628K
        # positions train / 2454 steps/epoch). CONSEQUENCE : le PolicyAccuracy de ce
        # nouveau dataset n'est PAS comparable terme a terme aux baselines depth=8
        # ci-dessus (31.18%/31.05%/35.28%/35.53%) - un professeur plus profond trouve
        # des coups moins "evidents", plus durs a imiter ; une accuracy plus basse ici
        # ne signifierait pas une regression du modele. Nouvelle baseline a partir de
        # ce run, pas une suite directe des runs precedents.
        #
        # dropout_rate 0.1 -> 0.25 (2026-08-07) : test de regularisation suite au run
        # token_dim=64->128 (voir doc de recherche technique 2026-08-06) - ce run a
        # confirme token_dim comme vrai levier de capacite (Val 29.3%->35.3%) mais avec
        # un gap train/val qui se creuse nettement en fin de run (1.9pt -> 7.2pt final,
        # Val qui plafonne epoch 13-15 pendant que Train continue). Un seul levier
        # change a la fois (discipline suivie sur num_bottleneck_tokens et token_dim
        # avant ca) : token_dim/num_bottleneck_tokens inchanges ici.
        #
        # weight_decay 5e-5 -> 5e-4 (2026-08-08, Option 1) : teste seul sur token_dim=192,
        # effet nul dans le bruit (Train +0.11pt, Val +0.06pt, Gap +0.05pt - voir
        # chess-search-teacher-strategy.md). Remis a 5e-5 (valeur d'origine) : pas de
        # raison de garder un changement sans effet mesure pendant qu'on isole le
        # prochain levier (dropout ci-dessous) - un seul changement a la fois par
        # rapport a la derniere base propre (dim=192/wd=5e-5).
        #
        # dropout_rate 0.25 -> 0.35 (2026-08-08, Option 2) : le levier precedent
        # (0.1->0.25, run 2026-08-07 dim=128/depth=8) avait deja montre le meme
        # symptome que weight_decay - gap qui se referme (7.23->4.37pt) mais Val qui
        # bouge a peine (+0.25pt) - attentes modestes assumees pour ce test, pas un
        # levier cense faire progresser le Val, plutot un point de donnees peu couteux
        # avant d'investir dans le label smoothing (Option 3, code non encore ecrit).
        # decay_steps ci-dessous : repli jamais utilise en pratique pour le domaine
        # echecs (trainer.py:171-180 le recalcule automatiquement depuis le vrai
        # volume sur disque x epochs - _count_real_train_samples ne matche jamais les
        # chunks echecs, deferred-work.md 2026-08-07). Valeurs recalculees a la main
        # pour epochs=25 et le volume depth=12 (1 402 252 positions) : gpu (batch=256)
        # 4929 steps/epoch x 25 = 123225 ; tpu (batch=128) 9859 steps/epoch x 25 = 246475.
        #
        # 2026-08-08 : un test epochs=35/decay_steps recalcule (172515/345065) a ete
        # tente sur la meilleure config connue (dim=192/dropout=0.35/label_smoothing=0.2,
        # Val=28.00%) pour voir si Val avait encore de la marge (pas de palier net a
        # l'epoch 25). Test INVALIDE : lance par reprise depuis le checkpoint de l'ancien
        # schedule (123225) sans le vider d'abord - en plus du bug d'off-by-one deja
        # connu (deferred-work.md), le changement de decay_steps en reprise cree une
        # vraie discontinuite de LR (le checkpoint etait converge sous l'ancien schedule,
        # le nouveau schedule calcule un LR plus haut/moins decru au meme step) - le
        # modele n'a jamais retrouve son optimum (patience=8 declenchee epoch 32, jamais
        # au-dessus de 0.2800). Resultat non concluant, PAS un signe que plus d'epochs
        # n'aiderait pas - juste un test contamine. Non retente (Aymeric prefere garder
        # la config propre connue plutot que refaire un test proprement, decision
        # explicite). Revert a epochs=25/decay_steps 123225/246475 ci-dessous.
        "tpu": {
            "micro_batch_size": 128,
            "accum_steps": 1,
            "learning_rate": 4e-4,
            "weight_decay": 5e-5,
            "dropout_rate": 0.35,
            "warmup_steps": 200,
            "decay_steps": 51950,
        },
        "gpu": {
            "micro_batch_size": 256,
            "accum_steps": 1,
            "learning_rate": 8e-4,
            "weight_decay": 5e-5,
            "dropout_rate": 0.35,
            "warmup_steps": 200,
            "decay_steps": 25975,
        },
        "optimizer": "adamw",
        "lr_schedule": "cosine",
        "epochs": 25,
        "patience": 8,
        "eval_batch_size": 64,
        "save_dir": "./checkpoints_chess_search_teacher",
    },

    "CHESS_MOVE_TOKEN": {
        # === SPIKE / PROVISOIRE (Epic 11, 2026-08-10) — statut expérimental ===
        # NE PAS traiter comme une config stable au même titre que CHESS_SEARCH_TEACHER/
        # CHESS_LEGAL_MOVES/CHESS_NO_HISTORY tant que le spike côté chess_ai n'est pas
        # jugé concluant (CAP-4 ET CAP-5, chess_ai/_bmad-output/specs/
        # spec-chess-move-token-poc/SPEC.md - Success signal). Voir le spine dédié
        # (_bmad-output/planning-artifacts/architecture/architecture-chess-move-token-2026-08-10/
        # ARCHITECTURE-SPINE.md, AD-26 à AD-33) et docs/spike-chess-move-token-dataset-schema.md
        # pour le contrat de données. Décision Aymeric/Winston 2026-08-10 : priorité
        # vitesse d'expérimentation, décisions ci-dessous réversibles selon résultats.
        #
        # Transformer causal (décodeur) sur l'historique de coups (move_tokens) d'une
        # partie - prédit le coup suivant (même espace policy 4672 que le reste du
        # domaine échecs, AD-22 hérité). PAS le modèle chess_cnn_attention_* existant :
        # entrée séquentielle (pas un plan 8x8), aucune tête value (AD-26).
        "task_type": "chess_move_token",
        "num_classes": 4672,
        # input_shape=(1,) : forme du dummy d'INIT uniquement (Trainer.create_train_state,
        # trainer.py:145 - jnp.ones((1,)+input_shape)) - PAS la longueur réelle des
        # séquences (longueur FIXE calculée depuis les vraies données par le chargeur,
        # AD-28 - une seule forme de batch pour tout le run, une seule compilation
        # JIT). Un décodeur transformer n'a aucun paramètre dont la forme dépend de la
        # longueur de séquence (Embed/Dense/Attention sont tous par-token) - une longueur
        # factice de 1 suffit à initialiser correctement tous les params.
        "input_shape": (1,),
        "model_name": "chess_move_token_transformer",
        # num_layers/d_model/num_heads : hyperparamètres du transformer - forwardés
        # conditionnellement par main.py comme token_dim/num_bottleneck_tokens pour les
        # modèles chess_cnn_attention_*. d_model % num_heads == 0, validé par le modèle
        # à l'appel.
        #
        # num_layers 4->6 / d_model 128->192 (Aymeric/Winston, 2026-08-10, après le Run
        # 2 - 416 306 positions, 15 epochs, Val PolicyAccuracy plafonne à 5.54%, gap
        # train/val sain 1.76pt) : 1 994 304 -> 4 468 672 params (2.24x, mesuré via
        # model.init(), voir chess-move-token.mmd). d_model=192 reprend le token_dim
        # convergent de CHESS_SEARCH_TEACHER (pas un nouveau chiffre inventé).
        # num_layers privilégié sur d_model comme premier levier : contrairement à
        # chess_cnn_attention_policy_value (position déjà encodée en plans), ce modèle
        # doit reconstruire l'état du jeu depuis la séquence brute avant de pouvoir
        # l'exploiter - une tâche compositionnelle que la profondeur sert mieux que la
        # largeur seule. Bonus mesuré : à ce ratio, embedding+tête policy tombent à
        # 40.3% du total (contre 60.2% à 4/128) - la capacité ajoutée va au corps du
        # réseau, pas à l'overhead fixe du vocabulaire (4674 tokens).
        "num_layers": 6,
        "d_model": 192,
        "num_heads": 4,
        # compute_dtype (Aymeric/Winston, 2026-08-11, précision mixte TPU) : "bfloat16"
        # pour le calcul des couches Dense/Attention (poids "maîtres" restent en
        # float32, AD-33-adjacent mais indépendant - voir docstring
        # ChessMoveTokenTransformer). Diagnostic réel (nvtop + logs TPU, 2026-08-10/11) :
        # le modèle tournait entièrement en float32 malgré le print trompeur "TPU:
        # Utilisation de float16" (qui ne concerne QUE le cast de l'entrée, jamais le
        # calcul du modèle) - les TPU sont conçus autour de bfloat16, un modèle 100%
        # float32 en perd l'essentiel de l'avantage. Checkpoint-compatible (dtype des
        # poids stockés inchangé) - mais rappel : la reprise d'entraînement ne doit de
        # toute façon pas être utilisée sur ce domaine (mécanisme jugé peu fiable,
        # verdict Aymeric rétro Epic 10, 2026-08-09 - voir deferred-work.md). Gain
        # attendu : partiel, pas 2-3x (une partie du temps par step est dispatch/
        # pipeline, pas calcul) - à mesurer sur le prochain run, pas garanti a priori.
        "compute_dtype": "bfloat16",
        # output_prefix porte ici le chemin LITTÉRAL du fichier spike unique (pas un
        # préfixe de glob `_chunk*.npz` comme les autres domaines échecs, AD-27) -
        # ChessMoveTokenDataset (data_management.py) le consomme comme tel.
        "output_prefix": f"{DATA_ROOT}/chunks/chess_move_token_spike/chess_move_token_spike.npz",
        "val_split": 0.1,
        # loss_params.label_smoothing 0.0 -> 0.2 (Aymeric, 2026-08-10, apres le 1er run
        # 10 epochs) : reponse au surapprentissage observe (Train PolicyAccuracy=33.5%
        # vs Val=0.35% a l'epoch 6, Val Loss qui augmente des l'epoch 1 - signature
        # classique, 2871 exemples pour ~2M parametres). Meme valeur que
        # CHESS_SEARCH_TEACHER (coincidence de convergence, pas copie). Cause racine
        # = volume de donnees (dataset spike, 20 parties) - une regeneration a plus
        # grande echelle est en cours cote chess_ai ; ce reglage attenue le symptome
        # en attendant, ne le resout pas.
        "loss_params": {
            "label_smoothing": 0.2,
        },

        # === Hyperparamètres GPU/TPU ===
        # HISTORIQUE (pour comprendre les valeurs ci-dessous, pas à ré-appliquer) :
        # Run 1 (20 parties/3190 positions) -> surapprentissage sévère (Train 33.5%/Val
        # 0.35%). Run 2 (régénération 137x, 416 306 positions, batch=128 après un OOM
        # réel à 256 sur GPU local - self-attention O(B×H×L²), L=404 -
        # decay_steps=43905/15 epochs) -> plateau propre, Val PolicyAccuracy 5.54%, gap
        # sain 1.76pt (voir chess-move-token.mmd, E1-E3). Conclusion Aymeric/Winston
        # 2026-08-10 : volume de données = facteur limitant principal.
        #
        # RUN 3 (2026-08-11) : dataset régénéré côté chess_ai à l'échelle cible -
        # mesuré RÉELLEMENT (pas estimé) depuis le fichier .npz réel via le motif de
        # reset des longueurs d'historique (une partie = une séquence croissante
        # 4,5,6...  qui retombe à 4 au début de la suivante) : **9980 parties,
        # 1 393 040 positions** (139,58 positions/partie en moyenne) - quasiment la
        # parité avec CHESS_SEARCH_TEACHER (10 000 parties, 1 402 252 positions).
        # val_split=0.1 -> 139 304 val / 1 253 736 train.
        #
        # micro_batch_size : gpu 128 -> **64** (Aymeric, 2026-08-11 - 128 restait trop
        # haut, nouveau crash sur le GPU local même après le fix Run 2 : le modèle est
        # désormais plus gros aussi, 6/192 au lieu de 4/128, AD-28/self-attention
        # O(B×H×L²) toujours en cause). tpu reste à 256 (Aymeric compte entraîner sur
        # TPU Colab suite aux crashs répétés en local - aucun historique d'OOM constaté
        # à cette échelle sur TPU, mémoire par cœur différente d'une carte locale).
        #
        # decay_steps RECALCULÉ depuis le volume réel ci-dessus (steps/epoch =
        # train_positions // micro_batch_size, x epochs=15 ci-dessous) :
        # gpu (batch=64) 1253736//64=19589 steps/epoch x 15 = 293835 ;
        # tpu (batch=256) 1253736//256=4897 steps/epoch x 15 = 73455.
        # _count_real_train_samples (trainer.py:45) ne matche jamais les fichiers
        # échecs (même limitation documentée pour CHESS_SEARCH_TEACHER) - ces valeurs
        # sont donc bien le repli réellement utilisé.
        #
        # max_seq_len recalculé automatiquement par ChessMoveTokenDataset depuis les
        # vraies données à chaque chargement (AD-28) - rien à changer ici (toujours
        # 404 sur ce nouveau fichier, même max qu'au Run 2), une seule compilation JIT
        # quel que soit le dataset.
        #
        # dropout_rate=0.3 / label_smoothing=0.2 (voir loss_params ci-dessus) : hérités
        # du Run 1, CONSERVÉS inchangés pour le Run 3 - le volume est désormais à
        # parité avec CHESS_SEARCH_TEACHER, donc plus loin de la cause du
        # surapprentissage initial (2871 exemples) qu'ils visaient à corriger ; à
        # réévaluer seulement si le Run 3 montre un gap qui se creuse anormalement,
        # pas par précaution préventive.
        #
        # decay_steps recalculés une 2e fois (Aymeric, 2026-08-11) pour epochs=15->25
        # (voir === Entraînement === ci-dessous, alignement complet sur
        # CHESS_SEARCH_TEACHER maintenant que le volume est à parité) : gpu (batch=64)
        # 19589 steps/epoch x 25 = 489725 ; tpu (batch=256) 4897 steps/epoch x 25 =
        # 122425.
        "tpu": {
            "micro_batch_size": 256,
            "accum_steps": 1,
            "learning_rate": 3e-4,
            "weight_decay": 5e-5,
            "dropout_rate": 0.3,
            "warmup_steps": 200,
            "decay_steps": 122425,
        },
        "gpu": {
            "micro_batch_size": 64,
            "accum_steps": 1,
            "learning_rate": 3e-4,
            "weight_decay": 5e-5,
            "dropout_rate": 0.3,
            "warmup_steps": 200,
            "decay_steps": 489725,
        },

        # === Entraînement ===
        # epochs=15 -> 25 (Aymeric, 2026-08-11, Run 3) : alignement complet sur
        # CHESS_SEARCH_TEACHER (epochs=25/patience=8, déjà identique) maintenant que le
        # volume de données est à parité (1 393 040 vs 1 402 252 positions) - une vraie
        # tentative de convergence comparable, pas juste "plus de données que le smoke-
        # test initial" (raison du 15 précédent). patience=8 déjà alignée, inchangée.
        # Reste réversible/ajustable (priorité vitesse d'expérimentation, spine
        # architecture-chess-move-token-2026-08-10) - pas un verdict figé.
        "optimizer": "adamw",
        "lr_schedule": "cosine",
        "epochs": 25,
        "patience": 8,

        # === Évaluation ===
        "eval_batch_size": 64,

        # === Sauvegarde ===
        "save_dir": "./checkpoints_chess_move_token",
    },

    "CHESS_TOKEN": {
        # === SPIKE / PROVISOIRE (spec-chess-token-candidate-model, 2026-08-13) — statut
        # expérimental === NE PAS traiter comme une config stable au même titre que
        # CHESS_SEARCH_TEACHER/CHESS_LEGAL_MOVES/CHESS_NO_HISTORY tant que le verdict
        # CAP-4/CAP-5/CAP-6 (accuracy/paramètres/convergence, mesures de référence, pas
        # un pass/fail contre le checkpoint 28,00%) n'est pas tranché. Voir
        # _bmad-output/specs/spec-chess-token-candidate-model/SPEC.md +
        # _bmad-output/implementation-artifacts/spec-chess-token-candidate-model.md
        # (tâches/Design Notes) pour le contrat complet.
        #
        # Tronc auto-attention PURE sur les 64 tokens-case (remplace le tronc CNN
        # existant) + bottleneck K=8/token_dim=64 (défauts model_library.py, décision
        # explicite d'Aymeric de repartir "à neuf" plutôt que de chercher une
        # comparabilité stricte avec le checkpoint 28,00%, K=16/token_dim=196) + tête
        # de scoring sur les 50 candidats légaux du dataset (remplace Dense(4672),
        # ChessTokenCandidateModel, model_library.py).
        "task_type": "chess_token",
        # num_classes porte ici MAX_CANDIDATES=50 (contrat .npz chess_token_candidate_spike,
        # côté chess_ai) - PAS un nombre de classes au sens habituel, même convention que
        # num_classes=4672 sur les configs chess_cnn_attention_*/chess_move_token
        # (create_chess_token_candidate_model fait le pont, model_library.py).
        "num_classes": 50,
        # input_shape=(120,) : forme du dummy d'INIT uniquement (Trainer.create_train_state,
        # trainer.py:145 - jnp.ones((1,)+input_shape)), PAS une longueur de séquence.
        # 120 = 64 (token_position) + 6 (global_flags) + 50 (candidate_moves) - les 3
        # champs du contrat .npz que le MODÈLE consomme, concaténés en un seul tenseur
        # par ChessTokenCandidateDataset (data_management.py) car Trainer._train_step/
        # _eval_step n'acheminent jamais qu'UNE SEULE entrée `images` vers `apply_fn`
        # (vérifié en lisant trainer.py:312-313/429-430 avant de trancher, plutôt que
        # de supposer un support multi-entrée qui n'existe pas dans ce pipeline -
        # même situation que le tenseur unique (sequence) de CHESS_MOVE_TOKEN
        # ci-dessus, généralisée à 3 champs concaténés au lieu d'1 seul).
        # candidate_mask (N,50) N'EST PAS inclus ici : le modèle n'en a pas besoin en
        # interne (slots de padding neutralisés via `safe_moves`, voir docstring
        # ChessTokenCandidateModel) - il voyage côté targets (dict), pas côté images.
        "input_shape": (120,),
        "model_name": "chess_token_candidate_model",
        "token_dim": 128,
        # token_dim 64->128 (Aymeric, 2026-08-14, après le run 2 - Val PolicyAccuracy
        # plafonnée à 19,21% dès l'epoch ~40, train/val plafonnant ENSEMBLE, gap sain
        # 1,7pt) : SEUL levier changé ce run (K=8/num_heads=4/num_trunk_layers=2 restent
        # aux défauts de la classe) - discipline "un seul levier à la fois" reprise de
        # CHESS_SEARCH_TEACHER (E1-E5, chess-cnn-attention-policy-value.mmd). Précédent
        # direct dans ce même domaine (CHESS_SEARCH_TEACHER, mêmes tests E1/E2) : K
        # 8->16 n'avait RIEN changé à l'accuracy (31,18%->31,05%), alors que token_dim
        # 64->128 avait fait sauter la Val de 29,37%->35,28% - token_dim est le vrai
        # levier de capacité de ce domaine, pas K. Réserve à surveiller (même précédent) :
        # ce même doublement avait fait passer le gap train/val de 1,9 à 7,2pt sur
        # CHESS_SEARCH_TEACHER - gap de départ ici plus sain (1,7pt), donc marge, mais
        # à re-mesurer sur ce run, pas supposé acquis. num_trunk_layers reste sans ligne
        # de forwarding dédiée dans main.py (pas utilisé ce run non plus).
        #
        # output_prefix porte ici le chemin LITTÉRAL du fichier spike unique (pas un
        # préfixe de glob `_chunk*.npz` comme les 4 domaines échecs du contrat stable) -
        # même convention que CHESS_MOVE_TOKEN. Run réel généré côté chess_ai
        # (2026-08-13) : 3000 parties, seed=42, 412 574 positions (professeur Stockfish
        # profondeur 12, mêmes opening_plies=4/max_halfmoves=400 que CHESS_SEARCH_TEACHER
        # - voir token-candidate-dataset-schema.md, chess_ai).
        #
        # RÉGÉNÉRATION 2026-08-13 (Aymeric, après le 1er run complet 25 epochs -
        # Val PolicyAccuracy plafonnée à 16,20%, train/val plafonnant ENSEMBLE dès
        # l'epoch ~20, gap sain 0,9pt - lecture Winston : signature de plafond de
        # CAPACITÉ du modèle plutôt que de famine de données, mais décision Aymeric
        # de tripler le volume en 1er levier quand même) : dataset régénéré ~3x,
        # **1 244 231 positions** mesurées directement dans le fichier réel (pas
        # estimé) - à garder à l'esprit pour interpréter le prochain run si la Val
        # ne bouge que marginalement malgré le volume x3 (cf. lecture Winston
        # ci-dessus, memlog spec-chess-token-candidate-model).
        "output_prefix": f"{DATA_ROOT}/chunks/chess_token_candidate_spike/chess_token_candidate_spike.npz",
        "val_split": 0.1,
        # loss_params.label_smoothing 0.0->0.2 (Aymeric, 2026-08-14, apres un
        # overfitting marque observe au run 3, token_dim 64->128) : compute_chess_token_
        # candidate_loss implemente sa PROPRE variante masquee du label smoothing
        # (voir sa docstring, loss_functions.py) - PAS une reutilisation de smooth_labels/
        # compute_chess_policy_loss (repartirait la masse sur des slots de padding).
        # Meme valeur que CHESS_MOVE_TOKEN a sa creation (dataset_configs.py ci-dessus,
        # 2026-08-10) - meme motivation (overfitting), pas une coincidence recherchee.
        "loss_params": {
            "label_smoothing": 0.2,
        },
        #
        # === Hyperparamètres GPU/TPU ===
        # PROVISOIRE - point de départ non tuné empiriquement (comme CHESS_LEGAL_MOVES/
        # CHESS_MOVE_TOKEN à leur création), pas des valeurs mesurées sur un run réel de
        # CE modèle. learning_rate=3e-4 aligné sur CHESS_MOVE_TOKEN (autre tronc
        # attention, pas un CNN comme CHESS_SEARCH_TEACHER qui tolère 4e-4/8e-4) -
        # dropout_rate 0.1->0.3 (Aymeric, 2026-08-14, même run/même motivation que
        # label_smoothing ci-dessus - overfitting marqué après token_dim 64->128,
        # gap train/val attendu en hausse par rapport aux runs précédents à token_dim=64
        # - voir memlog spec-chess-token-candidate-model). Ancienne valeur (0.1,
        # défaut de la classe model_library.py) conservée en historique : pas le 0.35
        # de CHESS_SEARCH_TEACHER (ce tronc reste structurellement plus petit - L=64
        # tokens, attention O(L²) largement moins coûteuse que les L=404 de
        # CHESS_MOVE_TOKEN qui avait forcé micro_batch_size=64 sur GPU local - aucun
        # historique d'OOM attendu à cette échelle, confirmé par smoke test réel
        # batch=256/token_dim=128, 2026-08-14).
        #
        # decay_steps RECALCULÉS 2026-08-13 (Aymeric, après régénération x3 ci-dessus +
        # gpu.micro_batch_size 128->256 + epochs 25->50, ce dernier motivé par la courbe
        # du 1er run qui progressait encore, quoique lentement, à l'epoch 25 - pas
        # encore franchement plafonnée au sens strict malgré la lecture "plafond de
        # capacité" du memlog). Volume RÉEL mesuré directement dans le fichier régénéré
        # (1 244 231 positions, PAS une extrapolation x3 du chiffre précédent - même
        # discipline que la mesure initiale) : val_split=0.1 -> n_val=max(1,int(1244231*0.1))
        # =124423, train=1119808, même formule que ChessTokenCandidateDataset.__init__.
        # steps/epoch = train // micro_batch_size, x epochs=50 ci-dessous : gpu ET tpu
        # (batch=256 désormais identique des deux côtés) 1119808//256=4374 x50=218700.
        # _count_real_train_samples (trainer.py:45) ne matche jamais les fichiers échecs
        # à fichier unique (même limitation documentée pour CHESS_SEARCH_TEACHER/
        # CHESS_MOVE_TOKEN) - cette valeur codée en dur est donc bien le repli réellement
        # utilisé ; à recalculer de nouveau si le dataset ou micro_batch_size rechangent.
        "tpu": {
            "micro_batch_size": 256,
            "accum_steps": 1,
            "learning_rate": 3e-4,
            "weight_decay": 5e-5,
            "dropout_rate": 0.3,
            "warmup_steps": 200,
            "decay_steps": 218700,
        },
        "gpu": {
            "micro_batch_size": 256,
            "accum_steps": 1,
            "learning_rate": 3e-4,
            "weight_decay": 5e-5,
            "dropout_rate": 0.3,
            "warmup_steps": 200,
            "decay_steps": 218700,
        },

        # === Entraînement ===
        # epochs=25->50 (Aymeric, 2026-08-13, après le 1er run - voir décalage de
        # décision ci-dessus) / patience=8 inchangée : même mécanisme d'arrêt anticipé
        # standard que les autres configs échecs (CAP-6 du SPEC - PAS de détection sur
        # mesure d'un décrochage contre la baseline CNN, décision explicite "on repart
        # à neuf").
        "optimizer": "adamw",
        "lr_schedule": "cosine",
        "epochs": 50,
        "patience": 8,

        # Pas de "eval_batch_size" ici (retiré, code review 2026-08-13) : cette clé
        # n'était que du boilerplate copié depuis les configs sœurs (section
        # "=== Évaluation ==="), jamais lue par le chemin chess_token
        # (get_datasets()/ChessTokenStrategy.generate_reports rebatchent la
        # validation avec le même batch_size que train, pas eval_batch_size -
        # seule ClassificationStrategy.generate_reports la consomme,
        # task_strategies.py) - clé morte, retirée plutôt que gardée pour une
        # cohérence de forme purement cosmétique.

        # === Sauvegarde ===
        "save_dir": "./checkpoints_chess_token",
    },

    "CHESS_TOKEN_1_MOVE": {
        # === SPIKE / PROVISOIRE (spec-chess-token-1-move, 2026-08-15) — statut
        # expérimental === NE PAS traiter comme une config stable au même titre que
        # CHESS_SEARCH_TEACHER/CHESS_LEGAL_MOVES tant que le verdict (accuracy jointe
        # face au 28,00% val PolicyAccuracy de CHESS_SEARCH_TEACHER) n'est pas tranché.
        # Voir _bmad-output/implementation-artifacts/spec-chess-token-1-move.md
        # (tâches/Design Notes/Verification) + docs/contract-chess-ai-training-interface.md
        # §3 CHESS_TOKEN_1_MOVE pour le contrat complet.
        #
        # Diagnostic motivant ce domaine (contrat §3) : CHESS_TOKEN (ci-dessus, spike
        # non concluant) score 50 candidats PRÉ-FILTRÉS par python-chess côté chess_ai -
        # par construction, ce scorer ne peut JAMAIS produire de coup illégal, ce qui
        # empêche de mesurer si le réseau "connaît" les règles. Ce domaine prédit
        # directement le coup dans l'espace complet (from_square 64 x move_type 73 =
        # 4672, 2 têtes indépendantes) SANS filtrage préalable - l'illégalité redevient
        # un échec mesurable (aucun masquage de légalité côté modèle/loss, jamais - voir
        # ChessTokenOneMoveModel/compute_chess_token_1_move_loss).
        #
        # Tronc auto-attention PURE sur les 64 tokens-case, copié PARTIELLEMENT de
        # ChessTokenCandidateModel (mêmes token_embed/pos_embed/global_flags_proj/blocs
        # auto-attention) mais SANS son étage bottleneck (pas de queries apprises K, pas
        # de cross-/auto-attention vers des tokens latents - contrainte explicite du
        # contrat) - le tronc alimente directement 2 têtes Dense indépendantes via un
        # simple mean-pool sur les 64 tokens (ChessTokenOneMoveModel, model_library.py).
        # === v2 (spec-chess-token-1-move-v2, 2026-08-16) === v1 éliminée (plateau
        # JointMoveAccuracy=5,10% dès l'epoch ~34/50, train≈val = plafond structurel,
        # voir contrat §3 "Résultat 2026-08-15"). Cette entrée est éditée EN PLACE
        # (pas de nouveau domaine _V2 - décision explicite Aymeric) pour tester 3
        # leviers combinés en un seul run : (1) conditionnement léger move_type sur
        # from_square (teacher-forcing en training uniquement - voir
        # ChessTokenOneMoveModel, model_library.py), (2) read_token appris remplaçant
        # le mean-pool, (3) tronc élargi (token_dim 32->64, num_trunk_layers 1->2, voir
        # ci-dessous). ATTENTION - AUCUN mécanisme automatique ne préserve l'ancien
        # checkpoint : _get_export_path (task_strategies.py) dérive le nom de fichier de
        # `dataset_name` seul (best_model_chess_token_1_move.pkl, IDENTIQUE v1/v2) - tout
        # nouveau training ÉCRASE silencieusement le précédent. Le checkpoint v1 a
        # d'ailleurs déjà été perdu ainsi (2026-08-16, smoke-test v2 lancé avant le
        # `cp` manuel prévu - trouvaille Blind Hunter, ce commentaire promettait à tort
        # une sauvegarde automatique). Si un checkpoint mérite d'être conservé : le
        # copier MANUELLEMENT (`cp best_model_chess_token_1_move.pkl <nom>.pkl`) AVANT de
        # relancer un training, jamais après.
        "task_type": "chess_token_1_move",
        # num_classes NE PORTE RIEN DE DIRECT ICI (Design Notes du SPEC) : ce modèle a 2
        # têtes de tailles DIFFÉRENTES (from_square=64, move_type=73, codées en dur dans
        # ChessTokenOneMoveModel, propriété fixe du schéma move_to_index existant) - PAS
        # un seul nombre de classes comme num_classes=50 sur CHESS_TOKEN ci-dessus ou
        # num_classes=4672 sur CHESS_MOVE_TOKEN/CHESS_LEGAL_MOVES. Convention choisie ici
        # (cohérente avec le précédent num_classes=50=MAX_CANDIDATES de CHESS_TOKEN, où
        # num_classes porte déjà un sens détourné de son nom générique) : valeur
        # SENTINELLE -1 (jamais une taille de tête réelle, ne peut pas être confondue
        # avec un num_classes valide d'un autre domaine) - create_chess_token_one_move_model
        # (model_library.py) l'ignore explicitement et ne la transmet jamais à
        # ChessTokenOneMoveModel. La plomberie générique de main.py (model_kwargs =
        # {"num_classes": ..., "dropout_rate": ...}) construit et transmet cette clé sans
        # condition pour TOUS les domaines - la factory est le point où elle est
        # neutralisée, pas cette config.
        "num_classes": -1,
        # input_shape=(71,) : forme du dummy d'INIT uniquement (Trainer.create_train_state,
        # trainer.py:145 - jnp.ones((1,)+input_shape)), PAS une longueur de séquence.
        # 71 = 64 (token_position) + 6 (global_flags) + 1 (from_square_teacher, v2,
        # spec-chess-token-1-move-v2 - PAS candidate_moves/candidate_mask, toujours pas
        # packés ici, ChessTokenOneMoveModel ne les consomme jamais en entrée). 70->71
        # (v2) : from_square_teacher = move_index // 73, packé comme 71e colonne par
        # ChessTokenOneMoveDataset (data_management.py) - label réel du from_square joué,
        # utilisé par le modèle UNIQUEMENT en teacher-forcing (training=True) pour
        # conditionner move_type_head ; ignoré et remplacé par argmax(from_square_logits)
        # quand training=False (jamais de fuite de label en évaluation - voir
        # ChessTokenOneMoveModel.__call__, model_library.py).
        "input_shape": (71,),
        "model_name": "chess_token_one_move_model",
        # token_dim/num_heads/num_trunk_layers : v1 avait délibérément un tronc ALLÉGÉ
        # (token_dim=32/num_trunk_layers=1) pour que token_embed/pos_embed pèsent une
        # part SIGNIFICATIVE des paramètres totaux (objectif chess_bottleneck_genetic.py
        # côté chess_ai, contrat §3) - conservé en v2 (read_token/conditionnement
        # n'ajoutent pas de nouvelle table indépendante, voir docstring
        # ChessTokenOneMoveModel). token_dim 32->64/num_trunk_layers 1->2 (v2, levier 3
        # du diagnostic 2026-08-15, contrat §3 "Piste v2" option 3 - "confirmé dans ce
        # domaine, CHESS_SEARCH_TEACHER 64->128 = +6pts") : tronc élargi, testé combiné
        # avec les leviers 1 (conditionnement) et 2 (read_token) plutôt qu'isolé. Valeurs
        # NON TUNÉES individuellement (point de départ du run combiné v2, pas un
        # hyperparamètre optimisé) - forwardées à la factory via les branches génériques
        # déjà existantes dans main.py (if "token_dim"/"num_heads"/"num_trunk_layers" in
        # config), aucune modification de main.py nécessaire pour ces 3 clés.
        "token_dim": 64,
        "num_heads": 4,
        "num_trunk_layers": 2,
        #
        # output_prefix : MÊME fichier .npz que CHESS_TOKEN ci-dessus (chemin LITTÉRAL
        # du fichier spike unique, pas un préfixe de glob `_chunk*.npz`) - réutilisé TEL
        # QUEL (contrat §3, correction 2026-08-15 : PAS chess_search_teacher, dont les
        # plans binaires encode_position sont incompatibles avec token_embed/pos_embed).
        # 1 244 231 positions (professeur Stockfish profondeur 12, même professeur que
        # CHESS_SEARCH_TEACHER), même volume que CHESS_TOKEN ci-dessus.
        "output_prefix": f"{DATA_ROOT}/chunks/chess_token_candidate_spike/chess_token_candidate_spike.npz",
        "val_split": 0.1,
        # loss_params : point de départ NEUTRE et NON TUNÉ (aucun run réel de ce domaine
        # au moment de la création de cette config) - poids égal entre les 2 têtes
        # (from_square_weight=move_type_weight=1.0, ni tête privilégiée ni pénalisée) et
        # aucun label smoothing (0.0, contrairement à CHESS_TOKEN/CHESS_MOVE_TOKEN qui
        # l'ont activé APRÈS avoir observé un overfitting sur un run réel - rien de tel
        # observé ici pour l'instant, à revisiter après le 1er run complet).
        "loss_params": {
            "from_square_weight": 1.0,
            "move_type_weight": 1.0,
            "label_smoothing": 0.0,
        },
        #
        # === Hyperparamètres GPU/TPU ===
        # PROVISOIRE - point de départ non tuné empiriquement (comme CHESS_TOKEN à sa
        # création), pas des valeurs mesurées sur un run réel de CE modèle.
        # learning_rate=3e-4 aligné sur CHESS_TOKEN/CHESS_MOVE_TOKEN (même famille de
        # tronc auto-attention, pas un CNN comme CHESS_SEARCH_TEACHER qui tolère des
        # learning rates plus élevés). dropout_rate=0.1 = défaut de la classe
        # ChessTokenOneMoveModel (PAS le 0.3 de CHESS_TOKEN, qui l'avait augmenté APRÈS
        # un overfitting observé sur un run réel à token_dim=128 - ce tronc est plus
        # étroit/allégé dès le départ, moins de risque d'overfitting a priori, à
        # re-mesurer sur le 1er run réel plutôt que supposé acquis).
        #
        # decay_steps : MÊME volume/MÊME micro_batch_size que CHESS_TOKEN ci-dessus
        # (train=1 119 808, val_split=0.1 sur 1 244 231 positions mesurées directement
        # dans le fichier réel - même formule que ChessTokenOneMoveDataset.__init__) :
        # steps/epoch = 1119808 // 256 = 4374, x epochs=50 ci-dessous = 218700 - même
        # discipline "recalculé pour le nombre d'epochs choisi" que CHESS_TOKEN (Tasks &
        # Acceptance du SPEC), epochs=50 choisi ici pour rester comparable au même
        # volume/même batch size, pas une extrapolation. _count_real_train_samples
        # (trainer.py:45) ne matche jamais les fichiers échecs à fichier unique (même
        # limitation documentée pour CHESS_TOKEN/CHESS_SEARCH_TEACHER/CHESS_MOVE_TOKEN) -
        # cette valeur codée en dur est donc bien le repli réellement utilisé ; à
        # recalculer de nouveau si le dataset ou micro_batch_size rechangent.
        "tpu": {
            "micro_batch_size": 256,
            "accum_steps": 1,
            "learning_rate": 3e-4,
            "weight_decay": 5e-5,
            "dropout_rate": 0.1,
            "warmup_steps": 200,
            "decay_steps": 218700,
        },
        "gpu": {
            "micro_batch_size": 256,
            "accum_steps": 1,
            "learning_rate": 3e-4,
            "weight_decay": 5e-5,
            "dropout_rate": 0.1,
            "warmup_steps": 200,
            "decay_steps": 218700,
        },

        # === Entraînement ===
        # epochs=50/patience=8 : même mécanisme d'arrêt anticipé standard que les autres
        # configs échecs (aucune détection sur mesure d'un décrochage contre la baseline
        # CNN, même discipline que CHESS_TOKEN).
        "optimizer": "adamw",
        "lr_schedule": "cosine",
        "epochs": 50,
        "patience": 8,

        # === Sauvegarde ===
        "save_dir": "./checkpoints_chess_token_1_move",
    },
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

    # value_head_trained (Epic 10, 2026-08-04) : dérivé automatiquement de
    # loss_params.value_weight plutôt qu'un littéral séparé par config - élimine tout
    # risque de désynchronisation entre les deux si l'un est modifié sans l'autre.
    # Scope volontairement restreint aux configs qui déclarent déjà value_weight
    # (CHESS_NO_HISTORY, CHESS_SEARCH_TEACHER) - ne pollue pas les configs sans tête
    # value (CHESS_LEGAL_MOVES, domaines non-échecs). Consommé par le checkpoint "export
    # pur" (TaskStrategy.export_model, sérialise ce dict tel quel) - traçabilité pour tout
    # futur consommateur (contrat #2.3), sans modification de export_model/trainer.py.
    if "loss_params" in config and "value_weight" in config["loss_params"]:
        # > 0 (pas != 0) : un value_weight negatif serait une erreur de config (inverse le
        # signal de la loss), jamais une "tete entrainee" legitime - revue de code 2026-08-04.
        config["value_head_trained"] = config["loss_params"]["value_weight"] > 0

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
    # class_names/image_size redevenus optionnels (2026-07-27, Story 9.3, voir
    # validate_config) - CHESS_NO_HISTORY n'a ni l'un ni l'autre (num_classes y porte le
    # littéral 4672, la shape est un littéral en dur dans la config, pas dérivée d'un module).
    print(f"Noms: {config.get('class_names', 'N/A')}")
    print(f"Image size: {config.get('image_size', 'N/A')}")
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
            print(f"   ✓ {config['num_classes']} classes: {config.get('class_names', 'N/A')}")
        except Exception as e:
            print(f"   ❌ Erreur: {e}")
    
    print("\n🎯 Affichage d'une config complète:")
    print_config("FIGHTERJET_CLASSIFICATION")
