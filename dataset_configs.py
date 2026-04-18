"""
Configuration centralisée pour tous les datasets
Permet de gérer facilement plusieurs datasets avec leurs paramètres spécifiques
"""

import numpy as np


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
        "num_classes": 35,
        "class_names": ['a4', 'a10', 'a400m', 'alphajet', 'b1b', 'b2', 'b52', 'c5', 'c130', 'c17', 'f4', 'f14', 'f15', 'f16', 'f18', 'f22', 'f35', 'f117', 'flanker', 'gripen', 'harrier', 'hawk', 'hawkeye', 'jaguar', 'mig29', 'mirage2000', 'miragef1', 'mustang', 'rafale', 'spitfire', 'su57', 'sr71', 'tornado', 'typhoon', 'v22'],
        "data_dir": "/home/aobled/Downloads/_balanced_dataset_split",
        "output_prefix": "./data/chunks/classification/dataset_classification",
        "chunk_size": 27000,
        "image_size": (128, 128),
        "grayscale": True,  # ✅ GRAYSCALE (3× plus rapide, même accuracy)
        "augmentation_params": {
            "flip_h": True,
            "flip_v": False,
            "rotation_factor": 0.10,      # Adouci (était 0.15, 0.12 avant)
            "zoom_factor": 0.10,          # Adouci (était 0.15)
            "translation_factor": 0.06,   # Divisé par 2 (0.12 -> 0.06) : Assez pour garder l'invariance anti-scintillement sans détruire la cible
            "brightness_delta": 0.10,
            "contrast_factor": 0.10
        },
        
        "mean": None,
        "std": None,
        "mean_std_path": "./data/chunks/classification/dataset_classification_meanstd.npz",
        
        # === Modèle ===
        "model_name": "sophisticated_cnn_128_plus",  # ✅ OPTIMAL: Version optimisée+ (4M params, 88% val)
        
        # === Hyperparamètres TPU ===
        "tpu": {
            "micro_batch_size": 128,   # ✅ Optimal testé
            "accum_steps": 4,          # ✅ Batch effectif: 512 (sweet spot)
            "learning_rate": 8e-3,     # 🔥 NOUVEAU: LR augmenté pour apprentissage plus agressif
            "weight_decay": 5e-5,      # ✅ Optimal trouvé
            "dropout_rate": 0.0,       # 🔥 NOUVEAU: Pas de dropout pour exploiter le potentiel
        },
        
        # === Hyperparamètres GPU ===
        "gpu": {
            "micro_batch_size": 128,   # ✅ Testé, fonctionne
            "accum_steps": 4,          # Batch effectif: 512
            "learning_rate": 8e-4,     # 🔥 NOUVEAU: LR/10 pour GPU (augmenté aussi)
            "weight_decay": 5e-5,
            "dropout_rate": 0.0,       # 🔥 NOUVEAU: Pas de dropout pour GPU aussi
        },
        
        # === Entraînement ===
        "epochs": 40,              # 40
        "patience": 5,
        "warmup_steps": 1200,      # Rendu explicite (était hérité du Trainer)
        "decay_steps": 6000,       # Le LR chute vite et stagne à 0 pour fine-tuning après ~15 epochs
        "label_smoothing": 0.1,    # ✅ Aide légèrement
        "mixup_alpha": 0.05,        # ✅ OPTIMAL: Mixup doux (meilleur compromis trouvé)
        
        # === Évaluation ===
        "eval_batch_size": 128,
        "eval_use_subset": True,
        "eval_max_subset": 100000,  
        
        # === Sauvegarde ===
        "checkpoint_path": "best_model.pkl",
        "confusion_matrix_path": "confusion_matrix.png",
    },
    
    "FIGHTERJET_VIT": {
        # === CONFIG VISION TRANSFORMER 128×128 ===
        # Test si attention globale (tokens) améliore pour classes similaires
        # Objectif : Capturer relations longue distance (nez+ailes+empennage)
        
        # === Données ===
        "num_classes": 11,
        "class_names": ['a10', 'c17', 'f14', 'f15', 'f16', 'f18', 'f22', 'f35', 'mirage2000', 'rafale', 'typhoon'],
        "data_dir": "/home/aobled/Downloads/_balanced_dataset_split",  # 🎯 Dataset Stretched (meilleur trouvé)
        "output_prefix": "./data/chunks/dataset_chunked_vit",
        "chunk_size": 15000,
        "image_size": (128, 128),
        "grayscale": True,
        "augmentation_params": {
            "flip_h": True,
            "flip_v": True,
            "rotation_factor": 0.30,      
            "zoom_factor": 0.25,          
            "translation_factor": 0.15,   
            "brightness_delta": 0.30,
            "contrast_factor": 0.30
        },
        
        "mean": None,
        "std": None,
        "mean_std_path": "./data/chunks/dataset_chunked_vit_meanstd.npz",
        
        # === Modèle ===
        "model_name": "tiny_vit_plus_balanced",  # 🤖 ViT : patch_size=8, 256 tokens, 2.8M params
        
        # === Hyperparamètres TPU ===
        "tpu": {
            "micro_batch_size": 128,   # Optimal pour 128×128
            "accum_steps": 4,          # Batch effectif: 512 (optimal testé)
            "learning_rate": 3e-3,     # 🤖 ViT préfère LR plus bas (÷2 vs CNN)
            "weight_decay": 1e-4,      # 🤖 ViT a besoin plus de régularisation (×2)
            "dropout_rate": 0.2,       # 🤖 ViT overfit facilement
        },
        
        # === Hyperparamètres GPU ===
        "gpu": {
            "micro_batch_size": 128,
            "accum_steps": 4,
            "learning_rate": 3e-4,     # ViT LR/10 pour GPU
            "weight_decay": 1e-4,
            "dropout_rate": 0.2,
        },
        
        # === Entraînement ===
        "epochs": 80,              # 🤖 80 epochs (reprise depuis 40, attention commence à se fixer)
        "patience": 10,            # Augmenté pour laisser plus de temps (ViT lent)
        "label_smoothing": 0.15,   # 🤖 ViT bénéficie plus du label smoothing
        "mixup_alpha": 0.0,
        
        # === Évaluation ===
        "eval_batch_size": 128,
        "eval_use_subset": True,
        "eval_max_subset": 3000,
        
        # === Sauvegarde ===
        "checkpoint_path": "best_model_vit.pkl",
        "confusion_matrix_path": "confusion_matrix_vit.png",
    },
    
    "FIGHTERJET_HYBRID_VIT": {
        # === CONFIG HYBRID VISION TRANSFORMER 128×128 ===
        # CNN Stem + Transformer pour combiner inductive bias local + attention globale
        # Objectif : Meilleure performance que ViT pur sur petit dataset (60-75% attendu)
        
        # === Données ===
        "num_classes": 11,
        "class_names": ['a10', 'c17', 'f14', 'f15', 'f16', 'f18', 'f22', 'f35', 'mirage2000', 'rafale', 'typhoon'],
        "data_dir": "/home/aobled/Downloads/_balanced_dataset_split",  # 🎯 Dataset Stretched
        "output_prefix": "./data/chunks/dataset_chunked_hybrid",
        "chunk_size": 15000,
        "image_size": (128, 128),
        "grayscale": True,
        "augmentation_params": {
            "flip_h": True,
            "flip_v": True,
            "rotation_factor": 0.30,      
            "zoom_factor": 0.25,          
            "translation_factor": 0.15,   
            "brightness_delta": 0.30,
            "contrast_factor": 0.30
        },
        
        "mean": None,
        "std": None,
        "mean_std_path": "./data/chunks/dataset_chunked_hybrid_meanstd.npz",
        
        # === Modèle ===
        "model_name": "hybrid_tiny_vit",  # 🤖 Hybrid : Conv Stem + Transformer
        
        # === Hyperparamètres TPU ===
        "tpu": {
            "micro_batch_size": 128,   # Optimal pour 128×128
            "accum_steps": 4,          # Batch effectif: 512
            "learning_rate": 3e-4,     # 🤖 LR suggéré pour Hybrid (AdamW optimal)
            "weight_decay": 5e-2,      # 🤖 WD suggéré pour Hybrid (0.05)
            "dropout_rate": 0.1,       # Dropout dans le modèle
        },
        
        # === Hyperparamètres GPU ===
        "gpu": {
            "micro_batch_size": 32,    # Réduit pour GPU 733 MB VRAM
            "accum_steps": 16,         # Maintenir batch effectif 512
            "learning_rate": 3e-5,     # LR/10 pour GPU
            "weight_decay": 5e-2,
            "dropout_rate": 0.1,
        },
        
        # === Entraînement ===
        "epochs": 60,              # 🤖 60 epochs pour Hybrid (plus rapide que ViT pur)
        "patience": 10,            # Patience élevée pour laisser converger
        "label_smoothing": 0.15,   # Label smoothing modéré
        "mixup_alpha": 0.0,
        
        # === Évaluation ===
        "eval_batch_size": 128,
        "eval_use_subset": True,
        "eval_max_subset": 3000,
        
        # === Sauvegarde ===
        "checkpoint_path": "best_model_hybrid.pkl",
        "confusion_matrix_path": "confusion_matrix_hybrid.png",
    },
    
    "FIGHTERJET_LETTERBOX": {
        # === CONFIG LETTERBOX 128×128 (pour référence) ===
        # Dataset avec ratio préservé + padding miroir
        # Résultat : 77.66% (moins bon que stretched)
        
        # === Données ===
        "task_type": "classification",
        "num_classes": 11,
        "class_names": ['a10', 'c17', 'f14', 'f15', 'f16', 'f18', 'f22', 'f35', 'mirage2000', 'rafale', 'typhoon'],
        "data_dir": "/home/aobled/Downloads/_balanced_dataset_split_letterbox",
        "output_prefix": "./data/chunks/dataset_chunked_letterbox",
        "chunk_size": 15000,
        "image_size": (128, 128),
        "grayscale": True,
        "augmentation_params": {
            "flip_h": True,
            "flip_v": False,
            "rotation_factor": 0.12,
            "zoom_factor": 0.10,
            "translation_factor": 0.0,
            "brightness_delta": 0.10,
            "contrast_factor": 0.10
        },
        
        "mean": None,
        "std": None,
        "mean_std_path": "./data/chunks/dataset_chunked_letterbox_meanstd.npz",
        
        # === Modèle ===
        "model_name": "sophisticated_cnn_128",
        
        # === Hyperparamètres TPU ===
        "tpu": {
            "micro_batch_size": 128,
            "accum_steps": 4,
            "learning_rate": 6e-3,
            "weight_decay": 5e-5,
            "dropout_rate": 0.15,
        },
        
        # === Hyperparamètres GPU ===
        "gpu": {
            "micro_batch_size": 128,
            "accum_steps": 4,
            "learning_rate": 6e-4,
            "weight_decay": 5e-5,
            "dropout_rate": 0.15,
        },
        
        # === Entraînement ===
        "epochs": 30,
        "patience": 5,
        "label_smoothing": 0.1,
        "mixup_alpha": 0.0,
        
        # === Évaluation ===
        "eval_batch_size": 128,
        "eval_use_subset": True,
        "eval_max_subset": 3000,
        
        # === Sauvegarde ===
        "checkpoint_path": "best_model_letterbox.pkl",
        "confusion_matrix_path": "confusion_matrix_letterbox.png",
    },
    
    "FIGHTERJET_DETECTION": {
        # === CONFIG DETECTION D'AVIONS ===
        "task_type": "detection",
        
        # === Données ===
        "num_classes": 1,  # Single Class Object Detection
        "class_names": ['aircraft'],
        "output_prefix": "./data/chunks/detection/dataset_detection",
        "image_size": (224, 224),
        "grayscale": True,
        
        # === Augmentation de Données ===
        "augmentation_params": {
            "flip_h": True,
            "flip_v": True,
            "rotation_factor": 0.0,
            "zoom_factor": 0.25,          # ±25% (était 20%) : Pour apprendre le multi-échelles
            "translation_factor": 0.15,   # Shifting ±15% (était 10%) : Désaxer les cibles
            "brightness_delta": 0.30,     # Beaucoup plus agressif (casser la dominante ciel gris clair)
            "contrast_factor": 0.30       # Très agressif (simule contre-jour et nuages sombres)
        },
        
        # === Modèle ---
        "model_name": "aircraft_detector_v7_advanced",
        "grid_size": 14,      # 224 / 16 = 14
        
        # === Hyperparamètres TPU ===
        "tpu": {
            "micro_batch_size": 32,    # Batch ramené à 32 pour plus de stochasticité (YOLO)
            "accum_steps": 2,          # Accumulation pour simuler 64
            "learning_rate": 4e-4,     # LR légèrement remonté
            "weight_decay": 5e-5,
            "dropout_rate": 0.05,
        },
        
        # === Hyperparamètres GPU ===
        "gpu": {
            "micro_batch_size": 16,
            "accum_steps": 2,
            "learning_rate": 2e-4,
            "weight_decay": 5e-5,
            "dropout_rate": 0.05,
        },
        
        # === Entraînement ===
        "epochs": 30,
        "patience": 5,
        "warmup_steps": 2000,          # Préchauffage lent sur ~4 epochs
        "decay_steps": 10000,          # 🔥 CORRIGÉ : Le LR chutera maintenant vers zéro autour de l'epoch 25 (au lieu de 200)
        
        # === Évaluation/Visualization ===
        "eval_batch_size": 16,
        "vis_freq": 5,
        
        # === Sauvegarde ===
        "checkpoint_path": "./checkpoints_detection",
        "save_dir": "./checkpoints_detection",
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
    print_config("FIGHTERJET_CLASSES")
