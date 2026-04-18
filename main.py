"""
Version POO de l'entraînement
Architecture orientée objet pour meilleure organisation et maintenance
"""

import os
# Supprimé: Ne pas désactiver la pré-allocation XLA sur TPU, cela cause une fragmentation mémoire (Crashes silencieux)
# os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
# os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import jax
import jax.numpy as jnp
import gc
import psutil
from tqdm import tqdm

# Import des modules
from model_library import get_model
from dataset_configs import get_dataset_config, print_config as print_dataset_config
from trainer import Trainer
from task_strategies import ClassificationStrategy, DetectionStrategy


# ======================
# Hardware init
# ======================
backend = jax.default_backend()
if backend == "tpu":
    print("🚀 TPU détecté - Optimisations activées")
    jax.config.update("jax_enable_x64", False)
    dtype = jnp.float16
    print("📊 TPU: Utilisation de float16")
else:
    print("🖥️  GPU détecté")
    jax.config.update("jax_platform_name", "gpu")
    dtype = jnp.float16
    print("📊 GPU: Utilisation de float16")

print("Backend JAX:", backend)
print("Devices:", jax.devices())
device = jax.devices()[0]
print("Using device:", device)


# ======================
# Fonction principale
# ======================

def main(dataset_name="FIGHTERJET_9CLASSES"):
    """
    Fonction principale d'entraînement - Version POO
    
    Args:
        dataset_name: Nom du dataset à utiliser (défini dans dataset_configs.py)
    """
    # 🔧 CHARGER LA CONFIGURATION DU DATASET
    print(f"\n📊 Chargement de la configuration: {dataset_name}")
    config = get_dataset_config(dataset_name)
    print_dataset_config(dataset_name)
    
    # Extraire les paramètres essentiels
    num_classes = config["num_classes"]
    class_names = config["class_names"]
    
    # === 1. GESTION DES DONNÉES ===
    print(f"\n📁 GESTION DES DONNÉES")
    print("=" * 60)
    
    from data_management import get_datasets
    
    # Obtenir les paramètres backend-specific
    backend_config = config[backend]
    micro_batch_size = backend_config["micro_batch_size"]
    
    # === CRÉATION DU PIPELINE UNIFIÉ ===
    train_ds, val_ds = get_datasets(config, backend_config)
    
    # Vérification des datasets
    print("\n🔍 Vérification des datasets...")
    sample_train = next(iter(train_ds.as_numpy_iterator()))
    if val_ds:
        sample_val = next(iter(val_ds.as_numpy_iterator()))
        print(f"📊 Train: shape={sample_train[0].shape}, targets={sample_train[1].shape}")
        print(f"📊 Val: shape={sample_val[0].shape}, targets={sample_val[1].shape}")
    
    train_dataset_final = train_ds
    val_dataset_final = val_ds
    
    # === 2. CRÉATION DU MODÈLE ===
    print(f"\n🏗️  CRÉATION DU MODÈLE")
    print("=" * 60)
    
    model_name = config["model_name"]
    dropout_rate = backend_config["dropout_rate"]
    
    print(f"Modèle: {model_name}")
    print(f"Classes: {num_classes}")
    print(f"Dropout: {dropout_rate}")
    
    model = get_model(model_name, num_classes=num_classes, dropout_rate=dropout_rate)
    
    # 4. INSTANCIATION DE LA STRATEGIE (Injection de dépendance)
    task_type = config.get("task_type", "classification")
    if task_type == "classification":
        print("🎯 Application de la logique d'entraînement : CLASSIFICATION")
        strategy = ClassificationStrategy(
            num_classes=num_classes,
            label_smoothing=config.get("label_smoothing", 0.0),
            mixup_alpha=config.get("mixup_alpha", 0.0)
        )
    else:
        print("🎯 Application de la logique d'entraînement : DETECTION")
        strategy = DetectionStrategy()

    # 5. INITIALISATION DU TRAINER
    print("\n🎯 CRÉATION DU TRAINER")
    print("=" * 60)
    trainer = Trainer(
        model=model,
        config=config,
        backend=backend,
        strategy=strategy,
        dtype=dtype
    )
    
    # === 4. ENTRAÎNEMENT ===
    print(f"\n🚀 LANCEMENT DE L'ENTRAÎNEMENT")
    print("=" * 60)
    
    rng = jax.random.PRNGKey(42)
    
    # Monitoring RAM avant entraînement
    memory = psutil.virtual_memory()
    print(f"💾 RAM avant entraînement: {memory.percent:.1f}%")
    
    final_state, best_val_metric = trainer.train(
        train_dataset=train_dataset_final,
        val_dataset=val_dataset_final,
        rng=rng,
        resume_from_checkpoint=config.get("resume_training", True)
    )
    
    # Garbage collection si RAM élevée
    memory = psutil.virtual_memory()
    if memory.percent > 85:
        print("🧹 RAM élevée, garbage collection...")
        gc.collect()
        memory = psutil.virtual_memory()
        print(f"💾 RAM après GC: {memory.percent:.1f}%")
    
    # === 5. GÉNÉRATION DES MÉTRIQUES (Confusion/Detection) ===
    print(f"\n📊 GÉNÉRATION DES MÉTRIQUES DÉLÉGUÉE À LA STRATÉGIE")
    print("=" * 60)
    
    strategy.generate_reports(val_ds, final_state, model, config)
    
    print(f"\n🏁 Programme terminé")
    print(f"   Meilleur score validation (Accuracy ou Loss): {best_val_metric:.4f}")


if __name__ == "__main__":
    import sys
    
    # Permettre de spécifier le dataset en ligne de commande
    # Usage: python main_poo.py [DATASET_NAME]
    # Exemple: python main_poo.py FIGHTERJET_9CLASSES
    if len(sys.argv) > 1:
        dataset_name = sys.argv[1]
        print(f"🎯 Dataset spécifié: {dataset_name}")
    else:
        dataset_name = "FIGHTERJET_CLASSIFICATION"  # Défaut
        print(f"🎯 Dataset par défaut: {dataset_name}")
    
    main(dataset_name)

