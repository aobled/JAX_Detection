"""
Version POO de l'entraînement
Architecture orientée objet pour meilleure organisation et maintenance
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("MPLBACKEND", "Agg")

import jax
import jax.numpy as jnp
import gc
import psutil
from tqdm import tqdm

# Import des modules
from model_library import get_model, MODELS, resolve_compute_dtype, build_kwargs_from_config, MODEL_FORWARDED_CONFIG_KEYS
from dataset_configs import get_dataset_config, print_config as print_dataset_config
from trainer import Trainer
from task_strategies import STRATEGIES, STRATEGY_FORWARDED_CONFIG_KEYS


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

# compute_dtype (spec-compute-dtype-hardware, 2026-08-17, AD-1/AD-2 ; etendu a
# nn.BatchNorm/nn.LayerNorm par l'amendement AD-4, spec-compute-dtype-batchnorm-
# layernorm, 2026-08-17) : dtype de CALCUL des couches matmul lourdes (nn.Conv/
# nn.Dense) ET de normalisation (nn.BatchNorm/nn.LayerNorm - jamais nn.Embed), derive
# UNE SEULE FOIS ici depuis le meme backend deja detecte ci-dessus - bfloat16 sur TPU,
# float32 partout ailleurs
# (GPU inclus - prudence de projet documentee, pas une limite materielle, voir
# SPEC.md Constraints). Distinct de `dtype` ci-dessus (cast de l'ENTREE uniquement,
# mecanisme separe, jamais touche ici). Deja un jnp.dtype resolu (pas une string) -
# aucune config de dataset ne doit plus jamais declarer "compute_dtype" (AD-1).
# Logique extraite dans model_library.resolve_compute_dtype() (testable en isolation,
# voir tests/test_compute_dtype_hardware.py) - main.py ne fait plus que l'appeler.
compute_dtype = resolve_compute_dtype(backend)
# Print de validation (2026-08-17, suite a une question d'Aymeric) : la seule chose
# qu'un vrai run peut apprendre que les tests locaux ne couvrent pas deja, c'est si
# jax.default_backend() detecte reellement "tpu" dans CE runtime (Colab) et resout
# bien bfloat16 - la propagation reelle du dtype dans les couches est deja prouvee en
# local (tests/test_compute_dtype_hardware.py, independante du materiel). Distinct du
# print "Backend JAX" ci-dessous (celui-ci existait deja, ne mentionne pas compute_dtype).
print(f"🔢 compute_dtype (calcul Conv/Dense/BatchNorm/LayerNorm): {compute_dtype.__name__}")

print("Backend JAX:", backend)
print("Devices:", jax.devices())
device = jax.devices()[0]
print("Using device:", device)


# ======================
# Fonction principale
# ======================

def main(dataset_name="FIGHTERJET_CLASSIFICATION"):
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
    # class_names optionnel (2026-07-27, Story 9.3) : les configs échecs n'en ont pas
    # (espace de 4672 coups, pas de noms de classe) - consommateurs réels (Trainer,
    # generate_reports des strategies de classification) le lisent déjà via
    # .get()/directement sur config, cette variable locale n'est utilisée nulle part
    # ailleurs dans main.py.
    class_names = config.get("class_names")
    
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
    def _shape_repr(x):
        # targets est un tenseur unique (classification/detection/kepler) ou un dict
        # {HEATMAP_KEY, SIZE_KEY} (detection_centernet, Story 7.5) - generique, pas
        # specifique a un task_type (meme classe de correctif que trainer.py, Story 7.6).
        if isinstance(x, dict):
            return {k: v.shape for k, v in x.items()}
        return x.shape

    print("\n🔍 Vérification des datasets...")
    sample_train = next(iter(train_ds.as_numpy_iterator()))
    if val_ds:
        sample_val = next(iter(val_ds.as_numpy_iterator()))
        print(f"📊 Train: shape={_shape_repr(sample_train[0])}, targets={_shape_repr(sample_train[1])}")
        print(f"📊 Val: shape={_shape_repr(sample_val[0])}, targets={_shape_repr(sample_val[1])}")
    
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
    
    # Construction des kwargs modele via l'introspection centralisee (Story 12.1,
    # AD-21) : MODEL_FORWARDED_CONFIG_KEYS.get(model_name, ()) (canal config,
    # forwarding inconditionnel scope par modele - model_library.py, source unique
    # partagee avec les tests, remplace les anciennes branches if "X" in config) +
    # compute_dtype/dropout_rate/num_classes (canal overrides, forwarding strict par
    # introspection de signature - AD-3 herite, compute_dtype jamais lu depuis config
    # - AD-1 herite). Lookup via .get() (pas MODELS[model_name]) : un model_name
    # invalide/mal orthographie ne doit pas planter ici avec un KeyError brut - on
    # laisse get_model() ci-dessous lever son ValueError explicite habituel (liste
    # les modeles disponibles).
    target_factory = MODELS.get(model_name)
    model_kwargs, forwarded_overrides = build_kwargs_from_config(
        target_factory,
        config,
        config_keys=MODEL_FORWARDED_CONFIG_KEYS.get(model_name, ()),
        compute_dtype=compute_dtype,
        dropout_rate=dropout_rate,
        num_classes=num_classes,
    )
    compute_dtype_injected = "compute_dtype" in forwarded_overrides
    print(f"🔢 compute_dtype pour '{model_name}': {'injecte (' + compute_dtype.__name__ + ')' if compute_dtype_injected else 'non applicable (modele non adapte)'}")
    model = get_model(model_name, **model_kwargs)
    
    # 4. INSTANCIATION DE LA STRATEGIE (Injection de dépendance) - dispatch via
    # STRATEGIES (task_strategies.py, Story 12.2, AD-21/AD-17) et construction des
    # kwargs via le meme helper que model_kwargs (build_kwargs_from_config, Story
    # 12.1) : num_classes en overrides strict (ClassificationStrategy/KeplerStrategy
    # uniquement), le reste via STRATEGY_FORWARDED_CONFIG_KEYS scope par task_type -
    # aucune des 9 classes Strategy n'a de **kwargs, un forwarding non scope
    # planterait immediatement (meme piege que model_kwargs, Story 12.1).
    task_type = config.get("task_type", "classification")
    strategy_cls = STRATEGIES.get(task_type)
    if strategy_cls is None:
        raise ValueError(f"task_type '{task_type}' non reconnu.")
    strategy_kwargs, _ = build_kwargs_from_config(
        strategy_cls,
        config,
        config_keys=STRATEGY_FORWARDED_CONFIG_KEYS.get(task_type, ()),
        num_classes=num_classes,
    )
    print(f"🎯 Strategy: {task_type} -> {strategy_cls.__name__}")
    strategy = strategy_cls(**strategy_kwargs)

    # 5. INITIALISATION DU TRAINER
    print("\n🎯 CRÉATION DU TRAINER")
    print("=" * 60)
    # dtype=jnp.int32 pour chess_move_token (AD-29, spine
    # architecture-chess-move-token-2026-08-10) - trainer.py caste INCONDITIONNELLEMENT
    # toute entree en `dtype` avant tout hook Strategy (trainer.py:313/430). Le
    # `float16` par defaut (ligne 35-41 ci-dessus) ne represente exactement les entiers
    # que jusqu'a ~2048 - les tokens de ce domaine (espace 0-4673, y compris BOS/PAD)
    # seraient silencieusement corrompus sinon. Branche locale a main.py uniquement -
    # trainer.py n'est pas modifie (zero-touch preserve, meme discipline qu'ailleurs).
    # chess_token (spec-chess-token-candidate-model, 2026-08-13) : meme raisonnement
    # que chess_move_token ci-dessus (AD-29 herite) - packed_features encode des
    # indices entiers (token_position/global_flags/candidate_moves, jusqu'a 4671) que
    # le cast float16 par defaut corromprait silencieusement au-dela de ~2048.
    trainer_dtype = jnp.int32 if task_type in ("chess_move_token", "chess_token") else dtype
    trainer = Trainer(
        model=model,
        config=config,
        backend=backend,
        strategy=strategy,
        dtype=trainer_dtype
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

    # Libere immediatement le runtime Colab en fin de script (2026-07-26) - sans ca,
    # la VM/GPU-TPU reste allouee (et facturee sur Pro/Pay-as-you-go) jusqu'au
    # timeout d'inactivite (~90min) ou la limite de session (12-24h), meme si le
    # training est termine et que personne ne regarde. exit(0)/sys.exit ne suffit
    # pas : ca tue le kernel, pas la VM (Colab en redemarre un automatiquement).
    # Sans effet en local (ImportError attendu, google.colab n'existe que sur Colab).
    try:
        from google.colab import runtime as colab_runtime
        print("\n🔌 Fin de script sur Colab : déconnexion du runtime (runtime.unassign)...")
        colab_runtime.unassign()
    except ImportError:
        pass

