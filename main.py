"""
Version POO de l'entraînement
Architecture orientée objet pour meilleure organisation et maintenance
"""

import os
# Supprimé: Ne pas désactiver la pré-allocation XLA sur TPU, cela cause une fragmentation mémoire (Crashes silencieux)
# os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
# os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# Sans serveur X11 : OpenCV (plugins Qt embarqués) + matplotlib évite xcb / crash
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("MPLBACKEND", "Agg")

import inspect
import jax
import jax.numpy as jnp
import gc
import psutil
from tqdm import tqdm

# Import des modules
from model_library import get_model, MODELS, resolve_compute_dtype
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

# compute_dtype (spec-compute-dtype-hardware, 2026-08-17, AD-1/AD-2) : dtype de CALCUL
# des couches matmul lourdes (nn.Conv/nn.Dense), derive UNE SEULE FOIS ici depuis le
# meme backend deja detecte ci-dessus - bfloat16 sur TPU, float32 partout ailleurs
# (GPU inclus - prudence de projet documentee, pas une limite materielle, voir
# SPEC.md Constraints). Distinct de `dtype` ci-dessus (cast de l'ENTREE uniquement,
# mecanisme separe, jamais touche ici). Deja un jnp.dtype resolu (pas une string) -
# aucune config de dataset ne doit plus jamais declarer "compute_dtype" (AD-1).
# Logique extraite dans model_library.resolve_compute_dtype() (testable en isolation,
# voir tests/test_compute_dtype_hardware.py) - main.py ne fait plus que l'appeler.
compute_dtype = resolve_compute_dtype(backend)

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
    
    model_kwargs = {"num_classes": num_classes, "dropout_rate": dropout_rate}
    if "heatmap_prior" in config:
        # aircraft_detector_centernet uniquement (Story 7.2 addendum) - les autres factories
        # (sophisticated_cnn_*) n'ont pas de **kwargs de secours et leveraient un TypeError
        # si on leur passait un argument inattendu, d'ou le passage conditionnel
        model_kwargs["heatmap_prior"] = config["heatmap_prior"]
    if "num_bottleneck_tokens" in config:
        # modeles chess_cnn_attention_* uniquement (test K du bottleneck, 2026-07-27) -
        # meme discipline de forwarding conditionnel que heatmap_prior ci-dessus
        model_kwargs["num_bottleneck_tokens"] = config["num_bottleneck_tokens"]
    if "token_dim" in config:
        # modeles chess_cnn_attention_* uniquement (test capacite token_dim, recherche
        # technique 2026-08-06) - meme discipline de forwarding conditionnel que
        # num_bottleneck_tokens ci-dessus
        model_kwargs["token_dim"] = config["token_dim"]
    if "num_layers" in config:
        # chess_move_token_transformer uniquement (Epic 11, spike) - meme discipline
        # de forwarding conditionnel que token_dim/num_bottleneck_tokens ci-dessus
        model_kwargs["num_layers"] = config["num_layers"]
    if "d_model" in config:
        # chess_move_token_transformer uniquement (Epic 11, spike)
        model_kwargs["d_model"] = config["d_model"]
    if "num_heads" in config:
        # chess_move_token_transformer (Epic 11, spike) ET chess_token_candidate_model
        # (spec-chess-token-candidate-model, 2026-08-13, spike) - les deux modeles
        # exposent un champ num_heads, cette branche generique les sert tous les deux.
        # Ne pas la restreindre/supprimer en pensant qu'elle ne sert qu'a
        # chess_move_token_transformer.
        model_kwargs["num_heads"] = config["num_heads"]
    if "num_trunk_layers" in config:
        # chess_token_candidate_model uniquement (spec-chess-token-candidate-model,
        # 2026-08-13, spike) - meme discipline de forwarding conditionnel que
        # num_bottleneck_tokens/token_dim/num_heads ci-dessus. Sans cette branche, la
        # cle "num_trunk_layers" dans CHESS_TOKEN (dataset_configs.py) serait
        # silencieusement ignoree malgre le commentaire de cette config qui la liste
        # comme ajustable.
        model_kwargs["num_trunk_layers"] = config["num_trunk_layers"]
    # compute_dtype (spec-compute-dtype-hardware, 2026-08-17, AD-3) : injection par
    # introspection de signature, PAS un forwarding conditionnel par config comme les
    # branches ci-dessus. `compute_dtype` n'est jamais lu depuis `config` (AD-1) - la
    # seule question est "la factory cible declare-t-elle un parametre NOMME
    # compute_dtype ?" (pas juste **kwargs, qui absorberait silencieusement la valeur
    # sans jamais la transmettre au constructeur - piege deja present sur
    # aircraft_detector_unet/centernet/centernet_lite, voir architecture spine AD-3).
    # Modeles non adaptes (9 sur 12) : rien injecte, comportement inchange. Lookup via
    # .get() (pas MODELS[model_name]) : un model_name invalide/mal orthographie ne doit
    # pas planter ici avec un KeyError brut - on laisse get_model() ci-dessous lever son
    # ValueError explicite habituel (liste les modeles disponibles), comme avant cette
    # feature (revue Edge Case Hunter/Blind Hunter, 2026-08-17).
    target_factory = MODELS.get(model_name)
    if target_factory is not None and "compute_dtype" in inspect.signature(target_factory).parameters:
        model_kwargs["compute_dtype"] = compute_dtype
    model = get_model(model_name, **model_kwargs)
    
    # 4. INSTANCIATION DE LA STRATEGIE (Injection de dépendance)
    task_type = config.get("task_type", "classification")
    loss_method = config.get("loss_method", "cross_entropy")
    loss_params = config.get("loss_params", {})
    metric_method = config.get("metric_method", "accuracy")
    report_method = config.get("report_method", "confusion_matrix")
    
    if task_type == "classification":
        print("🎯 Application de la logique d'entraînement : CLASSIFICATION")
        strategy = ClassificationStrategy(
            num_classes=num_classes,
            label_smoothing=config.get("label_smoothing", 0.0),
            mixup_alpha=config.get("mixup_alpha", 0.0),
            loss_method=loss_method,
            loss_params=loss_params,
            metric_method=metric_method,
            report_method=report_method
        )
    elif task_type == "detection":
        print("🎯 Application de la logique d'entraînement : DETECTION")
        strategy = DetectionStrategy(
            loss_method=loss_method,
            loss_params=loss_params,
            metric_method=metric_method,
            report_method=report_method
        )
    elif task_type == "kepler":
        print("🎯 Application de la logique d'entraînement : KEPLER 1D")
        from task_strategies import KeplerStrategy
        strategy = KeplerStrategy(
            num_classes=num_classes,
            loss_method=loss_method,
            loss_params=loss_params,
            metric_method=metric_method,
            report_method=report_method
        )
    elif task_type == "detection_centernet":
        print("🎯 Application de la logique d'entraînement : DETECTION CENTERNET")
        from task_strategies import CenterNetDetectionStrategy
        # CenterNetDetectionStrategy n'a pas de dispatch interne (une seule methode de
        # perte/metrique, Story 7.6) - signature reelle (loss_params uniquement, plus de
        # metric_threshold depuis l'addendum post-hoc 2026-07-18 : HeatmapActivation est
        # une moyenne continue, pas un seuil dur), pas loss_method/metric_method/
        # report_method comme les 3 branches ci-dessus.
        strategy = CenterNetDetectionStrategy(loss_params=loss_params)
    elif task_type == "chess_policy_value":
        print("🎯 Application de la logique d'entraînement : CHESS POLICY+VALUE")
        from task_strategies import ChessPolicyValueStrategy
        # ChessPolicyValueStrategy n'a pas de dispatch interne (une seule methode de
        # perte/metrique, Story 9.3) - meme pattern que CenterNetDetectionStrategy
        # ci-dessus (loss_params uniquement, pas loss_method/metric_method/report_method).
        strategy = ChessPolicyValueStrategy(loss_params=loss_params)
    elif task_type == "chess_legal_moves":
        print("🎯 Application de la logique d'entraînement : CHESS COUPS LÉGAUX (multi-label)")
        from task_strategies import ChessLegalMovesStrategy
        # Pas de loss_method/metric_method/report_method : meme discipline que
        # ChessPolicyValueStrategy ci-dessus (une seule methode de perte/metrique).
        # metric_threshold : optionnel, expose via config suite au balayage de seuil
        # du 1er run (2026-08-02) - F1=0.60 a seuil 0.3 contre 0.53 au defaut 0.5.
        # loss_params={"pos_weight": ...} : ponderation de la classe positive dans
        # la BCE (2e run, 2026-08-02) - voir compute_chess_legal_moves_loss.
        strategy = ChessLegalMovesStrategy(
            metric_threshold=config.get("metric_threshold", 0.5),
            loss_params=loss_params,
        )
    elif task_type == "chess_move_token":
        print("🎯 Application de la logique d'entraînement : CHESS MOVE-TOKEN (policy-only, spike)")
        from task_strategies import ChessMoveTokenStrategy
        # Pas de loss_method/metric_method/report_method : meme discipline que
        # ChessPolicyValueStrategy/ChessLegalMovesStrategy ci-dessus.
        strategy = ChessMoveTokenStrategy(loss_params=loss_params)
    elif task_type == "chess_token":
        print("🎯 Application de la logique d'entraînement : CHESS TOKEN (scoring candidats, spike)")
        from task_strategies import ChessTokenStrategy
        # Pas de loss_method/metric_method/report_method : meme discipline que
        # ChessMoveTokenStrategy/ChessPolicyValueStrategy ci-dessus.
        strategy = ChessTokenStrategy(loss_params=loss_params)
    elif task_type == "chess_token_1_move":
        print("🎯 Application de la logique d'entraînement : CHESS TOKEN 1-MOVE (tête factorisée, spike)")
        from task_strategies import ChessTokenOneMoveStrategy
        # Pas de loss_method/metric_method/report_method : meme discipline que
        # ChessTokenStrategy/ChessMoveTokenStrategy ci-dessus.
        strategy = ChessTokenOneMoveStrategy(loss_params=loss_params)
    else:
        raise ValueError(f"task_type '{task_type}' non reconnu.")

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

