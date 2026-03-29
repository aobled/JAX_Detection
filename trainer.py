"""
Classe Trainer pour orchestrer l'entraînement
Architecture orientée objet pour meilleure organisation
"""

import time
import gc
import jax
import jax.numpy as jnp
import optax
from tqdm import tqdm
from typing import Tuple, Optional

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

from model_library import TrainStateWithBatchStats
from checkpoint_manager import CheckpointManager
from utils import smooth_labels, mixup_batch, tree_add, tree_div, batch_stats_div, count_parameters, get_model_size_mb
from reporting import TrainingVisualizer, ViTAttentionVisualizer
from loss_functions import compute_grid_loss


class Trainer:
    """
    Classe principale pour l'entraînement des modèles JAX/Flax
    
    Responsabilités:
    - Création de l'état d'entraînement
    - Exécution des epochs
    - Évaluation
    - Early stopping
    - Gestion des checkpoints
    """
    
    def __init__(self, model, config: dict, backend: str, strategy, dtype=jnp.float32):
        """
        Initialise le trainer
        
        Args:
            model: Modèle JAX/Flax
            config: Configuration du dataset (depuis dataset_configs)
            backend: Backend JAX ('tpu' ou 'gpu')
            strategy: Stratégie polymorphique d'exécution des pertes et métriques (TaskStrategy)
            dtype: Type de données (float16 ou float32)
        """
        self.model = model
        self.config = config
        self.backend = backend
        self.strategy = strategy
        self.dtype = dtype
        
        # Extraire les paramètres backend-specific
        backend_config = config[backend]
        self.micro_batch_size = backend_config["micro_batch_size"]
        self.accum_steps = backend_config["accum_steps"]
        self.learning_rate = backend_config["learning_rate"]
        self.weight_decay = backend_config["weight_decay"]
        
        # Paramètres d'entraînement
        self.epochs = config["epochs"]
        # Suppression de task_type (désormais géré par strategy)
        self.model_name = config["model_name"]
        
        # Image configuration
        self.grayscale = config.get("grayscale", False)
        self.num_channels = 1 if self.grayscale else 3
        
        # État d'entraînement
        self.state = None
        self.best_val_acc = float('-inf')
        self.patience_counter = 0
        self.current_epoch = 0
        self.schedule = None  # Pour extraire le LR
        
        # Historique d'entraînement pour visualisation
        self.history = {
            'epochs': [],
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'learning_rate': []
        }
        
        # Batch test fixe pour visualisation attention (si ViT)
        self.test_batch_for_viz = None
        self.class_names = config.get("class_names", [])
        
        # Checkpoint manager
        from checkpoint_manager import CheckpointManager
        ckpt_path = config.get("checkpoint_path", "best_model.pkl")
        if ckpt_path.endswith(".pkl"):
            ckpt_path = ckpt_path.replace(".pkl", "_training_state.pkl")
        self.checkpoint_manager = CheckpointManager(ckpt_path)        
        # Fonctions JIT
        self._train_step = None
        self._eval_step = None
    
    def create_train_state(self, rng):
        """
        Crée l'état d'entraînement initial
        
        Args:
            rng: Clé RNG JAX
        
        Returns:
            TrainStateWithBatchStats
        """
        image_size = self.config["image_size"]
        # 🎨 Dummy input adaptatif : 1 canal (grayscale) ou 3 canaux (RGB)
        dummy_input = jnp.ones((1, image_size[0], image_size[1], self.num_channels), jnp.float32)
        
        # Initialiser le modèle
        variables = self.model.init(rng, dummy_input, training=True)
        params = variables["params"]
        batch_stats = variables.get("batch_stats", {})
        
        # Schedule d'apprentissage dynamique
        # Warmup et decay adaptés depuis le fichier de config selon la taille du dataset
        warmup_steps = self.config.get("warmup_steps", 1200)
        decay_steps = self.config.get("decay_steps", 6000)
        
        self.schedule = optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=self.learning_rate,
            warmup_steps=warmup_steps,
            decay_steps=decay_steps,
            end_value=1e-6
        )
        
        tx = optax.adamw(self.schedule, weight_decay=self.weight_decay)
        
        state = TrainStateWithBatchStats.create(
            apply_fn=self.model.apply,
            params=params,
            tx=tx,
            batch_stats=batch_stats
        )
        
        # Afficher les infos du modèle
        total_params = count_parameters(state.params)
        model_size_mb = get_model_size_mb(state.params)
        print(f"📊 TAILLE DU MODÈLE:")
        print(f"   Paramètres totaux: {total_params:,}")
        print(f"   Taille: {model_size_mb:.1f} MB")
        
        return state
    
    def _create_train_step(self):
        """Crée la fonction de train step JIT avec délégation de stratégie."""
        
        @jax.jit
        def train_step(params, batch_stats, images, targets, rng):
            # Délégation complète du prétraitement (Mixup, Cast, Label Smoothing)
            rng, mix_rng, drop_rng = jax.random.split(rng, 3)
            images, targets, use_onehot_labels = self.strategy.preprocess_batch(
                images, targets, is_training=True, rng=mix_rng
            )
            
            def loss_fn(params):
                vars = {'params': params, 'batch_stats': batch_stats}
                outputs, new_batch_stats = self.state.apply_fn(
                    vars, images, training=True,
                    rngs={'dropout': drop_rng},
                    mutable=['batch_stats']
                )
                
                # Délégation du calcul de la perte
                loss = self.strategy.compute_loss(outputs, targets, use_onehot_labels=use_onehot_labels)
                return loss, (outputs, new_batch_stats)
            
            (loss, (outputs, new_batch_stats)), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
            
            # Délégation du calcul interne de l'Accuracy
            acc = self.strategy.compute_metrics(outputs, targets)
            
            return loss, grads, acc, new_batch_stats
        
        return train_step
    
    def _create_eval_step(self):
        """Crée la fonction d'évaluation JIT avec délégation de stratégie."""

        @jax.jit
        def eval_step(state, images, targets, rng):
            images, targets, _ = self.strategy.preprocess_batch(
                images, targets, is_training=False
            )
            
            vars = {"params": state.params}
            if state.batch_stats is not None and state.batch_stats != {}:
                vars["batch_stats"] = state.batch_stats
            
            rngs = {"dropout": rng}
            outputs, _ = state.apply_fn(vars, images, training=False, mutable=["batch_stats"], rngs=rngs)
            
            loss = self.strategy.compute_loss(outputs, targets, use_onehot_labels=False)
            score = self.strategy.compute_metrics(outputs, targets)
                
            return loss, score
        
        return eval_step
    
    def train_epoch(self, dataset, rng) -> Tuple[float, float]:
        """
        Entraîne une epoch avec accumulation de gradients
        
        Args:
            dataset: Dataset d'entraînement (list si cache, ou tf.data.Dataset)
            rng: Clé RNG
        
        Returns:
            tuple: (train_loss, train_acc, rng)
        """
        if self._train_step is None:
            self._train_step = self._create_train_step()
        
        # Accumulateurs
        grads_accum = None
        batch_stats_accum = None
        micro_count = 0
        
        total_loss = 0.0
        total_acc = 0.0
        nb_updates = 0
        
        # Itérer sur le dataset (cache ou TensorFlow)
        if isinstance(dataset, list):
            batch_iterator = tqdm(dataset, desc="Training")
        else:
            batch_iterator = tqdm(dataset.as_numpy_iterator(), desc="Training")
        
        for batch in batch_iterator:
            images_np, labels_np = batch
            images = jnp.array(images_np, dtype=self.dtype)
            labels = jnp.array(labels_np) # dtype handles by strategy
            
            # Monitoring agressif de la RAM dans la boucle
            if PSUTIL_AVAILABLE and micro_count == 0 and nb_updates % 50 == 0:
                mem = psutil.virtual_memory()
                if mem.percent > 90:
                    print(f"\n⚠️ CRITICAL RAM: {mem.percent}% -> Forcing GC")
                    gc.collect()
            
            # Split RNG
            rng, step_rng = jax.random.split(rng)
            
            # Train step: Acc délégué depuis l'intérieur du JAX JIT
            loss, grads, step_acc, new_batch_stats = self._train_step(
                self.state.params, self.state.batch_stats, images, labels, step_rng
            )
            
            # Accumulation de gradients (utiliser tree_map pour éviter les références qui s'accumulent)
            if grads_accum is None:
                grads_accum = grads
            else:
                grads_accum = jax.tree_util.tree_map(lambda a, b: a + b, grads_accum, grads)
                
            # Forcer libération
            del grads
            
            # Accumulation de batch_stats
            if new_batch_stats is not None and new_batch_stats != {}:
                new_bs = new_batch_stats['batch_stats']
                if batch_stats_accum is None:
                    batch_stats_accum = new_bs
                else:
                    batch_stats_accum = jax.tree_util.tree_map(
                        lambda a, b: a + b, batch_stats_accum, new_bs
                    )
            
            # Métriques
            total_loss += float(loss)
            total_acc += float(step_acc)
            micro_count += 1
            
            # Appliquer quand on atteint accum_steps
            if micro_count == self.accum_steps:
                # Moyenne des gradients
                grads_mean = tree_div(grads_accum, float(self.accum_steps))
                self.state = self.state.apply_gradients(grads=grads_mean)
                
                # Moyenne des batch_stats
                if batch_stats_accum is not None:
                    averaged_bs = batch_stats_div(batch_stats_accum, float(self.accum_steps))
                    self.state = self.state.replace(batch_stats=averaged_bs)
                
                # 🔥 SYNCHRONISATION JAX (CRUCIAL POUR TPU) 🔥
                # Force le TPU à exécuter les opérations en attente.
                # Empêche Python de saturer la VRAM/HBM du TPU avec des milliers de requêtes asynchrones en epoch 2 (quand le cache OS rend la lecture ultra-rapide).
                self.state = jax.block_until_ready(self.state)
                
                # Reset accumulateurs
                grads_accum = None
                batch_stats_accum = None
                micro_count = 0
                nb_updates += 1
        
        # Appliquer les gradients restants
        if micro_count > 0:
            grads_mean = tree_div(grads_accum, float(micro_count))
            self.state = self.state.apply_gradients(grads=grads_mean)
            if batch_stats_accum is not None:
                averaged_bs = batch_stats_div(batch_stats_accum, float(micro_count))
                self.state = self.state.replace(batch_stats=averaged_bs)
            nb_updates += 1
            
            # Synchronisation finale
            self.state = jax.block_until_ready(self.state)
        
        mean_loss = total_loss / ((nb_updates * self.accum_steps) if nb_updates > 0 else 1)
        mean_acc = total_acc / ((nb_updates * self.accum_steps) if nb_updates > 0 else 1)
        
        return mean_loss, mean_acc, rng
    
    def evaluate(self, dataset) -> Tuple[float, float]:
        """
        Évalue le modèle sur un dataset
        
        Args:
            dataset: Dataset de validation (list si cache, ou tf.data.Dataset)
        
        Returns:
            tuple: (val_loss, val_acc)
        """
        if self._eval_step is None:
            self._eval_step = self._create_eval_step()
        
        val_loss = 0.0
        val_acc = 0.0
        num_batches = 0
        
        # Clé RNG fixe pour l'évaluation
        eval_rng = jax.random.PRNGKey(42)
        
        # Itérer sur le dataset (cache ou TensorFlow)
        if isinstance(dataset, list):
            batch_iterator = dataset
        else:
            batch_iterator = dataset.as_numpy_iterator()
        
        for batch in batch_iterator:
            images_np, labels_np = batch
            images = jnp.array(images_np, dtype=self.dtype)
            labels = jnp.array(labels_np) # Preprocessed by strategy
            
            # Évaluation
            loss, acc = self._eval_step(self.state, images, labels, eval_rng)
            val_loss += float(loss)
            val_acc += float(acc)
            num_batches += 1
        
        return val_loss / num_batches, val_acc / num_batches
    
    def train(self, train_dataset, val_dataset, rng, resume_from_checkpoint: bool = True):
        """
        Lance l'entraînement complet
        
        Args:
            train_dataset: Dataset d'entraînement
            val_dataset: Dataset de validation
            rng: Clé RNG initiale
            resume_from_checkpoint: Si True, reprend depuis le checkpoint existant
        
        Returns:
            tuple: (final_state, best_val_acc)
        """
        # Initialiser l'état d'entraînement
        rng, init_rng = jax.random.split(rng)
        self.state = self.create_train_state(init_rng)
        
        # Reprendre depuis checkpoint si demandé
        start_epoch = 0
        if resume_from_checkpoint:
            checkpoint = self.checkpoint_manager.load()
            if checkpoint is not None:
                self.state, self.best_val_acc, self.patience_counter, start_epoch, rng = \
                    self.checkpoint_manager.resume_training(
                        checkpoint, self.model, self.learning_rate, self.weight_decay
                    )
                if self.state is None:
                    print("❌ Impossible de reprendre, démarrage depuis zéro")
                    start_epoch = 0
                    self.best_val_acc = 0.0
                    self.patience_counter = 0
            else:
                print("🚀 Démarrage d'un nouvel entraînement")
        
        # Boucle d'entraînement
        for epoch in range(start_epoch, self.epochs):
            self.current_epoch = epoch
            print(f"\nEpoch {epoch+1}/{self.epochs} - training...")
            
            # Monitoring RAM périodique
            if PSUTIL_AVAILABLE and epoch % 5 == 0:
                memory = psutil.virtual_memory()
                print(f"💾 RAM: {memory.percent:.1f}% ({memory.used/1024**3:.1f}GB/{memory.total/1024**3:.1f}GB)")
            
            # Entraînement
            start_time = time.time()
            train_loss, train_acc, rng = self.train_epoch(train_dataset, rng)
            train_time = time.time() - start_time
            
            print(f"Train | Loss={train_loss:.4f} | Acc={train_acc:.4f} | Time={train_time:.1f}s")
            
            # Évaluation
            val_loss, val_acc = self.evaluate(val_dataset)
            print(f"Val   | Loss={val_loss:.4f} | Acc={val_acc:.4f}")
            
            # Extraire le learning rate actuel depuis le schedule
            current_step = int(self.state.step)
            current_lr = float(self.schedule(current_step))
            
            # Stocker dans l'historique
            self.history['epochs'].append(epoch + 1)
            self.history['train_loss'].append(float(train_loss))
            self.history['train_acc'].append(float(train_acc) * 100)  # En pourcentage
            self.history['val_loss'].append(float(val_loss))
            self.history['val_acc'].append(float(val_acc) * 100)  # En pourcentage
            self.history['learning_rate'].append(current_lr)
            
            # Générer et sauvegarder la visualisation à chaque epoch
            visualizer = TrainingVisualizer(
                history=self.history,
                model_name=self.model_name,
                num_params=count_parameters(self.state.params)
            )
            visualizer.plot_training_curves(epoch_start=0, save_path=f"{self.model_name}.png")
            
            # Visualisation attention maps (tous les 5 epochs pour ViT)
            is_vit = 'vit' in self.model_name.lower()
            if is_vit and (epoch % 5 == 0 or epoch == self.epochs - 1):
                # Stocker un batch test fixe au premier epoch
                if self.test_batch_for_viz is None:
                    try:
                        if isinstance(val_dataset, list):
                            # Dataset en cache
                            if len(val_dataset) > 0:
                                self.test_batch_for_viz = val_dataset[0]  # Premier batch
                        else:
                            # TensorFlow Dataset
                            for batch in val_dataset.take(1).as_numpy_iterator():
                                self.test_batch_for_viz = batch
                                break
                    except:
                        pass
                
                if self.test_batch_for_viz is not None:
                    try:
                        test_images, test_labels = self.test_batch_for_viz
                        test_images = jnp.array(test_images[:4], dtype=self.dtype)  # 4 images
                        test_labels = jnp.array(test_labels[:4], dtype=jnp.int32)
                        
                        attention_viz = ViTAttentionVisualizer(
                            state=self.state,
                            model=self.model,
                            class_names=self.class_names
                        )
                        attention_viz.visualize_attention(
                            test_images=test_images,
                            test_labels=test_labels,
                            epoch=epoch + 1,
                            save_path=f"attention_epoch_{epoch+1:03d}.png",
                            num_samples=8
                        )
                    except Exception as e:
                        print(f"⚠️ Attention visualization skipped: {e}")
            
            # Garbage collection agressif en fin d'epoch
            if PSUTIL_AVAILABLE:
                memory = psutil.virtual_memory()
                if memory.percent > 75:  # Baissé de 85 à 75 pour être plus agressif
                    print("🧹 Nettoyage mémoire de fin d'epoch...")
                    # Nettoyer les caches internes de JAX (optionnel mais utile sur TPU)
                    # jax.clear_backends() # Attention, peut ralentir si utilisé trop souvent
                    gc.collect()
                    memory = psutil.virtual_memory()
                    print(f"💾 RAM après GC: {memory.percent:.1f}%")
            
            # Early stopping
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.patience_counter = 0
                print(f"[✓] New best model saved (val_acc={val_acc:.4f})")
                
                # Sauvegarder le checkpoint Orbax
                self.checkpoint_manager.save(
                    self.state, self.best_val_acc, self.patience_counter,
                    epoch, rng, self.model_name, self.config["num_classes"]
                )
                
                # --- EXPORT PKL ---
                task_type = self.config.get("task_type", "classification")
                if task_type == "classification":
                    pkl_path = self.config.get("checkpoint_path", "best_model.pkl")
                    if "checkpoints" in pkl_path and not pkl_path.endswith('.pkl'):
                        pkl_path = "best_model_classification.pkl"
                else:
                    pkl_path = "best_model_detection.pkl"  # Attendu par le script d'inférence
                
                try:
                    import pickle
                    
                    # Convertir les tenseurs XLA/TPU en Numpy natif CPU pour éviter tous les problèmes de portabilité
                    params_cpu = jax.device_get(self.state.params)
                    batch_stats_cpu = jax.device_get(self.state.batch_stats) if self.state.batch_stats is not None else {}
                    
                    model_dict = {
                        'params': params_cpu,
                        'batch_stats': batch_stats_cpu,
                        'config': self.config 
                    }
                    with open(pkl_path, 'wb') as f:
                        pickle.dump(model_dict, f)
                    print(f"   [💾] Export pur PKL généré: {pkl_path} (12 Mo)")
                    
                    # Libérer la mémoire des copies numpy
                    del params_cpu, batch_stats_cpu
                except Exception as e:
                    print(f"   [⚠️] Erreur d'export PKL: {e}")
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.config.get("patience", 10):
                    print(f"Early stopping at epoch {epoch+1}")
                    break
        
        print(f"\n🎯 ENTRAÎNEMENT TERMINÉ")
        print(f"   Meilleure accuracy validation: {self.best_val_acc:.4f}")
        
        return self.state, self.best_val_acc
    
    def get_state(self):
        """Retourne l'état d'entraînement actuel"""
        return self.state
    
    def get_best_val_acc(self):
        """Retourne la meilleure accuracy de validation"""
        return self.best_val_acc

