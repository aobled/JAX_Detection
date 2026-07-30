"""
Classe Trainer pour orchestrer l'entraînement
Architecture orientée objet pour meilleure organisation
"""

import time
import gc
import glob
import math
import jax
import jax.numpy as jnp
import numpy as np
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
from reporting import TrainingVisualizer
from loss_functions import compute_grid_loss


def _count_real_train_samples(output_prefix):
    """Compte le nombre reel d'echantillons dans les chunks .npz d'entrainement
    (glob {output_prefix}_train_chunk*.npz). Utilise la premiere cle du npz (peu
    importe son nom - 'image'/'images'/'heatmap' - le premier axe est toujours N
    quel que soit le schema de chunk, classification/detection/detection_centernet/
    kepler). Retourne None si aucun chunk trouve (config sans donnees reelles sur
    disque, ex. contexte de test) - l'appelant doit alors se rabattre sur une valeur
    par defaut, jamais planter ici.

    Ajoute 2026-07-21 suite a 3 incidents reels de decay_steps code en dur devenu
    stale (CIFAR10 Epic 5 Addendum 3, JAX_DETECTOR rv7 apres activation de
    zoom_augment_probability, JAX_DETECTOR v8 apres changement de micro_batch_size)
    - voir deferred-work.md. Remplace le recalcul manuel par une mesure automatique
    a chaque construction du Trainer.
    """
    chunk_files = sorted(glob.glob(f"{output_prefix}_train_chunk*.npz"))
    if not chunk_files:
        return None
    total = 0
    for path in chunk_files:
        with np.load(path) as data:
            if not data.files:
                continue
            total += data[data.files[0]].shape[0]
    return total if total > 0 else None


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
        self.backend_config = backend_config  # conserve pour create_train_state (warmup/decay_steps)
        self.micro_batch_size = backend_config["micro_batch_size"]
        self.accum_steps = backend_config["accum_steps"]
        self.learning_rate = backend_config["learning_rate"]
        self.weight_decay = backend_config["weight_decay"]
        
        # Paramètres d'entraînement
        self.epochs = config["epochs"]
        # Suppression de task_type (désormais géré par strategy)
        self.model_name = config["model_name"]
        
        # État d'entraînement
        self.state = None
        self.best_val_metric = float('-inf') if self.strategy.optimization_mode == "max" else float('inf')
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
        
        self.class_names = config.get("class_names", [])
        
        # Checkpoint manager
        ckpt_path = self.strategy.get_training_state_path(self.config)
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
        # "input_shape" (2026-07-30, remplace l'ancien couple image_size+num_channels/
        # grayscale + branche if/elif par task_type) : forme complète de l'entrée modele,
        # hors batch - (H, W, C) pour une image, (longueur, canaux) pour une sequence 1D
        # (Kepler), (8, 8, NUM_PLANES) pour les echecs, etc. Source unique et 100%
        # generique - Trainer n'a plus besoin de connaitre la famille de forme de chaque
        # domaine (aucune branche if/elif ici, contrairement a avant). image_size/
        # grayscale restent utilises ailleurs (data_management.py, dataset_builder/) pour
        # le retraitement d'image reel (resize) - role distinct, jamais mélangé avec
        # celui-ci desormais.
        input_shape = self.config["input_shape"]
        dummy_input = jnp.ones((1,) + tuple(input_shape), jnp.float32)
        
        # Initialiser le modèle
        variables = self.model.init(rng, dummy_input, training=True)
        params = variables["params"]
        batch_stats = variables.get("batch_stats", {})
        
        # Schedule d'apprentissage dynamique
        # Warmup et decay adaptés depuis le fichier de config selon la taille du dataset.
        # Nichés sous tpu/gpu (pas top-level) : ce sont des comptes de STEPS, qui dépendent
        # du steps/epoch (donc du micro_batch_size, backend-specific) - contrairement à
        # epochs/patience (comptes d'EPOCHS, backend-independants, restent top-level).
        # (2026-07-18, migration - voir dataset_configs.py pour la justification complète)
        warmup_steps = self.backend_config.get("warmup_steps", 1200)

        # decay_steps calcule automatiquement depuis le vrai volume de donnees sur
        # disque (2026-07-21) plutot que lu tel quel depuis la config - elimine la
        # classe de bug "steps/epoch mesure une fois a la main, jamais remis a jour
        # quand le dataset ou micro_batch_size change" (3 incidents reels, voir
        # deferred-work.md). Repli sur la config si aucun chunk trouve (ex. tests).
        # decay_epochs (optionnel) decouple la duree du decay du budget total
        # d'epochs - certaines configs veulent une longue queue de fine-tuning a
        # bas LR (decay court) plutot qu'un decay etale sur tout l'entrainement.
        # Repli sur self.epochs si absent (comportement inchange pour les configs
        # existantes). Ajoute 2026-07-26 suite a la comparaison Plus/Lite sur
        # FIGHTERJET_CLASSIFICATION (voir deferred-work.md).
        decay_epochs = self.config.get("decay_epochs", self.epochs)

        real_train_samples = _count_real_train_samples(self.config.get("output_prefix", ""))
        if real_train_samples:
            real_steps_per_epoch = real_train_samples // self.micro_batch_size
            decay_steps = real_steps_per_epoch * decay_epochs
            print(
                f"📐 decay_steps calculé automatiquement : {real_train_samples} échantillons réels, "
                f"batch={self.micro_batch_size} -> {real_steps_per_epoch} steps/epoch × {decay_epochs} decay_epochs = {decay_steps}"
            )
        else:
            decay_steps = self.backend_config.get("decay_steps", 6000)
            print(f"⚠️  decay_steps automatique indisponible (aucun chunk .npz trouvé), repli sur la config : {decay_steps}")

        lr_schedule_type = self.config.get("lr_schedule", "cosine")
        if lr_schedule_type == "cosine":
            self.schedule = optax.warmup_cosine_decay_schedule(
                init_value=0.0,
                peak_value=self.learning_rate,
                warmup_steps=warmup_steps,
                decay_steps=decay_steps,
                end_value=1e-6
            )
        elif lr_schedule_type == "constant":
            self.schedule = optax.constant_schedule(self.learning_rate)
        else:
            raise ValueError(f"Schedule '{lr_schedule_type}' non supporté.")
        
        optimizer_type = self.config.get("optimizer", "adamw")
        if optimizer_type == "adamw":
            tx = optax.adamw(self.schedule, weight_decay=self.weight_decay)
        elif optimizer_type == "sgd":
            # Le SGD standard utilise un momentum optionnel, on reste basique ici
            tx = optax.sgd(self.schedule)
        elif optimizer_type == "adam":
            tx = optax.adam(self.schedule)
        else:
            raise ValueError(f"Optimizer '{optimizer_type}' non supporté.")
        
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
        total_microbatches = 0
        
        # Itérer sur le dataset (cache ou TensorFlow)
        if isinstance(dataset, list):
            batch_iterator = tqdm(dataset, desc="Training")
        else:
            batch_iterator = tqdm(dataset.as_numpy_iterator(), desc="Training")
        
        for batch in batch_iterator:
            images_np, labels_np = batch
            images = jnp.array(images_np, dtype=self.dtype)
            labels = jax.tree_util.tree_map(jnp.array, labels_np) # dtype handles by strategy
            
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
            loss_value = float(loss)
            if not math.isfinite(loss_value):
                print(f"\n⚠️ Loss non-finie détectée ({loss_value}) - le run a probablement divergé")
            total_loss += loss_value
            total_acc += float(step_acc)
            micro_count += 1
            total_microbatches += 1
            
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
        
        if total_microbatches == 0:
            print("⚠️ Epoch sans aucun batch traité (dataset d'entraînement vide) - loss/accuracy à 0.0")
            return 0.0, 0.0, rng

        mean_loss = total_loss / total_microbatches
        mean_acc = total_acc / total_microbatches

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
            labels = jax.tree_util.tree_map(jnp.array, labels_np) # Preprocessed by strategy
            
            # Évaluation
            loss, acc = self._eval_step(self.state, images, labels, eval_rng)
            val_loss += float(loss)
            val_acc += float(acc)
            num_batches += 1
        
        if num_batches == 0:
            print("⚠️ Évaluation sans aucun batch de validation traité (dataset de validation vide)")
            return 0.0, 0.0

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
                self.state, self.best_val_metric, self.patience_counter, start_epoch, rng = \
                    self.checkpoint_manager.resume_training(
                        checkpoint, self.model, self.schedule, self.weight_decay
                    )
                if self.state is None:
                    print("❌ Impossible de reprendre, démarrage depuis zéro")
                    start_epoch = 0
                    self.best_val_metric = float('-inf') if self.strategy.optimization_mode == "max" else float('inf')
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
            
            print(f"Train | Loss={train_loss:.4f} | {self.strategy.primary_metric_name}={train_acc:.4f} | Time={train_time:.1f}s")
            
            # Évaluation
            start_time = time.time()
            val_loss, val_acc = self.evaluate(val_dataset)
            val_time = time.time() - start_time
            print(f"Val   | Loss={val_loss:.4f} | {self.strategy.primary_metric_name}={val_acc:.4f} | Time={val_time:.1f}s")
            
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
            # Dérivé de dataset_name (pas de model_name) : deux configs peuvent réutiliser la même architecture
            # (ex. CIFAR10 et FIGHTERJET_CLASSIFICATION partagent sophisticated_cnn_128_plus) sans s'écraser.
            visualizer.plot_training_curves(epoch_start=0, save_path=f"{self.config.get('dataset_name', 'unknown').lower()}.png")
            

            
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
            current_metric = val_acc if self.strategy.optimization_mode == "max" else val_loss
            is_better = (current_metric > self.best_val_metric) if self.strategy.optimization_mode == "max" else (current_metric < self.best_val_metric)
            
            if is_better:
                self.best_val_metric = current_metric
                self.patience_counter = 0
                print(f"[✓] New best model saved ({self.strategy.primary_metric_name}={current_metric:.4f})")
                
                # Sauvegarder le checkpoint Orbax
                self.checkpoint_manager.save(
                    self.state, self.best_val_metric, self.patience_counter,
                    epoch, rng, self.model_name, self.config["num_classes"]
                )
                
                # --- EXPORT PKL ---
                self.strategy.export_model(self.state, self.config)
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.config.get("patience", 10):
                    print(f"Early stopping at epoch {epoch+1}")
                    break
        
        print(f"\n🎯 ENTRAÎNEMENT TERMINÉ")
        print(f"   Meilleur {self.strategy.primary_metric_name} validation: {self.best_val_metric:.4f}")
        
        return self.state, self.best_val_metric
    
    def get_state(self):
        """Retourne l'état d'entraînement actuel"""
        return self.state
    
    def get_best_val_metric(self):
        """Retourne la meilleure métrique de validation"""
        return self.best_val_metric

