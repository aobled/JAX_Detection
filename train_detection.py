"""
Script d'entraînement pour la détection d'avions (Single Class Object Detection)
Utilise :
- Modèle : AircraftDetector (Grid-based / YOLO-like)
- Loss : compute_grid_loss
- Données : DetectionDataset (images 224x224, list of boxes)
"""

import os
import time
import pickle  # 💾 Pour sauvegarder le modèle
import jax
import jax.numpy as jnp
import optax
import numpy as np
import gc
from tqdm import tqdm
from tqdm import tqdm
import flax.linen as nn

# Imports locaux
from model_library import AircraftDetector, TrainStateWithBatchStats, get_model # ✅ Added get_model
from data_management import DetectionDataset
from loss_functions import compute_grid_loss, compute_grid_loss_multilevel, compute_v7_loss
from reporting import DetectionReporter

# Configuration par défaut
CONFIG = {
    "model_name": "aircraft_detector_v3", # 🚀 V3 Optimized Nano
    "image_size": (224, 224), # ✅ 224x224 (Plus rapide, suffisant pour V2)
    "grid_size": 7,      # 224 / 32 = 7
    "batch_size": 32,    # Standard
    "learning_rate": 2e-4,  # 🔥 Augmenté pour détection (meilleure convergence)
    "weight_decay": 5e-5,  # 🔥 Réduit pour laisser plus de flexibilité
    "dropout_rate": 0.05,  # 🔥 Réduit pour détection (moins de régularisation)
    "warmup_steps": 2000,  # 🔥 Augmenté pour 224×224 (convergence plus lente)
    "epochs": 30,
    "data_prefix": "./data/chunks/detection", # Output de tools/detection_dataset_tools.py
    "vis_freq": 5,       # Visualiser tous les X epochs
    "save_dir": "./checkpoints_detection",
    "grayscale": True,  # 🎨 Grayscale recommandé: 3× moins de mémoire, même performance (comme en classification)
}

# 🔍 DIAGNOSTIC JAX LORS DU LANCEMENT SUR COLAB
try:
    print(f"DEBUG: JAX Devices: {jax.devices()}")
    print(f"DEBUG: JAX Backend: {jax.default_backend()}")
except:
    print("DEBUG: Could not list JAX devices")

class DetectionTrainer:
    def __init__(self, config):
        self.config = config
        self.image_size = config["image_size"]
        self.grid_size = config["grid_size"]
        
        # 1. Préparation du Modèle
        dropout_rate = config.get("dropout_rate", 0.05)  # 🔥 Dropout configurable
        # 🔥 UTILISER LA FACTORY POUR CHARGER V2 (avant c'était hardcodé V1)
        self.model = get_model(config["model_name"], dropout_rate=dropout_rate)
        
        # 2. Dataset
        self.dataset_manager = DetectionDataset(
            output_prefix=config["data_prefix"],
            image_size=self.image_size,
            batch_size=config["batch_size"],
            grayscale=config.get("grayscale", False)  # 🎨 Support grayscale
        )
        
        # 3. Reporter (Visu)
        self.reporter = DetectionReporter(
            image_size=self.image_size,
            grid_size=self.grid_size
        )
        
        # État initial
        self.state = None
        self.best_val_loss = float('inf') # 📉 Pour tracker le meilleur modèle
        os.makedirs(config["save_dir"], exist_ok=True)

    def create_train_state(self, rng):
        """Initialise les paramètres et l'optimizer"""
        print("🔧 Initialisation du modèle...")
        # 🎨 Adapter le nombre de canaux selon grayscale ou RGB
        num_channels = 1 if self.config.get("grayscale", False) else 3
        dummy_input = jnp.ones((1, *self.image_size, num_channels), jnp.float32)
        
        variables = self.model.init(rng, dummy_input, training=True)
        params = variables["params"]
        batch_stats = variables.get("batch_stats", {})
        
        # 🔥 Scheduler optimisé pour détection
        # Estimation: ~3000 steps/epoch avec batch_size=16
        # Total: ~150,000 steps pour 50 epochs
        # SÉCURITÉ: S'assurer que epochs >= 1 pour le scheduler
        n_epochs_sched = max(1, self.config["epochs"])
        total_steps_estimate = n_epochs_sched * 3000  # Estimation conservatrice
        warmup_steps = self.config.get("warmup_steps", 1000)
        weight_decay = self.config.get("weight_decay", 5e-5) # Restored
        
        # SÉCURITÉ: decay_steps doit être le nombre TOTAL de steps pour optax
        # et doit être supérieur à warmup_steps
        decay_steps = max(total_steps_estimate, warmup_steps + 100)
        
        schedule = optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=self.config["learning_rate"],
            warmup_steps=warmup_steps,
            decay_steps=decay_steps,
            end_value=self.config["learning_rate"] * 0.01  # 🔥 Decay jusqu'à 1% du LR initial
        )
        
        tx = optax.adamw(learning_rate=schedule, weight_decay=weight_decay)
        
        return TrainStateWithBatchStats.create(
            apply_fn=self.model.apply,
            params=params,
            tx=tx,
            batch_stats=batch_stats
        )

    @jax.jit
    def train_step(state, images, target_boxes, rng):
        """Une étape d'apprentissage"""
        def loss_fn(params):
            vars = {'params': params, 'batch_stats': state.batch_stats}
            
            # Forward pass
            pred_grid, new_batch_stats = state.apply_fn(
                vars, images, training=True,
                mutable=['batch_stats'],
                rngs={'dropout': rng}
            )
            
            # Loss calculation router
            if isinstance(pred_grid, (tuple, list)):
                if len(pred_grid) == 3:
                    loss = compute_v7_loss(pred_grid, target_boxes)
                elif len(pred_grid) == 2:
                    loss = compute_grid_loss_multilevel(pred_grid, target_boxes)
                else:
                    raise ValueError(f"Nombre de grilles en sortie non supporté: {len(pred_grid)}")
            else:
                loss = compute_grid_loss(pred_grid, target_boxes)
            
            return loss, (pred_grid, new_batch_stats)
        
        (loss, (pred_grid, new_batch_stats)), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
        state = state.apply_gradients(grads=grads)
        state = state.replace(batch_stats=new_batch_stats['batch_stats'])
        
        return state, loss, pred_grid

    @jax.jit
    def val_step(state, images, target_boxes):
        """Une étape de validation (JIT)"""
        vars = {'params': state.params, 'batch_stats': state.batch_stats}
        pred_grid = state.apply_fn(vars, images, training=False)
        
        # Loss calculation router
        if isinstance(pred_grid, (tuple, list)):
            if len(pred_grid) == 3:
                loss = compute_v7_loss(pred_grid, target_boxes)
            elif len(pred_grid) == 2:
                loss = compute_grid_loss_multilevel(pred_grid, target_boxes)
            else:
                raise ValueError(f"Nombre de grilles en sortie non supporté: {len(pred_grid)}")
        else:
            loss = compute_grid_loss(pred_grid, target_boxes)
            
        return loss

    def run(self):
        """Boucle principale"""
        rng = jax.random.PRNGKey(0)
        rng, init_rng = jax.random.split(rng)
        
        # Créer dataset TensorFlow
        train_ds = self.dataset_manager.create_tf_dataset('train', augment=True)
        val_ds = self.dataset_manager.create_tf_dataset('val', augment=False)
        
        if train_ds is None:
            print("❌ Erreur: Pas de données d'entraînement.")
            print("   Vérifiez que vous avez bien généré les données avec tools/fighterjet_detection_dataset_tools.py")
            return

        # 🚨 Si epochs=0, on ne fait que vérifier que les chunks sont OK
        if self.config["epochs"] == 0:
            print("✅ Mode vérification: epochs=0, vérification des chunks uniquement")
            print(f"   Train chunks: {len(self.dataset_manager.train_chunks)}")
            print(f"   Val chunks: {len(self.dataset_manager.val_chunks)}")
            # Tester un batch pour vérifier que tout fonctionne
            if train_ds:
                sample_batch = next(iter(train_ds.as_numpy_iterator()))
                print(f"   ✅ Test batch OK: images shape={sample_batch[0].shape}, boxes shape={sample_batch[1].shape}")
            return
        
        # 🔧 Initialisation du state SEULEMENT si on entraîne
        self.state = self.create_train_state(init_rng)
        print(f"🚀 Démarrage de l'entraînement sur {self.config['epochs']} epochs.")
        
        for epoch in range(self.config["epochs"]):
            # --- TRAINING ---
            start_time = time.time()
            epoch_loss = 0.0
            count = 0
            
            pbar = tqdm(train_ds.as_numpy_iterator(), desc=f"Epoch {epoch+1} [Train]")
            for batch_img, batch_boxes in pbar:
                batch_img = jnp.array(batch_img)
                batch_boxes = jnp.array(batch_boxes)
                
                rng, step_rng = jax.random.split(rng)
                self.state, loss, _ = DetectionTrainer.train_step(
                    self.state, batch_img, batch_boxes, step_rng
                )
                
                epoch_loss += float(loss)
                count += 1
                pbar.set_postfix(loss=f"{float(loss):.4f}")
            
            train_loss = epoch_loss / count if count > 0 else 0
            
            # --- VALIDATION ---
            val_loss = 0.0
            val_count = 0
            
            if val_ds:
                for batch_img, batch_boxes in val_ds.as_numpy_iterator():
                    batch_img = jnp.array(batch_img)
                    batch_boxes = jnp.array(batch_boxes)
                    
                    # 🚀 Utilisation de la version JIT
                    v_loss = DetectionTrainer.val_step(self.state, batch_img, batch_boxes)
                    
                    val_loss += float(v_loss)
                    val_count += 1
            
            avg_val_loss = val_loss / val_count if val_count > 0 else 0
            duration = time.time() - start_time
            
            print(f"Stats: Train Loss={train_loss:.4f} | Val Loss={avg_val_loss:.4f} | Time={duration:.1f}s")
            
            # --- VISUALIZATION (Validation visuelle) ---
            if (epoch + 1) % self.config["vis_freq"] == 0 and val_ds:
                print("🎨 Génération des visualisations de contrôle (sur validation)...")
                # Prendre un batch de validation pour visualiser
                for vis_imgs, vis_boxes in val_ds.take(1).as_numpy_iterator():
                    vis_imgs = jnp.array(vis_imgs)
                    vars = {'params': self.state.params, 'batch_stats': self.state.batch_stats}
                    pred_grid = self.state.apply_fn(vars, vis_imgs, training=False)
                    
                    self.reporter.visualize_batch(
                        images=np.array(vis_imgs),
                        predictions=np.array(pred_grid),
                        targets=np.array(vis_boxes),
                        save_path=f"{self.config['save_dir']}/vis_epoch_{epoch+1:02d}.png",
                        conf_threshold=0.5
                    )
                    break
            
            # --- SAVE BEST MODEL ---
            if val_ds and avg_val_loss < self.best_val_loss:
                print(f"📉 Validation Loss améliorée ({self.best_val_loss:.4f} -> {avg_val_loss:.4f})")
                self.best_val_loss = avg_val_loss
                
                save_path = os.path.join(self.config['save_dir'], "best_model_detection.pkl")
                
                # Sauvegarder les params et la config
                # JAX Array -> Numpy pour portabilité facile et pickle
                params_cpu = jax.device_get(self.state.params)
                batch_stats_cpu = jax.device_get(self.state.batch_stats)
                
                with open(save_path, "wb") as f:
                    pickle.dump({
                        'params': params_cpu,
                        'batch_stats': batch_stats_cpu,
                        'config': self.config,
                        'epoch': epoch + 1,
                        'val_loss': avg_val_loss
                    }, f)
                print(f"💾 Nouveau meilleur modèle sauvegardé : {save_path}")
                # 🧹 Libération mémoire immédiate
                del params_cpu
                gc.collect()

            # --- SAVE CHECKPOINT ---
            # Sauvegarde périodique (backup)
            if (epoch + 1) % 10 == 0:
                save_path = f"{self.config['save_dir']}/{self.config['model_name']}_epoch{epoch+1}"
                # Pour l'instant on ne save pas vraiment, juste placeholder
                # On pourrait utiliser orbax ou pickle simple
                print(f"💾 Checkpoint (placeholder) : {save_path}")
            
            # 🧹 Nettoyage RAM fin d'epoch pour éviter OOM sur Colab
            gc.collect()

if __name__ == "__main__":
    # Vérification des pré-requis
    if not os.path.exists("./data/chunks/detection"):
        print("⚠️ AVERTISSEMENT: Le dossier ./data/chunks/detection n'existe pas.")
        print("   Veuillez exécuter tools/fighterjet_detection_dataset_tools.py d'abord.")
    
    trainer = DetectionTrainer(CONFIG)
    trainer.run()
