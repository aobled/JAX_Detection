"""
Gestion des checkpoints et sauvegarde/chargement des modèles
Classe pour gérer la persistance de l'état d'entraînement
"""

import pickle
import os
from typing import Optional, Dict, Any
import jax
import optax
from flax.training import train_state


class CheckpointManager:
    """
    Gestionnaire de checkpoints pour l'entraînement
    
    Responsable de :
    - Sauvegarde de l'état d'entraînement
    - Chargement des checkpoints
    - Reprise d'entraînement
    """
    
    def __init__(self, checkpoint_path: str = "best_model.pkl"):
        """
        Initialise le gestionnaire de checkpoints
        
        Args:
            checkpoint_path: Chemin du fichier de checkpoint
        """
        self.checkpoint_path = checkpoint_path
        self.checkpoint_dir = os.path.dirname(checkpoint_path)
        
        # Créer le dossier si nécessaire
        if self.checkpoint_dir and not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir, exist_ok=True)
    
    def save(self, state, best_val_acc: float, patience_counter: int, 
             epoch: int, rng, model_name: str, num_classes: int):
        """
        Sauvegarde complète du modèle pour reprise d'entraînement
        
        Args:
            state: État d'entraînement (TrainState)
            best_val_acc: Meilleure accuracy de validation
            patience_counter: Compteur de patience pour early stopping
            epoch: Numéro de l'epoch actuelle
            rng: Clé RNG actuelle
            model_name: Nom du modèle
            num_classes: Nombre de classes
        """
        checkpoint = {
            'model_state': {
                'params': state.params,
                'batch_stats': state.batch_stats,
                'step': state.step,
                'opt_state': state.opt_state
            },
            'training_state': {
                'best_val_acc': best_val_acc,
                'patience_counter': patience_counter,
                'epoch': epoch,
                'rng': rng
            },
            'model_info': {
                'model_name': model_name,
                'num_classes': num_classes
            }
        }
        
        with open(self.checkpoint_path, "wb") as f:
            pickle.dump(checkpoint, f)
        
        # Affichage minimal pour ne pas polluer les logs
        # print(f"💾 Checkpoint sauvegardé: {self.checkpoint_path}")
    
    def load(self) -> Optional[Dict[str, Any]]:
        """
        Charge un checkpoint pour reprendre l'entraînement
        
        Returns:
            Dictionnaire du checkpoint ou None si non trouvé
        """
        try:
            with open(self.checkpoint_path, "rb") as f:
                checkpoint = pickle.load(f)
            print(f"📂 Checkpoint chargé: {self.checkpoint_path}")
            return checkpoint
        except FileNotFoundError:
            print(f"⚠️  Aucun checkpoint trouvé: {self.checkpoint_path}")
            return None
    
    def resume_training(self, checkpoint, model, learning_rate: float, weight_decay: float):
        """
        Reprend l'entraînement depuis un checkpoint
        
        Args:
            checkpoint: Dictionnaire du checkpoint
            model: Modèle JAX
            learning_rate: Learning rate
            weight_decay: Weight decay
        
        Returns:
            tuple: (state, best_val_acc, patience_counter, epoch, rng) ou (None, None, None, None, None)
        """
        if checkpoint is None:
            return None, None, None, None, None
        
        # Recréer l'état d'entraînement
        rng = checkpoint['training_state']['rng']
        rng, init_rng = jax.random.split(rng)
        
        # Recréer l'optimiseur
        schedule = optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=learning_rate,
            warmup_steps=500,
            decay_steps=5000,
            end_value=1e-6
        )
        tx = optax.adamw(schedule, weight_decay=weight_decay)
        
        # Import TrainStateWithBatchStats
        from model_library import TrainStateWithBatchStats
        
        # Restaurer l'état
        state = TrainStateWithBatchStats.create(
            apply_fn=model.apply,
            params=checkpoint['model_state']['params'],
            tx=tx,
            batch_stats=checkpoint['model_state']['batch_stats']
        )
        
        # Restaurer l'état de l'optimiseur
        state = state.replace(opt_state=checkpoint['model_state']['opt_state'])
        
        # Restaurer les variables d'entraînement
        best_val_acc = checkpoint['training_state']['best_val_acc']
        patience_counter = checkpoint['training_state']['patience_counter']
        epoch = checkpoint['training_state']['epoch']
        
        print(f"🔄 Entraînement repris depuis l'epoch {epoch+1}")
        print(f"   Meilleure accuracy: {best_val_acc:.4f}")
        print(f"   Patience counter: {patience_counter}")
        
        return state, best_val_acc, patience_counter, epoch, rng
    
    def exists(self) -> bool:
        """Vérifie si un checkpoint existe"""
        return os.path.exists(self.checkpoint_path)
    
    def delete(self):
        """Supprime le checkpoint existant"""
        if self.exists():
            os.remove(self.checkpoint_path)
            print(f"🗑️  Checkpoint supprimé: {self.checkpoint_path}")

