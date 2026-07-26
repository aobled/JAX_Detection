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
    
    def save(self, state, best_val_metric: float, patience_counter: int, 
             epoch: int, rng, model_name: str, num_classes: int):
        """
        Sauvegarde un checkpoint
        
        Args:
            state: État JAX d'entraînement
            best_val_metric: Meilleur score de validation (Accuracy ou Loss)
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
                'best_val_metric': best_val_metric,
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
        except (FileNotFoundError, pickle.UnpicklingError, EOFError) as e:
            print(f"⚠️  Aucun checkpoint valide trouvé: {self.checkpoint_path} ({e})")
            return None
    
    def resume_training(self, checkpoint, model, schedule, weight_decay: float):
        """
        Reprend l'entraînement depuis un checkpoint

        Args:
            checkpoint: Dictionnaire du checkpoint
            model: Modèle JAX
            schedule: Schedule de learning rate (le même objet que celui calculé
                dans Trainer.create_train_state - corrige 2026-07-26 un schedule
                distinct code en dur ici (warmup=500/decay=5000) qui pilotait
                silencieusement l'optimiseur restauré, desynchronise du LR affiche
                dans les logs (self.schedule cote Trainer)
            weight_decay: Weight decay

        Returns:
            tuple: (state, best_val_metric, patience_counter, epoch, rng) ou (None, None, None, None, None)
        """
        if checkpoint is None:
            return None, None, None, None, None

        # Recréer l'état d'entraînement
        rng = checkpoint['training_state']['rng']
        rng, init_rng = jax.random.split(rng)

        # Recréer l'optimiseur avec le même schedule que celui utilisé avant l'interruption
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
        state = state.replace(opt_state=checkpoint['model_state'].get('opt_state'))
        
        # Restaurer les variables d'entraînement
        # Rétrocompatibilité : Les anciens checkpoints sauvaient "best_val_acc" 
        # (même si c'était techniquement une loss négative en détection).
        best_val_metric = checkpoint['training_state'].get('best_val_metric', checkpoint['training_state'].get('best_val_acc', float('-inf')))
        patience_counter = checkpoint['training_state']['patience_counter']
        epoch = checkpoint['training_state']['epoch']
        
        print(f"🏁 Reprise d'entraînement à l'epoch {epoch}")
        print(f"   Meilleure métrique (Acc/Loss): {best_val_metric:.4f}")
        print(f"   Patience utilisée: {patience_counter}")
        
        return state, best_val_metric, patience_counter, epoch, rng
    
    def exists(self) -> bool:
        """Vérifie si un checkpoint existe"""
        return os.path.exists(self.checkpoint_path)
    
    def delete(self):
        """Supprime le checkpoint existant"""
        if self.exists():
            os.remove(self.checkpoint_path)
            print(f"🗑️  Checkpoint supprimé: {self.checkpoint_path}")

