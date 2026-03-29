import jax
import jax.numpy as jnp
import optax
from abc import ABC, abstractmethod
from loss_functions import compute_grid_loss
from utils import mixup_batch, smooth_labels

class TaskStrategy(ABC):
    @abstractmethod
    def preprocess_batch(self, images, targets, is_training, rng=None):
        """Prétraite les données (Cast, Mixup, Label Smoothing...) à l'intérieur du JIT."""
        pass
        
    @abstractmethod
    def compute_loss(self, outputs, targets, **kwargs):
        """Calcule la perte du réseau."""
        pass
        
    @abstractmethod
    def compute_metrics(self, outputs, targets):
        """Calcule la ou les métriques d'évaluation."""
        pass
        
    @abstractmethod
    def generate_reports(self, val_ds, final_state, model, config):
        """Génère les rapports post-entraînement (Matrice de confusion, Visualisation Boxes...)."""
        pass

class ClassificationStrategy(TaskStrategy):
    def __init__(self, num_classes: int, label_smoothing: float = 0.0, mixup_alpha: float = 0.0):
        self.num_classes = num_classes
        self.label_smoothing = label_smoothing
        self.mixup_alpha = mixup_alpha

    def preprocess_batch(self, images, targets, is_training, rng=None):
        targets = jnp.array(targets, dtype=jnp.int32)
        use_onehot = False
        
        if not is_training:
            return images, targets, use_onehot
            
        if self.mixup_alpha > 0 and rng is not None:
             images, targets = mixup_batch(images, targets, self.mixup_alpha, self.num_classes, rng)
             use_onehot = True
        elif self.label_smoothing > 0:
             targets = smooth_labels(targets, self.num_classes, self.label_smoothing)
             use_onehot = True
             
        return images, targets, use_onehot
        
    def compute_loss(self, outputs, targets, use_onehot_labels=False, **kwargs):
        if use_onehot_labels:
            return optax.softmax_cross_entropy(outputs, targets).mean()
        else:
            return optax.softmax_cross_entropy_with_integer_labels(outputs, targets).mean()
            
    def compute_metrics(self, outputs, targets):
        # Si targets est one_hot, on doit le convertir pour l'accuracy,
        # mais on calcule les métriques généralement sur les vrais labels int32.
        # En training, targets peut être mixup (onehot floats).
        if len(targets.shape) > 1 and targets.shape[-1] == outputs.shape[-1]:
            # Targets est one-hot
            true_classes = jnp.argmax(targets, axis=-1)
        else:
            true_classes = targets
        return (jnp.argmax(outputs, axis=-1) == true_classes).mean()
        
    def generate_reports(self, val_ds, final_state, model, config):
        from reporting import Reporter as ModelReporter # Utilise la classe Reporter
        reporter = ModelReporter(class_names=config["class_names"])
        try:
            # Déterminer le bon fichier PKL engendré par le Trainer
            pkl_path = config.get("checkpoint_path", "best_model.pkl")
            if "checkpoints" in pkl_path and not pkl_path.endswith('.pkl'):
                pkl_path = "best_model_classification.pkl"
                
            reporter.confusion_matrix_from_pkl(
                dataset=val_ds,
                pkl_path=pkl_path,
                confusion_matrix_png_path=config.get("confusion_matrix_path", "confusion_matrix.png"),
                use_subset=config.get("eval_use_subset", False),
                batch_size=config.get("eval_batch_size", 32),
                max_subset=config.get("eval_max_subset", 1000)
            )
            print(f"✅ Matrice de confusion générée avec succès depuis l'export pur (pkl) !")
        except Exception as e:
            print(f"❌ Erreur metrics: {e}")


class DetectionStrategy(TaskStrategy):
    def preprocess_batch(self, images, targets, is_training, rng=None):
        targets = jnp.array(targets, dtype=jnp.float32)
        return images, targets, False
        
    def compute_loss(self, outputs, targets, **kwargs):
        return compute_grid_loss(outputs, targets)
        
    def compute_metrics(self, outputs, targets):
        # En détection, pour le système de Checkpoint qui "maximise", 
        # on retourne l'inverse mathématique de la Loss
        return -compute_grid_loss(outputs, targets)
        
    def generate_reports(self, val_ds, final_state, model, config):
        from reporting import DetectionReporter
        import numpy as np
        reporter = DetectionReporter(
            image_size=config["image_size"],
            grid_size=config.get("grid_size", 7)
        )
        try:
            for vis_imgs, vis_boxes in val_ds.take(1).as_numpy_iterator():
                vars = {'params': final_state.params, 'batch_stats': final_state.batch_stats}
                pred_grid = final_state.apply_fn(vars, vis_imgs, training=False)
                reporter.visualize_batch(
                    images=np.array(vis_imgs),
                    predictions=np.array(pred_grid),
                    targets=np.array(vis_boxes),
                    save_path="final_detection_vis.png",
                    conf_threshold=0.5
                )
                break
            print("✅ Visualisation de détection finale générée avec succès!")
        except Exception as e:
            print(f"❌ Erreur lors de la visualisation: {e}")
