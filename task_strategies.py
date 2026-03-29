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
