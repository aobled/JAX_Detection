"""
Fonctions utilitaires JAX pures (jittables)
Séparation des utilitaires pour meilleure organisation
"""

import jax
import jax.numpy as jnp


# ======================
# Utilitaires de manipulation d'arbres JAX
# ======================

def smooth_labels(labels, num_classes, factor=0.1):
    """
    Label smoothing pour régularisation
    
    Args:
        labels: Labels entiers
        num_classes: Nombre de classes
        factor: Facteur de smoothing (0.1 = 10% de smoothing)
    
    Returns:
        Labels smoothés (one-hot avec smoothing)
    """
    labels_oh = jax.nn.one_hot(labels, num_classes)
    return labels_oh * (1 - factor) + factor / num_classes


def mixup_batch(images, labels, alpha, num_classes, rng):
    """
    Applique Mixup : mélange linéaire de paires d'images et labels
    
    Mixup régularise en créant des exemples synthétiques :
    x_mixed = λ × x_i + (1-λ) × x_j
    y_mixed = λ × y_i + (1-λ) × y_j
    
    où λ ~ Beta(α, α)
    
    Args:
        images: Batch d'images (B, H, W, C)
        labels: Labels entiers (B,)
        alpha: Paramètre de la distribution Beta (0.2-0.4 recommandé)
        num_classes: Nombre de classes
        rng: Clé RNG JAX
    
    Returns:
        tuple: (images_mixées, labels_mixés_onehot)
    """
    batch_size = images.shape[0]
    
    # Tirer λ depuis Beta(alpha, alpha)
    rng, lambda_rng = jax.random.split(rng)
    lambda_val = jax.random.beta(lambda_rng, alpha, alpha)
    
    # Permuter le batch pour créer les paires
    rng, perm_rng = jax.random.split(rng)
    indices = jax.random.permutation(perm_rng, batch_size)
    
    # Mélanger les images
    mixed_images = lambda_val * images + (1 - lambda_val) * images[indices]
    
    # Mélanger les labels (convertir en one-hot d'abord)
    labels_onehot = jax.nn.one_hot(labels, num_classes)
    mixed_labels = lambda_val * labels_onehot + (1 - lambda_val) * labels_onehot[indices]
    
    return mixed_images, mixed_labels


def tree_add(tree_a, tree_b):
    """
    Addition de deux arbres JAX (pytrees)
    
    Args:
        tree_a: Premier arbre
        tree_b: Deuxième arbre
    
    Returns:
        Arbre résultant de l'addition
    """
    return jax.tree_util.tree_map(lambda a, b: a + b, tree_a, tree_b)


def tree_div(tree, scalar):
    """
    Division d'un arbre JAX par un scalaire
    
    Args:
        tree: Arbre JAX
        scalar: Scalaire diviseur
    
    Returns:
        Arbre divisé
    """
    return jax.tree_util.tree_map(lambda a: a / scalar, tree)


def batch_stats_div(batch_stats, denom):
    """
    Division des batch_stats par un dénominateur
    
    Args:
        batch_stats: Batch statistics (ou None)
        denom: Dénominateur
    
    Returns:
        Batch stats divisés (ou None)
    """
    if batch_stats is None:
        return None
    return jax.tree_util.tree_map(lambda x: x / denom, batch_stats)


# ======================
# Fonctions d'entraînement JIT
# ======================

def create_eval_step():
    """
    Crée une fonction d'évaluation JIT
    
    Returns:
        Fonction eval_step jittée
    """
    @jax.jit
    def eval_step(state, images, labels, rng=None):
        """Étape d'évaluation"""
        vars = {"params": state.params}
        if state.batch_stats is not None and state.batch_stats != {}:
            vars["batch_stats"] = state.batch_stats
        
        # Pour l'évaluation, on utilise training=False
        if rng is None:
            rng = jax.random.PRNGKey(42)
        
        rngs = {"dropout": rng}
        logits, _ = state.apply_fn(vars, images, training=False, mutable=["batch_stats"], rngs=rngs)
        
        import optax
        loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels).mean()
        accuracy = jnp.mean(jnp.argmax(logits, axis=-1) == labels)
        return loss, accuracy
    
    return eval_step


def create_train_step():
    """
    Crée une fonction de train step JIT
    
    Returns:
        Fonction train_step jittée
    """
    @jax.jit
    def train_step(params, batch_stats, apply_fn, images, labels, rng):
        """Étape d'entraînement"""
        import optax
        
        def loss_fn(params):
            vars = {'params': params, 'batch_stats': batch_stats}
            logits, new_batch_stats = apply_fn(
                vars, images, training=True, 
                rngs={'dropout': rng}, 
                mutable=['batch_stats']
            )
            loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels).mean()
            return loss, (logits, new_batch_stats)
        
        (loss, (logits, new_batch_stats)), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        return loss, grads, logits, new_batch_stats
    
    return train_step


# ======================
# Utilitaires de calcul
# ======================

def count_parameters(params):
    """
    Compte le nombre de paramètres dans un pytree
    
    Args:
        params: Paramètres du modèle (pytree)
    
    Returns:
        Nombre total de paramètres
    """
    return sum(x.size for x in jax.tree_util.tree_flatten(params)[0])


def get_model_size_mb(params):
    """
    Calcule la taille du modèle en MB
    
    Args:
        params: Paramètres du modèle
    
    Returns:
        Taille en MB
    """
    total_params = count_parameters(params)
    return total_params * 4 / (1024 * 1024)  # float32 = 4 bytes

