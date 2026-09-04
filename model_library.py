"""
Librairie des modèles de deep learning
Contient tous les modèles utilisés pour l'entraînement
"""

import inspect
import math
from typing import Any, Optional

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.training import train_state
import flax

from detection_target_encoding import HEATMAP_KEY, SIZE_KEY


class SeparableConv(nn.Module):
    """Convolution séparable (Depthwise + Pointwise)"""
    filters: int
    kernel_size: tuple
    strides: tuple = (1, 1)
    padding: str = "SAME"
    use_bias: bool = False
    # compute_dtype (spec-compute-dtype-hardware, 2026-08-17, AD-4/AD-5) : dtype de
    # CALCUL des deux nn.Conv internes uniquement - poids stockes (param_dtype)
    # inchanges, restent float32 par defaut flax. Doit etre passe explicitement par
    # CHAQUE appelant (AD-5), jamais herite implicitement.
    compute_dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, x, training=True):
        # Depthwise convolution
        x = nn.Conv(
            features=x.shape[-1],
            kernel_size=self.kernel_size,
            strides=self.strides,
            padding=self.padding,
            feature_group_count=x.shape[-1],
            use_bias=self.use_bias,
            kernel_init=nn.initializers.kaiming_normal(),
            dtype=self.compute_dtype
        )(x)
        # Pointwise convolution
        x = nn.Conv(
            features=self.filters,
            kernel_size=(1, 1),
            padding=self.padding,
            use_bias=self.use_bias,
            kernel_init=nn.initializers.kaiming_normal(),
            dtype=self.compute_dtype
        )(x)
        return x


class SEBlock(nn.Module):
    """Squeeze-and-Excitation Block"""
    reduction: int = 16
    # compute_dtype (spec-compute-dtype-hardware, 2026-08-17, AD-4/AD-5) : cf.
    # SeparableConv ci-dessus - s'applique ici aux deux nn.Dense internes.
    compute_dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, x, training=True):
        B, H, W, C = x.shape
        # Global Average Pooling
        gap = jnp.mean(x, axis=(1, 2))
        # Squeeze
        se_dense1 = nn.Dense(C // self.reduction, use_bias=True, dtype=self.compute_dtype)(gap)
        se_act1 = nn.silu(se_dense1)
        # Excite
        se_dense2 = nn.Dense(C, use_bias=True, dtype=self.compute_dtype)(se_act1)
        se_sigmoid = nn.sigmoid(se_dense2)
        # Scale
        se_broadcast = jnp.expand_dims(jnp.expand_dims(se_sigmoid, 1), 1)
        return x * se_broadcast


class SpatialAttention(nn.Module):
    """Spatial Attention Module"""
    # compute_dtype (spec-compute-dtype-hardware, 2026-08-17, AD-4/AD-5) : cf.
    # SeparableConv ci-dessus - s'applique ici au nn.Conv interne.
    compute_dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, x, training=True):
        # Spatial attention
        spatial_attn = nn.Conv(
            features=1,
            kernel_size=(1, 1),
            padding="SAME",
            use_bias=True,
            kernel_init=nn.initializers.kaiming_normal(),
            dtype=self.compute_dtype
        )(x)
        spatial_attn = nn.sigmoid(spatial_attn)
        return x * spatial_attn


class SophisticatedCNN128Plus(nn.Module):
    """
    CNN sophistiqué OPTIMISÉ+ pour images 128×128
    
    Améliorations par rapport à SophisticatedCNN128:
    - Multi-scale feature fusion: Concat features de différentes résolutions
    - CBAM attention: Channel + Spatial attention combinée
    - Classification head amélioré: 2 couches au lieu d'1
    - Skip connections dans le head
    - Paramètres: 1.26M mesuré (num_classes=32) — corrigé 2026-07-26, le "~4M" d'origine
      était faux (jamais vérifié par comptage réel, voir memoire "verifier contre logs")

    Objectif: Dépasser 90% de validation
    Val attendue: 90-92% (vs 87% avec modèle standard)
    """
    num_classes: int = 2
    dropout_rate: float = 0.0
    # compute_dtype (spec-compute-dtype-rollout-classification, 2026-08-17, AD-1/AD-4/
    # AD-5/AD-6 ; AD-4 amende par spec-compute-dtype-batchnorm-layernorm, 2026-08-17) :
    # dtype de CALCUL des nn.Conv/nn.Dense ET nn.BatchNorm/nn.LayerNorm (jamais
    # nn.Embed) - derive du materiel par main.py (bfloat16 sur TPU, float32 sinon),
    # injecte par introspection de signature. Poids stockes (checkpoint) restent
    # float32 dans tous les cas (AD-6) - champ dataclass statique, jamais un
    # nn.Param, absent du pytree de parametres. Meme regle que
    # SophisticatedCNN128Lite/32Plus deja adaptes.
    compute_dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, x, training=True):
        # Input: (B, 128, 128, C) où C=1 (grayscale) ou C=3 (RGB)

        # === STOCKAGE DES FEATURES POUR MULTI-SCALE FUSION ===
        multi_scale_features = []

        # === BLOC 0: Conv initiale (adaptatif au nombre de canaux) ===
        x = nn.Conv(64, (3, 3), padding="SAME", use_bias=False,
                   kernel_init=nn.initializers.kaiming_normal(), dtype=self.compute_dtype)(x)
        x = nn.BatchNorm(use_running_average=not training, dtype=self.compute_dtype)(x)
        x = nn.silu(x)

        # === BLOC 1: 96 canaux (NOUVEAU pour 128×128) ===
        x = SeparableConv(96, (3, 3), compute_dtype=self.compute_dtype)(x, training)
        x = nn.BatchNorm(use_running_average=not training, dtype=self.compute_dtype)(x)
        x = nn.silu(x)

        # Residual connection
        residual = nn.Conv(96, (1, 1), padding="SAME", use_bias=False,
                          kernel_init=nn.initializers.kaiming_normal(), dtype=self.compute_dtype)(x)
        residual = nn.BatchNorm(use_running_average=not training, dtype=self.compute_dtype)(residual)

        x = SeparableConv(96, (3, 3), compute_dtype=self.compute_dtype)(x, training)
        x = nn.BatchNorm(use_running_average=not training, dtype=self.compute_dtype)(x)
        x = x + residual
        x = nn.silu(x)

        # Max Pool 1: 128×128 → 64×64
        x = nn.max_pool(x, (2, 2), strides=(2, 2))

        # === BLOC 2: 128 canaux ===
        x = SeparableConv(128, (3, 3), compute_dtype=self.compute_dtype)(x, training)
        x = nn.BatchNorm(use_running_average=not training, dtype=self.compute_dtype)(x)
        x = nn.silu(x)

        # Residual connection
        residual = nn.Conv(128, (1, 1), padding="SAME", use_bias=False,
                          kernel_init=nn.initializers.kaiming_normal(), dtype=self.compute_dtype)(x)
        residual = nn.BatchNorm(use_running_average=not training, dtype=self.compute_dtype)(residual)

        x = SeparableConv(128, (3, 3), compute_dtype=self.compute_dtype)(x, training)
        x = nn.BatchNorm(use_running_average=not training, dtype=self.compute_dtype)(x)
        x = x + residual
        x = nn.silu(x)

        # Max Pool 2: 64×64 → 32×32
        x = nn.max_pool(x, (2, 2), strides=(2, 2))

        # === BLOC 3: 256 canaux ===
        x = SeparableConv(256, (3, 3), compute_dtype=self.compute_dtype)(x, training)
        x = nn.BatchNorm(use_running_average=not training, dtype=self.compute_dtype)(x)
        x = nn.silu(x)

        x = SeparableConv(256, (3, 3), compute_dtype=self.compute_dtype)(x, training)
        x = nn.BatchNorm(use_running_average=not training, dtype=self.compute_dtype)(x)
        x = nn.silu(x)

        # SE Attention
        x = SEBlock(reduction=16, compute_dtype=self.compute_dtype)(x, training)

        # Max Pool 3: 32×32 → 16×16
        x = nn.max_pool(x, (2, 2), strides=(2, 2))

        # === BLOC 4: 512 canaux ===
        x = nn.Conv(384, (1, 1), padding="SAME", use_bias=False,
                   kernel_init=nn.initializers.kaiming_normal(), dtype=self.compute_dtype)(x)
        x = nn.BatchNorm(use_running_average=not training, dtype=self.compute_dtype)(x)
        x = nn.silu(x)

        x = SeparableConv(512, (3, 3), compute_dtype=self.compute_dtype)(x, training)
        x = nn.BatchNorm(use_running_average=not training, dtype=self.compute_dtype)(x)
        x = nn.silu(x)

        # Residual connection
        residual = nn.Conv(512, (1, 1), padding="SAME", use_bias=False,
                          kernel_init=nn.initializers.kaiming_normal(), dtype=self.compute_dtype)(x)
        residual = nn.BatchNorm(use_running_average=not training, dtype=self.compute_dtype)(residual)

        x = nn.Conv(512, (1, 1), padding="SAME", use_bias=False,
                   kernel_init=nn.initializers.kaiming_normal(), dtype=self.compute_dtype)(x)
        x = nn.BatchNorm(use_running_average=not training, dtype=self.compute_dtype)(x)
        x = x + residual
        x = nn.silu(x)

        # SE Attention
        x = SEBlock(reduction=16, compute_dtype=self.compute_dtype)(x, training)

        # Spatial Attention
        x = SpatialAttention(compute_dtype=self.compute_dtype)(x, training)

        # Global Average Pooling
        x = jnp.mean(x, axis=(1, 2))  # (B, 512)

        # Classification head
        x = nn.LayerNorm(dtype=self.compute_dtype)(x)
        x = nn.Dense(384, use_bias=True, dtype=self.compute_dtype)(x)
        x = nn.silu(x)
        x = nn.Dropout(self.dropout_rate, deterministic=not training)(x)

        # Recast explicite en float32 avant retour, quel que soit compute_dtype - la
        # loss (optax.softmax_cross_entropy*, task_strategies.py::ClassificationStrategy)
        # ne doit jamais recevoir des logits en precision reduite (meme garde que
        # SophisticatedCNN128Lite/32Plus/ChessMoveTokenTransformer, pour la meme raison :
        # stabilite numerique du softmax/cross-entropy).
        return nn.Dense(self.num_classes, use_bias=True, dtype=self.compute_dtype)(x).astype(jnp.float32)


def create_sophisticated_cnn_128_plus(num_classes=2, dropout_rate=0.0, compute_dtype=jnp.float32):
    """Crée une instance de SophisticatedCNN128Plus optimisé+ pour images 128×128. Valide compute_dtype (voir _validate_compute_dtype)."""
    _validate_compute_dtype(compute_dtype)
    return SophisticatedCNN128Plus(num_classes=num_classes, dropout_rate=dropout_rate, compute_dtype=compute_dtype)


class SophisticatedCNN128Lite(nn.Module):
    """
    CNN sophistiqué OPTIMISÉ+ pour images 128×128 — variante LITE (vitesse d'inférence)

    Même topologie que SophisticatedCNN128Plus (SeparableConv, résiduelles, CBAM,
    tête à 2 couches), largeurs réduites aux deux points chauds identifiés (Winston,
    2026-07-26) :
    - Bloc 1 : 96 → 64 canaux — tourne à pleine résolution 128×128 avant tout
      maxpool, donc le principal point chaud en FLOPs (pas en paramètres) malgré
      son faible nombre de canaux.
    - Bloc 4 : pic 512 → 384 canaux (×0.75, même ratio que le conv de compression
      384/512 d'origine) — principal point chaud en paramètres (convs 1×1 à large
      canal sur une carte déjà réduite à 16×16).
    Blocs 0/2/3 et tête de classification inchangés par choix (portée limitée aux
    deux blocs validés). SophisticatedCNN128Plus reste inchangé pour comparaison
    et retour arrière.
    """
    num_classes: int = 2
    dropout_rate: float = 0.0
    # compute_dtype (spec-compute-dtype-hardware, 2026-08-17, AD-1/AD-4/AD-6 ; AD-4
    # amende par spec-compute-dtype-batchnorm-layernorm, 2026-08-17) : dtype de CALCUL
    # des nn.Conv/nn.Dense ET nn.BatchNorm/nn.LayerNorm (jamais nn.Embed) - derive du
    # materiel par main.py (bfloat16 sur TPU, float32 sinon), injecte par introspection
    # de signature. Poids stockes (checkpoint) restent float32 dans tous les cas
    # (AD-6) - champ dataclass statique, jamais un nn.Param, absent du pytree de
    # parametres.
    compute_dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, x, training=True):
        # Input: (B, 128, 128, C) où C=1 (grayscale) ou C=3 (RGB)

        # === BLOC 0: Conv initiale (inchangé) ===
        x = nn.Conv(64, (3, 3), padding="SAME", use_bias=False,
                   kernel_init=nn.initializers.kaiming_normal(), dtype=self.compute_dtype)(x)
        x = nn.BatchNorm(use_running_average=not training, dtype=self.compute_dtype)(x)
        x = nn.silu(x)

        # === BLOC 1: 64 canaux (LITE: 96 -> 64, pleine résolution 128×128) ===
        x = SeparableConv(64, (3, 3), compute_dtype=self.compute_dtype)(x, training)
        x = nn.BatchNorm(use_running_average=not training, dtype=self.compute_dtype)(x)
        x = nn.silu(x)

        residual = nn.Conv(64, (1, 1), padding="SAME", use_bias=False,
                          kernel_init=nn.initializers.kaiming_normal(), dtype=self.compute_dtype)(x)
        residual = nn.BatchNorm(use_running_average=not training, dtype=self.compute_dtype)(residual)

        x = SeparableConv(64, (3, 3), compute_dtype=self.compute_dtype)(x, training)
        x = nn.BatchNorm(use_running_average=not training, dtype=self.compute_dtype)(x)
        x = x + residual
        x = nn.silu(x)

        # Max Pool 1: 128×128 → 64×64
        x = nn.max_pool(x, (2, 2), strides=(2, 2))

        # === BLOC 2: 128 canaux (inchangé) ===
        x = SeparableConv(128, (3, 3), compute_dtype=self.compute_dtype)(x, training)
        x = nn.BatchNorm(use_running_average=not training, dtype=self.compute_dtype)(x)
        x = nn.silu(x)

        residual = nn.Conv(128, (1, 1), padding="SAME", use_bias=False,
                          kernel_init=nn.initializers.kaiming_normal(), dtype=self.compute_dtype)(x)
        residual = nn.BatchNorm(use_running_average=not training, dtype=self.compute_dtype)(residual)

        x = SeparableConv(128, (3, 3), compute_dtype=self.compute_dtype)(x, training)
        x = nn.BatchNorm(use_running_average=not training, dtype=self.compute_dtype)(x)
        x = x + residual
        x = nn.silu(x)

        # Max Pool 2: 64×64 → 32×32
        x = nn.max_pool(x, (2, 2), strides=(2, 2))

        # === BLOC 3: 256 canaux (inchangé) ===
        x = SeparableConv(256, (3, 3), compute_dtype=self.compute_dtype)(x, training)
        x = nn.BatchNorm(use_running_average=not training, dtype=self.compute_dtype)(x)
        x = nn.silu(x)

        x = SeparableConv(256, (3, 3), compute_dtype=self.compute_dtype)(x, training)
        x = nn.BatchNorm(use_running_average=not training, dtype=self.compute_dtype)(x)
        x = nn.silu(x)

        # SE Attention
        x = SEBlock(reduction=16, compute_dtype=self.compute_dtype)(x, training)

        # Max Pool 3: 32×32 → 16×16
        x = nn.max_pool(x, (2, 2), strides=(2, 2))

        # === BLOC 4: 384 canaux (LITE: pic 512 -> 384, x0.75) ===
        x = nn.Conv(288, (1, 1), padding="SAME", use_bias=False,
                   kernel_init=nn.initializers.kaiming_normal(), dtype=self.compute_dtype)(x)
        x = nn.BatchNorm(use_running_average=not training, dtype=self.compute_dtype)(x)
        x = nn.silu(x)

        x = SeparableConv(384, (3, 3), compute_dtype=self.compute_dtype)(x, training)
        x = nn.BatchNorm(use_running_average=not training, dtype=self.compute_dtype)(x)
        x = nn.silu(x)

        # Residual connection
        residual = nn.Conv(384, (1, 1), padding="SAME", use_bias=False,
                          kernel_init=nn.initializers.kaiming_normal(), dtype=self.compute_dtype)(x)
        residual = nn.BatchNorm(use_running_average=not training, dtype=self.compute_dtype)(residual)

        x = nn.Conv(384, (1, 1), padding="SAME", use_bias=False,
                   kernel_init=nn.initializers.kaiming_normal(), dtype=self.compute_dtype)(x)
        x = nn.BatchNorm(use_running_average=not training, dtype=self.compute_dtype)(x)
        x = x + residual
        x = nn.silu(x)

        # SE Attention
        x = SEBlock(reduction=16, compute_dtype=self.compute_dtype)(x, training)

        # Spatial Attention
        x = SpatialAttention(compute_dtype=self.compute_dtype)(x, training)

        # Global Average Pooling
        x = jnp.mean(x, axis=(1, 2))  # (B, 384)

        # Classification head (inchangé)
        x = nn.LayerNorm(dtype=self.compute_dtype)(x)
        x = nn.Dense(384, use_bias=True, dtype=self.compute_dtype)(x)
        x = nn.silu(x)
        x = nn.Dropout(self.dropout_rate, deterministic=not training)(x)

        # Recast explicite en float32 avant retour, quel que soit compute_dtype - la
        # loss (optax.softmax_cross_entropy*, task_strategies.py::ClassificationStrategy)
        # ne doit jamais recevoir des logits en precision reduite (meme garde que
        # ChessMoveTokenTransformer, model_library.py, pour la meme raison : stabilite
        # numerique du softmax/cross-entropy, pas seulement du softmax d'attention).
        return nn.Dense(self.num_classes, use_bias=True, dtype=self.compute_dtype)(x).astype(jnp.float32)


def create_sophisticated_cnn_128_lite(num_classes=2, dropout_rate=0.0, compute_dtype=jnp.float32):
    """Crée une instance de SophisticatedCNN128Lite (Blocs 1+4 allégés pour la vitesse d'inférence). Valide compute_dtype (voir _validate_compute_dtype)."""
    _validate_compute_dtype(compute_dtype)
    return SophisticatedCNN128Lite(num_classes=num_classes, dropout_rate=dropout_rate, compute_dtype=compute_dtype)


class SophisticatedCNN32Plus(nn.Module):
    """
    CNN sophistiqué OPTIMISÉ+ pour images 32×32 (ex. CIFAR-10)

    Variante réduite de SophisticatedCNN128Plus : même architecture (SeparableConv,
    résiduelles, SE Attention, Spatial Attention, tête GAP), mais canaux et profondeur
    de pooling adaptés à une entrée 16× plus petite en pixels — 2 max-pools (32→16→8)
    au lieu de 3, canaux à peu près divisés par 2 à chaque étage (pic à 256 au lieu de 512).
    """
    num_classes: int = 10
    dropout_rate: float = 0.0
    # compute_dtype (spec-compute-dtype-hardware, 2026-08-17, AD-1/AD-4/AD-6 ; AD-4
    # amende par spec-compute-dtype-batchnorm-layernorm, 2026-08-17) : cf.
    # SophisticatedCNN128Lite ci-dessus - meme regle (Conv/Dense/BatchNorm/LayerNorm),
    # meme garantie checkpoint.
    compute_dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, x, training=True):
        # Input: (B, 32, 32, C) où C=1 (grayscale) ou C=3 (RGB)

        # === BLOC 0: Conv initiale ===
        x = nn.Conv(32, (3, 3), padding="SAME", use_bias=False,
                   kernel_init=nn.initializers.kaiming_normal(), dtype=self.compute_dtype)(x)
        x = nn.BatchNorm(use_running_average=not training, dtype=self.compute_dtype)(x)
        x = nn.silu(x)

        # === BLOC 1: 48 canaux ===
        x = SeparableConv(48, (3, 3), compute_dtype=self.compute_dtype)(x, training)
        x = nn.BatchNorm(use_running_average=not training, dtype=self.compute_dtype)(x)
        x = nn.silu(x)

        residual = nn.Conv(48, (1, 1), padding="SAME", use_bias=False,
                          kernel_init=nn.initializers.kaiming_normal(), dtype=self.compute_dtype)(x)
        residual = nn.BatchNorm(use_running_average=not training, dtype=self.compute_dtype)(residual)

        x = SeparableConv(48, (3, 3), compute_dtype=self.compute_dtype)(x, training)
        x = nn.BatchNorm(use_running_average=not training, dtype=self.compute_dtype)(x)
        x = x + residual
        x = nn.silu(x)

        # Max Pool 1: 32×32 → 16×16
        x = nn.max_pool(x, (2, 2), strides=(2, 2))

        # === BLOC 2: 64 canaux ===
        x = SeparableConv(64, (3, 3), compute_dtype=self.compute_dtype)(x, training)
        x = nn.BatchNorm(use_running_average=not training, dtype=self.compute_dtype)(x)
        x = nn.silu(x)

        residual = nn.Conv(64, (1, 1), padding="SAME", use_bias=False,
                          kernel_init=nn.initializers.kaiming_normal(), dtype=self.compute_dtype)(x)
        residual = nn.BatchNorm(use_running_average=not training, dtype=self.compute_dtype)(residual)

        x = SeparableConv(64, (3, 3), compute_dtype=self.compute_dtype)(x, training)
        x = nn.BatchNorm(use_running_average=not training, dtype=self.compute_dtype)(x)
        x = x + residual
        x = nn.silu(x)

        # Max Pool 2: 16×16 → 8×8
        x = nn.max_pool(x, (2, 2), strides=(2, 2))

        # === BLOC 3: 128 canaux ===
        x = SeparableConv(128, (3, 3), compute_dtype=self.compute_dtype)(x, training)
        x = nn.BatchNorm(use_running_average=not training, dtype=self.compute_dtype)(x)
        x = nn.silu(x)

        x = SeparableConv(128, (3, 3), compute_dtype=self.compute_dtype)(x, training)
        x = nn.BatchNorm(use_running_average=not training, dtype=self.compute_dtype)(x)
        x = nn.silu(x)

        # SE Attention
        x = SEBlock(reduction=16, compute_dtype=self.compute_dtype)(x, training)

        # Pas de 3e max-pool (contrairement à 128-Plus) : reste à 8×8, suffisant vu la taille d'entrée native

        # === BLOC 4: bottleneck 192 -> 256 canaux ===
        x = nn.Conv(192, (1, 1), padding="SAME", use_bias=False,
                   kernel_init=nn.initializers.kaiming_normal(), dtype=self.compute_dtype)(x)
        x = nn.BatchNorm(use_running_average=not training, dtype=self.compute_dtype)(x)
        x = nn.silu(x)

        x = SeparableConv(256, (3, 3), compute_dtype=self.compute_dtype)(x, training)
        x = nn.BatchNorm(use_running_average=not training, dtype=self.compute_dtype)(x)
        x = nn.silu(x)

        residual = nn.Conv(256, (1, 1), padding="SAME", use_bias=False,
                          kernel_init=nn.initializers.kaiming_normal(), dtype=self.compute_dtype)(x)
        residual = nn.BatchNorm(use_running_average=not training, dtype=self.compute_dtype)(residual)

        x = nn.Conv(256, (1, 1), padding="SAME", use_bias=False,
                   kernel_init=nn.initializers.kaiming_normal(), dtype=self.compute_dtype)(x)
        x = nn.BatchNorm(use_running_average=not training, dtype=self.compute_dtype)(x)
        x = x + residual
        x = nn.silu(x)

        # SE Attention
        x = SEBlock(reduction=16, compute_dtype=self.compute_dtype)(x, training)

        # Spatial Attention
        x = SpatialAttention(compute_dtype=self.compute_dtype)(x, training)

        # Global Average Pooling
        x = jnp.mean(x, axis=(1, 2))  # (B, 256)

        # Classification head
        x = nn.LayerNorm(dtype=self.compute_dtype)(x)
        x = nn.Dense(192, use_bias=True, dtype=self.compute_dtype)(x)
        x = nn.silu(x)
        x = nn.Dropout(self.dropout_rate, deterministic=not training)(x)

        # Recast explicite en float32 avant retour, quel que soit compute_dtype - meme
        # garde que SophisticatedCNN128Lite/ChessMoveTokenTransformer ci-dessus.
        return nn.Dense(self.num_classes, use_bias=True, dtype=self.compute_dtype)(x).astype(jnp.float32)


def create_sophisticated_cnn_32_plus(num_classes=2, dropout_rate=0.0, compute_dtype=jnp.float32):
    """Crée une instance de SophisticatedCNN32Plus, variante réduite pour images 32×32 (ex. CIFAR-10). Valide compute_dtype (voir _validate_compute_dtype)."""
    _validate_compute_dtype(compute_dtype)
    return SophisticatedCNN32Plus(num_classes=num_classes, dropout_rate=dropout_rate, compute_dtype=compute_dtype)


# SophisticatedCNN64Plus (variante 64×64) testée et rejetée le 2026-07-15 : -6.1 pts d'accuracy
# vs 128×128 (0.8910 vs 0.9522), dégradation sur les 32 classes sans exception. Supprimée -
# récupérable via l'historique git si le sujet est un jour repris avec une vraie raison de le faire.


class AircraftDetectorUNet(nn.Module):
    """
    Détecteur d'avions par Segmentation Sémantique (U-Net)
    Input: (B, 224, 224, C)
    Output: (B, 224, 224, 1) Mask de probabilités (0 à 1)
    """
    dropout_rate: float = 0.2
    # compute_dtype (spec-compute-dtype-detection, 2026-08-17, AD-1/AD-3/AD-4/AD-6) :
    # dtype de CALCUL de CHAQUE nn.Conv/nn.BatchNorm - meme regle que
    # SophisticatedCNN32Plus/128Lite/128Plus ci-dessus (Conv/Dense/BatchNorm/
    # LayerNorm suivent compute_dtype, jamais nn.Embed - non applicable ici, pas de
    # nn.Embed dans cette classe). Poids stockes (checkpoint) restent float32 dans
    # tous les cas (AD-6). La sortie (masque, passe par nn.sigmoid) est recastee
    # float32 AVANT la non-linearite, pas seulement avant le retour - cas non
    # couvert par le precedent classification, la non-linearite ne doit jamais
    # recevoir des logits en precision reduite.
    compute_dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, x, training=True):
        # --- ENCODER ---
        # Block 1 (224x224 -> 112x112)
        x1 = nn.Conv(32, (3, 3), padding="SAME", dtype=self.compute_dtype)(x)
        x1 = nn.BatchNorm(use_running_average=not training, dtype=self.compute_dtype)(x1)
        x1 = nn.silu(x1)
        x1 = nn.Conv(32, (3, 3), padding="SAME", dtype=self.compute_dtype)(x1)
        x1 = nn.BatchNorm(use_running_average=not training, dtype=self.compute_dtype)(x1)
        x1 = nn.silu(x1)
        p1 = nn.max_pool(x1, window_shape=(2, 2), strides=(2, 2)) # 112x112

        # Block 2 (112x112 -> 56x56)
        x2 = nn.Conv(64, (3, 3), padding="SAME", dtype=self.compute_dtype)(p1)
        x2 = nn.BatchNorm(use_running_average=not training, dtype=self.compute_dtype)(x2)
        x2 = nn.silu(x2)
        x2 = nn.Conv(64, (3, 3), padding="SAME", dtype=self.compute_dtype)(x2)
        x2 = nn.BatchNorm(use_running_average=not training, dtype=self.compute_dtype)(x2)
        x2 = nn.silu(x2)
        p2 = nn.max_pool(x2, window_shape=(2, 2), strides=(2, 2)) # 56x56

        # Block 3 (56x56 -> 28x28)
        x3 = nn.Conv(128, (3, 3), padding="SAME", dtype=self.compute_dtype)(p2)
        x3 = nn.BatchNorm(use_running_average=not training, dtype=self.compute_dtype)(x3)
        x3 = nn.silu(x3)
        x3 = nn.Conv(128, (3, 3), padding="SAME", dtype=self.compute_dtype)(x3)
        x3 = nn.BatchNorm(use_running_average=not training, dtype=self.compute_dtype)(x3)
        x3 = nn.silu(x3)
        p3 = nn.max_pool(x3, window_shape=(2, 2), strides=(2, 2)) # 28x28

        # --- BOTTLENECK ---
        # 28x28
        b = nn.Conv(256, (3, 3), padding="SAME", dtype=self.compute_dtype)(p3)
        b = nn.BatchNorm(use_running_average=not training, dtype=self.compute_dtype)(b)
        b = nn.silu(b)
        b = nn.Conv(256, (3, 3), padding="SAME", dtype=self.compute_dtype)(b)
        b = nn.BatchNorm(use_running_average=not training, dtype=self.compute_dtype)(b)
        b = nn.silu(b)

        # Application du Dropout au bottleneck (le seul endroit où il est vraiment efficace sur un U-Net)
        b = nn.Dropout(self.dropout_rate, deterministic=not training)(b)

        # --- DECODER ---
        # Up 1 (28x28 -> 56x56)
        u1 = jax.image.resize(b, shape=(b.shape[0], x3.shape[1], x3.shape[2], b.shape[3]), method='bilinear')
        u1 = nn.Conv(128, (2, 2), padding="SAME", dtype=self.compute_dtype)(u1)
        u1 = jnp.concatenate([u1, x3], axis=-1)
        u1 = nn.Conv(128, (3, 3), padding="SAME", dtype=self.compute_dtype)(u1)
        u1 = nn.BatchNorm(use_running_average=not training, dtype=self.compute_dtype)(u1)
        u1 = nn.silu(u1)
        u1 = nn.Conv(128, (3, 3), padding="SAME", dtype=self.compute_dtype)(u1)
        u1 = nn.BatchNorm(use_running_average=not training, dtype=self.compute_dtype)(u1)
        u1 = nn.silu(u1)

        # Up 2 (56x56 -> 112x112)
        u2 = jax.image.resize(u1, shape=(u1.shape[0], x2.shape[1], x2.shape[2], u1.shape[3]), method='bilinear')
        u2 = nn.Conv(64, (2, 2), padding="SAME", dtype=self.compute_dtype)(u2)
        u2 = jnp.concatenate([u2, x2], axis=-1)
        u2 = nn.Conv(64, (3, 3), padding="SAME", dtype=self.compute_dtype)(u2)
        u2 = nn.BatchNorm(use_running_average=not training, dtype=self.compute_dtype)(u2)
        u2 = nn.silu(u2)
        u2 = nn.Conv(64, (3, 3), padding="SAME", dtype=self.compute_dtype)(u2)
        u2 = nn.BatchNorm(use_running_average=not training, dtype=self.compute_dtype)(u2)
        u2 = nn.silu(u2)

        # Up 3 (112x112 -> 224x224)
        u3 = jax.image.resize(u2, shape=(u2.shape[0], x1.shape[1], x1.shape[2], u2.shape[3]), method='bilinear')
        u3 = nn.Conv(32, (2, 2), padding="SAME", dtype=self.compute_dtype)(u3)
        u3 = jnp.concatenate([u3, x1], axis=-1)
        u3 = nn.Conv(32, (3, 3), padding="SAME", dtype=self.compute_dtype)(u3)
        u3 = nn.BatchNorm(use_running_average=not training, dtype=self.compute_dtype)(u3)
        u3 = nn.silu(u3)
        u3 = nn.Conv(32, (3, 3), padding="SAME", dtype=self.compute_dtype)(u3)
        u3 = nn.BatchNorm(use_running_average=not training, dtype=self.compute_dtype)(u3)
        u3 = nn.silu(u3)

        # --- OUTPUT ---
        # Mask 224x224x1 - recast explicite en float32 AVANT nn.sigmoid (pas apres) :
        # la non-linearite ne doit jamais recevoir des logits en precision reduite,
        # meme garde-principe que le recast pre-loss des modeles de classification,
        # applique ici avant l'activation plutot qu'avant une loss externe.
        out = nn.Conv(1, (1, 1), padding="SAME", dtype=self.compute_dtype)(u3)
        out = out.astype(jnp.float32)
        return nn.sigmoid(out)


def create_aircraft_detector_unet(dropout_rate=0.2, compute_dtype=jnp.float32, **kwargs):
    """Factory for UNet Detector.

    compute_dtype accepte explicitement comme parametre NOMME (AD-3) pour que
    l'introspection stricte de main.py (inspect.signature) cible cette factory -
    **kwargs seul resterait invisible a cette detection. Valide compute_dtype (voir
    _validate_compute_dtype). Bug preexistant **kwargs non transmis au constructeur
    (documente, ex. num_classes) volontairement PAS corrige ici - hors scope.
    """
    _validate_compute_dtype(compute_dtype)
    return AircraftDetectorUNet(dropout_rate=dropout_rate, compute_dtype=compute_dtype)


class AircraftDetectorCenterNet(nn.Module):
    """
    Détecteur d'avions par point central (style CenterNet, anchor-free)
    Même famille d'architecture que AircraftDetectorUNet (AD-10 : pas de backbone+FPN).
    Input: (B, H, W, C)
    Output: dict {HEATMAP_KEY: (B, H, W, 1), SIZE_KEY: (B, H, W, 2)} — même résolution
    (H, W) que l'entrée (stride=1), pas de sous-échantillonnage, pour rester compatible
    avec le schéma de cibles de detection_target_encoding.py (Story 7.1).

    heatmap_prior : proportion attendue de pixels positifs (gt_heatmap==1.0) dans le
    dataset cible — sert à initialiser le biais de la tête heatmap (voir Story 7.2,
    addendum post-hoc 2026-07-17). Défaut 0.01 = valeur générique du papier RetinaNet
    (Lin et al. 2018, §3.3) ; JAX_DETECTOR utilise sa valeur mesurée réellement
    (dataset_configs.py, ~2.68e-5, très inférieure au défaut générique car un seul pixel
    par objet sur une grille 224×224, contrairement aux milliers d'ancres du papier
    d'origine).
    """
    dropout_rate: float = 0.2
    heatmap_prior: float = 0.01
    # compute_dtype (spec-compute-dtype-detection, 2026-08-17, AD-1/AD-3/AD-4/AD-6) :
    # meme regle que AircraftDetectorUNet ci-dessus - CHAQUE nn.Conv/nn.BatchNorm de
    # cette classe (encodeur, bottleneck dilate, branche de contexte global,
    # decodeur) suit compute_dtype. Les DEUX tetes de sortie (heatmap, size) sont
    # recastees float32 avant retour - heatmap AVANT nn.sigmoid (meme garde que
    # AircraftDetectorUNet), size directement (pas d'activation a proteger, mais
    # coherence float32 avec le reste des sorties de ce projet).
    compute_dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, x, training: bool = True):
        # --- ENCODER --- (identique à AircraftDetectorUNet)
        # Block 1 (H,W -> H/2,W/2)
        x1 = nn.Conv(32, (3, 3), padding="SAME", dtype=self.compute_dtype)(x)
        x1 = nn.BatchNorm(use_running_average=not training, dtype=self.compute_dtype)(x1)
        x1 = nn.silu(x1)
        x1 = nn.Conv(32, (3, 3), padding="SAME", dtype=self.compute_dtype)(x1)
        x1 = nn.BatchNorm(use_running_average=not training, dtype=self.compute_dtype)(x1)
        x1 = nn.silu(x1)
        p1 = nn.max_pool(x1, window_shape=(2, 2), strides=(2, 2))

        # Block 2 (H/2,W/2 -> H/4,W/4)
        x2 = nn.Conv(64, (3, 3), padding="SAME", dtype=self.compute_dtype)(p1)
        x2 = nn.BatchNorm(use_running_average=not training, dtype=self.compute_dtype)(x2)
        x2 = nn.silu(x2)
        x2 = nn.Conv(64, (3, 3), padding="SAME", dtype=self.compute_dtype)(x2)
        x2 = nn.BatchNorm(use_running_average=not training, dtype=self.compute_dtype)(x2)
        x2 = nn.silu(x2)
        p2 = nn.max_pool(x2, window_shape=(2, 2), strides=(2, 2))

        # Block 3 (H/4,W/4 -> H/8,W/8)
        x3 = nn.Conv(128, (3, 3), padding="SAME", dtype=self.compute_dtype)(p2)
        x3 = nn.BatchNorm(use_running_average=not training, dtype=self.compute_dtype)(x3)
        x3 = nn.silu(x3)
        x3 = nn.Conv(128, (3, 3), padding="SAME", dtype=self.compute_dtype)(x3)
        x3 = nn.BatchNorm(use_running_average=not training, dtype=self.compute_dtype)(x3)
        x3 = nn.silu(x3)
        p3 = nn.max_pool(x3, window_shape=(2, 2), strides=(2, 2))

        # --- BOTTLENECK ---
        # Convolutions dilatees (2026-07-19, hypothese "champ receptif" - voir
        # deferred-work.md et jax-single-pass.mmd) : le bottleneck non-dilate (RF
        # theorique ~68px/224, ~30%) est structurellement trop etroit pour les boites
        # plein-cadre (~47% du dataset detection/train a une aire >=50% de l'image,
        # mesure reelle 2026-07-19 via reporting_global_boxes_size). La dilatation
        # agrandit le RF theorique a ~132px/224 (~59%) sans changer la resolution
        # spatiale du bottleneck (28x28) ni le nombre de parametres (dilation != taille
        # de noyau). N'affecte pas AircraftDetectorUNet (AD-20, code non partage malgre
        # l'architecture jumelle).
        b = nn.Conv(256, (3, 3), kernel_dilation=(2, 2), padding="SAME", dtype=self.compute_dtype)(p3)
        b = nn.BatchNorm(use_running_average=not training, dtype=self.compute_dtype)(b)
        b = nn.silu(b)
        b = nn.Conv(256, (3, 3), kernel_dilation=(4, 4), padding="SAME", dtype=self.compute_dtype)(b)
        b = nn.BatchNorm(use_running_average=not training, dtype=self.compute_dtype)(b)
        b = nn.silu(b)

        # Branche de contexte global (2026-07-22, complement a la dilatation ci-dessus -
        # voir deferred-work.md et jax-single-pass.mmd). La dilatation agrandit le champ
        # receptif de facon finie (~30%->~59%, mesure reelle : +18% sur HeatmapActivation,
        # mais insuffisant sur les boites plein-cadre 70-100% de l'image). Le pooling
        # global, lui, garantit par construction une couverture a 100% de l'image, quelle
        # que soit la taille de l'objet - independant de tout empilement de couches.
        # Moyenne spatiale globale du bottleneck -> (B,1,1,256), projetee (Conv 1x1,
        # equivalent a une couche dense sur une entree 1x1) puis rediffusee a chaque
        # position spatiale et fusionnee (concat) avec les features locales avant
        # projection retour a 256 canaux.
        context = jnp.mean(b, axis=(1, 2), keepdims=True)  # (B,1,1,256)
        context = nn.Conv(256, (1, 1), padding="SAME", dtype=self.compute_dtype)(context)
        context = nn.silu(context)
        context = jnp.broadcast_to(context, b.shape)  # (B,28,28,256)
        b = jnp.concatenate([b, context], axis=-1)  # (B,28,28,512) : local + global
        b = nn.Conv(256, (1, 1), padding="SAME", dtype=self.compute_dtype)(b)  # projection retour a 256ch
        b = nn.BatchNorm(use_running_average=not training, dtype=self.compute_dtype)(b)
        b = nn.silu(b)

        b = nn.Dropout(self.dropout_rate, deterministic=not training)(b)

        # --- DECODER --- (identique à AircraftDetectorUNet)
        # Up 1
        u1 = jax.image.resize(b, shape=(b.shape[0], x3.shape[1], x3.shape[2], b.shape[3]), method='bilinear')
        u1 = nn.Conv(128, (2, 2), padding="SAME", dtype=self.compute_dtype)(u1)
        u1 = jnp.concatenate([u1, x3], axis=-1)
        u1 = nn.Conv(128, (3, 3), padding="SAME", dtype=self.compute_dtype)(u1)
        u1 = nn.BatchNorm(use_running_average=not training, dtype=self.compute_dtype)(u1)
        u1 = nn.silu(u1)
        u1 = nn.Conv(128, (3, 3), padding="SAME", dtype=self.compute_dtype)(u1)
        u1 = nn.BatchNorm(use_running_average=not training, dtype=self.compute_dtype)(u1)
        u1 = nn.silu(u1)

        # Up 2
        u2 = jax.image.resize(u1, shape=(u1.shape[0], x2.shape[1], x2.shape[2], u1.shape[3]), method='bilinear')
        u2 = nn.Conv(64, (2, 2), padding="SAME", dtype=self.compute_dtype)(u2)
        u2 = jnp.concatenate([u2, x2], axis=-1)
        u2 = nn.Conv(64, (3, 3), padding="SAME", dtype=self.compute_dtype)(u2)
        u2 = nn.BatchNorm(use_running_average=not training, dtype=self.compute_dtype)(u2)
        u2 = nn.silu(u2)
        u2 = nn.Conv(64, (3, 3), padding="SAME", dtype=self.compute_dtype)(u2)
        u2 = nn.BatchNorm(use_running_average=not training, dtype=self.compute_dtype)(u2)
        u2 = nn.silu(u2)

        # Up 3
        u3 = jax.image.resize(u2, shape=(u2.shape[0], x1.shape[1], x1.shape[2], u2.shape[3]), method='bilinear')
        u3 = nn.Conv(32, (2, 2), padding="SAME", dtype=self.compute_dtype)(u3)
        u3 = jnp.concatenate([u3, x1], axis=-1)
        u3 = nn.Conv(32, (3, 3), padding="SAME", dtype=self.compute_dtype)(u3)
        u3 = nn.BatchNorm(use_running_average=not training, dtype=self.compute_dtype)(u3)
        u3 = nn.silu(u3)
        u3 = nn.Conv(32, (3, 3), padding="SAME", dtype=self.compute_dtype)(u3)
        u3 = nn.BatchNorm(use_running_average=not training, dtype=self.compute_dtype)(u3)
        u3 = nn.silu(u3)

        # --- OUTPUT : deux têtes paralleles ---
        # Heatmap de centres (B,H,W,1), valeurs [0,1] (Story 7.1)
        # Biais initial non nul (RetinaNet, Lin et al. 2018 §3.3) : l'init Flax par defaut
        # (biais=0 -> sigmoid(0)=0.5 partout) fait que le volume massif de gradient de
        # fond (pixels negatifs >> positifs) noie le signal des rares pixels-centres avant
        # que le reseau ait pu apprendre a les differencier - collapse observe
        # empiriquement en execution reelle (Story 7.8 : predictions quasi identiques aux
        # centres et au fond apres 1 epoch, archive/diagnose_heatmap_predictions.py). Corrige en
        # demarrant sigmoid(biais) = heatmap_prior (la vraie proportion de positifs),
        # au lieu de 0.5 non-informatif.
        assert 0.0 < self.heatmap_prior < 1.0, (
            f"heatmap_prior doit etre dans (0,1) - log(p/(1-p)) indefini sinon, recu {self.heatmap_prior}"
        )
        heatmap_bias_init = math.log(self.heatmap_prior / (1.0 - self.heatmap_prior))
        heatmap = nn.Conv(1, (1, 1), padding="SAME", bias_init=nn.initializers.constant(heatmap_bias_init), dtype=self.compute_dtype)(u3)
        # Recast explicite en float32 AVANT nn.sigmoid (meme garde que AircraftDetectorUNet) :
        # la non-linearite ne doit jamais recevoir des logits en precision reduite.
        heatmap = heatmap.astype(jnp.float32)
        heatmap = nn.sigmoid(heatmap)

        # Regression de taille (B,H,W,2) largeur/hauteur, pas d'activation
        # (convention CenterNet standard - la perte, Story 7.3, gere la positivite).
        # Recast explicite en float32 directement (pas de non-linearite a proteger ici,
        # mais coherence float32 avec le reste des sorties de ce projet - AD-6).
        size = nn.Conv(2, (1, 1), padding="SAME", dtype=self.compute_dtype)(u3)
        size = size.astype(jnp.float32)

        return {HEATMAP_KEY: heatmap, SIZE_KEY: size}


def create_aircraft_detector_centernet(dropout_rate=0.2, heatmap_prior=0.01, compute_dtype=jnp.float32, **kwargs):
    """Factory for CenterNet Detector.

    compute_dtype accepte explicitement comme parametre NOMME (AD-3), meme
    raisonnement que create_aircraft_detector_unet ci-dessus. Valide compute_dtype
    (voir _validate_compute_dtype). Bug preexistant **kwargs non transmis au
    constructeur volontairement PAS corrige ici - hors scope.
    """
    _validate_compute_dtype(compute_dtype)
    return AircraftDetectorCenterNet(dropout_rate=dropout_rate, heatmap_prior=heatmap_prior, compute_dtype=compute_dtype)


# MiniUNet/conv_block/create_aircraft_detector_miniunet supprimés le 2026-07-15 : non utilisés par
# aucune des 4 configs actives (seule référence était une ligne commentée dans dataset_configs.py),
# aucun .pkl versionné n'en dépendait (vérifié : best_model_fighterjet_detection.pkl est aircraft_detector_unet).
# Récupérable via l'historique git si besoin.


class ChessCnnAttentionPolicyValue(nn.Module):
    """
    Modèle échecs : CNN 8×8 (sans pooling) -> bottleneck de K tokens appris
    (Perceiver/TokenLearner-style, cross-attention) -> auto-attention entre
    tokens -> têtes policy + value (FR5, AD-23, Story 9.2).

    Input : (B, 8, 8, C) - planes position+historique du contrat .npz échecs
    (cote chess_ai, C=29 en pratique, mais le
    nombre de canaux d'entrée n'est pas codé en dur ici - inféré par la
    première conv, comme tous les autres modèles de ce fichier). La dimension
    spatiale est validée contre 8 (BOARD_SIZE, contrat .npz échecs) plutôt que
    supposée.
    Output : {"policy": (B, num_moves), "value": (B,)} - miroir du pattern
    dict de AircraftDetectorCenterNet (HEATMAP_KEY/SIZE_KEY), AD-24.

    Pas de maxpool (contrairement aux CNN images de ce fichier, ex.
    SophisticatedCNN128Plus/Lite) : le plateau 8×8 est déjà petit, un
    empilement de convs 3×3 couvre déjà tout le champ réceptif
    (_bmad-output/planning-artifacts/briefs/brief-jax_supervised_training-2026-07-27/brief.md,
    § "Architecture envisagée") - pooler détruirait l'identité des cases avant
    le bottleneck, alors que la tête policy (AD-22) a besoin de distinguer les
    64 cases source.

    num_moves : taille de la tête policy - doit venir de la constante 4672
    (contrat .npz échecs, côté chess_ai) côté appelant, jamais un littéral
    dupliqué ici (AD-22). Passe par le kwarg générique `num_classes` de model_kwargs
    (main.py) via create_chess_cnn_attention_policy_value ci-dessous - le champ
    interne s'appelle num_moves pour la clarté, la factory fait le pont.
    Validé > 0 à l'appel (garde-fou contre un défaut générique du type
    num_classes=2, utilisé par d'autres modèles de ce fichier).

    token_dim/num_bottleneck_tokens/num_heads : hyperparamètres d'architecture,
    valeurs de départ (proposition initiale d'Aymeric, memlog du brief
    ci-dessus, entrée du 2026-07-27 "Architecture proposée par Aymeric",
    K=8/D=64) - ajustables via la factory (voir create_chess_cnn_attention_policy_value),
    pas des contraintes fixées par le PRD/spine. token_dim doit être divisible
    par num_heads (validé à l'appel).
    """
    num_moves: int
    dropout_rate: float = 0.1
    token_dim: int = 64
    num_bottleneck_tokens: int = 8
    num_heads: int = 4

    @nn.compact
    def __call__(self, x, training: bool = True):
        assert self.token_dim % self.num_heads == 0, (
            f"token_dim ({self.token_dim}) doit etre divisible par num_heads ({self.num_heads})"
        )
        assert self.num_moves > 0, (
            f"num_moves doit etre > 0 (recu {self.num_moves}) - as-tu bien passe num_classes=4672 "
            f"(contrat .npz echecs, cote chess_ai) plutot qu'un defaut generique de ce fichier (ex. num_classes=2) ?"
        )
        assert x.shape[1] == 8 and x.shape[2] == 8, (
            f"attendu un plateau 8x8 (contrat .npz echecs, cote chess_ai), "
            f"recu {x.shape[1]}x{x.shape[2]}"
        )

        # --- BACKBONE CNN 8×8, aucun maxpool (voir docstring) ---
        x = nn.Conv(self.token_dim, (3, 3), padding="SAME", use_bias=False,
                    kernel_init=nn.initializers.kaiming_normal())(x)
        x = nn.BatchNorm(use_running_average=not training)(x)
        x = nn.silu(x)

        # 2 blocs résiduels - pattern repris caractère pour caractère de
        # SophisticatedCNN128Lite (model_library.py, cf. Dev Notes story 9.2) : le
        # skip part de x APRES la première SeparableConv+BN+silu du bloc (pas de
        # l'entrée brute du bloc), une seconde SeparableConv+BN suit avant l'addition.
        # Avec la conv initiale ci-dessus, 5 convs 3×3 au total (RF=11, couvre
        # largement les 8×8 du plateau).
        for _ in range(2):
            x = SeparableConv(self.token_dim, (3, 3))(x, training)
            x = nn.BatchNorm(use_running_average=not training)(x)
            x = nn.silu(x)

            residual = nn.Conv(self.token_dim, (1, 1), padding="SAME", use_bias=False,
                                kernel_init=nn.initializers.kaiming_normal())(x)
            residual = nn.BatchNorm(use_running_average=not training)(residual)

            x = SeparableConv(self.token_dim, (3, 3))(x, training)
            x = nn.BatchNorm(use_running_average=not training)(x)
            x = x + residual
            x = nn.silu(x)

        # --- BOTTLENECK : 64 tokens case (8×8 aplati) -> K tokens appris ---
        batch_size = x.shape[0]
        # Dimension spatiale derivee de x.shape (deja validee ci-dessus contre
        # 8), jamais un litteral "8*8" duplique.
        board_tokens = x.reshape(batch_size, x.shape[1] * x.shape[2], self.token_dim)  # (B, 64, D)

        queries = self.param(
            "bottleneck_queries",
            nn.initializers.normal(0.02),
            (self.num_bottleneck_tokens, self.token_dim),
        )
        queries = jnp.broadcast_to(
            queries[None, :, :], (batch_size, self.num_bottleneck_tokens, self.token_dim)
        )

        # Cross-attention Perceiver-style : les K requêtes apprises interrogent les
        # 64 tokens case (AD-23).
        tokens = nn.MultiHeadDotProductAttention(num_heads=self.num_heads)(
            inputs_q=queries, inputs_kv=board_tokens
        )  # (B, K, D)

        # Auto-attention standard entre les K tokens du bottleneck, sans biais
        # géométrique (AD-23, différé - voir spine Deferred).
        tokens = nn.MultiHeadDotProductAttention(num_heads=self.num_heads)(inputs_q=tokens)  # (B, K, D)

        pooled = jnp.mean(tokens, axis=1)  # (B, D)
        pooled = nn.Dropout(self.dropout_rate, deterministic=not training)(pooled)

        # --- TÊTES ---
        # Policy : logits bruts (B, num_moves) - la cross-entropy (Story 9.3) attend
        # des logits, pas une distribution déjà normalisée. Pas de masquage des coups
        # illégaux (AD-22, tranché en Story 9.1).
        policy_logits = nn.Dense(self.num_moves)(pooled)

        # Value : scalaire (B,) borné [-1, 1] (AD-24).
        value = nn.Dense(1)(pooled)
        value = nn.tanh(value)
        value = jnp.squeeze(value, axis=-1)

        return {"policy": policy_logits, "value": value}


def create_chess_cnn_attention_policy_value(num_classes, dropout_rate=0.1, **kwargs):
    """
    Factory pour le modèle échecs. `num_classes` (nom imposé par la plomberie
    model_kwargs générique de main.py, AD-22) porte en réalité la taille de l'espace
    de coups (4672, contrat .npz échecs côté chess_ai) - PAS un nombre de classes au sens habituel. Piège connu de ce fichier
    (create_aircraft_detector_centernet ci-dessus laisse tomber num_classes
    dans **kwargs sans jamais l'utiliser, correct pour CenterNet mono-classe
    mais casserait AD-22 ici) : cette factory lit et utilise num_classes.

    **kwargs transmis tel quel à ChessCnnAttentionPolicyValue - permet de
    surcharger les hyperparamètres "ajustables" documentés sur la classe
    (token_dim, num_bottleneck_tokens, num_heads) sans modifier cette factory.
    """
    return ChessCnnAttentionPolicyValue(num_moves=num_classes, dropout_rate=dropout_rate, **kwargs)


class ChessCnnAttentionLegalMoves(nn.Module):
    """
    Duplique le backbone de ChessCnnAttentionPolicyValue ci-dessus (CNN 8x8 sans
    maxpool -> bottleneck cross-attention Perceiver-style -> auto-attention) mais
    sans tete value : tache multi-label (predire l'ensemble des coups legaux d'une
    position, pas le seul coup joue) - copie deliberee plutot qu'un flag sur la
    classe policy+value (convention deja en place dans ce fichier, cf.
    CHESS_NO_HISTORY dans dataset_configs.py qui est une copie de CHESS, pas un
    flag).

    Input : (B, 8, 8, C) - meme contrat .npz echecs (position+historique) que
    ChessCnnAttentionPolicyValue.
    Output : (B, num_moves) - logits bruts, UN SEUL tenseur (pas un dict) : sigmoid
    binary cross-entropy attend des logits par coup, chaque coup etant
    independamment legal/illegal (pas une seule classe correcte comme la policy).
    """
    num_moves: int
    dropout_rate: float = 0.1
    token_dim: int = 64
    num_bottleneck_tokens: int = 8
    num_heads: int = 4

    @nn.compact
    def __call__(self, x, training: bool = True):
        assert self.token_dim % self.num_heads == 0, (
            f"token_dim ({self.token_dim}) doit etre divisible par num_heads ({self.num_heads})"
        )
        assert self.num_moves > 0, (
            f"num_moves doit etre > 0 (recu {self.num_moves}) - as-tu bien passe num_classes=4672 "
            f"(contrat .npz echecs, cote chess_ai) plutot qu'un defaut generique de ce fichier (ex. num_classes=2) ?"
        )
        assert x.shape[1] == 8 and x.shape[2] == 8, (
            f"attendu un plateau 8x8 (contrat .npz echecs, cote chess_ai), "
            f"recu {x.shape[1]}x{x.shape[2]}"
        )

        # --- BACKBONE CNN 8×8, identique a ChessCnnAttentionPolicyValue ---
        x = nn.Conv(self.token_dim, (3, 3), padding="SAME", use_bias=False,
                    kernel_init=nn.initializers.kaiming_normal())(x)
        x = nn.BatchNorm(use_running_average=not training)(x)
        x = nn.silu(x)

        for _ in range(2):
            x = SeparableConv(self.token_dim, (3, 3))(x, training)
            x = nn.BatchNorm(use_running_average=not training)(x)
            x = nn.silu(x)

            residual = nn.Conv(self.token_dim, (1, 1), padding="SAME", use_bias=False,
                                kernel_init=nn.initializers.kaiming_normal())(x)
            residual = nn.BatchNorm(use_running_average=not training)(residual)

            x = SeparableConv(self.token_dim, (3, 3))(x, training)
            x = nn.BatchNorm(use_running_average=not training)(x)
            x = x + residual
            x = nn.silu(x)

        # --- BOTTLENECK : 64 tokens case (8×8 aplati) -> K tokens appris ---
        batch_size = x.shape[0]
        board_tokens = x.reshape(batch_size, x.shape[1] * x.shape[2], self.token_dim)  # (B, 64, D)

        queries = self.param(
            "bottleneck_queries",
            nn.initializers.normal(0.02),
            (self.num_bottleneck_tokens, self.token_dim),
        )
        queries = jnp.broadcast_to(
            queries[None, :, :], (batch_size, self.num_bottleneck_tokens, self.token_dim)
        )

        tokens = nn.MultiHeadDotProductAttention(num_heads=self.num_heads)(
            inputs_q=queries, inputs_kv=board_tokens
        )  # (B, K, D)

        tokens = nn.MultiHeadDotProductAttention(num_heads=self.num_heads)(inputs_q=tokens)  # (B, K, D)

        pooled = jnp.mean(tokens, axis=1)  # (B, D)
        pooled = nn.Dropout(self.dropout_rate, deterministic=not training)(pooled)

        # --- TETE UNIQUE : logits bruts (B, num_moves) ---
        # Sigmoid BCE (ChessLegalMovesStrategy) attend des logits, pas une
        # distribution deja normalisee - pas de tete value (multi-label, pas de
        # notion de valeur de position ici).
        legal_move_logits = nn.Dense(self.num_moves)(pooled)

        return legal_move_logits


def create_chess_cnn_attention_legal_moves(num_classes, dropout_rate=0.1, **kwargs):
    """
    Factory pour ChessCnnAttentionLegalMoves - meme contrat que
    create_chess_cnn_attention_policy_value ci-dessus (num_classes porte la taille
    de l'espace de coups, 4672, pas un nombre de classes au sens habituel).
    """
    return ChessCnnAttentionLegalMoves(num_moves=num_classes, dropout_rate=dropout_rate, **kwargs)


class ChessMoveTokenTransformer(nn.Module):
    """
    Modèle chess_move_token (Epic 11, spike — AD-26/AD-28/AD-29/AD-30/AD-31/AD-33,
    spine architecture-chess-move-token-2026-08-10) : décodeur transformer CAUSAL sur
    une séquence de coup-tokens (historique de la partie), qui prédit le coup suivant
    (tête policy, 4672 classes). Première utilisation d'un masque causal dans ce
    fichier — toute l'attention existante ci-dessus (ChessCnnAttentionPolicyValue/
    ChessCnnAttentionLegalMoves) est un encodeur bidirectionnel Perceiver-style, pas
    causal.

    Input : (B, L) — séquence de tokens PADDÉE À GAUCHE (AD-28, construite par
    ChessMoveTokenDataset/data_management.py) : BOS_TOKEN_ID=4672 en tête d'historique
    réel, PAD_TOKEN_ID=4673 à gauche. Espace d'entrée (embedding) = 4674
    (num_moves + BOS + PAD, AD-30).

    x peut arriver en float32 (dummy d'initialisation de Trainer.create_train_state,
    trainer.py:145 — hors du chemin couvert par le fix dtype=jnp.int32 d'AD-29, qui ne
    couvre que la boucle d'entraînement réelle) : caste explicitement en int32 dès la
    première ligne, indépendamment du dtype fourni par l'appelant (AD-33, trouvé par
    exécution réelle — nn.Embed.init() lève sinon ValueError sur une entrée flottante).

    Output : tenseur unique (B, num_moves) — PAS un dict {"policy", "value"} (AD-26,
    ce domaine n'a aucune tête value, contrairement à ChessCnnAttentionPolicyValue).

    Lecture : état caché du DERNIER token de la séquence (indice -1) — grâce au
    padding à gauche (AD-28), c'est toujours le dernier token RÉEL, quelle que soit la
    longueur réelle de la séquence. Jamais un pooling moyen.

    num_layers/d_model/num_heads/dropout_rate : hyperparamètres, valeurs de départ
    (Story 11.2, non tunées empiriquement) — ajustables via dataset_configs.py, pas
    des contraintes fixées par le spine (Deferred, spine).

    compute_dtype (2026-08-11, précision mixte TPU) : dtype de CALCUL des couches à
    matmul lourd (`Dense`/`MultiHeadDotProductAttention`) — PAS le dtype de stockage
    des poids (`param_dtype`, jamais touché ici, reste `float32` par défaut flax :
    poids "maîtres" stables, seul le calcul transitoire passe en précision réduite).
    Distinct et indépendant du cast `int32` des tokens d'entrée (AD-33) - deux dtypes
    différents à deux endroits différents du pipeline, ne jamais confondre. Un
    checkpoint sauvegardé avant ce changement reste chargeable sans conversion : la
    forme/le dtype des paramètres stockés ne change pas. `nn.Embed`/`nn.LayerNorm`
    volontairement laissés en `float32` par défaut (gather non compute-bound pour
    Embed ; `force_float32_reductions=True` par défaut sur LayerNorm de toute façon) -
    seuls Dense/Attention bénéficient réellement d'un calcul en précision réduite sur
    TPU. `force_fp32_for_softmax=True` fixé explicitement sur l'attention (le défaut
    flax est `False` - vérifié - le softmax sur une séquence de longueur 404 reste en
    fp32 par prudence numérique, pratique standard précision mixte). Sortie
    (`policy_logits`) explicitement recastée en `float32` avant retour, quel que soit
    `compute_dtype` - `compute_chess_policy_loss` (cross-entropy) ne doit jamais
    recevoir des logits en précision réduite.
    """
    num_moves: int  # taille de la tête policy (4672) — passe par num_classes générique (main.py), jamais un littéral dupliqué (AD-22 hérité)
    dropout_rate: float = 0.1
    num_layers: int = 4
    d_model: int = 128
    num_heads: int = 4
    compute_dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, x, training: bool = True):
        assert self.d_model % self.num_heads == 0, (
            f"d_model ({self.d_model}) doit etre divisible par num_heads ({self.num_heads})"
        )
        assert self.num_moves > 0, (
            f"num_moves doit etre > 0 (recu {self.num_moves}) - as-tu bien passe num_classes=4672 "
            f"(meme convention que ChessCnnAttentionPolicyValue) ?"
        )

        # AD-33 : caste explicitement en int32, independamment du dtype fourni par
        # l'appelant (dummy d'init float32 de Trainer OU vraies donnees deja int32
        # apres le fix AD-29) - idempotent, sans cout, testee par execution reelle.
        # AUCUN rapport avec compute_dtype ci-dessous (indices de tokens vs precision
        # de calcul des activations) - ne jamais fusionner ces deux dtypes.
        tokens = jnp.asarray(x, dtype=jnp.int32)

        # Import paresseux (pas en tete de fichier) : model_library.py est importe des
        # le tout debut de main.py (avant jax/tensorflow proprement inities), alors que
        # data_management.py ne l'est que paresseusement DANS main() pour positionner
        # CUDA_VISIBLE_DEVICES avant le premier "import tensorflow" (data_management.py,
        # commentaire d'en-tete) - un import top-level ici casserait cet ordre. Sans
        # risque au call-site reel : data_management est deja charge par main.py avant
        # la creation du modele. Source unique des constantes (AD-30) - jamais un
        # litteral 4673 duplique ici.
        from data_management import CHESS_MOVE_TOKEN_PAD

        vocab_size = self.num_moves + 2  # + BOS + PAD (AD-30)
        embed = nn.Embed(num_embeddings=vocab_size, features=self.d_model)
        h = embed(tokens)  # (B, L, D) - float32 (gather non compute-bound, voir docstring)

        # Masque : causal (nn.make_causal_mask) combine au masque de padding, derive
        # directement du tenseur d'entree (tokens != PAD_TOKEN_ID) - jamais un second
        # tenseur de masque achemine via Strategy/Trainer (AD-28, apply_fn a une
        # signature fixe a un seul tenseur d'entree).
        causal_mask = nn.make_causal_mask(tokens)
        padding_mask = nn.make_attention_mask(tokens != CHESS_MOVE_TOKEN_PAD, tokens != CHESS_MOVE_TOKEN_PAD)
        mask = nn.combine_masks(causal_mask, padding_mask)

        for _ in range(self.num_layers):
            residual = h
            h = nn.LayerNorm()(h)
            h = nn.MultiHeadDotProductAttention(
                num_heads=self.num_heads,
                dtype=self.compute_dtype,
                force_fp32_for_softmax=True,  # defaut flax=False - softmax en fp32 par prudence numerique (L=404)
            )(inputs_q=h, mask=mask)
            h = nn.Dropout(self.dropout_rate, deterministic=not training)(h)
            h = residual + h

            residual = h
            h = nn.LayerNorm()(h)
            h = nn.Dense(4 * self.d_model, dtype=self.compute_dtype)(h)
            h = nn.gelu(h)
            h = nn.Dense(self.d_model, dtype=self.compute_dtype)(h)
            h = nn.Dropout(self.dropout_rate, deterministic=not training)(h)
            h = residual + h

        h = nn.LayerNorm()(h)

        # Lecture du DERNIER token (toujours reel grace au padding a gauche, AD-28) -
        # jamais un pooling moyen sur la sequence (AD-28).
        last_hidden = h[:, -1, :]  # (B, D)

        # (B, num_moves) - jamais BOS/PAD en sortie (AD-30), aucun masquage des coups
        # illegaux (AD-22 herite). Recast explicite en float32 avant retour, quel que
        # soit compute_dtype - compute_chess_policy_loss (cross-entropy) ne doit
        # jamais recevoir des logits en precision reduite (perte/instabilite possible
        # du softmax de la loss, distinct du softmax interne de l'attention deja
        # protege ci-dessus par force_fp32_for_softmax).
        policy_logits = nn.Dense(self.num_moves, dtype=self.compute_dtype)(last_hidden)
        return policy_logits.astype(jnp.float32)


def create_chess_move_token_transformer(num_classes, dropout_rate=0.1, compute_dtype="float32", **kwargs):
    """
    Factory pour chess_move_token_transformer. `num_classes` (nom imposé par la
    plomberie model_kwargs générique de main.py) porte la taille de l'espace de coups
    (4672) — même convention que create_chess_cnn_attention_policy_value ci-dessus.

    compute_dtype : chaîne ("float32"/"bfloat16"/"float16") OU un jnp.dtype déjà
    résolu. Historiquement une string uniquement (appel direct/manuel, littéral
    sérialisable) ; accepte aussi un jnp.dtype depuis spec-compute-dtype-hardware
    (2026-08-17, AD-3) — l'injection par introspection de main.py cible cette
    factory (paramètre nommé compute_dtype) et lui transmet le jnp.dtype DÉJÀ
    résolu par le mécanisme central (AD-2), jamais une string. Erreur explicite
    dans LES DEUX cas si la valeur ne correspond à rien de connu, plutôt qu'un
    échec silencieux plus loin dans le graphe (revue Edge Case Hunter, 2026-08-17 :
    le chemin non-string acceptait n'importe quel objet sans validation avant ce
    fix).
    """
    _VALID_DTYPES = (jnp.float32, jnp.bfloat16, jnp.float16)
    if isinstance(compute_dtype, str):
        resolved_dtype = getattr(jnp, compute_dtype, None)
        if resolved_dtype is None or resolved_dtype not in _VALID_DTYPES:
            raise ValueError(
                f"compute_dtype='{compute_dtype}' inconnu de jax.numpy - attendu 'float32'/'bfloat16'/'float16'"
            )
    elif compute_dtype in _VALID_DTYPES:
        resolved_dtype = compute_dtype
    else:
        raise ValueError(
            f"compute_dtype={compute_dtype!r} non reconnu - attendu une string "
            f"('float32'/'bfloat16'/'float16') ou un de {_VALID_DTYPES}"
        )
    return ChessMoveTokenTransformer(
        num_moves=num_classes, dropout_rate=dropout_rate, compute_dtype=resolved_dtype, **kwargs
    )


# Constante d'encodage de coup (chess_ai/chess_target_encoding.py:243, NUM_MOVES =
# 64 * 73 = 4672) - PAS un hyperparametre ajustable, une propriete fixe du schema
# move_to_index existant (AD-18 herite) : index = from_square*73 + move_type. Nommee
# ici plutot qu'un litteral "73" duplique dans ChessTokenCandidateModel ci-dessous.
NUM_MOVE_TYPES = 73


class ChessTokenCandidateModel(nn.Module):
    """
    Modèle chess_token (spec-chess-token-candidate-model, 2026-08-13, spike) : tronc
    auto-attention PURE sur les 64 tokens-case (remplace le tronc CNN de
    ChessCnnAttentionPolicyValue/ChessCnnAttentionLegalMoves ci-dessus), même
    bottleneck (cross-attention K tokens appris + auto-attention, K=8/token_dim=64
    par défaut - copié tel quel, convention "copy the class, change the head" déjà en
    place dans ce fichier, cf. docstring ChessCnnAttentionLegalMoves), mais tête de
    sortie qui SCORE 50 coups candidats fournis par le dataset au lieu d'une
    Dense(4672) fixe (CAP-1/CAP-5 du SPEC - la Dense(4672) actuelle porte 78% des
    paramètres du modèle policy+value existant, non masquée à l'entraînement).

    Input (x) : UN SEUL tenseur (B, 64+6+num_candidates) - contrainte de plomberie
    réelle, pas un choix arbitraire (vérifié en lisant trainer.py avant d'implémenter,
    voir _bmad-output/implementation-artifacts/spec-chess-token-candidate-model.md,
    section input_shape) : Trainer._train_step/_eval_step appellent toujours
    `apply_fn(vars, images, ...)` avec `images` = UN SEUL `jnp.array(images_np, ...)`
    (trainer.py:312-313/429-430) - jamais plusieurs entrées nommées. Les 3 champs du
    contrat .npz que le MODÈLE doit consommer (token_position (N,64), global_flags
    (N,6), candidate_moves (N,50)) sont donc concaténés en un seul vecteur par
    ChessTokenCandidateDataset (data_management.py) et RE-découpés ici, dans l'ordre
    exact [token_position(64) | global_flags(6) | candidate_moves(num_candidates)].
    candidate_mask (N,50) N'EST PAS packé dans cette entrée - il ne sert qu'au masquage
    de la loss/métriques (ChessTokenStrategy), jamais à un calcul interne du modèle
    (les slots de padding sont neutralisés ici via `safe_moves`, voir plus bas), donc
    il voyage côté "targets" (pytree, `jax.tree_util.tree_map` le laisse passer tel
    quel, trainer.py:314/431), pas côté "images".

    x peut arriver en float32 (dummy d'initialisation de Trainer.create_train_state,
    trainer.py:145, jamais modifié - zero-touch AD-33) : casté explicitement en int32
    dès la première ligne, même pattern que ChessMoveTokenTransformer ci-dessus
    (model_library.py:1142-1147).

    global_flags (trait/roques/répétition, binaires 0/1 - voir companion schema
    token-candidate-dataset-schema.md) : gap non couvert littéralement par le SPEC
    (Design Notes ne mentionnent que token_position/candidate_moves) - décision prise
    ici (documentée dans le rapport d'implémentation) : projeté en un biais de
    dimension token_dim (nn.Dense) et AJOUTÉ (broadcast) à chacun des 64 tokens-case
    AVANT le tronc auto-attention, plutôt que traité comme un 65e token séparé - garde
    le tronc à exactement 64 tokens (lecture littérale de "auto-attention pure sur les
    64 tokens-case" du SPEC), tout en laissant l'auto-attention du tronc propager ce
    contexte global (droits de roque, répétition) dans chaque représentation de case.

    Représentation d'un coup candidat DÉCOMPOSÉE (jamais nn.Embed(NUM_MOVES=4672, -),
    voir Design Notes du SPEC et .memlog.md - CORRECTIF Winston explicite, une table
    sur 4672 réintroduirait exactement la structure que CAP-5 cherche à éliminer) :
    from_square = index // 73 RÉUTILISE la même table d'embedding positionnel `pos_embed`
    (64, token_dim) déjà utilisée pour le tronc (poids partagés, 0 paramètre
    supplémentaire - même instance de sous-module flax appelée deux fois, pas deux
    tables indépendantes) ; move_type = index % 73 utilise une petite table dédiée
    nn.Embed(73, 32), indépendante de 4672. Slots de padding (candidate_moves == -1,
    candidate_mask == 0 côté dataset) : jamais décodés tels quels (index négatif) -
    substitués par 0 via `safe_moves` avant division/modulo (indices d'embedding
    toujours valides), leur score est de toute façon exclu par le masquage additif
    -1e9 côté loss (compute_chess_token_candidate_loss, loss_functions.py) - aucun
    impact sur la loss ni le gradient.

    Sortie : (B, num_candidates) logits bruts - UN SEUL tenseur (pas de tête value,
    même famille que ChessCnnAttentionLegalMoves/ChessMoveTokenTransformer). Chaque
    slot est scoré par produit scalaire (mis à l'échelle) entre sa représentation
    projetée et la représentation de position poolée (moyenne des K tokens du
    bottleneck) - style attention à une seule tête, peu de paramètres additionnels
    (cohérent avec l'objectif CAP-5 de réduction de paramètres).

    num_candidates : taille de la tête de scoring (50, MAX_CANDIDATES du contrat .npz
    côté chess_ai) - jamais un littéral dupliqué ici, passe par le kwarg générique
    `num_classes` (main.py) via create_chess_token_candidate_model ci-dessous, même
    convention que num_moves sur ChessCnnAttentionPolicyValue (le nom interne du champ
    reste `num_candidates` pour la clarté, la factory fait le pont).

    token_dim/num_bottleneck_tokens/num_heads : mêmes défauts que
    ChessCnnAttentionPolicyValue (K=8/D=64) - décision explicite d'Aymeric (SPEC,
    2026-08-13) de repartir sur la config par défaut plutôt que de chercher une
    comparabilité stricte avec le checkpoint 28,00% (K=16/token_dim=196). D=64
    également utilisé pour nn.Embed(13,64)/nn.Embed(64,64) (alimentent le tronc SANS
    projection intermédiaire, décision Winston/Aymeric du memlog).

    num_trunk_layers : nombre de blocs auto-attention du NOUVEAU tronc (remplace les 5
    convs 3x3 de ChessCnnAttentionPolicyValue) - hyperparamètre de départ non tuné
    (comme token_dim/num_bottleneck_tokens ci-dessus), valeur de départ 2 (léger,
    laisse le bottleneck faire le gros du travail de compression K=8) - ajustable via
    la factory, pas une contrainte fixée par le SPEC (silencieux sur ce point).
    """
    num_candidates: int
    dropout_rate: float = 0.1
    token_dim: int = 64
    num_bottleneck_tokens: int = 8
    num_heads: int = 4
    num_trunk_layers: int = 2

    @nn.compact
    def __call__(self, x, training: bool = True):
        # Verifie AVANT le modulo ci-dessous : num_heads<=0 (ex. 0 passe par erreur via
        # **kwargs, main.py) provoquerait sinon un ZeroDivisionError peu clair (ou un
        # resultat de modulo trompeur pour un num_heads negatif) plutot qu'un message
        # explicite.
        assert self.num_heads > 0, f"num_heads doit etre > 0 (recu {self.num_heads})"
        assert self.token_dim % self.num_heads == 0, (
            f"token_dim ({self.token_dim}) doit etre divisible par num_heads ({self.num_heads})"
        )
        assert self.num_candidates > 0, (
            f"num_candidates doit etre > 0 (recu {self.num_candidates}) - as-tu bien passe "
            f"num_classes=50 (MAX_CANDIDATES, contrat .npz chess_token_candidate_spike) plutot "
            f"qu'un defaut generique de ce fichier (ex. num_classes=2) ?"
        )

        # AD-33 (pattern ChessMoveTokenTransformer, model_library.py:1142-1147) : caste
        # explicitement en int32, independamment du dtype fourni par l'appelant (dummy
        # d'init float32 de Trainer.create_train_state, trainer.py:145 - jamais modifie).
        packed = jnp.asarray(x, dtype=jnp.int32)

        expected_width = 64 + 6 + self.num_candidates
        assert packed.shape[-1] == expected_width, (
            f"attendu une entree packee (B, 64+6+num_candidates={expected_width}) "
            f"[token_position(64) | global_flags(6) | candidate_moves({self.num_candidates})] "
            f"(ChessTokenCandidateDataset, data_management.py), recu (B, {packed.shape[-1]})"
        )

        # --- DÉCOUPAGE de l'entrée packée (voir docstring - un seul tenseur d'entree,
        # contrainte reelle de Trainer._train_step/_eval_step) ---
        token_position = packed[:, :64]                                    # (B, 64) int32 in [0,12]
        global_flags = packed[:, 64:70].astype(jnp.float32)                # (B, 6) 0.0/1.0
        candidate_moves = packed[:, 70:70 + self.num_candidates]           # (B, num_candidates) int32 in [-1,4671]

        # --- EMBEDDINGS token + position, alimentent le tronc SANS projection (SPEC) ---
        token_embed = nn.Embed(num_embeddings=13, features=self.token_dim, name="token_embed")
        pos_embed = nn.Embed(num_embeddings=64, features=self.token_dim, name="pos_embed")

        board_tokens = token_embed(token_position)  # (B, 64, D)
        position_ids = jnp.arange(64)
        board_tokens = board_tokens + pos_embed(position_ids)[None, :, :]  # (B, 64, D)

        # global_flags (trait/roques/repetition) : biais additif diffuse a chaque case
        # AVANT le tronc (voir docstring - decision non couverte litteralement par le
        # SPEC, tronc garde exactement 64 tokens).
        global_bias = nn.Dense(self.token_dim, name="global_flags_proj")(global_flags)  # (B, D)
        h = board_tokens + global_bias[:, None, :]  # (B, 64, D)

        # --- TRONC : auto-attention PURE sur les 64 tokens-case (remplace le tronc CNN
        # de ChessCnnAttentionPolicyValue/ChessCnnAttentionLegalMoves) ---
        for _ in range(self.num_trunk_layers):
            residual = h
            h = nn.LayerNorm()(h)
            h = nn.MultiHeadDotProductAttention(num_heads=self.num_heads)(inputs_q=h)  # bidirectionnel, pas de masque
            h = nn.Dropout(self.dropout_rate, deterministic=not training)(h)
            h = residual + h

            residual = h
            h = nn.LayerNorm()(h)
            h = nn.Dense(4 * self.token_dim)(h)
            h = nn.gelu(h)
            h = nn.Dense(self.token_dim)(h)
            h = nn.Dropout(self.dropout_rate, deterministic=not training)(h)
            h = residual + h

        h = nn.LayerNorm()(h)
        board_tokens = h  # (B, 64, D)

        # --- BOTTLENECK : copie EXACTE du pattern ChessCnnAttentionPolicyValue/
        # ChessCnnAttentionLegalMoves (K tokens appris, cross-attention puis
        # auto-attention) - convention "copy the class" de ce fichier, pas de mixin. ---
        batch_size = board_tokens.shape[0]
        queries = self.param(
            "bottleneck_queries",
            nn.initializers.normal(0.02),
            (self.num_bottleneck_tokens, self.token_dim),
        )
        queries = jnp.broadcast_to(
            queries[None, :, :], (batch_size, self.num_bottleneck_tokens, self.token_dim)
        )

        tokens = nn.MultiHeadDotProductAttention(num_heads=self.num_heads)(
            inputs_q=queries, inputs_kv=board_tokens
        )  # (B, K, D)
        tokens = nn.MultiHeadDotProductAttention(num_heads=self.num_heads)(inputs_q=tokens)  # (B, K, D)

        pooled = jnp.mean(tokens, axis=1)  # (B, D)
        pooled = nn.Dropout(self.dropout_rate, deterministic=not training)(pooled)

        # --- TÊTE DE SCORING CANDIDATS (remplace Dense(num_moves)) ---
        # Slots de padding (candidate_moves == -1) neutralises AVANT toute
        # division/modulo - jamais un index d'embedding negatif (voir docstring, sans
        # impact sur la loss : ces slots sont exclus par candidate_mask cote
        # compute_chess_token_candidate_loss). Borne haute egalement gardee (pas
        # seulement >= 0) : le contrat reel de candidate_moves est [0,4672) (NUM_MOVES=
        # 64*73, chess_ai/chess_target_encoding.py:243), PAS [0,4660] (4660 n'etait que
        # le max observe sur le seul run de generation reel disponible, pas une borne
        # contractuelle) - une valeur hors bornes serait sinon divisee/modulee par
        # NUM_MOVE_TYPES avec un resultat non defini par le contrat plutot que neutralisee.
        safe_moves = jnp.where((candidate_moves >= 0) & (candidate_moves < 4672), candidate_moves, 0)  # (B, num_candidates)
        from_square = safe_moves // NUM_MOVE_TYPES  # (B, num_candidates) in [0, 64)
        move_type = safe_moves % NUM_MOVE_TYPES     # (B, num_candidates) in [0, 73)

        # from_square : REUTILISE pos_embed (meme instance de sous-module que ci-dessus,
        # poids partages - AUCUNE nouvelle table sur 4672, decision explicite du SPEC).
        from_square_repr = pos_embed(from_square)  # (B, num_candidates, D)
        move_type_embed = nn.Embed(num_embeddings=NUM_MOVE_TYPES, features=32, name="move_type_embed")
        move_type_repr = move_type_embed(move_type)  # (B, num_candidates, 32)

        candidate_repr = jnp.concatenate([from_square_repr, move_type_repr], axis=-1)  # (B, num_candidates, D+32)
        candidate_proj = nn.Dense(self.token_dim, name="candidate_proj")(candidate_repr)  # (B, num_candidates, D)

        # Score = produit scalaire mis a l'echelle entre chaque candidat projete et la
        # representation de position poolee - style attention a une seule tete, peu de
        # parametres additionnels (candidate_proj + move_type_embed + global_flags_proj,
        # tous largement < Dense(4672) qu'ils remplacent, cf. CAP-5).
        scale = jnp.sqrt(jnp.asarray(self.token_dim, dtype=jnp.float32))
        logits = jnp.einsum("bsd,bd->bs", candidate_proj, pooled) / scale  # (B, num_candidates)

        return logits.astype(jnp.float32)


def create_chess_token_candidate_model(num_classes, dropout_rate=0.1, **kwargs):
    """
    Factory pour ChessTokenCandidateModel. `num_classes` (nom impose par la plomberie
    model_kwargs generique de main.py) porte en realite MAX_CANDIDATES (50, contrat
    .npz chess_token_candidate_spike cote chess_ai) - PAS un nombre de classes au sens
    habituel, meme convention que create_chess_cnn_attention_policy_value ci-dessus
    (num_moves=4672) et create_chess_cnn_attention_legal_moves.

    **kwargs transmis tel quel a ChessTokenCandidateModel - permet de surcharger
    token_dim/num_bottleneck_tokens/num_heads/num_trunk_layers sans modifier cette
    factory (meme discipline que les factories chess_cnn_attention_* ci-dessus).
    """
    return ChessTokenCandidateModel(num_candidates=num_classes, dropout_rate=dropout_rate, **kwargs)


class ChessTokenOneMoveModel(nn.Module):
    """
    Modèle chess_token_1_move (spec-chess-token-1-move-v2, 2026-08-16, spike, contrat
    chess_ai §3 CHESS_TOKEN_1_MOVE "Piste v2") : tronc auto-attention PURE sur les 64
    tokens-case, copié PARTIELLEMENT de ChessTokenCandidateModel ci-dessus (mêmes
    embeddings token_embed/pos_embed, même biais global_flags additif, mêmes blocs
    auto-attention) mais SANS son étage bottleneck (pas de queries apprises
    `bottleneck_queries`, pas de cross-/auto-attention vers K tokens latents, ligne
    1393-1411 ci-dessus délibérément NON reprise - contrainte explicite du contrat,
    "Never" du SPEC v1 ET v2) - le tronc alimente directement 2 têtes de sortie.

    v1 (spec-chess-token-1-move, 2026-08-15) éliminée : plateau `JointMoveAccuracy` à
    5,10% dès l'epoch ~34/50, train≈val (plafond structurel, pas overfitting).
    Diagnostic (contrat §3, "Résultat 2026-08-15") : suspect principal = la contrainte
    v1 "2 têtes prédites indépendamment, sans conditionnement" (SUPERSEDED, voir
    spec-chess-token-1-move.md), qui forçait le modèle à marginaliser sur l'identité de
    la pièce en case de départ pour prédire move_type, alors que ce dernier en dépend
    presque entièrement aux échecs. v2 (spec-chess-token-1-move-v2, 2026-08-16) combine
    3 leviers en UN SEUL run (pas 3 runs isolés) : (1) conditionnement léger de
    move_type_head sur from_square (CI-DESSOUS), (2) read_token appris remplaçant le
    mean-pool (CI-DESSOUS), (3) tronc élargi (dataset_configs.py : token_dim 32->64,
    num_trunk_layers 1->2 - AUCUN changement dans cette classe, déjà en kwargs).

    Pourquoi pas de bottleneck ici : CAP-1/CAP-5 côté chess-token-candidate-poc avaient
    déjà motivé le passage à un tronc auto-attention pur (pas de CNN), mais leur tête de
    scoring à 50 candidats pré-filtrés par python-chess ne peut, par construction, JAMAIS
    produire de coup illégal - ce qui empêche de mesurer si le réseau "connaît" les
    règles (diagnostic chess_ai, contrat §3). Ce modèle prédit directement le coup dans
    l'espace complet (from_square 64 x move_type 73 = 4672), sans filtrage préalable :
    l'illégalité redevient un échec mesurable (aucun masquage de légalité côté
    modèle/loss - "Never" du SPEC).

    Input (x) : UN SEUL tenseur (B, 71) = [token_position(64) | global_flags(6) |
    from_square_teacher(1)] (v2 - 70 en v1, +1 colonne) - contrainte de plomberie réelle
    héritée de ChessTokenCandidateModel (Trainer._train_step/_eval_step n'acheminent
    jamais qu'UNE SEULE entrée `images` vers `apply_fn`, voir sa docstring pour la
    référence trainer.py exacte) - packé par ChessTokenOneMoveDataset
    (data_management.py). PAS de candidate_moves ici (contrairement à
    ChessTokenCandidateModel) : ce modèle ne voit jamais les 50 candidats légaux du
    dataset, seulement la position brute (+ le from_square RÉEL en 71e colonne, réservé
    au conditionnement teacher-forcing ci-dessous) - c'est précisément ce qui rend
    l'illégalité mesurable (voir ci-dessus).

    x peut arriver en float32 (dummy d'initialisation de Trainer.create_train_state,
    trainer.py:145, jamais modifié - AD-33) : casté explicitement en int32 dès la
    première ligne, même pattern que ChessTokenCandidateModel/ChessMoveTokenTransformer
    ci-dessus.

    global_flags (trait/roques/répétition) : même traitement que ChessTokenCandidateModel
    (biais additif de dimension token_dim, projeté par nn.Dense puis ajouté à chacun des
    64 tokens-case AVANT le tronc) - même décision, même motivation (garder le tronc à
    exactement 64 tokens-case, laisser l'auto-attention propager ce contexte global).

    token_embed = nn.Embed(13, token_dim, name="token_embed") / pos_embed =
    nn.Embed(64, token_dim, name="pos_embed") : NOMS EXACTS imposés par le contrat
    (CAP-6 côté chess_ai cible ces noms de paramètres pour la mutation génétique de
    chess_bottleneck_genetic.py) - ne jamais renommer ni remplacer par une autre forme
    de couche. pos_embed est désormais RÉUTILISÉ (v2, poids partagés, 0 paramètre
    supplémentaire - même instance de sous-module flax appelée deux fois, exactement le
    pattern déjà en place sur ChessTokenCandidateModel pour from_square des candidats,
    voir sa docstring) pour embedder le from_square de conditionnement ci-dessous - en
    v1 pos_embed ne servait QUE le tronc, il n'y avait rien d'autre à encoder.

    read_token (v2, nouveau `self.param`, PAS un nn.Embed) : un unique vecteur appris de
    dimension token_dim, inséré comme 65e token de la séquence auto-attention AVANT le
    tronc (concaténé à board_tokens, PAS un étage de cross-attention séparé - reste
    conforme à la contrainte "pas de bottleneck" ci-dessus, contrat §3 "Piste v2" option
    2). Après le tronc, sa représentation en sortie (h[:, 64, :], PAS h[:, :64].mean())
    remplace le mean-pool v1 comme représentation poolée de la position - laisse
    l'auto-attention "choisir" quoi lire dans les 64 tokens-case plutôt qu'une moyenne
    uniforme fixe.

    Conditionnement move_type_head sur from_square (v2, contrat §3 "Piste v2" option 1) :
    from_square_logits reste calculé UNIQUEMENT depuis `pooled` (NON conditionné - un
    coup illégal doit rester possible, objectif contractuel "illégalité mesurable"
    préservé). move_type_head, elle, reçoit en entrée la concaténation [pooled ;
    pos_embed(from_square_cond)] où from_square_cond = from_square_teacher (colonne 71,
    label RÉEL, teacher-forcing) quand training=True, et argmax(from_square_logits)
    quand training=False - JAMAIS le label réel utilisé en évaluation (fuite qui
    invaliderait le diagnostic "illégalité mesurable" du contrat, I/O Matrix du SPEC v2).
    Branche Python-level `if training:` (PAS jnp.where) : `training` arrive ici comme un
    bool Python STATIQUE, jamais une valeur tracée JAX - vérifié en lisant trainer.py
    avant d'implémenter (Trainer._train_step/_eval_step appellent toujours
    `apply_fn(vars, images, training=True/False, ...)` avec un littéral Python, jamais un
    argument passé à travers `jax.jit` - trainer.py:239-240/272), cohérent avec
    `deterministic=not training` déjà utilisé pour le dropout dans cette même classe.

    Sortie : dict {"from_square": (B,64), "move_type": (B,73)} - PAS un tenseur unique
    (contrairement à ChessTokenCandidateModel/ChessCnnAttentionLegalMoves/
    ChessMoveTokenTransformer) : 2 têtes Dense sur `pooled` (from_square) et
    [pooled ; embedding(from_square_cond)] (move_type, v2 - conditionnement à SENS
    UNIQUE, from_square ne dépend jamais de move_type). from_square/move_type sont des
    tailles FIXES (64/NUM_MOVE_TYPES=73, propriété du schéma move_to_index existant, PAS
    un hyperparamètre) - jamais paramétrées via num_classes (voir
    create_chess_token_one_move_model ci-dessous, Design Notes du SPEC).

    token_dim/num_heads/num_trunk_layers : tronc v1 délibérément ALLÉGÉ (valeurs de
    départ plus petites que les défauts token_dim=64/num_trunk_layers=2 de
    ChessTokenCandidateModel) - objectif explicite du contrat (§3) : que
    token_embed/pos_embed pèsent une part SIGNIFICATIVE des paramètres totaux (pour que
    chess_bottleneck_genetic.py, côté chess_ai, ait un espace de mutation avec un impact
    comportemental mesurable - 5 tentatives précédentes négatives sur un sous-espace
    mutable trop petit, chess_ai/docs/chess_ai_global_conclusions.md §4). Défauts de
    CETTE CLASSE inchangés (token_dim=32, num_heads=4, num_trunk_layers=1) : le tronc
    élargi de v2 (token_dim=64, num_trunk_layers=2) est appliqué UNIQUEMENT via les
    kwargs de dataset_configs.py (entrée CHESS_TOKEN_1_MOVE), pas un changement de
    défaut ici - aucune autre classe/usage de ChessTokenOneMoveModel n'est affecté.
    """
    dropout_rate: float = 0.1
    token_dim: int = 32
    num_heads: int = 4
    num_trunk_layers: int = 1

    @nn.compact
    def __call__(self, x, training: bool = True):
        assert self.num_heads > 0, f"num_heads doit etre > 0 (recu {self.num_heads})"
        assert self.token_dim % self.num_heads == 0, (
            f"token_dim ({self.token_dim}) doit etre divisible par num_heads ({self.num_heads})"
        )

        # AD-33 (pattern ChessTokenCandidateModel/ChessMoveTokenTransformer ci-dessus) :
        # caste explicitement en int32, independamment du dtype fourni par l'appelant
        # (dummy d'init float32 de Trainer.create_train_state, trainer.py:145 - jamais
        # modifie).
        packed = jnp.asarray(x, dtype=jnp.int32)

        expected_width = 71  # v2 : 70 (v1) + 1 (from_square_teacher)
        assert packed.shape[-1] == expected_width, (
            f"attendu une entree packee (B, 71) [token_position(64) | global_flags(6) | "
            f"from_square_teacher(1)] (ChessTokenOneMoveDataset, data_management.py), "
            f"recu (B, {packed.shape[-1]})"
        )

        # --- DÉCOUPAGE de l'entrée packée ---
        token_position = packed[:, :64]                      # (B, 64) int32 in [0,12]
        global_flags = packed[:, 64:70].astype(jnp.float32)  # (B, 6) 0.0/1.0
        from_square_teacher = packed[:, 70]                  # (B,) int32 in [0,64) - v2, teacher-forcing UNIQUEMENT (voir plus bas)

        batch_size = packed.shape[0]

        # --- EMBEDDINGS token + position, alimentent le tronc SANS projection (noms
        # EXACTS imposes par le contrat, voir docstring) ---
        token_embed = nn.Embed(num_embeddings=13, features=self.token_dim, name="token_embed")
        pos_embed = nn.Embed(num_embeddings=64, features=self.token_dim, name="pos_embed")

        board_tokens = token_embed(token_position)  # (B, 64, D)
        position_ids = jnp.arange(64)
        board_tokens = board_tokens + pos_embed(position_ids)[None, :, :]  # (B, 64, D)

        # global_flags (trait/roques/repetition) : biais additif diffuse a chaque case
        # AVANT le tronc - meme decision que ChessTokenCandidateModel ci-dessus.
        global_bias = nn.Dense(self.token_dim, name="global_flags_proj")(global_flags)  # (B, D)
        h = board_tokens + global_bias[:, None, :]  # (B, 64, D)

        # --- read_token (v2, voir docstring) : 65e token de la sequence, AVANT le
        # tronc - PAS de position_ids/pos_embed pour ce token (ce n'est pas une case du
        # plateau), juste un vecteur appris directement, broadcast sur le batch. ---
        read_token = self.param(
            "read_token", nn.initializers.normal(0.02), (self.token_dim,)
        )
        read_token = jnp.broadcast_to(read_token[None, None, :], (batch_size, 1, self.token_dim))
        h = jnp.concatenate([h, read_token], axis=1)  # (B, 65, D)

        # --- TRONC : auto-attention PURE sur les 65 tokens (64 cases + read_token,
        # copie du pattern ChessTokenCandidateModel ci-dessus, SANS le bottleneck qui
        # suit normalement) - le read_token participe a l'auto-attention comme n'importe
        # quel autre token, bidirectionnel (pas de masque causal). ---
        for _ in range(self.num_trunk_layers):
            residual = h
            h = nn.LayerNorm()(h)
            h = nn.MultiHeadDotProductAttention(num_heads=self.num_heads)(inputs_q=h)  # bidirectionnel, pas de masque
            h = nn.Dropout(self.dropout_rate, deterministic=not training)(h)
            h = residual + h

            residual = h
            h = nn.LayerNorm()(h)
            h = nn.Dense(4 * self.token_dim)(h)
            h = nn.gelu(h)
            h = nn.Dense(self.token_dim)(h)
            h = nn.Dropout(self.dropout_rate, deterministic=not training)(h)
            h = residual + h

        h = nn.LayerNorm()(h)  # (B, 65, D)

        # --- PAS DE BOTTLENECK ICI (contrainte du contrat, voir docstring classe) :
        # la representation en sortie du read_token (dernier token, index 64) remplace
        # le mean-pool v1 - laisse l'auto-attention "choisir" quoi lire dans les 64
        # tokens-case plutot qu'une moyenne uniforme fixe (v2). ---
        pooled = h[:, 64, :]  # (B, D) - sortie du read_token, PAS jnp.mean(h[:, :64], axis=1)
        pooled = nn.Dropout(self.dropout_rate, deterministic=not training)(pooled)

        # --- TETE from_square : NON CONDITIONNEE, calculee UNIQUEMENT depuis `pooled`
        # (contrat v2 "Always" : from_square doit rester libre, un coup illegal doit
        # rester possible - objectif "illegalite mesurable" preserve). ---
        from_square_logits = nn.Dense(64, name="from_square_head")(pooled)  # (B, 64)

        # --- Conditionnement move_type_head sur from_square (v2, voir docstring) :
        # teacher-forcing (label REEL, colonne 71 packee) UNIQUEMENT quand training=True ;
        # argmax(from_square_logits) quand training=False - JAMAIS le label reel en
        # evaluation (anti-fuite, I/O Matrix du SPEC v2). Branche Python-level (`if
        # training:`), PAS jnp.where : `training` est un bool Python STATIQUE ici (voir
        # docstring classe pour la verification trainer.py), donc un `if` classique
        # trace une seule des deux branches, sans cout ni ambiguite a l'execution. ---
        if training:
            from_square_cond = from_square_teacher  # (B,) int32 in [0,64) - label reel (teacher-forcing)
        else:
            from_square_cond = jnp.argmax(from_square_logits, axis=-1)  # (B,) int32 in [0,64) - jamais le label

        # pos_embed REUTILISE (meme instance que ci-dessus, poids partages - AUCUNE
        # nouvelle table sur 64, meme pattern que ChessTokenCandidateModel pour
        # from_square des candidats).
        cond_embed = pos_embed(from_square_cond)  # (B, D)
        move_type_input = jnp.concatenate([pooled, cond_embed], axis=-1)  # (B, 2*D)
        move_type_logits = nn.Dense(NUM_MOVE_TYPES, name="move_type_head")(move_type_input)  # (B, 73)

        return {
            "from_square": from_square_logits.astype(jnp.float32),
            "move_type": move_type_logits.astype(jnp.float32),
        }


def create_chess_token_one_move_model(num_classes=None, dropout_rate=0.1, **kwargs):
    """
    Factory pour ChessTokenOneMoveModel. `num_classes` (nom imposé par la plomberie
    model_kwargs générique de main.py - main.py construit toujours
    model_kwargs={"num_classes": ..., "dropout_rate": ...} même pour ce modèle) est
    IGNORÉ ICI, délibérément : ce modèle a 2 têtes de tailles FIXES (from_square=64,
    move_type=NUM_MOVE_TYPES=73, codées en dur dans ChessTokenOneMoveModel, propriété du
    schéma move_to_index existant) - "un nombre de classes" unique n'a pas de sens
    direct ici, contrairement à create_chess_token_candidate_model (num_classes=50=
    MAX_CANDIDATES) ou create_chess_move_token_transformer (num_classes=4672=NUM_MOVES)
    ci-dessus (Design Notes du SPEC). Convention choisie (dataset_configs.py, entrée
    CHESS_TOKEN_1_MOVE) : num_classes porte une valeur SENTINELLE documentée dans la
    config (jamais consommée ici, jamais transmise à ChessTokenOneMoveModel).

    **kwargs transmis tel quel à ChessTokenOneMoveModel - permet de surcharger
    token_dim/num_heads/num_trunk_layers sans modifier cette factory, même discipline
    que create_chess_token_candidate_model ci-dessus.
    """
    return ChessTokenOneMoveModel(dropout_rate=dropout_rate, **kwargs)


class Kepler1DConvNet(nn.Module):
    """
    Réseau de Neurones Convolutif 1D pour l'analyse de Séries Temporelles.
    Spécialement conçu pour détecter les creux de luminosité (transit) dans les données Kepler.
    """
    num_classes: int = 2
    dropout_rate: float = 0.3
    # compute_dtype (spec-compute-dtype-rollout-classification, 2026-08-17, AD-1/AD-4/
    # AD-6) : dtype de CALCUL des nn.Conv/nn.Dense uniquement - pas de nn.BatchNorm/
    # nn.LayerNorm dans cette classe (verifie, rien a exclure ici, AD-4 n'a donc rien
    # a exclure). Derive du materiel par main.py (bfloat16 sur TPU, float32 sinon),
    # injecte par introspection de signature. Poids stockes (checkpoint) restent
    # float32 dans tous les cas (AD-6) - champ dataclass statique, jamais un nn.Param.
    compute_dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, x, training: bool):
        # x est de shape (Batch, SequenceLength, 1) -> ex: (B, 3197, 1)

        # Bloc 1 (Détection de motifs locaux)
        x = nn.Conv(features=32, kernel_size=(11,), padding='SAME', dtype=self.compute_dtype)(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2,), strides=(2,))

        # Bloc 2 (Extraction de features temporelles)
        x = nn.Conv(features=64, kernel_size=(5,), padding='SAME', dtype=self.compute_dtype)(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2,), strides=(2,))

        # Bloc 3
        x = nn.Conv(features=128, kernel_size=(5,), padding='SAME', dtype=self.compute_dtype)(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2,), strides=(2,))

        # Bloc 4
        x = nn.Conv(features=256, kernel_size=(3,), padding='SAME', dtype=self.compute_dtype)(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2,), strides=(2,))

        # Global Average Pooling 1D (On moyenne sur le temps restant)
        x = jnp.mean(x, axis=1) # (Batch, 256)

        # Classification Head
        x = nn.Dense(features=64, dtype=self.compute_dtype)(x)
        x = nn.relu(x)
        x = nn.Dropout(self.dropout_rate, deterministic=not training)(x)
        x = nn.Dense(features=self.num_classes, dtype=self.compute_dtype)(x)

        # Recast explicite en float32 avant retour, quel que soit compute_dtype - meme
        # garde que SophisticatedCNN128Plus/Lite/32Plus/ChessMoveTokenTransformer,
        # KeplerStrategy (task_strategies.py:375) utilise cross_entropy par defaut
        # comme ClassificationStrategy (meme risque de precision reduite dans la loss).
        return x.astype(jnp.float32)


def create_kepler_1d_cnn(num_classes=2, dropout_rate=0.3, compute_dtype=jnp.float32, **kwargs):
    """Crée une instance de Kepler1DConvNet.

    num_classes/dropout_rate/compute_dtype acceptes explicitement comme parametres
    NOMMES (AD-3, Story 12.1) : **kwargs seul forwarderait fonctionnellement la
    valeur, mais le canal `overrides` strict de build_kwargs_from_config
    (model_library.py) ne detecte que les parametres nommes explicites, jamais la
    simple presence d'un **kwargs catch-all - un parametre nomme est necessaire pour
    que num_classes/dropout_rate/compute_dtype soient reellement transmis depuis
    dataset_configs.py (trouve en revue de code, Story 12.1 : cette factory etait la
    seule a ne nommer ni num_classes ni dropout_rate, masque uniquement parce que les
    defauts de Kepler1DConvNet coincidaient avec la config JAX_KEPLER actuelle -
    tout futur changement de ces valeurs en config aurait ete silencieusement
    ignore). Valide compute_dtype (voir _validate_compute_dtype).
    """
    _validate_compute_dtype(compute_dtype)
    return Kepler1DConvNet(num_classes=num_classes, dropout_rate=dropout_rate, compute_dtype=compute_dtype, **kwargs)


MODELS = {
    'aircraft_detector_unet': create_aircraft_detector_unet, # Semantic Segmentation U-Net
    'aircraft_detector_centernet': create_aircraft_detector_centernet, # CenterNet (point central, anchor-free)
    'chess_cnn_attention_policy_value': create_chess_cnn_attention_policy_value, # CNN 8x8 + bottleneck attention, policy+value (Epic 9)
    'chess_cnn_attention_legal_moves': create_chess_cnn_attention_legal_moves, # meme backbone, sans tete value, multi-label coups legaux
    'chess_move_token_transformer': create_chess_move_token_transformer, # decodeur causal sur move_tokens, policy-only (Epic 11, spike)
    'chess_token_candidate_model': create_chess_token_candidate_model, # tronc auto-attention pure + tete scoring 50 candidats (spec-chess-token-candidate-model, spike)
    'chess_token_one_move_model': create_chess_token_one_move_model, # tronc auto-attention pure SANS bottleneck + 2 tetes independantes from_square/move_type (spec-chess-token-1-move, spike)
    'sophisticated_cnn_128_plus': create_sophisticated_cnn_128_plus,
    'sophisticated_cnn_128_lite': create_sophisticated_cnn_128_lite,
    'sophisticated_cnn_32_plus': create_sophisticated_cnn_32_plus,
    'kepler_1d_cnn': create_kepler_1d_cnn,
}


# Cles de dataset_configs.py forwardees sans condition vers model_kwargs quand
# presentes, par model_name (Story 12.1, AD-21 - revue de code : evite qu'une
# liste plate cross-modeles se retrouve dupliquee entre main.py et les tests,
# et scope chaque cle au(x) seul(s) modele(s) qui la consomment reellement, au
# lieu d'un simple commentaire pour le documenter). Source unique, importee
# telle quelle par main.py (build_kwargs_from_config) et par les tests -
# jamais redefinie independamment. Absent d'un model_name : aucune cle
# supplementaire forwardee (get() avec defaut () cote appelant).
MODEL_FORWARDED_CONFIG_KEYS = {
    'aircraft_detector_centernet': ('heatmap_prior',),
    'chess_cnn_attention_policy_value': ('num_bottleneck_tokens', 'token_dim'),
    'chess_cnn_attention_legal_moves': ('num_bottleneck_tokens', 'token_dim'),
    'chess_move_token_transformer': ('num_layers', 'd_model', 'num_heads'),
    'chess_token_candidate_model': ('token_dim', 'num_heads', 'num_trunk_layers'),
    'chess_token_one_move_model': ('token_dim', 'num_heads', 'num_trunk_layers'),
}


def resolve_compute_dtype(backend):
    """
    Derive compute_dtype (dtype de CALCUL des couches matmul lourdes, AD-1/AD-2,
    spec-compute-dtype-hardware 2026-08-17) depuis le backend JAX detecte -
    bfloat16 sur TPU, float32 partout ailleurs (GPU inclus - prudence de projet
    documentee, pas une limite materielle, voir SPEC.md Constraints).

    Extrait de main.py dans une fonction pure importable/testable (revue Blind
    Hunter, 2026-08-17 : la ligne qui fait tout le travail de cette feature
    n'etait exercee par aucun test, main.py ayant des effets de bord au chargement
    du module - hardware init, prints - qui rendent son import direct depuis les
    tests indesirable).
    """
    return jnp.bfloat16 if backend == "tpu" else jnp.float32


def build_kwargs_from_config(target, config, config_keys=(), **overrides):
    """
    Construit les kwargs d'une factory/classe (Story 12.1, AD-21) selon deux
    canaux distincts, jamais confondus :

    1. `config_keys` (explicite, fourni par l'appelant) : chaque nom present
       dans `config` est forwarde SANS CONDITION - meme comportement que les
       branches `if "X" in config: kwargs["X"] = config["X"]` qu'il remplace
       (main.py, avant migration). Pas d'introspection de signature ici : le
       contrat "cette cle de config est destinee a ce modele" reste, comme
       avant, la responsabilite de l'auteur de dataset_configs.py - jamais
       "tout `config`" (la plupart des cles d'un DATASET_CONFIGS - task_type,
       optimizer, tpu/gpu, loss_params... - ne sont pas des kwargs modele et
       feraient planter un nn.Module qui n'a pas son propre `**kwargs`, ex.
       ChessCnnAttentionLegalMoves).
    2. `overrides` (valeurs CALCULEES par l'appelant - compute_dtype,
       dropout_rate, num_classes) : forwarde uniquement si `target` nomme le
       parametre EXPLICITEMENT dans sa signature - jamais absorbe par un
       `**kwargs` catch-all (discipline stricte heritee d'AD-3,
       architecture-compute-dtype-hardware-2026-08-17). Une cle presente a la
       fois dans `config_keys`/`config` et dans `overrides` : `overrides`
       gagne toujours, silencieusement.

    `target` peut etre None (ex. MODELS.get(model_name) sur un nom invalide) -
    retourne alors des kwargs vides plutot que de planter ici ; get_model()
    leve son ValueError explicite habituel en aval.

    Retourne (kwargs, forwarded_override_keys) : le second element (frozenset)
    liste les cles d'`overrides` reellement forwardees - reutilise par
    l'appelant pour un print diagnostique fidele (ex. compute_dtype), sans
    dupliquer independamment la verification de signature.
    """
    if target is None:
        return {}, frozenset()

    parameters = inspect.signature(target).parameters

    kwargs = {}
    for key in config_keys:
        if key in config:
            kwargs[key] = config[key]

    forwarded_override_keys = set()
    for key, value in overrides.items():
        param = parameters.get(key)
        if param is not None and param.kind is not inspect.Parameter.VAR_KEYWORD:
            kwargs[key] = value
            forwarded_override_keys.add(key)

    return kwargs, frozenset(forwarded_override_keys)


_VALID_COMPUTE_DTYPES = (jnp.float32, jnp.bfloat16, jnp.float16)


def _validate_compute_dtype(compute_dtype):
    """
    Valide qu'un compute_dtype deja resolu (jnp.dtype, PAS une string - voir
    create_chess_move_token_transformer pour le seul chemin qui accepte encore une
    string, precedent historique preserve tel quel) est l'un des dtypes de calcul
    reconnus. Erreur explicite plutot qu'un echec silencieux/confus plus loin dans
    le graphe (revue Blind Hunter, 2026-08-17 : create_sophisticated_cnn_32_plus/
    128_lite/128_plus et create_kepler_1d_cnn n'avaient aucune validation - un
    compute_dtype invalide type jnp.int32 aurait ete accepte sans erreur, produisant
    des activations en entier bien avant tout crash comprehensible).
    Appelee par toutes les factories non-chess adoptant compute_dtype - source unique
    de validation, pas dupliquee par factory.
    """
    if compute_dtype not in _VALID_COMPUTE_DTYPES:
        raise ValueError(
            f"compute_dtype={compute_dtype!r} non reconnu - attendu un de {_VALID_COMPUTE_DTYPES}"
        )


def get_model(model_name, **kwargs):
    """
    Factory function pour obtenir un modèle par nom
    
    Args:
        model_name: Nom du modèle ('sophisticated_cnn_128_plus', 'aircraft_detector_unet', 'kepler_1d_cnn')
        **kwargs: Arguments supplémentaires pour le modèle
    
    Returns:
        Instance du modèle
    """
    if model_name not in MODELS:
        raise ValueError(f"Modèle '{model_name}' non trouvé. Modèles disponibles: {list(MODELS.keys())}")
    
    return MODELS[model_name](**kwargs)


def list_available_models():
    """Retourne la liste des modèles disponibles"""
    return list(MODELS.keys())


def get_model_info(model_name):
    """
    Retourne les informations sur un modèle

    Args:
        model_name: Nom du modèle

    Returns:
        Dict avec les informations du modèle
    """
    model_info = {
        'aircraft_detector_unet': {
            'name': 'AircraftDetectorUNet',
            'description': 'U-Net pour la détection par Segmentation Sémantique',
            'params': '~1.5M',
            'size': '~6MB',
            'best_for': 'Détection pixel-perfect, ignore la gestion des ancres.'
        },
        'kepler_1d_cnn': {
            'name': 'Kepler1DConvNet',
            'description': 'Réseau Convolutif 1D profond pour détecter les motifs de baisse de lumière dans les séries temporelles stellaires.',
            'params': '~150K',
            'size': '~1MB',
            'best_for': 'Données astronomiques (séries temporelles 1D) pour la recherche d\'exoplanètes.'
        },
        'aircraft_detector_centernet': {
            'name': 'AircraftDetectorCenterNet',
            'description': 'Détection par point central (heatmap de centres + régression de taille), anchor-free, style CenterNet/CornerNet.',
            'params': '2,125,539 mesuré',
            'size': 'non mesuré ici (voir checkpoint)',
            'best_for': 'Détection multi-instances (formations serrées), remplace la segmentation U-Net (AD-9/AD-10).'
        },
        'sophisticated_cnn_128_plus': {
            'name': 'SophisticatedCNN128Plus',
            'description': 'CNN avec convolutions séparables, résiduelles, SE + Spatial Attention, pour images 128×128.',
            'params': '~4M',
            'size': 'non mesuré ici (voir checkpoint)',
            'best_for': 'Classification fine-grained (FIGHTERJET_CLASSIFICATION).'
        },
        'sophisticated_cnn_128_lite': {
            'name': 'SophisticatedCNN128Lite',
            'description': 'Variante allégée de SophisticatedCNN128Plus (Bloc 1: 96->64 canaux, Bloc 4: pic 512->384) pour réduire la latence d\'inférence, mêmes attentions SE/Spatial.',
            'params': '834,089 mesuré (num_classes=32)',
            'size': '3.2 MB',
            'best_for': 'Classification fine-grained (FIGHTERJET_CLASSIFICATION) quand la vitesse d\'inférence prime sur le dernier point d\'accuracy. Validé 2026-07-26 : 0.9451 val vs 0.9521 pour Plus à schedule égal (-0.7pt pour -34% params).'
        },
        'sophisticated_cnn_32_plus': {
            'name': 'SophisticatedCNN32Plus',
            'description': 'Variante réduite de SophisticatedCNN128Plus pour images 32×32 (canaux/profondeur de pooling adaptés).',
            'params': 'non mesuré ici (voir checkpoint)',
            'size': 'non mesuré ici (voir checkpoint)',
            'best_for': 'Classification sur petites images (ex. CIFAR-10).'
        },
        'chess_cnn_attention_policy_value': {
            'name': 'ChessCnnAttentionPolicyValue',
            'description': 'CNN 8×8 sans pooling + bottleneck de tokens appris (Perceiver-style, cross-attention) + auto-attention + têtes policy/value (FR5, AD-23, Epic 9).',
            'params': '382,017 mesuré (num_classes=4672, token_dim=64, K=8)',
            'size': 'non mesuré ici (voir checkpoint)',
            'best_for': 'Domaine échecs (policy+value) - preuve de généralisation du pipeline, pas encore entraîné (Story 9.2).'
        },
        'chess_cnn_attention_legal_moves': {
            'name': 'ChessCnnAttentionLegalMoves',
            'description': 'Même backbone que ChessCnnAttentionPolicyValue, sans tête value - sortie unique (B, 4672) pour prédire l\'ensemble des coups légaux (multi-label, sigmoid BCE).',
            'params': '381,952 mesuré (num_classes=4672, token_dim=64, K=8) - légèrement moins que policy_value (382,017, une tête en moins)',
            'size': 'non mesuré ici (voir checkpoint)',
            'best_for': 'Domaine échecs - test de plomberie sur une tâche multi-label, pas un objectif de qualité de jeu.'
        }
    }

    if model_name not in model_info:
        raise ValueError(f"Modèle '{model_name}' non trouvé")

    return model_info[model_name]


class TrainStateWithBatchStats(train_state.TrainState):
    """TrainState étendu pour gérer batch_stats"""
    batch_stats: Optional[flax.core.FrozenDict] = None
