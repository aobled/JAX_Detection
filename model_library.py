"""
Librairie des modèles de deep learning
Contient tous les modèles utilisés pour l'entraînement
"""

import math
from typing import Optional

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
            kernel_init=nn.initializers.kaiming_normal()
        )(x)
        # Pointwise convolution
        x = nn.Conv(
            features=self.filters,
            kernel_size=(1, 1),
            padding=self.padding,
            use_bias=self.use_bias,
            kernel_init=nn.initializers.kaiming_normal()
        )(x)
        return x


class SEBlock(nn.Module):
    """Squeeze-and-Excitation Block"""
    reduction: int = 16

    @nn.compact
    def __call__(self, x, training=True):
        B, H, W, C = x.shape
        # Global Average Pooling
        gap = jnp.mean(x, axis=(1, 2))
        # Squeeze
        se_dense1 = nn.Dense(C // self.reduction, use_bias=True)(gap)
        se_act1 = nn.silu(se_dense1)
        # Excite
        se_dense2 = nn.Dense(C, use_bias=True)(se_act1)
        se_sigmoid = nn.sigmoid(se_dense2)
        # Scale
        se_broadcast = jnp.expand_dims(jnp.expand_dims(se_sigmoid, 1), 1)
        return x * se_broadcast


class SpatialAttention(nn.Module):
    """Spatial Attention Module"""
    
    @nn.compact
    def __call__(self, x, training=True):
        # Spatial attention
        spatial_attn = nn.Conv(
            features=1,
            kernel_size=(1, 1),
            padding="SAME",
            use_bias=True,
            kernel_init=nn.initializers.kaiming_normal()
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

    @nn.compact
    def __call__(self, x, training=True):
        # Input: (B, 128, 128, C) où C=1 (grayscale) ou C=3 (RGB)
        
        # === STOCKAGE DES FEATURES POUR MULTI-SCALE FUSION ===
        multi_scale_features = []
        
        # === BLOC 0: Conv initiale (adaptatif au nombre de canaux) ===
        x = nn.Conv(64, (3, 3), padding="SAME", use_bias=False,
                   kernel_init=nn.initializers.kaiming_normal())(x)
        x = nn.BatchNorm(use_running_average=not training)(x)
        x = nn.silu(x)
        
        # === BLOC 1: 96 canaux (NOUVEAU pour 128×128) ===
        x = SeparableConv(96, (3, 3))(x, training)
        x = nn.BatchNorm(use_running_average=not training)(x)
        x = nn.silu(x)
        
        # Residual connection
        residual = nn.Conv(96, (1, 1), padding="SAME", use_bias=False,
                          kernel_init=nn.initializers.kaiming_normal())(x)
        residual = nn.BatchNorm(use_running_average=not training)(residual)
        
        x = SeparableConv(96, (3, 3))(x, training)
        x = nn.BatchNorm(use_running_average=not training)(x)
        x = x + residual
        x = nn.silu(x)
        
        # Max Pool 1: 128×128 → 64×64
        x = nn.max_pool(x, (2, 2), strides=(2, 2))
        
        # === BLOC 2: 128 canaux ===
        x = SeparableConv(128, (3, 3))(x, training)
        x = nn.BatchNorm(use_running_average=not training)(x)
        x = nn.silu(x)
        
        # Residual connection
        residual = nn.Conv(128, (1, 1), padding="SAME", use_bias=False,
                          kernel_init=nn.initializers.kaiming_normal())(x)
        residual = nn.BatchNorm(use_running_average=not training)(residual)
        
        x = SeparableConv(128, (3, 3))(x, training)
        x = nn.BatchNorm(use_running_average=not training)(x)
        x = x + residual
        x = nn.silu(x)
        
        # Max Pool 2: 64×64 → 32×32
        x = nn.max_pool(x, (2, 2), strides=(2, 2))
        
        # === BLOC 3: 256 canaux ===
        x = SeparableConv(256, (3, 3))(x, training)
        x = nn.BatchNorm(use_running_average=not training)(x)
        x = nn.silu(x)
        
        x = SeparableConv(256, (3, 3))(x, training)
        x = nn.BatchNorm(use_running_average=not training)(x)
        x = nn.silu(x)
        
        # SE Attention
        x = SEBlock(reduction=16)(x, training)
        
        # Max Pool 3: 32×32 → 16×16
        x = nn.max_pool(x, (2, 2), strides=(2, 2))
        
        # === BLOC 4: 512 canaux ===
        x = nn.Conv(384, (1, 1), padding="SAME", use_bias=False,
                   kernel_init=nn.initializers.kaiming_normal())(x)
        x = nn.BatchNorm(use_running_average=not training)(x)
        x = nn.silu(x)
        
        x = SeparableConv(512, (3, 3))(x, training)
        x = nn.BatchNorm(use_running_average=not training)(x)
        x = nn.silu(x)
        
        # Residual connection
        residual = nn.Conv(512, (1, 1), padding="SAME", use_bias=False,
                          kernel_init=nn.initializers.kaiming_normal())(x)
        residual = nn.BatchNorm(use_running_average=not training)(residual)
        
        x = nn.Conv(512, (1, 1), padding="SAME", use_bias=False,
                   kernel_init=nn.initializers.kaiming_normal())(x)
        x = nn.BatchNorm(use_running_average=not training)(x)
        x = x + residual
        x = nn.silu(x)
        
        # SE Attention
        x = SEBlock(reduction=16)(x, training)
        
        # Spatial Attention
        x = SpatialAttention()(x, training)
        
        # Global Average Pooling
        x = jnp.mean(x, axis=(1, 2))  # (B, 512)
        
        # Classification head
        x = nn.LayerNorm()(x)
        x = nn.Dense(384, use_bias=True)(x)
        x = nn.silu(x)
        x = nn.Dropout(self.dropout_rate, deterministic=not training)(x)
        
        return nn.Dense(self.num_classes, use_bias=True)(x)


def create_sophisticated_cnn_128_plus(num_classes=2, dropout_rate=0.0):
    """Crée une instance de SophisticatedCNN128Plus optimisé+ pour images 128×128"""
    return SophisticatedCNN128Plus(num_classes=num_classes, dropout_rate=dropout_rate)


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

    @nn.compact
    def __call__(self, x, training=True):
        # Input: (B, 128, 128, C) où C=1 (grayscale) ou C=3 (RGB)

        # === BLOC 0: Conv initiale (inchangé) ===
        x = nn.Conv(64, (3, 3), padding="SAME", use_bias=False,
                   kernel_init=nn.initializers.kaiming_normal())(x)
        x = nn.BatchNorm(use_running_average=not training)(x)
        x = nn.silu(x)

        # === BLOC 1: 64 canaux (LITE: 96 -> 64, pleine résolution 128×128) ===
        x = SeparableConv(64, (3, 3))(x, training)
        x = nn.BatchNorm(use_running_average=not training)(x)
        x = nn.silu(x)

        residual = nn.Conv(64, (1, 1), padding="SAME", use_bias=False,
                          kernel_init=nn.initializers.kaiming_normal())(x)
        residual = nn.BatchNorm(use_running_average=not training)(residual)

        x = SeparableConv(64, (3, 3))(x, training)
        x = nn.BatchNorm(use_running_average=not training)(x)
        x = x + residual
        x = nn.silu(x)

        # Max Pool 1: 128×128 → 64×64
        x = nn.max_pool(x, (2, 2), strides=(2, 2))

        # === BLOC 2: 128 canaux (inchangé) ===
        x = SeparableConv(128, (3, 3))(x, training)
        x = nn.BatchNorm(use_running_average=not training)(x)
        x = nn.silu(x)

        residual = nn.Conv(128, (1, 1), padding="SAME", use_bias=False,
                          kernel_init=nn.initializers.kaiming_normal())(x)
        residual = nn.BatchNorm(use_running_average=not training)(residual)

        x = SeparableConv(128, (3, 3))(x, training)
        x = nn.BatchNorm(use_running_average=not training)(x)
        x = x + residual
        x = nn.silu(x)

        # Max Pool 2: 64×64 → 32×32
        x = nn.max_pool(x, (2, 2), strides=(2, 2))

        # === BLOC 3: 256 canaux (inchangé) ===
        x = SeparableConv(256, (3, 3))(x, training)
        x = nn.BatchNorm(use_running_average=not training)(x)
        x = nn.silu(x)

        x = SeparableConv(256, (3, 3))(x, training)
        x = nn.BatchNorm(use_running_average=not training)(x)
        x = nn.silu(x)

        # SE Attention
        x = SEBlock(reduction=16)(x, training)

        # Max Pool 3: 32×32 → 16×16
        x = nn.max_pool(x, (2, 2), strides=(2, 2))

        # === BLOC 4: 384 canaux (LITE: pic 512 -> 384, x0.75) ===
        x = nn.Conv(288, (1, 1), padding="SAME", use_bias=False,
                   kernel_init=nn.initializers.kaiming_normal())(x)
        x = nn.BatchNorm(use_running_average=not training)(x)
        x = nn.silu(x)

        x = SeparableConv(384, (3, 3))(x, training)
        x = nn.BatchNorm(use_running_average=not training)(x)
        x = nn.silu(x)

        # Residual connection
        residual = nn.Conv(384, (1, 1), padding="SAME", use_bias=False,
                          kernel_init=nn.initializers.kaiming_normal())(x)
        residual = nn.BatchNorm(use_running_average=not training)(residual)

        x = nn.Conv(384, (1, 1), padding="SAME", use_bias=False,
                   kernel_init=nn.initializers.kaiming_normal())(x)
        x = nn.BatchNorm(use_running_average=not training)(x)
        x = x + residual
        x = nn.silu(x)

        # SE Attention
        x = SEBlock(reduction=16)(x, training)

        # Spatial Attention
        x = SpatialAttention()(x, training)

        # Global Average Pooling
        x = jnp.mean(x, axis=(1, 2))  # (B, 384)

        # Classification head (inchangé)
        x = nn.LayerNorm()(x)
        x = nn.Dense(384, use_bias=True)(x)
        x = nn.silu(x)
        x = nn.Dropout(self.dropout_rate, deterministic=not training)(x)

        return nn.Dense(self.num_classes, use_bias=True)(x)


def create_sophisticated_cnn_128_lite(num_classes=2, dropout_rate=0.0):
    """Crée une instance de SophisticatedCNN128Lite (Blocs 1+4 allégés pour la vitesse d'inférence)"""
    return SophisticatedCNN128Lite(num_classes=num_classes, dropout_rate=dropout_rate)


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

    @nn.compact
    def __call__(self, x, training=True):
        # Input: (B, 32, 32, C) où C=1 (grayscale) ou C=3 (RGB)

        # === BLOC 0: Conv initiale ===
        x = nn.Conv(32, (3, 3), padding="SAME", use_bias=False,
                   kernel_init=nn.initializers.kaiming_normal())(x)
        x = nn.BatchNorm(use_running_average=not training)(x)
        x = nn.silu(x)

        # === BLOC 1: 48 canaux ===
        x = SeparableConv(48, (3, 3))(x, training)
        x = nn.BatchNorm(use_running_average=not training)(x)
        x = nn.silu(x)

        residual = nn.Conv(48, (1, 1), padding="SAME", use_bias=False,
                          kernel_init=nn.initializers.kaiming_normal())(x)
        residual = nn.BatchNorm(use_running_average=not training)(residual)

        x = SeparableConv(48, (3, 3))(x, training)
        x = nn.BatchNorm(use_running_average=not training)(x)
        x = x + residual
        x = nn.silu(x)

        # Max Pool 1: 32×32 → 16×16
        x = nn.max_pool(x, (2, 2), strides=(2, 2))

        # === BLOC 2: 64 canaux ===
        x = SeparableConv(64, (3, 3))(x, training)
        x = nn.BatchNorm(use_running_average=not training)(x)
        x = nn.silu(x)

        residual = nn.Conv(64, (1, 1), padding="SAME", use_bias=False,
                          kernel_init=nn.initializers.kaiming_normal())(x)
        residual = nn.BatchNorm(use_running_average=not training)(residual)

        x = SeparableConv(64, (3, 3))(x, training)
        x = nn.BatchNorm(use_running_average=not training)(x)
        x = x + residual
        x = nn.silu(x)

        # Max Pool 2: 16×16 → 8×8
        x = nn.max_pool(x, (2, 2), strides=(2, 2))

        # === BLOC 3: 128 canaux ===
        x = SeparableConv(128, (3, 3))(x, training)
        x = nn.BatchNorm(use_running_average=not training)(x)
        x = nn.silu(x)

        x = SeparableConv(128, (3, 3))(x, training)
        x = nn.BatchNorm(use_running_average=not training)(x)
        x = nn.silu(x)

        # SE Attention
        x = SEBlock(reduction=16)(x, training)

        # Pas de 3e max-pool (contrairement à 128-Plus) : reste à 8×8, suffisant vu la taille d'entrée native

        # === BLOC 4: bottleneck 192 -> 256 canaux ===
        x = nn.Conv(192, (1, 1), padding="SAME", use_bias=False,
                   kernel_init=nn.initializers.kaiming_normal())(x)
        x = nn.BatchNorm(use_running_average=not training)(x)
        x = nn.silu(x)

        x = SeparableConv(256, (3, 3))(x, training)
        x = nn.BatchNorm(use_running_average=not training)(x)
        x = nn.silu(x)

        residual = nn.Conv(256, (1, 1), padding="SAME", use_bias=False,
                          kernel_init=nn.initializers.kaiming_normal())(x)
        residual = nn.BatchNorm(use_running_average=not training)(residual)

        x = nn.Conv(256, (1, 1), padding="SAME", use_bias=False,
                   kernel_init=nn.initializers.kaiming_normal())(x)
        x = nn.BatchNorm(use_running_average=not training)(x)
        x = x + residual
        x = nn.silu(x)

        # SE Attention
        x = SEBlock(reduction=16)(x, training)

        # Spatial Attention
        x = SpatialAttention()(x, training)

        # Global Average Pooling
        x = jnp.mean(x, axis=(1, 2))  # (B, 256)

        # Classification head
        x = nn.LayerNorm()(x)
        x = nn.Dense(192, use_bias=True)(x)
        x = nn.silu(x)
        x = nn.Dropout(self.dropout_rate, deterministic=not training)(x)

        return nn.Dense(self.num_classes, use_bias=True)(x)


def create_sophisticated_cnn_32_plus(num_classes=2, dropout_rate=0.0):
    """Crée une instance de SophisticatedCNN32Plus, variante réduite pour images 32×32 (ex. CIFAR-10)"""
    return SophisticatedCNN32Plus(num_classes=num_classes, dropout_rate=dropout_rate)


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

    @nn.compact
    def __call__(self, x, training=True):
        # --- ENCODER ---
        # Block 1 (224x224 -> 112x112)
        x1 = nn.Conv(32, (3, 3), padding="SAME")(x)
        x1 = nn.BatchNorm(use_running_average=not training)(x1)
        x1 = nn.silu(x1)
        x1 = nn.Conv(32, (3, 3), padding="SAME")(x1)
        x1 = nn.BatchNorm(use_running_average=not training)(x1)
        x1 = nn.silu(x1)
        p1 = nn.max_pool(x1, window_shape=(2, 2), strides=(2, 2)) # 112x112
        
        # Block 2 (112x112 -> 56x56)
        x2 = nn.Conv(64, (3, 3), padding="SAME")(p1)
        x2 = nn.BatchNorm(use_running_average=not training)(x2)
        x2 = nn.silu(x2)
        x2 = nn.Conv(64, (3, 3), padding="SAME")(x2)
        x2 = nn.BatchNorm(use_running_average=not training)(x2)
        x2 = nn.silu(x2)
        p2 = nn.max_pool(x2, window_shape=(2, 2), strides=(2, 2)) # 56x56
        
        # Block 3 (56x56 -> 28x28)
        x3 = nn.Conv(128, (3, 3), padding="SAME")(p2)
        x3 = nn.BatchNorm(use_running_average=not training)(x3)
        x3 = nn.silu(x3)
        x3 = nn.Conv(128, (3, 3), padding="SAME")(x3)
        x3 = nn.BatchNorm(use_running_average=not training)(x3)
        x3 = nn.silu(x3)
        p3 = nn.max_pool(x3, window_shape=(2, 2), strides=(2, 2)) # 28x28
        
        # --- BOTTLENECK ---
        # 28x28
        b = nn.Conv(256, (3, 3), padding="SAME")(p3)
        b = nn.BatchNorm(use_running_average=not training)(b)
        b = nn.silu(b)
        b = nn.Conv(256, (3, 3), padding="SAME")(b)
        b = nn.BatchNorm(use_running_average=not training)(b)
        b = nn.silu(b)
        
        # Application du Dropout au bottleneck (le seul endroit où il est vraiment efficace sur un U-Net)
        b = nn.Dropout(self.dropout_rate, deterministic=not training)(b)
        
        # --- DECODER ---
        # Up 1 (28x28 -> 56x56)
        u1 = jax.image.resize(b, shape=(b.shape[0], x3.shape[1], x3.shape[2], b.shape[3]), method='bilinear')
        u1 = nn.Conv(128, (2, 2), padding="SAME")(u1)
        u1 = jnp.concatenate([u1, x3], axis=-1)
        u1 = nn.Conv(128, (3, 3), padding="SAME")(u1)
        u1 = nn.BatchNorm(use_running_average=not training)(u1)
        u1 = nn.silu(u1)
        u1 = nn.Conv(128, (3, 3), padding="SAME")(u1)
        u1 = nn.BatchNorm(use_running_average=not training)(u1)
        u1 = nn.silu(u1)
        
        # Up 2 (56x56 -> 112x112)
        u2 = jax.image.resize(u1, shape=(u1.shape[0], x2.shape[1], x2.shape[2], u1.shape[3]), method='bilinear')
        u2 = nn.Conv(64, (2, 2), padding="SAME")(u2)
        u2 = jnp.concatenate([u2, x2], axis=-1)
        u2 = nn.Conv(64, (3, 3), padding="SAME")(u2)
        u2 = nn.BatchNorm(use_running_average=not training)(u2)
        u2 = nn.silu(u2)
        u2 = nn.Conv(64, (3, 3), padding="SAME")(u2)
        u2 = nn.BatchNorm(use_running_average=not training)(u2)
        u2 = nn.silu(u2)
        
        # Up 3 (112x112 -> 224x224)
        u3 = jax.image.resize(u2, shape=(u2.shape[0], x1.shape[1], x1.shape[2], u2.shape[3]), method='bilinear')
        u3 = nn.Conv(32, (2, 2), padding="SAME")(u3)
        u3 = jnp.concatenate([u3, x1], axis=-1)
        u3 = nn.Conv(32, (3, 3), padding="SAME")(u3)
        u3 = nn.BatchNorm(use_running_average=not training)(u3)
        u3 = nn.silu(u3)
        u3 = nn.Conv(32, (3, 3), padding="SAME")(u3)
        u3 = nn.BatchNorm(use_running_average=not training)(u3)
        u3 = nn.silu(u3)
        
        # --- OUTPUT ---
        # Mask 224x224x1
        out = nn.Conv(1, (1, 1), padding="SAME")(u3)
        return nn.sigmoid(out)


def create_aircraft_detector_unet(dropout_rate=0.2, **kwargs):
    """Factory for UNet Detector"""
    return AircraftDetectorUNet(dropout_rate=dropout_rate)


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

    @nn.compact
    def __call__(self, x, training: bool = True):
        # --- ENCODER --- (identique à AircraftDetectorUNet)
        # Block 1 (H,W -> H/2,W/2)
        x1 = nn.Conv(32, (3, 3), padding="SAME")(x)
        x1 = nn.BatchNorm(use_running_average=not training)(x1)
        x1 = nn.silu(x1)
        x1 = nn.Conv(32, (3, 3), padding="SAME")(x1)
        x1 = nn.BatchNorm(use_running_average=not training)(x1)
        x1 = nn.silu(x1)
        p1 = nn.max_pool(x1, window_shape=(2, 2), strides=(2, 2))

        # Block 2 (H/2,W/2 -> H/4,W/4)
        x2 = nn.Conv(64, (3, 3), padding="SAME")(p1)
        x2 = nn.BatchNorm(use_running_average=not training)(x2)
        x2 = nn.silu(x2)
        x2 = nn.Conv(64, (3, 3), padding="SAME")(x2)
        x2 = nn.BatchNorm(use_running_average=not training)(x2)
        x2 = nn.silu(x2)
        p2 = nn.max_pool(x2, window_shape=(2, 2), strides=(2, 2))

        # Block 3 (H/4,W/4 -> H/8,W/8)
        x3 = nn.Conv(128, (3, 3), padding="SAME")(p2)
        x3 = nn.BatchNorm(use_running_average=not training)(x3)
        x3 = nn.silu(x3)
        x3 = nn.Conv(128, (3, 3), padding="SAME")(x3)
        x3 = nn.BatchNorm(use_running_average=not training)(x3)
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
        b = nn.Conv(256, (3, 3), kernel_dilation=(2, 2), padding="SAME")(p3)
        b = nn.BatchNorm(use_running_average=not training)(b)
        b = nn.silu(b)
        b = nn.Conv(256, (3, 3), kernel_dilation=(4, 4), padding="SAME")(b)
        b = nn.BatchNorm(use_running_average=not training)(b)
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
        context = nn.Conv(256, (1, 1), padding="SAME")(context)
        context = nn.silu(context)
        context = jnp.broadcast_to(context, b.shape)  # (B,28,28,256)
        b = jnp.concatenate([b, context], axis=-1)  # (B,28,28,512) : local + global
        b = nn.Conv(256, (1, 1), padding="SAME")(b)  # projection retour a 256ch
        b = nn.BatchNorm(use_running_average=not training)(b)
        b = nn.silu(b)

        b = nn.Dropout(self.dropout_rate, deterministic=not training)(b)

        # --- DECODER --- (identique à AircraftDetectorUNet)
        # Up 1
        u1 = jax.image.resize(b, shape=(b.shape[0], x3.shape[1], x3.shape[2], b.shape[3]), method='bilinear')
        u1 = nn.Conv(128, (2, 2), padding="SAME")(u1)
        u1 = jnp.concatenate([u1, x3], axis=-1)
        u1 = nn.Conv(128, (3, 3), padding="SAME")(u1)
        u1 = nn.BatchNorm(use_running_average=not training)(u1)
        u1 = nn.silu(u1)
        u1 = nn.Conv(128, (3, 3), padding="SAME")(u1)
        u1 = nn.BatchNorm(use_running_average=not training)(u1)
        u1 = nn.silu(u1)

        # Up 2
        u2 = jax.image.resize(u1, shape=(u1.shape[0], x2.shape[1], x2.shape[2], u1.shape[3]), method='bilinear')
        u2 = nn.Conv(64, (2, 2), padding="SAME")(u2)
        u2 = jnp.concatenate([u2, x2], axis=-1)
        u2 = nn.Conv(64, (3, 3), padding="SAME")(u2)
        u2 = nn.BatchNorm(use_running_average=not training)(u2)
        u2 = nn.silu(u2)
        u2 = nn.Conv(64, (3, 3), padding="SAME")(u2)
        u2 = nn.BatchNorm(use_running_average=not training)(u2)
        u2 = nn.silu(u2)

        # Up 3
        u3 = jax.image.resize(u2, shape=(u2.shape[0], x1.shape[1], x1.shape[2], u2.shape[3]), method='bilinear')
        u3 = nn.Conv(32, (2, 2), padding="SAME")(u3)
        u3 = jnp.concatenate([u3, x1], axis=-1)
        u3 = nn.Conv(32, (3, 3), padding="SAME")(u3)
        u3 = nn.BatchNorm(use_running_average=not training)(u3)
        u3 = nn.silu(u3)
        u3 = nn.Conv(32, (3, 3), padding="SAME")(u3)
        u3 = nn.BatchNorm(use_running_average=not training)(u3)
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
        heatmap = nn.Conv(1, (1, 1), padding="SAME", bias_init=nn.initializers.constant(heatmap_bias_init))(u3)
        heatmap = nn.sigmoid(heatmap)

        # Regression de taille (B,H,W,2) largeur/hauteur, pas d'activation
        # (convention CenterNet standard - la perte, Story 7.3, gere la positivite)
        size = nn.Conv(2, (1, 1), padding="SAME")(u3)

        return {HEATMAP_KEY: heatmap, SIZE_KEY: size}


def create_aircraft_detector_centernet(dropout_rate=0.2, heatmap_prior=0.01, **kwargs):
    """Factory for CenterNet Detector"""
    return AircraftDetectorCenterNet(dropout_rate=dropout_rate, heatmap_prior=heatmap_prior)


# MiniUNet/conv_block/create_aircraft_detector_miniunet supprimés le 2026-07-15 : non utilisés par
# aucune des 4 configs actives (seule référence était une ligne commentée dans dataset_configs.py),
# aucun .pkl versionné n'en dépendait (vérifié : best_model_detection.pkl est aircraft_detector_unet).
# Récupérable via l'historique git si besoin.


class Kepler1DConvNet(nn.Module):
    """
    Réseau de Neurones Convolutif 1D pour l'analyse de Séries Temporelles.
    Spécialement conçu pour détecter les creux de luminosité (transit) dans les données Kepler.
    """
    num_classes: int = 2
    dropout_rate: float = 0.3

    @nn.compact
    def __call__(self, x, training: bool):
        # x est de shape (Batch, SequenceLength, 1) -> ex: (B, 3197, 1)
        
        # Bloc 1 (Détection de motifs locaux)
        x = nn.Conv(features=32, kernel_size=(11,), padding='SAME')(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2,), strides=(2,))
        
        # Bloc 2 (Extraction de features temporelles)
        x = nn.Conv(features=64, kernel_size=(5,), padding='SAME')(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2,), strides=(2,))
        
        # Bloc 3
        x = nn.Conv(features=128, kernel_size=(5,), padding='SAME')(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2,), strides=(2,))
        
        # Bloc 4
        x = nn.Conv(features=256, kernel_size=(3,), padding='SAME')(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2,), strides=(2,))
        
        # Global Average Pooling 1D (On moyenne sur le temps restant)
        x = jnp.mean(x, axis=1) # (Batch, 256)
        
        # Classification Head
        x = nn.Dense(features=64)(x)
        x = nn.relu(x)
        x = nn.Dropout(self.dropout_rate, deterministic=not training)(x)
        x = nn.Dense(features=self.num_classes)(x)
        
        return x

def create_kepler_1d_cnn(**kwargs):
    return Kepler1DConvNet(**kwargs)


MODELS = {
    'aircraft_detector_unet': create_aircraft_detector_unet, # Semantic Segmentation U-Net
    'aircraft_detector_centernet': create_aircraft_detector_centernet, # CenterNet (point central, anchor-free)
    'sophisticated_cnn_128_plus': create_sophisticated_cnn_128_plus,
    'sophisticated_cnn_128_lite': create_sophisticated_cnn_128_lite,
    'sophisticated_cnn_32_plus': create_sophisticated_cnn_32_plus,
    'kepler_1d_cnn': create_kepler_1d_cnn,
}

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
            'params': 'non mesuré ici (voir checkpoint)',
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
        }
    }

    if model_name not in model_info:
        raise ValueError(f"Modèle '{model_name}' non trouvé")

    return model_info[model_name]


class TrainStateWithBatchStats(train_state.TrainState):
    """TrainState étendu pour gérer batch_stats"""
    batch_stats: Optional[flax.core.FrozenDict] = None
