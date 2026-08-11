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


class AircraftDetectorCenterNetLite(nn.Module):
    """
    Variante allegee de AircraftDetectorCenterNet (vitesse d'inference).

    Le bottleneck dilate + branche de contexte global (2026-07-19/07-22) concentre
    51% des parametres du modele (1.08M/2.13M mesure) ET c'est le mecanisme qui a
    resolu la chute d'IoU sur les boites plein-cadre (70-100% de l'image) - INTACT
    ici, caractere pour caractere identique a AircraftDetectorCenterNet. Aucune
    modification ne touche Conv_6-9 (dilatation, contexte global) ni les tetes de
    sortie (biais heatmap_prior notamment).

    Seul l'encoder/decoder ("jumeau" AircraftDetectorUNet, AD-10, jamais touche par
    les fix plein-cadre) est modifie : les convolutions 3x3 pleines deviennent des
    SeparableConv (depthwise+pointwise, deja utilisee et validee sur les modeles de
    classification de ce fichier - voir SophisticatedCNN128Lite). Les convs 2x2 de
    pont avant chaque concat (fusion skip connection) restent des nn.Conv pleines,
    volontairement - elles ne font que reconcilier les canaux avant fusion, pas de
    l'extraction spatiale.

    Risque distinct de la classification (Winston, 2026-07-26) : ici la tache est
    une prediction dense (heatmap+taille par pixel) avec des skip connections qui
    transportent le detail spatial de l'encoder directement au decodeur - contrairement
    a un classifieur qui finit par un Global Average Pooling, la factorisation
    depthwise/pointwise peut couter de la finesse spatiale exactement la ou elle sert.
    A valider par un vrai entrainement contre la reference actuelle (IoU moyen 0.7003,
    82.5% GOOD, voir deferred-work.md) - surveiller specifiquement le bucket 70-100%
    plein-cadre, le cas que la dilatation+contexte a sauve.
    """
    dropout_rate: float = 0.2
    heatmap_prior: float = 0.01

    @nn.compact
    def __call__(self, x, training: bool = True):
        # --- ENCODER --- (LITE: nn.Conv -> SeparableConv, sinon identique)
        # Block 1 (H,W -> H/2,W/2)
        x1 = SeparableConv(32, (3, 3))(x, training)
        x1 = nn.BatchNorm(use_running_average=not training)(x1)
        x1 = nn.silu(x1)
        x1 = SeparableConv(32, (3, 3))(x1, training)
        x1 = nn.BatchNorm(use_running_average=not training)(x1)
        x1 = nn.silu(x1)
        p1 = nn.max_pool(x1, window_shape=(2, 2), strides=(2, 2))

        # Block 2 (H/2,W/2 -> H/4,W/4)
        x2 = SeparableConv(64, (3, 3))(p1, training)
        x2 = nn.BatchNorm(use_running_average=not training)(x2)
        x2 = nn.silu(x2)
        x2 = SeparableConv(64, (3, 3))(x2, training)
        x2 = nn.BatchNorm(use_running_average=not training)(x2)
        x2 = nn.silu(x2)
        p2 = nn.max_pool(x2, window_shape=(2, 2), strides=(2, 2))

        # Block 3 (H/4,W/4 -> H/8,W/8)
        x3 = SeparableConv(128, (3, 3))(p2, training)
        x3 = nn.BatchNorm(use_running_average=not training)(x3)
        x3 = nn.silu(x3)
        x3 = SeparableConv(128, (3, 3))(x3, training)
        x3 = nn.BatchNorm(use_running_average=not training)(x3)
        x3 = nn.silu(x3)
        p3 = nn.max_pool(x3, window_shape=(2, 2), strides=(2, 2))

        # --- BOTTLENECK --- (inchangé, caractère pour caractère - voir docstring)
        b = nn.Conv(256, (3, 3), kernel_dilation=(2, 2), padding="SAME")(p3)
        b = nn.BatchNorm(use_running_average=not training)(b)
        b = nn.silu(b)
        b = nn.Conv(256, (3, 3), kernel_dilation=(4, 4), padding="SAME")(b)
        b = nn.BatchNorm(use_running_average=not training)(b)
        b = nn.silu(b)

        context = jnp.mean(b, axis=(1, 2), keepdims=True)  # (B,1,1,256)
        context = nn.Conv(256, (1, 1), padding="SAME")(context)
        context = nn.silu(context)
        context = jnp.broadcast_to(context, b.shape)  # (B,28,28,256)
        b = jnp.concatenate([b, context], axis=-1)  # (B,28,28,512) : local + global
        b = nn.Conv(256, (1, 1), padding="SAME")(b)  # projection retour a 256ch
        b = nn.BatchNorm(use_running_average=not training)(b)
        b = nn.silu(b)

        b = nn.Dropout(self.dropout_rate, deterministic=not training)(b)

        # --- DECODER --- (LITE: convs 3x3 post-concat -> SeparableConv ; ponts 2x2 inchangés)
        # Up 1
        u1 = jax.image.resize(b, shape=(b.shape[0], x3.shape[1], x3.shape[2], b.shape[3]), method='bilinear')
        u1 = nn.Conv(128, (2, 2), padding="SAME")(u1)
        u1 = jnp.concatenate([u1, x3], axis=-1)
        u1 = SeparableConv(128, (3, 3))(u1, training)
        u1 = nn.BatchNorm(use_running_average=not training)(u1)
        u1 = nn.silu(u1)
        u1 = SeparableConv(128, (3, 3))(u1, training)
        u1 = nn.BatchNorm(use_running_average=not training)(u1)
        u1 = nn.silu(u1)

        # Up 2
        u2 = jax.image.resize(u1, shape=(u1.shape[0], x2.shape[1], x2.shape[2], u1.shape[3]), method='bilinear')
        u2 = nn.Conv(64, (2, 2), padding="SAME")(u2)
        u2 = jnp.concatenate([u2, x2], axis=-1)
        u2 = SeparableConv(64, (3, 3))(u2, training)
        u2 = nn.BatchNorm(use_running_average=not training)(u2)
        u2 = nn.silu(u2)
        u2 = SeparableConv(64, (3, 3))(u2, training)
        u2 = nn.BatchNorm(use_running_average=not training)(u2)
        u2 = nn.silu(u2)

        # Up 3
        u3 = jax.image.resize(u2, shape=(u2.shape[0], x1.shape[1], x1.shape[2], u2.shape[3]), method='bilinear')
        u3 = nn.Conv(32, (2, 2), padding="SAME")(u3)
        u3 = jnp.concatenate([u3, x1], axis=-1)
        u3 = SeparableConv(32, (3, 3))(u3, training)
        u3 = nn.BatchNorm(use_running_average=not training)(u3)
        u3 = nn.silu(u3)
        u3 = SeparableConv(32, (3, 3))(u3, training)
        u3 = nn.BatchNorm(use_running_average=not training)(u3)
        u3 = nn.silu(u3)

        # --- OUTPUT : deux têtes paralleles --- (inchangé, voir AircraftDetectorCenterNet)
        assert 0.0 < self.heatmap_prior < 1.0, (
            f"heatmap_prior doit etre dans (0,1) - log(p/(1-p)) indefini sinon, recu {self.heatmap_prior}"
        )
        heatmap_bias_init = math.log(self.heatmap_prior / (1.0 - self.heatmap_prior))
        heatmap = nn.Conv(1, (1, 1), padding="SAME", bias_init=nn.initializers.constant(heatmap_bias_init))(u3)
        heatmap = nn.sigmoid(heatmap)

        size = nn.Conv(2, (1, 1), padding="SAME")(u3)

        return {HEATMAP_KEY: heatmap, SIZE_KEY: size}


def create_aircraft_detector_centernet_lite(dropout_rate=0.2, heatmap_prior=0.01, **kwargs):
    """Factory for CenterNet Detector Lite (encoder/decoder en SeparableConv, bottleneck inchangé)"""
    return AircraftDetectorCenterNetLite(dropout_rate=dropout_rate, heatmap_prior=heatmap_prior)


# MiniUNet/conv_block/create_aircraft_detector_miniunet supprimés le 2026-07-15 : non utilisés par
# aucune des 4 configs actives (seule référence était une ligne commentée dans dataset_configs.py),
# aucun .pkl versionné n'en dépendait (vérifié : best_model_detection.pkl est aircraft_detector_unet).
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
    """
    num_moves: int  # taille de la tête policy (4672) — passe par num_classes générique (main.py), jamais un littéral dupliqué (AD-22 hérité)
    dropout_rate: float = 0.1
    num_layers: int = 4
    d_model: int = 128
    num_heads: int = 4

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
        h = embed(tokens)  # (B, L, D)

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
            h = nn.MultiHeadDotProductAttention(num_heads=self.num_heads)(inputs_q=h, mask=mask)
            h = nn.Dropout(self.dropout_rate, deterministic=not training)(h)
            h = residual + h

            residual = h
            h = nn.LayerNorm()(h)
            h = nn.Dense(4 * self.d_model)(h)
            h = nn.gelu(h)
            h = nn.Dense(self.d_model)(h)
            h = nn.Dropout(self.dropout_rate, deterministic=not training)(h)
            h = residual + h

        h = nn.LayerNorm()(h)

        # Lecture du DERNIER token (toujours reel grace au padding a gauche, AD-28) -
        # jamais un pooling moyen sur la sequence (AD-28).
        last_hidden = h[:, -1, :]  # (B, D)

        # (B, num_moves) - jamais BOS/PAD en sortie (AD-30), aucun masquage des coups
        # illegaux (AD-22 herite).
        policy_logits = nn.Dense(self.num_moves)(last_hidden)
        return policy_logits


def create_chess_move_token_transformer(num_classes, dropout_rate=0.1, **kwargs):
    """
    Factory pour chess_move_token_transformer. `num_classes` (nom imposé par la
    plomberie model_kwargs générique de main.py) porte la taille de l'espace de coups
    (4672) — même convention que create_chess_cnn_attention_policy_value ci-dessus.
    """
    return ChessMoveTokenTransformer(num_moves=num_classes, dropout_rate=dropout_rate, **kwargs)


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
    'aircraft_detector_centernet_lite': create_aircraft_detector_centernet_lite, # CenterNet, encoder/decoder en SeparableConv
    'chess_cnn_attention_policy_value': create_chess_cnn_attention_policy_value, # CNN 8x8 + bottleneck attention, policy+value (Epic 9)
    'chess_cnn_attention_legal_moves': create_chess_cnn_attention_legal_moves, # meme backbone, sans tete value, multi-label coups legaux
    'chess_move_token_transformer': create_chess_move_token_transformer, # decodeur causal sur move_tokens, policy-only (Epic 11, spike)
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
            'params': '2,125,539 mesuré',
            'size': 'non mesuré ici (voir checkpoint)',
            'best_for': 'Détection multi-instances (formations serrées), remplace la segmentation U-Net (AD-9/AD-10).'
        },
        'aircraft_detector_centernet_lite': {
            'name': 'AircraftDetectorCenterNetLite',
            'description': 'Variante de AircraftDetectorCenterNet: encoder/decoder en SeparableConv (comme SophisticatedCNN128Lite), bottleneck dilaté+contexte global strictement inchangé (AD-20-like: ne pas régresser sur le fix plein-cadre).',
            'params': 'non mesuré ici (voir checkpoint) - a l\'essai, non entraîné',
            'size': 'non mesuré ici (voir checkpoint)',
            'best_for': 'A valider: candidat vitesse d\'inférence pour JAX_DETECTOR si l\'IoU sur le bucket plein-cadre (70-100%) tient face à la référence (0.7003 IoU moyen, 82.5% GOOD).'
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
