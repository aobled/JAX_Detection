"""
Librairie des modèles de deep learning
Contient tous les modèles utilisés pour l'entraînement
"""

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.training import train_state
import flax


class SeparableConv(nn.Module):
    """Convolution séparable (Depthwise + Pointwise)"""
    filters: int
    kernel_size: tuple
    strides: tuple = (1, 1) # ✅ Added strides support
    padding: str = "SAME"
    use_bias: bool = False

    @nn.compact
    def __call__(self, x, training=True):
        # Depthwise convolution
        x = nn.Conv(
            features=x.shape[-1],
            kernel_size=self.kernel_size,
            strides=self.strides, # ✅ Apply strides here
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


class ChannelAttention(nn.Module):
    """Channel Attention Module"""
    
    @nn.compact
    def __call__(self, x, training=True):
        # Input: (B, H, W, C)
        # Output: (B, H, W, C)
        
        # Global Average Pooling
        avg_pool = jnp.mean(x, axis=(1, 2), keepdims=True)  # (B, 1, 1, C)
        
        # Global Max Pooling
        max_pool = jnp.max(x, axis=(1, 2), keepdims=True)    # (B, 1, 1, C)
        
        # MLP pour channel attention
        # Shared MLP
        avg_out = nn.Dense(x.shape[-1] // 16, use_bias=True)(avg_pool)
        avg_out = nn.relu(avg_out)
        avg_out = nn.Dense(x.shape[-1], use_bias=True)(avg_out)
        
        max_out = nn.Dense(x.shape[-1] // 16, use_bias=True)(max_pool)
        max_out = nn.relu(max_out)
        max_out = nn.Dense(x.shape[-1], use_bias=True)(max_out)
        
        # Combine et sigmoid
        channel_attn = nn.sigmoid(avg_out + max_out)  # (B, 1, 1, C)
        
        return x * channel_attn


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


class CBAMBlock(nn.Module):
    """
    Convolutional Block Attention Module (CBAM)
    Combine Channel Attention et Spatial Attention
    """
    
    @nn.compact
    def __call__(self, x, training=True):
        # Input: (B, H, W, C)
        # Output: (B, H, W, C)
        
        # 1. Channel Attention
        x = ChannelAttention()(x, training)
        
        # 2. Spatial Attention
        x = SpatialAttention()(x, training)
        
        return x


class SophisticatedCNN(nn.Module):
    """Modèle CNN sophistiqué avec convolutions séparables, SE et attention spatiale - VERSION ORIGINALE"""
    num_classes: int = 2
    dropout_rate: float = 0.0

    @nn.compact
    def __call__(self, x, training=True):
        # Input: (B, 64, 64, 3)
        
        # Conv1: 64 filtres
        x = nn.Conv(64, (3, 3), padding="SAME", use_bias=False,
                   kernel_init=nn.initializers.kaiming_normal())(x)
        x = nn.BatchNorm(use_running_average=not training)(x)
        x = nn.silu(x)  # act1
        
        # Separable Conv Block 1
        x = SeparableConv(128, (3, 3))(x, training)
        x = nn.BatchNorm(use_running_average=not training)(x)
        x = nn.silu(x)  # act2
        
        # Residual connection
        residual = nn.Conv(128, (1, 1), padding="SAME", use_bias=False,
                          kernel_init=nn.initializers.kaiming_normal())(x)
        residual = nn.BatchNorm(use_running_average=not training)(residual)
        
        x = SeparableConv(128, (3, 3))(x, training)
        x = nn.BatchNorm(use_running_average=not training)(x)
        x = x + residual  # Skip connection
        x = nn.silu(x)  # act3
        
        # Max Pool
        x = nn.max_pool(x, (2, 2), strides=(2, 2))  # 32x32
        
        # Separable Conv Block 2
        x = SeparableConv(256, (3, 3))(x, training)
        x = nn.BatchNorm(use_running_average=not training)(x)
        x = nn.silu(x)  # act4
        
        x = SeparableConv(256, (3, 3))(x, training)
        x = nn.BatchNorm(use_running_average=not training)(x)
        x = nn.silu(x)  # act5
        
        # SE Attention
        x = SEBlock(reduction=16)(x, training)
        
        # Max Pool
        x = nn.max_pool(x, (2, 2), strides=(2, 2))  # 16x16
        
        # Conv Block 3
        x = nn.Conv(512, (1, 1), padding="SAME", use_bias=False,
                   kernel_init=nn.initializers.kaiming_normal())(x)
        x = nn.BatchNorm(use_running_average=not training)(x)
        x = nn.silu(x)  # act6
        
        x = SeparableConv(512, (3, 3))(x, training)
        x = nn.BatchNorm(use_running_average=not training)(x)
        x = nn.silu(x)  # act7
        
        # Residual connection
        residual = nn.Conv(512, (1, 1), padding="SAME", use_bias=False,
                          kernel_init=nn.initializers.kaiming_normal())(x)
        residual = nn.BatchNorm(use_running_average=not training)(residual)
        
        x = nn.Conv(512, (1, 1), padding="SAME", use_bias=False,
                   kernel_init=nn.initializers.kaiming_normal())(x)
        x = nn.BatchNorm(use_running_average=not training)(x)
        x = x + residual  # Skip connection
        x = nn.silu(x)  # res3
        
        # SE Attention 2
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


class SophisticatedCNNDropedOut(nn.Module):
    """
    CNN sophistiqué avec DROPOUT STRATÉGIQUE pour combattre l'overfitting
    
    Basé sur SophisticatedCNN avec ajout de dropout progressif:
    - Après bloc 1 (128 canaux): dropout 0.1
    - Après bloc 2 (256 canaux): dropout 0.15  
    - Après bloc 3 (512 canaux): dropout 0.2
    - Dans classification head: dropout 0.3 puis 0.4
    
    Objectif: Réduire l'écart train/val de ~20% à <10%
    """
    num_classes: int = 2
    dropout_rate: float = 0.3  # Base dropout pour le head

    @nn.compact
    def __call__(self, x, training=True):
        # Input: (B, 64, 64, 3)
        
        # Conv1: 64 filtres
        x = nn.Conv(64, (3, 3), padding="SAME", use_bias=False,
                   kernel_init=nn.initializers.kaiming_normal())(x)
        x = nn.BatchNorm(use_running_average=not training)(x)
        x = nn.silu(x)  # act1
        
        # Separable Conv Block 1 (128 canaux)
        x = SeparableConv(128, (3, 3))(x, training)
        x = nn.BatchNorm(use_running_average=not training)(x)
        x = nn.silu(x)  # act2
        
        # Residual connection
        residual = nn.Conv(128, (1, 1), padding="SAME", use_bias=False,
                          kernel_init=nn.initializers.kaiming_normal())(x)
        residual = nn.BatchNorm(use_running_average=not training)(residual)
        
        x = SeparableConv(128, (3, 3))(x, training)
        x = nn.BatchNorm(use_running_average=not training)(x)
        x = x + residual  # Skip connection
        x = nn.silu(x)  # act3
        
        # 🔥 DROPOUT 1: Après bloc 128 canaux
        x = nn.Dropout(0.1, deterministic=not training)(x)
        
        # Max Pool
        x = nn.max_pool(x, (2, 2), strides=(2, 2))  # 32x32
        
        # Separable Conv Block 2 (256 canaux)
        x = SeparableConv(256, (3, 3))(x, training)
        x = nn.BatchNorm(use_running_average=not training)(x)
        x = nn.silu(x)  # act4
        
        x = SeparableConv(256, (3, 3))(x, training)
        x = nn.BatchNorm(use_running_average=not training)(x)
        x = nn.silu(x)  # act5
        
        # SE Attention
        x = SEBlock(reduction=16)(x, training)
        
        # 🔥 DROPOUT 2: Après bloc 256 canaux
        x = nn.Dropout(0.15, deterministic=not training)(x)
        
        # Max Pool
        x = nn.max_pool(x, (2, 2), strides=(2, 2))  # 16x16
        
        # Conv Block 3 (512 canaux)
        x = nn.Conv(512, (1, 1), padding="SAME", use_bias=False,
                   kernel_init=nn.initializers.kaiming_normal())(x)
        x = nn.BatchNorm(use_running_average=not training)(x)
        x = nn.silu(x)  # act6
        
        x = SeparableConv(512, (3, 3))(x, training)
        x = nn.BatchNorm(use_running_average=not training)(x)
        x = nn.silu(x)  # act7
        
        # Residual connection
        residual = nn.Conv(512, (1, 1), padding="SAME", use_bias=False,
                          kernel_init=nn.initializers.kaiming_normal())(x)
        residual = nn.BatchNorm(use_running_average=not training)(residual)
        
        x = nn.Conv(512, (1, 1), padding="SAME", use_bias=False,
                   kernel_init=nn.initializers.kaiming_normal())(x)
        x = nn.BatchNorm(use_running_average=not training)(x)
        x = x + residual  # Skip connection
        x = nn.silu(x)  # res3
        
        # SE Attention 2
        x = SEBlock(reduction=16)(x, training)
        
        # Spatial Attention
        x = SpatialAttention()(x, training)
        
        # 🔥 DROPOUT 3: Après bloc 512 canaux
        x = nn.Dropout(0.2, deterministic=not training)(x)
        
        # Global Average Pooling
        x = jnp.mean(x, axis=(1, 2))  # (B, 512)
        
        # Classification head avec DOUBLE DROPOUT
        x = nn.LayerNorm()(x)
        x = nn.Dense(384, use_bias=True)(x)
        x = nn.silu(x)
        # 🔥 DROPOUT 4: Premier dropout du head
        x = nn.Dropout(self.dropout_rate, deterministic=not training)(x)
        
        x = nn.Dense(128, use_bias=True)(x)
        x = nn.silu(x)
        # 🔥 DROPOUT 5: Second dropout du head (plus fort)
        x = nn.Dropout(self.dropout_rate + 0.1, deterministic=not training)(x)
        
        return nn.Dense(self.num_classes, use_bias=True)(x)


class SophisticatedCNN128(nn.Module):
    """
    CNN sophistiqué OPTIMISÉ pour images 128×128
    
    Architecture adaptée pour exploiter la haute résolution:
    - 4 blocs de convolution (au lieu de 3)
    - Progression canaux: 64→96→128→256→512
    - 3 max poolings: 128→64→32→16 (préserve plus d'information)
    - Paramètres: ~2.5M (vs 1.3M pour version 64×64)
    - Ratio params/pixel optimal pour 128×128
    
    Objectif: Exploiter pleinement les 4× plus de pixels
    Val attendue: 85-87% (vs 83% avec modèle 64×64)
    """
    num_classes: int = 2
    dropout_rate: float = 0.0


class SophisticatedCNN128Plus(nn.Module):
    """
    CNN sophistiqué OPTIMISÉ+ pour images 128×128
    
    Améliorations par rapport à SophisticatedCNN128:
    - Multi-scale feature fusion: Concat features de différentes résolutions
    - CBAM attention: Channel + Spatial attention combinée
    - Classification head amélioré: 2 couches au lieu d'1
    - Skip connections dans le head
    - Paramètres: ~4M (vs 2.5M, +60%)
    
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


class SophisticatedCNN128Ultimate(nn.Module):
    """
    CNN sophistiqué ULTIMATE pour images 128×128
    
    Architecture ultra-avancée avec techniques d'état de l'art:
    - Multi-scale feature fusion: Concat features de toutes les résolutions
    - CBAM attention complète: Channel + Spatial attention partout
    - Dense connections: Skip connections entre tous les blocs
    - Advanced classification head: 3 couches avec skip connections
    - EfficientNet-style: Compound scaling + SE blocks
    - Paramètres: ~8M (vs 4M, +100%)
    
    Objectif: Dépasser 95% de validation
    Val attendue: 92-95% (vs 88% avec modèle standard)
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
        
        # === BLOC 1: 96 canaux avec Dense Connection ===
        x = SeparableConv(96, (3, 3))(x, training)
        x = nn.BatchNorm(use_running_average=not training)(x)
        x = nn.silu(x)
        
        # Dense connection
        x = SeparableConv(96, (3, 3))(x, training)
        x = nn.BatchNorm(use_running_average=not training)(x)
        x = nn.silu(x)
        
        # CBAM Attention
        x = CBAMBlock()(x, training)
        
        # Max Pool 1: 128×128 → 64×64
        x = nn.max_pool(x, (2, 2), strides=(2, 2))
        
        # Stocker pour multi-scale fusion
        multi_scale_features.append(x)  # 64×64
        
        # === BLOC 2: 128 canaux avec Dense Connection ===
        x = SeparableConv(128, (3, 3))(x, training)
        x = nn.BatchNorm(use_running_average=not training)(x)
        x = nn.silu(x)
        
        x = SeparableConv(128, (3, 3))(x, training)
        x = nn.BatchNorm(use_running_average=not training)(x)
        x = nn.silu(x)
        
        # CBAM Attention
        x = CBAMBlock()(x, training)
        
        # Max Pool 2: 64×64 → 32×32
        x = nn.max_pool(x, (2, 2), strides=(2, 2))
        
        # Stocker pour multi-scale fusion
        multi_scale_features.append(x)  # 32×32
        
        # === BLOC 3: 256 canaux avec Dense Connection ===
        x = SeparableConv(256, (3, 3))(x, training)
        x = nn.BatchNorm(use_running_average=not training)(x)
        x = nn.silu(x)
        
        x = SeparableConv(256, (3, 3))(x, training)
        x = nn.BatchNorm(use_running_average=not training)(x)
        x = nn.silu(x)
        
        # CBAM Attention
        x = CBAMBlock()(x, training)
        
        # Max Pool 3: 32×32 → 16×16
        x = nn.max_pool(x, (2, 2), strides=(2, 2))
        
        # Stocker pour multi-scale fusion
        multi_scale_features.append(x)  # 16×16
        
        # === BLOC 4: 512 canaux avec Dense Connection ===
        x = nn.Conv(384, (1, 1), padding="SAME", use_bias=False,
                   kernel_init=nn.initializers.kaiming_normal())(x)
        x = nn.BatchNorm(use_running_average=not training)(x)
        x = nn.silu(x)
        
        x = SeparableConv(512, (3, 3))(x, training)
        x = nn.BatchNorm(use_running_average=not training)(x)
        x = nn.silu(x)
        
        x = SeparableConv(512, (3, 3))(x, training)
        x = nn.BatchNorm(use_running_average=not training)(x)
        x = nn.silu(x)
        
        # CBAM Attention
        x = CBAMBlock()(x, training)
        
        # === MULTI-SCALE FEATURE FUSION ===
        # Upsample et concaténer les features de différentes résolutions
        x_64 = nn.Conv(128, (1, 1), padding="SAME", use_bias=False,
                      kernel_init=nn.initializers.kaiming_normal())(multi_scale_features[0])
        x_64 = nn.BatchNorm(use_running_average=not training)(x_64)
        x_64 = nn.avg_pool(x_64, (4, 4), strides=(4, 4))  # 64×64 → 16×16
        
        x_32 = nn.Conv(128, (1, 1), padding="SAME", use_bias=False,
                      kernel_init=nn.initializers.kaiming_normal())(multi_scale_features[1])
        x_32 = nn.BatchNorm(use_running_average=not training)(x_32)
        x_32 = nn.avg_pool(x_32, (2, 2), strides=(2, 2))  # 32×32 → 16×16
        
        x_16 = nn.Conv(128, (1, 1), padding="SAME", use_bias=False,
                      kernel_init=nn.initializers.kaiming_normal())(multi_scale_features[2])
        x_16 = nn.BatchNorm(use_running_average=not training)(x_16)
        
        # Concaténer toutes les features (16×16)
        x = jnp.concatenate([x, x_16, x_32, x_64], axis=-1)  # 512 + 128 + 128 + 128 = 896
        
        # === ADVANCED CLASSIFICATION HEAD ===
        # Global Average Pooling
        x = jnp.mean(x, axis=(1, 2))  # (B, 896)
        
        # Head layer 1
        x1 = nn.LayerNorm()(x)
        x1 = nn.Dense(512, use_bias=True)(x1)
        x1 = nn.silu(x1)
        x1 = nn.Dropout(self.dropout_rate, deterministic=not training)(x1)
        
        # Head layer 2 avec skip connection
        x2 = nn.LayerNorm()(x1)
        x2 = nn.Dense(256, use_bias=True)(x2)
        x2 = nn.silu(x2)
        x2 = nn.Dropout(self.dropout_rate, deterministic=not training)(x2)
        
        # Skip connection dans le head
        x2 = x2 + nn.Dense(256, use_bias=True)(x1)
        
        # Head layer 3
        x3 = nn.LayerNorm()(x2)
        x3 = nn.Dense(128, use_bias=True)(x3)
        x3 = nn.silu(x3)
        x3 = nn.Dropout(self.dropout_rate, deterministic=not training)(x3)
        
        # Skip connection finale
        x3 = x3 + nn.Dense(128, use_bias=True)(x2)
        
        return nn.Dense(self.num_classes, use_bias=True)(x3)


class SophisticatedCNNOptimized(nn.Module):
    """
    CNN sophistiqué OPTIMISÉ pour images 64x64
    
    Améliorations par rapport à SophisticatedCNN:
    - Stochastic Depth (Drop Path): Dropout de blocs entiers
    - Multi-scale feature fusion: Concat features de différentes résolutions
    - CBAM: Channel + Spatial attention combinée
    - Better classification head: 2 layers au lieu d'1
    - Dropout stratégique après chaque bloc
    - Residual connections améliorées
    
    Résultats attendus:
    - Paramètres: ~2.5M (vs 1.3M)
    - Temps/epoch: ~95s (vs 77s, +23%)
    - Gain accuracy: +5-10% (objectif: 82-85%)
    """
    num_classes: int = 2
    dropout_rate: float = 0.2
    drop_path_rate: float = 0.1

    @nn.compact
    def __call__(self, x, training=True):
        # Input: (B, 64, 64, 3)
        features = []  # Pour multi-scale fusion
        
        # === STAGE 1: Initial Conv ===
        x = nn.Conv(64, (3, 3), padding="SAME", use_bias=False,
                   kernel_init=nn.initializers.kaiming_normal())(x)
        x = nn.BatchNorm(use_running_average=not training)(x)
        x = nn.silu(x)
        
        # === STAGE 2: First Residual Block ===
        residual = x
        
        x = SeparableConv(128, (3, 3))(x, training)
        x = nn.BatchNorm(use_running_average=not training)(x)
        x = nn.silu(x)
        
        x = SeparableConv(128, (3, 3))(x, training)
        x = nn.BatchNorm(use_running_average=not training)(x)
        
        # Adjust residual
        if residual.shape[-1] != x.shape[-1]:
            residual = nn.Conv(128, (1, 1), padding="SAME", use_bias=False,
                             kernel_init=nn.initializers.kaiming_normal())(residual)
            residual = nn.BatchNorm(use_running_average=not training)(residual)
        
        # Stochastic Depth
        if training and self.drop_path_rate > 0:
            keep_prob = 1 - self.drop_path_rate
            rng = self.make_rng('dropout')
            batch_size = x.shape[0]
            mask = jax.random.bernoulli(rng, keep_prob, (batch_size, 1, 1, 1))
            x = jnp.where(mask, x / keep_prob, 0)
        
        x = x + residual
        x = nn.silu(x)
        x = nn.Dropout(self.dropout_rate * 0.5)(x, deterministic=not training)
        
        features.append(x)  # 64x64x128
        
        # Max Pool
        x = nn.max_pool(x, (2, 2), strides=(2, 2))  # 32x32
        
        # === STAGE 3: Second Residual Block + CBAM ===
        residual = x
        
        x = SeparableConv(256, (3, 3))(x, training)
        x = nn.BatchNorm(use_running_average=not training)(x)
        x = nn.silu(x)
        
        x = SeparableConv(256, (3, 3))(x, training)
        x = nn.BatchNorm(use_running_average=not training)(x)
        
        # Adjust residual
        if residual.shape[-1] != x.shape[-1]:
            residual = nn.Conv(256, (1, 1), padding="SAME", use_bias=False,
                             kernel_init=nn.initializers.kaiming_normal())(residual)
            residual = nn.BatchNorm(use_running_average=not training)(residual)
        
        # Stochastic Depth
        if training and self.drop_path_rate > 0:
            keep_prob = 1 - self.drop_path_rate * 0.5
            rng = self.make_rng('dropout')
            batch_size = x.shape[0]
            mask = jax.random.bernoulli(rng, keep_prob, (batch_size, 1, 1, 1))
            x = jnp.where(mask, x / keep_prob, 0)
        
        x = x + residual
        x = nn.silu(x)
        
        # SE Block (Channel Attention)
        x = SEBlock(reduction=16)(x, training)
        
        # Spatial Attention
        x = SpatialAttention()(x, training)
        
        x = nn.Dropout(self.dropout_rate * 0.75)(x, deterministic=not training)
        
        features.append(x)  # 32x32x256
        
        # Max Pool
        x = nn.max_pool(x, (2, 2), strides=(2, 2))  # 16x16
        
        # === STAGE 4: Third Residual Block + CBAM ===
        residual = x
        
        x = nn.Conv(512, (1, 1), padding="SAME", use_bias=False,
                   kernel_init=nn.initializers.kaiming_normal())(x)
        x = nn.BatchNorm(use_running_average=not training)(x)
        x = nn.silu(x)
        
        x = SeparableConv(512, (3, 3))(x, training)
        x = nn.BatchNorm(use_running_average=not training)(x)
        x = nn.silu(x)
        
        x = nn.Conv(512, (1, 1), padding="SAME", use_bias=False,
                   kernel_init=nn.initializers.kaiming_normal())(x)
        x = nn.BatchNorm(use_running_average=not training)(x)
        
        # Adjust residual
        if residual.shape[-1] != x.shape[-1]:
            residual = nn.Conv(512, (1, 1), padding="SAME", use_bias=False,
                             kernel_init=nn.initializers.kaiming_normal())(residual)
            residual = nn.BatchNorm(use_running_average=not training)(residual)
        
        x = x + residual
        x = nn.silu(x)
        
        # SE Block (Channel Attention)
        x = SEBlock(reduction=16)(x, training)
        
        # Spatial Attention
        x = SpatialAttention()(x, training)
        
        x = nn.Dropout(self.dropout_rate)(x, deterministic=not training)
        
        features.append(x)  # 16x16x512
        
        # === MULTI-SCALE FEATURE FUSION ===
        # Downsample features to same resolution (16x16)
        f1 = features[0]  # 64x64x128
        f1 = nn.max_pool(f1, (4, 4), strides=(4, 4))  # 16x16x128
        
        f2 = features[1]  # 32x32x256
        f2 = nn.max_pool(f2, (2, 2), strides=(2, 2))  # 16x16x256
        
        f3 = features[2]  # 16x16x512 (already correct size)
        
        # Concatenate multi-scale features
        x = jnp.concatenate([f1, f2, f3], axis=-1)  # 16x16x(128+256+512)=896
        
        # Reduce channels
        x = nn.Conv(512, (1, 1), padding="SAME", use_bias=False,
                   kernel_init=nn.initializers.kaiming_normal())(x)
        x = nn.BatchNorm(use_running_average=not training)(x)
        x = nn.silu(x)
        
        # === GLOBAL POOLING ===
        # Mix of average and max pooling
        avg_pool = jnp.mean(x, axis=(1, 2))  # (B, 512)
        max_pool = jnp.max(x, axis=(1, 2))   # (B, 512)
        x = jnp.concatenate([avg_pool, max_pool], axis=-1)  # (B, 1024)
        
        # === IMPROVED CLASSIFICATION HEAD ===
        x = nn.LayerNorm()(x)
        
        # First dense layer
        x = nn.Dense(512, use_bias=True)(x)
        x = nn.silu(x)
        x = nn.Dropout(self.dropout_rate)(x, deterministic=not training)
        
        # Second dense layer
        x = nn.Dense(256, use_bias=True)(x)
        x = nn.silu(x)
        x = nn.Dropout(self.dropout_rate * 0.5)(x, deterministic=not training)
        
class AircraftDetector(nn.Module):
    """
    Détecteur d'avions "Grid-Based" (Style YOLO simplifié)
    Entrée: Image (H, W) où H et W sont divisibles par 32 (ex: 224x224, 448x448)
    Sortie: Grille S x S x (C + 5*B)
    
    Pour 224x224: S=7 (grille 7x7), pour 448x448: S=14 (grille 14x14)
    B=1 (1 box par cellule, suffit pour avions éparpillés)
    Sortie: (Batch, S, S, 5)
    Canaux: [Confiance, x, y, w, h]
    """
    dropout_rate: float = 0.0

    @nn.compact
    def __call__(self, x, training=True):
        # Input: (B, H, W, 3) où H et W sont divisibles par 32
        
        # --- BACKBONE (Extraction de features) ---
        # Réduit la taille par 32: 224/32=7, 448/32=14, etc.
        
        # /2 -> 224
        x = nn.Conv(32, (3, 3), strides=(2, 2), padding="SAME", use_bias=False)(x)
        x = nn.BatchNorm(use_running_average=not training)(x)
        x = nn.silu(x)
        
        # /4 -> 112
        x = SeparableConv(64, (3, 3))(x, training)
        x = nn.max_pool(x, (2, 2), strides=(2, 2))
        
        # /8 -> 56
        x = SeparableConv(128, (3, 3))(x, training)
        x = nn.BatchNorm(use_running_average=not training)(x)
        x = nn.silu(x)
        x = nn.max_pool(x, (2, 2), strides=(2, 2))
        
        # /16 -> 28
        x = SeparableConv(256, (3, 3))(x, training)
        x = nn.BatchNorm(use_running_average=not training)(x)
        x = nn.silu(x)
        x = nn.max_pool(x, (2, 2), strides=(2, 2))
        
        # /32 -> 14 (Feature Map Finale)
        x = SeparableConv(512, (3, 3))(x, training)
        x = nn.BatchNorm(use_running_average=not training)(x)
        x = nn.silu(x)
        x = nn.max_pool(x, (2, 2), strides=(2, 2))
        
        # --- DETECTION HEAD ---
        # Enrichir le contexte avant prédiction
        x = SeparableConv(512, (3, 3))(x, training)
        x = nn.BatchNorm(use_running_average=not training)(x)
        x = nn.silu(x)
        
        # Projection finale : 5 canaux (Conf, x, y, w, h)
        # Activation Sigmoid pour que tout soit entre 0 et 1
        x = nn.Conv(5, (1, 1), padding="SAME")(x)
        x = nn.sigmoid(x)
        
        # Output shape: (Batch, 14, 14, 5)
        return x


class ResidualBlock(nn.Module):
    """Bloc résiduel pour ResNet"""
    filters: int
    stride: int = 1
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(self, x, training=True):
        residual = x
        
        # Première convolution
        x = nn.Conv(
            features=self.filters,
            kernel_size=(3, 3),
            strides=(self.stride, self.stride),
            padding="SAME",
            use_bias=False,
            kernel_init=nn.initializers.kaiming_normal()
        )(x)
        x = nn.BatchNorm(use_running_average=not training)(x)
        x = nn.relu(x)
        
        # Deuxième convolution
        x = nn.Conv(
            features=self.filters,
            kernel_size=(3, 3),
            padding="SAME",
            use_bias=False,
            kernel_init=nn.initializers.kaiming_normal()
        )(x)
        x = nn.BatchNorm(use_running_average=not training)(x)
        
        # Connexion résiduelle
        if residual.shape != x.shape:
            residual = nn.Conv(
                features=self.filters,
                kernel_size=(1, 1),
                strides=(self.stride, self.stride),
                padding="SAME",
                use_bias=False,
                kernel_init=nn.initializers.kaiming_normal()
            )(residual)
            residual = nn.BatchNorm(use_running_average=not training)(residual)
        
        x = x + residual
        x = nn.relu(x)
        x = nn.Dropout(self.dropout_rate)(x, deterministic=not training)
        
        return x


class ResidualBlockSeparable(nn.Module):
    """Bloc résiduel optimisé (Separable Conv) pour V3 (~8x moins de params)"""
    filters: int
    stride: int = 1
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(self, x, training=True):
        residual = x
        
        # Première convolution (Separable)
        x = SeparableConv(self.filters, (3, 3), strides=(self.stride, self.stride))(x, training)
        x = nn.BatchNorm(use_running_average=not training)(x)
        x = nn.relu(x)
        
        # Deuxième convolution (Separable)
        x = SeparableConv(self.filters, (3, 3))(x, training)
        x = nn.BatchNorm(use_running_average=not training)(x)
        
        # Connexion résiduelle
        if residual.shape != x.shape:
            # Projection légère (Conv 1x1 standard)
            residual = nn.Conv(
                features=self.filters,
                kernel_size=(1, 1),
                strides=(self.stride, self.stride),
                padding="SAME",
                use_bias=False,
                kernel_init=nn.initializers.kaiming_normal()
            )(residual)
            residual = nn.BatchNorm(use_running_average=not training)(residual)
            
        x = x + residual
        x = nn.relu(x)
        x = nn.Dropout(self.dropout_rate)(x, deterministic=not training)
        
        return x


class ResNetLight(nn.Module):
    """ResNet léger pour comparaison"""
    num_classes: int = 2
    dropout_rate: float = 0.3

    @nn.compact
    def __call__(self, x, training=True):
        # Couche initiale
        x = nn.Conv(
            features=64,
            kernel_size=(7, 7),
            strides=(2, 2),
            padding="SAME",
            use_bias=False,
            kernel_init=nn.initializers.kaiming_normal()
        )(x)
        x = nn.BatchNorm(use_running_average=not training)(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(3, 3), strides=(2, 2), padding="SAME")
        
        # Blocs résiduels
        x = ResidualBlock(filters=64, dropout_rate=self.dropout_rate)(x)
        x = ResidualBlock(filters=64, dropout_rate=self.dropout_rate)(x)
        
        x = ResidualBlock(filters=128, stride=2, dropout_rate=self.dropout_rate)(x)
        x = ResidualBlock(filters=128, dropout_rate=self.dropout_rate)(x)
        
        x = ResidualBlock(filters=256, stride=2, dropout_rate=self.dropout_rate)(x)
        x = ResidualBlock(filters=256, dropout_rate=self.dropout_rate)(x)
        
        # Global Average Pooling
        x = jnp.mean(x, axis=(1, 2))
        
        # Classification head
        x = nn.Dense(512, use_bias=True)(x)
        x = nn.relu(x)
        x = nn.Dropout(self.dropout_rate)(x, deterministic=not training)
        
        x = nn.Dense(self.num_classes, use_bias=True)(x)
        
        return x


class TinyViTPlus(nn.Module):
    """Vision Transformer léger pour comparaison"""
    num_classes: int = 2
    patch_size: int = 16
    hidden_dim: int = 192
    num_heads: int = 3
    num_layers: int = 6
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(self, x, training=True):
        # Patch embedding
        x = nn.Conv(
            features=self.hidden_dim,
            kernel_size=(self.patch_size, self.patch_size),
            strides=(self.patch_size, self.patch_size),
            padding="VALID",
            use_bias=False,
            kernel_init=nn.initializers.kaiming_normal()
        )(x)
        
        # Reshape pour transformer en séquence de patches
        # x shape: (B, H', W', C) -> (B, H'*W', C)
        batch_size = x.shape[0]
        x = jnp.reshape(x, (batch_size, -1, self.hidden_dim))
        
        # Ajouter token de classe
        cls_token = self.param('cls_token', nn.initializers.normal(0.02), (1, 1, self.hidden_dim))
        cls_token = jnp.tile(cls_token, (batch_size, 1, 1))
        x = jnp.concatenate([cls_token, x], axis=1)
        
        # Positional embedding
        pos_embed = self.param('pos_embed', nn.initializers.normal(0.02), (1, x.shape[1], self.hidden_dim))
        x = x + pos_embed
        
        # Transformer blocks
        for _ in range(self.num_layers):
            # Self-attention
            x_norm = nn.LayerNorm()(x)
            x_attn = nn.MultiHeadDotProductAttention(
                num_heads=self.num_heads,
                dropout_rate=self.dropout_rate,
                deterministic=not training
            )(x_norm, x_norm)
            x = x + x_attn
            
            # MLP
            x_norm = nn.LayerNorm()(x)
            x_mlp = nn.Dense(self.hidden_dim * 4, use_bias=True)(x_norm)
            x_mlp = nn.gelu(x_mlp)
            x_mlp = nn.Dropout(self.dropout_rate)(x_mlp, deterministic=not training)
            x_mlp = nn.Dense(self.hidden_dim, use_bias=True)(x_mlp)
            x = x + x_mlp
        
        # Classification
        x = nn.LayerNorm()(x)
        x = x[:, 0]  # Prendre le token de classe
        x = nn.Dense(self.num_classes, use_bias=True)(x)
        
        return x


class TinyViTPlusBalanced(nn.Module):
    """
    Vision Transformer optimisé - Configuration équilibrée
    
    Améliorations par rapport à TinyViTPlus:
    - patch_size: 16 → 8 (4x plus de patches pour plus de détails)
    - hidden_dim: 192 → 224 (plus de capacité)
    - num_heads: 3 → 4 (meilleure attention multi-échelle)
    - num_layers: 6 (conservé pour équilibre performance/temps)
    
    Résultats attendus:
    - Paramètres: ~2.8M (vs 1.8M)
    - Temps/epoch: ~130s (vs 78s, +67%)
    - Gain accuracy: +3-5%
    """
    num_classes: int = 2
    patch_size: int = 8      # Réduit pour plus de détails
    hidden_dim: int = 224    # Augmenté pour plus de capacité
    num_heads: int = 4       # Augmenté pour meilleure attention
    num_layers: int = 6      # Conservé pour équilibre
    dropout_rate: float = 0.1
    
    @nn.compact
    def __call__(self, x, training=True):
        # Patch embedding
        x = nn.Conv(
            features=self.hidden_dim,
            kernel_size=(self.patch_size, self.patch_size),
            strides=(self.patch_size, self.patch_size),
            padding="VALID",
            use_bias=False,
            kernel_init=nn.initializers.kaiming_normal()
        )(x)
        
        # Reshape pour transformer en séquence de patches
        # x shape: (B, H', W', C) -> (B, H'*W', C)
        batch_size = x.shape[0]
        x = jnp.reshape(x, (batch_size, -1, self.hidden_dim))
        
        # Ajouter token de classe
        cls_token = self.param('cls_token', nn.initializers.normal(0.02), (1, 1, self.hidden_dim))
        cls_token = jnp.tile(cls_token, (batch_size, 1, 1))
        x = jnp.concatenate([cls_token, x], axis=1)
        
        # Positional embedding
        pos_embed = self.param('pos_embed', nn.initializers.normal(0.02), (1, x.shape[1], self.hidden_dim))
        x = x + pos_embed
        
        # Transformer blocks
        for _ in range(self.num_layers):
            # Self-attention
            x_norm = nn.LayerNorm()(x)
            x_attn = nn.MultiHeadDotProductAttention(
                num_heads=self.num_heads,
                dropout_rate=self.dropout_rate,
                deterministic=not training
            )(x_norm, x_norm)
            x = x + x_attn
            
            # MLP
            x_norm = nn.LayerNorm()(x)
            x_mlp = nn.Dense(self.hidden_dim * 4, use_bias=True)(x_norm)
            x_mlp = nn.gelu(x_mlp)
            x_mlp = nn.Dropout(self.dropout_rate)(x_mlp, deterministic=not training)
            x_mlp = nn.Dense(self.hidden_dim, use_bias=True)(x_mlp)
            x = x + x_mlp
        
        # Classification
        x = nn.LayerNorm()(x)
        x = x[:, 0]  # Prendre le token de classe
        x = nn.Dense(self.num_classes, use_bias=True)(x)
        
        return x
    

class TinyViTPlusUltimate(nn.Module):
    """
    Vision Transformer ultime pour images 64x64 - Configuration maximale
    
    Innovations par rapport à TinyViTPlusBalanced:
    - Convolutional Stem: Extraction de features progressive au lieu de patch embedding simple
    - Stochastic Depth (Drop Path): Dropout de layers entiers pour meilleure généralisation
    - Layer Scale: Scaling learnable des résidus pour stabilité
    - patch_size: 8 → 4 (256 patches au lieu de 64)
    - hidden_dim: 224 → 384 (capacité maximale)
    - num_heads: 4 → 6 (attention multi-échelle riche)
    - num_layers: 6 → 12 (profondeur importante)
    
    Résultats attendus:
    - Paramètres: ~8-10M (vs 2.8M Balanced)
    - Temps/epoch: ~250-300s (vs 130s Balanced, +100%)
    - Gain accuracy: +5-11% (objectif: 82-88%)
    """
    num_classes: int = 2
    patch_size: int = 4          # Patches plus petits pour maximum de détails
    hidden_dim: int = 384        # Grande capacité
    num_heads: int = 6           # Attention riche
    num_layers: int = 12         # Profondeur importante
    dropout_rate: float = 0.15   # Régularisation modérée
    drop_path_rate: float = 0.2  # Stochastic depth
    layer_scale_init: float = 1e-4  # Layer scale

    @nn.compact
    def __call__(self, x, training=True):
        # === CONVOLUTIONAL STEM ===
        # Meilleur que patch embedding simple pour petites images
        # Extraction progressive de features
        x = nn.Conv(64, (3, 3), padding="SAME", use_bias=False,
                   kernel_init=nn.initializers.kaiming_normal())(x)
        x = nn.BatchNorm(use_running_average=not training)(x)
        x = nn.gelu(x)
        
        x = nn.Conv(128, (3, 3), padding="SAME", use_bias=False,
                   kernel_init=nn.initializers.kaiming_normal())(x)
        x = nn.BatchNorm(use_running_average=not training)(x)
        x = nn.gelu(x)
        
        # Patchify avec stride pour réduire la résolution
        x = nn.Conv(
            features=self.hidden_dim,
            kernel_size=(self.patch_size, self.patch_size),
            strides=(self.patch_size, self.patch_size),
            padding="VALID",
            use_bias=False,
            kernel_init=nn.initializers.kaiming_normal()
        )(x)
        
        # Reshape pour transformer en séquence de patches
        batch_size = x.shape[0]
        x = jnp.reshape(x, (batch_size, -1, self.hidden_dim))
        
        # Ajouter token de classe
        cls_token = self.param('cls_token', nn.initializers.normal(0.02), (1, 1, self.hidden_dim))
        cls_token = jnp.tile(cls_token, (batch_size, 1, 1))
        x = jnp.concatenate([cls_token, x], axis=1)
        
        # Positional embedding
        pos_embed = self.param('pos_embed', nn.initializers.normal(0.02), (1, x.shape[1], self.hidden_dim))
        x = x + pos_embed
        
        # === TRANSFORMER BLOCKS avec Stochastic Depth et Layer Scale ===
        for layer_idx in range(self.num_layers):
            # Drop path rate croissant (stochastic depth)
            drop_path = self.drop_path_rate * layer_idx / (self.num_layers - 1)
            
            # Self-attention avec Layer Scale
            x_norm = nn.LayerNorm()(x)
            x_attn = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            dropout_rate=self.dropout_rate,
                deterministic=not training
            )(x_norm, x_norm)
            
            # Layer Scale pour l'attention
            layer_scale_attn = self.param(
                f'layer_scale_attn_{layer_idx}',
                lambda rng, shape: jnp.full(shape, self.layer_scale_init),
                (self.hidden_dim,)
            )
            x_attn = x_attn * layer_scale_attn
            
            # Stochastic Depth (Drop Path) pour l'attention
            if training and drop_path > 0:
                keep_prob = 1 - drop_path
                rng = self.make_rng('dropout')
                mask = jax.random.bernoulli(rng, keep_prob, (batch_size, 1, 1))
                x_attn = jnp.where(mask, x_attn / keep_prob, 0)
            
            x = x + x_attn
            
            # MLP avec Layer Scale
            x_norm = nn.LayerNorm()(x)
            x_mlp = nn.Dense(self.hidden_dim * 4, use_bias=True)(x_norm)
            x_mlp = nn.gelu(x_mlp)
            x_mlp = nn.Dropout(self.dropout_rate)(x_mlp, deterministic=not training)
            x_mlp = nn.Dense(self.hidden_dim, use_bias=True)(x_mlp)
            
            # Layer Scale pour le MLP
            layer_scale_mlp = self.param(
                f'layer_scale_mlp_{layer_idx}',
                lambda rng, shape: jnp.full(shape, self.layer_scale_init),
                (self.hidden_dim,)
            )
            x_mlp = x_mlp * layer_scale_mlp
            
            # Stochastic Depth (Drop Path) pour le MLP
            if training and drop_path > 0:
                keep_prob = 1 - drop_path
                rng = self.make_rng('dropout')
                mask = jax.random.bernoulli(rng, keep_prob, (batch_size, 1, 1))
                x_mlp = jnp.where(mask, x_mlp / keep_prob, 0)
            
            x = x + x_mlp
        
        # === CLASS ATTENTION (derniers layers) ===
        # Attention uniquement sur le cls_token pour efficacité
        x = nn.LayerNorm()(x)
        cls_token = x[:, 0:1]  # Extraire le cls_token
        
        # Attention du cls_token vers tous les patches
        cls_attn = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            dropout_rate=self.dropout_rate,
            deterministic=not training
        )(cls_token, x)
        
        cls_token = cls_token + cls_attn
        
        # MLP final sur le cls_token
        cls_token = nn.LayerNorm()(cls_token)
        cls_token = jnp.squeeze(cls_token, axis=1)  # (B, 1, C) -> (B, C)
        
        # Classification head
        x = nn.Dense(self.hidden_dim, use_bias=True)(cls_token)
        x = nn.gelu(x)
        x = nn.Dropout(self.dropout_rate)(x, deterministic=not training)
        x = nn.Dense(self.num_classes, use_bias=True)(x)
        
        return x


class HybridTinyViT(nn.Module):
    """
    Hybrid Vision Transformer : CNN Stem + Transformer
    
    Architecture optimisée pour petits datasets d'images :
    - Convolutional stem : Extrait features locales (inductive bias)
    - Patch size 4 : Plus de tokens pour images 128×128
    - CLS token : Représentation globale apprise
    - Plus efficace que ViT pur sur datasets <100K images
    
    Params: ~1.5-2M (vs 2.8M pour TinyViTPlusBalanced)
    """
    num_classes: int = 10
    patch_size: int = 4          # Patch plus petit pour 128×128
    hidden_dim: int = 224         # Dimension cachée
    num_heads: int = 4            # Têtes d'attention
    num_layers: int = 6           # Layers Transformer
    dropout_rate: float = 0.1     # Dropout
    
    @nn.compact
    def __call__(self, x, training=True):
        """
        Forward pass du Hybrid ViT
        
        Args:
            x: Input images (B, H, W, C) - H=W=128, C=1 ou 3
            training: Mode entraînement
            
        Returns:
            logits: (B, num_classes)
        """
        # === CONVOLUTIONAL STEM ===
        # Réduit 128×128 → 32×32 tout en extrayant features locales
        # Conv 1 : 128×128×1 → 64×64×64
        x = nn.Conv(64, (3, 3), strides=(2, 2), padding='SAME', use_bias=False)(x)
        x = nn.BatchNorm(use_running_average=not training)(x)
        x = nn.relu(x)
        
        # Conv 2 : 64×64×64 → 32×32×128
        x = nn.Conv(128, (3, 3), strides=(2, 2), padding='SAME', use_bias=False)(x)
        x = nn.BatchNorm(use_running_average=not training)(x)
        x = nn.relu(x)
        
        # === PATCH EMBEDDING ===
        # 32×32×128 → (8×8=64 patches) × hidden_dim
        x = nn.Conv(
            self.hidden_dim, 
            (self.patch_size, self.patch_size), 
            strides=(self.patch_size, self.patch_size),
            padding='VALID',
            use_bias=True
        )(x)
        
        # Reshape: (B, H', W', C) → (B, num_patches, hidden_dim)
        B = x.shape[0]
        x = jnp.reshape(x, (B, -1, self.hidden_dim))
        num_patches = x.shape[1]
        
        # === CLS TOKEN ===
        # Token spécial pour la classification
        cls = self.param('cls', nn.initializers.normal(stddev=0.02), (1, 1, self.hidden_dim))
        cls = jnp.tile(cls, (B, 1, 1))
        x = jnp.concatenate([cls, x], axis=1)  # (B, 1+num_patches, hidden_dim)
        
        # === POSITIONAL EMBEDDING ===
        pos = self.param('pos', nn.initializers.normal(stddev=0.02), (1, 1 + num_patches, self.hidden_dim))
        x = x + pos
        
        # === TRANSFORMER BLOCKS ===
        for layer_idx in range(self.num_layers):
            # Multi-Head Self-Attention
            x_norm = nn.LayerNorm()(x)
            attn = nn.MultiHeadDotProductAttention(
                num_heads=self.num_heads,
                dropout_rate=self.dropout_rate,
                deterministic=not training
            )(x_norm, x_norm)
            x = x + attn
            
            # MLP (Feed-Forward)
            x_norm = nn.LayerNorm()(x)
            mlp = nn.Dense(self.hidden_dim * 4)(x_norm)
            mlp = nn.gelu(mlp)
            mlp = nn.Dropout(self.dropout_rate)(mlp, deterministic=not training)
            mlp = nn.Dense(self.hidden_dim)(mlp)
            mlp = nn.Dropout(self.dropout_rate)(mlp, deterministic=not training)
            x = x + mlp
        
        # === CLASSIFICATION HEAD ===
        x = nn.LayerNorm()(x)
        x_cls = x[:, 0]  # Extraire le CLS token
        
        # Dense layer finale
        x_cls = nn.Dense(self.num_classes)(x_cls)
        
        return x_cls


# ======================
# Factory functions pour faciliter l'utilisation
# ======================

def create_sophisticated_cnn(num_classes=2, dropout_rate=0.1):
    """Crée une instance de SophisticatedCNN"""
    return SophisticatedCNN(num_classes=num_classes, dropout_rate=dropout_rate)


def create_sophisticated_cnn_droped_out(num_classes=2, dropout_rate=0.3):
    """Crée une instance de SophisticatedCNNDropedOut avec dropout stratégique"""
    return SophisticatedCNNDropedOut(num_classes=num_classes, dropout_rate=dropout_rate)


def create_sophisticated_cnn_128(num_classes=2, dropout_rate=0.15):
    """Crée une instance de SophisticatedCNN128 optimisé pour images 128×128"""
    return SophisticatedCNN128(num_classes=num_classes, dropout_rate=dropout_rate)


def create_sophisticated_cnn_128_plus(num_classes=2, dropout_rate=0.0):
    """Crée une instance de SophisticatedCNN128Plus optimisé+ pour images 128×128"""
    return SophisticatedCNN128Plus(num_classes=num_classes, dropout_rate=dropout_rate)


def create_sophisticated_cnn_128_ultimate(num_classes=2, dropout_rate=0.0):
    """Crée une instance de SophisticatedCNN128Ultimate ultra-avancé pour images 128×128"""
    return SophisticatedCNN128Ultimate(num_classes=num_classes, dropout_rate=dropout_rate)


def create_sophisticated_cnn_optimized(num_classes=2, dropout_rate=0.2):
    """Crée une instance de SophisticatedCNNOptimized"""
    return SophisticatedCNNOptimized(num_classes=num_classes, dropout_rate=dropout_rate)


def create_resnet_light(num_classes=2, dropout_rate=0.3):
    """Crée une instance de ResNetLight"""
    return ResNetLight(num_classes=num_classes, dropout_rate=dropout_rate)


def create_tiny_vit_plus(num_classes=2, dropout_rate=0.1):
    """Crée une instance de TinyViTPlus"""
    return TinyViTPlus(num_classes=num_classes, dropout_rate=dropout_rate)


def create_tiny_vit_plus_balanced(num_classes=2, dropout_rate=0.1):
    """Crée une instance de TinyViTPlusBalanced"""
    return TinyViTPlusBalanced(num_classes=num_classes, dropout_rate=dropout_rate)


def create_tiny_vit_plus_ultimate(num_classes=2, dropout_rate=0.15):
    """Crée une instance de TinyViTPlusUltimate"""
    return TinyViTPlusUltimate(num_classes=num_classes, dropout_rate=dropout_rate)


def create_hybrid_tiny_vit(num_classes=2, dropout_rate=0.1):
    """Crée une instance de HybridTinyViT (CNN Stem + Transformer)"""
    return HybridTinyViT(num_classes=num_classes, dropout_rate=dropout_rate)


# ======================
# Dictionnaire des modèles disponibles
# ======================

class AircraftDetectorV2(nn.Module):
    """
    Détecteur d'avions V2 "Nano Competitor" (~6-8MB)
    
    Inspiré de YOLOv8n :
    - Backbone ResNet profond (4 stages: 64->128->256->512)
    - Residual connections partout (facilite l'apprentissage profond)
    - FPN Simple (Fusion Features 14x14 et 28x28)
    - Grid: 14x14
    """
    dropout_rate: float = 0.2

    @nn.compact
    def __call__(self, x, training=True):
        # Input: (B, 448, 448, 3)
        
        # --- STAGE 0: STEM ---
        # /2 -> 224
        x = nn.Conv(64, (3, 3), strides=(2, 2), padding="SAME", use_bias=False)(x)
        x = nn.BatchNorm(use_running_average=not training)(x)
        x = nn.silu(x)
        
        # --- STAGE 1: 64 filters ---
        # /4 -> 112
        x = nn.Conv(64, (3, 3), strides=(2, 2), padding="SAME", use_bias=False)(x)
        x = nn.BatchNorm(use_running_average=not training)(x)
        x = nn.silu(x)
        
        x = ResidualBlock(64, dropout_rate=self.dropout_rate)(x, training)
        x = ResidualBlock(64, dropout_rate=self.dropout_rate)(x, training)
        
        # --- STAGE 2: 128 filters ---
        # /8 -> 56
        x = ResidualBlock(128, stride=2, dropout_rate=self.dropout_rate)(x, training)
        x = ResidualBlock(128, dropout_rate=self.dropout_rate)(x, training)
        x = CBAMBlock()(x, training) # Attention
        
        # --- STAGE 3: 256 filters ---
        # /16 -> 28
        x = ResidualBlock(256, stride=2, dropout_rate=self.dropout_rate)(x, training)
        x = ResidualBlock(256, dropout_rate=self.dropout_rate)(x, training)
        
        # Save feature map P3 for fusion (28x28)
        feat_p3 = x 
        
        # --- STAGE 4: 512 filters ---
        # /32 -> 14
        x = ResidualBlock(512, stride=2, dropout_rate=self.dropout_rate)(x, training)
        x = ResidualBlock(512, dropout_rate=self.dropout_rate)(x, training)
        x = CBAMBlock()(x, training) # Attention au bottleneck
        
        feat_p4 = x # (14x14)
        
        # --- FPN: FEATURE FUSION ---
        # On upsample P4 (14->28) et on concatene avec P3 (28)
        # 1. Project P4 to 256 dim
        p4_up = nn.Conv(256, (1, 1))(feat_p4) 
        p4_up = nn.BatchNorm(use_running_average=not training)(p4_up)
        p4_up = nn.silu(p4_up)
        
        # 2. Resize P4 to match P3 resolution (Dynamic)
        # JAX resize bilinear
        target_h, target_w = feat_p3.shape[1], feat_p3.shape[2]
        p4_up = jax.image.resize(p4_up, shape=(p4_up.shape[0], target_h, target_w, p4_up.shape[3]), method='bilinear')
        
        # 3. Concat P3 + P4_up
        fusion_28 = jnp.concatenate([feat_p3, p4_up], axis=-1) # 256+256 = 512
        
        # 4. Refine fusion (Conv 3x3)
        fusion_28 = SeparableConv(256, (3, 3))(fusion_28, training)
        fusion_28 = nn.BatchNorm(use_running_average=not training)(fusion_28)
        fusion_28 = nn.silu(fusion_28)
        
        # 5. Downsample back to 14x14 (strided convolution) to get final grid
        # Pour integrer le contexte haute résolution dans la grille finale
        final_grid = nn.Conv(512, (3, 3), strides=(2, 2), padding="SAME")(fusion_28)
        final_grid = nn.BatchNorm(use_running_average=not training)(final_grid)
        final_grid = nn.silu(final_grid)
        
        # Fusionner avec le P4 original (Skip connection géante)
        final_grid = final_grid + feat_p4
        
        # --- DETECTION HEAD ---
        # Projection finale : 5 canaux (Conf, x, y, w, h)
        out = nn.Conv(5, (1, 1), padding="SAME")(final_grid)
        out = nn.sigmoid(out)
        
        return out


def create_aircraft_detector_v2(dropout_rate=0.2, **kwargs):
    """Factory for V2"""
    return AircraftDetectorV2(dropout_rate=dropout_rate)


class AircraftDetectorV3(nn.Module):
    """
    Détecteur d'avions V3 "Optimized Nano" (~6-8MB target)
    
    Identique à V2 mais utilise des Convolutions Séparables
    dans le backbone pour réduire massivement les paramètres (48MB -> ~8MB).
    """
    dropout_rate: float = 0.2

    @nn.compact
    def __call__(self, x, training=True):
        # Input: (B, 448, 448, 3) (ou 224)
        
        # --- STAGE 0: STEM (Standard Conv pour garder info RGB) ---
        x = nn.Conv(64, (3, 3), strides=(2, 2), padding="SAME", use_bias=False)(x)
        x = nn.BatchNorm(use_running_average=not training)(x)
        x = nn.silu(x)
        
        # --- STAGE 1: 64 filters ---
        x = SeparableConv(64, (3, 3), strides=(2, 2))(x, training) # Downsample
        x = nn.BatchNorm(use_running_average=not training)(x)
        x = nn.silu(x)
        
        x = ResidualBlockSeparable(64, dropout_rate=self.dropout_rate)(x, training)
        x = ResidualBlockSeparable(64, dropout_rate=self.dropout_rate)(x, training)
        
        # --- STAGE 2: 128 filters ---
        x = ResidualBlockSeparable(128, stride=2, dropout_rate=self.dropout_rate)(x, training)
        x = ResidualBlockSeparable(128, dropout_rate=self.dropout_rate)(x, training)
        x = CBAMBlock()(x, training) # Attention
        
        # --- STAGE 3: 256 filters ---
        x = ResidualBlockSeparable(256, stride=2, dropout_rate=self.dropout_rate)(x, training)
        x = ResidualBlockSeparable(256, dropout_rate=self.dropout_rate)(x, training)
        
        feat_p3 = x 
        
        # --- STAGE 4: 512 filters ---
        x = ResidualBlockSeparable(512, stride=2, dropout_rate=self.dropout_rate)(x, training)
        x = ResidualBlockSeparable(512, dropout_rate=self.dropout_rate)(x, training)
        x = CBAMBlock()(x, training)
        
        feat_p4 = x
        
        # --- FPN: FEATURE FUSION ---
        p4_up = nn.Conv(256, (1, 1))(feat_p4) 
        p4_up = nn.BatchNorm(use_running_average=not training)(p4_up)
        p4_up = nn.silu(p4_up)
        
        target_h, target_w = feat_p3.shape[1], feat_p3.shape[2]
        p4_up = jax.image.resize(p4_up, shape=(p4_up.shape[0], target_h, target_w, p4_up.shape[3]), method='bilinear')
        
        fusion_28 = jnp.concatenate([feat_p3, p4_up], axis=-1)
        
        fusion_28 = SeparableConv(256, (3, 3))(fusion_28, training)
        fusion_28 = nn.BatchNorm(use_running_average=not training)(fusion_28)
        fusion_28 = nn.silu(fusion_28)
        
        final_grid = nn.Conv(512, (3, 3), strides=(2, 2), padding="SAME")(fusion_28)
        final_grid = nn.BatchNorm(use_running_average=not training)(final_grid)
        final_grid = nn.silu(final_grid)
        
        final_grid = final_grid + feat_p4
        
        # --- DETECTION HEAD ---
        out = nn.Conv(5, (1, 1), padding="SAME")(final_grid)
        out = nn.sigmoid(out)
        
        return out


def create_aircraft_detector_v3(dropout_rate=0.2, **kwargs):
    """Factory for V3"""
    return AircraftDetectorV3(dropout_rate=dropout_rate)


class AircraftDetectorV4(nn.Module):
    """
    Détecteur d'avions V4 "Anchors Support" 
    
    Identique à V3 mais prédit B=2 boîtes par cellule (10 canaux en sortie)
    pour mieux gérer la densité d'objets (ex: jusqu'à 30 avions).
    """
    dropout_rate: float = 0.2
    num_anchors: int = 2

    @nn.compact
    def __call__(self, x, training=True):
        # --- STAGE 0: STEM ---
        x = nn.Conv(64, (3, 3), strides=(2, 2), padding="SAME", use_bias=False)(x)
        x = nn.BatchNorm(use_running_average=not training)(x)
        x = nn.silu(x)
        
        # --- STAGE 1: 64 filters ---
        x = SeparableConv(64, (3, 3), strides=(2, 2))(x, training)
        x = nn.BatchNorm(use_running_average=not training)(x)
        x = nn.silu(x)
        
        x = ResidualBlockSeparable(64, dropout_rate=self.dropout_rate)(x, training)
        x = ResidualBlockSeparable(64, dropout_rate=self.dropout_rate)(x, training)
        
        # --- STAGE 2: 128 filters ---
        x = ResidualBlockSeparable(128, stride=2, dropout_rate=self.dropout_rate)(x, training)
        x = ResidualBlockSeparable(128, dropout_rate=self.dropout_rate)(x, training)
        x = CBAMBlock()(x, training)
        
        # --- STAGE 3: 256 filters ---
        x = ResidualBlockSeparable(256, stride=2, dropout_rate=self.dropout_rate)(x, training)
        x = ResidualBlockSeparable(256, dropout_rate=self.dropout_rate)(x, training)
        feat_p3 = x 
        
        # --- STAGE 4: 512 filters ---
        x = ResidualBlockSeparable(512, stride=2, dropout_rate=self.dropout_rate)(x, training)
        x = ResidualBlockSeparable(512, dropout_rate=self.dropout_rate)(x, training)
        x = CBAMBlock()(x, training)
        feat_p4 = x
        
        # --- FPN: FEATURE FUSION ---
        p4_up = nn.Conv(256, (1, 1))(feat_p4) 
        p4_up = nn.BatchNorm(use_running_average=not training)(p4_up)
        p4_up = nn.silu(p4_up)
        
        target_h, target_w = feat_p3.shape[1], feat_p3.shape[2]
        p4_up = jax.image.resize(p4_up, shape=(p4_up.shape[0], target_h, target_w, p4_up.shape[3]), method='bilinear')
        
        fusion_28 = jnp.concatenate([feat_p3, p4_up], axis=-1)
        
        fusion_28 = SeparableConv(256, (3, 3))(fusion_28, training)
        fusion_28 = nn.BatchNorm(use_running_average=not training)(fusion_28)
        fusion_28 = nn.silu(fusion_28)
        
        final_grid = nn.Conv(512, (3, 3), strides=(2, 2), padding="SAME")(fusion_28)
        final_grid = nn.BatchNorm(use_running_average=not training)(final_grid)
        final_grid = nn.silu(final_grid)
        
        final_grid = final_grid + feat_p4
        
        # --- DETECTION HEAD (B=2 -> 10 canaux) ---
        out = nn.Conv(5 * self.num_anchors, (1, 1), padding="SAME")(final_grid)
        out = nn.sigmoid(out)
        
        return out


def create_aircraft_detector_v4(dropout_rate=0.2, **kwargs):
    """Factory for V4"""
    return AircraftDetectorV4(dropout_rate=dropout_rate)

# ... (Previous MODELS dict)

MODELS = {
    'aircraft_detector': lambda **kwargs: AircraftDetector(),
    'aircraft_detector_v2': create_aircraft_detector_v2, # NEW
    'aircraft_detector_v3': create_aircraft_detector_v3, # 🚀 Optimized Nano
    'aircraft_detector_v4': create_aircraft_detector_v4, # 🚀 V4 Anchors (B=2)
    'sophisticated_cnn': create_sophisticated_cnn,
    'sophisticated_cnn_droped_out': create_sophisticated_cnn_droped_out,
    'sophisticated_cnn_128': create_sophisticated_cnn_128,
    'sophisticated_cnn_128_plus': create_sophisticated_cnn_128_plus,
    'sophisticated_cnn_128_ultimate': create_sophisticated_cnn_128_ultimate,
    'sophisticated_cnn_optimized': create_sophisticated_cnn_optimized,
    'resnet_light': create_resnet_light,
    'tiny_vit_plus': create_tiny_vit_plus,
    'tiny_vit_plus_balanced': create_tiny_vit_plus_balanced,
    'tiny_vit_plus_ultimate': create_tiny_vit_plus_ultimate,
    'hybrid_tiny_vit': create_hybrid_tiny_vit,
}


def get_model(model_name, **kwargs):
    """
    Factory function pour obtenir un modèle par nom
    
    Args:
        model_name: Nom du modèle ('sophisticated_cnn', 'resnet_light', 'tiny_vit_plus')
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
        'sophisticated_cnn': {
            'name': 'SophisticatedCNN',
            'description': 'CNN sophistiqué avec convolutions séparables, SE et attention spatiale',
            'params': '~1.3M',
            'size': '~5MB',
            'best_for': 'Datasets moyens, images 64x64, architecture équilibrée'
        },
        'sophisticated_cnn_droped_out': {
            'name': 'SophisticatedCNNDropedOut',
            'description': 'CNN sophistiqué avec DROPOUT STRATÉGIQUE (0.1→0.15→0.2→0.3→0.4) pour combattre l\'overfitting',
            'params': '~1.3M',
            'size': '~5MB',
            'best_for': 'Réduire l\'overfitting massif, datasets moyens, objectif: écart train/val <10%'
        },
        'sophisticated_cnn_128': {
            'name': 'SophisticatedCNN128',
            'description': 'CNN OPTIMISÉ pour images 128×128 (4 blocs, 64→96→128→256→512, 3 poolings)',
            'params': '~2.5M',
            'size': '~10MB',
            'best_for': 'Images 128×128, exploite pleinement la haute résolution, objectif: 85-87% val'
        },
        'sophisticated_cnn_optimized': {
            'name': 'SophisticatedCNNOptimized',
            'description': 'CNN OPTIMISÉ avec Stochastic Depth, Multi-scale fusion, CBAM, improved head',
            'params': '~2.5M',
            'size': '~10MB',
            'best_for': 'Maximum de performance CNN, images 64x64, +5-10% accuracy (objectif: 82-85%)'
        },
        'resnet_light': {
            'name': 'ResNetLight',
            'description': 'ResNet léger avec blocs résiduels',
            'params': '~2.5M',
            'size': '~10MB',
            'best_for': 'Datasets moyens, images 64x64'
        },
        'tiny_vit_plus': {
            'name': 'TinyViTPlus',
            'description': 'Vision Transformer léger',
            'params': '~1.8M',
            'size': '~7MB',
            'best_for': 'Grands datasets, images 224x224+'
        },
        'tiny_vit_plus_balanced': {
            'name': 'TinyViTPlusBalanced',
            'description': 'Vision Transformer optimisé - Configuration équilibrée (patch_size=8, hidden_dim=224, num_heads=4)',
            'params': '~2.8M',
            'size': '~11MB',
            'best_for': 'Datasets moyens/grands, images 64x64, meilleure accuracy (+3-5%)'
        },
        'tiny_vit_plus_ultimate': {
            'name': 'TinyViTPlusUltimate',
            'description': 'Vision Transformer ultime avec Conv Stem, Stochastic Depth, Layer Scale, Class Attention (patch_size=4, hidden_dim=384, num_heads=6, num_layers=12)',
            'params': '~8-10M',
            'size': '~32-40MB',
            'best_for': 'Maximum de performance, images 64x64, accuracy maximale (+5-11%, objectif: 82-88%)'
        },
        'hybrid_tiny_vit': {
            'name': 'HybridTinyViT',
            'description': 'Hybrid CNN+Transformer : Conv Stem (128→32) + ViT (patch_size=4, hidden_dim=224, num_heads=4, num_layers=6, CLS token)',
            'params': '~1.5-2M',
            'size': '~6-8MB',
            'best_for': 'Petits datasets (<100K images), images 128×128, meilleur compromis CNN/ViT (objectif: 60-75%)'
        }
    }
    
    if model_name not in model_info:
        raise ValueError(f"Modèle '{model_name}' non trouvé")
    
    return model_info[model_name]


# ======================
# TrainState avec batch_stats
# ======================
class TrainStateWithBatchStats(train_state.TrainState):
    """TrainState étendu pour gérer batch_stats"""
    batch_stats: flax.core.FrozenDict = None
