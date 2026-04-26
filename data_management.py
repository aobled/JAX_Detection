"""
Gestion des chunks de données pour l'entraînement
Séparation de la logique de création/vérification des chunks
"""

import os
import glob
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from PIL import Image
import tqdm
from typing import Tuple, Optional


class ChunkManager:
    """
    Gestionnaire des chunks de données
    Responsable UNIQUEMENT du chargement des chunks (la création se fait via _dataset_tools.py)
    """
    def __init__(self, output_prefix: str, image_size: tuple = (128, 128), grayscale: bool = False):
        self.output_prefix = output_prefix
        self.image_size = image_size
        self.grayscale = grayscale
        
        # Chemins des chunks
        self.train_chunks = sorted(glob.glob(f"{output_prefix}_train_chunk*.npz"))
        self.val_chunks = sorted(glob.glob(f"{output_prefix}_val_chunk*.npz"))
        self.mean_std_path = f"{output_prefix}_meanstd.npz"
        
        mode_str = "Grayscale (1 canal)" if grayscale else "RGB (3 canaux)"
        print(f"📦 Classification Dataset: {len(self.train_chunks)} train chunks, {len(self.val_chunks)} val chunks [{mode_str}]")
    
    def get_chunk_statistics(self) -> dict:
        """Retourne les statistiques des chunks"""
        stats = {
            'train_chunks': len(self.train_chunks),
            'val_chunks': len(self.val_chunks),
            'train_samples': 0,
            'val_samples': 0,
            'train_classes': [],
            'val_classes': []
        }
        
        # ✅ CORRECTION: Compter TOUS les chunks, pas seulement les 3 premiers
        print("📊 Calcul des statistiques complètes...")
        
        # Compter tous les chunks de train
        for chunk_path in self.train_chunks:
            try:
                with np.load(chunk_path) as data:
                    chunk_samples = len(data['label'])
                    stats['train_samples'] += chunk_samples
                    stats['train_classes'].append(np.bincount(data['label']))
                    print(f"  Train chunk {os.path.basename(chunk_path)}: {chunk_samples} échantillons")
            except Exception as e:
                print(f"  Erreur lecture {chunk_path}: {e}")
                
        # Compter tous les chunks de validation
        for chunk_path in self.val_chunks:
            try:
                with np.load(chunk_path) as data:
                    chunk_samples = len(data['label'])
                    stats['val_samples'] += chunk_samples
                    stats['val_classes'].append(np.bincount(data['label']))
                    print(f"  Val chunk {os.path.basename(chunk_path)}: {chunk_samples} échantillons")
            except Exception as e:
                print(f"  Erreur lecture {chunk_path}: {e}")
                
        return stats
    
    def create_tf_datasets(self, micro_batch_size: int = 32, augment: bool = True, augmentation_params: dict = None) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
        """
        Crée les datasets TensorFlow pour l'entraînement
        
        Args:
            augmentation_params: Dictionnaire de paramètres pour l'augmentation
        """
        def create_chunked_tf_dataset(split: str, batch_size: int, augment: bool = True, augmentation_params: dict = None) -> tf.data.Dataset:
            """Crée un dataset TensorFlow à partir des chunks"""
            chunk_files = sorted(glob.glob(f"{self.output_prefix}_{split}_chunk*.npz"))
            
            if not chunk_files:
                raise FileNotFoundError(f"No chunk files found for {split}")
            
            # Charger mean/std pour la classification
            if not os.path.exists(self.mean_std_path):
                raise FileNotFoundError(f"Missing mean_std file: {self.mean_std_path}. Did you run fighterjet_classification_dataset_tools.py?")
            meanstd = np.load(self.mean_std_path)
            mean = meanstd['mean'].astype(np.float32)
            std = meanstd['std'].astype(np.float32)
            
            def gen():
                for file_path in chunk_files:
                    with np.load(file_path) as data:
                        images = data["image"].astype(np.float32)
                        labels = data["label"].astype(np.int32)
                        for img, lab in zip(images, labels):
                            yield img, lab
            
            # 🎨 Adapter la signature selon RGB ou Grayscale
            num_channels = 1 if self.grayscale else 3
            output_signature = (
                tf.TensorSpec(shape=self.image_size + (num_channels,), dtype=tf.float32),
                tf.TensorSpec(shape=(), dtype=tf.int32),
            )
            
            dataset = tf.data.Dataset.from_generator(gen, output_signature=output_signature)
            
            if split == "train" and augment:
                dataset = dataset.shuffle(4096)  # ⚡ Augmenté pour 224×224 (meilleur mélange)
                
                if augmentation_params is None:
                    augmentation_params = {}
                
                # Construction dynamique du pipeline d'augmentation
                aug_layers = []
                
                if augmentation_params.get("flip_h", False):
                    aug_layers.append(layers.RandomFlip("horizontal"))
                if augmentation_params.get("flip_v", False):
                    aug_layers.append(layers.RandomFlip("vertical"))
                    
                rot_factor = augmentation_params.get("rotation_factor", 0.0)
                if rot_factor > 0.0:
                    aug_layers.append(layers.RandomRotation(rot_factor, fill_mode="reflect"))
                    
                zoom_factor = augmentation_params.get("zoom_factor", 0.0)
                if zoom_factor > 0.0:
                    aug_layers.append(layers.RandomZoom(zoom_factor, fill_mode="reflect"))
                    
                trans_factor = augmentation_params.get("translation_factor", 0.0)
                if trans_factor > 0.0:
                    aug_layers.append(layers.RandomTranslation(trans_factor, trans_factor, fill_mode="reflect"))
                    
                bright_delta = augmentation_params.get("brightness_delta", 0.0)
                if bright_delta > 0.0:
                    aug_layers.append(layers.RandomBrightness(bright_delta, value_range=(0.0, 1.0)))
                    
                cont_factor = augmentation_params.get("contrast_factor", 0.0)
                if cont_factor > 0.0:
                    aug_layers.append(layers.RandomContrast(cont_factor))
                
                data_augmentation = tf.keras.Sequential(aug_layers)
                
                def aug_norm_fn(img, lab):
                    img = tf.expand_dims(img, axis=0)
                    img = data_augmentation(img)
                    img = tf.squeeze(img, axis=0)
                    img = tf.clip_by_value(img, 0.0, 1.0)
                    # ✅ CORRECTION: Convertir mean et std en Tensors TensorFlow
                    mean_tensor = tf.constant(mean, dtype=tf.float32)
                    std_tensor = tf.constant(std, dtype=tf.float32)
                    img = (img - mean_tensor) / std_tensor
                    return img, lab
                
                dataset = dataset.map(aug_norm_fn, num_parallel_calls=tf.data.AUTOTUNE)
            else:
                def norm_fn(img, lab):
                    # ✅ CORRECTION: Convertir mean et std en Tensors TensorFlow
                    mean_tensor = tf.constant(mean, dtype=tf.float32)
                    std_tensor = tf.constant(std, dtype=tf.float32)
                    img = (img - mean_tensor) / std_tensor
                    return img, lab
                
                dataset = dataset.map(norm_fn, num_parallel_calls=tf.data.AUTOTUNE)
            
            return dataset.batch(batch_size, drop_remainder=False).prefetch(tf.data.AUTOTUNE)
        
        # Créer les datasets
        train_ds = create_chunked_tf_dataset('train', micro_batch_size, augment=augment, augmentation_params=augmentation_params)
        val_ds = create_chunked_tf_dataset('val', micro_batch_size, augment=False, augmentation_params=None)
        
        return train_ds, val_ds
    
    def ensure_chunks_ready(self, micro_batch_size: int = 32, augment: bool = True, augmentation_params: dict = None) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
        """
        Point d'entrée principal : vérifie que les chunks existent et crée les TF Datasets
        """
        if not self.train_chunks or not self.val_chunks:
            error_msg = (
                f"\n❌ ERREUR: Chunks introuvables pour la classification !\n"
                f"   Je m'attendais à trouver {self.output_prefix}_[train|val]_chunk*.npz\n"
                f"💡 LANCEZ D'ABORD : python fighterjet_classification_dataset_tools.py"
            )
            print(error_msg)
            exit(1)
            
        print("✅ Chunks de classification trouvés")
        
        # Afficher les statistiques
        stats = self.get_chunk_statistics()
        print(f"📊 STATISTIQUES:")
        print(f"   Train: {stats['train_chunks']} chunks, ~{stats['train_samples']} échantillons")
        print(f"   Val: {stats['val_chunks']} chunks, ~{stats['val_samples']} échantillons")
        
        return self.create_tf_datasets(
            micro_batch_size=micro_batch_size,
            augment=augment,
            augmentation_params=augmentation_params
        )
class DetectionDataset:
    """
    Gestionnaire de dataset pour la détection d'objets
    Charge les chunks générés par tools/fighterjet_detection_dataset_tools.py
    """
    def __init__(self, output_prefix: str, image_size: tuple = (224, 224), batch_size: int = 16, grayscale: bool = False, augmentation_params: dict = None):
        self.output_prefix = output_prefix
        self.image_size = image_size
        self.batch_size = batch_size
        self.grayscale = grayscale  # 🎨 Support grayscale
        self.augmentation_params = augmentation_params if augmentation_params is not None else {}
        
        # Repérer les chunks
        self.train_chunks = sorted(glob.glob(f"{output_prefix}_train_chunk*.npz"))
        self.val_chunks = sorted(glob.glob(f"{output_prefix}_val_chunk*.npz"))
        
        mode_str = "Grayscale (1 canal)" if self.grayscale else "RGB (3 canaux)"
        print(f"📦 Detection Dataset: {len(self.train_chunks)} train chunks, {len(self.val_chunks)} val chunks [{mode_str}]")

    def create_tf_dataset(self, split='train', augment=True):
        """
        Crée un dataset TensorFlow qui retourne (image, boxes)
        Image: (image_size[0], image_size[1], C) où C=1 (grayscale) ou 3 (RGB)
        Boxes: (MAX_BOXES, 5)  [conf=1, x, y, w, h]
        """
        chunks = self.train_chunks if split == 'train' else self.val_chunks
        if not chunks:
            error_msg = (
                f"\n❌ ERREUR: Chunks introuvables pour la détection !\n"
                f"   Je m'attendais à trouver {self.output_prefix}_[split]_chunk*.npz\n"
                f"💡 LANCEZ D'ABORD : python fighterjet_detection_dataset_tools.py"
            )
            print(error_msg)
            exit(1)
            
        def gen():
            for chunk_path in chunks:
                with np.load(chunk_path) as data:
                    images = data['images'] # (N, H, W, C)
                    masks = data['masks']   # (N, H, W, 1)
                    
                    for img, mask in zip(images, masks):
                        yield img, mask

        # 🎨 Adapter le nombre de canaux selon grayscale ou RGB
        num_channels = 1 if self.grayscale else 3
        output_signature = (
            tf.TensorSpec(shape=self.image_size + (num_channels,), dtype=tf.float32),
            tf.TensorSpec(shape=self.image_size + (1,), dtype=tf.float32) # Masque binaire
        )
        
        ds = tf.data.Dataset.from_generator(gen, output_signature=output_signature)
        
        if split == 'train' and augment:
            ds = ds.shuffle(1000)
            # Todo: Augmentation complexe pour detection (flip boxes...)
            # Pour l'instant on fait simple : Flip horizontal uniquement
            
            def augment_fn(img, mask):
                # --- 1. Flips (Vertical & Horizontal) ---
                flip_v_enabled = self.augmentation_params.get("flip_v", False)
                if flip_v_enabled:
                    do_flip_v = tf.random.uniform([]) > 0.5
                    img = tf.cond(do_flip_v, lambda: tf.image.flip_up_down(img), lambda: img)
                    mask = tf.cond(do_flip_v, lambda: tf.image.flip_up_down(mask), lambda: mask)
                
                flip_h_enabled = self.augmentation_params.get("flip_h", False)
                if flip_h_enabled:
                    do_flip_h = tf.random.uniform([]) > 0.5
                    img = tf.cond(do_flip_h, lambda: tf.image.flip_left_right(img), lambda: img)
                    mask = tf.cond(do_flip_h, lambda: tf.image.flip_left_right(mask), lambda: mask)
                
                # --- 2. Translation (Shift) ---
                trans_factor = self.augmentation_params.get("translation_factor", 0.0)
                if trans_factor > 0.0:
                    do_translate = tf.random.uniform([]) > 0.5
                    shift_x = tf.random.uniform([], -trans_factor, trans_factor)
                    shift_y = tf.random.uniform([], -trans_factor, trans_factor)
                    
                    img_h = tf.shape(img)[0]
                    img_w = tf.shape(img)[1]
                    
                    def apply_translation(i, sx, sy):
                        px = tf.cast(sx * tf.cast(img_w, tf.float32), tf.int32)
                        py = tf.cast(sy * tf.cast(img_h, tf.float32), tf.int32)
                        
                        pad_h = tf.cast(0.20 * tf.cast(img_h, tf.float32), tf.int32)
                        pad_w = tf.cast(0.20 * tf.cast(img_w, tf.float32), tf.int32)
                        
                        padded_img = tf.pad(i, paddings=[[pad_h, pad_h], [pad_w, pad_w], [0, 0]], mode='REFLECT')
                        start_y = pad_h - py
                        start_x = pad_w - px
                        return tf.image.crop_to_bounding_box(padded_img, start_y, start_x, img_h, img_w)
                    
                    img = tf.cond(do_translate, lambda: apply_translation(img, shift_x, shift_y), lambda: img)
                    # Translation identique sur le masque avec fond noir (mode CONSTANT)
                    def apply_translation_mask(m, sx, sy):
                        px = tf.cast(sx * tf.cast(img_w, tf.float32), tf.int32)
                        py = tf.cast(sy * tf.cast(img_h, tf.float32), tf.int32)
                        
                        pad_h = tf.cast(0.20 * tf.cast(img_h, tf.float32), tf.int32)
                        pad_w = tf.cast(0.20 * tf.cast(img_w, tf.float32), tf.int32)
                        
                        padded_mask = tf.pad(m, paddings=[[pad_h, pad_h], [pad_w, pad_w], [0, 0]], mode='CONSTANT')
                        start_y = pad_h - py
                        start_x = pad_w - px
                        return tf.image.crop_to_bounding_box(padded_mask, start_y, start_x, img_h, img_w)

                    mask = tf.cond(do_translate, lambda: apply_translation_mask(mask, shift_x, shift_y), lambda: mask)
                
                # --- 3. Zoom (Scale) ---
                zoom_factor = self.augmentation_params.get("zoom_factor", 0.0)
                if zoom_factor > 0.0:
                    do_zoom = tf.random.uniform([]) > 0.5
                    scale = tf.random.uniform([], 1.0 - zoom_factor, 1.0 + zoom_factor)
                    
                    def apply_zoom(i, cur_scale):
                        crop_frac = 1.0 / cur_scale
                        crop_frac = tf.clip_by_value(crop_frac, 0.1, 1.0)
                        i_cropped = tf.image.central_crop(i, crop_frac)
                        img_h_local = tf.shape(i)[0]
                        img_w_local = tf.shape(i)[1]
                        target_shape = tf.cast([img_h_local, img_w_local], tf.int32)
                        return tf.image.resize(i_cropped, target_shape)
                    
                    img = tf.cond(do_zoom, lambda: apply_zoom(img, scale), lambda: img)
                    mask = tf.cond(do_zoom, lambda: apply_zoom(mask, scale), lambda: mask)
                
                # --- 4. Augmentation couleur (uniquement sur l'image) ---
                bright_delta = self.augmentation_params.get("brightness_delta", 0.0)
                if bright_delta > 0.0:
                    img = tf.image.random_brightness(img, bright_delta)
                    
                cont_factor = self.augmentation_params.get("contrast_factor", 0.0)
                if cont_factor > 0.0:
                    lower_cont = 1.0 - cont_factor
                    lower_cont = max(0.1, lower_cont) # Prevent contrast going below 0.1
                    upper_cont = 1.0 + cont_factor
                    img = tf.image.random_contrast(img, lower_cont, upper_cont)
                
                # S'assurer que le masque reste strictement binaire ou borné après resize
                mask = tf.clip_by_value(mask, 0.0, 1.0)
                
                return img, mask

                
            ds = ds.map(augment_fn)
            
        ds = ds.batch(self.batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)
        return ds

    def get_dataset(self):
        train_ds = self.create_tf_dataset('train', augment=True)
        val_ds = self.create_tf_dataset('val', augment=False)
        return train_ds, val_ds

def get_datasets(config: dict, backend_config: dict) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    """
    Fonction factory unifiée pour charger les datasets selon le type de tâche.
    
    Args:
        config (dict): Configuration globale du dataset
        backend_config (dict): Configuration spécifique au backend (TPU/GPU)
        
    Returns:
        tuple: (train_dataset, validation_dataset)
    """
    task_type = config.get("task_type", "classification")
    print(f"🔄 Initialisation du pipeline de données pour la tâche : {task_type.upper()}")
    
    aug_params = config.get("augmentation_params", {})
    
    if task_type == "classification":
        chunk_manager = ChunkManager(
            output_prefix=config["output_prefix"],
            image_size=config["image_size"],
            grayscale=config.get("grayscale", False)
        )
        return chunk_manager.ensure_chunks_ready(
            micro_batch_size=backend_config["micro_batch_size"],
            augment=True,
            augmentation_params=aug_params
        )
        
    elif task_type == "detection":
        dataset_manager = DetectionDataset(
            output_prefix=config["output_prefix"],
            image_size=config["image_size"],
            batch_size=backend_config["micro_batch_size"],
            grayscale=config.get("grayscale", False),
            augmentation_params=aug_params
        )
        train_ds = dataset_manager.create_tf_dataset('train', augment=True)
        val_ds = dataset_manager.create_tf_dataset('val', augment=False)
        
        # Mode vérification avec epochs=0 géré au niveau de get_datasets
        if config.get("epochs", 1) == 0:
            print("✅ Mode vérification: epochs=0, vérification des chunks requise")
        
        return train_ds, val_ds
        
    else:
        raise ValueError(f"Task type inconnu: {task_type}")
