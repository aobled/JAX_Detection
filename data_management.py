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
    Responsable de la création, vérification et chargement des chunks
    """
    
    def __init__(self, data_dir: str, output_prefix: str, chunk_size: int = 1000, image_size: tuple = (64, 64), class_names: list = None, grayscale: bool = False):
        self.data_dir = data_dir
        self.output_prefix = output_prefix
        self.chunk_size = chunk_size
        self.image_size = image_size
        self.grayscale = grayscale  # 🎨 Support images grayscale
        
        # 🔧 CLASSES DYNAMIQUES
        if class_names is None:
            # Détection automatique des classes depuis les dossiers
            self.class_names = self._detect_classes()
        else:
            self.class_names = class_names
        
        print(f"📁 Classes détectées: {self.class_names}")
        print(f"🎨 Mode: {'Grayscale (1 canal)' if grayscale else 'RGB (3 canaux)'}")
        
        # Chemins des chunks
        self.train_chunks = sorted(glob.glob(f"{output_prefix}_train_chunk*.npz"))
        self.val_chunks = sorted(glob.glob(f"{output_prefix}_val_chunk*.npz"))
    
    def _detect_classes(self) -> list:
        """Détecte automatiquement les classes depuis les dossiers train/val"""
        train_dir = os.path.join(self.data_dir, 'train')
        if os.path.exists(train_dir):
            classes = sorted([d for d in os.listdir(train_dir) 
                            if os.path.isdir(os.path.join(train_dir, d))])
            print(f"🔍 Détection automatique: {classes}")
            return classes
        else:
            print("⚠️  Dossier train non trouvé, exit")
            exit(0)
        
    def check_chunks_exist(self) -> bool:
        """Vérifie si les chunks existent"""
        return len(self.train_chunks) > 0 and len(self.val_chunks) > 0
    
    def verify_chunks_quality(self) -> bool:
        """
        Vérifie la qualité des chunks (équilibrage des classes)
        Retourne True si les chunks sont valides
        """
        if not self.check_chunks_exist():
            return False
            
        print("🔍 Vérification de la qualité des chunks...")
        val_has_both_classes = True
        
        # Vérifier les premiers chunks de validation
        for i, chunk_path in enumerate(self.val_chunks[:5]):
            try:
                with np.load(chunk_path) as data:
                    labels = data['label']
                    class_counts = np.bincount(labels)
                    print(f"Chunk val {i}: {len(labels)} échantillons, classes {class_counts}")
                    
                    # Vérifier si ce chunk contient les deux classes
                    if len(class_counts) < 2:
                        val_has_both_classes = False
                        print(f"⚠️  Chunk val {i} ne contient qu'une seule classe")
                        
            except Exception as e:
                print(f"Erreur chunk val {i}: {e}")
                val_has_both_classes = False
        
        return val_has_both_classes
    
    def should_recreate_chunks(self) -> bool:
        """
        Détermine si les chunks doivent être recréés
        """
        if not self.check_chunks_exist():
            print("❌ Aucun chunk trouvé")
            return True
            
        if not self.verify_chunks_quality():
            print("❌ Chunks existants sont défaillants")
            return True
            
        print("✅ Chunks existants sont valides")
        return False
    
    def cleanup_old_chunks(self):
        """Supprime les anciens chunks"""
        print("🗑️  Suppression des anciens chunks...")
        old_chunks = glob.glob(f"{self.output_prefix}_*_chunk*.npz")
        for chunk in old_chunks:
            try:
                os.remove(chunk)
                print(f"Supprimé: {chunk}")
            except Exception as e:
                print(f"Erreur suppression {chunk}: {e}")
    
    def create_chunked_npz(self):
        """
        Crée les chunks avec répartition équilibrée - VERSION OPTIMISÉE MÉMOIRE
        Charge les images progressivement au lieu de tout en RAM
        """
        print("🔄 Création des chunks avec répartition équilibrée (optimisé mémoire)...")
        
        # Créer le dossier de sortie si nécessaire
        output_dir = os.path.dirname(self.output_prefix)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        for split in ['train', 'val']:
            split_dir = os.path.join(self.data_dir, split)
            
            # 🔄 COLLECTER LES CHEMINS DES FICHIERS (pas les images)
            print(f"\n🔄 COLLECTE DES CHEMINS POUR {split}:")
            file_paths = []
            file_labels = []
            
            for label, class_name in enumerate(self.class_names):
                class_path = os.path.join(split_dir, class_name)
                print(f"Collecte {class_name}...")
                
                for img_file in os.listdir(class_path):
                    file_paths.append(os.path.join(class_path, img_file))
                    file_labels.append(label)
            
            # 🔀 MÉLANGER LES CHEMINS (pas les images)
            print(f"Mélange de {len(file_paths)} chemins...")
            indices = np.arange(len(file_paths))
            np.random.shuffle(indices)
            
            file_paths = [file_paths[i] for i in indices]
            file_labels = [file_labels[i] for i in indices]
            
            # Afficher la distribution des classes
            class_counts = [file_labels.count(i) for i in range(len(self.class_names))]
            print(f"Après mélange - Distribution: {dict(zip(self.class_names, class_counts))}")
            
            # Calcul des statistiques pour le train (sur un échantillon)
            num_channels = 1 if self.grayscale else 3
            mean_accumulator = np.zeros(num_channels, dtype=np.float64)
            std_accumulator = np.zeros(num_channels, dtype=np.float64)
            
            if split == 'train':
                print("Calcul des statistiques sur un échantillon...")
                sample_size = min(1000, len(file_paths))
                for i in tqdm.tqdm(range(sample_size), desc="Stats"):
                    try:
                        # 🎨 Chargement RGB ou Grayscale
                        img_mode = 'L' if self.grayscale else 'RGB'
                        img = Image.open(file_paths[i]).convert(img_mode)
                        img_array = np.array(img.resize(self.image_size), dtype=np.float32) / 255.0
                        
                        # Si grayscale, ajouter dimension canal
                        if self.grayscale and img_array.ndim == 2:
                            img_array = np.expand_dims(img_array, axis=-1)  # (H, W) → (H, W, 1)
                        
                        mean_accumulator += img_array.mean(axis=(0, 1))
                        std_accumulator += img_array.std(axis=(0, 1))
                    except:
                        pass
                
                mean = mean_accumulator / sample_size
                std = std_accumulator / sample_size
                np.savez(f"{self.output_prefix}_meanstd.npz", mean=mean.astype(np.float32), std=std.astype(np.float32))
                print(f"Train mean: {mean}, std: {std}")
            
            # 💾 CRÉATION DES CHUNKS PROGRESSIVEMENT
            num_chunks = (len(file_paths) + self.chunk_size - 1) // self.chunk_size
            print(f"Création de {num_chunks} chunks...")
            
            for chunk_id in range(num_chunks):
                start = chunk_id * self.chunk_size
                end = min((chunk_id + 1) * self.chunk_size, len(file_paths))
                
                # Charger seulement les images de CE chunk
                chunk_images = []
                chunk_labels = []
                
                for i in tqdm.tqdm(range(start, end), desc=f"Chunk {chunk_id}/{num_chunks}"):
                    try:
                        # 🎨 Chargement RGB ou Grayscale
                        img_mode = 'L' if self.grayscale else 'RGB'
                        img = Image.open(file_paths[i]).convert(img_mode)
                        img_array = np.array(img.resize(self.image_size), dtype=np.float32) / 255.0
                        
                        # Si grayscale, ajouter dimension canal
                        if self.grayscale and img_array.ndim == 2:
                            img_array = np.expand_dims(img_array, axis=-1)  # (H, W) → (H, W, 1)
                        
                        chunk_images.append(img_array)
                        chunk_labels.append(file_labels[i])
                    except Exception as e:
                        print(f"Erreur sur {file_paths[i]}: {e}")
                
                # Convertir en numpy et sauvegarder
                chunk_images_np = np.stack(chunk_images)
                chunk_labels_np = np.array(chunk_labels, dtype=np.int32)
                
                # Vérifier l'équilibrage
                chunk_class_counts = np.bincount(chunk_labels_np, minlength=len(self.class_names))
                print(f"  Chunk {chunk_id}: {len(chunk_labels_np)} échantillons, classes {chunk_class_counts}")
                
                np.savez_compressed(
                    f"{self.output_prefix}_{split}_chunk{chunk_id}.npz",
                    image=chunk_images_np,
                    label=chunk_labels_np
                )
                
                # Libérer la mémoire
                del chunk_images, chunk_labels, chunk_images_np, chunk_labels_np
            
            print(f"[✓] {split}: {len(file_paths)} images → {num_chunks} chunks")
    
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
    
    def create_tf_datasets(self, micro_batch_size: int = 32, augment: bool = True, aggressive_aug: bool = False) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
        """
        Crée les datasets TensorFlow pour l'entraînement
        
        Args:
            aggressive_aug: Si True, utilise augmentation agressive pour ViT
        """
        def create_chunked_tf_dataset(split: str, batch_size: int, augment: bool = True, aggressive_aug: bool = False) -> tf.data.Dataset:
            """Crée un dataset TensorFlow à partir des chunks"""
            chunk_files = sorted(glob.glob(f"{self.output_prefix}_{split}_chunk*.npz"))
            
            if not chunk_files:
                raise FileNotFoundError(f"No chunk files found for {split}")
            
            # Charger mean/std
            meanstd = np.load(f"{self.output_prefix}_meanstd.npz")
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
                
                # 🔥 CHOIX D'AUGMENTATION selon besoin
                # Mode "aggressive" pour ViT (simuler dataset ×10)
                # Mode "moderate" pour CNN (équilibre)
                
                if aggressive_aug:
                    # 🔥 AUGMENTATION TRÈS AGRESSIVE (pour Vision Transformers)
                    # Simule un dataset 10× plus grand
                    data_augmentation = tf.keras.Sequential([
                        layers.RandomFlip("horizontal"),
                        layers.RandomFlip("vertical"),                         # 🔥 Nouveau
                        layers.RandomRotation(0.30, fill_mode="reflect"),      # 🔥 ±30° (vs ±12°)
                        layers.RandomZoom(0.25, fill_mode="reflect"),          # 🔥 ±25% (vs ±10%)
                        layers.RandomTranslation(0.15, 0.15, fill_mode="reflect"),  # 🔥 Translation ±15%
                        layers.RandomBrightness(0.30, value_range=(0.0, 1.0)), # 🔥 ±30% (vs ±10%)
                        layers.RandomContrast(0.30),                           # 🔥 ±30% (vs ±10%)
                    ])
                else:
                    # 🎯 AUGMENTATION MODÉRÉE (pour CNNs)
                    data_augmentation = tf.keras.Sequential([
                        layers.RandomFlip("horizontal"),
                        layers.RandomRotation(0.12, fill_mode="reflect"),      # ±12°
                        layers.RandomZoom(0.10, fill_mode="reflect"),          # ±10%
                        layers.RandomBrightness(0.10, value_range=(0.0, 1.0)), # ±10%
                        layers.RandomContrast(0.10),                           # ±10%
                    ])
                
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
        train_ds = create_chunked_tf_dataset('train', micro_batch_size, augment=augment, aggressive_aug=aggressive_aug)
        val_ds = create_chunked_tf_dataset('val', micro_batch_size, augment=False, aggressive_aug=False)
        
        return train_ds, val_ds
    
    def ensure_chunks_ready(self, micro_batch_size: int = 32, augment: bool = True, aggressive_aug: bool = False) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
        """
        Point d'entrée principal : s'assure que les chunks sont prêts et retourne les datasets
        """
        print("🔍 VÉRIFICATION DES CHUNKS...")
        
        # Mettre à jour les chemins des chunks
        self.train_chunks = sorted(glob.glob(f"{self.output_prefix}_train_chunk*.npz"))
        self.val_chunks = sorted(glob.glob(f"{self.output_prefix}_val_chunk*.npz"))
        
        if self.should_recreate_chunks():
            print("\n🚨 RECRÉATION DES CHUNKS...")
            self.cleanup_old_chunks()
            self.create_chunked_npz()
            
            # Mettre à jour les chemins après création
            self.train_chunks = sorted(glob.glob(f"{self.output_prefix}_train_chunk*.npz"))
            self.val_chunks = sorted(glob.glob(f"{self.output_prefix}_val_chunk*.npz"))
        else:
            print("✅ Utilisation des chunks existants")
        
        # Afficher les statistiques
        stats = self.get_chunk_statistics()
        print(f"📊 STATISTIQUES:")
        print(f"   Train: {stats['train_chunks']} chunks, ~{stats['train_samples']} échantillons")
        print(f"   Val: {stats['val_chunks']} chunks, ~{stats['val_samples']} échantillons")
        
        # 🔥 CORRECTION: Retourner les datasets après s'être assuré que les chunks existent
        return self.create_tf_datasets(
            micro_batch_size=micro_batch_size,
            augment=augment,
            aggressive_aug=aggressive_aug
        )
class DetectionDataset:
    """
    Gestionnaire de dataset pour la détection d'objets
    Charge les chunks générés par tools/fighterjet_detection_dataset_tools.py
    """
    def __init__(self, output_prefix: str, image_size: tuple = (224, 224), batch_size: int = 16, grayscale: bool = False):
        self.output_prefix = output_prefix
        self.image_size = image_size
        self.batch_size = batch_size
        self.grayscale = grayscale  # 🎨 Support grayscale
        
        # Repérer les chunks
        self.train_chunks = sorted(glob.glob(os.path.join(output_prefix, "detection_train_chunk*.npz")))
        self.val_chunks = sorted(glob.glob(os.path.join(output_prefix, "detection_val_chunk*.npz")))
        
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
            print(f"⚠️ No chunks for {split}")
            return None
            
        def gen():
            for chunk_path in chunks:
                with np.load(chunk_path) as data:
                    images = data['images'] # (N, H, W, C) où H,W = image_size, C=1 (grayscale) ou 3 (RGB)
                    boxes = data['boxes']   # (N, 30, 5)
                    
                    for img, box in zip(images, boxes):
                        yield img, box

        # 🎨 Adapter le nombre de canaux selon grayscale ou RGB
        num_channels = 1 if self.grayscale else 3
        output_signature = (
            tf.TensorSpec(shape=self.image_size + (num_channels,), dtype=tf.float32),
            tf.TensorSpec(shape=(None, 5), dtype=tf.float32) # (30, 5) variable in creation but fixed here
        )
        
        ds = tf.data.Dataset.from_generator(gen, output_signature=output_signature)
        
        if split == 'train' and augment:
            ds = ds.shuffle(1000)
            # Todo: Augmentation complexe pour detection (flip boxes...)
            # Pour l'instant on fait simple : Flip horizontal uniquement
            
            def augment_fn(img, boxes):
                # boxes: (N, 5) -> [has, cx, cy, w, h]
                has_obj = boxes[:, 0:1]
                cx = boxes[:, 1:2]
                cy = boxes[:, 2:3]
                w = boxes[:, 3:4]
                h = boxes[:, 4:5]
                
                # --- 1. Flips (Vertical & Horizontal) ---
                do_flip_v = tf.random.uniform([]) > 0.5
                img = tf.cond(do_flip_v, lambda: tf.image.flip_up_down(img), lambda: img)
                cy = tf.where(do_flip_v, 1.0 - cy, cy)
                
                do_flip_h = tf.random.uniform([]) > 0.5
                img = tf.cond(do_flip_h, lambda: tf.image.flip_left_right(img), lambda: img)
                cx = tf.where(do_flip_h, 1.0 - cx, cx)
                
                # --- 2. Translation (Shift) ---
                do_translate = tf.random.uniform([]) > 0.5
                shift_x = tf.random.uniform([], -0.1, 0.1)
                shift_y = tf.random.uniform([], -0.1, 0.1)
                
                img_h = tf.shape(img)[0]
                img_w = tf.shape(img)[1]
                
                def apply_translation(i, sx, sy):
                    # Convert shift to pixels
                    px = tf.cast(sx * tf.cast(img_w, tf.float32), tf.int32)
                    py = tf.cast(sy * tf.cast(img_h, tf.float32), tf.int32)
                    
                    # Pad the image symmetrically (larger than needed)
                    # We pad by max possible shift (10% + safety margin)
                    pad_h = tf.cast(0.15 * tf.cast(img_h, tf.float32), tf.int32)
                    pad_w = tf.cast(0.15 * tf.cast(img_w, tf.float32), tf.int32)
                    
                    # Pad using REFLECT mode
                    padded_img = tf.pad(i, paddings=[[pad_h, pad_h], [pad_w, pad_w], [0, 0]], mode='REFLECT')
                    
                    # Define new crop start (center pad_h/pad_w, minus the shift)
                    start_y = pad_h - py
                    start_x = pad_w - px
                    
                    # Crop back to original size
                    shifted_img = tf.image.crop_to_bounding_box(padded_img, start_y, start_x, img_h, img_w)
                    return shifted_img
                
                img = tf.cond(do_translate, 
                              lambda: apply_translation(img, shift_x, shift_y), 
                              lambda: img)
                
                cx = tf.where(do_translate, cx + shift_x, cx)
                cy = tf.where(do_translate, cy + shift_y, cy)
                
                # --- 3. Zoom (Scale) ---
                # Zoom in (scale > 1) ou out (scale < 1) de +/- 20%
                do_zoom = tf.random.uniform([]) > 0.5
                scale = tf.random.uniform([], 0.8, 1.2)
                
                # Pour zoomer avec tf.image, on crop au centre puis redimensionne (Zoom In)
                # Ou on pad puis redimensionne (Zoom Out). tf.image.central_crop est simple.
                def apply_zoom(i, cur_scale):
                    # Central crop & resize
                    crop_frac = 1.0 / cur_scale # Si scale=1.2 (zoom in), on crop 83% de l'image
                    crop_frac = tf.clip_by_value(crop_frac, 0.1, 1.0)
                    i_cropped = tf.image.central_crop(i, crop_frac)
                    # We use img_h and img_w which are defined above translation
                    target_shape = tf.cast([img_h, img_w], tf.int32)
                    return tf.image.resize(i_cropped, target_shape)
                
                img = tf.cond(do_zoom, lambda: apply_zoom(img, scale), lambda: img)
                
                # Ajuster les boîtes (le centre change car on recadre par rapport au centre)
                # Le centre de l'image est 0.5, 0.5. L'écart au centre est multiplié par le scale.
                cx_dist = cx - 0.5
                cy_dist = cy - 0.5
                
                new_cx = 0.5 + (cx_dist * scale)
                new_cy = 0.5 + (cy_dist * scale)
                new_w = w * scale
                new_h = h * scale
                
                cx = tf.where(do_zoom, new_cx, cx)
                cy = tf.where(do_zoom, new_cy, cy)
                w = tf.where(do_zoom, new_w, w)
                h = tf.where(do_zoom, new_h, h)
                
                # --- Séparation des boxes invalides ---
                # Si une boîte sort complètement du cadre, on annule has_obj
                is_invalid = (cx < 0.0) | (cx > 1.0) | (cy < 0.0) | (cy > 1.0)
                has_obj = tf.where(is_invalid, tf.zeros_like(has_obj), has_obj)
                
                # Clip les dimensions pour rester [0, 1]
                cx = tf.clip_by_value(cx, 0.0, 1.0)
                cy = tf.clip_by_value(cy, 0.0, 1.0)
                w = tf.clip_by_value(w, 0.0, 1.0)
                h = tf.clip_by_value(h, 0.0, 1.0)
                
                boxes = tf.concat([has_obj, cx, cy, w, h], axis=-1)
                
                # --- 4. Augmentation couleur (Dernier car non-destructeur pour boxes) ---
                img = tf.image.random_brightness(img, 0.2)
                img = tf.image.random_contrast(img, 0.8, 1.2)
                
                return img, boxes

                
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
    
    if task_type == "classification":
        chunk_manager = ChunkManager(
            data_dir=config["data_dir"],
            output_prefix=config["output_prefix"],
            chunk_size=config["chunk_size"],
            image_size=config["image_size"],
            class_names=config["class_names"],
            grayscale=config.get("grayscale", False)
        )
        aggressive_aug = config.get("aggressive_augmentation", False)
        return chunk_manager.ensure_chunks_ready(
            micro_batch_size=backend_config["micro_batch_size"],
            augment=True,
            aggressive_aug=aggressive_aug
        )
        
    elif task_type == "detection":
        dataset_manager = DetectionDataset(
            output_prefix=config["output_prefix"],
            image_size=config["image_size"],
            batch_size=backend_config["micro_batch_size"],
            grayscale=config.get("grayscale", False)
        )
        train_ds = dataset_manager.create_tf_dataset('train', augment=True)
        val_ds = dataset_manager.create_tf_dataset('val', augment=False)
        
        # Mode vérification avec epochs=0 géré au niveau de get_datasets
        if config.get("epochs", 1) == 0:
            print("✅ Mode vérification: epochs=0, vérification des chunks requise")
        
        return train_ds, val_ds
        
    else:
        raise ValueError(f"Task type inconnu: {task_type}")
