"""
Gestion des chunks de données pour l'entraînement
Séparation de la logique de création/vérification des chunks
"""

import os
import glob

# Ce module n'utilise TensorFlow que pour le pipeline tf.data (chargement/augmentation
# CPU) — le calcul GPU/TPU de l'entraînement est géré par JAX seul, importé et déjà
# initialisé plus tôt dans main.py au moment où ce module est chargé (import paresseux,
# à l'intérieur de main()). CUDA_VISIBLE_DEVICES doit donc être positionné ICI, juste
# avant le tout premier `import tensorflow`, PAS dans le bloc d'init précoce de main.py
# (qui s'exécute avant `import jax` — y toucher aveuglerait aussi JAX).
#
# Deux couches, vérifiées empiriquement, chacune couvrant un cas que l'autre ne couvre pas :
#   1. CUDA_VISIBLE_DEVICES avant l'import : efficace sur runtime TPU (JAX y utilise le
#      driver TPU, jamais CUDA — rien à "réclamer" avant TF) ou sur toute machine où rien
#      n'a encore touché CUDA dans ce process. Confirmé par test : TF ne voit alors aucun GPU.
#   2. tf.config.set_visible_devices() après import, en secours : sur une machine GPU où
#      JAX a déjà initialisé son propre contexte CUDA avant que ce module ne charge (le cas
#      local le plus exigeant testé), la variable d'environnement seule n'a plus d'effet sur
#      ce que TF découvre ensuite — cet appel API reste le seul moyen de forcer TF à ne pas
#      retenir ce GPU pour ses propres opérations, même s'il ne supprime pas un éventuel
#      warning déjà émis pendant l'import. Best-effort, ne doit jamais faire échouer l'entraînement.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import numpy as np
import tensorflow as tf

try:
    tf.config.set_visible_devices([], 'GPU')
except RuntimeError as _e:
    print(f"⚠️  Impossible de masquer le GPU à TensorFlow (non bloquant) : {_e}")

from tensorflow.keras import layers
from PIL import Image
import tqdm
from typing import Tuple, Optional

from detection_target_encoding import HEATMAP_KEY, SIZE_KEY


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
            
            num_channels = 1 if self.grayscale else 3
            
            def gen():
                for file_path in chunk_files:
                    with np.load(file_path) as data:
                        images = data["image"].astype(np.float32)
                        labels = data["label"].astype(np.int32)
                        for img, lab in zip(images, labels):
                            # Kepler / grayscale 2D dans le NPZ : (H, W) → (H, W, C)
                            if tuple(img.shape) == self.image_size:
                                img = np.expand_dims(img, axis=-1)
                            yield img, lab
            
            # 🎨 Adapter la signature selon RGB ou Grayscale
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
                
                if len(aug_layers) > 0:
                    data_augmentation = tf.keras.Sequential(aug_layers)
                    
                    def aug_norm_fn(img, lab):
                        img = tf.expand_dims(img, axis=0)
                        img = data_augmentation(img)
                        img = tf.squeeze(img, axis=0)
                        
                        # --- Augmentation Custom : Pixelation ---
                        pixelation_factor = augmentation_params.get("pixelation_factor", 0.0)
                        if pixelation_factor > 0.0:
                            do_pixelate = tf.random.uniform([]) > 0.5
                            def apply_pix(i):
                                original_shape = tf.shape(i)[:2]
                                random_factor = tf.random.uniform([], 2.0, tf.maximum(2.1, pixelation_factor))
                                new_h = tf.cast(tf.cast(original_shape[0], tf.float32) / random_factor, tf.int32)
                                new_w = tf.cast(tf.cast(original_shape[1], tf.float32) / random_factor, tf.int32)
                                small = tf.image.resize(i, [new_h, new_w], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
                                return tf.image.resize(small, original_shape, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
                            
                            img = tf.cond(do_pixelate, lambda: apply_pix(img), lambda: img)
                        
                        img = tf.clip_by_value(img, 0.0, 1.0)
                        # ✅ CORRECTION: Convertir mean et std en Tensors TensorFlow
                        mean_tensor = tf.constant(mean, dtype=tf.float32)
                        std_tensor = tf.constant(std, dtype=tf.float32)
                        img = (img - mean_tensor) / std_tensor
                        return img, lab
                    
                    dataset = dataset.map(aug_norm_fn, num_parallel_calls=tf.data.AUTOTUNE)
                else:
                    def norm_fn_train(img, lab):
                        mean_tensor = tf.constant(mean, dtype=tf.float32)
                        std_tensor = tf.constant(std, dtype=tf.float32)
                        img = (img - mean_tensor) / std_tensor
                        return img, lab
                    
                    dataset = dataset.map(norm_fn_train, num_parallel_calls=tf.data.AUTOTUNE)
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
            hint = (
                "python tools/kepler_dataset_tools.py"
                if "kepler" in self.output_prefix.lower()
                else "python fighterjet_classification_dataset_tools.py"
            )
            error_msg = (
                f"\n❌ ERREUR: Chunks introuvables pour la classification !\n"
                f"   Je m'attendais à trouver {self.output_prefix}_[train|val]_chunk*.npz\n"
                f"💡 Vérifiez output_prefix dans dataset_configs.py, ou lancez : {hint}"
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


def _apply_geometric_and_color_augmentation(
    img, extra_tensors, extra_pad_modes, extra_interp_methods, extra_rescale_on_zoom,
    augmentation_params,
):
    """
    Logique d'augmentation géométrique (flip V/H, translation pad+crop, zoom) et
    couleur (brightness/contrast) partagée entre DetectionDataset et
    CenterNetDetectionDataset (auparavant dupliquée ~120 lignes à l'identique,
    avec dérive : voir extra_interp_methods/extra_rescale_on_zoom ci-dessous).
    AD-17 : les deux classes restent dédiées et dispatchées séparément par
    task_type - seule cette géométrie commune, indépendante du format de payload
    (masque binaire vs heatmap+taille), est mutualisée ici.

    `img` utilise toujours pad_mode='REFLECT' et interpolation='bilinear' pour la
    translation/le zoom (identique historiquement dans les deux classes).

    `extra_tensors` : tenseurs additionnels à transformer en synchronisation avec
    `img` (ex. [mask] ou [heatmap, size]). `extra_pad_modes`/`extra_interp_methods`/
    `extra_rescale_on_zoom` sont des listes parallèles (un réglage par tenseur) :
    - extra_interp_methods : 'nearest' préserve les valeurs exactes (pic heatmap
      ==1.0, cellule de taille ponctuelle) - nécessaire pour heatmap/size, pas pour
      un masque binaire classique qui tolère l'interpolation bilinéaire historique
      de DetectionDataset.
    - extra_rescale_on_zoom : True uniquement pour la carte `size` de
      CenterNetDetectionDataset - après zoom, la magnitude stockée (en pixels) doit
      être rescalée par le facteur de zoom réellement appliqué (après clamp), sinon
      elle devient fausse silencieusement. Ne s'applique jamais au masque ni au
      heatmap (valeurs de présence/position, pas de magnitude).

    Retourne (img, extra_tensors) après géométrie ; le clip final (masque/heatmap
    dans [0,1], jamais sur `size`) reste à la charge de l'appelant.
    """
    n_extra = len(extra_tensors)
    if not (len(extra_pad_modes) == len(extra_interp_methods) == len(extra_rescale_on_zoom) == n_extra):
        raise ValueError(
            "extra_tensors, extra_pad_modes, extra_interp_methods et extra_rescale_on_zoom "
            f"doivent avoir la même longueur (reçu {n_extra}, {len(extra_pad_modes)}, "
            f"{len(extra_interp_methods)}, {len(extra_rescale_on_zoom)}) - un `zip()` sur des "
            "listes de longueurs différentes tronquerait silencieusement au lieu d'échouer."
        )

    all_tensors = [img] + list(extra_tensors)

    # --- 1. Flips (Vertical & Horizontal) ---
    flip_v_enabled = augmentation_params.get("flip_v", False)
    if flip_v_enabled:
        do_flip_v = tf.random.uniform([]) > 0.5
        all_tensors = [
            tf.cond(do_flip_v, lambda t=t: tf.image.flip_up_down(t), lambda t=t: t)
            for t in all_tensors
        ]

    flip_h_enabled = augmentation_params.get("flip_h", False)
    if flip_h_enabled:
        do_flip_h = tf.random.uniform([]) > 0.5
        all_tensors = [
            tf.cond(do_flip_h, lambda t=t: tf.image.flip_left_right(t), lambda t=t: t)
            for t in all_tensors
        ]

    # --- 2. Translation (Shift) --- décalage entier pad+crop
    trans_factor = augmentation_params.get("translation_factor", 0.0)
    if trans_factor > 0.0:
        do_translate = tf.random.uniform([]) > 0.5
        shift_x = tf.random.uniform([], -trans_factor, trans_factor)
        shift_y = tf.random.uniform([], -trans_factor, trans_factor)

        img_h = tf.shape(img)[0]
        img_w = tf.shape(img)[1]

        def apply_translation(t, sx, sy, pad_mode):
            px = tf.cast(sx * tf.cast(img_w, tf.float32), tf.int32)
            py = tf.cast(sy * tf.cast(img_h, tf.float32), tf.int32)

            pad_h = tf.cast(trans_factor * tf.cast(img_h, tf.float32), tf.int32) + 1
            pad_w = tf.cast(trans_factor * tf.cast(img_w, tf.float32), tf.int32) + 1

            padded = tf.pad(t, paddings=[[pad_h, pad_h], [pad_w, pad_w], [0, 0]], mode=pad_mode)
            start_y = pad_h - py
            start_x = pad_w - px
            return tf.image.crop_to_bounding_box(padded, start_y, start_x, img_h, img_w)

        pad_modes = ["REFLECT"] + list(extra_pad_modes)
        all_tensors = [
            tf.cond(
                do_translate,
                lambda t=t, pm=pm: apply_translation(t, shift_x, shift_y, pm),
                lambda t=t: t,
            )
            for t, pm in zip(all_tensors, pad_modes)
        ]

    # --- 3. Zoom (Scale) ---
    zoom_factor = augmentation_params.get("zoom_factor", 0.0)
    if zoom_factor > 0.0:
        do_zoom = tf.random.uniform([]) > 0.5
        scale = tf.random.uniform([], 1.0 - zoom_factor, 1.0 + zoom_factor)

        def _effective_zoom_scale(cur_scale):
            # crop_frac est clampe a [0.1, 1.0] - pour cur_scale < 1 (zoom arriere), le
            # clamp a 1.0 annule tout crop reel, l'image reste inchangee malgre un
            # cur_scale != 1.0. Le facteur de zoom REELLEMENT applique est donc
            # 1/crop_frac (apres clamp), pas cur_scale brut.
            crop_frac = tf.clip_by_value(1.0 / cur_scale, 0.1, 1.0)
            return 1.0 / crop_frac

        def _zoom_crop_and_resize(t, cur_scale, method):
            crop_frac = tf.clip_by_value(1.0 / cur_scale, 0.1, 1.0)
            t_cropped = tf.image.central_crop(t, crop_frac)
            t_h_local = tf.shape(t)[0]
            t_w_local = tf.shape(t)[1]
            target_shape = tf.cast([t_h_local, t_w_local], tf.int32)
            return tf.image.resize(t_cropped, target_shape, method=method)

        interp_methods = ["bilinear"] + list(extra_interp_methods)
        rescale_flags = [False] + list(extra_rescale_on_zoom)

        new_tensors = []
        for t, method, rescale in zip(all_tensors, interp_methods, rescale_flags):
            if rescale:
                def _zoomed_and_rescaled(t=t, method=method):
                    resized = _zoom_crop_and_resize(t, scale, method)
                    return resized * _effective_zoom_scale(scale)
                new_t = tf.cond(do_zoom, _zoomed_and_rescaled, lambda t=t: t)
            else:
                new_t = tf.cond(
                    do_zoom,
                    lambda t=t, method=method: _zoom_crop_and_resize(t, scale, method),
                    lambda t=t: t,
                )
            new_tensors.append(new_t)
        all_tensors = new_tensors

    img = all_tensors[0]
    extra_tensors = all_tensors[1:1 + n_extra]

    # --- 4. Augmentation couleur (uniquement sur l'image) ---
    bright_delta = augmentation_params.get("brightness_delta", 0.0)
    if bright_delta > 0.0:
        img = tf.image.random_brightness(img, bright_delta)

    cont_factor = augmentation_params.get("contrast_factor", 0.0)
    if cont_factor > 0.0:
        lower_cont = 1.0 - cont_factor
        lower_cont = max(0.1, lower_cont)  # Prevent contrast going below 0.1
        upper_cont = 1.0 + cont_factor
        img = tf.image.random_contrast(img, lower_cont, upper_cont)

    return img, extra_tensors


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
                img, (mask,) = _apply_geometric_and_color_augmentation(
                    img, [mask],
                    extra_pad_modes=["CONSTANT"],
                    extra_interp_methods=["bilinear"],
                    extra_rescale_on_zoom=[False],
                    augmentation_params=self.augmentation_params,
                )
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


class CenterNetDetectionDataset:
    """
    Gestionnaire de dataset pour la détection JAX_DETECTOR (heatmap+taille, AD-9/AD-17/AD-18).
    Charge les chunks générés par dataset_builder/jax_detector_dataset_tools.py (Story 7.4).
    Classe séparée de DetectionDataset (AD-17) - ne modifie ni n'étend celle-ci.
    """
    def __init__(self, output_prefix: str, image_size: tuple = (224, 224), batch_size: int = 16, grayscale: bool = False, augmentation_params: dict = None):
        self.output_prefix = output_prefix
        self.image_size = image_size
        self.batch_size = batch_size
        self.grayscale = grayscale
        self.augmentation_params = augmentation_params if augmentation_params is not None else {}

        # Repérer les chunks - même pattern que DetectionDataset, fonctionne quel que soit
        # le préfixe littéral choisi par la Story 7.4 tant que output_prefix correspond
        # (dépendance sur la config JAX_DETECTOR, Story 7.7)
        self.train_chunks = sorted(glob.glob(f"{output_prefix}_train_chunk*.npz"))
        self.val_chunks = sorted(glob.glob(f"{output_prefix}_val_chunk*.npz"))

        mode_str = "Grayscale (1 canal)" if self.grayscale else "RGB (3 canaux)"
        print(f"📦 CenterNet Detection Dataset: {len(self.train_chunks)} train chunks, {len(self.val_chunks)} val chunks [{mode_str}]")

    def create_tf_dataset(self, split='train', augment=True):
        """
        Crée un dataset TensorFlow qui retourne (image, {HEATMAP_KEY: heatmap, SIZE_KEY: size})
        Image: (image_size[0], image_size[1], C) où C=1 (grayscale) ou 3 (RGB)
        heatmap: (image_size[0], image_size[1], 1) ; size: (image_size[0], image_size[1], 2)
        """
        chunks = self.train_chunks if split == 'train' else self.val_chunks
        if not chunks:
            error_msg = (
                f"\n❌ ERREUR: Chunks introuvables pour la détection JAX_DETECTOR !\n"
                f"   Je m'attendais à trouver {self.output_prefix}_[split]_chunk*.npz\n"
                f"💡 LANCEZ D'ABORD : python dataset_builder/jax_detector_dataset_tools.py"
            )
            print(error_msg)
            exit(1)

        def gen():
            for chunk_path in chunks:
                with np.load(chunk_path) as data:
                    images = data['images']            # (N, H, W, C)
                    heatmaps = data[HEATMAP_KEY]        # (N, H, W, 1)
                    sizes = data[SIZE_KEY]              # (N, H, W, 2)

                    for img, heatmap, size in zip(images, heatmaps, sizes):
                        yield img, {HEATMAP_KEY: heatmap, SIZE_KEY: size}

        num_channels = 1 if self.grayscale else 3
        output_signature = (
            tf.TensorSpec(shape=self.image_size + (num_channels,), dtype=tf.float32),
            {
                HEATMAP_KEY: tf.TensorSpec(shape=self.image_size + (1,), dtype=tf.float32),
                SIZE_KEY: tf.TensorSpec(shape=self.image_size + (2,), dtype=tf.float32),
            }
        )

        ds = tf.data.Dataset.from_generator(gen, output_signature=output_signature)

        if split == 'train' and augment:
            ds = ds.shuffle(1000)

            def augment_fn(img, targets):
                heatmap = targets[HEATMAP_KEY]
                size = targets[SIZE_KEY]

                # heatmap/size : interpolation 'nearest' (préserve pic heatmap==1.0 et cellule
                # de taille ponctuelle, Story 7.1/7.3) + padding CONSTANT (fond noir) sur
                # translation ; size seul est rescalé par le facteur de zoom réellement
                # appliqué (voir _apply_geometric_and_color_augmentation).
                img, (heatmap, size) = _apply_geometric_and_color_augmentation(
                    img, [heatmap, size],
                    extra_pad_modes=["CONSTANT", "CONSTANT"],
                    extra_interp_methods=["nearest", "nearest"],
                    extra_rescale_on_zoom=[False, True],
                    augmentation_params=self.augmentation_params,
                )

                # Filet de sécurité : le heatmap reste borné [0,1] (déjà le cas par construction,
                # cohérent avec DetectionDataset.clip_by_value sur son masque). JAMAIS appliqué à
                # `size` (Task 4bis) - la carte de taille porte des magnitudes en pixels, pas des
                # valeurs [0,1] ; un clip [0,1] écraserait silencieusement tout le signal de taille.
                heatmap = tf.clip_by_value(heatmap, 0.0, 1.0)

                return img, {HEATMAP_KEY: heatmap, SIZE_KEY: size}

            ds = ds.map(augment_fn)

        ds = ds.batch(self.batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)
        return ds

    def get_dataset(self):
        train_ds = self.create_tf_dataset('train', augment=True)
        val_ds = self.create_tf_dataset('val', augment=False)
        return train_ds, val_ds


class ChessPolicyValueDataset:
    """
    Gestionnaire de dataset pour le domaine échecs (policy+value, AD-17/AD-18, Story 9.3).
    Charge les chunks .npz générés côté chess_ai (génération du dataset échecs retirée de
    ce repo, cf. spec-chess-npz-boundary-cleanup, 2026-08-01 — chess_ai est désormais
    l'unique source de vérité).
    Classe séparée des loaders existants (AD-17) - ne les modifie ni ne les étend.

    Déviation délibérée de la convention "split côté producteur" des autres loaders
    (ChunkManager/DetectionDataset/CenterNetDetectionDataset attendent des chunks
    {output_prefix}_train_chunk*.npz / {output_prefix}_val_chunk*.npz déjà séparés) :
    le générateur côté chess_ai ne produit qu'un seul flux
    {output_prefix}_chunk*.npz, sans split. Le split se fait donc ICI, au chargement,
    par fraction de CHUNKS entiers (pas d'exemple par exemple) - limite connue, acceptée
    pour une preuve de généricité (voir Dev Notes de la Story 9.3). Les chunks sont
    triés numériquement (pas lexicographiquement - "chunk10" ne doit pas passer avant
    "chunk2") puis mélangés avec une graine fixe (reproductible d'un run à l'autre)
    avant de prélever la fraction val - évite qu'un split "toujours les derniers par
    ordre alphabétique" biaise systématiquement vers les dernières parties du fichier
    PGN source (trouvé en code review, 2026-07-27).

    Pas d'augmentation de données (contrairement à DetectionDataset/CenterNetDetectionDataset) :
    flip/zoom/translation n'ont pas de sens géométrique direct sur un plateau encodé en
    planes - non demandé par le PRD/spine.
    """
    def __init__(self, output_prefix: str, batch_size: int = 32, val_split: float = 0.1, shuffle_seed: int = 42,
                 num_planes: int = 29):
        self.output_prefix = output_prefix
        self.batch_size = batch_size
        # num_planes (2026-07-29, test d'ablation historique, voir deferred-work.md) :
        # jusqu'ici 29 (NUM_PLANES) etait code en dur dans output_signature (create_tf_dataset)
        # au lieu d'etre derive des donnees reelles - cassait silencieusement tout dataset
        # dont les positions n'ont pas exactement 29 canaux (ex. CHESS_NO_HISTORY, 19
        # canaux). Defaut 29 : jamais utilise en pratique, CHESS_NO_HISTORY (seule config
        # echecs restante depuis le retrait de CHESS le 2026-08-02) fournit toujours
        # num_channels explicitement (voir get_datasets ci-dessous).
        self.num_planes = num_planes

        def _chunk_index(path):
            # Extrait l'entier de "..._chunk{N}.npz" - tri numerique, jamais lexicographique
            # (le generateur cote chess_ai n'ecrit pas d'index zero-pad).
            stem = os.path.basename(path).rsplit("_chunk", 1)[-1]
            return int(stem[:-len(".npz")])

        all_chunks = sorted(glob.glob(f"{output_prefix}_chunk*.npz"), key=_chunk_index)
        if not all_chunks:
            error_msg = (
                f"\n❌ ERREUR: Chunks introuvables pour le domaine échecs !\n"
                f"   Je m'attendais à trouver {output_prefix}_chunk*.npz\n"
                f"💡 Génère-les d'abord côté chess_ai (ce repo ne génère plus les .npz échecs)."
            )
            print(error_msg)
            exit(1)

        # Scan leger (cles uniquement via NpzFile.files, jamais les tableaux - quasi
        # gratuit meme sur des centaines de chunks) execute UNE FOIS a la construction,
        # plutot qu'un warning par chunk repete a chaque epoch dans gen() ci-dessous
        # (bruit de log signale par Aymeric, 2026-08-05). Detecte aussi une incoherence
        # (certains chunks avec "value", d'autres sans) qui serait un vrai bug de
        # generateur cote chess_ai - jamais tolere silencieusement.
        chunks_missing_value = [
            c for c in all_chunks
            if "value" not in np.load(c).files
        ]
        if chunks_missing_value and len(chunks_missing_value) != len(all_chunks):
            raise ValueError(
                f"{len(chunks_missing_value)}/{len(all_chunks)} chunks de "
                f"{output_prefix} n'ont pas la clé 'value', les autres si - dataset "
                f"incohérent (régression probable côté générateur chess_ai). "
                f"Premier chunk concerné : {os.path.basename(chunks_missing_value[0])}"
            )
        self.has_value = (len(chunks_missing_value) == 0)
        if not self.has_value:
            print(f"ℹ️  {len(all_chunks)}/{len(all_chunks)} chunks sans clé 'value' - "
                  f"value factice (0.0) utilisée pour tous (dataset sans issue de "
                  f"partie complétée, ex. chess_search_teacher - attendu, voir "
                  f"loss_params.value_weight de la config)")

        # Melange reproductible (graine fixe) avant le split - evite le biais "toujours
        # les derniers chunks numeriquement" d'un simple tri croissant + slice de queue.
        shuffled_chunks = list(all_chunks)
        np.random.RandomState(shuffle_seed).shuffle(shuffled_chunks)

        n_val = 0 if val_split <= 0 else (max(1, int(len(shuffled_chunks) * val_split)) if len(shuffled_chunks) > 1 else 0)
        self.train_chunks = shuffled_chunks[:-n_val] if n_val else shuffled_chunks
        self.val_chunks = shuffled_chunks[-n_val:] if n_val else []

        print(f"📦 Chess Policy+Value Dataset: {len(self.train_chunks)} train chunks, "
              f"{len(self.val_chunks)} val chunks (split côté chargement, val_split={val_split}, "
              f"mélange reproductible seed={shuffle_seed})")

    def create_tf_dataset(self, split='train'):
        """
        Crée un dataset TensorFlow qui retourne (position, {"policy": policy, "value": value})
        position: (8, 8, num_planes) ; policy: scalaire int32 ; value: scalaire float32

        gen() yield un CHUNK entier par itération (pas un exemple), suivi d'un
        .unbatch() - pas un exemple par exemple comme avant (2026-08-07, diagnostic
        Winston, deferred-work.md 2026-08-05 : pipeline CPU-bound, ~1,26M
        franchissements Python/epoch sur le dataset chess_search_teacher depth=12,
        ramenes a ~127 par cette regle). RAM negligeable pour ce domaine (~1,1 Mo/chunk
        chess, contrairement a un dataset d'images ou garder un chunk entier en
        memoire serait risque - voir deferred-work.md pour la mise en garde generale).
        Le point d'insertion de .unbatch() (avant .shuffle()) preserve exactement le
        buffer de melange par-exemple d'avant (1000 exemples individuels), pas par chunk.
        """
        chunks = self.train_chunks if split == 'train' else self.val_chunks
        if not chunks:
            raise ValueError(f"Aucun chunk '{split}' disponible pour le domaine échecs "
                              f"(output_prefix={self.output_prefix}, val_split trop faible ou trop peu de chunks)")

        def gen():
            for chunk_path in chunks:
                with np.load(chunk_path) as data:
                    positions = data["position"]  # (N, 8, 8, num_planes)
                    # Cast explicite en int32 (une fois par chunk, pas par exemple) :
                    # le generateur cote chess_ai sauvegarde deja "policy" en int32,
                    # mais output_signature ci-dessous l'exige STRICTEMENT
                    # (tf.data.Dataset.from_generator) - defense explicite plutot que
                    # de compter silencieusement sur le producteur.
                    policies = data["policy"].astype(np.int32)  # (N,)
                    # "value" absente pour les datasets professeur (ex. chess_search_teacher,
                    # contrat #2.6 - positions d'auto-jeu, pas d'issue de partie completee) :
                    # value factice a 0.0 plutot qu'un KeyError (Epic 10, 2026-08-04). Ne
                    # change rien pour les datasets qui fournissent deja "value"
                    # (CHESS_NO_HISTORY) - meme forme/dtype produits dans les deux cas.
                    # self.has_value deja verifie chunk par chunk une seule fois dans
                    # __init__ (coherence garantie, exception levee sinon) - pas la peine
                    # de re-verifier ni de reavertir a chaque epoch ici (revue de code,
                    # 2026-08-05).
                    if self.has_value:
                        values = data["value"].astype(np.float32)  # (N,)
                        if len(values) != len(positions):
                            raise ValueError(
                                f"{chunk_path}: 'value' a {len(values)} exemples, "
                                f"'position' en a {len(positions)} - chunk malforme, "
                                f"jamais tronque silencieusement via zip()"
                            )
                    else:
                        values = np.zeros(len(positions), dtype=np.float32)

                    yield positions, {"policy": policies, "value": values}

        output_signature = (
            tf.TensorSpec(shape=(None, 8, 8, self.num_planes), dtype=tf.float32),
            {
                "policy": tf.TensorSpec(shape=(None,), dtype=tf.int32),
                "value": tf.TensorSpec(shape=(None,), dtype=tf.float32),
            }
        )

        ds = tf.data.Dataset.from_generator(gen, output_signature=output_signature)
        ds = ds.unbatch()

        if split == 'train':
            ds = ds.shuffle(1000)

        ds = ds.batch(self.batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)
        return ds

    def get_dataset(self):
        train_ds = self.create_tf_dataset('train')
        val_ds = self.create_tf_dataset('val') if self.val_chunks else None
        return train_ds, val_ds


class ChessLegalMovesDataset:
    """
    Gestionnaire de dataset pour la tache multi-label "coups legaux" (contrat .npz
    chess_legal_moves, cote chess_ai). Duplique ChessPolicyValueDataset ci-dessus
    (meme tri numerique des chunks, meme melange reproductible + split par fraction
    de CHUNKS entiers - convention deja en place, pas une nouvelle regle) mais
    retourne (position, legal_mask) au lieu de (position, {"policy", "value"}).

    Classe separee plutot qu'une generalisation de ChessPolicyValueDataset (flag
    "avec/sans value") : meme discipline que le reste de ce fichier (CHESS_NO_HISTORY
    est une copie de config, pas un flag sur une classe).
    """
    def __init__(self, output_prefix: str, batch_size: int = 32, val_split: float = 0.1, shuffle_seed: int = 42,
                 num_planes: int = 29, num_moves: int = 4672):
        self.output_prefix = output_prefix
        self.batch_size = batch_size
        self.num_planes = num_planes
        self.num_moves = num_moves

        def _chunk_index(path):
            stem = os.path.basename(path).rsplit("_chunk", 1)[-1]
            return int(stem[:-len(".npz")])

        all_chunks = sorted(glob.glob(f"{output_prefix}_chunk*.npz"), key=_chunk_index)
        if not all_chunks:
            error_msg = (
                f"\n❌ ERREUR: Chunks introuvables pour le domaine échecs (coups légaux) !\n"
                f"   Je m'attendais à trouver {output_prefix}_chunk*.npz\n"
                f"💡 Génère-les d'abord côté chess_ai (ce repo ne génère plus les .npz échecs)."
            )
            print(error_msg)
            exit(1)

        shuffled_chunks = list(all_chunks)
        np.random.RandomState(shuffle_seed).shuffle(shuffled_chunks)

        n_val = 0 if val_split <= 0 else (max(1, int(len(shuffled_chunks) * val_split)) if len(shuffled_chunks) > 1 else 0)
        self.train_chunks = shuffled_chunks[:-n_val] if n_val else shuffled_chunks
        self.val_chunks = shuffled_chunks[-n_val:] if n_val else []

        print(f"📦 Chess Legal Moves Dataset: {len(self.train_chunks)} train chunks, "
              f"{len(self.val_chunks)} val chunks (split côté chargement, val_split={val_split}, "
              f"mélange reproductible seed={shuffle_seed})")

    def create_tf_dataset(self, split='train'):
        """
        Crée un dataset TensorFlow qui retourne (position, legal_mask).
        position: (8, 8, num_planes) ; legal_mask: (4672,) float32 (cast depuis
        l'int8 du .npz - sigmoid BCE, ChessLegalMovesStrategy, attend un dtype flottant).
        """
        chunks = self.train_chunks if split == 'train' else self.val_chunks
        if not chunks:
            raise ValueError(f"Aucun chunk '{split}' disponible pour le domaine échecs (coups légaux) "
                              f"(output_prefix={self.output_prefix}, val_split trop faible ou trop peu de chunks)")

        def gen():
            for chunk_path in chunks:
                with np.load(chunk_path) as data:
                    positions = data["position"]      # (N, 8, 8, num_planes)
                    legal_masks = data["legal_mask"]  # (N, 4672) int8

                    # Valide une fois par CHUNK (pas par exemple - cout negligeable) que
                    # les formes reelles correspondent a la config attendue - sinon
                    # l'erreur ne surgirait qu'en aval, comme une erreur de shape TF
                    # opaque au milieu du pipeline tf.data (trouve en revue adversariale).
                    assert positions.shape[-1] == self.num_planes, (
                        f"{chunk_path} : position a {positions.shape[-1]} canaux, "
                        f"attendu {self.num_planes} (num_channels de la config)"
                    )
                    assert legal_masks.shape[-1] == self.num_moves, (
                        f"{chunk_path} : legal_mask a {legal_masks.shape[-1]} coups, "
                        f"attendu {self.num_moves} (num_classes de la config)"
                    )

                    for pos, mask in zip(positions, legal_masks):
                        yield pos, mask.astype(np.float32)

        output_signature = (
            tf.TensorSpec(shape=(8, 8, self.num_planes), dtype=tf.float32),
            tf.TensorSpec(shape=(self.num_moves,), dtype=tf.float32),
        )

        ds = tf.data.Dataset.from_generator(gen, output_signature=output_signature)

        if split == 'train':
            ds = ds.shuffle(1000)

        ds = ds.batch(self.batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)
        return ds

    def get_dataset(self):
        train_ds = self.create_tf_dataset('train')
        val_ds = self.create_tf_dataset('val') if self.val_chunks else None
        return train_ds, val_ds


# Tokens spéciaux du domaine chess_move_token (Epic 11, AD-30 — spine
# architecture-chess-move-token-2026-08-10). Source unique : tout autre module
# (model_library.py) importe ces constantes, ne les redéfinit jamais comme littéraux
# indépendants. Espace d'entrée (embedding) = NUM_MOVES(4672) + 2 = 4674 ; la tête de
# sortie policy reste strictement sur les 4672 classes existantes (AD-22 hérité),
# BOS/PAD n'y apparaissent jamais.
CHESS_MOVE_TOKEN_BOS = 4672
CHESS_MOVE_TOKEN_PAD = 4673


class ChessMoveTokenDataset:
    """
    Gestionnaire de dataset pour le domaine chess_move_token (Epic 11, spike — AD-27/
    AD-28/AD-30, spine architecture-chess-move-token-2026-08-10). Contrairement à tous
    les chargeurs échecs ci-dessus, le dataset spike est un FICHIER UNIQUE (pas de
    convention `_chunk*.npz`) au format CSR : `move_tokens` (tokens à plat, toutes
    positions concaténées) + `move_token_offsets` (bornes N+1 pour N positions).

    Retourne (sequence, policy) — pas un dict {"policy", "value"} comme
    ChessPolicyValueDataset : ce domaine n'a pas de tête value (AD-26).
    """
    def __init__(self, npz_path: str, batch_size: int = 32, val_split: float = 0.1, shuffle_seed: int = 42):
        self.npz_path = npz_path
        self.batch_size = batch_size

        if not os.path.exists(npz_path):
            error_msg = (
                f"\n❌ ERREUR: Dataset spike chess_move_token introuvable !\n"
                f"   Je m'attendais à trouver {npz_path}\n"
                f"💡 Génère-le d'abord côté chess_ai (chess_move_token_spike_dataset.py)."
            )
            print(error_msg)
            exit(1)

        # Fichier unique et petit (spike, 3190 positions/~310K tokens) - chargé
        # entièrement en mémoire une fois, contrairement aux chargeurs chunkés
        # ci-dessus (pas de contrainte de streaming pertinente ici).
        with np.load(npz_path) as data:
            self.policy = data["policy"].astype(np.int32)                      # (N,)
            self.move_tokens = data["move_tokens"].astype(np.int32)            # (T,) à plat
            self.move_token_offsets = data["move_token_offsets"].astype(np.int64)  # (N+1,) bornes CSR

        n_examples = len(self.policy)
        assert len(self.move_token_offsets) == n_examples + 1, (
            f"{npz_path} : move_token_offsets a {len(self.move_token_offsets)} entrées, "
            f"attendu {n_examples + 1} (N+1 bornes CSR pour {n_examples} positions) - "
            f"dataset spike malformé"
        )

        # Mélange reproductible des EXEMPLES (pas de chunks - un seul fichier, AD-27)
        # avant le split fraction train/val - même convention que
        # ChessPolicyValueDataset/ChessLegalMovesDataset (évite le biais d'ordre déjà
        # documenté le 2026-07-27), au grain de l'exemple plutôt que du chunk.
        indices = np.arange(n_examples)
        np.random.RandomState(shuffle_seed).shuffle(indices)

        n_val = 0 if val_split <= 0 else (max(1, int(n_examples * val_split)) if n_examples > 1 else 0)
        self.train_indices = indices[:-n_val] if n_val else indices
        self.val_indices = indices[-n_val:] if n_val else np.array([], dtype=np.int64)

        # Longueur de séquence FIXE et GLOBALE (pas un padding dynamique par batch,
        # contrairement à la 1ère version d'AD-28) - trouvé par exécution réelle
        # (Aymeric, 2026-08-10, nvtop) : `Trainer._train_step` est `@jax.jit`
        # (trainer.py:229) et cache par FORME d'entrée - un padding par-batch (forme
        # `(B, L)` avec L variable, ancien `padded_shapes=([None], [])`) déclenchait
        # une recompilation XLA a quasiment CHAQUE step (GPU a 0% pendant ~5s,
        # visible dans nvtop, ~3.7s/it au lieu de quelques dizaines de ms). Calculée
        # depuis les VRAIES données (pas un littéral 301 en dur, qui deviendrait
        # silencieusement faux si le dataset spike changeait) : max(longueur
        # d'historique) + 1 pour le BOS préfixé (voir create_tf_dataset). Une seule
        # forme de batch pour tout le run -> une seule compilation JIT.
        history_lengths = self.move_token_offsets[1:] - self.move_token_offsets[:-1]
        self.max_seq_len = int(history_lengths.max()) + 1  # +1 = BOS

        print(f"📦 Chess Move-Token Dataset (spike) : {len(self.train_indices)} positions train, "
              f"{len(self.val_indices)} positions val (split côté chargement, val_split={val_split}, "
              f"mélange reproductible seed={shuffle_seed}), longueur de séquence fixe={self.max_seq_len} "
              f"(historique max + BOS, une seule compilation JIT pour tout le run)")

    def create_tf_dataset(self, split='train'):
        """
        Crée un dataset TensorFlow qui retourne (sequence, policy).
        sequence: (L,) int32, longueur variable, paddée à GAUCHE par batch (AD-28) ;
        policy: scalaire int32 (index du coup joué, espace 4672).
        """
        indices = self.train_indices if split == 'train' else self.val_indices
        if len(indices) == 0:
            raise ValueError(f"Aucune position '{split}' disponible pour chess_move_token "
                              f"(npz_path={self.npz_path}, val_split trop faible ou trop peu de positions)")

        def gen():
            for i in indices:
                start, end = self.move_token_offsets[i], self.move_token_offsets[i + 1]
                history = self.move_tokens[start:end]
                # BOS préfixé à l'historique (AD-30) AVANT le reverse de la recette de
                # padding à gauche ci-dessous (AD-28) - il se retrouve donc juste après
                # le padding une fois la séquence batchée, jamais tronqué.
                sequence = np.concatenate(([CHESS_MOVE_TOKEN_BOS], history)).astype(np.int32)
                yield sequence, self.policy[i]

        output_signature = (
            tf.TensorSpec(shape=(None,), dtype=tf.int32),
            tf.TensorSpec(shape=(), dtype=tf.int32),
        )
        ds = tf.data.Dataset.from_generator(gen, output_signature=output_signature)

        if split == 'train':
            ds = ds.shuffle(1000)

        # Recette de padding à GAUCHE (AD-28) - tf.data.Dataset.padded_batch ne padde
        # nativement qu'à droite (vérifié) : on inverse chaque séquence avant padding
        # (donc "avant" en ordre inversé = "après" une fois re-inversée), on padde, puis
        # on re-inverse le batch obtenu. Le dernier token réel est ainsi toujours à
        # l'indice -1 du tenseur, quelle que soit la longueur réelle de la séquence.
        #
        # padded_shapes=([self.max_seq_len], []) - longueur FIXE et GLOBALE (pas
        # `[None]`, qui padderait dynamiquement à la longueur max DU BATCH COURANT
        # uniquement) - voir le commentaire sur self.max_seq_len (__init__) : une
        # forme de batch variable ferait recompiler `Trainer._train_step` (@jax.jit)
        # à quasiment chaque step (trouvé par exécution réelle, GPU à 0% ~5s entre
        # chaque step dans nvtop). Toutes les séquences réelles sont <= max_seq_len
        # par construction (calculé depuis les mêmes données) - jamais tronquées.
        ds = ds.map(lambda seq, label: (tf.reverse(seq, axis=[0]), label))
        ds = ds.padded_batch(
            self.batch_size,
            padded_shapes=([self.max_seq_len], []),
            padding_values=(np.int32(CHESS_MOVE_TOKEN_PAD), np.int32(0)),
            drop_remainder=True,
        )
        ds = ds.map(lambda seq_batch, label_batch: (tf.reverse(seq_batch, axis=[1]), label_batch))

        ds = ds.prefetch(tf.data.AUTOTUNE)
        return ds

    def get_dataset(self):
        train_ds = self.create_tf_dataset('train')
        val_ds = self.create_tf_dataset('val') if len(self.val_indices) else None
        return train_ds, val_ds


class ChessTokenCandidateDataset:
    """
    Gestionnaire de dataset pour le domaine chess_token (spec-chess-token-candidate-model,
    2026-08-13, spike, CAP-3). Fichier unique .npz (pas de chunks, même convention que
    ChessMoveTokenDataset ci-dessus, contrairement aux 4 chargeurs échecs chunkés) - contrat
    de données : companion token-candidate-dataset-schema.md (chess_ai) : `token_position`
    (N,64) int32 dans [0,12], `global_flags` (N,6) float32 binaire, `candidate_moves` (N,50)
    int32 dans [-1,4671] (sentinel -1 = padding ; [0,4671] = [0,4672) = espace complet
    move_to_index, NUM_MOVES=4672 chess_ai/chess_target_encoding.py:243 - PAS 4660, qui n'était
    que le max observé sur le seul run de génération réel disponible au moment de l'écriture,
    pas la borne réelle du contrat), `candidate_mask` (N,50) int8, `candidate_label`
    (N,) int32 dans [0,50) - index de SLOT, pas un index move_to_index.

    Retourne (packed_features, {"candidate_label": ..., "candidate_mask": ...}) - PAS
    (packed_features, candidate_label) comme ChessMoveTokenDataset : packed_features
    concatène [token_position | global_flags | candidate_moves] en UN SEUL tenseur
    (B, 64+6+num_candidates) int32, seule entrée consommée par
    ChessTokenCandidateModel.__call__ (model_library.py, voir sa docstring - contrainte réelle
    de Trainer._train_step/_eval_step, `apply_fn(vars, images, ...)` avec UN SEUL tenseur
    `images`, jamais plusieurs entrées nommées). candidate_mask NE fait PAS partie de
    packed_features (le modèle n'en a pas besoin en interne, les slots de padding sont
    neutralisés côté modèle via `safe_moves`) - il voyage côté targets, pour le masquage de la
    loss/métrique (ChessTokenStrategy, task_strategies.py) - même patron dict-target que
    CenterNetDetectionStrategy ({HEATMAP_KEY, SIZE_KEY}), pas un nouveau mécanisme.
    """
    def __init__(self, npz_path: str, batch_size: int = 32, val_split: float = 0.1, shuffle_seed: int = 42,
                 num_candidates: int = 50):
        self.npz_path = npz_path
        self.batch_size = batch_size
        self.num_candidates = num_candidates

        if not os.path.exists(npz_path):
            error_msg = (
                f"\n❌ ERREUR: Dataset spike chess_token introuvable !\n"
                f"   Je m'attendais à trouver {npz_path}\n"
                f"💡 Génère-le d'abord côté chess_ai (chess_token_candidate_spike_dataset.py)."
            )
            print(error_msg)
            exit(1)

        # Fichier unique, chargé entièrement en mémoire une fois (412 574 positions au run
        # 2026-08-13 - packed_features (N, 120) int32 ~198 Mo, largement dans le budget RAM
        # habituel de ce repo, même discipline que ChessMoveTokenDataset ci-dessus, pas de
        # streaming chunké pertinent ici). token_position/global_flags/candidate_moves sont
        # supprimés (del) juste après la construction de packed_features ci-dessous : ils ne
        # sont lus nulle part ailleurs dans cette classe (vérifié - create_tf_dataset ne
        # consomme que packed_features/candidate_label/candidate_mask), les garder vivants
        # doublerait inutilement ~190 Mo de RAM pour la durée de vie de l'objet (code review,
        # 2026-08-13).
        required_keys = ["token_position", "global_flags", "candidate_moves", "candidate_mask", "candidate_label"]
        with np.load(npz_path) as data:
            for key in required_keys:
                if key not in data.files:
                    error_msg = (
                        f"\n❌ ERREUR: Dataset spike chess_token malformé !\n"
                        f"   Clé '{key}' absente de {npz_path} (clés présentes : {list(data.files)})\n"
                        f"💡 Vérifie que le fichier a bien été généré par chess_token_candidate_spike_dataset.py "
                        f"(côté chess_ai) avec le schéma attendu (token_position, global_flags, candidate_moves, "
                        f"candidate_mask, candidate_label)."
                    )
                    print(error_msg)
                    exit(1)

            raw_token_position = data["token_position"]
            raw_global_flags = data["global_flags"]
            raw_candidate_moves = data["candidate_moves"]
            raw_candidate_mask = data["candidate_mask"]
            raw_candidate_label = data["candidate_label"]

            # global_flags doit être binaire (0/1, voir companion schema) - vérifié AVANT le
            # cast .astype(np.int32) ci-dessous, sinon un bruit non-binaire (ex. 0.3) serait
            # silencieusement tronqué à 0 par le cast entier sans jamais lever d'erreur.
            assert np.all(np.isin(raw_global_flags, [0, 1])), (
                f"{npz_path} : global_flags contient des valeurs hors {{0,1}} (binaire attendu) - "
                f"dataset chess_token malformé (min={raw_global_flags.min()}, max={raw_global_flags.max()})"
            )

            token_position = raw_token_position.astype(np.int32)    # (N, 64)
            # global_flags est binaire (0.0/1.0, voir companion schema) - cast int32 sans
            # perte (déjà vérifié ci-dessus), permet de packer avec token_position/
            # candidate_moves dans UN SEUL tenseur int32 (voir docstring classe).
            global_flags = raw_global_flags.astype(np.int32)        # (N, 6)
            candidate_moves = raw_candidate_moves.astype(np.int32)  # (N, num_candidates)
            self.candidate_mask = raw_candidate_mask.astype(np.int32)    # (N, num_candidates)
            self.candidate_label = raw_candidate_label.astype(np.int32)  # (N,)

        n_examples = len(self.candidate_label)
        assert token_position.shape == (n_examples, 64), (
            f"{npz_path} : token_position a shape {token_position.shape}, attendu (N, 64) - "
            f"dataset chess_token malformé"
        )
        assert global_flags.shape == (n_examples, 6), (
            f"{npz_path} : global_flags a shape {global_flags.shape}, attendu (N, 6) - "
            f"dataset chess_token malformé"
        )
        assert candidate_moves.shape == (n_examples, self.num_candidates), (
            f"{npz_path} : candidate_moves a shape {candidate_moves.shape}, attendu "
            f"(N, {self.num_candidates}) - num_candidates de la config ne correspond pas au fichier réel"
        )
        assert self.candidate_mask.shape == candidate_moves.shape, (
            f"{npz_path} : candidate_mask {self.candidate_mask.shape} != candidate_moves "
            f"{candidate_moves.shape} - dataset chess_token malformé"
        )
        assert self.candidate_label.shape == (n_examples,), (
            f"{npz_path} : candidate_label a shape {self.candidate_label.shape}, attendu (N,) - "
            f"dataset chess_token malformé"
        )
        assert np.all((token_position >= 0) & (token_position < 13)), (
            f"{npz_path} : token_position contient des valeurs hors [0,13) - dataset chess_token "
            f"malformé (min={token_position.min()}, max={token_position.max()})"
        )
        assert np.all(self.candidate_label >= 0), (
            f"{npz_path} : candidate_label contient des valeurs negatives - dataset chess_token malformé "
            f"(min={self.candidate_label.min()})"
        )
        # candidate_label pointe un index de SLOT (pas un index move_to_index) - il ne doit
        # jamais pointer un slot de padding (candidate_mask==0), sinon la loss/l'argmax
        # traiteraient silencieusement un slot factice comme le coup du professeur.
        assert np.all(self.candidate_mask[np.arange(n_examples), self.candidate_label] == 1), (
            f"{npz_path} : au moins un candidate_label pointe un slot marqué comme padding "
            f"(candidate_mask==0) - dataset chess_token malformé"
        )

        # Vecteur packé UNE FOIS à la construction (pas recalculé par exemple dans gen(),
        # ni à chaque epoch) - concat numpy vectorisé, coût négligeable devant le chargement
        # du .npz lui-même. Ordre = [token_position(64) | global_flags(6) |
        # candidate_moves(num_candidates)], celui attendu par ChessTokenCandidateModel.
        self.packed_features = np.concatenate(
            [token_position, global_flags, candidate_moves], axis=1
        ).astype(np.int32)  # (N, 64+6+num_candidates)
        # token_position/global_flags/candidate_moves ne sont plus référencés nulle part
        # ailleurs dans cette classe une fois packed_features construit (voir commentaire
        # RAM ci-dessus) - suppression explicite plutôt que de les laisser vivre en double
        # jusqu'à la fin de vie de l'objet.
        del token_position, global_flags, candidate_moves

        # Mélange reproductible des EXEMPLES (pas de chunks - un seul fichier) avant le
        # split fraction train/val - même convention que ChessMoveTokenDataset ci-dessus.
        indices = np.arange(n_examples)
        np.random.RandomState(shuffle_seed).shuffle(indices)

        n_val = 0 if val_split <= 0 else (max(1, int(n_examples * val_split)) if n_examples > 1 else 0)
        self.train_indices = indices[:-n_val] if n_val else indices
        self.val_indices = indices[-n_val:] if n_val else np.array([], dtype=np.int64)

        print(f"📦 Chess Token Candidate Dataset (spike) : {len(self.train_indices)} positions train, "
              f"{len(self.val_indices)} positions val (split côté chargement, val_split={val_split}, "
              f"mélange reproductible seed={shuffle_seed}), num_candidates={self.num_candidates}")

    def create_tf_dataset(self, split='train'):
        """
        Crée un dataset TensorFlow qui retourne (packed_features, {"candidate_label",
        "candidate_mask"}). packed_features : (64+6+num_candidates,) int32 - toutes les
        formes sont FIXES (pas de padding dynamique nécessaire, contrairement à
        ChessMoveTokenDataset) - `ds.batch` simple suffit (même pattern que
        ChessLegalMovesDataset, data_management.py:804-848).
        """
        indices = self.train_indices if split == 'train' else self.val_indices
        if len(indices) == 0:
            raise ValueError(f"Aucune position '{split}' disponible pour chess_token "
                              f"(npz_path={self.npz_path}, val_split trop faible ou trop peu de positions)")
        # drop_remainder=True (ci-dessous) yielderait silencieusement ZÉRO batch si ce split
        # a moins d'exemples que batch_size - la boucle train/eval correspondante ne ferait
        # alors rien du tout, sans aucune erreur (piège classique tf.data). Erreur explicite
        # ici plutôt qu'un repli sur drop_remainder=False (qui risquerait un batch de forme
        # différente et une recompilation JIT dans ce pipeline très sensible aux formes,
        # cf. ChessMoveTokenDataset.__init__ ci-dessus) - erreur bruyante et immédiate,
        # cohérente avec le reste de ce fichier.
        if len(indices) < self.batch_size:
            raise ValueError(
                f"Split '{split}' de chess_token n'a que {len(indices)} exemple(s), inférieur à "
                f"batch_size={self.batch_size} : avec drop_remainder=True cela produirait "
                f"silencieusement ZÉRO batch (npz_path={self.npz_path}). Réduis batch_size ou "
                f"augmente val_split/le volume de données."
            )

        def gen():
            for i in indices:
                yield (
                    self.packed_features[i],
                    {
                        "candidate_label": self.candidate_label[i],
                        "candidate_mask": self.candidate_mask[i],
                    },
                )

        output_signature = (
            tf.TensorSpec(shape=(64 + 6 + self.num_candidates,), dtype=tf.int32),
            {
                "candidate_label": tf.TensorSpec(shape=(), dtype=tf.int32),
                "candidate_mask": tf.TensorSpec(shape=(self.num_candidates,), dtype=tf.int32),
            },
        )
        ds = tf.data.Dataset.from_generator(gen, output_signature=output_signature)

        if split == 'train':
            ds = ds.shuffle(1000)

        ds = ds.batch(self.batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)
        return ds

    def get_dataset(self):
        train_ds = self.create_tf_dataset('train')
        val_ds = self.create_tf_dataset('val') if len(self.val_indices) else None
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
    
    if task_type in ["classification", "kepler"]:
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

    elif task_type == "detection_centernet":
        dataset_manager = CenterNetDetectionDataset(
            output_prefix=config["output_prefix"],
            image_size=config["image_size"],
            batch_size=backend_config["micro_batch_size"],
            grayscale=config.get("grayscale", False),
            augmentation_params=aug_params
        )
        train_ds = dataset_manager.create_tf_dataset('train', augment=True)
        val_ds = dataset_manager.create_tf_dataset('val', augment=False)

        if config.get("epochs", 1) == 0:
            print("✅ Mode vérification: epochs=0, vérification des chunks requise")

        return train_ds, val_ds

    elif task_type == "chess_policy_value":
        dataset_manager = ChessPolicyValueDataset(
            output_prefix=config["output_prefix"],
            batch_size=backend_config["micro_batch_size"],
            val_split=config.get("val_split", 0.1),
            # Defaut 29 = ancien num_channels de la config CHESS (avec historique, retiree
            # le 2026-08-02) - CHESS_NO_HISTORY fournit toujours num_channels explicitement,
            # ce defaut n'est en pratique jamais utilise.
            num_planes=config.get("num_channels", 29),
        )
        return dataset_manager.get_dataset()

    elif task_type == "chess_legal_moves":
        dataset_manager = ChessLegalMovesDataset(
            output_prefix=config["output_prefix"],
            batch_size=backend_config["micro_batch_size"],
            val_split=config.get("val_split", 0.1),
            num_planes=config.get("num_channels", 29),
            num_moves=config.get("num_classes", 4672),
        )
        return dataset_manager.get_dataset()

    elif task_type == "chess_move_token":
        # "output_prefix" porte ici le chemin LITTÉRAL du fichier spike unique (pas un
        # préfixe de glob `_chunk*.npz` comme les autres domaines échecs) - AD-27.
        dataset_manager = ChessMoveTokenDataset(
            npz_path=config["output_prefix"],
            batch_size=backend_config["micro_batch_size"],
            val_split=config.get("val_split", 0.1),
        )
        return dataset_manager.get_dataset()

    elif task_type == "chess_token":
        # "output_prefix" porte ici le chemin LITTÉRAL du fichier spike unique (pas un
        # préfixe de glob `_chunk*.npz` comme les autres domaines échecs) - même
        # convention que chess_move_token ci-dessus.
        dataset_manager = ChessTokenCandidateDataset(
            npz_path=config["output_prefix"],
            batch_size=backend_config["micro_batch_size"],
            val_split=config.get("val_split", 0.1),
            num_candidates=config.get("num_classes", 50),
        )
        return dataset_manager.get_dataset()

    else:
        raise ValueError(f"Task type inconnu: {task_type}")
