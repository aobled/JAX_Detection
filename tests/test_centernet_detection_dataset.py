"""
Test standalone pour CenterNetDetectionDataset (Story 7.5).
Execution: python3 test_centernet_detection_dataset.py
"""

import sys
import os

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, os.path.join(_REPO_ROOT, "dataset_builder"))

import json
import os
import shutil
import tempfile

import numpy as np
import tensorflow as tf
from PIL import Image

from data_management import CenterNetDetectionDataset
from detection_target_encoding import HEATMAP_KEY, SIZE_KEY
from jax_detector_dataset_tools import process_detector_dataset


IMAGE_SIZE = (224, 224)
# Bbox choisie pour retomber pres du centre de l'image target apres rescale, afin de
# rester dans la zone de central_crop meme a zoom_factor=0.2 (crop_frac ~0.87 au minimum)
RAW_BBOX = [280, 190, 80, 60]  # x, y, w, h en pixels source (image source 640x480)
ORIG_SIZE = (640, 480)


def _make_fake_dataset(root_dir):
    os.makedirs(root_dir, exist_ok=True)
    img = Image.new("RGB", ORIG_SIZE, color=(60, 90, 130))
    img_filename = "fake_0.jpg"
    img_path = os.path.join(root_dir, img_filename)
    img.save(img_path)

    annotation = {
        "image": {"file_name": img_filename},
        "annotation": {"bbox": RAW_BBOX},
    }
    with open(os.path.join(root_dir, "fake_0.json"), "w") as f:
        json.dump(annotation, f)


def _build_chunk(out_dir):
    src_dir = tempfile.mkdtemp(prefix="test_cnd_src_")
    try:
        _make_fake_dataset(src_dir)
        process_detector_dataset(
            root_dirs=[src_dir],
            output_dir=out_dir,
            split_name="train",
            target_size=IMAGE_SIZE,
            max_boxes=20,
            chunk_size=2000,
            grayscale=True,
        )
    finally:
        shutil.rmtree(src_dir, ignore_errors=True)


def test_load_without_augmentation_keys_and_shapes():
    out_dir = tempfile.mkdtemp(prefix="test_cnd_out1_")
    try:
        _build_chunk(out_dir)
        output_prefix = os.path.join(out_dir, "jax_detector_targets")

        dataset = CenterNetDetectionDataset(
            output_prefix=output_prefix,
            image_size=IMAGE_SIZE,
            batch_size=1,
            grayscale=True,
            augmentation_params={},
        )
        ds = dataset.create_tf_dataset('train', augment=False)

        img_batch, targets = next(iter(ds.take(1)))
        assert set(targets.keys()) == {HEATMAP_KEY, SIZE_KEY}
        assert img_batch.shape == (1, 224, 224, 1)
        assert targets[HEATMAP_KEY].shape == (1, 224, 224, 1)
        assert targets[SIZE_KEY].shape == (1, 224, 224, 2)

        heatmap = targets[HEATMAP_KEY].numpy()
        assert np.isclose(heatmap.max(), 1.0), "pic du heatmap non-augmenté doit être exactement 1.0"
        print("OK - test_load_without_augmentation_keys_and_shapes")
    finally:
        shutil.rmtree(out_dir, ignore_errors=True)


def test_zoom_preserves_heatmap_peak_and_rescales_size():
    out_dir = tempfile.mkdtemp(prefix="test_cnd_out2_")
    try:
        _build_chunk(out_dir)
        output_prefix = os.path.join(out_dir, "jax_detector_targets")

        # 1. Baseline sans augmentation : taille originale (avant zoom) au pixel du centre
        baseline_ds = CenterNetDetectionDataset(
            output_prefix=output_prefix, image_size=IMAGE_SIZE, batch_size=1,
            grayscale=True, augmentation_params={},
        )
        _, baseline_targets = next(iter(baseline_ds.create_tf_dataset('train', augment=False).take(1)))
        baseline_size = baseline_targets[SIZE_KEY].numpy()
        baseline_nonzero_w = baseline_size[..., 0][baseline_size[..., 0] > 0]
        assert baseline_nonzero_w.size >= 1
        original_w = float(baseline_nonzero_w.max())

        # 2. Forcer do_zoom=True et scale=1.15 fixe (Task 6 - pas de dépendance au tirage aléatoire)
        FIXED_SCALE = 1.15
        orig_uniform = tf.random.uniform

        def fake_uniform(shape, minval=0, maxval=1, **kwargs):
            if minval == 0 and maxval == 1:
                return tf.constant(0.99, dtype=tf.float32)  # force do_zoom = True (> 0.5)
            return tf.constant(FIXED_SCALE, dtype=tf.float32)  # force scale exacte

        zoom_ds = CenterNetDetectionDataset(
            output_prefix=output_prefix, image_size=IMAGE_SIZE, batch_size=1,
            grayscale=True, augmentation_params={"zoom_factor": 0.2},
        )
        tf.random.uniform = fake_uniform
        try:
            ds = zoom_ds.create_tf_dataset('train', augment=True)
            img_batch, targets = next(iter(ds.take(1)))
        finally:
            tf.random.uniform = orig_uniform

        heatmap = targets[HEATMAP_KEY].numpy()
        size = targets[SIZE_KEY].numpy()

        # Le pic du heatmap doit rester exactement 1.0 (nearest-neighbor, pas d'interpolation)
        assert np.isclose(heatmap.max(), 1.0), f"pic du heatmap altéré par le zoom : {heatmap.max()}"

        # La carte de taille doit rester ponctuelle (peu de pixels non-nuls), pas un flou diffus
        nonzero_count = int(np.sum(size[..., 0] > 0))
        assert 0 < nonzero_count <= 4, f"carte de taille non ponctuelle après zoom : {nonzero_count} pixels non-nuls"

        # La valeur non-nulle doit valoir taille_originale * scale (± tolérance d'arrondi pixel)
        augmented_w = float(size[..., 0][size[..., 0] > 0].max())
        expected_w = original_w * FIXED_SCALE
        assert abs(augmented_w - expected_w) <= 3.0, (
            f"taille augmentée {augmented_w} != taille_originale*scale {expected_w} (tolérance 3px)"
        )
        # Preuve que le clip [0,1] n'est PAS appliqué à la taille (Task 4bis) : la valeur
        # dépasse largement 1.0, contrairement à ce qu'un clip_by_value(0,1) produirait.
        assert augmented_w > 1.0

        print("OK - test_zoom_preserves_heatmap_peak_and_rescales_size")
    finally:
        shutil.rmtree(out_dir, ignore_errors=True)


def test_zoom_out_keeps_size_consistent_with_clamped_image():
    """
    Regression du bug MEDIUM trouve par la revue independante de la Story 7.5 :
    pour scale<1 (zoom arriere), crop_frac = clip(1/scale, 0.1, 1.0) est clampe a 1.0
    (l'image reste inchangee), mais une premiere version du code multipliait quand meme
    la taille par le `scale` brut - desynchronisant silencieusement la cible de taille
    de l'image sur ~la moitie des tirages de zoom. Le facteur reellement applique doit
    etre 1/crop_frac (donc 1.0 ici), pas `scale`.
    """
    out_dir = tempfile.mkdtemp(prefix="test_cnd_out3_")
    try:
        _build_chunk(out_dir)
        output_prefix = os.path.join(out_dir, "jax_detector_targets")

        baseline_ds = CenterNetDetectionDataset(
            output_prefix=output_prefix, image_size=IMAGE_SIZE, batch_size=1,
            grayscale=True, augmentation_params={},
        )
        _, baseline_targets = next(iter(baseline_ds.create_tf_dataset('train', augment=False).take(1)))
        baseline_size = baseline_targets[SIZE_KEY].numpy()
        baseline_nonzero_w = baseline_size[..., 0][baseline_size[..., 0] > 0]
        original_w = float(baseline_nonzero_w.max())

        FIXED_SCALE = 0.85  # < 1 : crop_frac = 1/0.85 clampe a 1.0, image inchangee
        orig_uniform = tf.random.uniform

        def fake_uniform(shape, minval=0, maxval=1, **kwargs):
            if minval == 0 and maxval == 1:
                return tf.constant(0.99, dtype=tf.float32)  # force do_zoom = True
            return tf.constant(FIXED_SCALE, dtype=tf.float32)

        zoom_ds = CenterNetDetectionDataset(
            output_prefix=output_prefix, image_size=IMAGE_SIZE, batch_size=1,
            grayscale=True, augmentation_params={"zoom_factor": 0.2},
        )
        tf.random.uniform = fake_uniform
        try:
            ds = zoom_ds.create_tf_dataset('train', augment=True)
            _, targets = next(iter(ds.take(1)))
        finally:
            tf.random.uniform = orig_uniform

        size = targets[SIZE_KEY].numpy()
        augmented_w = float(size[..., 0][size[..., 0] > 0].max())

        # Le facteur reellement applique est 1.0 (clamp), pas 0.85 : la taille doit rester
        # ~inchangee, pas 0.85x plus petite
        assert abs(augmented_w - original_w) <= 3.0, (
            f"taille modifiée alors que le zoom est clampé (image inchangée) : "
            f"attendu ~{original_w}, obtenu {augmented_w}"
        )
        print("OK - test_zoom_out_keeps_size_consistent_with_clamped_image")
    finally:
        shutil.rmtree(out_dir, ignore_errors=True)


if __name__ == "__main__":
    test_load_without_augmentation_keys_and_shapes()
    test_zoom_preserves_heatmap_peak_and_rescales_size()
    test_zoom_out_keeps_size_consistent_with_clamped_image()
    print("Tous les tests sont passés.")
