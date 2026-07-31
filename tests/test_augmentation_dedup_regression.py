"""
Regression pixel-level pour le refactor de dédup de l'augmentation entre
DetectionDataset et CenterNetDetectionDataset (data_management.py).

Fonctionnement capture/comparaison :
- Au premier lancement (avant refactor), aucune baseline n'existe encore sous
  tests/fixtures/augmentation_dedup_baseline/ : ce script les CAPTURE (comportement
  de référence, pré-refactor).
- Aux lancements suivants (après refactor), les baselines existent : ce script
  COMPARE la sortie actuelle à la baseline et échoue au moindre écart numérique.

Paramètres d'augmentation = valeurs réelles de production (FIGHTERJET_DETECTION
pour DetectionDataset, JAX_DETECTOR pour CenterNetDetectionDataset, dataset_configs.py).
Tirages aléatoires mockés (comme tests/test_centernet_detection_dataset.py) pour forcer
un chemin déterministe couvrant flip_v, flip_h, translation ET zoom simultanément.

Execution: python3 tests/test_augmentation_dedup_regression.py
"""

import sys
import os

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, os.path.join(_REPO_ROOT, "dataset_builder"))

import json
import shutil
import tempfile

import numpy as np
import tensorflow as tf
from PIL import Image

from data_management import DetectionDataset, CenterNetDetectionDataset
from detection_target_encoding import HEATMAP_KEY, SIZE_KEY
from jax_detector_dataset_tools import process_detector_dataset


IMAGE_SIZE = (224, 224)
BASELINE_DIR = os.path.join(_REPO_ROOT, "tests", "fixtures", "augmentation_dedup_baseline")

# Valeurs réelles dataset_configs.py::FIGHTERJET_DETECTION et ::JAX_DETECTOR (copiées
# telles quelles l'une de l'autre) pour la partie géométrique (flip/translation/zoom) -
# seuls domaines avec cette augmentation active sur ces deux classes.
#
# brightness_delta/contrast_factor forcés à 0.0 ici (au lieu de 0.15/0.30 en prod) :
# tf.image.random_brightness/random_contrast tirent leur aléa via le RNG interne de TF,
# PAS via tf.random.uniform (vérifié empiriquement - le mock ci-dessous ne les couvre
# pas), donc non-reproductibles à l'identique d'un run de process à l'autre. Ce n'est
# pas une perte de couverture : ces deux blocs de code sont strictement identiques
# (copiés-collés sans dérive) entre DetectionDataset et CenterNetDetectionDataset, avant
# comme après le refactor - le risque de régression pixel-level visé par ce test porte
# sur la géométrie (flip/translation/zoom), là où l'interpolation/le padding/le rescale
# de `size` diffèrent réellement entre les deux classes.
# rotation_factor délibérément absent : mort pour ces deux classes (aucune logique de
# rotation dans _apply_geometric_and_color_augmentation), l'inclure ici laisserait croire
# à une couverture qui n'existe pas.
AUGMENTATION_PARAMS = {
    "flip_h": True,
    "flip_v": True,
    "zoom_factor": 0.35,
    "translation_factor": 0.25,
    "brightness_delta": 0.0,
    "contrast_factor": 0.0,
}


def _fake_uniform(shape, minval=0, maxval=1, **kwargs):
    """Force tous les tirages booléens à True (0.99 > 0.5) et tous les tirages continus
    (shift_x, shift_y, scale) à 80% de l'intervalle - déterministe, couvre flip+
    translation+zoom en un seul passage.

    PAS la médiane de l'intervalle : pour un shift symétrique [-f, f], la médiane vaut
    exactement 0 (translation = no-op, ne distingue jamais REFLECT de CONSTANT) ; pour
    un scale [1-f, 1+f], la médiane vaut exactement 1.0 (zoom = no-op, crop_frac=1.0,
    ne distingue jamais nearest de bilinear, et _effective_zoom_scale(1.0)==1.0 rend le
    rescale de `size` indétectable - trouvé en revue adversariale, corrigé ici)."""
    if minval == 0 and maxval == 1:
        return tf.constant(0.99, dtype=tf.float32)
    return tf.constant(minval + 0.8 * (maxval - minval), dtype=tf.float32)


def _capture_or_compare(name, array):
    os.makedirs(BASELINE_DIR, exist_ok=True)
    path = os.path.join(BASELINE_DIR, f"{name}.npy")
    if not os.path.exists(path):
        np.save(path, array)
        print(f"  [baseline capturée] {name} -> {path}")
        return
    baseline = np.load(path)
    assert baseline.shape == array.shape, (
        f"{name}: forme différente baseline={baseline.shape} actuel={array.shape}"
    )
    max_diff = float(np.max(np.abs(baseline.astype(np.float64) - array.astype(np.float64))))
    assert np.array_equal(baseline, array), (
        f"{name}: sortie post-refactor différente de la baseline pré-refactor "
        f"(écart absolu max = {max_diff})"
    )
    print(f"  [OK] {name} identique à la baseline (écart max = {max_diff})")


# --- DetectionDataset (masque binaire) ---

def _build_detection_dataset_chunk(out_dir, n_samples=1):
    """Construit directement un chunk .npz {output_prefix}_train_chunk*.npz avec les
    clés attendues par DetectionDataset (images, masks) - pas besoin de passer par
    l'outil de génération, la classe ne lit que ces deux tableaux.

    n_samples=1 (comme le chunk CenterNet ci-dessous) : ds.shuffle(1000) utilise son
    propre RNG (pas tf.random.uniform, donc non mocké par _fake_uniform) - avec 2+
    échantillons, l'ORDRE du batch peut varier d'un run de process à l'autre et fausser
    la comparaison pixel-level (faux positif observé en pratique, rien à voir avec le
    refactor : diagnostiqué par un repro isolé montrant 0 écart entre ancienne et
    nouvelle logique sur un même process). Avec 1 seul échantillon, rien à réordonner.
    """
    h, w = IMAGE_SIZE
    rng = np.random.RandomState(42)
    images = rng.rand(n_samples, h, w, 1).astype(np.float32)
    masks = np.zeros((n_samples, h, w, 1), dtype=np.float32)
    # Bloc de 1 excentré (pas centré) pour que flip/translation/zoom produisent des
    # sorties visiblement différentes d'un no-op.
    masks[:, 60:100, 40:90, 0] = 1.0

    output_prefix = os.path.join(out_dir, "dataset_detection")
    np.savez(f"{output_prefix}_train_chunk0.npz", images=images, masks=masks)
    return output_prefix


def test_detection_dataset_augmentation_regression():
    out_dir = tempfile.mkdtemp(prefix="test_dd_regression_")
    try:
        output_prefix = _build_detection_dataset_chunk(out_dir)
        ds_manager = DetectionDataset(
            output_prefix=output_prefix,
            image_size=IMAGE_SIZE,
            batch_size=1,
            grayscale=True,
            augmentation_params=AUGMENTATION_PARAMS,
        )

        orig_uniform = tf.random.uniform
        tf.random.uniform = _fake_uniform
        try:
            ds = ds_manager.create_tf_dataset('train', augment=True)
            img_batch, mask_batch = next(iter(ds.take(1)))
        finally:
            tf.random.uniform = orig_uniform

        print("DetectionDataset - vérification pixel-level :")
        _capture_or_compare("detection_dataset_image", img_batch.numpy())
        _capture_or_compare("detection_dataset_mask", mask_batch.numpy())
    finally:
        shutil.rmtree(out_dir, ignore_errors=True)


# --- CenterNetDetectionDataset (heatmap+size) ---

RAW_BBOX = [280, 190, 80, 60]  # x, y, w, h en pixels source (image source 640x480)
ORIG_SIZE = (640, 480)


def _build_centernet_chunk(out_dir):
    src_dir = tempfile.mkdtemp(prefix="test_cnd_regression_src_")
    try:
        os.makedirs(src_dir, exist_ok=True)
        img = Image.new("RGB", ORIG_SIZE, color=(60, 90, 130))
        img_filename = "fake_0.jpg"
        img_path = os.path.join(src_dir, img_filename)
        img.save(img_path)
        annotation = {
            "image": {"file_name": img_filename},
            "annotation": {"bbox": RAW_BBOX},
        }
        with open(os.path.join(src_dir, "fake_0.json"), "w") as f:
            json.dump(annotation, f)

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
    return os.path.join(out_dir, "jax_detector_targets")


def test_centernet_detection_dataset_augmentation_regression():
    out_dir = tempfile.mkdtemp(prefix="test_cnd_regression_out_")
    try:
        output_prefix = _build_centernet_chunk(out_dir)
        ds_manager = CenterNetDetectionDataset(
            output_prefix=output_prefix,
            image_size=IMAGE_SIZE,
            batch_size=1,
            grayscale=True,
            augmentation_params=AUGMENTATION_PARAMS,
        )

        orig_uniform = tf.random.uniform
        tf.random.uniform = _fake_uniform
        try:
            ds = ds_manager.create_tf_dataset('train', augment=True)
            img_batch, targets = next(iter(ds.take(1)))
        finally:
            tf.random.uniform = orig_uniform

        print("CenterNetDetectionDataset - vérification pixel-level :")
        _capture_or_compare("centernet_dataset_image", img_batch.numpy())
        _capture_or_compare("centernet_dataset_heatmap", targets[HEATMAP_KEY].numpy())
        _capture_or_compare("centernet_dataset_size", targets[SIZE_KEY].numpy())
    finally:
        shutil.rmtree(out_dir, ignore_errors=True)


if __name__ == "__main__":
    test_detection_dataset_augmentation_regression()
    test_centernet_detection_dataset_augmentation_regression()
    print("Tous les tests sont passés.")
