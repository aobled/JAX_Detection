"""
Test standalone pour dataset_builder/jax_detector_dataset_tools.py (Story 7.4).
Execution: python3 tests/test_jax_detector_dataset_tools.py
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
from PIL import Image

from jax_detector_dataset_tools import process_detector_dataset, _generate_fullframe_zoom_variant
from detection_target_encoding import HEATMAP_KEY, SIZE_KEY, decode_detection_targets


def _make_fake_dataset(root_dir, n_images=3):
    os.makedirs(root_dir, exist_ok=True)
    known_boxes = []
    for i in range(n_images):
        w, h = 640, 480
        img = Image.new("RGB", (w, h), color=(50 + i * 10, 80, 120))
        img_filename = f"fake_{i}.jpg"
        img_path = os.path.join(root_dir, img_filename)
        img.save(img_path)

        bbox = [100 + i * 20, 150 + i * 10, 80, 60]  # x, y, w, h en pixels source
        known_boxes.append(bbox)

        annotation = {
            "image": {"file_name": img_filename},
            "annotation": {"bbox": bbox},
        }
        json_path = os.path.join(root_dir, f"fake_{i}.json")
        with open(json_path, "w") as f:
            json.dump(annotation, f)

    return known_boxes


def test_chunk_shapes_and_keys():
    tmp_dir = tempfile.mkdtemp(prefix="test_v2_src_")
    out_dir = tempfile.mkdtemp(prefix="test_v2_out_")
    try:
        n_images = 3
        target_size = (224, 224)
        _make_fake_dataset(tmp_dir, n_images=n_images)

        process_detector_dataset(
            root_dirs=[tmp_dir],
            output_dir=out_dir,
            split_name="test",
            target_size=target_size,
            max_boxes=20,
            chunk_size=2000,
            grayscale=True,
        )

        chunk_files = [f for f in os.listdir(out_dir) if f.startswith("jax_detector_targets_")]
        assert len(chunk_files) == 1, f"un seul chunk attendu, trouvé {chunk_files}"
        assert not any(f.startswith("dataset_detection_") for f in os.listdir(out_dir)), \
            "le nom du chunk ne doit pas matcher le glob de nettoyage de l'outil actuel"

        data = np.load(os.path.join(out_dir, chunk_files[0]))
        assert set(data.files) >= {"images", HEATMAP_KEY, SIZE_KEY}
        assert data["images"].shape == (n_images, 224, 224, 1), data["images"].shape
        assert data[HEATMAP_KEY].shape == (n_images, 224, 224, 1), data[HEATMAP_KEY].shape
        assert data[SIZE_KEY].shape == (n_images, 224, 224, 2), data[SIZE_KEY].shape

        print("OK - test_chunk_shapes_and_keys")
        return out_dir, chunk_files[0]
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_decode_roundtrip_on_extracted_example():
    tmp_dir = tempfile.mkdtemp(prefix="test_v2_src2_")
    out_dir = tempfile.mkdtemp(prefix="test_v2_out2_")
    try:
        target_size = (224, 224)
        _make_fake_dataset(tmp_dir, n_images=2)

        process_detector_dataset(
            root_dirs=[tmp_dir],
            output_dir=out_dir,
            split_name="test",
            target_size=target_size,
            max_boxes=20,
            chunk_size=2000,
            grayscale=True,
        )

        chunk_files = [f for f in os.listdir(out_dir) if f.startswith("jax_detector_targets_")]
        data = np.load(os.path.join(out_dir, chunk_files[0]))

        heatmaps_np = data[HEATMAP_KEY]
        sizes_np = data[SIZE_KEY]

        for i in range(heatmaps_np.shape[0]):
            boxes = decode_detection_targets(heatmaps_np[i], sizes_np[i])
            assert len(boxes) >= 1, f"exemple {i} : aucune boîte décodée depuis un chunk réel"
            for (x1, y1, x2, y2, score) in boxes:
                assert 0.0 <= x1 < x2 <= target_size[0] + 1
                assert 0.0 <= y1 < y2 <= target_size[1] + 1
                assert score > 0.0

        print("OK - test_decode_roundtrip_on_extracted_example")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        shutil.rmtree(out_dir, ignore_errors=True)


# --- Tests unitaires de _generate_fullframe_zoom_variant (I/O Matrix, spec augmentation zoom) ---
# Boite cible commune a plusieurs scenarios : [100, 150, 80, 60] dans une image 640x480.
# Geometrie attendue (verifiee explicitement, valeurs figees ci-dessous) avec les defauts
# margin_ratio=0.15, min_visible_ratio=0.3 :
#   margin_x = 0.5*0.15*80 = 6.0 ; margin_y = 0.5*0.15*60 = 4.5
#   crop_x0 = floor(100-6.0) = 94 ; crop_y0 = floor(150-4.5) = 145
#   crop_x1 = ceil(100+80+6.0) = 186 ; crop_y1 = ceil(150+60+4.5) = 215
#   crop_w = 92, crop_h = 70 ; boite cible remappee = [100-94, 150-145, 80, 60] = [6, 5, 80, 60]

def test_generate_fullframe_zoom_variant_single_box():
    result = _generate_fullframe_zoom_variant([[100, 150, 80, 60]], 640, 480)
    assert result == (94, 145, 92, 70, [[6, 5, 80, 60]]), result
    print("OK - test_generate_fullframe_zoom_variant_single_box")


def test_generate_fullframe_zoom_variant_far_box_excluded():
    # 2e boite loin du crop [94,145,186,215] -> aucun recouvrement, absente de remapped_boxes
    result = _generate_fullframe_zoom_variant([[100, 150, 80, 60], [500, 400, 20, 20]], 640, 480)
    crop_x0, crop_y0, crop_w, crop_h, remapped_boxes = result
    assert (crop_x0, crop_y0, crop_w, crop_h) == (94, 145, 92, 70)
    assert remapped_boxes == [[6, 5, 80, 60]], remapped_boxes
    print("OK - test_generate_fullframe_zoom_variant_far_box_excluded")


def test_generate_fullframe_zoom_variant_partial_box_below_threshold_dropped():
    # box [180,150,40,40] : intersection avec crop x-range [94,186] -> largeur visible 6/40 = 0.15 < 0.3
    result = _generate_fullframe_zoom_variant([[100, 150, 80, 60], [180, 150, 40, 40]], 640, 480)
    _, _, _, _, remapped_boxes = result
    assert remapped_boxes == [[6, 5, 80, 60]], \
        f"boite partiellement coupee sous le seuil doit etre ecartee, trouve {remapped_boxes}"
    print("OK - test_generate_fullframe_zoom_variant_partial_box_below_threshold_dropped")


def test_generate_fullframe_zoom_variant_partial_box_above_threshold_kept_clipped():
    # box [170,150,40,40] : intersection avec crop x-range [94,186] -> largeur visible 16/40 = 0.4 >= 0.3
    # gardee, clippee au bord du crop : [170-94, 150-145, 16, 40] = [76, 5, 16, 40]
    result = _generate_fullframe_zoom_variant([[100, 150, 80, 60], [170, 150, 40, 40]], 640, 480)
    _, _, _, _, remapped_boxes = result
    assert remapped_boxes == [[6, 5, 80, 60], [76, 5, 16, 40]], remapped_boxes
    print("OK - test_generate_fullframe_zoom_variant_partial_box_above_threshold_kept_clipped")


def test_generate_fullframe_zoom_variant_target_partially_out_of_bounds_returns_none():
    # Revue adversariale + edge-case 2026-07-19 : boite cible elle-meme partiellement hors
    # du cadre source (deja observe sur ce dataset, cf. detection_target_encoding.py) -
    # image 15x100, cible [-25,10,30,20] (visible en x: [0,5] sur 30px de large = 16.7%
    # < min_visible_ratio=0.3) -> aucune boite (cible incluse) ne survit au clip -> None,
    # pas une variante "plein cadre" a zero boite.
    result = _generate_fullframe_zoom_variant([[-25, 10, 30, 20]], 15, 100)
    assert result is None, result
    print("OK - test_generate_fullframe_zoom_variant_target_partially_out_of_bounds_returns_none")


def test_generate_fullframe_zoom_variant_degenerate_target_returns_none():
    # boite cible unique avec largeur nulle -> aucune variante generable
    assert _generate_fullframe_zoom_variant([[10, 10, 0, 5]], 640, 480) is None
    # hauteur negative
    assert _generate_fullframe_zoom_variant([[10, 10, 5, -1]], 640, 480) is None
    # toutes les boites degenerees (la "plus grande aire" reste degeneree)
    assert _generate_fullframe_zoom_variant([[10, 10, -5, 5], [20, 20, 0, 0]], 640, 480) is None
    # aucune boite du tout
    assert _generate_fullframe_zoom_variant([], 640, 480) is None
    print("OK - test_generate_fullframe_zoom_variant_degenerate_target_returns_none")


# --- Tests d'integration bout-en-bout de zoom_augment_probability ---

def test_zoom_augment_probability_zero_matches_default_behavior():
    # AC1 : zoom_augment_probability=0.0 (explicite) doit produire une sortie strictement
    # identique (memes formes, memes valeurs) a l'appel sans ce parametre (comportement
    # d'avant ce changement). Seed remise a zero avant chaque appel pour que le
    # np.random.shuffle interne soit reproductible entre les 2 runs.
    tmp_dir = tempfile.mkdtemp(prefix="test_v2_zoom0_src_")
    out_default = tempfile.mkdtemp(prefix="test_v2_zoom0_default_")
    out_explicit = tempfile.mkdtemp(prefix="test_v2_zoom0_explicit_")
    try:
        _make_fake_dataset(tmp_dir, n_images=3)
        target_size = (224, 224)

        np.random.seed(1234)
        process_detector_dataset(
            root_dirs=[tmp_dir], output_dir=out_default, split_name="test",
            target_size=target_size, max_boxes=20, chunk_size=2000, grayscale=True,
        )

        np.random.seed(1234)
        process_detector_dataset(
            root_dirs=[tmp_dir], output_dir=out_explicit, split_name="test",
            target_size=target_size, max_boxes=20, chunk_size=2000, grayscale=True,
            zoom_augment_probability=0.0,
        )

        f_default = [f for f in os.listdir(out_default) if f.startswith("jax_detector_targets_")][0]
        f_explicit = [f for f in os.listdir(out_explicit) if f.startswith("jax_detector_targets_")][0]
        d_default = np.load(os.path.join(out_default, f_default))
        d_explicit = np.load(os.path.join(out_explicit, f_explicit))

        assert d_default["images"].shape == (3, 224, 224, 1) == d_explicit["images"].shape
        assert np.array_equal(d_default["images"], d_explicit["images"])
        assert np.array_equal(d_default[HEATMAP_KEY], d_explicit[HEATMAP_KEY])
        assert np.array_equal(d_default[SIZE_KEY], d_explicit[SIZE_KEY])

        print("OK - test_zoom_augment_probability_zero_matches_default_behavior")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        shutil.rmtree(out_default, ignore_errors=True)
        shutil.rmtree(out_explicit, ignore_errors=True)


def test_zoom_augment_probability_one_doubles_entries_and_variant_fills_frame():
    # AC2 : zoom_augment_probability=1.0 sur une image a 1 boite -> 2 entrees, et la boite
    # decodee de la 2e (variante) couvre >= 60% de la surface de target_size.
    tmp_dir = tempfile.mkdtemp(prefix="test_v2_zoom1_src_")
    out_dir = tempfile.mkdtemp(prefix="test_v2_zoom1_out_")
    try:
        target_size = (224, 224)
        _make_fake_dataset(tmp_dir, n_images=1)

        process_detector_dataset(
            root_dirs=[tmp_dir], output_dir=out_dir, split_name="test",
            target_size=target_size, max_boxes=20, chunk_size=2000, grayscale=True,
            zoom_augment_probability=1.0,
        )

        chunk_files = [f for f in os.listdir(out_dir) if f.startswith("jax_detector_targets_")]
        data = np.load(os.path.join(out_dir, chunk_files[0]))
        assert data["images"].shape == (2, 224, 224, 1), \
            f"1 image + zoom_augment_probability=1.0 doit produire 2 entrees, trouve {data['images'].shape}"

        heatmaps_np = data[HEATMAP_KEY]
        sizes_np = data[SIZE_KEY]

        variant_boxes = decode_detection_targets(heatmaps_np[1], sizes_np[1])
        assert len(variant_boxes) == 1, f"1 seule boite attendue dans la variante, trouve {variant_boxes}"
        x1, y1, x2, y2, score = variant_boxes[0]
        area_fraction = ((x2 - x1) * (y2 - y1)) / (target_size[0] * target_size[1])
        assert area_fraction >= 0.6, f"boite variante couvre {area_fraction:.3f} < 0.6 du cadre cible"

        print("OK - test_zoom_augment_probability_one_doubles_entries_and_variant_fills_frame")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        shutil.rmtree(out_dir, ignore_errors=True)


def test_zoom_augment_excludes_far_box_from_variant_target():
    # AC3 : image a 2 boites dont une hors du crop calcule -> seule la boite visible
    # apparait dans la cible encodee de la variante (l'originale garde ses 2 boites).
    tmp_dir = tempfile.mkdtemp(prefix="test_v2_zoomfar_src_")
    out_dir = tempfile.mkdtemp(prefix="test_v2_zoomfar_out_")
    try:
        target_size = (224, 224)
        os.makedirs(tmp_dir, exist_ok=True)
        img = Image.new("RGB", (640, 480), color=(50, 80, 120))
        img_filename = "fake_multi.jpg"
        img.save(os.path.join(tmp_dir, img_filename))

        # boite cible (grande) + boite lointaine (hors du crop calcule pour la cible)
        target_bbox = [100, 150, 80, 60]
        far_bbox = [500, 400, 20, 20]
        for idx, bbox in enumerate([target_bbox, far_bbox]):
            annotation = {"image": {"file_name": img_filename}, "annotation": {"bbox": bbox}}
            with open(os.path.join(tmp_dir, f"fake_multi_{idx}.json"), "w") as f:
                json.dump(annotation, f)

        process_detector_dataset(
            root_dirs=[tmp_dir], output_dir=out_dir, split_name="test",
            target_size=target_size, max_boxes=20, chunk_size=2000, grayscale=True,
            zoom_augment_probability=1.0,
        )

        chunk_files = [f for f in os.listdir(out_dir) if f.startswith("jax_detector_targets_")]
        data = np.load(os.path.join(out_dir, chunk_files[0]))
        assert data["images"].shape == (2, 224, 224, 1), data["images"].shape

        heatmaps_np = data[HEATMAP_KEY]
        sizes_np = data[SIZE_KEY]

        original_boxes = decode_detection_targets(heatmaps_np[0], sizes_np[0])
        variant_boxes = decode_detection_targets(heatmaps_np[1], sizes_np[1])
        assert len(original_boxes) == 2, f"l'originale doit garder ses 2 boites, trouve {original_boxes}"
        assert len(variant_boxes) == 1, \
            f"la variante ne doit garder que la boite visible (cible), trouve {variant_boxes}"

        print("OK - test_zoom_augment_excludes_far_box_from_variant_target")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        shutil.rmtree(out_dir, ignore_errors=True)


def test_zoom_augment_degenerate_target_box_no_variant_no_crash():
    # Boite cible degeneree (largeur nulle) + zoom_augment_probability=1.0 -> l'image est
    # traitee normalement (1 seule entree, l'originale), aucune exception, aucune variante.
    tmp_dir = tempfile.mkdtemp(prefix="test_v2_zoomdeg_src_")
    out_dir = tempfile.mkdtemp(prefix="test_v2_zoomdeg_out_")
    try:
        target_size = (224, 224)
        os.makedirs(tmp_dir, exist_ok=True)
        img = Image.new("RGB", (640, 480), color=(50, 80, 120))
        img_filename = "fake_degenerate.jpg"
        img.save(os.path.join(tmp_dir, img_filename))

        degenerate_bbox = [100, 150, 0, 60]  # largeur nulle
        annotation = {"image": {"file_name": img_filename}, "annotation": {"bbox": degenerate_bbox}}
        with open(os.path.join(tmp_dir, "fake_degenerate.json"), "w") as f:
            json.dump(annotation, f)

        process_detector_dataset(
            root_dirs=[tmp_dir], output_dir=out_dir, split_name="test",
            target_size=target_size, max_boxes=20, chunk_size=2000, grayscale=True,
            zoom_augment_probability=1.0,
        )

        chunk_files = [f for f in os.listdir(out_dir) if f.startswith("jax_detector_targets_")]
        assert len(chunk_files) == 1
        data = np.load(os.path.join(out_dir, chunk_files[0]))
        assert data["images"].shape == (1, 224, 224, 1), \
            f"boite cible degeneree -> aucune variante, 1 seule entree attendue, trouve {data['images'].shape}"

        print("OK - test_zoom_augment_degenerate_target_box_no_variant_no_crash")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        shutil.rmtree(out_dir, ignore_errors=True)


def test_zoom_augment_rgb_path_no_crash():
    # Revue adversariale 2026-07-19 : le chemin RGB (grayscale=False) combine a la
    # variante zoom n'etait couvert par aucun test - _encode_and_append est partagee
    # entre original et variante, mais on verifie explicitement ici que ca tient sur les
    # 2 chemins (ndim==3, pas d'expand_dims), pas seulement en grayscale.
    tmp_dir = tempfile.mkdtemp(prefix="test_v2_zoomrgb_src_")
    out_dir = tempfile.mkdtemp(prefix="test_v2_zoomrgb_out_")
    try:
        target_size = (224, 224)
        _make_fake_dataset(tmp_dir, n_images=1)

        process_detector_dataset(
            root_dirs=[tmp_dir], output_dir=out_dir, split_name="test",
            target_size=target_size, max_boxes=20, chunk_size=2000, grayscale=False,
            zoom_augment_probability=1.0,
        )

        chunk_files = [f for f in os.listdir(out_dir) if f.startswith("jax_detector_targets_")]
        data = np.load(os.path.join(out_dir, chunk_files[0]))
        assert data["images"].shape == (2, 224, 224, 3), \
            f"RGB + variante attendu (2, 224, 224, 3), trouve {data['images'].shape}"

        print("OK - test_zoom_augment_rgb_path_no_crash")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        shutil.rmtree(out_dir, ignore_errors=True)


def test_zoom_augment_variant_triggers_chunk_save():
    # Revue adversariale 2026-07-19 : le 2e garde "chunk plein" (juste apres l'append de
    # la variante, distinct du garde apres l'original) n'etait jamais reellement declenche
    # par aucun test. chunk_size=2 + 1 image + zoom_augment_probability=1.0 -> l'original
    # (1/2) ne declenche pas le 1er garde, la variante (2/2) declenche le 2e.
    tmp_dir = tempfile.mkdtemp(prefix="test_v2_zoomchunk_src_")
    out_dir = tempfile.mkdtemp(prefix="test_v2_zoomchunk_out_")
    try:
        target_size = (224, 224)
        _make_fake_dataset(tmp_dir, n_images=1)

        process_detector_dataset(
            root_dirs=[tmp_dir], output_dir=out_dir, split_name="test",
            target_size=target_size, max_boxes=20, chunk_size=2, grayscale=True,
            zoom_augment_probability=1.0,
        )

        chunk_files = [f for f in os.listdir(out_dir) if f.startswith("jax_detector_targets_")]
        assert len(chunk_files) == 1, f"un seul chunk (plein a 2/2) attendu, trouve {chunk_files}"
        data = np.load(os.path.join(out_dir, chunk_files[0]))
        assert data["images"].shape == (2, 224, 224, 1), \
            f"chunk_size=2 : le chunk sauvegarde par le garde 'variante' doit contenir 2 entrees, trouve {data['images'].shape}"

        print("OK - test_zoom_augment_variant_triggers_chunk_save")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        shutil.rmtree(out_dir, ignore_errors=True)


if __name__ == "__main__":
    out_dir, chunk_file = test_chunk_shapes_and_keys()
    shutil.rmtree(out_dir, ignore_errors=True)
    test_decode_roundtrip_on_extracted_example()

    test_generate_fullframe_zoom_variant_single_box()
    test_generate_fullframe_zoom_variant_far_box_excluded()
    test_generate_fullframe_zoom_variant_partial_box_below_threshold_dropped()
    test_generate_fullframe_zoom_variant_partial_box_above_threshold_kept_clipped()
    test_generate_fullframe_zoom_variant_degenerate_target_returns_none()
    test_generate_fullframe_zoom_variant_target_partially_out_of_bounds_returns_none()

    test_zoom_augment_probability_zero_matches_default_behavior()
    test_zoom_augment_probability_one_doubles_entries_and_variant_fills_frame()
    test_zoom_augment_excludes_far_box_from_variant_target()
    test_zoom_augment_degenerate_target_box_no_variant_no_crash()
    test_zoom_augment_rgb_path_no_crash()
    test_zoom_augment_variant_triggers_chunk_save()

    print("Tous les tests sont passés.")
