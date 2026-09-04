"""
Diagnostic precis (discussion 2026-07-19, retour utilisateur) : CenterNet semble
excellent sur les scenes multi-avions (formation serree) mais rate un avion UNIQUE
occupant la quasi-totalite du cadre, surtout sur fond non-bleu (desert/terre) - alors que
l'ancien pipeline UNet s'en sortait bien sur ces memes cas. Compare les deux pipelines
contre la VERITE TERRAIN sur test_media/single_full_size_aircraft/ (14 images, 1 boite
chacune, deja annotees par l'utilisateur), et rapporte le TOP-3 des scores bruts (avant
filtrage valid_mask) du nouveau detecteur pour distinguer "il y a un pic faible pas assez
confiant" de "aucun pic pres de la vraie position, le modele ne voit vraiment rien".

Usage (depuis la racine du repo): python3 test/diagnose_single_full_size_aircraft.py
"""
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
import jax.numpy as jnp

from dataset_configs import get_dataset_config
from inference_utils import (
    load_detection_model, load_jax_model,
    decode_segmentation_and_detect, predict_crop,
    build_single_pass_predict_fn,
)

TEST_MEDIA_DIR = "test_media/single_full_size_aircraft"
# Lot ajoute le 2026-07-19 (apres le 1er passage de ce diagnostic) : avions uniques sur
# fond bleu/ciel, pour isoler si l'echec de recall vient du fond non-bleu (desert) ou de
# l'avion unique plein cadre en general.
BLUE_BACKGROUND_IMAGES = {
    "8abad40539bd5ad2.jpg", "a18bde660a9935ea.jpg", "a9d916a47d19cf60.jpg",
    "c69339e4c69b3866.jpg", "c787589827e758d8.jpg", "c99216e5699a1ee5.jpg",
}
DETECTION_CHECKPOINT_PATH = "best_model_fighterjet_detection.pkl"
CLASSIFIER_CHECKPOINT_PATH = "best_model_fighterjet_classification.pkl"
DETECTOR_CHECKPOINT_PATH = "best_model_jax_detector.pkl"
DETECTION_CONF_THRESHOLD = 0.8
BOX_AERA_MIN = 60
CANONICAL_WIDTH, CANONICAL_HEIGHT = 1920, 1080
IOU_MATCH_THRESHOLD = 0.3


def _xywh_to_xyxy(bbox):
    x, y, w, h = bbox
    return (x, y, x + w, y + h)


def _iou(box_a, box_b):
    xa1, ya1, xa2, ya2 = box_a
    xb1, yb1, xb2, yb2 = box_b
    x1, y1 = max(xa1, xb1), max(ya1, yb1)
    x2, y2 = min(xa2, xb2), min(ya2, yb2)
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, xa2 - xa1) * max(0.0, ya2 - ya1)
    area_b = max(0.0, xb2 - xb1) * max(0.0, yb2 - yb1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _discover_images():
    images = []
    for jpg_path in sorted(glob.glob(os.path.join(TEST_MEDIA_DIR, "*.jpg"))):
        base = os.path.splitext(os.path.basename(jpg_path))[0]
        json_matches = sorted(glob.glob(os.path.join(TEST_MEDIA_DIR, f"{base}_*.json")))
        if json_matches:
            images.append((os.path.basename(jpg_path), json_matches))
    return images


def _load_ground_truth(json_paths):
    gt_boxes = []
    for path in json_paths:
        with open(path) as f:
            data = json.load(f)
        ann = data["annotation"]
        gt_boxes.append({"box": _xywh_to_xyxy(ann["bbox"]), "class": ann["category_name"]})
    return gt_boxes


def _best_match(gt_box, detections):
    best_iou, best_det = 0.0, None
    for det in detections:
        iou = _iou(gt_box, det["box"])
        if iou > best_iou:
            best_iou, best_det = iou, det
    return (best_iou, best_det) if best_iou >= IOU_MATCH_THRESHOLD else (0.0, None)


def main():
    config_clf = get_dataset_config("FIGHTERJET_CLASSIFICATION")
    class_names = config_clf["class_names"]

    det_model, det_vars, det_config = load_detection_model(DETECTION_CHECKPOINT_PATH)
    clf_model, clf_vars, dataset_mean, dataset_std = load_jax_model(CLASSIFIER_CHECKPOINT_PATH, config_clf)
    predict_fn = build_single_pass_predict_fn(
        detector_checkpoint_path=DETECTOR_CHECKPOINT_PATH,
        classifier_checkpoint_path=CLASSIFIER_CHECKPOINT_PATH,
    )

    images = _discover_images()
    print(f"{len(images)} images trouvees dans {TEST_MEDIA_DIR}/")

    rows = []
    raw_score_rows = []

    for image_name, json_paths in images:
        image_path = os.path.join(TEST_MEDIA_DIR, image_name)
        img = cv2.imread(image_path)
        h, w = img.shape[:2]
        gt_boxes = _load_ground_truth(json_paths)

        # --- Ancien pipeline (UNet) ---
        old_detections = decode_segmentation_and_detect(
            img, det_model, det_vars, det_config,
            conf_threshold=DETECTION_CONF_THRESHOLD, box_aera_min=BOX_AERA_MIN,
        )
        old_dets = []
        for (x1, y1, x2, y2, det_score) in old_detections:
            crop = img[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            predicted_class, confidence = predict_crop(crop, clf_model, clf_vars, dataset_mean, dataset_std, config_clf)
            old_dets.append({"box": (float(x1), float(y1), float(x2), float(y2)),
                              "class": predicted_class, "confidence": confidence})

        # --- Nouveau pipeline (CenterNet Single-Pass) ---
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        scale_x = CANONICAL_WIDTH / w
        scale_y = CANONICAL_HEIGHT / h
        if (h, w) != (CANONICAL_HEIGHT, CANONICAL_WIDTH):
            img_gray = cv2.resize(img_gray, (CANONICAL_WIDTH, CANONICAL_HEIGHT))
        canonical_image = jnp.asarray(img_gray.astype(np.float32)[..., None])
        result = predict_fn(canonical_image)
        valid_mask = np.asarray(result["valid_mask"])
        boxes = np.asarray(result["boxes"])
        classes = np.asarray(result["classes"])
        class_scores = np.asarray(result["class_scores"])
        detection_scores = np.asarray(result["detection_scores"])
        new_dets = []
        for i in np.where(valid_mask)[0]:
            x1, y1, x2, y2 = boxes[i]
            new_dets.append({"box": (float(x1), float(y1), float(x2), float(y2)),
                              "class": class_names[int(classes[i])], "confidence": float(class_scores[i])})

        # --- Top-3 scores BRUTS (avant valid_mask) les plus proches de la verite terrain,
        # pour distinguer "pic faible mais present" de "rien pres de la vraie position" ---
        gt_box_canonical = None
        if gt_boxes:
            gx1, gy1, gx2, gy2 = gt_boxes[0]["box"]
            gt_box_canonical = (gx1 * scale_x, gy1 * scale_y, gx2 * scale_x, gy2 * scale_y)
        ious_all = np.array([
            _iou(gt_box_canonical, tuple(boxes[i])) if gt_box_canonical else 0.0
            for i in range(20)
        ])
        best_iou_idx = np.argsort(-ious_all)[:3]

        for gt in gt_boxes:
            gx1, gy1, gx2, gy2 = gt["box"]
            gt_box_scaled = (gx1 * scale_x, gy1 * scale_y, gx2 * scale_x, gy2 * scale_y)
            iou_old, det_old = _best_match(gt["box"], old_dets)
            # new_dets est dans le repere canonique 1920x1080 (contrat fixe de
            # build_single_pass_predict_fn), PAS dans le repere de l'image source (ex.
            # 640x427 ici) - comparer gt["box"] brut aurait ete un bug (2 reperes
            # differents, IoU quasi toujours nul par construction). Corrige 2026-07-19.
            iou_new, det_new = _best_match(gt_box_scaled, new_dets)
            rows.append({
                "image": image_name, "gt_class": gt["class"],
                "group": "bleu" if image_name in BLUE_BACKGROUND_IMAGES else "non-bleu",
                "iou_old": iou_old, "class_old": det_old["class"] if det_old else None,
                "conf_old": det_old["confidence"] if det_old else None,
                "iou_new": iou_new, "class_new": det_new["class"] if det_new else None,
                "conf_new": det_new["confidence"] if det_new else None,
            })

        raw_score_rows.append({
            "image": image_name,
            "best_candidates": [
                (float(ious_all[i]), float(detection_scores[i]), bool(valid_mask[i]))
                for i in best_iou_idx
            ],
        })

    # === Rapport detaille ===
    print(f"\n\n{'='*100}")
    print("DETAIL PAR BOITE DE VERITE TERRAIN")
    print(f"{'='*100}")
    header = f"{'image':28} {'fond':9} {'gt_class':10} {'IoU_old':8} {'class_old':11} {'conf_old':9} {'IoU_new':8} {'class_new':11} {'conf_new':9}"
    print(header)
    print("-" * len(header))
    for r in rows:
        c_old = f"{r['conf_old']*100:.1f}%" if r["conf_old"] is not None else "MANQUE"
        c_new = f"{r['conf_new']*100:.1f}%" if r["conf_new"] is not None else "MANQUE"
        cl_old = r["class_old"] or "-"
        cl_new = r["class_new"] or "-"
        print(f"{r['image']:28} {r['group']:9} {r['gt_class']:10} {r['iou_old']:.3f}    {cl_old:11} {c_old:9} "
              f"{r['iou_new']:.3f}    {cl_new:11} {c_new:9}")

    # === Statistiques agregees, globales puis par groupe fond bleu/non-bleu ===
    n_total = len(rows)
    n_found_old = sum(1 for r in rows if r["conf_old"] is not None)
    n_found_new = sum(1 for r in rows if r["conf_new"] is not None)
    print(f"\n\n{'='*70}")
    print("RECALL (IoU >= 0.3 vs verite terrain)")
    print(f"{'='*70}")
    print(f"Ancien pipeline (UNet)          : {n_found_old}/{n_total}")
    print(f"Nouveau pipeline (CenterNet)     : {n_found_new}/{n_total}")

    for group in ("non-bleu", "bleu"):
        group_rows = [r for r in rows if r["group"] == group]
        if not group_rows:
            continue
        g_old = sum(1 for r in group_rows if r["conf_old"] is not None)
        g_new = sum(1 for r in group_rows if r["conf_new"] is not None)
        print(f"\n  -- Fond {group} ({len(group_rows)} images) --")
        print(f"     Ancien pipeline (UNet)      : {g_old}/{len(group_rows)}")
        print(f"     Nouveau pipeline (CenterNet) : {g_new}/{len(group_rows)}")

    print(f"\n\n{'='*100}")
    print("TOP-3 CANDIDATS BRUTS (avant valid_mask) LES PLUS PROCHES DE LA VERITE TERRAIN")
    print("(IoU=0 partout => aucun pic pres de la vraie position, pas juste sous le seuil)")
    print(f"{'='*100}")
    for r in raw_score_rows:
        cands = ", ".join(f"IoU={iou:.2f}/score={score:.3f}/{'VALID' if valid else 'sous-seuil'}"
                           for iou, score, valid in r["best_candidates"])
        print(f"{r['image']:28} {cands}")


if __name__ == "__main__":
    main()
