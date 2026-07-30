"""
Story 9.4 : capture boxes/classes/scores JAX_DETECTOR sur test_media/testvid01,02,03.png
via le vrai chemin de production (build_single_pass_predict_fn, inference_utils.py).

Concu pour etre invoque depuis la racine du checkout cible (avant OU apres l'Epic 9) -
insere le cwd courant dans sys.path pour importer les modules de CE checkout, jamais un
chemin fige (script lui-meme vit hors du repo, scratchpad).

Usage: cd <checkout> && python3 capture_jax_detector_baseline.py <output.json>
"""
import json
import os
import sys

sys.path.insert(0, os.getcwd())

import numpy as np
from PIL import Image

from inference_utils import build_single_pass_predict_fn

IMAGES = ["testvid01.png", "testvid02.png", "testvid03.png"]


def load_grayscale(path):
    img = Image.open(path).convert("L")
    arr = np.asarray(img, dtype=np.float32)[..., None]
    return arr


def main(output_path):
    predict_fn = build_single_pass_predict_fn()
    results = {}

    for name in IMAGES:
        path = os.path.join("test_media", name)
        image = load_grayscale(path)
        out = predict_fn(image)

        valid_mask = np.asarray(out["valid_mask"])
        boxes = np.asarray(out["boxes"])
        classes = np.asarray(out["classes"])
        class_scores = np.asarray(out["class_scores"])
        detection_scores = np.asarray(out["detection_scores"])

        detections = []
        for i in range(boxes.shape[0]):
            if not bool(valid_mask[i]):
                continue
            detections.append({
                "box": [round(float(v), 4) for v in boxes[i]],
                "class": int(classes[i]),
                "class_score": round(float(class_scores[i]), 4),
                "detection_score": round(float(detection_scores[i]), 4),
            })

        results[name] = {"num_valid": len(detections), "detections": detections}
        print(f"{name}: {len(detections)} detections valides")

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Sauvegarde: {output_path}")


if __name__ == "__main__":
    main(sys.argv[1])
