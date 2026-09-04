"""
Assistant d'inférence JAX pour tools/boxes_process_manual_tkinter.py (extrait de
PhotoViewer, Étape 1 du refactor du 2026-07-24 - voir
refactor-boxes-process-manual-tkinter.md). Contrainte AD-20 : consomme
exactement le même pipeline FIGHTERJET_DETECTION/UNet legacy que l'ancienne
implémentation inline (load_detection_model/build_predict_fn/
decode_segmentation_and_detect_batch, best_model_fighterjet_detection.pkl + best_model_fighterjet_classification.pkl)
- aucun changement de comportement, seulement de l'organisation du code.
"""
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
from PIL import Image

from dataset_configs import get_dataset_config
from inference_utils import (
    load_detection_model, load_jax_model, build_predict_fn, build_clf_predict_fn,
    decode_segmentation_and_detect_batch, predict_crops_batch, get_iou,
)


class PredictionAssistant:
    """Charge les modèles JAX (détection UNet + classification, pipeline
    FIGHTERJET_DETECTION legacy, AD-20) et produit une prédiction visuelle sur
    une image PIL - séparé de l'état tkinter de PhotoViewer (canvas/titre gérés
    par l'appelant, jamais ici)."""

    def __init__(self):
        self.det_predict_fn = None
        self.clf_predict_fn = None
        self.det_config = None
        self.clf_config = None
        self.dataset_mean = None
        self.dataset_std = None
        self.loaded = False

    def load_models(self):
        print("🏗️ Chargement des modèles JAX en arrière-plan (Lazy Loading)...")

        self.clf_config = get_dataset_config("FIGHTERJET_CLASSIFICATION")
        script_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(script_dir)
        det_path = os.path.join(parent_dir, "best_model_fighterjet_detection.pkl")
        clf_path = os.path.join(parent_dir, "best_model_fighterjet_classification.pkl")

        det_model, det_vars, self.det_config = load_detection_model(det_path)
        clf_model, clf_vars, self.dataset_mean, self.dataset_std = load_jax_model(clf_path, self.clf_config)

        print("⚡ Compilation JIT...")
        self.det_predict_fn = build_predict_fn(det_model, det_vars)
        self.clf_predict_fn = build_clf_predict_fn(clf_model, clf_vars)

        self.loaded = True
        print("✅ Modèles JAX prêts !")

    def predict(self, original_image, true_boxes_x1y1x2y2):
        """
        original_image : PIL.Image (repère original, pas zoomé - même convention
        que PhotoViewer.original_image).
        true_boxes_x1y1x2y2 : liste de [x1, y1, x2, y2] déjà dézoomées (repère
        original), pour le calcul d'IoU affiché.

        Retourne (result_image, avg_iou, predictions) :
          - result_image : PIL.Image RGB, overlay heatmap+boîtes+labels, même
            taille que original_image (l'appelant gère le resize zoom).
          - avg_iou : float, IoU moyen (0-100) vs true_boxes_x1y1x2y2.
          - predictions : liste de {"bbox": [x, y, w, h], "class_name": str},
            repère original (même format que self.last_predictions avant
            extraction).
        """
        print("🔍 Analyse de l'image en cours...")
        cv_img = cv2.cvtColor(np.array(original_image), cv2.COLOR_RGB2BGR)

        batch_results = decode_segmentation_and_detect_batch(
            [cv_img],
            self.det_predict_fn, self.det_config,
            conf_threshold=0.3,
            box_aera_min=25
        )

        pred_boxes, target_heatmap_lr, _target_binary_mask = batch_results[0]

        crop_imgs = []
        valid_boxes = []
        img_h, img_w = cv_img.shape[:2]

        for box in pred_boxes:
            x1, y1, x2, y2, conf = box

            x_start = max(0, int(x1))
            y_start = max(0, int(y1))
            x_end = min(img_w, int(x2))
            y_end = min(img_h, int(y2))

            if x_end > x_start and y_end > y_start:
                crop = cv_img[y_start:y_end, x_start:x_end]
                if crop.size > 0:
                    crop_imgs.append(crop)
                    valid_boxes.append([conf, x_start, y_start, x_end - x_start, y_end - y_start])

        pred_boxes = valid_boxes

        class_predictions = predict_crops_batch(
            crop_imgs, self.clf_predict_fn, self.dataset_mean, self.dataset_std, self.clf_config
        )

        # --- Construction de la heatmap couleur (upscale basse résolution -> HD) ---
        target_heatmap_hd = cv2.resize(
            target_heatmap_lr, (img_w, img_h), interpolation=cv2.INTER_LINEAR
        )
        heatmap_uint8 = np.clip(target_heatmap_hd * 255, 0, 255).astype(np.uint8)

        heatmap_color = cv2.applyColorMap(
            heatmap_uint8,
            cv2.COLORMAP_JET
        )

        # Supprime les faibles activations visuellement
        mask = heatmap_uint8 > 40
        heatmap_color[~mask] = 0

        # --- Overlay transparent ---
        alpha = 0.45  # transparence heatmap

        result_img = cv2.addWeighted(
            cv_img,
            1.0,
            heatmap_color,
            alpha,
            0
        )

        predicted_bboxes_x1y1x2y2 = []
        predictions = []

        for i, box in enumerate(pred_boxes):
            conf, x, y, w, h = box
            x, y, w, h = int(x), int(y), int(w), int(h)
            pred_class, pred_conf = class_predictions[i]

            color = (0, 255, 0)
            cv2.rectangle(result_img, (x, y), (x+w, y+h), color, 2)
            label = f"{pred_class} ({100*pred_conf:.0f}%)"
            cv2.putText(result_img, label, (x, max(0, y-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            predicted_bboxes_x1y1x2y2.append([x, y, x+w, y+h])
            predictions.append({"bbox": [x, y, w, h], "class_name": pred_class})

        total_iou = 0.0
        matches = 0

        for true_box in true_boxes_x1y1x2y2:
            best_iou = 0.0
            for pred_box in predicted_bboxes_x1y1x2y2:
                iou = get_iou(true_box, pred_box)
                if iou > best_iou:
                    best_iou = iou
            total_iou += best_iou
            matches += 1

        avg_iou = (total_iou / matches * 100) if matches > 0 else 0.0

        result_img_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
        result_image = Image.fromarray(result_img_rgb)

        print(f"✅ Prédiction terminée. IoU: {avg_iou:.2f}%")
        return result_image, avg_iou, predictions
