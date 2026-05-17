#!/usr/bin/env python3
"""
Script simplifié pour évaluer UNIQUEMENT la détection de boxes d'un modèle JAX.

Ce script est conçu pour être appelé par genetic_algorithm2.py.
Il prend UN SEUL argument : --checkpoint (chemin vers le modèle à évaluer).

Sortie :
- Un JSON avec les métriques essentielles pour l'algorithme génétique.

Usage:
    python loss_detection_only_directory.py --checkpoint /chemin/vers/model.pkl
"""

import sys
import os
import time
import argparse
import json
import cv2
import numpy as np
import jax
import jax.numpy as jnp
import pickle
from tqdm import tqdm
from typing import List, Dict, Tuple

# Ajouter les répertoires parents pour importer les modules locaux
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model_library import get_model


# =================================================================================================
# CONFIGURATION FIXE (pour éviter les arguments inutiles)
# =================================================================================================
# Dossier d'évaluation (fixe, contient les images + JSON ground truth)
EVALUATION_DIR = "./tmp_test"

# Paramètres du modèle de détection (fixes)
DEFAULT_IMAGE_SIZE = (224, 224)
DEFAULT_CONF_THRESHOLD = 0.35   # old 0.7
DEFAULT_BOX_AREA_MIN = 30       # old 60
DEFAULT_NMS_THRESHOLD = 0.4
DEFAULT_IOU_THRESHOLD = 0.3     # old 0.5


# =================================================================================================
# FONCTIONS UTILITAIRES POUR DÉTECTION
# =================================================================================================

def load_detection_model(checkpoint_path: str) -> Tuple:
    """Charge le modèle JAX de DÉTECTION."""
    # Vérifier les chemins possibles
    possible_paths = [
        checkpoint_path,
        os.path.join(os.path.dirname(os.getcwd()), checkpoint_path),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), checkpoint_path),
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), checkpoint_path),
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            checkpoint_path = path
            break
    else:
        raise FileNotFoundError(f"Checkpoint non trouvé: {checkpoint_path}")

    print(f"🔍 Chargement du modèle depuis {checkpoint_path}...")
    with open(checkpoint_path, 'rb') as f:
        data_model = pickle.load(f)

    params = data_model['params']
    config_model = data_model.get('config', {})
    model_name = config_model.get('model_name', 'aircraft_detector_v3')
    
    print(f"   Modèle détecté: {model_name}")
    
    # Création du modèle
    model = get_model(model_name, dropout_rate=0.0)
    
    # Récupération des batch_stats
    batch_stats = data_model.get('batch_stats', {})
    if not batch_stats:
        if 'model_state' in data_model:
            batch_stats = data_model['model_state'].get('batch_stats', {})
    
    if not batch_stats:
        print("⚠️  'batch_stats' non trouvés, ré-initialisation...")
        rng = jax.random.PRNGKey(0)
        target_size = config_model.get("image_size", DEFAULT_IMAGE_SIZE)
        grayscale = config_model.get("grayscale", True)
        channels = 1 if grayscale else 3
        dummy_input = jnp.ones((1, *target_size, channels), jnp.float32)
        init_variables = model.init(rng, dummy_input, training=True)
        batch_stats = init_variables.get('batch_stats', {})
        print("   ✅ batch_stats ré-initialisés.")

    variables = {'params': params, 'batch_stats': batch_stats}
    return model, variables, config_model


def get_iou(box1: List[float], box2: List[float], format1: str = "xyxy", format2: str = "xyxy") -> float:
    """Calcule l'Intersection over Union (IoU) de deux boxes."""
    if format1 == "xywh":
        x1_1, y1_1, w1, h1 = box1
        x1_2 = x1_1 + w1
        y1_2 = y1_1 + h1
    else:
        x1_1, y1_1, x1_2, y1_2 = box1
    
    if format2 == "xywh":
        x2_1, y2_1, w2, h2 = box2
        x2_2 = x2_1 + w2
        y2_2 = y2_1 + h2
    else:
        x2_1, y2_1, x2_2, y2_2 = box2
    
    x_left = max(x1_1, x2_1)
    y_top = max(y1_1, y2_1)
    x_right = min(x1_2, x2_2)
    y_bottom = min(y1_2, y2_2)
    
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    area1 = (x1_2 - x1_1) * (y1_2 - y1_1)
    area2 = (x2_2 - x2_1) * (y2_2 - y2_1)
    union_area = area1 + area2 - intersection_area
    
    return intersection_area / union_area if union_area > 0 else 0.0


def non_max_suppression(boxes: List[List[float]], iou_threshold: float) -> List[List[float]]:
    """Applique le Non-Maximum Suppression (NMS)."""
    if not boxes:
        return []
    boxes = sorted(boxes, key=lambda x: x[4], reverse=True)
    kept_boxes = []
    for current_box in boxes:
        overlap = False
        for kept_box in kept_boxes:
            iou = get_iou(current_box[:4], kept_box[:4], format1="xyxy", format2="xyxy")
            if iou > iou_threshold:
                overlap = True
                break
        if not overlap:
            kept_boxes.append(current_box)
    return kept_boxes


def decode_segmentation_and_detect(img_bgr, model, variables, config_model,
                                    conf_threshold: float = DEFAULT_CONF_THRESHOLD,
                                    box_area_min: int = DEFAULT_BOX_AREA_MIN) -> List[List[float]]:
    """Exécute la détection par Segmentation Sémantique (U-Net)."""
    h_orig, w_orig = img_bgr.shape[:2]
    target_size = config_model.get("image_size", DEFAULT_IMAGE_SIZE)
    grayscale = config_model.get("grayscale", True)
    
    img_resized = cv2.resize(img_bgr, target_size)
    
    if grayscale:
        img_input = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
        img_input = img_input.astype(np.float32) / 255.0
        img_jax = img_input[np.newaxis, :, :, np.newaxis]
    else:
        img_input = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        img_input = img_input.astype(np.float32) / 255.0
        img_jax = img_input[np.newaxis, :, :, :]
    
    preds = model.apply(variables, jnp.array(img_jax), training=False)
    pred_mask = np.array(preds[0, :, :, 0])
    mask_resized = cv2.resize(pred_mask, (w_orig, h_orig), interpolation=cv2.INTER_CUBIC)
    
    strong_mask = (mask_resized > conf_threshold).astype(np.uint8) * 255
    weak_mask = (mask_resized > (conf_threshold * 0.4)).astype(np.uint8) * 255
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    expanded_strong = cv2.dilate(strong_mask, kernel, iterations=1)
    binary_mask = cv2.bitwise_and(expanded_strong, weak_mask)
    
    closing_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, closing_kernel, iterations=1)
    
    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    binary_mask = cv2.dilate(binary_mask, dilate_kernel, iterations=1)
    
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    final_detections = []
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < box_area_min:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        sub_mask = mask_resized[y:y+h, x:x+w]
        score = float(np.max(sub_mask)) if sub_mask.size > 0 else 1.0
        final_detections.append([x, y, x+w, y+h, score])
    
    return final_detections


# =================================================================================================
# FONCTIONS DE COMPARAISON ET CALCUL DE LOSS
# =================================================================================================

def load_ground_truth_boxes(image_path: str) -> List[Dict]:
    """Charge les ground truth boxes pour une image donnée."""
    json_dir = os.path.dirname(image_path)
    file_name = os.path.basename(image_path)
    base_name = os.path.splitext(file_name)[0]
    ground_truths = []
    
    for json_file in os.listdir(json_dir):
        if json_file.startswith(base_name) and json_file.endswith('.json'):
            json_path = os.path.join(json_dir, json_file)
            try:
                with open(json_path, 'r') as f:
                    data = json.load(f)
                annotation = data.get('annotation', {})
                if 'detection_score' not in annotation:
                    ground_truths.append(data)
            except (json.JSONDecodeError, KeyError):
                continue
    return ground_truths


def convert_xywh_to_xyxy(bbox: List[float]) -> List[float]:
    """Convertit une bbox de [x, y, w, h] à [x1, y1, x2, y2]."""
    x, y, w, h = bbox
    return [x, y, x + w, y + h]


def calculate_detection_loss(pred_boxes: List[List[float]], gt_boxes: List[Dict],
                             iou_threshold: float = DEFAULT_IOU_THRESHOLD) -> Dict:
    """Calcule la loss pour la détection uniquement."""
    results = {
        'localization_loss': 0.0,
        'false_positive_loss': 0.0,
        'false_negative_loss': 0.0,
        'total_loss': 0.0,
        'num_true_positives': 0,
        'num_false_positives': 0,
        'num_false_negatives': 0,
        'avg_iou': 0.0
    }
    
    if not gt_boxes:
        results['false_positive_loss'] = len(pred_boxes)
        results['num_false_positives'] = len(pred_boxes)
        results['total_loss'] = results['false_positive_loss']
        return results
    
    if not pred_boxes:
        results['false_negative_loss'] = len(gt_boxes)
        results['num_false_negatives'] = len(gt_boxes)
        results['total_loss'] = results['false_negative_loss']
        return results
    
    gt_boxes_xyxy = []
    for gt in gt_boxes:
        bbox_xywh = gt['annotation']['bbox']
        bbox_xyxy = convert_xywh_to_xyxy(bbox_xywh)
        gt_boxes_xyxy.append({'bbox': bbox_xyxy})
    
    iou_matrix = np.zeros((len(pred_boxes), len(gt_boxes_xyxy)))
    for i, pred_box in enumerate(pred_boxes):
        for j, gt_box in enumerate(gt_boxes_xyxy):
            iou_matrix[i, j] = get_iou(pred_box[:4], gt_box['bbox'], format1="xyxy", format2="xyxy")
    
    matched_preds = set()
    matched_gts = set()
    total_iou = 0.0
    sorted_pred_indices = sorted(range(len(pred_boxes)), key=lambda i: pred_boxes[i][4], reverse=True)
    
    for pred_idx in sorted_pred_indices:
        best_gt_idx = -1
        best_iou = 0.0
        for gt_idx in range(len(gt_boxes_xyxy)):
            if gt_idx in matched_gts:
                continue
            if iou_matrix[pred_idx, gt_idx] > best_iou:
                best_iou = iou_matrix[pred_idx, gt_idx]
                best_gt_idx = gt_idx
        
        if best_iou >= iou_threshold:
            matched_preds.add(pred_idx)
            matched_gts.add(best_gt_idx)
            total_iou += best_iou
            loc_loss = 1.0 - best_iou
            results['localization_loss'] += loc_loss
            results['num_true_positives'] += 1
    
    if results['num_true_positives'] > 0:
        results['avg_iou'] = total_iou / results['num_true_positives']
    
    for pred_idx in range(len(pred_boxes)):
        if pred_idx not in matched_preds:
            results['num_false_positives'] += 1
            results['false_positive_loss'] += 1.0
    
    for gt_idx in range(len(gt_boxes_xyxy)):
        if gt_idx not in matched_gts:
            results['num_false_negatives'] += 1
            results['false_negative_loss'] += 1.0
    
    num_gts = len(gt_boxes)
    if num_gts > 0:
        results['localization_loss'] /= num_gts
        results['false_positive_loss'] /= num_gts
        results['false_negative_loss'] /= num_gts
    
    results['total_loss'] = (
        results['localization_loss'] +
        results['false_positive_loss'] +
        results['false_negative_loss']
    )
    
    return results


# =================================================================================================
# ÉVALUATION GLOBALE
# =================================================================================================

def evaluate_model(checkpoint_path: str) -> Dict:
    """
    Évalue le modèle de détection sur le dataset fixe (EVALUATION_DIR).
    Retourne UNIQUEMENT les métriques essentielles pour genetic_algorithm2.py.
    """
    # 1. Chargement du modèle
    try:
        model, variables, config_model = load_detection_model(checkpoint_path)
        print("✅ Modèle chargé.")
    except Exception as e:
        print(f"❌ Erreur chargement modèle: {e}")
        sys.exit(1)
    
    # 2. Préparation des fichiers
    if not os.path.exists(EVALUATION_DIR):
        print(f"❌ Le dossier {EVALUATION_DIR} n'existe pas.")
        sys.exit(1)
    
    image_files = []
    for root, _, files in os.walk(EVALUATION_DIR):
        for file in files:
            if file.lower().endswith((".jpg", ".png", ".jpeg", ".bmp")):
                image_files.append(os.path.join(root, file))
    
    print(f"📂 Évaluation sur {len(image_files)} images dans {EVALUATION_DIR}")
    
    # 3. Initialisation des métriques globales
    global_metrics = {
        'total_images': 0,
        'total_gt_boxes': 0,
        'total_pred_boxes': 0,
        'total_true_positives': 0,
        'total_false_positives': 0,
        'total_false_negatives': 0,
        'total_localization_loss': 0.0,
        'total_false_positive_loss': 0.0,
        'total_false_negative_loss': 0.0,
        'total_iou': 0.0,
        'total_loss': 0.0
    }
    
    # 4. Boucle d'évaluation
    for img_path in tqdm(image_files, desc="Évaluation"):
        img = cv2.imread(img_path)
        if img is None:
            continue
        
        gt_boxes = load_ground_truth_boxes(img_path)
        global_metrics['total_gt_boxes'] += len(gt_boxes)
        
        detections = decode_segmentation_and_detect(img, model, variables, config_model)
        detections = non_max_suppression(detections, iou_threshold=DEFAULT_NMS_THRESHOLD)
        global_metrics['total_pred_boxes'] += len(detections)
        
        image_metrics = calculate_detection_loss(detections, gt_boxes, iou_threshold=DEFAULT_IOU_THRESHOLD)
        
        global_metrics['total_true_positives'] += image_metrics['num_true_positives']
        global_metrics['total_false_positives'] += image_metrics['num_false_positives']
        global_metrics['total_false_negatives'] += image_metrics['num_false_negatives']
        global_metrics['total_localization_loss'] += image_metrics['localization_loss']
        global_metrics['total_false_positive_loss'] += image_metrics['false_positive_loss']
        global_metrics['total_false_negative_loss'] += image_metrics['false_negative_loss']
        global_metrics['total_loss'] += image_metrics['total_loss']
        global_metrics['total_iou'] += image_metrics['avg_iou'] * image_metrics['num_true_positives']
        global_metrics['total_images'] += 1
    
    # 5. Calcul des métriques finales
    metrics = {}
    if global_metrics['total_images'] > 0:
        metrics['avg_loss'] = global_metrics['total_loss'] / global_metrics['total_images']
        metrics['total_loss'] = global_metrics['total_loss']
        metrics['avg_localization_loss'] = global_metrics['total_localization_loss'] / global_metrics['total_images']
        metrics['avg_false_positive_loss'] = global_metrics['total_false_positive_loss'] / global_metrics['total_images']
        metrics['avg_false_negative_loss'] = global_metrics['total_false_negative_loss'] / global_metrics['total_images']
    else:
        metrics['avg_loss'] = 0.0
        metrics['total_loss'] = 0.0
        metrics['avg_localization_loss'] = 0.0
        metrics['avg_false_positive_loss'] = 0.0
        metrics['avg_false_negative_loss'] = 0.0
    
    if global_metrics['total_true_positives'] > 0:
        metrics['avg_iou'] = global_metrics['total_iou'] / global_metrics['total_true_positives']
    else:
        metrics['avg_iou'] = 0.0
    
    # Métriques de précision/rappel
    if global_metrics['total_gt_boxes'] > 0:
        metrics['precision'] = global_metrics['total_true_positives'] / (
            global_metrics['total_true_positives'] + global_metrics['total_false_positives'] + 1e-10
        )
        metrics['recall'] = global_metrics['total_true_positives'] / (
            global_metrics['total_true_positives'] + global_metrics['total_false_negatives'] + 1e-10
        )
        metrics['f1_score'] = 2 * (
            metrics['precision'] * metrics['recall'] / (metrics['precision'] + metrics['recall'] + 1e-10)
        )
    else:
        metrics['precision'] = 0.0
        metrics['recall'] = 0.0
        metrics['f1_score'] = 0.0
    
    # Ajouter les comptes bruts
    metrics['total_images'] = global_metrics['total_images']
    metrics['total_gt_boxes'] = global_metrics['total_gt_boxes']
    metrics['total_pred_boxes'] = global_metrics['total_pred_boxes']
    metrics['total_true_positives'] = global_metrics['total_true_positives']
    metrics['total_false_positives'] = global_metrics['total_false_positives']
    metrics['total_false_negatives'] = global_metrics['total_false_negatives']
    
    return metrics


# =================================================================================================
# MAIN
# =================================================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Évaluer un modèle de détection sur un dataset fixe.")
    parser.add_argument("--checkpoint", type=str, default="../best_model_detection.pkl",
                        help="Chemin vers le checkpoint du modèle à évaluer (obligatoire)")
    parser.add_argument("--output_json", action="store_true", default=True,
                        help="Sortir les métriques au format JSON (activé par défaut)")
    
    args = parser.parse_args()
    
    # Évaluation
    start_time = time.time()
    metrics = evaluate_model(args.checkpoint)
    end_time = time.time()
    
    # Sortie JSON (toujours)
    print(json.dumps(metrics))
    
    # Affichage optionnel (pour débogage)
    if not args.output_json:
        print(f"\n⏱️  Temps d'exécution: {end_time - start_time:.2f} secondes")
        print(f"✅ Évaluation terminée!")
