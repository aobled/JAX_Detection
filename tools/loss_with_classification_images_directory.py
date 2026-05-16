#!/usr/bin/env python3
"""
Script pour évaluer un modèle de détection + classification JAX sur un dataset fixe.

Ce script:
1. Charge les modèles de détection et classification JAX
2. Parcourt les images dans INPUT_DIR
3. Pour chaque image, charge les ground truth boxes depuis les JSON correspondants
4. Effectue la prédiction de détection + classification
5. Compare les boxes prédites avec les ground truths
6. Calcule une loss combinée (détection + classification + IoU)

Format attendu des JSON ground truth:
{
    "image": {"file_name": "...", "width": ..., "height": ...},
    "annotation": {
        "bbox": [x, y, width, height],
        "category_name": "classe",
        "bbox_id": 0
    }
}

Usage:
    python loss_with_classification_images_directory.py --input_dir /chemin/vers/dataset
"""

import sys
import os

# Ajouter le répertoire parent en PRIORITÉ absolue (index 0) pour forcer Python
# à utiliser le model_library.py de JAX_Detection (et non celui de JAX_Classification si exécuté depuis l'autre dossier)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import argparse
import json
import cv2
import numpy as np
import jax
import jax.numpy as jnp
import pickle
from tqdm import tqdm
from typing import List, Dict, Tuple, Optional

# Ajouter le répertoire parent pour importer les modules locaux
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataset_configs import get_dataset_config
from model_library import get_model


# =================================================================================================
# CONFIGURATION PAR DÉFAUT
# =================================================================================================
DEFAULT_DATASET_NAME = "FIGHTERJET_CLASSIFICATION"
DEFAULT_CHECKPOINT_PATH = "best_model.pkl"
DEFAULT_DETECTION_CHECKPOINT_PATH = "best_model_detection.pkl"
DEFAULT_DETECTION_IMAGE_SIZE = (224, 224)
DEFAULT_CONFIDENCE_THRESHOLD = 0.7
DEFAULT_DETECTION_CONF_THRESHOLD = 0.7
DEFAULT_BOX_AREA_MIN = 60
DEFAULT_NMS_THRESHOLD = 0.4
DEFAULT_DEFAULT_CLASSE = "unknown"


# =================================================================================================
# FONCTIONS UTILITAIRES POUR JAX (CLASSIFICATION)
# =================================================================================================
def load_jax_model(checkpoint_path: str, config: Dict) -> Tuple:
    """Charge le modèle JAX de CLASSIFICATION."""
    if not os.path.exists(checkpoint_path):
        # Essayer de trouver le checkpoint dans le dossier parent
        parent_checkpoint = os.path.join(os.path.dirname(os.getcwd()), checkpoint_path)
        if os.path.exists(parent_checkpoint):
            checkpoint_path = parent_checkpoint
        else:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            parent_dir = os.path.dirname(script_dir)
            parent_checkpoint = os.path.join(parent_dir, checkpoint_path)
            if os.path.exists(parent_checkpoint):
                checkpoint_path = parent_checkpoint
            else:
                raise FileNotFoundError(f"Checkpoint non trouvé: {checkpoint_path}")

    print(f"🔍 Chargement du modèle CLASSIFICATION depuis {checkpoint_path}...")
    with open(checkpoint_path, 'rb') as f:
        model_data = pickle.load(f)

    # Extraction des paramètres
    if 'model_state' in model_data:
        params = model_data['model_state']['params']
        batch_stats = model_data['model_state'].get('batch_stats', {})
        model_info = model_data.get('model_info', {})
        model_name = model_info.get('model_name', config["model_name"])
    else:
        params = model_data['params']
        batch_stats = model_data.get('batch_stats', {})
        model_name = model_data.get('model_name', config["model_name"])
    
    num_classes = config["num_classes"]
    
    # Création du modèle
    model = get_model(model_name, num_classes=num_classes, dropout_rate=0.0)
    variables = {'params': params, 'batch_stats': batch_stats}
    
    # Chargement des stats de normalisation
    mean_std_path = config.get("mean_std_path", "./data/chunks/dataset_chunked_meanstd.npz")
    if not os.path.exists(mean_std_path):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(script_dir)
        mean_std_path_abs = os.path.join(parent_dir, mean_std_path)
        if os.path.exists(mean_std_path_abs):
            mean_std_path = mean_std_path_abs
    
    if os.path.exists(mean_std_path):
        with np.load(mean_std_path) as data:
            mean = data['mean']
            std = data['std']
            print("✅ Stats de normalisation chargées.")
            
            # Correction automatique : Si Grayscale mais stats RGB
            if config.get("grayscale", False):
                if isinstance(mean, np.ndarray) and mean.size == 3:
                     print("⚠️  Conversion des stats RGB -> Grayscale (mean/std)")
                     mean = np.mean(mean)
                     std = np.mean(std)
    else:
        print("⚠️  ATTENTION: Stats de normalisation non trouvées, utilisation de valeurs par défaut (0.5, 0.5)")
        mean = 0.5
        std = 0.5

    return model, variables, mean, std


def predict_crop(crop_img, model, variables, mean, std, config: Dict) -> Tuple[str, float]:
    """
    Prédit la classe d'un crop (image OpenCV BGR).
    Retourne: (nom_classe, confiance)
    """
    # 1. Prétraitement
    target_size = config["image_size"]
    grayscale = config.get("grayscale", False)
    
    # Resize
    crop_resized = cv2.resize(crop_img, target_size)
    
    # Conversion couleur et normalisation
    if grayscale:
        img_input = cv2.cvtColor(crop_resized, cv2.COLOR_BGR2GRAY)
        img_input = img_input.astype(np.float32) / 255.0
        img_normalized = (img_input - mean) / std
        img_jax = img_normalized[np.newaxis, :, :, np.newaxis]
    else:
        img_input = cv2.cvtColor(crop_resized, cv2.COLOR_BGR2RGB)
        img_input = img_input.astype(np.float32) / 255.0
        img_normalized = (img_input - mean) / std
        img_jax = img_normalized[np.newaxis, :, :, :]

    # 2. Inférence JAX
    logits = model.apply(variables, jnp.array(img_jax), training=False)
    probs = jax.nn.softmax(logits, axis=-1)
    
    # 3. Résultat
    pred_idx = int(jnp.argmax(probs))
    confidence = float(probs[0, pred_idx])
    
    return config["class_names"][pred_idx], confidence


# =================================================================================================
# FONCTIONS UTILITAIRES POUR DÉTECTION
# =================================================================================================
def load_detection_model(checkpoint_path: str) -> Tuple:
    """Charge le modèle JAX de DÉTECTION."""
    if not os.path.exists(checkpoint_path):
        parent_checkpoint = os.path.join(os.path.dirname(os.getcwd()), checkpoint_path)
        if os.path.exists(parent_checkpoint):
            checkpoint_path = parent_checkpoint
        else:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            parent_dir = os.path.dirname(script_dir)
            parent_checkpoint = os.path.join(parent_dir, checkpoint_path)
            if os.path.exists(parent_checkpoint):
                checkpoint_path = parent_checkpoint
            else:
                raise FileNotFoundError(f"Checkpoint détection non trouvé: {checkpoint_path}")

    print(f"🔍 Chargement du modèle DÉTECTION depuis {checkpoint_path}...")
    with open(checkpoint_path, 'rb') as f:
        data_model = pickle.load(f)

    params = data_model['params']
    config_model = data_model.get('config', {})
    model_name = config_model.get('model_name', 'aircraft_detector_v3')
    
    print(f"   Modèle détecté: {model_name}")
    
    # Création du modèle
    model = get_model(model_name, dropout_rate=0.0)
    
    # Récupération ou Initialisation des batch_stats
    batch_stats = data_model.get('batch_stats', {})
    
    if not batch_stats:
        if 'model_state' in data_model:
             batch_stats = data_model['model_state'].get('batch_stats', {})
    
    if not batch_stats:
        print("⚠️  ATTENTION: 'batch_stats' non trouvés dans le checkpoint !")
        print("   Tentative de ré-initialisation...")
        
        rng = jax.random.PRNGKey(0)
        target_size = config_model.get("image_size", DEFAULT_DETECTION_IMAGE_SIZE)
        grayscale = config_model.get("grayscale", True)
        channels = 1 if grayscale else 3
        dummy_input = jnp.ones((1, *target_size, channels), jnp.float32)
        
        init_variables = model.init(rng, dummy_input, training=True)
        batch_stats = init_variables.get('batch_stats', {})
        print("   ✅ Structure batch_stats ré-initialisée.")

    variables = {'params': params, 'batch_stats': batch_stats}

    return model, variables, config_model


def get_iou(box1: List[float], box2: List[float], format1: str = "xywh", format2: str = "xywh") -> float:
    """
    Calcule l'Intersection over Union (IoU) de deux boxes.
    
    Args:
        box1: [x1, y1, w1, h1] ou [x1, y1, x2, y2]
        box2: [x2, y2, w2, h2] ou [x1, y1, x2, y2]
        format1: "xywh" ou "xyxy"
        format2: "xywh" ou "xyxy"
    
    Returns:
        IoU entre 0 et 1
    """
    # Convertir en xyxy si nécessaire
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
    
    # Calcul de l'intersection
    x_left = max(x1_1, x2_1)
    y_top = max(y1_1, y2_1)
    x_right = min(x1_2, x2_2)
    y_bottom = min(y1_2, y2_2)
    
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    
    # Calcul des aires
    area1 = (x1_2 - x1_1) * (y1_2 - y1_1)
    area2 = (x2_2 - x2_1) * (y2_2 - y2_1)
    
    union_area = area1 + area2 - intersection_area
    
    return intersection_area / union_area if union_area > 0 else 0.0


def non_max_suppression(boxes: List[List[float]], iou_threshold: float) -> List[List[float]]:
    """
    Applique le Non-Maximum Suppression (NMS) pour supprimer les boîtes superposées.
    boxes: liste de [x1, y1, x2, y2, score]
    """
    if not boxes:
        return []
    
    # Trier les boîtes par score (le dernier élément) décroissant
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
                                    conf_threshold: float = 0.3, 
                                    box_aera_min: int = 225) -> List[List[float]]:
    """
    Exécute la détection par Segmentation Sémantique (U-Net).
    Retourne une liste de boxes [x1, y1, x2, y2, score].
    """
    h_orig, w_orig = img_bgr.shape[:2]
    
    # 1. Prétraitement
    target_size = config_model.get("image_size", DEFAULT_DETECTION_IMAGE_SIZE)
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
    
    # 2. Inférence U-Net
    preds = model.apply(variables, jnp.array(img_jax), training=False)
    pred_mask = np.array(preds[0, :, :, 0])  # (224, 224)
    
    # 3. Redimensionnement à la taille de l'image originale
    mask_resized = cv2.resize(pred_mask, (w_orig, h_orig), interpolation=cv2.INTER_CUBIC)
    
    # 4. Binarisation
    strong_mask = (mask_resized > conf_threshold).astype(np.uint8) * 255
    weak_mask = (mask_resized > (conf_threshold * 0.4)).astype(np.uint8) * 255
    
    # 5. Dilatation légère des zones fortes
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    expanded_strong = cv2.dilate(strong_mask, kernel, iterations=1)
    
    # Intersection avec weak_mask
    binary_mask = cv2.bitwise_and(expanded_strong, weak_mask)
    
    # Fermeture morphologique
    closing_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, closing_kernel, iterations=1)
    
    # Dilatation pour récupérer les zones faibles autour
    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    binary_mask = cv2.dilate(binary_mask, dilate_kernel, iterations=1)
    
    # 6. Extraction des Contours
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    final_detections = []
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < box_aera_min:
            continue
            
        x, y, w, h = cv2.boundingRect(contour)
        
        sub_mask = mask_resized[y:y+h, x:x+w]
        score = float(np.max(sub_mask)) if sub_mask.size > 0 else 1.0
        
        final_detections.append([x, y, x+w, y+h, score])
        
    return final_detections


# =================================================================================================
# FONCTIONS DE COMPARAISON ET CALCUL DE LOSS
# =================================================================================================
def load_ground_truth_boxes(image_path: str, json_dir: str = None) -> List[Dict]:
    """
    Charge les ground truth boxes pour une image donnée.
    
    Args:
        image_path: Chemin vers l'image
        json_dir: Dossier contenant les JSON (par défaut, même dossier que l'image)
    
    Returns:
        Liste de dictionnaires avec les annotations ground truth
    """
    if json_dir is None:
        json_dir = os.path.dirname(image_path)
    
    file_name = os.path.basename(image_path)
    base_name = os.path.splitext(file_name)[0]
    
    ground_truths = []
    
    # Chercher tous les JSON correspondant à cette image
    for json_file in os.listdir(json_dir):
        if json_file.startswith(base_name) and json_file.endswith('.json'):
            json_path = os.path.join(json_dir, json_file)
            try:
                with open(json_path, 'r') as f:
                    data = json.load(f)
                
                # Vérifier que c'est une annotation valide (pas une prédiction)
                # On considère que c'est un ground truth si pas de detection_score
                annotation = data.get('annotation', {})
                if 'detection_score' not in annotation:
                    ground_truths.append(data)
            except (json.JSONDecodeError, KeyError):
                continue
    
    return ground_truths


def convert_xywh_to_xyxy(bbox: List[float]) -> List[float]:
    """Convertit une bbox de [x, y, w, h] à [x1, y1, x2, y2]"""
    x, y, w, h = bbox
    return [x, y, x + w, y + h]


def match_boxes(pred_boxes: List[List[float]], gt_boxes: List[Dict], 
                iou_threshold: float = 0.5) -> List[Tuple[int, int, float]]:
    """
    Associe les boxes prédites aux ground truth boxes en utilisant l'IoU.
    
    Args:
        pred_boxes: Liste de [x1, y1, x2, y2, score]
        gt_boxes: Liste de dict avec 'bbox' (xywh) et 'category_name'
        iou_threshold: Seuil minimal pour considérer un match
    
    Returns:
        Liste de tuples (pred_idx, gt_idx, iou)
    """
    matches = []
    
    # Convertir les gt_boxes en xyxy
    gt_boxes_xyxy = []
    for gt in gt_boxes:
        bbox_xywh = gt['annotation']['bbox']
        bbox_xyxy = convert_xywh_to_xyxy(bbox_xywh)
        gt_boxes_xyxy.append({
            'bbox': bbox_xyxy,
            'category_name': gt['annotation']['category_name'],
            'bbox_id': gt['annotation'].get('bbox_id', 0)
        })
    
    # Pour chaque prédiction, trouver le meilleur match
    for pred_idx, pred_box in enumerate(pred_boxes):
        pred_xyxy = pred_box[:4]  # [x1, y1, x2, y2]
        
        best_iou = 0.0
        best_gt_idx = -1
        
        for gt_idx, gt_box in enumerate(gt_boxes_xyxy):
            iou = get_iou(pred_xyxy, gt_box['bbox'], format1="xyxy", format2="xyxy")
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx
        
        if best_iou >= iou_threshold:
            matches.append((pred_idx, best_gt_idx, best_iou))
    
    return matches


def calculate_detection_loss(pred_boxes: List[List[float]], gt_boxes: List[Dict], 
                             class_names: List[str], 
                             alpha: float = 1.0, 
                             beta: float = 1.0) -> Dict:
    """
    Calcule la loss combinée pour la détection et la classification.
    
    Args:
        pred_boxes: Liste de [x1, y1, x2, y2, score, class_name, class_confidence]
        gt_boxes: Liste de dict avec 'bbox' (xywh) et 'category_name'
        class_names: Liste des noms de classes
        alpha: Poids pour la loss de localisation (IoU)
        beta: Poids pour la loss de classification
    
    Returns:
        Dictionnaire avec les différentes composantes de la loss
    """
    results = {
        'total_loss': 0.0,
        'localization_loss': 0.0,
        'classification_loss': 0.0,
        'false_positive_loss': 0.0,
        'false_negative_loss': 0.0,
        'num_true_positives': 0,
        'num_false_positives': 0,
        'num_false_negatives': 0,
        'matched_pairs': [],
        'unmatched_preds': [],
        'unmatched_gts': []
    }
    
    if not gt_boxes:
        # Pas de ground truth, toutes les prédictions sont des faux positifs
        results['false_positive_loss'] = len(pred_boxes)
        results['num_false_positives'] = len(pred_boxes)
        results['total_loss'] = results['false_positive_loss']
        return results
    
    if not pred_boxes:
        # Pas de prédictions, tous les ground truths sont des faux négatifs
        results['false_negative_loss'] = len(gt_boxes)
        results['num_false_negatives'] = len(gt_boxes)
        results['total_loss'] = results['false_negative_loss']
        return results
    
    # Convertir les gt_boxes en xyxy
    gt_boxes_xyxy = []
    for gt in gt_boxes:
        bbox_xywh = gt['annotation']['bbox']
        bbox_xyxy = convert_xywh_to_xyxy(bbox_xywh)
        gt_boxes_xyxy.append({
            'bbox': bbox_xyxy,
            'category_name': gt['annotation']['category_name'],
            'bbox_id': gt['annotation'].get('bbox_id', 0)
        })
    
    # Créer une matrice d'IoU entre toutes les paires
    iou_matrix = np.zeros((len(pred_boxes), len(gt_boxes_xyxy)))
    for i, pred_box in enumerate(pred_boxes):
        for j, gt_box in enumerate(gt_boxes_xyxy):
            iou_matrix[i, j] = get_iou(pred_box[:4], gt_box['bbox'], format1="xyxy", format2="xyxy")
    
    # Trouver les meilleurs matches (Hungarian algorithm simplifié)
    # On utilise une approche greedy pour l'instant
    matched_preds = set()
    matched_gts = set()
    
    # Trier les prédictions par score décroissant
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
        
        if best_iou >= 0.5:  # Seuil IoU pour un match
            matched_preds.add(pred_idx)
            matched_gts.add(best_gt_idx)
            
            # Calcul de la loss pour ce match
            pred_box = pred_boxes[pred_idx]
            gt_box = gt_boxes_xyxy[best_gt_idx]
            
            # Loss de localisation (1 - IoU)
            loc_loss = 1.0 - best_iou
            
            # Loss de classification (0 si bonne classe, 1 sinon)
            pred_class = pred_box[5] if len(pred_box) > 5 else DEFAULT_DEFAULT_CLASSE
            gt_class = gt_box['category_name']
            
            # Confiance de la prédiction
            pred_confidence = pred_box[6] if len(pred_box) > 6 else 1.0
            
            if pred_class == gt_class:
                cls_loss = 0.0
            else:
                # Penalty plus fort si confiance élevée mais mauvaise classe
                cls_loss = 1.0 * (1.0 - (1.0 - pred_confidence))  # = pred_confidence
            
            # Loss combinée pour ce match
            match_loss = alpha * loc_loss + beta * cls_loss
            
            results['localization_loss'] += loc_loss
            results['classification_loss'] += cls_loss
            results['num_true_positives'] += 1
            results['matched_pairs'].append({
                'pred_idx': pred_idx,
                'gt_idx': best_gt_idx,
                'iou': best_iou,
                'pred_class': pred_class,
                'gt_class': gt_class,
                'pred_confidence': pred_confidence,
                'loss': match_loss
            })
    
    # Faux positifs (prédictions non matchées)
    for pred_idx in range(len(pred_boxes)):
        if pred_idx not in matched_preds:
            results['num_false_positives'] += 1
            results['false_positive_loss'] += 1.0
            results['unmatched_preds'].append({
                'pred_idx': pred_idx,
                'bbox': pred_boxes[pred_idx][:4],
                'score': pred_boxes[pred_idx][4],
                'class': pred_boxes[pred_idx][5] if len(pred_boxes[pred_idx]) > 5 else DEFAULT_DEFAULT_CLASSE
            })
    
    # Faux négatifs (ground truths non matchés)
    for gt_idx in range(len(gt_boxes_xyxy)):
        if gt_idx not in matched_gts:
            results['num_false_negatives'] += 1
            results['false_negative_loss'] += 1.0
            results['unmatched_gts'].append({
                'gt_idx': gt_idx,
                'bbox': gt_boxes_xyxy[gt_idx]['bbox'],
                'class': gt_boxes_xyxy[gt_idx]['category_name']
            })
    
    # Normalisation par le nombre total de ground truths
    num_gts = len(gt_boxes)
    if num_gts > 0:
        results['localization_loss'] /= num_gts
        results['classification_loss'] /= num_gts
        results['false_positive_loss'] /= num_gts
        results['false_negative_loss'] /= num_gts
    
    # Loss totale
    results['total_loss'] = (
        alpha * results['localization_loss'] +
        beta * results['classification_loss'] +
        results['false_positive_loss'] +
        results['false_negative_loss']
    )
    
    return results


# =================================================================================================
# FONCTION PRINCIPALE
# =================================================================================================
def evaluate_model(input_dir: str, 
                   dataset_name: str = DEFAULT_DATASET_NAME,
                   checkpoint_path: str = DEFAULT_CHECKPOINT_PATH,
                   detection_checkpoint_path: str = DEFAULT_DETECTION_CHECKPOINT_PATH,
                   detection_image_size: Tuple[int, int] = DEFAULT_DETECTION_IMAGE_SIZE,
                   confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
                   detection_conf_threshold: float = DEFAULT_DETECTION_CONF_THRESHOLD,
                   box_area_min: int = DEFAULT_BOX_AREA_MIN,
                   nms_threshold: float = DEFAULT_NMS_THRESHOLD,
                   iou_threshold: float = 0.5,
                   alpha: float = 1.0,
                   beta: float = 1.0) -> Dict:
    """
    Évalue le modèle sur un dataset.
    
    Args:
        input_dir: Dossier contenant les images et les JSON ground truth
        dataset_name: Nom du dataset pour la configuration
        checkpoint_path: Chemin vers le modèle de classification
        detection_checkpoint_path: Chemin vers le modèle de détection
        detection_image_size: Taille d'entrée pour le modèle de détection
        confidence_threshold: Seuil de confiance pour la classification
        detection_conf_threshold: Seuil de confiance pour la détection
        box_area_min: Aire minimale pour une box
        nms_threshold: Seuil IoU pour le NMS
        iou_threshold: Seuil IoU pour le matching
        alpha: Poids pour la loss de localisation
        beta: Poids pour la loss de classification
    
    Returns:
        Dictionnaire avec les métriques globales
    """
    # 1. Chargement de la config
    print("\n🏗️  Initialisation...")
    try:
        config = get_dataset_config(dataset_name)
        CLASS_NAMES = config["class_names"]
        print(f"✅ Configuration chargée: {dataset_name}")
        print(f"📊 Classes ({len(CLASS_NAMES)}): {CLASS_NAMES}")
    except Exception as e:
        print(f"❌ Erreur chargement config: {e}")
        sys.exit(1)
    
    # 2. Chargement des modèles
    try:
        det_model, det_vars, det_config = load_detection_model(detection_checkpoint_path)
        print("✅ Modèle de DÉTECTION JAX chargé.")
    except Exception as e:
        print(f"❌ Erreur chargement modèle détection: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    try:
        clf_model, clf_vars, dataset_mean, dataset_std = load_jax_model(checkpoint_path, config)
        print("✅ Modèle de CLASSIFICATION JAX chargé.")
    except Exception as e:
        print(f"❌ Erreur chargement modèle classification: {e}")
        sys.exit(1)
    
    # 3. Préparation des fichiers
    image_files = []
    if not os.path.exists(input_dir):
        print(f"❌ Le dossier {input_dir} n'existe pas.")
        sys.exit(1)
    
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith((".jpg", ".png", ".jpeg", ".bmp")):
                image_files.append(os.path.join(root, file))
    
    print(f"\n📂 Évaluation sur {len(image_files)} images dans {input_dir}")
    
    # 4. Initialisation des métriques globales
    global_metrics = {
        'total_images': 0,
        'total_gt_boxes': 0,
        'total_pred_boxes': 0,
        'total_true_positives': 0,
        'total_false_positives': 0,
        'total_false_negatives': 0,
        'total_localization_loss': 0.0,
        'total_classification_loss': 0.0,
        'total_false_positive_loss': 0.0,
        'total_false_negative_loss': 0.0,
        'total_loss': 0.0,
        'per_image_metrics': [],
        'per_class_metrics': {class_name: {
            'true_positives': 0,
            'false_positives': 0,
            'false_negatives': 0
        } for class_name in CLASS_NAMES}
    }
    
    # 5. Boucle d'évaluation
    for img_path in tqdm(image_files, desc="Évaluation des images"):
        root = os.path.dirname(img_path)
        file_name = os.path.basename(img_path)
        
        # Lire l'image
        img = cv2.imread(img_path)
        if img is None:
            print(f"⚠️ Impossible de lire {img_path}")
            continue
        
        h, w = img.shape[:2]
        
        # Charger les ground truth boxes
        gt_boxes = load_ground_truth_boxes(img_path, root)
        global_metrics['total_gt_boxes'] += len(gt_boxes)
        
        # --- DÉTECTION ---
        detections = decode_segmentation_and_detect(
            img, det_model, det_vars, det_config,
            conf_threshold=detection_conf_threshold,
            box_aera_min=box_area_min
        )
        
        # --- NMS ---
        detections = non_max_suppression(detections, iou_threshold=nms_threshold)
        
        # --- CLASSIFICATION DES CROPS ---
        pred_boxes_with_class = []
        for (x1, y1, x2, y2, score) in detections:
            # Extraire le crop pour classification
            crop = img[y1:y2, x1:x2]
            
            if crop.size == 0 or crop.shape[0] == 0 or crop.shape[1] == 0:
                continue
            
            # Classification
            predicted_class, confidence = predict_crop(crop, clf_model, clf_vars, dataset_mean, dataset_std, config)
            
            # Filtrage par confiance
            if confidence < confidence_threshold:
                predicted_class = DEFAULT_DEFAULT_CLASSE
            
            pred_boxes_with_class.append([x1, y1, x2, y2, score, predicted_class, confidence])
        
        global_metrics['total_pred_boxes'] += len(pred_boxes_with_class)
        
        # --- CALCUL DE LA LOSS POUR CETTE IMAGE ---
        image_metrics = calculate_detection_loss(
            pred_boxes_with_class, gt_boxes, CLASS_NAMES,
            alpha=alpha, beta=beta
        )
        
        # Mise à jour des métriques globales
        global_metrics['total_true_positives'] += image_metrics['num_true_positives']
        global_metrics['total_false_positives'] += image_metrics['num_false_positives']
        global_metrics['total_false_negatives'] += image_metrics['num_false_negatives']
        global_metrics['total_localization_loss'] += image_metrics['localization_loss']
        global_metrics['total_classification_loss'] += image_metrics['classification_loss']
        global_metrics['total_false_positive_loss'] += image_metrics['false_positive_loss']
        global_metrics['total_false_negative_loss'] += image_metrics['false_negative_loss']
        global_metrics['total_loss'] += image_metrics['total_loss']
        
        # Métriques par image
        global_metrics['per_image_metrics'].append({
            'image': file_name,
            'num_gt_boxes': len(gt_boxes),
            'num_pred_boxes': len(pred_boxes_with_class),
            'metrics': image_metrics
        })
        
        # Métriques par classe
        for match in image_metrics['matched_pairs']:
            pred_class = match['pred_class']
            gt_class = match['gt_class']
            
            if pred_class in global_metrics['per_class_metrics']:
                global_metrics['per_class_metrics'][pred_class]['true_positives'] += 1
            
            if gt_class in global_metrics['per_class_metrics']:
                global_metrics['per_class_metrics'][gt_class]['true_positives'] += 1
        
        for fp in image_metrics['unmatched_preds']:
            pred_class = fp['class']
            if pred_class in global_metrics['per_class_metrics']:
                global_metrics['per_class_metrics'][pred_class]['false_positives'] += 1
        
        for fn in image_metrics['unmatched_gts']:
            gt_class = fn['class']
            if gt_class in global_metrics['per_class_metrics']:
                global_metrics['per_class_metrics'][gt_class]['false_negatives'] += 1
        
        global_metrics['total_images'] += 1
    
    # 6. Calcul des métriques finales
    if global_metrics['total_images'] > 0:
        global_metrics['avg_loss'] = global_metrics['total_loss'] / global_metrics['total_images']
        global_metrics['avg_localization_loss'] = global_metrics['total_localization_loss'] / global_metrics['total_images']
        global_metrics['avg_classification_loss'] = global_metrics['total_classification_loss'] / global_metrics['total_images']
        global_metrics['avg_false_positive_loss'] = global_metrics['total_false_positive_loss'] / global_metrics['total_images']
        global_metrics['avg_false_negative_loss'] = global_metrics['total_false_negative_loss'] / global_metrics['total_images']
    
    # Calcul des métriques de précision/rappel
    if global_metrics['total_gt_boxes'] > 0:
        global_metrics['precision'] = global_metrics['total_true_positives'] / (
            global_metrics['total_true_positives'] + global_metrics['total_false_positives'] + 1e-10
        )
        global_metrics['recall'] = global_metrics['total_true_positives'] / (
            global_metrics['total_true_positives'] + global_metrics['total_false_negatives'] + 1e-10
        )
        global_metrics['f1_score'] = 2 * (
            global_metrics['precision'] * global_metrics['recall'] / (
                global_metrics['precision'] + global_metrics['recall'] + 1e-10
            )
        )
    
    # Métriques par classe
    for class_name, metrics in global_metrics['per_class_metrics'].items():
        tp = metrics['true_positives']
        fp = metrics['false_positives']
        fn = metrics['false_negatives']
        
        if tp + fp > 0:
            metrics['precision'] = tp / (tp + fp)
        else:
            metrics['precision'] = 0.0
            
        if tp + fn > 0:
            metrics['recall'] = tp / (tp + fn)
        else:
            metrics['recall'] = 0.0
            
        if metrics['precision'] + metrics['recall'] > 0:
            metrics['f1_score'] = 2 * (metrics['precision'] * metrics['recall'] / (metrics['precision'] + metrics['recall']))
        else:
            metrics['f1_score'] = 0.0
    
    return global_metrics


def print_metrics(metrics: Dict):
    """Affiche les métriques de manière lisible."""
    print("\n" + "="*80)
    print("📊 RÉSULTATS DE L'ÉVALUATION")
    print("="*80)
    
    print(f"\n📁 Dataset:")
    print(f"   - Images traitées: {metrics['total_images']}")
    print(f"   - Boxes ground truth: {metrics['total_gt_boxes']}")
    print(f"   - Boxes prédites: {metrics['total_pred_boxes']}")
    
    print(f"\n🎯 Métriques globales:")
    print(f"   - True Positives: {metrics['total_true_positives']}")
    print(f"   - False Positives: {metrics['total_false_positives']}")
    print(f"   - False Negatives: {metrics['total_false_negatives']}")
    
    print(f"\n📉 Losses:")
    print(f"   - Localization Loss: {metrics.get('avg_localization_loss', 0):.4f}")
    print(f"   - Classification Loss: {metrics.get('avg_classification_loss', 0):.4f}")
    print(f"   - False Positive Loss: {metrics.get('avg_false_positive_loss', 0):.4f}")
    print(f"   - False Negative Loss: {metrics.get('avg_false_negative_loss', 0):.4f}")
    print(f"   - Total Loss: {metrics.get('avg_loss', 0):.4f}")
    
    print(f"\n🎯 Métriques de performance:")
    print(f"   - Precision: {metrics.get('precision', 0):.4f}")
    print(f"   - Recall: {metrics.get('recall', 0):.4f}")
    print(f"   - F1 Score: {metrics.get('f1_score', 0):.4f}")
    
    print(f"\n📊 Métriques par classe:")
    print("-" * 60)
    print(f"{'Classe':<20} {'TP':<6} {'FP':<6} {'FN':<6} {'Precision':<10} {'Recall':<10} {'F1':<10}")
    print("-" * 60)
    
    for class_name, class_metrics in sorted(metrics['per_class_metrics'].items()):
        print(f"{class_name:<20} {class_metrics['true_positives']:<6} {class_metrics['false_positives']:<6} "
              f"{class_metrics['false_negatives']:<6} {class_metrics.get('precision', 0):<10.4f} "
              f"{class_metrics.get('recall', 0):<10.4f} {class_metrics.get('f1_score', 0):<10.4f}")
    
    print("="*80)


# =================================================================================================
# MAIN
# =================================================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Évaluer un modèle de détection + classification sur un dataset")
    parser.add_argument("--input_dir", type=str, required=True, 
                        help="Dossier contenant les images et les JSON ground truth")
    parser.add_argument("--dataset_name", type=str, default=DEFAULT_DATASET_NAME,
                        help="Nom du dataset pour la configuration")
    parser.add_argument("--checkpoint_path", type=str, default=DEFAULT_CHECKPOINT_PATH,
                        help="Chemin vers le modèle de classification")
    parser.add_argument("--detection_checkpoint_path", type=str, default=DEFAULT_DETECTION_CHECKPOINT_PATH,
                        help="Chemin vers le modèle de détection")
    parser.add_argument("--confidence_threshold", type=float, default=DEFAULT_CONFIDENCE_THRESHOLD,
                        help="Seuil de confiance pour la classification")
    parser.add_argument("--detection_conf_threshold", type=float, default=DEFAULT_DETECTION_CONF_THRESHOLD,
                        help="Seuil de confiance pour la détection")
    parser.add_argument("--box_area_min", type=int, default=DEFAULT_BOX_AREA_MIN,
                        help="Aire minimale pour une box")
    parser.add_argument("--nms_threshold", type=float, default=DEFAULT_NMS_THRESHOLD,
                        help="Seuil IoU pour le NMS")
    parser.add_argument("--iou_threshold", type=float, default=0.5,
                        help="Seuil IoU pour le matching")
    parser.add_argument("--alpha", type=float, default=1.0,
                        help="Poids pour la loss de localisation")
    parser.add_argument("--beta", type=float, default=1.0,
                        help="Poids pour la loss de classification")
    
    args = parser.parse_args()
    
    # Exécution de l'évaluation
    start_time = time.time()
    metrics = evaluate_model(
        input_dir=args.input_dir,
        dataset_name=args.dataset_name,
        checkpoint_path=args.checkpoint_path,
        detection_checkpoint_path=args.detection_checkpoint_path,
        confidence_threshold=args.confidence_threshold,
        detection_conf_threshold=args.detection_conf_threshold,
        box_area_min=args.box_area_min,
        nms_threshold=args.nms_threshold,
        iou_threshold=args.iou_threshold,
        alpha=args.alpha,
        beta=args.beta
    )
    end_time = time.time()
    
    # Affichage des résultats
    print_metrics(metrics)
    
    print(f"\n⏱️  Temps d'exécution: {end_time - start_time:.2f} secondes")
    print(f"✅ Évaluation terminée!")
