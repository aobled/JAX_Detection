import os
import sys
import json
import cv2
import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
from tqdm import tqdm

# --- Imports locaux ---
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset_configs import get_dataset_config
from inference_utils import (
    load_detection_model, decode_segmentation_and_detect_batch, get_iou
)

# --- Configuration ---
DATASET_PATH = '/home/aobled/Downloads/Aircraft_DATASET/detection'
CONFIG_NAME = "FIGHTERJET_DETECTION"
BATCH_SIZE = 32
DETECTION_CONF_THRESHOLD = 0.3
BOX_AERA_MIN = 225


def load_detection_audit_model(config):
    """Charge le modèle de détection JAX."""
    checkpoint_path = config.get("checkpoint_path", "best_model_fighterjet_detection.pkl")
    
    # Résoudre le chemin relatif
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)
    checkpoint_path_abs = os.path.join(parent_dir, checkpoint_path)
    if not os.path.exists(checkpoint_path_abs):
        raise FileNotFoundError(f"Modèle de détection introuvable : {checkpoint_path_abs}")
        
    print(f"📦 Chargement du modèle de détection depuis {checkpoint_path_abs}...")
    det_model, det_vars, det_config = load_detection_model(checkpoint_path_abs)
    
    # Compilation JIT
    print("🔥 Compilation JIT du graphe de détection...")
    predict_fn = jax.jit(lambda x: det_model.apply(det_vars, x, training=False))
    print("✅ Modèle de détection prêt pour l'inférence par batch.")
    
    return predict_fn, det_config


def convert_bbox_format(bbox_xywh, img_width, img_height, target_size=(224, 224)):
    """
    Convertit une bbox au format [x, y, w, h] (COCO) en [x1, y1, x2, y2] absolu
    dans l'espace de l'image redimensionnée.
    """
    x, y, w, h = bbox_xywh
    
    # Convertir en coordonnées absolues dans l'image originale
    x1_orig = x
    y1_orig = y
    x2_orig = x + w
    y2_orig = y + h
    
    # Redimensionner vers target_size
    scale_x = target_size[0] / img_width
    scale_y = target_size[1] / img_height
    
    x1 = int(x1_orig * scale_x)
    y1 = int(y1_orig * scale_y)
    x2 = int(x2_orig * scale_x)
    y2 = int(y2_orig * scale_y)
    
    return [x1, y1, x2, y2]


def calculate_mean_iou(true_boxes, pred_boxes):
    """
    Calcule le IoU moyen entre les true boxes et les predicted boxes.
    
    Args:
        true_boxes: Liste de boxes au format [x1, y1, x2, y2]
        pred_boxes: Liste de boxes au format [x1, y1, x2, y2]
    
    Returns:
        float: IoU moyen (entre 0 et 1)
    """
    if not true_boxes or not pred_boxes:
        return 0.0
    
    total_iou = 0.0
    matches = 0
    
    for true_box in true_boxes:
        best_iou = 0.0
        for pred_box in pred_boxes:
            iou = get_iou(true_box, pred_box)
            if iou > best_iou:
                best_iou = iou
        total_iou += best_iou
        matches += 1
    
    return (total_iou / matches) if matches > 0 else 0.0


def run_audit():
    config = get_dataset_config(CONFIG_NAME)
    
    predict_fn, det_config = load_detection_audit_model(config)
    
    results = []
    image_batch = []
    meta_batch = []
    
    def flush_batch():
        nonlocal image_batch, meta_batch, results
        if not image_batch:
            return
        
        # Prédiction par batch
        batch_results = decode_segmentation_and_detect_batch(
            image_batch,
            predict_fn,
            det_config,
            conf_threshold=DETECTION_CONF_THRESHOLD,
            box_aera_min=BOX_AERA_MIN
        )
        
        # Traitement des résultats
        for i, meta in enumerate(meta_batch):
            pred_boxes_x1y1x2y2, _, _ = batch_results[i]
            
            # Convertir les pred_boxes en format [x1, y1, x2, y2] (déjà dans ce format)
            pred_boxes = []
            for box in pred_boxes_x1y1x2y2:
                x1, y1, x2, y2, conf = box
                pred_boxes.append([int(x1), int(y1), int(x2), int(y2)])
            
            # Calculer le IoU moyen
            mean_iou = calculate_mean_iou(meta["true_boxes"], pred_boxes)
            
            # Enregistrement des résultats
            results.append({
                "image_name": meta["image_name"],
                "split": meta["split"],
                "num_true_boxes": len(meta["true_boxes"]),
                "num_pred_boxes": len(pred_boxes),
                "mean_iou": round(mean_iou, 4),
                "status": "GOOD" if mean_iou >= 0.5 else "POOR",
                "directory": meta["directory"],
                "image_path": meta["image_path"]
            })
        
        image_batch.clear()
        meta_batch.clear()

    print("🔍 Début du scan du dataset de détection...")
    for split in ["train", "val"]:
        split_dir = os.path.join(DATASET_PATH, split)
        if not os.path.exists(split_dir):
            continue
        
        # Parcours des fichiers JSON
        json_files = []
        for root, _, files in os.walk(split_dir):
            json_files.extend([
                os.path.join(root, f) 
                for f in files 
                if f.endswith('.json')
            ])
        
        # Grouper les JSON par image
        image_annotations = {}
        for json_path in tqdm(json_files, desc=f"Chargement annotations {split}", leave=False):
            try:
                with open(json_path, 'r') as f:
                    data = json.load(f)
                
                img_filename = data["image"]["file_name"]
                img_path = os.path.join(os.path.dirname(json_path), img_filename)
                
                if not os.path.exists(img_path):
                    continue
                
                if img_filename not in image_annotations:
                    image_annotations[img_filename] = {
                        "path": img_path,
                        "boxes": [],
                        "json_files": []
                    }
                
                bbox = data["annotation"]["bbox"]
                image_annotations[img_filename]["boxes"].append(bbox)
                image_annotations[img_filename]["json_files"].append(json_path)
                
            except Exception as e:
                print(f"Erreur sur {json_path}: {e}")
        
        # Traiter chaque image
        for img_filename, img_data in tqdm(image_annotations.items(), 
                                            desc=f"Traitement images {split}", 
                                            leave=False):
            img_path = img_data["path"]
            true_boxes_xywh = img_data["boxes"]
            
            try:
                # Lecture de l'image
                img = cv2.imread(img_path)
                if img is None:
                    continue
                
                img_height, img_width = img.shape[:2]
                
                # Convertir les true boxes en format [x1, y1, x2, y2] pour le calcul IoU
                true_boxes_x1y1x2y2 = []
                for bbox in true_boxes_xywh:
                    true_box = convert_bbox_format(
                        bbox, 
                        img_width, 
                        img_height,
                        det_config.get("image_size", (224, 224))
                    )
                    true_boxes_x1y1x2y2.append(true_box)
                
                # Redimensionner l'image pour le modèle
                target_size = det_config.get("image_size", (224, 224))
                img_resized = cv2.resize(img, target_size)
                
                image_batch.append(img_resized)
                meta_batch.append({
                    "image_name": img_filename,
                    "split": split,
                    "true_boxes": true_boxes_x1y1x2y2,
                    "directory": os.path.basename(os.path.dirname(img_path)),
                    "image_path": img_path
                })
                
                if len(image_batch) >= BATCH_SIZE:
                    flush_batch()
                    
            except Exception as e:
                print(f"Erreur sur {img_path}: {e}")
    
    # Vider le dernier batch restant
    flush_batch()
    
    # --- Création du CSV et reporting Pandas ---
    print("💾 Export des résultats vers audit_detection_results.csv...")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, "audit_detection_results.csv")
    
    df = pd.DataFrame(results)
    if len(df) == 0:
        print("❌ Aucune donnée analysée.")
        return
        
    df.to_csv(csv_path, index=False)
    
    total = len(df)
    poor_iou = len(df[df["status"] == "POOR"])
    avg_iou = df["mean_iou"].mean() * 100
    
    print("==================")
    print("📊 BILAN DE L'AUDIT DÉTECTION")
    print("====================")
    print(f"   Total analysé : {total} images")
    print(f"   IoU Moyen    : {avg_iou:.2f} %")
    print(f"   Images POOR (IoU < 0.5) : {poor_iou}")
    
    print("   Répartition de l'IoU par split :")
    split_iou = df.groupby('split')['mean_iou'].mean() * 100
    for s, iou in split_iou.items():
        print(f"     - {s.upper()} : {iou:.2f} %")
    
    if poor_iou > 0:
        print("⚠️ TOP 20 PIRES IMAGES (IoU le plus bas) :")
        worst = df.sort_values(by="mean_iou", ascending=True).head(20)
        print(worst[["image_name", "split", "num_true_boxes", "num_pred_boxes", "mean_iou"]].to_string(index=False))
    
    if len(df) > 0:
        print("✅ MEILLEURES IMAGES (IoU le plus élevé) :")
        best = df.sort_values(by="mean_iou", ascending=False).head(10)
        print(best[["image_name", "split", "num_true_boxes", "num_pred_boxes", "mean_iou"]].to_string(index=False))


if __name__ == "__main__":
    run_audit()
