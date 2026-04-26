import sys
import os
import cv2
import numpy as np
import jax
import jax.numpy as jnp
import pickle

# Configuration:
IMAGE_PATH = "/home/aobled/Downloads/8aab6779b09a496c.jpg" # Remplacer par l'image de test avec les gros et petits avions
OUTPUT_PATH = "/home/aobled/Downloads/heatmap_contouring_output.jpg"
DETECTION_CHECKPOINT_PATH = "best_model_detection.pkl"
HEATMAP_THRESHOLD = 0.01 # 30% de confiance minimum pour binariser la chaleur

# Ajouter le répertoire racine
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model_library import get_model

def load_detection_model(checkpoint_path):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint détection non trouvé: {checkpoint_path}")

    print(f"🔍 Chargement du modèle DÉTECTION depuis {checkpoint_path}...")
    with open(checkpoint_path, 'rb') as f:
        data_model = pickle.load(f)

    params = data_model['params']
    config_model = data_model.get('config', {})
    model_name = config_model.get('model_name', 'aircraft_detector_v7_advanced')
    
    print(f"   Modèle détecté: {model_name}")
    model = get_model(model_name, dropout_rate=0.0) 
    
    batch_stats = data_model.get('batch_stats', {})
    if not batch_stats and 'model_state' in data_model:
        batch_stats = data_model['model_state'].get('batch_stats', {})
             
    variables = {'params': params, 'batch_stats': batch_stats}
    return model, variables, config_model

def detect_by_contouring(img_bgr, model, variables, config_model, threshold=0.3):
    h_orig, w_orig = img_bgr.shape[:2]
    target_size = config_model.get("image_size", (224, 224))
    grayscale = config_model.get("grayscale", True)
    
    # 1. Prétraitement
    img_resized = cv2.resize(img_bgr, target_size)
    
    if grayscale:
        img_input = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
        img_input = img_input.astype(np.float32) / 255.0
        img_jax = img_input[np.newaxis, :, :, np.newaxis]
    else:
        img_input = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        img_input = img_input.astype(np.float32) / 255.0
        img_jax = img_input[np.newaxis, :, :, :]
        
    # 2. Inférence
    preds = model.apply(variables, jnp.array(img_jax), training=False)
    
    # 3. Extraction et Fusion des Heatmaps
    fused_heatmap = np.zeros((h_orig, w_orig), dtype=np.float32)
    
    for i, pred_grid in enumerate(preds):
        grid_np = np.array(pred_grid[0])
        S = grid_np.shape[0]
        C_pred = grid_np.shape[-1]
        
        if C_pred > 5:
            B_boxes = C_pred // 5
            grid_np = grid_np.reshape((S, S, B_boxes, 5))
            conf_map = np.max(grid_np[..., 0], axis=-1)
        else:
            conf_map = grid_np[..., 0] 
            
        # Redimensionner la carte de cette couche à la taille originale de l'image
        layer_heatmap = cv2.resize(conf_map, (w_orig, h_orig), interpolation=cv2.INTER_CUBIC)
        
        # Max Pooling: fusionner avec la carte globale
        fused_heatmap = np.maximum(fused_heatmap, layer_heatmap)
        
    # 4. Binarisation (Thresholding)
    # fused_heatmap est entre 0.0 et 1.0
    binary_mask = (fused_heatmap > threshold).astype(np.uint8) * 255
    
    # 5. Extraction des Contours
    # RETR_EXTERNAL: on ne veut que les contours extérieurs (pas les trous dans un gros avion)
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    boxes = []
    drawn_img = img_bgr.copy()
    
    for contour in contours:
        # Filtrer les micro-artefacts (bruit d'1 pixel)
        area = cv2.contourArea(contour)
        if area < 50: # Seuil minimum de taille en pixels pour être considéré comme un avion
            continue
            
        # Trouver la boîte englobante mathématiquement parfaite autour du contour
        x, y, w, h = cv2.boundingRect(contour)
        boxes.append([x, y, x+w, y+h])
        
        # Dessiner le rectangle sur l'image
        cv2.rectangle(drawn_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(drawn_img, "Avion", (x, max(y - 10, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
    # 6. Assemblage Visuel pour le test (Original Tracé | Heatmap Fusionnée | Masque Binaire)
    mask_bgr = cv2.cvtColor(binary_mask, cv2.COLOR_GRAY2BGR)
    heatmap_color = cv2.applyColorMap((fused_heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
    
    cv2.putText(drawn_img, "Bounding Boxes (Contouring)", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
    cv2.putText(heatmap_color, "Heatmap Fusion", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
    cv2.putText(mask_bgr, "Masque Binaire (Trouver les Iles)", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
    
    final_output = cv2.hconcat([drawn_img, heatmap_color, mask_bgr])
    
    # Redimensionner pour que ça s'affiche bien sur un écran
    scale_down = min(1.0, 1920 / final_output.shape[1]) 
    if scale_down < 1.0:
        new_w = int(final_output.shape[1] * scale_down)
        new_h = int(final_output.shape[0] * scale_down)
        final_output = cv2.resize(final_output, (new_w, new_h))
        
    return final_output, boxes

if __name__ == "__main__":
    if not os.path.exists(IMAGE_PATH):
        print(f"❌ Image de test non trouvée: {IMAGE_PATH}")
        print("   Change le chemin dans le script: IMAGE_PATH = '...'")
        sys.exit(1)
        
    print("🚀 Initialisation de la détection par Contouring...")
    det_model, det_vars, det_config = load_detection_model(DETECTION_CHECKPOINT_PATH)
    
    img_bgr = cv2.imread(IMAGE_PATH)
    if img_bgr is None:
        print("❌ Erreur de lecture de l'image (format non supporté ?)")
        sys.exit(1)
        
    final_image, final_boxes = detect_by_contouring(img_bgr, det_model, det_vars, det_config, threshold=HEATMAP_THRESHOLD)
    
    print(f"🎯 {len(final_boxes)} avions détectés par contouring !")
    
    cv2.imwrite(OUTPUT_PATH, final_image)
    print(f"✅ Terminé ! Image composite sauvegardée dans: {OUTPUT_PATH}")
