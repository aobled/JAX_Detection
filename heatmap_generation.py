import sys
import os
import cv2
import numpy as np
import jax
import jax.numpy as jnp
import pickle

# Configuration:
IMAGE_PATH = "/home/aobled/Downloads/test_image.png" # Remplacer par le chemin vers ton image de test
OUTPUT_PATH = "/home/aobled/Downloads/heatmap_output.jpg"
DETECTION_CHECKPOINT_PATH = "best_model_detection.pkl"

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

def generate_heatmap(img_bgr, model, variables, config_model):
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
    
    # preds contient 3 tenseurs: (grid_28, grid_14, grid_7)
    # (1, 28, 28, 5), (1, 14, 14, 5), (1, 7, 7, 5)
    
    heatmaps = []
    
    for i, pred_grid in enumerate(preds):
        grid_np = np.array(pred_grid[0]) # Enlever dimension Batch
        
        S = grid_np.shape[0]
        C_pred = grid_np.shape[-1]
        
        # Sur V7, la sortie est (S, S, 5). 
        # Si C_pred est multiple de 5, on fait attention au format (S, S, B_boxes, 5)
        # Mais le Anchor-Free V7 sort généralement juste (S, S, 5).
        if C_pred > 5:
            B_boxes = C_pred // 5
            grid_np = grid_np.reshape((S, S, B_boxes, 5))
            # Prendre la confiance maximum parmi toutes les ancres de la cellule
            conf_map = np.max(grid_np[..., 0], axis=-1)
        else:
            conf_map = grid_np[..., 0] # Prendre le canal de Confidence (0.0 à 1.0)
            
        # Normaliser visuellement (au cas où la confiance max est basse, pour bien voir les blobs)
        # Optionnel: on peut laisser la valeur brute si on veut voir la "vraie" certitude
        # Ici on utilise la valeur brute [0, 1] pour voir si le réseau est sûr de lui.
        conf_map_scaled = np.clip(conf_map * 255.0, 0, 255).astype(np.uint8)
        
        # Redimensionner à la taille d'origine
        heatmap_resized = cv2.resize(conf_map_scaled, (w_orig, h_orig), interpolation=cv2.INTER_CUBIC)
        
        # Appliquer la Colormap (Rouge = Hautement confiant, Bleu = 0)
        heatmap_color = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
        
        # Fusionner avec l'image originale
        alpha = 0.5 # Transparence
        blended = cv2.addWeighted(img_bgr, 1 - alpha, heatmap_color, alpha, 0)
        
        # Ajouter le titre de la grille
        cv2.putText(blended, f"Grille {S}x{S}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
        heatmaps.append(blended)
        
    # Combiner les images horizontalement
    # (Original | Grid 28 | Grid 14 | Grid 7)
    
    # Mettre un titre sur l'originale
    orig_drawn = img_bgr.copy()
    cv2.putText(orig_drawn, "Image Originale", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
    
    # Assemblage
    all_images = [orig_drawn] + heatmaps
    
    # Redimensionner tout pour que ça rentre sur un écran standard (horizontalement)
    scale_down = min(1.0, 1920 / (w_orig * len(all_images))) 
    
    final_output = cv2.hconcat(all_images)
    
    if scale_down < 1.0:
        new_w = int(final_output.shape[1] * scale_down)
        new_h = int(final_output.shape[0] * scale_down)
        final_output = cv2.resize(final_output, (new_w, new_h))
        
    return final_output

if __name__ == "__main__":
    if not os.path.exists(IMAGE_PATH):
        print(f"❌ Image de test non trouvée: {IMAGE_PATH}")
        print("   Change le chemin dans le script: IMAGE_PATH = '...'")
        sys.exit(1)
        
    print("🚀 Initialisation du générateur de Heatmaps...")
    det_model, det_vars, det_config = load_detection_model(DETECTION_CHECKPOINT_PATH)
    
    img_bgr = cv2.imread(IMAGE_PATH)
    if img_bgr is None:
        print("❌ Erreur de lecture de l'image (format non supporté ?)")
        sys.exit(1)
        
    final_image = generate_heatmap(img_bgr, det_model, det_vars, det_config)
    
    cv2.imwrite(OUTPUT_PATH, final_image)
    print(f"✅ Terminé ! Carte de chaleur sauvegardée dans: {OUTPUT_PATH}")
