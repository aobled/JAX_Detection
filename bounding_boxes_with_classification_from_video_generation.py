import sys
import os

# Ajouter le répertoire parent en PRIORITÉ absolue (index 0) pour forcer Python
# à utiliser le model_library.py de JAX_Detection (et non celui de JAX_Classification si exécuté depuis l'autre dossier)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Remplace YOLO import par JAX/Flax helpers si nécessaire
# from ultralytics import YOLO  <-- REMOVED
import cv2
import numpy as np
import jax
import jax.numpy as jnp
import pickle
from tqdm import tqdm

from dataset_configs import get_dataset_config
from model_library import get_model  # Uniquement get_model (pas besoin de la classe directe)

# =================================================================================================
# Input CONFIGURATION
# =================================================================================================
# 1. Configuration du dataset et du modèle de classification
DATASET_NAME = "FIGHTERJET_CLASSIFICATION"     # Nom de la config dans dataset_configs.py
CHECKPOINT_PATH = "best_model.pkl"      # Chemin vers le modèle de CLASSIFICATION

# 2. Configuration du modèle de détection
DETECTION_CHECKPOINT_PATH = "best_model_detection.pkl" # Chemin vers le modèle de DÉTECTION
DETECTION_IMAGE_SIZE = (224, 224)       # Taille d'entrée du modèle de détection

# 3. Configuration de la zone de détection
CROP_MARGIN_PERCENT = 5  # 15 = Ajoute 15% de marge autour de la détection pour le classifieur
BOX_AERA_MIN = 1225

# PRIORITÉ AU DOSSIER PARENT
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ==========================================================
# Detection CONFIGURATION
# ==========================================================
VIDEO_PATH = "/media/aobled/Elements/Python/videos/A-10, P-51, F-35, F-22 Oshkosh 2019 - short.mp4"
OUTPUT_DIR = "/home/aobled/Downloads/video_frames_annotated"

FRAME_STRIDE = 1  # 1 = toutes les frames
#CONFIDENCE_THRESHOLD = 0.5            # Seuil de confiance pour valider une CLASSIFICATION bet 0.96
DETECTION_CONF_THRESHOLD = 0.6          # Seuil pour considérer une détection valide (objectness + class) target 0.6
#TARGET_CLASS_LIST = ["f16", "a10", "b52", "b1b", "b2", "f22", "f15"]
TARGET_CLASS_LIST = ["a10","mustang", "f35", "f22", "c130"]

# 3. Chargement de la config dataset
try:
    config = get_dataset_config(DATASET_NAME)
    CLASS_NAMES = config["class_names"]
    print(f"✅ Configuration chargée: {DATASET_NAME}")
    print(f"📊 Classes ({len(CLASS_NAMES)}): {CLASS_NAMES}")
    #print(f"🔒 Seuil de confiance (Classification): {CONFIDENCE_THRESHOLD * 100}%")
except Exception as e:
    print(f"❌ Erreur chargement config: {e}")
    sys.exit(1)





# =================================================================================================
# FONCTIONS UTILITAIRES POUR JAX (CLASSIFICATION)
# =================================================================================================
def load_jax_model(checkpoint_path, config):
    """Charge le modèle JAX de CLASSIFICATION."""
    if not os.path.exists(checkpoint_path):
        # Essayer de trouver le checkpoint dans le dossier parent si on est dans tools/
        parent_checkpoint = os.path.join(os.path.dirname(os.getcwd()), checkpoint_path)
        if os.path.exists(parent_checkpoint):
            checkpoint_path = parent_checkpoint
        else:
            # Essayer de trouver le checkpoint dans le dossier parent du script
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
    # Tenter de résoudre le chemin relatif
    if not os.path.exists(mean_std_path):
        # Essayer chemin relatif au script
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
            
            # 🚑 CORRECTION AUTOMATIQUE : Si Grayscale mais stats RGB
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

def predict_crop(crop_img, model, variables, mean, std, config):
    """
    Prédit la classe d'un crop (image OpenCV BGR).
    Retourne: (nom_classe, confiance)
    """
    # 1. Prétraitement
    target_size = config["image_size"]  # (128, 128)
    grayscale = config.get("grayscale", False)
    
    # Resize
    crop_resized = cv2.resize(crop_img, target_size)
    
    # Conversion couleur et normalisation
    if grayscale:
        # BGR -> Gray
        img_input = cv2.cvtColor(crop_resized, cv2.COLOR_BGR2GRAY)
        img_input = img_input.astype(np.float32) / 255.0
        # Normalisation
        img_normalized = (img_input - mean) / std
        # Ajout dimensions: (H, W) -> (1, H, W, 1)
        img_jax = img_normalized[np.newaxis, :, :, np.newaxis]
    else:
        # BGR -> RGB
        img_input = cv2.cvtColor(crop_resized, cv2.COLOR_BGR2RGB)
        img_input = img_input.astype(np.float32) / 255.0
        # Normalisation
        img_normalized = (img_input - mean) / std
        # Ajout dimensions: (H, W, 3) -> (1, H, W, 3)
        img_jax = img_normalized[np.newaxis, :, :, :]

    # 2. Inférence JAX
    logits = model.apply(variables, jnp.array(img_jax), training=False)
    probs = jax.nn.softmax(logits, axis=-1)
    
    # 3. Résultat
    pred_idx = int(jnp.argmax(probs))
    confidence = float(probs[0, pred_idx])
    
    return config["class_names"][pred_idx], confidence


# =================================================================================================
# FONCTIONS UTILITAIRES POUR DÉTECTION (NOUVEAU)
# =================================================================================================
def load_detection_model(checkpoint_path):
    """Charge le modèle JAX de DÉTECTION."""
    if not os.path.exists(checkpoint_path):
        # Essayer de trouver le checkpoint dans le dossier parent si on est dans tools/
        parent_checkpoint = os.path.join(os.path.dirname(os.getcwd()), checkpoint_path)
        if os.path.exists(parent_checkpoint):
            checkpoint_path = parent_checkpoint
        else:
            # Essayer de trouver le checkpoint dans le dossier parent du script
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
        # Tenter de trouver dans 'model_state'
        if 'model_state' in data_model:
             batch_stats = data_model['model_state'].get('batch_stats', {})
             
    if not batch_stats:
        print("⚠️  ATTENTION: 'batch_stats' non trouvés dans le checkpoint !")
        print("   Le modèle utilise des BatchNorms mais les stats (moyenne/variance) n'ont pas été sauvegardées.")
        print("   🔧 Tentative de ré-initialisation (les stats seront à 0/1, ce qui peut affecter la performance).")
        
        # On doit ré-initialiser le modèle pour obtenir la structure des batch_stats
        rng = jax.random.PRNGKey(0)
        target_size = config_model.get("image_size", DETECTION_IMAGE_SIZE)
        grayscale = config_model.get("grayscale", True)
        channels = 1 if grayscale else 3
        dummy_input = jnp.ones((1, *target_size, channels), jnp.float32)
        
        # Init pour avoir la structure
        init_variables = model.init(rng, dummy_input, training=True)
        batch_stats = init_variables.get('batch_stats', {})
        print("   ✅ Structure batch_stats ré-initialisée.")

    variables = {'params': params, 'batch_stats': batch_stats}

    return model, variables, config_model

def get_iou(box1, box2):
    """Calcule l'Intersection over Union (IoU) de deux boxes [x1, y1, x2, y2]"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    union = area1 + area2 - intersection
    return intersection / union if union > 0 else 0

def non_max_suppression(boxes, iou_threshold):
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
            iou = get_iou(current_box[:4], kept_box[:4])
            if iou > iou_threshold:
                overlap = True
                break
        if not overlap:
            kept_boxes.append(current_box)
            
    return kept_boxes




def decode_segmentation_and_detect(img_bgr, model, variables, config_model, conf_threshold=0.3, box_aera_min=225):
    """
    Exécute la détection par Segmentation Sémantique (U-Net).
    Retourne une liste de boxes [x1, y1, x2, y2, score].
    """
    h_orig, w_orig = img_bgr.shape[:2]
    
    # 1. Prétraitement
    target_size = config_model.get("image_size", DETECTION_IMAGE_SIZE)
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
    # preds est (1, 224, 224, 1)
    
    pred_mask = np.array(preds[0, :, :, 0]) # (224, 224)
    
    # 3. Redimensionnement à la taille de l'image originale
    mask_resized = cv2.resize(pred_mask, (w_orig, h_orig), interpolation=cv2.INTER_CUBIC)
    
    # 4. Binarisation
    binary_mask = (mask_resized > conf_threshold).astype(np.uint8) * 255
    
    # --- SÉPARATION DES OBJETS PROCHES (ÉROSION) ---
    # Érosion légère pour "casser" les ponts (glu) entre les avions très proches
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary_mask = cv2.erode(binary_mask, kernel, iterations=1)
    
    # --- EXTRACTION ET MARGE PROPORTIONNELLE ---
    # On a supprimé cv2.dilate car l'effet en pixels absolus était trop faible sur du 1080p.
    # Il est bien plus performant et juste d'appliquer une marge en pourcentage sur la bounding box finale.
    
    # 5. Extraction des Contours
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    final_detections = []
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < box_aera_min: # Ignorer le bruit microscopique
            continue
            
        x, y, w, h = cv2.boundingRect(contour)
        
        # Le "score" peut être la valeur moyenne ou max de probabilité dans la box STRICTE
        sub_mask = mask_resized[y:y+h, x:x+w]
        score = float(np.max(sub_mask)) if sub_mask.size > 0 else 1.0
        
        # Application de la marge proportionnelle (ex: 15% de la taille de l'avion)
        margin_x = int(w * (CROP_MARGIN_PERCENT / 100.0))
        margin_y = int(h * (CROP_MARGIN_PERCENT / 100.0))
        
        x1 = max(0, x - margin_x)
        y1 = max(0, y - margin_y)
        x2 = min(w_orig, x + w + margin_x)
        y2 = min(h_orig, y + h + margin_y)
        
        final_detections.append((x1, y1, x2, y2, score))
        
    return final_detections, mask_resized

# =================================================================================================
# MAIN
# =================================================================================================



# ==========================================================
# ⚠️ COLLER ICI TES FONCTIONS EXISTANTES :
# - load_jax_model
# - load_detection_model
# - predict_crop
# - decode_grid_and_detect
# - non_max_suppression
# - get_iou
# ==========================================================


# ==========================================================
# MAIN
# ==========================================================
def build_quadrant_canvas(target_frame, target_heatmap, target_detections, clf_model, clf_vars, dataset_mean, dataset_std, config):
    canvas = np.zeros((1080, 1920, 3), dtype=np.uint8)
    
    # 1. Top-Left: Original
    tl_img = cv2.resize(target_frame, (960, 540))
    canvas[0:540, 0:960] = tl_img
    
    # 2. Bottom-Left: Heatmap
    hm_vis = (target_heatmap * 255).astype(np.uint8)
    hm_color = cv2.applyColorMap(hm_vis, cv2.COLORMAP_JET)
    bl_img = cv2.resize(hm_color, (960, 540))
    canvas[540:1080, 0:960] = bl_img
    
    # 3. Top-Right: Annotated
    draw_frame = target_frame.copy()
    
    # 4. Bottom-Right: Classification Crops
    br_canvas = np.zeros((540, 960, 3), dtype=np.uint8)
    crop_idx = 0
    grid_cols = 7
    grid_rows = 4
    cell_w = 128
    cell_h = 128
    
    for (x1, y1, x2, y2, det_score) in target_detections:
        crop = target_frame[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        predicted_class, confidence = predict_crop(crop, clf_model, clf_vars, dataset_mean, dataset_std, config)

        # Draw Top-Right
        color = (0, 255, 0) if predicted_class in TARGET_CLASS_LIST else (0, 0, 255)
        label = f"{predicted_class} ({confidence:.2f})"
        cv2.rectangle(draw_frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(draw_frame, label, (x1, max(y1 - 10, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Draw Bottom-Right
        if crop_idx < grid_cols * grid_rows:
            col = crop_idx % grid_cols
            row = crop_idx // grid_cols
            
            crop_resized = cv2.resize(crop, (cell_w, cell_h))
            if config.get("grayscale", False):
                crop_gray = cv2.cvtColor(crop_resized, cv2.COLOR_BGR2GRAY)
                crop_disp = cv2.cvtColor(crop_gray, cv2.COLOR_GRAY2BGR)
            else:
                crop_disp = crop_resized
                
            # On ne dessine PLUS sur crop_disp pour ne pas mordre sur l'image !
            
            x_start = col * cell_w + 10 + (col * 5)
            y_start = row * (cell_h + 20) + 20 + (row * 5) # +20 pour laisser de la place au texte au-dessus
            
            if x_start + cell_w <= 960 and y_start + cell_h <= 540:
                br_canvas[y_start:y_start+cell_h, x_start:x_start+cell_w] = crop_disp
                # Dessiner le texte sur le canvas noir, juste au-dessus de l'image
                cv2.putText(br_canvas, predicted_class, (x_start, y_start - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            crop_idx += 1
            
    # Place Top-Right on canvas
    tr_img = cv2.resize(draw_frame, (960, 540))
    canvas[0:540, 960:1920] = tr_img
    
    # Place Bottom-Right on canvas
    canvas[540:1080, 960:1920] = br_canvas
    
    return canvas

if __name__ == "__main__":
    
    
    print("🏗️ Chargement des modèles...")

    config = get_dataset_config(DATASET_NAME)

    det_model, det_vars, det_config = load_detection_model(DETECTION_CHECKPOINT_PATH)
    clf_model, clf_vars, dataset_mean, dataset_std = load_jax_model(CHECKPOINT_PATH, config)

    print("✅ Modèles chargés.")

    # ======================================================
    # OUVERTURE VIDEO
    # ======================================================
    cap = cv2.VideoCapture(VIDEO_PATH)

    if not cap.isOpened():
        print("❌ Impossible d'ouvrir la vidéo.")
        sys.exit(1)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"🎬 {total_frames} frames détectées")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    frame_id = 0
    saved_count = 0

    with tqdm(total=total_frames, desc="Traitement vidéo") as pbar:

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Skip frames si stride > 1
            if frame_id % FRAME_STRIDE != 0:
                frame_id += 1
                pbar.update(1)
                continue

            original_frame = frame.copy()

            # ==================================================
            # DÉTECTION
            # ==================================================
            # Inférence
            detections, heatmap = decode_segmentation_and_detect(
                original_frame, 
                det_model, det_vars, det_config, 
                conf_threshold=DETECTION_CONF_THRESHOLD,
                box_aera_min=BOX_AERA_MIN
            )

            target_idx = frame_id
            target_frame = original_frame
            target_detections = detections
            target_heatmap = heatmap
                
            # ==================================================
            # CONSTRUCTION DU CANVAS 4-QUARTS
            # ==================================================
            canvas = build_quadrant_canvas(
                target_frame, target_heatmap, target_detections, 
                clf_model, clf_vars, dataset_mean, dataset_std, config
            )

            # ==================================================
            # SAUVEGARDE IMAGE
            # ==================================================
            output_path = os.path.join(OUTPUT_DIR, f"frame_{target_idx:06d}.jpg")
            cv2.imwrite(output_path, canvas)
            saved_count += 1

            frame_id += 1
            pbar.update(1)

    cap.release()

    print("\n✅ Terminé !")
    print(f"📸 Images sauvegardées : {saved_count}")
    print(f"📂 Dossier de sortie : {OUTPUT_DIR}")
    
    
    # ======================================================
    # RECONSTRUCTION VIDÉO À PARTIR DES IMAGES
    # ======================================================
        
    print("\n🎬 Reconstruction de la vidéo finale...")
    
    cap_original = cv2.VideoCapture(VIDEO_PATH)
    fps = cap_original.get(cv2.CAP_PROP_FPS)
    cap_original.release()
    
    # Récupérer les dimensions à partir de la première image générée
    # (car le canvas des 4 quadrants force le 1920x1080)
    image_files = sorted([
        f for f in os.listdir(OUTPUT_DIR)
        if f.endswith(".jpg")
    ])
    
    if len(image_files) == 0:
        print("❌ Aucune image générée pour la reconstruction.")
        sys.exit(1)
        
    first_frame = cv2.imread(os.path.join(OUTPUT_DIR, image_files[0]))
    height, width = first_frame.shape[:2]
    
    # Codec robuste
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
    OUTPUT_VIDEO_PATH = os.path.join(OUTPUT_DIR, "reconstructed_video.mp4")
    
    video_writer = cv2.VideoWriter(
        OUTPUT_VIDEO_PATH,
        fourcc,
        fps,
        (width, height)
    )
    
    if not video_writer.isOpened():
        print("❌ ERREUR : Impossible d'ouvrir VideoWriter.")
        exit()
    
    # image_files est déjà chargé plus haut
    
    for img_name in tqdm(image_files, desc="Assemblage vidéo"):
        img_path = os.path.join(OUTPUT_DIR, img_name)
        frame = cv2.imread(img_path)
    
        if frame is None:
            continue
    
        video_writer.write(frame)
    
    video_writer.release()
    
    # Vérification finale
    if os.path.exists(OUTPUT_VIDEO_PATH):
        print("✅ Vidéo reconstruite avec succès !")
        print(f"🎥 Fichier : {OUTPUT_VIDEO_PATH}")
    else:
        print("❌ La vidéo n'a pas été créée.")