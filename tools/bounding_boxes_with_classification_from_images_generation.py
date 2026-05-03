
import sys
import os
import time

# Ajouter le répertoire parent en PRIORITÉ absolue (index 0) pour forcer Python
# à utiliser le model_library.py de JAX_Detection (et non celui de JAX_Classification si exécuté depuis l'autre dossier)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Remplace YOLO import par JAX/Flax helpers si nécessaire
# from ultralytics import YOLO  <-- REMOVED
import cv2
import json
import shutil
import numpy as np
import jax
import jax.numpy as jnp
from PIL import Image
import pickle
from tqdm import tqdm

from dataset_configs import get_dataset_config
from model_library import get_model  # Uniquement get_model (pas besoin de la classe directe)

# =================================================================================================
# CONFIGURATION
# =================================================================================================
# 1. Configuration du dataset et du modèle de classification
DATASET_NAME = "FIGHTERJET_CLASSIFICATION"     # Nom de la config dans dataset_configs.py
CHECKPOINT_PATH = "best_model.pkl"      # Chemin vers le modèle de CLASSIFICATION
INPUT_DIR = "/home/aobled/Downloads/tmp_multi/sr71"  # Dossier d'entrée (images à traiter)
CONFIDENCE_THRESHOLD = 0.7            # Seuil de confiance pour valider une CLASSIFICATION bet 0.96

# 2. Configuration du modèle de détection
DETECTION_CHECKPOINT_PATH = "best_model_detection.pkl" # Chemin vers le modèle de DÉTECTION
DETECTION_IMAGE_SIZE = (224, 224)       # Taille d'entrée du modèle de détection
DETECTION_CONF_THRESHOLD = 0.7          # Seuil pour considérer une détection valide (objectness + class) best 0.5

# 3. Configuration de la zone de détection
ELLIPSE_MARGIN_PERCENT = 5
ELLIPSE_ITERATIONS = 3

DEFAULT_CLASSE = "unknown"

# 3. Chargement de la config dataset
try:
    config = get_dataset_config(DATASET_NAME)
    CLASS_NAMES = config["class_names"]
    print(f"✅ Configuration chargée: {DATASET_NAME}")
    print(f"📊 Classes ({len(CLASS_NAMES)}): {CLASS_NAMES}")
    print(f"🔒 Seuil de confiance (Classification): {CONFIDENCE_THRESHOLD * 100}%")
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

def decode_segmentation_and_detect(img_bgr, model, variables, config_model, conf_threshold=0.3):
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
    
    # --- DILATATION MORPHOLOGIQUE ---
    # On gonfle organiquement la tache blanche pour inclure du contexte autour de l'avion
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ELLIPSE_MARGIN_PERCENT, ELLIPSE_MARGIN_PERCENT))
    binary_mask = cv2.dilate(binary_mask, kernel, iterations=ELLIPSE_ITERATIONS)
    
    # 5. Extraction des Contours
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    final_detections = []
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 50: # Ignorer le bruit microscopique
            continue
            
        x, y, w, h = cv2.boundingRect(contour)
        
        # Le "score" peut être la valeur moyenne ou max de probabilité dans la box
        sub_mask = mask_resized[y:y+h, x:x+w]
        score = float(np.max(sub_mask)) if sub_mask.size > 0 else 1.0
        
        final_detections.append((x, y, x+w, y+h, score))
        
    return final_detections

# =================================================================================================
# MAIN
# =================================================================================================

if __name__ == "__main__":
    # 1. Chargement des modèles
    print("\n🏗️  Initialisation...")
    
    # DÉTECTION (Custom JAX)
    try:
        det_model, det_vars, det_config = load_detection_model(DETECTION_CHECKPOINT_PATH)
        print("✅ Modèle de DÉTECTION JAX chargé.")
        print(f"   Config détection: Grid={det_config.get('grid_size', '?')}, Size={det_config.get('image_size', '?')}")
    except Exception as e:
        print(f"❌ Erreur chargement modèle détection: {e}")
        # traceback
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # CLASSIFICATION (JAX Classifier)
    try:
        clf_model, clf_vars, dataset_mean, dataset_std = load_jax_model(CHECKPOINT_PATH, config)
        print("✅ Modèle de CLASSIFICATION JAX chargé.")
    except Exception as e:
        print(f"❌ Erreur chargement modèle classification: {e}")
        sys.exit(1)


    # 2. Préparation des fichiers
    image_files = []
    if not os.path.exists(INPUT_DIR):
         print(f"❌ Le dossier {INPUT_DIR} n'existe pas.")
         sys.exit(1)
         
    for root, _, files in os.walk(INPUT_DIR):
        for file in files:
            if file.lower().endswith((".jpg", ".png", ".jpeg", ".bmp")):
                image_files.append(os.path.join(root, file))

    print(f"\n📂 Traitement de {len(image_files)} images dans {INPUT_DIR}")


    # 3. Boucle de traitement
    processed_count = 0
    planes_detected = 0

    for img_path in tqdm(image_files, desc="Traitement des images"):
        root = os.path.dirname(img_path)
        file_name = os.path.basename(img_path)
        
        # Lire l'image
        img = cv2.imread(img_path)
        if img is None:
            continue
        h, w = img.shape[:2]
        
        # --- DÉTECTION CUSTOM ---
        detections = decode_segmentation_and_detect(
            img, det_model, det_vars, det_config, 
            conf_threshold=DETECTION_CONF_THRESHOLD
        )
        
        json_files_created = []
        detected_classes = set()
        planes_in_image = 0
        
        for (x1, y1, x2, y2, score) in detections:
            # Note: Le modèle de détection ne prédit que "avion" (classe unique implicite)
            
            # Extraire le crop pour classification
            crop = img[y1:y2, x1:x2]
            
            if crop.size == 0 or crop.shape[0] == 0 or crop.shape[1] == 0:
                continue
                
            # --- CLASSIFICATION DU CROP ---
            predicted_class, confidence = predict_crop(crop, clf_model, clf_vars, dataset_mean, dataset_std, config)
            
            # Filtrage par confiance
            if confidence < CONFIDENCE_THRESHOLD:
                predicted_class = DEFAULT_CLASSE
            
            detected_classes.add(predicted_class)
            # ------------------------------
            
            # Préparer JSON
            bbox_float = [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]
            
            data = {
                "image": {
                    "file_name": file_name,
                    "width": w,
                    "height": h
                },
                "annotation": {
                    "file_name": file_name,
                    "bbox": bbox_float,
                    "category_name": predicted_class,  # Utilisation de la classe prédite !
                    "bbox_id": planes_in_image,
                    "detection_score": float(score), # Ajout du score de détection pour info
                    "classification_score": float(confidence)
                }
            }
            
            # Sauvegarde JSON
            base_name = os.path.splitext(file_name)[0]
            out_name = f"{base_name}_{planes_in_image}.json"
            out_path = os.path.join(root, out_name)
            
            with open(out_path, "w") as f:
                json.dump(data, f, indent=4)
                
            json_files_created.append(out_path)
            planes_in_image += 1
            planes_detected += 1
        
        # Organisation des fichiers (Déplacement)
        if planes_in_image > 0:
            # Déterminer le dossier de destination
            if len(detected_classes) == 1:
                # Une seule classe détectée -> Dossier au nom de la classe
                folder_name = list(detected_classes)[0]
            else:
                # Plusieurs classes détectées -> Dossier "multi"
                folder_name = "multi"
            
            dest_dir = os.path.join(INPUT_DIR, folder_name)
            os.makedirs(dest_dir, exist_ok=True)
            
            # Déplacer l'image
            try:
                shutil.move(img_path, os.path.join(dest_dir, file_name))
            except shutil.Error:
                pass # Évite de planter si l'image existe déjà (overwrite silencieux ou skip)
            
            # Déplacer les JSONs
            for jf in json_files_created:
                try:
                    shutil.move(jf, os.path.join(dest_dir, os.path.basename(jf)))
                except shutil.Error:
                    pass
                    
        processed_count += 1

    print(f"\n✅ Terminé !")
    print(f"   Images traitées: {processed_count}")
    print(f"   Avions détectés et classifiés: {planes_detected}")
