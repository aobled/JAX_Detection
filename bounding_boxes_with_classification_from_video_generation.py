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
DATASET_NAME = "FIGHTERJET_CLASSES"     # Nom de la config dans dataset_configs.py
CHECKPOINT_PATH = "best_model.pkl"      # Chemin vers le modèle de CLASSIFICATION

# 2. Configuration du modèle de détection
DETECTION_CHECKPOINT_PATH = "best_model_detection.pkl" # Chemin vers le modèle de DÉTECTION
DETECTION_IMAGE_SIZE = (224, 224)       # Taille d'entrée du modèle de détection

# PRIORITÉ AU DOSSIER PARENT
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ==========================================================
# Detection CONFIGURATION
# ==========================================================
VIDEO_PATH = "/home/aobled/Downloads/EAA AirVenture Oshkosh.mp4"
OUTPUT_DIR = "/home/aobled/Downloads/video_frames_annotated"

FRAME_STRIDE = 1  # 1 = toutes les frames

DATASET_NAME = "FIGHTERJET_CLASSES"
CHECKPOINT_PATH = "best_model.pkl"
DETECTION_CHECKPOINT_PATH = "best_model_detection.pkl"

CONFIDENCE_THRESHOLD = 0.5            # Seuil de confiance pour valider une CLASSIFICATION bet 0.96
DETECTION_CONF_THRESHOLD = 0.4          # Seuil pour considérer une détection valide (objectness + class) best 0.7
NMS_THRESHOLD = 0.5                     # Seuil IoU pour NMS best 0.4
DEFAULT_CLASSE = "unknown"
TARGET_CLASS_LIST = ["f22", "f35", "a10", "f16"]

# Paramètres de Lissage Temporel (Anti-Flickering / Tracking)
SMOOTHING_ENABLED = True
SMOOTHING_ALPHA = 0.6              # Ratio de lissage (ex: 0.7 = 70% de la détection actuelle + 30% d'historique)
SMOOTHING_MAX_MISSING_FRAMES = 4   # Nombre de frames passées mémorisées (pour pallier un raté de détection)
SMOOTHING_IOU_THRESHOLD = 0.5      # Seuil IoU pour associer la boîte frame T avec frame T-1


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


class TemporalSmoother:
    """Traque et lisse temporellement les bounding boxes (EMA Tracking basique)."""
    def __init__(self, alpha=0.7, max_missing_frames=5, iou_threshold=0.3):
        self.alpha = alpha
        self.max_missing_frames = max_missing_frames
        self.iou_threshold = iou_threshold
        self.trackers = {} # {id: {'box': [x1, y1, x2, y2], 'missing_count': 0, 'score': score}}
        self.next_id = 0

    def update(self, detections):
        """
        Met à jour les trajectoires et retourne les boîtes lissées.
        detections: list de tuples (x1, y1, x2, y2, score)
        """
        smoothed_detections = []
        matched_tracker_ids = set()

        for det in detections:
            x1, y1, x2, y2, score = det
            best_iou = 0
            best_id = -1
            
            # 1. Associer la nouvelle détection avec une trajectoire existante
            for t_id, t_info in self.trackers.items():
                if t_id in matched_tracker_ids:
                     continue
                iou = get_iou(det[:4], t_info['box'])
                if iou > best_iou:
                    best_iou = iou
                    best_id = t_id
            
            if best_iou >= self.iou_threshold:
                # 2a. Si ça correspond au même avion -> Lissage temporel (alpha)
                old_box = self.trackers[best_id]['box']
                nx1 = int(self.alpha * x1 + (1 - self.alpha) * old_box[0])
                ny1 = int(self.alpha * y1 + (1 - self.alpha) * old_box[1])
                nx2 = int(self.alpha * x2 + (1 - self.alpha) * old_box[2])
                ny2 = int(self.alpha * y2 + (1 - self.alpha) * old_box[3])
                
                self.trackers[best_id]['box'] = [nx1, ny1, nx2, ny2]
                self.trackers[best_id]['missing_count'] = 0
                self.trackers[best_id]['score'] = score
                matched_tracker_ids.add(best_id)
                smoothed_detections.append((nx1, ny1, nx2, ny2, score))
            else:
                # 2b. Nouvel avion détecté
                self.trackers[self.next_id] = {
                    'box': [x1, y1, x2, y2],
                    'missing_count': 0,
                    'score': score
                }
                matched_tracker_ids.add(self.next_id)
                smoothed_detections.append((x1, y1, x2, y2, score))
                self.next_id += 1
                
        # 3. Récupérer les "fantômes" (avions non détectés sur CETTE frame mais dans le passé récent)
        for t_id in list(self.trackers.keys()):
            if t_id not in matched_tracker_ids:
                self.trackers[t_id]['missing_count'] += 1
                if self.trackers[t_id]['missing_count'] > self.max_missing_frames:
                    # Vraiment disparu de l'écran -> on oublie
                    del self.trackers[t_id]
                else:
                    # Toujours censé être là -> on rend sa dernière position connue
                    box = self.trackers[t_id]['box']
                    score = self.trackers[t_id]['score']
                    smoothed_detections.append((box[0], box[1], box[2], box[3], score))

        return smoothed_detections


def non_max_suppression(boxes, scores, iou_threshold):
    """Applique NMS sur les boxes filtrées"""
    indices = np.argsort(scores)[::-1]
    keep = []
    
    while indices.size > 0:
        i = indices[0]
        keep.append(i)
        
        if indices.size == 1:
            break
            
        ious = np.array([get_iou(boxes[i], boxes[j]) for j in indices[1:]])
        indices = indices[1:][ious < iou_threshold]
        
    return keep

def decode_grid_and_detect(img_bgr, model, variables, config_model, conf_threshold=0.5, nms_threshold=0.4):
    """
    Exécute la détection sur une image.
    Retourne une liste de boxes [x1, y1, x2, y2, score] (coordonnées absolues).
    """
    h_orig, w_orig = img_bgr.shape[:2]
    
    # 1. Prétraitement
    target_size = config_model.get("image_size", DETECTION_IMAGE_SIZE)
    grayscale = config_model.get("grayscale", True) # Par défaut détection en grayscale
    
    img_resized = cv2.resize(img_bgr, target_size)
    
    if grayscale:
        img_input = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
        img_input = img_input.astype(np.float32) / 255.0
        # (H, W) -> (1, H, W, 1)
        img_jax = img_input[np.newaxis, :, :, np.newaxis]
    else:
        img_input = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        img_input = img_input.astype(np.float32) / 255.0
        # (H, W, 3) -> (1, H, W, 3)
        img_jax = img_input[np.newaxis, :, :, :]
    
    # 2. Inférence
    # Note: On suppose que le modèle de détection rend une grille (ex: 14x14x5) ou un tuple de grilles
    # avec pour chaque cellule: [conf, x, y, w, h] * B_boxes
    preds = model.apply(variables, jnp.array(img_jax), training=False)
    
    if isinstance(preds, (tuple, list)):
        preds_list = [np.array(p[0]) for p in preds] # Extraire l'élément du batch = 0
    else:
        preds_list = [np.array(preds[0])]
    
    boxes = []
    scores = []
    
    # 3. Décodage des grilles
    for pred_grid in preds_list:
        S = pred_grid.shape[0] # Taille de grille 
        C_pred = pred_grid.shape[-1]
        B_boxes = C_pred // 5
        
        pred_grid = pred_grid.reshape((S, S, B_boxes, 5))
        
        for row in range(S):
            for col in range(S):
                for b in range(B_boxes):
                    cell = pred_grid[row, col, b]
                    conf = cell[0]
                    
                    if conf > conf_threshold:
                        # Coordonnées relatives à la cellule (0-1) -> relatives à l'image (0-1)
                        bx = (col + cell[1]) / S
                        by = (row + cell[2]) / S
                        bw = cell[3]
                        bh = cell[4]
                        
                        # Conversion en pixels absolus sur l'image ORIGINALE
                        # x,y sont le centre de la boite
                        center_x = bx * w_orig
                        center_y = by * h_orig
                        width = bw * w_orig
                        height = bh * h_orig
                        
                        x1 = int(center_x - width / 2)
                        y1 = int(center_y - height / 2)
                        x2 = int(center_x + width / 2)
                        y2 = int(center_y + height / 2)
                        
                        # Clipper
                        x1 = max(0, min(x1, w_orig))
                        y1 = max(0, min(y1, h_orig))
                        x2 = max(0, min(x2, w_orig))
                        y2 = max(0, min(y2, h_orig))
                        
                        boxes.append([x1, y1, x2, y2])
                        scores.append(float(conf))
                
    if not boxes:
        return []
        
    # 4. NMS
    boxes_np = np.array(boxes)
    scores_np = np.array(scores)
    
    keep_indices = non_max_suppression(boxes_np, scores_np, nms_threshold)
    
    final_detections = []
    for i in keep_indices:
        x1, y1, x2, y2 = boxes_np[i]
        score = scores_np[i]
        final_detections.append((int(x1), int(y1), int(x2), int(y2), score))
        
    return final_detections

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

    smoother = TemporalSmoother(
        alpha=SMOOTHING_ALPHA,
        max_missing_frames=SMOOTHING_MAX_MISSING_FRAMES,
        iou_threshold=SMOOTHING_IOU_THRESHOLD
    ) if SMOOTHING_ENABLED else None

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
            detections = decode_grid_and_detect(
                frame,
                det_model,
                det_vars,
                det_config,
                conf_threshold=DETECTION_CONF_THRESHOLD,
                nms_threshold=NMS_THRESHOLD
            )

            if SMOOTHING_ENABLED and smoother is not None:
                detections = smoother.update(detections)

            # ==================================================
            # CLASSIFICATION + DRAW
            # ==================================================
            for (x1, y1, x2, y2, det_score) in detections:

                crop = original_frame[y1:y2, x1:x2]
                if crop.size == 0:
                    continue

                predicted_class, confidence = predict_crop(
                    crop,
                    clf_model,
                    clf_vars,
                    dataset_mean,
                    dataset_std,
                    config
                )

                # if (confidence < CONFIDENCE_THRESHOLD):
                #if (predicted_class not in TARGET_CLASS_LIST):
                #    predicted_class = DEFAULT_CLASSE

                # Couleur selon confiance
                #color = (0, 255, 0) if confidence >= CONFIDENCE_THRESHOLD else (0, 0, 255)
                
                # Couleur selon classe OK/KO
                color = (0, 255, 0) if predicted_class in (TARGET_CLASS_LIST) else (0, 0, 255)

                label = f"{predicted_class} ({confidence:.2f})"

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    frame,
                    label,
                    (x1, max(y1 - 10, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2
                )

            # ==================================================
            # SAUVEGARDE IMAGE
            # ==================================================
            output_path = os.path.join(OUTPUT_DIR, f"frame_{frame_id:06d}.jpg")
            cv2.imwrite(output_path, frame)
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
    width = int(cap_original.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap_original.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    cap_original.release()
    
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
    
    image_files = sorted([
        f for f in os.listdir(OUTPUT_DIR)
        if f.endswith(".jpg")
    ])
    
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