import sys
import os
import cv2
import numpy as np
import jax
import jax.numpy as jnp
import pickle
from tqdm import tqdm

# Ajouter le répertoire parent en PRIORITÉ absolue
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataset_configs import get_dataset_config
from model_library import get_model


# =================================================================================================
# Input CONFIGURATION
# =================================================================================================
DATASET_NAME = "FIGHTERJET_CLASSIFICATION"     # Nom de la config de CLASSIFICATION
CHECKPOINT_PATH = "best_model.pkl"      # Chemin vers le modèle de CLASSIFICATION

DETECTION_CHECKPOINT_PATH = "best_model_detection.pkl" # Chemin vers le modèle de DÉTECTION
DETECTION_IMAGE_SIZE = (224, 224)       # Taille d'entrée du modèle de détection

# ==========================================================
# Benchmark CONFIGURATION
# ==========================================================
IMAGE_PATH = "/home/aobled/Downloads/test_image.png"  # ⚠️ REMPLACER PAR LE CHEMIN DE L'IMAGE TEST
OUTPUT_DIR = "/home/aobled/Downloads/benchmark_results"

TARGET_CLASS_LIST = ["b2", "b52", "b1b", "f15", "f22", "a10", "f16"]
DEFAULT_CLASSE = "unknown"

# =================================================================================================
# FONCTIONS UTILITAIRES POUR JAX (COPIÉES DE LA VIDÉO)
# =================================================================================================
def load_jax_model(checkpoint_path, config):
    if not os.path.exists(checkpoint_path):
        parent_checkpoint = os.path.join(os.path.dirname(os.getcwd()), checkpoint_path)
        if os.path.exists(parent_checkpoint):
            checkpoint_path = parent_checkpoint
        else:
            raise FileNotFoundError(f"Checkpoint non trouvé: {checkpoint_path}")

    print(f"🔍 Chargement du modèle CLASSIFICATION depuis {checkpoint_path}...")
    with open(checkpoint_path, 'rb') as f:
        model_data = pickle.load(f)

    if 'model_state' in model_data:
        params = model_data['model_state']['params']
        batch_stats = model_data['model_state'].get('batch_stats', {})
    else:
        params = model_data['params']
        batch_stats = model_data.get('batch_stats', {})
    
    model_name = config.get("model_name", "sophisticated_cnn_128_plus")
    num_classes = config["num_classes"]
    model = get_model(model_name, num_classes=num_classes, dropout_rate=0.0)
    variables = {'params': params, 'batch_stats': batch_stats}
    
    mean_std_path = config.get("mean_std_path", "./data/chunks/dataset_chunked_meanstd.npz")
    if os.path.exists(mean_std_path):
        with np.load(mean_std_path) as data:
            mean = data['mean']
            std = data['std']
            if config.get("grayscale", False) and isinstance(mean, np.ndarray) and mean.size == 3:
                 mean = np.mean(mean)
                 std = np.mean(std)
    else:
        mean, std = 0.5, 0.5

    return model, variables, mean, std

def predict_crop(crop_img, model, variables, mean, std, config):
    target_size = config["image_size"]
    grayscale = config.get("grayscale", False)
    crop_resized = cv2.resize(crop_img, target_size)
    
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

    logits = model.apply(variables, jnp.array(img_jax), training=False)
    probs = jax.nn.softmax(logits, axis=-1)
    
    pred_idx = int(jnp.argmax(probs))
    confidence = float(probs[0, pred_idx])
    return config["class_names"][pred_idx], confidence

def load_detection_model(checkpoint_path):
    if not os.path.exists(checkpoint_path):
        parent_checkpoint = os.path.join(os.path.dirname(os.getcwd()), checkpoint_path)
        if os.path.exists(parent_checkpoint):
            checkpoint_path = parent_checkpoint
        else:
            raise FileNotFoundError(f"Checkpoint détection non trouvé: {checkpoint_path}")

    print(f"🔍 Chargement du modèle DÉTECTION depuis {checkpoint_path}...")
    with open(checkpoint_path, 'rb') as f:
        data_model = pickle.load(f)

    params = data_model['params']
    config_model = data_model.get('config', {})
    model_name = config_model.get('model_name', 'aircraft_detector_v7_advanced')
    model = get_model(model_name, dropout_rate=0.0) 
    
    batch_stats = data_model.get('batch_stats', {})
    if not batch_stats and 'model_state' in data_model:
         batch_stats = data_model['model_state'].get('batch_stats', {})
    if not batch_stats:
        rng = jax.random.PRNGKey(0)
        target_size = list(config_model.get("image_size", DETECTION_IMAGE_SIZE))
        channels = 1 if config_model.get("grayscale", True) else 3
        dummy_input = jnp.ones((1, target_size[0], target_size[1], channels), jnp.float32)
        init_variables = model.init(rng, dummy_input, training=True)
        batch_stats = init_variables.get('batch_stats', {})

    variables = {'params': params, 'batch_stats': batch_stats}
    return model, variables, config_model

def get_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    return intersection / union if union > 0 else 0

def non_max_suppression(boxes, scores, iou_threshold):
    indices = np.argsort(scores)[::-1]
    keep = []
    while indices.size > 0:
        i = indices[0]
        keep.append(i)
        if indices.size == 1: break
        ious = np.array([get_iou(boxes[i], boxes[j]) for j in indices[1:]])
        indices = indices[1:][ious < iou_threshold]
    return keep

def decode_grid_and_detect(img_bgr, model, variables, config_model, conf_threshold=0.5, nms_threshold=0.4):
    h_orig, w_orig = img_bgr.shape[:2]
    
    target_size = tuple(config_model.get("image_size", DETECTION_IMAGE_SIZE))
    grayscale = config_model.get("grayscale", True)
    
    img_resized = cv2.resize(img_bgr, target_size)
    if grayscale:
        img_input = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        img_jax = img_input[np.newaxis, :, :, np.newaxis]
    else:
        img_input = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img_jax = img_input[np.newaxis, :, :, :]
    
    preds = model.apply(variables, jnp.array(img_jax), training=False)
    preds_list = [np.array(p[0]) for p in preds] if isinstance(preds, (tuple, list)) else [np.array(preds[0])]
    
    boxes, scores = [], []
    for pred_grid in preds_list:
        S = pred_grid.shape[0]
        C_pred = pred_grid.shape[-1]
        B_boxes = C_pred // 5
        pred_grid = pred_grid.reshape((S, S, B_boxes, 5))
        
        for row in range(S):
            for col in range(S):
                for b in range(B_boxes):
                    cell = pred_grid[row, col, b]
                    conf = cell[0]
                    if conf > conf_threshold:
                        bx, by, bw, bh = (col + cell[1]) / S, (row + cell[2]) / S, cell[3], cell[4]
                        center_x, center_y = bx * w_orig, by * h_orig
                        width, height = bw * w_orig, bh * h_orig
                        
                        x1 = max(0, min(int(center_x - width / 2), w_orig))
                        y1 = max(0, min(int(center_y - height / 2), h_orig))
                        x2 = max(0, min(int(center_x + width / 2), w_orig))
                        y2 = max(0, min(int(center_y + height / 2), h_orig))
                        
                        boxes.append([x1, y1, x2, y2])
                        scores.append(float(conf))
                        
    if not boxes: return []
    
    boxes_np = np.array(boxes)
    scores_np = np.array(scores)
    keep_indices = non_max_suppression(boxes_np, scores_np, nms_threshold)
    
    return [(int(boxes_np[i][0]), int(boxes_np[i][1]), int(boxes_np[i][2]), int(boxes_np[i][3]), scores_np[i]) for i in keep_indices]

# =================================================================================================
# MAIN BENCHMARK LOOP
# =================================================================================================
if __name__ == "__main__":
    
    # 1. Chargements
    print("🏗️ Chargement de la configuration...")
    config = get_dataset_config(DATASET_NAME)
    
    print("🏗️ Chargement des modèles...")
    det_model, det_vars, det_config = load_detection_model(DETECTION_CHECKPOINT_PATH)
    clf_model, clf_vars, dataset_mean, dataset_std = load_jax_model(CHECKPOINT_PATH, config)
    
    print("✅ Initialisation OK.")
    
    # 2. Vérification Image
    if not os.path.exists(IMAGE_PATH):
        print(f"❌ Erreur: Image introuvable {IMAGE_PATH}")
        print("💡 N'oublie pas de définir IMAGE_PATH avec une image pertinente existante dans le script.")
        sys.exit(1)
        
    print(f"🖼️ Chargement de l'image de test: {IMAGE_PATH}")
    img_bgr = cv2.imread(IMAGE_PATH)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 3. Boucles de test
    conf_thresholds = np.arange(0.1, 1.05, 0.1)
    nms_thresholds = np.arange(0.1, 1.05, 0.1)
    
    total_iterations = len(conf_thresholds) * len(nms_thresholds)
    print(f"🚀 Lancement du benchmark sur {total_iterations} combinaisons (Conf from 0.0 to 1.0, NMS from 0.0 to 1.0)...")
    
    with tqdm(total=total_iterations, desc="Génération Benchmark") as pbar:
        for conf_t in conf_thresholds:
            for nms_t in nms_thresholds:
                
                # A. Sécurité pour les virgules flottantes (ex: 0.100000001)
                conf_t = round(conf_t, 1)
                nms_t = round(nms_t, 1)
                
                # Copier l'image fraîche pour chaque test
                frame = img_bgr.copy()
                
                # B. Lancer la détection avec les seuils spécifiques
                detections = decode_grid_and_detect(
                    frame,
                    det_model,
                    det_vars,
                    det_config,
                    conf_threshold=conf_t,
                    nms_threshold=nms_t
                )
                
                # C. Classifier les box trouvées et les dessiner
                for (x1, y1, x2, y2, det_score) in detections:
                    crop = img_bgr[y1:y2, x1:x2]
                    if crop.size == 0:
                        continue
                        
                    predicted_class, class_conf = predict_crop(
                        crop,
                        clf_model,
                        clf_vars,
                        dataset_mean,
                        dataset_std,
                        config
                    )
                    
                    color = (0, 255, 0) if predicted_class in TARGET_CLASS_LIST else (0, 0, 255)
                    label = f"{predicted_class} (Det:{det_score:.2f} Cls:{class_conf:.2f})"
                    
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, label, (x1, max(y1 - 10, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                # Ajouter un gros texte explicatif sur l'image entière
                info_text = f"CONF: {conf_t:.1f} | NMS: {nms_t:.1f} | Detections: {len(detections)}"
                cv2.putText(frame, info_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 3)
                
                # D. Sauvegarde de la frame annotée
                filename = f"DCT{int(conf_t*10):02d}_NT{int(nms_t*10):02d}.png"
                output_path = os.path.join(OUTPUT_DIR, filename)
                cv2.imwrite(output_path, frame)
                
                pbar.update(1)

    print(f"\n✅ Benchmark terminé avec succès ! Les {total_iterations} images sont dans {OUTPUT_DIR}")
