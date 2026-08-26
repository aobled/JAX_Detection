
import sys
import os

# Ajouter le répertoire parent en PRIORITÉ absolue (index 0) pour forcer Python
# à utiliser le model_library.py de jax_supervised_training (et non celui de JAX_Classification si exécuté depuis l'autre dossier)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Remplace YOLO import par JAX/Flax helpers si nécessaire
# from ultralytics import YOLO  <-- REMOVED
import cv2
import json
import shutil
import numpy as np
import jax.numpy as jnp
from tqdm import tqdm

from dataset_configs import get_dataset_config
from inference_utils import (
    build_single_pass_predict_fn, _rescale_boxes,
    load_detection_model, load_jax_model, decode_segmentation_and_detect, predict_crop,
)

# =================================================================================================
# CONFIGURATION
# =================================================================================================
# 1. Configuration du dataset et du modèle de classification
DATASET_NAME = "FIGHTERJET_CLASSIFICATION"     # Nom de la config dans dataset_configs.py
CHECKPOINT_PATH = "best_model.pkl"      # Chemin vers le modèle de CLASSIFICATION
INPUT_DIR = "/home/aobled/Downloads/tmp_nara/undetected"  # Dossier d'entrée (images à traiter)
CLASSIFICATION_CONFIDENCE_THRESHOLD = 0.5            # Seuil de confiance pour valider une CLASSIFICATION (0.0-1.0)

# 2. Backend de détection - rétrocompatibilité (2026-07-19, retour utilisateur : JAX_DETECTOR
# se comporte moins bien que l'ancien pipeline en pratique sur ce script). Deux choix :
#   "JAX_DETECTOR"         -> pipeline Single-Pass (CenterNet, Story 8.6), best_model_jax_detector.pkl
#   "FIGHTERJET_DETECTION" -> ancien pipeline (UNet+segmentation, AD-20, jamais modifié) -
#                             ratifié pour CE script par decode_segmentation_and_detect/
#                             predict_crop (voir leurs docstrings dans inference_utils.py,
#                             qui citent explicitement ce fichier comme consommateur).
DETECTOR_BACKEND = "JAX_DETECTOR"  # "JAX_DETECTOR" ou "FIGHTERJET_DETECTION"

# Checkpoint du détecteur - dépend du backend choisi ci-dessus.
DETECTOR_CHECKPOINT_PATH = (
    "best_model_jax_detector.pkl" if DETECTOR_BACKEND == "JAX_DETECTOR" else "best_model_detection.pkl"
)

# Repère canonique attendu par build_single_pass_predict_fn (AD-12) - uniquement utilisé
# par le backend JAX_DETECTOR ; les images d'entrée de ce script sont de résolution
# arbitraire, jamais garanties 1920x1080.
CANONICAL_WIDTH, CANONICAL_HEIGHT = 1920, 1080

# Seuils propres à l'ancien pipeline (FIGHTERJET_DETECTION) - mêmes valeurs par défaut que
# tools/audit_dataset_detection.py (AD-20, consommateur de référence). Sans équivalent
# direct côté JAX_DETECTOR, qui dérive son propre seuil depuis detection_score_threshold
# (Story 8.3). Note (mise à jour 2026-07-22) : JAX_DETECTOR a désormais aussi un NMS
# JAX-natif (`nms_iou_threshold`, voir ci-dessous) — AD-9 reste vrai pour un objet de
# taille normale (pic étroit, pas de doublon), mais plus pour un objet plein-cadre
# (voir deferred-work.md, `_gaussian_radius` produit un plateau large sur les grands
# objets, source de doublons que le NMS nettoie).
DETECTION_CONF_THRESHOLD = 0.05
BOX_AERA_MIN = 400

# Seuil IoU du NMS JAX-natif JAX_DETECTOR (même source que build_single_pass_predict_fn
# en interne, 2026-07-22) - récupéré ici uniquement pour être passé explicitement,
# sans dupliquer sa valeur en dur (AD-15).
NMS_IOU_THRESHOLD = get_dataset_config("JAX_DETECTOR")["nms_iou_threshold"]

DEFAULT_CLASSE = "unknown"

# 3. Chargement de la config dataset
try:
    config = get_dataset_config(DATASET_NAME)
    CLASS_NAMES = config["class_names"]
    print(f"✅ Configuration chargée: {DATASET_NAME}")
    print(f"📊 Classes ({len(CLASS_NAMES)}): {CLASS_NAMES}")
    print(f"🔒 Seuil de confiance (Classification): {CLASSIFICATION_CONFIDENCE_THRESHOLD * 100}%")
except Exception as e:
    print(f"❌ Erreur chargement config: {e}")
    sys.exit(1)


# =================================================================================================
# MAIN
# =================================================================================================

if __name__ == "__main__":
    # 1. Chargement des modèles - deux chemins possibles selon DETECTOR_BACKEND. Dans les
    # deux cas on construit un unique callable detect_and_classify(img_bgr) -> liste de
    # detections uniformes {box, detection_score, predicted_class, confidence}, pour garder
    # la boucle de traitement ci-dessous identique quel que soit le backend choisi.
    print(f"\n🏗️  Initialisation (backend détection: {DETECTOR_BACKEND})...")

    try:
        if DETECTOR_BACKEND == "JAX_DETECTOR":
            predict_fn = build_single_pass_predict_fn(
                detector_checkpoint_path=DETECTOR_CHECKPOINT_PATH,
                classifier_checkpoint_path=CHECKPOINT_PATH,
                nms_iou_threshold=NMS_IOU_THRESHOLD,
            )
            print("✅ Modèles DÉTECTION+CLASSIFICATION (Single-Pass) chargés.")

            def detect_and_classify(img_bgr):
                h, w = img_bgr.shape[:2]
                img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
                img_canonical = cv2.resize(img_gray, (CANONICAL_WIDTH, CANONICAL_HEIGHT))
                img_canonical = img_canonical.astype(np.float32)[..., None]

                result = predict_fn(jnp.asarray(img_canonical))

                boxes_native = _rescale_boxes(
                    result["boxes"], detector_size=(CANONICAL_WIDTH, CANONICAL_HEIGHT), original_size=(w, h)
                )
                boxes_native = np.asarray(boxes_native)
                valid_mask = np.asarray(result["valid_mask"])
                classes = np.asarray(result["classes"])
                class_scores = np.asarray(result["class_scores"])
                detection_scores = np.asarray(result["detection_scores"])

                detections = []
                for i in np.where(valid_mask)[0]:  # valid_mask seule autorité (AD-15/AC3 8.6)
                    x1, y1, x2, y2 = boxes_native[i]
                    detections.append({
                        "box": (float(x1), float(y1), float(x2), float(y2)),
                        "detection_score": float(detection_scores[i]),
                        "predicted_class": CLASS_NAMES[int(classes[i])],
                        "confidence": float(class_scores[i]),
                    })
                return detections

        elif DETECTOR_BACKEND == "FIGHTERJET_DETECTION":
            det_model, det_vars, det_config = load_detection_model(DETECTOR_CHECKPOINT_PATH)
            clf_model, clf_vars, clf_mean, clf_std = load_jax_model(CHECKPOINT_PATH, config)
            print("✅ Modèles DÉTECTION (UNet) + CLASSIFICATION chargés (ancien pipeline, AD-20).")

            def detect_and_classify(img_bgr):
                # decode_segmentation_and_detect retourne deja les boites en pleine
                # resolution source (pas de rescale separe necessaire, contrairement au
                # backend JAX_DETECTOR).
                raw_detections = decode_segmentation_and_detect(
                    img_bgr, det_model, det_vars, det_config,
                    conf_threshold=DETECTION_CONF_THRESHOLD, box_aera_min=BOX_AERA_MIN,
                )
                detections = []
                for x1, y1, x2, y2, score in raw_detections:
                    crop_img = img_bgr[int(y1):int(y2), int(x1):int(x2)]
                    if crop_img.size == 0:
                        continue
                    predicted_class, confidence = predict_crop(
                        crop_img, clf_model, clf_vars, clf_mean, clf_std, config
                    )
                    detections.append({
                        "box": (float(x1), float(y1), float(x2), float(y2)),
                        "detection_score": float(score),
                        "predicted_class": predicted_class,
                        "confidence": float(confidence),
                    })
                return detections

        else:
            raise ValueError(
                f"DETECTOR_BACKEND inconnu: {DETECTOR_BACKEND!r} "
                f"(attendu 'JAX_DETECTOR' ou 'FIGHTERJET_DETECTION')"
            )
    except Exception as e:
        print(f"❌ Erreur chargement des modèles: {e}")
        import traceback
        traceback.print_exc()
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

        # --- DÉTECTION + CLASSIFICATION (backend choisi via DETECTOR_BACKEND) ---
        detections = detect_and_classify(img)

        json_files_created = []
        detected_classes = set()
        planes_in_image = 0

        for det in detections:
            x1, y1, x2, y2 = det["box"]
            score = det["detection_score"]
            predicted_class = det["predicted_class"]
            confidence = det["confidence"]

            # Filtrage par confiance
            if confidence < CLASSIFICATION_CONFIDENCE_THRESHOLD:
                predicted_class = DEFAULT_CLASSE

            detected_classes.add(predicted_class)
            # ------------------------------

            # Préparer JSON (boîtes déjà en résolution source)
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
                    "detection_score": score, # Ajout du score de détection pour info
                    "classification_score": confidence
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
