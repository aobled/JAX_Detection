from ultralytics import YOLO
import cv2
import os
import json
import shutil
from tqdm import tqdm


# Charger YOLOv8
model = YOLO("yolov8n.pt")

input_dir = "/home/aobled/Downloads/Aircraft_DATASET/test/a_faire/c5"
category_name = 'c5'

# Récupérer la liste de toutes les images
image_files = []
for root, _, files in os.walk(input_dir):
    for file in files:
        if file.lower().endswith((".jpg", ".png", ".jpeg", ".png")):
            image_files.append(os.path.join(root, file))

print(f"Nombre d'images trouvées : {len(image_files)}")

# Parcours avec barre de progression
for img_path in tqdm(image_files, desc="Traitement des images"):
    root = os.path.dirname(img_path)
    file = os.path.basename(img_path)

    # Récupérer la catégorie = nom du dossier parent
    #category_name = os.path.basename(root)

    # Lire l’image pour récupérer largeur/hauteur
    img = cv2.imread(img_path)
    if img is None:
        continue
    h, w = img.shape[:2]

    # Détection
    results = model(img_path, verbose=False)[0]

    json_files = []
    box_id = 0

    for box in results.boxes:
        cls_id = int(box.cls.cpu().numpy())
        label = model.names[cls_id]

        if label == "airplane":
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            bbox = [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]

            data = {
                "image": {
                    "file_name": file,
                    "width": w,
                    "height": h
                },
                "annotation": {
                    "file_name": file,
                    "bbox": bbox,
                    "category_name": category_name,
                    "bbox_id": box_id
                }
            }

            base_name = os.path.splitext(file)[0]
            out_name = f"{base_name}_{box_id}.json"
            out_path = os.path.join(root, out_name)

            with open(out_path, "w") as f:
                json.dump(data, f, indent=4)

            json_files.append(out_path)
            box_id += 1

    # === Étape finale : déplacement selon le nombre de boxes ===
    dest_dir = os.path.join(input_dir, f"{box_id:02d}")
    os.makedirs(dest_dir, exist_ok=True)

    # Déplacer l’image
    shutil.move(img_path, os.path.join(dest_dir, file))

    # Déplacer les jsons (seulement si box_id > 0)
    for jf in json_files:
        shutil.move(jf, os.path.join(dest_dir, os.path.basename(jf)))
