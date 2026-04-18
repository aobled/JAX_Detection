import os
import json
import shutil
import tqdm
from PIL import Image

# 📁 INPUT
BASE_DIR = "/home/aobled/Downloads/tmp_a_traiter/archive"
IMAGES_DIR = os.path.join(BASE_DIR, "images")
LABELS_DIR = os.path.join(BASE_DIR, "labels")
CLASSES_PATH = os.path.join(BASE_DIR, "classes.txt")

# 📁 OUTPUT
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 🔹 Fonction fournie
def create_json_annotation(image_name, bbox, bbox_id, img_width, img_height, category_name):
    return {
        "image": {
            "file_name": image_name,
            "width": img_width,
            "height": img_height
        },
        "annotation": {
            "file_name": image_name,
            "bbox": bbox,
            "category_name": category_name,
            "bbox_id": bbox_id
        }
    }

# 🔹 Charger les classes
with open(CLASSES_PATH, "r") as f:
    classes = [c.strip() for c in f.read().split(",")]

print(f"📊 {len(classes)} classes chargées")

# 🔹 Parcours dataset
for split in ["train", "test", "validation"]:
    img_dir = os.path.join(IMAGES_DIR, split)
    lbl_dir = os.path.join(LABELS_DIR, split)

    for file in tqdm.tqdm(os.listdir(img_dir), desc=f"Processing {split}"):
        if not file.endswith(".jpg"):
            continue

        base_name = os.path.splitext(file)[0]
        img_path = os.path.join(img_dir, file)
        lbl_path = os.path.join(lbl_dir, base_name + ".txt")

        if not os.path.exists(lbl_path):
            print(f"⚠️ Pas de label pour {file}")
            continue

        # 🔹 Lire image (pour conversion pixels)
        img = Image.open(img_path)
        W, H = img.size

        classes_in_image = set()
        json_files = []

        with open(lbl_path, "r") as f:
            lines = f.readlines()

        for idx, line in enumerate(lines):
            parts = line.strip().split()
            if len(parts) != 5:
                continue

            class_id = int(parts[0])
            x_center = float(parts[1])
            y_center = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])

            class_name = classes[class_id]
            classes_in_image.add(class_name)

            # 🔥 Conversion YOLO → pixels
            xmin = int((x_center - width / 2) * W)
            ymin = int((y_center - height / 2) * H)
            w = int(width * W)
            h = int(height * H)

            bbox = [xmin, ymin, w, h]

            json_data = create_json_annotation(
                image_name=file,
                bbox=bbox,
                bbox_id=idx,
                img_width=W,
                img_height=H,
                category_name=class_name
            )

            json_filename = f"{base_name}_{idx}.json"
            json_path = os.path.join(BASE_DIR, json_filename)

            with open(json_path, "w") as jf:
                json.dump(json_data, jf, indent=4)

            json_files.append(json_path)

        # 🔹 Organisation des dossiers
        if len(classes_in_image) == 1:
            target_class = list(classes_in_image)[0]
            target_dir = os.path.join(OUTPUT_DIR, target_class)
        else:
            target_dir = os.path.join(OUTPUT_DIR, "_multi")

        os.makedirs(target_dir, exist_ok=True)

        # 🔹 Copier (SAFE)
        shutil.copy(img_path, os.path.join(target_dir, file))

        for jf in json_files:
            shutil.copy(jf, os.path.join(target_dir, os.path.basename(jf)))

        print(f"✅ {file} -> {len(lines)} boxes -> {target_dir}")