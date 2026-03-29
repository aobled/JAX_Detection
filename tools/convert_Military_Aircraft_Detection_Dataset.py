import os
import csv
import json
import shutil
import tqdm

# Répertoire contenant les images .jpg et les fichiers .csv
INPUT_DIR = "/home/aobled/Downloads/Military Aircraft Detection Dataset/dataset"
# Répertoire de sortie pour organiser les images
OUTPUT_DIR = "/home/aobled/Downloads/Military Aircraft Detection Dataset/dataset/output"

os.makedirs(OUTPUT_DIR, exist_ok=True)

for file in tqdm.tqdm(os.listdir(INPUT_DIR)):
    if file.endswith(".jpg"):
        base_name = os.path.splitext(file)[0]
        jpg_path = os.path.join(INPUT_DIR, file)
        csv_path = os.path.join(INPUT_DIR, base_name + ".csv")

        if not os.path.exists(csv_path):
            print(f"⚠️ Pas de CSV pour {file}, ignoré.")
            continue

        classes_in_csv = set()
        json_files_to_move = []

        with open(csv_path, newline='', encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

            for idx, row in enumerate(rows):
                classes_in_csv.add(row["class"])

                json_data = {
                    "image": {
                        "file_name": row["filename"] + ".jpg",
                        "width": int(row["width"]),
                        "height": int(row["height"])
                    },
                    "annotation": {
                        "file_name": row["filename"] + ".jpg",
                        "bbox": [
                            int(row["xmin"]),
                            int(row["ymin"]),
                            int(row["xmax"]) - int(row["xmin"]),
                            int(row["ymax"]) - int(row["ymin"])
                        ],
                        "category_name": row["class"],
                        "bbox_id": idx
                    }
                }

                json_filename = f"{base_name}_{idx}.json"
                json_path = os.path.join(INPUT_DIR, json_filename)

                with open(json_path, "w", encoding="utf-8") as jf:
                    json.dump(json_data, jf, indent=4, ensure_ascii=False)

                json_files_to_move.append(json_path)

        # Détermination du dossier cible selon les classes
        if len(classes_in_csv) == 1:
            class_name = list(classes_in_csv)[0]
            target_dir = os.path.join(OUTPUT_DIR, class_name)
        else:
            target_dir = os.path.join(OUTPUT_DIR, "_multi")

        os.makedirs(target_dir, exist_ok=True)

        # Déplacer l'image
        shutil.move(jpg_path, os.path.join(target_dir, file))
        # Déplacer les JSON associés
        for jf in json_files_to_move:
            shutil.move(jf, os.path.join(target_dir, os.path.basename(jf)))

        print(f"✅ Traité {file}, {len(rows)} boxes -> JSON générés et déplacés dans {target_dir}")
