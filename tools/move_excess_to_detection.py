import os
import glob
import random
import shutil
import tqdm

# ==========================================================
# CONFIGURATION
# ==========================================================
INPUT_DIR = "/home/aobled/Downloads/Aircraft_DATASET/classification/train/typhoon"
OUTPUT_DIR = "/home/aobled/Downloads/Aircraft_DATASET/detection/train/typhoon/single/"
IMAGES_NUMBER_THRESHOLD = 4000

# ==========================================================

def move_excess_images():
    print(f"📁 Dossier source : {INPUT_DIR}")
    print(f"📁 Dossier cible  : {OUTPUT_DIR}")
    print(f"🎯 Limite à conserver : {IMAGES_NUMBER_THRESHOLD} images\n")

    if not os.path.exists(INPUT_DIR):
        print(f"❌ Le dossier source n'existe pas : {INPUT_DIR}")
        return

    # 1. Lister toutes les images (.jpg et .png)
    image_extensions = ('*.jpg', '*.jpeg', '*.png')
    all_images = []
    for ext in tqdm.tqdm(image_extensions):
        all_images.extend(glob.glob(os.path.join(INPUT_DIR, ext)))

    total_images = len(all_images)
    print(f"📸 Nombre total d'images trouvées : {total_images}")

    # 2. Vérifier si un déplacement est nécessaire
    if total_images <= IMAGES_NUMBER_THRESHOLD:
        print("✅ Le nombre d'images est déjà sous le seuil. Aucun déplacement nécessaire.")
        return

    # 3. Séparer les images à garder et à déplacer
    random.shuffle(all_images)
    images_to_keep = all_images[:IMAGES_NUMBER_THRESHOLD]
    images_to_move = all_images[IMAGES_NUMBER_THRESHOLD:]

    print(f"📦 Images à conserver : {len(images_to_keep)}")
    print(f"🚚 Images à déplacer : {len(images_to_move)}")

    # Créer le dossier de sortie si nécessaire
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    moved_images_count = 0
    moved_json_count = 0

    # 4. Déplacer les images et leurs fichiers JSON associés
    for img_path in tqdm.tqdm(images_to_move):
        # Déplacer l'image
        img_name = os.path.basename(img_path)
        dest_img_path = os.path.join(OUTPUT_DIR, img_name)
        shutil.move(img_path, dest_img_path)
        moved_images_count += 1

        # Trouver et déplacer les JSON associés (ex: nomimage_0.json, nomimage_1.json, etc.)
        base_name = os.path.splitext(img_name)[0]
        json_pattern = os.path.join(INPUT_DIR, f"{base_name}*.json")
        json_files = glob.glob(json_pattern)

        for json_path in json_files:
            json_name = os.path.basename(json_path)
            dest_json_path = os.path.join(OUTPUT_DIR, json_name)
            shutil.move(json_path, dest_json_path)
            moved_json_count += 1

    print("\n✅ Opération terminée !")
    print(f"   -> {moved_images_count} images déplacées.")
    print(f"   -> {moved_json_count} fichiers JSON déplacés.")

if __name__ == "__main__":
    move_excess_images()
