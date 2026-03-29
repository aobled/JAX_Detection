import os
import cv2
import numpy as np

def detect_black_or_white_bands(root_dir, height_detection=10):
    """
    Parcourt tous les sous-dossiers et détecte les images ayant
    en bas à gauche ET à droite un rectangle noir ou blanc.

    Args:
        root_dir (str): dossier racine contenant les images (et sous-dossiers)
        height_detection (int): hauteur du rectangle à vérifier (en pixels)
    """

    # Extensions d'image valides
    valid_exts = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

    for subdir, _, files in os.walk(root_dir):
        for file in files:
            if not file.lower().endswith(valid_exts):
                continue

            path = os.path.join(subdir, file)
            img = cv2.imread(path)

            if img is None:
                continue

            h, w, _ = img.shape
            rect_height = min(height_detection, h)
            rect_width = w // 4  # un quart de la largeur

            # Coordonnées des rectangles
            bottom_y1 = h - rect_height
            left_x1 = 0
            right_x1 = w - rect_width

            # Extraire les régions à gauche et à droite
            left_rect = img[bottom_y1:h, left_x1:left_x1 + rect_width]
            right_rect = img[bottom_y1:h, right_x1:w]

            # Vérifier si la région est entièrement noire ou blanche
            def is_black_or_white(region, tol=5):
                mean_color = np.mean(region)
                return mean_color < tol or mean_color > 255 - tol

            import shutil
            import glob
            
            if is_black_or_white(left_rect) and is_black_or_white(right_rect):
                print(f"→ Image suspecte : {file}")
            
                # Création du dossier "errors" s'il n'existe pas
                error_dir = os.path.join(subdir, "errors")
                os.makedirs(error_dir, exist_ok=True)
            
                # Déplacer l'image suspecte
                src_img = os.path.join(subdir, file)
                dst_img = os.path.join(error_dir, file)
                shutil.move(src_img, dst_img)
            
                # Déplacer les fichiers JSON associés
                base_name, _ = os.path.splitext(file)
                json_pattern = os.path.join(subdir, f"{base_name}*.json")
            
                for json_file in glob.glob(json_pattern):
                    dst_json = os.path.join(error_dir, os.path.basename(json_file))
                    shutil.move(json_file, dst_json)



# Exemple d'utilisation :
if __name__ == "__main__":
    dossier_images = "/home/aobled/Downloads/Figtherjet_DATASET/"  # 🔁 à modifier
    detect_black_or_white_bands(dossier_images, height_detection=10)
