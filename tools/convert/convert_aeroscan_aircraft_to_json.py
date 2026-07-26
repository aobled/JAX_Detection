#!/usr/bin/env python3
"""
Convertisseur AeroScan (YOLO Format) vers format JSON personnalisé
Convertit le dataset AeroScan (88 classes d'avions militaires) vers le format JSON utilisé par ton outil d'annotation.
"""

import os
import json
from pathlib import Path
from PIL import Image
import tqdm

# ===== CONFIGURATION =====
ORIGIN_DIRECTORY = "/home/aobled/Downloads/tmp_test"  # Répertoire contenant images + annotations YOLO (.txt)
TARGET_DIRECTORY = "/home/aobled/Downloads/tmp_test/json"    # Répertoire de sortie pour les JSON

# Liste des 88 classes d'AeroScan (ordre = class_id)
AEROSCAN_CLASSES = [
    'a10', 'a400m', 'ag600', 'ah64', 'akinci', 'av8b', 'an124', 'an22', 'an225', 'an72',
    'b1', 'b2', 'b52', 'be200', 'c1', 'c130', 'c17', 'c2', 'c390', 'c5',
    'ch47', 'ch53', 'cl415', 'e2', 'e7', 'ef2000', 'emb314', 'f117', 'f14', 'f15',
    'f16', 'f18', 'f2', 'f22', 'f35', 'f4', 'fck1', 'h6', 'il76', 'j10',
    'j20', 'j35', 'j36', 'jas39', 'jf17', 'jh7', 'kaan', 'kc135', 'kf21', 'kj600',
    'ka27', 'ka52', 'mq9', 'mi24', 'mi26', 'mi28', 'mi8', 'mig29', 'mig31', 'mirage2000',
    'p3', 'rq4', 'rafale', 'sr71', 'su24', 'su25', 'su34', 'su47', 'su57',
    'tb001', 'tb2', 'tejas', 'tornado', 'tu160', 'tu22m', 'tu95', 'u2', 'uh60',
    'us2', 'v22', 'vulcan', 'wz7', 'x32', 'xb70', 'y20', 'yf23', 'z10', 'z19'
]

def get_image_dimensions(image_path):
    """Obtient les dimensions d'une image"""
    try:
        with Image.open(image_path) as img:
            return img.size  # (width, height)
    except Exception as e:
        print(f"⚠️ Erreur lors de la lecture de {image_path}: {e}")
        return None, None

def convert_yolo_to_bbox(yolo_coords, img_width, img_height):
    """
    Convertit les coordonnées YOLO (normalisées) vers le format bbox [x, y, w, h] en pixels absolus.
    YOLO format: class_id x_center y_center width height (tous normalisés 0-1)
    Notre format: [x, y, w, h] (coin supérieur gauche + dimensions)
    """
    class_id, x_center, y_center, width, height = yolo_coords

    # Conversion vers pixels absolus
    x_center_px = x_center * img_width
    y_center_px = y_center * img_height
    width_px = width * img_width
    height_px = height * img_height

    # Conversion vers format [x, y, w, h] (coin supérieur gauche)
    x = x_center_px - (width_px / 2)
    y = y_center_px - (height_px / 2)

    return int(class_id), [x, y, width_px, height_px]  # Retourne aussi le class_id

def create_json_annotation(image_name, bbox, bbox_id, img_width, img_height, class_name):
    """
    Crée une annotation JSON avec la classe réelle (pas "unknown").
    """
    return {
        "image": {
            "file_name": image_name,
            "width": img_width,
            "height": img_height
        },
        "annotation": {
            "file_name": image_name,
            "bbox": bbox,
            "category_name": class_name,  # <-- Utilise le nom de classe réel
            "bbox_id": bbox_id
        }
    }

def process_aeroscan_dataset():
    """Traite le dataset AeroScan et le convertit vers le format JSON"""
    print(f"🚀 Début de la conversion AeroScan vers JSON")
    print(f"📁 Répertoire source: {ORIGIN_DIRECTORY}")
    print(f"📁 Répertoire cible: {TARGET_DIRECTORY}")

    # Vérifier que le répertoire source existe
    if not os.path.exists(ORIGIN_DIRECTORY):
        print(f"❌ Erreur: Le répertoire source {ORIGIN_DIRECTORY} n'existe pas!")
        return

    # Créer le répertoire cible
    os.makedirs(TARGET_DIRECTORY, exist_ok=True)

    # Compteurs
    images_processed = 0
    total_annotations = 0
    skipped_images = 0  # Images sans annotation (fond)

    # Parcourir les images
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp')
    for file in tqdm.tqdm(os.listdir(ORIGIN_DIRECTORY)):
        if file.lower().endswith(image_extensions):
            image_path = os.path.join(ORIGIN_DIRECTORY, file)
            image_name = os.path.basename(file)
            base_name = os.path.splitext(image_name)[0]

            # Fichier d'annotation YOLO correspondant
            annotation_file = os.path.join(ORIGIN_DIRECTORY, f"{base_name}.txt")

            # Vérifier si le fichier d'annotation existe
            if not os.path.exists(annotation_file):
                print(f"⚠️  Aucune annotation trouvée pour {image_name} (image de fond ?)")
                skipped_images += 1
                continue

            # Lire les dimensions de l'image
            img_width, img_height = get_image_dimensions(image_path)
            if img_width is None or img_height is None:
                print(f"⚠️  Impossible de lire les dimensions de {image_name}, ignoré")
                continue

            # Lire les annotations YOLO
            with open(annotation_file, 'r') as f:
                lines = [line.strip() for line in f.readlines() if line.strip()]

            # Si pas d'annotation (fichier vide = image de fond)
            if not lines:
                print(f"⚠️  {image_name} est une image de fond (pas d'avion), ignorée")
                skipped_images += 1
                continue

            # Traiter chaque annotation
            for bbox_id, line in enumerate(lines):
                try:
                    # Parser la ligne YOLO
                    coords = [float(x) for x in line.split()]
                    if len(coords) != 5:
                        print(f"⚠️  Format invalide pour {image_name}: {line}")
                        continue

                    # Convertir les coordonnées et obtenir le class_id
                    class_id, bbox = convert_yolo_to_bbox(coords, img_width, img_height)

                    # Vérifier que le class_id est valide (0-87)
                    if class_id < 0 or class_id >= len(AEROSCAN_CLASSES):
                        print(f"⚠️  Class ID {class_id} invalide pour {image_name}, ignoré")
                        continue

                    # Obtenir le nom de la classe
                    class_name = AEROSCAN_CLASSES[int(class_id)]

                    # Créer l'annotation JSON
                    json_data = create_json_annotation(
                        image_name, bbox, bbox_id, img_width, img_height, class_name
                    )

                    # Sauvegarder le JSON
                    json_filename = f"{base_name}_{bbox_id}.json"
                    json_path = os.path.join(TARGET_DIRECTORY, json_filename)
                    with open(json_path, 'w', encoding='utf-8') as f:
                        json.dump(json_data, f, indent=4, ensure_ascii=False)

                    total_annotations += 1

                except Exception as e:
                    print(f"❌ Erreur pour {image_name}, ligne {bbox_id}: {e}")
                    continue

            images_processed += 1
            if images_processed % 100 == 0:
                print(f"📊 {images_processed} images traitées, {total_annotations} annotations créées")

    print(f"\n✅ Conversion terminée!")
    print(f"📊 {images_processed} images avec annotations traitées")
    print(f"📊 {total_annotations} annotations JSON créées")
    print(f"⏭️  {skipped_images} images de fond ignorées (pas d'avion)")
    print(f"📁 Résultat dans: {TARGET_DIRECTORY}")

def main():
    """Fonction principale"""
    print("=" * 60)
    print("🔄 CONVERTISSEUR AEROSCAN VERS JSON")
    print("=" * 60)
    print(f"📁 Répertoire source: {ORIGIN_DIRECTORY}")
    print(f"📁 Répertoire cible: {TARGET_DIRECTORY}")
    print(f"🏷️  Classes cibles: {len(AEROSCAN_CLASSES)} (ex: {AEROSCAN_CLASSES[:5]}...)")

    # Demander confirmation
    response = input("\n❓ Voulez-vous continuer? (y/N): ").strip().lower()
    if response not in ['y', 'yes', 'oui']:
        print("❌ Conversion annulée")
        return

    process_aeroscan_dataset()

if __name__ == "__main__":
    main()