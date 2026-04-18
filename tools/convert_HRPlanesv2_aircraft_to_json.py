#!/usr/bin/env python3
"""
Convertisseur HRPlanesv2 (YOLO Format) vers format JSON personnalisé
Convertit le dataset HRPlanesv2 vers le format JSON utilisé par l'outil d'annotation
"""

import os
import json
from pathlib import Path
from PIL import Image

# Configuration (à adapter selon tes besoins)
ORIGIN_DIRECTORY = "/home/aobled/Downloads/HRPlanesv2/"  # Répertoire contenant les images et annotations
TARGET_DIRECTORY = "/home/aobled/Downloads/HRPlanesv2/json"

def get_image_dimensions(image_path):
    """Obtient les dimensions d'une image"""
    try:
        with Image.open(image_path) as img:
            return img.size  # (width, height)
    except Exception as e:
        print(f"Erreur lors de la lecture de {image_path}: {e}")
        return None, None

def convert_yolo_to_bbox(yolo_coords, img_width, img_height):
    """
    Convertit les coordonnées YOLO (normalisées) vers le format bbox [x, y, w, h] en pixels absolus
    YOLO format: class_id x_center y_center width height (tous normalisés 0-1)
    Notre format: [x, y, w, h] (coin supérieur gauche + dimensions)
    """
    class_id, x_center, y_center, width, height = yolo_coords

    # Conversion vers pixels absolus
    x_center_px = x_center * img_width
    y_center_px = y_center * img_height
    width_px = width * img_width
    height_px = height * img_height

    # Conversion vers format [x, y, w, h] (coin supérieur gauche + dimensions)
    x = x_center_px - (width_px / 2)
    y = y_center_px - (height_px / 2)

    return [x, y, width_px, height_px]

def create_json_annotation(image_name, bbox, bbox_id, img_width, img_height):
    """
    Crée une annotation JSON au format utilisé par l'outil d'annotation
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
            "category_name": "unknown",  # Par défaut, comme demandé
            "bbox_id": bbox_id
        }
    }

def process_hrplanesv2_dataset():
    """Traite le dataset HRPlanesv2 et le convertit vers le format JSON"""

    print(f"🚀 Début de la conversion HRPlanesv2 vers JSON")
    print(f"📁 Répertoire source: {ORIGIN_DIRECTORY}")
    print(f"📁 Répertoire cible: {TARGET_DIRECTORY}")

    # Vérifier que le répertoire source existe
    if not os.path.exists(ORIGIN_DIRECTORY):
        print(f"❌ Erreur: Le répertoire source {ORIGIN_DIRECTORY} n'existe pas!")
        return

    # Créer le répertoire cible
    os.makedirs(TARGET_DIRECTORY, exist_ok=True)

    # Parcourir les images dans le répertoire source
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp')
    images_processed = 0
    total_annotations = 0

    # Traiter chaque image
    for file in os.listdir(ORIGIN_DIRECTORY):
        if file.lower().endswith(image_extensions):
            image_path = os.path.join(ORIGIN_DIRECTORY, file)
            image_name = os.path.basename(file)
            base_name = os.path.splitext(image_name)[0]

            # Chercher le fichier d'annotation correspondant (.txt)
            annotation_file = os.path.join(ORIGIN_DIRECTORY, f"{base_name}.txt")

            if os.path.exists(annotation_file):
                # Obtenir les dimensions de l'image
                img_width, img_height = get_image_dimensions(image_path)
                if img_width is None or img_height is None:
                    print(f"⚠️  Impossible de lire les dimensions de {image_name}, ignoré")
                    continue

                # Lire les annotations YOLO
                with open(annotation_file, 'r') as f:
                    lines = f.readlines()

                if not lines:
                    print(f"⚠️  Aucune annotation trouvée pour {image_name}")
                    continue

                # Traiter chaque annotation
                for bbox_id, line in enumerate(lines):
                    line = line.strip()
                    if not line:
                        continue

                    try:
                        # Parser la ligne YOLO
                        coords = [float(x) for x in line.split()]
                        if len(coords) != 5:
                            print(f"⚠️  Format d'annotation invalide pour {image_name}: {line}")
                            continue

                        # Convertir les coordonnées
                        bbox = convert_yolo_to_bbox(coords, img_width, img_height)

                        # Créer l'annotation JSON
                        json_data = create_json_annotation(
                            image_name, bbox, bbox_id, img_width, img_height
                        )

                        # Sauvegarder l'annotation JSON
                        json_filename = f"{base_name}_{bbox_id}.json"
                        json_path = os.path.join(TARGET_DIRECTORY, json_filename)

                        with open(json_path, 'w', encoding='utf-8') as f:
                            json.dump(json_data, f, indent=4, ensure_ascii=False)

                        total_annotations += 1

                    except Exception as e:
                        print(f"❌ Erreur lors du traitement de {image_name}, ligne {bbox_id}: {e}")
                        continue

                images_processed += 1
                if images_processed % 100 == 0:
                    print(f"📊 {images_processed} images traitées, {total_annotations} annotations créées")

    print(f"\n✅ Conversion terminée!")
    print(f"📊 {images_processed} images traitées")
    print(f"📊 {total_annotations} annotations créées")
    print(f"📁 Résultat dans: {TARGET_DIRECTORY}")

def main():
    """Fonction principale"""
    print("=" * 60)
    print("🔄 CONVERTISSEUR HRPLANESV2 VERS JSON")
    print("=" * 60)

    # Vérifier la configuration
    print(f"📁 Répertoire source: {ORIGIN_DIRECTORY}")
    print(f"📁 Répertoire cible: {TARGET_DIRECTORY}")

    # Demander confirmation
    response = input("\n❓ Voulez-vous continuer? (y/N): ").strip().lower()
    if response not in ['y', 'yes', 'oui']:
        print("❌ Conversion annulée")
        return

    # Lancer la conversion
    process_hrplanesv2_dataset()

if __name__ == "__main__":
    main()