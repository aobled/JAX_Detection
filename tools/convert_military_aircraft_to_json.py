#!/usr/bin/env python3
"""
Convertisseur Military Aircraft Detection Dataset (YOLO Format) vers format JSON personnalisé
Convertit le dataset Military Aircraft Detection Dataset vers le format JSON utilisé par l'outil d'annotation
"""

import os
import json
import shutil
from pathlib import Path
from PIL import Image

# Configuration
ORIGIN_DIRECTORY = "/home/aobled/Downloads/MilitaryOrigin"  # Répertoire contenant images/ et labels/
TARGET_DIRECTORY = "/home/aobled/Downloads/MilitaryTarget"

# Mapping des classes Military Aircraft Detection Dataset vers vos catégories existantes
CLASS_MAPPING = {
    0: "a10",              # A10 → a10
    1: "a400m",            # A400M → a400m
    2: "ag600",            # AG600 → ag600
    3: "harrier",          # AV8B → harrier
    4: "b1b",              # B1 → b1b
    5: "b2",              # B2 → b2
    6: "b52",              # B52 → b52
    7: "be200",            # Be200 → be200
    8: "c130",             # C130 → c130
    9: "c17",              # C17 → c17
    10: "c2",              # C2 → c2
    11: "c5",              # C5 → c5
    12: "hawkeye",         # E2 → hawkeye
    13: "e7",              # E7 → e7
    14: "typhoon",         # EF2000 → typhoon
    15: "f117",            # F117 → f117
    16: "f14",             # F14 → f14
    17: "f15",             # F15 → f15
    18: "f16",             # F16 → f16
    19: "f18",             # F18 → f18
    20: "f22",             # F22 → f22
    21: "f35",             # F35 → f35
    22: "f4",              # F4 → f4
    23: "j20",             # J20 → j20
    24: "gripen",          # JAS39 → gripen
    25: "mq9",             # MQ9 → mq9
    26: "mig31",           # Mig31 → mig31
    27: "mirage2000",      # Mirage2000 → mirage2000
    28: "p3",              # P3 → p3
    29: "rq4",             # RQ4 → rq4
    30: "rafale",          # Rafale → rafale
    31: "sr71",            # SR71 → sr71
    32: "su34",            # Su34 → su34
    33: "su57",            # Su57 → su57
    34: "tornado",         # Tornado → tornado
    35: "tu160",           # Tu160 → tu160
    36: "tu95",            # Tu95 → tu95
    37: "u2",              # U2 → u2
    38: "us2",             # US2 → us2
    39: "v22",             # V22 → v22
    40: "vulcan",          # Vulcan → vulcan
    41: "xb70",            # XB70 → xb70
    42: "yf23"             # YF23 → yf23
}

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
    Convertit les coordonnées YOLO (normalisées) vers le format bbox [x, y, w, h]
    
    YOLO format: class_id x_center y_center width height (tous normalisés 0-1)
    Notre format: [x, y, w, h] en pixels absolus
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
    
    return [x, y, width_px, height_px], int(class_id)

def create_json_annotation(image_name, bbox, category_name, bbox_id, img_width, img_height):
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
            "category_name": category_name,
            "bbox_id": bbox_id
        }
    }

def process_military_aircraft_dataset():
    """Traite le dataset Military Aircraft Detection et le convertit vers le format JSON"""
    
    print(f"🚀 Début de la conversion Military Aircraft Detection vers JSON")
    print(f"📁 Répertoire source: {ORIGIN_DIRECTORY}")
    print(f"📁 Répertoire cible: {TARGET_DIRECTORY}")
    
    # Vérifier que le répertoire source existe
    if not os.path.exists(ORIGIN_DIRECTORY):
        print(f"❌ Erreur: Le répertoire source {ORIGIN_DIRECTORY} n'existe pas!")
        return
    
    # Créer le répertoire cible
    os.makedirs(TARGET_DIRECTORY, exist_ok=True)
    
    # Parcourir les images dans les sous-répertoires train et val
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp')
    images_processed = 0
    total_annotations = 0
    
    # Traiter les répertoires train et val
    for split in ['aircraft_train', 'aircraft_val']:
        images_dir = os.path.join(ORIGIN_DIRECTORY, 'images', split)
        labels_dir = os.path.join(ORIGIN_DIRECTORY, 'labels', split)
        
        if not os.path.exists(images_dir):
            print(f"⚠️  Répertoire d'images {images_dir} n'existe pas, ignoré")
            continue
            
        if not os.path.exists(labels_dir):
            print(f"⚠️  Répertoire de labels {labels_dir} n'existe pas, ignoré")
            continue
        
        print(f"📂 Traitement du split: {split}")
        
        # Parcourir les images
        for file in os.listdir(images_dir):
            if file.lower().endswith(image_extensions):
                image_path = os.path.join(images_dir, file)
                image_name = os.path.basename(file)
                base_name = os.path.splitext(image_name)[0]
                
                # Chercher le fichier d'annotation correspondant (.txt)
                annotation_file = os.path.join(labels_dir, f"{base_name}.txt")
                
                print("TRAITEMENT = ", image_path, "annotation=", annotation_file)
                print("EXISTS = ", os.path.exists(annotation_file))
                
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
                            bbox, class_id = convert_yolo_to_bbox(coords, img_width, img_height)
                            
                            # Obtenir le nom de la catégorie
                            category_name = CLASS_MAPPING.get(class_id, f"class_{class_id}")
                            
                            # Créer l'annotation JSON
                            json_data = create_json_annotation(
                                image_name, bbox, category_name, bbox_id, img_width, img_height
                            )
                            
                            # Créer le répertoire de classe s'il n'existe pas
                            class_dir = os.path.join(TARGET_DIRECTORY, category_name)
                            os.makedirs(class_dir, exist_ok=True)
                            
                            # Copier l'image vers le répertoire de classe
                            target_image_path = os.path.join(class_dir, image_name)
                            if not os.path.exists(target_image_path):
                                shutil.copy2(image_path, target_image_path)
                            
                            # Sauvegarder l'annotation JSON
                            json_filename = f"{base_name}_{bbox_id}.json"
                            json_path = os.path.join(class_dir, json_filename)
                            
                            with open(json_path, 'w', encoding='utf-8') as f:
                                json.dump(json_data, f, indent=4, ensure_ascii=False)
                            
                            total_annotations += 1
                            
                        except Exception as e:
                            print(f"❌ Erreur lors du traitement de {image_name}, ligne {bbox_id}: {e}")
                            continue
                    
                    images_processed += 1
                    if images_processed % 1000 == 0:
                        print(f"📊 {images_processed} images traitées, {total_annotations} annotations créées")
    
    print(f"\n✅ Conversion terminée!")
    print(f"📊 {images_processed} images traitées")
    print(f"📊 {total_annotations} annotations créées")
    print(f"📁 Résultat dans: {TARGET_DIRECTORY}")
    
    # Afficher le résumé par classe
    print(f"\n📋 Résumé par classe:")
    for class_name in os.listdir(TARGET_DIRECTORY):
        class_path = os.path.join(TARGET_DIRECTORY, class_name)
        if os.path.isdir(class_path):
            json_files = [f for f in os.listdir(class_path) if f.endswith('.json')]
            images = [f for f in os.listdir(class_path) if f.lower().endswith(image_extensions)]
            print(f"  {class_name}: {len(images)} images, {len(json_files)} annotations")

def main():
    """Fonction principale"""
    print("=" * 60)
    print("🔄 CONVERTISSEUR MILITARY AIRCRAFT DETECTION VERS JSON")
    print("=" * 60)
    
    # Vérifier la configuration
    print(f"📁 Répertoire source: {ORIGIN_DIRECTORY}")
    print(f"📁 Répertoire cible: {TARGET_DIRECTORY}")
    print(f"🗺️  Mapping des classes: {len(CLASS_MAPPING)} classes")
    
    # Demander confirmation
    response = input("\n❓ Voulez-vous continuer? (y/N): ").strip().lower()
    if response not in ['y', 'yes', 'oui']:
        print("❌ Conversion annulée")
        return
    
    # Lancer la conversion
    process_military_aircraft_dataset()

if __name__ == "__main__":
    main()
