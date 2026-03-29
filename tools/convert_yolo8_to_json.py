#!/usr/bin/env python3
"""
Convertisseur YOLO8 vers format JSON personnalisé
Convertit le dataset Air Military Vehicle Dataset (YOLO8 Format) vers le format JSON utilisé par l'outil d'annotation
"""

import os
import json
import shutil
from pathlib import Path
from PIL import Image

# Configuration
ORIGIN_DIRECTORY = "/home/aobled/Downloads/origin-small-air-military-vehicle"  # À modifier selon votre chemin
TARGET_DIRECTORY = "/home/aobled/Downloads/target-small-air-military-vehicle"

# Mapping des classes YOLO8 vers vos catégories existantes
CLASS_MAPPING = {
    0: "c2",               # C2 Greyhound → c2
    1: "u2",               # U-2 Dragon Lady → u2
    2: "a10",              # A-10 Thunderbolt II → a10
    3: "b1b",              # B-1 Lancer → b1b
    4: "b52",              # B-52 Stratofortress → b52
    5: "f15",              # F-15 Eagle → f15
    6: "ka27",             # Ka-27 Helix → ka27
    7: "c130",             # C-130 Hercules → c130
    8: "us2",              # US-2 → us2
    9: "f16",              # F-16 Fighting Falcon → f16
    10: "tu22m",           # Tu-22M Backfire → tu22m
    11: "mig31",           # MiG-31 Foxhound → mig31
    12: "b2",              # B-2 Spirit → b2
    13: "hawkeye",         # E-2 Hawkeye → hawkeye
    14: "su34",            # Su-34 Fullback → su34
    15: "y20",             # Y-20 → y20
    16: "b21",             # B-21 Raider → b21
    17: "rafale",          # Rafale → rafale
    18: "f35",             # F-35 Lightning II → f35
    19: "mq9",             # MQ-9 Reaper → mq9
    20: "kj600",           # KJ-600 → kj600
    21: "v22",             # V-22 Osprey → v22
    22: "harrier",         # AV-8B Harrier II → harrier
    23: "f22",             # F-22 Raptor → f22
    24: "typhoon",         # Eurofighter Typhoon → typhoon
    25: "ch47",            # CH-47 Chinook → ch47
    26: "f18",             # F/A-18 Hornet → f18
    27: "mi24",            # Mi-24 Hind → mi24
    28: "su25",            # Su-25 Frogfoot → su25
    29: "su57",            # Su-57 Felon → su57
    30: "c5",              # C-5 Galaxy → c5
    31: "cl415",           # CL-415 → cl415
    32: "tu160",           # Tu-160 Blackjack → tu160
    33: "mirage2000",      # Mirage 2000 → mirage2000
    34: "p3",              # P-3 Orion → p3
    35: "e7",              # E-7 Wedgetail → e7
    36: "gripen",          # JAS 39 Gripen → gripen
    37: "j20",             # J-20 Mighty Dragon → j20
    38: "be200",           # Be-200 → be200
    39: "h6",              # H-6 → h6
    40: "f117",            # F-117 Nighthawk → f117
    41: "f4",              # F-4 Phantom II → f4
    42: "jh7",             # JH-7 → jh7
    43: "c390",            # C-390 Millennium → c390
    44: "ka52",            # Ka-52 Alligator → ka52
    45: "vulcan",          # Vulcan → vulcan
    46: "j10",             # J-10 Vigorous Dragon → j10
    47: "f14",             # F-14 Tomcat → f14
    48: "a400m",           # A400M Atlas → a400m
    49: "mig29",           # MiG-29 Fulcrum → mig29
    50: "an124",           # An-124 Ruslan → an124
    51: "sr71",            # SR-71 Blackbird → sr71
    52: "tu95",            # Tu-95 Bear → tu95
    53: "c17",             # C-17 Globemaster III → c17
    54: "tb2",             # TB2 Bayraktar → tb2
    55: "ah64",            # AH-64 Apache → ah64
    56: "mi28",            # Mi-28 Havoc → mi28
    57: "rq4",             # RQ-4 Global Hawk → rq4
    58: "su24",            # Su-24 Fencer → su24
    59: "tb001",           # TB001 → tb001
    60: "uh60",            # UH-60 Black Hawk → uh60
    61: "jf17",            # JF-17 Thunder → jf17
    62: "tornado",         # Tornado → tornado
    63: "an225",           # An-225 Mriya → an225
    64: "yf23",            # YF-23 Black Widow II → yf23
    65: "ag600",           # AG600 → ag600
    66: "xb70",            # XB-70 Valkyrie → xb70
    67: "wz7",             # WZ-7 → wz7
    68: "kc135",           # KC-135 Stratotanker → kc135
    69: "kf21",            # KF-21 Boramae → kf21
    70: "unknown",         # Mi-26 Halo → unknown (pas dans vos classes)
    71: "an72",            # An-72 Coaler → an72
    72: "an22",            # An-22 Antei → an22
    73: "z19"              # Z-19 → z19
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
    Convertit les coordonnées YOLO8 (normalisées) vers le format bbox [x, y, w, h]
    
    YOLO8 format: class_id x_center y_center width height (tous normalisés 0-1)
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

def process_yolo8_dataset():
    """Traite le dataset YOLO8 et le convertit vers le format JSON"""
    
    print(f"🚀 Début de la conversion YOLO8 vers JSON")
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
    
    for root, dirs, files in os.walk(ORIGIN_DIRECTORY):
        for file in files:
            if file.lower().endswith(image_extensions):
                image_path = os.path.join(root, file)
                image_name = os.path.basename(file)
                base_name = os.path.splitext(image_name)[0]
                
                # Chercher le fichier d'annotation correspondant (.txt)
                # Dans YOLO8, les images sont dans train/images/ et les labels dans train/labels/
                if "/images/" in image_path:
                    # Remplacer "images" par "labels" dans le chemin
                    labels_dir = image_path.replace("/images/", "/labels/")
                    labels_dir = os.path.dirname(labels_dir)
                    # Les fichiers d'annotation ont un préfixe "label_"
                    annotation_file = os.path.join(labels_dir, f"label_{base_name}.txt")
                else:
                    # Fallback : chercher dans le même répertoire
                    annotation_file = os.path.join(root, f"label_{base_name}.txt")
                
                print("TRAITEMENT = ", image_path, "annotation=", annotation_file)
                print("EXISTS = ", os.path.exists(annotation_file))
                print("ABSPATH = ", os.path.abspath(annotation_file))
                
                if os.path.exists(annotation_file):
                    # Obtenir les dimensions de l'image
                    img_width, img_height = get_image_dimensions(image_path)
                    if img_width is None or img_height is None:
                        print(f"⚠️  Impossible de lire les dimensions de {image_name}, ignoré")
                        continue
                    
                    # Lire les annotations YOLO8
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
                            # Parser la ligne YOLO8
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
                    if images_processed % 100 == 0:
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
    print("🔄 CONVERTISSEUR YOLO8 VERS JSON")
    print("=" * 60)
    
    # Vérifier la configuration
    print(f"📁 Répertoire source: {ORIGIN_DIRECTORY}")
    print(f"📁 Répertoire cible: {TARGET_DIRECTORY}")
    print(f"🗺️  Mapping des classes: {CLASS_MAPPING}")
    
    # Demander confirmation
    response = input("\n❓ Voulez-vous continuer? (y/N): ").strip().lower()
    if response not in ['y', 'yes', 'oui']:
        print("❌ Conversion annulée")
        return
    
    # Lancer la conversion
    process_yolo8_dataset()

if __name__ == "__main__":
    main()
