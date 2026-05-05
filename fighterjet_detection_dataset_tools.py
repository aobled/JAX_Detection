import os
import json
import glob
import numpy as np
from PIL import Image, ImageOps
import tqdm
from typing import List, Tuple, Dict
import cv2

def process_detection_dataset(
    root_dir: str,
    output_dir: str,
    split_name: str = "train",
    target_size: Tuple[int, int] = (224, 224),  # ✅ Valeur par défaut changée à 224x224
    max_boxes: int = 20,
    chunk_size: int = 2000,
    grayscale: bool = False  # 🎨 Support grayscale (3× moins de mémoire)
):
    """
    Traite le dataset pour la détection :
    1. Regroupe les annotations par image source
    2. Applique Letterbox
    3. Normalise les boxes [x, y, w, h] en [0-1] relatif à l'image target_size
    4. Sauvegarde en chunks .npz
    
    Args:
        grayscale: Si True, convertit les images en niveaux de gris (1 canal) au lieu de RGB (3 canaux)
    """
    mode_str = "Grayscale (1 canal)" if grayscale else "RGB (3 canaux)"
    print(f"🚀 Traitement Détection ({split_name}) : {root_dir} -> {target_size} [{mode_str}]")
    
    # Structure pour regrouper les infos par image
    # Clé: nom_image_source (ex: "8a3ab5634b9ab46c.jpg")
    # Valeur: { 'path': ..., 'boxes': [...] }
    images_data: Dict[str, Dict] = {}
    
    # 1. Scanner tous les JSONs pour regrouper par image
    json_files = glob.glob(os.path.join(root_dir, "**", "*.json"), recursive=True)
    print(f"📚 Analyse de {len(json_files)} fichiers JSON...")
    
    for json_file in tqdm.tqdm(json_files, desc="Grouping annotations"):
        try:
            with open(json_file, "r") as f:
                data = json.load(f)
            
            # Nom de l'image source (ex: "8a3ab5634b9ab46c.jpg")
            img_filename = data["image"]["file_name"]
            
            # Chemin complet de l'image (supposé dans le même dossier que le JSON)
            img_path = os.path.join(os.path.dirname(json_file), img_filename)
            
            # Vérifier si l'image existe
            if not os.path.exists(img_path):
                # Fallback: parfois l'image est un niveau au-dessus ou ailleurs
                continue

            if img_filename not in images_data:
                images_data[img_filename] = {
                    'path': img_path,
                    'boxes': []
                }
            
            # Récupérer la bbox (x, y, w, h) originale
            bbox = data["annotation"]["bbox"]
            images_data[img_filename]['boxes'].append(bbox)
            
        except Exception as e:
            # print(f"Skipping {json_file}: {e}")
            pass
            
    print(f"📊 {len(images_data)} images uniques trouvées avec annotations.")
    
    # 2. Créer les Chunks
    image_list = list(images_data.values())
    np.random.shuffle(image_list) # Mélanger pour l'entraînement
    
    processed_count = 0
    chunk_idx = 0
    
    current_chunk_images = []
    current_chunk_masks = [] # Masques de segmentation [H, W, 1]
    
    os.makedirs(output_dir, exist_ok=True)
    
    for i, item in enumerate(tqdm.tqdm(image_list, desc="Processing images")):
        img_path = item['path']
        raw_boxes = item['boxes'] # Liste de [x, y, w, h]
        
        try:
            with Image.open(img_path) as img:
                # 🎨 Convertir en RGB ou Grayscale selon le paramètre
                img_mode = 'L' if grayscale else 'RGB'
                img = img.convert(img_mode)
                orig_w, orig_h = img.size
                
                # --- STRETCHED RESIZING (au lieu de Letterbox) ---
                # On déforme l'image pour qu'elle remplisse le target_size
                new_img = img.resize(target_size, Image.Resampling.LANCZOS)
                
                # Normalisation Image
                img_array = np.array(new_img, dtype=np.float32) / 255.0 # [0-1]
                
                # 🎨 Si grayscale, ajouter dimension canal si nécessaire (H, W) → (H, W, 1)
                if grayscale and img_array.ndim == 2:
                    img_array = np.expand_dims(img_array, axis=-1)
                
                # Traitement des boxes -> Création du Masque (Segmentation)
                mask_array = np.zeros(target_size[::-1] + (1,), dtype=np.float32) # (224, 224, 1)
                
                for box in raw_boxes:
                    bx, by, bw, bh = box
                    
                    # Convertir les coordonnées originales en coordonnées target_size
                    x1 = int((bx / orig_w) * target_size[0])
                    y1 = int((by / orig_h) * target_size[1])
                    x2 = int(((bx + bw) / orig_w) * target_size[0])
                    y2 = int(((by + bh) / orig_h) * target_size[1])
                    
                    # Dessiner une ellipse (1.0) pleine sur le masque au lieu d'un rectangle
                    # Cela permet de couper les coins et d'éviter la fusion des masques pour des objets proches
                    center = ((x1 + x2) // 2, (y1 + y2) // 2)
                    axes = ((x2 - x1) // 2, (y2 - y1) // 2)
                    cv2.ellipse(mask_array, center, axes, 0, 0, 360, 1.0, -1)
                
                current_chunk_images.append(img_array)
                current_chunk_masks.append(mask_array)
                processed_count += 1
                
                # Sauvegarder si chunk plein
                if len(current_chunk_images) >= chunk_size:
                    _save_chunk(output_dir, split_name, chunk_idx, current_chunk_images, current_chunk_masks)
                    chunk_idx += 1
                    current_chunk_images = []
                    current_chunk_masks = []
                    
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            continue

    # Sauvegarder le dernier chunk partiel
    if len(current_chunk_images) > 0:
        _save_chunk(output_dir, split_name, chunk_idx, current_chunk_images, current_chunk_masks)

def _save_chunk(output_dir, split_name, chunk_idx, images, masks):
    images_np = np.array(images, dtype=np.float32) # (N, H, W, C)
    masks_np = np.array(masks, dtype=np.float32)   # (N, H, W, 1)
    
    out_path = os.path.join(output_dir, f"dataset_detection_{split_name}_chunk{chunk_idx}.npz")
    np.savez_compressed(out_path, images=images_np, masks=masks_np)
    print(f"💾 Chunk {chunk_idx} saved: {len(images)} images")


if __name__ == "__main__":
    import sys
    sys.path.append(os.path.dirname(__file__))
    from dataset_configs import get_dataset_config
    
    # Charger la configuration
    config = get_dataset_config("FIGHTERJET_DETECTION")
    
    # Configuration des chemins
    TRAIN_DIR = "/home/aobled/Downloads/Aircraft_DATASET/detection/train"
    VAL_DIR = "/home/aobled/Downloads/Aircraft_DATASET/detection/val"
    OUTPUT_DIR = os.path.dirname(config["output_prefix"])
    
    print("🚀 Démarrage de la préparation des données de DÉTECTION...")
    
    # 🗑️ Supprimer les anciens chunks si ils existent (pour éviter les conflits de taille)
    if os.path.exists(OUTPUT_DIR):
        old_chunks = glob.glob(os.path.join(OUTPUT_DIR, "dataset_detection_*_chunk*.npz"))
        if old_chunks:
            print(f"🗑️  Suppression de {len(old_chunks)} anciens chunks...")
            for chunk in old_chunks:
                try:
                    os.remove(chunk)
                    print(f"   Supprimé: {os.path.basename(chunk)}")
                except Exception as e:
                    print(f"   Erreur suppression {chunk}: {e}")
            print("✅ Anciens chunks supprimés\n")
    
    # 🎨 OPTION: Grayscale pour réduire la mémoire de 3×
    USE_GRAYSCALE = True  # ✅ Recommandé: 3× moins de mémoire, même performance
    
    # 1. Traiter le Training Set
    if os.path.exists(TRAIN_DIR):
        process_detection_dataset(
            TRAIN_DIR, 
            OUTPUT_DIR, 
            split_name="train", 
            target_size=config.get("image_size", (224, 224)),
            max_boxes=config.get("max_boxes", 20),
            chunk_size=19000,
            grayscale=config.get("grayscale", True)
        )
    else:
        print(f"❌ Dossier train non trouvé: {TRAIN_DIR}")

    # 2. Traiter le Val Set
    if os.path.exists(VAL_DIR):
        process_detection_dataset(
            VAL_DIR, 
            OUTPUT_DIR, 
            split_name="val", 
            target_size=config.get("image_size", (224, 224)),
            max_boxes=config.get("max_boxes", 20),
            chunk_size=19000,
            grayscale=config.get("grayscale", True)
        )
    else:
        print(f"❌ Dossier val non trouvé: {VAL_DIR}")
