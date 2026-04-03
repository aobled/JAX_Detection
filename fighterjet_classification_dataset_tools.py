import os
import json
import glob
import random
import numpy as np
from PIL import Image, ImageOps
import shutil
import tqdm


def calculate_iou(box1, box2):
    """
    Calcule l'Intersection over Union (IoU) entre deux bounding boxes
    
    Args:
        box1, box2: (x, y, w, h) - coordonnées des bounding boxes
    
    Returns:
        float: IoU entre 0 et 1
    """
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    
    # Calculer les coordonnées des rectangles
    x1_min, y1_min = x1, y1
    x1_max, y1_max = x1 + w1, y1 + h1
    x2_min, y2_min = x2, y2
    x2_max, y2_max = x2 + w2, y2 + h2
    
    # Calculer l'intersection
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)
    
    # Si pas d'intersection
    if inter_x_max <= inter_x_min or inter_y_max <= inter_y_min:
        return 0.0
    
    # Aire de l'intersection
    inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
    
    # Aire des deux rectangles
    area1 = w1 * h1
    area2 = w2 * h2
    
    # Aire de l'union
    union_area = area1 + area2 - inter_area
    
    # Éviter la division par zéro
    if union_area == 0:
        return 0.0
    
    return inter_area / union_area


def has_overlapping_boxes(json_files, overlap_threshold=0.1, categories = 'a10'):
    """
    Vérifie si une image contient des bounding boxes qui se superposent
    
    Args:
        json_files: Liste des fichiers JSON pour une image
        overlap_threshold: Seuil d'IoU pour considérer une superposition (0.1 = 10%)
    
    Returns:
        bool: True si des boxes se superposent au-dessus du seuil
    """
    boxes = []
    
    # Charger toutes les bounding boxes de l'image
    for json_file in json_files:
        try:
            with open(json_file, "r") as f:
                data = json.load(f)
            
            bbox = data["annotation"]["bbox"]
            category = data["annotation"]["category_name"]
            x, y, w, h = map(int, bbox)
            
            # Filtrer les classes valides
            if category in categories:
                boxes.append((x, y, w, h, category, json_file))
        except Exception as e:
            print(f"Erreur lecture {json_file}: {e}")
            continue
    
    # Vérifier les superpositions
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            box1 = boxes[i][:4]  # (x, y, w, h)
            box2 = boxes[j][:4]  # (x, y, w, h)
            
            iou = calculate_iou(box1, box2)
            
            if iou > overlap_threshold:
                print(f"[⚠️] Superposition détectée: {boxes[i][4]} vs {boxes[j][4]} (IoU={iou:.3f})")
                print(f"    Box1: {json_files[i] if i < len(json_files) else 'N/A'}")
                print(f"    Box2: {json_files[j] if j < len(json_files) else 'N/A'}")
                return True
    
    return False


def process_dataset_stretched(
    root_dir,
    output_dir,
    target_size=128,
    grayscale=False,  # 🎨 Option pour images en noir & blanc
    overlap_threshold=0.1,  # 🔥 Nouveau: Seuil de superposition (0.1 = 10%)
    categories = 'a10'
):
    """
    Traite les images du dataset avec resize stretched (sans préserver le ratio)
    PRÉSERVE LA RÉPARTITION train/val si présente dans le répertoire source
    
    Args:
        root_dir: Répertoire racine contenant les images et JSONs (peut contenir train/ et val/)
        output_dir: Répertoire de sortie (créera train/ et val/ automatiquement si nécessaire)
        target_size: Taille cible (carré)
        grayscale: Si True, convertit en niveaux de gris (L) au lieu de RGB
    """
    mode = "L" if grayscale else "RGB"
    mode_str = "Grayscale" if grayscale else "RGB"
    print(f"🎨 Mode de traitement: {mode_str}")
    
    # Vérifier si root_dir contient des sous-dossiers train/val
    train_dir = os.path.join(root_dir, "train")
    val_dir = os.path.join(root_dir, "val")
    
    has_train_val_split = os.path.isdir(train_dir) and os.path.isdir(val_dir)
    
    if has_train_val_split:
        print("📁 Structure train/val détectée - Répartition conservée")
        splits = ["train", "val"]
        base_dirs = [train_dir, val_dir]
    else:
        print("📁 Pas de structure train/val - Traitement global")
        splits = [None]  # Pas de split
        base_dirs = [root_dir]
    
    # Traiter chaque split
    for split, base_dir in zip(splits, base_dirs):
        split_desc = f"{mode_str} - {split}" if split else f"{mode_str}"
        
        # Définir le répertoire de sortie
        if split:
            output_split_dir = os.path.join(output_dir, split)
        else:
            output_split_dir = output_dir
        os.makedirs(output_split_dir, exist_ok=True)

        jpg_files = glob.glob(os.path.join(base_dir, "**", "*.jpg"), recursive=True)
        png_files = glob.glob(os.path.join(base_dir, "**", "*.png"), recursive=True)
        image_paths = jpg_files + png_files

        for image_path in tqdm.tqdm(image_paths, desc=f"Processing {split_desc}"):
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            image = Image.open(image_path).convert(mode)  # 🎨 RGB ou Grayscale

            json_pattern = os.path.join(os.path.dirname(image_path), f"{base_name}_*.json")
            json_files = glob.glob(json_pattern)

            # 🔥 NOUVEAU: Vérifier les superpositions AVANT de traiter les boxes
            #if has_overlapping_boxes(json_files, overlap_threshold):
            #    print(f"[⚠️] Image ignorée (superpositions): {os.path.basename(image_path)}")
            #    continue

            for json_file in json_files:
                with open(json_file, "r") as f:
                    data = json.load(f)

                bbox = data["annotation"]["bbox"]
                category = data["annotation"]["category_name"]
                x, y, w, h = map(int, bbox)
                
                # don't considere boxes too small, that is to say half of the target heigth ou width
                #if w <= target_size/8 or h <= target_size/8:
                #     print(f"[⚠️] Bbox too small (w/h ≤ {target_size/3.2}) : {json_file}")
                #     continue

                #if category not in ['a10', 'c17', 'f14', 'f15', 'f16', 'f18', 'f22', 'f35', 'mirage2000', 'rafale', 'typhoon']:
                if category not in categories:
                    continue

                cropped = image.crop((x, y, x + w, y + h))

                # Ici on redimensionne en target_size × target_size SANS respecter le ratio (stretched)
                stretched = cropped.resize((target_size, target_size), Image.Resampling.LANCZOS)

                # Création dossier classe
                class_dir = os.path.join(output_split_dir, category)
                os.makedirs(class_dir, exist_ok=True)

                # Sauvegarde avec nom JSON incluant la taille minimale
                min_size = min(w, h)
                size_prefix = f"{min_size:04d}_"  # 4 chiffres avec zéros non significatifs
                base_name = os.path.splitext(os.path.basename(json_file))[0]
                out_name = size_prefix + base_name + ".png"
                out_path = os.path.join(class_dir, out_name)
                stretched.save(out_path)

                #print(f"[✓] Sauvé (stretched) : {out_path}")


def analyze_overlap_statistics(root_dir, overlap_thresholds=[0.05, 0.1, 0.15, 0.2]):
    """
    Analyse les statistiques de superposition dans le dataset
    
    Args:
        root_dir: Répertoire racine contenant les images et JSONs
        overlap_thresholds: Liste des seuils à tester
    """
    print("🔍 ANALYSE DES SUPERPOSITIONS DANS LE DATASET")
    print("=" * 60)
    
    jpg_files = glob.glob(os.path.join(root_dir, "**", "*.jpg"), recursive=True)
    png_files = glob.glob(os.path.join(root_dir, "**", "*.png"), recursive=True)
    image_paths = jpg_files + png_files
    
    total_images = len(image_paths)
    images_with_multiple_boxes = 0
    
    print(f"📊 Total d'images: {total_images}")
    
    for threshold in overlap_thresholds:
        overlapping_images = 0
        
        for image_path in tqdm.tqdm(image_paths, desc=f"Seuil {threshold}"):
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            json_pattern = os.path.join(os.path.dirname(image_path), f"{base_name}_*.json")
            json_files = glob.glob(json_pattern)
            
            if len(json_files) > 1:
                images_with_multiple_boxes += 1
                
                if has_overlapping_boxes(json_files, threshold):
                    overlapping_images += 1
        
        percentage = (overlapping_images / total_images) * 100 if total_images > 0 else 0
        print(f"Seuil {threshold:4.2f}: {overlapping_images:4d} images ({percentage:5.1f}%) avec superpositions")
    
    multiple_percentage = (images_with_multiple_boxes / total_images) * 100 if total_images > 0 else 0
    print(f"\n📊 Images avec plusieurs boxes: {images_with_multiple_boxes} ({multiple_percentage:.1f}%)")
    print("=" * 60)


if __name__ == "__main__":
    # Exemple d'utilisation
    print("🚀 EXEMPLE D'UTILISATION")
    print("=" * 40)
    
    # Analyse des superpositions
    # analyze_overlap_statistics("/path/to/your/dataset")
    
    # Traitement avec détection de superposition
    # process_dataset_stretched(
    #     root_dir="/path/to/your/dataset",
    #     output_dir="/path/to/output",
    #     target_size=128,
    #     grayscale=True,
    #     overlap_threshold=0.1  # Ignorer les images avec >10% de superposition
    # )
    
    print("✅ Fonctions ajoutées:")
    print("  - calculate_iou(): Calcule l'IoU entre deux boxes")
    print("  - has_overlapping_boxes(): Détecte les superpositions")
    print("  - analyze_overlap_statistics(): Analyse les statistiques")
    print("  - process_dataset_stretched(): Maintenant avec détection de superposition")




# ========== 3. Création des fichiers NPZ ==========
from PIL import Image
def create_npz_splits(
    dataset_dir,
    output_prefix,
    image_size=(128, 128)
):
    for split in ['train', 'val']:
        split_dir = os.path.join(dataset_dir, split)
        class_names = sorted(os.listdir(split_dir))
        class_to_idx = {name: idx for idx, name in enumerate(class_names)}

        images, labels = [], []

        for class_name in tqdm.tqdm(class_names):
            class_path = os.path.join(split_dir, class_name)
            for path in glob.glob(os.path.join(class_path, "*.png")):
                img = Image.open(path).convert("RGB").resize(image_size)
                images.append(np.asarray(img))
                labels.append(class_to_idx[class_name])

        images = np.stack(images).astype(np.uint8)
        labels = np.array(labels).astype(np.uint8)

        out_file = f"{output_prefix}_{split}.npz"
        np.savez_compressed(out_file, image=images, label=labels)
        print(f"[✓] {split} : {len(images)} images sauvegardées dans {out_file}")


# Calcul du MEAN et STD
def calculate_normalization_stats(root_dir, sample_size=None, save_path=None):
    """
    Calcule la moyenne et l'écart-type du dataset de manière itérative (sans tout charger en RAM).
    
    Args:
        root_dir: Dossier racine des images.
        sample_size: Si défini, limite le calcul à un échantillon aléatoire de X images (ex: 5000).
        save_path: Si défini, sauvegarde les stats dans un fichier .npz (mean=..., std=...).
    """
    print(f"📊 Calcul des stats de normalisation sur {root_dir}")
    
    # Recherche de tous les fichiers .jpg et .png
    jpg_files = glob.glob(os.path.join(root_dir, "**", "*.jpg"), recursive=True)
    png_files = glob.glob(os.path.join(root_dir, "**", "*.png"), recursive=True)
    image_paths = jpg_files + png_files
    
    total_images = len(image_paths)
    if total_images == 0:
        raise ValueError("Aucune image trouvée.")

    # Échantillonnage si demandé
    if sample_size and sample_size < total_images:
        print(f"🎲 Échantillonnage activé : {sample_size} images sur {total_images}")
        random.shuffle(image_paths)
        image_paths = image_paths[:sample_size]
    else:
        print(f"📚 Traitement de toutes les images : {total_images}")

    # Variables pour le calcul itératif
    sum_rgb = np.zeros(3)
    sum_sq_rgb = np.zeros(3)
    count = 0

    for img_path in tqdm.tqdm(image_paths, desc="Calcul Mean/Std"):
        try:
             with Image.open(img_path) as img:
                # Convertir en RGB et normaliser [0, 1]
                img_array = np.array(img.convert("RGB")) / 255.0
                
                # Aplatir les dimensions H et W pour ne garder que les canaux
                # img_array shape: (H, W, 3) -> reshape -> (N, 3)
                pixels = img_array.reshape(-1, 3)
                
                # Sommes cumulées
                sum_rgb += pixels.sum(axis=0)
                sum_sq_rgb += (pixels ** 2).sum(axis=0)
                count += pixels.shape[0]
                
        except Exception as e:
            print(f"Erreur lecture {img_path}: {e}")
            continue

    # Calcul final
    mean = sum_rgb / count
    # Var = E[X^2] - (E[X])^2
    variance = (sum_sq_rgb / count) - (mean ** 2)
    # Std = sqrt(Var)
    std = np.sqrt(variance)

    print("-" * 30)
    print(f"✅ Résultat ({count} pixels analysés) :")
    print(f"Mean: {mean}")
    print(f"Std : {std}")
    print("-" * 30)

    # Sauvegarde si demandé
    if save_path:
        # Créer dossiers parents si nécessaire
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        np.savez(save_path, mean=mean.astype(np.float32), std=std.astype(np.float32))
        print(f"💾 Stats sauvegardées dans : {save_path}")

    return mean, std

# 🎨 GÉNÉRATION DU DATASET
TARGET_SIZE = 128
#CLASSES_NAMES = ['a10', 'a400m', 'alphajet', 'b1b', 'b2', 'c130', 'c17', 'f4', 'f14', 'f15', 'f16', 'f18', 'f22', 'f35', 'f117', 'flanker', 'gripen', 'harrier', 'hawk', 'hawkeye', 'jaguar', 'mig29', 'mirage2000', 'rafale', 'su57', 'tornado', 'typhoon']
#------------------------------
#--- Configuration download ---
#------------------------------
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset_configs import get_dataset_config

DATASET_PATH = '/home/aobled/Downloads/Aircraft_DATASET/classification'
DATASET_NAME = "FIGHTERJET_CLASSES"     # Nom de la config dans dataset_configs.py
try:
    config = get_dataset_config(DATASET_NAME)
    CLASS_NAMES = config["class_names"]
    print(f"📊 Classes ({len(CLASS_NAMES)}): {CLASS_NAMES}")
except Exception as e:
    print(f"❌ Erreur chargement config: {e}")
    sys.exit(1)
#------------------------------
process_dataset_stretched(root_dir="/home/aobled/Downloads/Aircraft_DATASET/classification", output_dir="/home/aobled/Downloads/_balanced_dataset_split", target_size=128, grayscale=True, overlap_threshold=0.15, categories = CLASS_NAMES)


# Étape 3 : conversion en fichiers NPZ JAX-friendly
IMAGE_SIZE = (TARGET_SIZE, TARGET_SIZE)
# create_npz_splits("/home/aobled/Downloads/_balanced_dataset_split", "fighterjet", image_size=IMAGE_SIZE)

# Calcul des stats sur un échantillon et sauvegarde pour l'entraînement
"""mean, std = calculate_normalization_stats(
    root_dir='/home/aobled/Downloads/_balanced_dataset_split/', 
    sample_size=10000,
    save_path='./data/chunks/dataset_chunked_meanstd.npz'
)"""
