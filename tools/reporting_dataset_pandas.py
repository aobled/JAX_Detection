import os
import json
import pandas as pd
from collections import defaultdict
import tqdm

def load_dataset_to_dataframe(target_root):
    data = []
    for split in ["train", "val"]:
        split_dir = os.path.join(target_root, split)
        if not os.path.exists(split_dir):
            continue
        for root, _, files in tqdm.tqdm(os.walk(split_dir)):
            image_files = defaultdict(list)
            for filename in files:
                if filename.endswith('.json'):
                    base_name = filename.split('_')[0]  # Ex: "8a2d52aa9175ab5e_0.json" -> "8a2d52aa9175ab5e"
                    image_files[base_name].append(filename)

            for base_name, json_files in image_files.items():
                image_path = None
                for json_file in json_files:
                    json_path = os.path.join(root, json_file)
                    try:
                        with open(json_path, 'r') as f:
                            annotation = json.load(f)
                            image_info = annotation.get("image", {})
                            ann_info = annotation.get("annotation", {})
                            bbox = ann_info.get("bbox", [])
                            image_filename = image_info.get("file_name", f"{base_name}.jpg")
                            data.append({
                                "image_path": json_path,
                                "image_filename": image_filename,
                                "base_image_name": base_name,  # <-- Ajout de cette colonne
                                "image_width": image_info.get("width", 0),
                                "image_height": image_info.get("height", 0),
                                "box_id": ann_info.get("bbox_id", 0),
                                #"box_class": ann_info.get("category_name", "unknown").lower(),
                                "box_class": ann_info.get("category_name", "unknown"),
                                "box_x": bbox[0],
                                "box_y": bbox[1],
                                "box_l": bbox[2],
                                "box_h": bbox[3],
                                "split": split,
                                "directory": os.path.basename(root)
                            })
                    except Exception as e:
                        print(f"Erreur lecture {json_path}: {e}")
                        continue
    return pd.DataFrame(data)

# GROUP BY class & split (train/val)
def reporting_groupby_class_and_split(df):
    class_counts = df.groupby(['box_class', 'split']).size().unstack(fill_value=0)
    class_counts['Total'] = class_counts.sum(axis=1)
    class_counts['Ratio'] = (class_counts['val'] / class_counts['Total'] * 100).round(1)
    
    # Réorganisation et affichage
    result = class_counts[['train', 'val', 'Total', 'Ratio']]
    result = result.sort_index()  # Trier par nom de classe
    
    print(result)
    result.to_csv("comptage_classes_train_val.csv")

# GROUP BY nombre de boxes
def reporting_groupby_box_count(df):
    # 1. Compter le nombre de boxe par image et par split
    box_counts = df.groupby(['base_image_name', 'split']).size().reset_index(name='box_count')
    
    # 2. Pivoter pour obtenir le nombre d'images par box_count et par split
    pivot = box_counts.pivot_table(
        index='box_count',
        columns='split',
        values='base_image_name',
        aggfunc='count',
        fill_value=0
    ).reset_index()
    
    # 3. Renommer les colonnes pour plus de clarté
    pivot.columns = ['box_count', 'train', 'val']
    
    # 4. Calculer le ratio val/(train+val) en pourcentage
    pivot['ratio'] = (pivot['val'] / (pivot['train'] + pivot['val'])) * 100
    
    # 5. Arrondir le ratio à 2 décimales
    pivot['ratio'] = pivot['ratio'].round(2).astype(str) + '%'
    
    # 6. Afficher le résultat
    print(pivot)

#  images avec une seule box dans le mauvais repertoire
def reporting_boxes_on_wrong_directory(df):
    single_box = (
        df[df['box_class'].isin(CLASS_NAMES)]
        .groupby('image_filename')
        .filter(lambda x: len(x) == 1)
        .query("box_class != directory")
    )
    COLUMNS = ['base_image_name', 'box_class', 'directory', 'split']
    print(single_box[COLUMNS])
    single_box[COLUMNS].to_csv("boxes_on_wrong_directory.csv", index=False)


#  images avec une seule box classe TRAGET_CLASS et taille TARGET_SIZE
def reporting_single_boxes_target_class_size(df, target_class, target_size):
    COLUMNS = ['base_image_name', 'box_class', 'directory', 'split']
    TRAGET_CLASS = target_class
    TARGET_SIZE = target_size
    
    # 1. Filtrer les images avec exactement une boxe (quel que soit le type)
    single_box_images = df.groupby('base_image_name').filter(lambda x: len(x) == 1)
    
    # 2. Parmi ces images, ne garder que celles où la boxe est de classe TRAGET_CLASS
    single_box_target = single_box_images[(single_box_images['box_class'] == TRAGET_CLASS) &
        ((single_box_images['box_l'] > TARGET_SIZE) &
        (single_box_images['box_h'] > TARGET_SIZE))
       ]
    
    # 3. Exporter le résultat
    print(single_box_target[COLUMNS])
    single_box_target[COLUMNS].to_csv("tmp_count1.csv", index=False)


#  specific classe images with size > 16px
def reporting_single_classe_images(df, target_class='b52', min_size=16):
    TARGET_CLASS = target_class
    TARGET_SIZE = min_size
    
    def get_base_image_name(json_path):
        filename = os.path.basename(json_path)
        return filename.split('_')[0]  # Ex: "8a2d52aa9175ab5e_0.json" -> "8a2d52aa9175ab5e"
    
    # Ajouter une colonne avec le nom de base de l'image
    df['base_image_name'] = df['image_path'].apply(
        lambda path: get_base_image_name(path.replace('.jpg', '_0.json'))  # On simule le nom du JSON
    )
    
    # 1. Filtrer les images avec au moins une boxe 
    hawkeye_images = df[df['box_class'] == TARGET_CLASS]['base_image_name'].unique()
    # 2. Pour chaque image, vérifier qu'elle n'a AUCUNE autre boxe d'une autre classe
    unique_hawkeye_images = []
    for image in hawkeye_images:
        boxes_in_image = df[df['base_image_name'] == image]
        if len(boxes_in_image) == 1 and boxes_in_image['box_class'].iloc[0] == TARGET_CLASS:
            unique_hawkeye_images.append(image)
    
    # 3. Filtrer ces images et appliquer la condition de taille
    single_box_filtered = df[
        (df['base_image_name'].isin(unique_hawkeye_images)) &
        ((df['box_l'] > TARGET_SIZE) &
        (df['box_h'] > TARGET_SIZE))
    ]
    
    # Columns selection
    COLUMNS = ['image_filename', 'box_class', 'directory', 'split']
    print(single_box_filtered[COLUMNS])
    # Export vers CSV
    single_box_filtered.to_csv("single_box_"+target_class+".csv", columns=COLUMNS, index=False)




#  specific classe searched
def reporting_all_images_in_class_list(df, class_list):
    searched_box = (
        df[df['box_class'].isin(class_list)]
    )
    # Columns selection
    COLUMNS = ['image_filename', 'box_class', 'directory', 'split']
    print(searched_box[COLUMNS])
    # Export vers CSV
    searched_box.to_csv(
        "searched_box.csv",
        columns=COLUMNS,
        index=False
    )

# Filtrer les boxes où image_width < 16 ou image_height < 16
def reporting_small_boxes(df, min_size=16):
    small_images_boxes = df[(df['box_l'] < min_size) | (df['box_h'] < min_size)]
    print("Boxes avec image_width ou image_height < 16 :")
    print(small_images_boxes[['image_filename', 'box_l', 'box_h', 'box_class', 'directory']])
    small_images_boxes.to_csv("boxes_small_images.csv", index=False)
 

#------------------------------
#--- Configuration download ---
#------------------------------
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset_configs import get_dataset_config

DATASET_PATH = '/home/aobled/Downloads/Aircraft_DATASET/detection'
DATASET_NAME = "FIGHTERJET_CLASSES"     # Nom de la config dans dataset_configs.py
try:
    config = get_dataset_config(DATASET_NAME)
    CLASS_NAMES = config["class_names"]
    print(f"📊 Classes ({len(CLASS_NAMES)}): {CLASS_NAMES}")
except Exception as e:
    print(f"❌ Erreur chargement config: {e}")
    sys.exit(1)
#------------------------------

# Charger les données
df = load_dataset_to_dataframe(DATASET_PATH)

#reporting_groupby_class_and_split(df)
reporting_groupby_box_count(df)
#reporting_boxes_on_wrong_directory(df)
#reporting_single_boxes_target_class_size(df, target_class='a10', target_size=2)
#reporting_single_classe_images(df, target_class='b52', min_size=16)
#reporting_all_images_in_class_list(df, class_list=['mig23'])
#reporting_small_boxes(df, min_size=16)
