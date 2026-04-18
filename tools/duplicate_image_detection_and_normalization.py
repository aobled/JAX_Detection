import os
import json
from PIL import Image
import imagehash
from collections import defaultdict
import tqdm
import sys
import re

def normalize_json(json_data, new_filename, image_path):
    with Image.open(image_path) as img:
        width, height = img.size

    if "image" not in json_data:
        json_data["image"] = {
            "file_name": new_filename,
            "width": width,
            "height": height
        }
    else:
        json_data["image"]["file_name"] = new_filename
        json_data["image"]["width"] = width
        json_data["image"]["height"] = height

    json_data["annotation"]["file_name"] = new_filename
    return json_data

def find_duplicates_optimized(directory, threshold=1):
    hashes = defaultdict(list)
    images = []

    for filename in tqdm.tqdm(os.listdir(directory), desc="calcule des hachages"):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            filepath = os.path.join(directory, filename)
            try:
                with Image.open(filepath) as img:
                    img_hash = imagehash.phash(img)
                    file_size = os.path.getsize(filepath)
                    img_size = img.size
                    hashes[img_hash].append(filename)
                    images.append((filename, img_hash, file_size, img_size))
            except Exception as e:
                print(f"Error processing {filename}: {e}")

    duplicates = {}
    processed_images = set()
    images.sort(key=lambda x: (x[2], x[3][0], x[3][1]), reverse=True)

    for i in tqdm.tqdm(range(len(images)), desc="comparaison des hachages"):
        if images[i][0] in processed_images:
            continue

        current_duplicates = []
        current_filename, current_hash, _, _ = images[i]

        for j in range(i + 1, len(images)):
            filename, hash, _, _ = images[j]
            distance = current_hash - hash

            if distance <= threshold:
                current_duplicates.append(filename)
                processed_images.add(filename)

        if current_duplicates:
            duplicates[current_filename] = current_duplicates
            processed_images.add(current_filename)

    return duplicates

def process_duplicates(directory, duplicates):
    for ref_image, dups in duplicates.items():
        print(f"Image à conserver : {ref_image}, Doublons : {dups}")

        # On vire les doublons
        for dup in dups:
            dup_path = os.path.join(directory, dup)
            os.remove(dup_path)

            base_name = dup.split('.')[0]
            for json_filename in os.listdir(directory):
                if json_filename.startswith(base_name) and json_filename.endswith('.json'):
                    json_path = os.path.join(directory, json_filename)
                    os.remove(json_path)


def process_all_files(directory):
    for filename in os.listdir(directory):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            filepath = os.path.join(directory, filename)
            try:
                with Image.open(filepath) as img:
                    focus_only_on_dups = 0

                    img_hash = imagehash.phash(img)
                    new_image_name = f"{img_hash}.{filename.split('.')[-1]}"
                    new_image_path = os.path.join(directory, new_image_name)
                    
                    # On ne fait rien si le l'image est déjà normalisée
                    if (filename != new_image_name):
                        print(">>>>>>>>>>>", filename, " ------- ", new_image_name)
                        os.rename(filepath, new_image_path)
    
                        base_name = filename.split('.')[0]
                        pattern = re.compile(rf"^{re.escape(base_name)}_([1-9]\d?|0)\.json$")
                        
                        for json_filename in os.listdir(directory):
                            if pattern.match(json_filename):
                                json_path = os.path.join(directory, json_filename)
                                with open(json_path, 'r') as json_file:
                                    json_data = json.load(json_file)
    
                                json_data = normalize_json(json_data, new_image_name, new_image_path)
    
                                new_json_name = f"{img_hash}_{json_filename.split('_')[-1]}"
                                new_json_path = os.path.join(directory, new_json_name)
                                with open(new_json_path, 'w') as json_file:
                                    json.dump(json_data, json_file, indent=4)
    
                                os.remove(json_path)

            except Exception as e:
                print(f"Error processing {filename}: {e}")

# Exemple d'utilisation
threshold = 1


directory = '/home/aobled/Downloads/Aircraft_DATASET/tmp_a_traiter/ag600'
duplicates = find_duplicates_optimized(directory, threshold)
process_duplicates(directory, duplicates)
process_all_files(directory)

