"""
Normalise les bounding boxes ([x, y, w, h]) des fichiers JSON d'annotation d'un
répertoire - retour utilisateur 2026-07-24 (voir tools/reporting_dataset_pandas.py
::reporting_incoherent_boxes) : plusieurs scripts de dataset_builder/ produisent des
coordonnées non arrondies et parfois hors image (ex. x négatif de -0.0003px après
conversion). Inspiré de tools/rename_category_in_json_files.py (même style de
parcours récursif, même façon d'écrire le JSON).
"""
import os
import json
import math


def normalize_bbox(bbox, image_width, image_height):
    """
    Normalise une boîte [x, y, w, h] : coin haut-gauche (x1, y1) arrondi vers le bas
    et clampé à 0 si négatif ; coin bas-droit (x2, y2 = x+w, y+h) arrondi vers le
    haut et clampé à la largeur/hauteur de l'image. Retourne une nouvelle liste
    [x, y, w, h] (entiers), ou None si la boîte était déjà normalisée.
    """
    x, y, w, h = bbox
    x2 = x + w
    y2 = y + h

    new_x1 = max(0, math.floor(x))
    new_y1 = max(0, math.floor(y))
    new_x2 = min(image_width, math.ceil(x2))
    new_y2 = min(image_height, math.ceil(y2))

    new_bbox = [new_x1, new_y1, new_x2 - new_x1, new_y2 - new_y1]
    if new_bbox == [x, y, w, h]:
        return None
    return new_bbox


def normalize_boxes_in_json_files(directory):
    """
    Parcourt récursivement `directory` et normalise le bbox de chaque fichier JSON
    d'annotation. Nécessite data['image']['width']/['height'] pour clamper le coin
    bas-droit - fichiers sans bbox ou sans dimensions d'image ignorés (avertissement).
    """
    updated = 0
    skipped = 0
    for root, _, files in os.walk(directory):
        for filename in files:
            if not filename.endswith('.json'):
                continue
            filepath = os.path.join(root, filename)
            with open(filepath, 'r') as f:
                data = json.load(f)

            image_info = data.get('image', {})
            image_width = image_info.get('width')
            image_height = image_info.get('height')
            bbox = data.get('annotation', {}).get('bbox')

            if not bbox or len(bbox) != 4 or not image_width or not image_height:
                print(f"⚠️  Ignoré (bbox ou dimensions manquantes) : {filepath}")
                skipped += 1
                continue

            new_bbox = normalize_bbox(bbox, image_width, image_height)
            if new_bbox is not None:
                data['annotation']['bbox'] = new_bbox
                with open(filepath, 'w') as f:
                    json.dump(data, f, indent=4, ensure_ascii=False)
                print(f"Normalisé : {filepath} ({bbox} → {new_bbox})")
                updated += 1

    print(f"\n{updated} fichier(s) normalisé(s), {skipped} ignoré(s).")


if __name__ == "__main__":
    directory = "/home/aobled/Downloads/Aircraft_DATASET/"
    normalize_boxes_in_json_files(directory)
