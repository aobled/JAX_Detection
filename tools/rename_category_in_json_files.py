import os
import json

def rename_category_in_json_files(directory, origin, target):
    """
    Remplace récursivement la valeur 'origin' par 'target' dans l'attribut 'category_name'
    de tous les fichiers JSON du répertoire et de ses sous-répertoires.
    """
    for root, _, files in os.walk(directory):
        for filename in files:
            if filename.endswith('.json'):
                filepath = os.path.join(root, filename)
                with open(filepath, 'r') as f:
                    data = json.load(f)

                # Vérifie et remplace si nécessaire
                if "annotation" in data and data["annotation"].get("category_name") == origin:
                    data["annotation"]["category_name"] = target
                    with open(filepath, 'w') as f:
                        json.dump(data, f, indent=2)
                    print(f"Mis à jour : {filepath} ({origin} → {target})")

# Exemple d'utilisation
directory = "/home/aobled/Downloads/Aircraft_DATASET/tmp_a_traiter/jas39"
origin = "jas39"
target = "gripen"

rename_category_in_json_files(directory, origin, target)