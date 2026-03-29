#!/usr/bin/env python3
import os
import re
import json
import argparse
import shutil
from collections import defaultdict
from tqdm import tqdm


def count_boxes_in_json(json_path: str) -> int:
    """Compte le nombre de bounding boxes dans un fichier JSON"""
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Format standard: {"annotation": {"bbox": [x, y, w, h], ...}}
        if isinstance(data, dict):
            if "annotation" in data and "bbox" in data["annotation"]:
                bbox = data["annotation"]["bbox"]
                if isinstance(bbox, list) and len(bbox) == 4:
                    return 1  # Une seule box dans ce JSON
            
            # Format alternatif: liste de boxes
            if "boxes" in data and isinstance(data["boxes"], list):
                return len(data["boxes"])
            
            # Format alternatif: {"bbox": [...]}
            if "bbox" in data and isinstance(data["bbox"], list):
                if len(data["bbox"]) == 4:
                    return 1
                # Si c'est une liste de listes
                if len(data["bbox"]) > 0 and isinstance(data["bbox"][0], list):
                    return len(data["bbox"])
    except Exception:
        pass
    return 0


def find_all_jsons_for_image(image_path: str) -> list:
    """Trouve tous les fichiers JSON associés à une image (tous formats)"""
    base, _ = os.path.splitext(image_path)
    dir_name = os.path.dirname(image_path)
    stem = os.path.basename(base)
    
    # Cherche tous les JSONs potentiellement associés :
    # 1. Format <stem>_<n>.json (ex: image_0.json, image_1.json)
    # 2. Format <stem>.json (ex: image.json)
    jsons = []
    if os.path.exists(dir_name):
        for name in os.listdir(dir_name):
            if not name.lower().endswith('.json'):
                continue
            
            # Vérifier si c'est un JSON associé à cette image
            name_stem, _ = os.path.splitext(name)
            if name_stem == stem:  # image.json
                jsons.append(os.path.join(dir_name, name))
            elif name_stem.startswith(stem + "_"):  # image_0.json, image_1.json
                # Vérifier que c'est bien le format image_<n>.json
                if re.match(rf"^{re.escape(stem)}_\d+$", name_stem):
                    jsons.append(os.path.join(dir_name, name))
    
    return sorted(jsons)


def find_box_jsons_for_image(image_path: str) -> list:
    """Trouve tous les fichiers JSON associés à une image et vérifie qu'ils contiennent chacun 1 box"""
    jsons = find_all_jsons_for_image(image_path)
    
    # Ne garder que les JSONs qui contiennent exactement 1 box
    valid_jsons = []
    for json_path in jsons:
        if count_boxes_in_json(json_path) == 1:
            valid_jsons.append(json_path)
    
    return sorted(valid_jsons)


def read_category_from_json(json_path: str, category_key: str = "category_name") -> str:
    """Lit la catégorie depuis data["annotation"][category_key] ou fallbacks"""
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if not isinstance(data, dict):
            return ""

        # Catégorie toujours dans data["annotation"]
        annotation = data.get("annotation", {})
        if isinstance(annotation, dict):
            # D'abord essayer avec la clé demandée
            if category_key in annotation:
                return str(annotation[category_key])
            # Fallbacks dans annotation si category_key n'existe pas
            for key in ("category_name", "category", "label", "class"):
                if key in annotation:
                    return str(annotation[key])
    except Exception:
        pass
    return ""


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def check_all_same_class(jsons: list, class_set: set, category_key: str = "category_name") -> tuple:
    """
    Vérifie si tous les JSONs (hors "unknown") sont de la même classe.
    Retourne (True, class_name) si oui, (False, None) sinon.
    """
    if len(jsons) == 0:
        return (False, None)
    
    classes_found = []
    for json_path in jsons:
        category = read_category_from_json(json_path, category_key).strip().lower()
        # Ignorer "unknown" dans la vérification
        if category and category != "unknown":
            classes_found.append(category)
    
    # Si aucune classe non-unknown, on ne peut pas déplacer
    if len(classes_found) == 0:
        return (False, None)
    
    # Vérifier que toutes les classes sont identiques
    unique_classes = set(classes_found)
    if len(unique_classes) != 1:
        return (False, None)
    
    # Vérifier que la classe est dans class_set
    class_name = unique_classes.pop()
    if class_name not in class_set:
        return (False, None)
    
    return (True, class_name)


def move_all_files(image_path: str, json_paths: list, dest_dir: str) -> None:
    """Déplace (pas copie) l'image et tous ses JSONs vers dest_dir"""
    ensure_dir(dest_dir)
    
    dest_image = os.path.join(dest_dir, os.path.basename(image_path))
    
    # Vérifier que l'image existe
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image source introuvable: {image_path}")
    
    # Déplacer l'image
    try:
        shutil.move(image_path, dest_image)
        
        # Vérification : le fichier source doit avoir disparu
        if os.path.exists(image_path):
            print(f"⚠️  ATTENTION: {image_path} existe encore après shutil.move()")
            if os.path.exists(dest_image):
                os.remove(image_path)
    except Exception as e:
        raise RuntimeError(f"Erreur lors du déplacement de {image_path}: {e}") from e
    
    # Déplacer tous les JSONs
    for json_path in json_paths:
        if not os.path.exists(json_path):
            print(f"⚠️  JSON introuvable, ignoré: {json_path}")
            continue
        
        dest_json = os.path.join(dest_dir, os.path.basename(json_path))
        try:
            shutil.move(json_path, dest_json)
            
            # Vérification : le fichier source doit avoir disparu
            if os.path.exists(json_path):
                print(f"⚠️  ATTENTION: {json_path} existe encore après shutil.move()")
                if os.path.exists(dest_json):
                    os.remove(json_path)
        except Exception as e:
            print(f"⚠️  Erreur lors du déplacement de {json_path}: {e}")


def move_pair(image_path: str, json_path: str, dest_dir: str) -> None:
    """Déplace (pas copie) l'image et son JSON vers dest_dir"""
    ensure_dir(dest_dir)
    
    dest_image = os.path.join(dest_dir, os.path.basename(image_path))
    dest_json = os.path.join(dest_dir, os.path.basename(json_path))
    
    # Vérifier que les fichiers source existent
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image source introuvable: {image_path}")
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"JSON source introuvable: {json_path}")
    
    # Déplacer avec shutil.move (supprime le fichier source après copie)
    try:
        shutil.move(image_path, dest_image)
        shutil.move(json_path, dest_json)
        
        # Vérification : les fichiers source doivent avoir disparu
        if os.path.exists(image_path):
            print(f"⚠️  ATTENTION: {image_path} existe encore après shutil.move()")
            # Supprimer manuellement si copie réussie mais source non supprimée
            if os.path.exists(dest_image):
                os.remove(image_path)
        if os.path.exists(json_path):
            print(f"⚠️  ATTENTION: {json_path} existe encore après shutil.move()")
            if os.path.exists(dest_json):
                os.remove(json_path)
                
    except Exception as e:
        # Si le déplacement échoue, essayer de supprimer ce qui a pu être copié
        if os.path.exists(dest_image) and not os.path.exists(image_path):
            # L'image a été déplacée mais le JSON a échoué, restaurer
            shutil.move(dest_image, image_path)
        raise RuntimeError(f"Erreur lors du déplacement de {image_path}: {e}") from e


def should_send_to_val(count_train: int, count_val: int, val_ratio: float) -> bool:
    total = count_train + count_val
    # Place en val si la proportion actuelle est en-dessous de la cible
    if total == 0:
        return False  # commence par train par défaut
    return (count_val * 1.0) / total < val_ratio


def build_fixed_dataset(
    source_root: str,
    target_root: str,
    class_names: list,
    val_ratio: float = 0.15,
    exts: tuple = (".jpg", ".jpeg", ".png", ".bmp"),
    category_key: str = "category_name",
):
    class_set = set(class_names)

    counts = {
        cls: {"train": 0, "val": 0, "total": 0}
        for cls in class_names
    }
    skipped_multi_class = 0  # Images avec plusieurs classes différentes
    skipped_no_class = 0  # Images sans classe valide
    skipped_no_json = 0  # Images sans annotation
    processed = 0

    # Collecte des images (récursif) pour une barre de progression fiable
    all_images = []
    for root, _, files in tqdm(os.walk(source_root), desc="Scan images", unit="dir"):
        for fname in files:
            if os.path.splitext(fname)[1].lower() in exts:
                all_images.append(os.path.join(root, fname))

    for image_path in tqdm(all_images, desc="Déplacement images mono-classe", unit="img"):
        # Trouver tous les JSONs associés à cette image
        all_jsons = find_all_jsons_for_image(image_path)
        
        if len(all_jsons) == 0:
            skipped_no_json += 1
            continue  # pas d'annotation, on ignore
        
        # Vérifier que chaque JSON contient exactement 1 box
        valid_jsons = []
        for json_path in all_jsons:
            if count_boxes_in_json(json_path) == 1:
                valid_jsons.append(json_path)
        
        if len(valid_jsons) == 0:
            skipped_no_json += 1
            continue  # aucun JSON valide (pas exactement 1 box)
        
        all_jsons = valid_jsons
        
        # Vérifier que tous les JSONs (hors "unknown") sont de la même classe
        is_valid, category = check_all_same_class(all_jsons, class_set, category_key)
        
        if not is_valid:
            # Vérifier si on a au moins une classe valide (pour distinguer multi-classe vs pas de classe)
            has_valid_class = False
            for json_path in all_jsons:
                cat = read_category_from_json(json_path, category_key).strip().lower()
                if cat and cat != "unknown" and cat in class_set:
                    has_valid_class = True
                    break
            
            if has_valid_class:
                # Plusieurs classes différentes
                skipped_multi_class += 1
            else:
                # Aucune classe valide trouvée (que "unknown" ou pas de classe)
                skipped_no_class += 1
            continue
        
        # Mise à jour compteurs globaux
        counts[category]["total"] += 1

        # Choix du split par classe (85/15)
        ctrain = counts[category]["train"]
        cval = counts[category]["val"]
        to_val = should_send_to_val(ctrain, cval, val_ratio)
        subset = "val" if to_val else "train"

        dest_dir = os.path.join(target_root, subset, category)
        move_all_files(image_path, all_jsons, dest_dir)

        counts[category][subset] += 1
        processed += 1

    # Résumé
    print("\n================ RÉSUMÉ =================")
    print(f"Target: {target_root}")
    print(f"Classes: {len(class_names)} | Ratio val: {int(val_ratio*100)}%")
    print(f"Images déplacées (mono-classe): {processed}")
    print(f"Ignorées (sans annotation): {skipped_no_json}")
    print(f"Ignorées (multi-classe): {skipped_multi_class}")
    print(f"Ignorées (pas de classe valide): {skipped_no_class}")
    print("-----------------------------------------")
    total_train = total_val = total_total = 0
    for cls in class_names:
        t = counts[cls]["train"]
        v = counts[cls]["val"]
        tot = counts[cls]["total"]
        total_train += t
        total_val += v
        total_total += tot
        print(f"{cls:12s}  total={tot:5d}  train={t:5d}  val={v:5d}")
    print("-----------------------------------------")
    print(f"GLOBAL         total={total_total:5d}  train={total_train:5d}  val={total_val:5d}")


def reporting_fixed_dataset(target_root: str, class_names: list = None, category_key: str = "category_name"):
    """
    Parcourt TOUS les JSONs dans le répertoire cible et lit leur contenu pour compter les boxes par classe.
    Affiche un premier tableau avec les classes cibles, puis un deuxième avec toutes les autres classes trouvées.
    """
    if not os.path.exists(target_root):
        print(f"⚠️  Répertoire cible introuvable: {target_root}")
        return
    
    train_dir = os.path.join(target_root, "train")
    val_dir = os.path.join(target_root, "val")
    
    # Dictionnaires pour compter les boxes par classe (train/val)
    # Structure: {class_name: {"train": count, "val": count}}
    class_counts = {}
    
    # Parcourir tous les JSONs dans train/
    if os.path.exists(train_dir):
        for root, dirs, files in os.walk(train_dir):
            for filename in files:
                if filename.lower().endswith('.json'):
                    json_path = os.path.join(root, filename)
                    category = read_category_from_json(json_path, category_key).strip().lower()
                    if category:
                        if category not in class_counts:
                            class_counts[category] = {"train": 0, "val": 0}
                        class_counts[category]["train"] += 1
    
    # Parcourir tous les JSONs dans val/
    if os.path.exists(val_dir):
        for root, dirs, files in os.walk(val_dir):
            for filename in files:
                if filename.lower().endswith('.json'):
                    json_path = os.path.join(root, filename)
                    category = read_category_from_json(json_path, category_key).strip().lower()
                    if category:
                        if category not in class_counts:
                            class_counts[category] = {"train": 0, "val": 0}
                        class_counts[category]["val"] += 1
    
    # Séparer les classes cibles des autres classes
    if class_names is None:
        # Si aucune classe cible fournie, toutes sont considérées comme cibles
        target_classes_set = set(class_counts.keys())
        other_classes_set = set()
    else:
        target_classes_set = {c.lower() for c in class_names}
        other_classes_set = set(class_counts.keys()) - target_classes_set
    
    # === PREMIER TABLEAU : Classes cibles ===
    print("\n" + "=" * 75)
    print("📊 RAPPORT DU DATASET FIXE - CLASSES CIBLES")
    print("=" * 75)
    print(f"Répertoire cible: {target_root}")
    print("-" * 75)
    print(f"{'Classe':<15s} {'Train':>8s} {'Val':>8s} {'Total':>8s} {'Ratio':>8s}")
    print("-" * 75)
    
    target_total_train = 0
    target_total_val = 0
    target_total_global = 0
    
    # Afficher les classes cibles dans l'ordre de class_names si fourni, sinon alphabétique
    if class_names:
        target_classes_ordered = [c for c in class_names if c.lower() in target_classes_set]
    else:
        target_classes_ordered = sorted(target_classes_set)
    
    for cls in target_classes_ordered:
        cls_lower = cls.lower()
        if cls_lower in class_counts:
            train = class_counts[cls_lower]["train"]
            val = class_counts[cls_lower]["val"]
            tot = train + val
            ratio = (val / tot * 100) if tot > 0 else 0.0
            print(f"{cls:<15s} {train:>8d} {val:>8d} {tot:>8d} {ratio:>7.1f}%")
            target_total_train += train
            target_total_val += val
            target_total_global += tot
        else:
            # Classe cible mais aucune box trouvée
            print(f"{cls:<15s} {0:>8d} {0:>8d} {0:>8d} {0.0:>7.1f}%")
    
    print("-" * 75)
    global_ratio = (target_total_val / target_total_global * 100) if target_total_global > 0 else 0.0
    print(f"{'TOTAL':<15s} {target_total_train:>8d} {target_total_val:>8d} {target_total_global:>8d} {global_ratio:>7.1f}%")
    print("=" * 75)
    
    # === DEUXIÈME TABLEAU : Autres classes trouvées ===
    if other_classes_set:
        print("\n" + "=" * 75)
        print("📊 AUTRES CLASSES TROUVÉES")
        print("=" * 75)
        print("-" * 75)
        print(f"{'Classe':<15s} {'Train':>8s} {'Val':>8s} {'Total':>8s} {'Ratio':>8s}")
        print("-" * 75)
        
        other_total_train = 0
        other_total_val = 0
        other_total_global = 0
        
        for cls in sorted(other_classes_set):
            train = class_counts[cls]["train"]
            val = class_counts[cls]["val"]
            tot = train + val
            ratio = (val / tot * 100) if tot > 0 else 0.0
            print(f"{cls:<15s} {train:>8d} {val:>8d} {tot:>8d} {ratio:>7.1f}%")
            other_total_train += train
            other_total_val += val
            other_total_global += tot
        
        print("-" * 75)
        other_global_ratio = (other_total_val / other_total_global * 100) if other_total_global > 0 else 0.0
        print(f"{'TOTAL':<15s} {other_total_train:>8d} {other_total_val:>8d} {other_total_global:>8d} {other_global_ratio:>7.1f}%")
        print("=" * 75)
    else:
        print("\n✅ Aucune autre classe trouvée (uniquement les classes cibles)")


def parse_args():
    p = argparse.ArgumentParser(description="Construire un dataset fixe (train/val) avec images à box unique.")
    p.add_argument("--source_root", default='/home/aobled/Downloads/Figtherjet_DATASET', help="Racine des images/annotations (scan récursif)")
    p.add_argument("--target_root", default='/home/aobled/Downloads/Aircraft_DATASET/classification', help="Dossier cible AIRCRAFT_DATASET avec train/ et val/")
    p.add_argument("--val_ratio", type=float, default=0.15, help="Ratio validation (défaut 0.15)")
    p.add_argument("--classes", nargs='*', default=[
        "a10","c17","f14","f15","f16","f18","f22","f35","mirage2000","rafale","typhoon"
    ], help="Liste des classes cibles (défaut: 11 classes fighterjet)")
    p.add_argument("--category_key", default="category_name", help="Clé JSON pour la classe (défaut: category_name ; fallbacks: category/label/class)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    """build_fixed_dataset(
        source_root=args.source_root,
        target_root=args.target_root,
        class_names=args.classes,
        val_ratio=args.val_ratio,
        category_key=args.category_key,
    )"""
    # Rapport final sur l'état du dataset fixe
    reporting_fixed_dataset(
        target_root=args.target_root,
        class_names=args.classes,
        category_key=args.category_key,
    )


