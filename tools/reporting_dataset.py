#!/usr/bin/env python3
import os
import json

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
    """
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
    """

if __name__ == "__main__":
    target_root = '/home/aobled/Downloads/Aircraft_DATASET/classification'
    classes = ["a10", "a400m", "alphajet", "b1b", "b2", "c130", "c17", "f4", "f14", "f15", "f16", "f18", "f22", "f35", "f117", "gripen", "harrier", "hawk", "hawkeye", "mig29", "mirage2000", "rafale", "su57", "tornado", "typhoon"]
    
    # Rapport final sur l'état du dataset fixe
    reporting_fixed_dataset(
        target_root=target_root,
        class_names=classes,
        category_key="category_name"
    )


