"""
Persistance JSON pour tools/boxes_process_manual_tkinter.py (extrait de
PhotoViewer, Étape 2 du refactor du 2026-07-24 - voir
refactor-boxes-process-manual-tkinter.md). Fonctions pures (aucun état) -
comportement identique à l'implémentation inline précédente, seulement
déplacé hors de PhotoViewer.
"""
import json


def validate_and_fix_bbox_coordinates(data, image_width, image_height):
    """
    Valide et corrige les coordonnées des boîtes pour s'assurer qu'elles sont dans les limites de l'image.
    Retourne les données corrigées.
    """
    if 'annotation' in data and 'bbox' in data['annotation']:
        bbox = data['annotation']['bbox']
        if len(bbox) == 4:
            x, y, w, h = bbox

            # Corriger les coordonnées négatives
            x = max(0, x)
            y = max(0, y)

            # Corriger les dimensions négatives
            w = max(1, w)  # Largeur minimale de 1 pixel
            h = max(1, h)  # Hauteur minimale de 1 pixel

            # S'assurer que la boîte ne dépasse pas les limites de l'image
            if x + w > image_width:
                w = max(1, image_width - x)
            if y + h > image_height:
                h = max(1, image_height - y)

            # Mettre à jour les coordonnées corrigées
            data['annotation']['bbox'] = [x, y, w, h]

            print(f"[✓] Coordonnées corrigées : [{x:.1f}, {y:.1f}, {w:.1f}, {h:.1f}]")

    return data


def ensure_json_consistency(data, image_name, image_width, image_height):
    """
    Assure la cohérence du JSON en ajoutant les champs manquants et en validant les coordonnées.
    """
    # S'assurer que la section 'image' existe et est complète
    if 'image' not in data:
        data['image'] = {}

    data['image']['file_name'] = image_name
    data['image']['width'] = image_width
    data['image']['height'] = image_height

    # S'assurer que la section 'annotation' existe et est complète
    if 'annotation' not in data:
        data['annotation'] = {}

    data['annotation']['file_name'] = image_name

    # Ajouter bbox_id s'il n'existe pas
    if 'bbox_id' not in data['annotation']:
        # Essayer d'extraire l'ID du nom de fichier
        try:
            bbox_id = int(data.get('annotation', {}).get('bbox_id', 0))
        except (ValueError, TypeError):
            bbox_id = 0
        data['annotation']['bbox_id'] = bbox_id

    # Valider et corriger les coordonnées des boîtes
    data = validate_and_fix_bbox_coordinates(data, image_width, image_height)

    return data


def save_json_with_consistency_check(file_path, data, image_name, image_width, image_height):
    """
    Sauvegarde un JSON avec vérification de cohérence et formatage correct.
    """
    # Assurer la cohérence du JSON
    data = ensure_json_consistency(data, image_name, image_width, image_height)

    # Sauvegarder avec formatage correct
    json_string = json.dumps(data, indent=4, ensure_ascii=False, separators=(',', ': '))
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(json_string)
        f.write('\n')  # Ajouter un retour à la ligne final
