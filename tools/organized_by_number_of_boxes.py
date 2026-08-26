import os
import shutil

def organiser_par_nombre_boxes(repertoire_source):
    for racine, _, fichiers in os.walk(repertoire_source):
        # Filtrer les fichiers .jpg
        for fichier in fichiers:
            if fichier.endswith('.jpg'):
                nom_base = os.path.splitext(fichier)[0]
                chemin_jpg = os.path.join(racine, fichier)

                # Trouver tous les fichiers .json associés
                json_associes = [f for f in fichiers if f.startswith(nom_base) and f.endswith('.json')]

                # Créer le sous-répertoire en fonction du nombre de boxes
                nb_boxes = len(json_associes)
                sous_repertoire = os.path.join(racine, str(nb_boxes))
                os.makedirs(sous_repertoire, exist_ok=True)

                # Déplacer le .jpg
                shutil.move(chemin_jpg, os.path.join(sous_repertoire, fichier))

                # Déplacer les .json associés
                for json_file in json_associes:
                    chemin_json = os.path.join(racine, json_file)
                    shutil.move(chemin_json, os.path.join(sous_repertoire, json_file))

                print(f"Déplacé {fichier} et {len(json_associes)} boxes dans {sous_repertoire}/")

if __name__ == "__main__":
    repertoire = "/home/aobled/Downloads/tmp_nara/multi/14"
    organiser_par_nombre_boxes(repertoire)
