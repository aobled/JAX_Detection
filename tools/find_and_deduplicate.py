import os
import shutil
import tqdm

"""
parcourt le répertoire directory_out, 
recherche les fichiers .png et .jpg, 
puis vérifie si un fichier portant le même nom (sans extension) existe déjà dans les sous-répertoires de directory_in. 
Si c'est le cas, il déplace le fichier de directory_out vers directory_garbage.
"""
def supprimer_images_existantes(directory_out, directory_in, directory_garbage):
    # Créer le répertoire "poubelle" s'il n'existe pas
    if not os.path.exists(directory_garbage):
        os.makedirs(directory_garbage)

    # Parcourir les fichiers dans directory_out
    for filename in tqdm.tqdm(os.listdir(directory_out)):
        if filename.lower().endswith(('.png', '.jpg')):
            # Extraire le nom sans extension
            name_without_ext = os.path.splitext(filename)[0]

            # Parcourir les sous-répertoires de directory_in
            for root, dirs, files in os.walk(directory_in):
                for file in files:
                    if os.path.splitext(file)[0] == name_without_ext:
                        # Si le fichier existe déjà dans directory_in, le déplacer vers la poubelle
                        src_path = os.path.join(directory_out, filename)
                        dst_path = os.path.join(directory_garbage, filename)
                        shutil.move(src_path, dst_path)
                        print(f"Fichier {filename} déplacé vers la poubelle.")
                        break  # Pas besoin de continuer à chercher une fois trouvé

if __name__ == "__main__":
    directory_out = '/home/aobled/Downloads/tmpTyphoon'
    directory_in = '/home/aobled/Downloads/Figtherjet_DATASET'
    directory_garbage = '/home/aobled/Downloads/tmpTyphoon/garbage'

    supprimer_images_existantes(directory_out, directory_in, directory_garbage)
    print("Traitement terminé.")
