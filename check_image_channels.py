#!/usr/bin/env python3
"""
Script pour vérifier le nombre de canaux des images du dataset
"""

import os
from PIL import Image
import numpy as np

def check_images_channels(directory, num_samples=20):
    """
    Vérifie le nombre de canaux de plusieurs images dans un répertoire
    """
    print(f"🔍 Vérification des images dans: {directory}")
    print("=" * 70)
    
    # Trouver des images
    images_found = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                images_found.append(os.path.join(root, file))
                if len(images_found) >= num_samples:
                    break
        if len(images_found) >= num_samples:
            break
    
    if not images_found:
        print(f"❌ Aucune image trouvée dans {directory}")
        return
    
    print(f"📊 Échantillon: {len(images_found)} images")
    print()
    
    # Analyser chaque image
    results = {1: 0, 3: 0, 4: 0}  # 1=Grayscale, 3=RGB, 4=RGBA
    
    for i, img_path in enumerate(images_found[:num_samples], 1):
        try:
            img = Image.open(img_path)
            
            # Mode PIL
            mode = img.mode
            
            # Convertir en array pour vérifier shape
            arr = np.array(img)
            
            # Déterminer le nombre de canaux
            if len(arr.shape) == 2:
                channels = 1  # Grayscale pur
            elif len(arr.shape) == 3:
                channels = arr.shape[2]
            else:
                channels = "?"
            
            results[channels] = results.get(channels, 0) + 1
            
            # Afficher quelques exemples
            if i <= 5:
                rel_path = os.path.relpath(img_path, directory)
                print(f"  {i}. {rel_path[:50]:<50} | Mode: {mode:5} | Shape: {arr.shape} | Canaux: {channels}")
                
                # Si RGB, vérifier si c'est du "faux grayscale" (R=G=B)
                if channels == 3:
                    is_fake_gray = np.allclose(arr[:,:,0], arr[:,:,1]) and np.allclose(arr[:,:,1], arr[:,:,2])
                    if is_fake_gray:
                        print(f"       ⚠️  RGB mais R=G=B (grayscale déguisé)")
        
        except Exception as e:
            print(f"  ❌ Erreur sur {img_path}: {e}")
    
    print()
    print("📊 RÉSUMÉ:")
    print("-" * 70)
    for ch, count in sorted(results.items()):
        if count > 0:
            percentage = (count / len(images_found[:num_samples])) * 100
            channel_name = {1: "Grayscale (1 canal)", 3: "RGB (3 canaux)", 4: "RGBA (4 canaux)"}.get(ch, f"{ch} canaux")
            print(f"  {channel_name}: {count}/{len(images_found[:num_samples])} images ({percentage:.0f}%)")
    
    print()
    print("🎯 CONCLUSION:")
    if results[3] > 0:
        print("  ⚠️  LES IMAGES SONT EN RGB (3 CANAUX) !")
        print("  → Même si elles apparaissent en noir & blanc, elles ont 3 canaux")
        print("  → Pour du vrai grayscale, il faut les convertir avec PIL.convert('L')")
    elif results[1] > 0:
        print("  ✅ LES IMAGES SONT EN VRAI GRAYSCALE (1 CANAL)")
    

def check_chunks_channels(chunks_dir="./data/chunks"):
    """
    Vérifie le nombre de canaux dans les chunks NPZ
    """
    print()
    print("🔍 Vérification des chunks NPZ...")
    print("=" * 70)
    
    chunk_files = [f for f in os.listdir(chunks_dir) if f.endswith('.npz') and 'train' in f]
    
    if not chunk_files:
        print(f"❌ Aucun chunk trouvé dans {chunks_dir}")
        return
    
    # Vérifier le premier chunk
    chunk_path = os.path.join(chunks_dir, chunk_files[0])
    print(f"📦 Chunk: {chunk_files[0]}")
    
    try:
        data = np.load(chunk_path)
        images = data['images']
        print(f"  Shape: {images.shape}")
        print(f"  → Batch: {images.shape[0]}, H: {images.shape[1]}, W: {images.shape[2]}, Canaux: {images.shape[3]}")
        
        if images.shape[3] == 1:
            print("  ✅ CHUNKS EN GRAYSCALE (1 canal)")
        elif images.shape[3] == 3:
            print("  ⚠️  CHUNKS EN RGB (3 canaux)")
            
    except Exception as e:
        print(f"  ❌ Erreur: {e}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Vérifier les canaux des images")
    parser.add_argument("--dir", default="/home/aobled/Downloads/_balanced_dataset_split/train/f15", 
                       help="Répertoire à vérifier")
    parser.add_argument("--samples", type=int, default=20, 
                       help="Nombre d'images à vérifier")
    parser.add_argument("--check_chunks", action="store_true", 
                       help="Vérifier aussi les chunks NPZ")
    
    args = parser.parse_args()
    
    # Vérifier les images sources
    check_images_channels(args.dir, args.samples)
    
    # Vérifier les chunks si demandé
    if args.check_chunks:
        check_chunks_channels()

