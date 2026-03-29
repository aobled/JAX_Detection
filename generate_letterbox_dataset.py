#!/usr/bin/env python3
"""
Script rapide pour générer le dataset Letterbox 128×128 Grayscale
À lancer en parallèle pendant l'entraînement 224×224
"""

import sys
sys.path.append('./tools')

from fighterjet_classification_dataset_tools import process_dataset_letterbox, balance_and_split_dataset

print("=" * 80)
print("🎯 GÉNÉRATION DATASET LETTERBOX 128×128 GRAYSCALE")
print("=" * 80)

# Paramètres
ROOT_DIR = "/home/aobled/Downloads/Figtherjet_DATASET"
OUTPUT_CROP = "/home/aobled/Downloads/_crop_classification_letterbox"
OUTPUT_BALANCED = "/home/aobled/Downloads/_balanced_dataset_split_letterbox"
TARGET_SIZE = 128
MAX_PER_CLASS = 6580

print(f"\n📋 Configuration:")
print(f"   Source : {ROOT_DIR}")
print(f"   Taille : {TARGET_SIZE}×{TARGET_SIZE}")
print(f"   Mode : Grayscale (1 canal)")
print(f"   Méthode : Letterboxing (ratio préservé + padding miroir)")
print(f"   Max/classe : {MAX_PER_CLASS}")

# Étape 1 : Extraction avec Letterbox
print(f"\n🔄 ÉTAPE 1/2 : Extraction avec Letterboxing...")
print(f"   Output : {OUTPUT_CROP}")
process_dataset_letterbox(
    root_dir=ROOT_DIR,
    output_dir=OUTPUT_CROP,
    target_size=TARGET_SIZE,
    grayscale=True,
    padding_mode='reflect'
)

# Étape 2 : Balance et split
print(f"\n⚖️  ÉTAPE 2/2 : Équilibrage et split train/val...")
print(f"   Output : {OUTPUT_BALANCED}")
balance_and_split_dataset(
    OUTPUT_CROP,
    OUTPUT_BALANCED,
    max_per_class=MAX_PER_CLASS
)

print("\n" + "=" * 80)
print("✅ DATASET LETTERBOX GÉNÉRÉ AVEC SUCCÈS !")
print("=" * 80)
print(f"\n📁 Dataset disponible dans : {OUTPUT_BALANCED}")
print(f"   → train/ : ~{MAX_PER_CLASS * 0.85 * 10:.0f} images")
print(f"   → val/   : ~{MAX_PER_CLASS * 0.15 * 10:.0f} images")

print(f"\n🚀 PROCHAINES ÉTAPES :")
print(f"   1. Générer les chunks :")
print(f"      python main.py FIGHTERJET_LETTERBOX")
print(f"   2. Uploader sur Google Drive (si Colab)")
print(f"   3. Lancer l'entraînement (epochs=30)")
print(f"\n🎯 Accuracy attendue : 86-87% (+2-4% vs stretched)")
print("=" * 80)

