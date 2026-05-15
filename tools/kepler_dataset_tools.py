import pandas as pd
import numpy as np
import pickle
import os
import tqdm

# --- CONFIGURATION ---
CSV_TRAIN_PATH = "/home/aobled/Downloads/kepler/exoTrain.csv"
CSV_TEST_PATH = "/home/aobled/Downloads/kepler/exoTest.csv"

OUTPUT_DIR = "/home/aobled/Desktop/Development/JAX_Detection/data/chunks/kepler" # Relative path
TRAIN_OUTPUT_PREFIX = os.path.join(OUTPUT_DIR, "dataset_kepler_train_chunk")
TEST_OUTPUT_PREFIX = os.path.join(OUTPUT_DIR, "dataset_kepler_val_chunk")

CHUNK_SIZE = 500

def parse_and_save_chunks(csv_path, output_prefix, chunk_size):
    if not os.path.exists(csv_path):
        print(f"❌ Fichier non trouvé: {csv_path}")
        return
        
    os.makedirs(os.path.dirname(output_prefix), exist_ok=True)
    
    print(f"📖 Lecture de {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Kaggle labels: 2 is Exoplanet, 1 is Non-Exoplanet.
    # Convert to 1 (Exoplanet) and 0 (Non-Exoplanet)
    labels = df['LABEL'].values - 1
    
    # Drop LABEL column to get only fluxes
    fluxes = df.drop('LABEL', axis=1).values # Shape (N, 3197)
    
    # Normalisation basique (Min-Max par étoile ou Z-score)
    # Les flux stellaires ont des amplitudes très différentes.
    # Un Z-score par étoile est standard en astronomie:
    means = np.mean(fluxes, axis=1, keepdims=True)
    stds = np.std(fluxes, axis=1, keepdims=True)
    fluxes = (fluxes - means) / (stds + 1e-8)
    
    # Reshape to (N, 3197, 1) for Conv1D compatibility (SequenceLength, Channels)
    # Note: data_management.py utilise la clé 'images', on la conserve.
    fluxes = fluxes[..., np.newaxis]
    
    num_samples = len(labels)
    num_chunks = int(np.ceil(num_samples / chunk_size))
    
    print(f"🚀 Génération de {num_chunks} chunks pour {num_samples} échantillons...")
    
    for i in tqdm.tqdm(range(num_chunks)):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, num_samples)
        
        chunk_images = fluxes[start_idx:end_idx]
        chunk_labels = labels[start_idx:end_idx]
        
        chunk_data = {
            'image': chunk_images.astype(np.float32), 
            'label': chunk_labels.astype(np.int32)
        }
        
        chunk_path = f"{output_prefix}_{i:04d}.npz"
        np.savez_compressed(chunk_path, **chunk_data)
            
    print(f"✅ Terminé pour {csv_path}")

if __name__ == "__main__":
    print("=== Génération des Chunks Kepler (Time-Series) ===")
    parse_and_save_chunks(CSV_TRAIN_PATH, TRAIN_OUTPUT_PREFIX, CHUNK_SIZE)
    parse_and_save_chunks(CSV_TEST_PATH, TEST_OUTPUT_PREFIX, CHUNK_SIZE)
    
    # Générer le fichier mean_std attendu par data_management.py
    # Puisque les données sont déjà standardisées par étoile (Z-score), 
    # la normalisation globale doit être neutre (mean=0, std=1)
    mean_std_path = os.path.join(OUTPUT_DIR, "dataset_kepler_meanstd.npz")
    print(f"🔧 Création du fichier de normalisation neutre: {mean_std_path}")
    np.savez_compressed(mean_std_path, mean=np.array([0.0]), std=np.array([1.0]))
    print("✅ Pipeline Kepler terminé !")
