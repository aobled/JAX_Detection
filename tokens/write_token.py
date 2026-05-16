from flax.traverse_util import flatten_dict
import pickle
import jax.numpy as jnp
import hashlib
import numpy as np

"""
Focus sur les tokens:
    query_tokens
        ↓
cross attention avec spatial features
        ↓
self attention entre tokens
        ↓
réinjection spatiale
"""

def get_unique_filename(tokens):
    tokens_bytes = tokens.tobytes()
    hash_object = hashlib.sha256(tokens_bytes)
    return f"tokens_{hash_object.hexdigest()}"

# 1. Charger le checkpoint
with open("../best_model_detection.pkl", "rb") as f:
    checkpoint = pickle.load(f)

# 2. Extraire les tokens originaux
tokens = checkpoint['params']['token_bottleneck']['query_tokens']
print(tokens)


"""
self attention entre tokens correspond à :
1 batch dimension technique
16 tokens
256 dimensions par token
"""
print(tokens.shape) # (1, 16, 256)
print(tokens.dtype) # float32
print(tokens[0,0,17])



"""
ENCODAGE / DECODAGE
Solution optimale : Hash SHA-256 + Format .npy
"""
# Générer un nom de fichier unique & Sauvegarder en .npy
token_file_origin = get_unique_filename(tokens)+".npy"
np.save(token_file_origin, tokens)
print(f"Fichier sauvegardé : {token_file_origin}")

# 3. Générer de NOUVEAUX tokens aléatoires
#loaded_tokens = np.load(token_file_origin)
#loaded_tokens[0, 0, 17] = -0.5  # Nouvelle valeur
#loaded_tokens = np.random.rand(1, 16, 256).astype(np.float32)  
loaded_tokens = np.load(token_file_origin)
noise = np.random.normal(0, 0.925, size=(256,))
loaded_tokens[0,1] += noise
#loaded_tokens[0, 15] = 0 # test de raz complet d'un token

# Sauvegarder à nouveau (nouveau hash = nouveau nom de fichier)
token_file_updated = get_unique_filename(loaded_tokens)+".npy"
np.save(token_file_updated, loaded_tokens)
print(f"Nouveau fichier : {token_file_updated}")



# 3. Mettre à jour le checkpoint
checkpoint['params']['token_bottleneck']['query_tokens'] = loaded_tokens # Reconstruire la structure
new_pkl = "../best_model_detectionv2.pkl"
with open(new_pkl, "wb") as f:
    pickle.dump(checkpoint, f)
print(f"Checkpoint sauvegardé avec nom unique : {new_pkl}")

