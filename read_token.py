from flax.traverse_util import flatten_dict
import pickle
import jax.numpy as jnp

with open("best_model_detection.pkl", "rb") as f:
    checkpoint = pickle.load(f)

print(checkpoint.keys())

params = checkpoint['params']['token_bottleneck']

flat = flatten_dict(params)

for k in flat.keys():
    print(k)



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


"""
self attention entre tokens correspond à :
1 batch dimension technique
16 tokens
256 dimensions par token
"""

tokens = params['query_tokens']
print(tokens.shape) # (1, 16, 256)
print(tokens.dtype) # float32
print(tokens[0,0,17])


norms = jnp.linalg.norm(tokens[0], axis=-1)
print(norms)



"""
ENCODAGE / DECODAGE
Solution optimale : Hash SHA-256 + Format .npy
"""
import hashlib
import numpy as np

def get_unique_filename(tokens):
    tokens_bytes = tokens.tobytes()
    hash_object = hashlib.sha256(tokens_bytes)
    return f"tokens_{hash_object.hexdigest()}.npy"

# Sauvegarder le tableau
# Exemple de tableau
#tokens = np.random.rand(1, 16, 256).astype(np.float32)

# Générer un nom de fichier unique
filename = get_unique_filename(tokens)

# Sauvegarder en .npy
np.save(filename, tokens)
print(f"Fichier sauvegardé : {filename}")

# Charger et modifier le tableau
loaded_tokens = np.load(filename)

# Modifier une valeur (exemple)
loaded_tokens[0, 0, 17] = -0.5  # Nouvelle valeur

# Sauvegarder à nouveau (nouveau hash = nouveau nom de fichier)
new_filename = get_unique_filename(loaded_tokens)
np.save(new_filename, loaded_tokens)
print(f"Nouveau fichier : {new_filename}")
