# Stratégie de Sauvegarde : Inférence Légère vs Entraînement Lourd (.pkl)

Durant l'entraînement de modèles avancés sur TPU (ou GPU), JAX crée et maintient en mémoire deux réalités très différentes :
1. **Les Matrices du Modèle** : Les poids purs (`params`) qui servent à prédire (l'intelligence finale).
2. **L'État de l'Entraînement** : Les historiques d'optimisation (Gradients passés, Moments Adam `mu` et `nu`, Epoch actuelle, Keys RNG JAX). Cet état pesait souvent **trois ou quatre fois plus lourd** que le modèle lui-même !

JAX Vision Framework implémente aujourd'hui une stratégie *Double Sauvegarde* pour gérer ces deux réalités distinctes.

---

## 1. L'Export Pur : L'Inférence Portable (Le Fichier Léger)

### Fonctionnement
Dès que le modèle bat son record de validation (`best_val_acc` max en classification, `loss` mini en détection), le `Trainer` déclenche un Export Pur au format `.pkl` (exemple : `best_model.pkl` ou `best_model_detection.pkl`).

### Contenu
Il s'agit d'un dictionnaire extrêmement propre et minimaliste, prévu pour être envoyé par Internet, chargé sur un Cloud, ou lu par une Raspberry Pi.
Il contient uniquement :
```python
{
    'params': params_cpu,                      # Les Matrices de poids converties pour le CPU
    'batch_stats': batch_stats_cpu,            # Les Moyennes Mobiles des BatchNorms
    'config': {                                # La Carte d'Identité du dataset
        'model_name': 'sophisticated_cnn',
        'num_classes': 8,
        'image_size': [224, 224]
        # ...
    }
}
```

### Usage
- Inférence pure (`bounding_boxes_with_classification_from_images_generation.py`).
- Génération de matrices de confusion post-entraînement dans `main.py` via `reporting.py`.
- **Ce fichier ne contient pas l'optimiseur.** Si on relance un entraînement avec, le Learning Rate sera réinitialisé et les Matrices Adam perdront tout leur élan (*Momentum*), détruisant la dynamique d'apprentissage.

---

## 2. Le TrainState Orbax/Flax : Le Cerveau Entier (Le Checkpoint Lourd)

### Fonctionnement
Parallèlement à l'export brut vu plus haut, le système s'appuie sur la classe `CheckpointManager` pour sauvegarder l'**intégralité de la RAM JAX** nécessaire à une reprise parfaite sur TPU, sans aucune altération mathématique de la courbe de *Loss*.

Ces fichiers lourds portent désormais le suffixe `_training_state.pkl` (exemple: `best_model_training_state.pkl`).

### Contenu
Il compresse l'entièreté du point d'entrée d'Optax/Flax :
```python
{
    'model_state': {
        'params': params,
        'batch_stats': batch_stats,
        'step': 14200,                        # Le nombre exact d'itérations passées
        'opt_state': opt_state                # Les moments de l'optimiseur AdamW
    },
    'training_state': {
        'best_val_acc': 0.957,                # Pour ne pas écraser une meilleure epoch à la reprise
        'patience_counter': 3,                # État de l'Early Stopping
        'epoch': 28,                          # Où s'est arrêté le compteur
        'rng': jnp.array([...])               # État du Générateur Aléatoire de JAX (Crucial)
    },
    'model_info': {
        'model_name': 'sophisticated_cnn',
        'num_classes': 8
    }
}
```

### Usage
- Résilience aux crashes sur Google Colab.
- Reprise longue durée des modèles de détection (qui ont besoin de 48 heures de TPU).
- **C'est ce fichier (et uniquement lui)** que `main.py` détecte automatiquement lors de sa routine de lancement : `if resume_from_checkpoint and self.checkpoint_manager.exists(): ...`

---

## Conclusion et Ségrégation

Auparavant, ces deux entités entraient en conflit : le `CheckpointManager` sauvegardait le mastodonte sous le nom de `best_model.pkl`, qui se faisait instantanément écraser une micro-seconde plus tard par l'Export Pur (le fichier de 12 Mo). Par conséquent, lors d'une tentative de reprise d'entraînement, le système lisait le fichier léger, cherchait la clé `"training_state"` (qui n'existe plus), et crashait !

Désormais, **la ségrégation est stricte** :
1. `best_model_classification.pkl` (Léger, pour vos scripts d'inférence Client).
2. `best_model_training_state.pkl` (Lourd, caché en arrière plan pour sauver la progression du TPU entre deux sessions Colab).
