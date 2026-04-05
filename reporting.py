"""
Module de reporting et visualisation des résultats d'entraînement
Adapté pour l'architecture JAX actuelle
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator
from scipy.integrate import simpson
import jax
import jax.numpy as jnp
from tqdm import tqdm
import pickle
import json
from model_library import get_model


class Reporter:
    """Classe pour générer des rapports et visualisations des modèles entraînés"""
    
    def __init__(self, class_names=None):
        """
        Initialise le reporter
        
        Args:
            class_names: Liste des noms de classes pour affichage dans la matrice de confusion
        """
        self.class_names = class_names or []
    
    def confusion_matrix_from_pkl(self, dataset, pkl_path, confusion_matrix_png_path, 
                                 use_subset=True, batch_size=32, max_subset=3000):
        """
        Crée une matrice de confusion du modèle chargé depuis un .pkl sur un dataset donné.
        Version optimisée pour JAX et votre architecture actuelle.
        
        Args:
            dataset: Dataset de validation (dict avec 'image' et 'label')
            pkl_path: Chemin vers le fichier .pkl du modèle sauvegardé
            confusion_matrix_png_path: Chemin de sauvegarde de la matrice
            use_subset: Utiliser un sous-ensemble pour les gros datasets
            batch_size: Taille des batches pour les prédictions
            max_subset: Nombre maximum d'échantillons pour le sous-ensemble
            
        Returns:
            tuple: (confusion_matrix, accuracy, macro_precision, macro_recall, macro_f1)
        """
        print("🔍 Chargement du modèle depuis le fichier .pkl...")
        
        # Charger le modèle sauvegardé
        with open(pkl_path, 'rb') as f:
            model_data = pickle.load(f)
        
        # Extraire les données du modèle (structure de main.py)
        if 'config' in model_data:
            # Nouvelle structure unifiée JAX_Detection
            params = model_data['params']
            batch_stats = model_data.get('batch_stats', {})
            model_name = model_data['config'].get('model_name', 'sophisticated_cnn')
            num_classes = model_data['config'].get('num_classes', len(self.class_names))
        elif 'model_state' in model_data:
            # Ancienne Structure de main.py
            model_state = model_data['model_state']
            params = model_state['params']
            batch_stats = model_state.get('batch_stats', {})
            
            # Utiliser les infos du modèle sauvegardé
            if 'model_info' in model_data:
                model_info = model_data['model_info']
                model_name = model_info.get('model_name', 'sophisticated_cnn')
                num_classes = model_info.get('num_classes', 2)
            else:
                # Fallback pour anciens fichiers
                model_name = model_data.get('model_name', 'sophisticated_cnn')
                num_classes = model_data.get('num_classes', 2)
        else:
            # Structure alternative (très ancienne)
            params = model_data['params']
            batch_stats = model_data.get('batch_stats', {})
            model_name = model_data.get('model_name', 'sophisticated_cnn')
            num_classes = model_data.get('num_classes', 2)
        
        print(f"🔍 Modèle sauvegardé: {model_name}, {num_classes} classes")
        print(f"⚠️  ATTENTION: Le fichier .pkl doit correspondre au modèle {model_name} avec {num_classes} classes")
        
        # Créer le modèle
        model = get_model(model_name, num_classes=num_classes, dropout_rate=0.0)
        
        # Collecter toutes les données du dataset TensorFlow
        all_images = []
        all_labels = []
        
        print("📊 Collecte des données du dataset...")
        for batch in dataset.as_numpy_iterator():
            batch_images, batch_labels = batch
            all_images.append(batch_images)
            all_labels.append(batch_labels)
        
        # Concaténer tous les batches
        images = np.concatenate(all_images, axis=0)
        labels = np.concatenate(all_labels, axis=0)
        
        print(f"📊 Dataset: {len(images)} images, {num_classes} classes")
        
        # Sous-ensemble intelligent pour les gros datasets
        if use_subset and len(images) > max_subset:
            num_total = len(images)
            num_subset = min(max_subset, num_total // 3)
            np.random.seed(42)  # Fixe pour reproductibilité
            subset_indices = np.random.choice(num_total, num_subset, replace=False)
            images = images[subset_indices]
            labels = labels[subset_indices]
            print(f"📉 Sous-ensemble: {len(images)} images sélectionnées")
        
        # Variables du modèle
        variables = {'params': params, 'batch_stats': batch_stats}
        dummy_rng = jax.random.PRNGKey(0)
        
        # Prédictions par batches
        all_logits = []
        print("🔮 Génération des prédictions...")
        
        for i in tqdm(range(0, len(images), batch_size), desc='Prédictions'):
            batch_images = jnp.array(images[i:i+batch_size], dtype=jnp.float32)
            
            # Appliquer le modèle en mode déterministe
            logits = model.apply(
                variables, 
                batch_images, 
                training=False,  # Mode évaluation
                rngs={'dropout': dummy_rng}
            )
            all_logits.append(np.array(logits))
        
        # Concaténer tous les logits
        all_logits = np.concatenate(all_logits, axis=0)
        predictions = np.argmax(all_logits, axis=-1)
        
        # Créer la matrice de confusion
        confusion_matrix = np.zeros((num_classes, num_classes), dtype=int)
        for true_label, pred_label in zip(labels, predictions):
            confusion_matrix[true_label, pred_label] += 1
        
        # Créer la visualisation
        self._plot_confusion_matrix(
            confusion_matrix, 
            confusion_matrix_png_path,
            num_classes
        )
        
        # Calculer les métriques
        accuracy = np.trace(confusion_matrix) / np.sum(confusion_matrix)
        precision = np.diag(confusion_matrix) / np.sum(confusion_matrix, axis=0)
        recall = np.diag(confusion_matrix) / np.sum(confusion_matrix, axis=1)
        
        # Éviter la division par zéro
        precision = np.nan_to_num(precision, nan=0.0)
        recall = np.nan_to_num(recall, nan=0.0)
        
        f1_score = 2 * (precision * recall) / (precision + recall)
        f1_score = np.nan_to_num(f1_score, nan=0.0)
        
        # Métriques globales
        macro_precision = np.mean(precision)
        macro_recall = np.mean(recall)
        macro_f1 = np.mean(f1_score)
        
        # Afficher les résultats
        print(f"\n📈 RÉSULTATS DE LA MATRICE DE CONFUSION:")
        print(f"Accuracy globale: {accuracy:.4f}")
        print(f"Macro Precision: {macro_precision:.4f}")
        print(f"Macro Recall: {macro_recall:.4f}")
        print(f"Macro F1: {macro_f1:.4f}")
        
        # Détails par classe
        print(f"\n📊 DÉTAILS PAR CLASSE:")
        for i in range(num_classes):
            class_name = self.class_names[i] if i < len(self.class_names) else f"Classe {i}"
            print(f"{class_name}: Precision={precision[i]:.3f}, Recall={recall[i]:.3f}, F1={f1_score[i]:.3f}")
        
        return confusion_matrix, accuracy, macro_precision, macro_recall, macro_f1
    
    def _plot_confusion_matrix(self, confusion_matrix, save_path, num_classes):
        """Crée et sauvegarde la visualisation de la matrice de confusion"""
        
        # Créer la figure
        plt.figure(figsize=(10, 8))
        
        # Noms des classes
        if self.class_names and len(self.class_names) >= num_classes:
            class_names = self.class_names[:num_classes]
        else:
            class_names = [f"Classe {i}" for i in range(num_classes)]
        
        # Afficher la matrice
        im = plt.imshow(confusion_matrix, cmap='Blues', interpolation='nearest')
        plt.colorbar(im, fraction=0.046, pad=0.04)
        
        # Ajouter les annotations
        for i in range(num_classes):
            for j in range(num_classes):
                value = confusion_matrix[i, j]
                color = 'white' if value > confusion_matrix.max() / 2 else 'black'
                plt.text(j, i, str(value), ha='center', va='center', 
                        color=color, fontsize=10, fontweight='bold')
        
        # Configuration des axes
        plt.xticks(range(num_classes), class_names, rotation=45, ha='right')
        plt.yticks(range(num_classes), class_names)
        plt.xlabel('Prédictions', fontsize=12)
        plt.ylabel('Vraies classes', fontsize=12)
        plt.title('Matrice de Confusion', fontsize=14, fontweight='bold')
        
        # Calculer et afficher les métriques
        accuracy = np.trace(confusion_matrix) / np.sum(confusion_matrix)
        precision = np.diag(confusion_matrix) / np.sum(confusion_matrix, axis=0)
        recall = np.diag(confusion_matrix) / np.sum(confusion_matrix, axis=1)
        
        precision = np.nan_to_num(precision, nan=0.0)
        recall = np.nan_to_num(recall, nan=0.0)
        
        f1_score = 2 * (precision * recall) / (precision + recall)
        f1_score = np.nan_to_num(f1_score, nan=0.0)
        
        macro_precision = np.mean(precision)
        macro_recall = np.mean(recall)
        macro_f1 = np.mean(f1_score)
        
        # Ajouter les métriques sur le graphique
        metrics_text = (f'Accuracy: {accuracy:.3f}\n'
                       f'Macro Precision: {macro_precision:.3f}\n'
                       f'Macro Recall: {macro_recall:.3f}\n'
                       f'Macro F1: {macro_f1:.3f}')
        
        plt.figtext(0.02, 0.02, metrics_text, fontsize=10, 
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"💾 Matrice de confusion sauvegardée: {save_path}")
    
    def confusion_matrix_from_state(self, dataset, state, model, confusion_matrix_png_path, 
                                  use_subset=True, batch_size=32, max_subset=3000):
        """
        Crée une matrice de confusion directement depuis l'état du modèle (sans fichier .pkl)
        Utile pour l'évaluation immédiate après l'entraînement.
        
        Args:
            dataset: Dataset de validation (TensorFlow Dataset)
            state: État du modèle (TrainState)
            model: Modèle JAX
            confusion_matrix_png_path: Chemin de sauvegarde
            use_subset: Utiliser un sous-ensemble
            batch_size: Taille des batches
            max_subset: Nombre maximum d'échantillons
            
        Returns:
            tuple: (confusion_matrix, accuracy, macro_precision, macro_recall, macro_f1)
        """
        print("🔍 Utilisation de l'état du modèle en mémoire...")
        
        # Variables du modèle
        variables = {'params': state.params, 'batch_stats': state.batch_stats}
        dummy_rng = jax.random.PRNGKey(0)
        
        # Collecter toutes les données du dataset TensorFlow
        all_images = []
        all_labels = []
        
        print("📊 Collecte des données du dataset...")
        for batch in dataset.as_numpy_iterator():
            batch_images, batch_labels = batch
            all_images.append(batch_images)
            all_labels.append(batch_labels)
        
        # Concaténer tous les batches
        images = np.concatenate(all_images, axis=0)
        labels = np.concatenate(all_labels, axis=0)
        num_classes = len(np.unique(labels))
        
        print(f"📊 Dataset: {len(images)} images, {num_classes} classes")
        
        # Sous-ensemble si nécessaire
        if use_subset and len(images) > max_subset:
            num_total = len(images)
            num_subset = min(max_subset, num_total // 3)
            np.random.seed(42)
            subset_indices = np.random.choice(num_total, num_subset, replace=False)
            images = images[subset_indices]
            labels = labels[subset_indices]
            print(f"📉 Sous-ensemble: {len(images)} images sélectionnées")
        
        # Prédictions par batches
        all_logits = []
        print("🔮 Génération des prédictions...")
        
        for i in tqdm(range(0, len(images), batch_size), desc='Prédictions'):
            batch_images = jnp.array(images[i:i+batch_size], dtype=jnp.float32)
            
            # Appliquer le modèle
            logits = model.apply(
                variables, 
                batch_images, 
                training=False,
                rngs={'dropout': dummy_rng}
            )
            all_logits.append(np.array(logits))
        
        # Concaténer et calculer les prédictions
        all_logits = np.concatenate(all_logits, axis=0)
        predictions = np.argmax(all_logits, axis=-1)
        
        # Créer la matrice de confusion
        confusion_matrix = np.zeros((num_classes, num_classes), dtype=int)
        for true_label, pred_label in zip(labels, predictions):
            confusion_matrix[true_label, pred_label] += 1
        
        # Créer la visualisation
        self._plot_confusion_matrix(confusion_matrix, confusion_matrix_png_path, num_classes)
        
        # Calculer les métriques
        accuracy = np.trace(confusion_matrix) / np.sum(confusion_matrix)
        precision = np.diag(confusion_matrix) / np.sum(confusion_matrix, axis=0)
        recall = np.diag(confusion_matrix) / np.sum(confusion_matrix, axis=1)
        
        precision = np.nan_to_num(precision, nan=0.0)
        recall = np.nan_to_num(recall, nan=0.0)
        
        f1_score = 2 * (precision * recall) / (precision + recall)
        f1_score = np.nan_to_num(f1_score, nan=0.0)
        
        macro_precision = np.mean(precision)
        macro_recall = np.mean(recall)
        macro_f1 = np.mean(f1_score)
        
        # Afficher les résultats
        print(f"\n📈 RÉSULTATS DE LA MATRICE DE CONFUSION:")
        print(f"Accuracy globale: {accuracy:.4f}")
        print(f"Macro Precision: {macro_precision:.4f}")
        print(f"Macro Recall: {macro_recall:.4f}")
        print(f"Macro F1: {macro_f1:.4f}")
        
        return confusion_matrix, accuracy, macro_precision, macro_recall, macro_f1
    
    def show_predictions_from_dir(self, pkl_path="best_model.pkl", 
                                  images_dir=None, 
                                  dataset_config=None, 
                                  images_per_grid=16, 
                                  grid_size=(4, 4),
                                  show_all=True,
                                  target_class=None,
                                  show_only_errors=False,
                                  log_only=False):
        """
        Affiche les prédictions du modèle sur des images d'un répertoire
        Traite TOUTES les images par paquets de 16
        
        Args:
            pkl_path: Chemin vers le fichier .pkl du modèle sauvegardé (défaut: "best_model.pkl")
            images_dir: Dossier contenant les images à prédire (défaut: premier dossier de val)
            dataset_config: Configuration du dataset (défaut: FIGHTERJET_8CLASSES)
            images_per_grid: Nombre d'images par grille (défaut: 16)
            grid_size: Taille de la grille (rows, cols) (défaut: 4x4)
            show_all: Si True, traite TOUTES les images par paquets (défaut: True)
            target_class: Classe cible attendue (ex: "c17") - si spécifié, affiche les erreurs
            show_only_errors: Si True avec target_class, affiche SEULEMENT les mauvaises prédictions
            log_only: Si True, sauvegarde dans log.txt au lieu de générer des PNG (défaut: False)
        """
        import os
        from PIL import Image
        
        # Valeurs par défaut pour Spyder
        if dataset_config is None:
            from dataset_configs import get_dataset_config
            dataset_config = get_dataset_config("FIGHTERJET_8CLASSES")
            print("📊 Configuration par défaut: FIGHTERJET_8CLASSES")
        
        if images_dir is None:
            # Utiliser le premier dossier de validation
            data_dir = dataset_config["data_dir"]
            val_dir = os.path.join(data_dir, 'val')
            first_class = dataset_config["class_names"][0]
            images_dir = os.path.join(val_dir, first_class)
            print(f"📁 Répertoire par défaut: {images_dir} (classe {first_class})")
        
        print(f"🔍 Chargement du modèle depuis {pkl_path}...")
        
        # Charger le modèle sauvegardé
        with open(pkl_path, 'rb') as f:
            model_data = pickle.load(f)
        
        # Extraire les données du modèle
        if 'model_state' in model_data:
            params = model_data['model_state']['params']
            batch_stats = model_data['model_state'].get('batch_stats', {})
            model_info = model_data.get('model_info', {})
            model_name = model_info.get('model_name', 'sophisticated_cnn')
            num_classes = model_info.get('num_classes', len(self.class_names))
        else:
            params = model_data['params']
            batch_stats = model_data.get('batch_stats', {})
            model_name = model_data.get('model_name', 'sophisticated_cnn')
            num_classes = model_data.get('num_classes', len(self.class_names))
        
        print(f"📊 Modèle: {model_name}, {num_classes} classes")
        
        # Créer le modèle
        model = get_model(model_name, num_classes=num_classes, dropout_rate=0.0)
        variables = {'params': params, 'batch_stats': batch_stats}
        
        # Charger les statistiques de normalisation
        mean_std_path = dataset_config.get("mean_std_path", "./data/chunks/dataset_chunked_meanstd.npz")
        with np.load(mean_std_path) as data:
            mean = data['mean']
            std = data['std']
        
        image_size = dataset_config["image_size"]
        grayscale = dataset_config.get("grayscale", False)
        
        # Charger TOUTES les images du répertoire
        print(f"📁 Chargement des images depuis {images_dir}...")
        print(f"🎨 Mode: {'Grayscale (1 canal)' if grayscale else 'RGB (3 canaux)'}")
        image_paths = sorted([
            os.path.join(images_dir, f)
            for f in os.listdir(images_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
        ])
        
        if not image_paths:
            print(f"❌ Aucune image trouvée dans {images_dir}")
            return
        
        total_images = len(image_paths)
        print(f"📊 {total_images} images trouvées")
        
        # Traiter par paquets de images_per_grid
        num_grids = (total_images + images_per_grid - 1) // images_per_grid if show_all else 1
        print(f"📊 {num_grids} grilles à générer (par paquets de {images_per_grid})")
        
        # Convertir target_class en index si spécifié
        target_class_idx = None
        if target_class:
            if target_class in self.class_names:
                target_class_idx = self.class_names.index(target_class)
                print(f"🎯 Classe cible: {target_class} (index {target_class_idx})")
            else:
                print(f"⚠️  Classe '{target_class}' non trouvée dans {self.class_names}")
                print(f"   Analyse sans filtrage...")
        
        all_predictions = []
        all_confidences = []
        all_image_data = []  # Stocker (path, img_pil, pred, proba) pour filtrage ultérieur
        
        for grid_idx in range(num_grids):
            start_idx = grid_idx * images_per_grid
            end_idx = min(start_idx + images_per_grid, total_images)
            batch_paths = image_paths[start_idx:end_idx]
            
            print(f"\n🔮 Traitement de la grille {grid_idx+1}/{num_grids} ({len(batch_paths)} images)...")
            
            # Préparer les images de ce paquet
            images = []
            original_images = []
            
            for path in batch_paths:
                # Charger l'image originale pour affichage (toujours en RGB pour matplotlib)
                img_pil_display = Image.open(path).convert('RGB')
                original_images.append(img_pil_display)
                
                # Préparer pour le modèle (selon config grayscale)
                img_pil_model = Image.open(path).convert('L' if grayscale else 'RGB')
                img_resized = img_pil_model.resize(image_size)
                img_array = np.array(img_resized, dtype=np.float32) / 255.0
                
                # Ajouter dimension canal si grayscale (H, W) → (H, W, 1)
                if grayscale and len(img_array.shape) == 2:
                    img_array = img_array[:, :, np.newaxis]
                
                # Normaliser comme les données d'entraînement
                img_normalized = (img_array - mean) / std
                images.append(img_normalized)
            
            # Convertir en batch
            images_batch = np.array(images)
            images_jnp = jnp.array(images_batch, dtype=jnp.float32)
            
            # Prédictions
            dummy_rng = jax.random.PRNGKey(0)
            logits = model.apply(
                variables,
                images_jnp,
                training=False,
                rngs={'dropout': dummy_rng}
            )
            
            predictions = np.argmax(logits, axis=-1)
            probas = jax.nn.softmax(logits, axis=-1)
            
            # Stocker pour le résumé global et filtrage
            for path, img_pil, pred, proba in zip(batch_paths, original_images, predictions, probas):
                all_predictions.append(pred)
                confidence = float(proba[pred]) * 100
                all_confidences.append(confidence)
                all_image_data.append((path, img_pil, pred, proba, confidence))
        
        # === FILTRAGE PAR CLASSE CIBLE ===
        filtered_image_data = all_image_data
        
        if target_class_idx is not None:
            # TOUJOURS filtrer pour ne garder que les ERREURS quand target_class est spécifié
            filtered_image_data = [
                data for data in all_image_data 
                if data[2] != target_class_idx  # data[2] = prediction
            ]
            
            accuracy_on_class = (len(all_image_data) - len(filtered_image_data)) / len(all_image_data) * 100
            
            print(f"\n🔍 FILTRAGE DES ERREURS:")
            print(f"   Total images: {len(all_image_data)}")
            print(f"   Correctes ({target_class}): {len(all_image_data) - len(filtered_image_data)} ({accuracy_on_class:.1f}%)")
            print(f"   Erreurs: {len(filtered_image_data)} ({100 - accuracy_on_class:.1f}%)")
            
            # Logger chaque erreur
            if filtered_image_data:
                print(f"\n📝 LISTE DES ERREURS:")
                for i, data in enumerate(filtered_image_data, 1):
                    path, img_pil, pred, proba, confidence = data
                    pred_class_name = self.class_names[pred] if pred < len(self.class_names) else f"Classe {pred}"
                    print(f"   {i}. {os.path.basename(path)}")
                    print(f"      → Prédit: {pred_class_name} (confiance: {confidence:.1f}%)")
                    print(f"      → Fichier: {path}")
        
        # Si pas d'images après filtrage, arrêter
        if not filtered_image_data:
            print(f"✅ Aucune erreur trouvée! Toutes les images sont correctement classifiées comme {target_class}")
            return
        
        # === MODE LOG ONLY ===
        if log_only:
            print(f"\n📝 MODE LOG ONLY - Sauvegarde dans fichier texte...")
            
            # Nom du fichier log
            if target_class:
                log_file = f"errors_{target_class}_log.txt"
            else:
                log_file = "predictions_log.txt"
            
            # Écrire dans le fichier
            with open(log_file, 'w', encoding='utf-8') as f:
                f.write("=" * 80 + "\n")
                f.write("ANALYSE DES PRÉDICTIONS\n")
                f.write("=" * 80 + "\n\n")
                
                if target_class:
                    f.write(f"Classe cible: {target_class}\n")
                    f.write(f"Total images: {len(all_image_data)}\n")
                    f.write(f"Correctes: {len(all_image_data) - len(filtered_image_data)}\n")
                    f.write(f"Erreurs: {len(filtered_image_data)}\n")
                    accuracy = (len(all_image_data) - len(filtered_image_data)) / len(all_image_data) * 100
                    f.write(f"Accuracy: {accuracy:.2f}%\n\n")
                
                f.write("target_class\tpred_class\tconfidence\tpath\n")
                
                for i, data in enumerate(filtered_image_data, 1):
                    path, img_pil, pred, proba, confidence = data
                    pred_class_name = self.class_names[pred] if pred < len(self.class_names) else f"Classe {pred}"
                    
                    f.write(f"{target_class if target_class else 'N/A'}\t")
                    f.write(f"{pred_class_name}\t")
                    f.write(f"{confidence:.2f}%\t")
                    f.write(f"{path}\n")
            
            print(f"✅ Log sauvegardé: {log_file}")
            print(f"   {len(filtered_image_data)} erreurs enregistrées")
            
            # Pas d'affichage graphique, on s'arrête ici
            return
        
        # === MODE GRAPHIQUE (comportement normal) ===
        # Régénérer les grilles avec les données filtrées
        num_filtered_grids = (len(filtered_image_data) + images_per_grid - 1) // images_per_grid
        print(f"\n📊 Génération de {num_filtered_grids} grilles...")
        
        for grid_idx in range(num_filtered_grids):
            start_idx = grid_idx * images_per_grid
            end_idx = min(start_idx + images_per_grid, len(filtered_image_data))
            grid_data = filtered_image_data[start_idx:end_idx]
            
            # Affichage de cette grille
            rows, cols = grid_size
            fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3.5))
            axes = axes.flatten()
            
            for idx, data in enumerate(grid_data):
                if idx >= rows * cols:
                    break
                
                path, img_pil, pred, proba, confidence = data
                
                ax = axes[idx]
                ax.imshow(img_pil)
                
                # Nom de classe et confiance
                class_name = self.class_names[pred] if pred < len(self.class_names) else f"Classe {pred}"
                
                # Couleur selon confiance OU selon erreur
                if target_class_idx is not None:
                    # Si classe cible spécifiée, rouge pour erreurs, vert pour correct
                    is_error = (pred != target_class_idx)
                    color = 'red' if is_error else 'green'
                    # Ajouter la vraie classe dans le titre si erreur
                    title = f"Prédit: {class_name}\n{confidence:.1f}%"
                    if is_error:
                        title = f"❌ {class_name}\n{confidence:.1f}%"
                    else:
                        title = f"✅ {class_name}\n{confidence:.1f}%"
                else:
                    # Code couleur normal selon confiance
                    color = 'green' if confidence > 70 else 'orange' if confidence > 40 else 'red'
                    title = f"{class_name}\n{confidence:.1f}%"
                
                ax.set_title(title, fontsize=10, fontweight='bold', color=color)
                ax.axis("off")
                
                # Afficher le nom du fichier en petit
                filename = os.path.basename(path)
                ax.text(0.5, -0.08, filename, transform=ax.transAxes,
                       ha='center', fontsize=7, style='italic')
            
            # Masquer les axes vides
            for idx in range(len(grid_data), rows * cols):
                axes[idx].axis('off')
            
            # Titre de la grille
            if target_class_idx is not None:
                title = f"ERREURS - Grille {grid_idx+1}/{num_filtered_grids} Classe attendue: {target_class}"
            else:
                title = f"Grille {grid_idx+1}/{num_filtered_grids}"
            
            fig.suptitle(title, fontsize=13, fontweight='bold', color='red' if target_class_idx is not None else 'black')
            
            plt.tight_layout()
            
            # Sauvegarder chaque grille
            if target_class_idx is not None:
                output_file = f"errors_{target_class}_grid_{grid_idx+1:03d}.png"
            else:
                output_file = f"predictions_grid_{grid_idx+1:03d}.png"
            plt.savefig(output_file, dpi=150, bbox_inches='tight')
            print(f"💾 Grille sauvegardée: {output_file}")
            
            # Afficher dans Spyder/Jupyter
            plt.show()
        
        # Afficher le résumé global
        print(f"\n" + "=" * 60)
        print(f"📊 RÉSUMÉ GLOBAL ({len(all_predictions)} images):")
        print("=" * 60)
        
        unique_preds, counts = np.unique(all_predictions, return_counts=True)
        for pred_class, count in zip(unique_preds, counts):
            class_name = self.class_names[pred_class] if pred_class < len(self.class_names) else f"Classe {pred_class}"
            percentage = count / len(all_predictions) * 100
            print(f"  {class_name}: {count} images ({percentage:.1f}%)")
        
        # Statistiques de confiance
        avg_confidence = np.mean(all_confidences)
        min_confidence = np.min(all_confidences)
        max_confidence = np.max(all_confidences)
        
        print(f"\n📊 STATISTIQUES DE CONFIANCE:")
        print(f"  Moyenne: {avg_confidence:.1f}%")
        print(f"  Min: {min_confidence:.1f}%")
        print(f"  Max: {max_confidence:.1f}%")
        
        # Images avec faible confiance
        low_conf_indices = [i for i, conf in enumerate(all_confidences) if conf < 50]
        if low_conf_indices:
            print(f"\n⚠️  {len(low_conf_indices)} images avec confiance < 50%:")
            for i in low_conf_indices[:5]:  # Afficher les 5 premières
                data = all_image_data[i]
                path, img_pil, pred, proba, confidence = data
                class_name = self.class_names[pred] if pred < len(self.class_names) else f"Classe {pred}"
                print(f"  - {os.path.basename(path)}: {class_name} ({confidence:.1f}%)")
        
        # Statistiques d'erreurs si classe cible spécifiée
        if target_class_idx is not None:
            errors = [data for data in all_image_data if data[2] != target_class_idx]
            correct = [data for data in all_image_data if data[2] == target_class_idx]
            
            accuracy_on_class = len(correct) / len(all_image_data) * 100 if all_image_data else 0
            
            print(f"\n🎯 ANALYSE POUR CLASSE '{target_class}':")
            print(f"  Correct: {len(correct)} images ({accuracy_on_class:.1f}%)")
            print(f"  Erreurs: {len(errors)} images ({100-accuracy_on_class:.1f}%)")
            
            if errors:
                # Analyser les confusions
                confusion_counts = {}
                for data in errors:
                    pred = data[2]
                    class_name = self.class_names[pred] if pred < len(self.class_names) else f"Classe {pred}"
                    confusion_counts[class_name] = confusion_counts.get(class_name, 0) + 1
                
                print(f"\n🔄 CONFUSIONS (classe {target_class} prédite comme):")
                for confused_class, count in sorted(confusion_counts.items(), key=lambda x: x[1], reverse=True):
                    percentage = count / len(errors) * 100
                    print(f"  → {confused_class}: {count} fois ({percentage:.1f}% des erreurs)")



class TrainingVisualizer:
    """
    Classe pour visualiser l'historique d'entraînement
    Génère des graphiques avec Loss, Accuracy et Learning Rate
    """
    
    def __init__(self, history, model_name, num_params):
        """
        Args:
            history: Dict avec 'epochs', 'train_loss', 'train_acc', 'val_loss', 'val_acc', 'learning_rate'
            model_name: Nom du modèle pour le titre
            num_params: Nombre de paramètres du modèle
        """
        self.history = history
        self.model_name = model_name
        self.num_params = num_params
    
    def plot_training_curves(self, epoch_start=0, save_path=None):
        """
        Génère le graphique de l'historique d'entraînement
        
        Args:
            epoch_start: Epoch de départ pour le zoom
            save_path: Chemin de sauvegarde (par défaut: model_name.png)
        """
        if not self.history['epochs']:
            print("⚠️  Aucun historique à visualiser")
            return
        
        # Déterminer le chemin de sauvegarde
        if save_path is None:
            save_path = f"{self.model_name}.png"
        
        # Préparer les données
        epochs = self.history['epochs'][epoch_start:]
        train_loss = self.history['train_loss'][epoch_start:]
        val_loss = self.history['val_loss'][epoch_start:]
        train_acc = self.history['train_acc'][epoch_start:]
        val_acc = self.history['val_acc'][epoch_start:]
        lr = self.history['learning_rate'][epoch_start:]
        
        if not epochs:
            print("⚠️  Pas assez de données pour visualiser")
            return
        
        # Créer la figure avec triple axe
        plt.rc('mathtext', default='regular')
        fig = plt.figure(figsize=(14, 7))
        ax = fig.add_subplot()
        ax2 = ax.twinx()
        
        # Plot Loss (train + val)
        ax.plot(epochs, train_loss, label='Train Loss', color='tab:orange', linewidth=2)
        ax.plot(epochs, val_loss, label='Val Loss', color='tab:orange', linewidth=2, linestyle='--', alpha=0.7)
        
        # Plot Accuracy (train + val) avec zone d'overfitting
        ax2.plot(epochs, train_acc, color='tab:red', label='Train Acc', linewidth=2, alpha=0.3)
        ax2.fill_between(epochs, train_acc, val_acc, color='tab:red', alpha=0.15, label='Overfitting zone')
        ax2.plot(epochs, val_acc, '-', color='tab:blue', label='Val Acc', linewidth=2.5)
        
        # Plot Learning Rate (échelle log)
        ax3 = ax.twinx()
        ax3.spines['right'].set_position(('outward', 60))
        ax3.set_yscale('log')
        ax3.plot(epochs, lr, color='lightgreen', label='LR', linewidth=2)
        ax3.set_ylabel('Learning Rate', color='green')
        
        # Ajuster les limites de LR
        if lr:
            lr_min, lr_max = min(lr), max(lr)
            ax3.set_ylim(bottom=lr_min/1.05, top=lr_max*1.05)
            ax3.yaxis.set_major_locator(LogLocator(base=10.0))
        
        # Couleurs des ticks
        ax.tick_params(axis='y', colors='tab:orange')
        ax2.tick_params(axis='y', colors='tab:blue')
        ax3.tick_params(axis='y', colors='green')
        
        # Labels et légendes
        ax.set_xlabel("Epochs", fontsize=12)
        ax.set_ylabel(f'Loss (final: {train_loss[-1]:.4f})', color='tab:orange', fontsize=11)
        ax2.set_ylabel('Accuracy %', color='tab:blue', fontsize=11)
        ax2.legend(loc='lower right', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Limites de l'axe x
        ax.set_xlim(left=epoch_start if epoch_start > 0 else epochs[0])
        ax2.set_xlim(left=epoch_start if epoch_start > 0 else epochs[0])
        ax3.set_xlim(left=epoch_start if epoch_start > 0 else epochs[0])
        
        # Calculer les statistiques
        best_train = round(max(train_acc), 2)
        best_val = round(max(val_acc), 2)
        final_train = round(train_acc[-1], 2)
        final_val = round(val_acc[-1], 2)
        
        # Calculer l'overfitting avec Simpson sur toute la courbe
        train_area = simpson(train_acc, dx=1) if len(train_acc) > 1 else train_acc[0]
        val_area = simpson(val_acc, dx=1) if len(val_acc) > 1 else val_acc[0]
        overfitting_area = train_area - val_area
        overfitting_percentage = round(100 * overfitting_area / val_area, 2) if val_area > 0 else 0
        
        # Calculer l'écart final
        final_gap = round(final_train - final_val, 2)
        
        # Annotation avec les statistiques
        annotation_text = (
            f"Model: {self.model_name}  |  "
            f"Params: {self.num_params:,}  |  "
            f"Final → Train: {final_train}% | Val: {final_val}% | Gap: {final_gap}%  |  "
            f"Best Val: {best_val}%  |  "
            f"Overfitting (Simpson): {overfitting_percentage}%"
        )
        
        ax.annotate(annotation_text, xy=(0.01, 1.02), xycoords='axes fraction', 
                   va='bottom', fontsize=10, bbox=dict(boxstyle='round,pad=0.5', 
                   facecolor='wheat', alpha=0.3))
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        #print(f"📊 Graphique sauvegardé: {save_path}")


# Alias pour compatibilité avec le code existant
ModelReporter = Reporter

class DetectionReporter:
    """Modèle de reporting spécifique pour la détection d'objets"""
    
    def __init__(self, image_size=(224, 224), grid_size=7):
        self.image_size = image_size
        self.grid_size = grid_size
        
    def visualize_batch(self, images, predictions, targets=None, save_path="detection_vis.png", conf_threshold=0.5):
        """
        Visualise un batch d'images avec les boxes prédites (rouge) et réelles (vert)
        
        Args:
            images: (B, H, W, 3)
            predictions: (B, S, S, 5) -> [conf, x, y, w, h]
            targets: (B, MaxBoxes, 5) -> [has, cx, cy, w, h] (Optionnel)
        """
        import matplotlib.patches as patches
        
        batch_size = images.shape[0]
        # Limiter à 16 images max
        n_vis = min(16, batch_size)
        rows = int(np.ceil(np.sqrt(n_vis)))
        cols = int(np.ceil(n_vis / rows))
        
        fig, axes = plt.subplots(rows, cols, figsize=(cols*4, rows*4))
        if n_vis == 1: axes = [axes]
        axes = np.array(axes).flatten()
        
        S = self.grid_size
        
        for i in range(n_vis):
            ax = axes[i]
            img = images[i]
            ax.imshow(img)
            
            # 1. Dessiner Ground Truth (Vert)
            if targets is not None:
                gt_boxes = targets[i] # (MaxBoxes, 5)
                for box in gt_boxes:
                    has_obj, cx, cy, w, h = box
                    if has_obj > 0.5:
                        # Convertir normalized -> pixel
                        pix_w = w * self.image_size[0]
                        pix_h = h * self.image_size[1]
                        pix_x = (cx * self.image_size[0]) - (pix_w / 2)
                        pix_y = (cy * self.image_size[1]) - (pix_h / 2)
                        
                        rect = patches.Rectangle((pix_x, pix_y), pix_w, pix_h, 
                                               linewidth=2, edgecolor='lime', facecolor='none')
                        ax.add_patch(rect)
            
            # 2. Dessiner Prédictions (Rouge)
            preds_list = predictions if isinstance(predictions, (tuple, list)) else [predictions]
            
            for pred_batch in preds_list:
                pred = pred_batch[i]
                S = pred.shape[0]  # Taille dynamique de la grille (ex: 14 ou 7)
                C_pred = pred.shape[-1]
                B_boxes = C_pred // 5
                
                # Reshape en (S, S, B_boxes, 5)
                pred = pred.reshape((S, S, B_boxes, 5))
                
                # Itérer sur la grille et les ancres
                for row in range(S):
                    for col in range(S):
                        for b in range(B_boxes):
                            cell = pred[row, col, b]
                            conf = cell[0]
                            
                            if conf > conf_threshold:
                                # Décoder la box
                                # x, y sont relatifs à la cellule (0-1)
                                # On veut cx, cy relatifs à l'image entière (0-1)
                                bx = (col + cell[1]) / S
                                by = (row + cell[2]) / S
                                bw = cell[3]
                                bh = cell[4]
                                
                                # Convertir en pixel
                                pix_w = bw * self.image_size[0]
                                pix_h = bh * self.image_size[1]
                                pix_x = (bx * self.image_size[0]) - (pix_w / 2)
                                pix_y = (by * self.image_size[1]) - (pix_h / 2)
                                
                                rect = patches.Rectangle((pix_x, pix_y), pix_w, pix_h, 
                                                       linewidth=2, edgecolor='red', facecolor='none')
                                ax.add_patch(rect)
                                ax.text(pix_x, pix_y - 5, f"{conf:.2f}", color='red', fontsize=8, fontweight='bold')
            
            ax.axis('off')
            ax.set_title(f"Image {i}")
            
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
            print(f"💾 Visualisation sauvegardée: {save_path}")
        plt.show() # Pour notebook/spyder
