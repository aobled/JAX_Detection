import jax
import jax.numpy as jnp
import optax
from abc import ABC, abstractmethod
from loss_functions import compute_grid_loss, compute_grid_loss_multilevel, compute_v7_loss, compute_segmentation_loss, compute_centernet_loss, compute_chess_policy_value_loss, compute_chess_policy_loss, compute_chess_value_loss
from detection_target_encoding import HEATMAP_KEY, SIZE_KEY
from chess_target_encoding import POLICY_KEY, VALUE_KEY
from utils import mixup_batch, smooth_labels

class TaskStrategy(ABC):
    @abstractmethod
    def preprocess_batch(self, images, targets, is_training, rng=None):
        """Prétraite les données (Cast, Mixup, Label Smoothing...) à l'intérieur du JIT."""
        pass
        
    @abstractmethod
    def compute_loss(self, outputs, targets, **kwargs):
        """Calcule la perte du réseau."""
        pass
        
    @abstractmethod
    def compute_metrics(self, outputs, targets):
        """Calcule la ou les métriques d'évaluation."""
        pass
        
    @abstractmethod
    def generate_reports(self, val_ds, final_state, model, config):
        """Génère les rapports post-entraînement (Matrice de confusion, Visualisation Boxes...)."""
        pass
        
    @property
    @abstractmethod
    def primary_metric_name(self) -> str:
        """Nom textuel de la métrique principale (ex: 'Accuracy', 'Score')."""
        pass
        
    @property
    @abstractmethod
    def optimization_mode(self) -> str:
        """Mode d'optimisation de la métrique ('max' ou 'min')."""
        pass
        
    def export_model(self, state, config):
        """Exporte le modèle (params et batch_stats) au format .pkl."""
        try:
            import pickle
            import jax
            
            pkl_path = self._get_export_path(config)
            
            # Convertir les tenseurs XLA/TPU en Numpy natif CPU pour éviter tous les problèmes de portabilité
            params_cpu = jax.device_get(state.params)
            batch_stats_cpu = jax.device_get(state.batch_stats) if state.batch_stats is not None else {}
            
            model_dict = {
                'params': params_cpu,
                'batch_stats': batch_stats_cpu,
                'config': config 
            }
            with open(pkl_path, 'wb') as f:
                pickle.dump(model_dict, f)
            print(f"   [💾] Export pur PKL généré: {pkl_path}")
            
            # Libérer la mémoire des copies numpy
            del params_cpu, batch_stats_cpu
        except Exception as e:
            print(f"   [⚠️] Erreur d'export PKL: {e}")
            
    def _get_export_path(self, config) -> str:
        """Retourne le chemin cible pour l'export .pkl.

        Défaut concret (2026-07-30, audit qualité Winston 2026-07-22, item 1) : les 5
        sous-classes existantes implémentaient exactement la même logique (nom dérivé de
        config["dataset_name"], Story 5.0) - la frontière d'abstraction était mal placée,
        c'était un défaut concret déguisé en méthode abstraite. Une sous-classe qui a
        vraiment besoin d'un chemin différent peut toujours la surcharger.
        """
        return config.get("checkpoint_path") or f"best_model_{config.get('dataset_name', 'unknown').lower()}.pkl"

    def get_training_state_path(self, config) -> str:
        """Retourne le chemin pour la sauvegarde de l'état d'entraînement complet (.pkl lourd).

        Défaut concret (2026-07-30, audit qualité Winston 2026-07-22, item 1) - même
        raisonnement que _get_export_path ci-dessus.
        """
        return config.get("training_state_path") or f"best_model_training_state_{config.get('dataset_name', 'unknown').lower()}.pkl"

class ClassificationStrategy(TaskStrategy):
    def __init__(self, num_classes: int, label_smoothing: float = 0.0, mixup_alpha: float = 0.0, loss_method: str = "cross_entropy", loss_params: dict = None, metric_method: str = "accuracy", report_method: str = "confusion_matrix"):
        self.num_classes = num_classes
        self.label_smoothing = label_smoothing
        self.mixup_alpha = mixup_alpha
        self.loss_method = loss_method
        self.loss_params = loss_params or {}
        self.metric_method = metric_method
        self.report_method = report_method



    @property
    def primary_metric_name(self) -> str:
        return "Accuracy"
        
    @property
    def optimization_mode(self) -> str:
        return "max"
        
    def preprocess_batch(self, images, targets, is_training, rng=None):
        targets = jnp.array(targets, dtype=jnp.int32)
        use_onehot = False
        
        if not is_training:
            return images, targets, use_onehot
            
        if self.mixup_alpha > 0 and rng is not None:
             images, targets = mixup_batch(images, targets, self.mixup_alpha, self.num_classes, rng)
             use_onehot = True
             if self.label_smoothing > 0:
                 # Composé sur les labels one-hot déjà mixés par mixup_batch (targets somme à 1 par ligne,
                 # la formule de smoothing standard s'applique identiquement à des labels durs ou mixés).
                 targets = targets * (1 - self.label_smoothing) + self.label_smoothing / self.num_classes
        elif self.label_smoothing > 0:
             targets = smooth_labels(targets, self.num_classes, self.label_smoothing)
             use_onehot = True

        return images, targets, use_onehot
        
    def compute_loss(self, outputs, targets, use_onehot_labels=False, **kwargs):
        if self.loss_method == "cross_entropy":
            if use_onehot_labels:
                return optax.softmax_cross_entropy(outputs, targets).mean()
            else:
                return optax.softmax_cross_entropy_with_integer_labels(outputs, targets).mean()
        elif self.loss_method == "focal_loss":
            from loss_functions import compute_focal_loss
            return compute_focal_loss(outputs, targets, use_onehot_labels=use_onehot_labels, **self.loss_params)
        else:
            raise ValueError(f"Méthode de loss '{self.loss_method}' non supportée pour la classification.")

            
    def compute_metrics(self, outputs, targets):
        # Si targets est one_hot, on doit le convertir pour l'accuracy,
        # mais on calcule les métriques généralement sur les vrais labels int32.
        # En training, targets peut être mixup (onehot floats).
        if len(targets.shape) > 1 and targets.shape[-1] == outputs.shape[-1]:
            # Targets est one-hot
            true_classes = jnp.argmax(targets, axis=-1)
        else:
            true_classes = targets
        return (jnp.argmax(outputs, axis=-1) == true_classes).mean()
        
    def generate_reports(self, val_ds, final_state, model, config):
        from reporting import Reporter as ModelReporter # Utilise la classe Reporter
        reporter = ModelReporter(class_names=config["class_names"])
        try:
            # Déterminer le bon fichier PKL engendré par le Trainer
            pkl_path = self._get_export_path(config)

            reporter.confusion_matrix_from_pkl(
                dataset=val_ds,
                pkl_path=pkl_path,
                confusion_matrix_png_path=config.get("confusion_matrix_path", "confusion_matrix.png"),
                use_subset=config.get("eval_use_subset", False),
                batch_size=config.get("eval_batch_size", 32),
                max_subset=config.get("eval_max_subset", 1000)
            )
            print(f"✅ Matrice de confusion générée avec succès depuis l'export pur (pkl) !")
        except Exception as e:
            print(f"❌ Erreur metrics: {e}")


class DetectionStrategy(TaskStrategy):
    def __init__(self, loss_method: str = "segmentation", loss_params: dict = None, metric_method: str = "segmentation_iou", report_method: str = "segmentation_heatmap"):
        self.loss_method = loss_method
        self.loss_params = loss_params or {}
        self.metric_method = metric_method
        self.report_method = report_method


    @property
    def primary_metric_name(self) -> str:
        return "IoU"
        
    @property
    def optimization_mode(self) -> str:
        return "max"
        
    def preprocess_batch(self, images, targets, is_training, rng=None):
        targets = jnp.array(targets, dtype=jnp.float32)
        return images, targets, False
        
    def compute_loss(self, outputs, targets, **kwargs):
        if self.loss_method == "segmentation":
            return compute_segmentation_loss(outputs, targets, **self.loss_params)
        elif self.loss_method == "grid":
            return compute_grid_loss(outputs, targets, **self.loss_params)
        elif self.loss_method == "grid_multilevel":
            return compute_grid_loss_multilevel(outputs, targets, **self.loss_params)
        elif self.loss_method == "v7":
            return compute_v7_loss(outputs, targets, **self.loss_params)
        else:
            raise ValueError(f"Méthode de loss '{self.loss_method}' non supportée pour la détection.")

        
        
    def compute_metrics(self, outputs, targets):
        """Calcule le mIoU (Mean Intersection over Union) binaire pour la segmentation."""
        if self.metric_method == "segmentation_iou":
            # Binarisation avec un seuil de 0.5
            threshold = 0.5
            preds = (outputs > threshold).astype(jnp.float32)
            targets = targets.astype(jnp.float32)
            
            # Calcul par image dans le batch (axes 1, 2, 3 correspondants à H, W, C)
            intersection = jnp.sum(preds * targets, axis=(1, 2, 3))
            union = jnp.sum(preds, axis=(1, 2, 3)) + jnp.sum(targets, axis=(1, 2, 3)) - intersection
            
            # S'il n'y a pas d'objet et qu'on a rien prédit, IoU = 1.0
            # Sinon IoU = intersection / union
            iou = jnp.where(
                union > 0, 
                intersection / union, 
                jnp.where(jnp.sum(targets, axis=(1, 2, 3)) == 0, 1.0, 0.0)
            )
            
            # Retourne l'IoU moyen du batch (entre 0 et 1)
            return jnp.mean(iou)
        elif self.metric_method == "yolo_iou":
            raise NotImplementedError("La métrique 'yolo_iou' doit être implémentée pour YOLO (calcul mAP ou IoU sur boxes).")
        else:
            raise ValueError(f"Méthode de métrique '{self.metric_method}' non supportée pour la détection.")

        
    def generate_reports(self, val_ds, final_state, model, config):
        if self.report_method == "segmentation_heatmap":
            import cv2
            import numpy as np
            try:
                batch_consumed = False
                for vis_imgs, vis_masks in val_ds.take(1).as_numpy_iterator():
                    batch_consumed = True
                    vars = {'params': final_state.params, 'batch_stats': final_state.batch_stats}
                    pred_masks = final_state.apply_fn(vars, vis_imgs, training=False)

                    # Sauvegarder juste un batch visuel pour debug
                    # Image 0
                    img0 = np.array(vis_imgs[0] * 255, dtype=np.uint8)
                    true0 = np.array(vis_masks[0] * 255, dtype=np.uint8)
                    pred0 = np.array(pred_masks[0] * 255, dtype=np.uint8)

                    # OpenCV a besoin d'un vrai tableau Numpy 2D (H, W) pour la ColorMap
                    pred0_flat = pred0[..., 0] if pred0.ndim == 3 and pred0.shape[-1] == 1 else pred0
                    heatmap = cv2.applyColorMap(pred0_flat, cv2.COLORMAP_JET)

                    # Conversion grayscale -> RGB si nécessaire pour la concatenation
                    if img0.shape[-1] == 1:
                        img0 = cv2.cvtColor(img0, cv2.COLOR_GRAY2BGR)
                    true0 = cv2.cvtColor(true0, cv2.COLOR_GRAY2BGR)

                    composite = cv2.hconcat([img0, true0, heatmap])
                    cv2.imwrite("final_detection_vis.png", composite)
                    break
                if batch_consumed:
                    print("✅ Visualisation de détection sémantique générée (final_detection_vis.png)")
                else:
                    print("⚠️  val_ds est vide - aucune visualisation générée (final_detection_vis.png absent)")
            except Exception as e:
                print(f"❌ Erreur lors de la visualisation sémantique: {e}")
        elif self.report_method == "yolo_boxes":
            raise NotImplementedError("Le rapport 'yolo_boxes' doit être implémenté pour YOLO (dessin des boxes au lieu d'une heatmap).")
        else:
            print(f"⚠️ Méthode de rapport '{self.report_method}' non supportée pour la détection.")


class CenterNetDetectionStrategy(TaskStrategy):
    """
    Stratégie dédiée à JAX_DETECTOR (heatmap+taille, AD-9/AD-17). Classe séparée de
    DetectionStrategy (approche masque/segmentation) - ne la modifie ni ne l'étend.
    outputs/targets sont des dicts {HEATMAP_KEY, SIZE_KEY} (Stories 7.2/7.3/7.5),
    jamais un tenseur unique comme le fait DetectionStrategy.
    """
    def __init__(self, loss_params: dict = None):
        self.loss_params = loss_params or {}

    @property
    def primary_metric_name(self) -> str:
        return "HeatmapActivation"

    @property
    def optimization_mode(self) -> str:
        return "max"

    def preprocess_batch(self, images, targets, is_training, rng=None):
        # targets est deja un dict {HEATMAP_KEY, SIZE_KEY} (Story 7.5, batche par tf.data) -
        # simple cast float32, pas de mixup/label smoothing (non pertinents pour la detection,
        # meme choix que DetectionStrategy.preprocess_batch)
        targets = jax.tree_util.tree_map(lambda t: jnp.asarray(t, dtype=jnp.float32), targets)
        return images, targets, False

    def compute_loss(self, outputs, targets, **kwargs):
        return compute_centernet_loss(outputs, targets, **self.loss_params)

    def compute_metrics(self, outputs, targets):
        """
        Metrique proxy JAX-native (HeatmapActivation) - pas un decode de boites complet.
        decode_detection_targets (Story 7.1) est NumPy pur, incompatible avec le JIT de
        trainer.py (voir Dev Notes de cette story) : une vraie precision/rappel de boites
        est le travail de l'Epic 8 (Story 8.3, decode JAX-natif pour l'inference), pas
        anticipe ici.

        Addendum post-hoc (2026-07-18) : remplace un ancien HeatmapRecall a seuil dur
        (fraction de pixels-centres reels ou pred>0.5) par la moyenne CONTINUE de la
        prediction aux vrais pixels-centres. Le seuil dur masquait un vrai progres en
        execution reelle (Story 7.8) - le modele apprenait deja une separation nette
        centres/fond (confirme par archive/diagnose_heatmap_predictions.py) alors que
        HeatmapRecall restait a 0.0000 plusieurs epochs de suite, tant qu'aucune
        prediction n'avait franchi 0.5. Cette metrique gate aussi la sauvegarde du
        checkpoint (trainer.py, optimization_mode="max") - un seuil dur y etait
        particulierement mal adapte : une progression reelle mais sous le seuil ne
        produisait jamais de "New best model saved". La version continue reste
        centree sur le heatmap uniquement (pas melangee a la taille comme le serait
        val_loss), compatible JIT (pas de decode de boites), et visible dans le
        reporting existant (train_acc/val_acc, TrainingVisualizer) sans aucun
        changement necessaire ailleurs.
        """
        gt_heatmap = targets[HEATMAP_KEY]
        pred_heatmap = outputs[HEATMAP_KEY]

        is_positive = (gt_heatmap == 1.0)
        num_pos = jnp.sum(is_positive.astype(jnp.float32))

        sum_pred_at_positives = jnp.sum(jnp.where(is_positive, pred_heatmap, 0.0))

        activation = jnp.where(num_pos > 0, sum_pred_at_positives / num_pos, 1.0)
        return activation

    def generate_reports(self, val_ds, final_state, model, config):
        report_method = getattr(self, "report_method", "centernet_heatmap")
        if report_method == "centernet_heatmap":
            import cv2
            import numpy as np
            try:
                batch_consumed = False
                for vis_imgs, vis_targets in val_ds.take(1).as_numpy_iterator():
                    batch_consumed = True
                    vars = {'params': final_state.params, 'batch_stats': final_state.batch_stats}
                    pred_outputs = final_state.apply_fn(vars, vis_imgs, training=False)

                    true_heatmap = vis_targets[HEATMAP_KEY]
                    pred_heatmap = pred_outputs[HEATMAP_KEY]

                    img0 = np.array(vis_imgs[0] * 255, dtype=np.uint8)
                    true0 = np.array(true_heatmap[0] * 255, dtype=np.uint8)
                    pred0 = np.array(pred_heatmap[0] * 255, dtype=np.uint8)

                    pred0_flat = pred0[..., 0] if pred0.ndim == 3 and pred0.shape[-1] == 1 else pred0
                    heatmap_vis = cv2.applyColorMap(pred0_flat, cv2.COLORMAP_JET)

                    if img0.shape[-1] == 1:
                        img0 = cv2.cvtColor(img0, cv2.COLOR_GRAY2BGR)
                    true0 = cv2.cvtColor(true0, cv2.COLOR_GRAY2BGR)

                    composite = cv2.hconcat([img0, true0, heatmap_vis])
                    cv2.imwrite("final_detection_centernet_vis.png", composite)
                    break
                if batch_consumed:
                    print("✅ Visualisation CenterNet générée (final_detection_centernet_vis.png)")
                else:
                    print("⚠️  val_ds est vide - aucune visualisation générée (final_detection_centernet_vis.png absent)")
            except Exception as e:
                print(f"❌ Erreur lors de la visualisation CenterNet: {e}")
        else:
            print(f"⚠️ Méthode de rapport '{report_method}' non supportée pour CenterNetDetectionStrategy.")


class KeplerStrategy(TaskStrategy):
    def __init__(self, num_classes: int, loss_method: str = "cross_entropy", loss_params: dict = None, metric_method: str = "accuracy", report_method: str = "lightcurves"):
        self.num_classes = num_classes
        self.loss_method = loss_method
        self.loss_params = loss_params or {}
        self.metric_method = metric_method
        self.report_method = report_method



    @property
    def primary_metric_name(self) -> str:
        return "Accuracy"
        
    @property
    def optimization_mode(self) -> str:
        return "max"
        
    def preprocess_batch(self, images, targets, is_training, rng=None):
        # Pour Kepler, on ne fait pas d'augmentation temporelle complexe pour le moment.
        images = jnp.array(images, dtype=jnp.float32)
        # Chunks (B, L, 1, C) depuis ChunkManager → Conv1D attend (B, L, C)
        if images.ndim == 4 and images.shape[2] == 1:
            images = images[..., 0, :]
        use_onehot = False
        return images, targets, use_onehot

    def compute_loss(self, outputs, targets, use_onehot_labels=False, **kwargs):
        # Identique à la classification, c'est un problème binaire (Exoplanet ou non)
        # **kwargs (2026-07-30, audit qualité Winston 2026-07-22, item 3) : signature
        # alignée sur la méthode abstraite (task_strategies.py:17) et sur
        # ClassificationStrategy (ligne 125) - latent jusqu'ici (trainer.py n'appelait
        # qu'avec use_onehot_labels=), mais rompait la promesse de généricité de
        # l'interface pour tout futur kwarg générique.
        if use_onehot_labels:
            loss = jnp.mean(optax.softmax_cross_entropy(logits=outputs, labels=targets))
        else:
            loss = jnp.mean(optax.softmax_cross_entropy_with_integer_labels(logits=outputs, labels=targets))
        return loss

    def compute_metrics(self, outputs, targets):
        predicted_classes = jnp.argmax(outputs, axis=-1)
        accuracy = jnp.mean(predicted_classes == targets)
        return accuracy

    def generate_reports(self, val_ds, final_state, model, config):
        print("   [📈] Génération des rapports Kepler (Courbes de lumière)...")
        
        try:
            import matplotlib.pyplot as plt
            import os
            import numpy as np
            
            # Prendre un seul batch de validation
            batch = next(val_ds.as_numpy_iterator())
            images = batch['images']
            labels = batch['labels']
            
            # Prédiction
            outputs, _ = final_state.apply_fn(
                {'params': final_state.params, 'batch_stats': final_state.batch_stats},
                images,
                training=False
            )
            predictions = np.argmax(outputs, axis=-1)
            
            # Tracer 4 exemples (2 exoplanètes, 2 non-exoplanètes si possible)
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            axes = axes.flatten()
            
            for i in range(min(4, len(images))):
                ax = axes[i]
                flux = images[i].squeeze() # Enlever la dimension channel
                true_label = labels[i]
                pred_label = predictions[i]
                
                # Couleur selon succès
                color = 'green' if true_label == pred_label else 'red'
                
                ax.plot(flux, color='black', alpha=0.7, linewidth=0.5)
                ax.set_title(f"True: {'Exoplanet' if true_label==1 else 'No'} | Pred: {'Exoplanet' if pred_label==1 else 'No'}", color=color)
                ax.set_xlabel("Time step")
                ax.set_ylabel("Normalized Flux")
                
            plt.tight_layout()
            report_path = config.get("confusion_matrix_path", "kepler_lightcurves_report.png")
            plt.savefig(report_path)
            plt.close()
            print(f"   [🖼️] Rapport généré : {report_path}")
            
        except Exception as e:
            print(f"   [⚠️] Erreur lors de la génération du rapport Matplotlib: {e}")


class ChessPolicyValueStrategy(TaskStrategy):
    """
    Stratégie dédiée au domaine échecs (policy+value, AD-17/AD-24, Story 9.3). Première
    stratégie à double tête/double loss de ce projet - teste la généricité réelle du
    pattern TaskStrategy. outputs/targets sont des dicts {POLICY_KEY, VALUE_KEY}
    (chess_target_encoding.py, AD-18), jamais un tenseur unique.
    """
    def __init__(self, loss_params: dict = None):
        self.loss_params = loss_params or {}

    @property
    def primary_metric_name(self) -> str:
        return "PolicyAccuracy"

    @property
    def optimization_mode(self) -> str:
        return "max"

    def preprocess_batch(self, images, targets, is_training, rng=None):
        # "images" est en realite la position echecs (nom generique herite de la
        # signature TaskStrategy - meme situation que CenterNetDetectionStrategy).
        # targets est deja un dict {POLICY_KEY, VALUE_KEY} (batche par tf.data,
        # Story 9.3) - simple cast, pas de mixup/label smoothing (non pertinents ici,
        # meme choix que CenterNetDetectionStrategy.preprocess_batch).
        targets = {
            POLICY_KEY: jnp.asarray(targets[POLICY_KEY], dtype=jnp.int32),
            VALUE_KEY: jnp.asarray(targets[VALUE_KEY], dtype=jnp.float32),
        }
        return images, targets, False

    def compute_loss(self, outputs, targets, **kwargs):
        return compute_chess_policy_value_loss(outputs, targets, **self.loss_params)

    def compute_metrics(self, outputs, targets):
        # Policy top-1 accuracy (primary_metric_name) - meme formule que
        # ClassificationStrategy.compute_metrics. La value head est entrainee (via
        # compute_loss) mais ne gate rien ici (AD-24).
        predicted = jnp.argmax(outputs[POLICY_KEY], axis=-1)
        return (predicted == targets[POLICY_KEY]).mean()

    def generate_reports(self, val_ds, final_state, model, config):
        # AD-24/AC3 : detail policy_loss/value_loss visible UNIQUEMENT ici (jamais dans
        # le log epoch-par-epoch de Trainer) - reutilise self.loss_params et les memes
        # sous-fonctions que compute_loss (compute_chess_policy_loss/compute_chess_value_loss),
        # jamais reimplemente localement.
        try:
            batch_consumed = False
            for batch_positions, batch_targets in val_ds.take(1).as_numpy_iterator():
                batch_consumed = True
                vars = {'params': final_state.params, 'batch_stats': final_state.batch_stats}
                outputs = final_state.apply_fn(vars, batch_positions, training=False)

                policy_targets = jnp.asarray(batch_targets[POLICY_KEY], dtype=jnp.int32)
                value_targets = jnp.asarray(batch_targets[VALUE_KEY], dtype=jnp.float32)

                policy_loss = compute_chess_policy_loss(outputs[POLICY_KEY], policy_targets)
                value_loss = compute_chess_value_loss(outputs[VALUE_KEY], value_targets)
                policy_weight = self.loss_params.get("policy_weight", 1.0)
                value_weight = self.loss_params.get("value_weight", 1.0)

                print(f"📊 Détail loss échecs (validation, 1 batch) : "
                      f"policy_loss={float(policy_loss):.4f} (poids={policy_weight}), "
                      f"value_loss={float(value_loss):.4f} (poids={value_weight})")
                break
            if not batch_consumed:
                print("⚠️  val_ds est vide - aucun détail policy/value loss généré")
        except Exception as e:
            print(f"❌ Erreur lors de la génération du rapport échecs: {e}")
