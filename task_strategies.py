import jax
import jax.numpy as jnp
import optax
from abc import ABC, abstractmethod
from loss_functions import (
    compute_centernet_loss,
    compute_chess_policy_value_loss, compute_chess_policy_loss,
    compute_chess_value_loss, compute_chess_legal_moves_loss,
    compute_chess_token_candidate_loss, compute_chess_token_1_move_loss,
    compute_chess_token_1_move_joint_accuracy, CHESS_TOKEN_1_MOVE_NUM_MOVE_TYPES,
    CLASSIFICATION_LOSS_FUNCTIONS, DETECTION_LOSS_FUNCTIONS,
)
from detection_target_encoding import HEATMAP_KEY, SIZE_KEY
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
        loss_fn = CLASSIFICATION_LOSS_FUNCTIONS.get(self.loss_method)
        if loss_fn is None:
            raise ValueError(f"Méthode de loss '{self.loss_method}' non supportée pour la classification.")
        return loss_fn(outputs, targets, use_onehot_labels=use_onehot_labels, **self.loss_params)

            
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
        loss_fn = DETECTION_LOSS_FUNCTIONS.get(self.loss_method)
        if loss_fn is None:
            raise ValueError(f"Méthode de loss '{self.loss_method}' non supportée pour la détection.")
        return loss_fn(outputs, targets, **self.loss_params)

        
        
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
    def __init__(self, num_classes: int, loss_params: dict = None, metric_method: str = "accuracy", report_method: str = "lightcurves"):
        self.num_classes = num_classes
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
    pattern TaskStrategy. outputs/targets sont des dicts {"policy", "value"}
    (contrat .npz cote chess_ai, AD-18), jamais un tenseur unique.
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
        # targets est deja un dict {"policy", "value"} (batche par tf.data,
        # Story 9.3) - simple cast, pas de mixup/label smoothing (non pertinents ici,
        # meme choix que CenterNetDetectionStrategy.preprocess_batch).
        targets = {
            "policy": jnp.asarray(targets["policy"], dtype=jnp.int32),
            "value": jnp.asarray(targets["value"], dtype=jnp.float32),
        }
        return images, targets, False

    def compute_loss(self, outputs, targets, **kwargs):
        return compute_chess_policy_value_loss(outputs, targets, **self.loss_params)

    def compute_metrics(self, outputs, targets):
        # Policy top-1 accuracy (primary_metric_name) - meme formule que
        # ClassificationStrategy.compute_metrics. La value head est entrainee (via
        # compute_loss) mais ne gate rien ici (AD-24).
        predicted = jnp.argmax(outputs["policy"], axis=-1)
        return (predicted == targets["policy"]).mean()

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

                policy_targets = jnp.asarray(batch_targets["policy"], dtype=jnp.int32)
                value_targets = jnp.asarray(batch_targets["value"], dtype=jnp.float32)

                policy_loss = compute_chess_policy_loss(outputs["policy"], policy_targets)
                value_loss = compute_chess_value_loss(outputs["value"], value_targets)
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


class ChessLegalMovesStrategy(TaskStrategy):
    """
    Tache multi-label : predire l'ensemble des coups legaux d'une position (pas le
    seul coup joue, contrairement a ChessPolicyValueStrategy ci-dessus). outputs
    est un tenseur unique (B, 4672) - PAS un dict {"policy", "value"} (pas de tete
    value, ChessCnnAttentionLegalMoves, model_library.py).
    """
    def __init__(self, metric_threshold: float = 0.5, loss_params: dict = None):
        # metric_threshold : seuil sur sigmoid(logits) pour decider "predit legal"
        # (compute_metrics) - n'affecte pas compute_loss (BCE continue, pas de seuil).
        self.metric_threshold = metric_threshold
        # loss_params : { "pos_weight": float } - meme convention que
        # ChessPolicyValueStrategy (policy_weight/value_weight) meme si un seul
        # terme ici. pos_weight=1.0 par defaut (BCE non ponderee, comportement
        # d'origine avant le 1er run reel du 2026-08-02).
        self.loss_params = loss_params or {}

    @property
    def primary_metric_name(self) -> str:
        return "LegalMoveF1"

    @property
    def optimization_mode(self) -> str:
        return "max"

    def preprocess_batch(self, images, targets, is_training, rng=None):
        # "images" est en realite la position echecs (nom generique herite de la
        # signature TaskStrategy). targets = legal_mask (B, 4672), int8 au
        # chargement (contrat .npz chess_legal_moves) - cast float32 pour la BCE.
        targets = jnp.asarray(targets, dtype=jnp.float32)
        return images, targets, False

    def compute_loss(self, outputs, targets, **kwargs):
        return compute_chess_legal_moves_loss(outputs, targets, pos_weight=self.loss_params.get("pos_weight", 1.0))

    def _precision_recall_f1(self, outputs, targets):
        # Factorise ici (compute_metrics ET generate_reports en avaient besoin,
        # copie-collee au premier jet - trouve en revue adversariale) : F1 sur la
        # classe "legal", PAS l'accuracy brute bit-a-bit - un plateau a ~20-40
        # coups legaux sur 4672, "tout illegal" donnerait deja ~99% d'accuracy
        # sans rien apprendre, inutilisable comme signal ici.
        predicted = (jax.nn.sigmoid(outputs) >= self.metric_threshold).astype(jnp.float32)
        true_positives = jnp.sum(predicted * targets)
        predicted_positives = jnp.sum(predicted)
        actual_positives = jnp.sum(targets)
        # Garde division-par-zero (jnp.where, pas de NaN) : un batch peut n'avoir
        # aucune prediction positive (debut d'entrainement) ou, en theorie, aucun
        # coup legal (mat/pat, jamais dans les positions initiales mais possible
        # sur d'autres positions du dataset).
        precision = jnp.where(predicted_positives > 0, true_positives / predicted_positives, 0.0)
        recall = jnp.where(actual_positives > 0, true_positives / actual_positives, 0.0)
        f1 = jnp.where(precision + recall > 0, 2 * precision * recall / (precision + recall), 0.0)
        return precision, recall, f1

    def compute_metrics(self, outputs, targets):
        _, _, f1 = self._precision_recall_f1(outputs, targets)
        return f1

    def generate_reports(self, val_ds, final_state, model, config):
        # Detail precision/rappel (1 batch validation) - meme convention que
        # ChessPolicyValueStrategy.generate_reports ci-dessus (jamais logue
        # epoch-par-epoque par Trainer, uniquement ici en fin d'entrainement).
        try:
            batch_consumed = False
            for batch_positions, batch_targets in val_ds.take(1).as_numpy_iterator():
                batch_consumed = True
                vars = {'params': final_state.params, 'batch_stats': final_state.batch_stats}
                outputs = final_state.apply_fn(vars, batch_positions, training=False)
                targets = jnp.asarray(batch_targets, dtype=jnp.float32)

                precision, recall, _ = self._precision_recall_f1(outputs, targets)

                print(f"📊 Détail coups légaux (validation, 1 batch) : "
                      f"precision={float(precision):.4f}, rappel={float(recall):.4f}")
                break
            if not batch_consumed:
                print("⚠️  val_ds est vide - aucun détail précision/rappel généré")
        except Exception as e:
            print(f"❌ Erreur lors de la génération du rapport coups légaux: {e}")


class ChessMoveTokenStrategy(TaskStrategy):
    """
    Tâche policy-only sur historique de coups (Epic 11, spike — AD-26, spine
    architecture-chess-move-token-2026-08-10) : prédit le coup suivant à partir de la
    séquence de coup-tokens (move_tokens) jouée jusqu'à la position courante. outputs
    est un tenseur unique (B, 4672) — PAS un dict {"policy", "value"} (aucune tête
    value, ChessMoveTokenTransformer, model_library.py) — même forme que
    ChessLegalMovesStrategy ci-dessus, pas ChessPolicyValueStrategy.

    compute_loss délègue intégralement à compute_chess_policy_loss (déjà existante,
    loss_functions.py) — aucune nouvelle fonction de loss créée (AD-26).
    """
    def __init__(self, loss_params: dict = None):
        # loss_params : { "label_smoothing": float } - meme convention que les
        # stratégies échecs ci-dessus (dict passé à l'instanciation).
        self.loss_params = loss_params or {}

    @property
    def primary_metric_name(self) -> str:
        return "PolicyAccuracy"

    @property
    def optimization_mode(self) -> str:
        return "max"

    def preprocess_batch(self, images, targets, is_training, rng=None):
        # "images" est en realite la sequence de move-tokens (nom generique herite de
        # la signature TaskStrategy - meme situation que ChessPolicyValueStrategy).
        # targets = index du coup joue (B,), deja int32 au chargement
        # (ChessMoveTokenDataset) - cast defensif, meme discipline que les autres
        # strategies echecs.
        targets = jnp.asarray(targets, dtype=jnp.int32)
        return images, targets, False

    def compute_loss(self, outputs, targets, **kwargs):
        return compute_chess_policy_loss(
            outputs, targets, label_smoothing=self.loss_params.get("label_smoothing", 0.0)
        )

    def compute_metrics(self, outputs, targets):
        # Policy top-1 accuracy (primary_metric_name) - meme formule que
        # ChessPolicyValueStrategy.compute_metrics ci-dessus.
        predicted = jnp.argmax(outputs, axis=-1)
        return (predicted == targets).mean()

    def generate_reports(self, val_ds, final_state, model, config):
        try:
            batch_consumed = False
            for batch_sequences, batch_targets in val_ds.take(1).as_numpy_iterator():
                batch_consumed = True
                vars = {'params': final_state.params, 'batch_stats': final_state.batch_stats}
                outputs = final_state.apply_fn(vars, batch_sequences, training=False)
                targets = jnp.asarray(batch_targets, dtype=jnp.int32)

                policy_loss = compute_chess_policy_loss(
                    outputs, targets, label_smoothing=self.loss_params.get("label_smoothing", 0.0)
                )
                acc = self.compute_metrics(outputs, targets)

                print(f"📊 Détail chess_move_token (validation, 1 batch) : "
                      f"policy_loss={float(policy_loss):.4f}, PolicyAccuracy={float(acc):.4f}")
                break
            if not batch_consumed:
                print("⚠️  val_ds est vide - aucun détail chess_move_token généré")
        except Exception as e:
            print(f"❌ Erreur lors de la génération du rapport chess_move_token: {e}")


class ChessTokenStrategy(TaskStrategy):
    """
    Tâche de scoring de candidats légaux (spec-chess-token-candidate-model,
    2026-08-13, spike, CAP-2) : prédit lequel des `candidate_mask.sum()` slots
    candidats réels (sur num_candidates=50 slots, dont certains sont du padding) est
    le coup choisi par le professeur (Stockfish profondeur 12, contrat .npz
    chess_token_candidate_spike côté chess_ai). outputs est un tenseur unique
    (B, num_candidates) - PAS un dict {"policy", "value"} (aucune tête value,
    ChessTokenCandidateModel, model_library.py) - même forme que
    ChessMoveTokenStrategy/ChessLegalMovesStrategy ci-dessus.

    targets est un DICT {"candidate_label": (B,) int32, "candidate_mask": (B, 50)} -
    PAS un tenseur unique (contrairement à ChessMoveTokenStrategy) : le masquage de la
    loss/métrique a besoin de candidate_mask en plus du label, et candidate_mask ne
    peut pas voyager côté "images" (le modèle n'en a pas besoin en interne, voir
    docstring ChessTokenCandidateModel - safe_moves neutralise déjà les slots de
    padding) - même patron dict-target que CenterNetDetectionStrategy
    ({HEATMAP_KEY, SIZE_KEY}, task_strategies.py), pas un nouveau mécanisme.

    compute_loss délègue intégralement à compute_chess_token_candidate_loss
    (loss_functions.py, nouvelle - CAP-2) - aucune loss existante ne masque sur un
    espace de sortie variable (50 slots), contrairement à ChessMoveTokenStrategy qui
    réutilise compute_chess_policy_loss telle quelle (AD-26).
    """
    def __init__(self, loss_params: dict = None):
        # loss_params.label_smoothing (2026-08-14, Aymeric, overfitting observe apres
        # token_dim 64->128) : lu ici et transmis a compute_chess_token_candidate_loss,
        # qui implemente sa PROPRE variante masquee (voir sa docstring) - PAS
        # smooth_labels/compute_chess_policy_loss (repartirait la masse sur des slots
        # de padding). Meme convention dict que les strategies echecs ci-dessus
        # (loss_params toujours accepte).
        self.loss_params = loss_params or {}

    @property
    def label_smoothing(self) -> float:
        return self.loss_params.get("label_smoothing", 0.0)

    @property
    def primary_metric_name(self) -> str:
        return "PolicyAccuracy"

    @property
    def optimization_mode(self) -> str:
        return "max"

    def preprocess_batch(self, images, targets, is_training, rng=None):
        # "images" est en realite l'entree packee (token_position|global_flags|
        # candidate_moves) - nom generique herite de la signature TaskStrategy (meme
        # situation que ChessMoveTokenStrategy/ChessPolicyValueStrategy).
        # targets = dict {"candidate_label": (B,), "candidate_mask": (B,50)} - deja
        # int32/int8 au chargement (ChessTokenCandidateDataset), cast defensif ici
        # (meme discipline que les autres strategies echecs) : candidate_label en
        # int32 (labels entiers, cross-entropy), candidate_mask en int32 (pas bool -
        # compute_chess_token_candidate_loss appelle .astype(bool) lui-meme, un
        # tenseur numerique reste plus simple a batcher/logger qu'un bool).
        targets = {
            "candidate_label": jnp.asarray(targets["candidate_label"], dtype=jnp.int32),
            "candidate_mask": jnp.asarray(targets["candidate_mask"], dtype=jnp.int32),
        }
        return images, targets, False

    def compute_loss(self, outputs, targets, **kwargs):
        return compute_chess_token_candidate_loss(
            outputs, targets["candidate_label"], targets["candidate_mask"],
            label_smoothing=self.label_smoothing,
        )

    def compute_metrics(self, outputs, targets):
        # Top-1 argmax MASQUE (primary_metric_name = PolicyAccuracy) : meme masquage
        # additif -1e9 que compute_chess_token_candidate_loss (loss_functions.py) -
        # sans lui, argmax pourrait selectionner un slot de padding (logit non
        # contraint, poids initialises aleatoirement) et gonfler/degonfler
        # artificiellement l'accuracy mesuree.
        #
        # PRECONDITION NON VERIFIEE ICI (meme choix que compute_chess_token_candidate_loss,
        # voir sa docstring, loss_functions.py) : si une ligne avait `candidate_mask` a 0
        # partout, `masked_outputs` vaudrait -1e9 partout et `jnp.argmax` retournerait le
        # slot 0 par tie-break, faussant silencieusement PolicyAccuracy. Cette methode est
        # appelee depuis `Trainer._train_step`/`_eval_step`, tous deux `@jax.jit`
        # (trainer.py:229/261) - un assert Python usuel sur un tracer y echouerait
        # (ConcretizationTypeError), d'ou l'absence deliberee de garde ici. Garanti en amont
        # par `ChessTokenCandidateDataset.__init__` (data_management.py) : `candidate_label`
        # pointe toujours un slot ou `candidate_mask == 1`, ce qui implique
        # `candidate_mask.sum(axis=-1) >= 1` pour chaque ligne du dataset reel.
        candidate_mask = targets["candidate_mask"]
        masked_outputs = jnp.where(candidate_mask.astype(bool), outputs, -1e9)
        predicted = jnp.argmax(masked_outputs, axis=-1)
        return (predicted == targets["candidate_label"]).mean()

    def generate_reports(self, val_ds, final_state, model, config):
        try:
            batch_consumed = False
            for batch_inputs, batch_targets in val_ds.take(1).as_numpy_iterator():
                batch_consumed = True
                vars = {'params': final_state.params, 'batch_stats': final_state.batch_stats}
                outputs = final_state.apply_fn(vars, batch_inputs, training=False)
                targets = {
                    "candidate_label": jnp.asarray(batch_targets["candidate_label"], dtype=jnp.int32),
                    "candidate_mask": jnp.asarray(batch_targets["candidate_mask"], dtype=jnp.int32),
                }

                loss = compute_chess_token_candidate_loss(
                    outputs, targets["candidate_label"], targets["candidate_mask"]
                )
                acc = self.compute_metrics(outputs, targets)

                print(f"📊 Détail chess_token (validation, 1 batch) : "
                      f"loss={float(loss):.4f}, PolicyAccuracy={float(acc):.4f}")
                break
            if not batch_consumed:
                print("⚠️  val_ds est vide - aucun détail chess_token généré")
        except Exception as e:
            print(f"❌ Erreur lors de la génération du rapport chess_token: {e}")


class ChessTokenOneMoveStrategy(TaskStrategy):
    """
    Tâche policy-only à tête factorisée (spec-chess-token-1-move, 2026-08-15, spike,
    contrat chess_ai §3 CHESS_TOKEN_1_MOVE) : prédit directement le coup joué dans
    l'espace complet (from_square 64 x move_type 73 = 4672), sans filtrage préalable par
    candidats légaux (contrairement à ChessTokenStrategy ci-dessus) - l'illégalité de
    coup redevient un échec mesurable (ChessTokenOneMoveModel, model_library.py, voir sa
    docstring pour le rationale complet).

    outputs est un DICT {"from_square": (B,64), "move_type": (B,73)} - PAS un tenseur
    unique (contrairement à ChessTokenStrategy/ChessMoveTokenStrategy/
    ChessLegalMovesStrategy ci-dessus) : 2 têtes indépendantes, aucun conditionnement de
    l'une sur l'autre (contrainte explicite du contrat).

    targets est un DICT {"move_index": (B,) int32} - le label RÉEL unique déjà dérivé
    par ChessTokenOneMoveDataset (data_management.py, candidate_moves[i,
    candidate_label[i]]) - PAS from_square/move_type précalculés : la décomposition
    divmod(move_index, 73) se fait à la volée dans compute_chess_token_1_move_loss/
    compute_chess_token_1_move_joint_accuracy (loss_functions.py), jamais ici.

    compute_loss délègue intégralement à compute_chess_token_1_move_loss (loss_functions.py,
    nouvelle) - aucune loss existante ne combine 2 têtes softmax indépendantes de tailles
    différentes sans masquage.

    ATTENTION lecture des résultats (v2, spec-chess-token-1-move-v2, trouvaille Blind
    Hunter 2026-08-16) : ChessTokenOneMoveModel v2 conditionne move_type sur from_square
    par teacher-forcing UNIQUEMENT quand training=True (argmax(from_square_logits) sinon,
    voir sa docstring). compute_metrics ci-dessous est appelé aussi bien par
    Trainer._train_step (training=True) que _eval_step (training=False) sur le MÊME
    outputs dict que compute_loss a déjà reçu - la JointMoveAccuracy loguée côté TRAIN
    reflète donc le chemin teacher-forcé (plus facile), PAS le chemin argmax utilisé côté
    VAL. Le diagnostic "train≈val => plafond de capacité, pas overfitting" (déjà utilisé
    pour v1) N'EST PLUS directement comparable train vs val pour v2 : seule la valeur VAL
    est comparable au plafond v1 (5,10%) ou entre runs v2. Un écart train>val ne doit pas
    être lu comme de l'overfitting classique sans tenir compte de cette asymétrie
    structurelle. Recalculer une accuracy train "libre" (argmax) nécessiterait un forward
    pass supplémentaire dans Trainer._train_step - hors scope (contrainte "ne pas toucher
    trainer.py" du spec v1/v2).

    compute_metrics DOIT rester un scalaire unique (compilé sous @jax.jit,
    Trainer._train_step/_eval_step, trainer.py:229/261, formaté directement avec :.4f -
    même contrainte que toutes les stratégies existantes ci-dessus) : retourne
    l'accuracy JOINTE (from_square ET move_type corrects simultanément sur le même
    exemple, primary_metric_name="JointMoveAccuracy") - LA métrique de comparaison
    contractuelle face à CHESS_SEARCH_TEACHER (28.00% val PolicyAccuracy, contrat chess_ai
    §3, ChessPolicyValueStrategy ci-dessus). L'accuracy PAR TÊTE (from_square seul /
    move_type seul - diagnostic, PAS la métrique de comparaison) n'est disponible que via
    generate_reports (hors-jit, en fin d'entraînement) - même discipline que
    ChessLegalMovesStrategy.generate_reports ci-dessus (precision/rappel en plus du F1
    retourné par compute_metrics).
    """
    def __init__(self, loss_params: dict = None):
        # loss_params : { "from_square_weight": float, "move_type_weight": float,
        # "label_smoothing": float } - même convention dict que les autres stratégies
        # échecs ci-dessus. Défauts (1.0/1.0/0.0) = comportement neutre : poids égal
        # entre les 2 têtes, pas de label smoothing - point de départ NON TUNÉ (comme
        # ChessLegalMovesStrategy.pos_weight/ChessTokenStrategy.label_smoothing à leur
        # création).
        self.loss_params = loss_params or {}

    @property
    def primary_metric_name(self) -> str:
        return "JointMoveAccuracy"

    @property
    def optimization_mode(self) -> str:
        return "max"

    def preprocess_batch(self, images, targets, is_training, rng=None):
        # "images" est en realite l'entree packee (token_position|global_flags) - nom
        # generique herite de la signature TaskStrategy (meme situation que
        # ChessTokenStrategy/ChessMoveTokenStrategy ci-dessus). targets = dict
        # {"move_index": (B,)}, deja int32 au chargement (ChessTokenOneMoveDataset) -
        # cast defensif ici, meme discipline que les autres strategies echecs.
        targets = {"move_index": jnp.asarray(targets["move_index"], dtype=jnp.int32)}
        return images, targets, False

    def compute_loss(self, outputs, targets, **kwargs):
        return compute_chess_token_1_move_loss(
            outputs, targets["move_index"],
            from_square_weight=self.loss_params.get("from_square_weight", 1.0),
            move_type_weight=self.loss_params.get("move_type_weight", 1.0),
            label_smoothing=self.loss_params.get("label_smoothing", 0.0),
        )

    def compute_metrics(self, outputs, targets):
        # Accuracy JOINTE (primary_metric_name) - voir docstring classe : seul scalaire
        # retourne ici, compatible @jax.jit (contrairement a un dict de metriques par
        # tete, qui casserait le formatage :.4f de Trainer).
        return compute_chess_token_1_move_joint_accuracy(outputs, targets["move_index"])

    def generate_reports(self, val_ds, final_state, model, config):
        # Detail par tete (from_square/move_type separement, diagnostic - PAS la
        # metrique de comparaison contractuelle) en plus de l'accuracy jointe - meme
        # convention que ChessLegalMovesStrategy.generate_reports ci-dessus
        # (precision/rappel en plus du F1 de compute_metrics).
        try:
            batch_consumed = False
            for batch_inputs, batch_targets in val_ds.take(1).as_numpy_iterator():
                batch_consumed = True
                vars = {'params': final_state.params, 'batch_stats': final_state.batch_stats}
                outputs = final_state.apply_fn(vars, batch_inputs, training=False)
                move_index = jnp.asarray(batch_targets["move_index"], dtype=jnp.int32)

                loss = compute_chess_token_1_move_loss(outputs, move_index)
                joint_acc = compute_chess_token_1_move_joint_accuracy(outputs, move_index)

                from_square_target, move_type_target = jnp.divmod(move_index, CHESS_TOKEN_1_MOVE_NUM_MOVE_TYPES)
                from_square_acc = (jnp.argmax(outputs["from_square"], axis=-1) == from_square_target).mean()
                move_type_acc = (jnp.argmax(outputs["move_type"], axis=-1) == move_type_target).mean()

                print(f"📊 Détail chess_token_1_move (validation, 1 batch) : "
                      f"loss={float(loss):.4f}, JointMoveAccuracy={float(joint_acc):.4f}, "
                      f"from_square_acc={float(from_square_acc):.4f}, move_type_acc={float(move_type_acc):.4f}")
                break
            if not batch_consumed:
                print("⚠️  val_ds est vide - aucun détail chess_token_1_move généré")
        except Exception as e:
            print(f"❌ Erreur lors de la génération du rapport chess_token_1_move: {e}")


# Dispatch task_type -> classe Strategy (Story 12.2, AD-21) - remplace le if/elif
# a 9 branches de main.py. Source unique du dispatch (AD-17 : meme litteral
# task_type que data_management.py/model_library.py MODELS).
STRATEGIES = {
    "classification": ClassificationStrategy,
    "detection": DetectionStrategy,
    "kepler": KeplerStrategy,
    "detection_centernet": CenterNetDetectionStrategy,
    "chess_policy_value": ChessPolicyValueStrategy,
    "chess_legal_moves": ChessLegalMovesStrategy,
    "chess_move_token": ChessMoveTokenStrategy,
    "chess_token": ChessTokenStrategy,
    "chess_token_1_move": ChessTokenOneMoveStrategy,
}


# Cles de dataset_configs.py forwardees sans condition vers les kwargs Strategy
# quand presentes, par task_type (Story 12.2, meme precedent que
# model_library.MODEL_FORWARDED_CONFIG_KEYS, Story 12.1) - AUCUNE des 9 classes
# Strategy ci-dessus n'a de **kwargs catch-all : un forwarding non scope par
# task_type (une liste plate partagee) leverait un TypeError des qu'une classe
# recevrait un champ qu'elle ne declare pas (ex. label_smoothing, propre a
# ClassificationStrategy, vers DetectionStrategy). Contenu derive des branches
# reelles de main.py avant migration - jamais un defaut ajoute ici qui ne soit
# pas deja le defaut propre du constructeur cible.
_LOSS_PARAMS_ONLY = ("loss_params",)  # les 5 strategies "une seule methode de perte/metrique" (voir classes ci-dessus)

STRATEGY_FORWARDED_CONFIG_KEYS = {
    "classification": ("label_smoothing", "mixup_alpha", "loss_method", "loss_params", "metric_method", "report_method"),
    "detection": ("loss_method", "loss_params", "metric_method", "report_method"),
    "kepler": ("loss_params", "metric_method", "report_method"),
    "detection_centernet": _LOSS_PARAMS_ONLY,
    "chess_policy_value": _LOSS_PARAMS_ONLY,
    "chess_legal_moves": ("metric_threshold", "loss_params"),
    "chess_move_token": _LOSS_PARAMS_ONLY,
    "chess_token": _LOSS_PARAMS_ONLY,
    "chess_token_1_move": _LOSS_PARAMS_ONLY,
}
