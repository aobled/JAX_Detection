import sys
import os
import threading
import queue
import concurrent.futures
import time

# Réduit les crashs cuDNN autotune (GTX 1660 Ti) — doit être défini avant import jax
os.environ.setdefault(
    "XLA_FLAGS",
    "--xla_gpu_strict_conv_algorithm_picker=true",
)

# Ajouter le répertoire parent en PRIORITÉ absolue (index 0) pour forcer Python
# à utiliser le model_library.py de jax_supervised_training (et non celui de JAX_Classification si exécuté depuis l'autre dossier)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Remplace YOLO import par JAX/Flax helpers si nécessaire
# from ultralytics import YOLO  <-- REMOVED
import cv2
import numpy as np
import jax
import jax.numpy as jnp
from tqdm import tqdm

from dataset_configs import get_dataset_config
from inference_utils import build_single_pass_predict_fn

# =================================================================================================
# Input CONFIGURATION
# =================================================================================================
# 1. Configuration du dataset et du modèle de classification
DATASET_NAME = "FIGHTERJET_CLASSIFICATION"     # Nom de la config dans dataset_configs.py
CHECKPOINT_PATH = "best_model.pkl"      # Chemin vers le modèle de CLASSIFICATION

# 2. Configuration du modèle de détection (Story 8.6 : JAX_DETECTOR remplace l'ancien
# best_model_detection.pkl/AircraftDetectorUNet - celui-ci reste disponible et
# fonctionnel pour l'ancien pipeline FIGHTERJET_DETECTION, AD-20, non touché ici).
DETECTOR_CHECKPOINT_PATH = "best_model_jax_detector.pkl"

CANVAS_WIDTH = 1920
CANVAS_HEIGHT = 1080
# Sigma du flou gaussien GLOBAL applique une seule fois a la heatmap synthetique (voir
# _render_synthetic_heatmap) - remplace le falloff calcule par-ellipse (cout ~ nombre de
# candidats x taille de boite, mesure jusqu'a 64% du temps total sur une scene dense,
# 2026-07-19) par un flou a cout fixe (~ taille du canevas), independant du contenu.
HEATMAP_BLUR_SIGMA = 14

# ==========================================================
# Detection CONFIGURATION
# ==========================================================
OUTPUT_DIR = "/home/aobled/Downloads/video_frames_annotated"
FRAME_STRIDE = 1  # 1 = toutes les frames
BATCH_SIZE = 8                         # Batch single-pass (réduire si OOM GPU, ex. GTX 1660 Ti 6 Go)
# Nombre max de boîtes détectées par frame (Top-K sur le heatmap, inference_utils.py::
# build_single_pass_predict_fn). Purement un choix d'inférence, figé à la construction de
# predict_fn ci-dessous - AUCUN réentraînement requis pour le changer. À adapter par vidéo
# selon le nombre réel d'avions attendus (discussion perf 2026-07-19, voir deferred-work.md) :
# plus petit = crop+classification plus rapide, mais toute frame avec plus de MAX_BOXES avions
# réels en perdrait silencieusement l'excédent (uniquement les pics de heatmap les moins confiants).
MAX_BOXES = 7

#VIDEO_PATH = "/home/aobled/Downloads/testvid.mp4"
#TARGET_CLASS_LIST = ["f15", "f22", "b1b", "b2", "b52", "a10", "f16"]
#VIDEO_PATH = "/home/aobled/Downloads/testvid2.mp4"
#TARGET_CLASS_LIST = ["f15", "rafale", "mirage2000"]
#VIDEO_PATH = "/home/aobled/Downloads/eaa2.mp4"
#TARGET_CLASS_LIST = ["f35", "a10", "f22", "f16", "c130"]
#VIDEO_PATH = "/home/aobled/Downloads/DEATH VALLEY - THE LAST SHOW OF FORCE  (4K).mp4"
#TARGET_CLASS_LIST = ["f18", "f15", "f16", "f35", "a10", "c17", "f22","harrier"]
#VIDEO_PATH = "/media/aobled/Elements/Python/videos/Awesome F-22 Raptor Tail slide in Full control.mp4"
#TARGET_CLASS_LIST = ["f22"]
#VIDEO_PATH = "/media/aobled/Elements/Python/videos/flypast over Paris for Bastille Day.mp4"
#TARGET_CLASS_LIST = ["alphajet", "rafale", "mirage2000", "f35", "f16", "f18", "hawkeye", "a400m", "typhoon"]
VIDEO_PATH = "/media/aobled/Elements/Python/videos/250th USA.mp4"
TARGET_CLASS_LIST = ["f22", "b2", "b1b", "f16"]



# 3. Chargement de la config dataset
try:
    config = get_dataset_config(DATASET_NAME)
    CLASS_NAMES = config["class_names"]
    print(f"✅ Configuration chargée: {DATASET_NAME}")
    print(f"📊 Classes ({len(CLASS_NAMES)}): {CLASS_NAMES}")
    #print(f"🔒 Seuil de confiance (Classification): {CONFIDENCE_THRESHOLD * 100}%")
except Exception as e:
    print(f"❌ Erreur chargement config: {e}")
    sys.exit(1)

# Seuil de detection JAX_DETECTOR (meme source que celui applique a valid_mask dans
# build_single_pass_predict_fn) - recupere ici uniquement pour recalibrer la colorimetrie
# du rendu heatmap (voir _remap_score_for_colormap), sans dupliquer sa valeur en dur.
DETECTION_SCORE_THRESHOLD = get_dataset_config("JAX_DETECTOR")["detection_score_threshold"]

# Seuil IoU du NMS JAX-natif (meme source que celui applique dans
# build_single_pass_predict_fn, 2026-07-22, voir deferred-work.md) - recupere ici pour
# etre passe explicitement a la construction de predict_fn ci-dessous, meme discipline
# que MAX_BOXES : un choix d'inference reglable sans reentrainement, a adapter au cas
# par cas (scenes tres denses -> abaisser ; sur-suppression suspectee -> relever).
NMS_IOU_THRESHOLD = get_dataset_config("JAX_DETECTOR")["nms_iou_threshold"]





def warmup_jit_predictors(batched_predict_fn, batch_size):
    """Compile les kernels JIT du single-pass (Story 8.6) une seule fois avant la boucle
    vidéo (Story 8.7, Task 2bis) - même précaution que l'ancien warmup séparé
    détection/classification, adaptée à l'unique callable batché (vmap) de la Story 8.6.
    Sans cela, le premier lot de la boucle chronométrée paierait le coût de compilation
    JIT complet, faussant la comparaison de débit AD-6 (Task 7) par un artefact de
    mesure, pas une vraie régression."""
    dummy_batch = jnp.zeros((batch_size, CANVAS_HEIGHT, CANVAS_WIDTH, 1), dtype=jnp.float32)
    print(f"   Warmup single-pass predict_fn (batch={batch_size})...")
    result = batched_predict_fn(dummy_batch)
    jax.tree_util.tree_map(lambda x: x.block_until_ready(), result)


def _draw_score_weighted_ellipse(heatmap_2d, center, semi_axes, score):
    """Remplit une ellipse PLEINE (teinte uniforme = `score`), centrée sur `center`=(cx,cy),
    de demi-axes `semi_axes`=(a,b).

    Dimensionnée pour ÉPOUSER l'étendue de la boîte (pas la formule CornerNet
    `_gaussian_radius` de `detection_target_encoding.py`/Story 7.1, essayée d'abord et
    écartée après contrôle visuel : elle produit un pic étroit adapté à l'entraînement
    d'un heatmap CenterNet, pas au rendu "masque plein" attendu ici, cf.
    `archive/old video detection render.png`). Remplissage plat via `cv2.ellipse` (natif,
    rapide) - PAS de falloff gaussien calculé ici (retire le `exp(-4*r²)` par-pixel du
    2026-07-19 matin : son coût scalait avec le nombre de candidats x taille de boîte,
    mesuré jusqu'à 64% du temps total sur une scène dense/3 vidéos comparées - régression
    "temps non-constant" contraire à la propriété attendue du Single-Pass, retour
    utilisateur 2026-07-19 soir). Le falloff visuel est maintenant obtenu par UN SEUL flou
    gaussien global appliqué par `_render_synthetic_heatmap` après tous les remplissages -
    coût fixe, indépendant du contenu. Écrit dans un patch local (ROI autour du centre,
    tronqué aux bords de l'image) puis blend max avec l'existant - jamais de plein-cadre
    alloué par détection. Utilisée uniquement pour la visualisation, jamais sur le chemin
    d'inférence critique.
    """
    h, w = heatmap_2d.shape[:2]
    cx, cy = int(round(center[0])), int(round(center[1]))
    a, b = max(int(round(semi_axes[0])), 1), max(int(round(semi_axes[1])), 1)

    x0, x1 = max(cx - a, 0), min(cx + a + 1, w)
    y0, y1 = max(cy - b, 0), min(cy + b + 1, h)
    if x1 <= x0 or y1 <= y0:
        return  # boite hors image, rien a dessiner

    patch = np.zeros((y1 - y0, x1 - x0), dtype=np.float32)
    cv2.ellipse(patch, (cx - x0, cy - y0), (a, b), 0, 0, 360, float(score), thickness=-1)
    region = heatmap_2d[y0:y1, x0:x1]
    np.maximum(region, patch, out=region)


def _render_synthetic_heatmap(boxes, detection_scores, indices, canvas_size):
    """Reconstruit une carte de chaleur visuelle à partir des boîtes/scores déjà exposés
    par le contrat de sortie fixe de `build_single_pass_predict_fn` (AD-15/AC3 Story 8.6 -
    inchangé) plutôt que d'exposer la carte dense interne du détecteur. Décidé avec
    l'utilisateur en remplacement de l'affichage simplifié initial de la Story 8.7
    (2026-07-18).

    `indices` : ensemble d'indices à dessiner - PAS forcément `valid_mask` seul. Les 20
    slots de `_top_k_boxes` (Story 8.3) sont déjà tous calculés avant filtrage par
    `detection_score_threshold` (confirmé lors du diagnostic threshold_sensitivity,
    2026-07-18) ; appeler cette fonction avec tous les indices (0..19) plutôt que
    `valid_mask` seul restaure la valeur exploratoire de l'ancien rendu heatmap dense
    UNet (voir "hésitations" sous le seuil, retour utilisateur 2026-07-19).

    Falloff "chaleur" obtenu par UN SEUL `cv2.GaussianBlur` global en fin de fonction
    (2026-07-19 soir), pas par ellipse - voir `_draw_score_weighted_ellipse`."""
    h, w = canvas_size
    heatmap = np.zeros((h, w), dtype=np.float32)
    for idx in indices:
        score = float(detection_scores[idx])
        if score < 0.02:
            continue  # quasi-invisible une fois recolorise, evite le cout de rendu pour rien
        x1, y1, x2, y2 = boxes[idx]
        box_w, box_h = x2 - x1, y2 - y1
        if box_w <= 0 or box_h <= 0:
            continue
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        semi_axes = (box_w / 2.0, box_h / 2.0)
        _draw_score_weighted_ellipse(heatmap, (cx, cy), semi_axes, score)
    heatmap = cv2.GaussianBlur(heatmap, (0, 0), sigmaX=HEATMAP_BLUR_SIGMA)
    return heatmap


def _remap_score_for_colormap(heatmap, threshold, confirmed_band_start=0.5):
    """Réétire la plage de scores utile avant application de la palette JET, pour que
    tout le spectre bleu->rouge soit exploité au lieu de sa seule moitié basse (les
    scores réels se concentrent autour de 0.5-0.75 - retour utilisateur 2026-07-19 :
    rendu "coincé entre le vert et le bleu clair").

    Deux bandes distinctes de part et d'autre de `threshold` (= `detection_score_threshold`,
    JAX_DETECTOR) :
    - [0, threshold)   -> [0, confirmed_band_start)          : candidats sous le seuil,
      tons froids (bleu/cyan) - "le modèle hésite ici".
    - [threshold, 1.0] -> [confirmed_band_start, 1.0]         : détections confirmées,
      tons chauds (vert->rouge). confirmed_band_start=0.5 (pas 0.35) : JET(0.35) reste
      cyan/bleu clair, pas assez distinct du fond - retour utilisateur 2026-07-19 (2e
      passe) : les détections confirmées restaient trop peu visibles. JET(0.5) est vert
      franc, garantit que toute détection confirmée est au moins verte.

    Le fond (heatmap==0, aucune détection) reste à 0 dans les deux cas -> inchangé,
    toujours JET(0)."""
    threshold = max(float(threshold), 1e-6)
    below = heatmap < threshold
    remapped = np.empty_like(heatmap)
    remapped[below] = (heatmap[below] / threshold) * confirmed_band_start

    above_span = max(1.0 - threshold, 1e-6)
    remapped[~below] = confirmed_band_start + (heatmap[~below] - threshold) / above_span * (1.0 - confirmed_band_start)
    return remapped


def build_quadrant_canvas(target_frame, results, frame_idx, config):
    """Construit le canvas 4 quadrants (Story 8.7, après migration Single-Pass, Story 8.6).

    `target_frame` DOIT déjà être au repère canonique 1920x1080 (voir
    `run_gpu_inference::_to_canonical_bgr`) - `results["boxes"]` est toujours exprimé
    dans ce même repère (`_rescale_boxes(original_size=(1920,1080))`, figé dans
    `build_single_pass_predict_fn`, Story 8.6, indépendant de la résolution native de la
    frame source). Passer ici une frame à une autre résolution désalignerait
    silencieusement l'affichage/le crop (trouvaille revue indépendante Story 8.7).

    Quadrants bas-gauche/haut-droit (2026-07-18, décidé avec l'utilisateur après examen
    de `archive/old video detection render.png`) : la carte de chaleur dense interne du
    détecteur n'est pas exposée par le contrat de sortie fixe de
    `build_single_pass_predict_fn` (AD-15/AC3, Story 8.6) - reconstruite ici à partir des
    boîtes/scores finaux (`_render_synthetic_heatmap`), sans étendre ce contrat déjà
    finalisé. Pas identique pixel pour pixel à l'activation interne réelle, mais
    visuellement proche de l'ancien rendu (blobs gaussiens colorés par score). Le
    quadrant bas-droit (grille de crops classifiés) perd aussi sa source directe
    (l'ancien chemin recevait le crop déjà extrait) - re-extrait ici via un simple
    slicing sur les boîtes rescalées, uniquement pour l'affichage, indépendant du chemin
    d'inférence JAX (qui ne réexpose jamais ses crops internes).

    Bas-gauche et haut-droit partagent la MEME heatmap exploratoire (tous les candidats,
    pas seulement `valid_mask`) depuis le 2026-07-19 (2e retour utilisateur - la version
    "haut-droit = confirmees seules uniquement" du 2026-07-19 matin retirait un vrai facteur
    d'analyse : impossible de voir les zones "presque detectees" superposees sur l'image
    reelle). Les boites vertes/rouges restent, elles, restreintes a `valid_indices` - seul
    le fond de chaleur redevient partage, un seul calcul/colorisation reutilise pour les
    deux quadrants (plus rapide que les 2 passages separes de la version precedente).
    """
    canvas = np.zeros((CANVAS_HEIGHT, CANVAS_WIDTH, 3), dtype=np.uint8)
    h_orig, w_orig = target_frame.shape[:2]

    valid_mask = np.asarray(results["valid_mask"][frame_idx])
    boxes = np.asarray(results["boxes"][frame_idx])
    classes = np.asarray(results["classes"][frame_idx])
    class_scores = np.asarray(results["class_scores"][frame_idx])
    detection_scores = np.asarray(results["detection_scores"][frame_idx])
    valid_indices = np.where(valid_mask)[0]  # valid_mask seule autorité (AD-15/AC3 8.6) - boites uniquement
    # Retri par position (y1 puis x1, "ordre de lecture" - meme convention que l'ancien
    # pipeline, sorted(key=lambda b: (y1, x1))) : valid_indices herite sinon de l'ordre de
    # _top_k_boxes (jax.lax.top_k, Story 8.3), qui trie par SCORE decroissant - un score qui
    # oscille legerement d'une frame a l'autre suffit a permuter 2 detections dans ce
    # classement, faisant "vibrer" la grille de crops bas-droite (chaque indice = une case
    # fixe) sans que les detections elles-memes n'aient bouge. Tri sur <=20 elements,
    # cout negligeable (retour utilisateur 2026-07-19).
    if valid_indices.size > 0:
        valid_indices = valid_indices[np.lexsort((boxes[valid_indices, 0], boxes[valid_indices, 1]))]
    all_indices = np.arange(detection_scores.shape[0])  # 20 slots bruts - heatmap partagee, voir docstring

    # 1. Top-Left: Original
    tl_img = cv2.resize(target_frame, (960, 540))
    canvas[0:540, 0:960] = tl_img

    # 2/3. Heatmap synthetique EXPLORATOIRE (tous les candidats bruts, gaussiens ponderes
    # par score - voir docstring), calculee UNE SEULE fois et partagee entre le quadrant
    # bas-gauche (fond plein) et l'overlay haut-droit (masque + alpha).
    synthetic_heatmap = _render_synthetic_heatmap(
        boxes, detection_scores, all_indices, (CANVAS_HEIGHT, CANVAS_WIDTH)
    )
    hm_vis = (np.clip(
        _remap_score_for_colormap(synthetic_heatmap, DETECTION_SCORE_THRESHOLD), 0.0, 1.0
    ) * 255).astype(np.uint8)
    hm_color = cv2.applyColorMap(hm_vis, cv2.COLORMAP_JET)
    bl_img = cv2.resize(hm_color, (960, 540))
    canvas[540:1080, 0:960] = bl_img

    # 3. Top-Right: Annotated + overlay heatmap synthetique (pas de contours
    # findContours, AD-9 - les positions des boites sont deja connues directement)
    tr_img = tl_img.copy()
    scale_x = 960 / w_orig
    scale_y = 540 / h_orig

    heatmap_mask = cv2.resize(
        (hm_vis > 40).astype(np.uint8), (960, 540), interpolation=cv2.INTER_NEAREST
    )
    heatmap_color_tr = bl_img.copy()  # meme heatmap partagee que le quadrant bas-gauche, voir docstring
    heatmap_color_tr[heatmap_mask == 0] = 0
    cv2.addWeighted(tr_img, 1.0, heatmap_color_tr, 0.45, 0, dst=tr_img)

    for idx in valid_indices:
        x1, y1, x2, y2 = boxes[idx]
        predicted_class = CLASS_NAMES[int(classes[idx])]
        color = (0, 255, 0) if predicted_class in TARGET_CLASS_LIST else (0, 0, 255)
        label = f"{predicted_class}"

        x1_md, y1_md = int(x1 * scale_x), int(y1 * scale_y)
        x2_md, y2_md = int(x2 * scale_x), int(y2 * scale_y)
        cv2.rectangle(tr_img, (x1_md, y1_md), (x2_md, y2_md), color, 2)
        cv2.putText(tr_img, label, (x1_md, max(y1_md - 10, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # 4. Bottom-Right: crops re-extraits pour l'affichage uniquement (slicing direct sur
    # les boîtes rescalées, boxes_orig - indépendant du chemin d'inférence JAX).
    br_canvas = np.zeros((540, 960, 3), dtype=np.uint8)
    crop_idx = 0
    grid_cols = 7
    grid_rows = 4
    cell_w = 128
    cell_h = 128

    for idx in valid_indices:
        if crop_idx >= grid_cols * grid_rows:
            break
        x1, y1, x2, y2 = boxes[idx]
        x1i, y1i = max(0, int(x1)), max(0, int(y1))
        x2i, y2i = min(w_orig, int(x2)), min(h_orig, int(y2))
        crop = target_frame[y1i:y2i, x1i:x2i]
        if crop.size == 0:
            continue

        crop_resized = cv2.resize(crop, (cell_w, cell_h))
        if config.get("grayscale", False):
            crop_gray = cv2.cvtColor(crop_resized, cv2.COLOR_BGR2GRAY)
            crop_disp = cv2.cvtColor(crop_gray, cv2.COLOR_GRAY2BGR)
        else:
            crop_disp = crop_resized

        predicted_class = CLASS_NAMES[int(classes[idx])]
        confidence = float(class_scores[idx])
        color = (0, 255, 0) if predicted_class in TARGET_CLASS_LIST else (0, 0, 255)

        col = crop_idx % grid_cols
        row = crop_idx // grid_cols
        x_start = col * cell_w + 10 + (col * 5)
        y_start = row * (cell_h + 20) + 20 + (row * 5)

        if x_start + cell_w <= 960 and y_start + cell_h <= 540:
            br_canvas[y_start:y_start + cell_h, x_start:x_start + cell_w] = crop_disp
            label = f"{predicted_class} ({100 * confidence:.1f}%)"
            cv2.putText(br_canvas, label, (x_start, y_start - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        crop_idx += 1

    canvas[0:540, 960:1920] = tr_img
    canvas[540:1080, 960:1920] = br_canvas

    return canvas


def run_gpu_inference(frames_buffer, batched_predict_fn):
    """Étape GPU du pipeline (Single-Pass, Story 8.6) : prétraitement + dispatch
    détection+classification, batchés via vmap - remplace
    decode_segmentation_and_detect_batch + classify_batch_detections.

    Pipelining GPU/CPU (2026-07-19, retour utilisateur : mesure "option A" montrant GPU
    et CPU strictement séquentiels, 0% de recouvrement, alors que le CPU de rendu est
    souvent plus rapide que le GPU - jusqu'à ~2x de débit laissé sur la table) :
    contrairement à la version précédente de cette fonction, AUCUN `jax.block_until_ready`
    ici - les résultats JAX (encore en calcul asynchrone côté GPU) sont poussés tels quels
    vers le thread de rendu (`render_thread_func`) par l'appelant. Le thread principal peut
    ainsi dispatcher le lot suivant sans attendre que ce lot-ci ait fini de calculer sur le
    GPU, pendant que le thread de rendu consomme les lots précédents. Retourne
    (canonical_frames, results, timing) - timing ne couvre QUE le prétraitement et le
    dispatch (rapide, asynchrone) : le temps de calcul GPU réel est désormais mesuré côté
    rendu (premier `np.asarray` sur `results`, dans `build_quadrant_canvas`, point où la
    synchronisation se produit réellement)."""
    # Frame source -> repere canonique BGR 1920x1080 - UNE SEULE FOIS, reutilisee a la
    # fois pour l'entree du modele (converted en gris ensuite) ET pour l'affichage/crop
    # (build_quadrant_canvas) - correction post-revue independante (Story 8.7) : passer
    # la frame BRUTE (resolution source arbitraire) a build_quadrant_canvas tout en
    # gardant des boites toujours exprimees en repere canonique 1920x1080
    # (_rescale_boxes(original_size=(1920,1080)), fige dans build_single_pass_predict_fn,
    # Story 8.6) desalignerait silencieusement l'affichage/le crop sur toute source non
    # nativement 1920x1080 - inoffensif ici car testvid.mp4 est deja 1920x1080, mais une
    # source differente aurait ete un bug latent. Un seul resize, jamais deux divergents.
    def _to_canonical_bgr(frame):
        if frame.shape[:2] != (CANVAS_HEIGHT, CANVAS_WIDTH):
            return cv2.resize(frame, (CANVAS_WIDTH, CANVAS_HEIGHT))
        return frame

    t_start = time.perf_counter()
    canonical_frames = [_to_canonical_bgr(frame) for frame in frames_buffer]
    # Pixels bruts [0,255] (AD-12/Story 8.2, jamais de /255.0 ici - normalisation interne
    # à build_single_pass_predict_fn).
    gray_frames = np.stack([
        cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)[..., None]
        for frame in canonical_frames
    ], axis=0)
    t_preprocess = time.perf_counter()

    results = batched_predict_fn(jnp.asarray(gray_frames))
    t_dispatch = time.perf_counter()

    timing = {
        "preprocess": t_preprocess - t_start,
        "gpu_dispatch": t_dispatch - t_preprocess,
    }
    return canonical_frames, results, timing


def render_batch_canvases(canonical_frames, results, n_frames, config):
    """Étape rendu CPU du pipeline - construit les `n_frames` canevas du lot. Point de
    synchronisation GPU réel (voir docstring `run_gpu_inference`) : le premier `np.asarray`
    sur `results`, dans `build_quadrant_canvas`, bloque jusqu'à ce que le calcul GPU
    dispatché soit terminé - `render_time` inclut donc potentiellement une attente
    résiduelle du GPU, pas seulement le rendu CPU pur."""
    t0 = time.perf_counter()

    def process_single_canvas(i):
        return build_quadrant_canvas(canonical_frames[i], results, i, config)

    with concurrent.futures.ThreadPoolExecutor() as executor:
        canvases = list(executor.map(process_single_canvas, range(n_frames)))

    render_time = time.perf_counter() - t0
    return canvases, render_time


def reader_thread_func(cap, input_queue, frame_stride):
    frame_id = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            input_queue.put(None)  # Signal de fin
            break

        if frame_id % frame_stride == 0:
            input_queue.put(frame.copy())

        frame_id += 1


def render_thread_func(render_queue, output_queue, config, timing_state):
    """Thread dédié au rendu CPU (2026-07-19, pipelining GPU/CPU - voir
    `run_gpu_inference`) - consomme les résultats GPU dispatchés par le thread principal,
    construit les canevas, alimente `output_queue` (déjà consommée par `writer_thread`).
    Tourne en parallèle du thread principal, qui peut dispatcher le lot GPU suivant pendant
    que ce thread rend le lot précédent. Relaie le signal de fin (`None`) vers
    `output_queue` pour terminer `writer_thread` proprement."""
    while True:
        item = render_queue.get()
        if item is None:
            output_queue.put(None)
            break

        canonical_frames, results, n_frames = item
        canvases, render_time = render_batch_canvases(canonical_frames, results, n_frames, config)
        for canvas in canvases:
            output_queue.put(canvas)

        timing_state["cpu_render"] += render_time
        timing_state["saved_count"] += n_frames
        timing_state["pbar"].update(n_frames)


def writer_thread_func(video_writer, output_queue):
    while True:
        canvas = output_queue.get()
        if canvas is None:
            break
        video_writer.write(canvas)

if __name__ == "__main__":
    
    
    print("🏗️ Chargement des modèles (JAX Single-Pass, Story 8.6)...")

    config = get_dataset_config(DATASET_NAME)

    predict_fn = build_single_pass_predict_fn(
        detector_checkpoint_path=DETECTOR_CHECKPOINT_PATH,
        classifier_checkpoint_path=CHECKPOINT_PATH,
        k=MAX_BOXES,
        nms_iou_threshold=NMS_IOU_THRESHOLD,
    )

    print("✅ Modèles chargés.")

    print("⚡ Compilation JAX (JIT)...")
    batched_predict_fn = jax.vmap(predict_fn)
    warmup_jit_predictors(batched_predict_fn, BATCH_SIZE)
    print("✅ Compilation terminée.")

    # ======================================================
    # OUVERTURE VIDEO
    # ======================================================
    cap = cv2.VideoCapture(VIDEO_PATH)

    if not cap.isOpened():
        print("❌ Impossible d'ouvrir la vidéo.")
        sys.exit(1)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if FRAME_STRIDE > 1:
        fps = fps / FRAME_STRIDE
    print(f"🎬 {total_frames} frames détectées")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    OUTPUT_VIDEO_PATH = os.path.join(OUTPUT_DIR, "reconstructed_video.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(
        OUTPUT_VIDEO_PATH, fourcc, fps, (CANVAS_WIDTH, CANVAS_HEIGHT)
    )
    if not video_writer.isOpened():
        print("❌ ERREUR : Impossible d'ouvrir VideoWriter.")
        cap.release()
        sys.exit(1)

    # Initialisation des Queues et Threads
    input_queue = queue.Queue(maxsize=BATCH_SIZE * 3)
    # render_queue borne a 2 lots : laisse le GPU dispatcher en avance sur le rendu
    # (recouvrement recherche) sans le laisser s'emballer indefiniment - chaque lot en
    # attente retient des tableaux JAX vivants en memoire GPU, deja limitee sur ce
    # materiel (2026-07-19, retour utilisateur : OOM a batch=16, marge memoire faible
    # meme a batch=8).
    render_queue = queue.Queue(maxsize=2)
    output_queue = queue.Queue(maxsize=BATCH_SIZE * 3)

    total_to_process = (total_frames + FRAME_STRIDE - 1) // FRAME_STRIDE
    pbar = tqdm(total=total_to_process, desc="Pipeline")

    # timing_state partage avec render_thread (2026-07-19, pipelining GPU/CPU) : chaque cle
    # n'est mutee que par un seul thread (preprocess/gpu_dispatch par le thread principal,
    # cpu_render/saved_count par render_thread) - pas de verrou necessaire, aucune ecriture
    # concurrente sur une meme cle.
    timing_state = {"preprocess": 0.0, "gpu_dispatch": 0.0, "cpu_render": 0.0, "saved_count": 0, "pbar": pbar}

    reader_thread = threading.Thread(target=reader_thread_func, args=(cap, input_queue, FRAME_STRIDE))
    render_thread = threading.Thread(target=render_thread_func, args=(render_queue, output_queue, config, timing_state))
    writer_thread = threading.Thread(target=writer_thread_func, args=(video_writer, output_queue))

    reader_thread.start()
    render_thread.start()
    writer_thread.start()

    frames_buffer = []
    t_wall_start = time.perf_counter()

    while True:
        frame = input_queue.get()
        if frame is None:
            break

        frames_buffer.append(frame)

        if len(frames_buffer) == BATCH_SIZE:
            canonical_frames, results, timing = run_gpu_inference(frames_buffer, batched_predict_fn)
            timing_state["preprocess"] += timing["preprocess"]
            timing_state["gpu_dispatch"] += timing["gpu_dispatch"]
            render_queue.put((canonical_frames, results, len(frames_buffer)))  # bloque si render_queue pleine (backpressure voulue)
            frames_buffer.clear()

    if len(frames_buffer) > 0:
        # Dernier lot partiel (taille != BATCH_SIZE) : paddé à BATCH_SIZE en répétant la
        # dernière frame réelle, pour ne JAMAIS présenter une 2e shape à jax.vmap - mesuré
        # à ~12s de recompilation JIT + recherche cuDNN bloquante en fin de run (retour
        # utilisateur 2026-07-19, timestamps du log) ; auparavant considéré comme un coût
        # ponctuel acceptable, en réalité significatif. Les résultats sur les frames de
        # padding sont ignorés : canonical_frames tronqué à n_real avant le rendu.
        n_real = len(frames_buffer)
        padded_buffer = frames_buffer + [frames_buffer[-1]] * (BATCH_SIZE - n_real)
        canonical_frames, results, timing = run_gpu_inference(padded_buffer, batched_predict_fn)
        canonical_frames = canonical_frames[:n_real]
        timing_state["preprocess"] += timing["preprocess"]
        timing_state["gpu_dispatch"] += timing["gpu_dispatch"]
        render_queue.put((canonical_frames, results, n_real))
        frames_buffer.clear()

    # Signal de fin : render_thread relaie vers output_queue -> writer_thread (voir
    # render_thread_func).
    render_queue.put(None)

    reader_thread.join()
    render_thread.join()
    writer_thread.join()

    t_wall_total = time.perf_counter() - t_wall_start
    saved_count = timing_state["saved_count"]
    pbar.close()

    cap.release()
    video_writer.release()

    print("\n✅ Terminé !")
    print(f"📸 Frames traitées : {saved_count}")
    print(f"📂 Dossier de sortie : {OUTPUT_DIR}")
    if os.path.exists(OUTPUT_VIDEO_PATH):
        print(f"🎥 Vidéo : {OUTPUT_VIDEO_PATH}")
    else:
        print("❌ La vidéo n'a pas été créée.")

    # Repartition temps (option A, 2026-07-19) - adaptee au pipelining GPU/CPU : les
    # buckets "preprocess"/"gpu_dispatch" (thread principal) et "cpu_render" (render_thread)
    # tournent desormais en PARALLELE, donc leur somme ne correspond plus au temps mur reel
    # (t_wall_total, mesure separement) - la comparer aux runs precedents (sequentiels,
    # somme == temps mur) montre directement le gain du recouvrement.
    if t_wall_total > 0 and saved_count > 0:
        print("\n⏱️  Pipelining GPU/CPU (2026-07-19) - buckets paralleles, ne pas sommer :")
        print(f"   Pré-traitement CPU (dispatch)   : {timing_state['preprocess']:7.2f}s")
        print(f"   Dispatch GPU (async, hors calcul): {timing_state['gpu_dispatch']:7.2f}s")
        print(f"   Rendu CPU (incl. attente GPU)   : {timing_state['cpu_render']:7.2f}s")
        print(f"   Temps mur total (réel, mesuré)  : {t_wall_total:7.2f}s")
        print(f"   Débit combiné réel              : {saved_count / t_wall_total:.2f} it/s")