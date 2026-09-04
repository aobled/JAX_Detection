"""
Story 8.6 : verifie build_single_pass_predict_fn (inference_utils.py) - assemblage
complet RESIZE -> detecteur -> pics/Top-K -> RESCALE -> CROP+normalisation ->
classification, sur les checkpoints reels JAX_DETECTOR (best_model_jax_detector.pkl) et
FIGHTERJET_CLASSIFICATION (best_model_fighterjet_classification.pkl).

Usage: python3 test_single_pass_predict_fn.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ast
import inspect

import jax.numpy as jnp
import numpy as np
from PIL import Image

from inference_utils import build_single_pass_predict_fn

# Image reelle (pas factice) - necessaire pour que valid_mask contienne un melange de
# slots valides/invalides et exerce reellement la verification de coherence ci-dessous
# (trouvaille en revue independante Story 8.6 : sur une image factice sans avion, le
# detecteur ne produit aucun score au-dessus du seuil, la partie "melange" de la
# verification de coherence ne s'executait jamais).
REAL_TEST_IMAGE_PATH = "test_media/testvid01.png"


def _load_real_test_image_grayscale():
    img = Image.open(REAL_TEST_IMAGE_PATH).convert("L")  # AD-12 : canonique grayscale
    arr = np.asarray(img, dtype=np.float32)[..., None]  # (1080,1920,1), pixels bruts
    return jnp.asarray(arr)


def test_end_to_end_output_contract():
    """Task 4 : image REELLE 1920x1080 grayscale (test_media, avions reels) ->
    predict_fn(image) -> verifie forme et cles exactes du dict de sortie, absence de
    NaN/crash, valid_mask coherent avec detection_scores (AC3)."""
    predict_fn = build_single_pass_predict_fn()

    image = _load_real_test_image_grayscale()
    result = predict_fn(image)

    expected_keys = {"boxes", "classes", "class_scores", "detection_scores", "valid_mask"}
    assert set(result.keys()) == expected_keys, f"cles inattendues: {result.keys()}"

    assert result["boxes"].shape == (20, 4), result["boxes"].shape
    assert result["classes"].shape == (20,), result["classes"].shape
    assert result["class_scores"].shape == (20,), result["class_scores"].shape
    assert result["detection_scores"].shape == (20,), result["detection_scores"].shape
    assert result["valid_mask"].shape == (20,), result["valid_mask"].shape
    assert result["valid_mask"].dtype == jnp.bool_, result["valid_mask"].dtype

    for key in expected_keys:
        assert not jnp.any(jnp.isnan(result[key].astype(jnp.float32))), f"NaN detecte dans {key}"

    # valid_mask doit etre monotone avec detection_scores : aucun slot valide ne doit
    # avoir un score <= un slot invalide (verification de coherence d'ordre, pas une
    # re-derivation exacte du seuil prive de la config).
    num_valid = int(jnp.sum(result["valid_mask"]))
    print(f"   -> {num_valid}/20 slots valides sur l'image reelle test_media/testvid01.png")
    if jnp.any(result["valid_mask"]) and jnp.any(~result["valid_mask"]):
        min_valid_score = jnp.min(jnp.where(result["valid_mask"], result["detection_scores"], jnp.inf))
        max_invalid_score = jnp.max(jnp.where(~result["valid_mask"], result["detection_scores"], -jnp.inf))
        assert min_valid_score > max_invalid_score, (
            f"incoherence valid_mask/detection_scores : score valide min {min_valid_score} "
            f"<= score invalide max {max_invalid_score}"
        )
        print(f"   -> coherence valid_mask/detection_scores exercee et verifiee (melange valide/invalide reel)")
    else:
        print(f"   -> ATTENTION: pas de melange valide/invalide sur cette image, verification de coherence non exercee")

    print("OK - test_end_to_end_output_contract (20 slots, cles/formes/types corrects, aucun NaN)")
    return predict_fn


def _called_function_names(source):
    """Extrait les noms de fonctions REELLEMENT APPELEES (ast.Call), pas une simple
    recherche de sous-chaine sur le texte source - une recherche naive donnerait un faux
    positif sur le docstring de build_single_pass_predict_fn lui-meme, qui NOMME ces
    fonctions interdites en prose pour documenter qu'elles ne sont pas appelees."""
    tree = ast.parse(source)
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                names.add(node.func.id)
            elif isinstance(node.func, ast.Attribute):
                names.add(node.func.attr)
    return names


def test_no_old_pipeline_functions_called():
    """Task 5 : confirme explicitement (analyse AST des appels reels dans le code source
    de la fonction ajoutee, pas une simple affirmation ni un grep naif sur le texte) qu'
    aucun appel a decode_segmentation_and_detect(_batch) ou non_max_suppression n'a lieu
    (AD-20 - ancien pipeline non touche)."""
    source = inspect.getsource(build_single_pass_predict_fn)
    called = _called_function_names(source)
    forbidden = {"decode_segmentation_and_detect", "decode_segmentation_and_detect_batch", "non_max_suppression"}
    violations = called & forbidden
    assert not violations, (
        f"{violations} reellement appele(s) dans build_single_pass_predict_fn - violation "
        f"AD-20, l'ancien pipeline ne doit jamais etre appele par ce nouveau chemin"
    )
    print("OK - test_no_old_pipeline_functions_called (analyse AST des appels reels, AD-20 confirme)")


def test_predict_fn_reusable_across_calls():
    """Task 1/3 : les modeles sont charges UNE SEULE FOIS a la construction (hors JIT) -
    predict_fn doit etre reutilisable sur plusieurs images sans recharger les modeles."""
    predict_fn = build_single_pass_predict_fn()
    image_a = jnp.zeros((1080, 1920, 1))
    image_b = jnp.full((1080, 1920, 1), 128.0)

    result_a = predict_fn(image_a)
    result_b = predict_fn(image_b)

    assert result_a["boxes"].shape == (20, 4)
    assert result_b["boxes"].shape == (20, 4)
    print("OK - test_predict_fn_reusable_across_calls (meme predict_fn, 2 images differentes)")


if __name__ == "__main__":
    test_end_to_end_output_contract()
    test_no_old_pipeline_functions_called()
    test_predict_fn_reusable_across_calls()
    print("Tous les tests sont passés.")
