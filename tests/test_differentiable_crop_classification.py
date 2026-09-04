"""
Story 8.5 : verifie _differentiable_crop/_normalize_crop_for_classifier (nouveaux,
inference_utils.py) composes avec load_jax_model/build_clf_predict_fn (existants, non
modifies) sur le checkpoint reel FIGHTERJET_CLASSIFICATION (best_model_fighterjet_classification.pkl).

Usage: python3 test_differentiable_crop_classification.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax.numpy as jnp

from inference_utils import (
    load_jax_model,
    build_clf_predict_fn,
    _differentiable_crop,
    _normalize_crop_for_classifier,
)
from dataset_configs import get_dataset_config

CHECKPOINT_PATH = "best_model_fighterjet_classification.pkl"  # FIGHTERJET_CLASSIFICATION (nommage derive de dataset_name, Story 5.0)


def test_load_jax_model_unmodified_for_classification():
    """Task 3 - meme discipline de verification que Story 8.2 pour load_detection_model :
    aucune modification de load_jax_model necessaire pour cette story."""
    config = get_dataset_config("FIGHTERJET_CLASSIFICATION")
    model, variables, mean, std = load_jax_model(CHECKPOINT_PATH, config)

    assert model is not None and variables is not None
    assert "params" in variables and "batch_stats" in variables
    assert mean is not None and std is not None
    print(f"OK - test_load_jax_model_unmodified_for_classification (mean={mean}, std={std})")
    return config, model, variables, mean, std


def test_build_clf_predict_fn_contract_on_20_slot_batch(model, variables):
    """Task 4 - build_clf_predict_fn (non modifie) retourne (probs, pred_indices) sur un
    batch (20,128,128,1), probs etant la distribution COMPLETE par slot (pas un score
    scalaire deja pret) - class_scores doit etre calcule explicitement en aval."""
    predict_fn = build_clf_predict_fn(model, variables)
    dummy_batch = jnp.zeros((20, 128, 128, 1))

    probs, pred_indices = predict_fn(dummy_batch)

    assert probs.shape == (20, 32), probs.shape  # 32 classes, FIGHTERJET_CLASSIFICATION
    assert pred_indices.shape == (20,), pred_indices.shape
    # probs = distribution complete, pas un score deja calcule : verifie qu'un gather
    # explicite (jnp.max) est necessaire pour obtenir class_scores (AC3).
    class_scores = jnp.max(probs, axis=-1)
    assert class_scores.shape == (20,), class_scores.shape
    assert jnp.allclose(jnp.sum(probs, axis=-1), 1.0, atol=1e-4), (
        "probs doit etre une distribution softmax valide (somme=1) par slot"
    )
    print("OK - test_build_clf_predict_fn_contract_on_20_slot_batch")
    return predict_fn


def test_differentiable_crop_geometry_only_no_normalization():
    """_differentiable_crop ne doit jamais normaliser (pixels bruts en entree ET en
    sortie) - seule _normalize_crop_for_classifier divise par 255."""
    raw_image = jnp.full((1080, 1920, 1), 200.0)
    boxes = jnp.array([[500.0, 300.0, 700.0, 450.0]] + [[0.0, 0.0, 0.0, 0.0]] * 19)

    crops = _differentiable_crop(raw_image, boxes, crop_size=(128, 128))
    assert crops.shape == (20, 128, 128, 1), crops.shape
    assert jnp.abs(crops[0].mean() - 200.0) < 1.0, (
        f"mean={crops[0].mean():.3f}, attendu ~200.0 (pixels bruts, pas normalises)"
    )
    print("OK - test_differentiable_crop_geometry_only_no_normalization")


def test_boxes_are_truncated_not_kept_floating():
    """Verifie que la troncature (jnp.trunc, decision Story 8.1/SPEC.md, voir correction
    des Dev Notes de cette story) est REELLEMENT appliquee, pas seulement documentee -
    une boite a coordonnees non-entieres doit produire EXACTEMENT le meme crop qu'une
    boite deja tronquee a la main, PAS le crop qu'on obtiendrait avec les coordonnees
    flottantes brutes (les deux resultats different reellement, ce test le prouve)."""
    raw_image = jnp.arange(1080 * 1920, dtype=jnp.float32).reshape(1080, 1920, 1) % 256.0

    fractional_box = jnp.array([[100.7, 100.3, 300.9, 250.1]] + [[0.0, 0.0, 0.0, 0.0]] * 19)
    pre_truncated_box = jnp.array([[100.0, 100.0, 300.0, 250.0]] + [[0.0, 0.0, 0.0, 0.0]] * 19)

    crop_from_fractional = _differentiable_crop(raw_image, fractional_box, crop_size=(128, 128))
    crop_from_pre_truncated = _differentiable_crop(raw_image, pre_truncated_box, crop_size=(128, 128))

    # Les deux doivent etre identiques : _differentiable_crop tronque en interne, donc
    # partir d'une boite fractionnaire ou d'une boite deja tronquee doit donner le meme
    # resultat (preuve que la troncature interne est reellement appliquee).
    assert jnp.array_equal(crop_from_fractional[0], crop_from_pre_truncated[0]), (
        "le crop issu d'une boite fractionnaire differe du crop issu de la meme boite "
        "pre-tronquee - la troncature interne (jnp.trunc) n'est pas appliquee correctement"
    )

    # Contre-preuve : si on simule le chemin SANS troncature (calcul manuel avec les
    # coordonnees flottantes brutes), le resultat DOIT differer - sinon ce test ne
    # discriminerait rien (le "self-confirming trap" deja evite en Story 8.4).
    def _crop_without_truncation(image, box, out_size=128):
        x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
        scale_x = (x2 - x1) / out_size
        scale_y = (y2 - y1) / out_size
        dst_y, dst_x = jnp.meshgrid(jnp.arange(out_size), jnp.arange(out_size), indexing="ij")
        src_x = x1 + (dst_x.astype(jnp.float32) + 0.5) * scale_x - 0.5
        src_y = y1 + (dst_y.astype(jnp.float32) + 0.5) * scale_y - 0.5
        from jax.scipy.ndimage import map_coordinates as _mc
        return _mc(image[..., 0], [src_y, src_x], order=1, mode="nearest")

    crop_without_trunc_manual = _crop_without_truncation(raw_image, fractional_box[0])
    assert not jnp.array_equal(crop_from_fractional[0, :, :, 0], crop_without_trunc_manual), (
        "le crop tronque et le crop non-tronque (calcul manuel de reference) sont "
        "identiques - ce test ne discrimine rien, la boite fractionnaire choisie ne "
        "produit pas un ecart mesurable, revoir les coordonnees de test"
    )
    print("OK - test_boxes_are_truncated_not_kept_floating (troncature interne prouvee, pas seulement documentee)")


def test_end_to_end_crop_normalize_classify_no_crash_including_degenerate_boxes(
    config, model, variables, mean, std, predict_fn
):
    """Task 5 - image source factice 1920x1080, 20 boites dont certaines degenerees
    (coordonnees a zero, AD-15 zero-padding des slots invalides) -> crop -> normalisation
    -> classification. Verifie l'absence de crash/NaN, y compris sur les boites
    degenerees (pas d'attente de resultat "sense" sur ces slots, valid_mask reste seule
    autorite en aval, Story 8.6)."""
    raw_image = jnp.asarray(
        (jnp.arange(1080 * 1920).reshape(1080, 1920, 1) % 256).astype(jnp.float32)
    )

    valid_boxes = [
        [100.0, 100.0, 300.0, 250.0],
        [1919.0, 1079.0, 1919.0, 1079.0],  # boite degeneree non-nulle (taille 0) au bord
        [0.0, 0.0, 50.0, 50.0],
    ]
    degenerate_zero_boxes = [[0.0, 0.0, 0.0, 0.0]] * (20 - len(valid_boxes))
    boxes = jnp.array(valid_boxes + degenerate_zero_boxes)
    assert boxes.shape == (20, 4)

    crops = _differentiable_crop(raw_image, boxes, crop_size=tuple(config["image_size"]))
    assert crops.shape == (20, *config["image_size"], 1), crops.shape
    assert not jnp.any(jnp.isnan(crops)), "NaN detecte dans les crops (avant meme la normalisation)"

    normalized = _normalize_crop_for_classifier(crops, mean, std)
    assert not jnp.any(jnp.isnan(normalized)), "NaN detecte apres normalisation"

    probs, pred_indices = predict_fn(normalized)
    assert not jnp.any(jnp.isnan(probs)), "NaN detecte dans les probabilites de classification"
    assert probs.shape == (20, config["num_classes"])
    assert pred_indices.shape == (20,)

    class_scores = jnp.max(probs, axis=-1)
    assert jnp.all(class_scores > 0.0) and jnp.all(class_scores <= 1.0), (
        "class_scores doit rester une probabilite valide, y compris sur les slots degeneres"
    )
    print(
        f"OK - test_end_to_end_crop_normalize_classify_no_crash_including_degenerate_boxes "
        f"(20 slots, dont {len(degenerate_zero_boxes)} degeneres a zero, aucun crash/NaN)"
    )


if __name__ == "__main__":
    config, model, variables, mean, std = test_load_jax_model_unmodified_for_classification()
    predict_fn = test_build_clf_predict_fn_contract_on_20_slot_batch(model, variables)
    test_differentiable_crop_geometry_only_no_normalization()
    test_boxes_are_truncated_not_kept_floating()
    test_end_to_end_crop_normalize_classify_no_crash_including_degenerate_boxes(
        config, model, variables, mean, std, predict_fn
    )
    print("Tous les tests sont passés.")
