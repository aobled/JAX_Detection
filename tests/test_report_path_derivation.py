"""
Story de normalisation des chemins de sauvegarde (2026-09-04) : verifie
TaskStrategy._get_report_path (task_strategies.py) - meme discipline que
_get_export_path/get_training_state_path (deja couverts par test_jax_detector_config.py/
test_centernet_detection_strategy.py), etendue au fallback confusion_matrix_path
utilise par ClassificationStrategy et KeplerStrategy.generate_reports.

Usage: python3 tests/test_report_path_derivation.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from task_strategies import ClassificationStrategy, KeplerStrategy


def test_classification_report_path_follows_dataset_name_pattern():
    strategy = ClassificationStrategy(num_classes=10)
    config = {"dataset_name": "CIFAR10"}
    assert strategy._get_report_path(config, "confusion_matrix") == "confusion_matrix_cifar10.png"
    print("OK - test_classification_report_path_follows_dataset_name_pattern")


def test_kepler_report_path_follows_dataset_name_pattern():
    strategy = KeplerStrategy(num_classes=2)
    config = {"dataset_name": "JAX_KEPLER"}
    assert strategy._get_report_path(config, "kepler_lightcurves_report") == "kepler_lightcurves_report_jax_kepler.png"
    print("OK - test_kepler_report_path_follows_dataset_name_pattern")


def test_explicit_confusion_matrix_path_overrides_derivation():
    strategy = ClassificationStrategy(num_classes=32)
    config = {"dataset_name": "FIGHTERJET_CLASSIFICATION", "confusion_matrix_path": "custom_report.png"}
    assert strategy._get_report_path(config, "confusion_matrix") == "custom_report.png"
    print("OK - test_explicit_confusion_matrix_path_overrides_derivation")


def test_falsy_confusion_matrix_path_falls_back_to_derivation():
    # `or` (pas `.get(key, default)`) : une cle presente mais vide/None doit quand meme
    # retomber sur la derivation auto, coherent avec _get_export_path/get_training_state_path
    # (edge case releve en revue de code, 2026-09-04).
    strategy = ClassificationStrategy(num_classes=10)
    for falsy_value in (None, ""):
        config = {"dataset_name": "CIFAR10", "confusion_matrix_path": falsy_value}
        assert strategy._get_report_path(config, "confusion_matrix") == "confusion_matrix_cifar10.png"
    print("OK - test_falsy_confusion_matrix_path_falls_back_to_derivation")


if __name__ == "__main__":
    test_classification_report_path_follows_dataset_name_pattern()
    test_kepler_report_path_follows_dataset_name_pattern()
    test_explicit_confusion_matrix_path_overrides_derivation()
    test_falsy_confusion_matrix_path_falls_back_to_derivation()
