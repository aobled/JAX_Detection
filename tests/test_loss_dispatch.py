"""
Test standalone pour le dispatch de loss par domaine (ClassificationStrategy /
DetectionStrategy, docs/loss_audit.md). Verifie que CLASSIFICATION_LOSS_FUNCTIONS et
DETECTION_LOSS_FUNCTIONS restent des registres separes : une loss_method du mauvais
domaine doit lever la ValueError de la Strategy, jamais s'executer silencieusement.
Execution: python3 test_loss_dispatch.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax.numpy as jnp

from task_strategies import ClassificationStrategy, DetectionStrategy


def test_classification_valid_method_returns_finite_loss():
    strategy = ClassificationStrategy(num_classes=3, loss_method="cross_entropy")
    outputs = jnp.array([[2.0, 0.5, 0.1]])
    targets = jnp.array([0])
    loss = strategy.compute_loss(outputs, targets, use_onehot_labels=False)
    assert jnp.isfinite(loss)
    print("OK - test_classification_valid_method_returns_finite_loss")


def test_classification_unknown_method_raises():
    strategy = ClassificationStrategy(num_classes=3, loss_method="does_not_exist")
    try:
        strategy.compute_loss(jnp.zeros((1, 3)), jnp.array([0]))
        assert False, "ValueError attendue pour une loss_method inconnue"
    except ValueError:
        pass
    print("OK - test_classification_unknown_method_raises")


def test_classification_detection_only_method_raises():
    # Regression: "grid" n'appartient qu'au domaine detection - doit rester absent
    # de CLASSIFICATION_LOSS_FUNCTIONS, pas s'executer avec un mauvais signal/forme.
    strategy = ClassificationStrategy(num_classes=3, loss_method="grid")
    try:
        strategy.compute_loss(jnp.zeros((1, 3)), jnp.array([0]))
        assert False, "ValueError attendue pour une loss_method du domaine detection"
    except ValueError:
        pass
    print("OK - test_classification_detection_only_method_raises")


def test_detection_valid_method_returns_finite_loss():
    strategy = DetectionStrategy(loss_method="segmentation")
    outputs = jnp.zeros((1, 4, 4, 1))
    targets = jnp.zeros((1, 4, 4, 1))
    loss = strategy.compute_loss(outputs, targets)
    assert jnp.isfinite(loss)
    print("OK - test_detection_valid_method_returns_finite_loss")


def test_detection_unknown_method_raises():
    strategy = DetectionStrategy(loss_method="does_not_exist")
    try:
        strategy.compute_loss(jnp.zeros((1, 4, 4, 1)), jnp.zeros((1, 4, 4, 1)))
        assert False, "ValueError attendue pour une loss_method inconnue"
    except ValueError:
        pass
    print("OK - test_detection_unknown_method_raises")


def test_detection_classification_only_method_raises():
    # Regression: "cross_entropy"/"focal_loss" n'appartiennent qu'au domaine classification -
    # avant le split de registre, ceci s'executait silencieusement avec la mauvaise loss.
    strategy = DetectionStrategy(loss_method="cross_entropy")
    try:
        strategy.compute_loss(jnp.zeros((1, 4, 4, 1)), jnp.zeros((1, 4, 4, 1)))
        assert False, "ValueError attendue pour une loss_method du domaine classification"
    except ValueError:
        pass
    print("OK - test_detection_classification_only_method_raises")


if __name__ == "__main__":
    test_classification_valid_method_returns_finite_loss()
    test_classification_unknown_method_raises()
    test_classification_detection_only_method_raises()
    test_detection_valid_method_returns_finite_loss()
    test_detection_unknown_method_raises()
    test_detection_classification_only_method_raises()
    print("Tous les tests sont passés.")
