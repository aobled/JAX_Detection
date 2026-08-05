"""
Test standalone pour la tolerance de ChessPolicyValueDataset a l'absence de "value"
dans les chunks .npz (Story 10.1, Epic 10 - dataset chess_search_teacher, contrat #2.6),
et pour le champ value_head_trained derive automatiquement (FR4).
Execution: python3 tests/test_chess_search_teacher_loader.py
"""

import sys
import os

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)

import shutil
import tempfile

import jax
import jax.numpy as jnp
import numpy as np

from data_management import ChessPolicyValueDataset
from dataset_configs import DATASET_CONFIGS, get_dataset_config
from loss_functions import compute_chess_policy_value_loss


# Source depuis la config reelle plutot qu'un litteral duplique - suit le contrat si
# num_channels change un jour (revue de code, 2026-08-04).
NUM_PLANES = DATASET_CONFIGS["CHESS_SEARCH_TEACHER"]["num_channels"]
N_EXAMPLES = 3
_POLICIES = np.array([10, 200, 4000], dtype=np.int32)
_VALUES = np.array([1.0, -1.0, 0.0], dtype=np.float32)


def _write_chunk(out_dir, prefix, with_value):
    os.makedirs(out_dir, exist_ok=True)
    positions = np.random.rand(N_EXAMPLES, 8, 8, NUM_PLANES).astype(np.float32)
    arrays = {"position": positions, "policy": _POLICIES}
    if with_value:
        arrays["value"] = _VALUES
    np.savez(os.path.join(out_dir, f"{prefix}_chunk0.npz"), **arrays)
    return os.path.join(out_dir, prefix)


def test_chunk_without_value_key_defaults_to_zero():
    """AC3 : un chunk sans clé "value" (contrat chess_search_teacher, #2.6) se charge
    sans KeyError, et produit value=0.0 pour chaque exemple."""
    out_dir = tempfile.mkdtemp(prefix="test_cst_novalue_")
    try:
        output_prefix = _write_chunk(out_dir, "chess_search_teacher", with_value=False)

        dataset = ChessPolicyValueDataset(output_prefix, batch_size=N_EXAMPLES, val_split=0.1, num_planes=NUM_PLANES)
        ds = dataset.create_tf_dataset('train')
        pos_batch, targets = next(iter(ds.take(1)))

        assert pos_batch.shape == (N_EXAMPLES, 8, 8, NUM_PLANES)
        assert targets["policy"].shape == (N_EXAMPLES,)
        assert targets["value"].shape == (N_EXAMPLES,)
        assert np.allclose(targets["value"].numpy(), 0.0), "value doit etre 0.0 partout quand la cle est absente"
        print("OK - test_chunk_without_value_key_defaults_to_zero")
    finally:
        shutil.rmtree(out_dir, ignore_errors=True)


def test_chunk_with_value_key_unaffected():
    """AC4 : non-regression - un chunk qui fournit deja "value" (format CHESS_NO_HISTORY,
    simule ici car aucun chunk reel n'existe en local) charge ses vraies valeurs, inchangees
    et correctement appariees a leur policy (pas seulement le meme multi-ensemble - revue de
    code 2026-08-04, un tri seul ne detecterait pas un desalignement position/value dans le
    zip)."""
    out_dir = tempfile.mkdtemp(prefix="test_cst_withvalue_")
    try:
        output_prefix = _write_chunk(out_dir, "chess_no_history_fake", with_value=True)
        expected_value_by_policy = dict(zip(_POLICIES.tolist(), _VALUES.tolist()))

        dataset = ChessPolicyValueDataset(output_prefix, batch_size=N_EXAMPLES, val_split=0.1, num_planes=NUM_PLANES)
        ds = dataset.create_tf_dataset('train')
        _, targets = next(iter(ds.take(1)))

        policies = targets["policy"].numpy().tolist()
        values = targets["value"].numpy().tolist()
        assert set(policies) == set(expected_value_by_policy), f"policies inattendues: {policies}"
        for pol, val in zip(policies, values):
            assert val == expected_value_by_policy[pol], (
                f"policy {pol}: value {val} != attendu {expected_value_by_policy[pol]} "
                f"(desalignement position/policy/value)"
            )
        print("OK - test_chunk_with_value_key_unaffected")
    finally:
        shutil.rmtree(out_dir, ignore_errors=True)


def test_dummy_value_never_produces_nan_loss():
    """AC5 (garde-fou pre-mortem) : value_weight=0.0 avec une cible/predite value dans
    [-1,1] ne produit jamais de NaN - 0.0 * NaN = NaN sinon, ce test verifie le cas reel
    plutot que de le supposer."""
    batch = 4
    outputs = {
        "policy": np.random.randn(batch, 4672).astype(np.float32),
        "value": np.tanh(np.random.randn(batch).astype(np.float32)),  # borne [-1,1], comme la tete reelle
    }
    targets = {
        "policy": np.array([0, 1, 2, 3], dtype=np.int32),
        "value": np.zeros(batch, dtype=np.float32),  # value factice (Story 10.1)
    }

    loss = compute_chess_policy_value_loss(outputs, targets, policy_weight=1.0, value_weight=0.0)

    assert not np.isnan(float(loss)), "la loss ne doit jamais etre NaN meme avec value_weight=0.0"
    print("OK - test_dummy_value_never_produces_nan_loss")


def test_dummy_value_produces_zero_gradient_on_value_head():
    """AC5 (2e moitie) : value_weight=0.0 annule reellement le gradient de la tete value,
    verifie via jax.grad plutot que suppose par la seule absence de NaN - revue de code
    2026-08-04 (Blind Hunter + Acceptance Auditor : la 1ere version ne testait que le NaN,
    jamais le gradient, malgre la story qui le revendiquait "verifie")."""
    batch = 4
    policy_targets = jnp.array([0, 1, 2, 3], dtype=jnp.int32)
    value_targets = jnp.zeros(batch, dtype=jnp.float32)  # value factice (Story 10.1)

    def loss_fn(value_pred, policy_logits):
        outputs = {"policy": policy_logits, "value": value_pred}
        targets = {"policy": policy_targets, "value": value_targets}
        return compute_chess_policy_value_loss(outputs, targets, policy_weight=1.0, value_weight=0.0)

    value_pred = jnp.tanh(jnp.array(np.random.randn(batch).astype(np.float32)))
    policy_logits = jnp.array(np.random.randn(batch, 4672).astype(np.float32))

    grad_wrt_value = jax.grad(loss_fn, argnums=0)(value_pred, policy_logits)

    assert np.allclose(np.asarray(grad_wrt_value), 0.0), (
        f"le gradient de la tete value doit etre nul quand value_weight=0.0, recu {grad_wrt_value}"
    )
    print("OK - test_dummy_value_produces_zero_gradient_on_value_head")


def test_value_head_trained_derived_correctly():
    """AC6/AC7 : value_head_trained derive automatiquement de loss_params.value_weight
    (get_dataset_config) - couverture automatisee (revue de code 2026-08-04 : la 1ere
    version ne verifiait ceci qu'a la main via un script python3 -c ponctuel, jamais dans
    un test persistant)."""
    assert get_dataset_config("CHESS_NO_HISTORY")["value_head_trained"] is True, (
        "CHESS_NO_HISTORY a value_weight=1.0 explicite -> value_head_trained doit etre True"
    )
    assert get_dataset_config("CHESS_SEARCH_TEACHER")["value_head_trained"] is False, (
        "CHESS_SEARCH_TEACHER a value_weight=0.0 -> value_head_trained doit etre False"
    )
    assert "value_head_trained" not in get_dataset_config("CHESS_LEGAL_MOVES"), (
        "CHESS_LEGAL_MOVES n'a pas de loss_params.value_weight (pas de tete value) -> "
        "value_head_trained ne doit pas etre ajoute du tout"
    )
    print("OK - test_value_head_trained_derived_correctly")


if __name__ == "__main__":
    test_chunk_without_value_key_defaults_to_zero()
    test_chunk_with_value_key_unaffected()
    test_dummy_value_never_produces_nan_loss()
    test_dummy_value_produces_zero_gradient_on_value_head()
    test_value_head_trained_derived_correctly()
    print("Tous les tests sont passés.")
