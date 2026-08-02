"""
Test de validation pour ChessCnnAttentionLegalMoves (tache multi-label "coups
legaux", spec-chess-legal-moves.md).

Script autonome - ce projet n'a pas de framework de test formel (meme convention
que tests/test_chess_model.py). Executer directement :
    python tests/test_chess_legal_moves_model.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
import jax.numpy as jnp

# Constantes du contrat .npz chess_legal_moves (cote chess_ai) - litteraux, meme
# convention que test_chess_model.py (plus d'import depuis un module local de codec).
NUM_MOVES = 4672
NUM_PLANES = 29

from model_library import (
    ChessCnnAttentionLegalMoves,
    create_chess_cnn_attention_legal_moves,
    get_model,
    MODELS,
)
from loss_functions import compute_chess_legal_moves_loss
from task_strategies import ChessLegalMovesStrategy

_BATCH_SIZE = 4


def _init_model(rng):
    model = ChessCnnAttentionLegalMoves(num_moves=NUM_MOVES, dropout_rate=0.1)
    x = jnp.zeros((_BATCH_SIZE, 8, 8, NUM_PLANES), dtype=jnp.float32)
    variables = model.init({"params": rng, "dropout": rng}, x, training=True)
    return model, variables, x


def test_output_shape_and_dtype():
    model, variables, x = _init_model(jax.random.PRNGKey(0))
    out, _ = model.apply(variables, x, training=True, mutable=["batch_stats"], rngs={"dropout": jax.random.PRNGKey(1)})

    # Tenseur unique (PAS un dict {"policy", "value"}) : pas de tete value ici.
    assert not isinstance(out, dict), f"attendu un tenseur unique, obtenu un dict : {type(out)}"
    assert out.shape == (_BATCH_SIZE, NUM_MOVES), f"attendu ({_BATCH_SIZE}, {NUM_MOVES}), obtenu {out.shape}"
    assert out.dtype == jnp.float32
    print(f"OK - shape/dtype corrects : logits {out.shape}")


def test_train_and_eval_apply_modes():
    model, variables, x = _init_model(jax.random.PRNGKey(0))

    out_train, new_state = model.apply(
        variables, x, training=True, mutable=["batch_stats"], rngs={"dropout": jax.random.PRNGKey(2)}
    )
    assert "batch_stats" in new_state, "batch_stats absent de la sortie mutable en mode entrainement"
    assert out_train.shape == (_BATCH_SIZE, NUM_MOVES)

    out_eval = model.apply(variables, x, training=False)
    assert out_eval.shape == (_BATCH_SIZE, NUM_MOVES)
    print("OK - apply_fn fonctionne en mode entrainement (batch_stats mutable) et en mode eval")


def test_get_model_factory_and_registry():
    assert "chess_cnn_attention_legal_moves" in MODELS, "modele absent du registre MODELS"

    model = get_model("chess_cnn_attention_legal_moves", num_classes=NUM_MOVES, dropout_rate=0.1)
    assert isinstance(model, ChessCnnAttentionLegalMoves)
    assert model.num_moves == NUM_MOVES, (
        f"num_classes non transmis correctement a num_moves : attendu {NUM_MOVES}, obtenu {model.num_moves}"
    )

    model_direct = create_chess_cnn_attention_legal_moves(num_classes=NUM_MOVES, dropout_rate=0.2)
    assert model_direct.num_moves == NUM_MOVES
    assert model_direct.dropout_rate == 0.2
    print("OK - get_model('chess_cnn_attention_legal_moves', num_classes=NUM_MOVES) fonctionne, "
          "num_classes correctement transmis a num_moves (pas silencieusement ignore)")


def test_end_to_end_differentiability():
    # Meme discipline que test_chess_model.py::test_end_to_end_differentiability :
    # entree ALEATOIRE requise (pas x=zeros), sinon backbone (Conv use_bias=False) et
    # bottleneck_queries produisent un gradient legitimement nul (degenerescence du
    # test, pas du modele).
    model = ChessCnnAttentionLegalMoves(num_moves=NUM_MOVES, dropout_rate=0.1)
    x = jax.random.normal(jax.random.PRNGKey(42), (_BATCH_SIZE, 8, 8, NUM_PLANES))
    init_rng = jax.random.PRNGKey(0)
    variables = model.init({"params": init_rng, "dropout": init_rng}, x, training=True)

    def loss_fn(params):
        vars_with_params = {**variables, "params": params}
        out, _ = model.apply(
            vars_with_params, x, training=True, mutable=["batch_stats"], rngs={"dropout": jax.random.PRNGKey(3)}
        )
        return jnp.sum(out)

    grads = jax.grad(loss_fn)(variables["params"])

    top_level_modules = list(variables["params"].keys())
    assert set(top_level_modules) == set(grads.keys()), "structure du gradient differente des parametres"

    dead_modules = []
    for module_name in top_level_modules:
        leaves = jax.tree_util.tree_leaves(grads[module_name])
        assert all(bool(jnp.all(jnp.isfinite(leaf))) for leaf in leaves), f"{module_name} : gradient contient NaN/Inf"
        if not any(bool(jnp.any(leaf != 0)) for leaf in leaves):
            dead_modules.append(module_name)

    assert not dead_modules, f"module(s) sans aucun gradient non-nul (branche morte suspectee) : {dead_modules}"
    print(f"OK - jax.grad de bout en bout reussit, chaque module ({len(top_level_modules)}: "
          f"{', '.join(top_level_modules)}) recoit au moins un gradient fini et non-nul")


def test_loss_is_finite():
    # compute_chess_legal_moves_loss (sigmoid BCE) sur logits/mask realistes -
    # verifie juste que la loss est un scalaire fini, pas une valeur precise.
    logits = jax.random.normal(jax.random.PRNGKey(7), (_BATCH_SIZE, NUM_MOVES))
    legal_mask = (jax.random.uniform(jax.random.PRNGKey(8), (_BATCH_SIZE, NUM_MOVES)) < 0.01).astype(jnp.float32)
    loss = compute_chess_legal_moves_loss(logits, legal_mask)
    assert loss.shape == (), f"attendu un scalaire, obtenu shape {loss.shape}"
    assert bool(jnp.isfinite(loss)), f"loss non finie : {loss}"
    print(f"OK - compute_chess_legal_moves_loss retourne un scalaire fini ({float(loss):.4f})")


def test_pos_weight_default_matches_unweighted_bce():
    # pos_weight=1.0 (defaut) doit rester exactement equivalent a la BCE non
    # ponderee d'origine (optax.sigmoid_binary_cross_entropy) - garde de
    # non-regression pour le comportement des configs qui ne fixent pas
    # loss_params.pos_weight.
    import optax
    logits = jax.random.normal(jax.random.PRNGKey(20), (_BATCH_SIZE, NUM_MOVES))
    legal_mask = (jax.random.uniform(jax.random.PRNGKey(21), (_BATCH_SIZE, NUM_MOVES)) < 0.01).astype(jnp.float32)

    loss_default = compute_chess_legal_moves_loss(logits, legal_mask)
    loss_reference = optax.sigmoid_binary_cross_entropy(logits, legal_mask).mean()

    assert bool(jnp.allclose(loss_default, loss_reference, atol=1e-5)), (
        f"pos_weight=1.0 devrait egaler la BCE non ponderee : {float(loss_default)} vs {float(loss_reference)}"
    )
    print(f"OK - pos_weight=1.0 (defaut) egale la BCE non ponderee ({float(loss_default):.4f})")


def test_pos_weight_penalizes_missed_positives_more():
    # Un pos_weight > 1 doit strictement augmenter la loss quand le modele rate
    # des vrais positifs (logits tres negatifs la ou legal_mask=1) - c'est
    # exactement le comportement recherche (pousser plus fort vers le rappel).
    logits = jnp.full((_BATCH_SIZE, NUM_MOVES), -10.0)  # predit "illegal" partout
    legal_mask = jnp.zeros((_BATCH_SIZE, NUM_MOVES), dtype=jnp.float32).at[:, :25].set(1.0)

    loss_unweighted = compute_chess_legal_moves_loss(logits, legal_mask, pos_weight=1.0)
    loss_weighted = compute_chess_legal_moves_loss(logits, legal_mask, pos_weight=2.0)

    assert bool(jnp.isfinite(loss_weighted)), f"loss ponderee non finie : {loss_weighted}"
    assert float(loss_weighted) > float(loss_unweighted), (
        f"pos_weight=2.0 devrait penaliser plus fort les faux negatifs : "
        f"{float(loss_weighted)} devrait etre > {float(loss_unweighted)}"
    )
    print(f"OK - pos_weight=2.0 penalise plus fort les faux negatifs "
          f"({float(loss_unweighted):.4f} -> {float(loss_weighted):.4f})")


def test_metrics_sane_on_realistic_sparse_mask():
    # Cas nominal (PAS le cas limite tout-a-zero ci-dessous) : ~25 coups legaux sur
    # 4672 (~0.5%, ordre de grandeur reel du dataset chess_legal_moves) - verifie que
    # F1 reste borne dans [0, 1] sur ce taux de sparsite realiste, pas seulement sur
    # le cas pathologique. Ne verifie pas une valeur precise (poids aleatoires).
    strategy = ChessLegalMovesStrategy()
    rng_logits, rng_mask = jax.random.PRNGKey(11), jax.random.PRNGKey(12)
    outputs = jax.random.normal(rng_logits, (_BATCH_SIZE, NUM_MOVES))
    targets = jnp.zeros((_BATCH_SIZE, NUM_MOVES), dtype=jnp.float32)
    for i in range(_BATCH_SIZE):
        idx = jax.random.choice(jax.random.fold_in(rng_mask, i), NUM_MOVES, shape=(25,), replace=False)
        targets = targets.at[i, idx].set(1.0)

    f1 = strategy.compute_metrics(outputs, targets)
    assert bool(jnp.isfinite(f1)), f"F1 non fini : {f1}"
    assert 0.0 <= float(f1) <= 1.0, f"F1 hors de [0,1] : {float(f1)}"
    print(f"OK - F1 reste dans [0,1] sur un mask realiste (~25/{NUM_MOVES} coups legaux), F1={float(f1):.4f}")


def test_metrics_no_nan_on_all_zero_mask():
    # Cas limite explicitement couvert par la spec (I/O Matrix) : legal_mask tout a
    # zero (aucun coup legal) ne doit jamais produire de NaN (division par zero sur
    # precision/rappel), meme si ce cas est improbable en pratique sur ce dataset.
    strategy = ChessLegalMovesStrategy()
    outputs = jnp.full((_BATCH_SIZE, NUM_MOVES), -10.0)  # sigmoid(-10) ~= 0 partout
    targets = jnp.zeros((_BATCH_SIZE, NUM_MOVES), dtype=jnp.float32)
    f1 = strategy.compute_metrics(outputs, targets)
    assert bool(jnp.isfinite(f1)), f"F1 non fini (NaN attendu evite) : {f1}"
    assert float(f1) == 0.0, f"attendu F1=0.0 quand aucune prediction et aucune cible positive, obtenu {float(f1)}"
    print(f"OK - compute_metrics ne produit pas de NaN sur legal_mask tout a zero (F1={float(f1)})")


if __name__ == "__main__":
    test_output_shape_and_dtype()
    test_train_and_eval_apply_modes()
    test_get_model_factory_and_registry()
    test_end_to_end_differentiability()
    test_loss_is_finite()
    test_pos_weight_default_matches_unweighted_bce()
    test_pos_weight_penalizes_missed_positives_more()
    test_metrics_sane_on_realistic_sparse_mask()
    test_metrics_no_nan_on_all_zero_mask()
    print("\nTous les tests du modele echecs (coups legaux) sont passes.")
