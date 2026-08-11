"""
Test de validation pour ChessMoveTokenTransformer (Epic 11, spike — AD-26/AD-28/
AD-29/AD-30/AD-31/AD-33, spine architecture-chess-move-token-2026-08-10).

Script autonome - ce projet n'a pas de framework de test formel (meme convention que
tests/test_chess_model.py / tests/test_chess_legal_moves_model.py). Executer
directement :
    python tests/test_chess_move_token_model.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
import jax.numpy as jnp
import numpy as np

NUM_MOVES = 4672

from model_library import (
    ChessMoveTokenTransformer,
    create_chess_move_token_transformer,
    get_model,
    MODELS,
)
from data_management import CHESS_MOVE_TOKEN_BOS, CHESS_MOVE_TOKEN_PAD
from task_strategies import ChessMoveTokenStrategy
from loss_functions import compute_chess_policy_loss

_BATCH_SIZE = 4
_SEQ_LEN = 6


def _init_model(rng):
    model = ChessMoveTokenTransformer(num_moves=NUM_MOVES, dropout_rate=0.1, num_layers=2, d_model=32, num_heads=4)
    x = jnp.zeros((_BATCH_SIZE, _SEQ_LEN), dtype=jnp.int32)
    variables = model.init({"params": rng, "dropout": rng}, x, training=True)
    return model, variables, x


def test_output_shape_and_dtype():
    model, variables, x = _init_model(jax.random.PRNGKey(0))
    out = model.apply(variables, x, training=False)

    assert not isinstance(out, dict), f"attendu un tenseur unique (pas de tete value, AD-26), obtenu {type(out)}"
    assert out.shape == (_BATCH_SIZE, NUM_MOVES), f"attendu ({_BATCH_SIZE}, {NUM_MOVES}), obtenu {out.shape}"
    assert out.dtype == jnp.float32
    print(f"OK - shape/dtype corrects : logits {out.shape}")


def test_init_with_float32_dummy_does_not_crash():
    # AD-33 (trouve par execution reelle pendant l'implementation) : Trainer.
    # create_train_state (trainer.py:145) construit son dummy d'init en FLOAT32
    # code en dur, independamment du dtype reellement utilise a l'entrainement -
    # nn.Embed.init() leve ValueError sur une entree flottante si le modele ne
    # caste pas explicitement en int32 lui-meme. Reproduit exactement le dummy
    # (1,)+input_shape avec input_shape=(1,) (CHESS_MOVE_TOKEN, dataset_configs.py).
    model = ChessMoveTokenTransformer(num_moves=NUM_MOVES, dropout_rate=0.1)
    dummy = jnp.ones((1, 1), jnp.float32)
    variables = model.init(jax.random.PRNGKey(0), dummy, training=True)
    assert variables is not None
    n_params = sum(x.size for x in jax.tree_util.tree_leaves(variables["params"]))
    print(f"OK - model.init() reussit avec un dummy float32 (1,1) comme Trainer.create_train_state "
          f"({n_params} params) - AD-33 non regresse")


def test_padding_invariance_same_content_different_padding_amount():
    # AD-28 : le dernier token REEL doit toujours etre lu (indice -1), quelle que
    # soit la quantite de padding a gauche. Meme contenu reel, 0 puis 3 PAD
    # supplementaires a gauche -> meme sortie (aux erreurs flottantes pres).
    #
    # ATTENTION (trouve pendant l'implementation) : ce test ne doit JAMAIS modifier
    # la VALEUR d'un slot deja marque PAD pour verifier l'invariance - le masque de
    # padding est derive dynamiquement de `tokens != PAD_TOKEN_ID`, donc changer la
    # valeur d'un slot PAD lui fait perdre son statut "pad" et invalide le test
    # (diff observee ~3.6 lors d'une 1ere tentative erronee, corrigee).
    model = ChessMoveTokenTransformer(num_moves=NUM_MOVES, dropout_rate=0.0, num_layers=2, d_model=32, num_heads=4)
    dummy = jnp.ones((1, 1), jnp.float32)
    variables = model.init(jax.random.PRNGKey(0), dummy, training=True)

    real = [CHESS_MOVE_TOKEN_BOS, 10, 20, 30]
    seq_no_pad = jnp.array(np.array([real], dtype=np.int32))
    seq_padded = jnp.array(np.array([[CHESS_MOVE_TOKEN_PAD] * 3 + real], dtype=np.int32))

    out_no_pad = model.apply(variables, seq_no_pad, training=False)
    out_padded = model.apply(variables, seq_padded, training=False)

    diff = float(jnp.abs(out_no_pad[0] - out_padded[0]).max())
    assert diff < 1e-4, f"le padding a gauche ne devrait pas changer la sortie (AD-28) : diff max {diff}"
    assert int(jnp.argmax(out_no_pad[0])) == int(jnp.argmax(out_padded[0]))
    print(f"OK - sortie invariante a la quantite de padding a gauche (diff max {diff:.2e})")


def test_earlier_real_tokens_affect_output():
    # Complement du test d'invariance ci-dessus : un token REEL (pas du padding)
    # plus tot dans l'historique doit, lui, changer la sortie - distingue "ignore
    # correctement le padding" de "ignore tout ce qui n'est pas le dernier token"
    # (bug qui passerait silencieusement le test d'invariance seul).
    model = ChessMoveTokenTransformer(num_moves=NUM_MOVES, dropout_rate=0.0, num_layers=2, d_model=32, num_heads=4)
    dummy = jnp.ones((1, 1), jnp.float32)
    variables = model.init(jax.random.PRNGKey(0), dummy, training=True)

    seq_a = jnp.array(np.array([[CHESS_MOVE_TOKEN_BOS, 10, 20, 30]], dtype=np.int32))
    seq_b = jnp.array(np.array([[CHESS_MOVE_TOKEN_BOS, 999, 20, 30]], dtype=np.int32))  # coup different tot dans l'historique

    out_a = model.apply(variables, seq_a, training=False)
    out_b = model.apply(variables, seq_b, training=False)

    diff = float(jnp.abs(out_a[0] - out_b[0]).max())
    assert diff > 1e-4, "un coup different plus tot dans l'historique devrait changer la sortie (masque trop agressif ?)"
    print(f"OK - un token reel plus tot dans l'historique change bien la sortie (diff max {diff:.4f})")


def test_dtype_survives_trainer_cast_path():
    # AD-29 (bug critique trouve en revue adversariale du spine) : Trainer caste
    # INCONDITIONNELLEMENT toute entree via jnp.array(images_np, dtype=self.dtype)
    # (trainer.py:313/430) avant tout hook Strategy. Verifie que passer par CE MEME
    # cast avec dtype=jnp.int32 (le fix, main.py) laisse les valeurs de tokens
    # bit-a-bit identiques - et, a titre de contraste documente, que le defaut
    # float16 des autres domaines les corromprait bel et bien.
    tokens = np.array([[CHESS_MOVE_TOKEN_PAD, CHESS_MOVE_TOKEN_BOS, 10, 4671, 4672, 4673]], dtype=np.int32)

    cast_int32 = jnp.array(tokens, dtype=jnp.int32)
    assert bool(jnp.all(cast_int32 == jnp.array(tokens))), "le cast int32 (AD-29) devrait etre un no-op parfait"

    cast_float16 = jnp.array(tokens, dtype=jnp.float16)
    corrupted = bool(jnp.any(cast_float16.astype(jnp.int32) != jnp.array(tokens)))
    assert corrupted, (
        "attendu : le cast float16 par defaut DEVRAIT corrompre au moins un token de cet echantillon "
        "(sinon le garde-fou AD-29 n'a plus de justification mesurable)"
    )
    print("OK - dtype=int32 (AD-29) preserve les tokens bit-a-bit ; float16 (defaut des autres domaines) "
          "les aurait corrompus, confirmant que le fix est necessaire")


def test_get_model_factory_and_registry():
    assert "chess_move_token_transformer" in MODELS, "modele absent du registre MODELS"

    model = get_model("chess_move_token_transformer", num_classes=NUM_MOVES, dropout_rate=0.1)
    assert isinstance(model, ChessMoveTokenTransformer)
    assert model.num_moves == NUM_MOVES, (
        f"num_classes non transmis correctement a num_moves : attendu {NUM_MOVES}, obtenu {model.num_moves}"
    )

    model_direct = create_chess_move_token_transformer(num_classes=NUM_MOVES, dropout_rate=0.2, num_layers=6)
    assert model_direct.num_moves == NUM_MOVES
    assert model_direct.dropout_rate == 0.2
    assert model_direct.num_layers == 6
    print("OK - get_model('chess_move_token_transformer', ...) fonctionne, num_classes transmis a num_moves, "
          "**kwargs (num_layers) transmis a la classe")


def test_end_to_end_differentiability():
    # Meme discipline que test_chess_model.py/test_chess_legal_moves_model.py :
    # entree ALEATOIRE dans l'espace de tokens valide (pas des zeros), sinon des
    # branches du modele produiraient un gradient legitimement nul (degenerescence
    # du test, pas du modele).
    model = ChessMoveTokenTransformer(num_moves=NUM_MOVES, dropout_rate=0.0, num_layers=2, d_model=32, num_heads=4)
    x = jax.random.randint(jax.random.PRNGKey(42), (_BATCH_SIZE, _SEQ_LEN), 0, NUM_MOVES)
    init_rng = jax.random.PRNGKey(0)
    variables = model.init({"params": init_rng, "dropout": init_rng}, x, training=True)

    def loss_fn(params):
        vars_with_params = {**variables, "params": params}
        out = model.apply(vars_with_params, x, training=True, rngs={"dropout": jax.random.PRNGKey(3)})
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


def test_strategy_loss_and_metrics():
    strategy = ChessMoveTokenStrategy()
    logits = jax.random.normal(jax.random.PRNGKey(5), (_BATCH_SIZE, NUM_MOVES))
    targets = jax.random.randint(jax.random.PRNGKey(6), (_BATCH_SIZE,), 0, NUM_MOVES)

    loss = strategy.compute_loss(logits, targets)
    assert loss.shape == (), f"attendu un scalaire, obtenu shape {loss.shape}"
    assert bool(jnp.isfinite(loss)), f"loss non finie : {loss}"

    ref_loss = compute_chess_policy_loss(logits, targets)
    assert bool(jnp.allclose(loss, ref_loss)), "compute_loss devrait deleguer integralement a compute_chess_policy_loss (AD-26)"

    metric = strategy.compute_metrics(logits, targets)
    assert 0.0 <= float(metric) <= 1.0, f"PolicyAccuracy hors de [0,1] : {float(metric)}"
    assert strategy.primary_metric_name == "PolicyAccuracy"
    assert strategy.optimization_mode == "max"
    print(f"OK - ChessMoveTokenStrategy.compute_loss delegue a compute_chess_policy_loss "
          f"(loss={float(loss):.4f}), compute_metrics dans [0,1] ({float(metric):.4f})")


def test_left_padding_dataset_loader_if_spike_available():
    # Test d'integration avec le VRAI fichier spike (pas un factice) - AD-27/AD-28.
    # Passe silencieusement (skip) si le fichier n'est pas present sur cette machine
    # (dataset spike, pas versionne) - la validation officielle par execution reelle
    # est couverte separement (Story 11.4).
    npz_path = "/home/aobled/Documents/data/chunks/chess_move_token_spike/chess_move_token_spike.npz"
    if not os.path.exists(npz_path):
        print("SKIP - dataset spike absent sur cette machine, test d'integration ignore")
        return

    from data_management import ChessMoveTokenDataset

    ds_mgr = ChessMoveTokenDataset(npz_path=npz_path, batch_size=8, val_split=0.1)
    train_ds, val_ds = ds_mgr.get_dataset()
    seqs, labels = next(iter(train_ds.as_numpy_iterator()))

    assert seqs.dtype == np.int32
    assert labels.dtype == np.int32
    assert (labels >= 0).all() and (labels < NUM_MOVES).all(), "labels hors de l'espace policy (0, NUM_MOVES)"
    assert (seqs == CHESS_MOVE_TOKEN_BOS).sum(axis=1).tolist() == [1] * seqs.shape[0], (
        "chaque sequence doit contenir exactement 1 BOS"
    )
    assert (seqs[:, -1] != CHESS_MOVE_TOKEN_PAD).all(), (
        "le dernier token de chaque sequence ne doit jamais etre PAD (recette de padding a gauche, AD-28)"
    )
    print(f"OK - ChessMoveTokenDataset (vrai fichier spike) : batch {seqs.shape}, "
          f"1 BOS/sequence, dernier token jamais PAD")


if __name__ == "__main__":
    test_output_shape_and_dtype()
    test_init_with_float32_dummy_does_not_crash()
    test_padding_invariance_same_content_different_padding_amount()
    test_earlier_real_tokens_affect_output()
    test_dtype_survives_trainer_cast_path()
    test_get_model_factory_and_registry()
    test_end_to_end_differentiability()
    test_strategy_loss_and_metrics()
    test_left_padding_dataset_loader_if_spike_available()
    print("\nTous les tests du modele echecs (move-token) sont passes.")
