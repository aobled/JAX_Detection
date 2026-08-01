"""
Test de validation pour ChessCnnAttentionPolicyValue (Story 9.2, FR5/AD-22/AD-23/AD-24).

Script autonome - ce projet n'a pas de framework de test formel (voir Dev Notes de la
story 9.2). Executer directement :
    python tests/test_chess_model.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
import jax.numpy as jnp

# Constantes du contrat .npz echecs (cote chess_ai, generation du dataset retiree de
# ce repo) - litteraux, plus d'import depuis un module local de codec.
NUM_MOVES = 4672
NUM_PLANES = 29
POLICY_KEY = "policy"
VALUE_KEY = "value"

from model_library import (
    ChessCnnAttentionPolicyValue,
    create_chess_cnn_attention_policy_value,
    get_model,
    MODELS,
)

_BATCH_SIZE = 4


def _init_model(rng):
    model = ChessCnnAttentionPolicyValue(num_moves=NUM_MOVES, dropout_rate=0.1)
    x = jnp.zeros((_BATCH_SIZE, 8, 8, NUM_PLANES), dtype=jnp.float32)
    variables = model.init({"params": rng, "dropout": rng}, x, training=True)
    return model, variables, x


def test_output_shapes_and_dtypes():
    model, variables, x = _init_model(jax.random.PRNGKey(0))
    out, _ = model.apply(variables, x, training=True, mutable=["batch_stats"], rngs={"dropout": jax.random.PRNGKey(1)})

    assert set(out.keys()) == {POLICY_KEY, VALUE_KEY}, f"cles inattendues : {out.keys()}"
    assert out[POLICY_KEY].shape == (_BATCH_SIZE, NUM_MOVES), (
        f"POLICY_KEY : attendu ({_BATCH_SIZE}, {NUM_MOVES}), obtenu {out[POLICY_KEY].shape}"
    )
    assert out[POLICY_KEY].dtype == jnp.float32
    assert out[VALUE_KEY].shape == (_BATCH_SIZE,), f"VALUE_KEY : attendu ({_BATCH_SIZE},), obtenu {out[VALUE_KEY].shape}"
    assert out[VALUE_KEY].dtype == jnp.float32
    print(f"OK - shapes/dtypes corrects : policy {out[POLICY_KEY].shape}, value {out[VALUE_KEY].shape}")


def test_value_bounded_in_tanh_range():
    # Verifie les bornes reelles sur plusieurs forward pass (poids initiaux aleatoires
    # varies), pas juste supposer que tanh() les garantit sans jamais l'observer. Chaque
    # init utilise les statistiques BatchNorm non-entrainees par defaut (mean=0, var=1) -
    # ce test couvre les bornes tanh() sur des activations pre-entrainement, pas un
    # comportement post-entrainement realiste (hors scope de cette story, pas encore de
    # boucle d'entrainement echecs - Story 9.3).
    model = ChessCnnAttentionPolicyValue(num_moves=NUM_MOVES, dropout_rate=0.1)
    all_values = []
    for seed in range(10):
        rng = jax.random.PRNGKey(seed)
        x = jax.random.normal(jax.random.PRNGKey(seed + 100), (_BATCH_SIZE, 8, 8, NUM_PLANES))
        variables = model.init({"params": rng, "dropout": rng}, x, training=False)
        out = model.apply(variables, x, training=False)
        all_values.append(out[VALUE_KEY])

    values = jnp.concatenate(all_values)
    assert bool(jnp.all(values >= -1.0)) and bool(jnp.all(values <= 1.0)), (
        f"value hors de [-1, 1] : min={float(values.min())}, max={float(values.max())}"
    )
    print(f"OK - value bornee dans [-1, 1] sur {len(values)} echantillons (10 inits non-entrainees x batch {_BATCH_SIZE}), "
          f"min={float(values.min()):.4f}, max={float(values.max()):.4f}")


def test_train_and_eval_apply_modes():
    # Les deux chemins reellement utilises par trainer.py (voir Dev Notes § Contrat Trainer).
    model, variables, x = _init_model(jax.random.PRNGKey(0))

    # Mode entrainement : batch_stats mutable (BatchNorm), rng dropout requis.
    out_train, new_state = model.apply(
        variables, x, training=True, mutable=["batch_stats"], rngs={"dropout": jax.random.PRNGKey(2)}
    )
    assert "batch_stats" in new_state, "batch_stats absent de la sortie mutable en mode entrainement"
    assert out_train[POLICY_KEY].shape == (_BATCH_SIZE, NUM_MOVES)

    # Mode eval : pas de mutation, pas de rng dropout necessaire (deterministic=True).
    out_eval = model.apply(variables, x, training=False)
    assert out_eval[POLICY_KEY].shape == (_BATCH_SIZE, NUM_MOVES)
    assert out_eval[VALUE_KEY].shape == (_BATCH_SIZE,)
    print("OK - apply_fn fonctionne en mode entrainement (batch_stats mutable) et en mode eval")


def test_get_model_factory_and_registry():
    assert "chess_cnn_attention_policy_value" in MODELS, "modele absent du registre MODELS"

    model = get_model("chess_cnn_attention_policy_value", num_classes=NUM_MOVES, dropout_rate=0.1)
    assert isinstance(model, ChessCnnAttentionPolicyValue)
    assert model.num_moves == NUM_MOVES, (
        f"num_classes non transmis correctement a num_moves : attendu {NUM_MOVES}, obtenu {model.num_moves}"
    )

    # Meme verification via l'appel direct a la factory (sans passer par get_model).
    model_direct = create_chess_cnn_attention_policy_value(num_classes=NUM_MOVES, dropout_rate=0.2)
    assert model_direct.num_moves == NUM_MOVES
    assert model_direct.dropout_rate == 0.2
    print("OK - get_model('chess_cnn_attention_policy_value', num_classes=NUM_MOVES) fonctionne, "
          "num_classes correctement transmis a num_moves (pas silencieusement ignore)")


def test_end_to_end_differentiability():
    # Confirme que le gradient se propage a travers CHAQUE composant du modele
    # (backbone, bottleneck_queries, cross-attention, self-attention, tetes policy
    # ET value individuellement) - pas seulement qu'AU MOINS UN parametre quelque
    # part recoit un gradient non nul (un OU logique global laisserait passer une
    # branche morte, ex. l'auto-attention court-circuitee). Entree ALEATOIRE requise
    # (pas x=zeros de _init_model) : avec un input exactement nul, tous les Conv du
    # backbone (use_bias=False) et le bottleneck_queries produisent un gradient
    # legitimement nul (cle/valeur d'attention toutes a zero -> softmax uniforme,
    # independant des requetes) - degenerescence du test, pas du modele. Trouve en
    # ecrivant ce test plus strict (voir Debug Log de cette story).
    model = ChessCnnAttentionPolicyValue(num_moves=NUM_MOVES, dropout_rate=0.1)
    x = jax.random.normal(jax.random.PRNGKey(42), (_BATCH_SIZE, 8, 8, NUM_PLANES))
    init_rng = jax.random.PRNGKey(0)
    variables = model.init({"params": init_rng, "dropout": init_rng}, x, training=True)

    def loss_fn(params):
        vars_with_params = {**variables, "params": params}
        out, _ = model.apply(
            vars_with_params, x, training=True, mutable=["batch_stats"], rngs={"dropout": jax.random.PRNGKey(3)}
        )
        return jnp.sum(out[POLICY_KEY]) + jnp.sum(out[VALUE_KEY])

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


if __name__ == "__main__":
    test_output_shapes_and_dtypes()
    test_value_bounded_in_tanh_range()
    test_train_and_eval_apply_modes()
    test_get_model_factory_and_registry()
    test_end_to_end_differentiability()
    print("\nTous les tests du modele echecs sont passes.")
