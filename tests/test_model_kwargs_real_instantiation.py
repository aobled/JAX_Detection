"""
Test de validation pour Story 12.1 (AD-21), AC #7 : verifie par instanciation
REELLE (model.init(), pas mockee) que les 5 factories qui ne recoivent leurs
hyperparametres **kwargs`-only (num_bottleneck_tokens/token_dim/num_layers/
d_model/num_trunk_layers) que via un `**kwargs` catch-all (jamais nomme
explicitement dans leur propre signature) recoivent bien la vraie valeur
configuree une fois passees par build_kwargs_from_config - pas juste
l'absence d'exception, mais un effet reel et mesurable sur la forme des
parametres (garde-fou critique trouve en Reviewer Gate du spine, voir
review-web-verify.md).

Methode : pour chaque factory, construit les kwargs depuis la vraie entree
dataset_configs.py, initialise le modele, puis reconstruit avec UNE SEULE
valeur d'hyperparametre deliberement differente (comparaison relative, comme
tests/test_compute_dtype_hardware.py pour compute_dtype) - le nombre total de
parametres DOIT differer, preuve que la valeur a reellement atteint le
nn.Module sous-jacent via le `**kwargs` de la factory.

Script autonome - meme convention que tests/test_chess_model.py et les autres
tests echecs (pas de framework de test formel impose). Executer directement :
    python tests/test_model_kwargs_real_instantiation.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
import jax.numpy as jnp

from model_library import MODELS, build_kwargs_from_config, MODEL_FORWARDED_CONFIG_KEYS
from dataset_configs import get_dataset_config
from utils import count_parameters as _param_count


def _init_from_real_config(dataset_name, x, extra_override_key=None, extra_override_value=None):
    config = get_dataset_config(dataset_name)
    backend_config = config.get("gpu", config.get("tpu"))
    model_name = config["model_name"]
    target_factory = MODELS.get(model_name)
    if extra_override_key is not None:
        config = dict(config)
        config[extra_override_key] = extra_override_value
    model_kwargs, _ = build_kwargs_from_config(
        target_factory,
        config,
        config_keys=MODEL_FORWARDED_CONFIG_KEYS.get(model_name, ()),
        compute_dtype=jnp.float32,
        dropout_rate=backend_config["dropout_rate"],
        num_classes=config["num_classes"],
    )
    model = target_factory(**model_kwargs)
    rng = jax.random.PRNGKey(0)
    variables = model.init({"params": rng, "dropout": rng}, x, training=True)
    return model_kwargs, variables


def test_chess_cnn_attention_policy_value_num_bottleneck_tokens_real_effect():
    x = jax.random.normal(jax.random.PRNGKey(1), (2, 8, 8, 19))  # CHESS_NO_HISTORY: num_channels=19
    kwargs_real, variables_real = _init_from_real_config("CHESS_NO_HISTORY", x)
    assert kwargs_real["num_bottleneck_tokens"] == 8, kwargs_real
    _, variables_alt = _init_from_real_config(
        "CHESS_NO_HISTORY", x, extra_override_key="num_bottleneck_tokens", extra_override_value=4
    )
    assert _param_count(variables_real) != _param_count(variables_alt), (
        "num_bottleneck_tokens=8 vs 4 doit produire un nombre de parametres different"
    )
    print("OK - chess_cnn_attention_policy_value : num_bottleneck_tokens reellement transmis (**kwargs)")


def test_chess_cnn_attention_legal_moves_num_bottleneck_tokens_real_effect():
    x = jax.random.normal(jax.random.PRNGKey(2), (2, 8, 8, 29))  # CHESS_LEGAL_MOVES: num_channels=29
    kwargs_real, variables_real = _init_from_real_config("CHESS_LEGAL_MOVES", x)
    assert kwargs_real["num_bottleneck_tokens"] == 8, kwargs_real
    _, variables_alt = _init_from_real_config(
        "CHESS_LEGAL_MOVES", x, extra_override_key="num_bottleneck_tokens", extra_override_value=4
    )
    assert _param_count(variables_real) != _param_count(variables_alt)
    print("OK - chess_cnn_attention_legal_moves : num_bottleneck_tokens reellement transmis (**kwargs)")


def test_chess_move_token_transformer_num_layers_real_effect():
    x = jax.random.randint(jax.random.PRNGKey(3), (2, 6), 0, 4672)  # (B, SEQ_LEN), vocab=NUM_MOVES
    kwargs_real, variables_real = _init_from_real_config("CHESS_MOVE_TOKEN", x)
    assert kwargs_real["num_layers"] == 6, kwargs_real
    _, variables_alt = _init_from_real_config(
        "CHESS_MOVE_TOKEN", x, extra_override_key="num_layers", extra_override_value=2
    )
    assert _param_count(variables_real) != _param_count(variables_alt), (
        "num_layers=6 vs 2 doit produire un nombre de parametres different"
    )
    print("OK - chess_move_token_transformer : num_layers reellement transmis (**kwargs)")


def test_chess_token_candidate_model_token_dim_real_effect():
    num_candidates = 50
    packed_width = 64 + 6 + num_candidates
    k1, k2, k3 = jax.random.split(jax.random.PRNGKey(4), 3)
    token_position = jax.random.randint(k1, (2, 64), 0, 13)
    global_flags = jax.random.randint(k2, (2, 6), 0, 2)
    candidate_moves = jax.random.randint(k3, (2, num_candidates), 0, 4672)
    x = jnp.concatenate([token_position, global_flags, candidate_moves], axis=1).astype(jnp.int32)
    assert x.shape[1] == packed_width

    kwargs_real, variables_real = _init_from_real_config("CHESS_TOKEN", x)
    assert kwargs_real["token_dim"] == 128, kwargs_real
    _, variables_alt = _init_from_real_config(
        "CHESS_TOKEN", x, extra_override_key="token_dim", extra_override_value=32
    )
    assert _param_count(variables_real) != _param_count(variables_alt), (
        "token_dim=128 vs 32 doit produire un nombre de parametres different"
    )
    print("OK - chess_token_candidate_model : token_dim reellement transmis (**kwargs)")


def test_chess_token_one_move_model_num_trunk_layers_real_effect():
    k1, k2, k3 = jax.random.split(jax.random.PRNGKey(5), 3)
    token_position = jax.random.randint(k1, (2, 64), 0, 13)
    global_flags = jax.random.randint(k2, (2, 6), 0, 2)
    from_square_teacher = jax.random.randint(k3, (2, 1), 0, 64)
    x = jnp.concatenate([token_position, global_flags, from_square_teacher], axis=1).astype(jnp.int32)

    kwargs_real, variables_real = _init_from_real_config("CHESS_TOKEN_1_MOVE", x)
    assert kwargs_real["num_trunk_layers"] == 2, kwargs_real
    _, variables_alt = _init_from_real_config(
        "CHESS_TOKEN_1_MOVE", x, extra_override_key="num_trunk_layers", extra_override_value=1
    )
    assert _param_count(variables_real) != _param_count(variables_alt), (
        "num_trunk_layers=2 vs 1 doit produire un nombre de parametres different"
    )
    print("OK - chess_token_one_move_model : num_trunk_layers reellement transmis (**kwargs)")


def test_kepler_1d_cnn_num_classes_and_dropout_rate_real_effect():
    """
    Trouve en revue de code (pas par une AC de la story) : create_kepler_1d_cnn
    etait la SEULE factory a ne nommer explicitement ni num_classes ni
    dropout_rate (seulement compute_dtype + **kwargs) - le canal overrides
    strict de build_kwargs_from_config ne les aurait donc jamais transmis,
    masque uniquement parce que les defauts de Kepler1DConvNet (num_classes=2,
    dropout_rate=0.3) coincident avec la config JAX_KEPLER actuelle. Corrige en
    nommant les deux parametres explicitement dans la factory (meme discipline
    qu'AD-3 pour compute_dtype). Ce test verifie l'effet reel avec des valeurs
    DELIBEREMENT differentes des defauts, pour ne pas laisser une coincidence
    masquer une regression future.
    """
    from model_library import MODELS, build_kwargs_from_config

    config = get_dataset_config("JAX_KEPLER")
    target_factory = MODELS.get(config["model_name"])
    model_kwargs, forwarded = build_kwargs_from_config(
        target_factory, config, config_keys=(),
        compute_dtype=jnp.float32, dropout_rate=0.5, num_classes=7,
    )
    assert forwarded == frozenset({"compute_dtype", "dropout_rate", "num_classes"}), forwarded
    model = target_factory(**model_kwargs)
    assert model.num_classes == 7, model.num_classes
    assert model.dropout_rate == 0.5, model.dropout_rate
    print("OK - kepler_1d_cnn : num_classes/dropout_rate reellement transmis (canal overrides)")


if __name__ == "__main__":
    test_chess_cnn_attention_policy_value_num_bottleneck_tokens_real_effect()
    test_chess_cnn_attention_legal_moves_num_bottleneck_tokens_real_effect()
    test_chess_move_token_transformer_num_layers_real_effect()
    test_chess_token_candidate_model_token_dim_real_effect()
    test_chess_token_one_move_model_num_trunk_layers_real_effect()
    test_kepler_1d_cnn_num_classes_and_dropout_rate_real_effect()
