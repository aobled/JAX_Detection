"""
Test de validation pour ChessTokenCandidateModel (spec-chess-token-candidate-model,
2026-08-13, spike, CAP-1/CAP-2).

Script autonome - ce projet n'a pas de framework de test formel (meme convention que
tests/test_chess_model.py / tests/test_chess_legal_moves_model.py /
tests/test_chess_move_token_model.py). Executer directement :
    python tests/test_chess_token_model.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
import jax.numpy as jnp
import numpy as np

NUM_CANDIDATES = 50
PACKED_WIDTH = 64 + 6 + NUM_CANDIDATES  # token_position(64) + global_flags(6) + candidate_moves(50)

from model_library import (
    ChessTokenCandidateModel,
    create_chess_token_candidate_model,
    get_model,
    MODELS,
    NUM_MOVE_TYPES,  # chess_ai/chess_target_encoding.py:243 - index = from_square*73 + move_type
)
from task_strategies import ChessTokenStrategy
from loss_functions import compute_chess_token_candidate_loss

_BATCH_SIZE = 4


def _random_packed_batch(rng_key, batch_size=_BATCH_SIZE, num_candidates=NUM_CANDIDATES):
    """
    Construit un batch packe (B, 64+6+num_candidates) VALIDE - token_position dans
    [0,12], global_flags binaire, candidate_moves dans [-1, 4671] (sentinel -1 =
    padding). Entree ALEATOIRE (pas des zeros) - meme discipline que
    tests/test_chess_move_token_model.py::test_end_to_end_differentiability, sinon des
    branches du modele produiraient un gradient legitimement nul (degenerescence du
    test, pas du modele).
    """
    k1, k2, k3 = jax.random.split(rng_key, 3)
    token_position = jax.random.randint(k1, (batch_size, 64), 0, 13)
    global_flags = jax.random.randint(k2, (batch_size, 6), 0, 2)
    candidate_moves = jax.random.randint(k3, (batch_size, num_candidates), 0, 4672)
    packed = jnp.concatenate([token_position, global_flags, candidate_moves], axis=1)
    return packed.astype(jnp.int32)


def _init_model(rng, num_trunk_layers=2):
    model = ChessTokenCandidateModel(
        num_candidates=NUM_CANDIDATES, dropout_rate=0.1, token_dim=32, num_bottleneck_tokens=4,
        num_heads=4, num_trunk_layers=num_trunk_layers,
    )
    x = _random_packed_batch(rng)
    variables = model.init({"params": rng, "dropout": rng}, x, training=True)
    return model, variables, x


def test_output_shape_and_dtype():
    model, variables, x = _init_model(jax.random.PRNGKey(0))
    out = model.apply(variables, x, training=False)

    assert not isinstance(out, dict), f"attendu un tenseur unique (pas de tete value), obtenu {type(out)}"
    assert out.shape == (_BATCH_SIZE, NUM_CANDIDATES), f"attendu ({_BATCH_SIZE}, {NUM_CANDIDATES}), obtenu {out.shape}"
    assert out.dtype == jnp.float32
    assert bool(jnp.all(jnp.isfinite(out))), "logits non finis"
    print(f"OK - shape/dtype corrects : logits {out.shape}, dtype={out.dtype}")


def test_init_with_float32_dummy_does_not_crash():
    # AD-33 (heritee de ChessMoveTokenTransformer) : Trainer.create_train_state
    # (trainer.py:145) construit son dummy d'init en FLOAT32 code en dur,
    # independamment du dtype reellement utilise a l'entrainement - nn.Embed.init()
    # leve ValueError sur une entree flottante si le modele ne caste pas explicitement
    # en int32 lui-meme. Reproduit exactement le dummy (1,)+input_shape avec
    # input_shape=(120,) (CHESS_TOKEN, dataset_configs.py).
    model = ChessTokenCandidateModel(num_candidates=NUM_CANDIDATES, dropout_rate=0.1)
    dummy = jnp.ones((1, PACKED_WIDTH), jnp.float32)
    variables = model.init(jax.random.PRNGKey(0), dummy, training=True)
    assert variables is not None
    n_params = sum(x.size for x in jax.tree_util.tree_leaves(variables["params"]))
    print(f"OK - model.init() reussit avec un dummy float32 (1,{PACKED_WIDTH}) comme "
          f"Trainer.create_train_state ({n_params} params) - AD-33 non regresse")


def test_dtype_survives_trainer_cast_path():
    # AD-29 (herite) : Trainer caste INCONDITIONNELLEMENT toute entree via
    # jnp.array(images_np, dtype=self.dtype) (trainer.py:313/430) avant tout hook
    # Strategy. candidate_moves va jusqu'a 4671 - un cast float16 par defaut
    # (autres domaines) corromprait silencieusement ces valeurs (float16 ne
    # represente exactement les entiers que jusqu'a ~2048), d'ou dtype=jnp.int32
    # pour task_type="chess_token" (main.py).
    packed = np.array([[0] * 64 + [1, 0, 1, 0, 1, 0] + [4671, 4660, -1] + [0] * 47], dtype=np.int32)

    cast_int32 = jnp.array(packed, dtype=jnp.int32)
    assert bool(jnp.all(cast_int32 == jnp.array(packed))), "le cast int32 devrait etre un no-op parfait"

    cast_float16 = jnp.array(packed, dtype=jnp.float16)
    corrupted = bool(jnp.any(cast_float16.astype(jnp.int32) != jnp.array(packed)))
    assert corrupted, (
        "attendu : le cast float16 par defaut DEVRAIT corrompre au moins une valeur de cet "
        "echantillon (sinon le garde-fou dtype=int32 n'a plus de justification mesurable)"
    )
    print("OK - dtype=int32 preserve les valeurs packees bit-a-bit ; float16 (defaut des "
          "autres domaines) les aurait corrompues, confirmant que le fix est necessaire")


def test_from_square_move_type_decomposition_at_boundaries():
    # Design Notes du SPEC : from_square = index // 73 doit rester dans [0,64), move_type
    # = index % 73 doit rester dans [0,73), aux BORNES exactes de l'espace de coups
    # (index=0 et index=4671=NUM_MOVES-1). Verifie a la fois l'arithmetique pure (le
    # contrat que le modele implemente) ET que le forward pass ne crashe pas quand ces
    # index extremes apparaissent reellement dans candidate_moves.
    for index, expected_from_square, expected_move_type in [(0, 0, 0), (4671, 63, 72)]:
        from_square = index // NUM_MOVE_TYPES
        move_type = index % NUM_MOVE_TYPES
        assert from_square == expected_from_square, f"index={index} : from_square={from_square}, attendu {expected_from_square}"
        assert move_type == expected_move_type, f"index={index} : move_type={move_type}, attendu {expected_move_type}"
        assert 0 <= from_square < 64, f"from_square {from_square} hors de [0,64) pour index={index}"
        assert 0 <= move_type < NUM_MOVE_TYPES, f"move_type {move_type} hors de [0,73) pour index={index}"

    # Forward pass reel avec ces index extremes dans candidate_moves (slots 0 et 1),
    # padding (-1) sur les slots restants.
    model = ChessTokenCandidateModel(num_candidates=NUM_CANDIDATES, dropout_rate=0.0, token_dim=32,
                                      num_bottleneck_tokens=4, num_heads=4, num_trunk_layers=1)
    token_position = np.zeros((1, 64), dtype=np.int32)
    global_flags = np.zeros((1, 6), dtype=np.int32)
    candidate_moves = np.full((1, NUM_CANDIDATES), -1, dtype=np.int32)
    candidate_moves[0, 0] = 0
    candidate_moves[0, 1] = 4671
    packed = np.concatenate([token_position, global_flags, candidate_moves], axis=1)
    x = jnp.array(packed)
    variables = model.init(jax.random.PRNGKey(0), x, training=True)
    out = model.apply(variables, x, training=False)
    assert out.shape == (1, NUM_CANDIDATES)
    assert bool(jnp.all(jnp.isfinite(out))), "logits non finis avec index de coup extremes (0 et 4671)"
    print("OK - decomposition from_square/move_type correcte aux bornes (index=0 -> (0,0), "
          "index=4671 -> (63,72)), forward pass stable avec ces index reels")


def test_masking_correctness_loss_and_argmax():
    # Batch SYNTHETIQUE construit a la main : quelques slots reels, le reste en
    # padding avec des logits ENORMES (jamais atteignables par le modele en pratique,
    # mais ca isole la propriete testee) - un masquage correct doit totalement ignorer
    # ces slots, autant pour la loss que pour l'argmax (compute_metrics).
    logits = jnp.array([
        [1.0, 2.0, 3.0] + [1000.0] * (NUM_CANDIDATES - 3),  # padding artificiellement "gagnant"
        [5.0, -1.0, 0.5] + [-999.0] * (NUM_CANDIDATES - 3),  # padding artificiellement "perdant"
    ])
    mask = jnp.array([
        [1, 1, 1] + [0] * (NUM_CANDIDATES - 3),
        [1, 1, 1] + [0] * (NUM_CANDIDATES - 3),
    ])
    label = jnp.array([2, 0])  # coup professeur = slot 2 (ligne 0), slot 0 (ligne 1)

    loss = compute_chess_token_candidate_loss(logits, label, mask)
    assert bool(jnp.isfinite(loss)), f"loss non finie : {loss}"

    # Valeur attendue a la main : cross-entropy restreinte aux 3 premiers logits de
    # chaque ligne (les slots masques, malgre des valeurs enormes, ne doivent PAS
    # entrer dans le softmax).
    manual_logits = logits[:, :3]
    manual_loss = -(
        manual_logits[jnp.arange(2), label] - jax.nn.logsumexp(manual_logits, axis=-1)
    ).mean()
    assert bool(jnp.allclose(loss, manual_loss, atol=1e-4)), (
        f"loss={float(loss):.6f} ne correspond pas au calcul a la main sur les 3 slots reels "
        f"({float(manual_loss):.6f}) - un slot de padding influence la loss"
    )

    strategy = ChessTokenStrategy()
    targets = {"candidate_label": label, "candidate_mask": mask}
    predicted_masked = strategy.compute_metrics(logits, targets)
    # ligne 0 : argmax masque doit choisir le slot 2 (label correct, malgre le padding
    # "gagnant" a 1000.0) ; ligne 1 : slot 0 (label correct, malgre -999.0 sur le padding).
    assert float(predicted_masked) == 1.0, (
        f"PolicyAccuracy attendue 1.0 (les 2 exemples ont leur veritable meilleur slot reel "
        f"choisi comme label) avec masquage correct, obtenu {float(predicted_masked)} - un "
        f"slot de padding a influence l'argmax"
    )

    # Contre-preuve : SANS masquage, l'argmax choisirait bien les slots de padding
    # (prouve que le test n'est pas degenere - le masquage a un effet mesurable).
    unmasked_argmax = jnp.argmax(logits, axis=-1)
    assert int(unmasked_argmax[0]) >= 3, "l'exemple synthetique doit avoir un padding gagnant sans masquage (sanity du test)"
    print(f"OK - masquage correct : loss={float(loss):.4f} (== calcul a la main sur slots reels "
          f"uniquement), argmax masque ignore totalement les slots de padding (accuracy=1.0 vs "
          f"argmax brut qui aurait choisi le slot {int(unmasked_argmax[0])})")


def test_get_model_factory_and_registry():
    assert "chess_token_candidate_model" in MODELS, "modele absent du registre MODELS"

    model = get_model("chess_token_candidate_model", num_classes=NUM_CANDIDATES, dropout_rate=0.1)
    assert isinstance(model, ChessTokenCandidateModel)
    assert model.num_candidates == NUM_CANDIDATES, (
        f"num_classes non transmis correctement a num_candidates : attendu {NUM_CANDIDATES}, "
        f"obtenu {model.num_candidates}"
    )

    model_direct = create_chess_token_candidate_model(num_classes=NUM_CANDIDATES, dropout_rate=0.2, token_dim=32)
    assert model_direct.num_candidates == NUM_CANDIDATES
    assert model_direct.dropout_rate == 0.2
    assert model_direct.token_dim == 32
    print("OK - get_model('chess_token_candidate_model', ...) fonctionne, num_classes transmis "
          "a num_candidates, **kwargs (token_dim) transmis a la classe")


def test_end_to_end_differentiability():
    model, variables, x = _init_model(jax.random.PRNGKey(42))
    label = jax.random.randint(jax.random.PRNGKey(7), (_BATCH_SIZE,), 0, NUM_CANDIDATES)
    mask = jnp.ones((_BATCH_SIZE, NUM_CANDIDATES), dtype=jnp.int32)

    def loss_fn(params):
        vars_with_params = {**variables, "params": params}
        out = model.apply(vars_with_params, x, training=True, rngs={"dropout": jax.random.PRNGKey(3)})
        return compute_chess_token_candidate_loss(out, label, mask)

    loss, grads = jax.value_and_grad(loss_fn)(variables["params"])
    assert bool(jnp.isfinite(loss)), f"loss non finie : {loss}"

    top_level_modules = list(variables["params"].keys())
    assert set(top_level_modules) == set(grads.keys()), "structure du gradient differente des parametres"

    dead_modules = []
    for module_name in top_level_modules:
        leaves = jax.tree_util.tree_leaves(grads[module_name])
        assert all(bool(jnp.all(jnp.isfinite(leaf))) for leaf in leaves), f"{module_name} : gradient contient NaN/Inf"
        if not any(bool(jnp.any(leaf != 0)) for leaf in leaves):
            dead_modules.append(module_name)

    assert not dead_modules, f"module(s) sans aucun gradient non-nul (branche morte suspectee) : {dead_modules}"
    print(f"OK - jax.grad de bout en bout reussit (loss={float(loss):.4f}), chaque module "
          f"({len(top_level_modules)}: {', '.join(top_level_modules)}) recoit au moins un "
          f"gradient fini et non-nul")


def test_end_to_end_differentiability_with_masking():
    # Complement de test_end_to_end_differentiability ci-dessus (code review, 2026-08-13) :
    # celui-ci utilise mask=jnp.ones(...) partout, donc n'exerce JAMAIS la branche
    # `safe_moves` de neutralisation du padding (model_library.py) sous gradient - un batch
    # entierement reel est un cas degenere du contrat reel (voir dataset : la plupart des
    # positions ont k<50 candidats reels). Ce test reprend le meme style de batch
    # PARTIELLEMENT masque que test_masking_correctness_loss_and_argmax (slots de padding
    # a candidate_moves=-1 ET candidate_mask=0, meme convention que le dataset reel) et
    # verifie que jax.grad reste fini et non-degenere a travers cette branche.
    n_real = 5  # < NUM_CANDIDATES=50 - la majorite des slots sont du padding, comme en pratique
    k1, k2, k3, k4 = jax.random.split(jax.random.PRNGKey(123), 4)

    token_position = jax.random.randint(k1, (_BATCH_SIZE, 64), 0, 13)
    global_flags = jax.random.randint(k2, (_BATCH_SIZE, 6), 0, 2)
    real_moves = jax.random.randint(k3, (_BATCH_SIZE, n_real), 0, 4672)
    padding_moves = jnp.full((_BATCH_SIZE, NUM_CANDIDATES - n_real), -1, dtype=jnp.int32)
    candidate_moves = jnp.concatenate([real_moves, padding_moves], axis=1).astype(jnp.int32)
    x = jnp.concatenate([token_position, global_flags, candidate_moves], axis=1).astype(jnp.int32)

    mask = jnp.concatenate([
        jnp.ones((_BATCH_SIZE, n_real), dtype=jnp.int32),
        jnp.zeros((_BATCH_SIZE, NUM_CANDIDATES - n_real), dtype=jnp.int32),
    ], axis=1)
    label = jax.random.randint(k4, (_BATCH_SIZE,), 0, n_real)  # toujours un slot reel (contrat dataset)

    model = ChessTokenCandidateModel(
        num_candidates=NUM_CANDIDATES, dropout_rate=0.1, token_dim=32, num_bottleneck_tokens=4,
        num_heads=4, num_trunk_layers=2,
    )
    variables = model.init({"params": jax.random.PRNGKey(0), "dropout": jax.random.PRNGKey(0)}, x, training=True)

    def loss_fn(params):
        vars_with_params = {**variables, "params": params}
        out = model.apply(vars_with_params, x, training=True, rngs={"dropout": jax.random.PRNGKey(3)})
        return compute_chess_token_candidate_loss(out, label, mask)

    loss, grads = jax.value_and_grad(loss_fn)(variables["params"])
    assert bool(jnp.isfinite(loss)), f"loss non finie (batch avec {NUM_CANDIDATES - n_real} slots de padding reels) : {loss}"

    top_level_modules = list(variables["params"].keys())
    dead_modules = []
    for module_name in top_level_modules:
        leaves = jax.tree_util.tree_leaves(grads[module_name])
        assert all(bool(jnp.all(jnp.isfinite(leaf))) for leaf in leaves), (
            f"{module_name} : gradient contient NaN/Inf avec un batch partiellement masque"
        )
        if not any(bool(jnp.any(leaf != 0)) for leaf in leaves):
            dead_modules.append(module_name)
    assert not dead_modules, (
        f"module(s) sans gradient non-nul avec un batch partiellement masque (padding reel, "
        f"candidate_moves=-1) : {dead_modules}"
    )
    print(f"OK - jax.grad de bout en bout reussit avec un batch REALISTE ({NUM_CANDIDATES - n_real}/"
          f"{NUM_CANDIDATES} slots en padding, candidate_moves=-1 dessus, exerce la branche "
          f"safe_moves sous gradient) (loss={float(loss):.4f}), gradients finis et non-degeneres "
          f"sur les {len(top_level_modules)} modules")


def test_pos_embed_weight_sharing():
    # Design Notes du SPEC : from_square DOIT reutiliser la MEME table pos_embed que le
    # tronc (poids partages, 0 parametre supplementaire) - PAS une nouvelle table
    # independante.
    model, variables, x = _init_model(jax.random.PRNGKey(0))
    params = variables["params"]
    embed_like_tables = [k for k in params.keys() if "embed" in k.lower() or "pos" in k.lower()]
    assert "pos_embed" in params, f"table pos_embed introuvable dans les params : {list(params.keys())}"

    # --- Verification 1 (COMPTAGE, proxy indirect) : une seule table de shape
    # (64, token_dim) dans tout le modele - si le modele avait par erreur cree une 2e
    # table positionnelle independante pour les candidats, on trouverait un module
    # supplementaire de meme shape. NE PROUVE PAS a lui seul une identite reelle (deux
    # tables DIFFERENTES pourraient coincidentellement avoir la meme shape sans etre
    # partagees) - voir Verification 2 ci-dessous pour une preuve structurelle directe.
    pos_embed_shape = params["pos_embed"]["embedding"].shape
    assert pos_embed_shape == (64, model.token_dim), f"pos_embed shape inattendue : {pos_embed_shape}"
    duplicate_tables = [
        k for k in embed_like_tables
        if k != "pos_embed" and "embedding" in params[k]
        and params[k]["embedding"].shape == pos_embed_shape
    ]
    assert not duplicate_tables, (
        f"table(s) positionnelle(s) dupliquee(s) trouvee(s) (poids NON partages) : {duplicate_tables}"
    )

    # --- Verification 2 (IDENTITE STRUCTURELLE, code review 2026-08-13) : inspection
    # directe du graphe de calcul JAX (jaxpr) de model.apply pour confirmer que la MEME
    # feuille de parametres "pos_embed/embedding" est bien lue par exactement DEUX
    # operations de lookup distinctes dans le MEME forward pass - une pour les positions
    # du plateau (indices = jnp.arange(64), 64 positions fixes), une pour from_square des
    # candidats (indices shape (batch, num_candidates)). Contrairement a la Verification 1
    # (qui ne peut que constater l'ABSENCE d'une 2e table), ceci prouve positivement que
    # c'est la MEME instance de tableau qui alimente les deux usages : jax.make_jaxpr
    # aplatit params UNE SEULE FOIS en invars, donc s'il n'y a bien qu'UN SEUL invar
    # "pos_embed/embedding" (confirme par la Verification 1) et que ce invar apparait
    # comme argument de deux equations de lookup differentes, ces deux lookups lisent
    # necessairement le meme tableau physique - pas une coincidence de shape.
    def apply_fn(p):
        return model.apply({"params": p}, x, training=False, rngs={"dropout": jax.random.PRNGKey(0)})

    jaxpr = jax.make_jaxpr(apply_fn)(params).jaxpr
    flat_paths = jax.tree_util.tree_flatten_with_path(params)[0]
    pos_embed_invar = None
    for (path, _), invar in zip(flat_paths, jaxpr.invars):
        path_str = "/".join(str(p) for p in path)
        if "pos_embed" in path_str and "embedding" in path_str:
            pos_embed_invar = invar
            break
    assert pos_embed_invar is not None, "invar pos_embed/embedding introuvable dans le jaxpr de model.apply"

    lookups = [eqn for eqn in jaxpr.eqns if pos_embed_invar in eqn.invars]
    assert len(lookups) == 2, (
        f"attendu exactement 2 operations lisant directement l'invar pos_embed/embedding "
        f"dans le jaxpr (positions du plateau + from_square candidats), obtenu {len(lookups)} - "
        f"preuve d'identite structurelle en echec (verifie une VRAIE identite de tableau, "
        f"pas seulement l'absence d'une 2e table de meme forme)"
    )
    other_arg_shapes = sorted(
        tuple(v.aval.shape for v in eqn.invars if v is not pos_embed_invar)[0]
        for eqn in lookups
    )
    # Les indices de l'autre operande de chaque lookup doivent correspondre exactement
    # aux deux usages attendus : (64,) = position_ids (arange(64), tronc) et
    # (batch, num_candidates) = from_square (tete de scoring candidats) - confirme que ce
    # sont bien CES deux usages precis, pas deux lookups non lies par coincidence.
    assert (64,) in other_arg_shapes, (
        f"aucun des 2 lookups pos_embed n'a d'indices de shape (64,) (position_ids du "
        f"tronc) : shapes observees {other_arg_shapes}"
    )
    assert (_BATCH_SIZE, NUM_CANDIDATES) in other_arg_shapes, (
        f"aucun des 2 lookups pos_embed n'a d'indices de shape ({_BATCH_SIZE}, {NUM_CANDIDATES}) "
        f"(from_square des candidats) : shapes observees {other_arg_shapes}"
    )
    print(f"OK - une seule table pos_embed {pos_embed_shape} dans tout le modele (Verification 1, "
          f"comptage) ET identite structurelle confirmee par inspection du jaxpr : le MEME invar "
          f"pos_embed/embedding est lu par exactement 2 lookups distincts (positions du plateau "
          f"(64,) + from_square candidats ({_BATCH_SIZE}, {NUM_CANDIDATES})) dans le meme forward "
          f"pass (Verification 2, preuve directe de partage, pas juste un comptage de tables)")


def test_strategy_loss_and_metrics():
    strategy = ChessTokenStrategy()
    assert strategy.primary_metric_name == "PolicyAccuracy"
    assert strategy.optimization_mode == "max"

    logits = jax.random.normal(jax.random.PRNGKey(5), (_BATCH_SIZE, NUM_CANDIDATES))
    label = jax.random.randint(jax.random.PRNGKey(6), (_BATCH_SIZE,), 0, NUM_CANDIDATES)
    mask = jnp.ones((_BATCH_SIZE, NUM_CANDIDATES), dtype=jnp.int32)
    targets_raw = {"candidate_label": np.array(label), "candidate_mask": np.array(mask)}

    images_out, targets_out, is_training_flag = strategy.preprocess_batch(None, targets_raw, True)
    assert is_training_flag is False
    assert targets_out["candidate_label"].dtype == jnp.int32
    assert targets_out["candidate_mask"].dtype == jnp.int32

    loss = strategy.compute_loss(logits, targets_out)
    assert loss.shape == (), f"attendu un scalaire, obtenu shape {loss.shape}"
    assert bool(jnp.isfinite(loss))

    ref_loss = compute_chess_token_candidate_loss(logits, targets_out["candidate_label"], targets_out["candidate_mask"])
    assert bool(jnp.allclose(loss, ref_loss)), "compute_loss devrait deleguer integralement a compute_chess_token_candidate_loss"

    metric = strategy.compute_metrics(logits, targets_out)
    assert 0.0 <= float(metric) <= 1.0, f"PolicyAccuracy hors de [0,1] : {float(metric)}"
    print(f"OK - ChessTokenStrategy.compute_loss delegue a compute_chess_token_candidate_loss "
          f"(loss={float(loss):.4f}), compute_metrics dans [0,1] ({float(metric):.4f})")


def test_real_dataset_and_forward_pass_if_spike_available():
    # Test d'integration avec le VRAI fichier spike (pas un factice). Passe
    # silencieusement (skip) si le fichier n'est pas present sur cette machine (meme
    # patron que tests/test_chess_move_token_model.py:268-291).
    npz_path = "/home/aobled/Documents/data/chunks/chess_token_candidate_spike/chess_token_candidate_spike.npz"
    if not os.path.exists(npz_path):
        print("SKIP - dataset spike absent sur cette machine, test d'integration ignore")
        return

    from data_management import ChessTokenCandidateDataset

    ds_mgr = ChessTokenCandidateDataset(npz_path=npz_path, batch_size=8, val_split=0.1)
    train_ds, val_ds = ds_mgr.get_dataset()
    packed, targets = next(iter(train_ds.as_numpy_iterator()))

    assert packed.dtype == np.int32
    assert packed.shape == (8, PACKED_WIDTH)
    assert targets["candidate_label"].dtype == np.int32
    assert targets["candidate_mask"].dtype == np.int32
    assert (targets["candidate_label"] >= 0).all() and (targets["candidate_label"] < NUM_CANDIDATES).all()
    # le label doit toujours pointer un slot marque valide dans candidate_mask (contrat .npz)
    label_is_valid_slot = targets["candidate_mask"][np.arange(8), targets["candidate_label"]]
    assert (label_is_valid_slot == 1).all(), "candidate_label pointe un slot masque (padding) - dataset malforme"

    model = ChessTokenCandidateModel(num_candidates=NUM_CANDIDATES, dropout_rate=0.1)
    variables = model.init({"params": jax.random.PRNGKey(0), "dropout": jax.random.PRNGKey(0)},
                            jnp.array(packed), training=True)
    out = model.apply(variables, jnp.array(packed), training=False)
    assert out.shape == (8, NUM_CANDIDATES)
    assert bool(jnp.all(jnp.isfinite(out))), "logits non finis sur un vrai batch"

    loss = compute_chess_token_candidate_loss(out, jnp.array(targets["candidate_label"]), jnp.array(targets["candidate_mask"]))
    assert bool(jnp.isfinite(loss)), f"loss non finie sur un vrai batch : {loss}"
    print(f"OK - ChessTokenCandidateDataset (vrai fichier spike) : batch {packed.shape}, "
          f"labels toujours sur un slot valide, forward pass + loss reussis (loss={float(loss):.4f})")


if __name__ == "__main__":
    test_output_shape_and_dtype()
    test_init_with_float32_dummy_does_not_crash()
    test_dtype_survives_trainer_cast_path()
    test_from_square_move_type_decomposition_at_boundaries()
    test_masking_correctness_loss_and_argmax()
    test_get_model_factory_and_registry()
    test_end_to_end_differentiability()
    test_end_to_end_differentiability_with_masking()
    test_pos_embed_weight_sharing()
    test_strategy_loss_and_metrics()
    test_real_dataset_and_forward_pass_if_spike_available()
    print("\nTous les tests du modele echecs (token candidate) sont passes.")
