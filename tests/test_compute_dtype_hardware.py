"""
Test de validation pour spec-compute-dtype-hardware (2026-08-17, AD-1..AD-7) :
generalisation de `compute_dtype` (precision mixte, derivee du materiel) a
SophisticatedCNN32Plus (CIFAR10) et SophisticatedCNN128Lite (FIGHTERJET_CLASSIFICATION),
au dela de son seul precedent (ChessMoveTokenTransformer, tests/test_chess_move_token_model.py).

Script autonome - ce projet n'a pas de framework de test formel (meme convention que
tests/test_chess_model.py / tests/test_chess_move_token_model.py /
tests/test_chess_token_model.py). Executer directement :
    python tests/test_compute_dtype_hardware.py
"""

import sys
import os
import inspect

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
import jax.numpy as jnp

from model_library import (
    SophisticatedCNN32Plus,
    SophisticatedCNN128Lite,
    SeparableConv,
    SEBlock,
    SpatialAttention,
    create_sophisticated_cnn_32_plus,
    create_sophisticated_cnn_128_lite,
    create_chess_move_token_transformer,
    create_kepler_1d_cnn,
    create_aircraft_detector_unet,
    get_model,
    resolve_compute_dtype,
    MODELS,
)


def _params_are_float32(variables):
    param_dtypes = set(str(l.dtype) for l in jax.tree_util.tree_leaves(variables["params"]))
    return param_dtypes == {"float32"}


def _all_variables_are_float32(variables):
    # AD-6 : verifie params ET tout autre collection mutable (ex. batch_stats de
    # nn.BatchNorm - moyennes/variances courantes) - _params_are_float32 ci-dessus ne
    # verifiait que "params", laissant batch_stats non couvert (revue Blind Hunter,
    # 2026-08-17 : SophisticatedCNN32Plus/128Lite utilisent nn.BatchNorm, dont l'etat
    # mutable vit dans une collection separee de "params").
    all_dtypes = set()
    for collection_name, collection in variables.items():
        for leaf in jax.tree_util.tree_leaves(collection):
            all_dtypes.add(str(leaf.dtype))
    return all_dtypes == {"float32"}


def test_sophisticated_cnn_32_plus_compute_dtype_really_observed():
    # AD-3/AD-4/AD-6 : compute_dtype doit REELLEMENT changer le dtype de calcul (pas
    # seulement etre accepte sans erreur) - poids stockes (params ET batch_stats)
    # restent float32 (checkpoint-compatible). La SORTIE finale est desormais TOUJOURS
    # float32 (recast explicite avant retour, revue Edge Case Hunter 2026-08-17 - meme
    # garde que ChessMoveTokenTransformer/loss cross-entropy) : la preuve d'effet reel
    # se fait donc sur une couche INTERNE (Conv_0, via capture_intermediates), pas sur
    # la sortie.
    x = jnp.zeros((2, 32, 32, 3), dtype=jnp.float32)

    model_f32 = SophisticatedCNN32Plus(num_classes=10, dropout_rate=0.0, compute_dtype=jnp.float32)
    variables_f32 = model_f32.init(
        {"params": jax.random.PRNGKey(0), "dropout": jax.random.PRNGKey(0)}, x, training=True
    )
    out_f32, mutated_f32 = model_f32.apply(
        variables_f32, x, training=False, mutable=["batch_stats", "intermediates"], capture_intermediates=True
    )
    assert out_f32.dtype == jnp.float32, f"sortie doit toujours etre float32 (recast explicite), obtenu {out_f32.dtype}"
    assert mutated_f32["intermediates"]["Conv_0"]["__call__"][0].dtype == jnp.float32
    assert _all_variables_are_float32(variables_f32), "poids/batch_stats doivent rester float32 (compute_dtype=float32)"

    model_bf16 = SophisticatedCNN32Plus(num_classes=10, dropout_rate=0.0, compute_dtype=jnp.bfloat16)
    variables_bf16 = model_bf16.init(
        {"params": jax.random.PRNGKey(0), "dropout": jax.random.PRNGKey(0)}, x, training=True
    )
    out_bf16, mutated_bf16 = model_bf16.apply(
        variables_bf16, x, training=False, mutable=["batch_stats", "intermediates"], capture_intermediates=True
    )
    assert out_bf16.dtype == jnp.float32, f"sortie doit rester float32 meme sous compute_dtype=bfloat16 (recast explicite), obtenu {out_bf16.dtype}"
    first_conv_dtype = mutated_bf16["intermediates"]["Conv_0"]["__call__"][0].dtype
    assert first_conv_dtype == jnp.bfloat16, f"compute_dtype=bfloat16 doit se propager a la 1ere couche Conv, obtenu {first_conv_dtype}"
    assert _all_variables_are_float32(variables_bf16), "poids/batch_stats doivent rester float32 (compat checkpoint) meme sous compute_dtype=bfloat16 (AD-6)"
    assert bool(jnp.all(jnp.isfinite(out_bf16))), "sortie non finie (NaN/Inf) avec compute_dtype=bfloat16"
    print("OK - SophisticatedCNN32Plus : compute_dtype reellement observe en interne (Conv_0), sortie/poids/batch_stats toujours float32")


def test_sophisticated_cnn_128_lite_compute_dtype_really_observed():
    # Meme preuve que ci-dessus, pour le second (et dernier) modele adopte dans ce
    # chantier (FIGHTERJET_CLASSIFICATION).
    x = jnp.zeros((2, 128, 128, 3), dtype=jnp.float32)

    model_f32 = SophisticatedCNN128Lite(num_classes=32, dropout_rate=0.0, compute_dtype=jnp.float32)
    variables_f32 = model_f32.init(
        {"params": jax.random.PRNGKey(0), "dropout": jax.random.PRNGKey(0)}, x, training=True
    )
    out_f32, mutated_f32 = model_f32.apply(
        variables_f32, x, training=False, mutable=["batch_stats", "intermediates"], capture_intermediates=True
    )
    assert out_f32.dtype == jnp.float32, f"sortie doit toujours etre float32 (recast explicite), obtenu {out_f32.dtype}"
    assert mutated_f32["intermediates"]["Conv_0"]["__call__"][0].dtype == jnp.float32
    assert _all_variables_are_float32(variables_f32), "poids/batch_stats doivent rester float32 (compute_dtype=float32)"

    model_bf16 = SophisticatedCNN128Lite(num_classes=32, dropout_rate=0.0, compute_dtype=jnp.bfloat16)
    variables_bf16 = model_bf16.init(
        {"params": jax.random.PRNGKey(0), "dropout": jax.random.PRNGKey(0)}, x, training=True
    )
    out_bf16, mutated_bf16 = model_bf16.apply(
        variables_bf16, x, training=False, mutable=["batch_stats", "intermediates"], capture_intermediates=True
    )
    assert out_bf16.dtype == jnp.float32, f"sortie doit rester float32 meme sous compute_dtype=bfloat16 (recast explicite), obtenu {out_bf16.dtype}"
    first_conv_dtype = mutated_bf16["intermediates"]["Conv_0"]["__call__"][0].dtype
    assert first_conv_dtype == jnp.bfloat16, f"compute_dtype=bfloat16 doit se propager a la 1ere couche Conv, obtenu {first_conv_dtype}"
    assert _all_variables_are_float32(variables_bf16), "poids/batch_stats doivent rester float32 (compat checkpoint) meme sous compute_dtype=bfloat16 (AD-6)"
    assert bool(jnp.all(jnp.isfinite(out_bf16))), "sortie non finie (NaN/Inf) avec compute_dtype=bfloat16"
    print("OK - SophisticatedCNN128Lite : compute_dtype reellement observe en interne (Conv_0), sortie/poids/batch_stats toujours float32")


def test_shared_submodules_compute_dtype_really_observed():
    # AD-5 : SeparableConv/SEBlock/SpatialAttention recoivent compute_dtype
    # explicitement a chaque site d'appel - verifie ici en isolation (pas seulement a
    # travers les classes parentes ci-dessus) que chaque sous-module propage
    # reellement compute_dtype a ses propres nn.Conv/nn.Dense internes.
    #
    # x est fourni dans le MEME dtype que compute_dtype pour chaque cas : c'est la
    # situation reelle a l'interieur de SophisticatedCNN32Plus/128Lite, ou l'entree de
    # chaque sous-module est deja le tenseur bfloat16 produit par la couche
    # precedente (elle-meme construite avec dtype=self.compute_dtype). SEBlock/
    # SpatialAttention combinent leur signal interne (calcule en compute_dtype) par
    # multiplication ELEMENTWISE avec x d'origine (x * se_broadcast /
    # x * spatial_attn) - si x et le signal interne different de dtype, la
    # promotion JAX standard (bfloat16 * float32 -> float32) masquerait le vrai
    # signal a observer ici (le calcul interne du sous-module), d'ou ce choix.
    def _x(dtype):
        return jnp.zeros((2, 8, 8, 16), dtype=dtype)

    sep_f32 = SeparableConv(24, (3, 3), compute_dtype=jnp.float32)
    v = sep_f32.init(jax.random.PRNGKey(0), _x(jnp.float32), training=True)
    assert sep_f32.apply(v, _x(jnp.float32), training=False).dtype == jnp.float32
    sep_bf16 = SeparableConv(24, (3, 3), compute_dtype=jnp.bfloat16)
    v = sep_bf16.init(jax.random.PRNGKey(0), _x(jnp.bfloat16), training=True)
    assert sep_bf16.apply(v, _x(jnp.bfloat16), training=False).dtype == jnp.bfloat16, "SeparableConv doit propager compute_dtype a ses 2 nn.Conv internes"

    se_f32 = SEBlock(reduction=4, compute_dtype=jnp.float32)
    v = se_f32.init(jax.random.PRNGKey(0), _x(jnp.float32), training=True)
    assert se_f32.apply(v, _x(jnp.float32), training=False).dtype == jnp.float32
    se_bf16 = SEBlock(reduction=4, compute_dtype=jnp.bfloat16)
    v = se_bf16.init(jax.random.PRNGKey(0), _x(jnp.bfloat16), training=True)
    assert se_bf16.apply(v, _x(jnp.bfloat16), training=False).dtype == jnp.bfloat16, "SEBlock doit propager compute_dtype a ses 2 nn.Dense internes"

    sa_f32 = SpatialAttention(compute_dtype=jnp.float32)
    v = sa_f32.init(jax.random.PRNGKey(0), _x(jnp.float32), training=True)
    assert sa_f32.apply(v, _x(jnp.float32), training=False).dtype == jnp.float32
    sa_bf16 = SpatialAttention(compute_dtype=jnp.bfloat16)
    v = sa_bf16.init(jax.random.PRNGKey(0), _x(jnp.bfloat16), training=True)
    assert sa_bf16.apply(v, _x(jnp.bfloat16), training=False).dtype == jnp.bfloat16, "SpatialAttention doit propager compute_dtype a son nn.Conv interne"
    print("OK - SeparableConv/SEBlock/SpatialAttention : compute_dtype reellement observe individuellement (AD-5)")


def test_factories_accept_and_forward_compute_dtype():
    # Les factories (pas seulement les classes) doivent accepter compute_dtype comme
    # parametre nomme explicite et le transmettre reellement au constructeur - pas
    # juste l'absorber silencieusement (AD-3d).
    model = create_sophisticated_cnn_32_plus(num_classes=10, dropout_rate=0.0, compute_dtype=jnp.bfloat16)
    assert model.compute_dtype == jnp.bfloat16, "create_sophisticated_cnn_32_plus doit transmettre compute_dtype au constructeur"

    model = create_sophisticated_cnn_128_lite(num_classes=32, dropout_rate=0.0, compute_dtype=jnp.bfloat16)
    assert model.compute_dtype == jnp.bfloat16, "create_sophisticated_cnn_128_lite doit transmettre compute_dtype au constructeur"

    # Meme chemin que main.py : get_model(model_name, **model_kwargs)
    model_via_registry = get_model("sophisticated_cnn_32_plus", num_classes=10, dropout_rate=0.0, compute_dtype=jnp.bfloat16)
    assert model_via_registry.compute_dtype == jnp.bfloat16

    # Defauts (sans compute_dtype explicite) doivent rester float32 - jamais bfloat16
    # par defaut hors injection explicite (comportement GPU/CPU inchange si un
    # appelant construit le modele sans passer par la derivation materielle).
    default_model = create_sophisticated_cnn_32_plus(num_classes=10, dropout_rate=0.0)
    assert default_model.compute_dtype == jnp.float32, "defaut de la factory doit rester float32"
    print("OK - create_sophisticated_cnn_32_plus/128_lite acceptent et transmettent reellement compute_dtype, defaut float32")


def test_resolve_compute_dtype_both_branches():
    # AD-1 : la ligne qui derive reellement compute_dtype depuis le backend (bfloat16
    # si tpu, float32 sinon) - extraite dans model_library.resolve_compute_dtype()
    # pour etre testable sans importer main.py (effets de bord au chargement du
    # module - hardware init, prints - revue Blind Hunter, 2026-08-17 : cette ligne
    # n'etait exercee par aucun test avant ce fix).
    assert resolve_compute_dtype("tpu") == jnp.bfloat16, "backend=tpu doit resoudre bfloat16"
    assert resolve_compute_dtype("gpu") == jnp.float32, "backend=gpu doit resoudre float32 (prudence, pas une limite materielle)"
    assert resolve_compute_dtype("cpu") == jnp.float32, "backend=cpu doit resoudre float32"
    print("OK - resolve_compute_dtype : bfloat16 sur tpu, float32 partout ailleurs")


def test_chess_move_token_transformer_factory_valid_string_still_works():
    # Non-regression du chemin string historique apres le fix qui a ajoute le chemin
    # jnp.dtype deja resolu (revue Blind Hunter, 2026-08-17 : ce chemin n'etait plus
    # re-teste apres modification de create_chess_move_token_transformer). Seule la
    # rejection de string invalide etait couverte par le test pre-existant
    # (test_compute_dtype_factory_rejects_unknown_string, tests/test_chess_move_token_model.py).
    model = create_chess_move_token_transformer(num_classes=4672, compute_dtype="bfloat16")
    assert model.compute_dtype == jnp.bfloat16, "chemin string valide doit toujours resoudre correctement post-fix"
    print("OK - create_chess_move_token_transformer : chemin string valide toujours fonctionnel apres le fix str/jnp.dtype")


def test_chess_move_token_transformer_factory_rejects_invalid_dtype_object():
    # Le chemin non-string (jnp.dtype deja resolu, AD-2) acceptait n'importe quel
    # objet sans validation avant ce fix (revue Edge Case Hunter, 2026-08-17) - une
    # valeur invalide (ex. jnp.int32, pas un dtype de calcul valide pour ce mecanisme)
    # doit lever une ValueError explicite, pas echouer silencieusement plus loin dans
    # le graphe.
    try:
        create_chess_move_token_transformer(num_classes=4672, compute_dtype=jnp.int32)
        assert False, "attendu ValueError sur un jnp.dtype non reconnu (jnp.int32)"
    except ValueError as e:
        assert "int32" in str(e) or "non reconnu" in str(e)
        print(f"OK - compute_dtype=jnp.int32 (objet invalide) rejete explicitement : {e}")


def test_introspection_targets_only_adapted_factories():
    # AD-3 : le mecanisme d'injection de main.py verifie la presence d'un parametre
    # NOMME "compute_dtype" dans la signature de la factory cible (inspect.signature)
    # - jamais un registre maintenu a la main. Ce test n'importe pas main.py comme
    # module (declenche la detection materielle + les prints au chargement) : il
    # verifie directement le contrat d'introspection sur les factories reelles de
    # MODELS, ce qui est equivalent a executer la condition de main.py pour chaque
    # modele.
    adapted = {"sophisticated_cnn_32_plus", "sophisticated_cnn_128_lite"}
    for model_name, factory in MODELS.items():
        has_named_param = "compute_dtype" in inspect.signature(factory).parameters
        if model_name in adapted:
            assert has_named_param, f"{model_name} doit declarer un parametre nomme compute_dtype"
        elif model_name == "chess_move_token_transformer":
            # precedent existant (Epic 11), volontairement inchange - garde le
            # parametre nomme, non touche par ce chantier (AD-2/AD-7).
            assert has_named_param, "chess_move_token_transformer doit garder son parametre compute_dtype existant"
        else:
            assert not has_named_param, (
                f"{model_name} ne doit PAS declarer de parametre nomme compute_dtype "
                f"(9 modeles non adaptes, rollout differe - Deferred de la spine)"
            )

    # Sanity explicite demandee par le spec : un exemple concret adapte vs un exemple
    # concret non adapte.
    assert "compute_dtype" in inspect.signature(create_sophisticated_cnn_32_plus).parameters
    assert "compute_dtype" not in inspect.signature(create_kepler_1d_cnn).parameters
    # aircraft_detector_unet illustre le piege **kwargs deja documente (AD-3c) :
    # accepte **kwargs en facade mais n'a pas de parametre NOMME compute_dtype -
    # l'introspection stricte (par nom, pas par **kwargs) ne doit pas le cibler.
    assert "compute_dtype" not in inspect.signature(create_aircraft_detector_unet).parameters
    print("OK - introspection cible exactement les 2 factories adaptees + le precedent chess_move_token_transformer, aucun autre")


if __name__ == "__main__":
    test_sophisticated_cnn_32_plus_compute_dtype_really_observed()
    test_sophisticated_cnn_128_lite_compute_dtype_really_observed()
    test_shared_submodules_compute_dtype_really_observed()
    test_factories_accept_and_forward_compute_dtype()
    test_resolve_compute_dtype_both_branches()
    test_chess_move_token_transformer_factory_valid_string_still_works()
    test_chess_move_token_transformer_factory_rejects_invalid_dtype_object()
    test_introspection_targets_only_adapted_factories()
