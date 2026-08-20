"""
Test de validation pour spec-compute-dtype-hardware (2026-08-17, AD-1..AD-7) :
generalisation de `compute_dtype` (precision mixte, derivee du materiel) a
SophisticatedCNN32Plus (CIFAR10) et SophisticatedCNN128Lite (FIGHTERJET_CLASSIFICATION),
au dela de son seul precedent (ChessMoveTokenTransformer, tests/test_chess_move_token_model.py).

Etendu par spec-compute-dtype-rollout-classification (2026-08-17, Groupe 1 du rollout) :
SophisticatedCNN128Plus (meme famille/sous-modules que 32Plus/128Lite) et Kepler1DConvNet
(plus simple, pas de sous-module partage, pas de nn.BatchNorm/nn.LayerNorm).

Etendu par spec-compute-dtype-batchnorm-layernorm (2026-08-17, amendement AD-4) :
SophisticatedCNN32Plus/128Lite/128Plus passent desormais dtype=self.compute_dtype a
CHAQUE nn.BatchNorm/nn.LayerNorm (plus jamais exclus, cf ancienne exception documentee
plus bas, desormais inversee pour ces 3 modeles). Consequence mesuree empiriquement
(pas supposee) : SEBlock/SpatialAttention, dont l'entree x vient maintenant d'un
BatchNorm qui ressort en compute_dtype (et non plus toujours float32), ressortent
elles aussi en compute_dtype de bout en bout - plus de "reset" a chaque bloc.

Etendu par spec-compute-dtype-detection (2026-08-17, dernier groupe du rollout) :
AircraftDetectorUNet (FIGHTERJET_DETECTION) et AircraftDetectorCenterNet (JAX_DETECTOR)
- meme famille d'architecture (Conv/BatchNorm uniquement, pas de LayerNorm/sous-module
partage), nouveau cas non couvert par le precedent classification : sortie(s) passant
par nn.sigmoid (masque UNet, tete heatmap CenterNet) - recast float32 AVANT la
non-linearite, pas juste avant le retour. Input reel 224x224 (dataset_configs.py,
JAX_DETECTOR/FIGHTERJET_DETECTION) ; tests ci-dessous utilisent 32x32 (divisible par 8,
survit aux 3 max-pools de l'encodeur, verifie empiriquement avant de figer ce choix -
uniquement pour la vitesse d'execution des tests, aucun impact sur le mecanisme teste).

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
from flax import linen as nn

from model_library import (
    SophisticatedCNN32Plus,
    SophisticatedCNN128Lite,
    SophisticatedCNN128Plus,
    Kepler1DConvNet,
    SeparableConv,
    SEBlock,
    SpatialAttention,
    AircraftDetectorUNet,
    AircraftDetectorCenterNet,
    create_sophisticated_cnn_32_plus,
    create_sophisticated_cnn_128_lite,
    create_sophisticated_cnn_128_plus,
    create_chess_move_token_transformer,
    create_kepler_1d_cnn,
    create_aircraft_detector_unet,
    create_aircraft_detector_centernet,
    create_chess_cnn_attention_policy_value,
    get_model,
    resolve_compute_dtype,
    MODELS,
)
from detection_target_encoding import HEATMAP_KEY, SIZE_KEY
from loss_functions import compute_segmentation_loss, compute_centernet_loss


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


_MATMUL_LAYER_PREFIXES = ("Conv_", "SeparableConv_", "Dense_")

# spec-compute-dtype-batchnorm-layernorm (2026-08-17, amendement AD-4) : BatchNorm/
# LayerNorm recoivent maintenant dtype=self.compute_dtype dans SophisticatedCNN32Plus/
# 128Lite/128Plus (les 3 seuls modeles ou cette verification renforcee est utilisee,
# voir include_normalization ci-dessous). SEBlock/SpatialAttention y sont inclus aussi :
# mesure empirique (pas suppose, cf _assert_all_matmul_layers_match_dtype) montrant
# qu'avec leur entree x desormais deja en compute_dtype (plus jamais float32 en sortie
# de BatchNorm), leur multiplication elementwise x*signal ne re-promeut plus vers
# float32 et ressort donc elle aussi en compute_dtype de bout en bout.
_NORMALIZATION_LAYER_PREFIXES = ("BatchNorm_", "LayerNorm_")
_ATTENTION_LAYER_PREFIXES = ("SEBlock_", "SpatialAttention_")


def _matmul_intermediates(mutated, extra_prefixes=()):
    # Filtre les intermediates captures (capture_intermediates=True) aux couches
    # matmul lourdes (Conv/SeparableConv/Dense, AD-4) - plus, si extra_prefixes est
    # fourni, les prefixes additionnels demandes (ex. BatchNorm_/LayerNorm_/SEBlock_/
    # SpatialAttention_ pour les 3 modeles amendes par AD-4 2026-08-17). Exclut
    # toujours Dropout_ (pas un concern dtype de calcul ici).
    prefixes = _MATMUL_LAYER_PREFIXES + tuple(extra_prefixes)
    return {
        name: val["__call__"][0]
        for name, val in mutated["intermediates"].items()
        if name.startswith(prefixes)
    }


def _assert_all_matmul_layers_match_dtype(mutated, expected_dtype, model_label, include_normalization=False, include_attention_submodules=None):
    # Revue Blind Hunter, 2026-08-17 : verifier seulement Conv_0 ne detecterait pas
    # un site d'appel qui aurait perdu dtype=self.compute_dtype plus loin dans le
    # reseau (ex. une des SeparableConv residuelles). Verifie ICI TOUTES les couches
    # matmul capturees, pas seulement la premiere.
    #
    # include_normalization=False (defaut, ex. Kepler1DConvNet qui n'a pas de
    # BatchNorm/LayerNorm) : ne verifie que Conv/SeparableConv/Dense, comme avant.
    #
    # include_normalization=True (SophisticatedCNN32Plus/128Lite/128Plus uniquement,
    # spec-compute-dtype-batchnorm-layernorm, 2026-08-17, amendement AD-4) : verifie EN
    # PLUS BatchNorm_*/LayerNorm_*/SEBlock_*/SpatialAttention_*. Avant cet amendement,
    # nn.BatchNorm n'etait JAMAIS appele avec dtype= et ressortait donc TOUJOURS en
    # float32 quel que soit le compute_dtype de la couche precedente, ce qui faisait
    # aussi ressortir SEBlock/SpatialAttention en float32 en cascade (leur
    # multiplication elementwise x*signal promeut vers float32 des que x l'est deja) -
    # cette ancienne exception est desormais explicitement testee comme INVERSEE : ces
    # 4 familles de couches doivent maintenant matcher expected_dtype tout comme
    # Conv/SeparableConv/Dense, la chaine de precision reduite ne se reinitialisant
    # plus a chaque bloc de normalisation (verifie empiriquement, pas suppose - cf
    # docstring de ce fichier).
    #
    # include_attention_submodules (spec-compute-dtype-detection, 2026-08-17) : par
    # defaut (None) suit include_normalization, comme avant l'ajout de ce parametre -
    # aucun changement de comportement pour les appelants existants (SophisticatedCNN*/
    # Kepler1DConvNet). Explicitement mis a False par AircraftDetectorUNet/CenterNet
    # (ci-dessous) : ces 2 classes n'ont AUCUN sous-module partage (pas de
    # SeparableConv/SEBlock/SpatialAttention, verifie en lisant le code - familles
    # Conv/BatchNorm uniquement), donc exiger des couches SEBlock_*/SpatialAttention_*
    # ferait echouer le test a coup sur pour un motif hors-sujet (sous-module absent,
    # pas un dtype incorrect).
    if include_attention_submodules is None:
        include_attention_submodules = include_normalization
    extra_prefixes = ()
    if include_normalization:
        extra_prefixes += _NORMALIZATION_LAYER_PREFIXES
    if include_attention_submodules:
        extra_prefixes += _ATTENTION_LAYER_PREFIXES
    layers = _matmul_intermediates(mutated, extra_prefixes=extra_prefixes)
    assert layers, f"{model_label}: aucune couche matmul capturee, capture_intermediates a-t-il fonctionne ?"
    # Revue Edge Case Hunter, 2026-08-17 : `assert layers` ci-dessus ne garantit que le
    # dict combine est non-vide (les entrees Conv/Dense suffisent a le satisfaire) - si
    # include_normalization=True et que la capture avait silencieusement rate les
    # prefixes BatchNorm_/LayerNorm_/SEBlock_/SpatialAttention_ (ex. suite a un
    # changement de comportement Flax), le test passerait SANS AVOIR VERIFIE l'invariant
    # que cet amendement existe pour prouver. Verification explicite par famille de
    # prefixe demandee.
    if include_normalization:
        found = [name for name in layers if name.startswith(_NORMALIZATION_LAYER_PREFIXES)]
        assert found, f"{model_label}: aucune couche capturee pour les prefixes {_NORMALIZATION_LAYER_PREFIXES} - capture_intermediates a-t-il bien cible ces couches ?"
    if include_attention_submodules:
        found = [name for name in layers if name.startswith(_ATTENTION_LAYER_PREFIXES)]
        assert found, f"{model_label}: aucune couche capturee pour les prefixes {_ATTENTION_LAYER_PREFIXES} - capture_intermediates a-t-il bien cible ces couches ?"
    mismatched = {name: arr.dtype for name, arr in layers.items() if arr.dtype != expected_dtype}
    assert not mismatched, (
        f"{model_label}: couches dont le dtype ne correspond pas a {expected_dtype} : {mismatched}"
    )


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
    _assert_all_matmul_layers_match_dtype(mutated_f32, jnp.float32, "SophisticatedCNN32Plus(compute_dtype=float32)", include_normalization=True)
    assert _all_variables_are_float32(variables_f32), "poids/batch_stats doivent rester float32 (compute_dtype=float32)"

    model_bf16 = SophisticatedCNN32Plus(num_classes=10, dropout_rate=0.0, compute_dtype=jnp.bfloat16)
    variables_bf16 = model_bf16.init(
        {"params": jax.random.PRNGKey(0), "dropout": jax.random.PRNGKey(0)}, x, training=True
    )
    out_bf16, mutated_bf16 = model_bf16.apply(
        variables_bf16, x, training=False, mutable=["batch_stats", "intermediates"], capture_intermediates=True
    )
    assert out_bf16.dtype == jnp.float32, f"sortie doit rester float32 meme sous compute_dtype=bfloat16 (recast explicite), obtenu {out_bf16.dtype}"
    _assert_all_matmul_layers_match_dtype(mutated_bf16, jnp.bfloat16, "SophisticatedCNN32Plus(compute_dtype=bfloat16)", include_normalization=True)
    assert _all_variables_are_float32(variables_bf16), "poids/batch_stats doivent rester float32 (compat checkpoint) meme sous compute_dtype=bfloat16 (AD-6)"
    assert bool(jnp.all(jnp.isfinite(out_bf16))), "sortie non finie (NaN/Inf) avec compute_dtype=bfloat16"

    # Revue Edge Case Hunter/Blind Hunter, 2026-08-17 : les verifications ci-dessus
    # utilisent toutes training=False (use_running_average=True sur nn.BatchNorm) - un
    # chemin de code DIFFERENT de training=True (reduction de stats EN DIRECT sur le
    # batch), le chemin reellement emprunte pendant l'entrainement, precisement celui
    # que cet amendement cible pour la vitesse. Verifie ici separement.
    out_bf16_train, mutated_bf16_train = model_bf16.apply(
        variables_bf16, x, training=True, rngs={"dropout": jax.random.PRNGKey(2)},
        mutable=["batch_stats", "intermediates"], capture_intermediates=True
    )
    _assert_all_matmul_layers_match_dtype(mutated_bf16_train, jnp.bfloat16, "SophisticatedCNN32Plus(compute_dtype=bfloat16, training=True)", include_normalization=True)
    assert all(l.dtype == jnp.float32 for l in jax.tree_util.tree_leaves(mutated_bf16_train["batch_stats"])), "batch_stats doit rester float32 apres une mise a jour EN DIRECT (training=True) meme sous compute_dtype=bfloat16"
    assert bool(jnp.all(jnp.isfinite(out_bf16_train))), "sortie non finie (NaN/Inf) en training=True avec compute_dtype=bfloat16"
    print("OK - SophisticatedCNN32Plus : compute_dtype reellement observe sur TOUTES les couches matmul (training=False ET True), sortie/poids/batch_stats toujours float32")


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
    _assert_all_matmul_layers_match_dtype(mutated_f32, jnp.float32, "SophisticatedCNN128Lite(compute_dtype=float32)", include_normalization=True)
    assert _all_variables_are_float32(variables_f32), "poids/batch_stats doivent rester float32 (compute_dtype=float32)"

    model_bf16 = SophisticatedCNN128Lite(num_classes=32, dropout_rate=0.0, compute_dtype=jnp.bfloat16)
    variables_bf16 = model_bf16.init(
        {"params": jax.random.PRNGKey(0), "dropout": jax.random.PRNGKey(0)}, x, training=True
    )
    out_bf16, mutated_bf16 = model_bf16.apply(
        variables_bf16, x, training=False, mutable=["batch_stats", "intermediates"], capture_intermediates=True
    )
    assert out_bf16.dtype == jnp.float32, f"sortie doit rester float32 meme sous compute_dtype=bfloat16 (recast explicite), obtenu {out_bf16.dtype}"
    _assert_all_matmul_layers_match_dtype(mutated_bf16, jnp.bfloat16, "SophisticatedCNN128Lite(compute_dtype=bfloat16)", include_normalization=True)
    assert _all_variables_are_float32(variables_bf16), "poids/batch_stats doivent rester float32 (compat checkpoint) meme sous compute_dtype=bfloat16 (AD-6)"
    assert bool(jnp.all(jnp.isfinite(out_bf16))), "sortie non finie (NaN/Inf) avec compute_dtype=bfloat16"

    # training=True (reduction BatchNorm EN DIRECT, cf SophisticatedCNN32Plus ci-dessus).
    out_bf16_train, mutated_bf16_train = model_bf16.apply(
        variables_bf16, x, training=True, rngs={"dropout": jax.random.PRNGKey(2)},
        mutable=["batch_stats", "intermediates"], capture_intermediates=True
    )
    _assert_all_matmul_layers_match_dtype(mutated_bf16_train, jnp.bfloat16, "SophisticatedCNN128Lite(compute_dtype=bfloat16, training=True)", include_normalization=True)
    assert all(l.dtype == jnp.float32 for l in jax.tree_util.tree_leaves(mutated_bf16_train["batch_stats"])), "batch_stats doit rester float32 apres une mise a jour EN DIRECT (training=True) meme sous compute_dtype=bfloat16"
    assert bool(jnp.all(jnp.isfinite(out_bf16_train))), "sortie non finie (NaN/Inf) en training=True avec compute_dtype=bfloat16"
    print("OK - SophisticatedCNN128Lite : compute_dtype reellement observe sur TOUTES les couches matmul (training=False ET True), sortie/poids/batch_stats toujours float32")


def test_sophisticated_cnn_128_plus_compute_dtype_really_observed():
    # Meme preuve que 32Plus/128Lite ci-dessus, pour SophisticatedCNN128Plus
    # (spec-compute-dtype-rollout-classification, Groupe 1 du rollout, 2026-08-17) :
    # meme famille/sous-modules (SeparableConv/SEBlock/SpatialAttention) que les 2
    # modeles deja adaptes, meme garde de recast float32 en sortie.
    x = jnp.zeros((2, 128, 128, 3), dtype=jnp.float32)

    model_f32 = SophisticatedCNN128Plus(num_classes=32, dropout_rate=0.0, compute_dtype=jnp.float32)
    variables_f32 = model_f32.init(
        {"params": jax.random.PRNGKey(0), "dropout": jax.random.PRNGKey(0)}, x, training=True
    )
    out_f32, mutated_f32 = model_f32.apply(
        variables_f32, x, training=False, mutable=["batch_stats", "intermediates"], capture_intermediates=True
    )
    assert out_f32.dtype == jnp.float32, f"sortie doit toujours etre float32 (recast explicite), obtenu {out_f32.dtype}"
    _assert_all_matmul_layers_match_dtype(mutated_f32, jnp.float32, "SophisticatedCNN128Plus(compute_dtype=float32)", include_normalization=True)
    assert _all_variables_are_float32(variables_f32), "poids/batch_stats doivent rester float32 (compute_dtype=float32)"

    model_bf16 = SophisticatedCNN128Plus(num_classes=32, dropout_rate=0.0, compute_dtype=jnp.bfloat16)
    variables_bf16 = model_bf16.init(
        {"params": jax.random.PRNGKey(0), "dropout": jax.random.PRNGKey(0)}, x, training=True
    )
    out_bf16, mutated_bf16 = model_bf16.apply(
        variables_bf16, x, training=False, mutable=["batch_stats", "intermediates"], capture_intermediates=True
    )
    assert out_bf16.dtype == jnp.float32, f"sortie doit rester float32 meme sous compute_dtype=bfloat16 (recast explicite), obtenu {out_bf16.dtype}"
    _assert_all_matmul_layers_match_dtype(mutated_bf16, jnp.bfloat16, "SophisticatedCNN128Plus(compute_dtype=bfloat16)", include_normalization=True)
    assert _all_variables_are_float32(variables_bf16), "poids/batch_stats doivent rester float32 (compat checkpoint) meme sous compute_dtype=bfloat16 (AD-6)"
    assert bool(jnp.all(jnp.isfinite(out_bf16))), "sortie non finie (NaN/Inf) avec compute_dtype=bfloat16"

    # training=True (reduction BatchNorm EN DIRECT, cf SophisticatedCNN32Plus ci-dessus).
    out_bf16_train, mutated_bf16_train = model_bf16.apply(
        variables_bf16, x, training=True, rngs={"dropout": jax.random.PRNGKey(2)},
        mutable=["batch_stats", "intermediates"], capture_intermediates=True
    )
    _assert_all_matmul_layers_match_dtype(mutated_bf16_train, jnp.bfloat16, "SophisticatedCNN128Plus(compute_dtype=bfloat16, training=True)", include_normalization=True)
    assert all(l.dtype == jnp.float32 for l in jax.tree_util.tree_leaves(mutated_bf16_train["batch_stats"])), "batch_stats doit rester float32 apres une mise a jour EN DIRECT (training=True) meme sous compute_dtype=bfloat16"
    assert bool(jnp.all(jnp.isfinite(out_bf16_train))), "sortie non finie (NaN/Inf) en training=True avec compute_dtype=bfloat16"
    print("OK - SophisticatedCNN128Plus : compute_dtype reellement observe sur TOUTES les couches matmul (training=False ET True), sortie/poids/batch_stats toujours float32")


def test_kepler_1d_cnn_compute_dtype_really_observed():
    # Meme preuve que ci-dessus pour Kepler1DConvNet (spec-compute-dtype-rollout-
    # classification, 2026-08-17) : pas de sous-module partage, pas de nn.BatchNorm/
    # nn.LayerNorm dans cette classe (verifie en lisant le code - donc pas de collection
    # "batch_stats" a verifier separement ici, contrairement aux 3 modeles CNN 2D
    # ci-dessus qui utilisent nn.BatchNorm : _all_variables_are_float32 reste correct a
    # appeler tel quel, elle itere simplement sur les collections presentes dans
    # `variables`, qui ne contiendra ici que "params").
    x = jnp.zeros((2, 256, 1), dtype=jnp.float32)

    model_f32 = Kepler1DConvNet(num_classes=2, dropout_rate=0.0, compute_dtype=jnp.float32)
    variables_f32 = model_f32.init(
        {"params": jax.random.PRNGKey(0), "dropout": jax.random.PRNGKey(0)}, x, training=True
    )
    out_f32, mutated_f32 = model_f32.apply(
        variables_f32, x, training=False, mutable=["intermediates"], capture_intermediates=True
    )
    assert out_f32.dtype == jnp.float32, f"sortie doit toujours etre float32 (recast explicite), obtenu {out_f32.dtype}"
    _assert_all_matmul_layers_match_dtype(mutated_f32, jnp.float32, "Kepler1DConvNet(compute_dtype=float32)")
    assert _all_variables_are_float32(variables_f32), "poids doivent rester float32 (compute_dtype=float32)"

    model_bf16 = Kepler1DConvNet(num_classes=2, dropout_rate=0.0, compute_dtype=jnp.bfloat16)
    variables_bf16 = model_bf16.init(
        {"params": jax.random.PRNGKey(0), "dropout": jax.random.PRNGKey(0)}, x, training=True
    )
    out_bf16, mutated_bf16 = model_bf16.apply(
        variables_bf16, x, training=False, mutable=["intermediates"], capture_intermediates=True
    )
    assert out_bf16.dtype == jnp.float32, f"sortie doit rester float32 meme sous compute_dtype=bfloat16 (recast explicite), obtenu {out_bf16.dtype}"
    # Kepler1DConvNet n'a pas de nn.BatchNorm (verifie) - contrairement aux 3 modeles
    # CNN 2D ci-dessus, TOUTES ses couches matmul devraient donc rester en
    # compute_dtype d'un bout a l'autre (pas de "reset" par BatchNorm entre blocs).
    _assert_all_matmul_layers_match_dtype(mutated_bf16, jnp.bfloat16, "Kepler1DConvNet(compute_dtype=bfloat16)")
    assert _all_variables_are_float32(variables_bf16), "poids doivent rester float32 (compat checkpoint) meme sous compute_dtype=bfloat16 (AD-6)"
    assert bool(jnp.all(jnp.isfinite(out_bf16))), "sortie non finie (NaN/Inf) avec compute_dtype=bfloat16"
    print("OK - Kepler1DConvNet : compute_dtype reellement observe sur TOUTES les couches matmul, sortie/poids toujours float32")


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


def test_aircraft_detector_unet_compute_dtype_really_observed():
    # spec-compute-dtype-detection (2026-08-17, dernier groupe du rollout) : meme
    # preuve que les modeles de classification ci-dessus, pour AircraftDetectorUNet
    # (FIGHTERJET_DETECTION). Entree reelle 224x224 (dataset_configs.py) ; 32x32 utilise
    # ici (divisible par 8, survit aux 3 max-pools de l'encodeur, verifie empiriquement -
    # seule la vitesse d'execution du test change, pas le mecanisme).
    #
    # include_attention_submodules=False : AircraftDetectorUNet n'a AUCUN sous-module
    # partage (pas de SeparableConv/SEBlock/SpatialAttention, verifie en lisant le
    # code) - seul BatchNorm (pas de LayerNorm non plus) suit compute_dtype ici.
    x = jnp.zeros((2, 32, 32, 3), dtype=jnp.float32)

    model_f32 = AircraftDetectorUNet(dropout_rate=0.0, compute_dtype=jnp.float32)
    variables_f32 = model_f32.init(
        {"params": jax.random.PRNGKey(0), "dropout": jax.random.PRNGKey(0)}, x, training=True
    )
    out_f32, mutated_f32 = model_f32.apply(
        variables_f32, x, training=False, mutable=["batch_stats", "intermediates"], capture_intermediates=True
    )
    assert out_f32.dtype == jnp.float32, f"masque doit toujours etre float32 (recast avant sigmoid), obtenu {out_f32.dtype}"
    _assert_all_matmul_layers_match_dtype(mutated_f32, jnp.float32, "AircraftDetectorUNet(compute_dtype=float32)", include_normalization=True, include_attention_submodules=False)
    assert _all_variables_are_float32(variables_f32), "poids/batch_stats doivent rester float32 (compute_dtype=float32)"

    model_bf16 = AircraftDetectorUNet(dropout_rate=0.0, compute_dtype=jnp.bfloat16)
    variables_bf16 = model_bf16.init(
        {"params": jax.random.PRNGKey(0), "dropout": jax.random.PRNGKey(0)}, x, training=True
    )
    out_bf16, mutated_bf16 = model_bf16.apply(
        variables_bf16, x, training=False, mutable=["batch_stats", "intermediates"], capture_intermediates=True
    )
    assert out_bf16.dtype == jnp.float32, f"masque doit rester float32 meme sous compute_dtype=bfloat16 (recast avant sigmoid), obtenu {out_bf16.dtype}"
    _assert_all_matmul_layers_match_dtype(mutated_bf16, jnp.bfloat16, "AircraftDetectorUNet(compute_dtype=bfloat16)", include_normalization=True, include_attention_submodules=False)
    assert _all_variables_are_float32(variables_bf16), "poids/batch_stats doivent rester float32 (compat checkpoint) meme sous compute_dtype=bfloat16 (AD-6)"
    assert bool(jnp.all(jnp.isfinite(out_bf16))), "masque non fini (NaN/Inf) avec compute_dtype=bfloat16"

    # training=True : reduction BatchNorm EN DIRECT sur le batch, chemin de code
    # different de training=False (use_running_average), cf modeles de classification
    # ci-dessus.
    out_bf16_train, mutated_bf16_train = model_bf16.apply(
        variables_bf16, x, training=True, rngs={"dropout": jax.random.PRNGKey(2)},
        mutable=["batch_stats", "intermediates"], capture_intermediates=True
    )
    assert out_bf16_train.dtype == jnp.float32, f"masque doit rester float32 en training=True (recast avant sigmoid), obtenu {out_bf16_train.dtype}"
    _assert_all_matmul_layers_match_dtype(mutated_bf16_train, jnp.bfloat16, "AircraftDetectorUNet(compute_dtype=bfloat16, training=True)", include_normalization=True, include_attention_submodules=False)
    assert all(l.dtype == jnp.float32 for l in jax.tree_util.tree_leaves(mutated_bf16_train["batch_stats"])), "batch_stats doit rester float32 apres une mise a jour EN DIRECT (training=True) meme sous compute_dtype=bfloat16"
    assert bool(jnp.all(jnp.isfinite(out_bf16_train))), "masque non fini (NaN/Inf) en training=True avec compute_dtype=bfloat16"
    print("OK - AircraftDetectorUNet : compute_dtype reellement observe sur TOUTES les couches Conv/BatchNorm (training=False ET True), masque/poids/batch_stats toujours float32")


def test_aircraft_detector_centernet_compute_dtype_really_observed():
    # Meme preuve que ci-dessus pour AircraftDetectorCenterNet (JAX_DETECTOR) - sortie
    # dict a 2 tetes (HEATMAP_KEY passe par nn.sigmoid, SIZE_KEY sans activation) : les
    # DEUX doivent rester float32, dans le cas float32 ET bfloat16.
    x = jnp.zeros((2, 32, 32, 3), dtype=jnp.float32)

    model_f32 = AircraftDetectorCenterNet(dropout_rate=0.0, heatmap_prior=0.01, compute_dtype=jnp.float32)
    variables_f32 = model_f32.init(
        {"params": jax.random.PRNGKey(0), "dropout": jax.random.PRNGKey(0)}, x, training=True
    )
    outputs_f32, mutated_f32 = model_f32.apply(
        variables_f32, x, training=False, mutable=["batch_stats", "intermediates"], capture_intermediates=True
    )
    assert outputs_f32[HEATMAP_KEY].dtype == jnp.float32, f"heatmap doit toujours etre float32 (recast avant sigmoid), obtenu {outputs_f32[HEATMAP_KEY].dtype}"
    assert outputs_f32[SIZE_KEY].dtype == jnp.float32, f"size doit toujours etre float32 (recast direct), obtenu {outputs_f32[SIZE_KEY].dtype}"
    _assert_all_matmul_layers_match_dtype(mutated_f32, jnp.float32, "AircraftDetectorCenterNet(compute_dtype=float32)", include_normalization=True, include_attention_submodules=False)
    assert _all_variables_are_float32(variables_f32), "poids/batch_stats doivent rester float32 (compute_dtype=float32)"

    model_bf16 = AircraftDetectorCenterNet(dropout_rate=0.0, heatmap_prior=0.01, compute_dtype=jnp.bfloat16)
    variables_bf16 = model_bf16.init(
        {"params": jax.random.PRNGKey(0), "dropout": jax.random.PRNGKey(0)}, x, training=True
    )
    outputs_bf16, mutated_bf16 = model_bf16.apply(
        variables_bf16, x, training=False, mutable=["batch_stats", "intermediates"], capture_intermediates=True
    )
    assert outputs_bf16[HEATMAP_KEY].dtype == jnp.float32, f"heatmap doit rester float32 meme sous compute_dtype=bfloat16 (recast avant sigmoid), obtenu {outputs_bf16[HEATMAP_KEY].dtype}"
    assert outputs_bf16[SIZE_KEY].dtype == jnp.float32, f"size doit rester float32 meme sous compute_dtype=bfloat16 (recast direct), obtenu {outputs_bf16[SIZE_KEY].dtype}"
    _assert_all_matmul_layers_match_dtype(mutated_bf16, jnp.bfloat16, "AircraftDetectorCenterNet(compute_dtype=bfloat16)", include_normalization=True, include_attention_submodules=False)
    assert _all_variables_are_float32(variables_bf16), "poids/batch_stats doivent rester float32 (compat checkpoint) meme sous compute_dtype=bfloat16 (AD-6)"
    assert bool(jnp.all(jnp.isfinite(outputs_bf16[HEATMAP_KEY]))), "heatmap non finie (NaN/Inf) avec compute_dtype=bfloat16"
    assert bool(jnp.all(jnp.isfinite(outputs_bf16[SIZE_KEY]))), "size non finie (NaN/Inf) avec compute_dtype=bfloat16"

    # training=True : reduction BatchNorm EN DIRECT sur le batch (cf AircraftDetectorUNet
    # ci-dessus).
    outputs_bf16_train, mutated_bf16_train = model_bf16.apply(
        variables_bf16, x, training=True, rngs={"dropout": jax.random.PRNGKey(2)},
        mutable=["batch_stats", "intermediates"], capture_intermediates=True
    )
    assert outputs_bf16_train[HEATMAP_KEY].dtype == jnp.float32, f"heatmap doit rester float32 en training=True (recast avant sigmoid), obtenu {outputs_bf16_train[HEATMAP_KEY].dtype}"
    assert outputs_bf16_train[SIZE_KEY].dtype == jnp.float32, f"size doit rester float32 en training=True (recast direct), obtenu {outputs_bf16_train[SIZE_KEY].dtype}"
    _assert_all_matmul_layers_match_dtype(mutated_bf16_train, jnp.bfloat16, "AircraftDetectorCenterNet(compute_dtype=bfloat16, training=True)", include_normalization=True, include_attention_submodules=False)
    assert all(l.dtype == jnp.float32 for l in jax.tree_util.tree_leaves(mutated_bf16_train["batch_stats"])), "batch_stats doit rester float32 apres une mise a jour EN DIRECT (training=True) meme sous compute_dtype=bfloat16"
    assert bool(jnp.all(jnp.isfinite(outputs_bf16_train[HEATMAP_KEY]))), "heatmap non finie (NaN/Inf) en training=True avec compute_dtype=bfloat16"
    assert bool(jnp.all(jnp.isfinite(outputs_bf16_train[SIZE_KEY]))), "size non finie (NaN/Inf) en training=True avec compute_dtype=bfloat16"
    print("OK - AircraftDetectorCenterNet : compute_dtype reellement observe sur TOUTES les couches Conv/BatchNorm (training=False ET True), heatmap/size/poids/batch_stats toujours float32")


def test_aircraft_detector_gradients_finite_under_bfloat16():
    # Revue Blind Hunter, 2026-08-17 : les 2 tests ci-dessus ne verifient que le FORWARD
    # pass (dtype/isfinite de la sortie) - jamais le GRADIENT a travers la vraie loss
    # (compute_segmentation_loss/compute_centernet_loss), la ou une perte de precision
    # bfloat16 se manifesterait le plus plausiblement (termes jnp.log/jnp.power pres de
    # la saturation) - risque de NaN en entrainement reel qu'un test forward-only ne
    # peut pas detecter. Verifie ici via jax.value_and_grad reel, pas suppose.
    #
    # heatmap_prior=0.0000268 : valeur REELLE de JAX_DETECTOR (dataset_configs.py:490),
    # pas le defaut 0.01 des tests ci-dessus - biais initial ~-10.5 (vs ~-4.6 pour le
    # defaut), le cas le plus extreme/pertinent en production, jamais exerce ci-dessus.
    x = jnp.zeros((2, 32, 32, 3), dtype=jnp.float32)

    model_unet = AircraftDetectorUNet(dropout_rate=0.0, compute_dtype=jnp.bfloat16)
    variables_unet = model_unet.init(
        {"params": jax.random.PRNGKey(0), "dropout": jax.random.PRNGKey(0)}, x, training=True
    )
    true_mask = jnp.zeros((2, 32, 32, 1), dtype=jnp.float32)

    def unet_loss_fn(params):
        pred_mask = model_unet.apply({**variables_unet, "params": params}, x, training=False)
        return compute_segmentation_loss(pred_mask, true_mask)

    unet_loss, unet_grads = jax.value_and_grad(unet_loss_fn)(variables_unet["params"])
    assert bool(jnp.isfinite(unet_loss)), f"AircraftDetectorUNet: loss non finie sous compute_dtype=bfloat16, obtenu {unet_loss}"
    unet_grad_leaves = jax.tree_util.tree_leaves(unet_grads)
    assert all(bool(jnp.all(jnp.isfinite(g))) for g in unet_grad_leaves), "AircraftDetectorUNet: gradient non fini (NaN/Inf) sous compute_dtype=bfloat16"
    assert all(g.dtype == jnp.float32 for g in unet_grad_leaves), "AircraftDetectorUNet: gradients doivent rester float32 (params float32, AD-6)"

    model_centernet = AircraftDetectorCenterNet(dropout_rate=0.0, heatmap_prior=0.0000268, compute_dtype=jnp.bfloat16)
    variables_centernet = model_centernet.init(
        {"params": jax.random.PRNGKey(0), "dropout": jax.random.PRNGKey(0)}, x, training=True
    )
    targets = {
        HEATMAP_KEY: jnp.zeros((2, 32, 32, 1), dtype=jnp.float32),
        SIZE_KEY: jnp.zeros((2, 32, 32, 2), dtype=jnp.float32),
    }

    def centernet_loss_fn(params):
        outputs = model_centernet.apply({**variables_centernet, "params": params}, x, training=False)
        return compute_centernet_loss(outputs, targets)

    centernet_loss, centernet_grads = jax.value_and_grad(centernet_loss_fn)(variables_centernet["params"])
    assert bool(jnp.isfinite(centernet_loss)), f"AircraftDetectorCenterNet: loss non finie (heatmap_prior reel extreme) sous compute_dtype=bfloat16, obtenu {centernet_loss}"
    centernet_grad_leaves = jax.tree_util.tree_leaves(centernet_grads)
    assert all(bool(jnp.all(jnp.isfinite(g))) for g in centernet_grad_leaves), "AircraftDetectorCenterNet: gradient non fini (NaN/Inf) sous compute_dtype=bfloat16 avec heatmap_prior reel extreme"
    assert all(g.dtype == jnp.float32 for g in centernet_grad_leaves), "AircraftDetectorCenterNet: gradients doivent rester float32 (params float32, AD-6)"
    print("OK - AircraftDetectorUNet/CenterNet : loss+gradients reels finis sous compute_dtype=bfloat16 (y compris heatmap_prior extreme reel de JAX_DETECTOR)")


def test_aircraft_detector_gradients_finite_under_bfloat16_training_true():
    # Revue de code (Story 12.1) : test_aircraft_detector_gradients_finite_under_bfloat16
    # ci-dessus n'exerce que training=False (BatchNorm sur moyennes glissantes figees a
    # l'init, aucune vraie statistique apprise) - jamais training=True (BatchNorm reduit
    # EN DIRECT sur le batch, dtype=self.compute_dtype, chemin de code REELLEMENT
    # utilise pendant un entrainement reel), ni combine a jax.value_and_grad. Meme motif
    # que ce test se donne lui-meme pour exister (perte de precision bfloat16 la ou elle
    # se manifesterait le plus plausiblement) - la reduction batch_stats EN DIRECT sous
    # bfloat16, retropropagee, est un chemin distinct du forward-only deja teste dans
    # test_aircraft_detector_unet/centernet_compute_dtype_really_observed (training=True
    # y est teste, mais sans grad) et du gradient deja teste ci-dessus (avec grad, mais
    # sans training=True). Ce test couvre l'intersection des deux, non testee ailleurs.
    x = jnp.zeros((2, 32, 32, 3), dtype=jnp.float32)

    model_unet = AircraftDetectorUNet(dropout_rate=0.0, compute_dtype=jnp.bfloat16)
    variables_unet = model_unet.init(
        {"params": jax.random.PRNGKey(0), "dropout": jax.random.PRNGKey(0)}, x, training=True
    )
    true_mask = jnp.zeros((2, 32, 32, 1), dtype=jnp.float32)

    def unet_loss_fn_train(params):
        pred_mask, _ = model_unet.apply(
            {**variables_unet, "params": params}, x, training=True,
            mutable=["batch_stats"], rngs={"dropout": jax.random.PRNGKey(3)},
        )
        return compute_segmentation_loss(pred_mask, true_mask)

    unet_loss, unet_grads = jax.value_and_grad(unet_loss_fn_train)(variables_unet["params"])
    assert bool(jnp.isfinite(unet_loss)), f"AircraftDetectorUNet (training=True): loss non finie sous compute_dtype=bfloat16, obtenu {unet_loss}"
    unet_grad_leaves = jax.tree_util.tree_leaves(unet_grads)
    assert all(bool(jnp.all(jnp.isfinite(g))) for g in unet_grad_leaves), "AircraftDetectorUNet (training=True): gradient non fini (NaN/Inf) sous compute_dtype=bfloat16"

    model_centernet = AircraftDetectorCenterNet(dropout_rate=0.0, heatmap_prior=0.0000268, compute_dtype=jnp.bfloat16)
    variables_centernet = model_centernet.init(
        {"params": jax.random.PRNGKey(0), "dropout": jax.random.PRNGKey(0)}, x, training=True
    )
    targets = {
        HEATMAP_KEY: jnp.zeros((2, 32, 32, 1), dtype=jnp.float32),
        SIZE_KEY: jnp.zeros((2, 32, 32, 2), dtype=jnp.float32),
    }

    def centernet_loss_fn_train(params):
        outputs, _ = model_centernet.apply(
            {**variables_centernet, "params": params}, x, training=True,
            mutable=["batch_stats"], rngs={"dropout": jax.random.PRNGKey(3)},
        )
        return compute_centernet_loss(outputs, targets)

    centernet_loss, centernet_grads = jax.value_and_grad(centernet_loss_fn_train)(variables_centernet["params"])
    assert bool(jnp.isfinite(centernet_loss)), f"AircraftDetectorCenterNet (training=True): loss non finie (heatmap_prior reel extreme) sous compute_dtype=bfloat16, obtenu {centernet_loss}"
    centernet_grad_leaves = jax.tree_util.tree_leaves(centernet_grads)
    assert all(bool(jnp.all(jnp.isfinite(g))) for g in centernet_grad_leaves), "AircraftDetectorCenterNet (training=True): gradient non fini (NaN/Inf) sous compute_dtype=bfloat16 avec heatmap_prior reel extreme"
    print("OK - AircraftDetectorUNet/CenterNet : loss+gradients reels finis sous compute_dtype=bfloat16 EN training=True (reduction BatchNorm en direct, retropropagee)")


def _last_conv_output_by_channels(mutated, num_channels):
    # Retrouve l'intermediate Conv_* dont le dernier axe a exactement num_channels
    # canaux (capture_intermediates=True) - selecteur robuste car UNet/CenterNet n'ont
    # chacun qu'une seule couche Conv de sortie a 1 canal (masque/heatmap) et
    # CenterNet une seule a 2 canaux (size), verifie empiriquement (aucune couche
    # intermediaire du reseau ne partage ce nombre de canaux en sortie).
    matches = [
        val["__call__"][0]
        for name, val in mutated["intermediates"].items()
        if name.startswith("Conv_") and val["__call__"][0].shape[-1] == num_channels
    ]
    assert len(matches) == 1, (
        f"attendu exactement 1 couche Conv de sortie a {num_channels} canal(aux), trouve {len(matches)}"
    )
    return matches[0]


def test_aircraft_detector_unet_sigmoid_computed_in_float32_not_bfloat16():
    # Revue de code (Story 12.1) : test_aircraft_detector_unet_compute_dtype_really_observed
    # ne verifie que le DTYPE de la sortie finale (float32) - un recast DEPLACE apres
    # nn.sigmoid (out = nn.sigmoid(conv_out).astype(jnp.float32), sigmoid calcule en
    # bfloat16 - PAS le code reel, voir model_library.py) produirait EXACTEMENT le meme
    # dtype final et passerait ce test-la sans le detecter, alors que c'est numeriquement
    # different (sigmoid en precision reduite est plus lossy). Reconstruit ici les deux
    # variantes depuis le logit brut capture (avant cast/sigmoid) et verifie laquelle la
    # sortie reelle du modele reproduit. Entree aleatoire (pas des zeros) : un logit
    # degenere pres de 0 peut rendre sigmoid(cast(x)) et sigmoid(x).astype(...)
    # indistinguables a l'atol du test (verifie empiriquement) - meme discipline que
    # tests/test_chess_token_model.py::_random_packed_batch.
    x = jax.random.normal(jax.random.PRNGKey(7), (2, 32, 32, 3), dtype=jnp.float32)
    model_bf16 = AircraftDetectorUNet(dropout_rate=0.0, compute_dtype=jnp.bfloat16)
    variables_bf16 = model_bf16.init(
        {"params": jax.random.PRNGKey(0), "dropout": jax.random.PRNGKey(0)}, x, training=True
    )
    out_bf16, mutated_bf16 = model_bf16.apply(
        variables_bf16, x, training=False, mutable=["batch_stats", "intermediates"], capture_intermediates=True
    )
    raw_logit_bf16 = _last_conv_output_by_channels(mutated_bf16, 1)
    assert raw_logit_bf16.dtype == jnp.bfloat16, f"logit brut attendu bfloat16, obtenu {raw_logit_bf16.dtype}"

    reconstructed_correct = nn.sigmoid(raw_logit_bf16.astype(jnp.float32))  # cast PUIS sigmoid (code reel)
    reconstructed_wrong_order = nn.sigmoid(raw_logit_bf16).astype(jnp.float32)  # sigmoid PUIS cast (alternative bugguee)

    assert jnp.allclose(out_bf16, reconstructed_correct, atol=1e-6), (
        "la sortie reelle du modele ne correspond pas a sigmoid(cast(logit)) - le recast float32 "
        "n'a pas lieu AVANT nn.sigmoid comme attendu"
    )
    assert not jnp.allclose(out_bf16, reconstructed_wrong_order, atol=1e-6), (
        "la sortie reelle du modele correspond a sigmoid(logit).astype(float32) (sigmoid calcule en "
        "bfloat16) - le test ne peut alors plus distinguer le bon ordre du mauvais, verifier que "
        "les deux reconstructions different reellement pour ce logit"
    )
    print("OK - AircraftDetectorUNet : nn.sigmoid est bien calcule en float32 (recast AVANT, pas apres)")


def test_aircraft_detector_centernet_sigmoid_computed_in_float32_not_bfloat16():
    # Meme preuve que ci-dessus pour la tete heatmap de AircraftDetectorCenterNet.
    # Entree aleatoire (pas des zeros), meme raison que le test UNet ci-dessus.
    x = jax.random.normal(jax.random.PRNGKey(7), (2, 32, 32, 3), dtype=jnp.float32)
    model_bf16 = AircraftDetectorCenterNet(dropout_rate=0.0, heatmap_prior=0.01, compute_dtype=jnp.bfloat16)
    variables_bf16 = model_bf16.init(
        {"params": jax.random.PRNGKey(0), "dropout": jax.random.PRNGKey(0)}, x, training=True
    )
    outputs_bf16, mutated_bf16 = model_bf16.apply(
        variables_bf16, x, training=False, mutable=["batch_stats", "intermediates"], capture_intermediates=True
    )
    raw_logit_bf16 = _last_conv_output_by_channels(mutated_bf16, 1)  # tete heatmap (1 canal)
    assert raw_logit_bf16.dtype == jnp.bfloat16, f"logit brut attendu bfloat16, obtenu {raw_logit_bf16.dtype}"

    reconstructed_correct = nn.sigmoid(raw_logit_bf16.astype(jnp.float32))
    reconstructed_wrong_order = nn.sigmoid(raw_logit_bf16).astype(jnp.float32)

    assert jnp.allclose(outputs_bf16[HEATMAP_KEY], reconstructed_correct, atol=1e-6), (
        "la heatmap reelle ne correspond pas a sigmoid(cast(logit)) - le recast float32 "
        "n'a pas lieu AVANT nn.sigmoid comme attendu"
    )
    assert not jnp.allclose(outputs_bf16[HEATMAP_KEY], reconstructed_wrong_order, atol=1e-6), (
        "la heatmap reelle correspond a sigmoid(logit).astype(float32) (sigmoid calcule en bfloat16) - "
        "verifier que les deux reconstructions different reellement pour ce logit"
    )
    print("OK - AircraftDetectorCenterNet : nn.sigmoid (tete heatmap) est bien calcule en float32 (recast AVANT, pas apres)")


def test_factories_accept_and_forward_compute_dtype():
    # Les factories (pas seulement les classes) doivent accepter compute_dtype comme
    # parametre nomme explicite et le transmettre reellement au constructeur - pas
    # juste l'absorber silencieusement (AD-3d).
    model = create_sophisticated_cnn_32_plus(num_classes=10, dropout_rate=0.0, compute_dtype=jnp.bfloat16)
    assert model.compute_dtype == jnp.bfloat16, "create_sophisticated_cnn_32_plus doit transmettre compute_dtype au constructeur"

    model = create_sophisticated_cnn_128_lite(num_classes=32, dropout_rate=0.0, compute_dtype=jnp.bfloat16)
    assert model.compute_dtype == jnp.bfloat16, "create_sophisticated_cnn_128_lite doit transmettre compute_dtype au constructeur"

    # Groupe 1 du rollout (spec-compute-dtype-rollout-classification, 2026-08-17).
    model = create_sophisticated_cnn_128_plus(num_classes=32, dropout_rate=0.0, compute_dtype=jnp.bfloat16)
    assert model.compute_dtype == jnp.bfloat16, "create_sophisticated_cnn_128_plus doit transmettre compute_dtype au constructeur"

    model = create_kepler_1d_cnn(num_classes=2, dropout_rate=0.0, compute_dtype=jnp.bfloat16)
    assert model.compute_dtype == jnp.bfloat16, "create_kepler_1d_cnn doit transmettre compute_dtype au constructeur"

    # Dernier groupe du rollout (spec-compute-dtype-detection, 2026-08-17).
    model = create_aircraft_detector_unet(dropout_rate=0.2, compute_dtype=jnp.bfloat16)
    assert model.compute_dtype == jnp.bfloat16, "create_aircraft_detector_unet doit transmettre compute_dtype au constructeur"

    model = create_aircraft_detector_centernet(dropout_rate=0.2, heatmap_prior=0.01, compute_dtype=jnp.bfloat16)
    assert model.compute_dtype == jnp.bfloat16, "create_aircraft_detector_centernet doit transmettre compute_dtype au constructeur"

    # Meme chemin que main.py : get_model(model_name, **model_kwargs)
    model_via_registry = get_model("sophisticated_cnn_32_plus", num_classes=10, dropout_rate=0.0, compute_dtype=jnp.bfloat16)
    assert model_via_registry.compute_dtype == jnp.bfloat16

    model_via_registry = get_model("sophisticated_cnn_128_plus", num_classes=32, dropout_rate=0.0, compute_dtype=jnp.bfloat16)
    assert model_via_registry.compute_dtype == jnp.bfloat16

    model_via_registry = get_model("kepler_1d_cnn", num_classes=2, dropout_rate=0.0, compute_dtype=jnp.bfloat16)
    assert model_via_registry.compute_dtype == jnp.bfloat16

    model_via_registry = get_model("aircraft_detector_unet", dropout_rate=0.2, compute_dtype=jnp.bfloat16)
    assert model_via_registry.compute_dtype == jnp.bfloat16

    model_via_registry = get_model("aircraft_detector_centernet", dropout_rate=0.2, heatmap_prior=0.01, compute_dtype=jnp.bfloat16)
    assert model_via_registry.compute_dtype == jnp.bfloat16

    # Defauts (sans compute_dtype explicite) doivent rester float32 - jamais bfloat16
    # par defaut hors injection explicite (comportement GPU/CPU inchange si un
    # appelant construit le modele sans passer par la derivation materielle).
    default_model = create_sophisticated_cnn_32_plus(num_classes=10, dropout_rate=0.0)
    assert default_model.compute_dtype == jnp.float32, "defaut de la factory doit rester float32"
    default_model = create_sophisticated_cnn_128_plus(num_classes=32, dropout_rate=0.0)
    assert default_model.compute_dtype == jnp.float32, "defaut de la factory doit rester float32"
    default_model = create_kepler_1d_cnn(num_classes=2, dropout_rate=0.0)
    assert default_model.compute_dtype == jnp.float32, "defaut de la factory doit rester float32"
    default_model = create_aircraft_detector_unet(dropout_rate=0.2)
    assert default_model.compute_dtype == jnp.float32, "defaut de la factory doit rester float32"
    default_model = create_aircraft_detector_centernet(dropout_rate=0.2, heatmap_prior=0.01)
    assert default_model.compute_dtype == jnp.float32, "defaut de la factory doit rester float32"
    print("OK - create_sophisticated_cnn_32_plus/128_lite/128_plus/kepler_1d_cnn/aircraft_detector_unet/centernet acceptent et transmettent reellement compute_dtype, defaut float32")


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
    # Etendu par spec-compute-dtype-rollout-classification (2026-08-17, Groupe 1) :
    # sophisticated_cnn_128_plus et kepler_1d_cnn rejoignent le set adapte. Pour
    # kepler_1d_cnn en particulier, create_kepler_1d_cnn a ete change de `(**kwargs)`
    # a `(compute_dtype=jnp.float32, **kwargs)` precisement pour que ce test passe :
    # **kwargs seul aurait fonctionne a l'execution (forwarding correct vers
    # Kepler1DConvNet) mais serait reste invisible a l'introspection stricte par nom
    # (AD-3), donc jamais reellement injecte par main.py.
    # Etendu par spec-compute-dtype-detection (2026-08-17, dernier groupe du rollout) :
    # aircraft_detector_unet/centernet rejoignent le set adapte, meme raisonnement -
    # create_aircraft_detector_unet/centernet passent de `(dropout_rate=..., **kwargs)`
    # a `(dropout_rate=..., compute_dtype=jnp.float32, **kwargs)`. Le bug preexistant
    # **kwargs-non-transmis-au-constructeur (AD-3c, ex. num_classes pour CenterNet)
    # reste entier pour ces 2 factories - seul compute_dtype devient un parametre
    # nomme explicite, hors scope de corriger le reste.
    adapted = {
        "sophisticated_cnn_32_plus", "sophisticated_cnn_128_lite", "sophisticated_cnn_128_plus", "kepler_1d_cnn",
        "aircraft_detector_unet", "aircraft_detector_centernet",
    }
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
                f"(modeles non adaptes, rollout differe - Deferred de la spine)"
            )

    # Sanity explicite demandee par le spec : un exemple concret adapte vs un exemple
    # concret non adapte.
    assert "compute_dtype" in inspect.signature(create_sophisticated_cnn_32_plus).parameters
    assert "compute_dtype" in inspect.signature(create_sophisticated_cnn_128_plus).parameters
    assert "compute_dtype" in inspect.signature(create_kepler_1d_cnn).parameters
    assert "compute_dtype" in inspect.signature(create_aircraft_detector_unet).parameters
    assert "compute_dtype" in inspect.signature(create_aircraft_detector_centernet).parameters
    # create_chess_cnn_attention_policy_value illustre le piege **kwargs toujours
    # documente (AD-3c, Deferred de la spine) : accepte **kwargs en facade mais n'a
    # pas de parametre NOMME compute_dtype - l'introspection stricte (par nom, pas
    # par **kwargs) ne doit pas le cibler. (aircraft_detector_unet/centernet
    # servaient auparavant de contre-exemple ici ; ils rejoignent desormais le set
    # adapte ci-dessus - spec-compute-dtype-detection, 2026-08-17.)
    assert "compute_dtype" not in inspect.signature(create_chess_cnn_attention_policy_value).parameters
    # f-string plutot qu'un chiffre code en dur (revue Blind Hunter, 2026-08-17) : ce
    # compte a deja du etre bascule manuellement de 4 a 6 lors de ce cycle - reste
    # synchronise automatiquement avec `adapted` si un futur modele rejoint le rollout.
    print(f"OK - introspection cible exactement les {len(adapted)} factories adaptees + le precedent chess_move_token_transformer, aucun autre")


if __name__ == "__main__":
    test_sophisticated_cnn_32_plus_compute_dtype_really_observed()
    test_sophisticated_cnn_128_lite_compute_dtype_really_observed()
    test_sophisticated_cnn_128_plus_compute_dtype_really_observed()
    test_kepler_1d_cnn_compute_dtype_really_observed()
    test_shared_submodules_compute_dtype_really_observed()
    test_aircraft_detector_unet_compute_dtype_really_observed()
    test_aircraft_detector_centernet_compute_dtype_really_observed()
    test_aircraft_detector_gradients_finite_under_bfloat16()
    test_aircraft_detector_gradients_finite_under_bfloat16_training_true()
    test_aircraft_detector_unet_sigmoid_computed_in_float32_not_bfloat16()
    test_aircraft_detector_centernet_sigmoid_computed_in_float32_not_bfloat16()
    test_factories_accept_and_forward_compute_dtype()
    test_resolve_compute_dtype_both_branches()
    test_chess_move_token_transformer_factory_valid_string_still_works()
    test_chess_move_token_transformer_factory_rejects_invalid_dtype_object()
    test_introspection_targets_only_adapted_factories()
