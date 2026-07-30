"""
Test de validation pour ChessPolicyValueStrategy / ChessPolicyValueDataset / entrée
CHESS (Story 9.3, FR4/FR6, AD-17/AD-24). Script autonome, meme convention que les
Stories 9.1/9.2. Executer directement :
    python tests/test_chess_task_strategy.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import contextlib
import glob
import io
import tempfile

import jax
import jax.numpy as jnp
import numpy as np

from chess_target_encoding import NUM_MOVES, NUM_PLANES, POSITION_KEY, POLICY_KEY, VALUE_KEY, BOARD_SIZE
from dataset_builder.chess_pgn_dataset_tools import build_chess_dataset
from data_management import ChessPolicyValueDataset
from task_strategies import ChessPolicyValueStrategy
from loss_functions import compute_chess_policy_value_loss
from model_library import get_model
from dataset_configs import get_dataset_config, validate_config, DATASET_CONFIGS, DATA_ROOT
from trainer import Trainer

_TEST_PGN = """[Event "Test1"]
[Result "1-0"]

1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6 5. O-O Be7 1-0

[Event "Test2"]
[Result "0-1"]

1. d4 d5 2. c4 e6 3. Nc3 Nf6 4. Bg5 Be7 0-1

[Event "Test3"]
[Result "1/2-1/2"]

1. e4 e5 2. Nf3 Nc6 3. Bc4 Bc5 1/2-1/2
"""
# 10 + 8 + 6 = 24 demi-coups au total


def _build_test_dataset(tmpdir):
    pgn_path = os.path.join(tmpdir, "test_archive.pgn")
    with open(pgn_path, "w", encoding="utf-8") as f:
        f.write(_TEST_PGN)
    output_prefix = os.path.join(tmpdir, "chess_targets")
    total = build_chess_dataset(pgn_path, output_prefix, chunk_size=5)  # 24 ex -> 5 chunks (5,5,5,5,4)
    return output_prefix, total


def test_chess_policy_value_dataset_loading():
    with tempfile.TemporaryDirectory() as tmpdir:
        output_prefix, total = _build_test_dataset(tmpdir)
        assert total == 24, f"attendu 24 exemples, obtenu {total}"

        # val_split=0.4 sur 5 chunks -> n_val = max(1, int(5*0.4)) = 2 chunks val, 3 train
        dataset = ChessPolicyValueDataset(output_prefix, batch_size=2, val_split=0.4)
        assert len(dataset.train_chunks) == 3, f"attendu 3 chunks train, obtenu {len(dataset.train_chunks)}"
        assert len(dataset.val_chunks) == 2, f"attendu 2 chunks val, obtenu {len(dataset.val_chunks)}"
        assert set(dataset.train_chunks).isdisjoint(set(dataset.val_chunks)), "chunks train/val se chevauchent"

        train_ds, val_ds = dataset.get_dataset()
        batch_positions, batch_targets = next(iter(train_ds.as_numpy_iterator()))
        assert batch_positions.shape == (2, BOARD_SIZE, BOARD_SIZE, NUM_PLANES), (
            f"shape position incorrecte: {batch_positions.shape}"
        )
        assert batch_positions.dtype == np.float32
        assert batch_targets[POLICY_KEY].shape == (2,)
        assert batch_targets[POLICY_KEY].dtype in (np.int32, np.int64)
        assert batch_targets[VALUE_KEY].shape == (2,)
        assert batch_targets[VALUE_KEY].dtype == np.float32

        assert val_ds is not None
        val_batch_positions, _ = next(iter(val_ds.as_numpy_iterator()))
        assert val_batch_positions.shape == (2, BOARD_SIZE, BOARD_SIZE, NUM_PLANES)

        print(f"OK - ChessPolicyValueDataset : {total} exemples -> 3 chunks train / 2 chunks val "
              f"(disjoints), shapes/dtypes de batch corrects")


def test_compute_chess_policy_value_loss_exact_values():
    # Valeurs construites a la main - policy uniforme (logits=0) sur 10 classes ->
    # cross-entropy = log(10). value_pred/value_targets connus -> MSE exact.
    outputs = {
        POLICY_KEY: jnp.zeros((3, 10)),
        VALUE_KEY: jnp.array([0.5, -0.2, 0.0]),
    }
    targets = {
        POLICY_KEY: jnp.array([0, 1, 2]),
        VALUE_KEY: jnp.array([1.0, -1.0, 0.0]),
    }
    expected_policy_loss = float(jnp.log(10.0))
    expected_value_loss = ((0.5 - 1.0) ** 2 + (-0.2 - (-1.0)) ** 2 + (0.0 - 0.0) ** 2) / 3.0
    expected_combined = expected_policy_loss * 1.0 + expected_value_loss * 1.0

    combined = float(compute_chess_policy_value_loss(outputs, targets, policy_weight=1.0, value_weight=1.0))
    assert abs(combined - expected_combined) < 1e-5, (
        f"attendu {expected_combined:.6f}, obtenu {combined:.6f}"
    )

    # Poids non-egaux : verifie que la ponderation s'applique correctement.
    weighted = float(compute_chess_policy_value_loss(outputs, targets, policy_weight=2.0, value_weight=0.5))
    expected_weighted = expected_policy_loss * 2.0 + expected_value_loss * 0.5
    assert abs(weighted - expected_weighted) < 1e-5, f"attendu {expected_weighted:.6f}, obtenu {weighted:.6f}"

    print(f"OK - compute_chess_policy_value_loss : valeurs exactes verifiees "
          f"(policy_loss=log(10)={expected_policy_loss:.4f}, value_loss={expected_value_loss:.4f}, "
          f"poids egaux et non-egaux)")


def test_end_to_end_strategy_with_real_model():
    # Modele reel (Story 9.2) + strategie reelle (Story 9.3) sur un batch reel du
    # dataset de test (Story 9.1) - le chemin complet preprocess_batch -> forward ->
    # compute_loss -> compute_metrics, valeurs finies.
    with tempfile.TemporaryDirectory() as tmpdir:
        output_prefix, _ = _build_test_dataset(tmpdir)
        dataset = ChessPolicyValueDataset(output_prefix, batch_size=4, val_split=0.4)
        train_ds, val_ds = dataset.get_dataset()
        batch_positions, batch_targets = next(iter(train_ds.as_numpy_iterator()))

        model = get_model("chess_cnn_attention_policy_value", num_classes=NUM_MOVES, dropout_rate=0.1)
        rng = jax.random.PRNGKey(0)
        variables = model.init({"params": rng, "dropout": rng}, jnp.asarray(batch_positions), training=True)

        strategy = ChessPolicyValueStrategy(loss_params={"policy_weight": 1.0, "value_weight": 1.0})

        positions, targets, use_onehot = strategy.preprocess_batch(
            jnp.asarray(batch_positions), batch_targets, is_training=True
        )
        assert use_onehot is False

        outputs, _ = model.apply(
            variables, positions, training=True, mutable=["batch_stats"], rngs={"dropout": jax.random.PRNGKey(1)}
        )

        loss = strategy.compute_loss(outputs, targets, use_onehot_labels=False)
        metric = strategy.compute_metrics(outputs, targets)

        assert jnp.isfinite(loss), f"loss non finie: {loss}"
        assert jnp.isfinite(metric), f"metrique non finie: {metric}"
        assert 0.0 <= float(metric) <= 1.0, f"policy accuracy hors [0,1]: {metric}"

        print(f"OK - chaine complete preprocess_batch->forward->compute_loss->compute_metrics : "
              f"loss={float(loss):.4f}, PolicyAccuracy={float(metric):.4f}")

        # generate_reports (AC3) - le corps est enveloppe dans un try/except (meme
        # convention que les autres strategies), donc "ne leve pas d'exception" est
        # toujours vrai meme si le calcul interne est casse : on capture stdout et on
        # verifie le CONTENU affiche (policy_loss/value_loss), pas juste l'absence d'erreur.
        class _FakeState:
            def __init__(self, params, batch_stats, apply_fn):
                self.params = params
                self.batch_stats = batch_stats
                self.apply_fn = apply_fn

        final_state = _FakeState(variables["params"], variables.get("batch_stats", {}), model.apply)

        captured = io.StringIO()
        with contextlib.redirect_stdout(captured):
            strategy.generate_reports(val_ds, final_state, model, config={})
        output = captured.getvalue()

        assert "policy_loss=" in output, f"detail policy_loss absent de la sortie de generate_reports: {output!r}"
        assert "value_loss=" in output, f"detail value_loss absent de la sortie de generate_reports: {output!r}"
        assert "Erreur" not in output, f"generate_reports a rencontre une erreur interne (try/except) : {output!r}"
        print(f"OK - generate_reports() affiche reellement le detail policy_loss/value_loss (AC3) : {output.strip()}")


def test_dataset_config_chess_entry():
    assert "CHESS" in DATASET_CONFIGS, "entree CHESS absente de DATASET_CONFIGS"
    config = get_dataset_config("CHESS")
    assert config["task_type"] == "chess_policy_value"
    assert config["num_classes"] == NUM_MOVES
    assert config["model_name"] == "chess_cnn_attention_policy_value"
    # input_shape PRESENT, image_size ABSENT (2026-07-30, remplace l'ancien couple
    # image_size+num_channels cote Trainer - voir dataset_configs.py::validate_config) :
    # Trainer.create_train_state le lit sans garde pour construire le dummy_input d'init.
    # num_channels reste present separement (role distinct : ChessPolicyValueDataset,
    # data_management.py, en a besoin pour la forme des tenseurs de position).
    assert "image_size" not in config, "CHESS ne devrait plus avoir image_size (retire 2026-07-30)"
    assert config["input_shape"] == (8, 8, NUM_PLANES)
    assert config["num_channels"] == NUM_PLANES
    assert "class_names" not in config, "CHESS ne devrait pas avoir class_names"
    # dropout_rate niche sous tpu/gpu (pas top-level) - bug trouve en code review
    # (main.py:106 le lit via backend_config, pas au niveau racine de la config).
    assert "dropout_rate" not in config, "dropout_rate ne devrait pas etre au niveau racine de CHESS"
    assert config["tpu"]["dropout_rate"] == 0.1
    assert config["gpu"]["dropout_rate"] == 0.1

    assert validate_config("CHESS", config) is True, "validate_config a rejete l'entree CHESS"

    # Non-regression : toutes les configs existantes passent toujours validate_config
    # apres le fix de la liste 'required' (image_size retire).
    for name, cfg in DATASET_CONFIGS.items():
        assert validate_config(name, cfg) is True, f"validate_config rejette desormais {name} (regression)"

    # Chemin negatif : validate_config doit toujours rejeter une config a laquelle il
    # manque un parametre reellement requis (num_classes/model_name) - la fonction reste
    # une vraie validation, pas une coquille vide apres le retrait de image_size.
    incomplete_config = {k: v for k, v in config.items() if k not in ("num_classes", "model_name")}
    assert validate_config("CHESS_INCOMPLETE_TEST", incomplete_config) is False, (
        "validate_config aurait du rejeter une config sans num_classes/model_name"
    )

    print(f"OK - entree CHESS presente/valide (input_shape/num_channels presents, image_size/class_names "
          f"absents, dropout_rate niche sous tpu/gpu), toutes les {len(DATASET_CONFIGS)} configs passent "
          f"validate_config (aucune regression), chemin negatif (config incomplete rejetee) verifie")


# test_trainer_change_is_the_single_authorized_deviation (Story 9.3/9.4) retire le
# 2026-07-30 : verifiait que trainer.py ne portait qu'un seul ecart autorise a AC2
# (Epic 9), diffe contre le commit pre-epic. Premisse obsolete depuis le refactor
# "input_shape" (voir dataset_configs.py::validate_config, meme date) - un changement
# plus large et deliberement voulu de trainer.py, plus une contrainte AC2 d'epic close.
# Couverture desormais assuree autrement : test_trainer_create_train_state_for_chess
# ci-dessous exerce le vrai chemin Trainer avec la config CHESS reelle, et le nombre de
# parametres attendu (382017) prouve indirectement que input_shape encode bien 29 canaux
# (une mauvaise valeur donnerait un premier conv de taille differente).


def test_trainer_create_train_state_for_chess():
    # Exerce le VRAI chemin Trainer (pas une reimplementation) avec la VRAIE config CHESS
    # (dataset_configs.py) - c'est precisement ce chemin que les bugs class_names/
    # dropout_rate/image_size/num_channels trouves en code review avaient casse, et
    # qu'aucun test precedent de cette story n'exercait. Construit Trainer + appelle
    # create_train_state (init du modele), sans lancer d'epoch d'entrainement.
    config = get_dataset_config("CHESS")
    # "gpu" fixe (pas jax.default_backend()) : ce test importe data_management (donc
    # tensorflow, qui manipule CUDA_VISIBLE_DEVICES pour eviter un conflit TF/JAX - voir
    # data_management.py, tete de fichier) apres que JAX ait deja ete importe ailleurs
    # dans ce script - l'ordre exact peut faire fluctuer default_backend() en cours de
    # process. La config CHESS n'a de toute facon que "tpu"/"gpu" (comme toutes les
    # configs du projet, cf. validate_config) - ce test valide l'integration CHESS, pas
    # la detection JAX elle-meme (hors scope, deja geree par main.py).
    backend = "gpu"
    backend_config = config[backend]

    model = get_model(config["model_name"], num_classes=config["num_classes"],
                       dropout_rate=backend_config["dropout_rate"])
    strategy = ChessPolicyValueStrategy(loss_params=config["loss_params"])
    trainer = Trainer(model, config, backend, strategy)

    # "input_shape" (2026-07-30) remplace l'ancien attribut Trainer.num_channels (retire) -
    # verifie directement la cle de config que create_train_state() consomme desormais.
    assert config["input_shape"] == (8, 8, NUM_PLANES), (
        f"config['input_shape'] attendu (8, 8, {NUM_PLANES}), obtenu {config['input_shape']}"
    )
    assert trainer.class_names == [], "Trainer.class_names attendu [] (CHESS n'en a pas, fallback .get())"

    state = trainer.create_train_state(jax.random.PRNGKey(0))
    n_params = sum(p.size for p in jax.tree_util.tree_leaves(state.params))
    assert n_params == 382017, f"nombre de parametres inattendu : {n_params} (attendu 382017, Story 9.2)"

    print(f"OK - Trainer(config=CHESS).create_train_state() reussit via le vrai chemin main.py "
          f"(input_shape={config['input_shape']}, {n_params} parametres, coherent avec Story 9.2)")


def test_real_chunks_compatibility_if_present():
    # Optionnel (voir Dev Notes/Portee exacte : pas d'entrainement complet ici) - si les
    # 139 vrais chunks d'Aymeric sont presents, verifie qu'UN seul se charge correctement
    # avec le schema attendu, sans lancer d'entrainement. Chemin lu depuis DATASET_CONFIGS
    # (pas un litteral duplique ici) - reste a jour si le chemin change un jour.
    real_prefix = DATASET_CONFIGS["CHESS"]["output_prefix"]
    real_chunks = sorted(glob.glob(f"{real_prefix}_chunk*.npz"))
    if not real_chunks:
        print("SKIP - chunks reels non trouves (pas bloquant, test optionnel)")
        return

    with np.load(real_chunks[0]) as data:
        assert set(data.files) == {POSITION_KEY, POLICY_KEY, VALUE_KEY}
        assert data[POSITION_KEY].shape[1:] == (BOARD_SIZE, BOARD_SIZE, NUM_PLANES)
    print(f"OK - {len(real_chunks)} chunk(s) reel(s) trouve(s), premier chunk ({real_chunks[0]}) "
          f"compatible avec le schema attendu")


if __name__ == "__main__":
    test_chess_policy_value_dataset_loading()
    test_compute_chess_policy_value_loss_exact_values()
    test_end_to_end_strategy_with_real_model()
    test_dataset_config_chess_entry()
    test_trainer_create_train_state_for_chess()
    test_real_chunks_compatibility_if_present()
    print("\nTous les tests de ChessPolicyValueStrategy/ChessPolicyValueDataset sont passes.")
