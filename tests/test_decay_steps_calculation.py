"""
Test de validation pour _resolve_lr_schedule_steps (trainer.py, 2026-08-21) :
priorite config-explicite/auto-calcul pour decay_steps, division par accum_steps
en steps d'OPTIMISEUR (ceil, pas floor - train_epoch flush aussi un groupe
incomplet en fin d'epoch), drop_remainder selon task_type, et ValueError
explicite si ni valeur explicite ni chunks detectables ou si le resultat serait
degenere. Remplace l'ancien mecanisme qui ecrasait silencieusement toute valeur
en dur des que des chunks etaient trouves, et ignorait accum_steps (bug confirme
en pratique sur FIGHTERJET_CLASSIFICATION : 6419 calcule au lieu de 6000
configure). Formule floor->ceil corrigee suite revue adversariale
(review_loop_iteration 1, voir Spec Change Log de
_bmad-output/implementation-artifacts/spec-decay-steps-generalization.md).

Script autonome - meme convention que les autres tests de ce projet (pas de
framework de test formel impose). Executer directement :
    python tests/test_decay_steps_calculation.py
"""

import math
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trainer import _resolve_lr_schedule_steps, _count_real_train_samples
from dataset_configs import DATASET_CONFIGS

# Datasets sans decay_steps explicite ET sans chunks sur disque aujourd'hui -
# etat connu et accepte (Aymeric, 2026-08-21) : ValueError attendue a
# l'entrainement plutot qu'un repli silencieux, pas une regression a corriger.
KNOWN_UNCOVERED_DATASETS = {"JAX_KEPLER"}


def test_explicit_decay_steps_wins_never_recomputed():
    """AC : decay_steps explicite en config -> utilise tel quel, meme si des
    echantillons reels (qui donneraient un resultat different) sont fournis."""
    warmup, decay = _resolve_lr_schedule_steps(
        backend_config={"decay_steps": 6000, "warmup_steps": 1200},
        task_type="classification",
        real_train_samples=117457,
        micro_batch_size=128,
        accum_steps=4,
        decay_epochs=7,
        dataset_name="FIGHTERJET_CLASSIFICATION",
        backend="tpu",
    )
    assert decay == 6000, decay
    assert warmup == 1200, warmup
    print("OK - decay_steps explicite jamais recalcule")


def test_explicit_decay_steps_zero_or_negative_rejected():
    """Une valeur explicite <= 0 est une config invalide, pas une volonte
    legitime de l'utilisateur - rejetee explicitement plutot que transmise
    telle quelle a optax."""
    for bad_value in (0, -5):
        try:
            _resolve_lr_schedule_steps(
                backend_config={"decay_steps": bad_value}, task_type="classification",
                real_train_samples=1000, micro_batch_size=128, accum_steps=1,
                decay_epochs=1, dataset_name="X", backend="gpu",
            )
            raise AssertionError(f"ValueError attendue pour decay_steps={bad_value}")
        except ValueError as e:
            assert "X" in str(e), str(e)
    print("OK - decay_steps explicite <= 0 rejete")


def test_auto_divides_by_accum_steps_using_ceil_not_floor():
    """AC : auto-calcul divise par accum_steps en arrondissant AU PLAFOND (pas
    floor) - reproduit le cas reel mesure sur FIGHTERJET_CLASSIFICATION
    (117457 echantillons, batch=128 -> 918 steps/epoch (ceil, classification),
    accum_steps=4 -> ceil(918/4)=230, pas floor=229 ; decay_epochs=7 -> 1610,
    pas 1603 ni 6419)."""
    warmup, decay = _resolve_lr_schedule_steps(
        backend_config={"warmup_steps": 1200},
        task_type="classification",
        real_train_samples=117457,
        micro_batch_size=128,
        accum_steps=4,
        decay_epochs=7,
        dataset_name="FIGHTERJET_CLASSIFICATION",
        backend="tpu",
    )
    assert decay == 1610, decay
    print(f"OK - decay_steps auto = {decay} (ceil, pas 1603 floor ni 6419 ancien bug)")


def test_ceil_vs_floor_explicit_minimal_case():
    """Cas minimal rond pour isoler sans ambiguite le comportement ceil :
    steps_per_epoch=10 (floor, task_type='detection'), accum_steps=3 ->
    ceil(10/3)=4, PAS floor=3 ; decay_epochs=2 -> 8, pas 6."""
    _, decay = _resolve_lr_schedule_steps(
        backend_config={}, task_type="detection", real_train_samples=1000,
        micro_batch_size=100, accum_steps=3, decay_epochs=2,
        dataset_name="X", backend="gpu",
    )
    assert decay == 8, decay
    print("OK - ceil(10/3)=4 confirme sur un cas minimal (pas floor=3)")


def test_auto_accum_steps_one_unaffected_by_ceil_change():
    """accum_steps=1 -> ceil(n/1)=n, identique au floor, comportement inchange
    (non-regression pour tous les domaines a accum_steps=1)."""
    warmup, decay = _resolve_lr_schedule_steps(
        backend_config={},
        task_type="detection_centernet",
        real_train_samples=127000,
        micro_batch_size=128,
        accum_steps=1,
        decay_epochs=15,
        dataset_name="JAX_DETECTOR",
        backend="tpu",
    )
    assert decay == (127000 // 128) * 15, decay
    print("OK - accum_steps=1 : ceil et floor coincident, calcul inchange")


def test_drop_remainder_false_for_classification_and_kepler():
    """classification/kepler utilisent drop_remainder=False (ChunkManager,
    data_management.py:224) -> ceil, pas floor."""
    _, decay_clf = _resolve_lr_schedule_steps(
        backend_config={}, task_type="classification", real_train_samples=1000,
        micro_batch_size=128, accum_steps=1, decay_epochs=1,
        dataset_name="X", backend="gpu",
    )
    _, decay_kepler = _resolve_lr_schedule_steps(
        backend_config={}, task_type="kepler", real_train_samples=1000,
        micro_batch_size=128, accum_steps=1, decay_epochs=1,
        dataset_name="JAX_KEPLER", backend="gpu",
    )
    expected = math.ceil(1000 / 128)
    assert decay_clf == expected, decay_clf
    assert decay_kepler == expected, decay_kepler
    print(f"OK - drop_remainder=False (ceil) pour classification/kepler ({expected})")


def test_drop_remainder_true_for_other_task_types():
    """Tout task_type hors classification/kepler utilise drop_remainder=True
    (floor pour steps_per_epoch) - detection, detection_centernet, et toute la
    famille chess_* (data_management.py: DetectionDataset, CenterNetDetectionDataset,
    ChessPolicyValueDataset, ChessLegalMovesDataset, ChessMoveTokenDataset,
    ChessTokenCandidateDataset, ChessTokenOneMoveDataset - toutes drop_remainder=True)."""
    for task_type, name in [
        ("chess_legal_moves", "CHESS_LEGAL_MOVES"),
        ("chess_move_token", "CHESS_MOVE_TOKEN"),
        ("chess_token", "CHESS_TOKEN"),
        ("chess_token_1_move", "CHESS_TOKEN_1_MOVE"),
    ]:
        _, decay = _resolve_lr_schedule_steps(
            backend_config={}, task_type=task_type, real_train_samples=1000,
            micro_batch_size=128, accum_steps=1, decay_epochs=1,
            dataset_name=name, backend="gpu",
        )
        assert decay == 1000 // 128, (task_type, decay)
    print("OK - drop_remainder=True (floor) sur toute la famille chess_* + detection")


def test_raises_valueerror_when_no_explicit_value_and_no_chunks():
    """AC : ni decay_steps explicite ni chunks detectables -> ValueError
    explicite citant dataset/backend, jamais de repli silencieux sur 6000."""
    try:
        _resolve_lr_schedule_steps(
            backend_config={},
            task_type="kepler",
            real_train_samples=None,
            micro_batch_size=32,
            accum_steps=4,
            decay_epochs=30,
            dataset_name="JAX_KEPLER",
            backend="gpu",
        )
        raise AssertionError("ValueError attendue, aucune exception levee")
    except ValueError as e:
        assert "JAX_KEPLER" in str(e), str(e)
        assert "gpu" in str(e), str(e)
        print(f"OK - ValueError explicite levee ({e})")


def test_raises_valueerror_on_invalid_accum_steps():
    """Defense en profondeur : accum_steps<=0 leve une ValueError claire plutot
    qu'une ZeroDivisionError - deja empeche en pratique par validate_config()
    pour toute config reelle (dataset_configs.py), mais cette fonction est
    appelable independamment (tests, futurs appelants)."""
    for bad_accum in (0, -1):
        try:
            _resolve_lr_schedule_steps(
                backend_config={}, task_type="detection", real_train_samples=1000,
                micro_batch_size=100, accum_steps=bad_accum, decay_epochs=1,
                dataset_name="X", backend="gpu",
            )
            raise AssertionError(f"ValueError attendue pour accum_steps={bad_accum}")
        except ValueError as e:
            assert "X" in str(e), str(e)
    print("OK - accum_steps invalide rejete explicitement")


def test_raises_valueerror_on_degenerate_zero_steps():
    """Volume de donnees trop faible pour produire un seul step d'optimiseur
    (steps_per_epoch tombe a 0 par floor) -> ValueError explicite plutot qu'un
    decay_steps=0 silencieux transmis a optax."""
    try:
        _resolve_lr_schedule_steps(
            backend_config={}, task_type="detection", real_train_samples=5,
            micro_batch_size=1000, accum_steps=1, decay_epochs=10,
            dataset_name="X", backend="gpu",
        )
        raise AssertionError("ValueError attendue, aucune exception levee")
    except ValueError as e:
        assert "X" in str(e), str(e)
        print(f"OK - ValueError explicite sur resultat degenere ({e})")


def test_all_dataset_configs_have_explicit_or_computable_decay_steps():
    """Audit systematique : pour chaque config reelle (tpu/gpu), decay_steps
    doit etre soit explicite, soit calculable depuis des chunks reels sur
    disque - sauf les datasets connus et acceptes comme non couverts
    aujourd'hui (KNOWN_UNCOVERED_DATASETS). Aurait detecte le trou JAX_KEPLER
    avant qu'il ne soit trouve en revue adversariale (Blind Hunter + Edge Case
    Hunter, independamment, 2026-08-21)."""
    gaps = []
    for name, cfg in DATASET_CONFIGS.items():
        real_samples = _count_real_train_samples(cfg.get("output_prefix", ""))
        for backend in ("tpu", "gpu"):
            if backend not in cfg:
                continue
            has_explicit = cfg[backend].get("decay_steps") is not None
            has_chunks = bool(real_samples)
            if not has_explicit and not has_chunks:
                gaps.append(f"{name}/{backend}")
    unexpected_gaps = [g for g in gaps if g.split("/")[0] not in KNOWN_UNCOVERED_DATASETS]
    assert not unexpected_gaps, f"Datasets sans decay_steps explicite ni chunks detectables : {unexpected_gaps}"
    known = [g for g in gaps if g.split("/")[0] in KNOWN_UNCOVERED_DATASETS]
    print(f"OK - aucun trou de couverture inattendu (connus et acceptes : {known})")


if __name__ == "__main__":
    test_explicit_decay_steps_wins_never_recomputed()
    test_explicit_decay_steps_zero_or_negative_rejected()
    test_auto_divides_by_accum_steps_using_ceil_not_floor()
    test_ceil_vs_floor_explicit_minimal_case()
    test_auto_accum_steps_one_unaffected_by_ceil_change()
    test_drop_remainder_false_for_classification_and_kepler()
    test_drop_remainder_true_for_other_task_types()
    test_raises_valueerror_when_no_explicit_value_and_no_chunks()
    test_raises_valueerror_on_invalid_accum_steps()
    test_raises_valueerror_on_degenerate_zero_steps()
    test_all_dataset_configs_have_explicit_or_computable_decay_steps()
