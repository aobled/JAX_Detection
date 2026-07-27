"""
Test de validation pour dataset_builder/chess_pgn_dataset_tools.py (Story 9.1,
AC 1/6/8). Script autonome, meme convention que tests/test_chess_target_encoding.py.
Executer directement :
    python tests/test_chess_pgn_dataset_tools.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import glob
import inspect
import tempfile

import numpy as np

import dataset_builder.chess_pgn_dataset_tools as chess_pgn_dataset_tools
from dataset_builder.chess_pgn_dataset_tools import build_chess_dataset
from chess_target_encoding import POSITION_KEY, POLICY_KEY, VALUE_KEY, NUM_PLANES

_TEST_PGN = """[Event "Test1"]
[Result "1-0"]

1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6 1-0

[Event "Test2"]
[Result "0-1"]

1. d4 d5 2. c4 e6 0-1

[Event "Test3"]
[Result "1/2-1/2"]

1. e4 e5 1/2-1/2
"""
# 8 + 4 + 2 = 14 demi-coups au total sur les 3 parties


def test_no_external_chess_engine_dependency():
    # AC 6 / NFR2 : verification statique - le module ne doit ni importer chess.engine
    # ni invoquer l'API moteur de python-chess (SimpleEngine), a aucun endroit. On
    # verifie l'usage reel de l'API moteur, pas la simple presence du mot "stockfish"
    # (qui apparait legitimement dans les commentaires documentant CETTE contrainte).
    source = inspect.getsource(chess_pgn_dataset_tools)
    assert "import chess.engine" not in source, "chess.engine importe - viole NFR2 (aucun moteur externe)"
    assert "SimpleEngine" not in source, "chess.engine.SimpleEngine utilise - viole NFR2 (aucun moteur externe)"
    print("OK - aucune dependance a un moteur d'echecs externe (verification statique du source)")


def test_build_dataset_example_count_and_chunking():
    with tempfile.TemporaryDirectory() as tmpdir:
        pgn_path = os.path.join(tmpdir, "test_archive.pgn")
        with open(pgn_path, "w", encoding="utf-8") as f:
            f.write(_TEST_PGN)

        output_prefix = os.path.join(tmpdir, "chess_targets")
        # chunk_size=5 sur 14 exemples -> 3 chunks (5, 5, 4)
        total = build_chess_dataset(pgn_path, output_prefix, chunk_size=5)

        assert total == 14, f"attendu 14 exemples (8+4+2 demi-coups), obtenu {total}"

        chunk_files = sorted(glob.glob(f"{output_prefix}_chunk*.npz"))
        assert len(chunk_files) == 3, f"attendu 3 chunks (tailles 5,5,4), obtenu {len(chunk_files)}: {chunk_files}"

        all_positions, all_policies, all_values = [], [], []
        expected_sizes = [5, 5, 4]
        for path, expected_n in zip(chunk_files, expected_sizes):
            data = np.load(path)
            assert set(data.files) == {POSITION_KEY, POLICY_KEY, VALUE_KEY}, (
                f"cles inattendues dans {path}: {data.files}"
            )
            n = data[POSITION_KEY].shape[0]
            assert n == expected_n, f"{path}: attendu {expected_n} exemples, obtenu {n}"
            assert data[POSITION_KEY].shape == (n, 8, 8, NUM_PLANES), (
                f"{path}: shape position incorrecte {data[POSITION_KEY].shape}"
            )
            assert data[POLICY_KEY].shape == (n,), f"{path}: shape policy incorrecte {data[POLICY_KEY].shape}"
            assert data[VALUE_KEY].shape == (n,), f"{path}: shape value incorrecte {data[VALUE_KEY].shape}"
            all_positions.append(data[POSITION_KEY])
            all_policies.append(data[POLICY_KEY])
            all_values.append(data[VALUE_KEY])

        values = np.concatenate(all_values)
        # Partie 1 (1-0, 8 demi-coups) : values 1,-1,1,-1,1,-1,1,-1
        # Partie 2 (0-1, 4 demi-coups) : values -1,1,-1,1
        # Partie 3 (1/2-1/2, 2 demi-coups) : values 0,0
        expected_values = [1, -1, 1, -1, 1, -1, 1, -1, -1, 1, -1, 1, 0, 0]
        assert list(values) == [float(v) for v in expected_values], (
            f"values incorrectes: {list(values)} (attendu {expected_values})"
        )
        print(f"OK - 3 parties (1-0, 0-1, 1/2-1/2) -> 14 exemples, 3 chunks (5/5/4), cles/shapes/values correctes")


if __name__ == "__main__":
    test_no_external_chess_engine_dependency()
    test_build_dataset_example_count_and_chunking()
    print("\nTous les tests de dataset_builder/chess_pgn_dataset_tools.py sont passes.")
