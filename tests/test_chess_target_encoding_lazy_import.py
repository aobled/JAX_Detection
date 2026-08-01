"""
Non-regression : chess_target_encoding.py (et les modules qui importent des constantes
depuis lui au top-level : model_library, task_strategies, dataset_configs,
loss_functions) doivent rester importables sans le package tiers `chess` (python-chess)
installe - un run CIFAR10/FIGHTERJET_*/KEPLER ne doit jamais exiger python-chess.
Protege contre la reintroduction d'un `import chess` module-level (voir docstring de
chess_target_encoding.py, section "Dependance python-chess : optionnelle au chargement
du module").

Execution: python3 tests/test_chess_target_encoding_lazy_import.py
"""

import os
import subprocess
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Execute dans un sous-process dedie (pas dans le process du test lui-meme) : simuler
# `chess` absent via sys.modules['chess'] = None doit s'appliquer AVANT tout import de
# chess_target_encoding, sans polluer le cache de modules du runner de tests.
_PROBE = """
import sys
sys.modules['chess'] = None  # tout `import chess` reel levera ModuleNotFoundError

import chess_target_encoding as cte
assert cte.NUM_PLANES == 29, f"NUM_PLANES attendu 29, obtenu {cte.NUM_PLANES}"
assert cte.NUM_MOVES == 4672, f"NUM_MOVES attendu 4672, obtenu {cte.NUM_MOVES}"

import model_library
import task_strategies
import dataset_configs
import loss_functions

try:
    cte.encode_position(None, [])
    raise SystemExit("ERREUR: encode_position aurait du lever ModuleNotFoundError")
except ModuleNotFoundError:
    pass

print("OK")
"""


def test_modules_importable_without_chess():
    result = subprocess.run(
        [sys.executable, "-c", _PROBE],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, (
        f"import sans `chess` a echoue (returncode={result.returncode})\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    assert "OK" in result.stdout, f"sortie inattendue:\n{result.stdout}\nstderr:\n{result.stderr}"
    print(
        "OK - chess_target_encoding/model_library/task_strategies/dataset_configs/"
        "loss_functions importables sans chess installe"
    )


if __name__ == "__main__":
    test_modules_importable_without_chess()
    print("Tous les tests sont passes.")
