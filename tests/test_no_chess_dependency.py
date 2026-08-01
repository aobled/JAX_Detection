"""
Non-regression : aucun module de ce repo ne doit dependre de `python-chess`, meme
indirectement - un run CIFAR10/FIGHTERJET_*/JAX_DETECTOR/KEPLER ne doit jamais exiger
`chess`. Remplace tests/test_chess_target_encoding_lazy_import.py (supprime avec
chess_target_encoding.py, cf. spec-chess-npz-boundary-cleanup) : garde-fou contre la
reintroduction d'un `import chess` dans model_library.py/task_strategies.py/
dataset_configs.py/loss_functions.py/data_management.py/main.py, desormais que la
generation du dataset echecs (et son besoin reel de `python-chess`) vit entierement
cote chess_ai.

Execution: python3 tests/test_no_chess_dependency.py
"""

import os
import subprocess
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Execute dans un sous-process dedie : simuler `chess` absent via
# sys.modules['chess'] = None doit s'appliquer AVANT tout import des modules testes,
# sans polluer le cache de modules du runner de tests.
_PROBE = """
import sys
sys.modules['chess'] = None  # tout `import chess` reel levera ModuleNotFoundError

import model_library
import task_strategies
import dataset_configs
import loss_functions
import data_management
import main

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
        "OK - model_library/task_strategies/dataset_configs/loss_functions/"
        "data_management/main importables sans chess installe"
    )


if __name__ == "__main__":
    test_modules_importable_without_chess()
    print("Tous les tests sont passes.")
