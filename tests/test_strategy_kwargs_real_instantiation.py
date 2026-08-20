"""
Test de validation pour Story 12.2 (AD-21), AC #7 : verifie par instanciation
REELLE (pas mockee) que chaque entree DATASET_CONFIGS reelle produit une
Strategy dont les attributs correspondent exactement a ce que produirait
l'ancien code main.py (config.get(champ, defaut) - le defaut etant celui du
CONSTRUCTEUR de la classe cible, jamais un defaut duplique cote main.py,
AC #4) - pas juste l'absence d'exception.

Approche generique (introspection de signature) plutot qu'une liste de
valeurs attendues transcrites a la main pour chacune des 11 configs reelles -
moins sujette a l'erreur de transcription, et couvre gratuitement les
configs qui partagent un task_type (ex. CHESS_NO_HISTORY et
CHESS_SEARCH_TEACHER, toutes deux chess_policy_value).

Script autonome - meme convention que les autres tests de ce projet. Executer
directement :
    python tests/test_strategy_kwargs_real_instantiation.py
"""

import sys
import os
import inspect

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from task_strategies import STRATEGIES, STRATEGY_FORWARDED_CONFIG_KEYS
from model_library import build_kwargs_from_config
from dataset_configs import DATASET_CONFIGS


def test_all_real_configs_produce_strategy_matching_config_or_class_default():
    checked_task_types = set()
    for dataset_name, config in DATASET_CONFIGS.items():
        task_type = config.get("task_type", "classification")
        strategy_cls = STRATEGIES.get(task_type)
        assert strategy_cls is not None, f"{dataset_name}: task_type '{task_type}' absent de STRATEGIES"

        strategy_kwargs, forwarded = build_kwargs_from_config(
            strategy_cls,
            config,
            config_keys=STRATEGY_FORWARDED_CONFIG_KEYS.get(task_type, ()),
            num_classes=config.get("num_classes"),
        )
        strategy = strategy_cls(**strategy_kwargs)

        sig = inspect.signature(strategy_cls.__init__)
        for param_name, param in sig.parameters.items():
            if param_name == "self":
                continue
            expected = config.get(param_name, param.default)
            if param_name == "loss_params":
                expected = expected or {}
            actual = getattr(strategy, param_name)
            assert actual == expected, (
                f"{dataset_name} ({task_type}) : attribut '{param_name}' attendu {expected!r}, obtenu {actual!r}"
            )
        checked_task_types.add(task_type)

    assert checked_task_types == set(STRATEGIES.keys()), (
        f"task_type non couverts par au moins une config reelle : {set(STRATEGIES.keys()) - checked_task_types}"
    )
    print(f"OK - {len(DATASET_CONFIGS)} configs reelles, {len(checked_task_types)} task_type : "
          f"kwargs Strategy reellement corrects (config ou defaut du constructeur, jamais un defaut main.py)")


def test_unknown_task_type_not_in_strategies():
    """AC #2 : precondition reelle du garde-fou main.py (STRATEGIES.get(task_type)
    is None -> raise ValueError(...), main.py:165-167) - un task_type absent de
    STRATEGIES doit retourner None, jamais lever un KeyError ni retourner une
    entree inattendue. Le raise lui-meme (main.py) est une ligne triviale, non
    dupliquee ici (une reimplementation locale du if/raise ne testerait que sa
    propre copie, pas le vrai code - revue de code, Story 12.2)."""
    assert STRATEGIES.get("nope") is None
    assert STRATEGIES.get("") is None
    print("OK - task_type inconnu absent de STRATEGIES (garantit le ValueError de main.py, jamais un KeyError)")


if __name__ == "__main__":
    test_all_real_configs_produce_strategy_matching_config_or_class_default()
    test_unknown_task_type_not_in_strategies()
