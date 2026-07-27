"""
Inference du modele echecs (checkpoint "export pur" de task_strategies.py::export_model,
cles 'params'/'batch_stats'/'config') pour chess_game.py - selection d'un coup joue par
le modele. S'appuie exclusivement sur chess_target_encoding.py (AD-18) pour
l'encodage/decodage, jamais de reimplementation locale de l'encodage ou de la legalite.

Hors perimetre de l'Epic 9 (PRD, Non-Goals : integration dans chess_game.py = banc de
test futur) - module d'exploration, pas trace par une story dediee.
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pickle

import numpy as np

from chess_target_encoding import encode_position, index_to_move, HISTORY_LENGTH, POLICY_KEY, VALUE_KEY
from model_library import get_model


def load_chess_model(checkpoint_path):
    """Charge un checkpoint 'export pur' et retourne (model, variables) prets pour
    model.apply(variables, batch, training=False)."""
    with open(checkpoint_path, "rb") as f:
        model_dict = pickle.load(f)

    config = model_dict["config"]
    model = get_model(config["model_name"], num_classes=config["num_classes"], dropout_rate=0.0)
    variables = {"params": model_dict["params"], "batch_stats": model_dict.get("batch_stats", {})}
    return model, variables


def select_model_move(model, variables, board):
    """
    Choisit un coup pour `board` (position au trait du joueur modele) : parcourt les
    index de la tete policy par score decroissant et retourne le premier qui correspond
    a un coup legal - index_to_move valide deja contre board.legal_moves (source unique
    de legalite, jamais reimplementee ici).

    Returns: (chess.Move, value) - value = sortie brute de la tete value (pas encore
    utilisee pour la selection, juste remontee pour affichage/debug eventuel).

    Leve ValueError si aucun index ne correspond a un coup legal (ne devrait arriver
    que si `board` n'a aucun coup legal - a l'appelant de verifier is_game_over() avant).
    """
    history = list(board.move_stack[-HISTORY_LENGTH:])
    planes = encode_position(board, history)
    batch = planes[None, ...]

    out = model.apply(variables, batch, training=False)
    policy_logits = np.asarray(out[POLICY_KEY][0])
    value = float(np.asarray(out[VALUE_KEY][0]))

    ranked_indices = np.argsort(policy_logits)[::-1]
    for idx in ranked_indices:
        try:
            move = index_to_move(int(idx), board)
            return move, value
        except ValueError:
            continue

    raise ValueError("Aucun coup legal trouve parmi les index de la tete policy (position sans coup legal ?)")
