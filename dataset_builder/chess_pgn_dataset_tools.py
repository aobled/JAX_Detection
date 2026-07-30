"""
Extraction du dataset echecs (position/policy/value) depuis des archives PGN
(Story 9.1, FR1/FR3, AD-25). Producteur du contrat defini par chess_target_encoding.py
(AD-18) - importe encode_position/move_to_index/POSITION_KEY/POLICY_KEY/VALUE_KEY,
ne reimplemente jamais ce format localement.

Source des archives : pgnmentor.com/files.html#players (par grand joueur, plusieurs
parties concatenees dans un meme fichier .pgn, victoires/defaites/nulles confondues).

Ne depend PAS de dataset_configs.py (aucune entree CHESS n'existe encore - Story 9.3
la cree et branchera cet outil dessus, meme principe que jax_detector_dataset_tools.py
lit sa config JAX_DETECTOR). build_chess_dataset() prend ses parametres explicitement.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ctypes
import gc
import glob

import numpy as np
import chess
import chess.pgn

from chess_target_encoding import (
    encode_position,
    move_to_index,
    POSITION_KEY,
    POLICY_KEY,
    VALUE_KEY,
    HISTORY_LENGTH,
)

_KNOWN_RESULTS = ("1-0", "0-1", "1/2-1/2")

# Frequence (en nombre de parties) du retour de progression pendant build_chess_dataset -
# permet de distinguer "en cours" de "bloque" sur une grosse archive a fichier unique.
_PROGRESS_EVERY_N_GAMES = 1000

try:
    _LIBC = ctypes.CDLL("libc.so.6")
except OSError:
    _LIBC = None  # non-Linux, garde defensive (meme pattern que jax_detector_dataset_tools.py)


def _release_freed_memory_to_os():
    gc.collect()
    if _LIBC is not None:
        _LIBC.malloc_trim(0)


def _value_for_mover(result: str, mover_is_white: bool) -> float:
    """
    Resultat de partie (+1/0/-1) du point de vue du joueur au trait a une position
    donnee. AD-25 : calcule et fige une seule fois ici (producteur) - jamais recalcule
    ni re-derive cote consommateur (data_management.py/ChessPolicyValueStrategy, Story 9.3).

    Un resultat "*" (partie inachevee/inconnue, en-tete PGN non standard) est traite
    comme une nulle (value=0) - decision de cette story, pas un cas documente par le PRD ;
    ces parties restent rares dans les archives pgnmentor (parties de tournois terminees).
    """
    if result == "1-0":
        return 1.0 if mover_is_white else -1.0
    if result == "0-1":
        return -1.0 if mover_is_white else 1.0
    return 0.0  # "1/2-1/2" (nulle) ou "*" (inconnu)


def _iter_game_examples(game: "chess.pgn.Game", include_history: bool = True):
    """
    Genere (position_planes, policy_index, value) pour chaque demi-coup de `game`,
    Blancs et Noirs confondus, sans filtrage par resultat de partie (FR3) - y compris
    les toutes premieres positions (historique < 5 demi-coups, padding a zero gere par
    encode_position).

    include_history: forwarde a encode_position (2026-07-29, test d'ablation historique,
    voir deferred-work.md) - defaut True, comportement inchange pour les appelants
    existants.
    """
    result = game.headers.get("Result", "*")
    board = game.board()
    history = []
    for move in game.mainline_moves():
        mover_is_white = board.turn == chess.WHITE
        value = _value_for_mover(result, mover_is_white)
        # Pre-decoupe aux HISTORY_LENGTH derniers coups avant l'appel : encode_position
        # ne consomme jamais que les 5 derniers, mais `history` grandit sans borne sur
        # toute la partie - repasser la liste entiere copierait un nombre croissant
        # d'elements a chaque demi-coup (O(n) par appel, O(n^2) sur la partie entiere)
        # pour un resultat identique.
        planes = encode_position(board, history[-HISTORY_LENGTH:], include_history=include_history)
        policy_idx = move_to_index(move, board)
        yield planes, policy_idx, value

        history.append(move)
        board.push(move)


def _save_chunk(output_prefix: str, chunk_idx: int, positions: list, policies: list, values: list) -> None:
    n = len(positions)

    # Empiler puis vider immediatement chaque liste source (meme discipline memoire que
    # jax_detector_dataset_tools.py::_save_chunk_v2, reutilisee par coherence). Les
    # tenseurs position (8,8,29 float32 ~7.4 Ko/exemple) sont un ordre de grandeur plus
    # legers qu'une image - non benchmarke a l'echelle d'une vraie archive pgnmentor
    # dans cette story, mais la marge par rapport au cas image (qui, lui, a deja cause
    # 2 crashs systeme sur ce projet a chunk_size trop eleve) est large.
    positions_np = np.array(positions, dtype=np.float32)  # (N, 8, 8, 29)
    positions.clear()
    policies_np = np.array(policies, dtype=np.int32)  # (N,)
    policies.clear()
    values_np = np.array(values, dtype=np.float32)  # (N,)
    values.clear()

    out_path = f"{output_prefix}_chunk{chunk_idx}.npz"
    np.savez_compressed(out_path, **{POSITION_KEY: positions_np, POLICY_KEY: policies_np, VALUE_KEY: values_np})
    print(f"[chunk {chunk_idx}] {n} positions -> {out_path}")

    del positions_np, policies_np, values_np
    _release_freed_memory_to_os()


def build_chess_dataset(pgn_paths, output_prefix: str, chunk_size: int = 5000, include_history: bool = True) -> int:
    """
    Construit le dataset echecs a partir d'une ou plusieurs archives PGN et l'ecrit en
    chunks .npz compresses (POSITION_KEY/POLICY_KEY/VALUE_KEY, source unique AD-18).

    Args:
        pgn_paths: chemin (str) ou liste de chemins vers des fichiers .pgn (chacun peut
            contenir plusieurs parties concatenees, format pgnmentor standard)
        output_prefix: prefixe des fichiers de sortie - chaque chunk ecrit
            "{output_prefix}_chunk{N}.npz"
        chunk_size: nombre d'exemples (positions) par chunk
        include_history: forwarde a chess_target_encoding.py::encode_position
            (2026-07-29, test d'ablation historique, voir deferred-work.md) - si False,
            les positions ecrites n'ont que NUM_POSITION_PLANES=19 canaux (pas
            NUM_PLANES=29). Defaut True, comportement inchange pour les appelants
            existants (ex. les 139 chunks deja generes pour la config CHESS).

    Returns: nombre total d'exemples (positions) produits, toutes archives/parties
        confondues.

    Aucun moteur d'echecs externe (Stockfish ou autre) n'est utilise a aucune etape
    (AD-25/NFR2) - seul python-chess sert a la legalite des coups et au rejeu PGN.
    """
    if isinstance(pgn_paths, str):
        pgn_paths = [pgn_paths]

    out_dir = os.path.dirname(output_prefix)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    chunk_idx = 0
    chunks_saved = 0
    total_examples = 0
    games_processed = 0
    games_skipped = 0
    games_with_parse_errors = 0
    games_unknown_result = 0
    positions, policies, values = [], [], []

    for pgn_path in pgn_paths:
        with open(pgn_path, encoding="utf-8", errors="replace") as pgn_file:
            while True:
                game = chess.pgn.read_game(pgn_file)
                if game is None:
                    break

                if game.errors:
                    games_with_parse_errors += 1
                    print(f"[avertissement] {pgn_path}: {len(game.errors)} erreur(s) de parsing sur une partie "
                          f"({game.headers.get('Event', '?')}) - coups recuperes traites du mieux possible")

                if game.headers.get("Result", "*") not in _KNOWN_RESULTS:
                    games_unknown_result += 1

                # Traite chaque partie dans une liste locale : si une erreur survient en
                # cours de rejeu (coup illegal, position incoherente), la partie entiere
                # est ecartee plutot que de laisser des exemples partiels contaminer le
                # dataset - une seule partie corrompue n'interrompt jamais l'import complet.
                try:
                    game_examples = list(_iter_game_examples(game, include_history=include_history))
                except (ValueError, chess.IllegalMoveError) as e:
                    games_skipped += 1
                    print(f"[avertissement] {pgn_path}: partie ignoree ({game.headers.get('Event', '?')}) - {e}")
                    continue

                for planes, policy_idx, value in game_examples:
                    positions.append(planes)
                    policies.append(policy_idx)
                    values.append(value)
                    total_examples += 1

                    if len(positions) >= chunk_size:
                        _save_chunk(output_prefix, chunk_idx, positions, policies, values)
                        chunk_idx += 1
                        chunks_saved += 1

                games_processed += 1
                if games_processed % _PROGRESS_EVERY_N_GAMES == 0:
                    print(f"... {games_processed} parties traitees, {total_examples} positions jusqu'ici")

    if positions:  # dernier chunk partiel
        _save_chunk(output_prefix, chunk_idx, positions, policies, values)
        chunks_saved += 1

    print(f"Dataset echecs : {total_examples} positions extraites depuis {len(pgn_paths)} archive(s) PGN, "
          f"{games_processed} partie(s) traitee(s), {chunks_saved} chunk(s)")
    if games_skipped:
        print(f"  {games_skipped} partie(s) ignoree(s) (erreur pendant le rejeu)")
    if games_with_parse_errors:
        print(f"  {games_with_parse_errors} partie(s) avec erreur(s) de parsing PGN (traitees du mieux possible)")
    if games_unknown_result:
        print(f"  {games_unknown_result} partie(s) a resultat inconnu (\"*\") - value repliee sur nulle (0.0)")

    return total_examples


if __name__ == "__main__":
    # Usage manuel autonome (regeneration/test hors pipeline complet) :
    #   python dataset_builder/chess_pgn_dataset_tools.py <archive.pgn> [<archive2.pgn> ...]
    # Sans argument (ex. "Run Python File" de VSCode) : traite tous les .pgn trouves
    # dans _DEFAULT_PGN_DIR (usage manuel de test d'Aymeric, pas un chemin de config).
    # output_prefix aligne sur dataset_configs.py::DATASET_CONFIGS["CHESS"]["output_prefix"]
    # (Story 9.3, branchement complet fait) - sous-dossier dedie, meme convention que
    # JAX_DETECTOR (chunks/jax_detector/jax_detector_targets_*.npz).
    _DEFAULT_PGN_DIR = "/home/aobled/Downloads/tmp_test_PGN"

    if len(sys.argv) >= 2:
        pgn_paths = sys.argv[1:]
    else:
        pgn_paths = sorted(glob.glob(os.path.join(_DEFAULT_PGN_DIR, "*.pgn")))
        if not pgn_paths:
            print(f"Aucun fichier .pgn trouve dans {_DEFAULT_PGN_DIR}")
            print("Usage: python chess_pgn_dataset_tools.py <archive.pgn> [<archive2.pgn> ...]")
            sys.exit(1)
        print(f"Aucun argument fourni - traitement de {len(pgn_paths)} fichier(s) .pgn trouve(s) dans {_DEFAULT_PGN_DIR}")

    build_chess_dataset(pgn_paths, output_prefix="/home/aobled/Documents/data/chunks/chess/chess")
