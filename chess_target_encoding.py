"""
Schema d'echange position/policy/value pour le domaine echecs (Epic 9, Story 9.1).

Module source unique de verite pour le format d'entree/sortie du domaine echecs -
miroir direct de detection_target_encoding.py (meme role, meme structure, AD-18
herite du spine parent). Consommateurs prevus :
  - dataset_builder/chess_pgn_dataset_tools.py (Story 9.1, producteur)
  - nouveau modele chess_cnn_attention_policy_value (Story 9.2, consommateur : dimensionne
    sa tete policy sur NUM_MOVES, importe la constante plutot que de la redefinir)
  - nouvelle TaskStrategy/chargeur data_management.py (Story 9.3, consommateur)

=== Contrat position (POSITION_KEY) ===

encode_position(board, move_history) -> np.ndarray, shape (8, 8, NUM_PLANES=29), float32.
Reperage (row, col) = (rank, file) - convention absolue, PAS de flip d'orientation
selon la couleur au trait (decision de cette story, voir Dev Notes de la story 9.1 -
plus simple pour cette premiere iteration ; un flip par couleur, qui unifierait les
motifs appris pour les deux couleurs, reste une piste d'amelioration future non
retenue ici, pas un oubli).

Plans 0-18 (position courante, 19 plans) :
    0-5   : pieces du joueur au trait (Roi, Dame, Tour, Fou, Cavalier, Pion), binaire
    6-11  : pieces de l'adversaire (memes 6 types, meme ordre)
    12    : trait - constante uniforme (1.0 si Blancs au trait, 0.0 si Noirs)
    13    : droit de roque court du joueur au trait
    14    : droit de roque long du joueur au trait
    15    : droit de roque court de l'adversaire
    16    : droit de roque long de l'adversaire
    17    : repetition - board.is_repetition(2), constante uniforme
    18    : coups legaux - case de DESTINATION des coups legaux du joueur au trait
            (simplification : n'encode pas la case source, ajustable si la Story 9.2
            montre qu'un signal plus fin aide le modele)

Plans 19-28 (historique des 5 derniers demi-coups, 2 plans/demi-coup, HISTORY_LENGTH=5) :
    Du plus ancien au plus recent. Chaque demi-coup = 1 plan case source + 1 plan case
    destination. Position avec moins de 5 demi-coups joues depuis le debut de la partie :
    les creneaux manquants sont des plans a zero (jamais une repetition de la position de
    depart, jamais une exclusion de la position du dataset - decision explicite, PRD FR2).

=== Contrat policy (POLICY_KEY) ===

Espace de coups fixe façon AlphaZero, NUM_MOVES = 4672 = 64 cases source x 73 types de
coup. move_to_index/index_to_move sont la source unique de cette conversion (AD-18) -
jamais reimplementees independamment cote Story 9.2/9.3. Voir docstrings de ces deux
fonctions pour le detail exact du schema (56 coups "dame" + 8 coups de cavalier + 9
sous-promotions).

=== Contrat value (VALUE_KEY) ===

Scalaire +1/0/-1, du point de vue du joueur au trait a la position consideree, nulles
incluses avec value=0. Calcule et fige une seule fois par le producteur
(dataset_builder/chess_pgn_dataset_tools.py) - jamais recalcule ou re-derive cote
consommateur (AD-25).

=== Persistance .npz ===

Cles canoniques POSITION_KEY/POLICY_KEY/VALUE_KEY. Pas de fonction save/load dediee ici
(contrairement a detection_target_encoding.py) : la persistance de ce domaine se fait
exclusivement par chunks (plusieurs exemples/fichier, np.savez_compressed cote
dataset_builder/chess_pgn_dataset_tools.py), jamais par exemple individuel - donc pas
de variante "single-example" a fournir ici. Toute future ecriture .npz, par lot ou non,
doit reutiliser ces memes constantes de cle plutot que d'en choisir de nouvelles.
"""

import chess
import numpy as np

# Cles .npz - source unique (AD-18, AD-25). Story 9.1 (producteur) et Story 9.3
# (consommateur) doivent les reutiliser plutot que des noms de cles inventes localement.
POSITION_KEY = "position"
POLICY_KEY = "policy"
VALUE_KEY = "value"

BOARD_SIZE = 8
HISTORY_LENGTH = 5  # nombre de demi-coups d'historique (fixe, PRD FR2)

_PIECE_TYPES = (chess.KING, chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT, chess.PAWN)

NUM_POSITION_PLANES = 2 * len(_PIECE_TYPES) + 1 + 4 + 1 + 1  # 12 + 1 + 4 + 1 + 1 = 19
NUM_HISTORY_PLANES = HISTORY_LENGTH * 2  # 10
NUM_PLANES = NUM_POSITION_PLANES + NUM_HISTORY_PLANES  # 29


def encode_position(board: chess.Board, move_history: list, include_history: bool = True) -> np.ndarray:
    """
    Encode une position + son historique de coups en planes (8, 8, NUM_PLANES).

    Args:
        board: position courante (le joueur au trait est board.turn).
        move_history: liste de chess.Move, du plus ancien au plus recent, jouee AVANT
            d'atteindre `board` (ex. les N derniers demi-coups de la partie a ce stade).
            Peut contenir moins de HISTORY_LENGTH elements (debut de partie) - le padding
            a zero des creneaux manquants est gere automatiquement ici.
        include_history: si False, n'encode que les NUM_POSITION_PLANES plans de position
            courante (retour de shape (8, 8, NUM_POSITION_PLANES), PAS NUM_PLANES) - les
            10 plans d'historique (source/destination des 5 derniers demi-coups) sont
            omis, pas mis a zero. Ajoute le 2026-07-29 (test d'ablation "l'historique
            sert-il a quelque chose ?", voir deferred-work.md) - defaut True, aucun
            changement de comportement pour les appelants existants qui ne le passent pas.

    Returns: np.ndarray float32, shape (8, 8, NUM_PLANES) si include_history (defaut),
        (8, 8, NUM_POSITION_PLANES) sinon. Voir docstring de module pour le detail exact
        plan par plan.
    """
    num_planes = NUM_PLANES if include_history else NUM_POSITION_PLANES
    planes = np.zeros((BOARD_SIZE, BOARD_SIZE, num_planes), dtype=np.float32)
    us = board.turn
    them = not us

    idx = 0
    for color in (us, them):
        for piece_type in _PIECE_TYPES:
            for square in board.pieces(piece_type, color):
                row, col = divmod(square, BOARD_SIZE)
                planes[row, col, idx] = 1.0
            idx += 1

    if us == chess.WHITE:
        planes[:, :, idx] = 1.0
    idx += 1

    planes[:, :, idx] = float(board.has_kingside_castling_rights(us)); idx += 1
    planes[:, :, idx] = float(board.has_queenside_castling_rights(us)); idx += 1
    planes[:, :, idx] = float(board.has_kingside_castling_rights(them)); idx += 1
    planes[:, :, idx] = float(board.has_queenside_castling_rights(them)); idx += 1

    planes[:, :, idx] = float(board.is_repetition(2)); idx += 1

    for legal_move in board.legal_moves:
        row, col = divmod(legal_move.to_square, BOARD_SIZE)
        planes[row, col, idx] = 1.0
    idx += 1

    assert idx == NUM_POSITION_PLANES, f"plans de position: attendu {NUM_POSITION_PLANES}, obtenu {idx}"

    if include_history:
        padding = [None] * max(0, HISTORY_LENGTH - len(move_history))
        padded_history = (padding + list(move_history))[-HISTORY_LENGTH:]
        for move in padded_history:
            if move is not None:
                fr, fc = divmod(move.from_square, BOARD_SIZE)
                tr, tc = divmod(move.to_square, BOARD_SIZE)
                planes[fr, fc, idx] = 1.0
                planes[tr, tc, idx + 1] = 1.0
            idx += 2

    assert idx == num_planes, f"total de plans: attendu {num_planes}, obtenu {idx}"
    return planes


# === Schema d'action AlphaZero (Silver et al., 2018) - 64 cases source x 73 types ===
#
# 56 coups "dame" : 8 directions x 7 distances (1 a 7 cases). Couvre tous les coups de
# tour/fou/dame/roi (roi = distance 1) et le roque (deplacement de roi de 2 cases,
# une des 56 combinaisons). Directions en (delta_file, delta_rank), ordre fixe :
_QUEEN_DIRECTIONS = (
    (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1),
)  # N, NE, E, SE, S, SW, W, NW

# 8 coups de cavalier (deplacements en L) :
_KNIGHT_DELTAS = (
    (1, 2), (2, 1), (2, -1), (1, -2), (-1, -2), (-2, -1), (-2, 1), (-1, 2),
)

# 9 sous-promotions : 3 directions relatives au sens de marche du joueur (tout droit,
# capture a gauche, capture a droite) x 3 pieces (Cavalier, Fou, Tour). Une promotion en
# Dame est un coup de pion normal (distance 1, deja couvert par les 56 coups "dame"), pas
# une sous-promotion.
_UNDERPROMOTION_PIECES = (chess.KNIGHT, chess.BISHOP, chess.ROOK)

NUM_MOVES = 64 * 73  # 4672


def _underpromotion_deltas(color: bool) -> tuple:
    """(tout droit, capture gauche, capture droite), en (delta_file, delta_rank) absolu."""
    forward = 1 if color == chess.WHITE else -1
    return ((0, forward), (-forward, forward), (forward, forward))


def move_to_index(move: chess.Move, board: chess.Board) -> int:
    """
    Encode un coup en index entier dans [0, NUM_MOVES). Source unique de cette
    conversion (AD-18) - jamais reimplementee independamment cote Story 9.2/9.3.

    La prise en passant est un coup de pion diagonal normal du point de vue de cet
    encodage (case source/destination) - couverte par les 56 coups "dame" sans
    traitement special.

    Leve ValueError si from_square == to_square (coup degenere/nul, ex. chess.Move.null())
    ou si `board.piece_at(move.from_square)` est None (move incoherent avec `board`).
    """
    from_sq, to_sq = move.from_square, move.to_square
    if from_sq == to_sq:
        raise ValueError(f"coup degenere : from_square == to_square == {from_sq} (coup nul ?)")

    from_file, from_rank = chess.square_file(from_sq), chess.square_rank(from_sq)
    to_file, to_rank = chess.square_file(to_sq), chess.square_rank(to_sq)
    delta_file, delta_rank = to_file - from_file, to_rank - from_rank

    if move.promotion is not None and move.promotion != chess.QUEEN:
        piece = board.piece_at(from_sq)
        if piece is None:
            raise ValueError(f"aucune piece en case source {from_sq} - move {move.uci()} incoherent avec board")
        move_type = 64 + _UNDERPROMOTION_PIECES.index(move.promotion) * 3 + \
            _underpromotion_deltas(piece.color).index((delta_file, delta_rank))
    elif (abs(delta_file), abs(delta_rank)) in ((1, 2), (2, 1)):
        move_type = 56 + _KNIGHT_DELTAS.index((delta_file, delta_rank))
    else:
        distance = max(abs(delta_file), abs(delta_rank))
        direction = (delta_file // distance, delta_rank // distance)
        move_type = _QUEEN_DIRECTIONS.index(direction) * 7 + (distance - 1)

    return from_sq * 73 + move_type


def index_to_move(index: int, board: chess.Board) -> chess.Move:
    """
    Decode un index [0, NUM_MOVES) en chess.Move legal sur `board`. Inverse de
    move_to_index - construit le coup candidat puis le valide contre board.legal_moves
    (seule source de verite sur la legalite, jamais reimplementee ici).

    Leve ValueError si aucun coup legal de `board` ne correspond a cet index (ex. index
    predit par un modele encore non entraine, hors du sous-ensemble legal de la position),
    y compris si `index` est hors de [0, NUM_MOVES) (ex. tete policy mal dimensionnee).
    """
    if not (0 <= index < NUM_MOVES):
        raise ValueError(f"index {index} hors de [0, {NUM_MOVES})")

    from_sq, move_type = divmod(index, 73)

    if move_type < 56:
        direction_idx, distance = divmod(move_type, 7)
        delta_file, delta_rank = _QUEEN_DIRECTIONS[direction_idx]
        distance += 1
        to_file = chess.square_file(from_sq) + delta_file * distance
        to_rank = chess.square_rank(from_sq) + delta_rank * distance
        promotion = None
    elif move_type < 64:
        delta_file, delta_rank = _KNIGHT_DELTAS[move_type - 56]
        to_file = chess.square_file(from_sq) + delta_file
        to_rank = chess.square_rank(from_sq) + delta_rank
        promotion = None
    else:
        sub_idx = move_type - 64
        piece_idx, dir_idx = divmod(sub_idx, 3)
        color = board.piece_at(from_sq).color if board.piece_at(from_sq) else chess.WHITE
        delta_file, delta_rank = _underpromotion_deltas(color)[dir_idx]
        to_file = chess.square_file(from_sq) + delta_file
        to_rank = chess.square_rank(from_sq) + delta_rank
        promotion = _UNDERPROMOTION_PIECES[piece_idx]

    if not (0 <= to_file < 8 and 0 <= to_rank < 8):
        raise ValueError(f"index {index}: case destination hors echiquier (file={to_file}, rank={to_rank})")

    to_sq = chess.square(to_file, to_rank)

    # Promotion en Dame implicite : un pion atteignant la derniere rangee via un coup
    # "dame" (move_type < 56, promotion=None ci-dessus) doit porter promotion=QUEEN pour
    # etre un coup legal valide (python-chess l'exige explicitement).
    piece = board.piece_at(from_sq)
    if promotion is None and piece is not None and piece.piece_type == chess.PAWN and to_rank in (0, 7):
        promotion = chess.QUEEN

    candidate = chess.Move(from_sq, to_sq, promotion=promotion)
    if candidate not in board.legal_moves:
        raise ValueError(f"index {index} -> {candidate.uci()} n'est pas un coup legal de cette position")
    return candidate
