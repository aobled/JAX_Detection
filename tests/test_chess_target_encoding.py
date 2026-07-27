"""
Test de round-trip pour chess_target_encoding.py (Story 9.1, AD-18/AD-22/AD-25).

Script autonome - ce projet n'a pas de framework de test formel (voir Dev Notes de la
story 9.1). Executer directement :
    python tests/test_chess_target_encoding.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import chess
import chess.pgn
import io

from chess_target_encoding import (
    encode_position,
    move_to_index,
    index_to_move,
    NUM_MOVES,
    NUM_PLANES,
    NUM_POSITION_PLANES,
    HISTORY_LENGTH,
    POSITION_KEY,
    POLICY_KEY,
    VALUE_KEY,
)


def test_contract_constants_pinned():
    # Epingle les constantes de contrat (AD-18/AD-22) comme litteraux - un refactor futur
    # de _PIECE_TYPES/HISTORY_LENGTH changerait silencieusement NUM_PLANES/NUM_MOVES sans
    # faire echouer aucun autre test (qui utilisent tous le symbole, pas la valeur), alors
    # que la Story 9.2 va importer NUM_MOVES pour dimensionner la tete policy du modele.
    assert NUM_MOVES == 4672, f"NUM_MOVES a change : {NUM_MOVES} (attendu 4672 = 64*73, contrat AD-22)"
    assert NUM_PLANES == 29, f"NUM_PLANES a change : {NUM_PLANES} (attendu 29 = 19 position + 10 historique)"

    # Coup connu, calcule a la main, epingle en dur : e2e4 depuis la position de depart.
    # from_square=e2=12, direction N=(0,1) index 0 dans _QUEEN_DIRECTIONS, distance=2 ->
    # move_type = 0*7 + (2-1) = 1. index = 12*73 + 1 = 877.
    board = chess.Board()
    e2e4 = chess.Move.from_uci("e2e4")
    idx = move_to_index(e2e4, board)
    assert idx == 877, f"e2e4 depuis la position de depart : attendu index 877, obtenu {idx}"
    print(f"OK - constantes de contrat epinglees (NUM_MOVES=4672, NUM_PLANES=29) + e2e4 -> index 877")


def _assert_move_roundtrip(board: chess.Board, move: chess.Move, label: str):
    idx = move_to_index(move, board)
    assert 0 <= idx < NUM_MOVES, f"[{label}] index {idx} hors de [0, {NUM_MOVES})"
    decoded = index_to_move(idx, board)
    assert decoded == move, f"[{label}] {move.uci()} -> index {idx} -> {decoded.uci()} (attendu {move.uci()})"


def test_roundtrip_all_legal_moves_start_position():
    board = chess.Board()
    for move in board.legal_moves:
        _assert_move_roundtrip(board, move, f"depart:{move.uci()}")
    print(f"OK - round-trip sur les {board.legal_moves.count()} coups legaux de la position de depart")


def test_roundtrip_castling():
    # Position avec roque disponible des deux cotes pour les Blancs
    board = chess.Board("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1")
    castling_moves = [m for m in board.legal_moves if board.is_castling(m)]
    assert len(castling_moves) == 2, f"attendu 2 coups de roque, obtenu {len(castling_moves)}"
    for move in castling_moves:
        _assert_move_roundtrip(board, move, f"roque:{move.uci()}")
    print(f"OK - round-trip roque ({[m.uci() for m in castling_moves]})")


def test_roundtrip_en_passant():
    board = chess.Board("rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3")
    ep_moves = [m for m in board.legal_moves if board.is_en_passant(m)]
    assert len(ep_moves) == 1, f"attendu 1 coup en passant, obtenu {len(ep_moves)}"
    _assert_move_roundtrip(board, ep_moves[0], f"en-passant:{ep_moves[0].uci()}")
    print(f"OK - round-trip prise en passant ({ep_moves[0].uci()})")


def test_roundtrip_promotion_all_pieces():
    # Pion blanc en a7, peut promouvoir tout droit (a8, case vide) - Dame + 3 sous-promotions
    board = chess.Board("7k/P7/8/8/8/8/8/K7 w - - 0 1")
    promo_moves = [m for m in board.legal_moves if m.promotion is not None]
    assert len(promo_moves) == 4, f"attendu 4 promotions (Q/R/B/N), obtenu {len(promo_moves)}"
    for move in promo_moves:
        _assert_move_roundtrip(board, move, f"promotion:{move.uci()}")
    print(f"OK - round-trip promotion, 4 pieces ({[m.uci() for m in promo_moves]})")


def test_roundtrip_promotion_with_capture():
    # Pion blanc en b7 peut promouvoir tout droit (b8) OU capturer en a8/c8
    board = chess.Board("n1n5/1P6/8/8/8/8/8/K6k w - - 0 1")
    promo_moves = [m for m in board.legal_moves if m.promotion is not None]
    assert len(promo_moves) == 12, f"attendu 12 promotions (3 destinations x 4 pieces), obtenu {len(promo_moves)}"
    for move in promo_moves:
        _assert_move_roundtrip(board, move, f"promotion-capture:{move.uci()}")
    print(f"OK - round-trip promotion avec capture, 12 coups ({[m.uci() for m in promo_moves]})")


def test_roundtrip_promotion_black():
    # Symetrique du test Blancs : pion noir en b2, promotion tout droit ou capture,
    # verifie que _underpromotion_deltas (logique couleur-relative) est correcte pour
    # les deux couleurs, pas seulement les Blancs.
    board = chess.Board("k7/8/8/8/8/8/1p6/N1N4K b - - 0 1")
    promo_moves = [m for m in board.legal_moves if m.promotion is not None]
    assert len(promo_moves) == 12, f"attendu 12 promotions (3 destinations x 4 pieces), obtenu {len(promo_moves)}"
    for move in promo_moves:
        _assert_move_roundtrip(board, move, f"promotion-noir:{move.uci()}")
    print(f"OK - round-trip promotion Noirs, 12 coups ({[m.uci() for m in promo_moves]})")


def test_roundtrip_random_game_exhaustive():
    # Partie semi-aleatoire plus longue (premier coup legal a chaque demi-coup, deterministe
    # via seed implicite de l'ordre de generation python-chess) - round-trip exhaustif sur
    # TOUS les coups legaux de CHAQUE position rencontree, forte probabilite de couvrir des
    # geometries variees (captures, echecs, coups de bord d'echiquier).
    import random
    random.seed(42)
    board = chess.Board()
    positions_checked = 0
    for _ in range(40):
        if board.is_game_over():
            break
        legal = list(board.legal_moves)
        for move in legal:
            _assert_move_roundtrip(board, move, f"random-game-ply{positions_checked}:{move.uci()}")
        positions_checked += 1
        board.push(random.choice(legal))
    print(f"OK - round-trip exhaustif sur {positions_checked} positions d'une partie semi-aleatoire (40 demi-coups max)")


def test_roundtrip_midgame_positions():
    # Quelques positions apres une petite sequence d'ouverture, coups varies (cavaliers,
    # fous, dame) - round-trip exhaustif sur tous les coups legaux de chaque position.
    board = chess.Board()
    opening = ["e2e4", "e7e5", "g1f3", "b8c6", "f1c4", "g8f6", "d2d3", "f8c5"]
    for uci in opening:
        board.push(chess.Move.from_uci(uci))
        for move in board.legal_moves:
            _assert_move_roundtrip(board, move, f"midgame:{uci}:{move.uci()}")
    print(f"OK - round-trip exhaustif sur {len(opening)} positions de milieu de partie")


def test_encode_position_shape_and_history_padding():
    board = chess.Board()
    planes = encode_position(board, move_history=[])
    assert planes.shape == (8, 8, NUM_PLANES), f"shape attendue (8,8,{NUM_PLANES}), obtenu {planes.shape}"

    # Position de depart, 0 demi-coup joue -> les HISTORY_LENGTH creneaux d'historique
    # (2 plans chacun, apres les NUM_POSITION_PLANES premiers) doivent etre entierement a zero.
    history_planes = planes[:, :, NUM_POSITION_PLANES:]
    assert history_planes.shape[-1] == HISTORY_LENGTH * 2
    assert history_planes.sum() == 0.0, "planes d'historique non nulles a la position de depart (0 coup joue)"

    # Plan trait (index 12) : Blancs au trait a la position de depart -> constante 1.0
    assert (planes[:, :, 12] == 1.0).all(), "plan trait incorrect a la position de depart (Blancs au trait)"
    print("OK - shape (8,8,29) + padding a zero de l'historique a la position de depart + plan trait")


def test_encode_position_partial_history_padding():
    board = chess.Board()
    moves = [chess.Move.from_uci(u) for u in ["e2e4", "e7e5"]]
    for m in moves:
        board.push(m)
    # 2 demi-coups joues -> 2 creneaux d'historique remplis (les 2 plus recents), 3 a zero
    planes = encode_position(board, move_history=moves)
    history_planes = planes[:, :, NUM_POSITION_PLANES:]
    slot_sums = [history_planes[:, :, i * 2:i * 2 + 2].sum() for i in range(HISTORY_LENGTH)]
    zero_slots = sum(1 for s in slot_sums if s == 0.0)
    nonzero_slots = sum(1 for s in slot_sums if s > 0.0)
    assert zero_slots == 3, f"attendu 3 creneaux vides (padding), obtenu {zero_slots} ({slot_sums})"
    assert nonzero_slots == 2, f"attendu 2 creneaux remplis, obtenu {nonzero_slots} ({slot_sums})"
    print(f"OK - padding partiel de l'historique (2 coups joues -> 2 creneaux remplis, 3 a zero, sommes={slot_sums})")


def test_dataset_example_count_matches_plies():
    # Exerce la vraie fonction de production (_iter_game_examples, dataset_builder/
    # chess_pgn_dataset_tools.py) plutot que de reimplementer la boucle et la logique de
    # signe de la value en double ici - evite toute divergence silencieuse entre deux
    # copies de la meme logique (finding de code review, Story 9.1).
    from dataset_builder.chess_pgn_dataset_tools import _iter_game_examples

    # Petite partie PGN de test (4 coups blancs, 4 coups noirs = 8 demi-coups), Blancs gagnent
    pgn_text = """[Event "Test"]
[Result "1-0"]

1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6 1-0
"""
    game = chess.pgn.read_game(io.StringIO(pgn_text))
    assert len(list(game.mainline_moves())) == 8, "attendu 8 demi-coups dans la partie de test"

    examples = list(_iter_game_examples(game))

    assert len(examples) == 8, f"attendu 8 exemples (== nb demi-coups), obtenu {len(examples)}"
    values = [value for _, _, value in examples]
    assert values == [1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0], (
        f"value doit alterner selon le joueur au trait (Blancs gagnent, 1-0): {values}"
    )
    print(f"OK - 8 demi-coups -> 8 exemples (via _iter_game_examples reel), value alterne correctement ({values})")


if __name__ == "__main__":
    test_contract_constants_pinned()
    test_roundtrip_all_legal_moves_start_position()
    test_roundtrip_castling()
    test_roundtrip_en_passant()
    test_roundtrip_promotion_all_pieces()
    test_roundtrip_promotion_with_capture()
    test_roundtrip_promotion_black()
    test_roundtrip_random_game_exhaustive()
    test_roundtrip_midgame_positions()
    test_encode_position_shape_and_history_padding()
    test_encode_position_partial_history_padding()
    test_dataset_example_count_matches_plies()
    print("\nTous les tests round-trip sont passes.")
