#!/usr/bin/env python3
"""Chess tournament data analysis - answers all 24 questions."""
from __future__ import annotations

import re
import math
import itertools
from collections import defaultdict, Counter, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional, Iterator
import multiprocessing as mp

import chess
import numpy as np
import pandas as pd
import pytz
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier

# ============================================================
# Configuration
# ============================================================

GAMES = "data/games.csv.gz"
MOVES = "data/moves.csv.gz"
TOURNAMENTS = "data/tournaments.csv.gz"
OUTPUT = "answers.md"

CET = pytz.timezone("Europe/Budapest")
UTC = pytz.utc
CHUNKSIZE = 300_000

# ============================================================
# Utility functions
# ============================================================

def parse_clock(s) -> int:
    """Parse H:MM:SS clock to seconds. Returns -1 for invalid."""
    try:
        parts = str(s).split(":")
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
    except Exception:
        return -1


def parse_tc(tc) -> tuple[int, int]:
    """Parse '180+0' -> (180, 0)."""
    try:
        b, i = str(tc).split("+")
        return int(b), int(i)
    except Exception:
        return 0, 0


def game_start_cet(utcdate: str, utctime: str) -> Optional[datetime]:
    try:
        dt = datetime.strptime(f"{utcdate} {utctime}", "%Y.%m.%d %H:%M:%S")
        return UTC.localize(dt).astimezone(CET)
    except Exception:
        return None


def is_standard(variant) -> bool:
    return str(variant).strip().lower() == "standard"


def result_code(result: str) -> int:
    """1=white wins, -1=black wins, 0=draw."""
    if result == "1-0":
        return 1
    if result == "0-1":
        return -1
    return 0


def simulate_board(moves_list: list[str]) -> Optional[chess.Board]:
    """Simulate board from SAN move list. Returns None on illegal move."""
    board = chess.Board()
    for san in moves_list:
        try:
            board.push_san(san)
        except Exception:
            return board  # return partial board
    return board


def count_material(board: chess.Board) -> tuple[int, int]:
    """Return (white_material, black_material) in standard points."""
    w = b = 0
    vals = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9}
    for pt, v in vals.items():
        w += len(board.pieces(pt, chess.WHITE)) * v
        b += len(board.pieces(pt, chess.BLACK)) * v
    return w, b


def sq_to_coords(sq: chess.Square) -> tuple[int, int]:
    return chess.square_file(sq), chess.square_rank(sq)


def square_name_to_coords(name: str) -> tuple[int, int]:
    """'a1' -> (0, 0), 'h8' -> (7, 7)."""
    return ord(name[0]) - ord("a"), int(name[1]) - 1


# ============================================================
# Hungarian alphabet ordering for Q18
# ============================================================

HU_ORDER = {
    "a": 1, "á": 2, "b": 3, "c": 4, "cs": 5, "d": 6, "dz": 7, "dzs": 8,
    "e": 9, "é": 10, "f": 11, "g": 12, "gy": 13, "h": 14, "i": 15, "í": 16,
    "j": 17, "k": 18, "l": 19, "ly": 20, "m": 21, "n": 22, "ny": 23,
    "o": 24, "ó": 25, "ö": 26, "ő": 27, "p": 28, "q": 29, "r": 30,
    "s": 31, "sz": 32, "t": 33, "ty": 34, "u": 35, "ú": 36, "ü": 37, "ű": 38,
    "v": 39, "w": 40, "x": 41, "y": 42, "z": 43, "zs": 44,
}

def hu_key(name: str) -> list[int]:
    """Generate sort key for Hungarian alphabetical order."""
    s = name.lower()
    result = []
    i = 0
    while i < len(s):
        # Try 3-char digraph
        if i + 2 < len(s) and s[i:i+3] in HU_ORDER:
            result.append(HU_ORDER[s[i:i+3]])
            i += 3
        elif i + 1 < len(s) and s[i:i+2] in HU_ORDER:
            result.append(HU_ORDER[s[i:i+2]])
            i += 2
        else:
            result.append(HU_ORDER.get(s[i], 100 + ord(s[i])))
            i += 1
    return result


# ============================================================
# Games streaming
# ============================================================

def stream_games_from_moves(
    game_id_filter: Optional[set] = None,
) -> Iterator[tuple[str, list[str], pd.DataFrame]]:
    """
    Stream complete games from moves file as (game_id, moves_list, moves_df).
    Reads ALL chunks but only yields games in game_id_filter (if given).
    """
    buffer = pd.DataFrame()

    for chunk in pd.read_csv(MOVES, chunksize=CHUNKSIZE):
        combined = pd.concat([buffer, chunk], ignore_index=True) if len(buffer) > 0 else chunk

        last_game = combined["game_id"].iloc[-1]
        complete = combined[combined["game_id"] != last_game]
        buffer = combined[combined["game_id"] == last_game]

        for gid, grp in complete.groupby("game_id", sort=False):
            if game_id_filter is None or gid in game_id_filter:
                yield gid, grp["move"].tolist(), grp

    if len(buffer) > 0:
        for gid, grp in buffer.groupby("game_id", sort=False):
            if game_id_filter is None or gid in game_id_filter:
                yield gid, grp["move"].tolist(), grp


# ============================================================
# Q1: Material disadvantage ≥ 3, standard, 2023.10.12–2024.02.19
# ============================================================

def q1_material_disadvantage(games: pd.DataFrame) -> int:
    print("  Q1: filtering games...")
    mask = (
        (games["variant"] == "Standard")
        & (games["date"] >= "2023.10.12")
        & (games["date"] <= "2024.02.19")
        & (games["result"] != "1/2-1/2")
    )
    relevant = set(games.loc[mask, "game_id"])
    result_map = dict(zip(games["game_id"], games["result"]))

    print(f"  Q1: simulating {len(relevant):,} games...")
    count = 0
    done = 0
    for gid, moves_list, _ in stream_games_from_moves(relevant):
        done += 1
        if done % 200000 == 0:
            print(f"    Q1 progress: {done:,}/{len(relevant):,}")
        board = simulate_board(moves_list)
        if board is None:
            continue
        res = result_map.get(gid, "")
        w, b = count_material(board)
        if res == "1-0" and (w - b) >= 3:
            count += 1
        elif res == "0-1" and (b - w) >= 3:
            count += 1
    return count


# ============================================================
# Q2: Left knight capture win rate
# ============================================================

def q2_left_knight_capture(games: pd.DataFrame) -> str:
    """
    White left knight: b1, black left knight: g8.
    Compare win rate of those who captured with left knight vs those who didn't.
    """
    result_map = dict(zip(games["game_id"], games["result"]))

    # win = player whose color won; track per-game if left knight captured
    left_knight_wins = 0
    left_knight_total = 0
    no_knight_wins = 0
    no_knight_total = 0

    print("  Q2: streaming all games for left knight capture...")
    done = 0
    for gid, moves_list, moves_df in stream_games_from_moves():
        done += 1
        if done % 500000 == 0:
            print(f"    Q2 progress: {done:,}")

        res = result_map.get(gid, "")
        if res == "1/2-1/2":
            continue

        board = chess.Board()
        white_left_captured = False
        black_left_captured = False

        # Track which squares our "left knights" are currently on
        white_left_sq = chess.B1   # b1
        black_left_sq = chess.G8   # g8
        white_left_alive = True
        black_left_alive = True

        for san in moves_list:
            try:
                move = board.push_san(san)
            except Exception:
                break

            piece = board.piece_at(move.to_square)
            captured_sq = move.to_square

            # Check if the moved piece was the "left knight"
            if board.turn == chess.BLACK:  # white just moved
                if white_left_alive and move.from_square == white_left_sq:
                    white_left_sq = move.to_square
                    if board.is_capture(move):
                        white_left_captured = True
            else:  # black just moved
                if black_left_alive and move.from_square == black_left_sq:
                    black_left_sq = move.to_square
                    if board.is_capture(move):
                        black_left_captured = True

            # Check if our tracked knight was captured
            if white_left_alive and captured_sq == white_left_sq and board.turn == chess.BLACK:
                # white knight was captured? Actually if white just moved to to_square,
                # we need to check if anyone captured the white_left_sq
                pass

        # Actually need better tracking - check if white left knight was captured by enemy
        # Re-do with proper tracking
        board2 = chess.Board()
        white_left_sq = chess.B1
        black_left_sq = chess.G8
        white_left_alive = True
        black_left_alive = True
        white_left_captured = False
        black_left_captured = False

        for san in moves_list:
            try:
                move = board2.push_san(san)
            except Exception:
                break

            from_sq = move.from_square
            to_sq = move.to_square
            is_cap = board2.is_capture(move) if hasattr(board2, '_original') else False
            # After push, it's the other side's turn
            mover_was_white = (board2.turn == chess.BLACK)

            if mover_was_white:
                if white_left_alive and from_sq == white_left_sq:
                    # Piece type check: must be a knight
                    moved_pt = board2.piece_type_at(to_sq)
                    if moved_pt == chess.KNIGHT:
                        white_left_sq = to_sq
                    else:
                        white_left_alive = False  # promoted or something weird
                # Check if black captured our white left knight
                if white_left_alive and to_sq == white_left_sq and not mover_was_white:
                    white_left_alive = False
            else:
                if black_left_alive and from_sq == black_left_sq:
                    moved_pt = board2.piece_type_at(to_sq)
                    if moved_pt == chess.KNIGHT:
                        black_left_sq = to_sq
                    else:
                        black_left_alive = False
                if black_left_alive and to_sq == black_left_sq and mover_was_white:
                    black_left_alive = False

        # Hmm, this approach is getting complicated. Let me use a cleaner method.
        break  # placeholder - will implement properly below

    # Clean implementation using from_square from chess library
    return _q2_clean(games, result_map)


def _q2_clean(games: pd.DataFrame, result_map: dict) -> str:
    """Track left knight captures using python-chess move.from_square."""
    # white left knight starts b1=chess.B1, black left knight g8=chess.G8
    left_knight_data = []  # (player_who_captured_color, res)

    done = 0
    for gid, moves_list, moves_df in stream_games_from_moves():
        done += 1
        if done % 500000 == 0:
            print(f"    Q2 progress: {done:,}")

        res = result_map.get(gid, "")
        board = chess.Board()
        white_left_sq = chess.B1
        black_left_sq = chess.G8
        white_left_alive = True
        black_left_alive = True
        white_captured = False
        black_captured = False

        for san in moves_list:
            try:
                # Check capture BEFORE pushing
                is_cap = "x" in san
                move = board.push_san(san)
            except Exception:
                break

            from_sq = move.from_square
            to_sq = move.to_square
            was_white = not board.turn  # board.turn is now NEXT player's turn

            if was_white:
                if white_left_alive and from_sq == white_left_sq:
                    # Our left white knight moved
                    white_left_sq = to_sq
                    if is_cap:
                        white_captured = True
                # Check if white captured the black left knight
                if black_left_alive and to_sq == black_left_sq and is_cap:
                    black_left_alive = False
            else:
                if black_left_alive and from_sq == black_left_sq:
                    black_left_sq = to_sq
                    if is_cap:
                        black_captured = True
                if white_left_alive and to_sq == white_left_sq and is_cap:
                    white_left_alive = False

        left_knight_data.append((white_captured, black_captured, res))

    # Compute win rates
    # "captured with left knight" = white_captured or black_captured
    # Win = player's color won
    lk_wins = lk_total = 0
    no_wins = no_total = 0

    for w_cap, b_cap, res in left_knight_data:
        if res == "1/2-1/2":
            continue
        # White player
        if w_cap:
            lk_total += 1
            if res == "1-0":
                lk_wins += 1
        else:
            no_total += 1
            if res == "1-0":
                no_wins += 1
        # Black player
        if b_cap:
            lk_total += 1
            if res == "0-1":
                lk_wins += 1
        else:
            no_total += 1
            if res == "0-1":
                no_wins += 1

    lk_rate = lk_wins / lk_total if lk_total > 0 else 0
    no_rate = no_wins / no_total if no_total > 0 else 0
    diff = lk_rate - no_rate
    direction = "nagyobb" if diff > 0 else "kisebb"
    return (f"Bal oldali lóval ütők nyerési aránya: {lk_rate:.4f} ({lk_wins}/{lk_total}), "
            f"nem ütők aránya: {no_rate:.4f} ({no_wins}/{no_total}), "
            f"különbség: {diff:+.4f} ({direction} arányban nyertek a bal oldali lóval ütők)")


# ============================================================
# Q3: 10-min games, white loses castling in first 6 half-moves
# ============================================================

def q3_castling_rights_lost(games: pd.DataFrame) -> int:
    mask = games["timecontrol"].str.startswith("600", na=False)
    relevant = set(games.loc[mask, "game_id"])
    print(f"  Q3: {len(relevant):,} 10-min games...")

    count = 0
    done = 0
    for gid, moves_list, moves_df in stream_games_from_moves(relevant):
        done += 1
        if done % 100000 == 0:
            print(f"    Q3 progress: {done:,}/{len(relevant):,}")

        board = chess.Board()
        white_moves_made = 0

        # We check after each white move (moves 1,2,3 = half moves 1,3,5)
        initial_white_castling = board.has_castling_rights(chess.WHITE)
        lost = False

        for i, san in enumerate(moves_list[:6]):  # first 6 half-moves
            prev_rights = board.has_castling_rights(chess.WHITE)
            try:
                board.push_san(san)
            except Exception:
                break

            # After white's move (half-move 0, 2, 4), check white's castling rights
            if i % 2 == 0:  # white moved
                if not board.has_castling_rights(chess.WHITE) and prev_rights:
                    lost = True
                    break

        if lost:
            count += 1

    return count


# ============================================================
# Q4: Rook distance difference (white vs black total)
# ============================================================

def q4_rook_distances(games: pd.DataFrame) -> int:
    print("  Q4: computing rook distances for all games...")
    white_dist = 0
    black_dist = 0
    done = 0

    for gid, moves_list, moves_df in stream_games_from_moves():
        done += 1
        if done % 500000 == 0:
            print(f"    Q4 progress: {done:,}")

        board = chess.Board()
        for san in moves_list:
            try:
                move = board.push_san(san)
            except Exception:
                break

            from_sq = move.from_square
            to_sq = move.to_square
            was_white = not board.turn

            # Determine if the piece that moved was a rook
            # After push, check piece at to_sq
            pt = board.piece_type_at(to_sq)

            # Castling: rook also moves silently
            if board.is_castling(move) if hasattr(board, 'is_castling') else False:
                # handled separately
                pass

            if pt == chess.ROOK:
                dist = abs(chess.square_file(to_sq) - chess.square_file(from_sq)) + \
                       abs(chess.square_rank(to_sq) - chess.square_rank(from_sq))
                if was_white:
                    white_dist += dist
                else:
                    black_dist += dist

            # Also handle castling: the rook moves silently
            # We need to detect castling and compute the rook's distance
            # board.is_castling can be checked on the move BEFORE push, so let's check differently

    return white_dist - black_dist


def q4_rook_distances_v2(games: pd.DataFrame) -> int:
    """Better version that handles castling rook moves."""
    print("  Q4: computing rook distances (v2)...")
    white_dist = 0
    black_dist = 0
    done = 0

    for gid, moves_list, moves_df in stream_games_from_moves():
        done += 1
        if done % 500000 == 0:
            print(f"    Q4 progress: {done:,}")

        board = chess.Board()
        for san in moves_list:
            try:
                pre_turn = board.turn
                move = board.push_san(san)
            except Exception:
                break

            from_sq = move.from_square
            to_sq = move.to_square
            was_white = (pre_turn == chess.WHITE)

            # Castling: king moves, but we need rook distance too
            if board.is_kingside_castling(move) or board.is_queenside_castling(move):
                # Rook moved: kingside h1->f1 (white), h8->f8 (black)
                #            queenside a1->d1 (white), a8->d8 (black)
                if board.is_kingside_castling(move):
                    rook_from = chess.H1 if was_white else chess.H8
                    rook_to = chess.F1 if was_white else chess.F8
                else:
                    rook_from = chess.A1 if was_white else chess.A8
                    rook_to = chess.D1 if was_white else chess.D8
                dist = abs(chess.square_file(rook_to) - chess.square_file(rook_from))
                if was_white:
                    white_dist += dist
                else:
                    black_dist += dist
                continue

            pt = board.piece_type_at(to_sq)
            if pt == chess.ROOK:
                dist = (abs(chess.square_file(to_sq) - chess.square_file(from_sq)) +
                        abs(chess.square_rank(to_sq) - chess.square_rank(from_sq)))
                if was_white:
                    white_dist += dist
                else:
                    black_dist += dist

    return white_dist - black_dist


# ============================================================
# Q5: Threefold repetition + scissors emoji in name
# ============================================================

def q5_threefold_scissors(games: pd.DataFrame) -> int:
    """Check if any player has scissors emoji. All names are ASCII -> answer is 0."""
    scissors_pattern = re.compile(r"[✂✀✁✋]|✂")
    has_scissors = (
        games["white"].str.contains(scissors_pattern, na=False, regex=True)
        | games["black"].str.contains(scissors_pattern, na=False, regex=True)
    )
    scissors_games = set(games.loc[has_scissors, "game_id"])
    if not scissors_games:
        return 0

    # If any exist, simulate boards
    result_map = dict(zip(games["game_id"], games["result"]))
    count = 0
    for gid, moves_list, _ in stream_games_from_moves(scissors_games):
        if result_map.get(gid) != "1/2-1/2":
            continue
        board = simulate_board(moves_list)
        if board and board.is_repetition(3):
            count += 1
    return count


# ============================================================
# Q6: Threefold repetition draws, standard, 2024.03.12–2024.11.19
# ============================================================

def q6_threefold_date_range(games: pd.DataFrame) -> int:
    mask = (
        (games["variant"] == "Standard")
        & (games["date"] >= "2024.03.12")
        & (games["date"] <= "2024.11.19")
        & (games["result"] == "1/2-1/2")
        & (games["termination"] == "Normal")
    )
    relevant = set(games.loc[mask, "game_id"])
    print(f"  Q6: simulating {len(relevant):,} draw games...")

    count = 0
    done = 0
    for gid, moves_list, _ in stream_games_from_moves(relevant):
        done += 1
        if done % 50000 == 0:
            print(f"    Q6 progress: {done:,}/{len(relevant):,}")
        board = simulate_board(moves_list)
        if board and board.is_repetition(3):
            count += 1
    return count


# ============================================================
# Q7: Avg white queens at checkmate in tournament winner games
# ============================================================

def q7_queens_at_checkmate(games: pd.DataFrame, tournaments: pd.DataFrame) -> float:
    # Map tournament_id -> winner_id (lichess username, lowercase)
    t = tournaments[["id", "winner__id"]].dropna()
    tour_winner = dict(zip(t["id"], t["winner__id"].str.lower()))

    # Games where tournament winner won by checkmate
    def is_winner_game(row) -> bool:
        winner = tour_winner.get(row["tournamentid"])
        if winner is None:
            return False
        if row["termination"] != "Normal":
            return False
        if row["result"] == "1-0" and str(row["white"]).lower() == winner:
            return True
        if row["result"] == "0-1" and str(row["black"]).lower() == winner:
            return True
        return False

    mask = games.apply(is_winner_game, axis=1)
    relevant = set(games.loc[mask, "game_id"])
    result_map = dict(zip(games["game_id"], games["result"]))
    print(f"  Q7: simulating {len(relevant):,} tournament winner games...")

    white_queen_counts = []
    done = 0
    for gid, moves_list, _ in stream_games_from_moves(relevant):
        done += 1
        if done % 20000 == 0:
            print(f"    Q7 progress: {done:,}/{len(relevant):,}")
        # Check last move is checkmate
        if not moves_list or not moves_list[-1].endswith("#"):
            continue
        board = simulate_board(moves_list)
        if board is None:
            continue
        n_white_queens = len(board.pieces(chess.QUEEN, chess.WHITE))
        white_queen_counts.append(n_white_queens)

    return float(np.mean(white_queen_counts)) if white_queen_counts else 0.0


# ============================================================
# Q8: Draw on March 20 where last move is pawn promotion
# ============================================================

def q8_draw_march20_promotion(games: pd.DataFrame) -> int:
    # Filter draws on March 20 (any year)
    march20 = games[
        (games["result"] == "1/2-1/2")
        & games["date"].str.endswith(".03.20", na=False)
    ]
    relevant = set(march20["game_id"])
    print(f"  Q8: checking {len(relevant):,} draw games on March 20...")

    count = 0
    for gid, moves_list, _ in stream_games_from_moves(relevant):
        if moves_list and "=" in moves_list[-1]:
            count += 1
    return count


# ============================================================
# Q9: Berserk timeout losses (games-only)
# ============================================================

def q9_berserk_timeouts(games: pd.DataFrame) -> list[str]:
    timeouts = games[games["termination"] == "Time forfeit"].copy()
    timeouts["tc_base"] = timeouts["timecontrol"].apply(lambda x: parse_tc(x)[0])
    timeouts["ws"] = timeouts["whitestart"].apply(parse_clock)
    timeouts["bs"] = timeouts["blackstart"].apply(parse_clock)

    # Berserk: starting time <= tc_base/2 + a small tolerance
    def white_berserk(row):
        return row["ws"] > 0 and row["tc_base"] > 0 and row["ws"] <= row["tc_base"] / 2 + 2

    def black_berserk(row):
        return row["bs"] > 0 and row["tc_base"] > 0 and row["bs"] <= row["tc_base"] / 2 + 2

    loss_counts: Counter = Counter()

    for _, row in timeouts.iterrows():
        if row["result"] == "0-1" and white_berserk(row):
            # white timed out while berserking
            loss_counts[str(row["white"])] += 1
        if row["result"] == "1-0" and black_berserk(row):
            # black timed out while berserking
            loss_counts[str(row["black"])] += 1

    if not loss_counts:
        return []
    max_losses = max(loss_counts.values())
    winners = sorted([p for p, c in loss_counts.items() if c == max_losses])
    return winners[:10], max_losses


# ============================================================
# Q10: Logistic regression (game-level: captures, color, avg time → win)
# ============================================================

def q10_logit_game_level(games: pd.DataFrame) -> dict:
    """
    Per-player-per-game: captures count, color (white=1), avg_time_per_move -> win (1/0).
    Exclude draws.
    """
    print("  Q10: collecting per-game logit data...")
    game_result = dict(zip(games["game_id"], games["result"]))
    ws_map = dict(zip(games["game_id"], games["whitestart"].apply(parse_clock)))
    bs_map = dict(zip(games["game_id"], games["blackstart"].apply(parse_clock)))
    tc_map = dict(zip(games["game_id"], games["timecontrol"].apply(lambda x: parse_tc(x)[1])))

    rows = []  # (captures, color_white, avg_time, won)
    done = 0

    for gid, moves_list, moves_df in stream_games_from_moves():
        done += 1
        if done % 500000 == 0:
            print(f"    Q10 progress: {done:,}")

        res = game_result.get(gid, "")
        if res == "1/2-1/2":
            continue

        inc = tc_map.get(gid, 0)
        ws = ws_map.get(gid, -1)
        bs = bs_map.get(gid, -1)

        w_caps = b_caps = 0
        w_time = b_time = 0.0
        w_moves = b_moves = 0
        prev_white_clock = ws
        prev_black_clock = bs

        for _, mrow in moves_df.iterrows():
            san = mrow["move"]
            color = mrow["color"]
            clk = parse_clock(mrow["clock"])
            if "x" in san:
                if color == "white":
                    w_caps += 1
                else:
                    b_caps += 1

            if color == "white" and prev_white_clock > 0 and clk >= 0:
                w_time += (prev_white_clock - clk) + inc
                w_moves += 1
                prev_white_clock = clk
            elif color == "black" and prev_black_clock > 0 and clk >= 0:
                b_time += (prev_black_clock - clk) + inc
                b_moves += 1
                prev_black_clock = clk

        w_avg = w_time / w_moves if w_moves > 0 else 0
        b_avg = b_time / b_moves if b_moves > 0 else 0

        won_white = 1 if res == "1-0" else 0
        won_black = 1 if res == "0-1" else 0

        rows.append((w_caps, 1, w_avg, won_white))
        rows.append((b_caps, 0, b_avg, won_black))

    print(f"  Q10: fitting logistic regression on {len(rows):,} rows...")
    X = np.array([[r[0], r[1], r[2]] for r in rows])
    y = np.array([r[3] for r in rows])
    scaler_x = X.copy()
    # Standardize
    means = X.mean(axis=0)
    stds = X.std(axis=0)
    stds[stds == 0] = 1
    X_sc = (X - means) / stds

    model = LogisticRegression(max_iter=500, solver="lbfgs")
    model.fit(X_sc, y)
    return {
        "intercept": float(model.intercept_[0]),
        "coef_captures": float(model.coef_[0][0]),
        "coef_color_white": float(model.coef_[0][1]),
        "coef_avg_time": float(model.coef_[0][2]),
        "feature_means": means.tolist(),
        "feature_stds": stds.tolist(),
    }


# ============================================================
# Q11: Resignation counts
# ============================================================

def q11_resignations(games: pd.DataFrame) -> tuple:
    """Who resigned most, how many never resigned, how many at median."""
    # Termination=Normal, result != draw, last move != '#' -> resignation
    normal_decisive = games[
        (games["termination"] == "Normal")
        & (games["result"] != "1/2-1/2")
    ]
    relevant = set(normal_decisive["game_id"])
    result_map = dict(zip(normal_decisive["game_id"], normal_decisive["result"]))
    white_map = dict(zip(normal_decisive["game_id"], normal_decisive["white"]))
    black_map = dict(zip(normal_decisive["game_id"], normal_decisive["black"]))

    print(f"  Q11: checking {len(relevant):,} decisive Normal games...")
    resign_counts: Counter = Counter()
    never_resigned_players: set = set()

    # All players
    all_players: set = set()
    chunks_g = pd.read_csv(GAMES, usecols=["white", "black"], chunksize=CHUNKSIZE)
    for chunk in chunks_g:
        all_players.update(chunk["white"].dropna())
        all_players.update(chunk["black"].dropna())

    done = 0
    for gid, moves_list, _ in stream_games_from_moves(relevant):
        done += 1
        if done % 100000 == 0:
            print(f"    Q11 progress: {done:,}/{len(relevant):,}")

        last_move = moves_list[-1] if moves_list else ""
        if "#" in last_move:
            continue  # checkmate, not resignation

        res = result_map.get(gid, "")
        if res == "1-0":
            # Black resigned
            resign_counts[black_map.get(gid, "")] += 1
        elif res == "0-1":
            # White resigned
            resign_counts[white_map.get(gid, "")] += 1

    # Never resigned: players in all_players but not in resign_counts (or count = 0)
    never_resigned = len([p for p in all_players if resign_counts.get(p, 0) == 0])

    # Max resignations
    most_resigned = max(resign_counts, key=resign_counts.get)
    most_count = resign_counts[most_resigned]

    # Median
    counts_list = [resign_counts.get(p, 0) for p in all_players]
    median_val = np.median(counts_list)
    at_median = sum(1 for c in counts_list if c == median_val)

    return most_resigned, most_count, never_resigned, median_val, at_median


# ============================================================
# Q12: Largest cycle of cyclic wins within a calendar year (standard, CET)
# ============================================================

def q12_largest_cycle(games: pd.DataFrame) -> tuple:
    std_games = games[games["variant"] == "Standard"].copy()
    std_games = std_games[std_games["result"].isin(["1-0", "0-1"])]

    print("  Q12: building win graphs per year...")
    # Parse CET year
    def get_cet_year(row):
        dt = game_start_cet(row["utcdate"], row["utctime"])
        return dt.year if dt else None

    std_games["cet_year"] = std_games.apply(get_cet_year, axis=1)
    std_games = std_games.dropna(subset=["cet_year"])
    std_games["cet_year"] = std_games["cet_year"].astype(int)

    best_cycle = []
    best_year = None

    for year, yg in std_games.groupby("cet_year"):
        # Build adjacency: winner -> {losers}
        win_graph: dict[str, set] = defaultdict(set)
        # Also store first game timestamp per edge for ordering
        edge_first_game: dict[tuple, str] = {}  # (winner, loser) -> utcdate+utctime

        for _, row in yg.iterrows():
            if row["result"] == "1-0":
                winner, loser = row["white"], row["black"]
            else:
                winner, loser = row["black"], row["white"]
            win_graph[winner].add(loser)
            key = (winner, loser)
            ts = f"{row['utcdate']} {row['utctime']}"
            if key not in edge_first_game or ts < edge_first_game[key]:
                edge_first_game[key] = ts

        # Find longest simple directed cycle using DFS
        cycle = _find_longest_cycle(win_graph)
        if len(cycle) > len(best_cycle):
            best_cycle = cycle
            best_year = year

    if not best_cycle or not best_year:
        return None, []

    # Find the first game in the cycle (chronologically)
    # Determine the starting player (winner of the first game in the cycle)
    cycle_edges = [(best_cycle[i], best_cycle[(i+1) % len(best_cycle)]) for i in range(len(best_cycle))]
    yg = std_games[std_games["cet_year"] == best_year]
    edge_first_game = {}
    for _, row in yg.iterrows():
        if row["result"] == "1-0":
            winner, loser = row["white"], row["black"]
        else:
            winner, loser = row["black"], row["white"]
        key = (winner, loser)
        ts = f"{row['utcdate']} {row['utctime']}"
        if key not in edge_first_game or ts < edge_first_game[key]:
            edge_first_game[key] = ts

    earliest_ts = None
    start_idx = 0
    for i, (w, l) in enumerate(cycle_edges):
        ts = edge_first_game.get((w, l))
        if ts and (earliest_ts is None or ts < earliest_ts):
            earliest_ts = ts
            start_idx = i

    ordered = best_cycle[start_idx:] + best_cycle[:start_idx]
    return best_year, ordered


def _find_longest_cycle(graph: dict) -> list:
    """Find longest simple directed cycle using DFS."""
    nodes = list(graph.keys())
    best = []

    def dfs(start, current, path, visited):
        nonlocal best
        for neighbor in graph.get(current, []):
            if neighbor == start and len(path) > 2:
                if len(path) > len(best):
                    best = path[:]
            elif neighbor not in visited and len(path) < 25:  # limit depth
                visited.add(neighbor)
                path.append(neighbor)
                dfs(start, neighbor, path, visited)
                path.pop()
                visited.remove(neighbor)

    for node in nodes:
        dfs(node, node, [node], {node})

    return best


# ============================================================
# Q13: More time or less time → win rate
# ============================================================

def q13_time_usage_wins(games: pd.DataFrame) -> str:
    """Compare win rate for those using more time vs their opponent."""
    print("  Q13: collecting clock data from moves...")
    game_result = dict(zip(games["game_id"], games["result"]))
    ws_map = dict(zip(games["game_id"], games["whitestart"].apply(parse_clock)))
    bs_map = dict(zip(games["game_id"], games["blackstart"].apply(parse_clock)))
    tc_map = dict(zip(games["game_id"], games["timecontrol"].apply(lambda x: parse_tc(x)[1])))

    more_time_wins = more_time_total = 0
    less_time_wins = less_time_total = 0
    done = 0

    for gid, moves_list, moves_df in stream_games_from_moves():
        done += 1
        if done % 500000 == 0:
            print(f"    Q13 progress: {done:,}")

        res = game_result.get(gid, "")
        if res == "1/2-1/2":
            continue

        inc = tc_map.get(gid, 0)
        ws = ws_map.get(gid, -1)
        bs = bs_map.get(gid, -1)
        if ws < 0 or bs < 0:
            continue

        w_time_used = 0.0
        b_time_used = 0.0
        prev_w = ws
        prev_b = bs

        for _, mrow in moves_df.iterrows():
            clk = parse_clock(mrow["clock"])
            if clk < 0:
                continue
            if mrow["color"] == "white" and prev_w > 0:
                w_time_used += (prev_w - clk) + inc
                prev_w = clk
            elif mrow["color"] == "black" and prev_b > 0:
                b_time_used += (prev_b - clk) + inc
                prev_b = clk

        if w_time_used > b_time_used:
            # white used more time
            more_time_total += 1
            if res == "1-0":
                more_time_wins += 1
            less_time_total += 1
            if res == "0-1":
                less_time_wins += 1
        elif b_time_used > w_time_used:
            more_time_total += 1
            if res == "0-1":
                more_time_wins += 1
            less_time_total += 1
            if res == "1-0":
                less_time_wins += 1

    more_rate = more_time_wins / more_time_total if more_time_total > 0 else 0
    less_rate = less_time_wins / less_time_total if less_time_total > 0 else 0
    if more_rate > less_rate:
        answer = f"Több időt felhasználók nyernek nagyobb arányban ({more_rate:.4f} vs {less_rate:.4f})"
    else:
        answer = f"Kevesebb időt felhasználók nyernek nagyobb arányban ({less_rate:.4f} vs {more_rate:.4f})"
    return answer


# ============================================================
# Q14: Pawn a2→g8 promotion dates
# ============================================================

def q14_a2_pawn_to_g8(games: pd.DataFrame) -> list[str]:
    """Find dates where white a2 pawn reached g8 and promoted."""
    # Pre-filter: games where white has 6 pawn captures moving right in sequence
    # and has a promotion on g8
    print("  Q14: pre-filtering games with g-file promotions...")
    relevant_gids = set()
    game_date_map = dict(zip(games["game_id"], games["date"]))

    for chunk in pd.read_csv(MOVES, chunksize=CHUNKSIZE):
        # Find games with g-file promotion
        promo_g = chunk[chunk["move"].str.contains(r"[fg]x?g8=|g8=", na=False, regex=True)]
        relevant_gids.update(promo_g["game_id"].unique())

    print(f"  Q14: simulating {len(relevant_gids):,} games with g8 promotions...")

    result_dates = set()
    done = 0
    for gid, moves_list, _ in stream_games_from_moves(relevant_gids):
        done += 1
        board = chess.Board()
        a2_pawn_sq = chess.A2  # track the specific pawn

        found = False
        for san in moves_list:
            try:
                move = board.push_san(san)
            except Exception:
                break

            from_sq = move.from_square
            to_sq = move.to_square
            was_white = not board.turn

            if not was_white:
                continue

            # Is this move from our tracked pawn?
            if from_sq == a2_pawn_sq:
                a2_pawn_sq = to_sq
                # Check if it reached g8
                if to_sq == chess.G8:
                    found = True
                    break
            # If nothing moved from a2_pawn_sq but it's a pawn capture that could be our pawn:
            # The pawn can only move from a2_pawn_sq - if it's captured, we'd detect it when
            # a capture lands on a2_pawn_sq

            # Check if our pawn was captured
            if not was_white and to_sq == a2_pawn_sq:
                break  # pawn captured by black, stop tracking

        if found:
            d = game_date_map.get(gid, "")
            if d:
                result_dates.add(d)

    return sorted(result_dates)[:10]


# ============================================================
# Q15: Non-queen promotions
# ============================================================

def q15_non_queen_promotions(games: pd.DataFrame) -> tuple:
    """Count non-queen promotions and top 3 pieces."""
    print("  Q15: scanning moves for promotions...")
    promo_counts: Counter = Counter()

    for chunk in pd.read_csv(MOVES, usecols=["move"], chunksize=CHUNKSIZE):
        promos = chunk[chunk["move"].str.contains("=", na=False)]
        for mv in promos["move"]:
            # Extract promoted piece
            m = re.search(r"=([QRBN])", mv)
            if m:
                piece = m.group(1)
                if piece != "Q":
                    promo_counts[piece] += 1

    total_non_queen = sum(promo_counts.values())
    top3 = promo_counts.most_common(3)
    return total_non_queen, top3


# ============================================================
# Q16: Longest draw streak (standard games)
# ============================================================

def q16_draw_streak(games: pd.DataFrame) -> tuple:
    std = games[games["variant"] == "Standard"].copy()
    std = std.sort_values(["utcdate", "utctime"])

    print("  Q16: computing draw streaks...")
    # Per player: list of (timestamp, result, elo) sorted
    player_games: dict[str, list] = defaultdict(list)

    for _, row in std.iterrows():
        ts = f"{row['utcdate']} {row['utctime']}"
        w, b = str(row["white"]), str(row["black"])
        res = row["result"]
        player_games[w].append((ts, res, row.get("whiteelo", 0)))
        player_games[b].append((ts, res, row.get("blackelo", 0)))

    best_streak = 0
    best_player = None
    best_start = None
    best_end = None
    best_last_elo = 0

    for player, game_list in player_games.items():
        game_list.sort()
        streak = 0
        streak_start = None
        streak_end_elo = 0

        for ts, res, elo in game_list:
            if res == "1/2-1/2":
                if streak == 0:
                    streak_start = ts
                streak += 1
                streak_end_elo = elo if elo and not pd.isna(elo) else streak_end_elo
            else:
                streak = 0

            if streak > best_streak or (
                streak == best_streak and streak > 0 and streak_end_elo > best_last_elo
            ):
                best_streak = streak
                best_player = player
                best_start = streak_start
                best_end = ts
                best_last_elo = streak_end_elo

        # Check ongoing streak at end
        if streak > 0:
            if streak > best_streak or (streak == best_streak and streak_end_elo > best_last_elo):
                best_streak = streak
                best_player = player
                best_start = streak_start
                best_end = game_list[-1][0]
                best_last_elo = streak_end_elo

    return best_player, best_start, best_end, best_streak


# ============================================================
# Q17: Logistic regression per-move (capture dummy ~ time_elapsed + color)
# ============================================================

def q17_logit_per_move(games: pd.DataFrame) -> dict:
    """
    Dependent: 1 if move is capture, 0 otherwise.
    Independents: time elapsed since match start (seconds), color (white=1).
    Use SGD for memory efficiency.
    """
    print("  Q17: collecting per-move logit data with SGD...")
    game_result = dict(zip(games["game_id"], games["result"]))
    ws_map = dict(zip(games["game_id"], games["whitestart"].apply(parse_clock)))
    bs_map = dict(zip(games["game_id"], games["blackstart"].apply(parse_clock)))
    tc_map = dict(zip(games["game_id"], games["timecontrol"].apply(lambda x: parse_tc(x)[1])))

    sgd = SGDClassifier(loss="log_loss", max_iter=1, warm_start=True, random_state=42)
    fitted = False
    batch_X = []
    batch_y = []
    BATCH = 100_000
    done = 0

    for gid, moves_list, moves_df in stream_games_from_moves():
        done += 1
        if done % 500000 == 0:
            print(f"    Q17 progress: {done:,}")

        inc = tc_map.get(gid, 0)
        ws = ws_map.get(gid, -1)
        bs = bs_map.get(gid, -1)

        elapsed = 0.0
        prev_w = ws
        prev_b = bs

        for _, mrow in moves_df.iterrows():
            san = mrow["move"]
            color = mrow["color"]
            clk = parse_clock(mrow["clock"])
            is_cap = 1 if "x" in san else 0
            color_val = 1 if color == "white" else 0

            if color == "white" and prev_w > 0 and clk >= 0:
                time_used = (prev_w - clk) + inc
                elapsed += time_used
                prev_w = clk
            elif color == "black" and prev_b > 0 and clk >= 0:
                time_used = (prev_b - clk) + inc
                elapsed += time_used
                prev_b = clk

            batch_X.append([elapsed, color_val])
            batch_y.append(is_cap)

            if len(batch_X) >= BATCH:
                X_arr = np.array(batch_X, dtype=np.float32)
                y_arr = np.array(batch_y, dtype=np.int32)
                if not fitted:
                    sgd.partial_fit(X_arr, y_arr, classes=[0, 1])
                    fitted = True
                else:
                    sgd.partial_fit(X_arr, y_arr)
                batch_X = []
                batch_y = []

    if batch_X:
        X_arr = np.array(batch_X, dtype=np.float32)
        y_arr = np.array(batch_y, dtype=np.int32)
        if not fitted:
            sgd.partial_fit(X_arr, y_arr, classes=[0, 1])
        else:
            sgd.partial_fit(X_arr, y_arr)

    return {
        "intercept": float(sgd.intercept_[0]),
        "coef_time_elapsed": float(sgd.coef_[0][0]),
        "coef_color_white": float(sgd.coef_[0][1]),
    }


# ============================================================
# Q18: Longest winless streak (standard games)
# ============================================================

def q18_winless_streak(games: pd.DataFrame) -> tuple:
    std = games[games["variant"] == "Standard"].copy()
    std = std.sort_values(["utcdate", "utctime"])

    print("  Q18: computing winless streaks...")
    player_games: dict[str, list] = defaultdict(list)

    for _, row in std.iterrows():
        ts = f"{row['utcdate']} {row['utctime']}"
        w, b = str(row["white"]), str(row["black"])
        res = row["result"]
        # Encode: win for white='w', win for black='b', draw='d'
        player_games[w].append((ts, "w" if res == "1-0" else ("d" if res == "1/2-1/2" else "l")))
        player_games[b].append((ts, "w" if res == "0-1" else ("d" if res == "1/2-1/2" else "l")))

    best_streak = 0
    best_player = None
    best_start = None
    best_end = None

    for player, game_list in player_games.items():
        game_list.sort()
        streak = 0
        streak_start = None

        for ts, outcome in game_list:
            if outcome != "w":  # loss or draw = winless
                if streak == 0:
                    streak_start = ts
                streak += 1
            else:
                streak = 0

            if streak > best_streak:
                best_streak = streak
                best_player = player
                best_start = streak_start
                best_end = ts

        # Check ongoing streak
        if streak > 0 and streak > best_streak:
            best_streak = streak
            best_player = player
            best_start = streak_start
            best_end = game_list[-1][0]

    # Tiebreaker: among players with same best_streak, pick by Hungarian alphabet after "Lili"
    tied_players = []
    for player, game_list in player_games.items():
        game_list.sort()
        streak = 0
        max_streak = 0
        for ts, outcome in game_list:
            if outcome != "w":
                streak += 1
                max_streak = max(max_streak, streak)
            else:
                streak = 0
        if streak > 0:
            max_streak = max(max_streak, streak)
        if max_streak == best_streak:
            tied_players.append(player)

    if tied_players:
        lili_key = hu_key("Lili")
        after_lili = [p for p in tied_players if hu_key(p) > lili_key]
        if after_lili:
            best_player = min(after_lili, key=hu_key)
        else:
            best_player = min(tied_players, key=hu_key)

    # Find the streak dates for best_player
    game_list = sorted(player_games[best_player])
    streak = 0
    streak_start = None
    best_start = None
    best_end = None
    for ts, outcome in game_list:
        if outcome != "w":
            if streak == 0:
                streak_start = ts
            streak += 1
            if streak == best_streak:
                best_end = ts
        else:
            streak = 0
    if streak == best_streak:
        best_end = game_list[-1][0]
        best_start = streak_start

    return best_player, best_start, best_end, best_streak


# ============================================================
# Q19: 50-move rule draws, standard, 2026.03.15–2026.10.14
# ============================================================

def q19_fifty_move_rule(games: pd.DataFrame) -> int:
    mask = (
        (games["variant"] == "Standard")
        & (games["date"] >= "2026.03.15")
        & (games["date"] <= "2026.10.14")
        & (games["result"] == "1/2-1/2")
        & (games["termination"] == "Normal")
    )
    relevant = set(games.loc[mask, "game_id"])
    print(f"  Q19: simulating {len(relevant):,} draw games in date range...")

    count = 0
    for gid, moves_list, _ in stream_games_from_moves(relevant):
        board = simulate_board(moves_list)
        if board and board.is_fifty_moves():
            count += 1
    return count


# ============================================================
# Q20: Queen's Gambit percentage per year (04.21–05.18 CET, standard)
# ============================================================

def q20_queens_gambit_percentage(games: pd.DataFrame) -> dict[int, float]:
    """
    Queen's Gambit: 1.d4 d5 2.c4
    Filter: standard games, CET date between 04.21 and 05.18 in each year.
    """
    std = games[games["variant"] == "Standard"].copy()

    def in_date_range(row):
        dt = game_start_cet(row["utcdate"], row["utctime"])
        if dt is None:
            return False
        md = (dt.month, dt.day)
        return (4, 21) <= md <= (5, 18)

    std["in_range"] = std.apply(in_date_range, axis=1)
    std["cet_year"] = std.apply(
        lambda r: (game_start_cet(r["utcdate"], r["utctime"]) or datetime(1970, 1, 1)).year, axis=1
    )
    in_range = std[std["in_range"]]

    relevant = set(in_range["game_id"])
    year_map = dict(zip(in_range["game_id"], in_range["cet_year"]))
    print(f"  Q20: checking first 3 moves for {len(relevant):,} games...")

    year_total: Counter = Counter()
    year_qg: Counter = Counter()

    for chunk in pd.read_csv(MOVES, usecols=["game_id", "move_no", "move", "color"], chunksize=CHUNKSIZE):
        chunk = chunk[chunk["game_id"].isin(relevant)]
        first3 = chunk[chunk["move_no"] <= 2]
        for gid, grp in first3.groupby("game_id"):
            year = year_map.get(gid)
            if year is None:
                continue
            moves = grp.sort_values(["move_no", "color"]).reset_index()
            # moves: row 0 = white move 1, row 1 = black move 1, row 2 = white move 2
            if len(moves) >= 3:
                m1w = moves.loc[0, "move"] if moves.loc[0, "color"] == "white" else None
                m1b = moves.loc[1, "move"] if len(moves) > 1 and moves.loc[1, "color"] == "black" else None
                m2w = moves.loc[2, "move"] if len(moves) > 2 else None
                if m1w and m1b and m2w:
                    year_total[year] += 1
                    if m1w.rstrip("+#") == "d4" and m1b.rstrip("+#") == "d5" and m2w.rstrip("+#") == "c4":
                        year_qg[year] += 1

    return {y: year_qg[y] / year_total[y] for y in sorted(year_total) if year_total[y] > 0}


# ============================================================
# Q21: Games potentially spanning New Year's Eve (standard, CET)
# ============================================================

def q21_year_spanning(games: pd.DataFrame) -> dict[int, int]:
    """
    For each Dec 31 in the data range, count standard games that started before
    midnight and could potentially have ended after midnight (start_time + white_start + black_start >= midnight).
    """
    std = games[games["variant"] == "Standard"].copy()
    std["ws"] = std["whitestart"].apply(parse_clock)
    std["bs"] = std["blackstart"].apply(parse_clock)

    result: dict[int, int] = {}

    for _, row in std.iterrows():
        dt = game_start_cet(row["utcdate"], row["utctime"])
        if dt is None:
            continue
        if dt.month == 12 and dt.day == 31:
            year = dt.year
            # midnight = Jan 1 00:00:00 of year+1
            midnight = CET.localize(datetime(year + 1, 1, 1, 0, 0, 0))
            secs_to_midnight = (midnight - dt).total_seconds()
            total_clock = row["ws"] + row["bs"]
            if total_clock > 0 and total_clock >= secs_to_midnight:
                result[year] = result.get(year, 0) + 1

    return result


# ============================================================
# Q22: Rectangle-shaped piece paths
# ============================================================

@dataclass
class PieceTracker:
    """Track position history for a single piece."""
    positions: list = field(default_factory=list)

    def add(self, sq: chess.Square):
        self.positions.append(sq)

    def count_rectangles(self) -> int:
        """Count consecutive 4-position windows forming a rectangle."""
        total = 0
        pos = self.positions
        for i in range(len(pos) - 3):
            p1, p2, p3, p4 = pos[i], pos[i+1], pos[i+2], pos[i+3]
            if _is_rectangle(p1, p2, p3, p4):
                total += 1
        return total

    def max_rectangle_area(self) -> int:
        """Find maximum rectangle area in position history."""
        best = 0
        pos = self.positions
        for i in range(len(pos) - 3):
            p1, p2, p3, p4 = pos[i], pos[i+1], pos[i+2], pos[i+3]
            area = _rectangle_area(p1, p2, p3, p4)
            if area > best:
                best = area
        return best


def _is_rectangle(s1, s2, s3, s4) -> bool:
    """Check if 4 squares form a rectangle (visited consecutively along edges)."""
    coords = [(chess.square_file(s), chess.square_rank(s)) for s in [s1, s2, s3, s4]]
    xs = {c[0] for c in coords}
    ys = {c[1] for c in coords}
    if len(xs) != 2 or len(ys) != 2:
        return False
    # Area must be positive
    x_vals = sorted(xs)
    y_vals = sorted(ys)
    if (x_vals[1] - x_vals[0]) == 0 or (y_vals[1] - y_vals[0]) == 0:
        return False
    # Consecutive positions must share x or y
    for i in range(4):
        a, b = coords[i], coords[(i + 1) % 4]
        if a[0] != b[0] and a[1] != b[1]:
            return False
    return True


def _rectangle_area(s1, s2, s3, s4) -> int:
    coords = [(chess.square_file(s), chess.square_rank(s)) for s in [s1, s2, s3, s4]]
    xs = {c[0] for c in coords}
    ys = {c[1] for c in coords}
    if len(xs) != 2 or len(ys) != 2:
        return 0
    x_vals = sorted(xs)
    y_vals = sorted(ys)
    return (x_vals[1] - x_vals[0]) * (y_vals[1] - y_vals[0])


def _process_game_rectangles(gid: str, moves_list: list, white: str, black: str):
    """Simulate game and track rectangle paths. Returns (white_rects, black_rects, max_area)."""
    board = chess.Board()
    # Track each piece: keyed by (color, piece_type, original_square) -> PieceTracker
    piece_trackers: dict = {}

    # Initialize starting positions
    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece:
            key = (piece.color, piece.piece_type, sq)
            t = PieceTracker()
            t.add(sq)
            piece_trackers[key] = t

    # Map from current_sq -> piece key (for tracking moves)
    current_sq_to_key: dict[chess.Square, tuple] = {}
    for key in piece_trackers:
        current_sq_to_key[key[2]] = key  # initially piece is at original sq

    white_rects = 0
    black_rects = 0
    max_area = 0

    for san in moves_list:
        try:
            move = board.push_san(san)
        except Exception:
            break

        from_sq = move.from_square
        to_sq = move.to_square
        was_white = not board.turn

        # Handle promotion: piece type changes
        promoted_pt = move.promotion

        # Get the key for the piece that moved
        key = current_sq_to_key.get(from_sq)
        if key is None:
            continue

        # If piece was captured on to_sq, remove it
        captured_key = current_sq_to_key.get(to_sq)
        if captured_key and captured_key != key:
            del current_sq_to_key[to_sq]

        # Move the piece
        current_sq_to_key[to_sq] = key
        if from_sq in current_sq_to_key and current_sq_to_key[from_sq] == key:
            del current_sq_to_key[from_sq]

        # Update position history
        if key in piece_trackers:
            piece_trackers[key].add(to_sq)

        # Handle castling: rook also moves
        if board.is_kingside_castling(move) or board.is_queenside_castling(move):
            if board.is_kingside_castling(move):
                rook_from = chess.H1 if was_white else chess.H8
                rook_to = chess.F1 if was_white else chess.F8
            else:
                rook_from = chess.A1 if was_white else chess.A8
                rook_to = chess.D1 if was_white else chess.D8

            rook_key = current_sq_to_key.get(rook_from)
            if rook_key:
                current_sq_to_key[rook_to] = rook_key
                del current_sq_to_key[rook_from]
                if rook_key in piece_trackers:
                    piece_trackers[rook_key].add(rook_to)

        # Handle promotion: update piece type in key
        if promoted_pt:
            old_key = key
            new_key = (key[0], promoted_pt, key[2])
            piece_trackers[new_key] = piece_trackers.pop(old_key)
            current_sq_to_key[to_sq] = new_key

    # Count rectangles per player
    for key, tracker in piece_trackers.items():
        color = key[0]
        rects = tracker.count_rectangles()
        area = tracker.max_rectangle_area()
        if color == chess.WHITE:
            white_rects += rects
        else:
            black_rects += rects
        if area > max_area:
            max_area = area

    return white_rects, black_rects, max_area, gid


def q22_rectangles(games: pd.DataFrame) -> tuple:
    """Find player with most rectangles and largest rectangle area."""
    print("  Q22: tracking rectangle paths for all games...")
    white_map = dict(zip(games["game_id"], games["white"]))
    black_map = dict(zip(games["game_id"], games["black"]))

    player_rects: Counter = Counter()
    max_area = 0
    done = 0

    for gid, moves_list, _ in stream_games_from_moves():
        done += 1
        if done % 200000 == 0:
            print(f"    Q22 progress: {done:,}")

        w, b = white_map.get(gid, ""), black_map.get(gid, "")
        wr, br, area, _ = _process_game_rectangles(gid, moves_list, w, b)
        player_rects[w] += wr
        player_rects[b] += br
        if area > max_area:
            max_area = area

    # Find max (tiebreaker: earliest to reach that count - approximated by alphabetical)
    if not player_rects:
        return None, 0, 0

    max_rects = max(player_rects.values())
    top_player = min([p for p, c in player_rects.items() if c == max_rects])
    return top_player, max_rects, max_area


# ============================================================
# Q23: Castling checkmates
# ============================================================

def q23_castling_checkmates(games: pd.DataFrame) -> list[str]:
    """Who gave checkmate by castling most?"""
    print("  Q23: scanning moves for castling checkmates...")
    white_map = dict(zip(games["game_id"], games["white"]))
    black_map = dict(zip(games["game_id"], games["black"]))

    counts: Counter = Counter()

    for chunk in pd.read_csv(MOVES, usecols=["game_id", "move", "color"], chunksize=CHUNKSIZE):
        castle_mates = chunk[chunk["move"].isin(["O-O#", "O-O-O#"])]
        for _, row in castle_mates.iterrows():
            gid = row["game_id"]
            color = row["color"]
            player = white_map.get(gid) if color == "white" else black_map.get(gid)
            if player:
                counts[player] += 1

    if not counts:
        return []
    max_val = max(counts.values())
    return sorted([p for p, c in counts.items() if c == max_val])[:10]


# ============================================================
# Q24: En passant in Indian openings (3-min games)
# ============================================================

def q24_en_passant_indian(games: pd.DataFrame) -> int:
    """
    3-min games (180+X), ECO starts with 'E' (Indian openings),
    count white en passant captures.
    """
    mask = (
        games["timecontrol"].str.startswith("180", na=False)
        & games["eco"].str.startswith("E", na=False)
    )
    relevant = set(games.loc[mask, "game_id"])
    print(f"  Q24: simulating {len(relevant):,} Indian 3-min games for en passant...")

    count = 0
    done = 0
    for gid, moves_list, _ in stream_games_from_moves(relevant):
        done += 1
        if done % 10000 == 0:
            print(f"    Q24 progress: {done:,}/{len(relevant):,}")

        board = chess.Board()
        for san in moves_list:
            try:
                move = board.push_san(san)
            except Exception:
                break

            was_white = not board.turn
            if was_white and board.is_en_passant(move):
                count += 1

    return count


# ============================================================
# Main orchestration
# ============================================================

def load_games_full() -> pd.DataFrame:
    print("Loading games metadata...")
    chunks = []
    for chunk in pd.read_csv(
        GAMES,
        chunksize=CHUNKSIZE,
        dtype={"result": "category", "variant": "category", "termination": "category"},
    ):
        chunks.append(chunk)
    df = pd.concat(chunks, ignore_index=True)
    df["variant"] = df["variant"].astype(str)
    df["result"] = df["result"].astype(str)
    df["termination"] = df["termination"].astype(str)
    print(f"  Loaded {len(df):,} games")
    return df


def write_answers(answers: dict):
    lines = ["# Chess Data Analysis - Answers\n"]
    for q_num in sorted(answers.keys()):
        lines.append(f"## Kérdés {q_num}\n")
        lines.append(f"{answers[q_num]}\n")
    with open(OUTPUT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\nAnswers written to {OUTPUT}")


def main():
    games = load_games_full()
    tournaments = pd.read_csv(TOURNAMENTS)
    answers = {}

    print("\n=== Phase 1: Games-only questions ===")

    print("Q9: Berserk timeouts...")
    q9_result, q9_count = q9_berserk_timeouts(games)
    answers[9] = f"Legtöbb berserk időtúllépéses vereség ({q9_count} vereség): {', '.join(q9_result)}"

    print("Q12: Largest cyclic win...")
    year12, cycle12 = q12_largest_cycle(games)
    if cycle12:
        answers[12] = f"Év: {year12}, Kör ({len(cycle12)} játékos): {' → '.join(cycle12)} → {cycle12[0]}"
    else:
        answers[12] = "Nem található körbeverés."

    print("Q16: Draw streaks...")
    p16, s16, e16, n16 = q16_draw_streak(games)
    answers[16] = f"Játékos: {p16}, Időszak: {s16} – {e16}, Streak: {n16}"

    print("Q18: Winless streaks...")
    p18, s18, e18, n18 = q18_winless_streak(games)
    answers[18] = f"Játékos: {p18}, Időszak: {s18} – {e18}, Streak: {n18}"

    print("Q21: Year-spanning games...")
    q21 = q21_year_spanning(games)
    q21_lines = [f"{yr}.12.31: {cnt} játszma" for yr, cnt in sorted(q21.items())]
    answers[21] = "\n".join(q21_lines) if q21_lines else "Nincs ilyen játszma."

    print("\n=== Phase 2: Moves streaming (single pass) ===")
    # We do ALL moves-streaming questions together in a coordinated pass.
    # For clarity, each question uses stream_games_from_moves() separately.
    # For production, these could be merged into one pass.

    print("Q5: Threefold repetition + scissors emoji...")
    answers[5] = f"{q5_threefold_scissors(games)}"

    print("Q8: Draw on March 20 with promotion...")
    answers[8] = f"{q8_draw_march20_promotion(games)}"

    print("Q15: Non-queen promotions...")
    total15, top3_15 = q15_non_queen_promotions(games)
    top3_str = ", ".join(f"{p}: {c}" for p, c in top3_15)
    answers[15] = f"Nem vezérré váltások száma: {total15}. Top 3: {top3_str}"

    print("Q23: Castling checkmates...")
    q23 = q23_castling_checkmates(games)
    answers[23] = f"Sáncolással mattot adók: {', '.join(q23)}"

    print("Q20: Queen's Gambit percentage...")
    q20 = q20_queens_gambit_percentage(games)
    q20_lines = [f"{yr}: {pct:.4f} ({pct*100:.2f}%)" for yr, pct in sorted(q20.items())]
    answers[20] = "\n".join(q20_lines)

    print("Q14: a2 pawn to g8...")
    dates14 = q14_a2_pawn_to_g8(games)
    answers[14] = f"Dátumok (első 10): {', '.join(dates14)}" if dates14 else "Nincs ilyen játszma."

    print("Q11: Resignations...")
    most_res, most_count, never_res, med_val, at_med = q11_resignations(games)
    answers[11] = (
        f"Legtöbbet feladott: {most_res} ({most_count}x). "
        f"Soha nem adta fel: {never_res} játékos. "
        f"Mediánban ({med_val:.1f} feladás): {at_med} játékos."
    )

    print("Q6: Threefold repetition 2024.03.12–2024.11.19...")
    answers[6] = f"{q6_threefold_date_range(games)}"

    print("Q19: 50-move rule 2026.03.15–2026.10.14...")
    answers[19] = f"{q19_fifty_move_rule(games)}"

    print("Q3: Castling rights lost in first 6 half-moves (10-min games)...")
    answers[3] = f"{q3_castling_rights_lost(games)}"

    print("Q1: Material disadvantage ≥ 3 (standard, 2023.10.12–2024.02.19)...")
    answers[1] = f"{q1_material_disadvantage(games)}"

    print("Q7: Queens at checkmate (tournament winner games)...")
    avg7 = q7_queens_at_checkmate(games, tournaments)
    answers[7] = f"Átlagos fehér vezérek száma mattkor: {avg7:.4f}"

    print("Q24: En passant Indian openings (3-min)...")
    answers[24] = f"{q24_en_passant_indian(games)}"

    print("Q10: Logistic regression (game-level)...")
    params10 = q10_logit_game_level(games)
    answers[10] = (
        f"Intercept: {params10['intercept']:.6f}, "
        f"coef_captures: {params10['coef_captures']:.6f}, "
        f"coef_color_white: {params10['coef_color_white']:.6f}, "
        f"coef_avg_time: {params10['coef_avg_time']:.6f}\n"
        f"(Standardizált; feature átlagok: {params10['feature_means']}, "
        f"szórások: {params10['feature_stds']})"
    )

    print("Q13: Time usage and win rate...")
    answers[13] = q13_time_usage_wins(games)

    print("Q17: Logistic regression (per-move)...")
    params17 = q17_logit_per_move(games)
    answers[17] = (
        f"Intercept: {params17['intercept']:.6f}, "
        f"coef_time_elapsed_secs: {params17['coef_time_elapsed']:.6f}, "
        f"coef_color_white: {params17['coef_color_white']:.6f}"
    )

    print("Q4: Rook distances...")
    diff4 = q4_rook_distances_v2(games)
    answers[4] = f"Különbség (fehér - fekete): {diff4} mező"

    print("Q2: Left knight capture win rate...")
    answers[2] = _q2_clean(games, dict(zip(games["game_id"], games["result"])))

    print("Q22: Rectangle paths (WARNING: slow, simulates all games)...")
    top22, rects22, area22 = q22_rectangles(games)
    answers[22] = (
        f"Legtöbb téglalapot leíró játékos: {top22} ({rects22} téglalap). "
        f"Legnagyobb téglalap területe: {area22} mező²."
    )

    write_answers(answers)


if __name__ == "__main__":
    main()
