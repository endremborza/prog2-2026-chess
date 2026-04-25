#!/usr/bin/env python3
"""Chess tournament data analysis - answers all 24 questions."""
from __future__ import annotations

import re
from collections import defaultdict, Counter
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Iterator

import chess
import numpy as np
import pandas as pd
import pytz
from sklearn.linear_model import LogisticRegression, SGDClassifier

# ============================================================
# Config
# ============================================================

GAMES = "data/games.csv.gz"
MOVES = "data/moves.csv.gz"
TOURNAMENTS = "data/tournaments.csv.gz"
OUTPUT = "answers.md"

CET = pytz.timezone("Europe/Budapest")
UTC = pytz.utc
CHUNKSIZE = 300_000

# ============================================================
# Utilities
# ============================================================

def parse_clock(s) -> int:
    """H:MM:SS → seconds. -1 on error."""
    try:
        p = str(s).split(":")
        return int(p[0]) * 3600 + int(p[1]) * 60 + int(p[2])
    except Exception:
        return -1


def parse_tc(tc) -> tuple[int, int]:
    """'180+0' → (180, 0)."""
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


def simulate_board(moves_list: list[str]) -> Optional[chess.Board]:
    board = chess.Board()
    for san in moves_list:
        try:
            board.push_san(san)
        except Exception:
            return board
    return board


def count_material(board: chess.Board) -> tuple[int, int]:
    vals = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9}
    w = sum(len(board.pieces(pt, chess.WHITE)) * v for pt, v in vals.items())
    b = sum(len(board.pieces(pt, chess.BLACK)) * v for pt, v in vals.items())
    return w, b


# ============================================================
# Hungarian alphabet key for Q18
# ============================================================

_HU = {
    "a": 1, "á": 2, "b": 3, "c": 4, "cs": 5, "d": 6, "dz": 7, "dzs": 8,
    "e": 9, "é": 10, "f": 11, "g": 12, "gy": 13, "h": 14, "i": 15, "í": 16,
    "j": 17, "k": 18, "l": 19, "ly": 20, "m": 21, "n": 22, "ny": 23,
    "o": 24, "ó": 25, "ö": 26, "ő": 27, "p": 28, "q": 29, "r": 30,
    "s": 31, "sz": 32, "t": 33, "ty": 34, "u": 35, "ú": 36, "ü": 37, "ű": 38,
    "v": 39, "w": 40, "x": 41, "y": 42, "z": 43, "zs": 44,
}

def hu_key(name: str) -> list[int]:
    s = name.lower()
    result = []
    i = 0
    while i < len(s):
        for length in (3, 2, 1):
            if s[i:i+length] in _HU:
                result.append(_HU[s[i:i+length]])
                i += length
                break
        else:
            result.append(200 + ord(s[i]))
            i += 1
    return result


# ============================================================
# Move streaming
# ============================================================

def stream_games(game_ids: Optional[set] = None) -> Iterator[tuple[str, list[str], pd.DataFrame]]:
    """Yield (game_id, moves_list, moves_df) for each complete game.
    Reads all chunks; filters to game_ids if provided."""
    buf = pd.DataFrame()
    for chunk in pd.read_csv(MOVES, chunksize=CHUNKSIZE):
        combined = pd.concat([buf, chunk], ignore_index=True) if len(buf) else chunk
        last = combined["game_id"].iloc[-1]
        complete = combined[combined["game_id"] != last]
        buf = combined[combined["game_id"] == last]
        for gid, grp in complete.groupby("game_id", sort=False):
            if game_ids is None or gid in game_ids:
                yield gid, grp["move"].tolist(), grp
    if len(buf):
        for gid, grp in buf.groupby("game_id", sort=False):
            if game_ids is None or gid in game_ids:
                yield gid, grp["move"].tolist(), grp


# ============================================================
# Q1: Material disadvantage ≥ 3, standard, 2023.10.12–2024.02.19
# ============================================================

def q1_material_disadvantage(games: pd.DataFrame) -> int:
    mask = (
        (games["variant"] == "Standard")
        & (games["date"] >= "2023.10.12")
        & (games["date"] <= "2024.02.19")
        & (games["result"] != "1/2-1/2")
    )
    gids = set(games.loc[mask, "game_id"])
    result_map = dict(zip(games.loc[mask, "game_id"], games.loc[mask, "result"]))
    print(f"  Q1: simulating {len(gids):,} games")
    count = done = 0
    for gid, moves_list, _ in stream_games(gids):
        done += 1
        if done % 200_000 == 0:
            print(f"    Q1 {done:,}/{len(gids):,}")
        board = simulate_board(moves_list)
        if board is None:
            continue
        w, b = count_material(board)
        res = result_map.get(gid, "")
        if (res == "1-0" and w - b >= 3) or (res == "0-1" and b - w >= 3):
            count += 1
    return count


# ============================================================
# Q2: Left knight capture win rate
# ============================================================

def q2_left_knight_capture(games: pd.DataFrame) -> str:
    """White left knight = b1, black left knight = g8."""
    result_map = dict(zip(games["game_id"], games["result"]))
    lk_wins = lk_total = no_wins = no_total = 0
    done = 0
    for gid, moves_list, _ in stream_games():
        done += 1
        if done % 500_000 == 0:
            print(f"    Q2 {done:,}")
        res = result_map.get(gid, "")
        if res == "1/2-1/2":
            continue
        board = chess.Board()
        # Track left knight squares (None if captured/gone)
        wlk_sq: Optional[chess.Square] = chess.B1
        blk_sq: Optional[chess.Square] = chess.G8
        wlk_captured = blk_captured = False
        for san in moves_list:
            try:
                move = board.parse_san(san)
                is_cap = board.is_capture(move)
                is_ep = board.is_en_passant(move)
                mover = board.turn
                board.push(move)
            except Exception:
                break
            from_sq, to_sq = move.from_square, move.to_square
            if mover == chess.WHITE:
                # Did white left knight move?
                if wlk_sq is not None and from_sq == wlk_sq:
                    pt = board.piece_type_at(to_sq)
                    if pt == chess.KNIGHT:
                        if is_cap:
                            wlk_captured = True
                        wlk_sq = to_sq
                    else:
                        wlk_sq = None  # piece on that square is no longer a knight
                # Did white capture black's left knight?
                if blk_sq is not None and is_cap and to_sq == blk_sq:
                    blk_sq = None
            else:
                if blk_sq is not None and from_sq == blk_sq:
                    pt = board.piece_type_at(to_sq)
                    if pt == chess.KNIGHT:
                        if is_cap:
                            blk_captured = True
                        blk_sq = to_sq
                    else:
                        blk_sq = None
                if wlk_sq is not None and is_cap and to_sq == wlk_sq:
                    wlk_sq = None

        # White player stats
        if wlk_captured:
            lk_total += 1
            if res == "1-0":
                lk_wins += 1
        else:
            no_total += 1
            if res == "1-0":
                no_wins += 1
        # Black player stats
        if blk_captured:
            lk_total += 1
            if res == "0-1":
                lk_wins += 1
        else:
            no_total += 1
            if res == "0-1":
                no_wins += 1

    lk_rate = lk_wins / lk_total if lk_total else 0
    no_rate = no_wins / no_total if no_total else 0
    diff = lk_rate - no_rate
    direction = "nagyobb" if diff > 0 else "kisebb"
    return (f"Bal lóval ütők nyerési aránya: {lk_rate:.4f} ({lk_wins}/{lk_total}), "
            f"nem ütők: {no_rate:.4f} ({no_wins}/{no_total}), "
            f"különbség: {diff:+.4f} — bal lóval ütők {direction} arányban nyertek")


# ============================================================
# Q3: 10-min games, white loses castling in first 6 half-moves
# ============================================================

def q3_castling_rights_lost(games: pd.DataFrame) -> int:
    gids = set(games.loc[games["timecontrol"].str.startswith("600", na=False), "game_id"])
    print(f"  Q3: {len(gids):,} 10-min games")
    count = done = 0
    for gid, moves_list, _ in stream_games(gids):
        done += 1
        if done % 100_000 == 0:
            print(f"    Q3 {done:,}/{len(gids):,}")
        board = chess.Board()
        lost = False
        for i, san in enumerate(moves_list[:6]):
            had_rights = board.has_castling_rights(chess.WHITE)
            try:
                board.push_san(san)
            except Exception:
                break
            if i % 2 == 0 and had_rights and not board.has_castling_rights(chess.WHITE):
                lost = True
                break
        if lost:
            count += 1
    return count


# ============================================================
# Q4: Rook distance difference (white total − black total)
# ============================================================

def q4_rook_distances(games: pd.DataFrame) -> int:
    print("  Q4: computing rook distances for all games")
    white_dist = black_dist = done = 0
    for gid, moves_list, _ in stream_games():
        done += 1
        if done % 500_000 == 0:
            print(f"    Q4 {done:,}")
        board = chess.Board()
        for san in moves_list:
            try:
                move = board.parse_san(san)
                is_castle = board.is_castling(move)
                is_ks = board.is_kingside_castling(move) if is_castle else False
                mover = board.turn
                board.push(move)
            except Exception:
                break
            from_sq, to_sq = move.from_square, move.to_square
            if is_castle:
                # Rook also moves: compute rook from/to
                if mover == chess.WHITE:
                    rook_from, rook_to = (chess.H1, chess.F1) if is_ks else (chess.A1, chess.D1)
                else:
                    rook_from, rook_to = (chess.H8, chess.F8) if is_ks else (chess.A8, chess.D8)
                d = abs(chess.square_file(rook_to) - chess.square_file(rook_from))
                if mover == chess.WHITE:
                    white_dist += d
                else:
                    black_dist += d
                continue
            pt = board.piece_type_at(to_sq)
            if pt == chess.ROOK:
                d = (abs(chess.square_file(to_sq) - chess.square_file(from_sq)) +
                     abs(chess.square_rank(to_sq) - chess.square_rank(from_sq)))
                if mover == chess.WHITE:
                    white_dist += d
                else:
                    black_dist += d
    return white_dist - black_dist


# ============================================================
# Q5: Threefold repetition with scissors emoji in player name
# ============================================================

def q5_threefold_scissors(games: pd.DataFrame) -> int:
    scissors = re.compile(r"[✂✀✁✃]|✂")
    has_sc = games["white"].str.contains(scissors, na=False) | games["black"].str.contains(scissors, na=False)
    gids = set(games.loc[has_sc, "game_id"])
    if not gids:
        return 0
    result_map = dict(zip(games["game_id"], games["result"]))
    count = 0
    for gid, moves_list, _ in stream_games(gids):
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
    gids = set(games.loc[mask, "game_id"])
    print(f"  Q6: simulating {len(gids):,} draw games")
    count = done = 0
    for gid, moves_list, _ in stream_games(gids):
        done += 1
        if done % 50_000 == 0:
            print(f"    Q6 {done:,}/{len(gids):,}")
        board = simulate_board(moves_list)
        if board and board.is_repetition(3):
            count += 1
    return count


# ============================================================
# Q7: Average white queens at checkmate in tournament winner games
# ============================================================

def q7_queens_at_checkmate(games: pd.DataFrame, tournaments: pd.DataFrame) -> float:
    t = tournaments[["id", "winner__id"]].dropna()
    tour_winner = {row["id"]: str(row["winner__id"]).lower() for _, row in t.iterrows()}

    def winner_won(row):
        w = tour_winner.get(row["tournamentid"])
        if not w or row["termination"] != "Normal":
            return False
        return (row["result"] == "1-0" and str(row["white"]).lower() == w) or \
               (row["result"] == "0-1" and str(row["black"]).lower() == w)

    mask = games.apply(winner_won, axis=1)
    gids = set(games.loc[mask, "game_id"])
    print(f"  Q7: simulating {len(gids):,} tournament-winner games")
    counts = []
    done = 0
    for gid, moves_list, _ in stream_games(gids):
        done += 1
        if done % 20_000 == 0:
            print(f"    Q7 {done:,}/{len(gids):,}")
        if not moves_list or "#" not in moves_list[-1]:
            continue
        board = simulate_board(moves_list)
        if board:
            counts.append(len(board.pieces(chess.QUEEN, chess.WHITE)))
    return float(np.mean(counts)) if counts else 0.0


# ============================================================
# Q8: Draw on March 20 with pawn promotion as last move
# ============================================================

def q8_draw_march20_promotion(games: pd.DataFrame) -> int:
    gids = set(games.loc[
        (games["result"] == "1/2-1/2") & games["date"].str.endswith(".03.20", na=False),
        "game_id"
    ])
    print(f"  Q8: checking {len(gids):,} draw games on March 20")
    return sum(1 for _, ml, _ in stream_games(gids) if ml and "=" in ml[-1])


# ============================================================
# Q9: Berserk timeout losses (games only)
# ============================================================

def q9_berserk_timeouts(games: pd.DataFrame) -> tuple[list[str], int]:
    tf = games[games["termination"] == "Time forfeit"].copy()
    tf["tc_b"] = tf["timecontrol"].apply(lambda x: parse_tc(x)[0])
    tf["ws"] = tf["whitestart"].apply(parse_clock)
    tf["bs"] = tf["blackstart"].apply(parse_clock)
    counts: Counter = Counter()
    for _, row in tf.iterrows():
        tc = row["tc_b"]
        if tc <= 0:
            continue
        half = tc / 2
        if row["result"] == "0-1" and 0 < row["ws"] <= half + 2:
            counts[str(row["white"])] += 1
        if row["result"] == "1-0" and 0 < row["bs"] <= half + 2:
            counts[str(row["black"])] += 1
    if not counts:
        return [], 0
    max_v = max(counts.values())
    return sorted(p for p, c in counts.items() if c == max_v)[:10], max_v


# ============================================================
# Q10: Logistic regression — game-level captures, color, avg time → win
# ============================================================

def q10_logit_game(games: pd.DataFrame) -> dict:
    print("  Q10: collecting per-game features")
    result_map = dict(zip(games["game_id"], games["result"]))
    ws_map = dict(zip(games["game_id"], games["whitestart"].apply(parse_clock)))
    bs_map = dict(zip(games["game_id"], games["blackstart"].apply(parse_clock)))
    inc_map = dict(zip(games["game_id"], games["timecontrol"].apply(lambda x: parse_tc(x)[1])))

    rows: list[tuple] = []
    done = 0
    for gid, _, mdf in stream_games():
        done += 1
        if done % 500_000 == 0:
            print(f"    Q10 {done:,}")
        res = result_map.get(gid, "")
        if res == "1/2-1/2":
            continue
        inc = inc_map.get(gid, 0)
        ws, bs = ws_map.get(gid, -1), bs_map.get(gid, -1)
        wcap = bcap = 0
        wt = bt = 0.0
        wn = bn = 0
        pw, pb = ws, bs
        for _, r in mdf.iterrows():
            clk = parse_clock(r["clock"])
            c = r["color"]
            if "x" in str(r["move"]):
                if c == "white":
                    wcap += 1
                else:
                    bcap += 1
            if c == "white" and pw > 0 and clk >= 0:
                wt += (pw - clk) + inc; wn += 1; pw = clk
            elif c == "black" and pb > 0 and clk >= 0:
                bt += (pb - clk) + inc; bn += 1; pb = clk
        wa = wt / wn if wn else 0
        ba = bt / bn if bn else 0
        rows.append((wcap, 1, wa, 1 if res == "1-0" else 0))
        rows.append((bcap, 0, ba, 1 if res == "0-1" else 0))

    print(f"  Q10: fitting on {len(rows):,} rows")
    X = np.array([[r[0], r[1], r[2]] for r in rows], dtype=float)
    y = np.array([r[3] for r in rows])
    mu, sd = X.mean(0), X.std(0)
    sd[sd == 0] = 1
    Xs = (X - mu) / sd
    m = LogisticRegression(max_iter=1000, solver="lbfgs")
    m.fit(Xs, y)
    return {"intercept": float(m.intercept_[0]),
            "coef_captures": float(m.coef_[0][0]),
            "coef_color_white": float(m.coef_[0][1]),
            "coef_avg_time_per_move": float(m.coef_[0][2]),
            "feature_means": mu.tolist(), "feature_stds": sd.tolist()}


# ============================================================
# Q11: Resignations
# ============================================================

def q11_resignations(games: pd.DataFrame) -> tuple:
    normal = games[(games["termination"] == "Normal") & (games["result"] != "1/2-1/2")]
    gids = set(normal["game_id"])
    res_map = dict(zip(normal["game_id"], normal["result"]))
    white_map = dict(zip(normal["game_id"], normal["white"]))
    black_map = dict(zip(normal["game_id"], normal["black"]))
    print(f"  Q11: checking {len(gids):,} decisive Normal games")

    resign: Counter = Counter()
    played_decisive: set = set()

    done = 0
    for gid, moves_list, _ in stream_games(gids):
        done += 1
        if done % 100_000 == 0:
            print(f"    Q11 {done:,}/{len(gids):,}")
        last = moves_list[-1] if moves_list else ""
        res = res_map.get(gid, "")
        w, b = white_map.get(gid, ""), black_map.get(gid, "")
        played_decisive.add(w)
        played_decisive.add(b)
        if "#" not in last:  # resignation (not checkmate)
            if res == "1-0":
                resign[b] += 1
            elif res == "0-1":
                resign[w] += 1

    # All players who appeared in any game
    all_players: set = set()
    for chunk in pd.read_csv(GAMES, usecols=["white", "black"], chunksize=CHUNKSIZE):
        all_players.update(chunk["white"].dropna())
        all_players.update(chunk["black"].dropna())

    never = sum(1 for p in all_players if resign.get(p, 0) == 0)
    most = max(resign, key=resign.get) if resign else ""
    most_count = resign[most] if resign else 0
    counts_arr = [resign.get(p, 0) for p in all_players]
    med = float(np.median(counts_arr))
    at_med = sum(1 for c in counts_arr if c == med)
    return most, most_count, never, med, at_med


# ============================================================
# Q12: Largest cyclic win within a calendar year (standard, CET)
# ============================================================

def _find_longest_cycle(graph: dict[str, set]) -> list[str]:
    """DFS cycle search, depth-limited."""
    best: list[str] = []

    def dfs(start: str, cur: str, path: list[str], visited: set[str]):
        nonlocal best
        for nb in graph.get(cur, set()):
            if nb == start and len(path) >= 3:
                if len(path) > len(best):
                    best = path[:]
            elif nb not in visited and len(path) < 20:
                visited.add(nb)
                path.append(nb)
                dfs(start, nb, path, visited)
                path.pop()
                visited.remove(nb)

    for node in list(graph):
        dfs(node, node, [node], {node})
    return best


def q12_largest_cycle(games: pd.DataFrame) -> tuple:
    std = games[(games["variant"] == "Standard") & games["result"].isin(["1-0", "0-1"])].copy()
    print(f"  Q12: {len(std):,} standard decisive games")

    def cet_year(row):
        dt = game_start_cet(str(row["utcdate"]), str(row["utctime"]))
        return dt.year if dt else None

    std["yr"] = std.apply(cet_year, axis=1)
    std = std.dropna(subset=["yr"])
    std["yr"] = std["yr"].astype(int)

    best_cycle: list[str] = []
    best_year = None

    for year, yg in std.groupby("yr"):
        print(f"    Q12 year {year}: {len(yg):,} games")
        graph: dict[str, set] = defaultdict(set)
        edge_ts: dict[tuple, str] = {}
        for _, row in yg.iterrows():
            w, b = str(row["white"]), str(row["black"])
            winner, loser = (w, b) if row["result"] == "1-0" else (b, w)
            graph[winner].add(loser)
            key = (winner, loser)
            ts = f"{row['utcdate']} {row['utctime']}"
            if key not in edge_ts or ts < edge_ts[key]:
                edge_ts[key] = ts
        cycle = _find_longest_cycle(graph)
        if len(cycle) > len(best_cycle):
            best_cycle = cycle
            best_year = year

    if not best_cycle:
        return None, []

    # Find first game in cycle to determine starting player
    yg = std[std["yr"] == best_year]
    edge_ts: dict[tuple, str] = {}
    for _, row in yg.iterrows():
        w, b = str(row["white"]), str(row["black"])
        winner, loser = (w, b) if row["result"] == "1-0" else (b, w)
        key = (winner, loser)
        ts = f"{row['utcdate']} {row['utctime']}"
        if key not in edge_ts or ts < edge_ts[key]:
            edge_ts[key] = ts

    n = len(best_cycle)
    edges = [(best_cycle[i], best_cycle[(i+1) % n]) for i in range(n)]
    earliest = min(edges, key=lambda e: edge_ts.get(e, "9999"))
    start_idx = edges.index(earliest)
    ordered = best_cycle[start_idx:] + best_cycle[:start_idx]
    return best_year, ordered


# ============================================================
# Q13: Does using more or less time correlate with winning?
# ============================================================

def q13_time_usage_wins(games: pd.DataFrame) -> str:
    print("  Q13: collecting time usage")
    result_map = dict(zip(games["game_id"], games["result"]))
    ws_map = dict(zip(games["game_id"], games["whitestart"].apply(parse_clock)))
    bs_map = dict(zip(games["game_id"], games["blackstart"].apply(parse_clock)))
    inc_map = dict(zip(games["game_id"], games["timecontrol"].apply(lambda x: parse_tc(x)[1])))

    more_wins = more_total = less_wins = less_total = 0
    done = 0
    for gid, _, mdf in stream_games():
        done += 1
        if done % 500_000 == 0:
            print(f"    Q13 {done:,}")
        res = result_map.get(gid, "")
        if res == "1/2-1/2":
            continue
        inc = inc_map.get(gid, 0)
        pw, pb = ws_map.get(gid, -1), bs_map.get(gid, -1)
        if pw < 0 or pb < 0:
            continue
        wt = bt = 0.0
        for _, r in mdf.iterrows():
            clk = parse_clock(r["clock"])
            if clk < 0:
                continue
            if r["color"] == "white" and pw >= 0:
                wt += (pw - clk) + inc; pw = clk
            elif r["color"] == "black" and pb >= 0:
                bt += (pb - clk) + inc; pb = clk
        if wt == bt:
            continue
        white_used_more = wt > bt
        if white_used_more:
            more_total += 1
            if res == "1-0":
                more_wins += 1
            less_total += 1
            if res == "0-1":
                less_wins += 1
        else:
            more_total += 1
            if res == "0-1":
                more_wins += 1
            less_total += 1
            if res == "1-0":
                less_wins += 1

    mr = more_wins / more_total if more_total else 0
    lr = less_wins / less_total if less_total else 0
    who = "Több időt felhasználók" if mr > lr else "Kevesebb időt felhasználók"
    return f"{who} nyernek nagyobb arányban (több: {mr:.4f}, kevesebb: {lr:.4f})"


# ============================================================
# Q14: White a2 pawn reaches g8 and promotes — dates
# ============================================================

def q14_a2_to_g8(games: pd.DataFrame) -> list[str]:
    # Pre-filter: games with g8 promotion by white
    print("  Q14: pre-filtering g8 promotions")
    pre_gids: set = set()
    for chunk in pd.read_csv(MOVES, usecols=["game_id", "move", "color"], chunksize=CHUNKSIZE):
        wg8 = chunk[(chunk["color"] == "white") & chunk["move"].str.contains(r"g8=", na=False, regex=True)]
        pre_gids.update(wg8["game_id"].unique())

    date_map = dict(zip(games["game_id"], games["date"]))
    print(f"  Q14: simulating {len(pre_gids):,} games with g8 white promotions")

    result_dates: set = set()
    for gid, moves_list, _ in stream_games(pre_gids):
        board = chess.Board()
        # Index of the original a2 pawn; we track it as it moves
        a2_pawn = chess.A2
        found = False
        for san in moves_list:
            try:
                move = board.parse_san(san)
                mover = board.turn
                board.push(move)
            except Exception:
                break
            if mover != chess.WHITE:
                continue
            from_sq, to_sq = move.from_square, move.to_square
            if from_sq == a2_pawn:
                # Check it's still a pawn (or just promoted)
                a2_pawn = to_sq
                if to_sq == chess.G8:
                    found = True
                    break
            # Check if our tracked pawn was captured by black
            if mover == chess.WHITE:
                pass  # white moved, black hasn't captured yet
        # Also check if pawn was captured by black (a black move landing on a2_pawn)
        # This is handled implicitly: if board has no piece at a2_pawn after a black move
        # We approximate: if found is True, the pawn reached g8
        if found:
            result_dates.add(date_map.get(gid, ""))

    return sorted(d for d in result_dates if d)[:10]


# ============================================================
# Q15: Non-queen promotions
# ============================================================

def q15_non_queen_promotions() -> tuple[int, list]:
    print("  Q15: scanning all moves for promotions")
    counts: Counter = Counter()
    for chunk in pd.read_csv(MOVES, usecols=["move"], chunksize=CHUNKSIZE):
        for mv in chunk["move"].dropna():
            m = re.search(r"=([RBNQ])", mv)
            if m and m.group(1) != "Q":
                counts[m.group(1)] += 1
    total = sum(counts.values())
    return total, counts.most_common(3)


# ============================================================
# Q16: Longest draw streak (standard games)
# ============================================================

def q16_draw_streak(games: pd.DataFrame) -> tuple:
    std = games[games["variant"] == "Standard"].sort_values(["utcdate", "utctime"])
    player_games: dict[str, list] = defaultdict(list)
    for _, row in std.iterrows():
        ts = f"{row['utcdate']} {row['utctime']}"
        w, b = str(row["white"]), str(row["black"])
        res = row["result"]
        welo = row.get("whiteelo") or 0
        belo = row.get("blackelo") or 0
        player_games[w].append((ts, res, welo))
        player_games[b].append((ts, res, belo))

    best_n = 0
    best_player = best_start = best_end = None
    best_elo = 0

    for player, glist in player_games.items():
        glist.sort()
        streak = 0
        s_start = s_elo = None
        for ts, res, elo in glist:
            if res == "1/2-1/2":
                if streak == 0:
                    s_start = ts
                streak += 1
                s_elo = elo or s_elo
            else:
                streak = 0
            if streak > best_n or (streak == best_n and streak > 0 and (s_elo or 0) > best_elo):
                best_n = streak
                best_player = player
                best_start = s_start
                best_end = ts
                best_elo = s_elo or 0
        # Ongoing streak
        if streak > 0:
            if streak > best_n or (streak == best_n and (s_elo or 0) > best_elo):
                best_n = streak
                best_player = player
                best_start = s_start
                best_end = glist[-1][0]
                best_elo = s_elo or 0

    return best_player, best_start, best_end, best_n


# ============================================================
# Q17: Logistic regression per-move (capture ~ time_elapsed + color)
# ============================================================

def q17_logit_move(games: pd.DataFrame) -> dict:
    print("  Q17: per-move logistic regression with SGD")
    ws_map = dict(zip(games["game_id"], games["whitestart"].apply(parse_clock)))
    bs_map = dict(zip(games["game_id"], games["blackstart"].apply(parse_clock)))
    inc_map = dict(zip(games["game_id"], games["timecontrol"].apply(lambda x: parse_tc(x)[1])))

    sgd = SGDClassifier(loss="log_loss", random_state=42)
    fitted = False
    bX: list = []
    by: list = []
    BATCH = 200_000
    done = 0

    for gid, _, mdf in stream_games():
        done += 1
        if done % 500_000 == 0:
            print(f"    Q17 {done:,}")
        inc = inc_map.get(gid, 0)
        pw, pb = ws_map.get(gid, -1), bs_map.get(gid, -1)
        elapsed = 0.0
        for _, r in mdf.iterrows():
            clk = parse_clock(r["clock"])
            c = r["color"]
            if c == "white" and pw > 0 and clk >= 0:
                elapsed += (pw - clk) + inc; pw = clk
            elif c == "black" and pb > 0 and clk >= 0:
                elapsed += (pb - clk) + inc; pb = clk
            bX.append([elapsed, 1 if c == "white" else 0])
            by.append(1 if "x" in str(r["move"]) else 0)
            if len(bX) >= BATCH:
                Xa = np.array(bX, dtype=np.float32)
                ya = np.array(by, dtype=np.int32)
                if not fitted:
                    sgd.partial_fit(Xa, ya, classes=[0, 1])
                    fitted = True
                else:
                    sgd.partial_fit(Xa, ya)
                bX, by = [], []

    if bX:
        Xa, ya = np.array(bX, dtype=np.float32), np.array(by, dtype=np.int32)
        if not fitted:
            sgd.partial_fit(Xa, ya, classes=[0, 1])
        else:
            sgd.partial_fit(Xa, ya)

    return {"intercept": float(sgd.intercept_[0]),
            "coef_time_elapsed_sec": float(sgd.coef_[0][0]),
            "coef_color_white": float(sgd.coef_[0][1])}


# ============================================================
# Q18: Longest winless streak (standard games)
# ============================================================

def q18_winless_streak(games: pd.DataFrame) -> tuple:
    std = games[games["variant"] == "Standard"].sort_values(["utcdate", "utctime"])
    player_games: dict[str, list] = defaultdict(list)
    for _, row in std.iterrows():
        ts = f"{row['utcdate']} {row['utctime']}"
        w, b = str(row["white"]), str(row["black"])
        res = row["result"]
        player_games[w].append((ts, "win" if res == "1-0" else "nowin"))
        player_games[b].append((ts, "win" if res == "0-1" else "nowin"))

    # Find all players' max winless streak
    player_max: dict[str, int] = {}
    player_info: dict[str, tuple] = {}

    for player, glist in player_games.items():
        glist.sort()
        best = cur = 0
        s_start = None
        best_start = best_end = None
        for ts, outcome in glist:
            if outcome == "nowin":
                if cur == 0:
                    s_start = ts
                cur += 1
                if cur > best:
                    best = cur
                    best_start = s_start
                    best_end = ts
            else:
                cur = 0
        # Check ongoing
        if cur > best:
            best = cur
            best_start = s_start
            best_end = glist[-1][0]
        player_max[player] = best
        player_info[player] = (best_start, best_end)

    if not player_max:
        return None, None, None, 0

    global_best = max(player_max.values())
    tied = [p for p, v in player_max.items() if v == global_best]

    lili_key = hu_key("Lili")
    after = [p for p in tied if hu_key(p) > lili_key]
    winner = min(after, key=hu_key) if after else min(tied, key=hu_key)

    s, e = player_info[winner]
    return winner, s, e, global_best


# ============================================================
# Q19: 50-move rule draws, standard, 2026.03.15–2026.10.14
# ============================================================

def q19_fifty_move(games: pd.DataFrame) -> int:
    mask = (
        (games["variant"] == "Standard")
        & (games["date"] >= "2026.03.15")
        & (games["date"] <= "2026.10.14")
        & (games["result"] == "1/2-1/2")
        & (games["termination"] == "Normal")
    )
    gids = set(games.loc[mask, "game_id"])
    print(f"  Q19: simulating {len(gids):,} draw games")
    count = 0
    for gid, moves_list, _ in stream_games(gids):
        board = simulate_board(moves_list)
        if board and board.is_fifty_moves():
            count += 1
    return count


# ============================================================
# Q20: Queen's Gambit % by year (04.21–05.18 CET, standard)
# ============================================================

def q20_queens_gambit(games: pd.DataFrame) -> dict[int, float]:
    std = games[games["variant"] == "Standard"].copy()

    def in_range_year(row):
        dt = game_start_cet(str(row["utcdate"]), str(row["utctime"]))
        if dt is None:
            return None, None
        md = (dt.month, dt.day)
        if (4, 21) <= md <= (5, 18):
            return dt.year, True
        return dt.year, False

    print(f"  Q20: parsing {len(std):,} standard game dates")
    yr_col = []
    in_col = []
    for _, row in std.iterrows():
        yr, ok = in_range_year(row)
        yr_col.append(yr)
        in_col.append(ok)
    std["yr"] = yr_col
    std["in_range"] = in_col
    in_r = std[std["in_range"] == True]
    gids = set(in_r["game_id"])
    yr_map = dict(zip(in_r["game_id"], in_r["yr"]))
    print(f"  Q20: checking first moves for {len(gids):,} games")

    year_total: Counter = Counter()
    year_qg: Counter = Counter()

    for gid, moves_list, _ in stream_games(gids):
        yr = yr_map.get(gid)
        if yr is None or len(moves_list) < 3:
            continue
        m1 = moves_list[0].rstrip("+#")  # white move 1
        m2 = moves_list[1].rstrip("+#")  # black move 1
        m3 = moves_list[2].rstrip("+#")  # white move 2
        year_total[yr] += 1
        if m1 == "d4" and m2 == "d5" and m3 == "c4":
            year_qg[yr] += 1

    return {yr: year_qg[yr] / year_total[yr] for yr in sorted(year_total) if year_total[yr] > 0}


# ============================================================
# Q21: Standard games potentially spanning New Year's Eve
# ============================================================

def q21_year_spanning(games: pd.DataFrame) -> dict[int, int]:
    std = games[games["variant"] == "Standard"].copy()
    std["ws"] = std["whitestart"].apply(parse_clock)
    std["bs"] = std["blackstart"].apply(parse_clock)
    result: dict[int, int] = {}
    for _, row in std.iterrows():
        dt = game_start_cet(str(row["utcdate"]), str(row["utctime"]))
        if dt is None or dt.month != 12 or dt.day != 31:
            continue
        yr = dt.year
        midnight = CET.localize(datetime(yr + 1, 1, 1))
        secs = (midnight - dt).total_seconds()
        total = (row["ws"] or 0) + (row["bs"] or 0)
        if total > 0 and total >= secs:
            result[yr] = result.get(yr, 0) + 1
    return result


# ============================================================
# Q22: Rectangle-shaped piece paths
# ============================================================

def _is_rect(s1, s2, s3, s4) -> bool:
    f = [chess.square_file(s) for s in (s1, s2, s3, s4)]
    r = [chess.square_rank(s) for s in (s1, s2, s3, s4)]
    if len(set(f)) != 2 or len(set(r)) != 2:
        return False
    if (max(f) - min(f)) * (max(r) - min(r)) == 0:
        return False
    for i in range(4):
        a = (f[i], r[i])
        b = (f[(i+1)%4], r[(i+1)%4])
        if a[0] != b[0] and a[1] != b[1]:
            return False
    return True


def _rect_area(s1, s2, s3, s4) -> int:
    f = [chess.square_file(s) for s in (s1, s2, s3, s4)]
    r = [chess.square_rank(s) for s in (s1, s2, s3, s4)]
    return (max(f) - min(f)) * (max(r) - min(r))


def _sim_rectangles(moves_list: list[str]) -> tuple[int, int, int]:
    """Return (white_rects, black_rects, max_area) for a game."""
    board = chess.Board()
    # Each piece tracked by its original square (unique piece ID)
    # history[orig_sq] = list of squares visited
    history: dict[chess.Square, list[chess.Square]] = {}
    # sq_to_orig: current square → original square (piece identity)
    sq_to_orig: dict[chess.Square, chess.Square] = {}

    for sq in chess.SQUARES:
        if board.piece_at(sq):
            history[sq] = [sq]
            sq_to_orig[sq] = sq

    white_rects = black_rects = max_area = 0

    for san in moves_list:
        try:
            move = board.parse_san(san)
            is_castle = board.is_castling(move)
            is_ks = board.is_kingside_castling(move) if is_castle else False
            is_ep = board.is_en_passant(move)
            mover = board.turn
            board.push(move)
        except Exception:
            break

        from_sq, to_sq = move.from_square, move.to_square
        orig = sq_to_orig.get(from_sq)
        if orig is None:
            continue

        # Remove captured piece from tracking
        if not is_ep:
            cap_orig = sq_to_orig.pop(to_sq, None)
        else:
            # En passant: captured pawn is on same rank as from_sq, file of to_sq
            ep_sq = chess.square(chess.square_file(to_sq), chess.square_rank(from_sq))
            sq_to_orig.pop(ep_sq, None)
            sq_to_orig.pop(to_sq, None)  # just in case

        # Move piece
        del sq_to_orig[from_sq]
        sq_to_orig[to_sq] = orig
        history.setdefault(orig, [from_sq]).append(to_sq)

        # Handle castling rook
        if is_castle:
            if mover == chess.WHITE:
                rf, rt = (chess.H1, chess.F1) if is_ks else (chess.A1, chess.D1)
            else:
                rf, rt = (chess.H8, chess.F8) if is_ks else (chess.A8, chess.D8)
            rorig = sq_to_orig.pop(rf, None)
            if rorig is not None:
                sq_to_orig[rt] = rorig
                history.setdefault(rorig, [rf]).append(rt)

        # Count rectangles in the piece's history
        h = history[orig]
        if len(h) >= 4:
            rects = sum(1 for i in range(len(h)-3) if _is_rect(h[i], h[i+1], h[i+2], h[i+3]))
            area = max((_rect_area(h[i], h[i+1], h[i+2], h[i+3]) for i in range(len(h)-3)
                        if _is_rect(h[i], h[i+1], h[i+2], h[i+3])), default=0)
            # Determine color: mover is the player who just moved this piece
            if mover == chess.WHITE:
                white_rects += rects
            else:
                black_rects += rects
            max_area = max(max_area, area)

    return white_rects, black_rects, max_area


def q22_rectangles(games: pd.DataFrame) -> tuple:
    print("  Q22: rectangle paths (this is slow — simulates all games)")
    white_map = dict(zip(games["game_id"], games["white"]))
    black_map = dict(zip(games["game_id"], games["black"]))
    player_rects: Counter = Counter()
    global_max = 0
    done = 0
    for gid, moves_list, _ in stream_games():
        done += 1
        if done % 200_000 == 0:
            print(f"    Q22 {done:,}")
        wr, br, area = _sim_rectangles(moves_list)
        player_rects[white_map.get(gid, "")] += wr
        player_rects[black_map.get(gid, "")] += br
        if area > global_max:
            global_max = area
    if not player_rects:
        return None, 0, 0
    top_val = max(player_rects.values())
    # Tiebreaker: first to reach (approximated as alphabetical)
    top_p = min(p for p, c in player_rects.items() if c == top_val and p)
    return top_p, top_val, global_max


# ============================================================
# Q23: Castling checkmates
# ============================================================

def q23_castle_checkmates(games: pd.DataFrame) -> list[str]:
    print("  Q23: scanning moves for castling checkmates")
    white_map = dict(zip(games["game_id"], games["white"]))
    black_map = dict(zip(games["game_id"], games["black"]))
    counts: Counter = Counter()
    for chunk in pd.read_csv(MOVES, usecols=["game_id", "move", "color"], chunksize=CHUNKSIZE):
        cm = chunk[chunk["move"].isin(["O-O#", "O-O-O#"])]
        for _, row in cm.iterrows():
            gid, color = row["game_id"], row["color"]
            p = white_map.get(gid) if color == "white" else black_map.get(gid)
            if p:
                counts[str(p)] += 1
    if not counts:
        return []
    top = max(counts.values())
    return sorted(p for p, c in counts.items() if c == top)[:10]


# ============================================================
# Q24: En passant by white in Indian openings (3-min games)
# ============================================================

def q24_en_passant_indian(games: pd.DataFrame) -> int:
    mask = (
        games["timecontrol"].str.startswith("180", na=False)
        & games["eco"].str.startswith("E", na=False)
    )
    gids = set(games.loc[mask, "game_id"])
    print(f"  Q24: simulating {len(gids):,} Indian 3-min games")
    count = done = 0
    for gid, moves_list, _ in stream_games(gids):
        done += 1
        if done % 10_000 == 0:
            print(f"    Q24 {done:,}/{len(gids):,}")
        board = chess.Board()
        for san in moves_list:
            try:
                move = board.parse_san(san)
                is_ep = board.is_en_passant(move)
                mover = board.turn
                board.push(move)
            except Exception:
                break
            if mover == chess.WHITE and is_ep:
                count += 1
    return count


# ============================================================
# Output
# ============================================================

def write_answers(answers: dict):
    lines = ["# Chess Data Analysis — Answers\n"]
    for q in sorted(answers):
        lines.append(f"## {q}. kérdés\n\n{answers[q]}\n")
    with open(OUTPUT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\nWritten to {OUTPUT}")


# ============================================================
# Main
# ============================================================

def load_games() -> pd.DataFrame:
    print("Loading games...")
    parts = []
    for chunk in pd.read_csv(GAMES, chunksize=CHUNKSIZE,
                              dtype={"result": "category", "variant": "category",
                                     "termination": "category"}):
        parts.append(chunk)
    df = pd.concat(parts, ignore_index=True)
    df["result"] = df["result"].astype(str)
    df["variant"] = df["variant"].astype(str)
    df["termination"] = df["termination"].astype(str)
    print(f"  {len(df):,} games")
    return df


def main():
    games = load_games()
    tournaments = pd.read_csv(TOURNAMENTS)
    ans = {}

    # --- Games-only (fast) ---
    print("\n=== Games-only questions ===")

    print("Q9"); winners9, cnt9 = q9_berserk_timeouts(games)
    ans[9] = f"Legtöbb berserk timeout vereség ({cnt9}x): {', '.join(winners9)}"

    print("Q16"); p16, s16, e16, n16 = q16_draw_streak(games)
    ans[16] = f"Játékos: {p16} | {s16} – {e16} | {n16} parti"

    print("Q18"); p18, s18, e18, n18 = q18_winless_streak(games)
    ans[18] = f"Játékos: {p18} | {s18} – {e18} | {n18} parti"

    print("Q21"); q21 = q21_year_spanning(games)
    ans[21] = "\n".join(f"{yr}: {c}" for yr, c in sorted(q21.items())) or "0"

    # --- Moves-pattern-matching (single scans, no board sim) ---
    print("\n=== Pattern matching ===")

    print("Q5"); ans[5] = str(q5_threefold_scissors(games))

    print("Q8"); ans[8] = str(q8_draw_march20_promotion(games))

    print("Q15"); tot15, top15 = q15_non_queen_promotions()
    ans[15] = f"Nem vezérre váltások: {tot15} | Top 3: " + ", ".join(f"{p}:{c}" for p, c in top15)

    print("Q23"); ans[23] = ", ".join(q23_castle_checkmates(games))

    # --- Board simulation (filtered sets) ---
    print("\n=== Board simulation (filtered) ===")

    print("Q1"); ans[1] = str(q1_material_disadvantage(games))

    print("Q3"); ans[3] = str(q3_castling_rights_lost(games))

    print("Q6"); ans[6] = str(q6_threefold_date_range(games))

    print("Q7"); avg7 = q7_queens_at_checkmate(games, tournaments)
    ans[7] = f"{avg7:.4f}"

    print("Q12"); yr12, cycle12 = q12_largest_cycle(games)
    ans[12] = (f"Év: {yr12} | " + " → ".join(cycle12) + f" → {cycle12[0]}") if cycle12 else "Nincs"

    print("Q19"); ans[19] = str(q19_fifty_move(games))

    print("Q24"); ans[24] = str(q24_en_passant_indian(games))

    print("Q14"); dates14 = q14_a2_to_g8(games)
    ans[14] = ", ".join(dates14) if dates14 else "Nincs"

    # --- Full-scan moves questions ---
    print("\n=== Full moves scan ===")

    print("Q4"); diff4 = q4_rook_distances(games)
    ans[4] = f"Fehér - Fekete bástya távolság: {diff4} mező"

    print("Q10"); p10 = q10_logit_game(games)
    ans[10] = (f"Intercept: {p10['intercept']:.6f}, "
               f"captures: {p10['coef_captures']:.6f}, "
               f"white: {p10['coef_color_white']:.6f}, "
               f"avg_time: {p10['coef_avg_time_per_move']:.6f} "
               f"(standardizált; μ={p10['feature_means']}, σ={p10['feature_stds']})")

    print("Q11"); most11, cnt11, never11, med11, atmed11 = q11_resignations(games)
    ans[11] = (f"Legtöbbet feladott: {most11} ({cnt11}x) | "
               f"Soha nem adta fel: {never11} | Mediánban ({med11:.1f}): {atmed11}")

    print("Q13"); ans[13] = q13_time_usage_wins(games)

    print("Q17"); p17 = q17_logit_move(games)
    ans[17] = (f"Intercept: {p17['intercept']:.6f}, "
               f"time_elapsed: {p17['coef_time_elapsed_sec']:.6f}, "
               f"white: {p17['coef_color_white']:.6f}")

    print("Q20"); q20 = q20_queens_gambit(games)
    ans[20] = "\n".join(f"{yr}: {r:.4f} ({r*100:.2f}%)" for yr, r in sorted(q20.items()))

    print("Q2"); ans[2] = q2_left_knight_capture(games)

    # Q22 is very slow — simulate all games
    print("Q22"); top22, n22, area22 = q22_rectangles(games)
    ans[22] = f"Játékos: {top22} ({n22} téglalap) | Legnagyobb terület: {area22}"

    write_answers(ans)


if __name__ == "__main__":
    main()
