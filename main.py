#!/usr/bin/env python3
"""Chess tournament data analysis - answers all 24 questions.

Memory design:
- Never load full games DataFrame into memory.
- GameIndex builds compact numpy arrays (binary-search lookup) in one pass: ~3 GB.
- Moves streamed in chunks: ~1 GB per chunk.
- Peak total: well under 16 GB.
"""

from __future__ import annotations

import re
from collections import defaultdict, Counter
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
CHUNKSIZE = 300_000

CET = pytz.timezone("Europe/Budapest")
UTC = pytz.utc

# ============================================================
# Utilities
# ============================================================


def parse_clock(s) -> int:
    """H:MM:SS → seconds.  Returns -1 on error."""
    try:
        p = str(s).split(":")
        return int(p[0]) * 3600 + int(p[1]) * 60 + int(p[2])
    except Exception:
        return -1


def _parse_clock_vec(series: "pd.Series") -> "pd.Series":
    """Vectorized H:MM:SS → seconds for a pandas Series."""
    filled = series.fillna("0:00:00").astype(str)
    parts = filled.str.split(":", expand=True)
    try:
        return (
            parts[0].astype(int) * 3600
            + parts[1].astype(int) * 60
            + parts[2].astype(int)
        )
    except Exception:
        return filled.apply(parse_clock)


def parse_tc(tc) -> tuple[int, int]:
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
    vals = {
        chess.PAWN: 1,
        chess.KNIGHT: 3,
        chess.BISHOP: 3,
        chess.ROOK: 5,
        chess.QUEEN: 9,
    }
    w = sum(len(board.pieces(pt, chess.WHITE)) * v for pt, v in vals.items())
    b = sum(len(board.pieces(pt, chess.BLACK)) * v for pt, v in vals.items())
    return w, b


# ============================================================
# Hungarian key for Q18
# ============================================================

_HU = {
    "a": 1,
    "á": 2,
    "b": 3,
    "c": 4,
    "cs": 5,
    "d": 6,
    "dz": 7,
    "dzs": 8,
    "e": 9,
    "é": 10,
    "f": 11,
    "g": 12,
    "gy": 13,
    "h": 14,
    "i": 15,
    "í": 16,
    "j": 17,
    "k": 18,
    "l": 19,
    "ly": 20,
    "m": 21,
    "n": 22,
    "ny": 23,
    "o": 24,
    "ó": 25,
    "ö": 26,
    "ő": 27,
    "p": 28,
    "q": 29,
    "r": 30,
    "s": 31,
    "sz": 32,
    "t": 33,
    "ty": 34,
    "u": 35,
    "ú": 36,
    "ü": 37,
    "ű": 38,
    "v": 39,
    "w": 40,
    "x": 41,
    "y": 42,
    "z": 43,
    "zs": 44,
}


def hu_key(name: str) -> list[int]:
    s = name.lower()
    result = []
    i = 0
    while i < len(s):
        for L in (3, 2, 1):
            if s[i : i + L] in _HU:
                result.append(_HU[s[i : i + L]])
                i += L
                break
        else:
            result.append(200 + ord(s[i]))
            i += 1
    return result


# ============================================================
# GameIndex: compact numpy binary-search lookup
# ============================================================


class GameIndex:
    """Compact game metadata index built in a single pass.

    All arrays are sorted by game_id (S14 bytes) so lookups are O(log n).
    Does NOT store white/black player names (handled separately per question).
    """

    # Termination codes
    TERM_NORMAL = 0
    TERM_TIMEFORFEIT = 1
    TERM_ABANDONED = 2
    TERM_OTHER = 3

    # Result codes
    RES_DRAW = 0
    RES_WHITE = 1
    RES_BLACK = 2

    def build(self) -> None:
        # Minimal columns: drop "date" (use utcdate), "tournamentid" (scanned separately)
        # Use int16 for wstarts/bstarts/welos/belos to save memory
        needed = [
            "game_id",
            "result",
            "variant",
            "utcdate",
            "utctime",
            "termination",
            "timecontrol",
            "whitestart",
            "blackstart",
            "eco",
            "whiteelo",
            "blackelo",
            "white",
            "black",
        ]
        gid_parts: list[np.ndarray] = []
        res_parts: list[np.ndarray] = []
        std_parts: list[np.ndarray] = []
        utcdate_parts: list[np.ndarray] = []
        utctime_parts: list[np.ndarray] = []
        term_parts: list[np.ndarray] = []
        tcbase_parts: list[np.ndarray] = []
        ws_parts: list[np.ndarray] = []
        bs_parts: list[np.ndarray] = []
        eco_parts: list[np.ndarray] = []
        welo_parts: list[np.ndarray] = []
        belo_parts: list[np.ndarray] = []
        # Player name indices
        player_dict: dict[str, int] = {}  # name → int id
        widx_parts: list[np.ndarray] = []
        bidx_parts: list[np.ndarray] = []

        _res_map = {
            "1-0": self.RES_WHITE,
            "0-1": self.RES_BLACK,
            "1/2-1/2": self.RES_DRAW,
        }
        _term_map = {
            "Normal": self.TERM_NORMAL,
            "Time forfeit": self.TERM_TIMEFORFEIT,
            "Abandoned": self.TERM_ABANDONED,
        }

        print("  Building GameIndex (single pass through games)...")
        total = 0
        for chunk in pd.read_csv(
            GAMES,
            usecols=needed,
            chunksize=CHUNKSIZE,
            dtype={"whiteelo": "Int32", "blackelo": "Int32"},
        ):
            n = len(chunk)
            total += n

            gid_parts.append(chunk["game_id"].to_numpy().astype("S14"))
            res_parts.append(
                chunk["result"]
                .map(_res_map)
                .fillna(self.RES_DRAW)
                .astype(np.int8)
                .to_numpy()
            )
            std_parts.append(
                (chunk["variant"].str.strip().str.lower() == "standard").to_numpy()
            )
            utcdate_parts.append(chunk["utcdate"].fillna("").to_numpy().astype("S10"))
            utctime_parts.append(
                _parse_clock_vec(chunk["utctime"]).astype(np.int32).to_numpy()
            )
            term_parts.append(
                chunk["termination"]
                .map(_term_map)
                .fillna(self.TERM_OTHER)
                .astype(np.int8)
                .to_numpy()
            )
            tcbase_parts.append(
                chunk["timecontrol"]
                .str.split("+", expand=True)[0]
                .fillna(0)
                .astype(np.int16)
                .to_numpy()
            )
            ws_parts.append(
                _parse_clock_vec(chunk["whitestart"])
                .clip(-1, 32767)
                .astype(np.int16)
                .to_numpy()
            )
            bs_parts.append(
                _parse_clock_vec(chunk["blackstart"])
                .clip(-1, 32767)
                .astype(np.int16)
                .to_numpy()
            )
            eco_parts.append(chunk["eco"].fillna("").to_numpy().astype("S3"))
            welo_parts.append(
                chunk["whiteelo"].fillna(0).clip(0, 32767).astype(np.int16).to_numpy()
            )
            belo_parts.append(
                chunk["blackelo"].fillna(0).clip(0, 32767).astype(np.int16).to_numpy()
            )
            # Player indices (vectorized via Series.map for speed)
            w_names = chunk["white"].fillna("").astype(str)
            b_names = chunk["black"].fillna("").astype(str)
            for p in pd.concat([w_names, b_names]).unique():
                if p not in player_dict:
                    player_dict[p] = len(player_dict)
            widx_parts.append(w_names.map(player_dict).astype(np.int32).to_numpy())
            bidx_parts.append(b_names.map(player_dict).astype(np.int32).to_numpy())

        print(f"  Concatenating {total:,} games...")
        gids_raw = np.concatenate(gid_parts)
        del gid_parts
        order = np.argsort(gids_raw)
        self.gids: np.ndarray = gids_raw[order].copy()
        del gids_raw

        def _sort(parts):
            arr = np.concatenate(parts)
            del parts[:]
            return arr[order]

        self.results = _sort(res_parts)
        self.is_std = _sort(std_parts)
        self.utcdates = _sort(utcdate_parts)
        self.utctimes = _sort(utctime_parts)
        self.terms = _sort(term_parts)
        self.tcbases = _sort(tcbase_parts)
        self.wstarts = _sort(ws_parts)
        self.bstarts = _sort(bs_parts)
        self.ecos = _sort(eco_parts)
        self.welos = _sort(welo_parts)
        self.belos = _sort(belo_parts)
        self.widxs = _sort(widx_parts)
        self.bidxs = _sort(bidx_parts)
        # Alias: dates = utcdates (UTC date used as proxy for local date)
        self.dates = self.utcdates
        # Build player names array (indexed by player_dict values)
        names = sorted(player_dict, key=player_dict.get)
        self.player_names: np.ndarray = np.array(names, dtype="U40")
        del player_dict, names, order
        print(
            f"  GameIndex ready: {len(self.gids):,} games, "
            f"~{self.gids.nbytes // 1024 // 1024 + self.results.nbytes // 1024 // 1024 + self.dates.nbytes // 1024 // 1024 + self.tcbases.nbytes // 1024 // 1024 * 10} MB"
        )

    def _idx(self, gid: str) -> int:
        target = np.array([gid], dtype="S14")[0]
        i = np.searchsorted(self.gids, target)
        if i < len(self.gids) and self.gids[i] == target:
            return int(i)
        return -1

    def result(self, gid: str) -> int:
        i = self._idx(gid)
        return int(self.results[i]) if i >= 0 else -1

    def game_ids_where(self, mask: np.ndarray) -> set[str]:
        return set(g.decode() for g in self.gids[mask])

    def filter_mask(
        self,
        *,
        std: bool = False,
        date_gte: str = "",
        date_lte: str = "",
        result_ne_draw: bool = False,
        result_eq_draw: bool = False,
        term_normal: bool = False,
        tc_starts: str = "",
        eco_starts: str = "",
    ) -> np.ndarray:
        mask = np.ones(len(self.gids), dtype=bool)
        if std:
            mask &= self.is_std
        if date_gte:
            mask &= self.dates >= date_gte.encode()
        if date_lte:
            mask &= self.dates <= date_lte.encode()
        if result_ne_draw:
            mask &= self.results != self.RES_DRAW
        if result_eq_draw:
            mask &= self.results == self.RES_DRAW
        if term_normal:
            mask &= self.terms == self.TERM_NORMAL
        if tc_starts:
            pref = tc_starts.encode()[: len(tc_starts)]
            mask &= (
                np.frompyfunc(lambda x: x.startswith(pref), 1, 1)(
                    np.frombuffer(
                        self.tcbases.tobytes() if False else b"", dtype=np.int32
                    )
                ).astype(bool)
                if False
                else self._tc_startswith(tc_starts)
            )
        if eco_starts:
            pref = eco_starts.encode()[:1]
            mask &= np.array([e[:1] == pref for e in self.ecos])
        return mask

    def _tc_startswith(self, prefix: str) -> np.ndarray:
        """Filter by timecontrol base that starts with given string (in seconds)."""
        p = int(prefix)
        return self.tcbases == p

    def by_date_endswith(self, suffix: str) -> np.ndarray:
        suf = suffix.encode()
        return np.array([d.endswith(suf) for d in self.dates])

    def utc_in_cet_month_day_range(self, start_md: tuple, end_md: tuple) -> np.ndarray:
        """Filter where CET month-day of game start is in [start_md, end_md]."""
        # For each game, decode utcdate+utctime to CET and check month-day
        result = np.zeros(len(self.gids), dtype=bool)
        for i in range(len(self.gids)):
            ud = self.utcdates[i].decode()
            ut_secs = int(self.utctimes[i])
            if not ud:
                continue
            try:
                # Quick check: parse just date and add time
                dt = datetime.strptime(ud, "%Y.%m.%d")
                dt_utc = UTC.localize(dt).replace(
                    hour=ut_secs // 3600,
                    minute=(ut_secs % 3600) // 60,
                    second=ut_secs % 60,
                )
                dt_cet = dt_utc.astimezone(CET)
                md = (dt_cet.month, dt_cet.day)
                result[i] = start_md <= md <= end_md
            except Exception:
                pass
        return result

    def player_at(self, i: int, color: str) -> str:
        """Return white or black player name for game at sorted index i."""
        pid = int(self.widxs[i]) if color == "white" else int(self.bidxs[i])
        return str(self.player_names[pid])

    def get_at(self, i: int) -> dict:
        return {
            "game_id": self.gids[i].decode(),
            "result": int(self.results[i]),
            "is_std": bool(self.is_std[i]),
            "date": self.dates[i].decode(),
            "utcdate": self.utcdates[i].decode(),
            "utctime_secs": int(self.utctimes[i]),
            "term": int(self.terms[i]),
            "tcbase": int(self.tcbases[i]),
            "wstart": int(self.wstarts[i]),
            "bstart": int(self.bstarts[i]),
            "eco": self.ecos[i].decode(),
            "welo": int(self.welos[i]),
            "belo": int(self.belos[i]),
        }


# ============================================================
# Moves streaming
# ============================================================


def stream_games(
    gids_filter: Optional[set] = None,
) -> Iterator[tuple[str, list[str], pd.DataFrame]]:
    buf = pd.DataFrame()
    for chunk in pd.read_csv(MOVES, chunksize=CHUNKSIZE):
        combined = pd.concat([buf, chunk], ignore_index=True) if len(buf) else chunk
        last = combined["game_id"].iloc[-1]
        complete = combined[combined["game_id"] != last]
        buf = combined[combined["game_id"] == last]
        for gid, grp in complete.groupby("game_id", sort=False):
            if gids_filter is None or gid in gids_filter:
                yield gid, grp["move"].tolist(), grp
    if len(buf):
        for gid, grp in buf.groupby("game_id", sort=False):
            if gids_filter is None or gid in gids_filter:
                yield gid, grp["move"].tolist(), grp


# ============================================================
# Q1: Material disadvantage ≥ 3, standard, 2023.10.12–2024.02.19
# ============================================================


def q1_material_disadvantage(gi: GameIndex) -> int:
    mask = gi.filter_mask(
        std=True, date_gte="2023.10.12", date_lte="2024.02.19", result_ne_draw=True
    )
    gids = gi.game_ids_where(mask)
    res = {gi.gids[i].decode(): gi.results[i] for i in np.where(mask)[0]}
    print(f"  Q1: {len(gids):,} games")
    count = done = 0
    for gid, ml, _ in stream_games(gids):
        done += 1
        if done % 200_000 == 0:
            print(f"    Q1 {done:,}/{len(gids):,}")
        board = simulate_board(ml)
        if board is None:
            continue
        w, b = count_material(board)
        r = res.get(gid, 0)
        if (r == GameIndex.RES_WHITE and w - b >= 3) or (
            r == GameIndex.RES_BLACK and b - w >= 3
        ):
            count += 1
    return count


# ============================================================
# Q2: Left knight capture win rate
# ============================================================


def q2_left_knight(gi: GameIndex) -> str:
    res_arr = gi.results
    gids_arr = gi.gids
    lk_wins = lk_total = no_wins = no_total = 0
    done = 0
    for gid, ml, _ in stream_games():
        done += 1
        if done % 500_000 == 0:
            print(f"    Q2 {done:,}")
        idx = np.searchsorted(gids_arr, np.array([gid], dtype="S14")[0])
        if idx >= len(gids_arr) or gids_arr[idx].decode() != gid:
            continue
        r = int(res_arr[idx])
        if r == GameIndex.RES_DRAW:
            continue

        board = chess.Board()
        wlk: Optional[chess.Square] = chess.B1
        blk: Optional[chess.Square] = chess.G8
        w_cap = b_cap = False
        for san in ml:
            try:
                move = board.parse_san(san)
                is_cap = board.is_capture(move)
                mover = board.turn
                board.push(move)
            except Exception:
                break
            from_sq, to_sq = move.from_square, move.to_square
            if mover == chess.WHITE:
                if wlk is not None and from_sq == wlk:
                    pt = board.piece_type_at(to_sq)
                    if pt == chess.KNIGHT:
                        if is_cap:
                            w_cap = True
                        wlk = to_sq
                    else:
                        wlk = None
                if blk is not None and is_cap and to_sq == blk:
                    blk = None
            else:
                if blk is not None and from_sq == blk:
                    pt = board.piece_type_at(to_sq)
                    if pt == chess.KNIGHT:
                        if is_cap:
                            b_cap = True
                        blk = to_sq
                    else:
                        blk = None
                if wlk is not None and is_cap and to_sq == wlk:
                    wlk = None

        for cap, won in (
            (w_cap, r == GameIndex.RES_WHITE),
            (b_cap, r == GameIndex.RES_BLACK),
        ):
            if cap:
                lk_total += 1
                if won:
                    lk_wins += 1
            else:
                no_total += 1
                if won:
                    no_wins += 1

    lk_r = lk_wins / lk_total if lk_total else 0
    no_r = no_wins / no_total if no_total else 0
    diff = lk_r - no_r
    dir_s = "nagyobb" if diff > 0 else "kisebb"
    return (
        f"Bal lóval ütők nyerési aránya: {lk_r:.4f} ({lk_wins}/{lk_total}), "
        f"nem ütők: {no_r:.4f} ({no_wins}/{no_total}), "
        f"különbség: {diff:+.4f} → bal lóval ütők {dir_s} arányban nyertek"
    )


# ============================================================
# Q3: 10-min games, white loses castling in first 6 half-moves
# ============================================================


def q3_castling_lost(gi: GameIndex) -> int:
    mask = gi.tcbases == 600
    gids = gi.game_ids_where(mask)
    print(f"  Q3: {len(gids):,} 10-min games")
    count = done = 0
    for gid, ml, _ in stream_games(gids):
        done += 1
        if done % 100_000 == 0:
            print(f"    Q3 {done:,}/{len(gids):,}")
        board = chess.Board()
        for i, san in enumerate(ml[:6]):
            had = board.has_castling_rights(chess.WHITE)
            try:
                board.push_san(san)
            except Exception:
                break
            if i % 2 == 0 and had and not board.has_castling_rights(chess.WHITE):
                count += 1
                break
    return count


# ============================================================
# Q4: Rook distance difference (white total – black total)
# ============================================================


def q4_rook_distances() -> int:
    print("  Q4: rook distances over all games")
    white_d = black_d = done = 0
    for gid, ml, _ in stream_games():
        done += 1
        if done % 500_000 == 0:
            print(f"    Q4 {done:,}")
        board = chess.Board()
        for san in ml:
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
                rf, rt = (
                    ((chess.H1, chess.F1) if is_ks else (chess.A1, chess.D1))
                    if mover == chess.WHITE
                    else ((chess.H8, chess.F8) if is_ks else (chess.A8, chess.D8))
                )
                d = abs(chess.square_file(rt) - chess.square_file(rf))
            elif board.piece_type_at(to_sq) == chess.ROOK:
                d = abs(chess.square_file(to_sq) - chess.square_file(from_sq)) + abs(
                    chess.square_rank(to_sq) - chess.square_rank(from_sq)
                )
            else:
                continue
            if mover == chess.WHITE:
                white_d += d
            else:
                black_d += d
    return white_d - black_d


# ============================================================
# Q5: Threefold repetition + scissors emoji
# ============================================================


def q5_threefold_scissors() -> int:
    scissors = re.compile(r"[✂✀✁✃✄]")
    found_gids: set[str] = set()
    for chunk in pd.read_csv(
        GAMES, usecols=["game_id", "white", "black", "result"], chunksize=CHUNKSIZE
    ):
        has_sc = chunk["white"].str.contains(scissors, na=False, regex=True) | chunk[
            "black"
        ].str.contains(scissors, na=False, regex=True)
        sub = chunk[has_sc & (chunk["result"] == "1/2-1/2")]
        found_gids.update(sub["game_id"])
    if not found_gids:
        return 0
    count = 0
    for gid, ml, _ in stream_games(found_gids):
        board = simulate_board(ml)
        if board and board.is_repetition(3):
            count += 1
    return count


# ============================================================
# Q6: Threefold repetition draws, standard, 2024.03.12–2024.11.19
# ============================================================


def q6_threefold_range(gi: GameIndex) -> int:
    mask = (
        gi.is_std
        & (gi.dates >= b"2024.03.12")
        & (gi.dates <= b"2024.11.19")
        & (gi.results == gi.RES_DRAW)
        & (gi.terms == gi.TERM_NORMAL)
    )
    gids = gi.game_ids_where(mask)
    print(f"  Q6: {len(gids):,} draw games")
    count = done = 0
    for gid, ml, _ in stream_games(gids):
        done += 1
        if done % 50_000 == 0:
            print(f"    Q6 {done:,}/{len(gids):,}")
        board = simulate_board(ml)
        if board and board.is_repetition(3):
            count += 1
    return count


# ============================================================
# Q7: Avg white queens at checkmate in tournament-winner games
# ============================================================


def q7_queens_at_mate(gi: GameIndex) -> float:
    tours = pd.read_csv(TOURNAMENTS, usecols=["id", "winner__id"]).dropna()
    tour_winner = {
        str(row["id"]): str(row["winner__id"]).lower() for _, row in tours.iterrows()
    }

    winner_games: set[str] = set()
    for chunk in pd.read_csv(
        GAMES,
        usecols=["game_id", "white", "black", "result", "tournamentid", "termination"],
        chunksize=CHUNKSIZE,
    ):
        sub = chunk[(chunk["termination"] == "Normal") & (chunk["result"] != "1/2-1/2")]
        for _, row in sub.iterrows():
            w = tour_winner.get(str(row["tournamentid"]))
            if not w:
                continue
            if (row["result"] == "1-0" and str(row["white"]).lower() == w) or (
                row["result"] == "0-1" and str(row["black"]).lower() == w
            ):
                winner_games.add(row["game_id"])

    print(f"  Q7: {len(winner_games):,} tournament-winner games")
    counts = []
    done = 0
    for gid, ml, _ in stream_games(winner_games):
        done += 1
        if done % 20_000 == 0:
            print(f"    Q7 {done:,}/{len(winner_games):,}")
        if not ml or "#" not in ml[-1]:
            continue
        board = simulate_board(ml)
        if board:
            counts.append(len(board.pieces(chess.QUEEN, chess.WHITE)))
    return float(np.mean(counts)) if counts else 0.0


# ============================================================
# Q8: Draw on March 20 with pawn promotion as last move
# ============================================================


def q8_draw_march20_promo(gi: GameIndex) -> int:
    mask = (gi.results == gi.RES_DRAW) & np.array(
        [d.endswith(b".03.20") for d in gi.dates]
    )
    gids = gi.game_ids_where(mask)
    print(f"  Q8: {len(gids):,} draw games on March 20")
    return sum(1 for _, ml, _ in stream_games(gids) if ml and "=" in ml[-1])


# ============================================================
# Q9: Berserk timeout losses (games-only)
# ============================================================


def q9_berserk_timeouts(gi: GameIndex) -> tuple[list[str], int]:
    counts: Counter = Counter()
    for chunk in pd.read_csv(
        GAMES,
        usecols=[
            "result",
            "white",
            "black",
            "termination",
            "timecontrol",
            "whitestart",
            "blackstart",
        ],
        chunksize=CHUNKSIZE,
    ):
        sub = chunk[chunk["termination"] == "Time forfeit"]
        sub = sub.copy()
        sub["tc_b"] = sub["timecontrol"].apply(lambda x: parse_tc(x)[0])
        sub["ws"] = sub["whitestart"].apply(parse_clock)
        sub["bs"] = sub["blackstart"].apply(parse_clock)
        for _, row in sub.iterrows():
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
    top = max(counts.values())
    return sorted(p for p, c in counts.items() if c == top)[:10], top


# ============================================================
# Q10: Logistic regression game-level (captures, color, avg_time → win)
# ============================================================


def q10_logit_game(gi: GameIndex) -> dict:
    """Two-pass: compute feature statistics, then fit with SGD (memory-efficient)."""
    print("  Q10: collecting per-game features (SGD, two-pass)")
    gids_s = gi.gids
    res_a = gi.results
    ws_a, bs_a = gi.wstarts, gi.bstarts

    # Pass 1: running mean/variance via Welford's algorithm
    n = 0
    M = np.zeros(3, dtype=np.float64)  # means
    S = np.zeros(3, dtype=np.float64)  # sum of squared deviations (for variance)

    def _update_welford(x: np.ndarray):
        nonlocal n, M, S
        n += 1
        delta = x - M
        M += delta / n
        delta2 = x - M
        S += delta * delta2

    done = 0
    for gid, _, mdf in stream_games():
        done += 1
        idx = np.searchsorted(gids_s, np.array([gid], dtype="S14")[0])
        if idx >= len(gids_s) or gids_s[idx].decode() != gid:
            continue
        r = int(res_a[idx])
        if r == GameIndex.RES_DRAW:
            continue
        ws, bs = int(ws_a[idx]), int(bs_a[idx])
        wcap = bcap = 0
        wt = bt = 0.0
        wn = bn = 0
        pw, pb = ws, bs
        for _, row in mdf.iterrows():
            clk = parse_clock(row["clock"])
            c = row["color"]
            if "x" in str(row["move"]):
                if c == "white":
                    wcap += 1
                else:
                    bcap += 1
            if c == "white" and pw > 0 and clk >= 0:
                wt += pw - clk
                wn += 1
                pw = clk
            elif c == "black" and pb > 0 and clk >= 0:
                bt += pb - clk
                bn += 1
                pb = clk
        _update_welford(np.array([wcap, 1, wt / wn if wn else 0]))
        _update_welford(np.array([bcap, 0, bt / bn if bn else 0]))

    mu = M.copy()
    sd = np.sqrt(S / n) if n > 1 else np.ones(3)
    sd[sd == 0] = 1

    print(f"  Q10: pass 2 fitting SGD (n_samples={n:,})")
    sgd = SGDClassifier(loss="log_loss", random_state=42)
    fitted = False
    bX: list = []
    by: list = []
    BATCH = 200_000

    for gid, _, mdf in stream_games():
        idx = np.searchsorted(gids_s, np.array([gid], dtype="S14")[0])
        if idx >= len(gids_s) or gids_s[idx].decode() != gid:
            continue
        r = int(res_a[idx])
        if r == GameIndex.RES_DRAW:
            continue
        ws, bs = int(ws_a[idx]), int(bs_a[idx])
        wcap = bcap = 0
        wt = bt = 0.0
        wn = bn = 0
        pw, pb = ws, bs
        for _, row in mdf.iterrows():
            clk = parse_clock(row["clock"])
            c = row["color"]
            if "x" in str(row["move"]):
                if c == "white":
                    wcap += 1
                else:
                    bcap += 1
            if c == "white" and pw > 0 and clk >= 0:
                wt += pw - clk
                wn += 1
                pw = clk
            elif c == "black" and pb > 0 and clk >= 0:
                bt += pb - clk
                bn += 1
                pb = clk
        bX.extend(
            [
                [
                    (wcap - mu[0]) / sd[0],
                    (1 - mu[1]) / sd[1],
                    (wt / wn - mu[2] if wn else -mu[2] / sd[2]),
                ],
                [
                    (bcap - mu[0]) / sd[0],
                    (0 - mu[1]) / sd[1],
                    (bt / bn - mu[2] if bn else -mu[2] / sd[2]),
                ],
            ]
        )
        by.extend(
            [1 if r == GameIndex.RES_WHITE else 0, 1 if r == GameIndex.RES_BLACK else 0]
        )
        if len(bX) >= BATCH:
            Xa = np.array(bX, dtype=np.float32)
            ya = np.array(by, dtype=np.int32)
            if not fitted:
                sgd.partial_fit(Xa, ya, classes=[0, 1])
                fitted = True
            else:
                sgd.partial_fit(Xa, ya)
            bX = []
            by = []
    if bX:
        Xa = np.array(bX, dtype=np.float32)
        ya = np.array(by, dtype=np.int32)
        if not fitted:
            sgd.partial_fit(Xa, ya, classes=[0, 1])
        else:
            sgd.partial_fit(Xa, ya)

    return {
        "intercept": float(sgd.intercept_[0]),
        "coef_captures": float(sgd.coef_[0][0]),
        "coef_color_white": float(sgd.coef_[0][1]),
        "coef_avg_time": float(sgd.coef_[0][2]),
        "means": mu.tolist(),
        "stds": sd.tolist(),
    }


# ============================================================
# Q11: Resignations
# ============================================================


def q11_resignations(gi: GameIndex) -> tuple:
    mask = (gi.terms == gi.TERM_NORMAL) & (gi.results != gi.RES_DRAW)
    gids = gi.game_ids_where(mask)
    res_map = {gi.gids[i].decode(): int(gi.results[i]) for i in np.where(mask)[0]}
    w_map: dict[str, str] = {}
    b_map: dict[str, str] = {}
    # Get all unique player names from GameIndex player_names array
    all_players: set[str] = set(gi.player_names.tolist())
    # Get white/black for the relevant games
    for chunk in pd.read_csv(
        GAMES, usecols=["game_id", "white", "black"], chunksize=CHUNKSIZE
    ):
        sub = chunk[chunk["game_id"].isin(gids)]
        w_map.update(zip(sub["game_id"], sub["white"]))
        b_map.update(zip(sub["game_id"], sub["black"]))

    print(
        f"  Q11: checking {len(gids):,} decisive Normal games, "
        f"{len(all_players):,} unique players"
    )
    resign: Counter = Counter()
    done = 0
    for gid, ml, _ in stream_games(gids):
        done += 1
        if done % 100_000 == 0:
            print(f"    Q11 {done:,}/{len(gids):,}")
        if ml and "#" in ml[-1]:
            continue  # checkmate, not resignation
        r = res_map.get(gid, -1)
        if r == GameIndex.RES_WHITE:
            resign[b_map.get(gid, "")] += 1
        elif r == GameIndex.RES_BLACK:
            resign[w_map.get(gid, "")] += 1

    never = sum(1 for p in all_players if resign.get(p, 0) == 0)
    most = max(resign, key=resign.get) if resign else ""
    most_c = resign[most] if resign else 0
    arr = [resign.get(p, 0) for p in all_players]
    med = float(np.median(arr))
    at_med = sum(1 for c in arr if c == med)
    return most, most_c, never, med, at_med


# ============================================================
# Q12: Largest cyclic win within a calendar year (standard, CET)
# ============================================================


def _longest_cycle(graph: dict[int, set]) -> list[int]:
    best: list[int] = []

    def dfs(start, cur, path, vis):
        nonlocal best
        for nb in graph.get(cur, set()):
            if nb == start and len(path) >= 3:
                if len(path) > len(best):
                    best = path[:]
            elif nb not in vis and len(path) < 20:
                vis.add(nb)
                path.append(nb)
                dfs(start, nb, path, vis)
                path.pop()
                vis.remove(nb)

    for node in list(graph):
        dfs(node, node, [node], {node})
    return best


def q12_cyclic_win(gi: GameIndex) -> tuple:
    """Build win graphs using integer player IDs; avoid large timestamp dicts."""
    name_to_id = {name: i for i, name in enumerate(gi.player_names)}

    mask = gi.is_std & (gi.results != gi.RES_DRAW)
    print(f"  Q12: processing {mask.sum():,} standard decisive games")

    # Build year→graph using player IDs (int32) for memory efficiency
    year_graph: dict[int, dict[int, set]] = defaultdict(lambda: defaultdict(set))

    std_dec_idx = np.where(mask)[0]
    udates = gi.utcdates[std_dec_idx]  # S10 "YYYY.MM.DD"
    utimes = gi.utctimes[std_dec_idx]  # int32 seconds from midnight
    results_sub = gi.results[std_dec_idx]
    widxs_sub = gi.widxs[std_dec_idx]
    bidxs_sub = gi.bidxs[std_dec_idx]

    # Vectorized UTC year extraction from S10 bytes "YYYY.MM.DD"
    # First 4 bytes are year ASCII digits
    year_bytes = udates.view("u1").reshape(-1, 10)[:, :4]  # shape (n, 4)
    utc_years = (
        (year_bytes[:, 0].astype(np.int16) - 48) * 1000
        + (year_bytes[:, 1].astype(np.int16) - 48) * 100
        + (year_bytes[:, 2].astype(np.int16) - 48) * 10
        + (year_bytes[:, 3].astype(np.int16) - 48)
    )
    # Adjust UTC→CET: if utctime + 3600 >= 86400 and date is Dec 31, year += 1
    # (simplified: only matters at midnight transitions, very rare for Q12)
    # For correctness, adjust only Dec 31 late-night games
    date_bytes = udates.view("u1").reshape(-1, 10)
    is_dec31 = (
        (date_bytes[:, 5] == ord("1"))
        & (date_bytes[:, 6] == ord("2"))
        & (date_bytes[:, 8] == ord("3"))
        & (date_bytes[:, 9] == ord("1"))
    )
    cet_years = utc_years.copy()
    cet_years[(is_dec31) & (utimes >= 82800)] += (
        1  # 82800 = 23*3600, after 23:00 UTC = +1 in CET
    )

    for j in range(len(std_dec_idx)):
        yr = int(cet_years[j])
        r = int(results_sub[j])
        wid = int(widxs_sub[j])
        bid = int(bidxs_sub[j])
        winner, loser = (wid, bid) if r == GameIndex.RES_WHITE else (bid, wid)
        year_graph[yr][winner].add(loser)

    best_ids: list[int] = []
    best_yr = None
    for yr, g in year_graph.items():
        c = _longest_cycle(g)
        if len(c) > len(best_ids):
            best_ids = c
            best_yr = yr

    if not best_ids:
        return None, []

    # Convert player IDs back to names
    best = [str(gi.player_names[pid]) for pid in best_ids]

    # Find first game in cycle (scan games file for cycle players in best_yr)
    cycle_set = set(best)
    n = len(best)
    first_ts = "9999"
    first_winner = best[0]
    for chunk in pd.read_csv(
        GAMES,
        usecols=["white", "black", "result", "utcdate", "utctime", "variant"],
        chunksize=CHUNKSIZE,
    ):
        sub = chunk[
            (chunk["variant"] == "Standard")
            & chunk["white"].isin(cycle_set)
            & chunk["black"].isin(cycle_set)
        ]
        for _, row in sub.iterrows():
            w, b = str(row["white"]), str(row["black"])
            if row["result"] == "1-0":
                winner, loser = w, b
            else:
                winner, loser = b, w
            # Check if this is a cycle edge
            if winner in cycle_set and loser == best[(best.index(winner) + 1) % n]:
                dt = game_start_cet(str(row["utcdate"]), str(row["utctime"]))
                if dt and dt.year == best_yr:
                    ts = f"{row['utcdate']} {row['utctime']}"
                    if ts < first_ts:
                        first_ts = ts
                        first_winner = winner

    si = best.index(first_winner)
    return best_yr, best[si:] + best[:si]


# ============================================================
# Q13: More vs less time used → win rate
# ============================================================


def q13_time_usage(gi: GameIndex) -> str:
    print("  Q13: collecting time usage")
    gids_s = gi.gids
    res_a = gi.results
    ws_a, bs_a = gi.wstarts, gi.bstarts

    more_w = more_t = less_w = less_t = 0
    done = 0
    for gid, _, mdf in stream_games():
        done += 1
        if done % 500_000 == 0:
            print(f"    Q13 {done:,}")
        idx = np.searchsorted(gids_s, np.array([gid], dtype="S14")[0])
        if idx >= len(gids_s) or gids_s[idx].decode() != gid:
            continue
        r = int(res_a[idx])
        if r == GameIndex.RES_DRAW:
            continue
        pw, pb = int(ws_a[idx]), int(bs_a[idx])
        if pw < 0 or pb < 0:
            continue
        wt = bt = 0.0
        for _, row in mdf.iterrows():
            clk = parse_clock(row["clock"])
            if clk < 0:
                continue
            if row["color"] == "white" and pw >= 0:
                wt += pw - clk
                pw = clk
            elif row["color"] == "black" and pb >= 0:
                bt += pb - clk
                pb = clk
        if wt == bt:
            continue
        if wt > bt:
            more_t += 1
            less_t += 1
            if r == GameIndex.RES_WHITE:
                more_w += 1
            if r == GameIndex.RES_BLACK:
                less_w += 1
        else:
            more_t += 1
            less_t += 1
            if r == GameIndex.RES_BLACK:
                more_w += 1
            if r == GameIndex.RES_WHITE:
                less_w += 1

    mr = more_w / more_t if more_t else 0
    lr = less_w / less_t if less_t else 0
    who = "Több időt felhasználók" if mr > lr else "Kevesebb időt felhasználók"
    return f"{who} nyernek nagyobb arányban (több: {mr:.4f}, kevesebb: {lr:.4f})"


# ============================================================
# Q14: White a2 pawn reaches g8 and promotes
# ============================================================


def q14_a2_to_g8(gi: GameIndex) -> list[str]:
    # Pre-filter: white promotions on g8
    pre: set[str] = set()
    for chunk in pd.read_csv(
        MOVES, usecols=["game_id", "move", "color"], chunksize=CHUNKSIZE
    ):
        sub = chunk[
            (chunk["color"] == "white")
            & chunk["move"].str.contains(r"g8=", na=False, regex=True)
        ]
        pre.update(sub["game_id"].unique())
    date_map: dict[str, str] = {}
    for chunk in pd.read_csv(GAMES, usecols=["game_id", "date"], chunksize=CHUNKSIZE):
        sub = chunk[chunk["game_id"].isin(pre)]
        date_map.update(zip(sub["game_id"], sub["date"]))
    print(f"  Q14: {len(pre):,} games with g8 white promotions")

    hits: set[str] = set()
    for gid, ml, _ in stream_games(pre):
        board = chess.Board()
        tracked = chess.A2
        found = False
        for san in ml:
            try:
                move = board.parse_san(san)
                mover = board.turn
                board.push(move)
            except Exception:
                break
            if mover != chess.WHITE:
                continue
            f, t = move.from_square, move.to_square
            if f == tracked:
                tracked = t
                if t == chess.G8:
                    found = True
                    break
            elif t == tracked:
                # Our pawn might have been captured if black lands here — but
                # we check only white moves here, so this won't trigger incorrectly
                pass
        if found and gid in date_map:
            hits.add(date_map[gid])
    return sorted(hits)[:10]


# ============================================================
# Q15: Non-queen promotions
# ============================================================


def q15_non_queen_promos() -> tuple[int, list]:
    counts: Counter = Counter()
    for chunk in pd.read_csv(MOVES, usecols=["move"], chunksize=CHUNKSIZE):
        for mv in chunk["move"].dropna():
            m = re.search(r"=([RBNQ])", str(mv))
            if m and m.group(1) != "Q":
                counts[m.group(1)] += 1
    return sum(counts.values()), counts.most_common(3)


# ============================================================
# Q16/Q18 helpers: per-player streak computation from GameIndex
# ============================================================


def _build_player_sorted_games(gi: GameIndex) -> tuple:
    """Return sorted (pids, is_draw, utcdates, utctimes, elos) for std games.

    Sorted by (player_id, utcdate, utctime).  White and black records
    are combined: one entry per (player, game) pair.
    Peak memory: ~3.5 GB during build; retained ~2.4 GB sorted arrays.
    """
    import gc

    std = gi.is_std
    n = std.sum()
    idx = np.where(std)[0]

    # White player records
    w_pids = gi.widxs[idx]  # int32
    w_draw = gi.results[idx] == GameIndex.RES_DRAW
    w_wd = gi.widxs[idx] != gi.bidxs[idx]  # unused, just placeholders
    w_dates = gi.utcdates[idx]  # S10
    w_times = gi.utctimes[idx]  # int32
    w_elos = gi.welos[idx]  # int32
    w_wins = gi.results[idx] == GameIndex.RES_WHITE  # white won?

    # Black player records
    b_pids = gi.bidxs[idx]
    b_draw = gi.results[idx] == GameIndex.RES_DRAW
    b_dates = gi.utcdates[idx]
    b_times = gi.utctimes[idx]
    b_elos = gi.belos[idx]
    b_wins = gi.results[idx] == GameIndex.RES_BLACK

    del idx
    gc.collect()

    # Combine
    all_pids = np.concatenate([w_pids, b_pids])
    all_draw = np.concatenate([w_draw, b_draw])
    all_dates = np.concatenate([w_dates, b_dates])
    all_times = np.concatenate([w_times, b_times])
    all_elos = np.concatenate([w_elos, b_elos])
    all_wins = np.concatenate([w_wins, b_wins])
    del w_pids, b_pids, w_draw, b_draw, w_dates, b_dates
    del w_times, b_times, w_elos, b_elos, w_wins, b_wins
    gc.collect()

    # Sort by (player_id, date, time)
    order = np.lexsort((all_times, all_dates, all_pids))
    all_pids = all_pids[order]
    all_draw = all_draw[order]
    all_dates = all_dates[order]
    all_times = all_times[order]
    all_elos = all_elos[order]
    all_wins = all_wins[order]
    del order
    gc.collect()

    return all_pids, all_draw, all_wins, all_dates, all_times, all_elos


def _compute_streaks(pids, cond, dates, times, elos, gi, tiebreak_elo=True) -> tuple:
    """Compute best streak where cond[i] is True, grouping by player.

    Returns (best_pid, best_start_date_bytes, best_end_date_bytes, best_n, best_elo,
             all_tied_pids) where all_tied_pids is a list of (pid, start_b, end_b, n, elo)
             for all players reaching best_n.
    """
    best_n = 0
    best_pid = -1
    best_s = b""
    best_e = b""
    best_elo = 0
    all_tied: list = []

    boundaries = np.concatenate([[0], np.where(np.diff(pids))[0] + 1, [len(pids)]])
    for k in range(len(boundaries) - 1):
        s, e = int(boundaries[k]), int(boundaries[k + 1])
        pid = int(pids[s])
        pc, pd_, pe = cond[s:e], dates[s:e], elos[s:e]

        st = 0
        ss_idx = 0
        ss_elo = 0
        for j in range(e - s):
            if pc[j]:
                if st == 0:
                    ss_idx = j
                st += 1
                ss_elo = int(pe[j]) if pe[j] > 0 else ss_elo
            else:
                if st > 0:
                    # Update candidate
                    if st > best_n:
                        best_n = st
                        all_tied = [
                            (pid, pd_[ss_idx], pd_[j - 1 if j > 0 else 0], st, ss_elo)
                        ]
                        if tiebreak_elo:
                            best_pid = pid
                            best_s = pd_[ss_idx]
                            best_e = pd_[j - 1 if j > 0 else 0]
                            best_elo = ss_elo
                        else:
                            best_pid = pid
                            best_s = pd_[ss_idx]
                            best_e = pd_[j - 1 if j > 0 else 0]
                    elif st == best_n:
                        all_tied.append(
                            (pid, pd_[ss_idx], pd_[j - 1 if j > 0 else 0], st, ss_elo)
                        )
                        if tiebreak_elo and ss_elo > best_elo:
                            best_pid = pid
                            best_s = pd_[ss_idx]
                            best_e = pd_[j - 1 if j > 0 else 0]
                            best_elo = ss_elo
                st = 0
        if st > 0:  # ongoing streak at end
            j = e - s - 1
            if st > best_n:
                best_n = st
                all_tied = [(pid, pd_[ss_idx], pd_[j], st, ss_elo)]
                best_pid = pid
                best_s = pd_[ss_idx]
                best_e = pd_[j]
                best_elo = ss_elo
            elif st == best_n:
                all_tied.append((pid, pd_[ss_idx], pd_[j], st, ss_elo))
                if tiebreak_elo and ss_elo > best_elo:
                    best_pid = pid
                    best_s = pd_[ss_idx]
                    best_e = pd_[j]
                    best_elo = ss_elo

    return best_pid, best_s, best_e, best_n, best_elo, all_tied


# ============================================================
# Q16: Longest draw streak (standard)
# ============================================================


def q16_draw_streak(gi: GameIndex) -> tuple:
    print("  Q16: building per-player sorted games from GameIndex")
    pids, is_draw, _, dates, times, elos = _build_player_sorted_games(gi)
    best_pid, best_s, best_e, best_n, _, _ = _compute_streaks(
        pids, is_draw, dates, times, elos, gi, tiebreak_elo=True
    )
    import gc

    del pids, is_draw, dates, times, elos
    gc.collect()
    if best_pid < 0:
        return None, None, None, 0
    p = str(gi.player_names[best_pid])
    s = best_s.decode() if isinstance(best_s, bytes) else str(best_s)
    e = best_e.decode() if isinstance(best_e, bytes) else str(best_e)
    return p, s, e, best_n


# ============================================================
# Q17: Logistic regression per-move (capture ~ time_elapsed + color)
# ============================================================


def q17_logit_move(gi: GameIndex) -> dict:
    print("  Q17: SGD per-move logit")
    gids_s = gi.gids
    ws_a, bs_a = gi.wstarts, gi.bstarts
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
        idx = np.searchsorted(gids_s, np.array([gid], dtype="S14")[0])
        if idx >= len(gids_s) or gids_s[idx].decode() != gid:
            continue
        pw, pb = int(ws_a[idx]), int(bs_a[idx])
        elapsed = 0.0
        for _, row in mdf.iterrows():
            clk = parse_clock(row["clock"])
            c = row["color"]
            if c == "white" and pw > 0 and clk >= 0:
                elapsed += pw - clk
                pw = clk
            elif c == "black" and pb > 0 and clk >= 0:
                elapsed += pb - clk
                pb = clk
            bX.append([elapsed, 1 if c == "white" else 0])
            by.append(1 if "x" in str(row["move"]) else 0)
            if len(bX) >= BATCH:
                Xa = np.array(bX, dtype=np.float32)
                ya = np.array(by, dtype=np.int32)
                if not fitted:
                    sgd.partial_fit(Xa, ya, classes=[0, 1])
                    fitted = True
                else:
                    sgd.partial_fit(Xa, ya)
                bX = []
                by = []
    if bX:
        Xa, ya = np.array(bX, dtype=np.float32), np.array(by, dtype=np.int32)
        if not fitted:
            sgd.partial_fit(Xa, ya, classes=[0, 1])
        else:
            sgd.partial_fit(Xa, ya)
    return {
        "intercept": float(sgd.intercept_[0]),
        "coef_time_elapsed": float(sgd.coef_[0][0]),
        "coef_color_white": float(sgd.coef_[0][1]),
    }


# ============================================================
# Q18: Longest winless streak (standard)
# ============================================================


def q18_winless_streak(gi: GameIndex) -> tuple:
    print("  Q18: building per-player sorted games from GameIndex")
    pids, is_draw, is_win, dates, times, elos = _build_player_sorted_games(gi)
    is_winless = ~is_win
    import gc

    del is_win, is_draw, elos
    gc.collect()

    _, _, _, best_n, _, all_tied = _compute_streaks(
        pids,
        is_winless,
        dates,
        times,
        np.zeros(len(pids), dtype=np.int32),
        gi,
        tiebreak_elo=False,
    )
    del pids, is_winless, dates, times
    gc.collect()

    if best_n == 0:
        return None, None, None, 0

    # Tiebreak: Hungarian alphabet order after "Lili"
    lk = hu_key("Lili")
    tied_named = [(str(gi.player_names[pid]), s, e) for pid, s, e, n, _ in all_tied]
    after = [(p, s, e) for p, s, e in tied_named if hu_key(p) > lk]
    if after:
        winner_p, winner_s, winner_e = min(after, key=lambda x: hu_key(x[0]))
    else:
        winner_p, winner_s, winner_e = min(tied_named, key=lambda x: hu_key(x[0]))

    s = winner_s.decode() if isinstance(winner_s, bytes) else str(winner_s)
    e = winner_e.decode() if isinstance(winner_e, bytes) else str(winner_e)
    return winner_p, s, e, best_n


# ============================================================
# Q19: 50-move rule draws, standard, 2026.03.15–2026.10.14
# ============================================================


def q19_fifty_move(gi: GameIndex) -> int:
    mask = (
        gi.is_std
        & (gi.dates >= b"2026.03.15")
        & (gi.dates <= b"2026.10.14")
        & (gi.results == gi.RES_DRAW)
        & (gi.terms == gi.TERM_NORMAL)
    )
    gids = gi.game_ids_where(mask)
    print(f"  Q19: {len(gids):,} draw games")
    count = 0
    for gid, ml, _ in stream_games(gids):
        board = simulate_board(ml)
        if board and board.is_fifty_moves():
            count += 1
    return count


# ============================================================
# Q20: Queen's Gambit % per year (04.21–05.18 CET, standard)
# ============================================================


def q20_queens_gambit(gi: GameIndex) -> dict[int, float]:
    # No pre-filter to avoid large game_id set; check is_std inline
    yr_total: Counter = Counter()
    yr_qg: Counter = Counter()

    print(f"  Q20: scanning all moves, checking standard games in date range")
    for gid, ml, _ in stream_games():  # no filter = all games
        idx = gi._idx(gid)
        if idx < 0 or not gi.is_std[idx]:
            continue
        ud = gi.utcdates[idx].decode()
        ut = int(gi.utctimes[idx])
        if not ud:
            continue
        try:
            dt = datetime.strptime(ud, "%Y.%m.%d")
            dt_utc = UTC.localize(dt).replace(
                hour=ut // 3600, minute=(ut % 3600) // 60, second=ut % 60
            )
            dt_cet = dt_utc.astimezone(CET)
        except Exception:
            continue
        md = (dt_cet.month, dt_cet.day)
        if not ((4, 21) <= md <= (5, 18)):
            continue
        yr = dt_cet.year
        if len(ml) < 3:
            continue
        yr_total[yr] += 1
        if (
            ml[0].rstrip("+#") == "d4"
            and ml[1].rstrip("+#") == "d5"
            and ml[2].rstrip("+#") == "c4"
        ):
            yr_qg[yr] += 1

    return {yr: yr_qg[yr] / yr_total[yr] for yr in sorted(yr_total) if yr_total[yr] > 0}


# ============================================================
# Q21: Standard games potentially spanning New Year's Eve
# ============================================================


def q21_year_spanning(gi: GameIndex) -> dict[int, int]:
    mask = gi.is_std & np.array([d.endswith(b".12.31") for d in gi.dates])
    result: dict[int, int] = {}
    idxs = np.where(mask)[0]
    for i in idxs:
        ud = gi.utcdates[i].decode()
        ut = int(gi.utctimes[i])
        if not ud:
            continue
        try:
            dt = datetime.strptime(ud, "%Y.%m.%d")
            dt_utc = UTC.localize(dt).replace(
                hour=ut // 3600, minute=(ut % 3600) // 60, second=ut % 60
            )
            dt_cet = dt_utc.astimezone(CET)
        except Exception:
            continue
        if dt_cet.month != 12 or dt_cet.day != 31:
            continue
        yr = dt_cet.year
        midnight = CET.localize(datetime(yr + 1, 1, 1))
        secs = (midnight - dt_cet).total_seconds()
        total = int(gi.wstarts[i]) + int(gi.bstarts[i])
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
        b = (f[(i + 1) % 4], r[(i + 1) % 4])
        if a[0] != b[0] and a[1] != b[1]:
            return False
    return True


def _rect_area(s1, s2, s3, s4) -> int:
    f = [chess.square_file(s) for s in (s1, s2, s3, s4)]
    r = [chess.square_rank(s) for s in (s1, s2, s3, s4)]
    return (max(f) - min(f)) * (max(r) - min(r))


def _sim_rects(ml: list[str]) -> tuple[int, int, int]:
    board = chess.Board()
    hist: dict[chess.Square, list] = {}  # orig_sq → position history
    sq_orig: dict[chess.Square, chess.Square] = {}  # cur_sq → orig_sq

    for sq in chess.SQUARES:
        if board.piece_at(sq):
            hist[sq] = [sq]
            sq_orig[sq] = sq

    wr = br = best_area = 0

    for san in ml:
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
        orig = sq_orig.get(from_sq)
        if orig is None:
            continue

        # Remove captured piece
        if not is_ep:
            sq_orig.pop(to_sq, None)
        else:
            ep_sq = chess.square(chess.square_file(to_sq), chess.square_rank(from_sq))
            sq_orig.pop(ep_sq, None)

        sq_orig.pop(from_sq, None)
        sq_orig[to_sq] = orig
        if orig not in hist:
            hist[orig] = [from_sq]
        hist[orig].append(to_sq)

        # Castling rook
        if is_castle:
            if mover == chess.WHITE:
                rf, rt = (chess.H1, chess.F1) if is_ks else (chess.A1, chess.D1)
            else:
                rf, rt = (chess.H8, chess.F8) if is_ks else (chess.A8, chess.D8)
            ro = sq_orig.pop(rf, None)
            if ro is not None:
                sq_orig[rt] = ro
                if ro not in hist:
                    hist[ro] = [rf]
                hist[ro].append(rt)

        # Check new rectangles in this piece's history
        h = hist[orig]
        if len(h) >= 4:
            for i in range(len(h) - 3):
                if _is_rect(h[i], h[i + 1], h[i + 2], h[i + 3]):
                    area = _rect_area(h[i], h[i + 1], h[i + 2], h[i + 3])
                    if mover == chess.WHITE:
                        wr += 1
                    else:
                        br += 1
                    best_area = max(best_area, area)

    return wr, br, best_area


def q22_rectangles(gi: GameIndex) -> tuple:
    print("  Q22: rectangle paths (slow — all games, single pass)")
    gids_s = gi.gids
    player_rects: Counter = Counter()
    global_max = 0
    done = 0
    for gid, ml, _ in stream_games():
        done += 1
        if done % 200_000 == 0:
            print(f"    Q22 {done:,}")
        wr, br, area = _sim_rects(ml)
        # Look up player names from GameIndex
        idx = np.searchsorted(gids_s, np.array([gid], dtype="S14")[0])
        if idx < len(gids_s) and gids_s[idx].decode() == gid:
            w = gi.player_at(idx, "white")
            b = gi.player_at(idx, "black")
        else:
            w = b = ""
        player_rects[w] += wr
        player_rects[b] += br
        global_max = max(global_max, area)

    if not player_rects:
        return None, 0, 0
    top = max(player_rects.values())
    best_p = min(p for p, c in player_rects.items() if c == top and p)
    return best_p, top, global_max


# ============================================================
# Q23: Castling checkmates
# ============================================================


def q23_castle_mates(gi: GameIndex) -> list[str]:
    print("  Q23: scanning castling checkmates (single pass)")
    # Single moves scan: collect (game_id, color) for castling checkmates
    castle_games: dict[str, str] = {}
    for chunk in pd.read_csv(
        MOVES, usecols=["game_id", "move", "color"], chunksize=CHUNKSIZE
    ):
        sub = chunk[chunk["move"].isin(["O-O#", "O-O-O#"])]
        for _, row in sub.iterrows():
            castle_games[str(row["game_id"])] = str(row["color"])

    print(f"  Q23: {len(castle_games):,} castling checkmates found")
    counts: Counter = Counter()
    gids_s = gi.gids
    for gid, color in castle_games.items():
        idx = np.searchsorted(gids_s, np.array([gid], dtype="S14")[0])
        if idx < len(gids_s) and gids_s[idx].decode() == gid:
            counts[gi.player_at(idx, color)] += 1

    if not counts:
        return []
    top = max(counts.values())
    return sorted(p for p, c in counts.items() if c == top)[:10]


# ============================================================
# Q24: En passant by white in 3-min Indian openings (ECO E)
# ============================================================


def q24_en_passant_indian(gi: GameIndex) -> int:
    mask = (gi.tcbases == 180) & np.array([e[:1] == b"E" for e in gi.ecos])
    gids = gi.game_ids_where(mask)
    print(f"  Q24: {len(gids):,} Indian 3-min games")
    count = done = 0
    for gid, ml, _ in stream_games(gids):
        done += 1
        if done % 10_000 == 0:
            print(f"    Q24 {done:,}/{len(gids):,}")
        board = chess.Board()
        for san in ml:
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


def write_answers(ans: dict) -> None:
    lines = ["# Chess Data Analysis — Answers\n"]
    for q in sorted(ans):
        lines.append(f"## {q}. kérdés\n\n{ans[q]}\n")
    with open(OUTPUT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\nWritten to {OUTPUT}")


# ============================================================
# Main
# ============================================================


def main() -> None:
    gi = GameIndex()
    gi.build()

    ans: dict[int, str] = {}

    # ---- Games-only (no moves needed) ----
    print("\n=== Games-only questions ===")

    print("Q9")
    w9, c9 = q9_berserk_timeouts(gi)
    ans[9] = f"Legtöbb berserk timeout vereség ({c9}x): {', '.join(w9)}"

    print("Q21")
    q21 = q21_year_spanning(gi)
    ans[21] = "\n".join(f"{yr}: {c}" for yr, c in sorted(q21.items())) or "0"

    # ---- Questions reading games file (no moves) ----
    print("\n=== Games-file questions ===")

    print("Q16")
    p16, s16, e16, n16 = q16_draw_streak(gi)
    ans[16] = f"{p16} | {s16} – {e16} | {n16} parti"

    print("Q18")
    p18, s18, e18, n18 = q18_winless_streak(gi)
    ans[18] = f"{p18} | {s18} – {e18} | {n18} parti"

    print("Q12")
    yr12, cyc12 = q12_cyclic_win(gi)
    ans[12] = (
        (f"Év: {yr12} | " + " → ".join(cyc12) + f" → {cyc12[0]}") if cyc12 else "Nincs"
    )

    # ---- Pattern matching (no board sim) ----
    print("\n=== Pattern matching ===")

    print("Q5")
    ans[5] = str(q5_threefold_scissors())
    print("Q8")
    ans[8] = str(q8_draw_march20_promo(gi))
    print("Q15")
    tot15, top15 = q15_non_queen_promos()
    ans[15] = f"Nem vezérre: {tot15} | Top 3: " + ", ".join(
        f"{p}:{c}" for p, c in top15
    )
    print("Q23")
    ans[23] = ", ".join(q23_castle_mates(gi))

    # ---- Board simulation (filtered) ----
    print("\n=== Board simulation (filtered) ===")

    print("Q1")
    ans[1] = str(q1_material_disadvantage(gi))
    print("Q3")
    ans[3] = str(q3_castling_lost(gi))
    print("Q6")
    ans[6] = str(q6_threefold_range(gi))
    print("Q7")
    a7 = q7_queens_at_mate(gi)
    ans[7] = f"{a7:.4f}"
    print("Q19")
    ans[19] = str(q19_fifty_move(gi))
    print("Q24")
    ans[24] = str(q24_en_passant_indian(gi))
    print("Q14")
    d14 = q14_a2_to_g8(gi)
    ans[14] = ", ".join(d14) if d14 else "Nincs"

    # ---- Full-scan moves questions ----
    print("\n=== Full moves scan ===")

    print("Q4")
    diff4 = q4_rook_distances()
    ans[4] = f"Fehér − Fekete bástya távolság összege: {diff4} mező"

    print("Q10")
    p10 = q10_logit_game(gi)
    ans[10] = (
        f"Intercept: {p10['intercept']:.6f}, "
        f"captures: {p10['coef_captures']:.6f}, "
        f"white: {p10['coef_color_white']:.6f}, "
        f"avg_time: {p10['coef_avg_time']:.6f} "
        f"(std; μ={[f'{v:.3f}' for v in p10['means']]}, "
        f"σ={[f'{v:.3f}' for v in p10['stds']]})"
    )

    print("Q11")
    m11, c11, nv11, med11, am11 = q11_resignations(gi)
    ans[11] = (
        f"Legtöbbet feladott: {m11} ({c11}x) | "
        f"Soha nem adta fel: {nv11} | Mediánban ({med11:.1f}): {am11}"
    )

    print("Q13")
    ans[13] = q13_time_usage(gi)

    print("Q17")
    p17 = q17_logit_move(gi)
    ans[17] = (
        f"Intercept: {p17['intercept']:.6f}, "
        f"time_elapsed: {p17['coef_time_elapsed']:.6f}, "
        f"white: {p17['coef_color_white']:.6f}"
    )

    print("Q20")
    q20 = q20_queens_gambit(gi)
    ans[20] = "\n".join(
        f"{yr}: {r:.4f} ({r * 100:.2f}%)" for yr, r in sorted(q20.items())
    )

    print("Q2")
    ans[2] = q2_left_knight(gi)

    print("Q22")
    top22, n22, area22 = q22_rectangles(gi)
    ans[22] = f"Játékos: {top22} ({n22} téglalap) | Legnagyobb terület: {area22}"

    write_answers(ans)


if __name__ == "__main__":
    main()
