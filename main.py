#!/usr/bin/env python3
"""Chess data analysis — answers all 24 questions.

Memory & I/O design
-------------------
1. Build a compact ``GameIndex`` (numpy arrays sorted by game_id) in a single
   pass over ``games.csv.gz``. ~3 GB.
2. Pure-metadata questions (Q9, Q12, Q16, Q18, Q21) finalize from the index
   alone — no moves file scan.
3. Every other question runs in **one** streaming pass over ``moves.csv.gz``.
   Per-game work is dispatched to a list of ``Question`` handlers; each game
   is walked at most once, and the walker emits ``MoveEvent`` records that
   handlers consume.
4. ``iterrows()`` is avoided everywhere. Per-game data arrives as plain Python
   lists (SANs + parsed clocks).

Peak memory is roughly 4–6 GB.
"""

from __future__ import annotations

import re
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Iterator, Optional

import chess
import numpy as np
import pandas as pd
import pytz
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm

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

SCISSORS_RE = re.compile(r"[✂✀✁✃✄]")
PROMO_RE = re.compile(r"=([RBNQ])")
STARTING_PIECE_SQUARES = tuple(sq for sq in chess.SQUARES if chess.Board().piece_at(sq))

# ============================================================
# Helpers
# ============================================================


def parse_clock(s) -> int:
    """``H:MM:SS`` → seconds. Returns -1 on error."""
    try:
        h, m, sec = str(s).split(":")
        return int(h) * 3600 + int(m) * 60 + int(sec)
    except Exception:
        return -1


def parse_clock_vec(series: pd.Series) -> np.ndarray:
    """Vectorised ``H:MM:SS`` → seconds. Invalid rows yield -1."""
    parts = series.fillna("").astype(str).str.split(":", expand=True)
    if parts.shape[1] < 3:
        return np.full(len(series), -1, dtype=np.int32)
    h = pd.to_numeric(parts[0], errors="coerce")
    m = pd.to_numeric(parts[1], errors="coerce")
    s = pd.to_numeric(parts[2], errors="coerce")
    out = (h * 3600 + m * 60 + s).fillna(-1).astype(np.int32).to_numpy()
    return out


def parse_tc_base(s) -> int:
    """Time control base in seconds (the part before the ``+``)."""
    try:
        return int(str(s).split("+")[0])
    except Exception:
        return 0


def utc_to_cet(udate: str, utime_secs: int) -> Optional[datetime]:
    if not udate:
        return None
    try:
        dt = datetime.strptime(udate, "%Y.%m.%d").replace(
            hour=utime_secs // 3600,
            minute=(utime_secs % 3600) // 60,
            second=utime_secs % 60,
        )
        return UTC.localize(dt).astimezone(CET)
    except Exception:
        return None


def count_material(board: chess.Board) -> tuple[int, int]:
    vals = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9}
    w = sum(len(board.pieces(pt, chess.WHITE)) * v for pt, v in vals.items())
    b = sum(len(board.pieces(pt, chess.BLACK)) * v for pt, v in vals.items())
    return w, b


def simulate_board(sans: list[str]) -> Optional[chess.Board]:
    """Replay a SAN list onto a fresh board; tolerates malformed moves."""
    board = chess.Board()
    for san in sans:
        try:
            board.push_san(san)
        except Exception:
            return board
    return board


# Hungarian alphabet collation (Q18)
_HU = {
    "a": 1, "á": 2, "b": 3, "c": 4, "cs": 5, "d": 6, "dz": 7, "dzs": 8,
    "e": 9, "é": 10, "f": 11, "g": 12, "gy": 13, "h": 14, "i": 15, "í": 16,
    "j": 17, "k": 18, "l": 19, "ly": 20, "m": 21, "n": 22, "ny": 23, "o": 24,
    "ó": 25, "ö": 26, "ő": 27, "p": 28, "q": 29, "r": 30, "s": 31, "sz": 32,
    "t": 33, "ty": 34, "u": 35, "ú": 36, "ü": 37, "ű": 38, "v": 39, "w": 40,
    "x": 41, "y": 42, "z": 43, "zs": 44,
}


def hu_key(name: str) -> tuple[int, ...]:
    s = name.lower()
    out: list[int] = []
    i = 0
    while i < len(s):
        for L in (3, 2, 1):
            if s[i : i + L] in _HU:
                out.append(_HU[s[i : i + L]])
                i += L
                break
        else:
            out.append(200 + ord(s[i]))
            i += 1
    return tuple(out)


# ============================================================
# GameIndex: compact, sorted-by-game_id metadata
# ============================================================


class GameIndex:
    """All per-game metadata kept as parallel numpy arrays sorted by gid.

    Lookups by gid are O(log n) via binary search. No moves data here.
    """

    TERM_NORMAL, TERM_TIMEFORFEIT, TERM_ABANDONED, TERM_OTHER = 0, 1, 2, 3
    RES_DRAW, RES_WHITE, RES_BLACK = 0, 1, 2

    _RES_MAP = {"1-0": RES_WHITE, "0-1": RES_BLACK, "1/2-1/2": RES_DRAW}
    _TERM_MAP = {"Normal": TERM_NORMAL, "Time forfeit": TERM_TIMEFORFEIT, "Abandoned": TERM_ABANDONED}

    def build(self) -> None:
        cols = [
            "game_id", "result", "variant", "utcdate", "utctime", "termination",
            "timecontrol", "whitestart", "blackstart", "eco", "whiteelo", "blackelo",
            "white", "black", "tournamentid",
        ]
        # Pre-load tournament winners (small) so we can flag tournament-winner games inline.
        tdf = pd.read_csv(TOURNAMENTS, usecols=["id", "winner__id"]).dropna(subset=["winner__id"])
        tour_winner = dict(zip(tdf["id"].astype(str), tdf["winner__id"].astype(str).str.lower()))
        del tdf

        parts: dict[str, list] = defaultdict(list)
        player_ids: dict[str, int] = {}

        print("Building GameIndex (single pass over games)...")
        # Estimate total chunks for tqdm by guessing; tqdm tolerates total=None.
        with tqdm(desc="games", unit=" rows", unit_scale=True) as pb:
            for chunk in pd.read_csv(
                GAMES, usecols=cols, chunksize=CHUNKSIZE,
                dtype={"whiteelo": "Int32", "blackelo": "Int32"},
            ):
                pb.update(len(chunk))
                parts["gid"].append(chunk["game_id"].to_numpy().astype("S14"))
                parts["res"].append(
                    chunk["result"].map(self._RES_MAP).fillna(self.RES_DRAW)
                    .astype(np.int8).to_numpy()
                )
                parts["std"].append(
                    (chunk["variant"].astype(str).str.strip().str.lower() == "standard").to_numpy()
                )
                parts["udate"].append(chunk["utcdate"].fillna("").to_numpy().astype("S10"))
                parts["utime"].append(parse_clock_vec(chunk["utctime"]))
                parts["term"].append(
                    chunk["termination"].map(self._TERM_MAP).fillna(self.TERM_OTHER)
                    .astype(np.int8).to_numpy()
                )
                parts["tcbase"].append(
                    chunk["timecontrol"].astype(str).str.split("+", expand=True)[0]
                    .pipe(pd.to_numeric, errors="coerce").fillna(0).clip(0, 32767)
                    .astype(np.int16).to_numpy()
                )
                parts["ws"].append(parse_clock_vec(chunk["whitestart"]).astype(np.int32))
                parts["bs"].append(parse_clock_vec(chunk["blackstart"]).astype(np.int32))
                parts["eco"].append(chunk["eco"].fillna("").to_numpy().astype("S3"))
                parts["welo"].append(
                    chunk["whiteelo"].fillna(0).clip(0, 32767).astype(np.int16).to_numpy()
                )
                parts["belo"].append(
                    chunk["blackelo"].fillna(0).clip(0, 32767).astype(np.int16).to_numpy()
                )

                # Tournament-winner-decisive flag (computed inline so we never re-scan games).
                tids = chunk["tournamentid"].astype(str)
                expected = tids.map(tour_winner)
                wlow = chunk["white"].fillna("").astype(str).str.lower()
                blow = chunk["black"].fillna("").astype(str).str.lower()
                rcol = chunk["result"]
                is_tw = (
                    ((rcol == "1-0") & (wlow == expected))
                    | ((rcol == "0-1") & (blow == expected))
                ).to_numpy()
                parts["tw"].append(is_tw)

                # Player IDs (vectorised).
                w_names = chunk["white"].fillna("").astype(str).to_numpy()
                b_names = chunk["black"].fillna("").astype(str).to_numpy()
                for n in w_names:
                    if n not in player_ids:
                        player_ids[n] = len(player_ids)
                for n in b_names:
                    if n not in player_ids:
                        player_ids[n] = len(player_ids)
                parts["wid"].append(np.fromiter((player_ids[n] for n in w_names), dtype=np.int32, count=len(w_names)))
                parts["bid"].append(np.fromiter((player_ids[n] for n in b_names), dtype=np.int32, count=len(b_names)))

        print(f"  concatenating {sum(len(p) for p in parts['gid']):,} games...")
        gids_raw = np.concatenate(parts.pop("gid"))
        order = np.argsort(gids_raw, kind="stable")
        self.gids = gids_raw[order].copy()
        del gids_raw

        def take(key: str) -> np.ndarray:
            arr = np.concatenate(parts.pop(key))
            return arr[order]

        self.results = take("res")
        self.is_std = take("std")
        self.utcdates = take("udate")
        self.utctimes = take("utime")
        self.terms = take("term")
        self.tcbases = take("tcbase")
        self.wstarts = take("ws")
        self.bstarts = take("bs")
        self.ecos = take("eco")
        self.welos = take("welo")
        self.belos = take("belo")
        self.is_tour_winner = take("tw")
        self.widxs = take("wid")
        self.bidxs = take("bid")

        names_sorted = sorted(player_ids, key=player_ids.get)
        self.player_names = np.array(names_sorted, dtype=object)
        del player_ids, names_sorted, order

        # Derived: scissors-emoji games (used by Q5).
        sc = np.array([bool(SCISSORS_RE.search(n)) for n in self.player_names], dtype=bool)
        self.has_scissors = sc[self.widxs] | sc[self.bidxs]

        # Lazy gid-string cache for faster idx lookups during moves pass.
        self._gid_to_idx: Optional[dict[bytes, int]] = None

        mb = sum(getattr(self, k).nbytes for k in (
            "gids", "results", "is_std", "utcdates", "utctimes", "terms",
            "tcbases", "wstarts", "bstarts", "ecos", "welos", "belos",
            "is_tour_winner", "widxs", "bidxs", "has_scissors",
        )) // 1024 // 1024
        print(f"  GameIndex ready: {len(self.gids):,} games, ~{mb} MB metadata")

    # --- lookup ---------------------------------------------------------
    def build_idx_lookup(self) -> None:
        """Build a hashtable gid → idx for O(1) lookups during moves pass.

        Uses bytes (S14) as keys; ~80 bytes per entry × 60M ≈ 4.5 GB. Skip if
        memory is tight and rely on ``binsearch`` instead.
        """
        self._gid_to_idx = {bytes(g): i for i, g in enumerate(self.gids)}

    def idx(self, gid_b: bytes) -> int:
        if self._gid_to_idx is not None:
            return self._gid_to_idx.get(gid_b, -1)
        i = np.searchsorted(self.gids, gid_b)
        if i < len(self.gids) and self.gids[i] == gid_b:
            return int(i)
        return -1

    def player_at(self, idx: int, color: str) -> str:
        pid = int(self.widxs[idx]) if color == "white" else int(self.bidxs[idx])
        return str(self.player_names[pid])

    # --- mask helpers ---------------------------------------------------
    def cet_year(self) -> np.ndarray:
        """Per-game CET year as int16, derived from utcdate + utctime."""
        n = len(self.gids)
        ud_bytes = self.utcdates.view("u1").reshape(n, 10)
        utc_year = (
            (ud_bytes[:, 0].astype(np.int16) - 48) * 1000
            + (ud_bytes[:, 1].astype(np.int16) - 48) * 100
            + (ud_bytes[:, 2].astype(np.int16) - 48) * 10
            + (ud_bytes[:, 3].astype(np.int16) - 48)
        )
        # CET = UTC+1 (or +2 in summer, but year-rollover only relevant on Dec 31).
        is_dec31 = (
            (ud_bytes[:, 5] == ord("1")) & (ud_bytes[:, 6] == ord("2"))
            & (ud_bytes[:, 8] == ord("3")) & (ud_bytes[:, 9] == ord("1"))
        )
        rolled = is_dec31 & (self.utctimes >= 23 * 3600)
        utc_year[rolled] += 1
        return utc_year


# ============================================================
# Streaming moves: yields (gid_bytes, sans, clocks) per game
# ============================================================


def stream_games() -> Iterator[tuple[bytes, list[str], list[int]]]:
    """Stream complete games from moves.csv.gz in file order.

    Avoids pandas groupby/iterrows; uses numpy slicing on chunk arrays.
    """
    cur_gid: Optional[bytes] = None
    cur_sans: list[str] = []
    cur_clocks: list[int] = []

    for chunk in pd.read_csv(
        MOVES, usecols=["game_id", "move", "clock"], chunksize=CHUNKSIZE,
    ):
        gids = chunk["game_id"].to_numpy().astype("S14")
        moves = chunk["move"].fillna("").to_numpy(dtype=object)
        clocks = parse_clock_vec(chunk["clock"])

        n = len(gids)
        # Boundary indices where gid changes (within the chunk).
        diff = np.concatenate(([True], gids[1:] != gids[:-1]))
        starts = np.flatnonzero(diff)
        ends = np.append(starts[1:], n)

        for s, e in zip(starts, ends):
            g = bytes(gids[s])
            if g == cur_gid:
                cur_sans.extend(moves[s:e].tolist())
                cur_clocks.extend(clocks[s:e].tolist())
            else:
                if cur_gid is not None:
                    yield cur_gid, cur_sans, cur_clocks
                cur_gid = g
                cur_sans = moves[s:e].tolist()
                cur_clocks = clocks[s:e].tolist()

    if cur_gid is not None:
        yield cur_gid, cur_sans, cur_clocks


# ============================================================
# Walker: replays one game once, dispatches MoveEvent to handlers
# ============================================================


@dataclass(slots=True)
class MoveEvent:
    ply: int
    san: str
    move: chess.Move
    from_sq: int
    to_sq: int
    mover: bool
    is_capture: bool
    is_castle: bool
    is_kingside: bool
    is_en_passant: bool
    ep_captured_sq: int
    promotion: Optional[int]


def walk_game(sans: list[str], walkers: list["Question"], gd: "GameData") -> Optional[chess.Board]:
    """Replay sans, dispatch MoveEvents to walker handlers, return final board.

    Tolerates malformed SANs by stopping early and returning the board so far.
    """
    board = chess.Board()
    for ply, san in enumerate(sans):
        try:
            move = board.parse_san(san)
        except Exception:
            return board
        is_castle = board.is_castling(move)
        is_kingside = is_castle and board.is_kingside_castling(move)
        is_ep = board.is_en_passant(move)
        is_capture = board.is_capture(move) or is_ep
        mover = board.turn
        from_sq = move.from_square
        to_sq = move.to_square
        ep_captured = -1
        if is_ep:
            ep_captured = chess.square(chess.square_file(to_sq), chess.square_rank(from_sq))
        board.push(move)
        ev = MoveEvent(
            ply=ply, san=san, move=move,
            from_sq=from_sq, to_sq=to_sq, mover=mover,
            is_capture=is_capture, is_castle=is_castle, is_kingside=is_kingside,
            is_en_passant=is_ep, ep_captured_sq=ep_captured,
            promotion=move.promotion,
        )
        for q in walkers:
            q.on_move(gd, ev)
    return board


# ============================================================
# Question protocol + per-game data bag
# ============================================================


@dataclass(slots=True)
class GameData:
    gid: bytes
    idx: int
    sans: list[str]
    clocks: list[int]


class Question:
    """Subclass and override what you need.

    - ``relevant(idx)``: whether this game contributes to the question.
    - ``needs_walk``: emit per-move events.
    - ``needs_board``: hand back the final ``chess.Board`` to ``on_end``.
    - ``on_end(gd, board)``: per-game finalisation. ``board`` may be None.
    - ``finalize()``: return the final answer string.
    """

    name: str = ""
    needs_walk: bool = False
    needs_board: bool = False

    def __init__(self, gi: GameIndex):
        self.gi = gi

    def relevant(self, idx: int) -> bool:
        return idx >= 0

    def begin_game(self, gd: GameData) -> None:  # for stateful walkers
        pass

    def on_move(self, gd: GameData, ev: MoveEvent) -> None:
        pass

    def on_end(self, gd: GameData, board: Optional[chess.Board]) -> None:
        pass

    def finalize(self) -> str:
        return ""


# ============================================================
# Reservoir sampler + utilities used by Q10/Q17
# ============================================================


class Reservoir:
    """Numerically stable reservoir sampler for (features, label) rows."""

    def __init__(self, k: int, n_features: int, seed: int = 42):
        self.k = k
        self.X = np.empty((k, n_features), dtype=np.float32)
        self.y = np.empty(k, dtype=np.int8)
        self.n = 0
        self.rng = np.random.default_rng(seed)

    def add(self, x: list[float], label: int) -> None:
        i = self.n
        self.n += 1
        if i < self.k:
            self.X[i] = x
            self.y[i] = label
        else:
            j = int(self.rng.integers(0, self.n))
            if j < self.k:
                self.X[j] = x
                self.y[j] = label

    def fit(self) -> LogisticRegression:
        m = min(self.n, self.k)
        clf = LogisticRegression(max_iter=2000, n_jobs=-1)
        clf.fit(self.X[:m], self.y[:m])
        return clf


# ============================================================
# Per-game-with-moves questions
# ============================================================


class Q1MaterialDisadvantage(Question):
    """Q1: material disadvantage ≥ 3 in standard 2023.10.12–2024.02.19."""

    name, needs_board = "Q1", True

    def __init__(self, gi: GameIndex):
        super().__init__(gi)
        self._mask = (
            gi.is_std
            & (gi.utcdates >= b"2023.10.12") & (gi.utcdates <= b"2024.02.19")
            & (gi.results != gi.RES_DRAW)
        )
        self._count = 0

    def relevant(self, idx: int) -> bool:
        return idx >= 0 and bool(self._mask[idx])

    def on_end(self, gd: GameData, board: Optional[chess.Board]) -> None:
        if board is None:
            return
        w, b = count_material(board)
        r = int(self.gi.results[gd.idx])
        if (r == self.gi.RES_WHITE and w - b >= 3) or (r == self.gi.RES_BLACK and b - w >= 3):
            self._count += 1

    def finalize(self) -> str:
        return str(self._count)


class Q2LeftKnight(Question):
    """Q2: win-rate gap between players who captured with their left knight vs not."""

    name, needs_walk = "Q2", True

    def __init__(self, gi: GameIndex):
        super().__init__(gi)
        self._lk_w = self._lk_t = self._no_w = self._no_t = 0
        self._wlk = self._blk = -1
        self._wcap = self._bcap = False

    def relevant(self, idx: int) -> bool:
        return idx >= 0 and int(self.gi.results[idx]) != self.gi.RES_DRAW

    def begin_game(self, gd: GameData) -> None:
        self._wlk, self._blk = chess.B1, chess.G8
        self._wcap = self._bcap = False

    def on_move(self, gd: GameData, ev: MoveEvent) -> None:
        # Left-knight tracking. Only updates when a knight moves *from* the tracked square.
        if ev.mover == chess.WHITE:
            if self._wlk >= 0 and ev.from_sq == self._wlk:
                # SAN starting with 'N' means a knight moved.
                if ev.san.startswith("N"):
                    if ev.is_capture:
                        self._wcap = True
                    self._wlk = ev.to_sq
                else:
                    self._wlk = -1
            if self._blk >= 0 and ev.is_capture and ev.to_sq == self._blk:
                self._blk = -1
        else:
            if self._blk >= 0 and ev.from_sq == self._blk:
                if ev.san.startswith("N"):
                    if ev.is_capture:
                        self._bcap = True
                    self._blk = ev.to_sq
                else:
                    self._blk = -1
            if self._wlk >= 0 and ev.is_capture and ev.to_sq == self._wlk:
                self._wlk = -1

    def on_end(self, gd: GameData, board: Optional[chess.Board]) -> None:
        r = int(self.gi.results[gd.idx])
        for cap, won in ((self._wcap, r == self.gi.RES_WHITE), (self._bcap, r == self.gi.RES_BLACK)):
            if cap:
                self._lk_t += 1
                if won:
                    self._lk_w += 1
            else:
                self._no_t += 1
                if won:
                    self._no_w += 1

    def finalize(self) -> str:
        lk = self._lk_w / self._lk_t if self._lk_t else 0.0
        no = self._no_w / self._no_t if self._no_t else 0.0
        return (
            f"Bal lóval ütők nyerési aránya: {lk:.4f} ({self._lk_w}/{self._lk_t}), "
            f"nem ütők: {no:.4f} ({self._no_w}/{self._no_t}), "
            f"különbség: {lk - no:+.4f}"
        )


class Q3CastlingLost(Question):
    """Q3: 10-min games where white loses castling rights within first 6 plies."""

    name, needs_walk = "Q3", True

    def __init__(self, gi: GameIndex):
        super().__init__(gi)
        self._mask = gi.tcbases == 600
        self._count = 0
        self._board: Optional[chess.Board] = None
        self._had: bool = False

    def relevant(self, idx: int) -> bool:
        return idx >= 0 and bool(self._mask[idx])

    def begin_game(self, gd: GameData) -> None:
        self._board = chess.Board()
        self._had = True

    def on_move(self, gd: GameData, ev: MoveEvent) -> None:
        # We replay our own mini-board only for the first 6 plies so we can read castling rights.
        if ev.ply >= 6 or self._board is None:
            return
        had = self._board.has_castling_rights(chess.WHITE)
        try:
            self._board.push(ev.move)
        except Exception:
            self._board = None
            return
        if ev.mover == chess.WHITE and had and not self._board.has_castling_rights(chess.WHITE):
            self._count += 1
            self._board = None  # done with this game

    def finalize(self) -> str:
        return str(self._count)


class Q4RookDistance(Question):
    """Q4: aggregate file/rank distance moved by rooks (white minus black)."""

    name, needs_walk = "Q4", True

    def __init__(self, gi: GameIndex):
        super().__init__(gi)
        self._wd = self._bd = 0

    def on_move(self, gd: GameData, ev: MoveEvent) -> None:
        if ev.is_castle:
            d = 2 if ev.is_kingside else 3  # H→F is 2, A→D is 3
        elif ev.san[:1] == "R":
            d = abs(chess.square_file(ev.to_sq) - chess.square_file(ev.from_sq)) + abs(
                chess.square_rank(ev.to_sq) - chess.square_rank(ev.from_sq)
            )
        else:
            return
        if ev.mover == chess.WHITE:
            self._wd += d
        else:
            self._bd += d

    def finalize(self) -> str:
        return f"Fehér − Fekete bástya távolság: {self._wd - self._bd} mező (fehér: {self._wd}, fekete: {self._bd})"


class Q5ScissorsThreefold(Question):
    """Q5: threefold-repetition draw with at least one scissors-emoji player."""

    name, needs_board = "Q5", True

    def __init__(self, gi: GameIndex):
        super().__init__(gi)
        self._mask = gi.has_scissors & (gi.results == gi.RES_DRAW)
        self._count = 0

    def relevant(self, idx: int) -> bool:
        return idx >= 0 and bool(self._mask[idx])

    def on_end(self, gd: GameData, board: Optional[chess.Board]) -> None:
        if board is not None and board.is_repetition(3):
            self._count += 1

    def finalize(self) -> str:
        return str(self._count)


class Q6ThreefoldDateRange(Question):
    """Q6: threefold-repetition draws (standard, 2024.03.12–2024.11.19, Normal)."""

    name, needs_board = "Q6", True

    def __init__(self, gi: GameIndex):
        super().__init__(gi)
        self._mask = (
            gi.is_std
            & (gi.utcdates >= b"2024.03.12") & (gi.utcdates <= b"2024.11.19")
            & (gi.results == gi.RES_DRAW) & (gi.terms == gi.TERM_NORMAL)
        )
        self._count = 0

    def relevant(self, idx: int) -> bool:
        return idx >= 0 and bool(self._mask[idx])

    def on_end(self, gd: GameData, board: Optional[chess.Board]) -> None:
        if board is not None and board.is_repetition(3):
            self._count += 1

    def finalize(self) -> str:
        return str(self._count)


class Q7QueensAtMate(Question):
    """Q7: average # white queens on the board at checkmate, in tournament-winner games."""

    name, needs_board = "Q7", True

    def __init__(self, gi: GameIndex):
        super().__init__(gi)
        self._mask = (
            gi.is_tour_winner & (gi.terms == gi.TERM_NORMAL) & (gi.results != gi.RES_DRAW)
        )
        self._sum = 0
        self._n = 0

    def relevant(self, idx: int) -> bool:
        return idx >= 0 and bool(self._mask[idx])

    def on_end(self, gd: GameData, board: Optional[chess.Board]) -> None:
        if not gd.sans or "#" not in gd.sans[-1] or board is None:
            return
        self._sum += len(board.pieces(chess.QUEEN, chess.WHITE))
        self._n += 1

    def finalize(self) -> str:
        return f"{(self._sum / self._n) if self._n else 0.0:.4f} ({self._n} parti)"


class Q8DrawMarch20Promo(Question):
    """Q8: draws on March 20 where the last move is a pawn promotion to queen."""

    name = "Q8"

    def __init__(self, gi: GameIndex):
        super().__init__(gi)
        self._mask = (gi.results == gi.RES_DRAW) & np.array(
            [d.endswith(b".03.20") for d in gi.utcdates]
        )
        self._count = 0

    def relevant(self, idx: int) -> bool:
        return idx >= 0 and bool(self._mask[idx])

    def on_end(self, gd: GameData, board: Optional[chess.Board]) -> None:
        if gd.sans and "=Q" in gd.sans[-1]:
            self._count += 1

    def finalize(self) -> str:
        return str(self._count)


class Q10WinLogit(Question):
    """Q10: per-player logit (captures, color, avg seconds/move) → won?"""

    name, needs_walk = "Q10", True

    def __init__(self, gi: GameIndex):
        super().__init__(gi)
        self._sample = Reservoir(k=2_000_000, n_features=3)
        self._wcap = self._bcap = 0
        self._wt = self._bt = 0.0
        self._wn = self._bn = 0
        self._pw = self._pb = -1
        self._skip = False

    def relevant(self, idx: int) -> bool:
        return idx >= 0 and int(self.gi.results[idx]) != self.gi.RES_DRAW

    def begin_game(self, gd: GameData) -> None:
        self._wcap = self._bcap = 0
        self._wt = self._bt = 0.0
        self._wn = self._bn = 0
        self._pw = int(self.gi.wstarts[gd.idx])
        self._pb = int(self.gi.bstarts[gd.idx])
        self._skip = self._pw < 0 or self._pb < 0

    def on_move(self, gd: GameData, ev: MoveEvent) -> None:
        if self._skip:
            return
        if ev.is_capture:
            if ev.mover == chess.WHITE:
                self._wcap += 1
            else:
                self._bcap += 1
        clk = gd.clocks[ev.ply] if ev.ply < len(gd.clocks) else -1
        if clk < 0:
            return
        if ev.mover == chess.WHITE and self._pw >= 0:
            self._wt += self._pw - clk
            self._wn += 1
            self._pw = clk
        elif ev.mover == chess.BLACK and self._pb >= 0:
            self._bt += self._pb - clk
            self._bn += 1
            self._pb = clk

    def on_end(self, gd: GameData, board: Optional[chess.Board]) -> None:
        if self._skip:
            return
        r = int(self.gi.results[gd.idx])
        wavg = (self._wt / self._wn) if self._wn else 0.0
        bavg = (self._bt / self._bn) if self._bn else 0.0
        self._sample.add([float(self._wcap), 1.0, wavg], 1 if r == self.gi.RES_WHITE else 0)
        self._sample.add([float(self._bcap), 0.0, bavg], 1 if r == self.gi.RES_BLACK else 0)

    def finalize(self) -> str:
        if self._sample.n == 0:
            return "Nincs adat"
        clf = self._sample.fit()
        c = clf.coef_[0]
        return (
            f"Intercept: {clf.intercept_[0]:.6f}, captures: {c[0]:.6f}, "
            f"white: {c[1]:.6f}, avg_time: {c[2]:.6f}  "
            f"(n_samples={min(self._sample.n, self._sample.k):,} of {self._sample.n:,})"
        )


class Q11Resignations(Question):
    """Q11: most-frequent resigner, # who never resigned, # at the median."""

    name = "Q11"

    def __init__(self, gi: GameIndex):
        super().__init__(gi)
        self._mask = (gi.terms == gi.TERM_NORMAL) & (gi.results != gi.RES_DRAW)
        self._counts: dict[int, int] = defaultdict(int)

    def relevant(self, idx: int) -> bool:
        return idx >= 0 and bool(self._mask[idx])

    def on_end(self, gd: GameData, board: Optional[chess.Board]) -> None:
        if not gd.sans or "#" in gd.sans[-1]:
            return
        r = int(self.gi.results[gd.idx])
        loser = int(self.gi.bidxs[gd.idx]) if r == self.gi.RES_WHITE else int(self.gi.widxs[gd.idx])
        self._counts[loser] += 1

    def finalize(self) -> str:
        if not self._counts:
            return "Nincs"
        # Build full per-player count array (every player ever seen, default 0).
        n_players = len(self.gi.player_names)
        all_counts = np.zeros(n_players, dtype=np.int64)
        for pid, c in self._counts.items():
            all_counts[pid] = c
        most_pid = int(np.argmax(all_counts))
        most_n = int(all_counts[most_pid])
        never = int(np.sum(all_counts == 0))
        med = float(np.median(all_counts))
        at_med = int(np.sum(all_counts == med))
        return (
            f"Legtöbbet feladott: {self.gi.player_names[most_pid]} ({most_n}x) | "
            f"Soha nem adta fel: {never} | Mediánban ({med:.1f}): {at_med}"
        )


class Q13TimeUsage(Question):
    """Q13: do players who use *more* or *less* time win at higher rate?"""

    name, needs_walk = "Q13", True

    def __init__(self, gi: GameIndex):
        super().__init__(gi)
        self._more_w = self._more_t = self._less_w = self._less_t = 0
        self._wt = self._bt = 0.0
        self._pw = self._pb = -1
        self._skip = False

    def relevant(self, idx: int) -> bool:
        return idx >= 0 and int(self.gi.results[idx]) != self.gi.RES_DRAW

    def begin_game(self, gd: GameData) -> None:
        self._wt = self._bt = 0.0
        self._pw = int(self.gi.wstarts[gd.idx])
        self._pb = int(self.gi.bstarts[gd.idx])
        self._skip = self._pw < 0 or self._pb < 0

    def on_move(self, gd: GameData, ev: MoveEvent) -> None:
        if self._skip:
            return
        clk = gd.clocks[ev.ply] if ev.ply < len(gd.clocks) else -1
        if clk < 0:
            return
        if ev.mover == chess.WHITE and self._pw >= 0:
            self._wt += self._pw - clk
            self._pw = clk
        elif ev.mover == chess.BLACK and self._pb >= 0:
            self._bt += self._pb - clk
            self._pb = clk

    def on_end(self, gd: GameData, board: Optional[chess.Board]) -> None:
        if self._skip or self._wt == self._bt:
            return
        r = int(self.gi.results[gd.idx])
        white_used_more = self._wt > self._bt
        more_won = (r == self.gi.RES_WHITE) if white_used_more else (r == self.gi.RES_BLACK)
        self._more_t += 1
        self._less_t += 1
        if more_won:
            self._more_w += 1
        else:
            self._less_w += 1

    def finalize(self) -> str:
        mr = self._more_w / self._more_t if self._more_t else 0
        lr = self._less_w / self._less_t if self._less_t else 0
        who = "Több időt felhasználók" if mr > lr else "Kevesebb időt felhasználók"
        return f"{who} nyernek nagyobb arányban (több: {mr:.4f}, kevesebb: {lr:.4f})"


class Q14A2ToG8(Question):
    """Q14: dates where the white pawn originally on a2 reached g8 and promoted."""

    name, needs_walk = "Q14", True

    def __init__(self, gi: GameIndex):
        super().__init__(gi)
        self._dates: set[str] = set()
        self._tracked: int = -1
        self._reached: bool = False

    def begin_game(self, gd: GameData) -> None:
        self._tracked = chess.A2
        self._reached = False

    def on_move(self, gd: GameData, ev: MoveEvent) -> None:
        if self._tracked < 0 or self._reached:
            return
        # Captured (regular)?
        if ev.is_capture and ev.to_sq == self._tracked and ev.mover == chess.BLACK:
            self._tracked = -1
            return
        # Captured by en passant?
        if ev.is_en_passant and ev.ep_captured_sq == self._tracked:
            self._tracked = -1
            return
        # Our pawn moved?
        if ev.mover == chess.WHITE and ev.from_sq == self._tracked:
            self._tracked = ev.to_sq
            if ev.to_sq == chess.G8 and ev.promotion is not None:
                self._reached = True

    def on_end(self, gd: GameData, board: Optional[chess.Board]) -> None:
        if self._reached:
            self._dates.add(self.gi.utcdates[gd.idx].decode())

    def finalize(self) -> str:
        return ", ".join(sorted(self._dates)[:10]) if self._dates else "Nincs"


class Q15NonQueenPromos(Question):
    """Q15: how often promotions chose something other than a queen + top 3 alternatives."""

    name = "Q15"

    def __init__(self, gi: GameIndex):
        super().__init__(gi)
        self._counts: Counter = Counter()

    def on_end(self, gd: GameData, board: Optional[chess.Board]) -> None:
        for san in gd.sans:
            m = PROMO_RE.search(san)
            if m and m.group(1) != "Q":
                self._counts[m.group(1)] += 1

    def finalize(self) -> str:
        total = sum(self._counts.values())
        top3 = self._counts.most_common(3)
        return f"Nem vezérre: {total} | Top 3: " + ", ".join(f"{p}:{c}" for p, c in top3)


class Q17MoveLogit(Question):
    """Q17: per-move logit (capture? ~ time_elapsed_seconds + white_dummy)."""

    name, needs_walk = "Q17", True

    def __init__(self, gi: GameIndex):
        super().__init__(gi)
        self._sample = Reservoir(k=3_000_000, n_features=2)
        self._pw = self._pb = -1
        self._elapsed = 0.0
        self._skip = False

    def begin_game(self, gd: GameData) -> None:
        self._pw = int(self.gi.wstarts[gd.idx])
        self._pb = int(self.gi.bstarts[gd.idx])
        self._elapsed = 0.0
        self._skip = self._pw < 0 or self._pb < 0

    def on_move(self, gd: GameData, ev: MoveEvent) -> None:
        if self._skip:
            return
        clk = gd.clocks[ev.ply] if ev.ply < len(gd.clocks) else -1
        if clk >= 0:
            if ev.mover == chess.WHITE and self._pw >= 0:
                self._elapsed += self._pw - clk
                self._pw = clk
            elif ev.mover == chess.BLACK and self._pb >= 0:
                self._elapsed += self._pb - clk
                self._pb = clk
        self._sample.add(
            [self._elapsed, 1.0 if ev.mover == chess.WHITE else 0.0],
            1 if ev.is_capture else 0,
        )

    def finalize(self) -> str:
        if self._sample.n == 0:
            return "Nincs adat"
        clf = self._sample.fit()
        c = clf.coef_[0]
        return (
            f"Intercept: {clf.intercept_[0]:.6f}, time_elapsed: {c[0]:.6f}, "
            f"white: {c[1]:.6f}  "
            f"(n_samples={min(self._sample.n, self._sample.k):,} of {self._sample.n:,})"
        )


class Q19FiftyMove(Question):
    """Q19: 50-move-rule draws in standard 2026.03.15–2026.10.14."""

    name, needs_board = "Q19", True

    def __init__(self, gi: GameIndex):
        super().__init__(gi)
        self._mask = (
            gi.is_std
            & (gi.utcdates >= b"2026.03.15") & (gi.utcdates <= b"2026.10.14")
            & (gi.results == gi.RES_DRAW) & (gi.terms == gi.TERM_NORMAL)
        )
        self._count = 0

    def relevant(self, idx: int) -> bool:
        return idx >= 0 and bool(self._mask[idx])

    def on_end(self, gd: GameData, board: Optional[chess.Board]) -> None:
        if board is not None and board.is_fifty_moves():
            self._count += 1

    def finalize(self) -> str:
        return str(self._count)


class Q20QueensGambit(Question):
    """Q20: per-year Queen's-Gambit ratio (CET 04.21–05.18, standard)."""

    name = "Q20"

    def __init__(self, gi: GameIndex):
        super().__init__(gi)
        # Pre-compute CET (year, day-in-window) per game once.
        years_cet = gi.cet_year()
        # We need CET month/day too. Cheaper to compute on-demand for the small
        # subset that passes std + raw-date pre-filter.
        std = gi.is_std
        # UTC may be off by one day from CET; broaden pre-filter by ±1 day.
        ud = gi.utcdates
        # Months in window: roughly 04 or 05 (UTC may shift to 03/06 at edges).
        m_bytes = ud.view("u1").reshape(len(ud), 10)
        month = (m_bytes[:, 5].astype(np.int8) - 48) * 10 + (m_bytes[:, 6].astype(np.int8) - 48)
        self._pre_mask = std & ((month == 4) | (month == 5) | (month == 3) | (month == 6))
        self._years_cet = years_cet
        self._yr_total: Counter = Counter()
        self._yr_qg: Counter = Counter()

    def relevant(self, idx: int) -> bool:
        return idx >= 0 and bool(self._pre_mask[idx])

    def on_end(self, gd: GameData, board: Optional[chess.Board]) -> None:
        idx = gd.idx
        ud = self.gi.utcdates[idx].decode()
        ut = int(self.gi.utctimes[idx])
        cet = utc_to_cet(ud, ut)
        if cet is None:
            return
        md = (cet.month, cet.day)
        if not ((4, 21) <= md <= (5, 18)):
            return
        yr = cet.year
        if len(gd.sans) < 3:
            return
        self._yr_total[yr] += 1
        a, b, c = (s.rstrip("+#") for s in gd.sans[:3])
        if a == "d4" and b == "d5" and c == "c4":
            self._yr_qg[yr] += 1

    def finalize(self) -> str:
        if not self._yr_total:
            return "Nincs adat"
        return "\n".join(
            f"{yr}: {self._yr_qg[yr] / t:.4f} ({self._yr_qg[yr] / t * 100:.2f}%)"
            for yr, t in sorted(self._yr_total.items()) if t > 0
        )


class Q22Rectangles(Question):
    """Q22: who completed the most rectangles, and the largest rectangle area."""

    name, needs_walk = "Q22", True

    def __init__(self, gi: GameIndex):
        super().__init__(gi)
        self._counts: dict[int, int] = defaultdict(int)
        self._max_area = 0
        # Per-game state.
        self._sq_orig: dict[int, int] = {}
        self._hist: dict[int, list[int]] = {}
        self._w_rect = 0
        self._b_rect = 0

    def begin_game(self, gd: GameData) -> None:
        self._sq_orig = {sq: sq for sq in STARTING_PIECE_SQUARES}
        self._hist = {sq: [sq] for sq in STARTING_PIECE_SQUARES}
        self._w_rect = self._b_rect = 0

    def on_move(self, gd: GameData, ev: MoveEvent) -> None:
        # Update piece tracking.
        if ev.is_en_passant:
            self._sq_orig.pop(ev.ep_captured_sq, None)
        else:
            self._sq_orig.pop(ev.to_sq, None)
        orig = self._sq_orig.pop(ev.from_sq, None)
        if orig is None:
            # Castling: still need to move the rook below.
            if ev.is_castle:
                self._handle_castle_rook(ev)
            return
        self._sq_orig[ev.to_sq] = orig
        self._hist.setdefault(orig, [ev.from_sq]).append(ev.to_sq)

        if ev.is_castle:
            self._handle_castle_rook(ev)

        h = self._hist[orig]
        if len(h) >= 4:
            a, b, c, d = h[-4], h[-3], h[-2], h[-1]
            if self._is_rect(a, b, c, d):
                area = self._rect_area(a, b, c, d)
                if ev.mover == chess.WHITE:
                    self._w_rect += 1
                else:
                    self._b_rect += 1
                if area > self._max_area:
                    self._max_area = area

    def _handle_castle_rook(self, ev: MoveEvent) -> None:
        if ev.mover == chess.WHITE:
            rf, rt = (chess.H1, chess.F1) if ev.is_kingside else (chess.A1, chess.D1)
        else:
            rf, rt = (chess.H8, chess.F8) if ev.is_kingside else (chess.A8, chess.D8)
        ro = self._sq_orig.pop(rf, None)
        if ro is not None:
            self._sq_orig[rt] = ro
            self._hist.setdefault(ro, [rf]).append(rt)

    @staticmethod
    def _is_rect(a: int, b: int, c: int, d: int) -> bool:
        sqs = (a, b, c, d)
        files = [chess.square_file(s) for s in sqs]
        ranks = [chess.square_rank(s) for s in sqs]
        if len(set(files)) != 2 or len(set(ranks)) != 2:
            return False
        if (max(files) - min(files)) * (max(ranks) - min(ranks)) == 0:
            return False
        # Adjacent points must share a file or a rank.
        for i in range(4):
            j = (i + 1) % 4
            if files[i] != files[j] and ranks[i] != ranks[j]:
                return False
        return True

    @staticmethod
    def _rect_area(a: int, b: int, c: int, d: int) -> int:
        sqs = (a, b, c, d)
        files = [chess.square_file(s) for s in sqs]
        ranks = [chess.square_rank(s) for s in sqs]
        return (max(files) - min(files)) * (max(ranks) - min(ranks))

    def on_end(self, gd: GameData, board: Optional[chess.Board]) -> None:
        wpid = int(self.gi.widxs[gd.idx])
        bpid = int(self.gi.bidxs[gd.idx])
        if self._w_rect:
            self._counts[wpid] += self._w_rect
        if self._b_rect:
            self._counts[bpid] += self._b_rect

    def finalize(self) -> str:
        if not self._counts:
            return "Nincs"
        top_pid = max(self._counts, key=self._counts.get)
        top_n = self._counts[top_pid]
        return (
            f"Játékos: {self.gi.player_names[top_pid]} ({top_n} téglalap) | "
            f"Legnagyobb terület: {self._max_area}"
        )


class Q23CastleMate(Question):
    """Q23: who delivered checkmate by castling most often?"""

    name = "Q23"

    def __init__(self, gi: GameIndex):
        super().__init__(gi)
        self._counts: Counter = Counter()

    def on_end(self, gd: GameData, board: Optional[chess.Board]) -> None:
        if not gd.sans:
            return
        last = gd.sans[-1]
        if last not in ("O-O#", "O-O-O#"):
            return
        # White moves on odd plies (1-indexed): index parity determines mover.
        color = "white" if (len(gd.sans) - 1) % 2 == 0 else "black"
        self._counts[self.gi.player_at(gd.idx, color)] += 1

    def finalize(self) -> str:
        if not self._counts:
            return ""
        top = max(self._counts.values())
        return ", ".join(sorted(p for p, c in self._counts.items() if c == top)[:10])


class Q24EnPassantIndian(Question):
    """Q24: en-passant captures by white in 3-min ECO-E* games."""

    name, needs_walk = "Q24", True

    def __init__(self, gi: GameIndex):
        super().__init__(gi)
        eco_starts_e = np.array([e[:1] == b"E" for e in gi.ecos])
        self._mask = (gi.tcbases == 180) & eco_starts_e
        self._count = 0

    def relevant(self, idx: int) -> bool:
        return idx >= 0 and bool(self._mask[idx])

    def on_move(self, gd: GameData, ev: MoveEvent) -> None:
        if ev.mover == chess.WHITE and ev.is_en_passant:
            self._count += 1

    def finalize(self) -> str:
        return str(self._count)


# ============================================================
# Pure-GameIndex questions (no moves needed)
# ============================================================


def q9_berserk_timeouts(gi: GameIndex) -> str:
    """Q9: most timeouts where the loser entered with ≤ tc/2 + 2 sec on the clock."""
    mask = (gi.terms == gi.TERM_TIMEFORFEIT) & (gi.tcbases > 0)
    half = gi.tcbases.astype(np.int32) // 2 + 2  # tc/2 + 2

    # White lost (result == BLACK) and their start clock ≤ half
    w_loser = mask & (gi.results == gi.RES_BLACK) & (gi.wstarts > 0) & (gi.wstarts <= half)
    b_loser = mask & (gi.results == gi.RES_WHITE) & (gi.bstarts > 0) & (gi.bstarts <= half)

    counts: dict[int, int] = defaultdict(int)
    for pid in gi.widxs[w_loser]:
        counts[int(pid)] += 1
    for pid in gi.bidxs[b_loser]:
        counts[int(pid)] += 1

    if not counts:
        return "Nincs"
    top = max(counts.values())
    winners = sorted(str(gi.player_names[pid]) for pid, c in counts.items() if c == top)[:10]
    return f"Legtöbb berserk timeout vereség ({top}x): {', '.join(winners)}"


def q21_year_spanning(gi: GameIndex) -> str:
    """Q21: standard games that *could* span midnight CET on Dec 31, by year."""
    is_dec31_utc = np.array([d.endswith(b".12.31") for d in gi.utcdates])
    mask = gi.is_std & is_dec31_utc

    result: dict[int, int] = defaultdict(int)
    idxs = np.flatnonzero(mask)
    for i in idxs:
        ud = gi.utcdates[i].decode()
        ut = int(gi.utctimes[i])
        cet = utc_to_cet(ud, ut)
        if cet is None or cet.month != 12 or cet.day != 31:
            continue
        midnight = CET.localize(datetime(cet.year + 1, 1, 1))
        secs_to_midnight = (midnight - cet).total_seconds()
        total_clock = int(gi.wstarts[i]) + int(gi.bstarts[i])
        if total_clock > 0 and total_clock >= secs_to_midnight:
            result[cet.year] += 1
    if not result:
        return "0"
    return "\n".join(f"{yr}: {c}" for yr, c in sorted(result.items()))


# ---- Q16 / Q18 (per-player streaks) ----


def _streak_arrays(gi: GameIndex):
    """One concatenated row per (player, std-game), sorted by (pid, date, time).

    Returns (pids, is_draw, is_win, dates, times, elos) — six numpy arrays of
    length 2 * #std_games.
    """
    std = gi.is_std
    sel = np.flatnonzero(std)
    res = gi.results[sel]
    pids = np.concatenate([gi.widxs[sel], gi.bidxs[sel]])
    is_draw = np.concatenate([res == gi.RES_DRAW, res == gi.RES_DRAW])
    is_win = np.concatenate([res == gi.RES_WHITE, res == gi.RES_BLACK])
    dates = np.concatenate([gi.utcdates[sel], gi.utcdates[sel]])
    times = np.concatenate([gi.utctimes[sel], gi.utctimes[sel]])
    elos = np.concatenate([gi.welos[sel], gi.belos[sel]])
    order = np.lexsort((times, dates, pids))
    return pids[order], is_draw[order], is_win[order], dates[order], times[order], elos[order]


def _scan_streaks(pids: np.ndarray, cond: np.ndarray, dates: np.ndarray, elos: np.ndarray):
    """Find longest run of True ``cond`` per player. Returns list of
    (pid, start_date_b, end_date_b, length, last_elo) for *every* player's
    longest run (so the caller can apply tiebreaks).
    """
    boundaries = np.concatenate(([0], np.flatnonzero(np.diff(pids)) + 1, [len(pids)]))
    # Going through Python lists is materially faster than poking numpy element-wise.
    pids_l = pids.tolist()
    cond_l = cond.tolist()
    elos_l = elos.tolist()
    out: list[tuple[int, bytes, bytes, int, int]] = []
    best_len = 0
    for k in range(len(boundaries) - 1):
        s, e = int(boundaries[k]), int(boundaries[k + 1])
        if e <= s:
            continue
        pid = pids_l[s]
        run = 0
        run_start = s
        run_elo = 0
        local_best = (s, s - 1, 0, 0)
        for j in range(s, e):
            if cond_l[j]:
                if run == 0:
                    run_start = j
                run += 1
                ej = elos_l[j]
                if ej > 0:
                    run_elo = ej
                if run > local_best[2]:
                    local_best = (run_start, j, run, run_elo)
            else:
                run = 0
                run_elo = 0
        if local_best[2] > 0 and local_best[2] >= best_len:
            if local_best[2] > best_len:
                out.clear()
                best_len = local_best[2]
            out.append(
                (pid, dates[local_best[0]], dates[local_best[1]], local_best[2], local_best[3])
            )
    return out, best_len


def q16_draw_streak(gi: GameIndex) -> str:
    pids, is_draw, _, dates, _, elos = _streak_arrays(gi)
    tied, best_n = _scan_streaks(pids, is_draw, dates, elos)
    del pids, is_draw, dates, elos
    if not tied:
        return "Nincs"
    # Tiebreak: highest end-of-streak elo.
    pid, s, e, n, _ = max(tied, key=lambda t: t[4])
    return f"{gi.player_names[pid]} | {s.decode()} – {e.decode()} | {n} parti"


def q18_winless_streak(gi: GameIndex) -> str:
    pids, _, is_win, dates, _, elos = _streak_arrays(gi)
    is_winless = ~is_win
    tied, best_n = _scan_streaks(pids, is_winless, dates, elos)
    del pids, is_win, is_winless, dates, elos
    if not tied:
        return "Nincs"
    # Tiebreak: name following "Lili" in Hungarian alphabet (smallest such name).
    lk = hu_key("Lili")
    named = [(str(gi.player_names[pid]), s, e, n) for pid, s, e, n, _ in tied]
    after = [t for t in named if hu_key(t[0]) > lk]
    pool = after if after else named
    name, s, e, n = min(pool, key=lambda t: hu_key(t[0]))
    return f"{name} | {s.decode()} – {e.decode()} | {n} parti"


# ---- Q12 (longest cycle in CET-year win graph) ----


def _longest_cycle(graph: dict[int, set[int]], time_budget_s: float, max_depth: int = 12) -> list[int]:
    """Iterative depth-limited DFS with global time budget.

    Returns the longest simple directed cycle found. Not guaranteed optimal.
    """
    best: list[int] = []
    deadline = time.time() + time_budget_s

    nodes = list(graph)
    # Random-ish order: shuffling avoids pathological repeated exploration of
    # the same dense neighbourhood.
    np.random.default_rng(0).shuffle(nodes)

    for start in nodes:
        if time.time() > deadline or len(best) >= max_depth:
            break
        stack = [(start, iter(graph[start]))]
        path = [start]
        in_path = {start}
        while stack:
            _, it = stack[-1]
            advanced = False
            for nb in it:
                if nb == start and len(path) >= 3:
                    if len(path) > len(best):
                        best = path[:]
                        if len(best) >= max_depth:
                            break
                elif nb not in in_path and len(path) < max_depth:
                    path.append(nb)
                    in_path.add(nb)
                    stack.append((nb, iter(graph.get(nb, ()))))
                    advanced = True
                    break
            if len(best) >= max_depth:
                break
            if not advanced:
                stack.pop()
                if path:
                    in_path.discard(path.pop())
    return best


def q12_cyclic_win(gi: GameIndex, time_budget_per_year_s: float = 60.0) -> str:
    """Find the largest cyclic win pattern within a CET calendar year (standard, decisive)."""
    print("Q12: building per-year win graphs...")
    mask = gi.is_std & (gi.results != gi.RES_DRAW)
    sel = np.flatnonzero(mask)
    # .tolist() once — much faster than per-element numpy indexing in tight Python loops.
    cet_years_l = gi.cet_year()[sel].tolist()
    res_l = gi.results[sel].tolist()
    wid_l = gi.widxs[sel].tolist()
    bid_l = gi.bidxs[sel].tolist()

    year_graph: dict[int, dict[int, set[int]]] = defaultdict(lambda: defaultdict(set))
    res_white = gi.RES_WHITE
    for yr, r, w, b in tqdm(
        zip(cet_years_l, res_l, wid_l, bid_l),
        total=len(res_l), desc="Q12 graph", unit=" games", smoothing=0.01,
    ):
        if r == res_white:
            year_graph[yr][w].add(b)
        else:
            year_graph[yr][b].add(w)

    print(f"Q12: years to scan: {sorted(year_graph)}; running depth-limited DFS...")
    best: list[int] = []
    best_yr: Optional[int] = None
    for yr in sorted(year_graph):
        cyc = _longest_cycle(year_graph[yr], time_budget_per_year_s)
        if len(cyc) > len(best):
            best = cyc
            best_yr = yr
        print(f"  year {yr}: best cycle length {len(cyc)} (overall best {len(best)})")

    if not best or best_yr is None:
        return "Nincs"

    cycle_set = set(best)
    n = len(best)

    # Find the earliest game (CET) that is one of the cycle edges.
    print("Q12: scanning for first cycle edge...")
    first_ts: Optional[bytes] = None
    first_winner: int = best[0]
    cycle_next = {pid: best[(i + 1) % n] for i, pid in enumerate(best)}
    sel_l = sel.tolist()
    for j in range(len(sel_l)):
        if cet_years_l[j] != best_yr:
            continue
        if res_l[j] == res_white:
            winner, loser = wid_l[j], bid_l[j]
        else:
            winner, loser = bid_l[j], wid_l[j]
        if cycle_next.get(winner) != loser:
            continue
        idx = sel_l[j]
        ts = bytes(gi.utcdates[idx]) + bytes(f":{int(gi.utctimes[idx]):06d}", "ascii")
        if first_ts is None or ts < first_ts:
            first_ts = ts
            first_winner = winner

    si = best.index(first_winner)
    rotated = best[si:] + best[:si]
    names = [str(gi.player_names[pid]) for pid in rotated]
    return f"Év: {best_yr} | " + " → ".join(names) + f" → {names[0]}"


# ============================================================
# Single-pass moves driver
# ============================================================


def run_moves_pass(gi: GameIndex, questions: list[Question], total: int) -> None:
    """One streamed pass over moves.csv.gz; dispatch each game to relevant questions."""
    walkers = [q for q in questions if q.needs_walk]
    boarders = [q for q in questions if q.needs_board and not q.needs_walk]

    pbar = tqdm(total=total, desc="moves pass", unit=" games", smoothing=0.01)
    for gid_b, sans, clocks in stream_games():
        pbar.update(1)
        idx = gi.idx(gid_b)
        if idx < 0:
            continue
        relevant_walkers = [q for q in walkers if q.relevant(idx)]
        relevant_boarders = [q for q in boarders if q.relevant(idx)]
        relevant_others = [
            q for q in questions
            if not q.needs_walk and not q.needs_board and q.relevant(idx)
        ]

        if not (relevant_walkers or relevant_boarders or relevant_others):
            continue

        gd = GameData(gid=gid_b, idx=idx, sans=sans, clocks=clocks)

        for q in relevant_walkers:
            q.begin_game(gd)

        board: Optional[chess.Board] = None
        if relevant_walkers:
            board = walk_game(sans, relevant_walkers, gd)
        elif relevant_boarders:
            board = simulate_board(sans)

        for q in relevant_walkers + relevant_boarders + relevant_others:
            q.on_end(gd, board)

    pbar.close()


# ============================================================
# Output + main
# ============================================================


def write_answers(answers: dict[int, str]) -> None:
    lines = ["# Chess Data Analysis — Answers\n"]
    for q in sorted(answers):
        lines.append(f"## {q}. kérdés\n\n{answers[q]}\n")
    with open(OUTPUT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\nWritten to {OUTPUT}")


def main() -> None:
    sys.setrecursionlimit(100_000)
    gi = GameIndex()
    gi.build()

    answers: dict[int, str] = {}

    # ---- Pure-GameIndex questions ----
    print("\n=== Pure-metadata questions ===")
    answers[9] = q9_berserk_timeouts(gi)
    print("  Q9 done")
    answers[21] = q21_year_spanning(gi)
    print("  Q21 done")
    answers[16] = q16_draw_streak(gi)
    print("  Q16 done")
    answers[18] = q18_winless_streak(gi)
    print("  Q18 done")
    answers[12] = q12_cyclic_win(gi)
    print("  Q12 done")

    # ---- Single moves pass ----
    print("\n=== Single moves pass ===")
    questions: list[Question] = [
        Q1MaterialDisadvantage(gi),
        Q2LeftKnight(gi),
        Q3CastlingLost(gi),
        Q4RookDistance(gi),
        Q5ScissorsThreefold(gi),
        Q6ThreefoldDateRange(gi),
        Q7QueensAtMate(gi),
        Q8DrawMarch20Promo(gi),
        Q10WinLogit(gi),
        Q11Resignations(gi),
        Q13TimeUsage(gi),
        Q14A2ToG8(gi),
        Q15NonQueenPromos(gi),
        Q17MoveLogit(gi),
        Q19FiftyMove(gi),
        Q20QueensGambit(gi),
        Q22Rectangles(gi),
        Q23CastleMate(gi),
        Q24EnPassantIndian(gi),
    ]
    run_moves_pass(gi, questions, total=len(gi.gids))

    # Map by class → question number.
    q_map = {
        Q1MaterialDisadvantage: 1, Q2LeftKnight: 2, Q3CastlingLost: 3,
        Q4RookDistance: 4, Q5ScissorsThreefold: 5, Q6ThreefoldDateRange: 6,
        Q7QueensAtMate: 7, Q8DrawMarch20Promo: 8, Q10WinLogit: 10,
        Q11Resignations: 11, Q13TimeUsage: 13, Q14A2ToG8: 14,
        Q15NonQueenPromos: 15, Q17MoveLogit: 17, Q19FiftyMove: 19,
        Q20QueensGambit: 20, Q22Rectangles: 22, Q23CastleMate: 23,
        Q24EnPassantIndian: 24,
    }
    for q in questions:
        answers[q_map[type(q)]] = q.finalize()

    write_answers(answers)


if __name__ == "__main__":
    main()
