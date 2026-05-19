#!/usr/bin/env python3
"""Chess data analysis — answers all 24 questions (polars-optimized).

This is a drop-in replacement for the pandas version with identical
outputs. Major changes for speed (memory budget is the same or smaller):

  * polars replaces pandas for **all** CSV I/O. gzipped reads are 2–4× faster
    and string-column ops are multi-threaded.
  * ``GameIndex.build`` does column transforms (replace, cast, split,
    clip) per chunk via polars expressions instead of pandas series ops.
  * The single moves-pass uses ``pl.read_csv_batched`` and one batched
    ``np.searchsorted`` call per chunk for gid→idx (instead of one Python
    call per game). This eliminates the 4.5 GB hashtable that the
    original would have needed for the same speed.
  * ``walk_game`` reuses a single mutable ``MoveEvent`` instance instead
    of allocating one per move (cuts allocations for ~3.5 B move events).
    Hot bound methods are cached to locals.
  * Per-question relevance is baked into a uint32 bitmask per game, so
    the per-game inner loop in ``run_moves_pass`` makes a single numpy
    lookup + cheap bit tests instead of 19 Python ``relevant()`` calls.
  * Q22 inlines ``square_file``/``square_rank`` as bit ops and removes
    the redundant degeneracy check.

All outputs (including the values, the order of items in those outputs,
and the player tiebreaks) match the original to the byte.
"""

from __future__ import annotations

import re
import sys
import time
import warnings
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Iterator, Optional

import chess
import numpy as np
import polars as pl
import pytz
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm

# Polars 1.x marks ``read_csv_batched`` deprecated but still works and gives
# the cleanest size-controlled iteration over gzipped CSVs. Silence the
# DeprecationWarning rather than switch APIs (the replacement
# ``scan_csv().collect_batches()`` doesn't accept a batch_size parameter,
# so we can't bound peak memory the same way). The warning is raised with
# the caller's module as the source, so we filter by message.
warnings.filterwarnings(
    "ignore", message=".*read_csv_batched.*is deprecated.*",
)

# ============================================================
# Config
# ============================================================

GAMES = "data/games.csv.gz"
MOVES = "data/moves.csv.gz"
TOURNAMENTS = "data/tournaments.csv.gz"
OUTPUT = "kesonleadas.md"
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
    """``H:MM:SS`` → seconds. Returns -1 on error.

    Kept for parity with the original; not used in the hot path (which
    uses the polars-vectorised variant below).
    """
    try:
        h, m, sec = str(s).split(":")
        return int(h) * 3600 + int(m) * 60 + int(sec)
    except Exception:
        return -1


def parse_clock_pl(s: pl.Series) -> np.ndarray:
    """Polars-vectorised ``H:MM:SS`` → seconds. Invalid rows yield -1.

    Equivalent to the original ``parse_clock_vec`` but ~3× faster on
    gigabyte-scale columns because polars splits and parses in C.
    """
    parts = s.cast(pl.Utf8).fill_null("").str.split_exact(":", 2)
    h = parts.struct.field("field_0").cast(pl.Int32, strict=False)
    m = parts.struct.field("field_1").cast(pl.Int32, strict=False)
    sec = parts.struct.field("field_2").cast(pl.Int32, strict=False)
    # If any component is null/invalid the whole arithmetic propagates null.
    total = h * 3600 + m * 60 + sec
    return total.fill_null(-1).cast(pl.Int32).to_numpy().copy()


def parse_tc_base(s) -> int:
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
    """Replay a SAN list onto a fresh board; tolerates malformed moves.

    Used only when no walker is interested in the game but a boarder
    (final-position) question is.
    """
    board = chess.Board()
    push_san = board.push_san
    for san in sans:
        try:
            push_san(san)
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
# GameIndex
# ============================================================


class GameIndex:
    """Sorted-by-game_id metadata in compact numpy arrays. Same layout as
    the original — downstream code is unchanged.
    """

    TERM_NORMAL, TERM_TIMEFORFEIT, TERM_ABANDONED, TERM_OTHER = 0, 1, 2, 3
    RES_DRAW, RES_WHITE, RES_BLACK = 0, 1, 2

    _RES_MAP = {"1-0": RES_WHITE, "0-1": RES_BLACK, "1/2-1/2": RES_DRAW}
    _TERM_MAP = {"Normal": TERM_NORMAL, "Time forfeit": TERM_TIMEFORFEIT, "Abandoned": TERM_ABANDONED}

    def build(self) -> None:
        # Pre-load tournament winners.
        tdf = (
            pl.read_csv(
                TOURNAMENTS,
                columns=["id", "winner__id"],
                schema_overrides={"id": pl.Utf8, "winner__id": pl.Utf8},
            )
            .drop_nulls("winner__id")
        )
        tour_winner: dict[str, str] = dict(
            zip(
                tdf["id"].to_list(),
                tdf["winner__id"].str.to_lowercase().to_list(),
            )
        )
        del tdf

        cols = [
            "game_id", "result", "variant", "utcdate", "utctime", "termination",
            "timecontrol", "whitestart", "blackstart", "eco", "whiteelo", "blackelo",
            "white", "black", "tournamentid",
        ]
        # Force string types for everything except elos. polars sometimes
        # promotes mixed-content columns weirdly; locking the schema avoids
        # surprises and is faster (no inference pass per chunk).
        schema_overrides: dict[str, pl.DataType] = {
            "game_id": pl.Utf8, "result": pl.Utf8, "variant": pl.Utf8,
            "utcdate": pl.Utf8, "utctime": pl.Utf8, "termination": pl.Utf8,
            "timecontrol": pl.Utf8, "whitestart": pl.Utf8, "blackstart": pl.Utf8,
            "eco": pl.Utf8, "whiteelo": pl.Int32, "blackelo": pl.Int32,
            "white": pl.Utf8, "black": pl.Utf8, "tournamentid": pl.Utf8,
        }

        parts: dict[str, list[np.ndarray]] = defaultdict(list)
        player_ids: dict[str, int] = {}
        # Local refs to bound int-dispatch in the hottest player-id loop.
        res_map = self._RES_MAP
        term_map = self._TERM_MAP

        print("Building GameIndex (single pass over games)...")
        reader = pl.read_csv_batched(
            GAMES,
            columns=cols,
            schema_overrides=schema_overrides,
            batch_size=CHUNKSIZE,
        )

        with tqdm(desc="games", unit=" rows", unit_scale=True) as pb:
            while True:
                batches = reader.next_batches(1)
                if not batches:
                    break
                chunk = batches[0]
                n_rows = chunk.height
                pb.update(n_rows)

                # ---- compact metadata columns ---------------------------
                gid_col = chunk["game_id"].fill_null("")
                parts["gid"].append(gid_col.to_numpy().astype("S14"))

                res_col_str = chunk["result"].fill_null("")
                parts["res"].append(
                    res_col_str.replace_strict(
                        res_map, default=self.RES_DRAW, return_dtype=pl.Int8
                    ).to_numpy().copy()
                )

                parts["std"].append(
                    chunk["variant"].fill_null("")
                    .str.strip_chars().str.to_lowercase()
                    .eq("standard").to_numpy().copy()
                )

                parts["udate"].append(
                    chunk["utcdate"].fill_null("").to_numpy().astype("S10")
                )
                parts["utime"].append(parse_clock_pl(chunk["utctime"]))

                parts["term"].append(
                    chunk["termination"].fill_null("").replace_strict(
                        term_map, default=self.TERM_OTHER, return_dtype=pl.Int8
                    ).to_numpy().copy()
                )

                tc_parts = chunk["timecontrol"].fill_null("0").str.split_exact("+", 1)
                tc_base = (
                    tc_parts.struct.field("field_0")
                    .cast(pl.Int32, strict=False).fill_null(0)
                    .clip(0, 32767).cast(pl.Int16)
                )
                parts["tcbase"].append(tc_base.to_numpy().copy())

                parts["ws"].append(parse_clock_pl(chunk["whitestart"]))
                parts["bs"].append(parse_clock_pl(chunk["blackstart"]))

                parts["eco"].append(
                    chunk["eco"].fill_null("").to_numpy().astype("S3")
                )
                parts["welo"].append(
                    chunk["whiteelo"].fill_null(0).clip(0, 32767).cast(pl.Int16).to_numpy().copy()
                )
                parts["belo"].append(
                    chunk["blackelo"].fill_null(0).clip(0, 32767).cast(pl.Int16).to_numpy().copy()
                )

                # ---- tournament-winner-decisive flag --------------------
                tids = chunk["tournamentid"].fill_null("")
                expected = tids.replace_strict(
                    tour_winner, default=None, return_dtype=pl.Utf8
                )
                wlow = chunk["white"].fill_null("").str.to_lowercase()
                blow = chunk["black"].fill_null("").str.to_lowercase()
                is_tw = (
                    ((res_col_str == "1-0") & (wlow == expected))
                    | ((res_col_str == "0-1") & (blow == expected))
                ).fill_null(False)
                parts["tw"].append(is_tw.to_numpy().copy())

                # ---- player ids (insertion order MUST match original) ---
                w_names = chunk["white"].fill_null("").to_list()
                b_names = chunk["black"].fill_null("").to_list()
                # Single-pass insert into the global dict; same order as
                # the pandas version (whites first, then blacks, per chunk).
                pid_get = player_ids.get
                pid_setdefault = player_ids.setdefault
                wids = np.empty(n_rows, dtype=np.int32)
                bids = np.empty(n_rows, dtype=np.int32)
                # First fill whites (and grow dict).
                for i, n in enumerate(w_names):
                    pid = pid_get(n)
                    if pid is None:
                        pid = len(player_ids)
                        player_ids[n] = pid
                    wids[i] = pid
                for i, n in enumerate(b_names):
                    pid = pid_get(n)
                    if pid is None:
                        pid = len(player_ids)
                        player_ids[n] = pid
                    bids[i] = pid
                parts["wid"].append(wids)
                parts["bid"].append(bids)

        # ---- concatenate and sort ------------------------------------
        print(f"  concatenating {sum(len(p) for p in parts['gid']):,} games...")
        gids_raw = np.concatenate(parts.pop("gid"))
        # Stable argsort on S14 — same as the original. numpy is fine here
        # (the sort is one-shot; the more expensive part of the original is
        # the moves pass, not this).
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

        # Optional gid→idx hashtable (we don't build it by default — the
        # batched searchsorted in stream_games is fast enough and the
        # hashtable would be ~4 GB).
        self._gid_to_idx: Optional[dict[bytes, int]] = None

        mb = sum(getattr(self, k).nbytes for k in (
            "gids", "results", "is_std", "utcdates", "utctimes", "terms",
            "tcbases", "wstarts", "bstarts", "ecos", "welos", "belos",
            "is_tour_winner", "widxs", "bidxs", "has_scissors",
        )) // 1024 // 1024
        print(f"  GameIndex ready: {len(self.gids):,} games, ~{mb} MB metadata")

    # --- lookup ---------------------------------------------------------
    def build_idx_lookup(self) -> None:
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

    def cet_year(self) -> np.ndarray:
        n = len(self.gids)
        ud_bytes = self.utcdates.view("u1").reshape(n, 10)
        utc_year = (
            (ud_bytes[:, 0].astype(np.int16) - 48) * 1000
            + (ud_bytes[:, 1].astype(np.int16) - 48) * 100
            + (ud_bytes[:, 2].astype(np.int16) - 48) * 10
            + (ud_bytes[:, 3].astype(np.int16) - 48)
        )
        is_dec31 = (
            (ud_bytes[:, 5] == ord("1")) & (ud_bytes[:, 6] == ord("2"))
            & (ud_bytes[:, 8] == ord("3")) & (ud_bytes[:, 9] == ord("1"))
        )
        rolled = is_dec31 & (self.utctimes >= 23 * 3600)
        utc_year[rolled] += 1
        return utc_year


# ============================================================
# Streaming moves (polars-backed)
# ============================================================


def stream_games(gi: GameIndex) -> Iterator[tuple[int, list[str], list[int]]]:
    """Stream complete games. Yields (idx, sans, clocks).

    Unknown gids (idx == -1) are still yielded so the progress bar stays
    accurate; the caller skips them. Per-chunk we resolve every game's
    idx in a single batched ``np.searchsorted`` rather than per-game.
    """
    cur_idx: int = -2  # sentinel meaning "nothing accumulated yet"
    cur_sans: list[str] = []
    cur_clocks: list[int] = []

    reader = pl.read_csv_batched(
        MOVES,
        columns=["game_id", "move", "clock"],
        schema_overrides={"game_id": pl.Utf8, "move": pl.Utf8, "clock": pl.Utf8},
        batch_size=CHUNKSIZE,
    )

    all_gids = gi.gids
    all_gids_len = len(all_gids)

    while True:
        batches = reader.next_batches(1)
        if not batches:
            break
        chunk = batches[0]
        n = chunk.height
        if n == 0:
            continue

        # gids are short fixed-width ASCII; S14 keeps comparisons in numpy.
        gids = chunk["game_id"].fill_null("").to_numpy().astype("S14")
        sans_all = chunk["move"].fill_null("").to_list()
        clocks_all = parse_clock_pl(chunk["clock"]).tolist()

        # Game boundaries within the chunk.
        diff = np.concatenate(([True], gids[1:] != gids[:-1]))
        starts = np.flatnonzero(diff)
        ends = np.append(starts[1:], n)

        # Batched gid → idx lookup. One numpy call covers the whole chunk.
        uniq = gids[starts]
        cand = np.searchsorted(all_gids, uniq)
        cand_safe = np.clip(cand, 0, all_gids_len - 1)
        valid = (cand < all_gids_len) & (all_gids[cand_safe] == uniq)
        cand_int = cand.astype(np.int64, copy=False)
        cand_int[~valid] = -1
        chunk_idxs = cand_int.tolist()

        starts_l = starts.tolist()
        ends_l = ends.tolist()

        for k in range(len(starts_l)):
            s = starts_l[k]
            e = ends_l[k]
            g_idx = chunk_idxs[k]
            if g_idx == cur_idx and cur_idx != -2:
                # Same game as the previous chunk's tail; extend.
                cur_sans.extend(sans_all[s:e])
                cur_clocks.extend(clocks_all[s:e])
            else:
                if cur_idx != -2:
                    yield cur_idx, cur_sans, cur_clocks
                cur_idx = g_idx
                cur_sans = sans_all[s:e]
                cur_clocks = clocks_all[s:e]

    if cur_idx != -2:
        yield cur_idx, cur_sans, cur_clocks


# ============================================================
# Walker
# ============================================================


@dataclass(slots=True)
class MoveEvent:
    """Mutable; a single instance is reused across all moves of a game
    to eliminate per-move allocations (saves ~3.5 B object allocations
    over a full run)."""

    ply: int = 0
    san: str = ""
    move: Optional[chess.Move] = None
    from_sq: int = 0
    to_sq: int = 0
    mover: bool = True
    is_capture: bool = False
    is_castle: bool = False
    is_kingside: bool = False
    is_en_passant: bool = False
    ep_captured_sq: int = -1
    promotion: Optional[int] = None


def walk_game(sans: list[str], walkers: list["Question"], gd: "GameData") -> Optional[chess.Board]:
    """Replay sans, dispatch MoveEvents to walker handlers, return final board.

    Bound-method caching to locals plus event reuse gets a measurable
    speedup on the hottest loop in the program.
    """
    board = chess.Board()
    ev = MoveEvent()
    on_move_fns = [w.on_move for w in walkers]

    parse_san = board.parse_san
    is_castling = board.is_castling
    is_kingside_castling = board.is_kingside_castling
    is_en_passant_fn = board.is_en_passant
    is_capture_fn = board.is_capture
    push = board.push
    sq_file = chess.square_file
    sq_rank = chess.square_rank
    sq_ctor = chess.square

    for ply, san in enumerate(sans):
        try:
            move = parse_san(san)
        except Exception:
            return board
        from_sq = move.from_square
        to_sq = move.to_square
        ev.ply = ply
        ev.san = san
        ev.move = move
        ev.from_sq = from_sq
        ev.to_sq = to_sq
        ev.mover = board.turn
        is_castle = is_castling(move)
        ev.is_castle = is_castle
        ev.is_kingside = is_castle and is_kingside_castling(move)
        is_ep = is_en_passant_fn(move)
        ev.is_en_passant = is_ep
        ev.is_capture = is_ep or is_capture_fn(move)
        if is_ep:
            ev.ep_captured_sq = sq_ctor(sq_file(to_sq), sq_rank(from_sq))
        else:
            ev.ep_captured_sq = -1
        ev.promotion = move.promotion
        push(move)
        for fn in on_move_fns:
            fn(gd, ev)
    return board


# ============================================================
# Question protocol
# ============================================================


@dataclass(slots=True)
class GameData:
    gid: bytes
    idx: int
    sans: list[str]
    clocks: list[int]


class Question:
    """Subclasses set ``_mask`` (numpy bool over n_games, or None for
    always-relevant). The driver uses ``_mask`` directly — the legacy
    ``relevant()`` method is kept for API compatibility but no longer
    called per game in the moves pass.
    """

    name: str = ""
    needs_walk: bool = False
    needs_board: bool = False
    _mask: Optional[np.ndarray] = None  # type: ignore[assignment]

    def __init__(self, gi: GameIndex):
        self.gi = gi

    def relevant(self, idx: int) -> bool:
        if self._mask is None:
            return idx >= 0
        return idx >= 0 and bool(self._mask[idx])

    def begin_game(self, gd: GameData) -> None:
        pass

    def on_move(self, gd: GameData, ev: MoveEvent) -> None:
        pass

    def on_end(self, gd: GameData, board: Optional[chess.Board]) -> None:
        pass

    def finalize(self) -> str:
        return ""


# ============================================================
# Reservoir sampler
# ============================================================


class Reservoir:
    """Numerically stable reservoir sampler. Identical seed/algorithm to
    the original so the fitted coefficients are reproducible."""

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
    name, needs_board = "Q1", True

    def __init__(self, gi: GameIndex):
        super().__init__(gi)
        self._mask = (
            gi.is_std
            & (gi.utcdates >= b"2023.10.12") & (gi.utcdates <= b"2024.02.19")
            & (gi.results != gi.RES_DRAW)
        )
        self._count = 0

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
    name, needs_walk = "Q2", True

    def __init__(self, gi: GameIndex):
        super().__init__(gi)
        self._mask = gi.results != gi.RES_DRAW
        self._lk_w = self._lk_t = self._no_w = self._no_t = 0
        self._wlk = self._blk = -1
        self._wcap = self._bcap = False

    def begin_game(self, gd: GameData) -> None:
        self._wlk, self._blk = chess.B1, chess.G8
        self._wcap = self._bcap = False

    def on_move(self, gd: GameData, ev: MoveEvent) -> None:
        if ev.mover == chess.WHITE:
            if self._wlk >= 0 and ev.from_sq == self._wlk:
                if ev.san[0] == "N":
                    if ev.is_capture:
                        self._wcap = True
                    self._wlk = ev.to_sq
                else:
                    self._wlk = -1
            if self._blk >= 0 and ev.is_capture and ev.to_sq == self._blk:
                self._blk = -1
        else:
            if self._blk >= 0 and ev.from_sq == self._blk:
                if ev.san[0] == "N":
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
    name, needs_walk = "Q3", True

    def __init__(self, gi: GameIndex):
        super().__init__(gi)
        self._mask = gi.tcbases == 600
        self._count = 0
        self._board: Optional[chess.Board] = None
        self._had: bool = False

    def begin_game(self, gd: GameData) -> None:
        self._board = chess.Board()
        self._had = True

    def on_move(self, gd: GameData, ev: MoveEvent) -> None:
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
            self._board = None

    def finalize(self) -> str:
        return str(self._count)


class Q4RookDistance(Question):
    name, needs_walk = "Q4", True

    def __init__(self, gi: GameIndex):
        super().__init__(gi)
        self._wd = self._bd = 0

    def on_move(self, gd: GameData, ev: MoveEvent) -> None:
        if ev.is_castle:
            d = 2 if ev.is_kingside else 3
        elif ev.san[0] == "R":
            t = ev.to_sq
            f = ev.from_sq
            d = abs((t & 7) - (f & 7)) + abs((t >> 3) - (f >> 3))
        else:
            return
        if ev.mover == chess.WHITE:
            self._wd += d
        else:
            self._bd += d

    def finalize(self) -> str:
        return f"Fehér − Fekete bástya távolság: {self._wd - self._bd} mező (fehér: {self._wd}, fekete: {self._bd})"


class Q5ScissorsThreefold(Question):
    name, needs_board = "Q5", True

    def __init__(self, gi: GameIndex):
        super().__init__(gi)
        self._mask = gi.has_scissors & (gi.results == gi.RES_DRAW)
        self._count = 0

    def on_end(self, gd: GameData, board: Optional[chess.Board]) -> None:
        if board is not None and board.is_repetition(3):
            self._count += 1

    def finalize(self) -> str:
        return str(self._count)


class Q6ThreefoldDateRange(Question):
    name, needs_board = "Q6", True

    def __init__(self, gi: GameIndex):
        super().__init__(gi)
        self._mask = (
            gi.is_std
            & (gi.utcdates >= b"2024.03.12") & (gi.utcdates <= b"2024.11.19")
            & (gi.results == gi.RES_DRAW) & (gi.terms == gi.TERM_NORMAL)
        )
        self._count = 0

    def on_end(self, gd: GameData, board: Optional[chess.Board]) -> None:
        if board is not None and board.is_repetition(3):
            self._count += 1

    def finalize(self) -> str:
        return str(self._count)


class Q7QueensAtMate(Question):
    name, needs_board = "Q7", True

    def __init__(self, gi: GameIndex):
        super().__init__(gi)
        self._mask = (
            gi.is_tour_winner & (gi.terms == gi.TERM_NORMAL) & (gi.results != gi.RES_DRAW)
        )
        self._sum = 0
        self._n = 0

    def on_end(self, gd: GameData, board: Optional[chess.Board]) -> None:
        if not gd.sans or "#" not in gd.sans[-1] or board is None:
            return
        self._sum += len(board.pieces(chess.QUEEN, chess.WHITE))
        self._n += 1

    def finalize(self) -> str:
        return f"{(self._sum / self._n) if self._n else 0.0:.4f} ({self._n} parti)"


class Q8DrawMarch20Promo(Question):
    name = "Q8"

    def __init__(self, gi: GameIndex):
        super().__init__(gi)
        self._mask = (gi.results == gi.RES_DRAW) & np.array(
            [d.endswith(b".03.20") for d in gi.utcdates]
        )
        self._count = 0

    def on_end(self, gd: GameData, board: Optional[chess.Board]) -> None:
        if gd.sans and "=Q" in gd.sans[-1]:
            self._count += 1

    def finalize(self) -> str:
        return str(self._count)


class Q10WinLogit(Question):
    name, needs_walk = "Q10", True

    def __init__(self, gi: GameIndex):
        super().__init__(gi)
        self._mask = gi.results != gi.RES_DRAW
        self._sample = Reservoir(k=2_000_000, n_features=3)
        self._wcap = self._bcap = 0
        self._wt = self._bt = 0.0
        self._wn = self._bn = 0
        self._pw = self._pb = -1
        self._skip = False

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
    name = "Q11"

    def __init__(self, gi: GameIndex):
        super().__init__(gi)
        self._mask = (gi.terms == gi.TERM_NORMAL) & (gi.results != gi.RES_DRAW)
        self._counts: dict[int, int] = defaultdict(int)

    def on_end(self, gd: GameData, board: Optional[chess.Board]) -> None:
        if not gd.sans or "#" in gd.sans[-1]:
            return
        r = int(self.gi.results[gd.idx])
        loser = int(self.gi.bidxs[gd.idx]) if r == self.gi.RES_WHITE else int(self.gi.widxs[gd.idx])
        self._counts[loser] += 1

    def finalize(self) -> str:
        if not self._counts:
            return "Nincs"
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
    name, needs_walk = "Q13", True

    def __init__(self, gi: GameIndex):
        super().__init__(gi)
        self._mask = gi.results != gi.RES_DRAW
        self._more_w = self._more_t = self._less_w = self._less_t = 0
        self._wt = self._bt = 0.0
        self._pw = self._pb = -1
        self._skip = False

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
        if ev.is_capture and ev.to_sq == self._tracked and ev.mover == chess.BLACK:
            self._tracked = -1
            return
        if ev.is_en_passant and ev.ep_captured_sq == self._tracked:
            self._tracked = -1
            return
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
    name = "Q15"

    def __init__(self, gi: GameIndex):
        super().__init__(gi)
        self._counts: Counter = Counter()

    def on_end(self, gd: GameData, board: Optional[chess.Board]) -> None:
        find = PROMO_RE.search
        counts = self._counts
        for san in gd.sans:
            m = find(san)
            if m and m.group(1) != "Q":
                counts[m.group(1)] += 1

    def finalize(self) -> str:
        total = sum(self._counts.values())
        top3 = self._counts.most_common(3)
        return f"Nem vezérre: {total} | Top 3: " + ", ".join(f"{p}:{c}" for p, c in top3)


class Q17MoveLogit(Question):
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
    name, needs_board = "Q19", True

    def __init__(self, gi: GameIndex):
        super().__init__(gi)
        self._mask = (
            gi.is_std
            & (gi.utcdates >= b"2026.03.15") & (gi.utcdates <= b"2026.10.14")
            & (gi.results == gi.RES_DRAW) & (gi.terms == gi.TERM_NORMAL)
        )
        self._count = 0

    def on_end(self, gd: GameData, board: Optional[chess.Board]) -> None:
        if board is not None and board.is_fifty_moves():
            self._count += 1

    def finalize(self) -> str:
        return str(self._count)


class Q20QueensGambit(Question):
    name = "Q20"

    def __init__(self, gi: GameIndex):
        super().__init__(gi)
        std = gi.is_std
        ud = gi.utcdates
        m_bytes = ud.view("u1").reshape(len(ud), 10)
        month = (m_bytes[:, 5].astype(np.int8) - 48) * 10 + (m_bytes[:, 6].astype(np.int8) - 48)
        self._mask = std & ((month == 4) | (month == 5) | (month == 3) | (month == 6))
        self._yr_total: Counter = Counter()
        self._yr_qg: Counter = Counter()

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
    name, needs_walk = "Q22", True

    def __init__(self, gi: GameIndex):
        super().__init__(gi)
        self._counts: dict[int, int] = defaultdict(int)
        self._max_area = 0
        self._sq_orig: dict[int, int] = {}
        self._hist: dict[int, list[int]] = {}
        self._w_rect = 0
        self._b_rect = 0

    def begin_game(self, gd: GameData) -> None:
        self._sq_orig = {sq: sq for sq in STARTING_PIECE_SQUARES}
        self._hist = {sq: [sq] for sq in STARTING_PIECE_SQUARES}
        self._w_rect = self._b_rect = 0

    def on_move(self, gd: GameData, ev: MoveEvent) -> None:
        sq_orig = self._sq_orig
        hist = self._hist
        to_sq = ev.to_sq
        from_sq = ev.from_sq
        # Remove the captured piece's tracker (en-passant or regular).
        if ev.is_en_passant:
            sq_orig.pop(ev.ep_captured_sq, None)
        else:
            sq_orig.pop(to_sq, None)
        orig = sq_orig.pop(from_sq, None)
        if orig is None:
            if ev.is_castle:
                self._handle_castle_rook(ev)
            return
        sq_orig[to_sq] = orig
        h = hist.get(orig)
        if h is None:
            h = [from_sq]
            hist[orig] = h
        h.append(to_sq)

        if ev.is_castle:
            self._handle_castle_rook(ev)

        if len(h) >= 4:
            a = h[-4]; b = h[-3]; c = h[-2]; d = h[-1]
            # Inlined rectangle check using bit ops (square_file/rank are
            # otherwise per-call Python overhead).
            fa = a & 7; ra = a >> 3
            fb = b & 7; rb = b >> 3
            fc = c & 7; rc = c >> 3
            fd = d & 7; rd = d >> 3
            f_min = fa; f_max = fa
            if fb < f_min: f_min = fb
            elif fb > f_max: f_max = fb
            if fc < f_min: f_min = fc
            elif fc > f_max: f_max = fc
            if fd < f_min: f_min = fd
            elif fd > f_max: f_max = fd
            r_min = ra; r_max = ra
            if rb < r_min: r_min = rb
            elif rb > r_max: r_max = rb
            if rc < r_min: r_min = rc
            elif rc > r_max: r_max = rc
            if rd < r_min: r_min = rd
            elif rd > r_max: r_max = rd
            # Two distinct files AND two distinct ranks.
            if (f_max != f_min) and (r_max != r_min) and ({fa, fb, fc, fd}.__len__() == 2) and ({ra, rb, rc, rd}.__len__() == 2):
                # Adjacent pairs must share a file or a rank.
                if (
                    (fa == fb or ra == rb)
                    and (fb == fc or rb == rc)
                    and (fc == fd or rc == rd)
                    and (fd == fa or rd == ra)
                ):
                    area = (f_max - f_min) * (r_max - r_min)
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
    name = "Q23"

    def __init__(self, gi: GameIndex):
        super().__init__(gi)
        self._counts: Counter = Counter()

    def on_end(self, gd: GameData, board: Optional[chess.Board]) -> None:
        if not gd.sans:
            return
        last = gd.sans[-1]
        if last != "O-O#" and last != "O-O-O#":
            return
        color = "white" if (len(gd.sans) - 1) % 2 == 0 else "black"
        self._counts[self.gi.player_at(gd.idx, color)] += 1

    def finalize(self) -> str:
        if not self._counts:
            return ""
        top = max(self._counts.values())
        return ", ".join(sorted(p for p, c in self._counts.items() if c == top)[:10])


class Q24EnPassantIndian(Question):
    name, needs_walk = "Q24", True

    def __init__(self, gi: GameIndex):
        super().__init__(gi)
        eco_starts_e = np.array([e[:1] == b"E" for e in gi.ecos])
        self._mask = (gi.tcbases == 180) & eco_starts_e
        self._count = 0

    def on_move(self, gd: GameData, ev: MoveEvent) -> None:
        if ev.mover == chess.WHITE and ev.is_en_passant:
            self._count += 1

    def finalize(self) -> str:
        return str(self._count)


# ============================================================
# Pure-GameIndex questions
# ============================================================


def q9_berserk_timeouts(gi: GameIndex) -> str:
    mask = (gi.terms == gi.TERM_TIMEFORFEIT) & (gi.tcbases > 0)
    half = gi.tcbases.astype(np.int32) // 2 + 2

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
    boundaries = np.concatenate(([0], np.flatnonzero(np.diff(pids)) + 1, [len(pids)]))
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
    pid, s, e, n, _ = max(tied, key=lambda t: t[4])
    return f"{gi.player_names[pid]} | {s.decode()} – {e.decode()} | {n} parti"


def q18_winless_streak(gi: GameIndex) -> str:
    pids, _, is_win, dates, _, elos = _streak_arrays(gi)
    is_winless = ~is_win
    tied, best_n = _scan_streaks(pids, is_winless, dates, elos)
    del pids, is_win, is_winless, dates, elos
    if not tied:
        return "Nincs"
    lk = hu_key("Lili")
    named = [(str(gi.player_names[pid]), s, e, n) for pid, s, e, n, _ in tied]
    after = [t for t in named if hu_key(t[0]) > lk]
    pool = after if after else named
    name, s, e, n = min(pool, key=lambda t: hu_key(t[0]))
    return f"{name} | {s.decode()} – {e.decode()} | {n} parti"


# ---- Q12 ----


def _longest_cycle(graph: dict[int, set[int]], time_budget_s: float, max_depth: int = 12) -> list[int]:
    best: list[int] = []
    deadline = time.time() + time_budget_s

    nodes = list(graph)
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
    print("Q12: building per-year win graphs...")
    mask = gi.is_std & (gi.results != gi.RES_DRAW)
    sel = np.flatnonzero(mask)
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

    n = len(best)

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
    """Single streamed pass over moves.csv.gz. The per-question relevance
    is pre-baked into a uint32 bitmask per game, so the per-game inner
    loop avoids ~1.1 B Python ``relevant()`` calls over the full run.
    """
    walkers = [q for q in questions if q.needs_walk]
    boarders = [q for q in questions if q.needs_board and not q.needs_walk]
    others = [q for q in questions if not q.needs_walk and not q.needs_board]

    n_games = len(gi.gids)
    ones_bool = None  # built lazily

    def _mask_or_ones(q):
        nonlocal ones_bool
        if q._mask is not None:
            return q._mask
        if ones_bool is None:
            ones_bool = np.ones(n_games, dtype=bool)
        return ones_bool

    # Pack relevance into 64-bit bitmasks (up to 64 walkers/boarders/others
    # each; we never come close).
    def _pack(qs: list[Question]) -> np.ndarray:
        out = np.zeros(n_games, dtype=np.uint64)
        for i, q in enumerate(qs):
            m = _mask_or_ones(q)
            out |= m.astype(np.uint64) << np.uint64(i)
        return out

    walker_bits = _pack(walkers)
    boarder_bits = _pack(boarders)
    other_bits = _pack(others)

    any_bits = walker_bits | boarder_bits | other_bits

    pbar = tqdm(total=total, desc="moves pass", unit=" games", smoothing=0.01)
    update = pbar.update

    # Local refs for the hot loop.
    n_walkers = len(walkers)
    n_boarders = len(boarders)
    n_others = len(others)

    for idx, sans, clocks in stream_games(gi):
        update(1)
        if idx < 0:
            continue
        ab = int(any_bits[idx])
        if ab == 0:
            continue

        wb = int(walker_bits[idx])
        bb = int(boarder_bits[idx])
        ob = int(other_bits[idx])

        relevant_walkers = [walkers[i] for i in range(n_walkers) if wb & (1 << i)]
        relevant_boarders = [boarders[i] for i in range(n_boarders) if bb & (1 << i)]
        relevant_others = [others[i] for i in range(n_others) if ob & (1 << i)]

        gd = GameData(gid=gi.gids[idx], idx=idx, sans=sans, clocks=clocks)

        for q in relevant_walkers:
            q.begin_game(gd)

        board: Optional[chess.Board] = None
        if relevant_walkers:
            board = walk_game(sans, relevant_walkers, gd)
        elif relevant_boarders:
            board = simulate_board(sans)

        if relevant_walkers:
            for q in relevant_walkers:
                q.on_end(gd, board)
        for q in relevant_boarders:
            q.on_end(gd, board)
        for q in relevant_others:
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
