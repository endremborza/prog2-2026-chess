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

import os
import re
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from functools import partial
from typing import Any, Iterator, Optional

# bulletchess replaces the old `chess` library for all board simulation.
# It is a C-backed library with identical semantics but far higher throughput.
import bulletchess as bc
from bulletchess import (
    Board, Move, Square, Piece, Color, PieceType,
    WHITE, BLACK,
    PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING,
    SQUARES,
    A1, A2, A8, B1, D1, F1, G8, H1, H8,
    F8, A8, D8,
    CHECKMATE, DRAW, CHECK,FIFTY_MOVE_TIMEOUT, THREEFOLD_REPETITION
)
import numpy as np
import pyarrow.parquet as pq
import polars as pl
import pytz
from concurrent.futures import ProcessPoolExecutor
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm

# ============================================================
# Config
# ============================================================

GAMES = "games.csv.gz"
MOVES = "moves.csv.gz"
TOURNAMENTS = "tournaments.csv.gz"
GAMES_PARQUET = "games.parquet"
MOVES_PARQUET = "moves.parquet"
TOURNAMENTS_PARQUET = "tournaments.parquet"
OUTPUT = "gabili.md"
# Lower the chunk size to avoid large in-memory Arrow/Polars batches
# which can spike peak memory on modest machines.
CHUNKSIZE = 300_000

CET = pytz.timezone("Europe/Budapest")
UTC = pytz.utc

SCISSORS_RE = re.compile(r"[✂✀✁✃✄]")
PROMO_RE = re.compile(r"=([RBNQ])")

# Build the set of starting squares using bulletchess.
# Board() gives the standard starting position; we collect every square
# that has a piece on it.
_START_BOARD = Board()
STARTING_PIECE_SQUARES: tuple[Square, ...] = tuple(
    sq for sq in SQUARES if _START_BOARD[sq] is not None
)
del _START_BOARD

# Bulletchess Square objects for the squares we reference by name in handlers.
# These are already imported from bulletchess above (A2, B1, G8, etc.).
# We also need integer indices for dict keys (bulletchess Squares are objects,
# not plain ints, so we use .index() where an int key is required).
_B1_IDX = B1.index()
_G8_IDX = G8.index()
_A2_IDX = A2.index()
_G8_IDX_INT = G8.index()

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


def parse_clock_vec(values: list[Any]) -> np.ndarray:
    """Convert H:MM:SS strings to seconds. Invalid rows yield -1."""
    out = np.empty(len(values), dtype=np.int32)
    for i, value in enumerate(values):
        try:
            if value is None:
                raise ValueError("missing")
            h, m, sec = str(value).split(":")
            out[i] = int(h) * 3600 + int(m) * 60 + int(sec)
        except Exception:
            out[i] = -1
    return out


def parse_clock_expr(expr: pl.Expr) -> pl.Expr:
    """Build a Polars expression that converts H:MM:SS into seconds."""
    parts = expr.cast(pl.Utf8).str.split(":")

    hours = parts.list.get(0).cast(pl.Int32, strict=False)
    minutes = parts.list.get(1).cast(pl.Int32, strict=False)
    seconds = parts.list.get(2).cast(pl.Int32, strict=False)

    total = hours * 3600 + minutes * 60 + seconds

    return pl.when(parts.list.len() == 3).then(total).otherwise(None).cast(pl.Int32).fill_null(-1)


def parse_tc_base(s) -> Any:
    """Time control base in seconds (the part before the ``+``)."""
    if isinstance(s, pl.Expr):
        return (
            s.cast(pl.Utf8)
            .str.split("+")
            .list.get(0)
            .cast(pl.Int32, strict=False)
            .fill_null(0)
            .clip(0, 32767)
            .cast(pl.Int16)
        )
    try:
        return int(str(s).split("+")[0])
    except Exception:
        return 0


def parquet_for(csv_path: str) -> str:
    if csv_path.endswith(".csv.gz"):
        return csv_path[:-7] + ".parquet"
    if csv_path.endswith(".csv"):
        return csv_path[:-4] + ".parquet"
    return csv_path + ".parquet"


def ensure_parquet(source: str, target: str, **kwargs: Any) -> str:
    if os.path.exists(target) and os.path.getmtime(target) >= os.path.getmtime(source):
        return target

    print(f"Converting {source} → {target}...")

    columns_to_keep = kwargs.pop("columns", None)

    if "dtypes" in kwargs:
        kwargs["schema_overrides"] = kwargs.pop("dtypes")

    lf = pl.scan_csv(source, **kwargs)

    if columns_to_keep:
        lf = lf.select(columns_to_keep)

    lf.sink_parquet(target)

    print(f"   wrote {target}")
    return target


def best_moves_source() -> str:
    """Return the parquet source for moves, converting CSV lazily if needed."""
    if os.path.exists(MOVES_PARQUET) and (
        not os.path.exists(MOVES) or os.path.getmtime(MOVES_PARQUET) >= os.path.getmtime(MOVES)
    ):
        return MOVES_PARQUET
    if os.path.exists(MOVES):
        ensure_parquet(
            MOVES,
            MOVES_PARQUET,
            columns=["game_id", "move", "clock"],
            dtypes={"game_id": pl.Utf8, "move": pl.Utf8, "clock": pl.Utf8},
            ignore_errors=True,
            infer_schema_length=100000,
        )
        return MOVES_PARQUET
    if os.path.exists(MOVES_PARQUET):
        return MOVES_PARQUET
    raise FileNotFoundError(f"Missing move data source: {MOVES} or {MOVES_PARQUET}")


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


_MATERIAL_PIECES = ((PAWN, 1), (KNIGHT, 3), (BISHOP, 3), (ROOK, 5), (QUEEN, 9))

def count_material(board: Board) -> tuple[int, int]:
    """Count material value for white and black using bulletchess Board indexing."""
    w = 0
    b = 0
    for pt, v in _MATERIAL_PIECES:
        w += len(board[WHITE, pt]) * v
        b += len(board[BLACK, pt]) * v
    return w, b


def simulate_board(sans: list[str]) -> Optional[Board]:
    """Replay a SAN list onto a fresh board using bulletchess; tolerates malformed moves."""
    board = Board()
    for san in sans:
        try:
            move = Move.from_san(san, board)
            board.apply(move)
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

        print("Building GameIndex (chunked pass over Parquet via Polars)...")
        tournaments_src = ensure_parquet(
            TOURNAMENTS,
            TOURNAMENTS_PARQUET,
            columns=["id", "winner__id"],
            dtypes={"id": pl.Utf8, "winner__id": pl.Utf8},
        )
        tdf = pl.read_parquet(tournaments_src, columns=["id", "winner__id"]).drop_nulls("winner__id").with_columns(
            pl.col("winner__id").str.to_lowercase().alias("winner_lower")
        )

        games_src = ensure_parquet(
            GAMES,
            GAMES_PARQUET,
            columns=cols,
            dtypes={
                "game_id": pl.Utf8, "result": pl.Utf8, "variant": pl.Utf8,
                "utcdate": pl.Utf8, "utctime": pl.Utf8, "termination": pl.Utf8,
                "timecontrol": pl.Utf8, "whitestart": pl.Utf8, "blackstart": pl.Utf8,
                "eco": pl.Utf8, "whiteelo": pl.Int32, "blackelo": pl.Int32,
                "white": pl.Utf8, "black": pl.Utf8, "tournamentid": pl.Utf8,
            },
        )

        pf = pq.ParquetFile(games_src)
        total_rows = pf.metadata.num_rows

        parts: dict[str, list] = defaultdict(list)
        player_ids: dict[str, int] = {}

        with tqdm(total=total_rows, desc="games", unit=" rows", unit_scale=True) as pb:
            for batch in pf.iter_batches(batch_size=CHUNKSIZE, columns=cols):

                chunk_df = pl.from_arrow(batch)

                processed = (
                    chunk_df
                    .with_columns([
                        pl.col("utcdate").fill_null("").alias("utcdate"),
                        parse_clock_expr(pl.col("utctime")).alias("utctime"),
                        parse_tc_base(pl.col("timecontrol")).alias("tcbase"),
                        parse_clock_expr(pl.col("whitestart")).alias("whitestart"),
                        parse_clock_expr(pl.col("blackstart")).alias("blackstart"),
                        pl.col("whiteelo").fill_null(0).clip(0, 32767).cast(pl.Int16).alias("whiteelo"),
                        pl.col("blackelo").fill_null(0).clip(0, 32767).cast(pl.Int16).alias("blackelo"),
                        pl.col("white").fill_null("").alias("white"),
                        pl.col("black").fill_null("").alias("black"),
                        pl.col("eco").fill_null("").alias("eco"),
                    ])
                    .join(tdf, left_on="tournamentid", right_on="id", how="left")
                    .with_columns([
                        pl.when(pl.col("result") == "1-0").then(self.RES_WHITE)
                        .when(pl.col("result") == "0-1").then(self.RES_BLACK)
                        .otherwise(self.RES_DRAW)
                        .cast(pl.Int8).alias("result"),
                        (pl.col("variant").str.strip_chars().str.to_lowercase() == "standard").alias("std"),
                        pl.when(pl.col("termination") == "Normal").then(self.TERM_NORMAL)
                        .when(pl.col("termination") == "Time forfeit").then(self.TERM_TIMEFORFEIT)
                        .when(pl.col("termination") == "Abandoned").then(self.TERM_ABANDONED)
                        .otherwise(self.TERM_OTHER)
                        .cast(pl.Int8).alias("termination"),
                    ])
                    .with_columns([
                        pl.when(
                            ((pl.col("result") == self.RES_WHITE) & (pl.col("white").str.to_lowercase() == pl.col("winner_lower")))
                            | ((pl.col("result") == self.RES_BLACK) & (pl.col("black").str.to_lowercase() == pl.col("winner_lower")))
                        ).then(True).otherwise(False).alias("is_tour_winner"),
                        (
                            pl.col("white").str.contains(SCISSORS_RE.pattern).fill_null(False)
                            | pl.col("black").str.contains(SCISSORS_RE.pattern).fill_null(False)
                        ).alias("has_scissors"),
                    ])
                )

                parts["gid"].append(processed["game_id"].to_numpy().astype("S14"))
                parts["res"].append(processed["result"].to_numpy())
                parts["std"].append(processed["std"].to_numpy())
                parts["udate"].append(processed["utcdate"].to_numpy().astype("S10"))
                parts["utime"].append(processed["utctime"].to_numpy())
                parts["term"].append(processed["termination"].to_numpy())
                parts["tcbase"].append(processed["tcbase"].to_numpy())
                parts["ws"].append(processed["whitestart"].to_numpy())
                parts["bs"].append(processed["blackstart"].to_numpy())
                parts["eco"].append(processed["eco"].to_numpy().astype("S3"))
                parts["welo"].append(processed["whiteelo"].to_numpy())
                parts["belo"].append(processed["blackelo"].to_numpy())
                parts["tw"].append(processed["is_tour_winner"].to_numpy())

                w_names = processed["white"].to_numpy().astype(str)
                b_names = processed["black"].to_numpy().astype(str)

                for n in w_names:
                    if n not in player_ids:
                        player_ids[n] = len(player_ids)
                for n in b_names:
                    if n not in player_ids:
                        player_ids[n] = len(player_ids)

                parts["wid"].append(np.fromiter((player_ids[n] for n in w_names), dtype=np.int32, count=len(w_names)))
                parts["bid"].append(np.fromiter((player_ids[n] for n in b_names), dtype=np.int32, count=len(b_names)))

                pb.update(len(chunk_df))

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

        self.player_names = np.array(sorted(player_ids, key=player_ids.get), dtype=object)
        del player_ids, order

        # Derived: scissors-emoji games (used by Q5).
        sc = np.array([bool(SCISSORS_RE.search(n)) for n in self.player_names], dtype=bool)
        self.has_scissors = sc[self.widxs] | sc[self.bidxs]

        self._gid_to_idx = None

        mb = sum(getattr(self, k).nbytes for k in (
            "gids", "results", "is_std", "utcdates", "utctimes", "terms",
            "tcbases", "wstarts", "bstarts", "ecos", "welos", "belos",
            "is_tour_winner", "widxs", "bidxs", "has_scissors",
        )) // 1024 // 1024
        print(f"  GameIndex ready: {len(self.gids):,} games, ~{mb} MB metadata")

    def build_idx_lookup(self) -> None:
        """Build an O(1) hash table gid→idx for the moves pass.

        ~80 bytes/entry × 60 M games ≈ 4–5 GB; call only if RAM allows.
        Falls back to binary search if not called.
        """
        self._gid_to_idx: dict[bytes, int] = {
            bytes(g): i for i, g in enumerate(self.gids)
        }

    def idx(self, gid: bytes) -> int:
        """Return the game index for a given game_id, or -1 if missing."""
        if self._gid_to_idx is not None:
            return self._gid_to_idx.get(gid, -1)
        i = int(np.searchsorted(self.gids, gid, side="left"))
        if i < len(self.gids) and self.gids[i] == gid:
            return i
        return -1

    def player_at(self, idx: int, color: str) -> str:
        if color == "white":
            return str(self.player_names[int(self.widxs[idx])])
        return str(self.player_names[int(self.bidxs[idx])])

    def cet_year(self) -> np.ndarray:
        if hasattr(self, "_cet_years"):
            return self._cet_years

        d_bytes = self.utcdates.view(np.uint8).reshape(-1, 10)
        years = (
            (d_bytes[:, 0].astype(np.int32) - 48) * 1000
            + (d_bytes[:, 1].astype(np.int32) - 48) * 100
            + (d_bytes[:, 2].astype(np.int32) - 48) * 10
            + (d_bytes[:, 3].astype(np.int32) - 48)
        )
        months = (
            (d_bytes[:, 5].astype(np.int32) - 48) * 10
            + (d_bytes[:, 6].astype(np.int32) - 48)
        )
        days = (
            (d_bytes[:, 8].astype(np.int32) - 48) * 10
            + (d_bytes[:, 9].astype(np.int32) - 48)
        )

        month_starts = np.array([0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334], dtype=np.int32)
        leap = ((years % 4 == 0) & (((years % 100) != 0) | ((years % 400) == 0))).astype(np.int32)
        day_of_year = days + month_starts[months - 1] + (leap & (months > 2))
        
        del leap  # Free memory immediately

        # sec_of_year safely fits in int32 (Max ~31.6M < 2.14B). No need for int64.
        sec_of_year = (day_of_year - 1) * 86400 + self.utctimes
        
        del day_of_year  # Free memory

        # Offsets are just 1 or 2, int8 is more than enough (1 byte instead of 4 or 8)
        offsets = np.ones_like(self.utctimes, dtype=np.int8)
        unique_years = np.unique(years)
        
        for yr in unique_years:
            start = self._last_sunday_utc(yr, 3)
            end = self._last_sunday_utc(yr, 10)
            start_sec = (start - 1) * 86400 + 3600
            end_sec = (end - 1) * 86400 + 3600

            # Evaluate conditions in-place bitwise to prevent massive temp array creation
            dst_mask = (years == yr)
            dst_mask &= (sec_of_year >= start_sec)
            dst_mask &= (sec_of_year < end_sec)
            offsets[dst_mask] = 2

        del sec_of_year  # Free memory

        # Compute next year crossings directly in-place
        next_year = (months == 12)
        next_year &= (days == 31)
        next_year &= ((self.utctimes + offsets) >= 86400)

        del months, days  # Free memory
        
        # Modify the 'years' array in-place, which acts as our final output
        years[next_year] += 1

        self._cet_years = years
        return self._cet_years
    
    @staticmethod
    def _last_sunday_utc(year: int, month: int) -> int:
        from datetime import date, timedelta

        d = date(year, month, 31)
        while d.weekday() != 6:
            d -= timedelta(days=1)
        return d.timetuple().tm_yday


# ============================================================
# Streaming moves: yields (gid_bytes, sans, clocks) per game
# ============================================================


def stream_games() -> Iterator[tuple[list[bytes], list[list[str]], list[list[int]]]]:
    """Stream batches of complete games from moves.parquet in file order."""
    source = best_moves_source()
    yield from stream_games_parquet(source)


def stream_games_parquet(path: str) -> Iterator[tuple[list[bytes], list[list[str]], list[list[int]]]]:
    """Stream games using Polars, yielding large batches to enable vectorized lookups."""
    parquet_file = pq.ParquetFile(path)

    cur_gid: Optional[bytes] = None
    cur_sans: list[str] = []
    cur_clocks: list[int] = []

    batch_gids = []
    batch_sans_list = []
    batch_clocks_list = []
    
    # Process 50k games at a time. High enough for fast NumPy lookups, low enough for minimal RAM.
    BATCH_YIELD_SIZE = 50_000 

    for batch in parquet_file.iter_batches(batch_size=CHUNKSIZE, columns=["game_id", "move", "clock"]):

        df = (
            pl.from_arrow(batch)
            .with_columns([
                pl.col("game_id").cast(pl.Binary),
                pl.col("move").fill_null("").cast(pl.Utf8),
                parse_clock_expr(pl.col("clock")).alias("clock")
            ])
            .group_by("game_id", maintain_order=True)
            .agg([
                pl.col("move"),
                pl.col("clock")
            ])
        )

        gids = df["game_id"].to_list()
        moves_list = df["move"].to_list()
        clocks_list = df["clock"].to_list()

        for g, m, c in zip(gids, moves_list, clocks_list):
            if g == cur_gid:
                cur_sans.extend(m)
                cur_clocks.extend(c)
            else:
                if cur_gid is not None:
                    batch_gids.append(cur_gid)
                    batch_sans_list.append(cur_sans)
                    batch_clocks_list.append(cur_clocks)
                    
                    if len(batch_gids) >= BATCH_YIELD_SIZE:
                        yield batch_gids, batch_sans_list, batch_clocks_list
                        batch_gids = []
                        batch_sans_list = []
                        batch_clocks_list = []

                cur_gid = g
                cur_sans = list(m)
                cur_clocks = list(c)

    # Yield whatever is left
    if cur_gid is not None:
        batch_gids.append(cur_gid)
        batch_sans_list.append(cur_sans)
        batch_clocks_list.append(cur_clocks)

    if batch_gids:
        yield batch_gids, batch_sans_list, batch_clocks_list


# ============================================================
# Walker: replays one game once, dispatches MoveEvent to handlers
# ============================================================


@dataclass(slots=True)
class MoveEvent:
    ply: int
    san: str
    move: Move                # bulletchess Move object
    from_sq: int              # integer index (Square.index())
    to_sq: int                # integer index
    mover: Color              # bulletchess WHITE or BLACK
    is_capture: bool
    is_castle: bool
    is_kingside: bool
    is_en_passant: bool
    ep_captured_sq: int       # integer index, or -1
    promotion: Optional[PieceType]   # bulletchess PieceType or None


# A single reusable MoveEvent is mutated in-place each ply to avoid
# allocating a new object per move. This is safe because on_move handlers
# only read the event (never store a reference to it).
_EV = MoveEvent(0, "", None, 0, 0, WHITE, False, False, False, False, -1, None)  # type: ignore[arg-type]


def _sq_file(sq_idx: int) -> int:
    """File (0=a … 7=h) from a square index (bulletchess: A1=0, B1=1, …)."""
    return sq_idx % 8


def _sq_rank(sq_idx: int) -> int:
    """Rank (0=1 … 7=8) from a square index."""
    return sq_idx // 8


def walk_game(sans: list[str], walkers: list["Question"], gd: "GameData") -> Optional[Board]:
    """Replay sans via bulletchess, dispatch MoveEvents to walker handlers, return final board.

    Tolerates malformed SANs by stopping early and returning the board so far.

    Performance: a single MoveEvent object (_EV) is mutated in-place each ply
    rather than allocating a new one. This eliminates ~hundreds of millions of
    heap allocations across the full dataset. Handlers must NOT hold references
    to the event object between calls.
    """
    board = Board()
    ev = _EV
    for ply, san in enumerate(sans):
        try:
            move = Move.from_san(san, board)
        except Exception:
            return board

        mover: Color = board.turn
        from_sq_obj = move.origin
        to_sq_obj = move.destination
        from_sq: int = from_sq_obj.index()
        to_sq: int = to_sq_obj.index()

        # Castle detection: king moves exactly 2 files.
        is_castle = False
        is_kingside = False
        mover_piece = board[from_sq_obj]
        if mover_piece is not None and mover_piece.piece_type == KING:
            file_diff = (to_sq & 7) - (from_sq & 7)  # inline _sq_file
            if file_diff == 2:
                is_castle = True
                is_kingside = True
            elif file_diff == -2:
                is_castle = True

        # En passant: pawn moves diagonally to an empty square.
        target_piece = board[to_sq_obj]
        is_ep = False
        ep_captured_sq = -1
        if (
            mover_piece is not None
            and mover_piece.piece_type == PAWN
            and target_piece is None
            and (from_sq & 7) != (to_sq & 7)   # different file
        ):
            is_ep = True
            ep_captured_sq = (from_sq >> 3 << 3) | (to_sq & 7)  # same rank as from, file of to

        is_capture = (target_piece is not None) or is_ep
        promotion: Optional[PieceType] = move.promotion

        board.apply(move)

        # Mutate the shared event object in-place (zero allocation).
        ev.ply = ply
        ev.san = san
        ev.move = move
        ev.from_sq = from_sq
        ev.to_sq = to_sq
        ev.mover = mover
        ev.is_capture = is_capture
        ev.is_castle = is_castle
        ev.is_kingside = is_kingside
        ev.is_en_passant = is_ep
        ev.ep_captured_sq = ep_captured_sq
        ev.promotion = promotion

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
    - ``needs_board``: hand back the final ``Board`` to ``on_end``.
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

    def begin_game(self, gd: GameData) -> None:
        pass

    def on_move(self, gd: GameData, ev: MoveEvent) -> None:
        pass

    def on_end(self, gd: GameData, board: Optional[Board]) -> None:
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

    def on_end(self, gd: GameData, board: Optional[Board]) -> None:
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
        # Track as integer indices (Square.index()); -1 means "gone/unknown".
        self._wlk: int = -1
        self._blk: int = -1
        self._wcap = self._bcap = False

    def relevant(self, idx: int) -> bool:
        return idx >= 0 and int(self.gi.results[idx]) != self.gi.RES_DRAW

    def begin_game(self, gd: GameData) -> None:
        # White's left knight starts on b1 (index 1), black's on g8 (index 62).
        self._wlk = _B1_IDX   # b1
        self._blk = _G8_IDX   # g8
        self._wcap = self._bcap = False

    def on_move(self, gd: GameData, ev: MoveEvent) -> None:
        if ev.mover == WHITE:
            if self._wlk >= 0 and ev.from_sq == self._wlk:
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

    def on_end(self, gd: GameData, board: Optional[Board]) -> None:
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
        # We track castling rights by replaying a mini-board via bulletchess.
        self._board: Optional[Board] = None
        # Whether white had castling rights before the current ply.
        # Bulletchess Board exposes castling rights via board[WHITE, KING] bitboard
        # and the legal moves check. Simplest approach: track via a fresh Board.

    def relevant(self, idx: int) -> bool:
        return idx >= 0 and bool(self._mask[idx])

    def begin_game(self, gd: GameData) -> None:
        self._board = Board()

    def _white_can_castle(self, board: Board) -> bool:
        """Return True if white still has at least one castling right."""
        # In bulletchess, castling rights are implicit: white can castle if the
        # legal moves include a king move of 2 squares. For efficiency we check
        # the standard rook positions directly.
        # A board in the starting configuration has H1 and A1 rooks; after the
        # king or rook moves those rights are revoked. We infer by asking whether
        # any legal move is a castling move (king moves 2 squares).
        if board.turn != WHITE:
            # Temporarily we want to check from white's perspective; but the
            # board's turn may be black. We look for white king + h1/a1 rook
            # presence as a proxy instead (correct for the first 6 plies where
            # the king has almost certainly not moved yet).
            wk = board[WHITE, KING]
            wr = board[WHITE, ROOK]
            # Castling rights survive as long as king is on e1 and at least one
            # rook is on its starting square. This is a reliable proxy for the
            # first 6 plies.
            e1_idx = 4  # E1 index
            h1_idx = 7  # H1 index
            a1_idx = 0  # A1 index
            king_ok = any(sq.index() == e1_idx for sq in wk)
            rook_ok = any(sq.index() in (h1_idx, a1_idx) for sq in wr)
            return king_ok and rook_ok
        else:
            # It is white's turn, so we can generate legal moves.
            for m in board.legal_moves():
                orig = m.origin.index()
                dest = m.destination.index()
                if orig == 4:  # E1 (king square)
                    if abs(dest - orig) == 2:
                        return True
            # Also check: king still on e1 and rook on h1/a1 is sufficient
            # even if we missed the legal-move check above (e.g. in check).
            wk = board[WHITE, KING]
            wr = board[WHITE, ROOK]
            king_ok = any(sq.index() == 4 for sq in wk)
            rook_ok = any(sq.index() in (0, 7) for sq in wr)
            return king_ok and rook_ok

    def on_move(self, gd: GameData, ev: MoveEvent) -> None:
        if ev.ply >= 6 or self._board is None:
            return

        had = self._board.castling_rights.any(WHITE)

        try:
            self._board.apply(ev.move)
        except Exception:
            self._board = None
            return

        has_now = self._board.castling_rights.any(WHITE)

        if ev.mover == WHITE and had and not has_now:
            self._count += 1
            self._board = None

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
            d = abs(_sq_file(ev.to_sq) - _sq_file(ev.from_sq)) + abs(
                _sq_rank(ev.to_sq) - _sq_rank(ev.from_sq)
            )
        else:
            return
        if ev.mover == WHITE:
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

    def on_end(self, gd: GameData, board: Optional[Board]) -> None:
        if board is not None and board in THREEFOLD_REPETITION:
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

    def on_end(self, gd: GameData, board: Optional[Board]) -> None:
        if board is not None and board in THREEFOLD_REPETITION:
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

    def on_end(self, gd: GameData, board: Optional[Board]) -> None:
        if not gd.sans or "#" not in gd.sans[-1] or board is None:
            return
        # bulletchess: board[WHITE, QUEEN] returns a Bitboard; len() gives piece count.
        self._sum += len(board[WHITE, QUEEN])
        self._n += 1

    def finalize(self) -> str:
        return f"{(self._sum / self._n) if self._n else 0.0:.4f} ({self._n} parti)"


class Q8DrawMarch20Promo(Question):
    """Q8: draws on March 20 where the last move is a pawn promotion to queen."""

    name = "Q8"

    def __init__(self, gi: GameIndex):
        super().__init__(gi)
        d_bytes = gi.utcdates.view(np.uint8).reshape(-1, 10)
        is_march20 = (
            (d_bytes[:, 5] == ord('0')) & (d_bytes[:, 6] == ord('3')) &
            (d_bytes[:, 8] == ord('2')) & (d_bytes[:, 9] == ord('0'))
        )
        self._mask = (gi.results == gi.RES_DRAW) & is_march20
        self._count = 0

    def relevant(self, idx: int) -> bool:
        return idx >= 0 and bool(self._mask[idx])

    def on_end(self, gd: GameData, board: Optional[Board]) -> None:
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
            if ev.mover == WHITE:
                self._wcap += 1
            else:
                self._bcap += 1
        clk = gd.clocks[ev.ply] if ev.ply < len(gd.clocks) else -1
        if clk < 0:
            return
        if ev.mover == WHITE and self._pw >= 0:
            self._wt += self._pw - clk
            self._wn += 1
            self._pw = clk
        elif ev.mover == BLACK and self._pb >= 0:
            self._bt += self._pb - clk
            self._bn += 1
            self._pb = clk

    def on_end(self, gd: GameData, board: Optional[Board]) -> None:
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

    def on_end(self, gd: GameData, board: Optional[Board]) -> None:
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
        if ev.mover == WHITE and self._pw >= 0:
            self._wt += self._pw - clk
            self._pw = clk
        elif ev.mover == BLACK and self._pb >= 0:
            self._bt += self._pb - clk
            self._pb = clk

    def on_end(self, gd: GameData, board: Optional[Board]) -> None:
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
        self._tracked: int = -1   # integer square index
        self._reached: bool = False

    def begin_game(self, gd: GameData) -> None:
        self._tracked = _A2_IDX   # a2 square index
        self._reached = False

    def on_move(self, gd: GameData, ev: MoveEvent) -> None:
        if self._tracked < 0 or self._reached:
            return
        # Captured (regular)?
        if ev.is_capture and ev.to_sq == self._tracked and ev.mover == BLACK:
            self._tracked = -1
            return
        # Captured by en passant?
        if ev.is_en_passant and ev.ep_captured_sq == self._tracked:
            self._tracked = -1
            return
        # Our pawn moved?
        if ev.mover == WHITE and ev.from_sq == self._tracked:
            self._tracked = ev.to_sq
            if ev.to_sq == _G8_IDX_INT and ev.promotion is not None:
                self._reached = True

    def on_end(self, gd: GameData, board: Optional[Board]) -> None:
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

    def on_end(self, gd: GameData, board: Optional[Board]) -> None:
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
            if ev.mover == WHITE and self._pw >= 0:
                self._elapsed += self._pw - clk
                self._pw = clk
            elif ev.mover == BLACK and self._pb >= 0:
                self._elapsed += self._pb - clk
                self._pb = clk
        self._sample.add(
            [self._elapsed, 1.0 if ev.mover == WHITE else 0.0],
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

    def on_end(self, gd: GameData, board: Optional[Board]) -> None:
        # bulletchess: halfmove_clock >= 100 means 50-move rule applies.
        if board is not None and board in FIFTY_MOVE_TIMEOUT:
            self._count += 1

    def finalize(self) -> str:
        return str(self._count)


class Q20QueensGambit(Question):
    """Q20: per-year Queen's-Gambit ratio (CET 04.21–05.18, standard)."""

    name = "Q20"

    def __init__(self, gi: GameIndex):
        super().__init__(gi)
        years_cet = gi.cet_year()
        std = gi.is_std
        ud = gi.utcdates
        m_bytes = ud.view("u1").reshape(len(ud), 10)
        month = (m_bytes[:, 5].astype(np.int8) - 48) * 10 + (m_bytes[:, 6].astype(np.int8) - 48)
        self._pre_mask = std & ((month == 4) | (month == 5) | (month == 3) | (month == 6))
        self._years_cet = years_cet
        self._yr_total: Counter = Counter()
        self._yr_qg: Counter = Counter()

    def relevant(self, idx: int) -> bool:
        return idx >= 0 and bool(self._pre_mask[idx])

    def on_end(self, gd: GameData, board: Optional[Board]) -> None:
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
        self._sq_orig: dict[int, int] = {}
        self._hist: dict[int, list[int]] = {}
        self._w_rect = 0
        self._b_rect = 0

    def begin_game(self, gd: GameData) -> None:
        # Use integer square indices as keys.
        self._sq_orig = {sq.index(): sq.index() for sq in STARTING_PIECE_SQUARES}
        self._hist = {sq.index(): [sq.index()] for sq in STARTING_PIECE_SQUARES}
        self._w_rect = self._b_rect = 0

    def on_move(self, gd: GameData, ev: MoveEvent) -> None:
        if ev.is_en_passant:
            self._sq_orig.pop(ev.ep_captured_sq, None)
        else:
            self._sq_orig.pop(ev.to_sq, None)
        orig = self._sq_orig.pop(ev.from_sq, None)
        if orig is None:
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
                if ev.mover == WHITE:
                    self._w_rect += 1
                else:
                    self._b_rect += 1
                if area > self._max_area:
                    self._max_area = area

    def _handle_castle_rook(self, ev: MoveEvent) -> None:
        # Square indices for rook castling endpoints.
        # H1=7, F1=5, A1=0, D1=3, H8=63, F8=61, A8=56, D8=59
        if ev.mover == WHITE:
            rf, rt = (7, 5) if ev.is_kingside else (0, 3)
        else:
            rf, rt = (63, 61) if ev.is_kingside else (56, 59)
        ro = self._sq_orig.pop(rf, None)
        if ro is not None:
            self._sq_orig[rt] = ro
            self._hist.setdefault(ro, [rf]).append(rt)

    @staticmethod
    def _is_rect(a: int, b: int, c: int, d: int) -> bool:
        sqs = (a, b, c, d)
        files = [_sq_file(s) for s in sqs]
        ranks = [_sq_rank(s) for s in sqs]
        if len(set(files)) != 2 or len(set(ranks)) != 2:
            return False
        if (max(files) - min(files)) * (max(ranks) - min(ranks)) == 0:
            return False
        for i in range(4):
            j = (i + 1) % 4
            if files[i] != files[j] and ranks[i] != ranks[j]:
                return False
        return True

    @staticmethod
    def _rect_area(a: int, b: int, c: int, d: int) -> int:
        sqs = (a, b, c, d)
        files = [_sq_file(s) for s in sqs]
        ranks = [_sq_rank(s) for s in sqs]
        return (max(files) - min(files)) * (max(ranks) - min(ranks))

    def on_end(self, gd: GameData, board: Optional[Board]) -> None:
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

    def on_end(self, gd: GameData, board: Optional[Board]) -> None:
        if not gd.sans:
            return
        last = gd.sans[-1]
        if last not in ("O-O#", "O-O-O#"):
            return
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
        if ev.mover == WHITE and ev.is_en_passant:
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
    d_bytes = gi.utcdates.view(np.uint8).reshape(-1, 10)
    is_dec31_utc = (
        (d_bytes[:, 5] == ord('1')) & (d_bytes[:, 6] == ord('2')) &
        (d_bytes[:, 8] == ord('3')) & (d_bytes[:, 9] == ord('1'))
    )
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
    """Sort player appearances by player, date, and time using NumPy only."""
    std = gi.is_std
    sel = np.flatnonzero(std)
    N = len(sel)

    pids = np.empty(2 * N, dtype=np.int32)
    pids[:N] = gi.widxs[sel]
    pids[N:] = gi.bidxs[sel]

    game_sel = np.empty(2 * N, dtype=np.int32)
    game_sel[:N] = sel
    game_sel[N:] = sel

    d_bytes = gi.utcdates.view(np.uint8).reshape(-1, 10)[sel]
    chron = (
        (d_bytes[:, 2].astype(np.int32) - 48) * 100000 +
        (d_bytes[:, 3].astype(np.int32) - 48) * 10000 +
        (d_bytes[:, 5].astype(np.int32) - 48) * 1000 +
        (d_bytes[:, 6].astype(np.int32) - 48) * 100 +
        (d_bytes[:, 8].astype(np.int32) - 48) * 10 +
        (d_bytes[:, 9].astype(np.int32) - 48)
    )
    chron_2n = np.empty(2 * N, dtype=np.int32)
    chron_2n[:N] = chron
    chron_2n[N:] = chron
    del chron, d_bytes

    times = gi.utctimes[sel]
    times_2n = np.empty(2 * N, dtype=np.int32)
    times_2n[:N] = times
    times_2n[N:] = times
    del times

    order = np.lexsort((times_2n, chron_2n, pids))

    pids_sorted = pids[order]
    game_sel_sorted = game_sel[order]
    del pids, game_sel, order

    res_sorted = gi.results[game_sel_sorted]
    is_white = (pids_sorted == gi.widxs[game_sel_sorted])
    is_draw = (res_sorted == gi.RES_DRAW)
    is_win = np.where(is_white, res_sorted == gi.RES_WHITE, res_sorted == gi.RES_BLACK)
    elos = np.where(is_white, gi.welos[game_sel_sorted], gi.belos[game_sel_sorted])

    return pids_sorted, is_draw, is_win, game_sel_sorted, elos


def _scan_streaks(gi: GameIndex, pids: np.ndarray, cond: np.ndarray, game_sel: np.ndarray, elos: np.ndarray):
    """Vectorized streak scanner that fetches strings lazily."""
    boundaries = np.concatenate(([0], np.flatnonzero(np.diff(pids)) + 1, [len(pids)]))
    out: list[tuple[int, bytes, bytes, int, int]] = []
    best_len = 0

    for k in range(len(boundaries) - 1):
        s, e = int(boundaries[k]), int(boundaries[k + 1])
        if e <= s:
            continue

        p_cond = cond[s:e]
        if not p_cond.any():
            continue

        padded = np.concatenate(([False], p_cond, [False]))
        edges = np.diff(padded.astype(np.int8))
        starts = np.flatnonzero(edges == 1)
        ends = np.flatnonzero(edges == -1)

        lengths = ends - starts
        max_idx = int(np.argmax(lengths))
        best_run_len = int(lengths[max_idx])

        if best_run_len > 0 and best_run_len >= best_len:
            if best_run_len > best_len:
                out.clear()
                best_len = best_run_len

            g_start = s + starts[max_idx]
            g_end = s + ends[max_idx] - 1

            r_elos = elos[g_start:g_end + 1]
            non_zero = r_elos[r_elos > 0]
            run_elo = int(non_zero[-1]) if len(non_zero) > 0 else 0

            idx_start = game_sel[g_start]
            idx_end = game_sel[g_end]

            out.append(
                (pids[s], gi.utcdates[idx_start], gi.utcdates[idx_end], best_run_len, run_elo)
            )

    return out, best_len


def q16_draw_streak(gi: GameIndex) -> str:
    pids, is_draw, _, game_sel, elos = _streak_arrays(gi)
    tied, best_n = _scan_streaks(gi, pids, is_draw, game_sel, elos)
    del pids, is_draw, game_sel, elos

    if not tied:
        return "Nincs"

    pid, s, e, n, _ = max(tied, key=lambda t: t[4])
    return f"{gi.player_names[pid]} | {s.decode()} – {e.decode()} | {n} parti"


def q18_winless_streak(gi: GameIndex) -> str:
    pids, _, is_win, game_sel, elos = _streak_arrays(gi)
    is_winless = ~is_win
    tied, best_n = _scan_streaks(gi, pids, is_winless, game_sel, elos)
    del pids, is_win, is_winless, game_sel, elos

    if not tied:
        return "Nincs"

    lk = hu_key("Lili")
    named = [(str(gi.player_names[pid]), s, e, n) for pid, s, e, n, _ in tied]
    after = [t for t in named if hu_key(t[0]) > lk]
    pool = after if after else named

    name, s, e, n = min(pool, key=lambda t: hu_key(t[0]))
    return f"{name} | {s.decode()} – {e.decode()} | {n} parti"


# ---- Q12 (longest cycle in CET-year win graph) ----


def _longest_cycle(graph: dict[int, set[int]], time_budget_s: float, max_depth: int = 12) -> list[int]:
    """Iterative depth-limited DFS with global time budget."""
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


def _best_cycle_for_year(year_graph_item: tuple[int, dict[int, set[int]]], time_budget_s: float) -> tuple[int, list[int]]:
    year, graph = year_graph_item
    return year, _longest_cycle(graph, time_budget_s)


def q12_cyclic_win(gi: GameIndex, time_budget_per_year_s: float = 60.0) -> str:
    """Find the largest cyclic win pattern within a CET calendar year (standard, decisive)."""
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

    print(f"Q12: years to scan: {sorted(year_graph)}; running depth-limited DFS in parallel...")
    best: list[int] = []
    best_yr: Optional[int] = None
    items = list(year_graph.items())
    max_workers = max(1, min(len(items), os.cpu_count() or 1) - 1)
    worker = partial(_best_cycle_for_year, time_budget_s=time_budget_per_year_s)
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for yr, cyc in executor.map(worker, items):
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
    """One streamed pass over moves.parquet using Batched Vectorized lookups."""
    walkers = [q for q in questions if q.needs_walk]
    boarders = [q for q in questions if q.needs_board and not q.needs_walk]
    others = [q for q in questions if not q.needs_walk and not q.needs_board]

    pbar = tqdm(total=total, desc="moves pass", unit=" games", smoothing=0.01)
    
    # Process batches instead of individual games
    for batch_gids, batch_sans, batch_clocks in stream_games():
        
        # 1. Vectorized GameIndex lookup (C-level speed)
        gids_arr = np.array(batch_gids, dtype="S14")
        indices = np.searchsorted(gi.gids, gids_arr, side="left")
        
        # Ensure we actually found a match
        valid_mask = (indices < len(gi.gids)) & (gi.gids[indices] == gids_arr)

        for i, is_valid in enumerate(valid_mask):
            if not is_valid:
                pbar.update(1)
                continue
                
            idx = indices[i]
            sans = batch_sans[i]
            clocks = batch_clocks[i]
            gid_b = batch_gids[i]
            
            pbar.update(1)

            relevant_walkers = [q for q in walkers if q.relevant(idx)]
            relevant_boarders = [q for q in boarders if q.relevant(idx)]
            relevant_others = [q for q in others if q.relevant(idx)]

            if not (relevant_walkers or relevant_boarders or relevant_others):
                continue

            gd = GameData(gid=gid_b, idx=idx, sans=sans, clocks=clocks)

            # 2. Fast C-level String Pre-filters (Kills generator overhead)
            if relevant_walkers:
                joined_sans = " ".join(sans)

                if any(isinstance(q, Q14A2ToG8) for q in relevant_walkers):
                    has_a_pawn = " a" in joined_sans or joined_sans.startswith("a")
                    has_promo = "=" in joined_sans
                    if not (has_a_pawn and has_promo):
                        relevant_walkers = [q for q in relevant_walkers if not isinstance(q, Q14A2ToG8)]

                if any(isinstance(q, Q4RookDistance) for q in relevant_walkers):
                    has_rook_move = "R" in joined_sans or "O-O" in joined_sans
                    if not has_rook_move:
                        relevant_walkers = [q for q in relevant_walkers if not isinstance(q, Q4RookDistance)]

            # Dispatch
            for q in relevant_walkers:
                q.begin_game(gd)

            board: Optional[Board] = None
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
    print("=== Converting CSV sources to Parquet if needed ===")
    best_moves_source()

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
    import time
    start_time = time.time()

    main()

    end_time = time.time()
    elapsed = end_time - start_time
    print(f"\n✅ All done! Total execution time: {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)")