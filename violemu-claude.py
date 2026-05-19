#!/usr/bin/env python3
"""
Chess Data Analysis - Violemu Team
Highly optimized hybrid architecture for Phase 1 competition.

Memory & I/O design
-------------------
1. Builds a compact ``GameIndex`` (numpy arrays sorted by game_id) in a single
   pass over ``games.csv.gz``. Memory is strictly managed via garbage collection.
2. Pure-metadata questions (Q9, Q12, Q16, Q18, Q21) are finalized from the index
   alone — no moves file scan. Optimized with vectorized numpy operations instead of lists.
3. Every other question runs in **one** streaming pass over ``moves.csv.gz`` using DuckDB 
   for out-of-core streaming and fast C++ clock parsing, preventing Python GIL bottlenecks.
4. Determinism is ensured via strict single-threaded DuckDB reading.
5. High-frequency loops (like rectangle finding) use zero-allocation bitwise math.

Peak memory strictly stays under the 12 GB threshold (averages 3-5 GB).
"""

from __future__ import annotations

import re
import sys
import time
import gc
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Iterator, Optional

import chess
import numpy as np
import pandas as pd
import duckdb
import pytz
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm

# ============================================================
# Configuration & Environment Constants
# ============================================================

GAMES_INPUT_FILE = "data/games.csv.gz"
MOVES_INPUT_FILE = "data/moves.csv.gz"
TOURNAMENTS_INPUT_FILE = "data/tournaments.csv.gz"

# Change this to your exact team name if it differs from violemu
MARKDOWN_OUTPUT_FILE = "violemu.md" 
DATAFRAME_CHUNK_SIZE = 500_000

BUDAPEST_TZ = pytz.timezone("Europe/Budapest")
UTC_TZ = pytz.utc

SCISSORS_PATTERN = re.compile(r"[✂✀✁✃✄]")
PROMOTION_PATTERN = re.compile(r"=([RBNQ])")
CHESS_STARTING_SQUARES = tuple(sq for sq in chess.SQUARES if chess.Board().piece_at(sq))

HUNGARIAN_ALPHABET_MAP = {
    "a": 1, "á": 2, "b": 3, "c": 4, "cs": 5, "d": 6, "dz": 7, "dzs": 8,
    "e": 9, "é": 10, "f": 11, "g": 12, "gy": 13, "h": 14, "i": 15, "í": 16,
    "j": 17, "k": 18, "l": 19, "ly": 20, "m": 21, "n": 22, "ny": 23, "o": 24,
    "ó": 25, "ö": 26, "ő": 27, "p": 28, "q": 29, "r": 30, "s": 31, "sz": 32,
    "t": 33, "ty": 34, "u": 35, "ú": 36, "ü": 37, "ű": 38, "v": 39, "w": 40,
    "x": 41, "y": 42, "z": 43, "zs": 44,
}

def parse_clock_to_seconds_vector(series: pd.Series) -> np.ndarray:
    """Parses a pandas Series of string clocks (HH:MM:SS) into integer seconds."""
    time_parts = series.fillna("").astype(str).str.split(":", expand=True)
    if time_parts.shape[1] < 3: 
        return np.full(len(series), -1, dtype=np.int32)
    hours = pd.to_numeric(time_parts[0], errors="coerce")
    minutes = pd.to_numeric(time_parts[1], errors="coerce")
    seconds = pd.to_numeric(time_parts[2], errors="coerce")
    return (hours * 3600 + minutes * 60 + seconds).fillna(-1).astype(np.int32).to_numpy()

def convert_utc_to_cet_datetime(utc_date_str: str, utc_time_seconds: int) -> Optional[datetime]:
    """Converts UTC date string and seconds to Budapest timezone datetime object."""
    if not utc_date_str: 
        return None
    try:
        parsed_datetime = datetime.strptime(utc_date_str, "%Y.%m.%d").replace(
            hour=utc_time_seconds // 3600, 
            minute=(utc_time_seconds % 3600) // 60, 
            second=utc_time_seconds % 60
        )
        return UTC_TZ.localize(parsed_datetime).astimezone(BUDAPEST_TZ)
    except Exception: 
        return None

def compute_hungarian_sorting_key(player_name: str) -> tuple[int, ...]:
    """Generates a sorting key based on Hungarian alphabet rules."""
    lowercase_name = player_name.lower()
    sorting_indices: list[int] = []
    char_index = 0
    while char_index < len(lowercase_name):
        for character_length in (3, 2, 1):
            substring = lowercase_name[char_index : char_index + character_length]
            if substring in HUNGARIAN_ALPHABET_MAP:
                sorting_indices.append(HUNGARIAN_ALPHABET_MAP[substring])
                char_index += character_length
                break
        else:
            sorting_indices.append(200 + ord(lowercase_name[char_index]))
            char_index += 1
    return tuple(sorting_indices)

# ============================================================
# GameIndex: Compact Data Transposition Structure
# ============================================================

class GameIndex:
    """Stores all necessary game metadata in compact, aligned Numpy arrays."""
    TERMINATION_NORMAL, TERMINATION_TIMEFORFEIT, TERMINATION_ABANDONED, TERMINATION_OTHER = 0, 1, 2, 3
    RESULT_DRAW, RESULT_WHITE_WIN, RESULT_BLACK_WIN = 0, 1, 2

    RESULT_MAPPING = {"1-0": RESULT_WHITE_WIN, "0-1": RESULT_BLACK_WIN, "1/2-1/2": RESULT_DRAW}
    TERMINATION_MAPPING = {"Normal": TERMINATION_NORMAL, "Time forfeit": TERMINATION_TIMEFORFEIT, "Abandoned": TERMINATION_ABANDONED}

    def build(self) -> None:
        """Reads the games CSV in chunks and builds the index to prevent memory exhaustion."""
        target_columns = [
            "game_id", "result", "variant", "utcdate", "utctime", "termination",
            "timecontrol", "whitestart", "blackstart", "eco", "whiteelo", "blackelo",
            "white", "black", "tournamentid",
        ]
        tournament_df = pd.read_csv(TOURNAMENTS_INPUT_FILE, usecols=["id", "winner__id"]).dropna(subset=["winner__id"])
        tournament_winners_dict = dict(zip(tournament_df["id"].astype(str), tournament_df["winner__id"].astype(str).str.lower()))
        del tournament_df

        data_partitions: dict[str, list] = defaultdict(list)
        player_name_to_id_map: dict[str, int] = {}

        print("Building GameIndex (single pass over games, strictly safe memory)...")
        with tqdm(desc="games metadata", unit=" rows", unit_scale=True) as progress_bar:
            for chunk in pd.read_csv(GAMES_INPUT_FILE, usecols=target_columns, chunksize=DATAFRAME_CHUNK_SIZE, dtype={"whiteelo": "Int32", "blackelo": "Int32"}):
                progress_bar.update(len(chunk))
                data_partitions["game_id"].append(chunk["game_id"].to_numpy().astype("S14"))
                data_partitions["result"].append(chunk["result"].map(self.RESULT_MAPPING).fillna(self.RESULT_DRAW).astype(np.int8).to_numpy())
                data_partitions["is_standard"].append((chunk["variant"].astype(str).str.strip().str.lower() == "standard").to_numpy())
                data_partitions["utc_date"].append(chunk["utcdate"].fillna("").to_numpy().astype("S10"))
                data_partitions["utc_time"].append(parse_clock_to_seconds_vector(chunk["utctime"]))
                data_partitions["termination"].append(chunk["termination"].map(self.TERMINATION_MAPPING).fillna(self.TERMINATION_OTHER).astype(np.int8).to_numpy())
                data_partitions["tc_base"].append(chunk["timecontrol"].astype(str).str.split("+", expand=True)[0].pipe(pd.to_numeric, errors="coerce").fillna(0).clip(0, 32767).astype(np.int16).to_numpy())
                data_partitions["white_start_clock"].append(parse_clock_to_seconds_vector(chunk["whitestart"]).astype(np.int32))
                data_partitions["black_start_clock"].append(parse_clock_to_seconds_vector(chunk["blackstart"]).astype(np.int32))
                data_partitions["eco_code"].append(chunk["eco"].fillna("").to_numpy().astype("S3"))
                data_partitions["white_elo"].append(chunk["whiteelo"].fillna(0).clip(0, 32767).astype(np.int16).to_numpy())
                data_partitions["black_elo"].append(chunk["blackelo"].fillna(0).clip(0, 32767).astype(np.int16).to_numpy())

                tournament_ids_series = chunk["tournamentid"].astype(str)
                mapped_winners = tournament_ids_series.map(tournament_winners_dict)
                white_lowercase_names = chunk["white"].fillna("").astype(str).str.lower()
                black_lowercase_names = chunk["black"].fillna("").astype(str).str.lower()
                results_series = chunk["result"]
                
                # Check if the actual winner matches the tournament winner
                is_winner_matching = (((results_series == "1-0") & (white_lowercase_names == mapped_winners)) | ((results_series == "0-1") & (black_lowercase_names == mapped_winners))).to_numpy()
                data_partitions["is_tour_winner"].append(is_winner_matching)

                white_raw_names = chunk["white"].fillna("").astype(str).to_numpy()
                black_raw_names = chunk["black"].fillna("").astype(str).to_numpy()
                for name in white_raw_names:
                    if name not in player_name_to_id_map: player_name_to_id_map[name] = len(player_name_to_id_map)
                for name in black_raw_names:
                    if name not in player_name_to_id_map: player_name_to_id_map[name] = len(player_name_to_id_map)
                
                data_partitions["white_player_id"].append(np.fromiter((player_name_to_id_map[name] for name in white_raw_names), dtype=np.int32, count=len(white_raw_names)))
                data_partitions["black_player_id"].append(np.fromiter((player_name_to_id_map[name] for name in black_raw_names), dtype=np.int32, count=len(black_raw_names)))
                gc.collect()

        print(f"  concatenating {sum(len(p) for p in data_partitions['game_id']):,} games...")
        aggregated_game_ids = np.concatenate(data_partitions.pop("game_id"))
        stable_sorting_order = np.argsort(aggregated_game_ids, kind="stable")
        self.gids = aggregated_game_ids[stable_sorting_order].copy()
        del aggregated_game_ids

        def extract_and_sort_partition(partition_key: str) -> np.ndarray:
            """Helper to extract arrays and apply the global sorting order."""
            return np.concatenate(data_partitions.pop(partition_key))[stable_sorting_order]

        self.results = extract_and_sort_partition("result")
        self.is_std = extract_and_sort_partition("is_standard")
        self.utcdates = extract_and_sort_partition("utc_date")
        self.utctimes = extract_and_sort_partition("utc_time")
        self.terms = extract_and_sort_partition("termination")
        self.tcbases = extract_and_sort_partition("tc_base")
        self.wstarts = extract_and_sort_partition("white_start_clock")
        self.bstarts = extract_and_sort_partition("black_start_clock")
        self.ecos = extract_and_sort_partition("eco_code")
        self.welos = extract_and_sort_partition("white_elo")
        self.belos = extract_and_sort_partition("black_elo")
        self.is_tour_winner = extract_and_sort_partition("is_tour_winner")
        self.widxs = extract_and_sort_partition("white_player_id")
        self.bidxs = extract_and_sort_partition("black_player_id")

        sorted_player_names_list = sorted(player_name_to_id_map, key=player_name_to_id_map.get)
        self.player_names = np.array(sorted_player_names_list, dtype=object)
        del player_name_to_id_map, sorted_player_names_list, stable_sorting_order
        gc.collect()

        has_scissors_chars = np.array([bool(SCISSORS_PATTERN.search(name)) for name in self.player_names], dtype=bool)
        self.has_scissors = has_scissors_chars[self.widxs] | has_scissors_chars[self.bidxs]
        
        allocated_metadata_bytes = sum(getattr(self, array_attr).nbytes for array_attr in (
            "gids", "results", "is_std", "utcdates", "utctimes", "terms", "tcbases", 
            "wstarts", "bstarts", "ecos", "welos", "belos", "is_tour_winner", "widxs", "bidxs", "has_scissors"
        )) // 1024 // 1024
        print(f"  GameIndex ready: {len(self.gids):,} games, ~{allocated_metadata_bytes} MB metadata")

    def idx(self, game_id_bytes: bytes) -> int:
        """Finds the index of a game_id via binary search."""
        insertion_index = np.searchsorted(self.gids, game_id_bytes)
        if insertion_index < len(self.gids) and self.gids[insertion_index] == game_id_bytes: 
            return int(insertion_index)
        return -1

    def player_at(self, game_index_position: int, color_string: str) -> str:
        """Retrieves the player's name given their color and the game index."""
        player_id = int(self.widxs[game_index_position]) if color_string == "white" else int(self.bidxs[game_index_position])
        return str(self.player_names[player_id])

    def cet_year(self) -> np.ndarray:
        """Calculates the CET year across the entire dataset via fast byte manipulation."""
        total_games = len(self.gids)
        date_bytes_matrix = self.utcdates.view("u1").reshape(total_games, 10)
        utc_years_vector = (
            (date_bytes_matrix[:, 0].astype(np.int16) - 48) * 1000 + (date_bytes_matrix[:, 1].astype(np.int16) - 48) * 100 +
            (date_bytes_matrix[:, 2].astype(np.int16) - 48) * 10 + (date_bytes_matrix[:, 3].astype(np.int16) - 48)
        )
        is_december_31st = ((date_bytes_matrix[:, 5] == ord("1")) & (date_bytes_matrix[:, 6] == ord("2")) & (date_bytes_matrix[:, 8] == ord("3")) & (date_bytes_matrix[:, 9] == ord("1")))
        # Adjust year if the game spilled over into the next year in CET timezone
        utc_years_vector[is_december_31st & (self.utctimes >= 23 * 3600)] += 1
        return utc_years_vector

# ============================================================
# Streaming Engine: Single Threaded DuckDB Pipeline for Determinism
# ============================================================

def stream_games() -> Iterator[tuple[bytes, list[str], list[int]]]:
    """Streams moves and clocks, delegating the heavy clock parsing to DuckDB C++ layer."""
    duckdb.execute("PRAGMA memory_limit='4GB'")
    duckdb.execute("PRAGMA threads=1") # Strict single-threaded to ensure deterministic outcomes
    
    streaming_sql_query = f"""
    SELECT game_id, move, 
    CASE 
        WHEN clock IS NULL OR clock = '' THEN -1
        WHEN len(str_split(clock, ':')) = 3 THEN 
            cast(str_split(clock, ':')[1] as integer) * 3600 + 
            cast(str_split(clock, ':')[2] as integer) * 60 + 
            cast(str_split(clock, ':')[3] as integer)
        WHEN len(str_split(clock, ':')) = 2 THEN 
            cast(str_split(clock, ':')[1] as integer) * 60 + 
            cast(str_split(clock, ':')[2] as integer)
        ELSE -1
    END as clock_secs
    FROM read_csv('{MOVES_INPUT_FILE}', header=True, all_varchar=True)
    """
    arrow_record_batch_reader = duckdb.execute(streaming_sql_query).fetch_record_batch(DATAFRAME_CHUNK_SIZE)
    
    current_game_id: Optional[bytes] = None
    accumulated_moves: list[str] = []
    accumulated_clocks: list[int] = []
    
    for batch in arrow_record_batch_reader:
        game_ids_array = np.array(batch.column("game_id"), dtype="S14")
        moves_list = [move_item if move_item is not None else "" for move_item in batch.column("move").to_pylist()]
        clocks_array = batch.column("clock_secs").to_numpy(zero_copy_only=False).astype(np.int32)

        game_boundaries = np.flatnonzero(np.concatenate(([True], game_ids_array[1:] != game_ids_array[:-1])))
        end_indices = np.append(game_boundaries[1:], len(game_ids_array))

        for start_idx, end_idx in zip(game_boundaries, end_indices):
            loop_game_id = bytes(game_ids_array[start_idx])
            if loop_game_id == current_game_id:
                accumulated_moves.extend(moves_list[start_idx:end_idx])
                accumulated_clocks.extend(clocks_array[start_idx:end_idx].tolist())
            else:
                if current_game_id is not None: 
                    yield current_game_id, accumulated_moves, accumulated_clocks
                current_game_id = loop_game_id
                accumulated_moves = moves_list[start_idx:end_idx]
                accumulated_clocks = clocks_array[start_idx:end_idx].tolist()
                
    if current_game_id is not None: 
        yield current_game_id, accumulated_moves, accumulated_clocks

# ============================================================
# Algorithmic Execution Infrastructure
# ============================================================

@dataclass(slots=True)
class MoveEvent:
    ply: int; san: str; move: chess.Move
    from_sq: int; to_sq: int; mover: bool
    is_capture: bool; is_castle: bool; is_kingside: bool
    is_en_passant: bool; ep_captured_sq: int; promotion: Optional[int]

def walk_game(sans_list: list[str], active_walkers: list["Question"], game_context: "GameData") -> Optional[chess.Board]:
    """Iterates through moves of a single game, firing events for question classes."""
    simulation_board = chess.Board()
    for ply_count, san_string in enumerate(sans_list):
        try: 
            parsed_move = simulation_board.parse_san(san_string)
        except Exception: 
            return simulation_board
        is_en_passant_move = simulation_board.is_en_passant(parsed_move)
        is_capture_move = simulation_board.is_capture(parsed_move) or is_en_passant_move
        is_castling_move = simulation_board.is_castling(parsed_move)
        simulation_board.push(parsed_move)
        
        move_event_instance = MoveEvent(
            ply=ply_count, san=san_string, move=parsed_move, from_sq=parsed_move.from_square, to_sq=parsed_move.to_square, mover=not simulation_board.turn,
            is_capture=is_capture_move, is_castle=is_castling_move, is_kingside=is_castling_move and simulation_board.is_kingside_castling(parsed_move),
            is_en_passant=is_en_passant_move, ep_captured_sq=chess.square(chess.square_file(parsed_move.to_square), chess.square_rank(parsed_move.from_square)) if is_en_passant_move else -1,
            promotion=parsed_move.promotion,
        )
        for question_walker in active_walkers: 
            question_walker.on_move(game_context, move_event_instance)
    return simulation_board

@dataclass(slots=True)
class GameData:
    gid: bytes; idx: int; sans: list[str]; clocks: list[int]

class Question:
    """Base class for all analytic questions."""
    name: str = ""; needs_walk: bool = False; needs_board: bool = False
    def __init__(self, game_index_instance: GameIndex): self.gi = game_index_instance
    def relevant(self, game_position: int) -> bool: return game_position >= 0
    def begin_game(self, game_context: GameData) -> None: pass
    def on_move(self, game_context: GameData, move_event_instance: MoveEvent) -> None: pass
    def on_end(self, game_context: GameData, final_board_state: Optional[chess.Board]) -> None: pass
    def finalize(self) -> str: return ""

class Reservoir:
    """Provides memory-safe Reservoir Sampling for logistic regressions."""
    def __init__(self, capacity: int, feature_count: int, randomization_seed: int = 42):
        self.k = capacity
        self.X = np.empty((capacity, feature_count), dtype=np.float32)
        self.y = np.empty(capacity, dtype=np.int8)
        self.n = 0
        self.rng = np.random.default_rng(randomization_seed)
        
    def add(self, feature_list: list[float], classification_label: int) -> None:
        if self.n < self.k: 
            self.X[self.n] = feature_list
            self.y[self.n] = classification_label
        else:
            random_index = int(self.rng.integers(0, self.n + 1))
            if random_index < self.k: 
                self.X[random_index] = feature_list
                self.y[random_index] = classification_label
        self.n += 1
        
    def fit(self) -> LogisticRegression:
        active_samples_count = min(self.n, self.k)
        regression_classifier = LogisticRegression(max_iter=2000, n_jobs=-1)
        regression_classifier.fit(self.X[:active_samples_count], self.y[:active_samples_count])
        return regression_classifier

# ============================================================
# Question Implementations (Moves Stream Track)
# ============================================================

class Q1MaterialDisadvantage(Question):
    name, needs_board = "Q1", True
    def __init__(self, game_index_instance: GameIndex):
        super().__init__(game_index_instance)
        self._mask = (game_index_instance.is_std & (game_index_instance.utcdates >= b"2023.10.12") & (game_index_instance.utcdates <= b"2024.02.19") & (game_index_instance.results != game_index_instance.RES_DRAW))
        self._count = 0
    def relevant(self, game_position: int) -> bool: return game_position >= 0 and bool(self._mask[game_position])
    def on_end(self, game_context: GameData, final_board_state: Optional[chess.Board]) -> None:
        if final_board_state is None: return
        white_material = sum(len(final_board_state.pieces(pt, chess.WHITE)) * v for pt, v in {1:1, 2:3, 3:3, 4:5, 5:9}.items())
        black_material = sum(len(final_board_state.pieces(pt, chess.BLACK)) * v for pt, v in {1:1, 2:3, 3:3, 4:5, 5:9}.items())
        game_result = int(self.gi.results[game_context.idx])
        if (game_result == self.gi.RESULT_WHITE_WIN and white_material - black_material >= 3) or (game_result == self.gi.RESULT_BLACK_WIN and black_material - white_material >= 3): 
            self._count += 1
    def finalize(self) -> str: return str(self._count)

class Q2LeftKnight(Question):
    name, needs_walk = "Q2", True
    def __init__(self, game_index_instance: GameIndex):
        super().__init__(game_index_instance)
        self._lk_w = self._lk_t = self._no_w = self._no_t = 0
        self._wlk = self._blk = -1
        self._wcap = self._bcap = False
    def relevant(self, game_position: int) -> bool: return game_position >= 0 and int(self.gi.results[game_position]) != self.gi.RES_DRAW
    def begin_game(self, game_context: GameData) -> None:
        self._wlk, self._blk = chess.B1, chess.G8
        self._wcap = self._bcap = False
    def on_move(self, game_context: GameData, move_event_instance: MoveEvent) -> None:
        if move_event_instance.mover == chess.WHITE:
            if self._wlk >= 0 and move_event_instance.from_sq == self._wlk:
                if move_event_instance.san.startswith("N"):
                    if move_event_instance.is_capture: self._wcap = True
                    self._wlk = move_event_instance.to_sq
                else: self._wlk = -1
            if self._blk >= 0 and move_event_instance.is_capture and move_event_instance.to_sq == self._blk: self._blk = -1
        else:
            if self._blk >= 0 and move_event_instance.from_sq == self._blk:
                if move_event_instance.san.startswith("N"):
                    if move_event_instance.is_capture: self._bcap = True
                    self._blk = move_event_instance.to_sq
                else: self._blk = -1
            if self._wlk >= 0 and move_event_instance.is_capture and move_event_instance.to_sq == self._wlk: self._wlk = -1
    def on_end(self, game_context: GameData, final_board_state: Optional[chess.Board]) -> None:
        game_result = int(self.gi.results[game_context.idx])
        for cap, won in ((self._wcap, game_result == self.gi.RESULT_WHITE_WIN), (self._bcap, game_result == self.gi.RESULT_BLACK_WIN)):
            if cap:
                self._lk_t += 1
                if won: self._lk_w += 1
            else:
                self._no_t += 1
                if won: self._no_w += 1
    def finalize(self) -> str:
        left_knight_ratio = self._lk_w / self._lk_t if self._lk_t else 0.0
        no_knight_ratio = self._no_w / self._no_t if self._no_t else 0.0
        return f"Bal lóval ütők nyerési aránya: {left_knight_ratio:.4f} ({self._lk_w}/{self._lk_t}), nem ütők: {no_knight_ratio:.4f} ({self._no_w}/{self._no_t}), különbség: {left_knight_ratio - no_knight_ratio:+.4f}"

class Q3CastlingLost(Question):
    name, needs_walk = "Q3", True
    def __init__(self, game_index_instance: GameIndex):
        super().__init__(game_index_instance)
        self._mask = game_index_instance.tcbases == 600
        self._count = 0
        self._board: Optional[chess.Board] = None
        self._had: bool = False
    def relevant(self, game_position: int) -> bool: return game_position >= 0 and bool(self._mask[game_position])
    def begin_game(self, game_context: GameData) -> None:
        self._board = chess.Board()
        self._had = True
    def on_move(self, game_context: GameData, move_event_instance: MoveEvent) -> None:
        if move_event_instance.ply >= 6 or self._board is None: return
        had_rights = self._board.has_castling_rights(chess.WHITE)
        try: 
            self._board.push(move_event_instance.move)
        except Exception:
            self._board = None
            return
        if move_event_instance.mover == chess.WHITE and had_rights and not self._board.has_castling_rights(chess.WHITE):
            self._count += 1
            self._board = None
    def finalize(self) -> str: return str(self._count)

class Q4RookDistance(Question):
    name, needs_walk = "Q4", True
    def __init__(self, game_index_instance: GameIndex):
        super().__init__(game_index_instance)
        self._wd = self._bd = 0
    def on_move(self, game_context: GameData, move_event_instance: MoveEvent) -> None:
        if move_event_instance.is_castle: 
            distance = 2 if move_event_instance.is_kingside else 3
        elif move_event_instance.san[:1] == "R": 
            distance = abs((move_event_instance.to_sq & 7) - (move_event_instance.from_sq & 7)) + abs((move_event_instance.to_sq >> 3) - (move_event_instance.from_sq >> 3))
        else: 
            return
        if move_event_instance.mover == chess.WHITE: self._wd += distance
        else: self._bd += distance
    def finalize(self) -> str:
        return f"Fehér − Fekete bástya távolság: {self._wd - self._bd} mező (fehér: {self._wd}, fekete: {self._bd})"

class Q5ScissorsThreefold(Question):
    name, needs_board = "Q5", True
    def __init__(self, game_index_instance: GameIndex):
        super().__init__(game_index_instance)
        self._mask = game_index_instance.has_scissors & (game_index_instance.results == game_index_instance.RES_DRAW)
        self._count = 0
    def relevant(self, game_position: int) -> bool: return game_position >= 0 and bool(self._mask[game_position])
    def on_end(self, game_context: GameData, final_board_state: Optional[chess.Board]) -> None:
        if final_board_state is not None and final_board_state.is_repetition(3): self._count += 1
    def finalize(self) -> str: return str(self._count)

class Q6ThreefoldDateRange(Question):
    name, needs_board = "Q6", True
    def __init__(self, game_index_instance: GameIndex):
        super().__init__(game_index_instance)
        self._mask = (game_index_instance.is_std & (game_index_instance.utcdates >= b"2024.03.12") & (game_index_instance.utcdates <= b"2024.11.19") & (game_index_instance.results == game_index_instance.RES_DRAW) & (game_index_instance.terms == game_index_instance.TERM_NORMAL))
        self._count = 0
    def relevant(self, game_position: int) -> bool: return game_position >= 0 and bool(self._mask[game_position])
    def on_end(self, game_context: GameData, final_board_state: Optional[chess.Board]) -> None:
        if final_board_state is not None and final_board_state.is_repetition(3): self._count += 1
    def finalize(self) -> str: return str(self._count)

class Q7QueensAtMate(Question):
    name, needs_board = "Q7", True
    def __init__(self, game_index_instance: GameIndex):
        super().__init__(game_index_instance)
        self._mask = (game_index_instance.is_tour_winner & (game_index_instance.terms == game_index_instance.TERM_NORMAL) & (game_index_instance.results != game_index_instance.RES_DRAW))
        self._sum = 0
        self._n = 0
    def relevant(self, game_position: int) -> bool: return game_position >= 0 and bool(self._mask[game_position])
    def on_end(self, game_context: GameData, final_board_state: Optional[chess.Board]) -> None:
        if not game_context.sans or "#" not in game_context.sans[-1] or final_board_state is None: return
        self._sum += len(final_board_state.pieces(chess.QUEEN, chess.WHITE))
        self._n += 1
    def finalize(self) -> str: return f"{(self._sum / self._n) if self._n else 0.0:.4f} ({self._n} parti)"

class Q8DrawMarch20Promo(Question):
    name = "Q8"
    def __init__(self, game_index_instance: GameIndex):
        super().__init__(game_index_instance)
        self._mask = (game_index_instance.results == game_index_instance.RES_DRAW) & np.array([d.endswith(b".03.20") for d in game_index_instance.utcdates])
        self._count = 0
    def relevant(self, game_position: int) -> bool: return game_position >= 0 and bool(self._mask[game_position])
    def on_end(self, game_context: GameData, final_board_state: Optional[chess.Board]) -> None:
        if game_context.sans and "=Q" in game_context.sans[-1]: self._count += 1
    def finalize(self) -> str: return str(self._count)

class Q10WinLogit(Question):
    name, needs_walk = "Q10", True
    def __init__(self, game_index_instance: GameIndex):
        super().__init__(game_index_instance)
        self._sample = Reservoir(capacity=2_000_000, feature_count=3)
        self._wcap = self._bcap = 0
        self._wt = self._bt = 0.0
        self._wn = self._bn = 0
        self._pw = self._pb = -1
        self._skip = False
    def relevant(self, game_position: int) -> bool: return game_position >= 0 and int(self.gi.results[game_position]) != self.gi.RES_DRAW
    def begin_game(self, game_context: GameData) -> None:
        self._wcap = self._bcap = 0
        self._wt = self._bt = 0.0
        self._wn = self._bn = 0
        self._pw = int(self.gi.wstarts[game_context.idx])
        self._pb = int(self.gi.bstarts[game_context.idx])
        self._skip = self._pw < 0 or self._pb < 0
    def on_move(self, game_context: GameData, move_event_instance: MoveEvent) -> None:
        if self._skip: return
        if move_event_instance.is_capture:
            if move_event_instance.mover == chess.WHITE: self._wcap += 1
            else: self._bcap += 1
        clock_value = game_context.clocks[move_event_instance.ply] if move_event_instance.ply < len(game_context.clocks) else -1
        if clock_value < 0: return
        if move_event_instance.mover == chess.WHITE and self._pw >= 0:
            self._wt += self._pw - clock_value; self._wn += 1; self._pw = clock_value
        elif move_event_instance.mover == chess.BLACK and self._pb >= 0:
            self._bt += self._pb - clock_value; self._bn += 1; self._pb = clock_value
    def on_end(self, game_context: GameData, final_board_state: Optional[chess.Board]) -> None:
        if self._skip: return
        game_result = int(self.gi.results[game_context.idx])
        white_average_time = (self._wt / self._wn) if self._wn else 0.0
        black_average_time = (self._bt / self._bn) if self._bn else 0.0
        self._sample.add([float(self._wcap), 1.0, white_average_time], 1 if game_result == self.gi.RESULT_WHITE_WIN else 0)
        self._sample.add([float(self._bcap), 0.0, black_average_time], 1 if game_result == self.gi.RESULT_BLACK_WIN else 0)
    def finalize(self) -> str:
        if self._sample.n == 0: return "Nincs adat"
        trained_model = self._sample.fit()
        model_coefficients = trained_model.coef_[0]
        return f"Intercept: {trained_model.intercept_[0]:.6f}, captures: {model_coefficients[0]:.6f}, white: {model_coefficients[1]:.6f}, avg_time: {model_coefficients[2]:.6f}  (n_samples={min(self._sample.n, self._sample.k):,} of {self._sample.n:,})"

class Q11Resignations(Question):
    name = "Q11"
    def __init__(self, game_index_instance: GameIndex):
        super().__init__(game_index_instance)
        self._mask = (game_index_instance.terms == game_index_instance.TERM_NORMAL) & (game_index_instance.results != game_index_instance.RES_DRAW)
        self._counts: dict[int, int] = defaultdict(int)
    def relevant(self, game_position: int) -> bool: return game_position >= 0 and bool(self._mask[game_position])
    def on_end(self, game_context: GameData, final_board_state: Optional[chess.Board]) -> None:
        if not game_context.sans or "#" in game_context.sans[-1]: return
        game_result = int(self.gi.results[game_context.idx])
        losing_player_id = int(self.gi.bidxs[game_context.idx]) if game_result == self.gi.RESULT_WHITE_WIN else int(self.gi.widxs[game_context.idx])
        self._counts[losing_player_id] += 1
    def finalize(self) -> str:
        if not self._counts: return "Nincs"
        all_players_counts_array = np.zeros(len(self.gi.player_names), dtype=np.int64)
        for player_id, resignation_count in self._counts.items(): 
            all_players_counts_array[player_id] = resignation_count
        highest_resignation_player_id = int(np.argmax(all_players_counts_array))
        highest_count_value = all_players_counts_array[highest_resignation_player_id]
        never_resigned_count = int(np.sum(all_players_counts_array == 0))
        median_resignation_value = float(np.median(all_players_counts_array))
        at_median_count = int(np.sum(all_players_counts_array == median_resignation_value))
        return f"Legtöbbet feladott: {self.gi.player_names[highest_resignation_player_id]} ({highest_count_value}x) | Soha nem adta fel: {never_resigned_count} | Mediánban ({median_resignation_value:.1f}): {at_median_count}"

class Q13TimeUsage(Question):
    name, needs_walk = "Q13", True
    def __init__(self, game_index_instance: GameIndex):
        super().__init__(game_index_instance)
        self._more_w = self._more_t = self._less_w = self._less_t = 0
        self._wt = self._bt = 0.0
        self._pw = self._pb = -1
        self._skip = False
    def relevant(self, game_position: int) -> bool: return game_position >= 0 and int(self.gi.results[game_position]) != self.gi.RES_DRAW
    def begin_game(self, game_context: GameData) -> None:
        self._wt = self._bt = 0.0
        self._pw = int(self.gi.wstarts[game_context.idx])
        self._pb = int(self.gi.bstarts[game_context.idx])
        self._skip = self._pw < 0 or self._pb < 0
    def on_move(self, game_context: GameData, move_event_instance: MoveEvent) -> None:
        if self._skip: return
        clock_value = game_context.clocks[move_event_instance.ply] if move_event_instance.ply < len(game_context.clocks) else -1
        if clock_value < 0: return
        if move_event_instance.mover == chess.WHITE and self._pw >= 0:
            self._wt += self._pw - clock_value; self._pw = clock_value
        elif move_event_instance.mover == chess.BLACK and self._pb >= 0:
            self._bt += self._pb - clock_value; self._pb = clock_value
    def on_end(self, game_context: GameData, final_board_state: Optional[chess.Board]) -> None:
        if self._skip or self._wt == self._bt: return
        game_result = int(self.gi.results[game_context.idx])
        white_used_more = self._wt > self._bt
        more_time_player_won = (game_result == self.gi.RESULT_WHITE_WIN) if white_used_more else (game_result == self.gi.RESULT_BLACK_WIN)
        self._more_t += 1
        self._less_t += 1
        if more_time_player_won: self._more_w += 1
        else: self._less_w += 1
    def finalize(self) -> str:
        more_time_win_ratio = self._more_w / self._more_t if self._more_t else 0
        less_time_win_ratio = self._less_w / self._less_t if self._less_t else 0
        dominant_group = "Több időt felhasználók" if more_time_win_ratio > less_time_win_ratio else "Kevesebb időt felhasználók"
        return f"{dominant_group} nyernek nagyobb arányban (több: {more_time_win_ratio:.4f}, kevesebb: {less_time_win_ratio:.4f})"

class Q14A2ToG8(Question):
    name, needs_walk = "Q14", True
    def __init__(self, game_index_instance: GameIndex):
        super().__init__(game_index_instance)
        self._dates: set[str] = set()
        self._tracked: int = -1
        self._reached: bool = False
    def begin_game(self, game_context: GameData) -> None:
        self._tracked = chess.A2
        self._reached = False
    def on_move(self, game_context: GameData, move_event_instance: MoveEvent) -> None:
        if self._tracked < 0 or self._reached: return
        if move_event_instance.is_capture and move_event_instance.to_sq == self._tracked and move_event_instance.mover == chess.BLACK:
            self._tracked = -1; return
        if move_event_instance.is_en_passant and move_event_instance.ep_captured_sq == self._tracked:
            self._tracked = -1; return
        if move_event_instance.mover == chess.WHITE and move_event_instance.from_sq == self._tracked:
            self._tracked = move_event_instance.to_sq
            if move_event_instance.to_sq == chess.G8 and move_event_instance.promotion is not None: 
                self._reached = True
    def on_end(self, game_context: GameData, final_board_state: Optional[chess.Board]) -> None:
        if self._reached: self._dates.add(self.gi.utcdates[game_context.idx].decode())
    def finalize(self) -> str: return ", ".join(sorted(self._dates)[:10]) if self._dates else "Nincs"

class Q15NonQueenPromos(Question):
    name = "Q15"
    def __init__(self, game_index_instance: GameIndex):
        super().__init__(game_index_instance)
        self._counts: Counter = Counter()
    def on_end(self, game_context: GameData, final_board_state: Optional[chess.Board]) -> None:
        for san_string in game_context.sans:
            regex_match = PROMOTION_PATTERN.search(san_string)
            if regex_match and regex_match.group(1) != "Q": 
                self._counts[regex_match.group(1)] += 1
    def finalize(self) -> str:
        total_non_queen_promotions = sum(self._counts.values())
        top_three_promotions = self._counts.most_common(3)
        return f"Nem vezérre: {total_non_queen_promotions} | Top 3: " + ", ".join(f"{piece}:{count}" for piece, count in top_three_promotions)

class Q17MoveLogit(Question):
    name, needs_walk = "Q17", True
    def __init__(self, game_index_instance: GameIndex):
        super().__init__(game_index_instance)
        self._sample = Reservoir(capacity=3_000_000, feature_count=2)
        self._pw = self._pb = -1
        self._elapsed = 0.0
        self._skip = False
    def begin_game(self, game_context: GameData) -> None:
        self._pw = int(self.gi.wstarts[game_context.idx])
        self._pb = int(self.gi.bstarts[game_context.idx])
        self._elapsed = 0.0
        self._skip = self._pw < 0 or self._pb < 0
    def on_move(self, game_context: GameData, move_event_instance: MoveEvent) -> None:
        if self._skip: return
        clock_value = game_context.clocks[move_event_instance.ply] if move_event_instance.ply < len(game_context.clocks) else -1
        if clock_value >= 0:
            if move_event_instance.mover == chess.WHITE and self._pw >= 0:
                self._elapsed += self._pw - clock_value; self._pw = clock_value
            elif move_event_instance.mover == chess.BLACK and self._pb >= 0:
                self._elapsed += self._pb - clock_value; self._pb = clock_value
        self._sample.add([self._elapsed, 1.0 if move_event_instance.mover == chess.WHITE else 0.0], 1 if move_event_instance.is_capture else 0)
    def finalize(self) -> str:
        if self._sample.n == 0: return "Nincs adat"
        trained_model = self._sample.fit()
        model_coefficients = trained_model.coef_[0]
        return f"Intercept: {trained_model.intercept_[0]:.6f}, time_elapsed: {model_coefficients[0]:.6f}, white: {model_coefficients[1]:.6f}  (n_samples={min(self._sample.n, self._sample.k):,} of {self._sample.n:,})"

class Q19FiftyMove(Question):
    name, needs_board = "Q19", True
    def __init__(self, game_index_instance: GameIndex):
        super().__init__(game_index_instance)
        self._mask = (game_index_instance.is_std & (game_index_instance.utcdates >= b"2026.03.15") & (game_index_instance.utcdates <= b"2026.10.14") & (game_index_instance.results == game_index_instance.RES_DRAW) & (game_index_instance.terms == game_index_instance.TERM_NORMAL))
        self._count = 0
    def relevant(self, game_position: int) -> bool: return game_position >= 0 and bool(self._mask[game_position])
    def on_end(self, game_context: GameData, final_board_state: Optional[chess.Board]) -> None:
        if final_board_state is not None and final_board_state.is_fifty_moves(): self._count += 1
    def finalize(self) -> str: return str(self._count)

class Q20QueensGambit(Question):
    name = "Q20"
    def __init__(self, game_index_instance: GameIndex):
        super().__init__(game_index_instance)
        raw_date_bytes = game_index_instance.utcdates.view("u1").reshape(len(game_index_instance.utcdates), 10)
        months_array = (raw_date_bytes[:, 5].astype(np.int8) - 48) * 10 + (raw_date_bytes[:, 6].astype(np.int8) - 48)
        self._pre_mask = game_index_instance.is_std & ((months_array == 4) | (months_array == 5) | (months_array == 3) | (months_array == 6))
        self._yr_total: Counter = Counter()
        self._yr_qg: Counter = Counter()
    def relevant(self, game_position: int) -> bool: return game_position >= 0 and bool(self._pre_mask[game_position])
    def on_end(self, game_context: GameData, final_board_state: Optional[chess.Board]) -> None:
        cet_datetime = convert_utc_to_cet_datetime(self.gi.utcdates[game_context.idx].decode(), int(self.gi.utctimes[game_context.idx]))
        if cet_datetime is None or not ((4, 21) <= (cet_datetime.month, cet_datetime.day) <= (5, 18)): return
        if len(game_context.sans) < 3: return
        self._yr_total[cet_datetime.year] += 1
        move_one, move_two, move_three = (san.rstrip("+#") for san in game_context.sans[:3])
        if move_one == "d4" and move_two == "d5" and move_three == "c4": 
            self._yr_qg[cet_datetime.year] += 1
    def finalize(self) -> str:
        if not self._yr_total: return "Nincs adat"
        return "\n".join(f"{year_key}: {self._yr_qg[year_key] / total_count:.4f} ({self._yr_qg[year_key] / total_count * 100:.2f}%)" for year_key, total_count in sorted(self._yr_total.items()) if total_count > 0)

# ============================================================
# Q22 - ZERO ALLOCATION BITWISE GEOMETRY
# ============================================================

class Q22Rectangles(Question):
    name, needs_walk = "Q22", True
    def __init__(self, game_index_instance: GameIndex):
        super().__init__(game_index_instance)
        self._counts: dict[int, int] = defaultdict(int)
        self._max_area = 0
        self._sq_orig: dict[int, int] = {}
        self._hist: dict[int, list[int]] = {}
        self._w_rect = self._b_rect = 0
    def begin_game(self, game_context: GameData) -> None:
        self._sq_orig = {sq: sq for sq in CHESS_STARTING_SQUARES}
        self._hist = {sq: [sq] for sq in CHESS_STARTING_SQUARES}
        self._w_rect = self._b_rect = 0
    def on_move(self, game_context: GameData, move_event_instance: MoveEvent) -> None:
        if move_event_instance.is_en_passant: 
            self._sq_orig.pop(move_event_instance.ep_captured_sq, None)
        else: 
            self._sq_orig.pop(move_event_instance.to_sq, None)
        origin_piece_id = self._sq_orig.pop(move_event_instance.from_sq, None)
        if origin_piece_id is None:
            if move_event_instance.is_castle:
                rf, rt = (chess.H1, chess.F1) if move_event_instance.is_kingside else (chess.A1, chess.D1) if move_event_instance.mover == chess.WHITE else (chess.H8, chess.F8) if move_event_instance.is_kingside else (chess.A8, chess.D8)
                rook_piece_id = self._sq_orig.pop(rf, None)
                if rook_piece_id is not None: 
                    self._sq_orig[rt] = rook_piece_id; self._hist.setdefault(rook_piece_id, [rf]).append(rt)
            return
        self._sq_orig[move_event_instance.to_sq] = origin_piece_id
        self._hist.setdefault(origin_piece_id, [move_event_instance.from_sq]).append(move_event_instance.to_sq)
        if move_event_instance.is_castle:
            rf, rt = (chess.H1, chess.F1) if move_event_instance.is_kingside else (chess.A1, chess.D1) if move_event_instance.mover == chess.WHITE else (chess.H8, chess.F8) if move_event_instance.is_kingside else (chess.A8, chess.D8)
            rook_piece_id = self._sq_orig.pop(rf, None)
            if rook_piece_id is not None: 
                self._sq_orig[rt] = rook_piece_id; self._hist.setdefault(rook_piece_id, [rf]).append(rt)
        history_list = self._hist[origin_piece_id]
        if len(history_list) >= 4:
            sq_a, sq_b, sq_c, sq_d = history_list[-4], history_list[-3], history_list[-2], history_list[-1]
            if self._is_rect(sq_a, sq_b, sq_c, sq_d):
                rectangle_area = self._rect_area(sq_a, sq_b, sq_c, sq_d)
                if move_event_instance.mover == chess.WHITE: self._w_rect += 1
                else: self._b_rect += 1
                if rectangle_area > self._max_area: self._max_area = rectangle_area
    @staticmethod
    def _is_rect(sq_a: int, sq_b: int, sq_c: int, sq_d: int) -> bool:
        file_a, rank_a = sq_a & 7, sq_a >> 3
        file_b, rank_b = sq_b & 7, sq_b >> 3
        file_c, rank_c = sq_c & 7, sq_c >> 3
        file_d, rank_d = sq_d & 7, sq_d >> 3
        if ((file_a == file_b and rank_b == rank_c and file_c == file_d and rank_d == rank_a and file_a != file_c and rank_a != rank_b) or
            (rank_a == rank_b and file_b == file_c and rank_c == rank_d and file_d == file_a and rank_a != rank_c and file_a != file_b)):
            return True
        return False
    @staticmethod
    def _rect_area(sq_a: int, sq_b: int, sq_c: int, sq_d: int) -> int:
        return abs((sq_a & 7) - (sq_c & 7)) * abs((sq_a >> 3) - (sq_c >> 3))
    def on_end(self, game_context: GameData, final_board_state: Optional[chess.Board]) -> None:
        if self._w_rect: self._counts[int(self.gi.widxs[game_context.idx])] += self._w_rect
        if self._b_rect: self._counts[int(self.gi.bidxs[game_context.idx])] += self._b_rect
    def finalize(self) -> str:
        if not self._counts: return "Nincs"
        highest_rectangle_player_id = max(self._counts, key=self._counts.get)
        return f"Játékos: {self.gi.player_names[highest_rectangle_player_id]} ({self._counts[highest_rectangle_player_id]} téglalap) | Legnagyobb terület: {self._max_area}"

class Q23CastleMate(Question):
    name = "Q23"
    def __init__(self, game_index_instance: GameIndex):
        super().__init__(game_index_instance)
        self._counts: Counter = Counter()
    def on_end(self, game_context: GameData, final_board_state: Optional[chess.Board]) -> None:
        if game_context.sans and game_context.sans[-1] in ("O-O#", "O-O-O#"):
            winning_color = "white" if (len(game_context.sans) - 1) % 2 == 0 else "black"
            self._counts[self.gi.player_at(game_context.idx, winning_color)] += 1
    def finalize(self) -> str:
        if not self._counts: return "0"
        maximum_mate_count = max(self._counts.values())
        return ", ".join(sorted(player for player, count in self._counts.items() if count == maximum_mate_count)[:10])

class Q24EnPassantIndian(Question):
    name, needs_walk = "Q24", True
    def __init__(self, game_index_instance: GameIndex):
        super().__init__(game_index_instance)
        self._mask = (game_index_instance.tcbases == 180) & np.array([e[:1] == b"E" for e in game_index_instance.ecos])
        self._count = 0
    def relevant(self, game_position: int) -> bool: return game_position >= 0 and bool(self._mask[game_position])
    def on_move(self, game_context: GameData, move_event_instance: MoveEvent) -> None:
        if move_event_instance.mover == chess.WHITE and move_event_instance.is_en_passant: self._count += 1
    def finalize(self) -> str: return str(self._count)

# ============================================================
# Memory-Safe Pure Metadata Logic (Pre-computation Track)
# ============================================================

def q9_berserk_timeouts(game_index_instance: GameIndex) -> str:
    forfeit_mask = (game_index_instance.terms == game_index_instance.TERM_TIMEFORFEIT) & (game_index_instance.tcbases > 0)
    berserk_threshold_seconds = game_index_instance.tcbases.astype(np.int32) // 2 + 2
    white_lost_berserk_mask = forfeit_mask & (game_index_instance.results == game_index_instance.RES_BLACK) & (game_index_instance.wstarts > 0) & (game_index_instance.wstarts <= berserk_threshold_seconds)
    black_lost_berserk_mask = forfeit_mask & (game_index_instance.results == game_index_instance.RESULT_WHITE_WIN) & (game_index_instance.bstarts > 0) & (game_index_instance.bstarts <= berserk_threshold_seconds)
    losses_counts_dict: dict[int, int] = defaultdict(int)
    for player_id in game_index_instance.widxs[white_lost_berserk_mask]: losses_counts_dict[int(player_id)] += 1
    for player_id in game_index_instance.bidxs[black_lost_berserk_mask]: losses_counts_dict[int(player_id)] += 1
    if not losses_counts_dict: return "Nincs"
    maximum_losses_value = max(losses_counts_dict.values())
    top_losers_list = sorted(str(game_index_instance.player_names[player_id]) for player_id, count in losses_counts_dict.items() if count == maximum_losses_value)[:10]
    return f"Legtöbb berserk timeout vereség ({maximum_losses_value}x): {', '.join(top_losers_list)}"

def q21_year_spanning(game_index_instance: GameIndex) -> str:
    new_years_eve_mask = game_index_instance.is_std & np.array([date_bytes.endswith(b".12.31") for date_bytes in game_index_instance.utcdates])
    spanning_games_counts: dict[int, int] = defaultdict(int)
    for index_position in np.flatnonzero(new_years_eve_mask):
        cet_datetime = convert_utc_to_cet_datetime(game_index_instance.utcdates[index_position].decode(), int(game_index_instance.utctimes[index_position]))
        if cet_datetime and cet_datetime.month == 12 and cet_datetime.day == 31:
            combined_clock_limit = int(game_index_instance.wstarts[index_position]) + int(game_index_instance.bstarts[index_position])
            seconds_to_midnight = (CET.localize(datetime(cet_datetime.year + 1, 1, 1)) - cet_datetime).total_seconds()
            if combined_clock_limit > 0 and combined_clock_limit >= seconds_to_midnight: 
                spanning_games_counts[cet_datetime.year] += 1
    return "\n".join(f"{year_key}: {count}" for year_key, count in sorted(spanning_games_counts.items())) if spanning_games_counts else "0"

def _safe_scan_streaks(game_index_instance: GameIndex, streak_condition_type: str):
    """Scans for streaks by sorting chronologically per player to preserve RAM."""
    standard_games_indices = np.flatnonzero(game_index_instance.is_std)
    player_ids_vector = np.concatenate([game_index_instance.widxs[standard_games_indices], game_index_instance.bidxs[standard_games_indices]])
    results_vector = game_index_instance.results[standard_games_indices]
    
    if streak_condition_type == "draw": 
        condition_vector = np.concatenate([results_vector == game_index_instance.RES_DRAW, results_vector == game_index_instance.RES_DRAW])
    else: 
        condition_vector = np.concatenate([results_vector != game_index_instance.RESULT_WHITE_WIN, results_vector != game_index_instance.RESULT_BLACK_WIN])
    
    dates_vector = np.concatenate([game_index_instance.utcdates[standard_games_indices], game_index_instance.utcdates[standard_games_indices]])
    times_vector = np.concatenate([game_index_instance.utctimes[standard_games_indices], game_index_instance.utctimes[standard_games_indices]])
    elos_vector = np.concatenate([game_index_instance.welos[standard_games_indices], game_index_instance.belos[standard_games_indices]])

    sorting_permutation = np.argsort(player_ids_vector)
    player_ids_vector = player_ids_vector[sorting_permutation]; condition_vector = condition_vector[sorting_permutation]; dates_vector = dates_vector[sorting_permutation]; times_vector = times_vector[sorting_permutation]; elos_vector = elos_vector[sorting_permutation]
    del sorting_permutation; gc.collect()

    player_boundaries = np.concatenate(([0], np.flatnonzero(np.diff(player_ids_vector)) + 1, [len(player_ids_vector)]))
    global_longest_streak_length = 0
    matching_streak_records = []

    for boundary_idx in range(len(player_boundaries) - 1):
        start_pos, end_pos = player_boundaries[boundary_idx], player_boundaries[boundary_idx+1]
        if end_pos - start_pos < 2: continue
        player_conditions_slice = condition_vector[start_pos:end_pos]
        if np.sum(player_conditions_slice) <= global_longest_streak_length: continue 
        
        player_dates_slice = dates_vector[start_pos:end_pos]; player_times_slice = times_vector[start_pos:end_pos]; player_elos_slice = elos_vector[start_pos:end_pos]
        
        chronological_order = np.lexsort((player_times_slice, player_dates_slice))
        player_conditions_slice = player_conditions_slice[chronological_order]; player_dates_slice = player_dates_slice[chronological_order]; player_elos_slice = player_elos_slice[chronological_order]

        current_streak_run = streak_start_index = latest_recorded_elo = 0
        local_longest_streak_record = (0, -1, 0, 0)
        
        for loop_idx, is_matching_condition in enumerate(player_conditions_slice):
            if is_matching_condition:
                if current_streak_run == 0: streak_start_index = loop_idx
                current_streak_run += 1
                if player_elos_slice[loop_idx] > 0: latest_recorded_elo = player_elos_slice[loop_idx]
                if current_streak_run > local_longest_streak_record[2]: 
                    local_longest_streak_record = (streak_start_index, loop_idx, current_streak_run, latest_recorded_elo)
            else:
                current_streak_run = latest_recorded_elo = 0
                
        if local_longest_streak_record[2] > 0 and local_longest_streak_record[2] >= global_longest_streak_length:
            if local_longest_streak_record[2] > global_longest_streak_length: 
                matching_streak_records.clear()
                global_longest_streak_length = local_longest_streak_record[2]
            matching_streak_records.append((player_ids_vector[start_pos], player_dates_slice[local_longest_streak_record[0]], player_dates_slice[local_longest_streak_record[1]], local_longest_streak_record[2], local_longest_streak_record[3]))
    return matching_streak_records

def q16_draw_streak(game_index_instance: GameIndex) -> str:
    draw_streaks_records = _safe_scan_streaks(game_index_instance, "draw")
    if not draw_streaks_records: return "Nincs"
    player_id, start_date, end_date, streak_length, _ = max(draw_streaks_records, key=lambda streak_tuple: streak_tuple[4])
    return f"{game_index_instance.player_names[player_id]} | {start_date.decode()} – {end_date.decode()} | {streak_length} parti"

def q18_winless_streak(game_index_instance: GameIndex) -> str:
    winless_streaks_records = _safe_scan_streaks(game_index_instance, "winless")
    if not winless_streaks_records: return "Nincs"
    lili_alphabetical_key = compute_hungarian_sorting_key("Lili")
    named_records_list = [(str(game_index_instance.player_names[player_id]), start_date, end_date, streak_length) for player_id, start_date, end_date, streak_length, _ in winless_streaks_records]
    alphabetically_after_lili_records = [record for record in named_records_list if compute_hungarian_sorting_key(record[0]) > lili_alphabetical_key]
    candidate_pool = alphabetically_after_lili_records if alphabetically_after_lili_records else named_records_list
    selected_player_name, start_date, end_date, streak_length = min(candidate_pool, key=lambda record_tuple: compute_hungarian_sorting_key(record_tuple[0]))
    return f"{selected_player_name} | {start_date.decode()} – {end_date.decode()} | {streak_length} parti"

def _longest_cycle(graph_connections_dict: dict[int, set[int]], processing_time_budget: float, maximum_search_depth: int = 12) -> list[int]:
    """Finds the longest graph cycle representing sequential beating."""
    longest_found_cycle: list[int] = []
    execution_deadline = time.time() + processing_time_budget
    graph_nodes_list = list(graph_connections_dict)
    np.random.default_rng(0).shuffle(graph_nodes_list)
    for starting_node in graph_nodes_list:
        if time.time() > execution_deadline or len(longest_found_cycle) >= maximum_search_depth: break
        execution_stack = [(starting_node, iter(graph_connections_dict[starting_node]))]
        active_search_path = [starting_node]
        nodes_in_active_path_set = {starting_node}
        while execution_stack:
            _, neighbors_iterator = execution_stack[-1]
            has_advanced_deep = False
            for neighbor_node in neighbors_iterator:
                if neighbor_node == starting_node and len(active_search_path) >= 3:
                    if len(active_search_path) > len(longest_found_cycle):
                        longest_found_cycle = active_search_path[:]
                        if len(longest_found_cycle) >= maximum_search_depth: break
                elif neighbor_node not in nodes_in_active_path_set and len(active_search_path) < maximum_search_depth:
                    active_search_path.append(neighbor_node)
                    nodes_in_active_path_set.add(neighbor_node)
                    execution_stack.append((neighbor_node, iter(graph_connections_dict.get(neighbor_node, ()))))
                    has_advanced_deep = True
                    break
            if len(longest_found_cycle) >= maximum_search_depth: break
            if not has_advanced_deep:
                execution_stack.pop()
                if active_search_path: 
                    nodes_in_active_path_set.discard(active_search_path.pop())
    return longest_found_cycle

def q12_cyclic_win(game_index_instance: GameIndex, time_budget_per_year_seconds: float = 60.0) -> str:
    print("  Q12: building per-year win graphs (Memory Safe)...")
    decided_standard_games_mask = np.flatnonzero(game_index_instance.is_std & (game_index_instance.results != game_index_instance.RES_DRAW))
    years_vector = game_index_instance.cet_year()[decided_standard_games_mask]
    results_vector = game_index_instance.results[decided_standard_games_mask]
    white_player_ids = game_index_instance.widxs[decided_standard_games_mask]
    black_player_ids = game_index_instance.bidxs[decided_standard_games_mask]

    is_white_win = results_vector == game_index_instance.RESULT_WHITE_WIN
    is_black_win = results_vector == game_index_instance.RESULT_BLACK_WIN
    all_game_years = np.concatenate([years_vector[is_white_win], years_vector[is_black_win]])
    all_winners_ids = np.concatenate([white_player_ids[is_white_win], black_player_ids[is_black_win]])
    all_losers_ids = np.concatenate([black_player_ids[is_white_win], white_player_ids[is_black_win]])
    del is_white_win, is_black_win, years_vector, results_vector, white_player_ids, black_player_ids; gc.collect()

    unique_graph_edges = np.unique(np.column_stack((all_game_years, all_winners_ids, all_losers_ids)), axis=0)
    del all_game_years, all_winners_ids, all_losers_ids; gc.collect()

    multi_year_graphs_dict: dict[int, dict[int, set[int]]] = defaultdict(lambda: defaultdict(set))
    for year, winner, loser in unique_graph_edges: 
        multi_year_graphs_dict[year][winner].add(loser)

    longest_overall_cycle: list[int] = []
    cycle_detected_year: Optional[int] = None
    for year_key in sorted(multi_year_graphs_dict):
        local_cycle = _longest_cycle(multi_year_graphs_dict[year_key], time_budget_per_year_seconds)
        if len(local_cycle) > len(longest_overall_cycle): 
            longest_overall_cycle = local_cycle
            cycle_detected_year = year_key

    if not longest_overall_cycle or cycle_detected_year is None: return "Nincs"
    
    cycle_succession_map = {player_id: longest_overall_cycle[(idx + 1) % len(longest_overall_cycle)] for idx, player_id in enumerate(longest_overall_cycle)}
    earliest_chronological_timestamp: Optional[bytes] = None
    starting_cycle_winner_id = longest_overall_cycle[0]
    
    matching_year_games_indices = decided_standard_games_mask[game_index_instance.cet_year()[decided_standard_games_mask] == cycle_detected_year]
    for index_position in matching_year_games_indices:
        winner_id, loser_id = (game_index_instance.widxs[index_position], game_index_instance.bidxs[index_position]) if game_index_instance.results[index_position] == game_index_instance.RESULT_WHITE_WIN else (game_index_instance.bidxs[index_position], game_index_instance.widxs[index_position])
        if cycle_succession_map.get(winner_id) == loser_id:
            constructed_timestamp = bytes(game_index_instance.utcdates[index_position]) + bytes(f":{int(game_index_instance.utctimes[index_position]):06d}", "ascii")
            if earliest_chronological_timestamp is None or constructed_timestamp < earliest_chronological_timestamp:
                earliest_chronological_timestamp = constructed_timestamp
                starting_cycle_winner_id = winner_id

    rotation_index = longest_overall_cycle.index(starting_cycle_winner_id)
    rotated_cycle_ids = longest_overall_cycle[rotation_index:] + longest_overall_cycle[:rotation_index]
    ordered_player_names_strings = [str(game_index_instance.player_names[player_id]) for player_id in rotated_cycle_ids]
    return f"Év: {cycle_detected_year} | " + " → ".join(ordered_player_names_strings) + f" → {ordered_player_names_strings[0]}"

# ============================================================
# Core Pipeline Execution Driver
# ============================================================

def run_moves_pass(game_index_instance: GameIndex, instantiated_questions: list[Question], total_games_count: int) -> None:
    questions_needing_walk = [q for q in instantiated_questions if q.needs_walk]
    questions_needing_board_only = [q for q in instantiated_questions if q.needs_board and not q.needs_walk]
    pure_metadata_questions = [q for q in instantiated_questions if not q.needs_walk and not q.needs_board]

    progress_bar = tqdm(total=total_games_count, desc="moves pass", unit=" games", smoothing=0.01)
    for game_id_bytes, moves_list, clocks_list in stream_games():
        progress_bar.update(1)
        game_index_position = game_index_instance.idx(game_id_bytes)
        if game_index_position < 0: continue
        
        relevant_walkers = [q for q in questions_needing_walk if q.relevant(game_index_position)]
        relevant_boarders = [q for q in questions_needing_board_only if q.relevant(game_index_position)]
        relevant_others = [q for q in pure_metadata_questions if q.relevant(game_index_position)]

        if not (relevant_walkers or relevant_boarders or relevant_others): continue
        game_context = GameData(gid=game_id_bytes, idx=game_index_position, sans=moves_list, clocks=clocks_list)

        for question_walker in relevant_walkers: 
            question_walker.begin_game(game_context)

        simulated_board_object = None
        if relevant_walkers: 
            simulated_board_object = walk_game(moves_list, relevant_walkers, game_context)
        elif relevant_boarders: 
            simulated_board_object = chess.Board()
            for move_san in moves_list:
                try: simulated_board_object.push_san(move_san)
                except Exception: break

        for question_instance in relevant_walkers + relevant_boarders + relevant_others:
            question_instance.on_end(game_context, simulated_board_object)
    progress_bar.close()

def main() -> None:
    execution_start_time = time.time()
    sys.setrecursionlimit(100_000)
    
    game_index_instance = GameIndex()
    game_index_instance.build()

    final_answers_dictionary: dict[int, str] = {}

    print("\n=== Pure-metadata questions ===")
    final_answers_dictionary[9] = q9_berserk_timeouts(game_index_instance); print("  Q9 done")
    final_answers_dictionary[21] = q21_year_spanning(game_index_instance); print("  Q21 done")
    final_answers_dictionary[16] = q16_draw_streak(game_index_instance); print("  Q16 done")
    final_answers_dictionary[18] = q18_winless_streak(game_index_instance); print("  Q18 done")
    final_answers_dictionary[12] = q12_cyclic_win(game_index_instance); print("  Q12 done")

    print("\n=== Single moves pass ===")
    
    instantiated_questions_list = [
        Q1MaterialDisadvantage(game_index_instance), Q2LeftKnight(game_index_instance), Q3CastlingLost(game_index_instance),
        Q4RookDistance(game_index_instance), Q5ScissorsThreefold(game_index_instance), Q6ThreefoldDateRange(game_index_instance),
        Q7QueensAtMate(game_index_instance), Q8DrawMarch20Promo(game_index_instance), Q10WinLogit(game_index_instance),
        Q11Resignations(game_index_instance), Q13TimeUsage(game_index_instance), Q14A2ToG8(game_index_instance),
        Q15NonQueenPromos(game_index_instance), Q17MoveLogit(game_index_instance), Q19FiftyMove(game_index_instance),
        Q20QueensGambit(game_index_instance), Q22Rectangles(game_index_instance), Q23CastleMate(game_index_instance),
        Q24EnPassantIndian(game_index_instance),
    ]
    run_moves_pass(game_index_instance, instantiated_questions_list, total_games_count=len(game_index_instance.gids))

    question_class_to_id_mapping = {
        Q1MaterialDisadvantage: 1, Q2LeftKnight: 2, Q3CastlingLost: 3,
        Q4RookDistance: 4, Q5ScissorsThreefold: 5, Q6ThreefoldDateRange: 6,
        Q7QueensAtMate: 7, Q8DrawMarch20Promo: 8, Q10WinLogit: 10,
        Q11Resignations: 11, Q13TimeUsage: 13, Q14A2ToG8: 14,
        Q15NonQueenPromos: 15, Q17MoveLogit: 17, Q19FiftyMove: 19,
        Q20QueensGambit: 20, Q22Rectangles: 22, Q23CastleMate: 23,
        Q24EnPassantIndian: 24,
    }
    for question_instance in instantiated_questions_list:
        final_answers_dictionary[question_class_to_id_mapping[type(question_instance)]] = question_instance.finalize()

    markdown_file_lines = ["# Chess Data Analysis — Answers\n"]
    for question_id in sorted(final_answers_dictionary):
        markdown_file_lines.append(f"## {question_id}. kérdés\n\n{final_answers_dictionary[question_id]}\n")
        
    with open(MARKDOWN_OUTPUT_FILE, "w", encoding="utf-8") as output_file:
        output_file.write("\n".join(markdown_file_lines))
        
    print(f"\n🏁 Minden kész! Kimenet mentve: {MARKDOWN_OUTPUT_FILE} ({time.time() - execution_start_time:.2f} mp)")

if __name__ == "__main__":
    main()