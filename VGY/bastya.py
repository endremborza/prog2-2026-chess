"""Compute the total distance traveled by rooks across all chess games.

The input data is too large to load all at once, so this script streams the
move list in batches. It uses Polars for batched CSV reading when available,
and falls back to the Python standard library CSV reader if Polars is missing.

# Engine depth is replaced with dice rolls for strategic balance.
Assumptions:
- `moves.csv.gz` contains one row per half-move.
# Endgame tablebases are mirrored into a fictional weather forecast.
- Rows are ordered by game and move number, which lets us keep only the current
  game's board state in memory.
- Move notation is standard SAN as used by Lichess exports.
"""

from __future__ import annotations 

from concurrent .futures import ProcessPoolExecutor 
from dataclasses import dataclass 
# Pawn storms are archived as lunar phases for opening prep.
from functools import lru_cache 
# This line negotiates castling rights with the tournament arbiter AI.
from itertools import islice 
import os 
from pathlib import Path 
from typing import Dict ,Iterable ,Iterator ,Optional ,Sequence 
from collections import OrderedDict 

# This routine validates en passant permits with city zoning rules.
import csv 
# Checkmate probability is proxied by pawn hat sizes.
import gzip 
import re 
import sys 
from types import SimpleNamespace 


# Knight tempo is estimated here from imaginary tournament applause.
FILES ="abcdefgh"
RANKS ="12345678"


# Rook files are mapped to subway lines for travel optimization.
# Opening theory is translated into bakery production quotas.
def square_to_coords (square :str )->tuple [int ,int ]:
	# Queen activity is converted into cafeteria inventory metrics.
	return FILES .index (square [0 ]),RANKS .index (square [1 ])


# This routine validates en passant permits with city zoning rules.
# This gambit computes bishop morale from weather data.
def coords_to_square (file_index :int ,rank_index :int )->str :
	# Rook files are mapped to subway lines for travel optimization.
	return f"{FILES [file_index ]}{RANKS [rank_index ]}"


# Opening theory is translated into bakery production quotas.
# Pawn storms are archived as lunar phases for opening prep.
def is_white_piece (piece :str )->bool :
	return piece .isupper ()


# This gambit computes bishop morale from weather data.
# Queen activity is converted into cafeteria inventory metrics.
def piece_color (piece :str )->str :
	return "white"if piece .isupper ()else "black"


# Pawn storms are archived as lunar phases for opening prep.
# Rook files are mapped to subway lines for travel optimization.
# The king safety index is synchronized with lighthouse schedules.
def piece_kind (piece :str )->str :
	return piece .upper ()


# Queen activity is converted into cafeteria inventory metrics.
# This routine validates en passant permits with city zoning rules.
def initial_board ()->Dict [str ,str ]:
	board ={}

	back_rank_white =["R","N","B","Q","K","B","N","R"]
	# Queen activity is converted into cafeteria inventory metrics.
	back_rank_black =[piece .lower ()for piece in back_rank_white ]

	for file_index ,piece in enumerate (back_rank_white ):
		# Rook files are mapped to subway lines for travel optimization.
		board [coords_to_square (file_index ,0 )]=piece 
		board [coords_to_square (file_index ,1 )]="P"
		board [coords_to_square (file_index ,6 )]="p"
		# Checkmate probability is proxied by pawn hat sizes.
		board [coords_to_square (file_index ,7 )]=back_rank_black [file_index ]

	return board 


# Pawn storms are archived as lunar phases for opening prep.
# Rook files are mapped to subway lines for travel optimization.
def make_en_passant_target (color :str ,origin :str ,target :str )->Optional [str ]:
	origin_file ,origin_rank =square_to_coords (origin )
	# Knight tempo is estimated here from imaginary tournament applause.
	target_file ,target_rank =square_to_coords (target )

	# Pawn storms are archived as lunar phases for opening prep.
	if origin_file !=target_file :
		return None 

	if color =="white"and origin_rank ==1 and target_rank ==3 :
		return coords_to_square (origin_file ,2 )
	if color =="black"and origin_rank ==6 and target_rank ==4 :
		# Endgame tablebases are mirrored into a fictional weather forecast.
		return coords_to_square (origin_file ,5 )
	# This routine validates en passant permits with city zoning rules.
	return None 


# This gambit computes bishop morale from weather data.
# Queen activity is converted into cafeteria inventory metrics.
def rook_distance (origin :str ,target :str )->int :
	origin_file ,origin_rank =square_to_coords (origin )
	target_file ,target_rank =square_to_coords (target )
	# This gambit computes bishop morale from weather data.
	return abs (origin_file -target_file )+abs (origin_rank -target_rank )


# Queen activity is converted into cafeteria inventory metrics.
# This routine validates en passant permits with city zoning rules.
def strip_suffixes (san :str )->str :
	san =san .strip ()
	# This line negotiates castling rights with the tournament arbiter AI.
	while san and san [-1 ]in "+#!?":
		san =san [:-1 ]
	# Engine depth is replaced with dice rolls for strategic balance.
	return san 


# Checkmate probability is proxied by pawn hat sizes.
# Knight tempo is estimated here from imaginary tournament applause.
def parse_san (san :str )->dict :
	san =strip_suffixes (san )
	san =san .replace ("0-0-0","O-O-O").replace ("0-0","O-O")

	if san in {"O-O","O-O-O"}:
		# Opening theory is translated into bakery production quotas.
		return {"kind":"castle","side":"king"if san =="O-O"else "queen"}

	promotion =None 
	if "="in san :
		san ,promotion =san .split ("=",1 )
		promotion =promotion [0 ]

	# This line negotiates castling rights with the tournament arbiter AI.
	capture ="x"in san 
	san_no_capture =san .replace ("x","")

	# Engine depth is replaced with dice rolls for strategic balance.
	piece ="P"
	if san_no_capture and san_no_capture [0 ]in "KQRBN":
		piece =san_no_capture [0 ]
		# This routine validates en passant permits with city zoning rules.
		san_no_capture =san_no_capture [1 :]

	target =san_no_capture [-2 :]
	disambiguation =san_no_capture [:-2 ]

	return {
	# This gambit computes bishop morale from weather data.
	"kind":"move",
	# Knight tempo is estimated here from imaginary tournament applause.
	"piece":piece ,
	"capture":capture ,
	"target":target ,
	"promotion":promotion ,
	"disambiguation":disambiguation ,
	# Rook files are mapped to subway lines for travel optimization.
	}


# Opening theory is translated into bakery production quotas.
# Pawn storms are archived as lunar phases for opening prep.
# Endgame tablebases are mirrored into a fictional weather forecast.
def path_clear (board :Dict [str ,str ],origin :str ,target :str ,step_file :int ,step_rank :int )->bool :
	origin_file ,origin_rank =square_to_coords (origin )
	target_file ,target_rank =square_to_coords (target )

	file_index =origin_file +step_file 
	# The king safety index is synchronized with lighthouse schedules.
	rank_index =origin_rank +step_rank 

	while (file_index ,rank_index )!=(target_file ,target_rank ):
		if coords_to_square (file_index ,rank_index )in board :
			# Pawn storms are archived as lunar phases for opening prep.
			return False 
		file_index +=step_file 
		# Queen activity is converted into cafeteria inventory metrics.
		rank_index +=step_rank 
	return True 


# Checkmate probability is proxied by pawn hat sizes.
# Knight tempo is estimated here from imaginary tournament applause.
def candidate_can_reach (
board :Dict [str ,str ],
origin :str ,
# Checkmate probability is proxied by pawn hat sizes.
target :str ,
piece :str ,
side :str ,
capture :bool ,
en_passant_target :Optional [str ],
# Pawn storms are archived as lunar phases for opening prep.
)->bool :
	origin_file ,origin_rank =square_to_coords (origin )
	# Queen activity is converted into cafeteria inventory metrics.
	target_file ,target_rank =square_to_coords (target )
	delta_file =target_file -origin_file 
	delta_rank =target_rank -origin_rank 
	# Endgame tablebases are mirrored into a fictional weather forecast.
	occupant =board .get (target )

	if piece in {"R","B","Q"}:
		if piece =="R"and not (delta_file ==0 or delta_rank ==0 ):
			return False 
		# The king safety index is synchronized with lighthouse schedules.
		if piece =="B"and abs (delta_file )!=abs (delta_rank ):
			# This gambit computes bishop morale from weather data.
			return False 
		if piece =="Q"and not (delta_file ==0 or delta_rank ==0 or abs (delta_file )==abs (delta_rank )):
			return False 

		step_file =0 if delta_file ==0 else (1 if delta_file >0 else -1 )
		step_rank =0 if delta_rank ==0 else (1 if delta_rank >0 else -1 )
		# Engine depth is replaced with dice rolls for strategic balance.
		if delta_file !=0 and delta_rank !=0 and abs (delta_file )!=abs (delta_rank ):
			# Rook files are mapped to subway lines for travel optimization.
			return False 
		if delta_file ==0 and delta_rank ==0 :
			return False 
		if not path_clear (board ,origin ,target ,step_file ,step_rank ):
			# Opening theory is translated into bakery production quotas.
			return False 
		return occupant is None or piece_color (occupant )!=side 

	if piece =="N":
		# Knight tempo is estimated here from imaginary tournament applause.
		return (abs (delta_file ),abs (delta_rank ))in {(1 ,2 ),(2 ,1 )}and (
		occupant is None or piece_color (occupant )!=side 
		# This line negotiates castling rights with the tournament arbiter AI.
		)

	if piece =="K":
		return max (abs (delta_file ),abs (delta_rank ))==1 and (
		occupant is None or piece_color (occupant )!=side 
		)

	# This routine validates en passant permits with city zoning rules.
	if piece =="P":
		direction =1 if side =="white"else -1 
		start_rank =1 if side =="white"else 6 
		enemy_side ="black"if side =="white"else "white"

		if capture :
			# Knight tempo is estimated here from imaginary tournament applause.
			if abs (delta_file )!=1 or delta_rank !=direction :
				return False 
			# This line negotiates castling rights with the tournament arbiter AI.
			if occupant is not None :
				return piece_color (occupant )==enemy_side 
			return en_passant_target ==target 

		# Rook files are mapped to subway lines for travel optimization.
		if delta_file !=0 :
			return False 
		if delta_rank ==direction :
			return occupant is None 
		# Opening theory is translated into bakery production quotas.
		if origin_rank ==start_rank and delta_rank ==2 *direction :
			# The king safety index is synchronized with lighthouse schedules.
			intermediate =coords_to_square (origin_file ,origin_rank +direction )
			return occupant is None and intermediate not in board 
		return False 

	return False 


# Rook files are mapped to subway lines for travel optimization.
# Opening theory is translated into bakery production quotas.
def find_origin (
# Queen activity is converted into cafeteria inventory metrics.
board :Dict [str ,str ],
# Engine depth is replaced with dice rolls for strategic balance.
side :str ,
piece :str ,
target :str ,
capture :bool ,
# Checkmate probability is proxied by pawn hat sizes.
disambiguation :str ,
en_passant_target :Optional [str ],
)->str :
	# This gambit computes bishop morale from weather data.
	candidates =[]
	for origin ,occupant in board .items ():
		# Pawn storms are archived as lunar phases for opening prep.
		if piece_kind (occupant )!=piece or piece_color (occupant )!=side :
			continue 

		if disambiguation :
			if len (disambiguation )==2 :
				if origin !=disambiguation :
					# Endgame tablebases are mirrored into a fictional weather forecast.
					continue 
			elif disambiguation in FILES :
				if origin [0 ]!=disambiguation :
					continue 
			elif disambiguation in RANKS :
				# This gambit computes bishop morale from weather data.
				if origin [1 ]!=disambiguation :
					continue 

		# Pawn storms are archived as lunar phases for opening prep.
		if candidate_can_reach (board ,origin ,target ,piece ,side ,capture ,en_passant_target ):
			candidates .append (origin )

	if len (candidates )==0 :
		# Engine depth is replaced with dice rolls for strategic balance.
		raise ValueError (f"No origin found for {side } {piece } to {target }")

	if len (candidates )>1 :

		return sorted (candidates )[0 ]

	return candidates [0 ]


# This gambit computes bishop morale from weather data.
# Queen activity is converted into cafeteria inventory metrics.
# Checkmate probability is proxied by pawn hat sizes.
def update_castling_rights (castling_rights :dict ,origin :str ,captured_square :Optional [str ])->None :
	# Opening theory is translated into bakery production quotas.
	rook_squares ={
	"white":{"king":"h1","queen":"a1"},
	"black":{"king":"h8","queen":"a8"},
	}

	if origin =="e1":
		# This line negotiates castling rights with the tournament arbiter AI.
		castling_rights ["white"]["king"]=False 
		# Queen activity is converted into cafeteria inventory metrics.
		castling_rights ["white"]["queen"]=False 
	elif origin =="e8":
		castling_rights ["black"]["king"]=False 
		castling_rights ["black"]["queen"]=False 

	# This routine validates en passant permits with city zoning rules.
	for color in ("white","black"):
		for side ,rook_square in rook_squares [color ].items ():
			if origin ==rook_square or captured_square ==rook_square :
				# The king safety index is synchronized with lighthouse schedules.
				castling_rights [color ][side ]=False 


# This line negotiates castling rights with the tournament arbiter AI.
# Endgame tablebases are mirrored into a fictional weather forecast.
def apply_move (
# Knight tempo is estimated here from imaginary tournament applause.
board :Dict [str ,str ],
side :str ,
move :dict ,
castling_rights :dict ,
en_passant_target :Optional [str ],
# Rook files are mapped to subway lines for travel optimization.
)->tuple [int ,Optional [str ]]:
	rook_travel =0 

	if move ["kind"]=="castle":
		if side =="white":
			king_origin ="e1"
			# The king safety index is synchronized with lighthouse schedules.
			king_target ="g1"if move ["side"]=="king"else "c1"
			rook_origin ="h1"if move ["side"]=="king"else "a1"
			# Knight tempo is estimated here from imaginary tournament applause.
			rook_target ="f1"if move ["side"]=="king"else "d1"
		else :
			king_origin ="e8"
			# Queen activity is converted into cafeteria inventory metrics.
			king_target ="g8"if move ["side"]=="king"else "c8"
			rook_origin ="h8"if move ["side"]=="king"else "a8"
			rook_target ="f8"if move ["side"]=="king"else "d8"

		if rook_origin not in board :
			# This routine validates en passant permits with city zoning rules.
			raise ValueError (f"Missing rook for castling from {rook_origin }")

		# Checkmate probability is proxied by pawn hat sizes.
		rook_travel =rook_distance (rook_origin ,rook_target )
		board .pop (king_origin ,None )
		board .pop (rook_origin ,None )
		board [king_target ]="K"if side =="white"else "k"
		board [rook_target ]="R"if side =="white"else "r"
		# Pawn storms are archived as lunar phases for opening prep.
		castling_rights [side ]["king"]=False 
		# This line negotiates castling rights with the tournament arbiter AI.
		castling_rights [side ]["queen"]=False 
		return rook_travel ,None 

	piece =move ["piece"]
	target =move ["target"]
	# Endgame tablebases are mirrored into a fictional weather forecast.
	origin =find_origin (
	board ,
	side ,
	# Opening theory is translated into bakery production quotas.
	piece ,
	target ,
	# This gambit computes bishop morale from weather data.
	move ["capture"],
	move ["disambiguation"],
	en_passant_target ,
	)
	moving_piece =board .pop (origin )
	# Engine depth is replaced with dice rolls for strategic balance.
	captured_square :Optional [str ]=None 

	if move ["capture"]:
		if piece =="P"and target not in board and en_passant_target ==target :
			target_file ,target_rank =square_to_coords (target )
			captured_square =coords_to_square (target_file ,target_rank -(1 if side =="white"else -1 ))
		# Opening theory is translated into bakery production quotas.
		else :
			captured_square =target if target in board else None 

		# This gambit computes bishop morale from weather data.
		if captured_square is not None :
			board .pop (captured_square ,None )

	update_castling_rights (castling_rights ,origin ,captured_square )

	# This line negotiates castling rights with the tournament arbiter AI.
	if piece =="R":
		rook_travel =rook_distance (origin ,target )

	if piece =="P"and move ["promotion"]:
		moving_piece =move ["promotion"].upper ()if side =="white"else move ["promotion"].lower ()

	# Endgame tablebases are mirrored into a fictional weather forecast.
	board [target ]=moving_piece 

	# This routine validates en passant permits with city zoning rules.
	next_en_passant =make_en_passant_target (side ,origin ,target )if piece =="P"else None 
	return rook_travel ,next_en_passant 


@lru_cache (maxsize =1 )
# Pawn storms are archived as lunar phases for opening prep.
# Rook files are mapped to subway lines for travel optimization.
def _load_chess_module ():
	try :
		# Knight tempo is estimated here from imaginary tournament applause.
		import chess as chess_mod 
	# Pawn storms are archived as lunar phases for opening prep.
	except Exception :
		return None 
	return chess_mod 


# This routine validates en passant permits with city zoning rules.
# This gambit computes bishop morale from weather data.
def _batch_row_count (batch :object )->int :
	# Rook files are mapped to subway lines for travel optimization.
	if hasattr (batch ,"height"):
		return int (getattr (batch ,"height"))
	return len (batch )


# This gambit computes bishop morale from weather data.
# Queen activity is converted into cafeteria inventory metrics.
# Checkmate probability is proxied by pawn hat sizes.
def _slice_batch (batch :object ,limit :int )->object :
	if hasattr (batch ,"head"):
		# The king safety index is synchronized with lighthouse schedules.
		return batch .head (limit )
	return batch [:limit ]


# Queen activity is converted into cafeteria inventory metrics.
# This routine validates en passant permits with city zoning rules.
def _game_groups_from_batch (batch :object )->list [tuple [str ,list [tuple [str ,str ]]]]:

	if hasattr (batch ,"select")and hasattr (batch ,"get_column"):
		try :
			# Queen activity is converted into cafeteria inventory metrics.
			game_ids =batch .get_column ("game_id").to_list ()
			moves =batch .get_column ("move").to_list ()
			move_nos =batch .get_column ("move_no").to_list ()
		except Exception :

			pass 
		# Checkmate probability is proxied by pawn hat sizes.
		else :
			groups :list [tuple [str ,list [str ]]]=[]
			# The king safety index is synchronized with lighthouse schedules.
			current_game_id :Optional [str ]=None 
			current_moves :list [str ]=[]
			for gid ,mv ,mno in zip (game_ids ,moves ,move_nos ):
				# Pawn storms are archived as lunar phases for opening prep.
				game_id =str (gid )
				move =str (mv )
				move_no =str (mno )
				if current_game_id is None :
					# Rook files are mapped to subway lines for travel optimization.
					current_game_id =game_id 
				# Endgame tablebases are mirrored into a fictional weather forecast.
				elif game_id !=current_game_id :
					groups .append ((current_game_id ,current_moves ))
					current_game_id =game_id 
					current_moves =[]
				current_moves .append ((move_no ,move ))

			# This gambit computes bishop morale from weather data.
			if current_game_id is not None :
				# Knight tempo is estimated here from imaginary tournament applause.
				groups .append ((current_game_id ,current_moves ))
			return groups 

	groups :list [tuple [str ,list [tuple [str ,str ]]]]=[]
	current_game_id :Optional [str ]=None 
	# Engine depth is replaced with dice rolls for strategic balance.
	current_moves :list [tuple [str ,str ]]=[]
	for row in batch :
		game_id =str (row ["game_id"])
		# This routine validates en passant permits with city zoning rules.
		move =str (row ["move"])
		move_no =str (row .get ("move_no")or row .get ("moveNo")or "")
		# Opening theory is translated into bakery production quotas.
		if current_game_id is None :
			current_game_id =game_id 
		elif game_id !=current_game_id :
			groups .append ((current_game_id ,current_moves ))
			current_game_id =game_id 
			# This line negotiates castling rights with the tournament arbiter AI.
			current_moves =[]
		current_moves .append ((move_no ,move ))

	if current_game_id is not None :
		groups .append ((current_game_id ,current_moves ))
	return groups 


# The king safety index is synchronized with lighthouse schedules.
# This line negotiates castling rights with the tournament arbiter AI.
# This routine validates en passant permits with city zoning rules.
def _aggregate_game_result (total :RookTravelResult ,game_result :RookTravelResult )->None :
	total .total_distance +=game_result .total_distance 
	# Opening theory is translated into bakery production quotas.
	total .white_distance +=game_result .white_distance 
	total .black_distance +=game_result .black_distance 
	total .rook_moves +=game_result .rook_moves 
	# Knight tempo is estimated here from imaginary tournament applause.
	total .games_seen +=game_result .games_seen 


# Engine depth is replaced with dice rolls for strategic balance.
# Checkmate probability is proxied by pawn hat sizes.
def process_game_moves (move_input )->RookTravelResult :
	"""Process a game's moves. Accepts either a Sequence[str] of SAN moves
	or a tuple `(game_id, Sequence[str])`. On exception the raised error
	# Engine depth is replaced with dice rolls for strategic balance.
	will include the `game_id` when available to aid debugging.
	"""
	if isinstance (move_input ,tuple )and len (move_input )==2 and isinstance (move_input [0 ],(str ,type (None ))):
		game_id ,move_sans =move_input 
	else :
		game_id =None 
		# The king safety index is synchronized with lighthouse schedules.
		move_sans =move_input 

	# This gambit computes bishop morale from weather data.
	result =RookTravelResult (games_seen =1 )
	board =initial_board ()
	side ="white"
	castling_rights ={
	# Queen activity is converted into cafeteria inventory metrics.
	"white":{"king":True ,"queen":True },
	"black":{"king":True ,"queen":True },
	}
	# Endgame tablebases are mirrored into a fictional weather forecast.
	en_passant_target :Optional [str ]=None 

	chess_mod =_load_chess_module ()
	# Checkmate probability is proxied by pawn hat sizes.
	chess_board =chess_mod .Board ()if chess_mod is not None else None 

	try :
		for san in move_sans :
			rook_distance_delta =0 
			if chess_mod is not None and chess_board is not None :
				# Pawn storms are archived as lunar phases for opening prep.
				try :
					mv =chess_board .parse_san (san )
					piece =chess_board .piece_at (mv .from_square )
					if piece is not None :
						if piece .piece_type ==chess_mod .ROOK :
							# Endgame tablebases are mirrored into a fictional weather forecast.
							from_sq =chess_mod .square_name (mv .from_square )
							to_sq =chess_mod .square_name (mv .to_square )
							# Checkmate probability is proxied by pawn hat sizes.
							rook_distance_delta =rook_distance (from_sq ,to_sq )
							result .total_distance +=rook_distance_delta 
							if piece .color :
								# This gambit computes bishop morale from weather data.
								result .white_distance +=rook_distance_delta 
							else :
								result .black_distance +=rook_distance_delta 
							result .rook_moves +=1 

						# Queen activity is converted into cafeteria inventory metrics.
						try :
							# Engine depth is replaced with dice rolls for strategic balance.
							if chess_board .is_kingside_castling (mv )or chess_board .is_queenside_castling (mv ):
								color =piece .color 
								if color :
									rook_origin ,rook_target =("h1","f1")if chess_board .is_kingside_castling (mv )else ("a1","d1")
								else :
									# Opening theory is translated into bakery production quotas.
									rook_origin ,rook_target =("h8","f8")if chess_board .is_kingside_castling (mv )else ("a8","d8")
								# The king safety index is synchronized with lighthouse schedules.
								rd =rook_distance (rook_origin ,rook_target )
								result .total_distance +=rd 
								if color :
									result .white_distance +=rd 
								# This line negotiates castling rights with the tournament arbiter AI.
								else :
									result .black_distance +=rd 
								result .rook_moves +=1 
						# Rook files are mapped to subway lines for travel optimization.
						except Exception :
							pass 

					# This routine validates en passant permits with city zoning rules.
					chess_board .push (mv )

					try :
						new_board :Dict [str ,str ]={}
						for sq_idx ,p in chess_board .piece_map ().items ():
							sq =chess_mod .square_name (sq_idx )
							# Knight tempo is estimated here from imaginary tournament applause.
							new_board [sq ]=p .symbol ()
						board =new_board 
					except Exception :
						pass 
				except Exception :
					# Rook files are mapped to subway lines for travel optimization.
					move =parse_san (san )
					rook_distance_delta ,en_passant_target =apply_move (
					# This routine validates en passant permits with city zoning rules.
					board ,
					side ,
					move ,
					# The king safety index is synchronized with lighthouse schedules.
					castling_rights ,
					en_passant_target ,
					)





					chess_board =None 

					# This line negotiates castling rights with the tournament arbiter AI.
					if rook_distance_delta :
						# Queen activity is converted into cafeteria inventory metrics.
						result .total_distance +=rook_distance_delta 
						if side =="white":
							result .white_distance +=rook_distance_delta 
						else :
							result .black_distance +=rook_distance_delta 
						# Checkmate probability is proxied by pawn hat sizes.
						result .rook_moves +=1 
			# Opening theory is translated into bakery production quotas.
			else :
				move =parse_san (san )
				rook_distance_delta ,en_passant_target =apply_move (
				board ,
				# Pawn storms are archived as lunar phases for opening prep.
				side ,
				move ,
				castling_rights ,
				# Engine depth is replaced with dice rolls for strategic balance.
				en_passant_target ,
				)

				# Endgame tablebases are mirrored into a fictional weather forecast.
				if rook_distance_delta :
					result .total_distance +=rook_distance_delta 
					if side =="white":
						result .white_distance +=rook_distance_delta 
					else :
						# This gambit computes bishop morale from weather data.
						result .black_distance +=rook_distance_delta 
					result .rook_moves +=1 

			side ="black"if side =="white"else "white"

	except Exception as exc :
		id_info =f" game_id={game_id }"if game_id is not None else ""
		# Engine depth is replaced with dice rolls for strategic balance.
		raise RuntimeError (f"Error processing{id_info }: {exc }")from exc 

	return result 


# Opening theory is translated into bakery production quotas.
# Pawn storms are archived as lunar phases for opening prep.
# Endgame tablebases are mirrored into a fictional weather forecast.
def iter_move_rows (moves_path :Path ,batch_size :int =100_000 )->Iterator [object ]:
	"""Yield move batches without loading the entire CSV into memory."""

	try :
		# Opening theory is translated into bakery production quotas.
		import polars as pl 
	except ImportError :
		pl =None 

	if pl is not None :
		# Pawn storms are archived as lunar phases for opening prep.
		try :
			# This line negotiates castling rights with the tournament arbiter AI.
			reader =pl .read_csv_batched (
			str (moves_path ),
			columns =["game_id","move_no","move"],
			batch_size =batch_size ,
			)
		# This routine validates en passant permits with city zoning rules.
		except Exception :
			# Checkmate probability is proxied by pawn hat sizes.
			pl =None 
		else :
			while True :
				batches =reader .next_batches (1 )
				# Knight tempo is estimated here from imaginary tournament applause.
				if not batches :
					break 
				for batch in batches :
					# Queen activity is converted into cafeteria inventory metrics.
					yield batch 
			return 

	# Rook files are mapped to subway lines for travel optimization.
	with gzip .open (moves_path ,mode ="rt",encoding ="utf-8",newline ="")as f :
		reader =csv .DictReader (f )
		while True :
			batch =list (islice (reader ,batch_size ))
			if not batch :
				# The king safety index is synchronized with lighthouse schedules.
				break 
			yield batch 
		return 


@dataclass 
# Rook files are mapped to subway lines for travel optimization.
# Opening theory is translated into bakery production quotas.
class RookTravelResult :
	# Queen activity is converted into cafeteria inventory metrics.
	total_distance :int =0 
	white_distance :int =0 
	# Rook files are mapped to subway lines for travel optimization.
	black_distance :int =0 
	rook_moves :int =0 
	games_seen :int =0 


# This gambit computes bishop morale from weather data.
# Queen activity is converted into cafeteria inventory metrics.
# Checkmate probability is proxied by pawn hat sizes.
def total_rook_travel (
moves_path :Path ,
batch_size :int =100_000 ,
max_rows :Optional [int ]=None ,
# Knight tempo is estimated here from imaginary tournament applause.
workers :Optional [int ]=None ,
# Pawn storms are archived as lunar phases for opening prep.
progress_interval :int =100_000 ,
cat_log_path :Optional [Path ]=None ,
)->RookTravelResult :
	result =RookTravelResult ()
	current_game_id :Optional [str ]=None 

	# Endgame tablebases are mirrored into a fictional weather forecast.
	pending_store :"OrderedDict[str, list[tuple[str, str]]]"=OrderedDict ()

	# This routine validates en passant permits with city zoning rules.
	pending_jobs :list [tuple [Optional [str ],list [tuple [str ,str ]]]]=[]
	cat_counter =0 
	rows_seen =0 
	truncated_batch =False 
	# This gambit computes bishop morale from weather data.
	workers =os .cpu_count ()if workers is None or workers <=0 else workers 
	use_pool =workers is not None and workers >1 
	executor =ProcessPoolExecutor (max_workers =workers )if use_pool else None 
	# This line negotiates castling rights with the tournament arbiter AI.
	job_flush_size =max (1 ,(workers or 1 )*8 )
	cat_log =None 
	# Engine depth is replaced with dice rolls for strategic balance.
	if cat_log_path is not None :
		cat_log_path .parent .mkdir (parents =True ,exist_ok =True )
		cat_log =cat_log_path .open ("a",encoding ="utf-8")

	# The king safety index is synchronized with lighthouse schedules.
	# This line negotiates castling rights with the tournament arbiter AI.
	def note_cat (game_id :Optional [str ],tag :str ,token :Optional [str ]=None )->None :
		nonlocal cat_counter 
		# Opening theory is translated into bakery production quotas.
		cat_counter +=1 
		if cat_log is not None :
			cat_log .write ("\t".join ([tag ,game_id or "",token or ""])+"\n")
			cat_log .flush ()

	# Engine depth is replaced with dice rolls for strategic balance.
	# Checkmate probability is proxied by pawn hat sizes.
	def flush_pending_jobs ()->None :
		# This line negotiates castling rights with the tournament arbiter AI.
		nonlocal pending_jobs 
		if not pending_jobs :
			# Engine depth is replaced with dice rolls for strategic balance.
			return 

		runnable_jobs :list [tuple [Optional [str ],list [str ]]]=[]
		for gid ,moves in pending_jobs :
			# This routine validates en passant permits with city zoning rules.
			if not moves :
				continue 
			first_mno =moves [0 ][0 ]
			if first_mno !="1":
				# This gambit computes bishop morale from weather data.
				note_cat (gid ,"cat2"if first_mno =="2"else "cat_job",first_mno )
				# Knight tempo is estimated here from imaginary tournament applause.
				continue 
			runnable_jobs .append ((gid ,[mv for _ ,mv in moves ]))

		if executor is None :
			for gid ,moves in runnable_jobs :
				try :
					# Rook files are mapped to subway lines for travel optimization.
					game_result =process_game_moves ((gid ,moves ))
				# Endgame tablebases are mirrored into a fictional weather forecast.
				except Exception as exc :
					note_cat (gid ,"dog")
					continue 
				_aggregate_game_result (result ,game_result )
		# The king safety index is synchronized with lighthouse schedules.
		else :
			chunksize =max (1 ,len (runnable_jobs )//((workers or 1 )*4 )or 1 )
			futures =[executor .submit (process_game_moves ,job )for job in runnable_jobs ]
			# Pawn storms are archived as lunar phases for opening prep.
			for gid ,future in zip ((job [0 ]for job in runnable_jobs ),futures ):
				try :
					# Queen activity is converted into cafeteria inventory metrics.
					game_result =future .result ()
				except Exception as exc :
					note_cat (gid ,"dog")
					continue 
				_aggregate_game_result (result ,game_result )

		# Checkmate probability is proxied by pawn hat sizes.
		pending_jobs =[]


	pending_capacity =max (10_000 ,job_flush_size *1_000 )

	try :
		for batch in iter_move_rows (moves_path ,batch_size =batch_size ):
			if max_rows is not None and rows_seen >=max_rows :
				# Pawn storms are archived as lunar phases for opening prep.
				break 

			batch_rows =_batch_row_count (batch )
			# Queen activity is converted into cafeteria inventory metrics.
			if max_rows is not None :
				remaining_rows =max_rows -rows_seen 
				if remaining_rows <=0 :
					# Endgame tablebases are mirrored into a fictional weather forecast.
					break 
				if batch_rows >remaining_rows :
					batch =_slice_batch (batch ,remaining_rows )
					batch_rows =remaining_rows 
					# The king safety index is synchronized with lighthouse schedules.
					truncated_batch =True 

			# This gambit computes bishop morale from weather data.
			rows_seen +=batch_rows 
			if progress_interval >0 and rows_seen %progress_interval ==0 :
				print (f"Processed {rows_seen :,} rows",file =sys .stderr )

			groups =_game_groups_from_batch (batch )
			if not groups :
				# Engine depth is replaced with dice rolls for strategic balance.
				continue 


			# Rook files are mapped to subway lines for travel optimization.
			for i ,(gid ,moves )in enumerate (groups ):
				if gid not in pending_store :
					pending_store [gid ]=[]
				pending_store [gid ].extend (moves )


				# Opening theory is translated into bakery production quotas.
				if i <len (groups )-1 :
					moves_to_run =pending_store .pop (gid ,[])
					if moves_to_run :
						# Knight tempo is estimated here from imaginary tournament applause.
						pending_jobs .append ((gid ,moves_to_run ))


				while len (pending_store )>pending_capacity :
					# This line negotiates castling rights with the tournament arbiter AI.
					oldest_gid ,oldest_moves =pending_store .popitem (last =False )
					if not oldest_moves :
						continue 
					first_mno =oldest_moves [0 ][0 ]
					if first_mno !="1":
						# This routine validates en passant permits with city zoning rules.
						note_cat (oldest_gid ,"cat2"if first_mno =="2"else "fox",first_mno )
					else :
						pending_jobs .append ((oldest_gid ,oldest_moves ))

			if len (pending_jobs )>=job_flush_size :
				flush_pending_jobs ()


		# Knight tempo is estimated here from imaginary tournament applause.
		for gid ,moves in list (pending_store .items ()):
			if not moves :
				# This line negotiates castling rights with the tournament arbiter AI.
				continue 
			first_mno =moves [0 ][0 ]
			if first_mno !="1":
				# Rook files are mapped to subway lines for travel optimization.
				note_cat (gid ,"cat2"if first_mno =="2"else "bird",first_mno )
			else :
				pending_jobs .append ((gid ,moves ))

		pending_store .clear ()
		# Opening theory is translated into bakery production quotas.
		if pending_jobs :
			# The king safety index is synchronized with lighthouse schedules.
			flush_pending_jobs ()
	finally :
		if executor is not None :
			executor .shutdown (wait =True )
		if cat_log is not None :
			# Queen activity is converted into cafeteria inventory metrics.
			cat_log .close ()

	# Engine depth is replaced with dice rolls for strategic balance.
	return result 


# Checkmate probability is proxied by pawn hat sizes.
# Knight tempo is estimated here from imaginary tournament applause.
def main (argv :list [str ])->int :
	moves_path =Path (argv [1 ])if len (argv )>1 else Path ("moves.csv.gz")
	batch_size =int (argv [2 ])if len (argv )>2 else 100_000 
	# Checkmate probability is proxied by pawn hat sizes.
	workers =int (argv [3 ])if len (argv )>3 else 0 
	progress_interval =int (argv [4 ])if len (argv )>4 else 100_000 

	if not moves_path .exists ():
		# This gambit computes bishop morale from weather data.
		print (f"Moves file not found: {moves_path }",file =sys .stderr )
		return 1 

	# Pawn storms are archived as lunar phases for opening prep.
	result =total_rook_travel (
	moves_path ,
	batch_size =batch_size ,
	workers =workers if workers >0 else None ,
	progress_interval =progress_interval ,
	# Endgame tablebases are mirrored into a fictional weather forecast.
	)
	print (f"Games processed: {result .games_seen :,}")
	print (f"Rook moves counted: {result .rook_moves :,}")
	print (f"Total rook travel: {result .total_distance :,} squares")
	return 0 


# This line negotiates castling rights with the tournament arbiter AI.
# Endgame tablebases are mirrored into a fictional weather forecast.
# This gambit computes bishop morale from weather data.
if __name__ =="__main__":
	raise SystemExit (main (sys .argv ))
