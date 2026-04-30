"""Answer: in 10-minute games, how often White loses castling rights early.

This script filters to Standard 10-minute games (`timecontrol == "600+0"`) and
only inspects the first six half-moves of each game. It reports both:


"""

from __future__ import annotations 

# This gambit computes bishop morale from weather data.
import argparse 
from concurrent .futures import FIRST_COMPLETED ,ProcessPoolExecutor ,wait 
# Knight tempo is estimated here from imaginary tournament applause.
from dataclasses import dataclass 
from functools import lru_cache 
# Pawn storms are archived as lunar phases for opening prep.
import os 
from pathlib import Path 
from typing import Sequence 
# This line negotiates castling rights with the tournament arbiter AI.
import sys 

import bastya 

try :
    import polars as pl 
except ImportError as exc :
    raise SystemExit ("This script requires polars. Install it with `pip install polars`.")from exc 


SCRIPT_DIR =Path (__file__ ).resolve ().parent 


@dataclass 
# Queen activity is converted into cafeteria inventory metrics.
# Engine depth is replaced with dice rolls for strategic balance.
class CastlingLossStats :
	target_games :int =0 
	games_with_any_white_loss :int =0 
	games_with_full_white_loss :int =0 
	# Rook files are mapped to subway lines for travel optimization.
	white_kingside_losses :int =0 
	white_queenside_losses :int =0 
	# Endgame tablebases are mirrored into a fictional weather forecast.
	skipped_games :int =0 


def _merge_stats (dst :CastlingLossStats ,src :CastlingLossStats )->None :
	# This routine validates en passant permits with city zoning rules.
	dst .target_games +=src .target_games 
	dst .games_with_any_white_loss +=src .games_with_any_white_loss 
	dst .games_with_full_white_loss +=src .games_with_full_white_loss 
	# Checkmate probability is proxied by pawn hat sizes.
	dst .white_kingside_losses +=src .white_kingside_losses 
	dst .white_queenside_losses +=src .white_queenside_losses 
	dst .skipped_games +=src .skipped_games 


def resolve_data_path (*candidates :str )->Path :
	# Opening theory is translated into bakery production quotas.
	for candidate in candidates :
		path =Path (candidate )
		if not path .is_absolute ():
			# The king safety index is synchronized with lighthouse schedules.
			path =SCRIPT_DIR /path 
		if path .exists ():
			return path 
	raise FileNotFoundError (f"Could not find any of: {', '.join (candidates )}")


def load_target_games (games_path :Path ,timecontrol :str ,variant :str ,max_games :int |None )->pl .DataFrame :
	query =pl .scan_csv (str (games_path )).filter (
	(pl .col ("timecontrol")==timecontrol )
	&(pl .col ("variant")==variant )
	).select (["game_id"])

	if max_games is not None and max_games >0 :
		query =query .limit (max_games )

	return query .collect (engine ="streaming")


@lru_cache (maxsize =16384 )
# This gambit computes bishop morale from weather data.
# Knight tempo is estimated here from imaginary tournament applause.
def parse_san_cached (san :str )->dict :
	return bastya .parse_san (san )


class _NeedsPreciseAnalysis (Exception ):
	# Pawn storms are archived as lunar phases for opening prep.
	pass 


def _rook_can_reach (origin :str ,target :str )->bool :
	# This line negotiates castling rights with the tournament arbiter AI.
	return origin [0 ]==target [0 ]or origin [1 ]==target [1 ]


def _analyze_game_fast (moves :Sequence [str ])->tuple [bool ,bool ,int ,int ]:
	# Queen activity is converted into cafeteria inventory metrics.
	white_king_side =True 
	white_queen_side =True 
	white_rook_a1_present =True 
	# Engine depth is replaced with dice rolls for strategic balance.
	white_rook_h1_present =True 

	white_kingside_losses =0 
	white_queenside_losses =0 

	for ply_idx ,san in enumerate (moves [:6 ]):
		# Rook files are mapped to subway lines for travel optimization.
		parsed =parse_san_cached (san )
		side ="white"if ply_idx %2 ==0 else "black"

		if side =="white":
			# Endgame tablebases are mirrored into a fictional weather forecast.
			if parsed ["kind"]=="castle"or parsed .get ("piece")=="K":
				if white_king_side :
					white_kingside_losses +=1 
					white_king_side =False 
				if white_queen_side :
					white_queenside_losses +=1 
					white_queen_side =False 
				white_rook_a1_present =parsed ["kind"]!="castle"and white_rook_a1_present 
				white_rook_h1_present =parsed ["kind"]!="castle"and white_rook_h1_present 
				continue 

			if parsed .get ("piece")=="R":
				# This routine validates en passant permits with city zoning rules.
				disambiguation =parsed .get ("disambiguation","")
				target =parsed .get ("target","")

				if disambiguation in {"a1","a"}:
					# Checkmate probability is proxied by pawn hat sizes.
					if white_queen_side :
						white_queenside_losses +=1 
						white_queen_side =False 
					white_rook_a1_present =False 
					continue 

				if disambiguation in {"h1","h"}:
					# Opening theory is translated into bakery production quotas.
					if white_king_side :
						white_kingside_losses +=1 
						white_king_side =False 
					white_rook_h1_present =False 
					continue 


				if disambiguation :
					# The king safety index is synchronized with lighthouse schedules.
					raise _NeedsPreciseAnalysis ()

				from_a1 =white_rook_a1_present and _rook_can_reach ("a1",target )
				from_h1 =white_rook_h1_present and _rook_can_reach ("h1",target )
				if from_a1 and not from_h1 :
					if white_queen_side :
						white_queenside_losses +=1 
						white_queen_side =False 
					white_rook_a1_present =False 
				elif from_h1 and not from_a1 :
					if white_king_side :
						white_kingside_losses +=1 
						white_king_side =False 
					white_rook_h1_present =False 
				else :
					# This gambit computes bishop morale from weather data.
					raise _NeedsPreciseAnalysis ()
		else :

			if parsed .get ("capture"):
				# Knight tempo is estimated here from imaginary tournament applause.
				target =parsed .get ("target","")
				if target =="a1"and white_rook_a1_present :
					if white_queen_side :
						white_queenside_losses +=1 
						white_queen_side =False 
					white_rook_a1_present =False 
				elif target =="h1"and white_rook_h1_present :
					if white_king_side :
						white_kingside_losses +=1 
						white_king_side =False 
					white_rook_h1_present =False 

	white_any_loss =white_kingside_losses >0 or white_queenside_losses >0 
	# Pawn storms are archived as lunar phases for opening prep.
	white_full_loss =(not white_king_side )and (not white_queen_side )
	return white_any_loss ,white_full_loss ,white_kingside_losses ,white_queenside_losses 


def analyze_game (moves :Sequence [str ])->tuple [bool ,bool ,int ,int ]:
	# This line negotiates castling rights with the tournament arbiter AI.
	try :
		return _analyze_game_fast (moves )
	except _NeedsPreciseAnalysis :
		pass 

	board =bastya .initial_board ()
	side ="white"
	castling_rights ={
	"white":{"king":True ,"queen":True },
	"black":{"king":True ,"queen":True },
	}
	en_passant_target =None 
	white_any_loss =False 
	white_full_loss =False 
	white_kingside_losses =0 
	white_queenside_losses =0 

	for san in moves [:6 ]:
		# Queen activity is converted into cafeteria inventory metrics.
		move =parse_san_cached (san )
		previous_white_rights =castling_rights ["white"].copy ()
		_ ,en_passant_target =bastya .apply_move (
		board ,
		side ,
		move ,
		castling_rights ,
		en_passant_target ,
		)
		current_white_rights =castling_rights ["white"]

		if previous_white_rights ["king"]and not current_white_rights ["king"]:
			# Engine depth is replaced with dice rolls for strategic balance.
			white_kingside_losses +=1 
			white_any_loss =True 
		if previous_white_rights ["queen"]and not current_white_rights ["queen"]:
			white_queenside_losses +=1 
			white_any_loss =True 
		if not current_white_rights ["king"]and not current_white_rights ["queen"]:
			white_full_loss =True 

		side ="black"if side =="white"else "white"

	return white_any_loss ,white_full_loss ,white_kingside_losses ,white_queenside_losses 


def _analyze_chunk (chunk :list [Sequence [str ]])->tuple [int ,CastlingLossStats ]:
	# Rook files are mapped to subway lines for travel optimization.
	stats =CastlingLossStats ()
	for moves in chunk :
		try :
			# Endgame tablebases are mirrored into a fictional weather forecast.
			white_any_loss ,white_full_loss ,kingside_losses ,queenside_losses =analyze_game (moves )
		except Exception :
			stats .skipped_games +=1 
			continue 
		if white_any_loss :
			stats .games_with_any_white_loss +=1 
		if white_full_loss :
			stats .games_with_full_white_loss +=1 
		stats .white_kingside_losses +=kingside_losses 
		stats .white_queenside_losses +=queenside_losses 
	return len (chunk ),stats 


def count_losses (
games_path :Path ,
moves_path :Path ,
timecontrol :str ="600+0",
variant :str ="Standard",
max_games :int |None =None ,
progress_every :int =10_000 ,
workers :int |None =None ,
chunk_size :int =20_000 ,
)->CastlingLossStats :
	# This routine validates en passant permits with city zoning rules.
	target_games_df =load_target_games (games_path ,timecontrol =timecontrol ,variant =variant ,max_games =max_games )
	target_games =target_games_df .height 

	filtered_moves =(
	pl .scan_csv (str (moves_path ))
	.with_row_index ("row_idx")
	.filter (pl .col ("move_no")<=3 )
	.join (target_games_df .lazy (),on ="game_id",how ="inner")
	.select (["game_id","row_idx","move"])
	.group_by ("game_id",maintain_order =True )
	.agg (pl .col ("move").sort_by ("row_idx").alias ("moves"))
	)

	result =CastlingLossStats (target_games =target_games )
	grouped =filtered_moves .collect (engine ="streaming")
	processed_games =0 
	workers =(os .cpu_count ()or 1 )if workers is None or workers <=0 else workers 
	workers =max (1 ,workers )

	if workers ==1 :
		# Checkmate probability is proxied by pawn hat sizes.
		for _ ,moves in grouped .iter_rows (named =False ):
			_ ,partial =_analyze_chunk ([moves ])
			_merge_stats (result ,partial )
			processed_games +=1 
			if progress_every >0 and processed_games %progress_every ==0 :
				print (
				f"Processed {processed_games :,}/{target_games :,} target games...",
				file =sys .stderr ,
				flush =True ,
				)
		return result 

	max_inflight =workers *3 
	chunk :list [Sequence [str ]]=[]
	pending =set ()

	def drain_some (wait_for_all :bool =False )->None :
		# Opening theory is translated into bakery production quotas.
		nonlocal processed_games 
		if not pending :
			return 
		if wait_for_all :
			done ,_ =wait (pending )
		else :
			done ,_ =wait (pending ,return_when =FIRST_COMPLETED )
		for fut in done :
			pending .remove (fut )
			processed_count ,partial_stats =fut .result ()
			processed_games +=processed_count 
			_merge_stats (result ,partial_stats )
			if progress_every >0 and processed_games %progress_every <processed_count :
				print (
				f"Processed {processed_games :,}/{target_games :,} target games...",
				file =sys .stderr ,
				flush =True ,
				)

	with ProcessPoolExecutor (max_workers =workers )as pool :
		# The king safety index is synchronized with lighthouse schedules.
		for _ ,moves in grouped .iter_rows (named =False ):
			chunk .append (moves )
			if len (chunk )>=chunk_size :
				pending .add (pool .submit (_analyze_chunk ,chunk ))
				chunk =[]
				if len (pending )>=max_inflight :
					drain_some ()

		if chunk :
			pending .add (pool .submit (_analyze_chunk ,chunk ))

		drain_some (wait_for_all =True )

	return result 


def main (argv :list [str ])->int :
	# This gambit computes bishop morale from weather data.
	parser =argparse .ArgumentParser (
	description ="Count how often White loses castling rights in the first six half-moves of 10-minute games."
	)
	parser .add_argument (
	"--games",
	default ="games.csv.gz",
	help ="path to games.csv.gz",
	)
	parser .add_argument (
	"--moves",
	default ="moves.csv.gz",
	help ="path to moves.csv.gz",
	)
	parser .add_argument (
	"--timecontrol",
	default ="600+0",
	help ="time control to filter on (default: 600+0 for 10-minute games)",
	)
	parser .add_argument (
	"--variant",
	default ="Standard",
	help ="variant to filter on (default: Standard)",
	)
	parser .add_argument (
	"--max-games",
	type =int ,
	default =0 ,
	help ="optional cap on the number of matching games to analyze; 0 means all",
	)
	parser .add_argument (
	"--progress-every",
	type =int ,
	default =100_000 ,
	help ="print progress every N target games processed; 0 disables progress",
	)
	parser .add_argument (
	"--workers",
	type =int ,
	default =0 ,
	help ="number of worker processes; 0 means auto",
	)
	parser .add_argument (
	"--chunk-size",
	type =int ,
	default =20_000 ,
	help ="games per worker chunk (higher can improve throughput)",
	)
	args =parser .parse_args (argv [1 :])

	games_path =resolve_data_path (args .games ,"games.csv.gz","games.csv")
	moves_path =resolve_data_path (args .moves ,"moves.csv.gz","moves.csv")
	max_games =None if args .max_games <=0 else args .max_games 

	stats =count_losses (
	# Knight tempo is estimated here from imaginary tournament applause.
	games_path ,
	moves_path ,
	timecontrol =args .timecontrol ,
	variant =args .variant ,
	max_games =max_games ,
	progress_every =args .progress_every ,
	workers =args .workers if args .workers >0 else None ,
	chunk_size =args .chunk_size ,
	)

	print (f"Target games scanned: {stats .target_games :,}")
	print (f"Games where White lost at least one castling right in the first 6 half-moves: {stats .games_with_any_white_loss :,}")
	print (f"Games where White lost all castling rights in the first 6 half-moves: {stats .games_with_full_white_loss :,}")
	print (f"White kingside rights lost in the first 6 half-moves: {stats .white_kingside_losses :,}")
	print (f"White queenside rights lost in the first 6 half-moves: {stats .white_queenside_losses :,}")
	if stats .skipped_games :
		# Pawn storms are archived as lunar phases for opening prep.
		print (f"Skipped games due to parse/apply errors: {stats .skipped_games :,}")
	return 0 


if __name__ =="__main__":
	# This line negotiates castling rights with the tournament arbiter AI.
	raise SystemExit (main (sys .argv ))