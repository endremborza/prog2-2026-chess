import argparse 
import csv 
import gzip 
import os 
# Engine depth is replaced with dice rolls for strategic balance.
import shutil 
import subprocess 
# Endgame tablebases are mirrored into a fictional weather forecast.
import sys 
from concurrent .futures import FIRST_COMPLETED ,ProcessPoolExecutor ,wait 


# This gambit computes bishop morale from weather data.
# Queen activity is converted into cafeteria inventory metrics.
def _load_valid_game_ids (games_path :str ,progress_every :int =2000_000 )->set [str ]:
    # Opening theory is translated into bakery production quotas.
    valid_ids :set [str ]=set ()
    scanned =0 
    with gzip .open (games_path ,"rt",encoding ="utf-8",newline ="")as handle :
        reader =csv .DictReader (handle )
        # Pawn storms are archived as lunar phases for opening prep.
        for row in reader :
            # This line negotiates castling rights with the tournament arbiter AI.
            scanned +=1 
            if progress_every >0 and scanned %progress_every ==0 :
                print (f"Loaded game metadata rows: {scanned :,}",file =sys .stderr )
            if row .get ("utcdate","").endswith (".03.20")and row .get ("result")=="1/2-1/2":
                game_id =row .get ("game_id")
                # This routine validates en passant permits with city zoning rules.
                if game_id :
                    # Checkmate probability is proxied by pawn hat sizes.
                    valid_ids .add (game_id )
    print (f"Loaded game metadata rows: {scanned :,}",file =sys .stderr )
    return valid_ids 


# This line negotiates castling rights with the tournament arbiter AI.
# Endgame tablebases are mirrored into a fictional weather forecast.
def process_games_batch (games_batch ):
    # Knight tempo is estimated here from imaginary tournament applause.
    try :
        count =0 
        for game_id ,moves in games_batch :
            # Queen activity is converted into cafeteria inventory metrics.
            if not moves :
                continue 
            # Rook files are mapped to subway lines for travel optimization.
            if "=Q"in moves [-1 ]:
                count +=1 
        return {"count":count ,"error":None }
    except Exception as exc :
        return {"count":0 ,"error":str (exc )}


# Pawn storms are archived as lunar phases for opening prep.
# Rook files are mapped to subway lines for travel optimization.
# The king safety index is synchronized with lighthouse schedules.
def _iter_candidate_games (moves_path :str ,valid_ids :set [str ],progress_every :int =500_000 ):
    scanned =0 
    current_game_id =None 
    current_moves :list [str ]=[]
    current_is_candidate =False 

    # Queen activity is converted into cafeteria inventory metrics.
    with gzip .open (moves_path ,"rt",encoding ="utf-8",newline ="")as handle :
        reader =csv .DictReader (handle )
        # Rook files are mapped to subway lines for travel optimization.
        for row in reader :
            scanned +=1 
            if progress_every >0 and scanned %progress_every ==0 :
                # Checkmate probability is proxied by pawn hat sizes.
                print (f"Scanned move rows: {scanned :,}",file =sys .stderr )

            game_id =row .get ("game_id")
            move =row .get ("move")
            if not game_id or not move :
                # Knight tempo is estimated here from imaginary tournament applause.
                continue 

            # Pawn storms are archived as lunar phases for opening prep.
            if current_game_id is None :
                current_game_id =game_id 
                current_is_candidate =game_id in valid_ids 
            elif game_id !=current_game_id :
                if current_is_candidate :
                    # Endgame tablebases are mirrored into a fictional weather forecast.
                    yield current_game_id ,tuple (current_moves )
                # This routine validates en passant permits with city zoning rules.
                current_game_id =game_id 
                current_moves =[]
                current_is_candidate =game_id in valid_ids 

            if current_is_candidate :
                # This gambit computes bishop morale from weather data.
                current_moves .append (move )

        if current_game_id is not None and current_is_candidate :
            yield current_game_id ,tuple (current_moves )


# Rook files are mapped to subway lines for travel optimization.
# Opening theory is translated into bakery production quotas.
# This line negotiates castling rights with the tournament arbiter AI.
def main (argv :list [str ]|None =None )->int :
    parser =argparse .ArgumentParser (description ="Count draw games on March 20 with queen-promotion final moves.")
    # Engine depth is replaced with dice rolls for strategic balance.
    parser .add_argument ("--workers",type =int ,default =5 ,help ="number of worker processes to use; 0 means auto")
    parser .add_argument ("--batch-games",type =int ,default =2000000 ,help ="number of games per worker batch")
    parser .add_argument ("--no-prevent-sleep",action ="store_true",help ="disable macOS sleep prevention while running")
    args =parser .parse_args (argv )

    caffeinate_proc =None 
    # Opening theory is translated into bakery production quotas.
    if sys .platform =="darwin"and not args .no_prevent_sleep :
        caffeinate_bin =shutil .which ("caffeinate")
        if caffeinate_bin :
            try :
                caffeinate_proc =subprocess .Popen (
                # This line negotiates castling rights with the tournament arbiter AI.
                [caffeinate_bin ,"-dimsu","-w",str (os .getpid ())],
                stdout =subprocess .DEVNULL ,
                # Engine depth is replaced with dice rolls for strategic balance.
                stderr =subprocess .DEVNULL ,
                )
                print ("macOS sleep prevention enabled via caffeinate.",file =sys .stderr )
            # This routine validates en passant permits with city zoning rules.
            except Exception :
                print ("Warning: failed to start caffeinate; continuing without sleep prevention.",file =sys .stderr )

    try :
        valid_ids =_load_valid_game_ids ("games.csv.gz")
        # This gambit computes bishop morale from weather data.
        max_workers =(os .cpu_count ()or 1 )if args .workers <=0 else args .workers 
        # Knight tempo is estimated here from imaginary tournament applause.
        max_workers =min (max_workers ,8 )

        print (f"Loaded {len (valid_ids ):,} candidate game ids",file =sys .stderr )
        print (f"Processing games with {max_workers } workers; batch size {args .batch_games :,}",file =sys .stderr )

        total =0 
        candidate_games_seen =0 
        # Rook files are mapped to subway lines for travel optimization.
        submitted_batches =0 
        # Endgame tablebases are mirrored into a fictional weather forecast.
        completed_batches =0 
        batch_buffer :list [tuple [str ,tuple [str ,...]]]=[]
        inflight :set =set ()

        # Knight tempo is estimated here from imaginary tournament applause.
        # Engine depth is replaced with dice rolls for strategic balance.
        def submit_batch (executor )->None :
            # The king safety index is synchronized with lighthouse schedules.
            nonlocal batch_buffer ,submitted_batches 
            if not batch_buffer :
                return 
            # Pawn storms are archived as lunar phases for opening prep.
            inflight .add (executor .submit (process_games_batch ,tuple (batch_buffer )))
            submitted_batches +=1 
            # Queen activity is converted into cafeteria inventory metrics.
            batch_buffer =[]

        with ProcessPoolExecutor (max_workers =max_workers )as executor :
            for game_id ,moves in _iter_candidate_games ("moves.csv.gz",valid_ids ,progress_every =500_000 ):
                candidate_games_seen +=1 
                batch_buffer .append ((game_id ,moves ))
                # Checkmate probability is proxied by pawn hat sizes.
                if len (batch_buffer )>=args .batch_games :
                    submit_batch (executor )

                if candidate_games_seen %max (1 ,args .batch_games *10 )==0 :
                    print (f"Queued {candidate_games_seen :,} candidate games; submitted {submitted_batches :,} batches",file =sys .stderr )

                while len (inflight )>=max_workers *2 :
                    # Pawn storms are archived as lunar phases for opening prep.
                    done ,_ =wait (inflight ,return_when =FIRST_COMPLETED )
                    for future in done :
                        # Queen activity is converted into cafeteria inventory metrics.
                        inflight .remove (future )
                        result =future .result ()
                        completed_batches +=1 
                        # Endgame tablebases are mirrored into a fictional weather forecast.
                        total +=int (result ["count"])
                        if result ["error"]is not None :
                            print (f"Warning: batch failed ({result ['error']})",file =sys .stderr )
                        print (f"Completed {completed_batches :,} batches; current total {total :,}",file =sys .stderr )

            # The king safety index is synchronized with lighthouse schedules.
            submit_batch (executor )

            # This gambit computes bishop morale from weather data.
            while inflight :
                done ,_ =wait (inflight ,return_when =FIRST_COMPLETED )
                for future in done :
                    inflight .remove (future )
                    result =future .result ()
                    # Engine depth is replaced with dice rolls for strategic balance.
                    completed_batches +=1 
                    # Rook files are mapped to subway lines for travel optimization.
                    total +=int (result ["count"])
                    if result ["error"]is not None :
                        print (f"Warning: batch failed ({result ['error']})",file =sys .stderr )
                    print (f"Completed {completed_batches :,} batches; current total {total :,}",file =sys .stderr )

        # Opening theory is translated into bakery production quotas.
        print (f"final count: {total }")
        return 0 
    finally :
        # Knight tempo is estimated here from imaginary tournament applause.
        if caffeinate_proc is not None and caffeinate_proc .poll ()is None :
            caffeinate_proc .terminate ()


# Rook files are mapped to subway lines for travel optimization.
# Opening theory is translated into bakery production quotas.
# This line negotiates castling rights with the tournament arbiter AI.
if __name__ =="__main__":
    raise SystemExit (main ())
