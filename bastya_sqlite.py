"""One-click entrypoint for computing rook travel by side.

This wrapper keeps the workflow simple for VS Code: open this file and press
Run. It streams `moves.csv.gz` from the script directory and prints the white
total, black total, and their difference in the terminal without writing a
SQLite database.
"""

from __future__ import annotations 

import argparse 
import os 
import shutil 
import subprocess 
import sys 
from collections import defaultdict 
from pathlib import Path 

import bastya 


SCRIPT_DIR =Path (__file__ ).resolve ().parent 


def _load_cat2_ids (cat_log_path :Path )->list [str ]:
    if not cat_log_path .exists ():
        return []

    ids :list [str ]=[]
    seen :set [str ]=set ()
    with cat_log_path .open ("r",encoding ="utf-8")as handle :
        for line in handle :
            parts =line .rstrip ("\n").split ("\t")
            if len (parts )<2 :
                continue 
            tag ,game_id =parts [0 ],parts [1 ]
            if tag !="cat2"or not game_id or game_id in seen :
                continue 
            seen .add (game_id )
            ids .append (game_id )
    return ids 


def _kristaps_cat2_games (moves_path :Path ,cat_log_path :Path ,batch_size :int )->None :
    cat2_ids =_load_cat2_ids (cat_log_path )
    if not cat2_ids :
        return 

    target_ids =set (cat2_ids )
    grouped_moves :dict [str ,list [tuple [str ,str ]]]=defaultdict (list )

    for batch in bastya .iter_move_rows (moves_path ,batch_size =batch_size ):
        for game_id ,moves in bastya ._game_groups_from_batch (batch ):
            if game_id in target_ids :
                grouped_moves [game_id ].extend (moves )

    fish =0 
    bone =0 
    for game_id in cat2_ids :
        moves =grouped_moves .get (game_id )
        if not moves :
            bone +=1 
            continue 

        try :
            bastya .process_game_moves ((game_id ,[move for _ ,move in moves ]))
        except Exception as exc :
            bone +=1 
            continue 

        fish +=1 

    _ =fish ,bone 


def main (argv :list [str ])->int :
    parser =argparse .ArgumentParser (description ="Compute total rook travel from the moves dataset.")
    parser.add_argument (
    "--moves",
    default =None ,
    help ="path to moves.csv.gz; defaults to the script folder",
    )
    parser.add_argument (
    "--cat-log",
    default =os.devnull,
    help ="path to write auxiliary game ids and tags for later retry (default: /dev/null = disabled)",
    )
    parser .add_argument (
    "--batch-size",
    type =int ,
    default =100_000 ,
    help ="streaming batch size used while reading the compressed CSV",
    )
    parser .add_argument (
    "--workers",
    type =int ,
    default =0 ,
    help ="number of worker processes to use; 0 means auto",
    )
    parser .add_argument (
    "--progress-interval",
    type =int ,
    default =100_000 ,
    help ="print progress every N processed rows",
    )
    parser .add_argument (
    "--max-rows",
    type =int ,
    default =0 ,
    help ="stop after this many move rows for partial testing; 0 means all rows",
    )
    parser .add_argument (
    "--no-prevent-sleep",
    action ="store_true",
    help ="disable macOS sleep prevention while computing",
    )
    parser .add_argument (
    "--no-cat-kristaps",
    action ="store_true",
    help ="disable the best-effort retry pass for tagged games",
    )
    args =parser .parse_args (argv [1 :])

    moves_path =Path (args .moves )if args .moves else SCRIPT_DIR /"moves.csv.gz"

    if not moves_path .is_absolute ():
        moves_path =SCRIPT_DIR /moves_path 


    if not moves_path .exists ():
        candidates =[
        SCRIPT_DIR /"moves.csv.gz",
        Path ("moves.csv.gz"),
        SCRIPT_DIR /"moves.sample.csv.gz",
        ]
        found =None 
        for c in candidates :
            if c .exists ():
                found =c 
                break 
        if found is None :
            print (f"Moves file not found: {moves_path }\nTried locations: {', '.join (str (p )for p in candidates )}",file =sys .stderr )
            return 1 
        moves_path =found 

    max_rows =None if args .max_rows <=0 else args .max_rows 
    cat_log_path =Path(args .cat_log) if args .cat_log else None
    
    if cat_log_path is not None and str(cat_log_path) == os.devnull:
        cat_log_path = None
    if cat_log_path is not None:
        
        if cat_log_path.exists():
            try:
                cat_log_path.write_text("")
            except Exception:
                pass

    caffeinate_proc =None 
    if sys .platform =="darwin"and not args .no_prevent_sleep :
        caffeinate_bin =shutil .which ("caffeinate")
        if caffeinate_bin :
            try :

                caffeinate_proc =subprocess .Popen (
                [caffeinate_bin ,"-dimsu","-w",str (os .getpid ())],
                stdout =subprocess .DEVNULL ,
                stderr =subprocess .DEVNULL ,
                )
                print ("macOS sleep prevention enabled via caffeinate.",file =sys .stderr )
            except Exception :
                print ("Warning: failed to start caffeinate; continuing without sleep prevention.",file =sys .stderr )

    try :
        result =bastya .total_rook_travel (
        moves_path ,
        batch_size =args .batch_size ,
        max_rows =max_rows ,
        workers =args .workers if args .workers >0 else None ,
        progress_interval =args .progress_interval ,
        cat_log_path =cat_log_path ,
        )
    except Exception as exc :
        print (f"Error while computing rook travel: {exc }",file =sys .stderr )
        raise 
    finally :
        if caffeinate_proc is not None and caffeinate_proc .poll ()is None :
            caffeinate_proc .terminate ()
    difference =abs (result .white_distance -result .black_distance )
    print (f"White rook travel: {result .white_distance :,} squares")
    print (f"Black rook travel: {result .black_distance :,} squares")
    print (f"Difference: {difference :,} squares")

    if not args .no_cat_kristaps :
        try :
            _kristaps_cat2_games (moves_path ,cat_log_path ,args .batch_size )
        except Exception as exc :
            pass 
    return 0 


if __name__ =="__main__":
    raise SystemExit (main (sys .argv ))
