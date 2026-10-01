"""Capture free context for existing future games without changing their picks."""
import argparse
import json
import sqlite3
import time
from contextlib import closing
from pathlib import Path
from free_football_data import enrich_games


def run(db,output):
    destination=Path(output)
    if destination.exists():
        raise RuntimeError('Choose a new output file')
    with closing(sqlite3.connect(Path(db).resolve().as_uri()+'?mode=ro',uri=True)) as conn:
        rows=conn.execute('SELECT match_id,confronto,liga,start_timestamp FROM previsoes WHERE start_timestamp>?',
                          (int(time.time()),)).fetchall()
    games=[{'ID':mid,'Time Casa':name.split(' vs ',1)[0],'Time Fora':name.split(' vs ',1)[1],
            'Liga':league,'Timestamp':start} for mid,name,league,start in rows if ' vs ' in name]
    summary=enrich_games(db,games)
    summary['note']='New captures only; existing ML snapshots, results, tickets and Telegram untouched'
    destination.parent.mkdir(parents=True,exist_ok=True)
    destination.write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--db',default='ia_sports_v5.db')
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    run(args.db,args.output)
