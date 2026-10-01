"""Record a locked learned challenger on new games, never change published picks."""
import json
import logging
import math
import sqlite3
import time
from contextlib import closing
from functools import lru_cache
from pathlib import Path

import joblib

ARTIFACT = Path(__file__).parent/'relatorios/correcao_20260911/cenarios/gate_25.joblib'


@lru_cache(maxsize=2)
def _artifact(path, mtime):
    result = joblib.load(path)
    if result.get('mode')!='shadow_only' or result.get('version')!='pick-scenarios-v1':
        raise ValueError('Not a compatible locked shadow model')
    return result


def record_shadow(db_path,match_id,kickoff,features,analysis,artifact_path=ARTIFACT):
    """Future-only snapshot; all failures are isolated from the active radar."""
    try:
        now=int(time.time())
        if not kickoff or now>=int(kickoff):
            return {'recorded':False,'reason':'not_pregame'}
        path=Path(artifact_path)
        if not path.is_file():
            return {'recorded':False,'reason':'no_artifact'}
        artifact=_artifact(str(path),path.stat().st_mtime_ns)
        if artifact['training_available_until']>=now:
            return {'recorded':False,'reason':'training_after_prediction'}
        row={'features':features,'base':analysis['base_probabilities'],'current':analysis['probabilities']}
        probabilities=artifact['model'].predict_proba([row])[0].tolist()
        if len(probabilities)!=3 or not all(math.isfinite(p) and p>=0 for p in probabilities):
            return {'recorded':False,'reason':'invalid_probabilities'}
        with closing(sqlite3.connect(db_path,timeout=10)) as conn:
            conn.execute('''CREATE TABLE IF NOT EXISTS pick_scenario_shadow (
                match_id TEXT PRIMARY KEY,captured_at INTEGER,kickoff INTEGER,
                candidate TEXT,training_available_until INTEGER,probabilities_json TEXT,
                original_probabilities_json TEXT,features_json TEXT)''')
            conn.execute('INSERT OR IGNORE INTO pick_scenario_shadow VALUES (?,?,?,?,?,?,?,?)',
                (str(match_id),now,int(kickoff),artifact['selected'],artifact['training_available_until'],
                 json.dumps(probabilities),json.dumps(row['current']),json.dumps(features)))
            conn.commit()
        return {'recorded':True,'candidate':artifact['selected']}
    except Exception as exc:
        logging.warning('Sombra de decisão indisponível (%s); pick ativo preservado',type(exc).__name__)
        return {'recorded':False,'reason':type(exc).__name__}
