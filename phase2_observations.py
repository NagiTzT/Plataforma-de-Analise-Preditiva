"""Append-only research observations. Failures must never change live decisions.

The observed timestamp is NOT a vendor quote timestamp. No market benchmark is
permitted solely because a previously cached price was observed before kickoff.
This sidecar never updates prediction, ticket, or training tables.
"""
import hashlib
from contextlib import closing
import json
import logging
import math
from pathlib import Path
import sqlite3
import threading
import time
import uuid

_cooldown = {}
_append_lock = threading.RLock()


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if hasattr(value, 'item'):
        return clean(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def new_run():
    return str(uuid.uuid4())


def market_probabilities(odds, observed_at, kickoff, suspended=False):
    """Benchmark of prices actually observed pregame, NOT reconstructed closing odds."""
    try:
        odds=[float(v) for v in odds]
        if (suspended or len(odds)!=3 or any(not math.isfinite(v) or v<=1 for v in odds)
                or not kickoff or not 0<float(observed_at)<float(kickoff)):
            return None
        implied=[1/v for v in odds]; total=sum(implied)
        return {'probabilities':[v/total for v in implied], 'overround':total-1,
                'method':'proportional_margin_removal', 'quote_age_unknown':True,
                'benchmark':'as_observed_pregame_not_closing_price'}
    except (TypeError,ValueError,OverflowError):
        return None


def observe_filter(db, before, after, stage):
    try:
        retained={str(g['ID']) for g in after}
        for g in before:
            if not g.get('Research_Run_ID'): continue
            append(db,g['Research_Run_ID'],str(g['ID']),stage,{
                'retained':str(g['ID']) in retained,
                'reason':None if str(g['ID']) in retained else stage,
                'features':None, 'feature_status':'not_evaluated_at_filter',
            })
    except Exception as exc:
        logging.warning('Phase2 filter capture failed (%s)',type(exc).__name__)


def finish_run(db, source_games, evaluated, approved, groups, operational_run_id, model_version, model_id):
    """Append decisions; missing inference is explicit, not fabricated features."""
    try:
        approved_ids = {str(g['ID']) for g in approved}
        evaluated_ids = {str(g['ID']) for g in evaluated}
        tickets = {str(g['ID']): group.get('Nome') for group in groups for g in group['Jogos']}
        by_run = {}
        for g in source_games:
            run = g.get('Research_Run_ID')
            if not run:
                continue
            mid = str(g['ID'])
            by_run.setdefault(run, []).append(mid)
            append(db, run, mid, 'decision', {
                'operational_run_id': operational_run_id,
                'model_version': model_version, 'model_id': model_id,
                'evaluated': mid in evaluated_ids, 'approved': mid in approved_ids,
                'quality_score': None, 'experimental_quality_gate': 'DISABLED',
                'reason': ('approved' if mid in approved_ids else
                           'operational_rule' if mid in evaluated_ids else 'no_prediction'),
                'ticket_id': tickets.get(mid),
            })
        for run, mids in by_run.items():
            ids = set(mids)
            append(db, run, '', 'run_finished', {
                'operational_run_id': operational_run_id,
                'model_version': model_version, 'model_id': model_id,
                'model_evaluated': len(ids & evaluated_ids),
                'operational_approved': len(ids & approved_ids),
                'operational_rejected': len((ids & evaluated_ids) - approved_ids),
                'quality_approved': None, 'quality_rejected': None,
                'quality_gate': 'DISABLED',
                'tickets_generated': len({tickets[m] for m in ids if m in tickets}),
                'leftovers': len((ids & approved_ids) - tickets.keys()),
            })
    except Exception as exc:
        logging.warning('Phase2 run not saved (%s)', type(exc).__name__)


def append(db, run_id, match_id, stage, payload):
    """First event at this key wins. Return False on failure/conflicting replay."""
    key = str(db)
    if time.monotonic() < _cooldown.get(key, 0):
        return False
    try:
        dest = Path(db).resolve().with_name('phase2_observations.db')
        raw = json.dumps(clean(payload), ensure_ascii=False, sort_keys=True, allow_nan=False)
        digest = hashlib.sha256(raw.encode()).hexdigest()
        with _append_lock:
            with closing(sqlite3.connect(dest, timeout=2.0)) as conn, conn:
                conn.execute('PRAGMA busy_timeout=2000')
                conn.execute('''CREATE TABLE IF NOT EXISTS observations (
                    run_id TEXT NOT NULL, match_id TEXT NOT NULL, stage TEXT NOT NULL,
                    observed_at REAL NOT NULL, sha256 TEXT NOT NULL, payload TEXT NOT NULL,
                    PRIMARY KEY(run_id, match_id, stage))''')
                conn.execute('''CREATE TRIGGER IF NOT EXISTS observations_no_update
                    BEFORE UPDATE ON observations BEGIN SELECT RAISE(ABORT, 'immutable'); END''')
                conn.execute('''CREATE TRIGGER IF NOT EXISTS observations_no_delete
                    BEFORE DELETE ON observations BEGIN SELECT RAISE(ABORT, 'immutable'); END''')
                existing = conn.execute('SELECT sha256 FROM observations WHERE run_id=? AND match_id=? AND stage=?',
                                        (str(run_id), str(match_id), stage)).fetchone()
                if existing:
                    return existing[0] == digest
                conn.execute('INSERT INTO observations VALUES (?,?,?,?,?,?)',
                             (str(run_id), str(match_id), stage, time.time(), digest, raw))
        return True
    except Exception as exc:
        _cooldown[key] = time.monotonic() + 10
        logging.warning('Phase2 observation not saved (%s); live decision unchanged', type(exc).__name__)
        return False
