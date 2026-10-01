"""Append-only recent-form evidence alongside the current operational cache.

Refreshing a cache must not erase the source that existed at prediction time.
The original capture timestamp is preserved; archival time is not backdated
into a claim that newly fetched data existed before a radar.
"""
import hashlib
import json


COLUMNS = ('match_id','side','provider','provider_team_id','captured_at',
           'cutoff_timestamp','freshest_match_timestamp','games_json','features_json','coverage')
CORE = ('match_id','side','provider','provider_team_id','captured_at','cutoff_timestamp','games_json')


def init_archive(conn):
    conn.execute('''CREATE TABLE IF NOT EXISTS pregame_recent_form_archive (
        evidence_hash TEXT PRIMARY KEY,
        match_id TEXT NOT NULL, side TEXT NOT NULL, provider TEXT NOT NULL,
        provider_team_id TEXT, captured_at INTEGER NOT NULL,
        cutoff_timestamp INTEGER NOT NULL, freshest_match_timestamp INTEGER,
        games_json TEXT NOT NULL, features_json TEXT NOT NULL, coverage REAL NOT NULL DEFAULT 0
    )''')
    conn.execute('''CREATE INDEX IF NOT EXISTS idx_recent_form_archive_target_capture
        ON pregame_recent_form_archive(match_id,side,captured_at)''')


def save_recent_snapshot(conn, values):
    """One caller-owned transaction: archive old/new evidence, then refresh."""
    values=tuple(values)
    if len(values)!=len(COLUMNS):
        raise ValueError('Unexpected recent-form snapshot schema')
    columns=','.join(COLUMNS)
    previous=conn.execute(f'SELECT {columns} FROM pregame_recent_form_snapshots WHERE match_id=? AND side=?',
                          values[:2]).fetchone()
    for observation in ([tuple(previous)] if previous is not None else [])+[values]:
        digest=hashlib.sha256(json.dumps(observation,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()
        conn.execute(f'INSERT OR IGNORE INTO pregame_recent_form_archive (evidence_hash,{columns}) '
                     f'VALUES ({",".join("?" for _ in range(len(COLUMNS)+1))})',(digest,*observation))
    conn.execute(f'INSERT OR REPLACE INTO pregame_recent_form_snapshots ({columns}) '
                 f'VALUES ({",".join("?" for _ in COLUMNS)})',values)


def read_recent_snapshots(conn, columns=CORE):
    """Read-only union, including legacy databases without an archive table."""
    tables={r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    seen=set(); result=[]
    for table in ('pregame_recent_form_snapshots','pregame_recent_form_archive'):
        if table not in tables:
            continue
        for values in conn.execute(f'SELECT {",".join(columns)} FROM {table}'):
            values=tuple(values)
            if values not in seen:
                seen.add(values)
                result.append(dict(zip(columns,values)))
    return result


def snapshot_index(snapshots):
    result={}
    for snapshot in snapshots:
        result.setdefault((str(snapshot['match_id']),snapshot['side']),[]).append(snapshot)
    return result


def snapshot_asof(index,row,side):
    """Latest matching pregame observation, refusing same-time contradictions."""
    eligible=[s for s in index.get((str(row['match_id']),side),[])
        if s.get('provider') in ('sofascore','allsports') and s.get('captured_at')
        and s['captured_at']<=row['captured_at'] and s['captured_at']<row['kickoff']
        and s.get('cutoff_timestamp')==row['kickoff']]
    if not eligible:
        return None
    newest=max(s['captured_at'] for s in eligible)
    eligible=[s for s in eligible if s['captured_at']==newest]
    if len({(s['provider'],s['provider_team_id'],s['games_json']) for s in eligible})!=1:
        return None
    return eligible[0]
