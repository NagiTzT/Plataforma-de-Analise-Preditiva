"""Bounded prospective covariates, isolated from operational predictions.

Reads the complete frozen market pool, balances acquisition across eligibility
groups and fills both teams' recent samples before starting another fixture.
Only statistics of finished previous matches are requested. The operational
database's team profiles, snapshots, picks and models are never updated.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import closing
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
from pathlib import Path
import sqlite3
import time

from allsports_api import fetch_recent_team_events_for_game, match_resource_url
from phase2_observations import append
from phase4_metadata import repair_frozen_metadata
from sofascore_intelligence import (
    _fixture_identity_matches, _profile_observed_metrics, _recent_event_rows,
    _statistics_metrics, _team_identity_name,
)

CONTEXT_VERSION = "phase4-covariates-v2"
MIN_SAMPLES = 3
METRICS = ("xg", "shots_on_target", "big_chances")
HISTORY_TTL = 12 * 3600
PARTIAL_METRICS_TTL = 24 * 3600


def _readonly(path):
    conn = sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True, timeout=15)
    conn.row_factory = sqlite3.Row
    return conn


def candidate_pool(db_path, now):
    sidecar = Path(db_path).resolve().with_name("phase2_observations.db")
    start = datetime.fromtimestamp(now, timezone(timedelta(hours=-3))).replace(
        hour=0, minute=0, second=0, microsecond=0).timestamp()
    with closing(_readonly(sidecar)) as conn:
        stages = list(conn.execute("""SELECT run_id,stage,json_extract(payload,'$.status') status
            FROM observations WHERE stage IN ('collection_started','collection_finished',
            'phase4_run_health_v1','phase4_shadow_health_v1','phase4_pipeline_error_v1')
            AND observed_at>=?""", (start,)))
        valid, errors = {}, set()
        for row in stages:
            if row['stage'] == 'phase4_pipeline_error_v1':
                errors.add(row['run_id'])
            elif row['stage'].endswith('health_v1') and row['status'] != 'PASS':
                continue
            else:
                valid.setdefault(row['run_id'], set()).add(row['stage'])
        required = {'collection_started','collection_finished','phase4_run_health_v1','phase4_shadow_health_v1'}
        runs = {run for run, found in valid.items() if required <= found and run not in errors}
        rows = conn.execute("""SELECT run_id,match_id,sha256,payload FROM observations
            WHERE stage='phase4_prefilter_v1' AND observed_at>=? ORDER BY observed_at""", (start,)).fetchall()
        done = {(r['run_id'], r['match_id']) for r in conn.execute(
            "SELECT run_id,match_id FROM observations WHERE stage='phase4_covariates_v2'")}
    games, seen = [], set()
    for row in rows:
        if row['run_id'] not in runs or (row['run_id'],row['match_id']) in done:
            continue
        if hashlib.sha256(row['payload'].encode()).hexdigest() != row['sha256']:
            continue
        d = json.loads(row['payload'])
        kickoff = float(d.get('event_time') or 0)
        if kickoff <= now or row['match_id'] in seen:
            continue
        seen.add(row['match_id'])
        games.append({'ID':row['match_id'], 'Timestamp':kickoff,
            'Time Casa':d.get('home_team'), 'Time Fora':d.get('away_team'),
            'Liga':d.get('league'), 'Research_Run_ID':row['run_id'],
            'Eligibility_Group':d.get('eligibility_group'), 'Parent_SHA256':row['sha256']})
    return games


def balanced_order(games):
    groups = [[], []]
    for game in games:
        group = 0 if game.get('Eligibility_Group') == 'A_ELIGIBLE' else 1
        key = hashlib.sha256((str(game.get('Research_Run_ID')) + ':' + str(game['ID'])).encode()).hexdigest()
        groups[group].append((key, game))
    for group in groups:
        group.sort(key=lambda item:item[0])
    ordered = []
    for i in range(max(map(len, groups), default=0)):
        for group in groups:
            if i < len(group):
                ordered.append(group[i][1])
    return ordered


def _safe_history(rows, team_name, target_id, now):
    valid = []
    seen = set()
    for row in rows:
        if not isinstance(row,dict):
            continue
        mid = str(row.get('match_id') or '')
        try:
            played = float(row.get('start_timestamp') or 0)
        except (TypeError,ValueError):
            continue
        if (not mid or mid==str(target_id) or mid in seen or not played
                or played + 3*3600 >= now or now-played > 180*86400
                or _team_identity_name(row.get('team_name')) != _team_identity_name(team_name)):
            continue
        seen.add(mid); valid.append(row)
    return sorted(valid,key=lambda row:row['start_timestamp'],reverse=True)[:5]


def _history(main, cache, game, side, now, get, host):
    name = game['Time Casa' if side=='home' else 'Time Fora']
    key = str(game.get('Liga') or '') + ':' + _team_identity_name(name)
    try:
        row = main.execute("""SELECT provider,captured_at,cutoff_timestamp,games_json
            FROM pregame_recent_form_snapshots WHERE match_id=? AND side=?""",
            (str(game['ID']),side)).fetchone()
    except sqlite3.OperationalError:
        row = None
    if (row and row['provider'] in {'allsports','sofascore'}
            and 0<=now-row['captured_at']<HISTORY_TTL
            and abs(float(row['cutoff_timestamp'])-game['Timestamp'])<=300):
        history = _safe_history(json.loads(row['games_json']),name,game['ID'],now)
        if len(history)>=3:
            return history,'operational_recent_snapshot'
    row = cache.execute('SELECT fetched_at,rows_json FROM recent_history WHERE team_key=?',(key,)).fetchone()
    if row and 0<=now-row[0]<HISTORY_TTL:
        history = _safe_history(json.loads(row[1]),name,game['ID'],now)
        return history,'research_history_cache'
    if get is None:
        return [],'history_not_observed'
    loaded = fetch_recent_team_events_for_game(host,game,side,get) or {}
    rows = _recent_event_rows(loaded.get('payload'),loaded.get('team_id'),name,int(now),limit=5)
    history = _safe_history(rows,name,game['ID'],now)
    cache.execute('INSERT OR REPLACE INTO recent_history VALUES (?,?,?)',
        (key,now,json.dumps(history,ensure_ascii=False)))
    cache.commit()
    return history,'allsports_recent' if history else 'history_unavailable'


def _profile_metrics(main, previous, now):
    columns = [f'{metric}_{direction}' for metric in METRICS for direction in ('for','against')]
    try:
        rows = main.execute('SELECT team_name,captured_at,metric_presence_json,'+
            ','.join(columns)+' FROM sofascore_team_match_profiles WHERE match_id=?',
            (str(previous['match_id']),)).fetchall()
    except sqlite3.OperationalError:
        return {}
    for row in rows:
        if (row['captured_at']>now or _team_identity_name(row['team_name'])!=_team_identity_name(previous['team_name'])):
            continue
        values = {k:row[k] for k in columns}
        observed = _profile_observed_metrics(values,row['metric_presence_json'])
        return {k:values[k] for k in observed if k in values and values[k] is not None}
    return {}


def _previous_metrics(main,cache,previous,now,get,host,summary):
    result = _profile_metrics(main,previous,now)
    if all(f'{metric}_{d}' in result for metric in METRICS for d in ('for','against')):
        summary['main_profile_hits']+=1
        return result,'local_measured_profile'
    mid=str(previous['match_id'])
    cached=cache.execute('SELECT fetched_at,metrics_json FROM previous_metrics WHERE match_id=?',(mid,)).fetchone()
    parsed={}
    if cached and cached[0]<=now:
        parsed=json.loads(cached[1])
    cache_fresh = bool(cached and 0<=now-cached[0]<PARTIAL_METRICS_TTL)
    cache_complete = bool(cached and all(parsed.get(f'{s}_{m}_available')
        for s in ('home','away') for m in METRICS))
    if cached and cached[0]<=now and (cache_fresh or cache_complete):
        summary['research_metric_cache_hits']+=1
    else:
        try:
            row=main.execute("""SELECT payload_json,fetched_at FROM sofascore_http_cache
                WHERE match_id=? AND resource='statistics' AND http_status=200""",(mid,)).fetchone()
        except sqlite3.OperationalError:
            row=None
        if row and row['fetched_at']<=now:
            parsed,_=_statistics_metrics(json.loads(row['payload_json']))
            summary['main_statistics_cache_hits']+=1
        if get is not None and not all(parsed.get(f'{s}_{m}_available') for s in ('home','away') for m in METRICS):
            payload=get(match_resource_url(host,mid,'statistics'))
            extra,_=_statistics_metrics(payload)
            for key,present in extra.items():
                if key.endswith('_available') and present and not parsed.get(key):
                    parsed[key]=1.0;parsed[key[:-10]]=extra[key[:-10]]
            cache.execute('INSERT OR REPLACE INTO previous_metrics VALUES (?,?,?)',
                (mid,now,json.dumps(parsed)))
            cache.commit()
    own='home' if previous.get('is_home') else 'away';other='away' if own=='home' else 'home'
    for metric in METRICS:
        for direction,side in [('for',own),('against',other)]:
            k=f'{metric}_{direction}'
            if k not in result and parsed.get(f'{side}_{metric}_available'):
                result[k]=float(parsed[f'{side}_{metric}'])
    return result,'measured_previous_fixture'


def collect_context(db_path, *, games=None, api_get=None, host='allsportsapi2.p.rapidapi.com',
                    max_requests=60, max_games=12, clock=time.time):
    """No HTTP is performed without an explicit callback and positive budget."""
    now=float(clock());games=candidate_pool(db_path,now) if games is None else list(games)
    summary=Counter(candidates=len(games),requests_attempted=0,snapshots_saved=0)
    dest=Path(db_path).resolve().with_name('phase4_context.db')
    with closing(sqlite3.connect(dest,timeout=10)) as cache, closing(_readonly(db_path)) as main:
        cache.execute('CREATE TABLE IF NOT EXISTS recent_history(team_key TEXT PRIMARY KEY,fetched_at REAL,rows_json TEXT)')
        cache.execute('CREATE TABLE IF NOT EXISTS previous_metrics(match_id TEXT PRIMARY KEY,fetched_at REAL,metrics_json TEXT)')
        cache.commit()
        for index,game in enumerate(balanced_order(games)):
            if clock()>=float(game['Timestamp']):
                summary['already_started']+=1;continue
            network_allowed=api_get is not None and index<max(0,max_games)
            def get(url):
                if (not network_allowed or summary['requests_attempted']>=max(0,max_requests)
                        or clock()>=float(game['Timestamp'])):
                    return None
                summary['requests_attempted']+=1
                try:
                    data=api_get(url)
                except Exception:
                    summary['source_errors']+=1;return None
                # Verify the current fixture used only to resolve provider IDs.
                if url.endswith('/match/'+str(game['ID'])):
                    event=(data or {}).get('event') if isinstance(data,dict) else None
                    if not isinstance(event,dict) or not _fixture_identity_matches(
                        game['Time Casa'],game['Time Fora'],game['Timestamp'],
                        (event.get('homeTeam') or {}).get('name',''),
                        (event.get('awayTeam') or {}).get('name',''),event.get('startTimestamp'))[0]:
                        summary['identity_rejections']+=1;return None
                return data
            features={};evidence={};histories={}
            for side in ('home','away'):
                rows,source=_history(main,cache,game,side,float(clock()),get if network_allowed else None,host)
                histories[side]=len(rows)
                samples={metric:[] for metric in METRICS}
                for previous in rows:
                    values,metric_source=_previous_metrics(main,cache,previous,float(clock()),
                        get if network_allowed else None,host,summary)
                    for metric in METRICS:
                        pair=(values.get(metric+'_for'),values.get(metric+'_against'))
                        if all(v is not None and math.isfinite(v) for v in pair):
                            samples[metric].append((str(previous['match_id']),previous['start_timestamp'],*pair,metric_source))
                    if all(len(v)>=MIN_SAMPLES for v in samples.values()):
                        break
                for metric,measured in samples.items():
                    features[f'research_{side}_{metric}_games']=len(measured)
                    evidence[f'{side}_{metric}']=[{'match_id':m[0],'event_time':m[1],
                        'source':m[4],'team_name':game['Time Casa' if side=='home' else 'Time Fora']}
                        for m in measured]
                    for j,direction in [(2,'for'),(3,'against')]:
                        if measured:
                            weights=[.85**i for i in range(len(measured))]
                            features[f'research_{side}_{metric}_{direction}']=sum(v[j]*w for v,w in zip(measured,weights))/sum(weights)
            families={metric:min(features.get(f'research_{side}_{metric}_games',0) for side in ('home','away'))>=MIN_SAMPLES for metric in METRICS}
            families['team_form']=min(histories.values(),default=0)>=MIN_SAMPLES
            if not any(families[m] for m in METRICS):
                summary['incomplete_pairs']+=1;continue
            captured=float(clock())
            if captured>=float(game['Timestamp']):
                summary['finished_after_kickoff']+=1;continue
            payload={'version':CONTEXT_VERSION,'parent_stage':'phase4_prefilter_v1',
                'parent_sha256':game['Parent_SHA256'],'captured_at':captured,
                'kickoff':game['Timestamp'],'eligibility_group':game['Eligibility_Group'],
                'data_coverage':{'families':families,'minimum_samples_per_team':MIN_SAMPLES},
                'features':features,'evidence':evidence,
                'separate_from_champion_inputs':True,'affects_production':False,
                'missing_reasons':{m:'fewer_than_three_measured_previous_matches_per_team'
                                   for m in METRICS if not families[m]}}
            if append(db_path,game['Research_Run_ID'],str(game['ID']),'phase4_covariates_v2',payload):
                summary['snapshots_saved']+=1
                summary['eligible_saved' if game['Eligibility_Group']=='A_ELIGIBLE' else 'non_eligible_saved']+=1
                for m in METRICS:summary[m+'_available']+=int(families[m])
            else:
                summary['append_failed']+=1
    return dict(summary)


def claim_daily_network_collection(db_path, now, max_requests, max_games):
    """A durable, conservative cap: no HTTP replay even after interruption."""
    day = datetime.fromtimestamp(now,timezone(timedelta(hours=-3))).date().isoformat()
    run_id='phase4-covariates:'+day
    sidecar=Path(db_path).resolve().with_name('phase2_observations.db')
    with closing(sqlite3.connect(sidecar,timeout=10)) as conn, conn:
        # Serialize the check/claim across parallel processes using the same
        # SQLite write lock as append(). Never update an observation.
        conn.execute('BEGIN IMMEDIATE')
        exists=conn.execute("SELECT 1 FROM observations WHERE run_id=? AND stage='phase4_covariate_collection_started_v2'",(run_id,)).fetchone()
        if exists:
            return run_id,False
        payload={'version':CONTEXT_VERSION,'date_brt':day,
            'max_requests':max_requests,'max_games':max_games,
            'affects_production':False,'interrupted_run_blocks_same_day_http_replay':True}
        raw=json.dumps(payload,ensure_ascii=False,sort_keys=True,allow_nan=False)
        conn.execute('INSERT INTO observations VALUES (?,?,?,?,?,?)',
            (run_id,'','phase4_covariate_collection_started_v2',now,
             hashlib.sha256(raw.encode()).hexdigest(),raw))
    return run_id,True


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--db',default='ia_sports_v5.db')
    parser.add_argument('--max-requests',type=int,default=60)
    parser.add_argument('--max-games',type=int,default=12)
    parser.add_argument('--repair-metadata',action='store_true')
    args=parser.parse_args()
    summary={}
    if args.repair_metadata:
        summary['metadata']=repair_frozen_metadata(args.db)
    collection_run=None
    if args.max_requests>0:
        collection_run,claimed=claim_daily_network_collection(args.db,time.time(),
            args.max_requests,args.max_games)
        if not claimed:
            args.max_requests=0
            summary['http_skipped']='daily_collection_already_claimed_use_local_cache_only'
    if args.max_requests>0:
        import robo_auto
        # Each call is bounded even when the existing helper rotates keys.
        def guarded_get(url):
            remaining=args.max_requests-robo_auto._api_metrics['http_requests']
            if remaining<=0:return None
            return robo_auto.safe_api_get(url,max_retries=1,max_http_requests=remaining)
        api_get=guarded_get
    else:
        api_get=None
    summary['context']=collect_context(args.db,api_get=api_get,
        max_requests=args.max_requests,max_games=args.max_games)
    if api_get is not None:
        summary['context']['actual_http_requests']=robo_auto._api_metrics['http_requests']
        append(args.db,collection_run,'','phase4_covariate_collection_finished_v2',
            {'version':CONTEXT_VERSION,'summary':summary,'outcomes_opened':False})
    print(json.dumps(summary,ensure_ascii=True))
