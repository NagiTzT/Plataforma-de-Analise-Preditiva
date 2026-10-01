"""Pure evidence cleaning. No requests, labels, odds or team-name guessing.

An event ID alone does not identify unique fixtures: providers can issue two
IDs for the same team/opponent, venue and kickoff. Conflicting observations
are quarantined, not resolved using the target outcome. Age is data freshness,
not a rest-days prediction feature. A finite age window is opt-in for research.
"""
from collections import Counter, defaultdict

from pregame_form_quality import number, sequence_features


def identity(value):
    return '' if value is None or isinstance(value, bool) else str(value).strip()


def clean_recent_history(events, before, target_id='', team_id='', max_age_days=None,
                         limit=10, end_buffer=3*3600):
    counts = Counter()
    accepted = []
    expected = identity(team_id)
    for item in events if isinstance(events,list) else []:
        counts['input_rows'] += 1
        if not isinstance(item,dict):
            counts['invalid_rows'] += 1
            continue
        mid = identity(item.get('match_id'))
        ts = number(item.get('start_timestamp'))
        gf,ga = number(item.get('goals_for')),number(item.get('goals_against'))
        home = item.get('is_home')
        if (not mid or ts is None or ts <= 0 or gf is None or ga is None
                or not gf.is_integer() or not ga.is_integer() or home not in (0,1)):
            counts['invalid_rows'] += 1
            continue
        own,opp = identity(item.get('team_id')),identity(item.get('opponent_id'))
        if (expected and own != expected) or (own and opp and own == opp):
            counts['identity_mismatch_rows'] += 1
            continue
        if mid == identity(target_id) or ts+end_buffer >= before:
            counts['not_available_before_prediction_rows'] += 1
            continue
        # Missing identities cannot support composite deduplication, but an
        # otherwise valid legacy row can still be deduplicated by event ID.
        composite = (own,opp,int(home),ts) if own and opp else None
        if composite is None:
            counts['missing_composite_identity_rows'] += 1
        row = dict(item,goals_for=gf,goals_against=ga,start_timestamp=ts)
        fact = (own,opp,int(home),ts,gf,ga)
        accepted.append((mid,composite,fact,row))
    by_id,by_fixture = defaultdict(set),defaultdict(set)
    for mid,fixture,fact,row in accepted:
        by_id[mid].add(fact)
        if fixture is not None:
            by_fixture[fixture].add(fact[-2:])
    bad_ids = {mid for mid,facts in by_id.items() if len(facts)>1}
    bad_fixtures = {fixture for fixture,scores in by_fixture.items() if len(scores)>1}
    # Propagate an ID contradiction to aliases of that fixture as well.
    bad_fixtures.update(fixture for mid,fixture,_,_ in accepted if mid in bad_ids and fixture is not None)
    chosen = {}
    for mid,fixture,fact,row in sorted(accepted,key=lambda item:item[0]):
        if mid in bad_ids or fixture in bad_fixtures:
            counts['conflicting_rows'] += 1
            continue
        key = ('fixture',fixture) if fixture is not None else ('event',mid)
        if key in chosen:
            counts['duplicate_rows'] += 1
            continue
        chosen[key] = row
    unique = sorted(chosen.values(),key=lambda row:(-row['start_timestamp'],identity(row['match_id'])))
    counts['unique_valid_rows'] = len(unique)
    counts['older_180d_rows'] = sum(before-r['start_timestamp']>180*86400 for r in unique)
    counts['older_365d_rows'] = sum(before-r['start_timestamp']>365*86400 for r in unique)
    if max_age_days is not None:
        recent = [r for r in unique if before-r['start_timestamp']<=max_age_days*86400]
        counts['stale_rows_excluded'] = len(unique)-len(recent)
    else:
        recent = unique
    rows = recent if limit is None else recent[:limit]
    counts['retained_rows'] = len(rows)
    counts['five_games_available'] = int(len(rows)>=5)
    return rows,dict(counts)


def fresh_sequence_features(events, side, before, target_id='', team_id=''):
    clean,quality = clean_recent_history(events,before,target_id,team_id,max_age_days=180)
    result = sequence_features(clean,side,before,target_id)
    # Keep a new namespace; no automatic change to the incumbent feature order.
    result = {k.replace('form_seq_','form_fresh180_',1):v for k,v in result.items()}
    for key in ('input_rows','duplicate_rows','conflicting_rows','retained_rows',
                'stale_rows_excluded','five_games_available'):
        result[f'form_fresh180_{side}_quality_{key}'] = float(quality.get(key,0))
    return result,quality
