"""Pregame-only form integrity and sequence features, without odds or identities.

Provider aggregates are not assumed to share a window. Explicit match rows are
the only source for last-five/last-ten comparisons; unknown data stays unknown.
"""
import math


def number(value):
    if isinstance(value, bool):
        return None
    try:
        value = float(value)
        return value if math.isfinite(value) and value >= 0 else None
    except (TypeError, ValueError):
        return None


def sfi_goal_integrity(raw):
    """Detect contradictory totals, means and complete goal-minute counters.

    This does not repair an API value from another possibly different window.
    A contradiction marks the aggregate as unsuitable for an attack/defense
    comparison. W/D/L remains independent and usable.
    """
    perf = raw.get('perf', {}) if isinstance(raw, dict) else {}
    if not isinstance(perf, dict):
        perf = {}
    # Keep measured zero distinct from an absent or invalid field. Legacy
    # flat payloads are supported without recursively reading live statistics.
    def observed(*keys):
        return next((number(container[key]) for container in (perf, raw)
                     if isinstance(container, dict) for key in keys
                     if key in container and number(container[key]) is not None), None)
    gf = observed('avg_goals_scored','average_goals_scored','goals_for_avg')
    ga = observed('avg_goals_conceded','average_goals_conceded','goals_against_avg')
    checked, conflicts = 0, 0
    for direction in ('scored', 'conceded'):
        total = number(perf.get('tot_goals_' + direction))
        avg = number(perf.get('avg_goals_' + direction))
        sequence = perf.get('l_5_matches', '')
        games = sum(letter in 'WDL' for letter in sequence) if isinstance(sequence, str) else 0
        if total is not None and avg is not None and games == 5:
            checked += 1
            conflicts += abs(total - avg * games) > .26  # two-decimal / one-decimal means
        buckets = [perf.get(f'goals_{direction}_{period}') for period in
                   ('0_15', '16_30', '31_45', '46_60', '61_75', '76_90')]
        values = [number(bucket[0]) if isinstance(bucket, (list, tuple)) and bucket else None
                  for bucket in buckets]
        if total is not None and all(value is not None for value in values):
            checked += 1
            conflicts += abs(total - sum(values)) > .01
    return {'goals_integrity_checked': float(checked > 0),
            'goals_available': float(gf is not None and ga is not None),
            'goals_inconsistent': float(conflicts > 0),
            'goals_integrity_conflicts': float(conflicts)}


def sfi_extended_performance(raw):
    """Extract documented SFI pregame aggregates without inventing zeros.

    Values remain in the provider's native scale.  The explicit availability
    flags let callers distinguish a measured zero from a missing field.
    """
    perf = raw.get('perf', {}) if isinstance(raw, dict) else {}
    if not isinstance(perf, dict):
        return {}
    aliases = {
        'avg_game_goals': ('avg_game_goals', 'average_game_goals'),
        'btts': ('btts', 'both_teams_to_score'),
        'over_1_5_team': ('o_1_5_team', 'over_1_5_team'),
        'over_0_5_game': ('o_0_5_game', 'over_0_5_game'),
        'over_1_5_game': ('o_1_5_game', 'over_1_5_game'),
        'over_2_5_game': ('o_2_5_game', 'over_2_5_game'),
        'over_3_5_game': ('o_3_5_game', 'over_3_5_game'),
    }
    result = {}
    for output, keys in aliases.items():
        value = next((number(perf[key]) for key in keys
                      if key in perf and number(perf[key]) is not None), None)
        result[output] = float(value) if value is not None else 0.0
        result[output + '_available'] = float(value is not None)
    return result


def valid_history(events, before, target_id=''):
    """Use completed score rows whose conservative end precedes capture time."""
    rows, seen = [], set()
    for item in events if isinstance(events, list) else []:
        if not isinstance(item, dict):
            continue
        mid = str(item.get('match_id') or '')
        ts = number(item.get('start_timestamp'))
        gf, ga = number(item.get('goals_for')), number(item.get('goals_against'))
        if (not mid or mid == str(target_id) or mid in seen or ts is None
                or ts + 3*3600 >= before or gf is None or ga is None
                or not gf.is_integer() or not ga.is_integer()
                or item.get('is_home') not in (0, 1, 0., 1.)):
            continue
        seen.add(mid)
        rows.append(dict(item, goals_for=gf, goals_against=ga))
    return sorted(rows, key=lambda row: row['start_timestamp'], reverse=True)[:10]


def sequence_features(events, side, before, target_id=''):
    rows = valid_history(events, before, target_id)
    result = {'games': float(len(rows)), 'available': float(len(rows) >= 5)}
    # Every row supplies the same schema; absent samples carry a count, not a
    # fabricated zero-goal claim. Models must use the available mask.
    for name, subset in (('last5', rows[:5]), ('previous5', rows[5:10]),
                         ('venue', [r for r in rows if bool(r['is_home']) == (side == 'home')])):
        n = len(subset)
        result[name + '_games'] = float(n)
        for metric in ('gf','ga','draw','zero_draw','scoring_draw','btts','total_variance','ppg'):
            result[name + '_' + metric] = 0.
        if not n:
            continue
        gf = [r['goals_for'] for r in subset]
        ga = [r['goals_against'] for r in subset]
        totals = [h+a for h,a in zip(gf,ga)]
        result.update({name+'_gf': sum(gf)/n, name+'_ga': sum(ga)/n,
            name+'_draw': sum(h==a for h,a in zip(gf,ga))/n,
            name+'_zero_draw': sum(h==a==0 for h,a in zip(gf,ga))/n,
            name+'_scoring_draw': sum(h==a and h>0 for h,a in zip(gf,ga))/n,
            name+'_btts': sum(h>0 and a>0 for h,a in zip(gf,ga))/n,
            name+'_total_variance': sum((g-sum(totals)/n)**2 for g in totals)/n,
            name+'_ppg': sum(3 if h>a else 1 if h==a else 0 for h,a in zip(gf,ga))/n})
    result['recent3_ppg'] = sum(3 if r['goals_for']>r['goals_against'] else
        1 if r['goals_for']==r['goals_against'] else 0 for r in rows[:3])/max(1,len(rows[:3]))
    comps = [str(r.get('tournament_id') or '') for r in rows if r.get('tournament_id')]
    result['competition_mix'] = 1-max((comps.count(c) for c in set(comps)),default=0)/max(1,len(comps))
    return {f'form_seq_{side}_{key}': float(value) for key,value in result.items()}


def sequence_vectors(features):
    """Directional attack/defense and separate low/high-scoring draw signals."""
    f = features
    side, draw = {}, {}
    def v(s,k):
        return number(f.get(f'form_seq_{s}_{k}')) or 0.
    coverage = min(v('home','available'),v('away','available'))
    draw['sequence_coverage'] = coverage
    for window in ('last5','previous5','venue'):
        rel = min(1.,v('home',window+'_games')/5,v('away',window+'_games')/5)*coverage
        draw[window+'_coverage'] = rel
        h = (v('home',window+'_gf')+v('away',window+'_ga'))/2
        a = (v('away',window+'_gf')+v('home',window+'_ga'))/2
        side[window+'_duel_gap'] = (h-a)*rel
        draw[window+'_total'] = (h+a)*rel
        draw[window+'_parity'] = math.exp(-abs(h-a))*rel
        for metric in ('draw','zero_draw','scoring_draw','btts','total_variance'):
            draw[window+'_'+metric] = (v('home',window+'_'+metric)+v('away',window+'_'+metric))/2*rel
        side[window+'_ppg_gap'] = (v('home',window+'_ppg')-v('away',window+'_ppg'))*rel
    rel = min(1.,v('home','previous5_games')/5,v('away','previous5_games')/5)*coverage
    for metric in ('gf','ga'):
        h = v('home','last5_'+metric)-v('home','previous5_'+metric)
        a = v('away','last5_'+metric)-v('away','previous5_'+metric)
        side[metric+'_trend_gap'] = (h-a)*rel
        draw[metric+'_trend_mean'] = (h+a)/2*rel
    draw['competition_mix'] = (v('home','competition_mix')+v('away','competition_mix'))/2*coverage
    return side,draw
