"""Prospective Phase 3 market/data capture helpers; no decision logic."""
import math
import re
import time
import hashlib

from phase2_observations import append
from phase2_observations import market_probabilities

MARKET_ALIASES={
 '1x2':('1x2','full time','match result'),
 'over_under_2_5':('over/under 2.5','total goals 2.5','over under 2.5'),
 'over_under_1_5':('over/under 1.5','total goals 1.5','over under 1.5'),
 'over_under_3_5':('over/under 3.5','total goals 3.5','over under 3.5'),
 'btts':('both teams to score','btts'),
 'asian_handicap':('asian handicap',),
 'double_chance':('double chance',),
 'draw_no_bet':('draw no bet','dnb'),
}


def decimal(value):
    try:
        if isinstance(value,(int,float)):
            result=float(value)
        elif '/' in str(value):
            a,b=str(value).split('/',1); result=1+float(a)/float(b)
        else: result=float(value)
        return result if math.isfinite(result) and result>1 else None
    except (TypeError,ValueError,ZeroDivisionError): return None


def market_kind(name):
    text=re.sub(r'\s+',' ',str(name or '').strip().lower())
    exact=next((kind for kind,names in MARKET_ALIASES.items() if any(alias in text for alias in names)),None)
    if exact: return exact
    if ('over' in text and 'under' in text) or 'total goals' in text: return 'over_under'
    return None


def market_line(value):
    """Preserve the provider line; never coerce distinct Asian/O-U lines."""
    if isinstance(value,(int,float)) and math.isfinite(float(value)):
        return float(value)
    match=re.search(r'(?<!\d)([+-]?\d+(?:[.,]\d+)?)(?!\d)',str(value or ''))
    if not match: return None
    try: return float(match.group(1).replace(',','.'))
    except ValueError: return None


def extract_markets(payload):
    """Read markets already present in a response. It never performs HTTP."""
    found=[]; stack=[payload]
    while stack:
        node=stack.pop()
        if isinstance(node,list): stack.extend(node); continue
        if not isinstance(node,dict): continue
        choices=node.get('choices')
        label=node.get('marketGroup') or node.get('marketName') or node.get('name')
        kind=market_kind(label)
        if isinstance(choices,list) and kind:
            line=(node.get('line') if node.get('line') is not None else
                  node.get('handicap') if node.get('handicap') is not None else
                  node.get('total'))
            line=market_line(line if line is not None else label)
            selections=[]
            for choice in choices:
                if not isinstance(choice,dict): continue
                price=decimal(choice.get('decimalValue') or choice.get('fractionalValue') or choice.get('odds'))
                name=str(choice.get('name') or choice.get('label') or '')
                choice_line=(choice.get('line') if choice.get('line') is not None else
                             choice.get('handicap') if choice.get('handicap') is not None else line)
                if price: selections.append({'name':name,'side':name.strip().lower(),
                    'line':market_line(choice_line if choice_line is not None else name),
                    'decimal_odd':price})
            family=('over_under' if kind.startswith('over_under') else kind)
            if selections: found.append({'market':kind,'market_family':family,'line':line,
                'provider_label':str(label),'selections':selections})
        for value in node.values():
            if isinstance(value,(list,dict)) and value is not choices: stack.append(value)
    # Preserve distinct lines. Exact replay of the same provider market is idempotent.
    unique={}
    for item in found:
        key=(item['market_family'],item.get('line'),item['provider_label'])
        unique.setdefault(key,item)
    return list(unique.values())


def capture_markets(db,run_id,match_id,payload,provider,kickoff,observed_at=None):
    observed_at=float(observed_at or time.time()); markets=extract_markets(payload)
    schema_fields=sorted(str(k) for k in payload) if isinstance(payload,dict) else []
    return append(db,run_id,str(match_id),'market_bundle_v1',{
      'provider':provider,'observed_at':observed_at,'kickoff':kickoff,
      'is_pregame_observation':bool(kickoff and observed_at<float(kickoff)),
      'pregame_invalid_reason':None if kickoff and observed_at<float(kickoff) else 'observed_at_not_before_kickoff',
      'vendor_quote_timestamp':None,
      'odds_age_seconds':float(kickoff)-observed_at if kickoff else None,
      'line_age':None,'markets':markets,
      'available_market_families':sorted({x['market_family'] for x in markets}),
      'provider_schema_fields':schema_fields,
      'provider_schema_hash':hashlib.sha256('|'.join(schema_fields).encode()).hexdigest(),
      'opening_price':None,'closing_price':None,'movement':None,
      'note':'Only markets present in this response; no additional request and no reconstructed movement.'})


def feature_source(key):
    if key.startswith('form_sfi_') or key.startswith('context_sfi_'): return 'soccer-football-info'
    if key.startswith(('sofa_','live_recent_','form_seq_','form_fresh180_')): return 'sofascore/cache'
    if key.startswith('allsports_'): return 'allsports'
    if key.startswith('free_'): return 'football-data.co.uk'
    if key.startswith(('elo_','liga_','temporada_','form_home_','form_away_','context_')): return 'local-derived'
    return 'local-model-input'


def coverage_summary(features,observed_at):
    groups={
      'team_form':any(float(features.get(k,0) or 0)>0 for k in ('context_sfi_available','sofa_pre_available','live_recent_available')),
      'elo_rating':any(k.startswith('elo_') for k in features),
      'home_away':any(t in k for k in features for t in ('mando','casa_','fora_','same_venue')),
      'league_profile':any(k.startswith(('liga_','competition_')) for k in features),
      'xg':min(float(features.get('sofa_roll_home_xg_games',0) or 0),float(features.get('sofa_roll_away_xg_games',0) or 0))>=3,
      'shots_on_target':min(float(features.get('sofa_roll_home_shots_on_target_for_games',0) or 0),float(features.get('sofa_roll_away_shots_on_target_for_games',0) or 0))>=3,
      'big_chances':min(float(features.get('sofa_roll_home_big_chances_for_games',0) or 0),float(features.get('sofa_roll_away_big_chances_for_games',0) or 0))>=3,
      'rest_schedule':any('rest' in k or 'descanso' in k for k in features),
      'lineups':False,'injuries_suspensions':False,'manager':False,
    }
    return {'score':sum(groups.values())/len(groups),'families':groups,
      'meaning':'Informational data availability only; never an approval score.',
      'provenance_families':{source:{'source':source,'observed_at':observed_at,
         'data_for_period_ending':None,'mode':'ORIGINAL_SNAPSHOT'}
         for source in sorted({feature_source(k) for k in features})}}


def capture_shadow(db,run_id,match_id,champion_probabilities,odds,kickoff,model_version):
    """Freeze predeclared challengers; missing challengers remain explicit."""
    now=time.time(); market=market_probabilities(odds,now,kickoff)
    candidates=[{'name':'CHAMPION','version':str(model_version),'probabilities':list(champion_probabilities),
                 'available':True,'source':'active frozen prediction'},
      {'name':'CHALLENGER_MARKET','version':'market-proportional-v1',
       'probabilities':market['probabilities'] if market else None,'available':bool(market),
       'source':'AllSports as-observed pregame 1X2','reason':None if market else 'invalid_or_not_pregame'},
      {'name':'CHALLENGER_STRENGTH','version':None,'probabilities':None,'available':False,
       'reason':'not frozen: retrospective strength signal not proven'},
      {'name':'CHALLENGER_DRAW','version':None,'probabilities':None,'available':False,
       'reason':'not frozen: draw signal not proven'},
      {'name':'CHALLENGER_FULL','version':None,'probabilities':None,'available':False,
       'reason':'not frozen: approved signal families unavailable'}]
    return append(db,run_id,str(match_id),'shadow_phase3_v1',{
      'frozen_at':now,'kickoff':kickoff,'candidates':candidates,
      'affects_production':False,'holdout_protocol':'20 new radars without tuning per frozen version'})
