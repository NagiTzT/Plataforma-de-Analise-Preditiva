"""Free bulk historical statistics, no keys/odds: football-data.co.uk.

Only documented result and match-stat columns are retained. These are stats of
PRIOR completed matches, never the current fixture's stats. Source identity and
capture time are persisted separately from the active model.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import logging
import math
import re
import sqlite3
import time
import unicodedata
import zipfile
from contextlib import closing
from datetime import datetime, timezone
from difflib import SequenceMatcher
from functools import lru_cache
from pathlib import Path
from urllib.parse import urljoin, urlparse

import numpy as np
import requests
from bs4 import BeautifulSoup

DEFAULT_DB = str(Path(__file__).with_name('free_football_data.db'))
INDEX = 'https://football-data.co.uk/downloadm.php'
METRICS = ('goals', 'shots', 'target', 'corners', 'fouls', 'yellow', 'red')
COLUMNS = {'goals':('FTHG','FTAG'), 'shots':('HS','AS'), 'target':('HST','AST'),
           'corners':('HC','AC'), 'fouls':('HF','AF'), 'yellow':('HY','AY'), 'red':('HR','AR')}
COUNTRIES = {'E':'england', 'SC':'scotland', 'D':'germany', 'I':'italy', 'SP':'spain',
             'F':'france', 'N':'netherlands', 'B':'belgium', 'P':'portugal', 'T':'turkey', 'G':'greece'}


def connect(db=DEFAULT_DB):
    conn = sqlite3.connect(db, timeout=15)
    conn.execute('PRAGMA journal_mode=WAL')
    conn.executescript('''CREATE TABLE IF NOT EXISTS sources (
        url TEXT PRIMARY KEY, fetched_at INTEGER, status INTEGER, sha256 TEXT);
        CREATE TABLE IF NOT EXISTS matches (
        fixture_key TEXT PRIMARY KEY, division TEXT, match_date TEXT,
        home TEXT, away TEXT, home_key TEXT, away_key TEXT, home_goals INTEGER,
        away_goals INTEGER, metrics_json TEXT, source_url TEXT, captured_at INTEGER);
        CREATE INDEX IF NOT EXISTS free_home_time ON matches(division,home_key,match_date);
        CREATE INDEX IF NOT EXISTS free_away_time ON matches(division,away_key,match_date);
        CREATE TABLE IF NOT EXISTS bulk_index (
            url TEXT PRIMARY KEY, fetched_at INTEGER, links_json TEXT);''')
    return conn


def number(value):
    try:
        value = float(value)
        return value if math.isfinite(value) and value >= 0 else None
    except (TypeError, ValueError):
        return None


def team_key(name):
    name = ''.join(ch for ch in unicodedata.normalize('NFKD',str(name)).casefold() if not unicodedata.combining(ch))
    name = re.sub(r'\butd\b','united',name)
    name = re.sub(r'\b(fc|cf|sc|afc)\b','',name)
    return re.sub(r'[^a-z0-9]','',name)


def team_category(name):
    # Never fuzzy-match youth, women or reserve sides to the senior men's team.
    text = str(name).casefold()
    age = re.search(r'\b(?:u|under)[ -]?(\d{1,2})\b', text)
    gender = bool(re.search(r'\b(women|womens|ladies|feminino|feminina|w)\b',text))
    reserve = bool(re.search(r'\b(reserves?|ii|iii|b)\b',text))
    return (age.group(1) if age else '', gender, reserve)


def parse_csv(body, source_url, captured_at):
    records = []
    for row in csv.DictReader(io.StringIO(body)):
        try:
            date = next(datetime.strptime(row['Date'].strip(), fmt).date()
                        for fmt in ('%d/%m/%Y','%d/%m/%y')
                        if len(row['Date'].strip().split('/')[-1]) == (4 if fmt.endswith('%Y') else 2))
            home, away, division = row['HomeTeam'].strip(), row['AwayTeam'].strip(), row['Div'].strip()
            hs, aws = number(row.get('FTHG')), number(row.get('FTAG'))
            if not home or not away or not division or hs is None or aws is None:
                continue
            if not hs.is_integer() or not aws.is_integer():
                continue
            outcome = 'H' if hs > aws else 'A' if aws > hs else 'D'
            if row.get('FTR') and row['FTR'] != outcome:
                continue
            metrics = {f'{side}_{metric}':number(row.get(column))
                       for metric, columns in COLUMNS.items() for side,column in zip(('home','away'),columns)}
            key = f'{division}:{date.isoformat()}:{team_key(home)}:{team_key(away)}'
            records.append((key,division,date.isoformat(),home,away,team_key(home),team_key(away),
                            int(hs),int(aws),json.dumps(metrics),source_url,int(captured_at)))
        except (KeyError, ValueError, AttributeError, StopIteration):
            continue
    return records


def collect_bulk(db=DEFAULT_DB, seasons=3, ttl=86400):
    now = int(time.time())
    # Global backoff prevents retrying a blocked provider through other files.
    with closing(connect(db)) as conn:
        blocked = conn.execute('SELECT MAX(fetched_at) FROM sources WHERE status IN (403,429)').fetchone()[0]
        if blocked and now-int(blocked) < 6*3600:
            return {'provider_backoff':1}
        cache = {r[0]:r[1:] for r in conn.execute('SELECT url,fetched_at,status FROM sources')}
        index_cache = conn.execute('SELECT fetched_at,links_json FROM bulk_index WHERE url=?',(INDEX,)).fetchone()
    summary = {'requests':0,'archives':0,'cache_hits':0,'parsed_matches':0,'failures':[]}
    if index_cache and now-index_cache[0] < ttl:
        urls = json.loads(index_cache[1])
    else:
        response = requests.get(INDEX, timeout=(4,25))
        summary['requests'] += 1
        with closing(connect(db)) as conn:
            conn.execute('INSERT OR REPLACE INTO sources VALUES (?,?,?,?)',
                (INDEX,now,response.status_code,hashlib.sha256(response.content).hexdigest()))
            conn.commit()
        if response.status_code != 200:
            summary['failures'].append({'url':INDEX,'status':response.status_code})
            return summary
        urls = []
        for anchor in BeautifulSoup(response.text,'html.parser').find_all('a'):
            url = urljoin(INDEX,anchor.get('href',''))
            if (urlparse(url).hostname in {'football-data.co.uk','www.football-data.co.uk'}
                    and re.search(r'/mmz4281/\d{4}/data\.zip$',url) and url not in urls):
                urls.append(url)
        if urls:
            with closing(connect(db)) as conn:
                conn.execute('INSERT OR REPLACE INTO bulk_index VALUES (?,?,?)',(INDEX,now,json.dumps(urls)))
                conn.commit()
    if not urls:
        raise ValueError('No published bulk download links found')
    for url in urls[:max(0,min(5,seasons))]:
        previous = cache.get(url)
        if previous and previous[1] == 200 and now-previous[0] < ttl:
            summary['cache_hits'] += 1
            continue
        result = requests.get(url,timeout=(4,40))
        summary['requests'] += 1
        records = []
        if result.status_code == 200:
            # Read the archive in memory: no arbitrary paths are extracted.
            if len(result.content) > 30_000_000:
                raise ValueError('Bulk archive exceeds expected size')
            with zipfile.ZipFile(io.BytesIO(result.content)) as archive:
                if sum(info.file_size for info in archive.infolist()) > 100_000_000:
                    raise ValueError('Expanded archive exceeds size bound')
                for member in archive.infolist():
                    if member.filename.lower().endswith('.csv'):
                        raw = archive.read(member)
                        try:
                            text = raw.decode('utf-8-sig')
                        except UnicodeDecodeError:
                            text = raw.decode('cp1252')
                        records.extend(parse_csv(text,url,now))
            if not records:
                raise ValueError('Provider returned no valid completed matches')
        with closing(connect(db)) as conn:
            # Do not erase evidence of when unchanged values first became known.
            conn.executemany('''INSERT INTO matches VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(fixture_key) DO UPDATE SET home_goals=excluded.home_goals,
                away_goals=excluded.away_goals,metrics_json=excluded.metrics_json,
                source_url=excluded.source_url,captured_at=excluded.captured_at
                WHERE matches.metrics_json != excluded.metrics_json''',records)
            conn.execute('INSERT OR REPLACE INTO sources VALUES (?,?,?,?)',
                (url,now,result.status_code,hashlib.sha256(result.content).hexdigest()))
            conn.commit()
        if result.status_code != 200:
            summary['failures'].append({'url':url,'status':result.status_code})
            if result.status_code in {403,429}:
                break
        else:
            summary['archives'] += 1
            summary['parsed_matches'] += len(records)
            print(f'Free source: {len(records)} finished matches from {url}',flush=True)
    return summary


class FreeStatistics:
    def __init__(self, db=DEFAULT_DB):
        with closing(connect(db)) as conn:
            conn.row_factory = sqlite3.Row
            self.rows = [dict(row) for row in conn.execute('SELECT * FROM matches ORDER BY match_date,fixture_key')]
        self.history, self.teams, self.categories = {}, {}, {}
        for row in self.rows:
            row['metrics'] = json.loads(row['metrics_json'])
            row['date_epoch'] = datetime.fromisoformat(row['match_date']).replace(tzinfo=timezone.utc).timestamp()
            for side in ('home','away'):
                self.history.setdefault((row['division'],row[f'{side}_key']),[]).append((row,side))
                self.teams.setdefault(row['division'],set()).add(row[f'{side}_key'])
                self.categories[row[f'{side}_key']] = team_category(row[side])

    @lru_cache(maxsize=8000)
    def match_teams(self, home, away, league=''):
        league = str(league).lower()
        found = []
        for division, teams in self.teams.items():
            country = 'england' if division.startswith('E') else COUNTRIES.get(re.sub(r'\d','',division))
            country_hits = [name for name in COUNTRIES.values() if name in league]
            if country_hits and country not in country_hits:
                continue
            keys, qualities = [], []
            for name in (home,away):
                key = team_key(name)
                compatible = [t for t in teams if self.categories[t] == team_category(name)]
                if key in compatible:
                    keys.append(key); qualities.append(1.0); continue
                ranked = sorted(((SequenceMatcher(None,key,t).ratio(),t) for t in compatible), reverse=True)
                if not ranked or ranked[0][0] < .88 or (len(ranked)>1 and ranked[0][0]-ranked[1][0] < .08):
                    break
                keys.append(ranked[0][1]); qualities.append(ranked[0][0])
            if len(keys)==2 and keys[0]!=keys[1]:
                found.append((sum(qualities),division,*keys))
        if not found:
            return None
        found.sort(reverse=True)
        # Promotion/relegation may yield two divisions for the same exact pair.
        # Use the division only if uniquely identified; never infer by result.
        if len(found)>1 and found[0][0]-found[1][0] < .08:
            return None
        return found[0][1:]

    def features(self, division, home, away, cutoff, historical=False):
        # Online: known capture required; historical reconstruction: 5-day lag
        # and explicit caveat because old provider publication times are unknown.
        features = {}
        for prefix, team, venue in (('home',home,'home'),('away',away,'away')):
            eligible = [(row,side) for row,side in self.history.get((division,team),[])
                if row['date_epoch'] + (5*86400 if historical else 86400) < cutoff
                and (historical or row['captured_at'] <= cutoff)]
            for subset, history in (('all',eligible[-10:]),('venue',[(r,s) for r,s in eligible if s==venue][-10:])):
                features[f'free_{prefix}_{subset}_games'] = float(len(history))
                for metric in METRICS:
                    for direction in ('for','against'):
                        samples = []
                        for i,(row,side) in enumerate(reversed(history)):
                            own = side if direction=='for' else ('away' if side=='home' else 'home')
                            value = row['metrics'].get(f'{own}_{metric}')
                            if value is not None:
                                samples.append((value,.85**i))
                        key = f'free_{prefix}_{subset}_{metric}_{direction}'
                        features[key+'_games'] = float(len(samples))
                        features[key] = sum(v*w for v,w in samples)/sum(w for v,w in samples) if samples else float('nan')
        features['free_both_available'] = float(min(features['free_home_all_games'],features['free_away_all_games']) >= 3)
        return features


def enrich_games(db_path, games, source_db=DEFAULT_DB):
    """Attach free features to radar games; no model switch or current scores."""
    try:
        try:
            summary = collect_bulk(source_db)
        except (requests.RequestException, ValueError, zipfile.BadZipFile) as exc:
            # Still use the last successful local collection on a network error.
            summary = {'refresh_failed':type(exc).__name__,'requests':None}
        source = FreeStatistics(source_db)
        now = int(time.time())
        captured = []
        for game in games:
            matched = source.match_teams(game.get('Time Casa',''),game.get('Time Fora',''),game.get('Liga',''))
            if not matched or int(game.get('Timestamp',0)) <= now:
                continue
            features = source.features(*matched,cutoff=now)
            features = {key:value for key,value in features.items() if math.isfinite(value)}
            captured.append((str(game['ID']),now,json.dumps(features,allow_nan=False),json.dumps(matched)))
        with closing(sqlite3.connect(db_path,timeout=15)) as conn:
            conn.execute('''CREATE TABLE IF NOT EXISTS free_pregame_snapshots (
                match_id TEXT PRIMARY KEY,captured_at INTEGER,features_json TEXT,identity_json TEXT)''')
            conn.executemany('INSERT OR IGNORE INTO free_pregame_snapshots VALUES (?,?,?,?)',captured)
            conn.commit()
        return {**summary,'linked_games':len(captured),'games':len(games)}
    except Exception as exc:
        logging.warning('Fonte gratuita indisponível (%s); radar mantém os dados existentes',type(exc).__name__)
        return {'source_failed':1,'linked_games':0}


def load_pregame_features(db_path, match_id, cutoff_timestamp=None):
    """Read-only, immutable per-fixture capture; no HTTP on prediction threads."""
    try:
        with closing(sqlite3.connect(Path(db_path).resolve().as_uri()+'?mode=ro',uri=True,timeout=5)) as conn:
            row = conn.execute('SELECT captured_at,features_json FROM free_pregame_snapshots WHERE match_id=?',
                               (str(match_id),)).fetchone()
        now = int(time.time())
        if (not row or row[0] > now or
                (cutoff_timestamp and row[0] >= int(cutoff_timestamp))):
            return {}
        features = json.loads(row[1])
        return {k:float(v) for k,v in features.items() if k.startswith('free_')
                and isinstance(v,(int,float)) and math.isfinite(v)}
    except (sqlite3.Error,ValueError,TypeError,AttributeError):
        return {}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--db',default=DEFAULT_DB)
    parser.add_argument('--seasons',type=int,default=3)
    parser.add_argument('--output',required=True)
    args = parser.parse_args()
    summary = collect_bulk(args.db,args.seasons)
    output = Path(args.output); output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary))
