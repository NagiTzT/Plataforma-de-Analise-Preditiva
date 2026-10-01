import collections
import concurrent.futures
import contextvars
import hashlib
import json
import logging
import math
import os
import random
import re
import sqlite3
import threading
import time
import unicodedata
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from difflib import get_close_matches
from io import BytesIO

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
import xgboost as xgb
from dotenv import load_dotenv
from api_key_config import configured_keys
from playwright.sync_api import sync_playwright
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, log_loss, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from ml_evolution import (
    MODEL_VERSION, balanced_sample_weights, ensure_evolution_tables, evaluate_evolution, get_active_confidence, get_active_model_identity,
    persist_evolution_result, record_skipped_evolution,
)
from ticket_engine import selecionar_grupos_bilhetes
from telegram_delivery import (
    claim_ticket_delivery,
    delivery_fingerprint,
    mark_ticket_failed,
    mark_ticket_sent,
)
from competition_context import competition_flags
from football_model_features import add_venue_comparison, enhanced_rolling_features, add_measured_sofa_duel
from historical_identity import resolve_training_team_name, training_team_aliases
from historical_opponent_strength import current_elo_ratings, historical_opponent_strengths
from football_results import regulation_score, event_finished, resolve_pick_side, settlement_status
from football_event_identity import event_matches_prediction, safe_event_timestamp
from pregame_metric_enrichment import enrich_recent_metrics_safely
from free_football_data import enrich_games as enrich_free_statistics, load_pregame_features as load_free_statistics
from pick_scenario_shadow import record_shadow as record_pick_scenario_shadow
from football_results import incident_goal_score
from radar_audit_scope import (
    ensure_radar_audit_schema,
    get_latest_radar_run_ids,
    radar_run_placeholders,
)
from radar_prediction_integrity import deduplicate_predictions
from operational_backtest import operational_snapshot_report
from allsports_api import (
    fetch_allsports_postmatch_resources,
    fetch_allsports_pregame_context,
    fetch_competition_events_for_date,
    fetch_football_events_for_date,
    fetch_recent_team_events_for_game,
    match_detail_url,
    match_odds_url,
    match_resource_url,
    match_winning_odds_url,
    matches_odds_date_url,
    radar_window_brt,
    schedule_dates_for_brt_window,
    team_matches_url,
    tournament_standings_url,
)
from soccer_football_info import (
    collect_soccer_radar_games,
    get_soccer_context_features,
    init_soccer_context_db,
    normalize_team_name,
    team_name_similarity,
)
from sofascore_intelligence import (
    ACTIVE_ANALYSIS_PROFILE,
    analyze_pregame_context,
    audit_match_postmortem,
    capture_pregame_contexts,
    get_event_for_audit,
    get_pregame_features as get_sofascore_pregame_features,
    init_sofascore_db,
    load_prediction_snapshot_features,
    load_prediction_snapshot_metadata,
    monitor_pregame_context_sources,
    reclassify_stored_postmortems,
    save_prediction_snapshot,
)

SOFASCORE_STANDINGS_BLACKLIST = set()
PLAYWRIGHT_SEMAPHORE = threading.Semaphore(4)
CACHE_ESTATISTICAS_SOFASCORE = {}
CACHE_STANDINGS = {}
_saved_league_mappings = set()
_saved_league_mappings_lock = threading.Lock()

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    
if "import_paused" not in st.session_state:
    st.session_state.import_paused = False
if "import_job_ids" not in st.session_state:
    st.session_state.import_job_ids = []   # lista de match_ids a processar
if "import_current_index" not in st.session_state:
    st.session_state.import_current_index = 0

# Lock para escritas no banco de dados (evita conflitos)
db_write_lock = threading.RLock()
# --- CONFIGURAÇÃO DE LOGS ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)

# --- CONFIGURAÇÃO DE INFRAESTRUTURA (via st.secrets) ---
load_dotenv()
try:
    RAPIDAPI_KEY = st.secrets.get("RAPIDAPI_KEY", "")
    RAPIDAPI_HOST = st.secrets.get("RAPIDAPI_HOST", "allsportsapi2.p.rapidapi.com")
    TELEGRAM_TOKEN = st.secrets["TELEGRAM_TOKEN"]
    TELEGRAM_CHAT_ID = st.secrets["TELEGRAM_CHAT_ID"]
except Exception as e:
    st.error("Erro ao carregar secrets. Verifique o arquivo .streamlit/secrets.toml")
    logger.error(f"Secrets não encontrados: {e}")
    st.stop()

RAPIDAPI_KEYS = configured_keys("RAPIDAPI_KEYS", "allsports")
if RAPIDAPI_KEY:
    RAPIDAPI_KEYS = list(dict.fromkeys([RAPIDAPI_KEY, *RAPIDAPI_KEYS]))
elif RAPIDAPI_KEYS:
    RAPIDAPI_KEY = RAPIDAPI_KEYS[0]
HEADERS = {"x-rapidapi-host": RAPIDAPI_HOST, "x-rapidapi-key": RAPIDAPI_KEY, "Content-Type": "application/json"}
RAPIDAPI_DAILY_LIMIT = 100
# O plano permite 5 req/s. 0,21 s mantém uma pequena margem de segurança.
RAPIDAPI_MIN_INTERVAL_SECONDS = float(os.getenv("RAPIDAPI_MIN_INTERVAL_SECONDS", "0.21"))
MIN_ML_CONFIDENCE = max(50, int(os.getenv("MIN_ML_CONFIDENCE", "50")))
MIN_DRAW_CONFIDENCE = int(os.getenv("MIN_DRAW_CONFIDENCE", "50"))
MAX_TICKETS_PER_RUN = max(0, int(os.getenv("MAX_TICKETS_PER_RUN", "0")))
PERMITIR_EMPATES_BILHETE = os.getenv("PERMITIR_EMPATES_BILHETE", "0") == "1"
PUBLICAR_TODAS_PREVISOES = True
USAR_MODELOS_POR_LIGA = os.getenv("USAR_MODELOS_POR_LIGA", "0") == "1"
ML_MIN_NEW_SAMPLES = max(20, int(os.getenv("ML_MIN_NEW_SAMPLES", "40")))
ML_CONTEXT_FEATURE_MIN_SAMPLES = max(
    50, int(os.getenv("ML_CONTEXT_FEATURE_MIN_SAMPLES", "200"))
)
DB_NAME = 'ia_sports_v5.db'

_rate_limiter = threading.Semaphore(4)
_last_request_time = 0
_rate_lock = threading.Lock()
_api_cache_lock = threading.Lock()
_api_response_cache = {}
_api_url_locks = {}
_api_metrics = {"logical_calls": 0, "http_requests": 0, "cache_hits": 0}
_key_rotation_lock = threading.Lock()
_key_rotation_cursor = 0
_no_keys_notice_lock = threading.Lock()
_no_keys_notice_date = None
_model_cache = {}
_model_cache_lock = threading.RLock()
MODEL_CACHE_TTL_SECONDS = max(1, int(os.getenv("MODEL_CACHE_TTL_SECONDS", "300")))
RAPIDAPI_CACHE_TTL_SECONDS = max(0, int(os.getenv("RAPIDAPI_CACHE_TTL_SECONDS", "300")))
_feature_cutoff_timestamp = contextvars.ContextVar("feature_cutoff_timestamp", default=None)
MAX_WORKERS = int(os.getenv("MAX_WORKERS", "4"))

db_write_lock = threading.RLock()

if "api_errors" not in st.session_state:
    st.session_state.api_errors = []

def get_brt_time():
    return datetime.now(timezone(timedelta(hours=-3)))

def get_db_connection():
    conn = sqlite3.connect(DB_NAME, timeout=60, check_same_thread=False)
    conn.execute("PRAGMA busy_timeout = 30000")
    return conn

def init_error_log_table():
    with get_db_connection() as conn:
        conn.execute('''CREATE TABLE IF NOT EXISTS api_error_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp DATETIME, endpoint TEXT, status_code TEXT, message TEXT)''')
init_error_log_table()

def log_api_error(endpoint, status_code, message=""):
    ts = get_brt_time().strftime("%Y-%m-%d %H:%M:%S")
    with get_db_connection() as conn:
        inserted = conn.execute("INSERT INTO api_error_log (timestamp, endpoint, status_code, message) VALUES (?,?,?,?)",
                                (ts, endpoint, str(status_code), message))
        # Evita crescimento ilimitado sem executar DELETE a cada erro.
        if inserted.lastrowid and inserted.lastrowid % 1000 == 0:
            conn.execute("DELETE FROM api_error_log WHERE id < ?", (max(0, inserted.lastrowid - 100000),))
    if "api_errors" not in st.session_state:
        st.session_state.api_errors = []
    st.session_state.api_errors.append({"timestamp": ts, "endpoint": endpoint, "status_code": str(status_code), "message": message})
    st.session_state.api_errors = st.session_state.api_errors[-500:]

def _notify_no_rapidapi_keys(endpoint):
    """Registra um único aviso diário, evitando spam entre workers."""
    global _no_keys_notice_date
    usage_date = get_brt_time().strftime("%Y-%m-%d")
    with _no_keys_notice_lock:
        if _no_keys_notice_date == usage_date:
            return
        _no_keys_notice_date = usage_date
    message = "Todas as chaves atingiram o limite diário ou estão temporariamente bloqueadas"
    log_api_error(endpoint, "NO_KEYS", message)
    logger.warning("RAPIDAPI: %s. A coleta atual será ignorada; o aplicativo continuará ativo.", message)

def _get_api_url_lock(url):
    with _api_cache_lock:
        return _api_url_locks.setdefault(url, threading.Lock())

def get_api_request_metrics(reset=False):
    """Retorna contadores reais da RapidAPI (tentativas HTTP, cache e chamadas lógicas)."""
    with _api_cache_lock:
        snapshot = dict(_api_metrics)
        if reset:
            for key in _api_metrics:
                _api_metrics[key] = 0
    return snapshot

def _rapidapi_key_id(key):
    return hashlib.sha256(key.encode('utf-8')).hexdigest()[:16]

def _rapidapi_header_int(headers, *names):
    for name in names:
        try:
            return int(headers.get(name))
        except (TypeError, ValueError, AttributeError):
            continue
    return None

def _rapidapi_reset_epoch(headers, now_epoch=None):
    """Converte o reset da RapidAPI (segundos restantes ou epoch) em epoch."""
    now_epoch = float(now_epoch or time.time())
    raw = None
    for name in ('x-ratelimit-requests-reset', 'X-RateLimit-Requests-Reset'):
        try:
            raw = float(headers.get(name))
            break
        except (TypeError, ValueError, AttributeError):
            continue
    if raw is None or raw <= 0:
        return 0.0
    if raw > 10_000_000_000:
        raw /= 1000.0
    return raw if raw > now_epoch - 86400 else now_epoch + raw

def _sync_rapidapi_response(key_id, idx, response, status_override=None, cooldown=0):
    """Faz o banco local obedecer ao saldo/reset informados pelo servidor."""
    global _key_rotation_cursor
    usage_date = get_brt_time().strftime("%Y-%m-%d")
    remaining = _rapidapi_header_int(
        response.headers,
        'x-ratelimit-requests-remaining', 'X-RateLimit-Requests-Remaining')
    server_limit = _rapidapi_header_int(
        response.headers,
        'x-ratelimit-requests-limit', 'X-RateLimit-Requests-Limit')
    reset_epoch = _rapidapi_reset_epoch(response.headers)
    if remaining is not None:
        effective_limit = int(server_limit or RAPIDAPI_DAILY_LIMIT)
        request_count = max(0, min(RAPIDAPI_DAILY_LIMIT, effective_limit - remaining))
    else:
        request_count = None
    if status_override:
        status = status_override
    else:
        status = 'exhausted' if remaining is not None and remaining <= 0 else 'active'
    if status == 'blocked':
        blocked_until = time.time() + max(1, cooldown)
    elif status in ('active', 'exhausted'):
        blocked_until = reset_epoch
    else:
        blocked_until = 0.0
    with _key_rotation_lock:
        conn = get_db_connection()
        try:
            conn.execute("""UPDATE rapidapi_key_usage SET
                request_count=COALESCE(?, request_count), status=?, blocked_until=?,
                last_http_status=? WHERE key_id=? AND usage_date=?""",
                (request_count, status, blocked_until, str(response.status_code),
                 key_id, usage_date))
            conn.commit()
        finally:
            conn.close()
        if status in ('exhausted', 'invalid', 'blocked'):
            _key_rotation_cursor = (idx + 1) % len(RAPIDAPI_KEYS)

def _reserve_rapidapi_key():
    """Reserva atomicamente uma chamada sem ultrapassar 100/dia por chave."""
    global _key_rotation_cursor
    usage_date = get_brt_time().strftime("%Y-%m-%d")
    now_epoch = time.time()
    with _key_rotation_lock:
        conn = get_db_connection()
        try:
            conn.execute("BEGIN IMMEDIATE")
            for offset in range(len(RAPIDAPI_KEYS)):
                idx = (_key_rotation_cursor + offset) % len(RAPIDAPI_KEYS)
                key = RAPIDAPI_KEYS[idx]
                key_id = _rapidapi_key_id(key)
                conn.execute("""INSERT OR IGNORE INTO rapidapi_key_usage
                    (key_id, usage_date, request_count, status, blocked_until)
                    VALUES (?,?,0,'active',0)""", (key_id, usage_date))
                count, status, blocked_until = conn.execute("""SELECT request_count, status, blocked_until
                    FROM rapidapi_key_usage WHERE key_id=? AND usage_date=?""",
                    (key_id, usage_date)).fetchone()
                blocked_until = float(blocked_until or 0)
                if blocked_until > 0 and blocked_until <= now_epoch and status != 'invalid':
                    if status == 'blocked':
                        status, blocked_until = 'active', 0.0
                    else:
                        count, status, blocked_until = 0, 'active', 0.0
                    conn.execute("""UPDATE rapidapi_key_usage
                        SET request_count=?, status=?, blocked_until=?
                        WHERE key_id=? AND usage_date=?""",
                        (count, status, blocked_until, key_id, usage_date))
                if count >= RAPIDAPI_DAILY_LIMIT or status in ('exhausted', 'invalid'):
                    continue
                if status == 'blocked' and float(blocked_until or 0) > now_epoch:
                    continue
                conn.execute("""UPDATE rapidapi_key_usage SET request_count=request_count+1,
                    status='active' WHERE key_id=? AND usage_date=?""",
                    (key_id, usage_date))
                _key_rotation_cursor = idx
                conn.commit()
                return key, key_id, idx
            conn.commit()
            return None, None, None
        finally:
            conn.close()

def _mark_rapidapi_key(key_id, idx, status, http_status, cooldown=0):
    global _key_rotation_cursor
    usage_date = get_brt_time().strftime("%Y-%m-%d")
    with _key_rotation_lock:
        with get_db_connection() as conn:
            conn.execute("""UPDATE rapidapi_key_usage SET status=?, blocked_until=?, last_http_status=?
                WHERE key_id=? AND usage_date=?""",
                (status, time.time() + cooldown if cooldown else 0, str(http_status), key_id, usage_date))
        if status in ('exhausted', 'invalid', 'blocked'):
            _key_rotation_cursor = (idx + 1) % len(RAPIDAPI_KEYS)

def get_rapidapi_key_usage():
    usage_date = get_brt_time().strftime("%Y-%m-%d")
    with get_db_connection() as conn:
        rows = {r[0]: r[1:] for r in conn.execute("""SELECT key_id, request_count, status,
            blocked_until, last_http_status FROM rapidapi_key_usage WHERE usage_date=?""", (usage_date,))}
    result = []
    for idx, key in enumerate(RAPIDAPI_KEYS, 1):
        count, status, blocked, last_status = rows.get(_rapidapi_key_id(key), (0, 'active', 0, None))
        if count >= RAPIDAPI_DAILY_LIMIT:
            status = 'exhausted'
        result.append({'key': idx, 'used': count, 'remaining': max(0, RAPIDAPI_DAILY_LIMIT-count),
                       'status': status, 'last_http_status': last_status})
    return result

def _cached_api_response(url):
    if RAPIDAPI_CACHE_TTL_SECONDS <= 0:
        return None
    with _api_cache_lock:
        cached = _api_response_cache.get(url)
        if cached and time.monotonic() - cached[0] <= RAPIDAPI_CACHE_TTL_SECONDS:
            _api_metrics["cache_hits"] += 1
            return cached[1]
        if cached:
            _api_response_cache.pop(url, None)
    return None

def safe_api_get(url, max_retries=2, timeout=20):
    global _last_request_time
    with _api_cache_lock:
        _api_metrics["logical_calls"] += 1
    cached = _cached_api_response(url)
    if cached is not None:
        return cached

    # Um lock por URL evita que os workers façam a mesma chamada simultaneamente.
    with _get_api_url_lock(url):
        cached = _cached_api_response(url)
        if cached is not None:
            return cached
        failures = 0
        while failures < max_retries:
            api_key, key_id, key_idx = _reserve_rapidapi_key()
            if not api_key:
                _notify_no_rapidapi_keys(url)
                return None
            with _rate_limiter:
                with _rate_lock:
                    now = time.time()
                    if now - _last_request_time < RAPIDAPI_MIN_INTERVAL_SECONDS:
                        time.sleep(RAPIDAPI_MIN_INTERVAL_SECONDS - (now - _last_request_time))
                    _last_request_time = time.time()
                try:
                    with _api_cache_lock:
                        _api_metrics["http_requests"] += 1
                    request_headers = dict(HEADERS)
                    request_headers["x-rapidapi-key"] = api_key
                    res = requests.get(url, headers=request_headers, timeout=timeout)
                    if res.status_code == 200:
                        _sync_rapidapi_response(key_id, key_idx, res)
                        data = res.json()
                        with _api_cache_lock:
                            _api_response_cache[url] = (time.monotonic(), data)
                        return data
                    elif res.status_code == 204:
                        _sync_rapidapi_response(key_id, key_idx, res)
                        return {}
                    elif res.status_code == 429:
                        remaining = res.headers.get('x-ratelimit-requests-remaining')
                        quota_exceeded = remaining == '0' or 'quota' in res.text.lower()
                        _sync_rapidapi_response(
                            key_id, key_idx, res,
                            status_override='exhausted' if quota_exceeded else 'blocked',
                            cooldown=0 if quota_exceeded else 2)
                        continue
                    elif res.status_code in (401, 403):
                        _sync_rapidapi_response(key_id, key_idx, res, status_override='invalid')
                        continue
                    elif res.status_code in (400, 404, 410):
                        _sync_rapidapi_response(key_id, key_idx, res)
                        # Erro de rota/parâmetro não melhora trocando de chave
                        # nem repetindo a mesma chamada.
                        log_api_error(url, res.status_code, res.text[:200])
                        return None
                    else:
                        _sync_rapidapi_response(key_id, key_idx, res)
                        log_api_error(url, res.status_code, res.text[:100])
                        failures += 1
                        time.sleep(1)
                except requests.exceptions.Timeout:
                    log_api_error(url, "Timeout", f"Falha {failures+1}")
                    failures += 1
                    time.sleep(2)
                except Exception as e:
                    log_api_error(url, "Exception", str(e))
                    failures += 1
                    time.sleep(1)
    return None

def usar_cutoff_temporal(func):
    """Impede que jogos posteriores ao alvo entrem nas features históricas."""
    def wrapper(match_id, *args, **kwargs):
        cutoff = None
        try:
            info = obter_info_partida(match_id)
            cutoff = float(info.get('startTimestamp') or 0) if info else None
        except Exception:
            cutoff = None
        token = _feature_cutoff_timestamp.set(cutoff or None)
        try:
            return func(match_id, *args, **kwargs)
        finally:
            _feature_cutoff_timestamp.reset(token)
    return wrapper

def extrair_fracional(frac_str):
    try:
        if not frac_str: return 0.0
        if '/' in str(frac_str):
            n, d = str(frac_str).split('/')
            return round(float(n)/float(d)+1.0, 2)
        return float(frac_str)
    except: return 0.0

def extrair_v_e_d(form_str):
    if not form_str: return 0,0,0
    m = re.search(r'(\d+)V\s*[-]?\s*(\d+)E\s*[-]?\s*(\d+)D', form_str, re.IGNORECASE)
    if m: return int(m.group(1)), int(m.group(2)), int(m.group(3))
    v=e=d=0
    mv = re.search(r'(\d+)V', form_str)
    if mv: v = int(mv.group(1))
    me = re.search(r'(\d+)E', form_str)
    if me: e = int(me.group(1))
    md = re.search(r'(\d+)D', form_str)
    if md: d = int(md.group(1))
    return v,e,d

# --- INICIALIZAÇÃO DO BANCO DE DADOS ---
def init_db():
    with get_db_connection() as conn:
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS elo_rating (
            team_id TEXT PRIMARY KEY,
            elo INTEGER,
            last_update DATETIME
        )''')
        c.execute('''CREATE TABLE IF NOT EXISTS cache_standings (
            id_torneio TEXT PRIMARY KEY,  -- "unique_tournament_id_season_id"
            dados_json TEXT,
            data_captura DATETIME)''')
        c.execute('''CREATE TABLE IF NOT EXISTS previsoes (
            id INTEGER PRIMARY KEY AUTOINCREMENT, match_id TEXT UNIQUE, timestamp DATETIME, confronto TEXT,
            liga TEXT, odd_casa REAL, odd_fora REAL, odd_empate REAL, vencedor_previsto TEXT, confianca INTEGER,
            scout_report TEXT, status_resultado TEXT DEFAULT 'PENDENTE', placar_real TEXT DEFAULT '-',
            aprendizado_ia TEXT, ticket_id TEXT, telegram_msg_id TEXT, data_jogo TEXT, hora_jogo TEXT,
            lesoes_jogadores TEXT)''')
        for col, tipo in [
            ('antecipado_detectado', 'INTEGER DEFAULT 0'), ('anulado', 'INTEGER DEFAULT 0'),
            ('telegram_enviado', 'INTEGER DEFAULT 0'), ('tournament_id', 'TEXT'),
            ('season_id', 'TEXT'), ('unique_tournament_id', 'TEXT'),
            ('quase_acerto_notificado', 'INTEGER DEFAULT 0'),
            ('selecionado_radar', 'INTEGER DEFAULT 0'),
             ('ml_model_version', 'INTEGER DEFAULT 0'), ('ml_model_id', 'TEXT'),
             ('start_timestamp', 'INTEGER DEFAULT 0'), ('audit_attempts', 'INTEGER DEFAULT 0'),
             ('audit_next_at', 'INTEGER DEFAULT 0'), ('audit_last_error', 'TEXT'),
             ('draw_risk_score', 'REAL DEFAULT 0'),
             ('context_quality_score', 'REAL DEFAULT 0'),
             ('context_conflict_score', 'REAL DEFAULT 0'),
             ('analysis_version', 'TEXT'), ('postmortem_verdict', 'TEXT'),
             ('postmortem_process_score', 'REAL'), ('postmortem_coverage', 'REAL')]:
            try:
                c.execute(f"ALTER TABLE previsoes ADD COLUMN {col} {tipo}")
            except sqlite3.OperationalError:
                pass
        c.execute('''CREATE TABLE IF NOT EXISTS autopsias_liga (
            liga TEXT PRIMARY KEY, relatorio_geral TEXT, jogos_processados INTEGER DEFAULT 0,
            ultima_atualizacao DATETIME)''')
        c.execute('''CREATE TABLE IF NOT EXISTS aprendizado_global (
            id INTEGER PRIMARY KEY AUTOINCREMENT, data_analise DATETIME, relatorio TEXT, jogos_analisados INTEGER)''')
        c.execute('''CREATE TABLE IF NOT EXISTS aprendizado_liga (
            liga TEXT PRIMARY KEY, data_analise DATETIME, total_jogos INTEGER, taxa_acerto REAL,
            xG_casa_medio REAL, xG_fora_medio REAL, posse_casa_medio REAL, posse_fora_medio REAL, regras TEXT)''')
        c.execute('''CREATE TABLE IF NOT EXISTS aprendizado_time (
            time TEXT PRIMARY KEY, liga TEXT, data_analise DATETIME, total_jogos INTEGER, taxa_acerto REAL,
            xG_medio REAL, posse_media REAL, chutes_medio REAL, motivo_principal TEXT)''')
        c.execute('''CREATE TABLE IF NOT EXISTS cache_xg_times (
            team_id TEXT, tournament_id TEXT, season_id TEXT, xg_medio REAL, gols_marcados_medio REAL,
            gols_sofridos_medio REAL, ultima_atualizacao DATETIME,
            PRIMARY KEY (team_id, tournament_id, season_id))''')
        c.execute('''CREATE TABLE IF NOT EXISTS cache_jogos_liga (
            tournament_id TEXT, season_id TEXT, last_match_id TEXT PRIMARY KEY,
            empates_acumulados INTEGER DEFAULT 0, total_jogos_acumulados INTEGER DEFAULT 0,
            ultima_atualizacao DATETIME)''')
        c.execute('''CREATE TABLE IF NOT EXISTS mapeamento_ligas (
            liga_nome TEXT PRIMARY KEY, tournament_id TEXT, season_id TEXT, ultima_atualizacao DATETIME)''')
        c.execute('''CREATE TABLE IF NOT EXISTS drift_reference (
            liga TEXT PRIMARY KEY, feature_stats TEXT, data_referencia DATETIME)''')
        c.execute('''CREATE TABLE IF NOT EXISTS estatisticas_medias_liga (
            liga TEXT PRIMARY KEY, num_jogos INTEGER, xg_casa_medio REAL, xg_fora_medio REAL,
            posse_casa_medio REAL, posse_fora_medio REAL, chutes_casa_medio REAL, chutes_fora_medio REAL,
            ef_of_home_medio REAL, ef_df_home_medio REAL, ef_of_away_medio REAL, ef_df_away_medio REAL,
            empates_medio REAL, avg_cards_medio REAL)''')
        c.execute('''CREATE TABLE IF NOT EXISTS training_data (
            match_id TEXT PRIMARY KEY, liga TEXT, data_jogo DATETIME, home_team TEXT, away_team TEXT,
            home_score INTEGER, away_score INTEGER, odd_casa REAL, odd_empate REAL, odd_fora REAL,
            features TEXT, usado_treinamento INTEGER DEFAULT 0, data_importacao DATETIME DEFAULT CURRENT_TIMESTAMP)''')
        c.execute("CREATE INDEX IF NOT EXISTS idx_training_data_date ON training_data(data_jogo)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_training_data_home_date ON training_data(home_team, data_jogo)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_training_data_away_date ON training_data(away_team, data_jogo)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_training_data_liga_date ON training_data(liga, data_jogo)")
        c.execute('''CREATE TABLE IF NOT EXISTS ml_team_ratings (
            team_name TEXT PRIMARY KEY, elo REAL NOT NULL, updated_at DATETIME NOT NULL)''')
        c.execute('''CREATE TABLE IF NOT EXISTS rapidapi_key_usage (
            key_id TEXT, usage_date TEXT, request_count INTEGER DEFAULT 0,
            status TEXT DEFAULT 'active', blocked_until REAL DEFAULT 0,
            last_http_status TEXT, PRIMARY KEY (key_id, usage_date))''')
        c.execute('''CREATE TABLE IF NOT EXISTS modelos_ml (
            liga TEXT PRIMARY KEY, data_treinamento DATETIME, num_amostras INTEGER,
            modelo_blob BLOB, scaler_params TEXT, feature_order TEXT, acuracia REAL, log_loss REAL)''')
        c.execute('''CREATE TABLE IF NOT EXISTS training_weights (
            match_id TEXT PRIMARY KEY, peso REAL DEFAULT 1.0, data_ultima_atualizacao DATETIME)''')
        # Tabela cache_estatisticas_partida
        c.execute('''CREATE TABLE IF NOT EXISTS cache_estatisticas_partida (
            match_id TEXT PRIMARY KEY,
            stats_json TEXT,
            data_captura DATETIME
        )''')
        for table, col, tipo in [
            ('training_data', 'tournament_id', 'TEXT'), ('training_data', 'season_id', 'TEXT'),
            ('training_data', 'unique_tournament_id', 'TEXT'),
            ('training_weights', 'error_margin', 'REAL DEFAULT 0')]:
            try:
                c.execute(f"ALTER TABLE {table} ADD COLUMN {col} {tipo}")
            except sqlite3.OperationalError:
                pass
        c.execute("CREATE INDEX IF NOT EXISTS idx_training_data_liga_used ON training_data(liga, usado_treinamento)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_previsoes_status ON previsoes(status_resultado, antecipado_detectado)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_previsoes_ticket ON previsoes(ticket_id)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_previsoes_radar_status ON previsoes(selecionado_radar, status_resultado)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_previsoes_model_status ON previsoes(ml_model_id, status_resultado)")
        c.execute("""CREATE INDEX IF NOT EXISTS idx_previsoes_audit
                     ON previsoes(status_resultado, start_timestamp, audit_next_at, audit_attempts)""")
        c.execute("CREATE INDEX IF NOT EXISTS idx_cache_stats_date ON cache_estatisticas_partida(data_captura)")
        ensure_radar_audit_schema(conn)
        c.execute("CREATE INDEX IF NOT EXISTS idx_api_error_timestamp ON api_error_log(timestamp)")
        try:
            c.execute("ALTER TABLE modelos_ml ADD COLUMN roc_auc REAL")
        except sqlite3.OperationalError:
            pass  # coluna já existe
        try:
            c.execute("ALTER TABLE modelos_ml ADD COLUMN model_version INTEGER DEFAULT 1")
        except sqlite3.OperationalError:
            pass
        ensure_evolution_tables(conn, MIN_ML_CONFIDENCE)
init_db()
init_soccer_context_db(DB_NAME)
init_sofascore_db(DB_NAME)

# ----------------------------------------------------------------------
# FUNÇÃO AUXILIAR COM PAGINAÇÃO (SUBSTITUI AS CHAMADAS ANTERIORES)
# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
# FUNÇÃO AUXILIAR PARA EXTRAIR ESTATÍSTICAS DA ALLSPORTS (SUA VERSÃO ATUAL)
# ----------------------------------------------------------------------
# ==================================================
# FUNÇÕES AUXILIARES PARA O FALLBACK
# ==================================================
def salvar_standings_cache(unique_tournament_id, season_id, dados):
    key = f"{unique_tournament_id}_{season_id}"
    with get_db_connection() as conn:
        conn.execute("INSERT OR REPLACE INTO cache_standings (id_torneio, dados_json, data_captura) VALUES (?,?,?)",
                     (key, json.dumps(dados), get_brt_time().strftime("%Y-%m-%d %H:%M:%S")))

def carregar_standings_cache(unique_tournament_id, season_id):
    key = f"{unique_tournament_id}_{season_id}"
    with get_db_connection() as conn:
        row = conn.execute("SELECT dados_json FROM cache_standings WHERE id_torneio = ?", (key,)).fetchone()
        if row:
            return json.loads(row[0])
    return None
def obter_prioridade_torneio(liga_name, is_knockout=0, is_final=0):
    """
    Retorna prioridade de 1 a 10 para o torneio.
    is_final: 1 se for final (detectado por palavras como 'final', 'cup final')
    """
    nivel = get_nivel_campeonato(liga_name)
    prioridade = nivel * 2  # nivel 5 -> 10, nivel 4 -> 8, nivel 3 -> 6, nivel 2 -> 4, nivel 1 -> 2
    if is_knockout:
        prioridade += 1
    if is_final:
        prioridade += 1
    return min(prioridade, 10)

def contar_jogos_ultimos_dias(team_id, dias=7):
    """Retorna quantos jogos o time disputou nos últimos X dias (apenas finalizados)."""
    referencia_ts = _feature_cutoff_timestamp.get()
    referencia = (datetime.fromtimestamp(referencia_ts, tz=timezone.utc)
                  if referencia_ts else get_brt_time())
    cutoff = referencia - timedelta(days=dias)
    eventos = obter_todos_ultimos_jogos(team_id, max_jogos=50)
    count = 0
    for ev in eventos:
        if ev.get('status', {}).get('type') == 'finished':
            ts = ev.get('startTimestamp', 0)
            if ts:
                dt_jogo = datetime.fromtimestamp(ts, tz=timezone.utc).astimezone(timezone(timedelta(hours=-3)))
                if dt_jogo >= cutoff:
                    count += 1
    return count

def obter_odds_com_fallback(match_id):
    """
    Tenta obter odds do mercado 1X2 na seguinte ordem:
    1. Endpoint v2.0: /api/match/{id}/odds/1/all
    2. Endpoint alternativo: /api/match/{id}/provider/1/winning-odds
    Retorna (odd_casa, odd_empate, odd_fora) ou (0,0,0) se nenhum funcionar.
    """
    # 1. Endpoint explícito da v2.0 (provedor 1, todos os mercados)
    odds_data = safe_api_get(match_odds_url(RAPIDAPI_HOST, match_id))
    if odds_data and 'markets' in odds_data:
        for market in odds_data['markets']:
            if market.get('marketGroup') == '1X2':
                odd_casa = odd_empate = odd_fora = 0.0
                for choice in market.get('choices', []):
                    if choice.get('name') == '1':
                        odd_casa = extrair_fracional(choice.get('fractionalValue'))
                    elif choice.get('name') == 'X':
                        odd_empate = extrair_fracional(choice.get('fractionalValue'))
                    elif choice.get('name') == '2':
                        odd_fora = extrair_fracional(choice.get('fractionalValue'))
                if odd_casa > 0 and odd_fora > 0:
                    return odd_casa, odd_empate, odd_fora
        # Se chegou aqui, não encontrou 1X2
        print(f"[ODDS] Endpoint padrão não retornou 1X2 para {match_id}")

    # 2. Endpoint alternativo: winning-odds
    alt_url = match_winning_odds_url(RAPIDAPI_HOST, match_id)
    alt_data = safe_api_get(alt_url)
    if alt_data:
        odd_casa = odd_empate = odd_fora = 0.0
        # Tentar diferentes estruturas de resposta
        if 'homeWin' in alt_data:
            odd_casa = extrair_fracional(alt_data.get('homeWin'))
            odd_empate = extrair_fracional(alt_data.get('draw'))
            odd_fora = extrair_fracional(alt_data.get('awayWin'))
        elif 'home' in alt_data:
            odd_casa = extrair_fracional(alt_data.get('home'))
            odd_empate = extrair_fracional(alt_data.get('draw')) or extrair_fracional(alt_data.get('tie'))
            odd_fora = extrair_fracional(alt_data.get('away'))
        elif 'choices' in alt_data:
            for choice in alt_data['choices']:
                if choice.get('name') == '1':
                    odd_casa = extrair_fracional(choice.get('fractionalValue'))
                elif choice.get('name') == 'X':
                    odd_empate = extrair_fracional(choice.get('fractionalValue'))
                elif choice.get('name') == '2':
                    odd_fora = extrair_fracional(choice.get('fractionalValue'))
        if odd_casa > 0 and odd_fora > 0:
            print(f"[ODDS] Endpoint winning-odds OK para {match_id}")
            return odd_casa, odd_empate, odd_fora

    print(f"[ODDS] Falha ao obter odds para {match_id}")
    return 0.0, 0.0, 0.0

def recalcular_todos_elos():
    """
    Recalcula o Elo rating de todos os times usando os jogos já finalizados
    na tabela training_data (ordem cronológica).
    """
    conn = get_db_connection()
    jogos = conn.execute("""
        SELECT home_team, away_team, home_score, away_score, data_jogo
        FROM training_data
        ORDER BY data_jogo ASC
    """).fetchall()
    conn.close()

    if not jogos:
        print("Nenhum jogo encontrado para recalcular Elo.")
        return

    # Dicionário para armazenar Elos
    elos = {}

    def obter_elo(team_name):
        return elos.get(team_name, 1500)

    def atualizar_elo(team_name, opponent_elo, resultado):
        elo_atual = obter_elo(team_name)
        expected = 1 / (1 + 10 ** ((opponent_elo - elo_atual) / 400))
        novo_elo = elo_atual + 32 * (resultado - expected)
        elos[team_name] = novo_elo
        return novo_elo

    total = len(jogos)
    print(f"Recalculando Elos para {total} jogos...")

    for idx, (home_team, away_team, home_score, away_score, _) in enumerate(jogos):
        # Determinar resultado
        if home_score > away_score:
            res_h, res_a = 1.0, 0.0
        elif home_score < away_score:
            res_h, res_a = 0.0, 1.0
        else:
            res_h, res_a = 0.5, 0.5

        elo_h = obter_elo(home_team)
        elo_a = obter_elo(away_team)

        atualizar_elo(home_team, elo_a, res_h)
        atualizar_elo(away_team, elo_h, res_a)

        if (idx + 1) % 1000 == 0:
            print(f"Processados {idx+1}/{total} jogos...")

    # Salvar resultados
    conn = get_db_connection()
    # Garantir tabela existe
    conn.execute("""CREATE TABLE IF NOT EXISTS elo_rating (
        team_id TEXT PRIMARY KEY,
        elo INTEGER,
        last_update DATETIME
    )""")
    for team_name, elo in elos.items():
        conn.execute("INSERT OR REPLACE INTO elo_rating (team_id, elo, last_update) VALUES (?, ?, ?)",
                     (team_name, int(elo), get_brt_time().isoformat()))
    conn.commit()
    conn.close()

    print(f"Elo recalculado para {len(elos)} times.")

def dias_ate_proximo_jogo_prioritario(team_id, prioridade_atual):
    """
    Retorna dias (float) até o próximo jogo do time que tenha prioridade maior que prioridade_atual.
    Se não houver, retorna 999.
    """
    cutoff = get_brt_time()
    eventos = obter_proximos_jogos(team_id, max_jogos=20)
    menor_dias = 999.0
    for ev in eventos:
        ts = ev.get('startTimestamp', 0)
        if ts and ts > cutoff.timestamp():
            tourn = ev.get('tournament', {})
            cat = tourn.get('category', {})
            liga_completa = f"{cat.get('name', '')} - {tourn.get('name', '')}"
            is_knockout = detectar_fase_mata_mata(liga_completa)[0]
            is_final = 1 if 'final' in liga_completa.lower() else 0
            prioridade = obter_prioridade_torneio(liga_completa, is_knockout, is_final)
            if prioridade > prioridade_atual:
                dias = (ts - cutoff.timestamp()) / 86400.0
                if dias < menor_dias:
                    menor_dias = dias
    return menor_dias

def obter_media_elo_adversarios(team_id, tournament_id, season_id, n=5):
    """
    Retorna a média do Elo dos últimos n adversários do time (na mesma competição).
    """
    eventos = obter_todos_ultimos_jogos(team_id, max_jogos=n*2)
    elos = []
    count = 0
    for ev in eventos:
        if ev.get('status', {}).get('type') != 'finished':
            continue
        ev_tourn_id = str(ev.get('tournament', {}).get('id', ''))
        if tournament_id and ev_tourn_id != tournament_id:
            continue
        is_home = str(ev.get('homeTeam', {}).get('id')) == str(team_id)
        if is_home:
            adv_id = str(ev.get('awayTeam', {}).get('id', ''))
        else:
            adv_id = str(ev.get('homeTeam', {}).get('id', ''))
        if adv_id:
            conn = get_db_connection()
            cur = conn.cursor()
            cur.execute("SELECT elo FROM elo_rating WHERE team_id = ?", (adv_id,))
            row = cur.fetchone()
            elo = row[0] if row else 1500
            conn.close()
            elos.append(elo)
        count += 1
        if count >= n:
            break
    if elos:
        return sum(elos) / len(elos)
    return 1500

def calcular_streak(team_id, tournament_id=None, n=5):
    """
    Retorna streak atual: positiva (vitórias consecutivas), negativa (derrotas consecutivas).
    Zero indica empate ou nenhum jogo.
    """
    eventos = obter_todos_ultimos_jogos(team_id, max_jogos=n)
    streak = 0
    for ev in eventos:
        if ev.get('status', {}).get('type') != 'finished':
            continue
        ev_tourn_id = str(ev.get('tournament', {}).get('id', ''))
        if tournament_id and ev_tourn_id != tournament_id:
            continue
        is_home = str(ev.get('homeTeam', {}).get('id')) == str(team_id)
        score = regulation_score(ev)
        if score is None:
            continue
        hs, aws = score
        if (hs > aws and is_home) or (aws > hs and not is_home):
            if streak >= 0:
                streak += 1
            else:
                break
        elif (hs < aws and is_home) or (aws < hs and not is_home):
            if streak <= 0:
                streak -= 1
            else:
                break
        else:  # empate
            break
    return streak

def fetch_flashscore_stats(match_id_allsports, home_team, away_team, match_date_timestamp):
    """
    Busca estatísticas do FlashScore usando os dados da partida.
    """
    match_date = datetime.fromtimestamp(match_date_timestamp).strftime("%Y-%m-%d")
    search_query = f"{home_team} {away_team} {match_date}".replace(" ", "%20")
    # URL da API de busca do FlashScore
    search_url = f"https://www.flashscore.com/?s={search_query}"

    stats = {}
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto(search_url)

        # 1. Aguarda o carregamento dos resultados da busca
        page.wait_for_selector(".event__match", timeout=10000)

        # 2. Extrai os dados do primeiro link de partida encontrado
        first_match_link = page.query_selector("a.event__match")
        if first_match_link:
            match_link = first_match_link.get_attribute("href")
            # O ID do jogo geralmente está na URL (ex: #gid_12345678)
            match_id_flashscore = match_link.split("#gid_")[-1]

            if match_id_flashscore:
                # 3. Faz a requisição para a API de estatísticas usando o ID correto
                stats_api_url = f"https://d.flashscore.com/x/feed/d_1_{match_id_flashscore}_en_1"
                page.goto(stats_api_url)
                content = page.content()
                stats = extract_flashscore_stats_from_json(content)

        browser.close()
    return stats

def extract_flashscore_stats_from_json(json_data):
    """
    Extrai os campos de estatística do JSON retornado pela API do FlashScore.
    """
    stats = {}
    # A estrutura do JSON do FlashScore é diferente da do SofaScore
    # Exemplo de extração (adapte conforme o retorno real):
    try:
        data = json.loads(json_data)
        # Exemplo para posse de bola (verifique os nomes dos campos reais)
        stats['posse_home'] = data.get('stat', {}).get('possessionHome', 0)
        stats['posse_away'] = data.get('stat', {}).get('possessionAway', 0)
        # ... extraia outros campos como xG, chutes, etc.
    except:
        pass
    return stats
# Cache para estatísticas do SofaScore (evita chamadas repetidas)
SOFASCORE_CACHE = {}
SOFASCORE_CACHE_LOCK = threading.Lock()

def extrair_estatisticas_do_json_sofascore(data):
    """Extrai estatísticas do JSON retornado pela API do SofaScore."""
    stats = {}
    if not data or 'statistics' not in data:
        return stats
    for period in data.get('statistics', []):
        if period.get('period') == 'ALL':
            for group in period.get('groups', []):
                for item in group.get('statisticsItems', []):
                    name = item.get('name')
                    def get_val(side):
                        val = item.get(f'{side}Value')
                        if val is not None:
                            return float(val)
                        raw = item.get(side, 0)
                        if isinstance(raw, (int, float)):
                            return float(raw)
                        if isinstance(raw, str):
                            raw = raw.replace('%', '').strip()
                            raw = re.sub(r'[^\d.-]', '', raw)
                            try:
                                return float(raw)
                            except:
                                return 0.0
                        return 0.0
                    if name == 'Ball possession':
                        stats['posse_home'] = get_val('home')
                        stats['posse_away'] = get_val('away')
                    elif name == 'Expected goals':
                        stats['xg_home'] = get_val('home')
                        stats['xg_away'] = get_val('away')
                    elif name == 'Total shots':
                        stats['chutes_home'] = get_val('home')
                        stats['chutes_away'] = get_val('away')
                    elif name == 'Shots on target':
                        stats['remates_gol_home'] = get_val('home')
                        stats['remates_gol_away'] = get_val('away')
                        stats['chutes_gol_home'] = stats['remates_gol_home']
                        stats['chutes_gol_away'] = stats['remates_gol_away']
                    elif name == 'Corner kicks':
                        stats['cantos_home'] = get_val('home')
                        stats['cantos_away'] = get_val('away')
                    elif name == 'Fouls':
                        stats['faltas_home'] = get_val('home')
                        stats['faltas_away'] = get_val('away')
                    elif name == 'Yellow cards':
                        stats['cartoes_home'] = get_val('home')
                        stats['cartoes_away'] = get_val('away')
    return stats

def salvar_estatisticas_cache(match_id, stats_dict):
    with get_db_connection() as conn:
        conn.execute("INSERT OR REPLACE INTO cache_estatisticas_partida (match_id, stats_json, data_captura) VALUES (?,?,?)",
                     (match_id, json.dumps(stats_dict), get_brt_time().strftime("%Y-%m-%d %H:%M:%S")))

def carregar_estatisticas_cache(match_id):
    with get_db_connection() as conn:
        row = conn.execute("SELECT stats_json FROM cache_estatisticas_partida WHERE match_id = ?", (match_id,)).fetchone()
        if row:
            return json.loads(row[0])
    return None

def buscar_estatisticas_sofascore(match_id):
    import concurrent.futures

    cache_key = str(match_id)
    if cache_key in CACHE_ESTATISTICAS_SOFASCORE:
        return CACHE_ESTATISTICAS_SOFASCORE[cache_key]
    cached = carregar_estatisticas_cache(match_id)
    if cached:
        CACHE_ESTATISTICAS_SOFASCORE[cache_key] = cached
        return cached

    def _fetch():
        try:
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True)
                page = browser.new_page()
                page.set_extra_http_headers({
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                    "Accept": "application/json",
                    "Referer": "https://www.sofascore.com/"
                })
                url = f"https://www.sofascore.com/api/v1/event/{match_id}/statistics"
                page.goto(url, timeout=15000)
                content = page.content()
                browser.close()
                start = content.find('{')
                end = content.rfind('}') + 1
                if start != -1 and end != 0:
                    json_str = content[start:end]
                    data = json.loads(json_str)
                    if 'statistics' in data:
                        return extrair_estatisticas_do_json_sofascore(data)
                return {}
        except Exception as e:
            print(f"[ERRO] _fetch stats: {e}")
            return {}

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(_fetch)
        try:
            result = future.result(timeout=25)
            salvar_estatisticas_cache(match_id, result)
            CACHE_ESTATISTICAS_SOFASCORE[cache_key] = result
            return result
        except concurrent.futures.TimeoutError:
            print(f"[TIMEOUT] Estatísticas para match {match_id}")
            salvar_estatisticas_cache(match_id, {})
            return {}
        except Exception as e:
            print(f"[ERRO] Estatísticas: {e}")
            return {}

def buscar_estatisticas_por_busca(home_team, away_team, timestamp):
    import concurrent.futures

    def _fetch():
        try:
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True)
                page = browser.new_page()
                page.set_extra_http_headers({
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                    "Accept": "application/json",
                    "Referer": "https://www.sofascore.com/"
                })
                dt = datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d")
                query = f"{home_team} {away_team} {dt}".replace(" ", "%20")
                search_url = f"https://www.sofascore.com/api/v1/search?q={query}"
                page.goto(search_url, timeout=15000)
                content = page.content()
                browser.close()
                match = re.search(r'(\{.*\})', content, re.DOTALL)
                if match:
                    data = json.loads(match.group(1))
                    for result in data.get('results', []):
                        if result.get('type') == 'event':
                            event_id = result.get('id')
                            if event_id:
                                return buscar_estatisticas_sofascore(event_id)
                return {}
        except Exception as e:
            print(f"[ERRO] busca por nome: {e}")
            return {}

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(_fetch)
        try:
            return future.result(timeout=25)
        except concurrent.futures.TimeoutError:
            print(f"[TIMEOUT] Busca por nome {home_team} vs {away_team}")
            return {}


def obter_info_partida(match_id):
    """
    Obtém informações detalhadas da partida usando a API AllSports.
    Retorna dicionário com home_id, away_id, tournament_id, season_id, unique_tournament_id, etc.
    """
    event_data = safe_api_get(match_detail_url(RAPIDAPI_HOST, match_id))
    if event_data and 'event' in event_data:
        ev = event_data['event']
        tournament = ev.get('tournament', {})
        unique_tournament = tournament.get('uniqueTournament', {})
        return {
            'match_id': match_id,
            'home_id': str(ev.get('homeTeam', {}).get('id', '')),
            'away_id': str(ev.get('awayTeam', {}).get('id', '')),
            'tournament_id': str(tournament.get('id', '')),
            'unique_tournament_id': str(unique_tournament.get('id', '')),
            'season_id': str(ev.get('season', {}).get('id', '')),
            'home_team': ev.get('homeTeam', {}).get('name', ''),
            'away_team': ev.get('awayTeam', {}).get('name', ''),
            'startTimestamp': ev.get('startTimestamp', 0),
            'liga': f"{tournament.get('category', {}).get('name', 'Mundo')} - {tournament.get('name', 'Liga')}"
        }
    return None

from playwright.sync_api import sync_playwright

import concurrent.futures

# Blacklist global para torneios que já falharam
SOFASCORE_BLACKLIST = set()

def get_standings_from_sofascore(unique_tournament_id, season_id):
    cache_key = f"standings_{unique_tournament_id}_{season_id}"
    if cache_key in CACHE_STANDINGS:
        return CACHE_STANDINGS[cache_key]
    cached = carregar_standings_cache(unique_tournament_id, season_id)
    if cached:
        CACHE_STANDINGS[cache_key] = cached
        return cached

    # Verifica blacklist
    if f"{unique_tournament_id}_{season_id}" in SOFASCORE_BLACKLIST:
        return None

    def _fetch():
        try:
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True)
                page = browser.new_page()
                page.set_extra_http_headers({
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                    "Accept": "application/json",
                    "Referer": "https://www.sofascore.com/"
                })
                url = f"https://www.sofascore.com/api/v1/unique-tournament/{unique_tournament_id}/season/{season_id}/standings/total"
                page.goto(url, timeout=15000)  # 15 segundos
                content = page.content()
                browser.close()
                # Extrai JSON
                try:
                    data = json.loads(content)
                except:
                    match = re.search(r'(\{.*\})', content, re.DOTALL)
                    if match:
                        data = json.loads(match.group(1))
                    else:
                        return None
                if data and 'standings' in data and data['standings']:
                    rows = data['standings'][0].get('rows', [])
                    standings_map = {}
                    for row in rows:
                        team = row.get('team', {})
                        team_id = team.get('id')
                        if team_id:
                            standings_map[team_id] = {
                                'position': row.get('position'),
                                'points': row.get('points'),
                                'matches': row.get('matches'),
                                'scores_for': row.get('scoresFor'),
                                'scores_against': row.get('scoresAgainst'),
                                'wins': row.get('wins'),
                                'draws': row.get('draws'),
                                'losses': row.get('losses'),
                                'name': team.get('name', '')
                            }
                    return standings_map
                return None
        except Exception as e:
            print(f"[ERRO] _fetch: {e}")
            return None

    # Executa com timeout de 25 segundos
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(_fetch)
        try:
            result = future.result(timeout=25)
            if result:
                salvar_standings_cache(unique_tournament_id, season_id, result)
                CACHE_STANDINGS[cache_key] = result
                return result
            else:
                # Falhou, adiciona à blacklist
                SOFASCORE_BLACKLIST.add(f"{unique_tournament_id}_{season_id}")
                return None
        except concurrent.futures.TimeoutError:
            print(f"[TIMEOUT] SofaScore para {unique_tournament_id}/{season_id} excedeu 25s")
            SOFASCORE_BLACKLIST.add(f"{unique_tournament_id}_{season_id}")
            return None
        except Exception as e:
            print(f"[ERRO] SofaScore: {e}")
            SOFASCORE_BLACKLIST.add(f"{unique_tournament_id}_{season_id}")
            return None

def obter_estatisticas_partida_all_sports(match_id):
    """Tenta obter estatísticas da AllSports via /api/match/{match_id}/statistics"""
    url = match_resource_url(RAPIDAPI_HOST, match_id, "statistics")
    data = safe_api_get(url, max_retries=1, timeout=8)
    if not data or 'statistics' not in data:
        return {}
    stats = {}
    try:
        for period in data.get('statistics', []):
            if period.get('period') == 'ALL':
                for group in period.get('groups', []):
                    for item in group.get('statisticsItems', []):
                        name = item.get('name')
                        def get_val(side):
                            val = item.get(f'{side}Value')
                            if val is not None:
                                return float(val)
                            raw = item.get(side, 0)
                            if isinstance(raw, (int, float)):
                                return float(raw)
                            if isinstance(raw, str):
                                raw = raw.replace('%', '').strip()
                                raw = re.sub(r'[^\d.-]', '', raw)
                                try:
                                    return float(raw)
                                except:
                                    return 0.0
                            return 0.0
                        if name == 'Expected goals':
                            stats['xg_home'] = get_val('home')
                            stats['xg_away'] = get_val('away')
                        elif name == 'Ball possession':
                            stats['posse_home'] = get_val('home')
                            stats['posse_away'] = get_val('away')
                        elif name == 'Total shots':
                            stats['chutes_home'] = get_val('home')
                            stats['chutes_away'] = get_val('away')
                        elif name == 'Shots on target':
                            stats['remates_gol_home'] = get_val('home')
                            stats['remates_gol_away'] = get_val('away')
                        elif name == 'Corner kicks':
                            stats['cantos_home'] = get_val('home')
                            stats['cantos_away'] = get_val('away')
                        elif name == 'Fouls':
                            stats['faltas_home'] = get_val('home')
                            stats['faltas_away'] = get_val('away')
                        elif name == 'Yellow cards':
                            stats['cartoes_home'] = get_val('home')
                            stats['cartoes_away'] = get_val('away')
        return stats
    except Exception as e:
        logger.error(f"Erro AllSports {match_id}: {e}")
        return {}

def obter_estatisticas_com_fallback(match_id, match_info=None):
    """
    Tenta obter estatísticas na ordem:
    1. AllSports (via API)
    2. SofaScore (via Playwright com timeout)
    3. Busca por nome (último recurso)
    Retorna dicionário com estatísticas (vazio se nenhum).
    """
    # 1. AllSports
    stats = obter_estatisticas_partida_all_sports(match_id)
    if stats and any(v != 0 for v in stats.values()):
        print(f"[ESTATS] {match_id}: AllSports OK")
        return stats

    # 2. SofaScore
    if not match_info:
        match_info = obter_info_partida(match_id)
    if match_info:
        stats = buscar_estatisticas_sofascore(match_id)
        if stats and any(v != 0 for v in stats.values()):
            print(f"[ESTATS] {match_id}: SofaScore OK")
            return stats
        # 3. Busca por nome/data
        home_team = match_info.get('home_team')
        away_team = match_info.get('away_team')
        start_ts = match_info.get('startTimestamp')
        if home_team and away_team and start_ts:
            stats = buscar_estatisticas_por_busca(home_team, away_team, start_ts)
            if stats and any(v != 0 for v in stats.values()):
                print(f"[ESTATS] {match_id}: SofaScore via busca OK")
                return stats
    print(f"[ESTATS] {match_id}: FALHA - sem estatísticas")
    return {}

def listar_features_zeradas(features_dict):
    zeros = [k for k, v in features_dict.items() if v == 0 or v is None]
    if zeros:
        return f"Features zeradas: {', '.join(zeros[:15])}" + ("..." if len(zeros)>15 else "")
    return "Todas as features não-zero."

def obter_todos_ultimos_jogos(team_id, max_jogos=50):
    """
    Retorna uma lista com todos os eventos finalizados do time,
    percorrendo as páginas da API enquanto houver 'hasNextPage' = true.
    """
    todos_eventos = []
    page = 0
    while len(todos_eventos) < max_jogos:
        data = safe_api_get(team_matches_url(RAPIDAPI_HOST, team_id, "previous", page))
        if not data or 'events' not in data:
            break
        eventos = data.get('events', [])
        cutoff_ts = _feature_cutoff_timestamp.get()
        finalizados = []
        for ev in eventos:
            event_ts = safe_event_timestamp(ev.get('startTimestamp'))
            if (ev.get('status', {}).get('type') == 'finished'
                    and event_ts
                    and (not cutoff_ts or event_ts < cutoff_ts)):
                finalizados.append(ev)
        todos_eventos.extend(finalizados)
        if not data.get('hasNextPage', False):
            break
        page += 1
        time.sleep(0.3)   # respeitar rate limit
    return todos_eventos[:max_jogos]

# ----------------------------------------------------------------------
# FUNÇÕES DE COLETA DE DADOS (AGORA USANDO obter_todos_ultimos_jogos)
# ----------------------------------------------------------------------
def analisar_ultimos_jogos_pro(team_id, limit=5, tipo='geral'):
    eventos = obter_todos_ultimos_jogos(team_id, max_jogos=limit*2)
    if not eventos: return "Form N/A"
    try:
        v=e=d=count=0
        for ev in eventos:
            if ev.get('status', {}).get('type') != 'finished': continue
            is_home = str(ev.get('homeTeam', {}).get('id')) == str(team_id)
            if tipo == 'casa' and not is_home: continue
            if tipo == 'fora' and is_home: continue
            score = regulation_score(ev)
            if score is None:
                continue
            hs, ast = score
            if hs == ast: e += 1
            elif (hs > ast and is_home) or (ast > hs and not is_home): v += 1
            else: d += 1
            count += 1
            if count >= limit: break
        return f"{v}V-{e}E-{d}D"
    except: return "Form N/A"

def analisar_ultimos_jogos_por_torneio(team_id, tournament_id, limit=5):
    eventos = obter_todos_ultimos_jogos(team_id, max_jogos=limit*2)
    if not eventos: return None
    try:
        v=e=d=count=0
        for ev in eventos:
            if ev.get('status', {}).get('type') != 'finished': continue
            ev_tournament_id = str(ev.get('tournament', {}).get('id', ''))
            if ev_tournament_id != str(tournament_id): continue
            is_home = str(ev.get('homeTeam', {}).get('id')) == str(team_id)
            score = regulation_score(ev)
            if score is None:
                continue
            hs, ast = score
            if hs == ast: e += 1
            elif (hs > ast and is_home) or (ast > hs and not is_home): v += 1
            else: d += 1
            count += 1
            if count >= limit: break
        if count == 0: return None
        return v, e, d
    except: return None

def obter_media_ppg_adversarios(team_id, tournament_id, season_id, n=5):
    eventos = obter_todos_ultimos_jogos(team_id, max_jogos=n*2)
    ppgs = []
    count = 0
    for ev in eventos:
        if ev.get('status', {}).get('type') != 'finished': continue
        is_home = str(ev.get('homeTeam', {}).get('id')) == str(team_id)
        if is_home: adv_id = str(ev.get('awayTeam', {}).get('id', ''))
        else: adv_id = str(ev.get('homeTeam', {}).get('id', ''))
        if adv_id:
            ev_tournament_id = str(ev.get('tournament', {}).get('id', ''))
            ev_season_id = str(ev.get('season', {}).get('id', ''))
            ppg_adv = obter_ppg_time(adv_id, ev_tournament_id, ev_season_id)
            ppgs.append(ppg_adv)
        count += 1
        if count >= n: break
    if ppgs: return sum(ppgs) / len(ppgs)
    return 0.0

def obter_ppg_time(team_id, tournament_id, season_id):
    if not tournament_id or not season_id: return 0.0
    try:
        data = safe_api_get(tournament_standings_url(RAPIDAPI_HOST, tournament_id, season_id))
        if data and 'standings' in data:
            for row in data['standings'][0].get('rows', []):
                if str(row.get('team', {}).get('id')) == str(team_id):
                    games = int(row.get('games', 0))
                    points = int(row.get('points', 0))
                    if games > 0: return points / games
                    return 0.0
    except: pass
    return 0.0

def obter_posicao_time(team_id, tournament_id, season_id):
    if not tournament_id or not season_id: return 0, 20
    try:
        data = safe_api_get(tournament_standings_url(RAPIDAPI_HOST, tournament_id, season_id))
        if data and 'standings' in data:
            rows = data['standings'][0].get('rows', [])
            total_times = len(rows)
            for row in rows:
                if str(row.get('team', {}).get('id')) == str(team_id):
                    return int(row.get('position', 0)), total_times
    except: pass
    return 0, 20

def obter_dias_descanso(team_id):
    eventos = obter_todos_ultimos_jogos(team_id, max_jogos=1)
    if eventos:
        ev = eventos[0]
        if ev.get('status', {}).get('type') == 'finished':
            last = datetime.fromtimestamp(ev['startTimestamp'], tz=timezone.utc)
            referencia_ts = _feature_cutoff_timestamp.get()
            referencia = (datetime.fromtimestamp(referencia_ts, tz=timezone.utc)
                          if referencia_ts else datetime.now(timezone.utc))
            return max(0, (referencia - last).days)
    return 7

def obter_estatisticas_media_time(team_id, n=5, tipo='geral'):
    eventos = obter_todos_ultimos_jogos(team_id, max_jogos=n*2)
    stats = {
        'gm': [], 'gs': [], 'xg': [], 'posse': [], 'chutes': [],
        'cantos': [], 'faltas': [], 'cartoes': [], 'remates_gol': []
    }
    count = 0
    for ev in eventos:
        if ev.get('status', {}).get('type') != 'finished':
            continue
        is_home = str(ev.get('homeTeam', {}).get('id')) == str(team_id)
        if tipo == 'casa' and not is_home:
            continue
        if tipo == 'fora' and is_home:
            continue
        score = regulation_score(ev)
        if score is None:
            continue
        hs, aws = score
        if is_home:
            stats['gm'].append(float(hs))
            stats['gs'].append(float(aws))
        else:
            stats['gm'].append(float(aws))
            stats['gs'].append(float(hs))
        match_id = ev.get('id')
        if match_id:
            est = obter_estatisticas_com_fallback(match_id, ev)
            if est:
                if is_home:
                    stats['xg'].append(est.get('xg_home', 0))
                    stats['posse'].append(est.get('posse_home', 0))
                    stats['chutes'].append(est.get('chutes_home', 0))
                    stats['cantos'].append(est.get('cantos_home', 0))
                    stats['faltas'].append(est.get('faltas_home', 0))
                    stats['cartoes'].append(est.get('cartoes_home', 0))
                    stats['remates_gol'].append(est.get('remates_gol_home', 0))
                else:
                    stats['xg'].append(est.get('xg_away', 0))
                    stats['posse'].append(est.get('posse_away', 0))
                    stats['chutes'].append(est.get('chutes_away', 0))
                    stats['cantos'].append(est.get('cantos_away', 0))
                    stats['faltas'].append(est.get('faltas_away', 0))
                    stats['cartoes'].append(est.get('cartoes_away', 0))
                    stats['remates_gol'].append(est.get('remates_gol_away', 0))
            else:
                for key in ['xg', 'posse', 'chutes', 'cantos', 'faltas', 'cartoes', 'remates_gol']:
                    stats[key].append(0.0)
        count += 1
        if count >= n:
            break
    result = {}
    for k, v in stats.items():
        result[k] = float(np.mean(v)) if v else 0.0
    return result

# Variável global para blacklist (colocar no topo do arquivo com outras globais)
SOFASCORE_BLACKLIST = set()

def buscar_classificacao_pro_detalhada(tourn_id, season_id, home_id, away_id, tourn_name='', unique_tourn_id=None, home_team='', away_team=''):
    def normalize_name(name):
        if not name:
            return ''
        name = name.lower()
        name = unicodedata.normalize('NFKD', name).encode('ASCII', 'ignore').decode('ascii')
        name = re.sub(r'[^\w\s]', '', name)
        name = re.sub(r'\s+', ' ', name).strip()
        return name

    print(f"[CLASSIF] INICIO: tourn_id={tourn_id}, season_id={season_id}, unique_tourn_id={unique_tourn_id}")

    # Na v2.0, a classificação de futebol usa o ID do tournament. As antigas
    # tentativas unique-tournament e group/tournament não existem para futebol.
    tentativas_allsports = [('normal', tourn_id)] if tourn_id and str(tourn_id) not in ('', '0') else []

    for tp, tid in tentativas_allsports:
        url = tournament_standings_url(RAPIDAPI_HOST, tid, season_id)
        print(f"[CLASSIF] Tentando {tp} (tournament): {url}")
        data = safe_api_get(url, max_retries=1, timeout=8)
        if data and 'standings' in data and data['standings']:
            rows = data['standings'][0].get('rows', [])
            total = len(rows) if rows else 20
            ppg_h = ppg_a = saldo_h = saldo_a = 0.0
            pos_h = pos_a = 0
            hp = ap = "-"
            for row in rows:
                team = row.get('team', {})
                tid_team = str(team.get('id'))
                games = int(row.get('games', 0))
                points = int(row.get('points', 0))
                gf = int(row.get('scoresFor', 0))
                ga = int(row.get('scoresAgainst', 0))
                if tid_team == str(home_id):
                    hp = str(row.get('position', '-'))
                    pos_h = int(row.get('position', 0))
                    ppg_h = points / games if games > 0 else 0.0
                    saldo_h = gf - ga
                if tid_team == str(away_id):
                    ap = str(row.get('position', '-'))
                    pos_a = int(row.get('position', 0))
                    ppg_a = points / games if games > 0 else 0.0
                    saldo_a = gf - ga
            if hp != "-" or ap != "-":
                print(f"[CLASSIF] {tp} OK: casa={hp}º ({ppg_h:.2f} PPG), fora={ap}º ({ppg_a:.2f} PPG)")
                return f"Casa: {hp}º | Fora: {ap}º", ppg_h, ppg_a, saldo_h, saldo_a, pos_h, pos_a, total, total
        else:
            print(f"[CLASSIF] {tp} sem dados ou resposta inválida")

    # 2. Fallback SofaScore (com cache e blacklist)
    effective_unique = unique_tourn_id if unique_tourn_id else tourn_id
    if effective_unique and season_id:
        key = f"{effective_unique}_{season_id}"
        if key in SOFASCORE_BLACKLIST:
            print(f"[CLASSIF] Pular SofaScore para {key} (já falhou antes)")
            print("[CLASSIF] Retornando valores padrão")
            return f"Casa: - | Fora: -", 0.0, 0.0, 0.0, 0.0, 0, 0, 20, 20

        print(f"[CLASSIF] Tentando SofaScore: unique={effective_unique}, season={season_id}")
        standings = get_standings_from_sofascore(effective_unique, season_id)
        if standings:
            home_stats = away_stats = None
            try:
                home_id_int = int(home_id) if home_id else None
                away_id_int = int(away_id) if away_id else None
            except:
                home_id_int = away_id_int = None

            if home_id_int and home_id_int in standings:
                home_stats = standings[home_id_int]
            if away_id_int and away_id_int in standings:
                away_stats = standings[away_id_int]

            # Busca textual se necessário
            if (home_stats is None or away_stats is None) and home_team and away_team:
                print(f"[CLASSIF] IDs não encontrados, tentando busca por nome: {home_team} / {away_team}")
                home_norm = normalize_name(home_team)
                away_norm = normalize_name(away_team)
                for team_id, stats in standings.items():
                    team_name = stats.get('name', '')
                    team_norm = normalize_name(team_name)
                    if team_norm == home_norm:
                        home_stats = stats
                        print(f"[CLASSIF] Encontrado casa: {team_name} (ID {team_id})")
                    if team_norm == away_norm:
                        away_stats = stats
                        print(f"[CLASSIF] Encontrado fora: {team_name} (ID {team_id})")
                    if home_stats and away_stats:
                        break

                # Fuzzy match (opcional)
                if not home_stats:
                    from difflib import get_close_matches
                    all_names = [normalize_name(s.get('name', '')) for s in standings.values()]
                    matches = get_close_matches(home_norm, all_names, n=1, cutoff=0.8)
                    if matches:
                        for team_id, stats in standings.items():
                            if normalize_name(stats.get('name', '')) == matches[0]:
                                home_stats = stats
                                print(f"[CLASSIF] Fuzzy match casa: {stats.get('name')}")
                                break
                if not away_stats:
                    from difflib import get_close_matches
                    all_names = [normalize_name(s.get('name', '')) for s in standings.values()]
                    matches = get_close_matches(away_norm, all_names, n=1, cutoff=0.8)
                    if matches:
                        for team_id, stats in standings.items():
                            if normalize_name(stats.get('name', '')) == matches[0]:
                                away_stats = stats
                                print(f"[CLASSIF] Fuzzy match fora: {stats.get('name')}")
                                break

            if home_stats and away_stats:
                pos_h = home_stats['position']
                pos_a = away_stats['position']
                ppg_h = home_stats['points'] / home_stats['matches'] if home_stats['matches'] > 0 else 0.0
                ppg_a = away_stats['points'] / away_stats['matches'] if away_stats['matches'] > 0 else 0.0
                saldo_h = home_stats['scores_for'] - home_stats['scores_against']
                saldo_a = away_stats['scores_for'] - away_stats['scores_against']
                total = len(standings)
                print(f"[CLASSIF] SofaScore OK: casa={pos_h}º ({ppg_h:.2f} PPG), fora={pos_a}º ({ppg_a:.2f} PPG)")
                return f"Casa: {pos_h}º | Fora: {pos_a}º", ppg_h, ppg_a, saldo_h, saldo_a, pos_h, pos_a, total, total
            else:
                print(f"[CLASSIF] SofaScore: times não encontrados. home={home_stats}, away={away_stats}")
                SOFASCORE_BLACKLIST.add(key)
        else:
            print("[CLASSIF] SofaScore retornou None")
            SOFASCORE_BLACKLIST.add(key)
    else:
        print(f"[CLASSIF] Fallback SofaScore não ativado: effective_unique={effective_unique}, season={season_id}")

    print("[CLASSIF] Retornando valores padrão")
    return f"Casa: - | Fora: -", 0.0, 0.0, 0.0, 0.0, 0, 0, 20, 20

def buscar_arbitro_estilo_detalhado(match_id):
    data = safe_api_get(match_detail_url(RAPIDAPI_HOST, match_id))
    try:
        ref_id = data.get('event', {}).get('referee', {}).get('id')
        if ref_id:
            ref = safe_api_get(f"https://{RAPIDAPI_HOST}/api/referee/{ref_id}/statistics")
            fouls = float(ref.get('statistics', {}).get('fouls', 0))
            yellows = float(ref.get('statistics', {}).get('yellowCards', 0))
            return f"Árbitro: Faltas/J {fouls} | Cartões/J {yellows}", fouls, yellows
    except: pass
    return "Árbitro N/A", 0.0, 0.0

def buscar_h2h(match_id):
    try:
        match_data = safe_api_get(match_detail_url(RAPIDAPI_HOST, match_id))
        if not match_data or 'event' not in match_data:
            return "H2H N/A", 0, 0, 0
        custom_id = match_data['event'].get('customId')
        if not custom_id:
            return "H2H N/A", 0, 0, 0
        data = safe_api_get(f"https://{RAPIDAPI_HOST}/api/match/{custom_id}/h2h")
        if data and 'events' in data:
            v_h = e = d = 0
            cutoff_ts = _feature_cutoff_timestamp.get()
            eventos_h2h = []
            for ev in data['events']:
                event_ts = safe_event_timestamp(ev.get('startTimestamp'))
                if event_ts and (not cutoff_ts or event_ts < cutoff_ts):
                    eventos_h2h.append(ev)
            for ev in eventos_h2h[:5]:
                if ev.get('status', {}).get('type') != 'finished':
                    continue
                score = regulation_score(ev)
                if score is None:
                    continue
                hs, ast = score
                if hs > ast:
                    v_h += 1
                elif hs == ast:
                    e += 1
                else:
                    d += 1
            return f"H2H (últ.5): {v_h}V {e}E {d}D", v_h, e, d
    except Exception as e:
        logger.error(f"Erro ao buscar H2H: {e}")
    return "H2H N/A", 0, 0, 0

def salvar_ids_liga(liga_nome, tournament_id, season_id):
    if not liga_nome or not tournament_id or not season_id: return
    cache_key = (str(liga_nome), str(tournament_id), str(season_id))
    with _saved_league_mappings_lock:
        if cache_key in _saved_league_mappings:
            return
    with get_db_connection() as conn:
        conn.execute('''INSERT OR REPLACE INTO mapeamento_ligas (liga_nome, tournament_id, season_id, ultima_atualizacao)
                        VALUES (?,?,?,?)''', (liga_nome, str(tournament_id), str(season_id), get_brt_time().strftime("%Y-%m-%d %H:%M:%S")))
    with _saved_league_mappings_lock:
        _saved_league_mappings.add(cache_key)

def obter_ids_liga(liga, modo='current'):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT tournament_id, season_id FROM mapeamento_ligas WHERE liga_nome = ? ORDER BY ultima_atualizacao DESC", (liga,))
    row = cursor.fetchone()
    if row: conn.close(); return str(row[0]), str(row[1])
    cursor.execute("SELECT DISTINCT tournament_id, season_id FROM previsoes WHERE liga = ? AND season_id != ''", (liga,))
    rows = cursor.fetchall()
    conn.close()
    if rows:
        rows_sorted = sorted(rows, key=lambda x: int(x[1]) if x[1].isdigit() else 0, reverse=True)
        if modo == 'current': return rows_sorted[0]
        else: return rows_sorted[1] if len(rows_sorted)>1 else rows_sorted[0]
    return None, None

# --- NOVAS FUNÇÕES AUXILIARES (CONTEXTO) ---
def get_nivel_campeonato(liga_name):
    texto = liga_name.lower()
    if any(x in texto for x in ['champions', 'libertadores', 'uefa', 'copa do mundo', 'world cup']): return 5
    if any(x in texto for x in ['premier league', 'la liga', 'serie a', 'bundesliga', 'primeira liga', 'brasileirão',
                                'ligue 1', 'eredivisie', 'super lig', 'süper lig', 'mls']): return 4
    if any(x in texto for x in ['copa', 'cup', 'fa cup', 'dfb pokal', 'coppa italia', 'copa del rey']): return 3
    if any(x in texto for x in ['serie b', 'segunda', 'championship', '2. bundesliga', 'segunda división',
                                'serie c', 'liga 2', '2. liga']): return 2
    return 1

def detectar_fase_mata_mata(liga_name):
    flags = competition_flags(liga_name)
    return int(flags['is_knockout']), int(flags['is_volta'])

def get_zonas_flags(posicao, total_times=20):
    if posicao <= 0 or total_times <= 0: return 0, 0
    reb = 1 if posicao >= total_times - 3 else 0
    clas = 1 if posicao <= 4 else 0
    return reb, clas
def calcular_peso_dinamico(data_jogo, error_margin):
    """
    data_jogo: datetime do jogo
    error_margin: erro absoluto entre a probabilidade prevista e o resultado real (0 a 1)
    Retorna peso final.
    """
    dias_desde = (get_brt_time() - data_jogo).days
    peso_temporal = max(0.5, 1.0 / (1 + 0.03 * dias_desde))  # decai com o tempo
    peso_erro = 1.0 + min(2.0, error_margin * 3)  # erros maiores aumentam peso até 3x
    return peso_temporal * peso_erro

def validar_training_data(limite=20):
    """
    Exibe amostra dos dados de treinamento e estatísticas de qualidade.
    """
    import pandas as pd
    import json
    from collections import Counter

    conn = get_db_connection()
    
    # 1. Estatísticas gerais
    df_stats = pd.read_sql_query("""
        SELECT 
            COUNT(*) as total_jogos,
            COUNT(DISTINCT liga) as total_ligas,
            SUM(CASE WHEN COALESCE(odd_casa, 0) <= 1.0 AND COALESCE(odd_fora, 0) <= 1.0
                     THEN 1 ELSE 0 END) as jogos_sem_odds,
            SUM(CASE WHEN usado_treinamento = 1 THEN 1 ELSE 0 END) as ja_usados,
            MIN(data_jogo) as data_mais_antiga,
            MAX(data_jogo) as data_mais_recente
        FROM training_data
    """, conn)
    
    print("\n" + "="*60)
    print("📊 ESTATÍSTICAS GERAIS DA BASE DE TREINO")
    print("="*60)
    for col in df_stats.columns:
        print(f"{col:25}: {df_stats.iloc[0][col]}")
    
    print("\n🧠 Política: odds ausentes ou zeradas são aceitas e não entram no ML.")

    # 2. Verificar features com vazamento
    df_feat = pd.read_sql_query("SELECT features FROM training_data LIMIT 100", conn)
    features_proibidas = [
        'xg_casa', 'xg_fora', 'posse_casa', 'posse_fora',
        'chutes_casa', 'chutes_fora', 'chutes_gol_casa', 'chutes_gol_fora',
        'escanteios_casa', 'escanteios_fora', 'faltas_casa', 'faltas_fora',
        'xg_diff', 'posse_diff', 'chutes_diff'
    ]
    contaminadas = []
    for _, row in df_feat.iterrows():
        feats = json.loads(row['features'])
        for p in features_proibidas:
            if p in feats and feats[p] != 0:
                contaminadas.append(p)
    if contaminadas:
        print(f"\n🚨 VAZAMENTO DETECTADO: {set(contaminadas)}")
        print("   Execute a evolução validada do modelo para corrigir.")
    else:
        print("\n✅ Nenhuma feature proibida encontrada nas amostras.")
    
    # 4. Distribuição de resultados (Casa/Empate/Fora)
    df_res = pd.read_sql_query("""
        SELECT 
            SUM(CASE WHEN home_score > away_score THEN 1 ELSE 0 END) as vitorias_casa,
            SUM(CASE WHEN home_score = away_score THEN 1 ELSE 0 END) as empates,
            SUM(CASE WHEN home_score < away_score THEN 1 ELSE 0 END) as vitorias_fora
        FROM training_data
    """, conn)
    total = df_res.sum(axis=1).iloc[0]
    print("\n📈 Distribuição dos resultados:")
    print(f"   Casa: {df_res.iloc[0]['vitorias_casa']} ({df_res.iloc[0]['vitorias_casa']/total*100:.1f}%)")
    print(f"   Empate: {df_res.iloc[0]['empates']} ({df_res.iloc[0]['empates']/total*100:.1f}%)")
    print(f"   Fora: {df_res.iloc[0]['vitorias_fora']} ({df_res.iloc[0]['vitorias_fora']/total*100:.1f}%)")
    
    # 5. Amostra dos jogos (últimos 20)
    df_sample = pd.read_sql_query("""
        SELECT match_id, liga, data_jogo, home_team, away_team, home_score, away_score,
               odd_casa, odd_fora, usado_treinamento
        FROM training_data
        ORDER BY data_jogo DESC
        LIMIT ?
    """, conn, params=(limite,))
    print(f"\n📋 ÚLTIMOS {len(df_sample)} JOGOS IMPORTADOS:")
    print(df_sample.to_string(index=False))
    
    # 6. Exemplo de features de um jogo
    print("\n🔍 EXEMPLO DE FEATURES (primeiro jogo da amostra):")
    if len(df_sample) > 0:
        primeiro_id = df_sample.iloc[0]['match_id']
        feat_row = pd.read_sql_query("SELECT features FROM training_data WHERE match_id = ?", conn, params=(primeiro_id,))
        if not feat_row.empty:
            feats = json.loads(feat_row.iloc[0]['features'])
            for k, v in list(feats.items())[:15]:
                print(f"   {k}: {v}")
    
    conn.close()
    
    # 7. Sugestão de ações
    print("\n" + "="*60)
    print("✅ RECOMENDAÇÕES:")
    if df_filtro.iloc[0]['fora_do_filtro'] > 0:
        print("   - Execute 'limpar_training_data_para_filtro_radar()' para remover jogos com odds baixas.")
    print("   - Use a evolução validada para treinar sem substituir um campeão melhor.")
    print("   - Verifique se as datas dos jogos cobrem o período desejado.")
    print("="*60)

# --- EXTRAÇÃO DE FEATURES (SEM ODDS, COM CONTEXTO E ESTILO) ---
@usar_cutoff_temporal
def extrair_features_basicas(match_id, home_id=None, away_id=None, tournament_id=None, season_id=None,
                             odd_casa=2.0, odd_empate=3.0, odd_fora=2.0, unique_tournament_id=None):
    print(f"[DEBUG features] INICIO: unique={unique_tournament_id}, tourn={tournament_id}, season={season_id}")

    f = {}

    # ========== 1. CLASSIFICAÇÃO (STANDINGS) ==========
    home_name = away_name = ''
    info = obter_info_partida(match_id)
    if info:
        home_name = info.get('home_team', '')
        away_name = info.get('away_team', '')

    effective_unique = unique_tournament_id
    if (not effective_unique or str(effective_unique) == '') and tournament_id:
        effective_unique = tournament_id
        print(f"[DEBUG features] unique vazio, usando tournament_id={tournament_id} como fallback")

    if effective_unique and str(effective_unique) not in ('', '0'):
        print(f"[DEBUG features] Chamando classificação com id={effective_unique}")
        resultado = buscar_classificacao_pro_detalhada(
            tournament_id, season_id, home_id, away_id, '', effective_unique,
            home_team=home_name, away_team=away_name)
        str_resumo, ppg_h, ppg_a, saldo_h, saldo_a, pos_h, pos_a, total_h, total_a = resultado
        print(f"[DEBUG features] Retorno: ppg_h={ppg_h}, ppg_a={ppg_a}, pos_h={pos_h}, pos_a={pos_a}")
        f.update({
            'ppg_home': ppg_h, 'ppg_away': ppg_a,
            'saldo_gols_home': saldo_h, 'saldo_gols_away': saldo_a,
            'posicao_home': pos_h, 'posicao_away': pos_a,
            'total_times_liga': total_h
        })
        reb_h, clas_h = get_zonas_flags(pos_h, total_h)
        reb_a, clas_a = get_zonas_flags(pos_a, total_a)
        f['zona_reb_home'] = reb_h
        f['zona_clas_home'] = clas_h
        f['zona_reb_away'] = reb_a
        f['zona_clas_away'] = clas_a
    else:
        f.update({
            'ppg_home': 0.0, 'ppg_away': 0.0,
            'saldo_gols_home': 0, 'saldo_gols_away': 0,
            'posicao_home': 0, 'posicao_away': 0,
            'zona_reb_home': 0, 'zona_clas_home': 0,
            'zona_reb_away': 0, 'zona_clas_away': 0,
            'total_times_liga': 20
        })
        print("[DEBUG features] Sem ID para classificação, usando padrão")

    f['ppg_diff'] = f['ppg_home'] - f['ppg_away']
    f['saldo_diff'] = f['saldo_gols_home'] - f['saldo_gols_away']
    f['posicao_diff'] = f['posicao_home'] - f['posicao_away']

    # ========== 2. ROLLING STATS (com fallback) ==========
    if home_id:
        form_home = analisar_ultimos_jogos_pro(home_id, 5, 'geral')
        v_h, e_h, d_h = extrair_v_e_d(form_home)
        rh = obter_estatisticas_media_time(home_id, 5, 'geral')
        f.update({
            'v_home_5': v_h, 'e_home_5': e_h, 'd_home_5': d_h,
            'gm_home_5': rh['gm'], 'gs_home_5': rh['gs'],
            'xg_home_5': rh['xg'], 'posse_home_5': rh['posse'], 'chutes_home_5': rh['chutes'],
            'dias_descanso_home': obter_dias_descanso(home_id)
        })
        f['dominancia_xg_home'] = rh['xg'] / (rh['xg'] + rh['gs'] + 0.01) if rh['xg'] + rh['gs'] > 0 else 0.0
        f['ef_ofensiva_home'] = rh['gm'] / (rh['xg'] + 0.01) if rh['xg'] > 0 else 0.0
        f['ef_defensiva_home'] = rh['gs'] / (rh['xg'] + 0.01) if rh['xg'] > 0 else 0.0

        rh3 = obter_estatisticas_media_time(home_id, 3, 'geral')
        f['gm_home_3'] = rh3['gm']; f['gs_home_3'] = rh3['gs']; f['xg_home_3'] = rh3['xg']

        rh_casa = obter_estatisticas_media_time(home_id, 5, 'casa')
        f['gm_home_casa_5'] = rh_casa['gm']; f['gs_home_casa_5'] = rh_casa['gs']
        f['xg_home_casa_5'] = rh_casa['xg']; f['posse_home_casa_5'] = rh_casa['posse']; f['chutes_home_casa_5'] = rh_casa['chutes']

        form_home_3 = analisar_ultimos_jogos_pro(home_id, 3, 'casa')
        v_h_casa3, e_h_casa3, d_h_casa3 = extrair_v_e_d(form_home_3)
        f['v_home_casa_3'] = v_h_casa3; f['e_home_casa_3'] = e_h_casa3; f['d_home_casa_3'] = d_h_casa3

        f['cantos_home_5'] = rh.get('cantos', 0.0)
        f['faltas_home_5'] = rh.get('faltas', 0.0)
        f['cartoes_home_5'] = rh.get('cartoes', 0.0)
        f['remates_gol_home_5'] = rh.get('remates_gol', 0.0)
        f['intensidade_home'] = (rh['chutes'] + rh.get('remates_gol', rh['chutes'])) / 2
        f['estilo_posse_home'] = rh['posse'] * rh['chutes']
        f['disciplina_home'] = rh.get('faltas', 0.0) + rh.get('cartoes', 0.0) * 2

        # ELO rating (busca no banco)
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT elo FROM elo_rating WHERE team_id = ?", (home_id,))
        row = cur.fetchone()
        f['elo_home'] = int(row[0]) if row and row[0] is not None else 1500
        conn.close()

        # Desempenho esperado (xG do time - média xG sofrido pelo adversário)
        if away_id and tournament_id:
            media_xg_sofrido_away = obter_media_xg_sofrido(away_id, tournament_id, n=5)
            f['desempenho_xg_home'] = float(f['xg_home_5'] - media_xg_sofrido_away)
        else:
            f['desempenho_xg_home'] = 0.0
    else:
        f.update({
            'v_home_5':0, 'e_home_5':0, 'd_home_5':0,
            'gm_home_5':0, 'gs_home_5':0,
            'xg_home_5':0, 'posse_home_5':0, 'chutes_home_5':0,
            'dias_descanso_home':7,
            'dominancia_xg_home':0, 'ef_ofensiva_home':0, 'ef_defensiva_home':0,
            'gm_home_3':0, 'gs_home_3':0, 'xg_home_3':0,
            'gm_home_casa_5':0, 'gs_home_casa_5':0, 'xg_home_casa_5':0,
            'posse_home_casa_5':0, 'chutes_home_casa_5':0,
            'v_home_casa_3':0, 'e_home_casa_3':0, 'd_home_casa_3':0,
            'cantos_home_5':0, 'faltas_home_5':0, 'cartoes_home_5':0, 'remates_gol_home_5':0,
            'intensidade_home':0, 'estilo_posse_home':0, 'disciplina_home':0,
            'elo_home':1500, 'desempenho_xg_home':0
        })

    # ========== 3. AWAY (simétrico) ==========
    if away_id:
        form_away = analisar_ultimos_jogos_pro(away_id, 5, 'geral')
        v_a, e_a, d_a = extrair_v_e_d(form_away)
        ra = obter_estatisticas_media_time(away_id, 5, 'geral')
        f.update({
            'v_away_5': v_a, 'e_away_5': e_a, 'd_away_5': d_a,
            'gm_away_5': ra['gm'], 'gs_away_5': ra['gs'],
            'xg_away_5': ra['xg'], 'posse_away_5': ra['posse'], 'chutes_away_5': ra['chutes'],
            'dias_descanso_away': obter_dias_descanso(away_id)
        })
        f['dominancia_xg_away'] = ra['xg'] / (ra['xg'] + ra['gs'] + 0.01) if ra['xg'] + ra['gs'] > 0 else 0.0
        f['ef_ofensiva_away'] = ra['gm'] / (ra['xg'] + 0.01) if ra['xg'] > 0 else 0.0
        f['ef_defensiva_away'] = ra['gs'] / (ra['xg'] + 0.01) if ra['xg'] > 0 else 0.0

        ra3 = obter_estatisticas_media_time(away_id, 3, 'geral')
        f['gm_away_3'] = ra3['gm']; f['gs_away_3'] = ra3['gs']; f['xg_away_3'] = ra3['xg']

        ra_fora = obter_estatisticas_media_time(away_id, 5, 'fora')
        f['gm_away_fora_5'] = ra_fora['gm']; f['gs_away_fora_5'] = ra_fora['gs']
        f['xg_away_fora_5'] = ra_fora['xg']; f['posse_away_fora_5'] = ra_fora['posse']; f['chutes_away_fora_5'] = ra_fora['chutes']

        form_away_3f = analisar_ultimos_jogos_pro(away_id, 3, 'fora')
        v_a_fora3, e_a_fora3, d_a_fora3 = extrair_v_e_d(form_away_3f)
        f['v_away_fora_3'] = v_a_fora3; f['e_away_fora_3'] = e_a_fora3; f['d_away_fora_3'] = d_a_fora3

        f['cantos_away_5'] = ra.get('cantos', 0.0)
        f['faltas_away_5'] = ra.get('faltas', 0.0)
        f['cartoes_away_5'] = ra.get('cartoes', 0.0)
        f['remates_gol_away_5'] = ra.get('remates_gol', 0.0)
        f['intensidade_away'] = (ra['chutes'] + ra.get('remates_gol', ra['chutes'])) / 2
        f['estilo_posse_away'] = ra['posse'] * ra['chutes']
        f['disciplina_away'] = ra.get('faltas', 0.0) + ra.get('cartoes', 0.0) * 2

        # ELO away
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT elo FROM elo_rating WHERE team_id = ?", (away_id,))
        row = cur.fetchone()
        f['elo_away'] = int(row[0]) if row and row[0] is not None else 1500
        conn.close()

        # Desempenho esperado away
        if home_id and tournament_id:
            media_xg_sofrido_home = obter_media_xg_sofrido(home_id, tournament_id, n=5)
            f['desempenho_xg_away'] = float(f['xg_away_5'] - media_xg_sofrido_home)
        else:
            f['desempenho_xg_away'] = 0.0
    else:
        f.update({
            'v_away_5':0, 'e_away_5':0, 'd_away_5':0,
            'gm_away_5':0, 'gs_away_5':0,
            'xg_away_5':0, 'posse_away_5':0, 'chutes_away_5':0,
            'dias_descanso_away':7,
            'dominancia_xg_away':0, 'ef_ofensiva_away':0, 'ef_defensiva_away':0,
            'gm_away_3':0, 'gs_away_3':0, 'xg_away_3':0,
            'gm_away_fora_5':0, 'gs_away_fora_5':0, 'xg_away_fora_5':0,
            'posse_away_fora_5':0, 'chutes_away_fora_5':0,
            'v_away_fora_3':0, 'e_away_fora_3':0, 'd_away_fora_3':0,
            'cantos_away_5':0, 'faltas_away_5':0, 'cartoes_away_5':0, 'remates_gol_away_5':0,
            'intensidade_away':0, 'estilo_posse_away':0, 'disciplina_away':0,
            'elo_away':1500, 'desempenho_xg_away':0
        })

    # ========== 4. DIFERENÇAS ==========
    f['descanso_relativo'] = f['dias_descanso_home'] - f['dias_descanso_away']
    for metrica in ['gm', 'gs', 'xg', 'posse', 'chutes', 'cantos', 'faltas', 'cartoes', 'remates_gol',
                    'intensidade', 'estilo_posse', 'disciplina', 'elo', 'desempenho_xg']:
        f[f'diff_{metrica}_5'] = f.get(f'{metrica}_home_5', 0) - f.get(f'{metrica}_away_5', 0)

    f['vantagem_ofensiva_casa'] = f['gm_home_casa_5'] / (f['gs_away_fora_5'] + 0.5) if f['gs_away_fora_5'] + 0.5 > 0 else 0.0
    f['vantagem_ofensiva_fora'] = f['gm_away_fora_5'] / (f['gs_home_casa_5'] + 0.5) if f['gs_home_casa_5'] + 0.5 > 0 else 0.0
    f['dominancia_diff'] = f['dominancia_xg_home'] - f['dominancia_xg_away']

    # ========== 5. ANÁLISE POR TORNEIO ==========
    if home_id and tournament_id:
        res_h = analisar_ultimos_jogos_por_torneio(home_id, tournament_id, 5)
        if res_h:
            v_h_t, e_h_t, d_h_t = res_h
            f['v_home_torneio_5'] = v_h_t
            f['e_home_torneio_5'] = e_h_t
            f['d_home_torneio_5'] = d_h_t
            f['pontos_home_torneio_5'] = (v_h_t * 3 + e_h_t) / 15.0
        else:
            f['v_home_torneio_5'] = 0
            f['e_home_torneio_5'] = 0
            f['d_home_torneio_5'] = 0
            f['pontos_home_torneio_5'] = 0.0
    else:
        f.update({'v_home_torneio_5':0, 'e_home_torneio_5':0, 'd_home_torneio_5':0, 'pontos_home_torneio_5':0.0})

    if away_id and tournament_id:
        res_a = analisar_ultimos_jogos_por_torneio(away_id, tournament_id, 5)
        if res_a:
            v_a_t, e_a_t, d_a_t = res_a
            f['v_away_torneio_5'] = v_a_t
            f['e_away_torneio_5'] = e_a_t
            f['d_away_torneio_5'] = d_a_t
            f['pontos_away_torneio_5'] = (v_a_t * 3 + e_a_t) / 15.0
        else:
            f['v_away_torneio_5'] = 0
            f['e_away_torneio_5'] = 0
            f['d_away_torneio_5'] = 0
            f['pontos_away_torneio_5'] = 0.0
    else:
        f.update({'v_away_torneio_5':0, 'e_away_torneio_5':0, 'd_away_torneio_5':0, 'pontos_away_torneio_5':0.0})

    # ========== 6. FORÇA DOS ADVERSÁRIOS (PPG) ==========
    if home_id and tournament_id and season_id:
        media_ppg_adv_home = obter_media_ppg_adversarios(home_id, tournament_id, season_id, 5)
        f['media_ppg_adv_home'] = media_ppg_adv_home
        f['gap_ppg_home'] = f.get('ppg_away', 0) - media_ppg_adv_home
        f['razao_ppg_home_adv'] = f.get('ppg_home', 0) / media_ppg_adv_home if media_ppg_adv_home > 0 else 1.0
    else:
        f['media_ppg_adv_home'] = 0.0
        f['gap_ppg_home'] = 0.0
        f['razao_ppg_home_adv'] = 1.0

    if away_id and tournament_id and season_id:
        media_ppg_adv_away = obter_media_ppg_adversarios(away_id, tournament_id, season_id, 5)
        f['media_ppg_adv_away'] = media_ppg_adv_away
        f['gap_ppg_away'] = f.get('ppg_home', 0) - media_ppg_adv_away
        f['razao_ppg_away_adv'] = f.get('ppg_away', 0) / media_ppg_adv_away if media_ppg_adv_away > 0 else 1.0
    else:
        f['media_ppg_adv_away'] = 0.0
        f['gap_ppg_away'] = 0.0
        f['razao_ppg_away_adv'] = 1.0

    # ========== 7. NÍVEL DO CAMPEONATO E MATA-MATA ==========
    liga_atual = ''
    if match_id:
        event_data = safe_api_get(match_detail_url(RAPIDAPI_HOST, match_id))
        if event_data and 'event' in event_data:
            t = event_data['event'].get('tournament', {})
            cat = t.get('category', {})
            liga_atual = f"{cat.get('name', '')} - {t.get('name', '')}"
    f['nivel_campeonato'] = get_nivel_campeonato(liga_atual) if liga_atual else 1
    knock, volta = detectar_fase_mata_mata(liga_atual) if liga_atual else (0, 0)
    f['is_knockout'] = knock
    f['is_volta'] = volta
    f.update(competition_flags(liga_atual))

    # ========== 8. H2H ==========
    try:
        _, v_h2h, e_h2h, d_h2h = buscar_h2h(match_id)
        f['v_h2h'] = v_h2h
        f['e_h2h'] = e_h2h
        f['d_h2h'] = d_h2h
    except:
        f['v_h2h'] = 0
        f['e_h2h'] = 0
        f['d_h2h'] = 0

    # ========== 9. NOVAS FEATURES: PRIORIDADE, DESGASTE, ELO ADV, STREAK, DIF GOLS ==========
    # 9.1 Prioridade do torneio
    is_final = 1 if 'final' in liga_atual.lower() else 0
    prioridade = obter_prioridade_torneio(liga_atual, f.get('is_knockout', 0), is_final)
    f['prioridade_torneio'] = prioridade

    # 9.2 Desgaste (jogos nos últimos 7 e 30 dias)
    if home_id:
        f['jogos_home_7d'] = contar_jogos_ultimos_dias(home_id, 7)
        f['jogos_home_30d'] = contar_jogos_ultimos_dias(home_id, 30)
        # Temporariamente desabilitado para evitar lentidão excessiva
        f['proximo_jogo_prioritario_home'] = 999.0
        # f['proximo_jogo_prioritario_home'] = dias_ate_proximo_jogo_prioritario(home_id, prioridade)
    else:
        f.update({'jogos_home_7d': 0, 'jogos_home_30d': 0, 'proximo_jogo_prioritario_home': 999.0})
    if away_id:
        f['jogos_away_7d'] = contar_jogos_ultimos_dias(away_id, 7)
        f['jogos_away_30d'] = contar_jogos_ultimos_dias(away_id, 30)
        f['proximo_jogo_prioritario_away'] = 999.0
        # f['proximo_jogo_prioritario_away'] = dias_ate_proximo_jogo_prioritario(away_id, prioridade)
    else:
        f.update({'jogos_away_7d': 0, 'jogos_away_30d': 0, 'proximo_jogo_prioritario_away': 999.0})

    # 9.3 Força do adversário ajustada por Elo (média dos últimos 5 adversários)
    if home_id and tournament_id and season_id:
        media_elo_adv_home = obter_media_elo_adversarios(home_id, tournament_id, season_id, 5)
        f['media_elo_adv_home'] = media_elo_adv_home
        f['gap_elo_home'] = f.get('elo_home', 1500) - media_elo_adv_home
        f['razao_elo_home_adv'] = f.get('elo_home', 1500) / media_elo_adv_home if media_elo_adv_home > 0 else 1.0
    else:
        f.update({'media_elo_adv_home': 1500, 'gap_elo_home': 0.0, 'razao_elo_home_adv': 1.0})

    if away_id and tournament_id and season_id:
        media_elo_adv_away = obter_media_elo_adversarios(away_id, tournament_id, season_id, 5)
        f['media_elo_adv_away'] = media_elo_adv_away
        f['gap_elo_away'] = f.get('elo_away', 1500) - media_elo_adv_away
        f['razao_elo_away_adv'] = f.get('elo_away', 1500) / media_elo_adv_away if media_elo_adv_away > 0 else 1.0
    else:
        f.update({'media_elo_adv_away': 1500, 'gap_elo_away': 0.0, 'razao_elo_away_adv': 1.0})

    # 9.4 Features de momento (diferença de gols nos últimos 3 jogos e streak)
    if home_id:
        f['diff_gols_3_home'] = f.get('gm_home_3', 0) - f.get('gs_home_3', 0)
        f['streak_home'] = calcular_streak(home_id, tournament_id, 5)
    else:
        f['diff_gols_3_home'] = 0
        f['streak_home'] = 0
    if away_id:
        f['diff_gols_3_away'] = f.get('gm_away_3', 0) - f.get('gs_away_3', 0)
        f['streak_away'] = calcular_streak(away_id, tournament_id, 5)
    else:
        f['diff_gols_3_away'] = 0
        f['streak_away'] = 0

    # Contexto pré-jogo persistido pela segunda fonte; muda imediatamente com
    # W/D/L novo e, sem mudança, expira em 12 horas. Não inclui odds.
    f.update(get_soccer_context_features(
        DB_NAME, match_id, home_name=home_name, away_name=away_name,
        league=liga_atual,
    ))

    # Garantir valores numéricos
    for key, val in f.items():
        if isinstance(val, str):
            try:
                f[key] = float(val) if '.' in val else int(val)
            except:
                f[key] = 0

    f['_feature_version'] = 3
    return f

def detectar_final(liga_name):
    texto = liga_name.lower()
    return 1 if any(p in texto for p in ['final', 'cup final', 'grand final']) else 0

def obter_proximos_jogos(team_id, max_jogos=20):
    """Retorna lista de próximos jogos do time."""
    todos = []
    page = 0
    while len(todos) < max_jogos:
        data = safe_api_get(team_matches_url(RAPIDAPI_HOST, team_id, "next", page))
        if not data or 'events' not in data:
            break
        eventos = data.get('events', [])
        todos.extend(eventos)
        if not data.get('hasNextPage', False):
            break
        page += 1
        time.sleep(0.3)
    return todos[:max_jogos]

def calcular_elo(team_id, tournament_id, season_id, opponent_elo, resultado, k_factor=32):
    """
    Calcula novo Elo para um time.
    resultado: 1 = vitória, 0.5 = empate, 0 = derrota
    Retorna novo_elo.
    """
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT elo FROM elo_rating WHERE team_id = ?", (team_id,))
    row = cur.fetchone()
    elo_atual = row[0] if row else 1500
    conn.close()

    expected = 1 / (1 + 10 ** ((opponent_elo - elo_atual) / 400))
    novo_elo = elo_atual + k_factor * (resultado - expected)
    return int(novo_elo)

def atualizar_elo_apos_partida(home_id, away_id, home_score, away_score):
    """
    Atualiza o Elo rating de ambos os times após uma partida finalizada.
    """
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT elo FROM elo_rating WHERE team_id = ?", (str(home_id),))
    elo_h = cur.fetchone()
    elo_h = elo_h[0] if elo_h else 1500
    cur.execute("SELECT elo FROM elo_rating WHERE team_id = ?", (str(away_id),))
    elo_a = cur.fetchone()
    elo_a = elo_a[0] if elo_a else 1500

    if home_score > away_score:
        res_h, res_a = 1.0, 0.0
    elif home_score < away_score:
        res_h, res_a = 0.0, 1.0
    else:
        res_h, res_a = 0.5, 0.5

    expected_h = 1 / (1 + 10 ** ((elo_a - elo_h) / 400))
    expected_a = 1 / (1 + 10 ** ((elo_h - elo_a) / 400))
    k = 32

    novo_elo_h = elo_h + k * (res_h - expected_h)
    novo_elo_a = elo_a + k * (res_a - expected_a)

    cur.execute("INSERT OR REPLACE INTO elo_rating (team_id, elo, last_update) VALUES (?, ?, ?)",
                (str(home_id), int(novo_elo_h), get_brt_time()))
    cur.execute("INSERT OR REPLACE INTO elo_rating (team_id, elo, last_update) VALUES (?, ?, ?)",
                (str(away_id), int(novo_elo_a), get_brt_time()))
    conn.commit()
    conn.close()
    logger.info(f"Elo atualizado: {home_id} {elo_h} -> {int(novo_elo_h)}, {away_id} {elo_a} -> {int(novo_elo_a)}")

def obter_media_xg_sofrido(team_id, tournament_id, n=5):
    eventos = obter_todos_ultimos_jogos(team_id, max_jogos=n*2)
    xg_sofrido = []
    count = 0
    for ev in eventos:
        if ev.get('status', {}).get('type') != 'finished':
            continue
        ev_tourn_id = str(ev.get('tournament', {}).get('id', ''))
        if tournament_id and ev_tourn_id != tournament_id:
            continue
        is_home = str(ev.get('homeTeam', {}).get('id')) == str(team_id)
        match_id = ev.get('id')
        if match_id:
            est = obter_estatisticas_com_fallback(match_id, ev)
            if est:
                if is_home:
                    xg_sofrido.append(est.get('xg_away', 0))
                else:
                    xg_sofrido.append(est.get('xg_home', 0))
        count += 1
        if count >= n:
            break
    if xg_sofrido:
        return sum(xg_sofrido) / len(xg_sofrido)
    return 1.2

def get_unique_tournament_and_season(event):
    unique_tournament = event.get('tournament', {}).get('uniqueTournament', {})
    unique_id = unique_tournament.get('id')
    if not unique_id:
        unique_id = event.get('tournament', {}).get('id')
    season_id = event.get('season', {}).get('id')
    return unique_id, season_id

def extrair_estatisticas_sofascore(data):
    """Extrai as estatísticas do JSON retornado pelo SofaScore."""
    stats = {}
    if not data or 'statistics' not in data:
        return stats
    for period in data.get('statistics', []):
        if period.get('period') == 'ALL':
            for group in period.get('groups', []):
                for item in group.get('statisticsItems', []):
                    name = item.get('name')
                    def get_val(side):
                        val = item.get(f'{side}Value')
                        if val is not None:
                            return float(val)
                        raw = item.get(side, 0)
                        if isinstance(raw, (int, float)):
                            return float(raw)
                        if isinstance(raw, str):
                            raw = raw.replace('%', '').strip()
                            raw = re.sub(r'[^\d.-]', '', raw)
                            try:
                                return float(raw)
                            except:
                                return 0.0
                        return 0.0
                    if name == 'Ball possession':
                        stats['posse_home'] = get_val('home')
                        stats['posse_away'] = get_val('away')
                    elif name == 'Expected goals':
                        stats['xg_home'] = get_val('home')
                        stats['xg_away'] = get_val('away')
                    elif name == 'Total shots':
                        stats['chutes_home'] = get_val('home')
                        stats['chutes_away'] = get_val('away')
                    elif name == 'Shots on target':
                        stats['remates_gol_home'] = get_val('home')
                        stats['remates_gol_away'] = get_val('away')
                        # alias para compatibilidade
                        stats['chutes_gol_home'] = stats['remates_gol_home']
                        stats['chutes_gol_away'] = stats['remates_gol_away']
                    elif name == 'Corner kicks':
                        stats['cantos_home'] = get_val('home')
                        stats['cantos_away'] = get_val('away')
                    elif name == 'Fouls':
                        stats['faltas_home'] = get_val('home')
                        stats['faltas_away'] = get_val('away')
                    elif name == 'Yellow cards':
                        stats['cartoes_home'] = get_val('home')
                        stats['cartoes_away'] = get_val('away')
    return stats

def buscar_estatisticas_sofascrape(match_id, home_team=None, away_team=None, match_date=None):
    """
    Busca estatísticas no SofaScore usando Playwright.
    Se home_team, away_team e match_date forem fornecidos, tenta busca.
    Caso contrário, tenta diretamente com o match_id.
    """
    cache_key = str(match_id)
    if cache_key in CACHE_ESTATISTICAS_SOFASCORE:
        return CACHE_ESTATISTICAS_SOFASCORE[cache_key]
    
    stats = {}
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            page.set_extra_http_headers({"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"})
            
            # Estratégia 1: tentar endpoint direto com o match_id
            url = f"https://www.sofascore.com/api/v1/event/{match_id}/statistics"
            page.goto(url, timeout=10000)
            content = page.content()
            start = content.find('{')
            end = content.rfind('}') + 1
            if start != -1 and end != 0:
                json_str = content[start:end]
                data = json.loads(json_str)
                stats = extrair_estatisticas_sofascore(data)
            
            # Se falhou e temos dados da partida, tenta busca por nome/data
            if not stats and home_team and away_team and match_date:
                dt = datetime.fromtimestamp(match_date).strftime("%Y-%m-%d")
                query = f"{home_team} {away_team} {dt}".replace(" ", "%20")
                search_url = f"https://www.sofascore.com/api/v1/search?q={query}"
                page.goto(search_url, timeout=10000)
                content = page.content()
                start = content.find('{')
                end = content.rfind('}') + 1
                if start != -1 and end != 0:
                    json_str = content[start:end]
                    data = json.loads(json_str)
                    # Pega o primeiro evento da busca
                    results = data.get('results', [])
                    for res in results:
                        if res.get('type') == 'event':
                            event_id = res.get('id')
                            if event_id:
                                # Busca estatísticas com esse ID
                                stats_url = f"https://www.sofascore.com/api/v1/event/{event_id}/statistics"
                                page.goto(stats_url, timeout=10000)
                                content = page.content()
                                start2 = content.find('{')
                                end2 = content.rfind('}') + 1
                                if start2 != -1 and end2 != 0:
                                    data2 = json.loads(content[start2:end2])
                                    stats = extrair_estatisticas_sofascore(data2)
                                if stats:
                                    break
            
            browser.close()
    except Exception as e:
        print(f"Erro no Sofascrape para {match_id}: {e}")
    
    # Armazena em cache (mesmo que vazio)
    CACHE_ESTATISTICAS_SOFASCORE[cache_key] = stats
    return stats

# ----------------------------------------------------------------------
# IMPORTAÇÃO DE JOGOS PARA TREINAMENTO
# ----------------------------------------------------------------------
def importar_jogos_para_treinamento(dias=7, progress_bar=None, status_text=None, aplicar_filtros=True, aplicar_filtro_odds=False,
                                    job_list=None, start_index=0):
    """
    Importa jogos da API. Se job_list e start_index forem fornecidos, retoma a importação.
    Agora inclui jogos sem odds (odd=0.0) na base de treinamento, mas ainda permite filtro
    opcional para odds baixas (quando as odds existem).
    """
    # ========== 1. Gerar lista de jogos (se não fornecida) ==========
    if job_list is None:
        logger.info(f"Importando jogos finalizados dos últimos {dias} dias...")
        agora = get_brt_time()
        datas = [(agora - timedelta(days=i)).strftime("%d/%m/%Y") for i in range(dias)]
        datas = list(set(datas))
        if not datas:
            print("Nenhuma data para consultar.")
            return 0

        blacklist = ["u17","u19","u20","u21","u22","u23","u24","sub-","sub17","sub19","sub20","sub21","sub23",
                     "amateur","amador","amadores","youth","juniors","aspirantes","reserva","reservas","reserve",
                     "reserves","woman","women","feminino","femenino","femmes","frauen"," w ","ladies","girls","sub","junior"]
        todos_eventos = []
        for d in datas:
            day, month, year = d.split('/')
            data = fetch_football_events_for_date(
                safe_api_get, RAPIDAPI_HOST, f"{day}/{month}/{year}"
            )
            if data and 'events' in data:
                eventos = data['events']
                if aplicar_filtros:
                    finalizados = [ev for ev in eventos 
                                   if ev.get('status', {}).get('type') == 'finished' 
                                   and not any(termo in (ev.get('tournament', {})
                                                          .get('category', {})
                                                          .get('name', '') + ' ' +
                                                          ev.get('tournament', {})
                                                          .get('name', '')).lower() 
                                              for termo in blacklist)]
                else:
                    finalizados = [ev for ev in eventos if ev.get('status', {}).get('type') == 'finished']
                todos_eventos.extend(finalizados)
                print(f"Data {d}: {len(eventos)} eventos totais, {len(finalizados)} finalizados")
            time.sleep(0.3)
        eventos_unicos = {str(ev.get('id')): ev for ev in todos_eventos}.values()
        job_list = list(eventos_unicos)
        start_index = 0

    total = len(job_list)
    if progress_bar:
        progress_bar.progress(0 if start_index == 0 else start_index/total)
    if status_text:
        status_text.text(f"Processando {total} jogos a partir do índice {start_index}...")

    stats = {"sem_odds": 0, "odds_baixas": 0, "ja_existe": 0, "erro": 0, "sem_classificacao": 0, "novos": 0}
    processados = start_index
    lock = threading.Lock()
    # ------------------------------------------------------------------
    # Função para listar features zeradas (apenas log)
    # ------------------------------------------------------------------
    def listar_features_zeradas(features_dict):
        zeros = [k for k, v in features_dict.items() if v == 0 or v is None]
        if zeros:
            return f"⚠️ Features zeradas: {', '.join(zeros[:10])}{'...' if len(zeros)>10 else ''}"
        return "✅ Todas as features não-zero."

    # ------------------------------------------------------------------
    # Função de processamento de cada jogo (thread)
    # ------------------------------------------------------------------
    def processar(ev):
        nonlocal stats
        match_id = str(ev.get('id'))
        try:
            conn_check = get_db_connection()
            cur = conn_check.cursor()
            cur.execute("SELECT 1 FROM training_data WHERE match_id=?", (match_id,))
            existe = cur.fetchone() is not None
            conn_check.close()
            if existe:
                with lock:
                    stats["ja_existe"] += 1
                return {'status':'exists'}
        except:
            pass

        home_team = ev.get('homeTeam', {}).get('name', 'Desconhecido')
        away_team = ev.get('awayTeam', {}).get('name', 'Desconhecido')
        home_id = str(ev.get('homeTeam', {}).get('id', ''))
        away_id = str(ev.get('awayTeam', {}).get('id', ''))
        tournament_id = str(ev.get('tournament', {}).get('id', ''))
        season_id = str(ev.get('season', {}).get('id', ''))
        unique_tournament_id = str(ev.get('tournament', {})
                                   .get('uniqueTournament', {})
                                   .get('id', ''))
        tourn_name = ev.get('tournament', {}).get('name', '')
        liga = f"{ev.get('tournament',{}).get('category',{}).get('name','Mundo')} - {tourn_name}"
        start_ts = safe_event_timestamp(ev.get('startTimestamp'))
        if not start_ts:
            return {'status': 'invalid_timestamp'}
        dt_jogo = datetime.fromtimestamp(start_ts, tz=timezone(timedelta(hours=-3)))
        score = regulation_score(ev)
        if score is None:
            return {'status': 'invalid_score'}
        hs, aws = score

        # Odds não fazem parte da coleta nem do modelo contextual.
        odd_casa = odd_empate = odd_fora = 0.0

        # Extrair features (com fallback)
        features = extrair_features_basicas(match_id, home_id, away_id,
                                            tournament_id, season_id,
                                            odd_casa, odd_empate, odd_fora,
                                            unique_tournament_id)
        print(f"[FEATURES] {match_id} ({home_team} vs {away_team}): {listar_features_zeradas(features)}")

        if features.get('ppg_home', 0) == 0 and features.get('ppg_away', 0) == 0:
            with lock:
                stats["sem_classificacao"] += 1

        try:
            with db_write_lock:
                conn = get_db_connection()
                cur = conn.cursor()
                cur.execute('''INSERT OR IGNORE INTO training_data
                    (match_id, liga, data_jogo, home_team, away_team, home_score, away_score,
                     odd_casa, odd_empate, odd_fora, features, tournament_id, season_id, unique_tournament_id)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)''',
                    (match_id, liga, dt_jogo.strftime("%Y-%m-%d %H:%M:%S"),
                     home_team, away_team, hs, aws,
                     odd_casa, odd_empate, odd_fora, json.dumps(features),
                     tournament_id, season_id, unique_tournament_id))
                cur.execute("INSERT OR IGNORE INTO training_weights (match_id, peso, data_ultima_atualizacao) VALUES (?,1.0,?)",
                            (match_id, dt_jogo.strftime("%Y-%m-%d %H:%M:%S")))
                conn.commit()
                conn.close()
            with lock:
                stats["novos"] += 1
            return {'status':'success'}
        except Exception as e:
            logger.error(f"Erro ao inserir {match_id}: {e}")
            with lock:
                stats["erro"] += 1
            return {'status':'error'}

    # ------------------------------------------------------------------
    # Execução paralela com ThreadPoolExecutor
    # ------------------------------------------------------------------
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(processar, job_list[i]): i for i in range(start_index, total)}
        for future in as_completed(futures):
            idx = futures[future]
            processados = idx + 1
            # Verifica pausa (apenas se estiver rodando no Streamlit)
            if 'st' in globals() and st.session_state.get("import_paused", False):
                st.session_state.import_job_ids = job_list[processados:]
                st.session_state.import_current_index = processados
                st.session_state.import_stats = stats
                st.warning("Importação pausada. Use 'Retomar' para continuar.")
                executor.shutdown(wait=False, cancel_futures=True)
                return 0

            if progress_bar:
                progress_bar.progress(processados / total)
            if status_text:
                status_text.text(f"Processados {processados}/{total} | Novos: {stats['novos']} | "
                                 f"Sem odds: {stats['sem_odds']} | Sem classificação: {stats['sem_classificacao']}")

    # Limpa estado de pausa após conclusão (apenas no Streamlit)
    if 'st' in globals():
        st.session_state.import_paused = False
        st.session_state.import_job_ids = []
        st.session_state.import_current_index = 0

    print("\n" + "="*80)
    print(f"📊 RESUMO DA IMPORTAÇÃO:")
    print(f"   Novos jogos inseridos: {stats['novos']}")
    print(f"   Sem odds (inseridos com odds=0): {stats['sem_odds']}")
    print(f"   Odds baixas (descartados): {stats['odds_baixas']}")
    print(f"   Já existentes (ignorados): {stats['ja_existe']}")
    print(f"   Erros: {stats['erro']}")
    print(f"   Sem classificação (features zeradas): {stats['sem_classificacao']}")
    print("="*80)

    logger.info(f"Importação concluída. Novos: {stats['novos']}. Ignorados: odds baixas={stats['odds_baixas']}, "
                f"já existentes={stats['ja_existe']}, erros={stats['erro']}, sem classificação={stats['sem_classificacao']}")
    return stats['novos']

# ----------------------------------------------------------------------
# FUNÇÕES DE TRANSFERÊNCIA E LIMPEZA
# ----------------------------------------------------------------------
def transferir_jogo_para_treinamento(match_id, evento=None):
    conn = None
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT match_id FROM training_data WHERE match_id = ?", (match_id,))
        if cur.fetchone():
            conn.close()
            return
        cur.execute("""SELECT odd_casa, odd_empate, odd_fora, placar_real, data_jogo,
                              vencedor_previsto, confronto, tournament_id, season_id, unique_tournament_id,
                              liga, timestamp
                       FROM previsoes WHERE match_id = ?""", (match_id,))
        row = cur.fetchone()
        if not row:
            conn.close()
            return
        (odd_casa, odd_empate, odd_fora, placar, data_jogo_str, vencedor, confronto,
         t_id, s_id, u_id, liga_salva, criado_em) = row
        if not placar or '-' not in placar:
            conn.close()
            return
        partes = placar.split('-')
        if len(partes) != 2:
            conn.close()
            return
        try:
            home_score = int(partes[0].strip())
            away_score = int(partes[1].strip())
        except:
            conn.close()
            return
        if ' vs ' in confronto:
            home_team, away_team = confronto.split(' vs ', 1)
        else:
            home_team, away_team = 'Casa', 'Fora'
        tournament_id = t_id if t_id else ''
        season_id = s_id if s_id else ''
        unique_tournament_id = u_id if u_id else ''
        # Reutiliza o evento do lote de auditoria; esta etapa não precisa consumir API.
        ev = evento.get('event', evento) if isinstance(evento, dict) else None
        if ev:
            tournament_id = tournament_id or str(ev.get('tournament', {}).get('id', ''))
            season_id = season_id or str(ev.get('season', {}).get('id', ''))
            unique_tournament_id = unique_tournament_id or str(ev.get('tournament', {}).get('uniqueTournament', {}).get('id', ''))
            liga = f"{ev.get('tournament', {}).get('category', {}).get('name', 'Mundo')} - {ev.get('tournament', {}).get('name', 'Liga')}"
            start_ts = safe_event_timestamp(ev.get('startTimestamp'))
        else:
            liga = liga_salva or 'Desconhecido'
            start_ts = 0
        if start_ts:
            dt_jogo = datetime.fromtimestamp(start_ts, tz=timezone(timedelta(hours=-3)))
        else:
            dt_jogo = get_brt_time()
            try:
                criado_dt = datetime.fromisoformat(str(criado_em))
                partes_data = str(data_jogo_str).split('/')
                if len(partes_data) >= 2:
                    dia, mes = int(partes_data[0]), int(partes_data[1])
                    ano = criado_dt.year + (1 if criado_dt.month == 12 and mes == 1 else 0)
                    dt_jogo = criado_dt.replace(year=ano, month=mes, day=dia)
            except (TypeError, ValueError):
                pass
        knock, volta = detectar_fase_mata_mata(liga)
        features = load_prediction_snapshot_features(DB_NAME, match_id)
        if features:
            features['_feature_version'] = max(6, int(features.get('_feature_version', 6) or 6))
        else:
            features = {'_feature_version': 3,
                        'nivel_campeonato': get_nivel_campeonato(liga),
                        'is_knockout': knock, 'is_volta': volta,
                        'prioridade_torneio': obter_prioridade_torneio(
                            liga, knock, int('final' in liga.lower()))}
        with db_write_lock:
            cur.execute('''INSERT OR IGNORE INTO training_data
                (match_id, liga, data_jogo, home_team, away_team, home_score, away_score,
                 odd_casa, odd_empate, odd_fora, features, tournament_id, season_id, unique_tournament_id)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)''',
                (match_id, liga, dt_jogo.strftime("%Y-%m-%d %H:%M:%S"),
                 home_team.strip(), away_team.strip(), home_score, away_score,
                 odd_casa, odd_empate, odd_fora, json.dumps(features),
                 tournament_id, season_id, unique_tournament_id))
            cur.execute("INSERT OR IGNORE INTO training_weights (match_id, peso, data_ultima_atualizacao) VALUES (?,1.0,?)",
                        (match_id, dt_jogo.strftime("%Y-%m-%d %H:%M:%S")))
            conn.commit()
        logger.info(f"Jogo {match_id} transferido (tournament_id={tournament_id}, season_id={season_id}, unique_tournament_id={unique_tournament_id})")
        conn.close()
    except Exception as e:
        logger.error(f"Erro transferir {match_id}: {e}")
    finally:
        if conn is not None:
            try:
                conn.close()
            except sqlite3.Error:
                pass

def importar_radar_para_treinamento():
    conn = get_db_connection()
    query = """
        SELECT p.match_id FROM previsoes p
        WHERE p.status_resultado IN ('GREEN ✅', 'RED ❌', 'GREEN ✅ (Antecipado)')
        AND p.match_id NOT IN (SELECT match_id FROM training_data)
    """
    cur = conn.execute(query)
    jogos = [row[0] for row in cur.fetchall()]
    conn.close()
    total = len(jogos)
    if total == 0:
        st.info("Nenhum jogo novo para transferir.")
        return 0
    progress_bar = st.progress(0)
    status_text = st.empty()
    transferidos = 0
    for i, mid in enumerate(jogos):
        status_text.text(f"Transferindo {i+1}/{total}: {mid}")
        transferir_jogo_para_treinamento(mid)
        transferidos += 1
        progress_bar.progress((i+1)/total)
    status_text.empty()
    progress_bar.empty()
    return transferidos

def preparar_toda_base_sem_odds():
    try:
        conn = get_db_connection()
        conn.execute("UPDATE training_data SET usado_treinamento = 0")
        conn.execute("DELETE FROM modelos_ml")
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        logger.error(f"Erro ao limpar training_data: {e}")
        return False


# Compatibilidade com atalhos antigos: agora nenhuma partida é apagada por odd.
limpar_training_data_para_filtro_radar = preparar_toda_base_sem_odds

# ----------------------------------------------------------------------
# FUNÇÕES DE TREINAMENTO
# ----------------------------------------------------------------------
def filtrar_features_sem_vazamento(features_dict):
    proibidas = [
        'xg_casa', 'xg_fora', 'posse_casa', 'posse_fora',
        'chutes_casa', 'chutes_fora', 'chutes_gol_casa', 'chutes_gol_fora',
        'escanteios_casa', 'escanteios_fora', 'faltas_casa', 'faltas_fora',
        'xg_diff', 'posse_diff', 'chutes_diff', 'chutes_gol_diff',
        'escanteios_diff', 'faltas_diff',
        'ppg_home', 'ppg_away', 'ppg_diff', 'saldo_gols_home', 'saldo_gols_away',
        'saldo_diff', 'posicao_home', 'posicao_away', 'posicao_diff', 'zona_reb_home',
        'zona_clas_home', 'zona_reb_away', 'zona_clas_away', 'media_ppg_adv_home',
        'media_ppg_adv_away', 'gap_ppg_home', 'gap_ppg_away', 'razao_ppg_home_adv',
        'razao_ppg_away_adv', 'elo_home', 'elo_away', 'diff_elo_5',
        'media_elo_adv_home', 'media_elo_adv_away', 'gap_elo_home', 'gap_elo_away',
        'razao_elo_home_adv', 'razao_elo_away_adv'
    ]
    return {k: v for k, v in features_dict.items() if k not in proibidas}

FEATURES_LEGADAS_SEGURAS = {
    'nivel_campeonato', 'is_knockout', 'is_volta', 'prioridade_torneio',
    'is_cup', 'is_qualifier', 'is_youth_or_reserve', 'is_lower_tier', 'is_women',
    'is_friendly', 'competition_family_league', 'competition_family_cup_group',
    'competition_family_knockout', 'competition_family_qualifier',
    'competition_family_friendly',
    'competition_volatility',
}

def _context_feature_has_temporal_coverage(registros, feature_name, min_samples):
    """Evita que uma fonte parcial funcione como marcador oculto de data."""
    total = len(registros)
    if total < 3:
        return False
    if feature_name.startswith('form_seq_'):
        available_key = ('form_seq_home_available' if feature_name.startswith('form_seq_home_')
                         else 'form_seq_away_available')
    elif feature_name.startswith(('form_sfi_', 'context_sfi_')):
        available_key = 'context_sfi_available'
    elif feature_name == 'sofa_pre_streaks_available':
        available_key = 'sofa_pre_streaks_available'
    elif feature_name.startswith('sofa_pre_'):
        available_key = 'sofa_pre_available'
    elif feature_name.startswith('sofa_roll_'):
        available_key = 'sofa_roll_available'
    elif feature_name.startswith('live_recent_'):
        available_key = 'live_recent_available'
    else:
        return True
    boundaries = (0, total // 3, (2 * total) // 3, total)
    for start, end in zip(boundaries, boundaries[1:]):
        period = registros[start:end]
        required = max(int(min_samples) // 4, int(math.ceil(len(period) * 0.02)))
        if sum(float(item.get(available_key, 0) or 0) > 0 for item in period) < required:
            return False
    return True

def _features_de_odds(odd_casa, odd_empate, odd_fora):
    odds = []
    for odd in (odd_casa, odd_empate, odd_fora):
        try:
            odd = float(odd)
            odds.append(odd if np.isfinite(odd) and odd > 1.01 else 0.0)
        except (TypeError, ValueError):
            odds.append(0.0)
    inversas = [1.0 / odd if odd > 1.01 else 0.0 for odd in odds]
    overround = sum(inversas)
    probs = ([value / overround for value in inversas]
             if overround > 0 else [1 / 3, 1 / 3, 1 / 3])
    return {
        'odd_casa_prejogo': odds[0], 'odd_empate_prejogo': odds[1],
        'odd_fora_prejogo': odds[2], 'prob_mercado_casa': probs[0],
        'prob_mercado_empate': probs[1], 'prob_mercado_fora': probs[2],
        'margem_mercado': max(0.0, overround - 1.0),
        'odds_validas': int(all(odd > 1.01 for odd in odds))
    }

def _rolling_features(jogos, prefix, draw_prior=0.27):
    return enhanced_rolling_features(jogos, prefix, draw_prior)


ODDS_ML_FEATURES = {
    'odd_casa_prejogo', 'odd_empate_prejogo', 'odd_fora_prejogo',
    'prob_mercado_casa', 'prob_mercado_empate', 'prob_mercado_fora',
    'margem_mercado', 'odds_validas',
}


def _sem_features_de_odds(features):
    return {k: v for k, v in features.items()
            if k not in ODDS_ML_FEATURES and not k.lower().startswith(('odd_', 'odds_', 'prob_mercado_'))}


def _enriquecer_duelo_de_estilos(features):
    """Transforma forma recente em força e encaixe de estilos, sem olhar preço de mercado."""
    def valor(nome, padrao=0.0):
        try:
            return float(features.get(nome, padrao) or 0.0)
        except (TypeError, ValueError):
            return float(padrao)
    home_gf = valor('form_home_10_gf', 1.30); home_ga = valor('form_home_10_ga', 1.30)
    away_gf = valor('form_away_10_gf', 1.30); away_ga = valor('form_away_10_ga', 1.30)
    ataque_home = home_gf - away_ga; ataque_away = away_gf - home_ga
    gols_esperados_home = max(0.0, (home_gf + away_ga) / 2.0)
    gols_esperados_away = max(0.0, (away_gf + home_ga) / 2.0)
    intensidade = gols_esperados_home + gols_esperados_away
    # Mantém a semântica das features do campeão; os ajustes novos recebem
    # nomes próprios em ``add_venue_comparison``.
    ppg_gap = valor('form_home_10_ppg', 1.35) - valor('form_away_10_ppg', 1.35)
    mando_gap = valor('form_home_casa_5_ppg', 1.35) - valor('form_away_fora_5_ppg', 1.35)
    prior_liga = max(0.10, min(0.50, valor('liga_prior_empate', 0.27)))
    tendencia_empate = (valor('form_home_10_draw_rate', prior_liga)
                         + valor('form_away_10_draw_rate', prior_liga)) / 2.0
    paridade_ppg = math.exp(-abs(ppg_gap) / 0.75)
    paridade_mando = math.exp(-abs(mando_gap) / 0.85)
    paridade_elo = math.exp(-abs(valor('elo_ml_diff')) / 140.0)
    equilibrio_ataques = math.exp(-abs(gols_esperados_home - gols_esperados_away) / 0.90)
    baixa_intensidade = max(0.0, min(1.0, (3.20 - intensidade) / 1.80))
    empate_composto = max(0.08, min(0.55,
        0.30 * prior_liga + 0.25 * tendencia_empate
        + 0.15 * (0.18 + 0.18 * paridade_ppg)
        + 0.10 * (0.18 + 0.18 * paridade_mando)
        + 0.10 * (0.18 + 0.18 * equilibrio_ataques)
        + 0.10 * (0.18 + 0.18 * baixa_intensidade)
    ))
    features.update({
        'context_forca_ppg_gap': ppg_gap,
        'context_forca_mando_gap': mando_gap,
        'context_ataque_home_vs_defesa_away': ataque_home,
        'context_ataque_away_vs_defesa_home': ataque_away,
        'context_encaixe_ofensivo_gap': ataque_home - ataque_away,
        'context_gols_esperados_home': gols_esperados_home,
        'context_gols_esperados_away': gols_esperados_away,
        'context_intensidade_gols': intensidade,
        'context_tendencia_empate': tendencia_empate,
        'context_paridade_ppg': paridade_ppg,
        'context_paridade_mando': paridade_mando,
        'context_paridade_elo': paridade_elo,
        'context_equilibrio_ataques': equilibrio_ataques,
        'context_baixa_intensidade': baixa_intensidade,
        'context_empate_composto': empate_composto,
        'context_vitorias_gap': valor('form_home_10_win_rate') - valor('form_away_10_win_rate'),
        'context_saldo_recente_gap': valor('form_home_10_saldo') - valor('form_away_10_saldo'),
    })
    add_venue_comparison(features)
    return features


def _enriquecer_duelo_sofa(features):
    return add_measured_sofa_duel(features)


def _timestamp_brt_sem_tz(value):
    ts = pd.Timestamp(value)
    if ts.tzinfo is not None:
        ts = ts.tz_convert('America/Sao_Paulo').tz_localize(None)
    return ts


def _features_calendario(datas_home, datas_away, data_jogo):
    atual = _timestamp_brt_sem_tz(data_jogo)
    def stats(datas):
        anteriores = []
        for data in datas:
            try:
                normalizada = _timestamp_brt_sem_tz(data)
            except (TypeError, ValueError):
                continue
            if normalizada < atual:
                anteriores.append(normalizada)
        jogos_7d = sum(data >= atual - pd.Timedelta(days=7) for data in anteriores)
        jogos_14d = sum(data >= atual - pd.Timedelta(days=14) for data in anteriores)
        return float(jogos_7d), float(jogos_14d)
    n7_h, n14_h = stats(datas_home); n7_a, n14_a = stats(datas_away)
    return {
        'calendario_jogos_7d_home': n7_h, 'calendario_jogos_7d_away': n7_a,
        'calendario_jogos_7d_gap': n7_h - n7_a,
        'calendario_jogos_14d_home': n14_h, 'calendario_jogos_14d_away': n14_a,
        'calendario_jogos_14d_gap': n14_h - n14_a,
    }


def _features_temporada_snapshot(stats_times, home_team, away_team, is_knockout=0):
    """Força e pressão de tabela calculadas apenas com rodadas anteriores."""
    def extrair(team):
        jogos, pontos, gf, ga = stats_times.get(team, [0, 0.0, 0.0, 0.0])
        ppg = (pontos + 1.35 * 4.0) / (jogos + 4.0)
        saldo = (gf - ga) / max(4.0, jogos + 2.0)
        gf_avg = (gf + 1.30 * 4.0) / (jogos + 4.0)
        ga_avg = (ga + 1.30 * 4.0) / (jogos + 4.0)
        return float(jogos), float(ppg), float(saldo), float(gf_avg), float(ga_avg)
    jogos_h, ppg_h, saldo_h, gf_h, ga_h = extrair(home_team)
    jogos_a, ppg_a, saldo_a, gf_a, ga_a = extrair(away_team)
    ranking = []
    for team, (jogos, pontos, gf, ga) in stats_times.items():
        if jogos:
            ranking.append((pontos / jogos + 0.12 * (gf - ga) / jogos, team))
    ranking.sort(reverse=True)
    if len(ranking) >= 4:
        posicoes = {team: 1.0 - idx / (len(ranking) - 1) for idx, (_, team) in enumerate(ranking)}
        rank_h = float(posicoes.get(home_team, 0.5)); rank_a = float(posicoes.get(away_team, 0.5))
    else:
        rank_h = rank_a = 0.5
    progresso = min(1.0, max(jogos_h, jogos_a) / 30.0)
    return {
        'temporada_ppg_home': ppg_h, 'temporada_ppg_away': ppg_a,
        'temporada_ppg_gap': ppg_h - ppg_a,
        'temporada_saldo_home': saldo_h, 'temporada_saldo_away': saldo_a,
        'temporada_saldo_gap': saldo_h - saldo_a,
        'temporada_gf_home': gf_h, 'temporada_ga_home': ga_h,
        'temporada_gf_away': gf_a, 'temporada_ga_away': ga_a,
        'temporada_expected_home': (gf_h + ga_a) / 2.0,
        'temporada_expected_away': (gf_a + ga_h) / 2.0,
        'temporada_expected_total': (gf_h + ga_a + gf_a + ga_h) / 2.0,
        'temporada_attack_defense_gap': (gf_h + ga_a - gf_a - ga_h) / 2.0,
        'temporada_sample_quality': min(1.0, min(jogos_h, jogos_a) / 10.0),
        'temporada_log_jogos_home': float(np.log1p(jogos_h)),
        'temporada_log_jogos_away': float(np.log1p(jogos_a)),
        'temporada_rank_home': rank_h, 'temporada_rank_away': rank_a,
        'temporada_rank_gap': rank_h - rank_a,
        'context_importancia_fase': max(float(bool(is_knockout)), progresso),
        'context_pressao_tabela': progresso * (0.5 + abs(rank_h - rank_a)),
    }

def _features_historicas_db(home_team, away_team, liga, cutoff_ts=None,
                            tournament_id=None, season_id=None, unique_tournament_id=None):
    """Agregados estritamente anteriores ao jogo; não consome a RapidAPI."""
    if not home_team or not away_team:
        return {}
    cutoff = (datetime.fromtimestamp(cutoff_ts, tz=timezone(timedelta(hours=-3))).strftime("%Y-%m-%d %H:%M:%S")
              if cutoff_ts else get_brt_time().strftime("%Y-%m-%d %H:%M:%S"))
    with get_db_connection() as conn:
        # Mantém exatamente a mesma identidade usada pelo robô. Sem isso o
        # dashboard consultava históricos vazios para aliases e, pior, podia
        # misturar time principal, base, reservas e feminino.
        home_original, away_original = home_team, away_team
        home_team, home_identity_confidence = resolve_training_team_name(
            conn, home_original
        )
        away_team, away_identity_confidence = resolve_training_team_name(
            conn, away_original
        )
        home_aliases = training_team_aliases(conn, home_team)
        away_aliases = training_team_aliases(conn, away_team)
        def team_stats(aliases):
            slots = ','.join('?' for _ in aliases)
            row = conn.execute(f"""SELECT COUNT(*), COALESCE(SUM(CASE
                WHEN home_team IN ({slots}) THEN CASE WHEN home_score>away_score THEN 3 WHEN home_score=away_score THEN 1 ELSE 0 END
                ELSE CASE WHEN away_score>home_score THEN 3 WHEN home_score=away_score THEN 1 ELSE 0 END END),0)
                FROM training_data WHERE data_jogo < ?
                  AND (home_team IN ({slots}) OR away_team IN ({slots}))""",
                (*aliases, cutoff, *aliases, *aliases)).fetchone()
            return int(row[0]), float(row[1])
        h_n, h_pts = team_stats(home_aliases)
        a_n, a_pts = team_stats(away_aliases)
        home_slots = ','.join('?' for _ in home_aliases)
        away_slots = ','.join('?' for _ in away_aliases)
        hh = conn.execute(f"""SELECT COUNT(*), COALESCE(SUM(CASE WHEN home_score>away_score THEN 3
            WHEN home_score=away_score THEN 1 ELSE 0 END),0) FROM training_data
            WHERE data_jogo < ? AND home_team IN ({home_slots})""",
            (cutoff, *home_aliases)).fetchone()
        aa = conn.execute(f"""SELECT COUNT(*), COALESCE(SUM(CASE WHEN away_score>home_score THEN 3
            WHEN home_score=away_score THEN 1 ELSE 0 END),0) FROM training_data
            WHERE data_jogo < ? AND away_team IN ({away_slots})""",
            (cutoff, *away_aliases)).fetchone()
        # Provedores usam nomes diferentes para a mesma competição. O ID
        # persistente evita perder todo o histórico e cair no prior neutro.
        if unique_tournament_id:
            league_column, league_value = "unique_tournament_id", str(unique_tournament_id)
        elif tournament_id:
            league_column, league_value = "tournament_id", str(tournament_id)
        else:
            league_column, league_value = "liga", liga
        lg = conn.execute(f"""SELECT COUNT(*), SUM(home_score>away_score),
            SUM(home_score=away_score), SUM(home_score<away_score)
            FROM training_data WHERE data_jogo < ? AND {league_column}=?""",
            (cutoff, league_value)).fetchone()
        def jogos_recentes(aliases):
            slots = ','.join('?' for _ in aliases)
            return conn.execute("""SELECT data_jogo, home_team, away_team, home_score, away_score, match_id
                FROM training_data WHERE data_jogo < ?
                  AND (home_team IN ({slots}) OR away_team IN ({slots}))
                ORDER BY data_jogo DESC LIMIT 50""".format(slots=slots),
                (cutoff, *aliases, *aliases)).fetchall()
        recentes_h = jogos_recentes(home_aliases)
        recentes_a = jogos_recentes(away_aliases)
        opponent_strengths = historical_opponent_strengths(conn)
        temporada_rows = []
        if season_id:
            if unique_tournament_id:
                filtro_id, valor_id = "unique_tournament_id=?", str(unique_tournament_id)
            elif tournament_id:
                filtro_id, valor_id = "tournament_id=?", str(tournament_id)
            else:
                filtro_id, valor_id = "liga=?", liga
            temporada_rows = conn.execute(
                f"""SELECT home_team, away_team, home_score, away_score FROM training_data
                    WHERE data_jogo < ? AND season_id=? AND {filtro_id}""",
                (cutoff, str(season_id), valor_id)).fetchall()
        else:
            if unique_tournament_id:
                season_column, season_value = "unique_tournament_id", str(unique_tournament_id)
            elif tournament_id:
                season_column, season_value = "tournament_id", str(tournament_id)
            else:
                season_column, season_value = "liga", liga
            temporada_rows = conn.execute(
                f"""SELECT home_team, away_team, home_score, away_score
                    FROM training_data
                    WHERE data_jogo < ? AND data_jogo >= datetime(?, '-400 days')
                      AND {season_column}=?""",
                (cutoff, cutoff, season_value)).fetchall()
        rating_names = {home_team, away_team}
        for _, recent_home, recent_away, _, _, _ in recentes_h + recentes_a:
            rating_names.update((recent_home, recent_away))
        rating_names = {name for name in rating_names if name}
        placeholders = ','.join('?' for _ in rating_names)
        ratings = dict(conn.execute(
            f"SELECT team_name, elo FROM ml_team_ratings WHERE team_name IN ({placeholders})",
            tuple(rating_names),
        ).fetchall()) if rating_names else {}
    def ppg(n, points, prior):
        return (float(points) + prior * 8.0) / (int(n) + 8.0)
    lg_n = int(lg[0] or 0)
    def perspectiva(rows, team_aliases, venue=None, limit=10):
        team_aliases = set(team_aliases)
        jogos = []
        for _, home, away, hs, aws, recent_match_id in rows:
            is_home = home in team_aliases
            if venue == 'home' and not is_home:
                continue
            if venue == 'away' and is_home:
                continue
            gf, ga = (hs, aws) if is_home else (aws, hs)
            pontos = 3 if gf > ga else (1 if gf == ga else 0)
            historical_pair = opponent_strengths.get(str(recent_match_id), (1.35, 1.35))
            opponent_ppg = float(historical_pair[0 if is_home else 1])
            jogos.append((pontos, float(gf), float(ga), opponent_ppg))
            if len(jogos) >= limit:
                break
        return jogos
    elo_h = float(ratings.get(home_team, 1500.0)); elo_a = float(ratings.get(away_team, 1500.0))
    elo_prob = 1.0 / (1.0 + 10 ** ((elo_a - (elo_h + 70.0)) / 400.0))
    result = {
        'historical_identity_home_confidence': float(home_identity_confidence),
        'historical_identity_away_confidence': float(away_identity_confidence),
        'historical_identity_both_linked': float(
            home_identity_confidence > 0 and away_identity_confidence > 0
        ),
        'hist_ppg_home_team': ppg(h_n, h_pts, 1.35),
        'hist_ppg_away_team': ppg(a_n, a_pts, 1.35),
        'hist_ppg_home_mandante': ppg(hh[0], hh[1], 1.55),
        'hist_ppg_away_visitante': ppg(aa[0], aa[1], 1.15),
        'hist_log_jogos_home': float(np.log1p(h_n)),
        'hist_log_jogos_away': float(np.log1p(a_n)),
        'liga_prior_casa': (float(lg[1] or 0) + 2.0) / (lg_n + 6.0),
        'liga_prior_empate': (float(lg[2] or 0) + 2.0) / (lg_n + 6.0),
        'liga_prior_fora': (float(lg[3] or 0) + 2.0) / (lg_n + 6.0),
        'elo_ml_home': elo_h, 'elo_ml_away': elo_a,
        'elo_ml_diff': elo_h - elo_a, 'elo_ml_prob_home': elo_prob,
    }
    draw_prior = result['liga_prior_empate']
    result.update(_rolling_features(perspectiva(recentes_h, home_aliases, limit=10), 'form_home_10', draw_prior))
    result.update(_rolling_features(perspectiva(recentes_a, away_aliases, limit=10), 'form_away_10', draw_prior))
    result.update(_rolling_features(perspectiva(recentes_h, home_aliases, venue='home', limit=5), 'form_home_casa_5', draw_prior))
    result.update(_rolling_features(perspectiva(recentes_a, away_aliases, venue='away', limit=5), 'form_away_fora_5', draw_prior))
    result.update(_rolling_features(perspectiva(recentes_h, home_aliases, venue='home', limit=10), 'form_home_casa_10', draw_prior))
    result.update(_rolling_features(perspectiva(recentes_a, away_aliases, venue='away', limit=10), 'form_away_fora_10', draw_prior))
    result.update(_features_calendario([r[0] for r in recentes_h], [r[0] for r in recentes_a], cutoff))
    _enriquecer_duelo_de_estilos(result)
    temporada_stats = {}
    for h_team, a_team, h_score, a_score in temporada_rows:
        h_team = (home_team if h_team in home_aliases else
                  away_team if h_team in away_aliases else h_team)
        a_team = (home_team if a_team in home_aliases else
                  away_team if a_team in away_aliases else a_team)
        if h_score > a_score: pontos_h, pontos_a = 3.0, 0.0
        elif h_score == a_score: pontos_h = pontos_a = 1.0
        else: pontos_h, pontos_a = 0.0, 3.0
        for team, pontos, gf, ga in ((h_team, pontos_h, h_score, a_score),
                                     (a_team, pontos_a, a_score, h_score)):
            stats = temporada_stats.setdefault(team, [0, 0.0, 0.0, 0.0])
            stats[0] += 1; stats[1] += pontos; stats[2] += float(gf); stats[3] += float(ga)
    knock, _ = detectar_fase_mata_mata(liga)
    result.update(_features_temporada_snapshot(temporada_stats, home_team, away_team, knock))
    return result

def preparar_dados_treinamento(liga=None):
    """
    Prepara X, y, feature_order, match_ids, pesos (dinâmicos) para treinamento.
    """
    conn = get_db_connection()
    if liga:
        query = """SELECT t.features, t.home_score, t.away_score, t.data_jogo, t.match_id,
                          COALESCE(w.peso, 1.0) as peso_base,
                          COALESCE(w.error_margin, 0.0) as error_margin,
                           t.odd_casa, t.odd_empate, t.odd_fora, t.liga, t.home_team, t.away_team,
                           t.tournament_id, t.season_id, t.unique_tournament_id,
                           sc.features_json AS sofa_archive_features,
                           sc.home_name AS sofa_archive_home,
                           sc.away_name AS sofa_archive_away,
                           sc.start_timestamp AS sofa_archive_start
                   FROM training_data t
                   LEFT JOIN training_weights w ON t.match_id = w.match_id
                   LEFT JOIN sofascore_pregame_context sc ON sc.match_id=t.match_id
                   WHERE t.liga=?
                   ORDER BY t.data_jogo ASC, t.match_id ASC"""
        params = (liga,)
    else:
        query = """SELECT t.features, t.home_score, t.away_score, t.data_jogo, t.match_id,
                          COALESCE(w.peso, 1.0) as peso_base,
                          COALESCE(w.error_margin, 0.0) as error_margin,
                           t.odd_casa, t.odd_empate, t.odd_fora, t.liga, t.home_team, t.away_team,
                           t.tournament_id, t.season_id, t.unique_tournament_id,
                           sc.features_json AS sofa_archive_features,
                           sc.home_name AS sofa_archive_home,
                           sc.away_name AS sofa_archive_away,
                           sc.start_timestamp AS sofa_archive_start
                   FROM training_data t
                   LEFT JOIN training_weights w ON t.match_id = w.match_id
                   LEFT JOIN sofascore_pregame_context sc ON sc.match_id=t.match_id
                   ORDER BY t.data_jogo ASC, t.match_id ASC"""
        params = ()
    df = pd.read_sql_query(query, conn, params=params)
    # Treino e inferência precisam enxergar a mesma equipe. A resolução é
    # somente textual/frequencial e não consulta placares posteriores.
    identity_names = set(df['home_team'].dropna().astype(str))
    identity_names.update(df['away_team'].dropna().astype(str))
    identity_map = {
        name: resolve_training_team_name(conn, name)[0]
        for name in identity_names if name
    }
    df['home_team'] = df['home_team'].map(
        lambda value: identity_map.get(str(value), value)
    )
    df['away_team'] = df['away_team'].map(
        lambda value: identity_map.get(str(value), value)
    )
    invalid_identity = df.apply(
        lambda row: (
            not normalize_team_name(row['home_team'])
            or not normalize_team_name(row['away_team'])
            or normalize_team_name(row['home_team']) == normalize_team_name(row['away_team'])
        ),
        axis=1,
    )
    if bool(invalid_identity.any()):
        logger.warning(
            "Treino: %d partida(s) ignorada(s) por identidade casa/fora inválida.",
            int(invalid_identity.sum()),
        )
        df = df.loc[~invalid_identity].copy()
    conn.close()
    if df.empty:
        return None, None, None, None, None

    y, match_ids, pesos, registros, radar_flags = [], [], [], [], []
    feature_keys = set()
    feature_support = collections.Counter()
    versoes_verificadas = 0
    historico_times, historico_casa, historico_fora, historico_ligas = {}, {}, {}, {}
    elo_ratings = {}
    rolling_times = collections.defaultdict(lambda: collections.deque(maxlen=10))
    rolling_casa = collections.defaultdict(lambda: collections.deque(maxlen=10))
    rolling_fora = collections.defaultdict(lambda: collections.deque(maxlen=10))
    datas_times = collections.defaultdict(lambda: collections.deque(maxlen=50))
    temporadas = collections.defaultdict(dict)
    now = get_brt_time()  # tz-aware

    def competition_history_key(row):
        for prefix, column in (("u:", "unique_tournament_id"),
                               ("t:", "tournament_id")):
            value = str(row.get(column) or "").strip()
            if value and value.lower() != "nan":
                return prefix + value
        return "n:" + str(row.get("liga") or "").strip()

    proibidas = [
        'xg_casa','xg_fora','posse_casa','posse_fora',
        'chutes_casa','chutes_fora','chutes_gol_casa','chutes_gol_fora',
        'escanteios_casa','escanteios_fora','faltas_casa','faltas_fora',
        'xg_diff','posse_diff','chutes_diff','chutes_gol_diff',
        'escanteios_diff','faltas_diff',
        'ppg_home','ppg_away','ppg_diff','saldo_gols_home','saldo_gols_away','saldo_diff',
        'posicao_home','posicao_away','posicao_diff','zona_reb_home','zona_clas_home',
        'zona_reb_away','zona_clas_away','media_ppg_adv_home','media_ppg_adv_away',
        'gap_ppg_home','gap_ppg_away','razao_ppg_home_adv','razao_ppg_away_adv',
        'elo_home','elo_away','diff_elo_5','media_elo_adv_home','media_elo_adv_away',
        'gap_elo_home','gap_elo_away','razao_elo_home_adv','razao_elo_away_adv'
    ]

    for _, row in df.iterrows():
        try:
            feats = json.loads(row['features']) if row['features'] else {}
        except (TypeError, ValueError, json.JSONDecodeError):
            feats = {}
        try:
            sofa_archive = json.loads(row.get('sofa_archive_features') or '{}')
        except (TypeError, ValueError, json.JSONDecodeError):
            sofa_archive = {}
        from pregame_archive_integrity import archive_matches_training_fixture
        if not archive_matches_training_fixture(
            row['home_team'], row['away_team'], row['data_jogo'],
            row.get('sofa_archive_home'), row.get('sofa_archive_away'),
            row.get('sofa_archive_start'),
        ):
            sofa_archive = {}
        try:
            feature_version = int(feats.pop('_feature_version', 0) or 0)
        except (TypeError, ValueError):
            feature_version = 0
        for p in proibidas:
            feats.pop(p, None)
        if feature_version < 2:
            # Bases antigas foram calculadas após o resultado e contêm rolling stats vazadas.
            feats = {k: v for k, v in feats.items() if k in FEATURES_LEGADAS_SEGURAS}
        else:
            versoes_verificadas += 1
        feats = _sem_features_de_odds(feats)
        for feature_name in list(feats):
            if feature_name.startswith('calendario_descanso_'):
                feats.pop(feature_name, None)
        # O snapshot histórico do SofaScore é uma fonte pré-jogo independente.
        # Ele entra depois da higienização da base legada para não ser descartado
        # junto com features antigas que poderiam conter vazamento temporal.
        if isinstance(sofa_archive, dict):
            feats.update({
                key: value for key, value in sofa_archive.items()
                if isinstance(value, (int, float, np.number))
                and (key.startswith('sofa_pre_') or key.startswith('sofa_roll_'))
            })
        _enriquecer_duelo_sofa(feats)
        knock, volta = detectar_fase_mata_mata(str(row['liga'] or ''))
        is_final = int('final' in str(row['liga'] or '').lower())
        feats.update({
            'nivel_campeonato': get_nivel_campeonato(str(row['liga'] or '')),
            'is_knockout': knock, 'is_volta': volta,
            'prioridade_torneio': obter_prioridade_torneio(str(row['liga'] or ''), knock, is_final),
        })
        feats.update(competition_flags(str(row['liga'] or '')))
        th = historico_times.get(row['home_team'], [0, 0.0])
        ta = historico_times.get(row['away_team'], [0, 0.0])
        hh = historico_casa.get(row['home_team'], [0, 0.0])
        aa = historico_fora.get(row['away_team'], [0, 0.0])
        league_history_key = competition_history_key(row)
        lg = historico_ligas.get(league_history_key, [0, 0, 0, 0])
        def ppg_hist(stats, prior):
            return (stats[1] + prior * 8.0) / (stats[0] + 8.0)
        opponent_h_pre = ppg_hist(ta, 1.35)
        opponent_a_pre = ppg_hist(th, 1.35)
        feats.update({
            'hist_ppg_home_team': ppg_hist(th, 1.35),
            'hist_ppg_away_team': ppg_hist(ta, 1.35),
            'hist_ppg_home_mandante': ppg_hist(hh, 1.55),
            'hist_ppg_away_visitante': ppg_hist(aa, 1.15),
            'hist_log_jogos_home': float(np.log1p(th[0])),
            'hist_log_jogos_away': float(np.log1p(ta[0])),
            'liga_prior_casa': (lg[1] + 2.0) / (lg[0] + 6.0),
            'liga_prior_empate': (lg[2] + 2.0) / (lg[0] + 6.0),
            'liga_prior_fora': (lg[3] + 2.0) / (lg[0] + 6.0),
        })
        elo_h = float(elo_ratings.get(row['home_team'], 1500.0))
        elo_a = float(elo_ratings.get(row['away_team'], 1500.0))
        elo_prob = 1.0 / (1.0 + 10 ** ((elo_a - (elo_h + 70.0)) / 400.0))
        feats.update({'elo_ml_home': elo_h, 'elo_ml_away': elo_a,
                      'elo_ml_diff': elo_h - elo_a, 'elo_ml_prob_home': elo_prob})
        draw_prior = feats['liga_prior_empate']
        recent_h = list(reversed(rolling_times[row['home_team']]))
        recent_a = list(reversed(rolling_times[row['away_team']]))
        venue_h = list(reversed(rolling_casa[row['home_team']]))
        venue_a = list(reversed(rolling_fora[row['away_team']]))
        feats.update(_rolling_features(recent_h, 'form_home_10', draw_prior))
        feats.update(_rolling_features(recent_a, 'form_away_10', draw_prior))
        feats.update(_rolling_features(venue_h[:5], 'form_home_casa_5', draw_prior))
        feats.update(_rolling_features(venue_a[:5], 'form_away_fora_5', draw_prior))
        feats.update(_rolling_features(venue_h, 'form_home_casa_10', draw_prior))
        feats.update(_rolling_features(venue_a, 'form_away_fora_10', draw_prior))
        data_contexto = _timestamp_brt_sem_tz(row['data_jogo'])
        feats.update(_features_calendario(datas_times[row['home_team']], datas_times[row['away_team']], data_contexto))
        _enriquecer_duelo_de_estilos(feats)
        chave_temporada = (str(row.get('unique_tournament_id') or row.get('tournament_id') or row['liga']),
                           str(row.get('season_id') or data_contexto.year))
        temporada_atual = temporadas[chave_temporada]
        feats.update(_features_temporada_snapshot(temporada_atual, row['home_team'], row['away_team'], knock))
        registros.append(feats)
        feature_keys.update(feats.keys())
        for feature_name in feats:
            supported = True
            if feature_name.startswith('form_sfi_') or feature_name.startswith('context_sfi_'):
                supported = float(feats.get('context_sfi_available', 0) or 0) > 0
            elif feature_name.startswith('sofa_pre_'):
                supported = float(feats.get('sofa_pre_available', 0) or 0) > 0
            elif feature_name.startswith('sofa_roll_home_'):
                supported = float(feats.get('sofa_roll_home_games', 0) or 0) > 0
            elif feature_name.startswith('sofa_roll_away_'):
                supported = float(feats.get('sofa_roll_away_games', 0) or 0) > 0
            elif feature_name == 'sofa_roll_available':
                supported = float(feats.get('sofa_roll_available', 0) or 0) > 0
            elif feature_name.startswith('sofa_roll_'):
                supported = float(feats.get('sofa_roll_available', 0) or 0) > 0
            elif feature_name.startswith('allsports_goal_home_'):
                supported = float(feats.get('allsports_goal_home_available', 0) or 0) > 0
            elif feature_name.startswith('allsports_goal_away_'):
                supported = float(feats.get('allsports_goal_away_available', 0) or 0) > 0
            elif feature_name.startswith('allsports_goal_'):
                supported = (
                    float(feats.get('allsports_goal_home_available', 0) or 0) > 0
                    and float(feats.get('allsports_goal_away_available', 0) or 0) > 0
                )
            if supported:
                feature_support[feature_name] += 1

        if row['home_score'] > row['away_score']:
            resultado, pontos_h, pontos_a = 0, 3.0, 0.0
        elif row['home_score'] == row['away_score']:
            resultado, pontos_h, pontos_a = 1, 1.0, 1.0
        else:
            resultado, pontos_h, pontos_a = 2, 0.0, 3.0
        y.append(resultado)
        for store, key, pontos in ((historico_times, row['home_team'], pontos_h),
                                   (historico_times, row['away_team'], pontos_a),
                                   (historico_casa, row['home_team'], pontos_h),
                                   (historico_fora, row['away_team'], pontos_a)):
            stats = store.setdefault(key, [0, 0.0]); stats[0] += 1; stats[1] += pontos
        liga_stats = historico_ligas.setdefault(league_history_key, [0, 0, 0, 0])
        liga_stats[0] += 1; liga_stats[resultado + 1] += 1
        hs, aws = max(0.0, float(row['home_score'])), max(0.0, float(row['away_score']))
        rolling_times[row['home_team']].append((pontos_h, hs, aws, opponent_h_pre))
        rolling_times[row['away_team']].append((pontos_a, aws, hs, opponent_a_pre))
        rolling_casa[row['home_team']].append((pontos_h, hs, aws, opponent_h_pre))
        rolling_fora[row['away_team']].append((pontos_a, aws, hs, opponent_a_pre))
        datas_times[row['home_team']].append(data_contexto)
        datas_times[row['away_team']].append(data_contexto)
        for team, pontos, gf, ga in ((row['home_team'], pontos_h, hs, aws),
                                     (row['away_team'], pontos_a, aws, hs)):
            stats_temp = temporada_atual.setdefault(team, [0, 0.0, 0.0, 0.0])
            stats_temp[0] += 1; stats_temp[1] += pontos; stats_temp[2] += gf; stats_temp[3] += ga
        score_h = 1.0 if resultado == 0 else (0.5 if resultado == 1 else 0.0)
        margem = max(1.0, float(np.log1p(abs(hs - aws))))
        elo_ratings[row['home_team']] = elo_h + 24.0 * margem * (score_h - elo_prob)
        elo_ratings[row['away_team']] = elo_a + 24.0 * margem * ((1.0 - score_h) - (1.0 - elo_prob))

        match_ids.append(row['match_id'])
        try:
            radar_flags.append(
                float(row['odd_casa'] or 0) > 1.99
                and float(row['odd_fora'] or 0) > 1.99
            )
        except (TypeError, ValueError):
            radar_flags.append(False)

        # Converte data_jogo para tz-aware (usando o mesmo fuso de now)
        data_jogo = pd.to_datetime(row['data_jogo'])
        if data_jogo.tzinfo is None:
            # data_jogo é persistida em horário BRT sem offset.
            data_jogo = data_jogo.tz_localize(now.tzinfo)
        dias_desde = (now - data_jogo).days
        peso_temporal = max(0.5, 1.0 / (1 + 0.03 * dias_desde))
        peso_final = float(row['peso_base']) * peso_temporal
        pesos.append(peso_final)

    # Só habilita rolling stats quando houver volume suficiente de snapshots sem vazamento.
    usar_features_estendidas = versoes_verificadas >= max(200, int(len(registros) * 0.10))
    if not usar_features_estendidas:
        feature_keys = {k for k in feature_keys
                        if k in FEATURES_LEGADAS_SEGURAS or k.startswith('hist_')
                        or k.startswith('liga_prior_') or k.startswith('elo_ml_')
                        or k.startswith(('form_', 'context_', 'calendario_', 'temporada_',
                                         'sofa_pre_', 'sofa_roll_'))}
    feature_keys = {k for k in feature_keys if k not in ODDS_ML_FEATURES
                    and not k.lower().startswith(('odd_', 'odds_', 'prob_mercado_'))}
    contextual_prefixes = (
        'form_seq_', 'form_sfi_', 'context_sfi_', 'sofa_pre_', 'sofa_roll_', 'live_recent_',
        'allsports_goal_'
    )
    feature_keys = {
        k for k in feature_keys
        if not k.startswith(contextual_prefixes)
        or (
            feature_support[k] >= ML_CONTEXT_FEATURE_MIN_SAMPLES
            and _context_feature_has_temporal_coverage(
                registros, k, ML_CONTEXT_FEATURE_MIN_SAMPLES
            )
        )
    }
    feature_order = sorted(feature_keys)
    X = np.array([
        [float(feats.get(k, 0.0) or 0.0) if isinstance(feats.get(k, 0.0), (int, float, np.number)) else 0.0
         for k in feature_order]
        for feats in registros
    ], dtype=np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    y = np.array(y)
    pesos = np.array(pesos, dtype=np.float32)
    # Balanceamento de rótulos é aplicado somente dentro da janela de fit.
    # Calcular aqui vazava a distribuição do holdout cronológico.
    global _ultimos_elos_treino, _ultimo_radar_mask_treino, _ultimos_registros_treino
    _ultimos_elos_treino = dict(elo_ratings)
    _ultimo_radar_mask_treino = np.asarray(radar_flags, dtype=bool)
    _ultimos_registros_treino = [dict(features) for features in registros]
    return X, y, feature_order, match_ids, pesos

def otimizar_xgboost(X_train, y_train, sample_weight=None, n_iter=20):
    param_grid = {
        'n_estimators': [200, 300, 500],
        'max_depth': [2, 3, 4, 5],
        'learning_rate': [0.01, 0.03, 0.05, 0.1],
        'subsample': [0.7, 0.8, 0.9],
        'colsample_bytree': [0.7, 0.8, 0.9],
        'gamma': [0, 0.1, 0.2],
        'reg_alpha': [0, 0.1],
        'reg_lambda': [1, 1.5]
    }
    xgb_base = xgb.XGBClassifier(objective='multi:softprob', num_class=3, eval_metric='mlogloss', random_state=42, n_jobs=-1)
    tscv = TimeSeriesSplit(n_splits=3)
    search = RandomizedSearchCV(xgb_base, param_grid, n_iter=n_iter, cv=tscv, scoring='neg_log_loss', n_jobs=-1, random_state=42, verbose=0)
    search.fit(X_train, y_train, sample_weight=sample_weight)
    logger.info(f"Melhores hiperparâmetros: {search.best_params_}")
    return search.best_estimator_

def treinar_modelo_liga_sem_vazamento(liga, min_amostras=100, acuracia_minima=0.35, usar_otimizacao=False):
    """
    Treina modelo XGBoost para uma liga específica, com validação cronológica,
    calibração, pesos de erro e cálculo de ROC-AUC multiclasse.
    Retorna (modelo, scaler, feature_order, acuracia, log_loss, roc_auc)
    """
    X, y, feature_order, match_ids, pesos = preparar_dados_treinamento(liga)
    if X is None or len(X) < min_amostras:
        logger.warning(f"Liga '{liga}': amostras insuficientes.")
        return None, None, None, 0.0, 0.0, 0.0

    if set(y) != {0, 1, 2}:
        logger.warning(f"Liga '{liga}': faltam classes reais; usando fallback global.")
        return None, None, None, 0.0, 0.0, 0.0

    xgb_params_no_early = {
        'objective': 'multi:softprob',
        'num_class': 3,
        'eval_metric': 'mlogloss',
        'random_state': 42,
        'n_jobs': -1,
        'max_depth': 3,
        'learning_rate': 0.03,
        'n_estimators': 350,
        'subsample': 0.85,
        'colsample_bytree': 0.75,
        'gamma': 0.05,
        'reg_alpha': 0.2,
        'reg_lambda': 4.0,
        'min_child_weight': 10,
    }

    tscv = TimeSeriesSplit(n_splits=min(5, max(2, len(X) // 30)))
    acc_scores, loss_scores, roc_auc_scores = [], [], []

    for train_idx, test_idx in tscv.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        w_train = pesos[train_idx]

        if set(y_train) != {0, 1, 2}:
            continue

        sample_weights = balanced_sample_weights(y_train, w_train)

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        model = xgb.XGBClassifier(**xgb_params_no_early)
        calibrated = model
        if len(X_train_s) >= 120 and min(np.bincount(y_train, minlength=3)) >= 5:
            try:
                calibrated = CalibratedClassifierCV(
                    model, method='sigmoid', cv=TimeSeriesSplit(n_splits=3))
                calibrated.fit(X_train_s, y_train, sample_weight=sample_weights)
            except (ValueError, xgb.core.XGBoostError):
                calibrated = model
                calibrated.fit(X_train_s, y_train, sample_weight=sample_weights)
        else:
            calibrated.fit(X_train_s, y_train, sample_weight=sample_weights)

        y_pred = calibrated.predict(X_test_s)
        y_proba = calibrated.predict_proba(X_test_s)
        acc = accuracy_score(y_test, y_pred)
        try:
            loss = log_loss(y_test, y_proba, labels=[0, 1, 2])
        except:
            loss = 0.0
        try:
            roc_auc = roc_auc_score(y_test, y_proba, multi_class='ovr', average='weighted')
        except:
            roc_auc = 0.0

        acc_scores.append(acc)
        loss_scores.append(loss)
        roc_auc_scores.append(roc_auc)

    if not acc_scores:
        return None, None, None, 0.0, 0.0, 0.0
    avg_acc = np.mean(acc_scores)
    avg_loss = np.mean(loss_scores)
    avg_roc_auc = np.mean(roc_auc_scores)

    if avg_acc < acuracia_minima:
        logger.warning(f"Modelo '{liga}' com acurácia baixa ({avg_acc:.2%}), mas será guardado na mesma.")

    # Treinar modelo final com todos os dados
    scaler_final = StandardScaler()
    X_full = scaler_final.fit_transform(X)
    final_weights = balanced_sample_weights(y, pesos)

    final_model_no_early = xgb.XGBClassifier(**xgb_params_no_early)
    final_model_no_early.fit(X_full, y, sample_weight=final_weights)

    calibrated_final = final_model_no_early
    if len(X_full) >= 120 and min(np.bincount(y, minlength=3)) >= 5:
        try:
            calibrated_final = CalibratedClassifierCV(
                final_model_no_early, method='sigmoid', cv=TimeSeriesSplit(n_splits=3))
            calibrated_final.fit(X_full, y, sample_weight=final_weights)
        except (ValueError, xgb.core.XGBoostError):
            calibrated_final = final_model_no_early
            calibrated_final.fit(X_full, y, sample_weight=final_weights)

    # Salvar modelo no banco
    buf = BytesIO()
    joblib.dump(calibrated_final, buf, compress=True)
    modelo_bytes = buf.getvalue()
    scaler_params = {'mean': scaler_final.mean_.tolist(), 'scale': scaler_final.scale_.tolist()}

    with db_write_lock:
        with get_db_connection() as conn:
            conn.execute('''INSERT OR REPLACE INTO modelos_ml
                (liga, data_treinamento, num_amostras, modelo_blob, scaler_params, feature_order, acuracia, log_loss, roc_auc, model_version)
                VALUES (?,?,?,?,?,?,?,?,?,?)''',
                (liga, get_brt_time().strftime("%Y-%m-%d %H:%M:%S"), len(X),
                modelo_bytes, json.dumps(scaler_params), json.dumps(feature_order),
                 avg_acc, avg_loss, avg_roc_auc, 6))
            # O marcador pertence ao campeão global e não pode ser consumido
            # por um artefato opcional de liga.
            conn.commit()

    with _model_cache_lock:
        _model_cache.clear()
    return calibrated_final, scaler_final, feature_order, avg_acc, avg_loss, avg_roc_auc

def _treinar_modelo_global_direto_legado_inseguro(min_amostras=50, usar_otimizacao=False):
    X, y, feature_order, _, pesos = preparar_dados_treinamento(liga=None)
    if X is None or len(X) < min_amostras:
        return None, None, None, 0.0, 0.0

    if set(y) != {0, 1, 2}:
        logger.warning("Modelo global: faltam classes reais; treino cancelado.")
        return None, None, None, 0.0, 0.0

    final_weights = balanced_sample_weights(y, pesos)

    xgb_params_no_early = {
        'objective': 'multi:softprob',
        'num_class': 3,
        'eval_metric': 'mlogloss',
        'random_state': 42,
        'n_jobs': -1,
        'max_depth': 3,
        'learning_rate': 0.03,
        'n_estimators': 350,
        'subsample': 0.85,
        'colsample_bytree': 0.75,
        'gamma': 0.05,
        'reg_alpha': 0.2,
        'reg_lambda': 4.0,
        'min_child_weight': 10,
    }

    # O scaler da avaliação também só enxerga o passado; o scaler final usa tudo.
    corte = max(1, int(len(X) * 0.80))
    scaler_avaliacao = StandardScaler()
    X_train_eval = scaler_avaliacao.fit_transform(X[:corte])
    X_test_eval = scaler_avaliacao.transform(X[corte:])
    avaliador = xgb.XGBClassifier(**xgb_params_no_early)
    avaliador.fit(X_train_eval, y[:corte], sample_weight=final_weights[:corte])
    proba_holdout = avaliador.predict_proba(X_test_eval)
    global_acc = accuracy_score(y[corte:], np.argmax(proba_holdout, axis=1))
    global_loss = log_loss(y[corte:], proba_holdout, labels=[0, 1, 2])
    try:
        global_roc = roc_auc_score(y[corte:], proba_holdout, multi_class='ovr', average='weighted')
    except ValueError:
        global_roc = 0.0

    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    model = xgb.XGBClassifier(**xgb_params_no_early)
    try:
        calibrated = CalibratedClassifierCV(
            model, method='sigmoid', cv=TimeSeriesSplit(n_splits=3))
        calibrated.fit(X_s, y, sample_weight=final_weights)
    except (ValueError, xgb.core.XGBoostError):
        calibrated = model
        calibrated.fit(X_s, y, sample_weight=final_weights)

    buf = BytesIO()
    joblib.dump(calibrated, buf, compress=True)
    modelo_bytes = buf.getvalue()
    scaler_params = {'mean': scaler.mean_.tolist(), 'scale': scaler.scale_.tolist()}

    with db_write_lock:
        with get_db_connection() as conn:
            conn.execute('''INSERT OR REPLACE INTO modelos_ml
                (liga, data_treinamento, num_amostras, modelo_blob, scaler_params, feature_order, acuracia, log_loss, roc_auc, model_version)
                VALUES (?,?,?,?,?,?,?,?,?,?)''',
                ('GLOBAL', get_brt_time().strftime("%Y-%m-%d %H:%M:%S"), len(X),
                 modelo_bytes, json.dumps(scaler_params), json.dumps(feature_order),
                  global_acc, global_loss, global_roc, MODEL_VERSION))
            conn.execute('''CREATE TABLE IF NOT EXISTS ml_team_ratings (
                team_name TEXT PRIMARY KEY, elo REAL NOT NULL, updated_at DATETIME NOT NULL)''')
            timestamp_rating = get_brt_time().strftime("%Y-%m-%d %H:%M:%S")
            conn.executemany("INSERT OR REPLACE INTO ml_team_ratings (team_name, elo, updated_at) VALUES (?,?,?)",
                             [(team, float(elo), timestamp_rating)
                              for team, elo in globals().get('_ultimos_elos_treino', {}).items()])
            conn.commit()

    with _model_cache_lock:
        _model_cache.clear()
    return calibrated, scaler, feature_order, global_acc, global_loss

def treinar_modelo_global_sem_vazamento(min_amostras=50, usar_otimizacao=False):
    """Compatibilidade segura: todo treino global passa pelo gate de evolução."""
    logger.warning(
        "Treino global direto desativado; executando campeão x desafiante."
    )
    return executar_evolucao_automatica(forcar=True)


def executar_evolucao_automatica(forcar=False):
    """Aprende dados novos, mas preserva o campeão quando o desafiante piora."""
    started_at = get_brt_time().strftime("%Y-%m-%d %H:%M:%S")
    with get_db_connection() as conn:
        total_samples = int(conn.execute("SELECT COUNT(*) FROM training_data").fetchone()[0])
        row = conn.execute("""SELECT num_amostras, modelo_blob, scaler_params,
                                      feature_order, data_treinamento, model_version
                               FROM modelos_ml WHERE liga='GLOBAL'""").fetchone()
        champion_samples = int(row[0] or 0) if row else 0
        unseen_ids = {
            str(item[0]) for item in conn.execute(
                "SELECT match_id FROM training_data WHERE usado_treinamento=0"
            ).fetchall()
        }
        last_attempt = conn.execute("""SELECT total_samples FROM ml_evolution_runs
            WHERE status IN ('PROMOTED','REJECTED') ORDER BY id DESC LIMIT 1""").fetchone()
        prediction_groups = {}
        try:
            for match_id, ticket_id, radar_run_id in conn.execute("""
                SELECT match_id, ticket_id, radar_run_id
                FROM previsoes
                WHERE ticket_id IS NOT NULL AND TRIM(ticket_id)!=''
                  AND radar_run_id IS NOT NULL AND TRIM(radar_run_id)!=''
                ORDER BY id
            """).fetchall():
                prediction_groups[str(match_id)] = (
                    str(ticket_id), str(radar_run_id)
                )
        except sqlite3.OperationalError:
            prediction_groups = {}
    reference_samples = max(champion_samples, int(last_attempt[0] or 0) if last_attempt else 0)
    new_samples = max(0, total_samples - reference_samples)
    if not forcar and row and new_samples < ML_MIN_NEW_SAMPLES:
        reason = f"Aguardando {ML_MIN_NEW_SAMPLES} novos resultados; disponíveis: {new_samples}"
        record_skipped_evolution(DB_NAME, total_samples, new_samples, reason)
        logger.info(reason)
        return {"status": "SKIPPED", "reason": reason, "new_samples": new_samples}

    X, y, feature_order, match_ids, pesos = preparar_dados_treinamento(liga=None)
    if X is None:
        reason = "Base de treinamento vazia"
        record_skipped_evolution(DB_NAME, total_samples, new_samples, reason)
        return {"status": "SKIPPED", "reason": reason}
    champion_artifact = None
    if row:
        try:
            champion_artifact = {
                "model_blob": row[1],
                "scaler_params": json.loads(row[2]),
                "feature_order": json.loads(row[3]),
                "trained_at": row[4], "model_version": row[5],
            }
        except (TypeError, ValueError, json.JSONDecodeError):
            champion_artifact = {
                "model_blob": row[1], "scaler_params": {}, "feature_order": []
            }
    comparison_mask = np.asarray(
        [str(match_id) in unseen_ids for match_id in match_ids], dtype=bool
    )
    ticket_groups = [
        prediction_groups.get(str(match_id), (None, None))[0]
        for match_id in match_ids
    ]
    radar_run_groups = [
        prediction_groups.get(str(match_id), (None, None))[1]
        for match_id in match_ids
    ]
    result = evaluate_evolution(
        X, y, feature_order, pesos,
        confidence_floor=MIN_ML_CONFIDENCE / 100.0,
        target_accuracy=0.50,
        radar_mask=globals().get('_ultimo_radar_mask_treino'),
        champion_artifact=champion_artifact,
        comparison_mask=comparison_mask,
        match_ids=match_ids,
        ticket_groups=ticket_groups,
        radar_run_groups=radar_run_groups,
        context_rows=globals().get('_ultimos_registros_treino'),
        probability_overlay=lambda probabilities, features: analyze_pregame_context(
            probabilities, features, analysis_profile=ACTIVE_ANALYSIS_PROFILE
        ),
    )
    persist_evolution_result(
        DB_NAME, result, len(X), new_samples, feature_order,
        team_ratings=globals().get('_ultimos_elos_treino', {}), started_at=started_at,
    )
    if result.get("promote"):
        with _model_cache_lock:
            _model_cache.clear()
    return result

def carregar_modelo_liga(liga):
    cache_key = liga if USAR_MODELOS_POR_LIGA else 'GLOBAL'
    now = time.monotonic()
    with _model_cache_lock:
        cached = _model_cache.get(cache_key)
        if cached and now - cached[0] <= MODEL_CACHE_TTL_SECONDS:
            return cached[1]
    with get_db_connection() as conn:
        row = None
        if USAR_MODELOS_POR_LIGA:
            row = conn.execute("SELECT liga, modelo_blob, scaler_params, feature_order FROM modelos_ml WHERE liga = ? AND COALESCE(model_version,1) >= 4", (liga,)).fetchone()
        if not row:
            row = conn.execute("SELECT liga, modelo_blob, scaler_params, feature_order FROM modelos_ml WHERE liga = 'GLOBAL' AND COALESCE(model_version,1) >= 4").fetchone()
        if not row:
            return None, None, None

    actual_liga, modelo_bytes, scaler_params_str, feat_order_str = row

    try:
        from model_artifact_integrity import load_model_artifact
        model, scaler, feature_order = load_model_artifact(
            modelo_bytes, scaler_params_str, feat_order_str
        )
    except Exception as e:
        logger.error(
            "Artefato ML '%s' inválido (%s: %s); preservado no banco para diagnóstico.",
            actual_liga, type(e).__name__, e,
        )
        if actual_liga != 'GLOBAL':
            return carregar_modelo_liga('GLOBAL')
        return None, None, None

    result = (model, scaler, feature_order)
    with _model_cache_lock:
        _model_cache[cache_key] = (time.monotonic(), result)
    return result

def prever_com_ml(match_id, home_id, away_id, tournament_id, season_id,
                  odd_casa, odd_empate, odd_fora, liga, unique_tournament_id='',
                  home_team=None, away_team=None, start_timestamp=None, research_run_id=None,
                  research_only=False, eligibility_group=None):
    model, scaler, feature_order = carregar_modelo_liga(liga)
    if model is None:
        return None
    active_version, active_id = get_active_model_identity(DB_NAME)
    frozen_model_identity = f"v{active_version}:{active_id}"
    features_leves = (FEATURES_LEGADAS_SEGURAS
                      | {k for k in feature_order
                         if k.startswith(('hist_', 'liga_prior_', 'elo_ml_', 'form_', 'context_',
                                          'calendario_', 'temporada_', 'sofa_pre_', 'sofa_roll_',
                                          'live_recent_', 'allsports_goal_'))})
    provided_start = safe_event_timestamp(start_timestamp)
    info = ({'home_team': home_team, 'away_team': away_team, 'liga': liga,
             'startTimestamp': provided_start}
            if home_team and away_team and provided_start else obter_info_partida(match_id))
    event_start = safe_event_timestamp(info.get('startTimestamp')) if info else 0
    if not event_start:
        logger.warning("ML: jogo %s ignorado por horário inicial ausente/inválido.", match_id)
        return None
    if int(time.time()) >= event_start:
        logger.warning("ML: jogo %s ignorado porque o horário pré-jogo já encerrou.", match_id)
        return None
    info = dict(info)
    info['startTimestamp'] = event_start
    if set(feature_order).issubset(features_leves):
        liga_atual = info.get('liga', liga) if info else liga
        knock, volta = detectar_fase_mata_mata(liga_atual)
        is_final = int('final' in liga_atual.lower())
        feats = {
            'nivel_campeonato': get_nivel_campeonato(liga_atual),
            'is_knockout': knock, 'is_volta': volta,
            'prioridade_torneio': obter_prioridade_torneio(liga_atual, knock, is_final),
        }
        feats.update(competition_flags(liga_atual))
    else:
        feats = extrair_features_basicas(match_id, home_id, away_id, tournament_id, season_id,
                                         odd_casa, odd_empate, odd_fora, unique_tournament_id)
    if info:
        feats.update(_features_historicas_db(
            info.get('home_team'), info.get('away_team'), liga, info.get('startTimestamp'),
            tournament_id=tournament_id, season_id=season_id,
            unique_tournament_id=unique_tournament_id))
    feats.update(get_soccer_context_features(
        DB_NAME, match_id,
        home_name=(info.get('home_team') if info else home_team),
        away_name=(info.get('away_team') if info else away_team),
        league=(info.get('liga', liga) if info else liga),
    ))
    feats.update(get_sofascore_pregame_features(
        DB_NAME, match_id,
        home_name=(info.get('home_team') if info else home_team) or '',
        away_name=(info.get('away_team') if info else away_team) or '',
        cutoff_timestamp=(info.get('startTimestamp') if info else start_timestamp),
    ))
    _enriquecer_duelo_sofa(feats)
    # New free-source fields are frozen for candidate evaluation only. The
    # incumbent's feature list/profile is not changed by this collection.
    feats.update(load_free_statistics(DB_NAME, match_id,
        (info.get('startTimestamp') if info else start_timestamp)))
    vec = []
    for feature_index, k in enumerate(feature_order):          # REMOVIDO o slicing [:10]
        # O descanso foi retirado. Em artefatos antigos, usar a média do scaler
        # neutraliza a variável sem criar uma entrada extrema padronizada.
        val = (float(scaler.mean_[feature_index])
               if k.startswith('calendario_descanso_') else feats.get(k, 0.0))
        try:
            vec.append(float(val))
        except (ValueError, TypeError):
            vec.append(0.0)
    X = np.array([vec])
    X_s = scaler.transform(X)
    proba_base = model.predict_proba(X_s)[0]
    analise_contextual = analyze_pregame_context(
        proba_base, feats, analysis_profile=ACTIVE_ANALYSIS_PROFILE
    )
    proba = np.asarray(analise_contextual['probabilities'], dtype=float)
    idx = int(np.argmax(proba))
    # Preserva a probabilidade calibrada na borda do corte de seleção.
    conf = round(float(proba[idx]) * 100.0, 2)
    vencedor = ['MANDANTE', 'EMPATE', 'VISITANTE'][idx]
    snapshot_analysis = dict(analise_contextual)
    snapshot_analysis['target_competition'] = {
        'provider': 'allsports',
        'tournament_id': str(tournament_id or ''),
        'unique_tournament_id': str(unique_tournament_id or ''),
        'season_id': str(season_id or ''),
        'name': str(liga or ''),
    }
    from phase2_observations import append as observe, new_run, clean
    from phase3_research import coverage_summary
    from phase4_research import capture_prediction_shadow
    capture_run_id = research_run_id or new_run()
    phase4_coverage = coverage_summary(feats, int(time.time()))
    prediction_stage = ('prediction_phase4_prefilter_v1' if research_only else 'prediction')
    observe(DB_NAME, capture_run_id, str(match_id), prediction_stage, {
        'features': feats, 'analysis': snapshot_analysis,
        'probabilities': proba.tolist(), 'pick': vencedor,
        'analysis_profile': ACTIVE_ANALYSIS_PROFILE,
        'model_version': frozen_model_identity,
        'model_class': type(model).__name__,
        'home_odd': odd_casa, 'draw_odd': odd_empate, 'away_odd': odd_fora,
        'provider': 'allsports', 'vendor_quote_timestamp': None,
        'home_team': home_team, 'away_team': away_team, 'league': liga,
        'season': str(season_id or ''), 'kickoff': start_timestamp,
        'quality_score': None, 'gate_decision': 'DISABLED',
        'missing_feature_keys': [k for k, v in feats.items() if clean(v) is None],
        'missing_indicators': {k: clean(v) is None for k, v in feats.items()},
        'data_coverage': phase4_coverage,
    })
    if not research_only:
        from phase3_research import capture_shadow
        capture_shadow(DB_NAME, capture_run_id, str(match_id), proba.tolist(),
                       [odd_casa, odd_empate, odd_fora],
                       (info.get('startTimestamp') if info else start_timestamp),
                       frozen_model_identity)
    capture_prediction_shadow(
        DB_NAME, capture_run_id, str(match_id), proba.tolist(),
        [odd_casa, odd_empate, odd_fora],
        (info.get('startTimestamp') if info else start_timestamp),
        frozen_model_identity, phase4_coverage,
        stage=('shadow_phase4_prefilter_champion_v1' if research_only
               else 'shadow_phase4_prediction_v1'),
        eligibility=eligibility_group,
    )
    if not research_only:
        save_prediction_snapshot(
            DB_NAME, match_id, (info.get('startTimestamp') if info else start_timestamp),
            feats, snapshot_analysis, vencedor,
        )
        record_pick_scenario_shadow(DB_NAME, match_id,
            (info.get('startTimestamp') if info else start_timestamp), feats, analise_contextual)
    fontes = []
    if feats.get('context_sfi_available', 0): fontes.append('forma Soccer atual')
    if feats.get('sofa_pre_available', 0): fontes.append('SofaScore pré-jogo')
    if feats.get('live_recent_available', 0): fontes.append('últimos 5 atualizados')
    if feats.get('sofa_roll_available', 0): fontes.append('desempenho detalhado anterior')
    fonte_texto = f" + {' + '.join(fontes)}" if fontes else ""
    relatorio = (f"🤖 ML sem odds{fonte_texto} "
                 f"({liga if liga != 'GLOBAL' else 'Global'}) | "
                 f"C:{proba[0]:.1%} E:{proba[1]:.1%} F:{proba[2]:.1%} | "
                 f"risco empate:{analise_contextual['draw_risk']:.0%} "
                 f"qualidade contexto:{analise_contextual['information_quality']:.0%}")
    return vencedor, conf, relatorio

def atualizar_peso_erro(match_id, errou=True):
    with db_write_lock:
        conn=get_db_connection(); cur=conn.cursor()
        cur.execute("SELECT peso FROM training_weights WHERE match_id=?",(match_id,))
        row=cur.fetchone()
        if row:
            novo_peso = 2.0 if errou else 1.0
            if row[0] != novo_peso:
                cur.execute("UPDATE training_weights SET peso=?, data_ultima_atualizacao=? WHERE match_id=?",
                            (novo_peso, get_brt_time().strftime("%Y-%m-%d %H:%M:%S"), match_id))
        else:
            novo_peso = 2.0 if errou else 1.0
            cur.execute("INSERT OR IGNORE INTO training_weights (match_id,peso,data_ultima_atualizacao) VALUES (?,?,?)",
                        (match_id, novo_peso, get_brt_time().strftime("%Y-%m-%d %H:%M:%S")))
        conn.commit(); conn.close()

# ----------------------------------------------------------------------
# FUNÇÃO ORIGINAL calcular_edge_super_python_v5
# ----------------------------------------------------------------------
def calcular_edge_super_python_v5(h_form_str, a_form_str, h2h_str, class_str, desfalques_str, streaks_str,
                                 stats_home, stats_away, odd_casa, odd_fora, odd_empate,
                                 form_rating_str, arbitro_str, dist_gols_h, dist_gols_a, pr_str,
                                 perf_h, perf_a, macro_liga, win_prob, tatica_str,
                                 ef_home, ef_away, liga, match_id, home_id, away_id, tournament_id, season_id,
                                 home_team=None, away_team=None, start_timestamp=None,
                                 unique_tournament_id='', research_run_id=None):
    pred_ml = prever_com_ml(match_id, home_id, away_id, tournament_id, season_id,
                           odd_casa, odd_empate, odd_fora, liga,
                           unique_tournament_id=unique_tournament_id,
                           home_team=home_team, away_team=away_team,
                           start_timestamp=start_timestamp, research_run_id=research_run_id)
    if pred_ml is not None:
        vencedor, confianca, relatorio = pred_ml
        return vencedor, confianca, relatorio, {'casa': 0.33, 'empate': 0.33, 'fora': 0.33}

    # Odds servem somente para o filtro >1,99. Se o artefato ML estiver
    # indisponível, o app não fabrica uma previsão baseada no mercado.
    return None, None, None, None

# ----------------------------------------------------------------------
# FUNÇÕES DO RADAR
# ----------------------------------------------------------------------
def extrair_dados_allsports(dados_json, bloco_odds, ts_inicio, ts_fim):
    jogos_extraidos = []
    lista_eventos = dados_json.get("events", dados_json.get("data", []))
    for jogo in lista_eventos:
        try:
            jogo_id = str(jogo.get("id",""))
            start_ts = safe_event_timestamp(jogo.get("startTimestamp"))
            if start_ts <= 0:
                logger.warning("Evento %s descartado: startTimestamp inválido", jogo_id)
                continue
            if not (ts_inicio <= start_ts < ts_fim):
                continue
            if jogo_id not in bloco_odds:
                continue
            dt_jogo_brt = datetime.fromtimestamp(start_ts, timezone(timedelta(hours=-3)))
            casa = jogo.get("homeTeam",{}).get("name","Casa")
            fora = jogo.get("awayTeam",{}).get("name","Fora")
            pais = jogo.get("tournament",{}).get("category",{}).get("name","Mundo")
            liga = f"{pais} - {jogo.get('tournament',{}).get('name','Liga')}"
            texto = f"{casa} {fora} {liga}".lower()
            if any(termo in texto for termo in BLACKLIST_TERMS):
                continue
            home_id = str(jogo.get("homeTeam",{}).get("id",""))
            away_id = str(jogo.get("awayTeam",{}).get("id",""))
            tourn_id = str(jogo.get("tournament",{}).get("id",""))
            season_id = str(jogo.get("season",{}).get("id",""))
            unique_tourn_id = str(jogo.get("tournament", {}).get("uniqueTournament", {}).get("id", ""))
            odds_jogo = bloco_odds.get(jogo_id, {}) if isinstance(bloco_odds, dict) else {}
            mercados = odds_jogo.get("choices", []) if isinstance(odds_jogo, dict) else []
            o_casa = o_fora = o_empate = 0.0
            for m in mercados:
                if m.get("name") == "1":
                    o_casa = extrair_fracional(m.get("fractionalValue") or m.get("value"))
                if m.get("name") == "2":
                    o_fora = extrair_fracional(m.get("fractionalValue") or m.get("value"))
                if m.get("name") == "X":
                    o_empate = extrair_fracional(m.get("fractionalValue") or m.get("value"))
            if o_casa > 1.99 and o_fora > 1.99:
                jogos_extraidos.append({
                    "ID": jogo_id, "Dia": dt_jogo_brt.strftime("%d/%m"), "Hora": dt_jogo_brt.strftime("%H:%M"),
                    "Liga": liga, "Time Casa": casa, "Odd Casa": o_casa, "Empate": o_empate, "Odd Fora": o_fora,
                    "Time Fora": fora, "Timestamp": start_ts, "Confronto": f"{casa} vs {fora}",
                    "Home_ID": home_id, "Away_ID": away_id, "Tournament_ID": tourn_id, "Season_ID": season_id,
                    "Unique_Tournament_ID": unique_tourn_id,
                })
                salvar_ids_liga(liga, tourn_id, season_id)
        except Exception as e:
            logger.warning("Evento AllSports %s descartado no parser: %s", jogo_id, e)
            continue
    return jogos_extraidos

def calcular_odd_multipla(jogos):
    odd=1.0
    for j in jogos:
        casa_nome=str(j.get('Confronto','')).split(' vs ')[0].strip().lower()
        pick=str(j.get('Vencedor Escolhido','')).strip().lower()
        if 'empate' in pick: odd_jogo=float(j.get('Empate',1.0))
        elif casa_nome in pick or pick in casa_nome: odd_jogo=float(j.get('Odd Casa',1.0))
        else: odd_jogo=float(j.get('Odd Fora',1.0))
        if odd_jogo<=1.0: odd_jogo=max(float(j.get('Odd Casa',1.0)),float(j.get('Odd Fora',1.0)))
        j['Odd_Calculada']=odd_jogo
        odd*=max(odd_jogo,1.0)
    return odd

def disparar_telegram(bilhetes, token, chat_id):
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    sucessos = 0
    conn = get_db_connection()
    cursor = conn.cursor()
    for b in bilhetes:
        b["_telegram_enviado"] = False
        odd_multipla = calcular_odd_multipla(b.get('Jogos', []))
        odds_disponiveis = all(float(j.get('Odd_Calculada', 0) or 0) > 1.0 for j in b.get('Jogos', []))
        prob_conjunta = float(b.get('Probabilidade Conjunta', 0.0) or np.prod([
            max(0.0, min(1.0, float(j.get('Confiança', 0)) / 100.0)) for j in b.get('Jogos', [])]))
        cabecalho_odd = (f"🔥 *Odd Múltipla: {odd_multipla:.2f}*" if odds_disponiveis
                         else "🧠 *Modelo contextual: odds ignoradas*")
        texto_msg = (f"⚡ *{b.get('Nome')}*\n{cabecalho_odd}\n"
                     f"📐 *Prob. conjunta estimada: {prob_conjunta:.2%}*\n━━━━━━━━━━━━━━━━━━\n")
        for j in b.get('Jogos', []):
            pick = j.get('Vencedor Escolhido', '').upper()
            odd_jogo = float(j.get('Odd_Calculada', 0) or 0)
            texto_odd = f" *(Odd: {odd_jogo:.2f})*" if odd_jogo > 1.0 else ""
            texto_msg += (f"⚽ *{j.get('Confronto')}*\n"
                          f"⏰ {j.get('Dia_Str')} {j.get('Hora BRT')} BRT | 🌍 🏆 {j.get('Liga_Exata', j.get('Liga', ''))}\n"
                          f"🎯 *Pick:* `{pick}`{texto_odd} - 📊 Confiança: {j.get('Confiança', 0)}%\n\n")
        ticket_id = str(b.get('Nome') or '')
        payload_hash = delivery_fingerprint(b.get('Jogos', []))
        autorizado, motivo, message_id_existente = claim_ticket_delivery(
            DB_NAME, ticket_id, payload_hash
        )
        if not autorizado:
            logger.info("Telegram ignorado para %s: envio %s.", ticket_id, motivo)
            if motivo == "sent":
                with db_write_lock:
                    cursor.execute("""UPDATE previsoes
                        SET telegram_msg_id=COALESCE(?,telegram_msg_id), telegram_enviado=1
                        WHERE ticket_id=?""", (message_id_existente, ticket_id))
                    conn.commit()
                b["_telegram_enviado"] = True
            continue
        for tentativa in range(3):
            try:
                res = requests.post(
                    url, json={"chat_id": chat_id, "text": texto_msg, "parse_mode": "Markdown"},
                    timeout=15)
                if res.status_code == 200:
                    sucessos += 1
                    msg_id = res.json().get('result', {}).get('message_id')
                    if msg_id:
                        mark_ticket_sent(DB_NAME, ticket_id, msg_id)
                        with db_write_lock:
                            cursor.execute("""UPDATE previsoes
                                SET telegram_msg_id = ?, telegram_enviado = 1
                                WHERE ticket_id = ?""", (str(msg_id), ticket_id))
                            conn.commit()
                    b["_telegram_enviado"] = True
                    time.sleep(1)
                    break
                elif res.status_code == 429:
                    time.sleep(res.json().get('parameters', {}).get('retry_after', 5) + 1)
                else:
                    mark_ticket_failed(DB_NAME, ticket_id, f"HTTP {res.status_code}: {res.text[:500]}")
                    time.sleep(2)
            except Exception as e:
                logger.error(f"Erro ao enviar Telegram: {e}")
                mark_ticket_failed(DB_NAME, ticket_id, repr(e), ambiguous=True)
                break
    conn.close()
    return sucessos

def atualizar_mensagem_telegram_por_bilhete(ticket_id):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT confronto, liga, vencedor_previsto, odd_casa, odd_fora, odd_empate, status_resultado, placar_real, telegram_msg_id, data_jogo, hora_jogo, confianca, anulado FROM previsoes WHERE ticket_id = ?", (ticket_id,))
    jogos = cursor.fetchall()
    conn.close()
    if not jogos: return
    msg_id_str = jogos[0][8]
    if not msg_id_str or msg_id_str == 'None': return
    try: msg_id = int(msg_id_str)
    except: return
    jogos_nao_anulados = [j for j in jogos if j[12] != 1]
    if not jogos_nao_anulados: return
    statuses = [j[6] for j in jogos_nao_anulados]
    todos_resolvidos = all('PENDENTE' not in status for status in statuses)
    todos_green = todos_resolvidos and all('GREEN' in status for status in statuses)
    odd_multipla = 1.0
    odds_disponiveis = True
    linhas_jogos = ""
    for j in jogos_nao_anulados:
        confronto, liga, pick, o_c, o_f, o_e, status, placar, _, d_j, h_j, conf, _ = j
        casa_nome = str(confronto).split(' vs ')[0].strip().lower()
        pick_lower = str(pick).strip().lower()
        if 'empate' in pick_lower: odd_jogo = float(o_e)
        elif casa_nome in pick_lower: odd_jogo = float(o_c)
        else: odd_jogo = float(o_f)
        if odd_jogo <= 1.0: odd_jogo = max(float(o_c), float(o_f))
        if odd_jogo <= 1.0:
            odds_disponiveis = False
            odd_jogo = 0.0
        odd_multipla *= max(odd_jogo, 1.0)
        icon = "✅" if "GREEN" in status else "❌" if "RED" in status else "⏳" if "PENDENTE" in status else "🚫" if "ANULADO" in status else "ℹ️"
        d_j_str = d_j if d_j else "--/--"; h_j_str = h_j if h_j else "--:--"
        linhas_jogos += f"⚽ *{confronto}* ({placar})\n"
        linhas_jogos += f"⏰ {d_j_str} {h_j_str} BRT | 🌍 🏆 {liga}\n"
        texto_odd = f" *(Odd: {odd_jogo:.2f})*" if odd_jogo > 1.0 else ""
        linhas_jogos += f"{icon} *Pick:* `{pick}`{texto_odd} - 📊 Confiança: {conf}% - {status.replace('ARQUIVADO ', '')}\n\n"
    cabecalho_odd = (f"🔥 *Odd Múltipla: {odd_multipla:.2f}*" if odds_disponiveis
                     else "🧠 *Modelo contextual: odds ignoradas*")
    titulo = (f"✅ GREEN | {ticket_id}" if todos_green else
              (f"🔴 RED | {ticket_id}" if todos_resolvidos else ticket_id))
    texto_msg = f"⚡ *{titulo}*\n{cabecalho_odd}\n━━━━━━━━━━━━━━━━━━\n" + linhas_jogos
    payload = {"chat_id": TELEGRAM_CHAT_ID, "message_id": msg_id, "text": texto_msg, "parse_mode": "Markdown"}
    for tentativa in range(3):
        try:
            response = requests.post(
                f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/editMessageText",
                json=payload, timeout=15,
            )
            if response.status_code == 200:
                return True
            if response.status_code == 400 and "message is not modified" in response.text.lower():
                return True
            if response.status_code == 429:
                retry_after = (response.json().get("parameters") or {}).get("retry_after", 2)
                time.sleep(min(15, max(1, int(retry_after))) + 1)
                continue
            logger.warning("Telegram não atualizou %s: HTTP %s %s", ticket_id,
                           response.status_code, response.text[:200])
        except requests.RequestException as exc:
            logger.warning("Falha ao atualizar %s no Telegram: %s", ticket_id, exc)
        if tentativa < 2:
            time.sleep(1 + tentativa)
    return False

def verificar_e_celebrar_green(ticket_id):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""CREATE TABLE IF NOT EXISTS telegram_tickets (
        ticket_id TEXT PRIMARY KEY, message_id TEXT, notificado INTEGER DEFAULT 0
    )""")
    ja_notificado = cursor.execute(
        "SELECT COALESCE(notificado,0) FROM telegram_tickets WHERE ticket_id=?",
        (str(ticket_id),),
    ).fetchone()
    if ja_notificado and int(ja_notificado[0] or 0) == 1:
        conn.close()
        return
    cursor.execute("SELECT confronto, vencedor_previsto, odd_casa, odd_fora, odd_empate, telegram_msg_id, status_resultado, anulado FROM previsoes WHERE ticket_id = ?", (ticket_id,))
    jogos = cursor.fetchall()
    conn.close()
    if not jogos: return
    jogos_nao_anulados = [j for j in jogos if j[7] != 1]
    if not jogos_nao_anulados: return
    statuses = [j[6] for j in jogos_nao_anulados]
    if any('PENDENTE' in s for s in statuses) or any('RED' in s for s in statuses): return
    msg_id_str = jogos[0][5]
    if not msg_id_str or msg_id_str == 'None': return
    odd_multipla = 1.0
    for j in jogos_nao_anulados:
        confronto, pick, o_c, o_f, o_e = j[0], j[1], j[2], j[3], j[4]
        casa_nome = str(confronto).split(' vs ')[0].strip().lower()
        pick_lower = str(pick).strip().lower()
        if 'empate' in pick_lower: odd_jogo = float(o_e)
        elif casa_nome in pick_lower: odd_jogo = float(o_c)
        else: odd_jogo = float(o_f)
        if odd_jogo <= 1.0: odd_jogo = max(float(o_c), float(o_f))
        odd_multipla *= max(odd_jogo, 1.0)
    texto_msg = f"✅ <b>BINGO! GREEN NO BOLSO!</b> 💰\n\nO bilhete <b>{ticket_id}</b> bateu com uma Odd Múltipla de <b>{odd_multipla:.2f}</b>! 🔥"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": texto_msg, "parse_mode": "HTML", "reply_to_message_id": int(msg_id_str)}
    for tentativa in range(3):
        try:
            res = requests.post(
                f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
                json=payload, timeout=15,
            )
            if res.status_code == 200:
                with db_write_lock:
                    with get_db_connection() as notify_conn:
                        notify_conn.execute(
                            """INSERT INTO telegram_tickets(ticket_id,message_id,notificado)
                               VALUES (?,?,1) ON CONFLICT(ticket_id) DO UPDATE SET
                               message_id=excluded.message_id, notificado=1""",
                            (str(ticket_id), str(res.json().get('result', {}).get('message_id') or '')),
                        )
                        notify_conn.commit()
                break
            time.sleep(2)
        except requests.RequestException:
            time.sleep(1)

def verificar_pagamento_antecipado(match_id, pick, home_name, away_name):
    try:
        data = safe_api_get(match_resource_url(RAPIDAPI_HOST, match_id, "incidents"), max_retries=1, timeout=6)
        if data:
            incidents = data.get('incidents', [])
            side = resolve_pick_side(None, pick, home_name, away_name)
            bet_on_home = side == 'MANDANTE'
            bet_on_away = side == 'VISITANTE'
            if not bet_on_home and not bet_on_away: return False
            for inc in incidents:
                score = incident_goal_score(inc)
                if score is not None:
                    h_score, a_score = score
                    if bet_on_home and (h_score - a_score) >= 2: return True
                    if bet_on_away and (a_score - h_score) >= 2: return True
    except: pass
    return False

# ======================================================================
# INTERFACE STREAMLIT
# ======================================================================
st.set_page_config(page_title="Nexus Quant | Python", layout="wide", initial_sidebar_state="collapsed")
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:opsz,wght@14..32,300;14..32,400;14..32,500;14..32,600;14..32,700&display=swap');
html, body, .stApp { background-color: #0a0c10; font-family: 'Inter', sans-serif; color: #e5e9f0; }
.stTabs [data-baseweb="tab-list"] { gap: 0px; background-color: #111316; border-bottom: 1px solid #2a2d34; padding: 0 1rem; }
.stTabs [data-baseweb="tab"] { height: 48px; padding: 0 1.5rem; font-weight: 500; letter-spacing: -0.01em; color: #8b8f9c; background-color: transparent; border-radius: 0; transition: all 0.2s ease; }
.stTabs [data-baseweb="tab"]:hover { color: #ffffff; background-color: #1a1d24; }
.stTabs [aria-selected="true"] { color: #ffffff !important; background-color: #0a0c10; border-bottom: 2px solid #ffffff !important; }
div.stButton > button { background-color: #1e2128; color: #e5e9f0; border: 1px solid #2f333d; border-radius: 10px; font-weight: 500; transition: all 0.2s ease; padding: 0.5rem 1rem; width: 100%; }
div.stButton > button:hover { background-color: #2a2e38; border-color: #4a4f5e; transform: translateY(-1px); }
div.stButton > button[kind="primary"] { background-color: #facc15; color: #0a0c10; border: none; font-weight: 600; }
div.stButton > button[kind="primary"]:hover { background-color: #eab308; transform: translateY(-1px); }
.match-card { background-color: #111316; border: 1px solid #2a2d34; border-radius: 16px; padding: 1.25rem; margin-bottom: 1rem; transition: all 0.2s ease; }
.match-card:hover { border-color: #3b3f4a; box-shadow: 0 4px 12px rgba(0,0,0,0.2); }
</style>
""", unsafe_allow_html=True)

st.title("Nexus Quant (Python Puro) - ML Refatorado")
st.caption("Dashboard Master | AllSports PRO V5 - Machine Learning Integrado (Nova Pipeline)")

if "resultados_v4" not in st.session_state:
    st.session_state.resultados_v4 = []
if "agregado_bil" not in st.session_state:
    st.session_state.agregado_bil = []
if "api_errors" not in st.session_state:
    st.session_state.api_errors = []

abas = ["📡 Radar", "🗄️ Logs", "📊 Auditoria", "🎫 Em Aberto", "🏦 Gestão P&L", 
        "📈 Macro Ligas", "📡 Log de Erros API", "🗃️ Database de Aprendizado", "Diagnostico",
        "🎓 Treinamento ML", "🔍 Avaliação", "📈 Backtest", "🧪 Teste de Estatísticas", 
        "📆 Relatório Diário", "📊 Monitor de Ligas"]
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab_db, tab_diagnostico, tab_ml, tab_avaliacao, tab9, tab_teste, tab_relatorio, tab_monitor_ligas = st.tabs(abas)

# ---------- ABA 1: RADAR ----------
with tab1:
    BLACKLIST_TERMS = [
        "u12","u13","u14","u15","u16","u17","u18","u19","u20","u21","u22","u23","u24",
        "u 12","u 13","u 14","u 15","u 16","u 17","u 18","u 19","u 20","u 21","u 22","u 23","u 24",
        "u-12","u-13","u-14","u-15","u-16","u-17","u-18","u-19","u-20","u-21","u-22","u-23","u-24",
        "sub12","sub13","sub14","sub15","sub16","sub17","sub18","sub19","sub20","sub21","sub22","sub23","sub24",
        "sub 12","sub 13","sub 14","sub 15","sub 16","sub 17","sub 18","sub 19","sub 20","sub 21","sub 22","sub 23","sub 24",
        "sub-12","sub-13","sub-14","sub-15","sub-16","sub-17","sub-18","sub-19","sub-20","sub-21","sub-22","sub-23","sub-24",
        "sub","junior","juvenil","youth","aspirantes","reserva","reservas","reserves","reserve",
        "amateur","amador","amadores",
        "woman","women","feminino","femenino","femenil","femmes","frauen","ladies","girls",
        " w ", "w's", "womens",
    ]

    if st.button("📡 Varrer Mercado (24h)", type="primary", width='stretch', key="btn_radar_varrer"):
        st.session_state.agregado_bil = []
        agora_brt = get_brt_time()
        start_time, end_time = radar_window_brt(agora_brt)
        datas_str = [day.strftime("%d/%m/%Y")
                     for day in schedule_dates_for_brt_window(start_time, end_time)]
        odds_payloads = []
        with st.status("A extrair ativos globais...", expanded=True) as status:
            for d_str in datas_str:
                res_odds = safe_api_get(matches_odds_date_url(RAPIDAPI_HOST, d_str), timeout=30)
                if res_odds:
                    odds_payloads.append(res_odds)
                    status.write(f"{d_str}: lote de odds atuais AllSports carregado.")
                else:
                    status.write(f"⚠️ {d_str}: odds atuais indisponíveis; data não será qualificada.")

            agenda_meta = None
            phase4_candidates = []
            phase4_scored = 0
            phase4_failed = 0
            try:
                def carregar_fallback_allsports(competicoes):
                    eventos, metadados = [], []
                    for data_texto in datas_str:
                        parcial = fetch_competition_events_for_date(
                            RAPIDAPI_HOST, data_texto, competicoes, safe_api_get,
                        )
                        eventos.extend(parcial.get("events") or [])
                        metadados.append(parcial.get("_meta") or {})
                    return {
                        "events": eventos,
                        "_meta": {
                            "dates": metadados,
                            "estimated_http_requests": sum(
                                int(item.get("estimated_http_requests", 0) or 0)
                                for item in metadados
                            ),
                            "tournaments_selected": sum(
                                int(item.get("tournaments_selected", 0) or 0)
                                for item in metadados
                            ),
                            "tournament_failures": sum(
                                int(item.get("tournament_failures", 0) or 0)
                                for item in metadados
                            ),
                        },
                    }

                finais, agenda_meta = collect_soccer_radar_games(
                    DB_NAME, start_time, end_time, odds_payloads,
                    min_home_odd=1.99, min_away_odd=1.99,
                    progress=lambda mensagem: status.write(f"⚠️ Agenda Soccer: {mensagem}"),
                    allsports_fallback_loader=carregar_fallback_allsports,
                )
                phase4_candidates = list(agenda_meta.get('research_candidates') or [])
                def score_phase4_candidate(candidate):
                    return prever_com_ml(
                        candidate['ID'], candidate.get('Home_ID'), candidate.get('Away_ID'),
                        candidate.get('Tournament_ID'), candidate.get('Season_ID'),
                        candidate['Odd Casa'], candidate['Empate'], candidate['Odd Fora'],
                        candidate['Liga'], candidate.get('Unique_Tournament_ID', ''),
                        home_team=candidate.get('Time Casa'), away_team=candidate.get('Time Fora'),
                        start_timestamp=candidate.get('Timestamp'),
                        research_run_id=candidate.get('Research_Run_ID'), research_only=True,
                        eligibility_group=candidate.get('Eligibility_Group'),
                    )
                if phase4_candidates:
                    with ThreadPoolExecutor(max_workers=min(3, len(phase4_candidates))) as pool:
                        futures = [pool.submit(score_phase4_candidate, item)
                                   for item in phase4_candidates]
                        for future in as_completed(futures):
                            try:
                                phase4_scored += int(future.result() is not None)
                            except Exception as phase4_exc:
                                phase4_failed += 1
                                logger.warning("Phase4 prefilter shadow failed: %s", phase4_exc)
                agenda_meta['phase4_prefilter_scored'] = phase4_scored
                agenda_meta['phase4_prefilter_failed'] = phase4_failed
                from phase4_research import finalize_shadow_health
                agenda_meta['phase4_shadow_health'] = finalize_shadow_health(
                    DB_NAME, agenda_meta.get('research_run_id'),
                    candidates=len(phase4_candidates), scored=phase4_scored,
                    failed=phase4_failed,
                )
                status.write(
                    f"Agenda Soccer: {agenda_meta.get('events_in_window', 0)} jogos na janela; "
                    f"{agenda_meta.get('http_requests', 0)} requisições novas e "
                    f"{agenda_meta.get('cache_hits', 0)} páginas do cache."
                )
                status.write(
                    f"Relação de odds: {agenda_meta.get('linked_to_current_odds', 0)} jogo(s) "
                    f"({agenda_meta.get('events_with_bet365_fid', 0)} evento(s) com fid, "
                    f"{agenda_meta.get('fallback_linked', 0)} pelo fallback); "
                    f"{agenda_meta.get('qualified_games', 0)} passaram pelo filtro >1,99."
                )
                status.write(
                    f"AllSports: {len(odds_payloads)}/{len(datas_str)} lote(s) de odds; "
                    "fallback seletivo ativado para campeonatos sem fid."
                )
                status.write(
                    "Phase 4 pré-filtro: "
                    f"{phase4_scored}/{len(phase4_candidates)} pontuado(s) pelo campeão em shadow; "
                    f"{agenda_meta.get('phase4_eligible', 0)} eligible, "
                    f"{agenda_meta.get('phase4_non_eligible', 0)} non-eligible; "
                    f"{phase4_failed} falha(s)."
                )
                fallback_meta = agenda_meta.get("fallback_meta") or {}
                if fallback_meta:
                    status.write(
                        "Fallback sem fid: "
                        f"{agenda_meta.get('fallback_linked', 0)} jogo(s) relacionado(s); "
                        f"{fallback_meta.get('tournaments_selected', 0)} torneio(s); "
                        f"~{fallback_meta.get('estimated_http_requests', 0)} requisição(ões); "
                        f"{fallback_meta.get('tournament_failures', 0)} falha(s)/cota."
                    )
                if agenda_meta.get('pages_missing'):
                    status.write(
                        f"⚠️ Faltaram {agenda_meta['pages_missing']} página(s) da agenda Soccer; "
                        "os jogos já coletados serão mantidos."
                    )
            except Exception as exc:
                if isinstance(agenda_meta, dict) and agenda_meta.get('research_run_id'):
                    try:
                        from phase2_observations import append as append_research_error
                        from phase4_research import finalize_shadow_health
                        append_research_error(
                            DB_NAME, agenda_meta['research_run_id'], '',
                            'phase4_pipeline_error_v1', {
                                'stage': 'prefilter_shadow_or_agenda_handoff',
                                'error_type': type(exc).__name__,
                                'affects_production': True,
                                'run_completed': False,
                            })
                        finalize_shadow_health(
                            DB_NAME, agenda_meta['research_run_id'],
                            candidates=len(phase4_candidates), scored=phase4_scored,
                            failed=max(phase4_failed, len(phase4_candidates) - phase4_scored),
                        )
                    except Exception:
                        logger.exception("Falha ao registrar erro estrutural da Phase 4")
                logger.error("Falha ao montar radar pela Soccer Football Info: %s", exc)
                finais = []
            if finais:
                conn = get_db_connection()
                cursor = conn.cursor()
                cursor.execute("SELECT match_id FROM previsoes")
                ids_salvos = [str(row[0]) for row in cursor.fetchall()]
                conn.close()
                research_before_existing = list(finais)
                finais = [j for j in finais if str(j["ID"]) not in ids_salvos]
                from phase2_observations import observe_filter
                observe_filter(DB_NAME, research_before_existing, finais, 'already_recorded_filter')

                finais_filtrados = []
                for j in finais:
                    texto = f"{j['Liga']} {j['Time Casa']} {j['Time Fora']}".lower()
                    if any(termo in texto for termo in BLACKLIST_TERMS):
                        continue
                    finais_filtrados.append(j)

                observe_filter(DB_NAME, finais, finais_filtrados, 'blacklist_filter')

                sofa_meta = capture_pregame_contexts(
                    DB_NAME, finais_filtrados,
                    progress=lambda mensagem: status.write(mensagem),
                    recent_form_fallback=lambda game, side, provider_team_id: (
                        fetch_recent_team_events_for_game(
                            RAPIDAPI_HOST, game, side, safe_api_get, provider_team_id
                        )
                    ),
                    pregame_payload_fallback=lambda game, need_form, need_streaks: (
                        fetch_allsports_pregame_context(
                            RAPIDAPI_HOST, game, safe_api_get,
                            need_form=(need_form and not bool(game.get('Soccer_Context_Matched'))),
                            need_streaks=need_streaks,
                            include_goal_distributions=False,
                        )
                    ),
                    season_context_fallback=lambda game: fetch_allsports_pregame_context(
                        RAPIDAPI_HOST, game, safe_api_get, need_form=False,
                        need_streaks=False, include_goal_distributions=True,
                    ),
                )
                free_meta = enrich_free_statistics(DB_NAME, finais_filtrados)
                status.write(
                    "Fonte gratuita (dados em sombra, sem trocar o campeão): "
                    f"{free_meta.get('linked_games', 0)}/{len(finais_filtrados)} vinculados; "
                    f"{free_meta.get('requests', 0)} requisição(ões)."
                )
                metric_meta = enrich_recent_metrics_safely(
                    DB_NAME, finais_filtrados,
                    fallback=lambda mid, resources: fetch_allsports_postmatch_resources(
                        RAPIDAPI_HOST, mid, resources, safe_api_get),
                )
                status.write(
                    "Estatísticas de jogos anteriores (até 40 consultas por radar por padrão): "
                    f"{metric_meta.get('lookups', 0)} consultado(s), "
                    f"{metric_meta.get('profile_cache_hits', 0)} no cache, "
                    f"{metric_meta.get('measured_matches', 0)} com estatísticas, "
                    f"{metric_meta.get('xg_matches', 0)} com xG; "
                    f"{metric_meta.get('allsports_http', 0)} requisição(ões) AllSports."
                )
                status.write(
                    f"SofaScore pré-jogo: {sofa_meta.get('available', 0)}/"
                    f"{len(finais_filtrados)} com forma; "
                    f"{sofa_meta.get('http_requests', 0)} requisições novas e "
                    f"{sofa_meta.get('cache_hits', 0) + sofa_meta.get('context_cache_hits', 0)} cache hits."
                )
                status.write(
                    "Forma viva (últimos 5): "
                    f"{sofa_meta.get('live_available', 0)}/{len(finais_filtrados)} com os dois times; "
                    f"{sofa_meta.get('live_provider_sofascore', 0)} lado(s) via SofaScore, "
                    f"{sofa_meta.get('live_provider_allsports', 0)} via AllSports; "
                    f"{sofa_meta.get('live_fallback_calls', 0)} fallback(s)."
                )
                status.write(
                    "Contexto alternativo AllSports: "
                    f"{sofa_meta.get('allsports_form_fallback', 0)} forma(s), "
                    f"{sofa_meta.get('allsports_streaks_fallback', 0)} streak(s), "
                    f"{sofa_meta.get('allsports_pregame_http', 0)} requisição(ões), "
                    f"{sofa_meta.get('allsports_pregame_cache_hits', 0)} cache hit(s)."
                )
                status.write(
                    "Distribuição sazonal AllSports: "
                    f"{sofa_meta.get('allsports_season_http', 0)} requisição(ões), "
                    f"{sofa_meta.get('allsports_season_cache_hits', 0)} cache hit(s)."
                )

                st.session_state.resultados_v4 = finais_filtrados
                status.update(label=f"Concluído. {len(finais_filtrados)} jogos na janela.", state="complete")
                time.sleep(1)
                st.rerun()
            else:
                st.session_state.resultados_v4 = []
                status.update(label="Falha ou filtro vazio.", state="error")

    if st.session_state.get("resultados_v4"):
        res_list = st.session_state.resultados_v4
        st.success(f"✅ **{len(res_list)} jogos qualificados**")
        df_view = pd.DataFrame(res_list)
        st.dataframe(
            df_view.drop(columns=["ID","Home_ID","Away_ID","Tournament_ID","Season_ID","Timestamp","Confronto"], errors="ignore"),
            width='stretch',
            hide_index=True
        )

        if st.button("🔮 Iniciar Análise Quantitativa V5 (Python)", type="primary", width='stretch', key="btn_analisar"):
            st.session_state.agregado_bil = []
            conn = get_db_connection()
            df_macros = pd.read_sql_query("SELECT liga, relatorio_geral FROM autopsias_liga", conn)
            dict_macros = dict(zip(df_macros["liga"], df_macros["relatorio_geral"]))
            conn.close()

            agregado_ind = []
            limiar_ml_ativo = (0.0 if PUBLICAR_TODAS_PREVISOES
                               else get_active_confidence(DB_NAME, MIN_ML_CONFIDENCE))
            ml_model_version, ml_model_id = get_active_model_identity(DB_NAME)
            with st.status("Analisando jogos com ML...", expanded=True) as analise_status:
                if PUBLICAR_TODAS_PREVISOES:
                    st.write("Política ativa: confiança informativa; todas as previsões entram na montagem.")
                else:
                    st.write(f"Política individual ativa: confiança mínima de {limiar_ml_ativo:.0f}%")
                for j in res_list:
                    texto = f"{j['Liga']} {j['Time Casa']} {j['Time Fora']}".lower()
                    if any(termo in texto for termo in BLACKLIST_TERMS):
                        continue

                    resultado_pred = calcular_edge_super_python_v5(
                        "", "", "",
                        "", "", "",
                        ("Stats Liga N/A", False), ("Stats Liga N/A", False),
                        j["Odd Casa"], j["Odd Fora"], j["Empate"],
                        "", "", "", "",
                        "", "", "", "",
                        dict_macros.get(j["Liga"], ""),
                        "", "",
                        (1.0, 1.0, False), (1.0, 1.0, False),
                        j["Liga"], match_id=j["ID"], home_id=j["Home_ID"], away_id=j["Away_ID"],
                        tournament_id=j["Tournament_ID"], season_id=j["Season_ID"],
                        home_team=j.get("Time Casa"), away_team=j.get("Time Fora"),
                        start_timestamp=j.get("Timestamp"),
                        unique_tournament_id=j.get("Unique_Tournament_ID", ""),
                        research_run_id=j.get("Research_Run_ID"),
                    )
                    if resultado_pred[0] is None:
                        continue

                    vencedor_pick, conf_calculada, edge_text, probs = resultado_pred
                    snapshot_meta = load_prediction_snapshot_metadata(DB_NAME, j["ID"])
                    if vencedor_pick == "MANDANTE":
                        odd_pick = j["Odd Casa"]
                        vencedor_oficial = j["Time Casa"]
                    elif vencedor_pick == "VISITANTE":
                        odd_pick = j["Odd Fora"]
                        vencedor_oficial = j["Time Fora"]
                    else:
                        odd_pick = j["Empate"]
                        vencedor_oficial = "Empate"

                    agregado_ind.append({
                        "ID_Jogo": j["ID"],
                        "Confronto": j["Confronto"],
                        "Liga": j["Liga"],
                        "Vencedor Escolhido": vencedor_oficial,
                        "Confiança": conf_calculada,
                        "Relatório Técnico": edge_text,
                        "Dia_Str": j["Dia"],
                        "Hora BRT": j["Hora"],
                        "Liga_Exata": j["Liga"],
                        "Odd Casa": j["Odd Casa"],
                        "Odd Fora": j["Odd Fora"],
                        "Empate": j["Empate"],
                        "ID": j["ID"],
                        "Risco_Empate": float(snapshot_meta.get("draw_risk", 0.0)),
                        "Qualidade_Contexto": float(snapshot_meta.get("information_quality", 0.0)),
                        "Conflito_Contexto": float(snapshot_meta.get("selection_conflict", 0.0)),
                        "Divergencia_Fontes": float(snapshot_meta.get("source_disagreement", 0.0)),
                        "Confiabilidade_Amostra": float(snapshot_meta.get("sample_reliability", 0.0)),
                        "Margem_Probabilidade": float(snapshot_meta.get("probability_margin", 0.0)),
                        "Entropia_Normalizada": float(snapshot_meta.get("normalized_entropy", 1.0)),
                        "Contexto_Detalhado_Ambos": float(snapshot_meta.get("detailed_both_available", 0.0)),
                        "Forma_Viva_Ambos": float(snapshot_meta.get("live_recent_both_available", 0.0)),
                        "Jogos_Recentes_Min": float(snapshot_meta.get("recent_games_min", 0.0)),
                        "Volatilidade_Competicao": float(snapshot_meta.get("competition_volatility", 0.08)),
                        "Corroboracao_Empate": int(snapshot_meta.get("draw_corroboration", 0)),
                        "Analysis_Version": snapshot_meta.get("source_version", ""),
                        "Odd_Calculada": odd_pick,
                        "Pick": vencedor_pick,
                        "Tournament_ID": j.get("Tournament_ID", ""),
                        "Season_ID": j.get("Season_ID", ""),
                        "Unique_Tournament_ID": j.get("Unique_Tournament_ID", ""),
                        "Timestamp": int(j.get("Timestamp", 0) or 0),
                    })

            analise_status.update(label="Análise concluída. Montando bilhetes...", state="running")

            jogos_unicos = deduplicate_predictions(agregado_ind)
            jogos_qualificados = [
                jogo for jogo in jogos_unicos
                if float(jogo.get("Confiança", 0)) >= limiar_ml_ativo
                and (PUBLICAR_TODAS_PREVISOES or PERMITIR_EMPATES_BILHETE or jogo.get("Pick") != "EMPATE")
                and (PUBLICAR_TODAS_PREVISOES or jogo.get("Pick") != "EMPATE"
                     or float(jogo.get("Confiança", 0)) >= MIN_DRAW_CONFIDENCE)
            ]
            jogos_ordenados = sorted(jogos_qualificados, key=lambda x: -x["Confiança"])
            st.session_state.selecoes_individuais = jogos_ordenados

            radar_run_id = get_brt_time().strftime("%Y%m%d_%H%M%S")
            run_id = radar_run_id.rsplit("_", 1)[-1]
            agregado_bil = selecionar_grupos_bilhetes(
                jogos_ordenados, min_confianca=limiar_ml_ativo,
                max_bilhetes=MAX_TICKETS_PER_RUN,
                permitir_empate=(PUBLICAR_TODAS_PREVISOES or PERMITIR_EMPATES_BILHETE), ligas_unicas=True,
                relaxar_ligas=True)
            for idx, bilhete in enumerate(agregado_bil, 1):
                bilhete["Nome"] = f"🛡️ Ticket Quant V5 {bilhete['Categoria']} #{idx} [#{run_id}]"
            from phase2_observations import finish_run
            finish_run(DB_NAME, st.session_state.resultados_v4, jogos_unicos, jogos_qualificados,
                       agregado_bil, radar_run_id, ml_model_version, ml_model_id)

            st.session_state.agregado_bil = agregado_bil

            with db_write_lock:
                conn = get_db_connection()
                cursor = conn.cursor()
                timestamp_atual = get_brt_time().strftime("%Y-%m-%d %H:%M:%S")
                ids_qualificados = {str(j["ID"]) for j in jogos_qualificados}
                for j in jogos_unicos:
                    try:
                        cursor.execute("""INSERT OR IGNORE INTO previsoes
                            (match_id, timestamp, confronto, liga, odd_casa, odd_fora, odd_empate,
                             vencedor_previsto, confianca, scout_report, data_jogo, hora_jogo,
                             telegram_enviado, selecionado_radar, tournament_id, season_id,
                             unique_tournament_id, ml_model_version, ml_model_id, start_timestamp,
                             draw_risk_score, context_quality_score, context_conflict_score, analysis_version,
                             radar_run_id)
                            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                            (j["ID"], timestamp_atual, j.get("Confronto"), j.get("Liga_Exata"),
                             j.get("Odd Casa", 0.0), j.get("Odd Fora", 0.0), j.get("Empate", 0.0),
                             j.get("Vencedor Escolhido"), j.get("Confiança", 0),
                             j.get("Relatório Técnico", ""), j.get("Dia_Str", ""), j.get("Hora BRT", ""),
                             1, int(str(j["ID"]) in ids_qualificados), j.get("Tournament_ID"),
                              j.get("Season_ID"), j.get("Unique_Tournament_ID"),
                              ml_model_version, ml_model_id, int(j.get("Timestamp", 0) or 0),
                              j.get("Risco_Empate", 0.0), j.get("Qualidade_Contexto", 0.0),
                              j.get("Conflito_Contexto", 0.0), j.get("Analysis_Version", ""),
                              radar_run_id))
                    except Exception as e:
                        st.error(f"Erro ao registrar previsão de aprendizado: {e}")
                for b in agregado_bil:
                    for j in b["Jogos"]:
                        try:
                            cursor.execute("""UPDATE previsoes SET ticket_id=?, data_jogo=?, hora_jogo=?,
                                confianca=?, telegram_enviado=0, selecionado_radar=1 WHERE match_id=?""",
                                           (b["Nome"], j.get("Dia_Str", ""), j.get("Hora BRT", ""), j.get("Confiança", 0), j["ID"]))
                        except Exception as e:
                            st.error(f"Erro ao salvar: {e}")
                conn.commit()
                conn.close()

            analise_status.update(
                label=(f"Concluído: {len(jogos_ordenados)} seleções individuais, "
                       f"{len(agregado_bil)} bilhetes 4/4."), state="complete")
            st.rerun()

    if st.session_state.get("selecoes_individuais"):
        selecoes = st.session_state.selecoes_individuais
        st.info(f"🎯 {len(selecoes)} previsões disponíveis para grupos completos de quatro. "
                "A confiança é exibida e usada para ordenar, mas não bloqueia a publicação.")
        with st.expander("Ver previsões consideradas"):
            st.dataframe(pd.DataFrame([{
                "Jogo": j.get("Confronto"), "Liga": j.get("Liga_Exata"),
                "Pick": j.get("Vencedor Escolhido"), "Confiança": j.get("Confiança"),
                "Odd": j.get("Odd_Calculada"),
            } for j in selecoes]), hide_index=True, width='stretch')

    if st.session_state.get("agregado_bil"):
        st.success(f"✅ **{len(st.session_state.agregado_bil)} bilhetes montados**")
        st.markdown("---")
        if st.button("📲 Disparar Bilhetes para o Telegram", type="secondary", width='stretch', key="btn_disparar_telegram"):
            sucessos = disparar_telegram(st.session_state.agregado_bil, TELEGRAM_TOKEN, TELEGRAM_CHAT_ID)
            st.success(f"✅ {sucessos} bilhetes enviados!")

        cols = st.columns(2)
        for idx, b in enumerate(st.session_state.agregado_bil):
            with cols[idx % 2]:
                odd_multipla = calcular_odd_multipla(b["Jogos"])
                st.markdown(f'<div class="match-card"><h4>{b["Nome"]} <span style="color:#facc15; font-size:0.9rem;">🔥 Odd: {odd_multipla:.2f}</span></h4>', unsafe_allow_html=True)
                for j in b["Jogos"]:
                    pick = j.get("Vencedor Escolhido", "").upper()
                    conf = j.get("Confiança", 0)
                    cor_confianca = "#22c55e" if int(conf) >= 80 else "#eab308" if int(conf) >= 60 else "#ef4444"
                    st.markdown(
                        f"<div style='background:#1a1d24; padding:12px; border-radius:12px; margin-top:8px; border:1px solid #2f333d;'>"
                        f"<div style='font-size:0.7rem; color:#8b8f9c; margin-bottom:4px; display:flex; justify-content:space-between;'>"
                        f"<span>🌍 {j.get('Liga_Exata','')} • ⏰ {j.get('Dia_Str','')} {j.get('Hora BRT','')} BRT</span>"
                        f"<span style='color:{cor_confianca}; font-weight:700;'>Conf: {conf}%</span></div>"
                        f"<div style='display:flex; align-items:center; justify-content:space-between;'>"
                        f"<span style='font-weight:600; font-size:0.95rem; color:#f8fafc;'>{j['Confronto']}</span></div>"
                        f"<div style='margin-top:8px; display:flex; justify-content:space-between; align-items:baseline;'>"
                        f"<div style='font-size:0.75rem; color:#8b8f9c; font-weight:500;'>Odd: {j.get('Odd_Calculada',1.0):.2f}</div>"
                        f"<div style='color:#facc15; font-weight:800; font-size:1.05rem;'>{pick}</div></div>"
                        f"<div style='font-size:0.75rem; color:#8b8f9c; margin-top:8px; border-top:1px dashed #2f333d; padding-top:6px; font-style:italic;'>💡 {j.get('Relatório Técnico','')}</div>"
                        f"</div>",
                        unsafe_allow_html=True
                    )
                st.markdown("</div>", unsafe_allow_html=True)

# ---------- ABA 2: LOGS ----------
with tab2:
    st.markdown("### 🗄️ Histórico e Logs")
    with st.expander("⚠️ Danger Zone (Zerar Tudo)"):
        if st.button("🗑️ ZERAR BANCO DE DADOS COMPLETO", type="secondary", key="btn_zerar_banco"):
            with get_db_connection() as conn:
                for t in ["previsoes","autopsias_liga","aprendizado_global","aprendizado_liga","aprendizado_time","cache_xg_times","cache_jogos_liga","mapeamento_ligas","drift_reference","api_error_log","estatisticas_medias_liga","training_data","modelos_ml","training_weights","ml_evolution_runs","ml_selection_policy","ml_live_monitoring"]:
                    conn.execute(f"DROP TABLE IF EXISTS {t}")
            init_db(); init_error_log_table()
            st.session_state.resultados_v4 = []; st.session_state.agregado_bil = []; st.session_state.selecoes_individuais = []
            if "api_errors" in st.session_state: st.session_state.api_errors = []
            st.success("Reset concluído!"); time.sleep(1); st.rerun()
    conn = get_db_connection()
    try: st.dataframe(pd.read_sql_query("SELECT timestamp, liga, confronto, vencedor_previsto, confianca, status_resultado, ticket_id FROM previsoes ORDER BY id DESC", conn), hide_index=True, width='stretch')
    except: pass
    conn.close()

# ---------- ABA 3: AUDITORIA ----------
with tab3:
    st.markdown("### ⚖️ Auditoria de Lucratividade")
    if st.button("🔍 Sincronizar Placar Real e Validar Pagamento Antecipado", type="primary", key="btn_sincronizar"):
        conn = get_db_connection(); cursor = conn.cursor()
        radar_run_ids = get_latest_radar_run_ids(conn, limit=2)
        if radar_run_ids:
            run_slots = radar_run_placeholders(radar_run_ids)
            cursor.execute(
                f"""SELECT match_id, vencedor_previsto, ticket_id FROM previsoes
                    WHERE status_resultado='PENDENTE' AND antecipado_detectado=0
                      AND radar_run_id IN ({run_slots})
                      AND COALESCE(start_timestamp,0)>0 AND start_timestamp<=?""",
                (*radar_run_ids, int(get_brt_time().timestamp()) - 7200),
            )
            pendentes = cursor.fetchall()
            st.info("Auditando os dois radares mais recentes: " + ", ".join(radar_run_ids))
        else:
            pendentes = []
            st.info("Nenhum lote de radar registrado para auditoria.")
        tickets_afetados = set(); finalizados_para_autopsia = []
        eventos_coletados = []
        with st.status("A verificar resultados oficiais...", expanded=True):
            # A coleta/cache usa conexões próprias. Ela precisa terminar antes
            # da transação que grava os placares para não bloquear o próprio DB.
            for pending_match_id, pending_previsto, pending_ticket_id in pendentes:
                try:
                    ev_sofa = get_event_for_audit(DB_NAME, pending_match_id)
                except Exception as exc:
                    logger.warning("Consulta SofaScore falhou no app para %s: %s",
                                   pending_match_id, exc)
                    ev_sofa = None
                if isinstance(ev_sofa, dict):
                    evento = ev_sofa
                else:
                    try:
                        res = safe_api_get(match_detail_url(RAPIDAPI_HOST, pending_match_id)) or {}
                    except Exception as exc:
                        logger.warning("Fallback RapidAPI falhou no app para %s: %s",
                                       pending_match_id, exc)
                        res = {}
                    evento = res.get("event")
                eventos_coletados.append(
                    (str(pending_match_id), pending_previsto, pending_ticket_id, evento)
                )

            with db_write_lock:
                for pending_match_id, pending_previsto, pending_ticket_id, ev in eventos_coletados:
                    try:
                        prediction_identity = cursor.execute(
                            "SELECT confronto,start_timestamp FROM previsoes WHERE match_id=?",
                            (str(pending_match_id),),
                        ).fetchone()
                        if (not prediction_identity or not event_matches_prediction(
                            ev, pending_match_id, prediction_identity[0], prediction_identity[1]
                        )):
                            continue
                        if ev.get("status", {}).get("type") != "finished":
                            continue
                        score = regulation_score(ev)
                        if score is None:
                            continue
                        h, a = score
                        previsto = pending_previsto
                        snapshot = cursor.execute(
                            "SELECT predicted_outcome FROM ml_prediction_snapshots WHERE match_id=?",
                            (str(pending_match_id),),
                        ).fetchone()
                        if ev.get('id') is not None and str(ev['id']) != str(pending_match_id):
                            continue
                        lado = resolve_pick_side(snapshot[0] if snapshot else None, previsto,
                            (ev.get('homeTeam') or {}).get('name'), (ev.get('awayTeam') or {}).get('name'))
                        if lado not in {"MANDANTE", "EMPATE", "VISITANTE"}:
                            continue
                        if h > a and lado == "MANDANTE": win = "GREEN ✅"
                        elif a > h and lado == "VISITANTE": win = "GREEN ✅"
                        elif h == a and lado == "EMPATE": win = "GREEN ✅"
                        else: win = "RED ❌"
                        cursor.execute("UPDATE previsoes SET status_resultado = ?, placar_real = ? WHERE match_id = ?", (win, f"{h}-{a}", pending_match_id))
                        finalizados_para_autopsia.append((pending_match_id, ev, previsto, win))
                        if pending_ticket_id: tickets_afetados.add(pending_ticket_id)
                    except Exception as exc:
                        logger.warning("Falha ao gravar auditoria no app para %s: %s",
                                       pending_match_id, exc)
                conn.commit()
        for m_id, ev, previsto, win in finalizados_para_autopsia:
            try:
                transferir_jogo_para_treinamento(m_id, ev)
                audit_match_postmortem(
                    DB_NAME, m_id, ev, previsto, win,
                    postmatch_payload_fallback=lambda mid, resources: (
                        fetch_allsports_postmatch_resources(
                            RAPIDAPI_HOST, mid, resources, safe_api_get,
                        )
                    ),
                )
            except Exception as exc:
                logger.warning("Autópsia SofaScore falhou no app para %s: %s", m_id, exc)
        if radar_run_ids:
            ids_escopo = [row[0] for row in cursor.execute(
                f"SELECT DISTINCT match_id FROM previsoes WHERE radar_run_id IN ({run_slots})",
                radar_run_ids,
            ).fetchall()]
            try:
                reclassify_stored_postmortems(DB_NAME, ids_escopo)
            except Exception as exc:
                logger.warning("Reclassificação local das autópsias falhou: %s", exc)
        try:
            monitor_pregame_context_sources(DB_NAME)
        except Exception as exc:
            logger.warning("Monitoramento das fontes pré-jogo falhou: %s", exc)
        for tid in tickets_afetados:
            atualizar_mensagem_telegram_por_bilhete(tid)
            verificar_e_celebrar_green(tid)
        conn.close()
        with db_write_lock:
            with get_db_connection() as ratings_conn:
                ratings = current_elo_ratings(ratings_conn)
                ratings_timestamp = get_brt_time().strftime("%Y-%m-%d %H:%M:%S")
                ratings_conn.executemany(
                    """INSERT OR REPLACE INTO ml_team_ratings
                       (team_name,elo,updated_at) VALUES (?,?,?)""",
                    [(team, float(elo), ratings_timestamp)
                     for team, elo in ratings.items()],
                )
                ratings_conn.commit()
        st.rerun()
    conn = get_db_connection()
    df_perf = pd.read_sql_query("SELECT status_resultado, ticket_id FROM previsoes WHERE status_resultado IN ('GREEN ✅', 'RED ❌') OR status_resultado LIKE 'GREEN ✅ (Antecipado)'", conn)
    if not df_perf.empty:
        greens_j = len(df_perf[df_perf["status_resultado"].str.contains("GREEN")])
        total_j = len(df_perf)
        wr_j = (greens_j / total_j) * 100
        st.metric("🎯 Win Rate (Jogo a Jogo)", f"{wr_j:.1f}%", f"{greens_j} Greens em {total_j} Jogos")
    try:
        df_postmortems = pd.read_sql_query(
            """SELECT p.confronto AS Jogo, p.status_resultado AS Resultado,
                      m.verdict AS Diagnostico,
                      ROUND(m.chosen_dominance * 100, 1) AS Dominio_da_selecao_pct,
                      ROUND(m.data_coverage * 100, 1) AS Cobertura_pct,
                      m.scoreline AS Placar
               FROM match_postmortems m JOIN previsoes p ON p.match_id=m.match_id
               ORDER BY m.audited_at DESC LIMIT 100""",
            conn,
        )
        if not df_postmortems.empty:
            st.markdown("#### 🧠 Diagnóstico de processo (SofaScore)")
            st.dataframe(df_postmortems, hide_index=True, width='stretch')
    except sqlite3.Error:
        pass
    conn.close()

# ---------- ABA 4: EM ABERTO ----------
with tab4:
    st.markdown("### 🎫 Bilhetes em Aberto")
    conn = get_db_connection()
    df_all_tickets = pd.read_sql_query(
        "SELECT id, match_id, confronto, liga, vencedor_previsto, odd_casa, odd_fora, odd_empate, "
        "confianca, status_resultado, ticket_id, data_jogo, hora_jogo, antecipado_detectado, anulado "
        "FROM previsoes WHERE ticket_id IS NOT NULL AND ticket_id != '' ORDER BY timestamp DESC",
        conn
    )
    conn.close()

    bilhetes_abertos = []
    for t_id, group in df_all_tickets.groupby("ticket_id"):
        statuses = group["status_resultado"].str.upper().tolist()
        if any("RED" in s for s in statuses): continue
        if not any("PENDENTE" in s for s in statuses): continue
        jogos_nao_anulados = group[group["anulado"] != 1]
        if len(jogos_nao_anulados) > 0:
            bilhetes_abertos.append((t_id, group))

    if not bilhetes_abertos:
        st.success("Não há bilhetes pendentes válidos.")
    else:
        cols = st.columns(2)
        for idx, (t_id, group) in enumerate(bilhetes_abertos):
            with cols[idx % 2]:
                jogos_list = []
                for _, row in group.iterrows():
                    if row["anulado"] == 1: continue
                    jogos_list.append({
                        "Confronto": row["confronto"],
                        "Vencedor Escolhido": row["vencedor_previsto"],
                        "Odd Casa": row["odd_casa"],
                        "Odd Fora": row["odd_fora"],
                        "Empate": row["odd_empate"],
                    })
                odd_multipla = calcular_odd_multipla(jogos_list)
                st.markdown(f"<div class='match-card' style='border-left: 4px solid #60a5fa;'><h4>⏳ {t_id} <br><span style='color:#60a5fa; font-size:0.9rem;'>Odd Múltipla: {odd_multipla:.2f}</span></h4>", unsafe_allow_html=True)

                for _, row in group.iterrows():
                    if row["anulado"] == 1: continue
                    pick = str(row["vencedor_previsto"]).upper()
                    casa_nome = str(row["confronto"]).split(" vs ")[0].strip().lower()
                    pick_lower = pick.lower()
                    if "empate" in pick_lower: odd_jogo = float(row["odd_empate"])
                    elif casa_nome in pick_lower: odd_jogo = float(row["odd_casa"])
                    else: odd_jogo = float(row["odd_fora"])
                    if odd_jogo <= 1.0: odd_jogo = max(float(row["odd_casa"]), float(row["odd_fora"]))

                    d_j = row["data_jogo"] if row["data_jogo"] else "--/--"
                    h_j = row["hora_jogo"] if row["hora_jogo"] else "--:--"
                    conf = row["confianca"] if row["confianca"] else "N/A"
                    status_jogo = str(row.get("status_resultado", "PENDENTE")).upper()
                    antecipado = row.get("antecipado_detectado", 0)

                    if "GREEN" in status_jogo: badge = "✅ GREEN"; cor = "#22c55e"
                    elif "RED" in status_jogo: badge = "❌ RED"; cor = "#ef4444"
                    elif status_jogo == "PENDENTE" and antecipado == 1: badge = "⚠️ ANTECIPADO"; cor = "#f59e0b"
                    else: badge = "⏳ PENDENTE"; cor = "#60a5fa"

                    st.markdown(f"""
                        <div style='background:#1a1d24; padding:12px; border-radius:12px; margin-top:8px; border:1px solid #2f333d;'>
                            <div style='font-size:0.7rem; color:#8b8f9c; margin-bottom:4px; display:flex; justify-content:space-between;'>
                                <span>🌍 {row['liga']} • ⏰ {d_j} {h_j} BRT</span>
                                <span style='color:{cor}; font-weight:700;'>Conf: {conf}%</span>
                            </div>
                            <div style='display:flex; align-items:center; justify-content:space-between;'>
                                <span style='font-weight:600; font-size:0.95rem; color:#f8fafc;'>{row['confronto']}</span>
                                <span style='background:#1e3a8a; color:{cor}; padding:2px 8px; border-radius:6px; font-size:0.7rem; font-weight:800;'>{badge}</span>
                            </div>
                            <div style='margin-top:8px; display:flex; justify-content:space-between; align-items:baseline;'>
                                <div style='font-size:0.75rem; color:#8b8f9c; font-weight:500;'>Odd: {odd_jogo:.2f}</div>
                                <div style='color:{cor}; font-weight:800; font-size:1.05rem;'>{pick}</div>
                            </div>
                        </div>
                    """, unsafe_allow_html=True)

                    col1, col2 = st.columns(2)

                    if status_jogo == "PENDENTE" and antecipado == 1:
                        with col1:
                            if st.button(f"✅ Confirmar Antecipado", key=f"confirm_ant_{row['id']}"):
                                with db_write_lock:
                                    conn = get_db_connection()
                                    cursor = conn.cursor()
                                    cursor.execute("UPDATE previsoes SET status_resultado = ?, antecipado_detectado = 0 WHERE id = ?",
                                                   ("GREEN ✅ (Antecipado)", row["id"]))
                                    conn.commit(); conn.close()
                                atualizar_mensagem_telegram_por_bilhete(t_id)
                                verificar_e_celebrar_green(t_id)
                                st.success("Pagamento antecipado confirmado!")
                                time.sleep(1); st.rerun()
                            with col2:
                                if st.button(f"🔍 Processar Resultado Real", key=f"process_real_{row['id']}"):
                                    with db_write_lock:
                                        conn = get_db_connection(); cursor = conn.cursor()
                                    novo_status = None; placar = "-"
                                    try:
                                        dados = safe_api_get(
                                            match_detail_url(RAPIDAPI_HOST, row["match_id"])
                                        ) or {}
                                        ev = dados.get("event", {})
                                        score = regulation_score(ev)
                                        if score is not None:
                                            h, a = score
                                            placar = f"{h}-{a}"
                                            home_name = ev.get("homeTeam", {}).get("name", "").lower()
                                            away_name = ev.get("awayTeam", {}).get("name", "").lower()
                                            snapshot = cursor.execute(
                                                "SELECT predicted_outcome FROM ml_prediction_snapshots WHERE match_id=?",
                                                (str(row["match_id"]),),
                                            ).fetchone()
                                            side = resolve_pick_side(snapshot[0] if snapshot else None,
                                                row["vencedor_previsto"], home_name, away_name)
                                            novo_status = settlement_status(score, side)
                                    except Exception as e:
                                        st.error(f"Erro ao buscar resultado: {e}")
                                    if novo_status:
                                        cursor.execute("UPDATE previsoes SET status_resultado = ?, placar_real = ?, antecipado_detectado = 0 WHERE id = ?",
                                                       (novo_status, placar, row["id"]))
                                        conn.commit(); conn.close()
                                        atualizar_mensagem_telegram_por_bilhete(t_id)
                                        if "GREEN" in novo_status: verificar_e_celebrar_green(t_id)
                                        st.success(f"Resultado processado: {novo_status} ({placar})")
                                    else:
                                        conn.close()
                                        st.warning("Resultado ainda não disponível.")
                                time.sleep(1); st.rerun()

                    if status_jogo == "PENDENTE":
                        with col2:
                            if st.button(f"❌ Anular Jogo", key=f"anular_{row['id']}"):
                                with db_write_lock:
                                    conn = get_db_connection()
                                    conn.execute("UPDATE previsoes SET anulado = 1, status_resultado = 'ANULADO' WHERE id = ?", (row["id"],))
                                    conn.commit(); conn.close()
                                atualizar_mensagem_telegram_por_bilhete(t_id)
                                st.success("Jogo anulado. Odd do bilhete recalculada.")
                                time.sleep(1); st.rerun()

                    if row["anulado"] == 1:
                        with col1:
                            if st.button(f"↩️ Reativar Jogo", key=f"reativar_{row['id']}"):
                                with db_write_lock:
                                    conn = get_db_connection()
                                    conn.execute("UPDATE previsoes SET anulado = 0, status_resultado = 'PENDENTE' WHERE id = ?", (row["id"],))
                                    conn.commit(); conn.close()
                                atualizar_mensagem_telegram_por_bilhete(t_id)
                                st.success("Jogo reativado!")
                                time.sleep(1); st.rerun()
                st.markdown("</div>", unsafe_allow_html=True)

# ---------- ABA 5: GESTÃO P&L ----------
with tab5:
    st.markdown("### 🏦 Gestão de Banca & P&L")
    conn = get_db_connection()
    df_jogos = pd.read_sql_query("SELECT timestamp, ticket_id, confronto, vencedor_previsto, odd_casa, odd_fora, odd_empate, status_resultado, anulado FROM previsoes WHERE ticket_id IS NOT NULL AND ticket_id != ''", conn)
    conn.close()
    if df_jogos.empty: st.info("Nenhum bilhete processado ainda.")
    else:
        stake = 1.0
        dados_banca = []
        for t_id, group in df_jogos.groupby("ticket_id"):
            grupo_nao_anulado = group[group["anulado"] != 1]
            if grupo_nao_anulado.empty: continue
            statuses = grupo_nao_anulado["status_resultado"].str.upper().tolist()
            if any("PENDENTE" in s for s in statuses): status_bilhete = "PENDENTE"
            elif any("RED" in s for s in statuses): status_bilhete = "RED ❌"
            else: status_bilhete = "GREEN ✅"
            jogos_list = []
            for _, row in grupo_nao_anulado.iterrows():
                jogos_list.append({"Confronto": row["confronto"], "Vencedor Escolhido": row["vencedor_previsto"], "Odd Casa": row["odd_casa"], "Odd Fora": row["odd_fora"], "Empate": row["odd_empate"]})
            odd_multipla = calcular_odd_multipla(jogos_list)
            if status_bilhete == "GREEN ✅": lucro = (stake * odd_multipla) - stake; retorno = stake * odd_multipla
            elif status_bilhete == "RED ❌": lucro = -stake; retorno = 0.0
            else: lucro = 0.0; retorno = 0.0
            data_criacao = pd.to_datetime(group["timestamp"].iloc[0])
            dados_banca.append({"Data": data_criacao.strftime("%d/%m/%Y"), "Mês": data_criacao.strftime("%m/%Y"), "Ticket": t_id, "Odd Múltipla": round(odd_multipla,2), "Status": status_bilhete, "Investimento (R$)": stake if status_bilhete!="PENDENTE" else 0.0, "Retorno (R$)": retorno, "Lucro Líquido (R$)": lucro})
        df_banca = pd.DataFrame(dados_banca)
        meses_disp = df_banca["Mês"].unique().tolist()
        if meses_disp:
            opcoes_mes = ["Todos"] + sorted(meses_disp, key=lambda x: (x.split("/")[1], x.split("/")[0]), reverse=True)
            mes_selecionado = st.selectbox("📅 Filtrar por Mês", opcoes_mes, key="tab5_filtro_mes")
        else: mes_selecionado = "Todos"
        df_view = df_banca[df_banca["Mês"] == mes_selecionado] if mes_selecionado != "Todos" else df_banca
        df_resolvidos = df_view[df_view["Status"] != "PENDENTE"]
        t_inv = df_resolvidos["Investimento (R$)"].sum(); t_ret = df_resolvidos["Retorno (R$)"].sum(); l_liq = df_resolvidos["Lucro Líquido (R$)"].sum()
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("🎯 Bilhetes Finalizados", f"{len(df_resolvidos)}")
        col2.metric("💰 Total Investido", f"R$ {t_inv:,.2f}".replace(",","v").replace(".",",").replace("v","."))
        col3.metric("📈 Lucro Líquido", f"R$ {l_liq:,.2f}".replace(",","v").replace(".",",").replace("v","."), delta=f"{(l_liq / t_inv * 100):+.1f}% ROI" if t_inv>0 else None)
        col4.metric("💵 Retorno Bruto", f"R$ {t_ret:,.2f}".replace(",","v").replace(".",",").replace("v","."))
        st.markdown("#### 📆 Evolução Diária")
        df_finalizados = df_view[df_view["Status"] != "PENDENTE"].copy()
        if not df_finalizados.empty:
            df_finalizados["Data_dt"] = pd.to_datetime(df_finalizados["Data"], format="%d/%m/%Y")
            df_diario = df_finalizados.groupby("Data_dt").agg(
                quantidade_bilhetes=("Ticket","count"),
                valor_ganhos=("Retorno (R$)", lambda x: x[df_finalizados.loc[x.index,"Status"]=="GREEN ✅"].sum()),
                valor_perdidos=("Investimento (R$)", lambda x: x[df_finalizados.loc[x.index,"Status"]=="RED ❌"].sum()),
                valor_apostado=("Investimento (R$)","sum")).reset_index()
            df_diario[["valor_ganhos","valor_perdidos"]] = df_diario[["valor_ganhos","valor_perdidos"]].fillna(0)
            fig_daily = go.Figure()
            fig_daily.add_trace(go.Scatter(x=df_diario["Data_dt"], y=df_diario["quantidade_bilhetes"], mode="lines+markers", name="Quantidade", line=dict(color="#ADD8E6")))
            fig_daily.add_trace(go.Scatter(x=df_diario["Data_dt"], y=df_diario["valor_ganhos"], mode="lines+markers", name="Ganho", line=dict(color="#90EE90")))
            fig_daily.add_trace(go.Scatter(x=df_diario["Data_dt"], y=df_diario["valor_perdidos"], mode="lines+markers", name="Perda", line=dict(color="#FFC0C0")))
            fig_daily.add_trace(go.Scatter(x=df_diario["Data_dt"], y=df_diario["valor_apostado"], mode="lines+markers", name="Apostado", line=dict(color="#FFFFE0")))
            fig_daily.update_layout(title="Evolução Diária", template="plotly_dark")
            st.plotly_chart(fig_daily, width='stretch')
            st.markdown("#### 📊 Lucro Líquido por Mês")
            df_finalizados["Mês_Ano"] = df_finalizados["Data_dt"].dt.to_period("M").astype(str)
            df_mensal = df_finalizados.groupby("Mês_Ano")["Lucro Líquido (R$)"].sum().reset_index()
            fig_monthly = go.Figure(go.Bar(x=df_mensal["Mês_Ano"], y=df_mensal["Lucro Líquido (R$)"],
                                           marker_color=["#22c55e" if v>=0 else "#ef4444" for v in df_mensal["Lucro Líquido (R$)"]],
                                           text=df_mensal["Lucro Líquido (R$)"].apply(lambda x: f"R$ {x:,.2f}"), textposition="outside"))
            fig_monthly.update_layout(title="Lucro Líquido Mensal", template="plotly_dark")
            st.plotly_chart(fig_monthly, width='stretch')
        st.markdown("#### 📋 Detalhamento por Bilhete")
        st.dataframe(df_view, width='stretch', hide_index=True)

# ---------- ABA 6: MACRO LIGAS ----------
with tab6:
    st.markdown("### 📈 Macro Ligas (Modelos ML)")
    conn = get_db_connection()
    df_macro = pd.read_sql_query("SELECT liga, data_treinamento, num_amostras, acuracia, log_loss FROM modelos_ml ORDER BY data_treinamento DESC", conn)
    conn.close()
    if df_macro.empty: st.info("Nenhum modelo treinado ainda.")
    else: st.dataframe(df_macro, width='stretch')

# ---------- ABA 7: LOG DE ERROS API ----------
with tab7:
    st.markdown("### 📡 Log de Erros da API")
    if st.session_state.get("api_errors"):
        df_errors = pd.DataFrame(st.session_state.api_errors)
        if not df_errors.empty:
            # Converte colunas problemáticas para string
            for col in df_errors.columns:
                if df_errors[col].dtype == 'object':
                    df_errors[col] = df_errors[col].astype(str)
            st.dataframe(df_errors, width='stretch', hide_index=True)
    else:
        st.success("Nenhum erro de API.")
    with st.expander("📁 Histórico (Banco)"):
        conn = get_db_connection()
        df_hist = pd.read_sql_query("""
            SELECT timestamp, endpoint, status_code, message 
            FROM api_error_log 
            ORDER BY id DESC 
            LIMIT 500
        """, conn)
        conn.close()
        if not df_hist.empty:
            for col in df_hist.columns:
                if df_hist[col].dtype == 'object':
                    df_hist[col] = df_hist[col].astype(str)
            st.dataframe(df_hist, width='stretch', hide_index=True)

# ---------- ABA TREINAMENTO ML ----------
with tab_ml:
    st.header("🎓 Pipeline de Treinamento (com Pesos de Erro)")

    # ======================== CONTROLE DE PAUSA/RETOMADA ========================
    st.subheader("⏯️ Controle de Importação")
    col_pause, col_resume = st.columns(2)
    with col_pause:
        if st.button("⏸️ Pausar Importação Atual", width='stretch'):
            st.session_state.import_paused = True
            st.warning("Importação pausada. Use 'Retomar' para continuar.")
    with col_resume:
        if st.button("▶️ Retomar Importação", width='stretch'):
            st.session_state.import_paused = False
            st.info("Retomando...")
            st.rerun()

    # Mostra aviso e botão de retomada específico se houver importação interrompida
    if st.session_state.get("import_job_ids", []):
        st.warning(f"⚠️ Importação interrompida com {len(st.session_state.import_job_ids)} jogos pendentes.")
        col_r1, col_r2 = st.columns(2)
        with col_r1:
            if st.button("▶️ Retomar Importação Pendente", width='stretch'):
                st.session_state.import_paused = False
                st.rerun()
        with col_r2:
            if st.button("🗑️ Cancelar e Limpar Progresso", width='stretch'):
                st.session_state.import_job_ids = []
                st.session_state.import_current_index = 0
                st.session_state.import_paused = False
                st.success("Progresso cancelado.")
                st.rerun()
        st.divider()

    # ======================== 1. IMPORTAR JOGOS DO RADAR ========================
    st.subheader("📥 Importar Jogos do Radar")
    st.caption("Transfere os jogos que o radar já apontou e que já possuem resultado (GREEN/RED) para a base de treino.")
    if st.button("📥 Importar Jogos do Radar para Treino", width='stretch'):
        transferidos = importar_radar_para_treinamento()
        if transferidos:
            st.success(f"{transferidos} jogos transferidos para a base de treino.")
        else:
            st.info("Nenhum jogo novo para transferir.")

    st.divider()

    # ======================== 2. IMPORTAÇÃO MANUAL VIA API ========================
    st.subheader("📥 Importação Manual (API)")
    col1, col2, col3 = st.columns(3)
    with col1:
        dias_import = st.slider("Dias para importar (até 800)", 1, 800, 30, key="ml_dias")
    with col2:
        forcar_reimportar = st.checkbox("🔄 Forçar reimportação de datas antigas", value=False,
                                        help="Ignora o controle de última data e reimporta todo o período.")
    with col3:
        if st.button("📥 Importar Jogos Finalizados (API)", width='stretch'):
            if st.session_state.get("import_job_ids"):
                st.warning("Há uma importação pausada. Clique em 'Retomar' primeiro.")
            else:
                prog = st.progress(0)
                status = st.empty()
                novos = importar_jogos_para_treinamento(dias=dias_import,
                                                        progress_bar=prog,
                                                        status_text=status,
                                                        aplicar_filtro_odds=False,
                                                        forcar_reimportacao=forcar_reimportar)
                prog.empty()
                status.empty()
                if novos:
                    st.success(f"{novos} novos jogos adicionados!")
                else:
                    st.info("Nenhum novo jogo.")

    st.divider()

    # ======================== 3. LIMPEZA E PREPARAÇÃO ========================
    st.subheader("🧹 Limpeza e Preparação dos Dados")
    st.caption("Preserva todos os jogos, zera o estado de treino e força um novo modelo contextual sem odds.")
    if st.button("🧹 Preparar toda a base para retreino sem odds", width='stretch'):
        if preparar_toda_base_sem_odds():
            st.success("Base preservada e preparada para retreino contextual.")
            st.info("Agora execute uma nova validação campeão x desafiante.")
        else:
            st.error("Erro ao limpar a base de dados.")

    st.divider()

    # ======================== 4. EVOLUÇÃO VALIDADA ========================
    st.subheader("🧬 Evolução Automática do Modelo")
    st.caption("O desafiante usa os dados novos, mas só substitui o campeão após passar por tune e por um holdout cronológico final.")
    col_re1, col_re2 = st.columns(2)
    with col_re1:
        if st.button("🧬 Evoluir com dados novos", width='stretch'):
            with st.spinner("Treinando desafiante e validando cronologicamente..."):
                resultado = executar_evolucao_automatica(forcar=False)
            if resultado.get("promote"):
                st.success("Desafiante aprovado e promovido a campeão.")
            elif resultado.get("status") == "SKIPPED":
                st.info(resultado.get("reason"))
            else:
                st.warning(f"Campeão preservado: {resultado.get('reason')}")

    with col_re2:
        if st.button("🧪 Forçar nova validação (preserva o campeão)", width='stretch'):
            with st.spinner("Executando campeão x desafiante em toda a base..."):
                resultado = executar_evolucao_automatica(forcar=True)
            if resultado.get("promote"):
                final = resultado.get("final", {})
                st.success(f"Novo campeão: {final.get('selected_accuracy', 0):.2%} "
                           f"em {final.get('selected', 0)} seleções do holdout.")
            else:
                st.warning(f"Desafiante rejeitado; campeão intacto. {resultado.get('reason')}")

    with get_db_connection() as conn:
        policy = pd.read_sql_query("SELECT * FROM ml_selection_policy WHERE scope='GLOBAL'", conn)
        evolution = pd.read_sql_query("""SELECT finished_at, status, chosen_config, reason,
            min_confidence, final_accuracy, final_selected, final_coverage, log_loss, promoted
            FROM ml_evolution_runs ORDER BY id DESC LIMIT 20""", conn)
        live_monitoring = pd.read_sql_query("""SELECT evaluated_at, samples, min_confidence,
            accuracy, recommended_confidence, status, details
            FROM ml_live_monitoring ORDER BY id DESC LIMIT 30""", conn)
    if not policy.empty:
        p = policy.iloc[0]
        st.info(f"Política ativa: mínimo {p['min_confidence']:.0f}% | estado {p['status']} | "
                f"acurácia validada: {p['validated_accuracy']:.2%}"
                if pd.notna(p['validated_accuracy']) else
                f"Política ativa: mínimo {p['min_confidence']:.0f}% | aguardando validação v7")
    if not evolution.empty:
        with st.expander("Histórico campeão x desafiante"):
            st.dataframe(evolution, hide_index=True, width='stretch')
    if not live_monitoring.empty:
        with st.expander("Validação dos resultados reais por versão"):
            st.dataframe(live_monitoring, hide_index=True, width='stretch')

    st.divider()

    # ======================== 5. DIAGNÓSTICO DE FEATURES PROIBIDAS ========================
    st.subheader("🔍 Diagnóstico de Features Proibidas nos Modelos Salvos")
    if st.button("Verificar Modelos com Vazamento", width='stretch'):
        conn = get_db_connection()
        rows = conn.execute("SELECT liga, feature_order FROM modelos_ml").fetchall()
        conn.close()
        proibidas = [
            'xg_casa', 'xg_fora', 'posse_casa', 'posse_fora',
            'chutes_casa', 'chutes_fora', 'chutes_gol_casa', 'chutes_gol_fora',
            'escanteios_casa', 'escanteios_fora', 'faltas_casa', 'faltas_fora',
            'xg_diff', 'posse_diff', 'chutes_diff', 'chutes_gol_diff',
            'escanteios_diff', 'faltas_diff'
        ]
        contaminados = []
        for liga, fo_json in rows:
            if fo_json:
                fo = json.loads(fo_json)
                encontradas = [f for f in proibidas if f in fo]
                if encontradas:
                    contaminados.append((liga, encontradas))
        if contaminados:
            st.error(f"🚨 {len(contaminados)} modelo(s) ainda contêm features proibidas:")
            for liga, vars_ in contaminados:
                st.write(f"**{liga}** → {', '.join(vars_)}")
            st.warning("Use o botão de validação campeão x desafiante acima para resolver.")
        else:
            st.success("✅ Nenhum modelo com features proibidas encontrado!")

    with st.expander("📊 Distribuição de Pesos de Erro"):
        conn = get_db_connection()
        pesos_df = pd.read_sql_query("""
            SELECT t.match_id, t.liga, w.peso 
            FROM training_weights w 
            JOIN training_data t USING (match_id) 
            WHERE w.peso>1.0 
            ORDER BY w.peso DESC
        """, conn)
        conn.close()
        if not pesos_df.empty:
            st.dataframe(pesos_df)
            st.plotly_chart(px.histogram(pesos_df, x="peso", nbins=20), width='stretch')
        else:
            st.info("Nenhum peso aumentado ainda.")

    # ======================== 6. RECALCULAR ELOS ========================
    st.subheader("🏆 Recalcular Elo Rating")
    st.caption("Recalcula os Elos de todos os times a partir dos jogos já importados na base de treino (ordem cronológica).")
    if st.button("🔄 Recalcular Elo Rating (todos os times)", width='stretch', type="primary"):
        with st.spinner("Recalculando Elos... Isso pode levar alguns segundos."):
            # Importar a função do robo_auto (ou definir localmente)
            try:
                from robo_auto import recalcular_todos_elos
            except ImportError:
                # Definir a função localmente se não estiver no robo_auto
                def recalcular_todos_elos():
                    conn = get_db_connection()
                    jogos = conn.execute("""
                        SELECT home_team, away_team, home_score, away_score, data_jogo
                        FROM training_data
                        ORDER BY data_jogo ASC
                    """).fetchall()
                    conn.close()
                    if not jogos:
                        st.warning("Nenhum jogo encontrado na base de treino.")
                        return
                    elos = {}
                    for home, away, hs, aws, _ in jogos:
                        res_h = 1.0 if hs > aws else (0.5 if hs == aws else 0.0)
                        res_a = 1.0 - res_h
                        elo_h = elos.get(home, 1500)
                        elo_a = elos.get(away, 1500)
                        expected_h = 1 / (1 + 10 ** ((elo_a - elo_h) / 400))
                        expected_a = 1 / (1 + 10 ** ((elo_h - elo_a) / 400))
                        elos[home] = elo_h + 32 * (res_h - expected_h)
                        elos[away] = elo_a + 32 * (res_a - expected_a)
                    conn = get_db_connection()
                    for team, elo in elos.items():
                        conn.execute("INSERT OR REPLACE INTO elo_rating (team_id, elo, last_update) VALUES (?, ?, ?)",
                                     (team, int(elo), get_brt_time().isoformat()))
                    conn.commit()
                    conn.close()
                    st.success(f"Elos recalculados para {len(elos)} times.")
            recalcular_todos_elos()
        st.success("Recálculo concluído!")

    # ======================== 7. PERIGO: LIMPEZA COMPLETA ========================
    with st.expander("⚠️ PERIGO: Limpeza Completa da Base de Treino"):
        st.warning("Isso apagará TODOS os jogos de treino, pesos e modelos. Use com EXTREMA cautela!")
        col_confirm1, col_confirm2 = st.columns(2)
        with col_confirm1:
            if st.button("🗑️ SIM, APAGAR TUDO (training_data, weights, modelos)", width='stretch', type="secondary"):
                with db_write_lock:
                    conn = get_db_connection()
                    conn.execute("DELETE FROM training_data")
                    conn.execute("DELETE FROM training_weights")
                    conn.execute("DELETE FROM modelos_ml")
                    conn.commit()
                    conn.close()
                st.success("Base de treino completamente limpa! Agora você pode reimportar os jogos.")
                st.balloons()
        with col_confirm2:
            st.caption("Clique no botão ao lado apenas se tiver certeza.")

# ---------- ABA AVALIAÇÃO ----------
with tab_avaliacao:
    st.header("🔍 Diagnóstico de Erros e Feedback (com EV e Backtest Cronológico)")
    st.markdown("Validação cronológica (treino no passado, teste no futuro) e Kelly Fracionado 25%.")

    conn = get_db_connection()
    df_modelos = pd.read_sql_query("SELECT liga FROM modelos_ml ORDER BY data_treinamento DESC", conn)
    conn.close()
    if df_modelos.empty:
        st.info("Nenhum modelo treinado ainda.")
    else:
        liga_avaliar = st.selectbox("Escolha a liga para avaliar", df_modelos['liga'].tolist(), key="avaliar_liga")
        if liga_avaliar:
            model, scaler_persistido, feature_order = carregar_modelo_liga(liga_avaliar)
            X, y, feat_order, match_ids, pesos = preparar_dados_treinamento(liga_avaliar)
            if X is None or len(X) == 0:
                st.warning("Sem dados.")
            else:
                split_idx = int(len(X) * 0.8)
                X_train_raw, X_test_raw = X[:split_idx], X[split_idx:]
                y_train, y_test = y[:split_idx], y[split_idx:]
                match_ids_train, match_ids_test = match_ids[:split_idx], match_ids[split_idx:]
                scaler_cv = StandardScaler()
                X_train = scaler_cv.fit_transform(X_train_raw)
                X_test = scaler_cv.transform(X_test_raw)
                class_weights = compute_class_weight('balanced', classes=np.array([0,1,2]), y=y_train)
                sample_weight_train = np.ones(len(y_train))
                for cls, w in zip([0,1,2], class_weights):
                    sample_weight_train[y_train == cls] *= w
                model_cv = xgb.XGBClassifier(objective='multi:softprob', num_class=3, eval_metric='mlogloss',
                                             random_state=42, n_jobs=-1, max_depth=6, learning_rate=0.05, n_estimators=300)
                model_cv.fit(X_train, y_train, sample_weight=sample_weight_train)
                y_pred = model_cv.predict(X_test)
                y_proba = model_cv.predict_proba(X_test)
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Matriz de Confusão")
                    cm = confusion_matrix(y_test, y_pred, labels=[0,1,2])
                    classes_nomes = ['Casa', 'Empate', 'Fora']
                    fig_cm = px.imshow(cm, text_auto=True, x=classes_nomes, y=classes_nomes, color_continuous_scale='Blues')
                    st.plotly_chart(fig_cm, width='stretch')
                with col2:
                    st.subheader("Relatório")
                    report = classification_report(y_test, y_pred, target_names=classes_nomes, output_dict=True)
                    st.dataframe(pd.DataFrame(report).transpose().style.format("{:.2f}"))

                st.markdown("---")
                st.subheader("📈 Backtest Financeiro (Kelly 25%)")
                odds_test_list = []
                if match_ids_test:
                    conn2 = get_db_connection()
                    placeholders = ','.join(['?']*len(match_ids_test))
                    query = f"SELECT match_id, odd_casa, odd_empate, odd_fora FROM training_data WHERE match_id IN ({placeholders})"
                    df_odds = pd.read_sql_query(query, conn2, params=match_ids_test)
                    conn2.close()
                    odds_map = {row['match_id']: (row['odd_casa'], row['odd_empate'], row['odd_fora']) for _, row in df_odds.iterrows()}
                    odds_test_list = [odds_map.get(mid, (2.0,3.0,2.0)) for mid in match_ids_test]
                if odds_test_list:
                    bankroll = 1000.0
                    banca_evolucao = [bankroll]
                    for proba, (odd_c, odd_e, odd_f), real in zip(y_proba, odds_test_list, y_test):
                        ev_c = proba[0]*odd_c - 1; ev_e = proba[1]*odd_e - 1; ev_f = proba[2]*odd_f - 1
                        idx_aposta = np.argmax([ev_c, ev_e, ev_f])
                        odd_apostada = odd_c if idx_aposta==0 else odd_e if idx_aposta==1 else odd_f
                        p = proba[idx_aposta]
                        b = odd_apostada - 1.0
                        kelly = (p * b - (1.0 - p)) / b if b > 0 else 0.0
                        if kelly > 0 and [ev_c, ev_e, ev_f][idx_aposta] > 0.05:
                            stake = bankroll * (kelly * 0.25)
                            if idx_aposta == real: bankroll += stake * odd_apostada - stake
                            else: bankroll -= stake
                        banca_evolucao.append(bankroll)
                    col_fin1, col_fin2 = st.columns(2)
                    with col_fin1:
                        st.metric("Lucro Final", f"R$ {bankroll-1000:.2f}")
                    with col_fin2:
                        st.line_chart(pd.DataFrame({'Banca': banca_evolucao}))

    st.markdown("---")
    st.subheader("🔍 Ajuste de Pesos (limite 8x)")
    ligas_erros = [r[0] for r in get_db_connection().execute("SELECT DISTINCT liga FROM training_data").fetchall()]
    liga_sel = st.selectbox("Liga", ["Todas"] + ligas_erros, key="diag_liga")
    if liga_sel == "Todas": liga_sel = None
    if st.button("🔎 Identificar Erros Recentes", width='stretch'):
        conn = get_db_connection()
        query = "SELECT match_id, liga, data_jogo, home_team, away_team, home_score, away_score, features FROM training_data WHERE usado_treinamento=0 ORDER BY data_jogo DESC LIMIT 200"
        df = pd.read_sql_query(query, conn)
        conn.close()
        erros = []
        for _, row in df.iterrows():
            feats = json.loads(row['features'])
            model, scaler, feature_order = carregar_modelo_liga(row['liga'])
            if model is None: continue
            vec = [feats.get(k, 0.0) for k in feature_order]
            X = np.array([vec]); X_s = scaler.transform(X)
            pred = np.argmax(model.predict_proba(X_s)[0])
            real = 0 if row['home_score'] > row['away_score'] else (1 if row['home_score'] == row['away_score'] else 2)
            if pred != real:
                peso_row = get_db_connection().execute("SELECT peso FROM training_weights WHERE match_id=?", (row['match_id'],)).fetchone()
                peso = peso_row[0] if peso_row else 1.0
                erros.append({"match_id": row['match_id'], "confronto": f"{row['home_team']} vs {row['away_team']}",
                              "placar": f"{row['home_score']}-{row['away_score']}", "predito": ['Casa','Empate','Fora'][pred],
                              "real": ['Casa','Empate','Fora'][real], "peso_atual": peso})
        st.session_state['erros_diag'] = erros
    if 'erros_diag' in st.session_state and st.session_state['erros_diag']:
        erros_df = pd.DataFrame(st.session_state['erros_diag'])
        st.dataframe(erros_df, width='stretch')
        if st.button("⏫ Dobrar Peso de TODOS (erros → peso 2.0)", type="primary"):
            for err in st.session_state['erros_diag']:
                atualizar_peso_erro(err['match_id'], errou=True)
            st.success(f"{len(st.session_state['erros_diag'])} pesos atualizados.")
            time.sleep(1); st.rerun()
        for err in st.session_state['erros_diag']:
            cols = st.columns([3,1,1])
            cols[0].write(f"{err['confronto']} ({err['placar']}) - Prev: {err['predito']} | Real: {err['real']}")
            if cols[1].button("⏫ Dobrar (erro)", key=f"up_{err['match_id']}"):
                atualizar_peso_erro(err['match_id'], errou=True)
                st.success("Peso marcado como erro (2.0)."); st.rerun()
            if cols[2].button("⏬ Resetar (acerto)", key=f"reset_{err['match_id']}"):
                atualizar_peso_erro(err['match_id'], errou=False)
                st.success("Peso resetado para 1.0."); st.rerun()

    st.markdown("---")
    st.subheader("📈 Importância das Features (SHAP)")
    if SHAP_AVAILABLE:
        liga_shap = st.selectbox("Liga para SHAP", ligas_erros, key="shap_liga")
        if st.button("Gerar SHAP"):
            model, scaler, feature_order = carregar_modelo_liga(liga_shap)
            if model:
                X, _, _, _, _ = preparar_dados_treinamento(liga_shap)
                if X is not None and len(X) > 0:
                    X_s = scaler.transform(X[:200])
                    explainer = shap.Explainer(model.estimator_ if hasattr(model,'estimator_') else model, X_s)
                    shap_values = explainer(X_s)
                    fig = shap.summary_plot(shap_values, features=X_s, feature_names=feature_order, show=False)
                    st.pyplot(fig)
    else: st.warning("Instale SHAP (`pip install shap`) para visualizar.")

# ---------- ABA BACKTEST ----------
with tab9:
    st.markdown("### 📈 Backtest – Simulação do Radar (Confiança Pura)")
    st.warning(
        "Este backtest utiliza a base de treino, aplica a blacklist do radar "
        "e monta os bilhetes com seleção baseada "
        "**exclusivamente na confiança** (probabilidade do modelo)."
    )

    with st.expander("🔍 Verificar Vazamento de Dados nos Modelos", expanded=False):
        if st.button("Analisar Features dos Modelos Salvos", key="btn_check_leakage"):
            conn = get_db_connection()
            modelos = pd.read_sql_query("SELECT liga, feature_order FROM modelos_ml", conn)
            conn.close()
            features_proibidas = [
                'xg_casa', 'xg_fora', 'posse_casa', 'posse_fora',
                'chutes_casa', 'chutes_fora', 'chutes_gol_casa', 'chutes_gol_fora',
                'escanteios_casa', 'escanteios_fora', 'faltas_casa', 'faltas_fora',
                'xg_diff', 'posse_diff', 'chutes_diff', 'chutes_gol_diff',
                'escanteios_diff', 'faltas_diff'
            ]
            contaminados = []
            for _, row in modelos.iterrows():
                if row['feature_order']:
                    fo = json.loads(row['feature_order'])
                    proib = [f for f in features_proibidas if f in fo]
                    if proib:
                        contaminados.append((row['liga'], proib))
            if contaminados:
                st.error("🚨 VAZAMENTO DE DADOS DETECTADO NOS SEGUINTES MODELOS:")
                for liga, vars_ in contaminados:
                    st.write(f"**{liga}** → contém: {', '.join(vars_)}")
            else:
                st.success("✅ Nenhum modelo contém features de vazamento.")

    with st.expander("🧹 Limpeza do Banco de Dados (Remover categorias de base e feminino)", expanded=False):
        st.caption("Remove jogos de base e futebol feminino.")
        termos_proibidos_full = [
            "u12","u13","u14","u15","u16","u17","u18","u19","u20","u21","u22","u23","u24",
            "u 12","u 13","u 14","u 15","u 16","u 17","u 18","u 19","u 20","u 21","u 22","u 23","u 24",
            "u-12","u-13","u-14","u-15","u-16","u-17","u-18","u-19","u-20","u-21","u-22","u-23","u-24",
            "sub12","sub13","sub14","sub15","sub16","sub17","sub18","sub19","sub20","sub21","sub22","sub23","sub24",
            "sub 12","sub 13","sub 14","sub 15","sub 16","sub 17","sub 18","sub 19","sub 20","sub 21","sub 22","sub 23","sub 24",
            "sub-12","sub-13","sub-14","sub-15","sub-16","sub-17","sub-18","sub-19","sub-20","sub-21","sub-22","sub-23","sub-24",
            "sub","junior","juvenil","youth","aspirantes","reserva","reservas","reserves","reserve",
            "amateur","amador","amadores",
            "woman","women","feminino","femenino","femenil","femmes","frauen","ladies","girls",
            " w ", "w's", "womens",
        ]
        if st.button("🔍 Pré‑visualizar", key="btn_preview_clean"):
            termos_escapados = [t.replace("'", "''") for t in termos_proibidos_full]
            condicoes_liga = " OR ".join([f"LOWER(liga) LIKE '%{t}%'" for t in termos_escapados])
            condicoes_home = " OR ".join([f"LOWER(home_team) LIKE '%{t}%'" for t in termos_escapados])
            condicoes_away = " OR ".join([f"LOWER(away_team) LIKE '%{t}%'" for t in termos_escapados])
            condicoes_sql = f"({condicoes_liga}) OR ({condicoes_home}) OR ({condicoes_away})"
            conn = get_db_connection()
            count = conn.execute(f"SELECT COUNT(*) FROM training_data WHERE {condicoes_sql}").fetchone()[0]
            conn.close()
            st.warning(f"**{count}** registos seriam removidos.") if count>0 else st.success("Nada a remover.")
        if st.button("🗑️ Executar Limpeza", type="secondary", key="btn_clean_db"):
            termos_escapados = [t.replace("'", "''") for t in termos_proibidos_full]
            condicoes_liga = " OR ".join([f"LOWER(liga) LIKE '%{t}%'" for t in termos_escapados])
            condicoes_home = " OR ".join([f"LOWER(home_team) LIKE '%{t}%'" for t in termos_escapados])
            condicoes_away = " OR ".join([f"LOWER(away_team) LIKE '%{t}%'" for t in termos_escapados])
            condicoes_sql = f"({condicoes_liga}) OR ({condicoes_home}) OR ({condicoes_away})"
            with db_write_lock:
                conn = get_db_connection()
                try:
                    conn.execute(f"DELETE FROM training_weights WHERE match_id IN (SELECT match_id FROM training_data WHERE {condicoes_sql})")
                    conn.execute(f"DELETE FROM training_data WHERE {condicoes_sql}")
                    conn.commit()
                    conn.close()
                    st.success(f"Removidos. Re‑treine os modelos.")
                except Exception as e:
                    conn.rollback(); conn.close()
                    st.error(f"Erro: {e}")

    st.caption("Critério: modelo contextual sem odds, bilhetes de **4 jogos** (sem repetir liga).")
    st.caption("Ordenação: **confiança decrescente**.")
    aposta_por_bilhete = 1.0

    if st.button("Executar Backtest Operacional (sem vazamento)", key="btn_bt_sim"):
        report = operational_snapshot_report(DB_NAME)
        st.subheader("📊 Picks realmente emitidos antes dos jogos")
        col1, col2, col3 = st.columns(3)
        col1.metric("Acertos individuais", report["correct_predictions"])
        col2.metric("Jogos resolvidos", report["resolved_predictions"])
        col3.metric("Assertividade", f"{report['individual_accuracy']:.2%}")
        col4, col5, col6 = st.columns(3)
        col4.metric("Bilhetes completos", report["complete_tickets"])
        col5.metric("Bilhetes 4/4", report["green_tickets"])
        col6.metric("Taxa 4/4", f"{report['ticket_green_rate']:.2%}")
        st.write("Recall por resultado:", report["recall_by_outcome"])
        st.write("Acertos por bilhete (0 a 4):", report["ticket_hits_distribution"])
        st.caption(
            "O cálculo usa o snapshot congelado no radar e o resultado gravado "
            "pela auditoria; nenhum jogo é previsto por um modelo que já o viu."
        )

    # Implementação histórica desativada: treinava em toda a base e reaplicava
    # o mesmo artefato aos próprios jogos de treino, portanto não era cega.
    if False:  # pragma: no cover - preservada somente para auditoria do legado
        conn = get_db_connection()

        termos_escapados = [t.replace("'", "''") for t in termos_proibidos_full]
        sql_partes = []
        for t in termos_escapados:
            sql_partes.append(f"(LOWER(liga) NOT LIKE '%{t}%' AND LOWER(home_team) NOT LIKE '%{t}%' AND LOWER(away_team) NOT LIKE '%{t}%')")
        sql_filtro = " AND ".join(sql_partes) if sql_partes else "1=1"

        query_datas = f"SELECT DISTINCT data_jogo FROM training_data WHERE {sql_filtro} ORDER BY data_jogo ASC"
        datas_df = pd.read_sql_query(query_datas, conn)
        datas_unicas = sorted(pd.to_datetime(datas_df['data_jogo']).dt.date.unique())
        total_datas = len(datas_unicas)

        if total_datas == 0:
            st.warning("Nenhum jogo encontrado com os filtros atuais.")
            conn.close()
            st.stop()

        progress_bar = st.progress(0)
        status_text = st.empty()
        capital = 0.0
        bilhetes_gerados = 0
        bilhetes_verdes = 0
        historico_bilhetes = []
        modelos_cache = {}

        for i, data_alvo in enumerate(datas_unicas):
            inicio = datetime.combine(data_alvo, datetime.min.time())
            fim = inicio + timedelta(days=1)
            query_dia = f"""
                SELECT match_id, data_jogo, home_team, away_team, home_score, away_score,
                       features, odd_casa, odd_empate, odd_fora, liga
                FROM training_data
                WHERE data_jogo >= ? AND data_jogo < ? AND ({sql_filtro})
                ORDER BY data_jogo ASC
            """
            df_dia = pd.read_sql_query(query_dia, conn, params=(inicio.strftime("%Y-%m-%d %H:%M:%S"),
                                                                  fim.strftime("%Y-%m-%d %H:%M:%S")))
            jogos_analisados = []
            for _, jogo in df_dia.iterrows():
                liga = jogo['liga']
                if liga not in modelos_cache:
                    model, scaler, feature_order = carregar_modelo_liga(liga)
                    modelos_cache[liga] = (model, scaler, feature_order) if model else None
                cached = modelos_cache[liga]
                if cached is None:
                    continue
                model, scaler, feature_order = cached

                feats = json.loads(jogo['features'])
                vec = [feats.get(k, 0.0) for k in feature_order]
                X = np.array([vec])
                X_s = scaler.transform(X)
                proba = model.predict_proba(X_s)[0]
                idx = np.argmax(proba)
                pick = ['MANDANTE', 'EMPATE', 'VISITANTE'][idx]
                confianca = round(float(proba[idx]) * 100.0, 2)

                # O backtest contextual mede acerto; ROI por odd não é calculado.
                odd_pick = 1.0

                jogos_analisados.append({
                    'confronto': f"{jogo['home_team']} vs {jogo['away_team']}",
                    'liga': liga,
                    'pick': pick,
                    'confianca': confianca,
                    'odd_pick': odd_pick,
                    'real': (jogo['home_score'], jogo['away_score'])
                })

            jogos_ordenados = sorted(jogos_analisados, key=lambda x: -x['confianca'])

            while len(jogos_ordenados) >= 4:
                ticket, ligas_ticket, restantes = [], {}, []
                for j in jogos_ordenados:
                    if len(ticket) == 4:
                        restantes.append(j)
                        continue
                    if ligas_ticket.get(j['liga'], 0) < 1:
                        ticket.append(j)
                        ligas_ticket[j['liga']] = 1
                    else:
                        restantes.append(j)

                if len(ticket) == 4:
                    odd_bilhete = np.prod([j['odd_pick'] for j in ticket])
                    bilhete_verde = all(
                        (j['pick'] == 'MANDANTE' and j['real'][0] > j['real'][1]) or
                        (j['pick'] == 'VISITANTE' and j['real'][1] > j['real'][0]) or
                        (j['pick'] == 'EMPATE' and j['real'][0] == j['real'][1])
                        for j in ticket
                    )
                    lucro = aposta_por_bilhete * odd_bilhete - aposta_por_bilhete if bilhete_verde else -aposta_por_bilhete
                    capital += lucro
                    bilhetes_gerados += 1
                    if bilhete_verde:
                        bilhetes_verdes += 1
                    historico_bilhetes.append({
                        'Data': data_alvo.strftime('%d/%m/%Y'),
                        'Jogos': [j['confronto'] for j in ticket],
                        'Picks': [j['pick'] for j in ticket],
                        'Odd Bilhete': round(odd_bilhete, 2),
                        'Verde?': 'Sim' if bilhete_verde else 'Não',
                        'Lucro (R$)': round(lucro, 2),
                        'Capital Acum. (R$)': round(1000.0 + capital, 2)
                    })
                    jogos_ordenados = restantes
                else:
                    if len(jogos_ordenados) >= 4:
                        ticket = jogos_ordenados[:4]
                        odd_bilhete = np.prod([j['odd_pick'] for j in ticket])
                        bilhete_verde = all(
                            (j['pick'] == 'MANDANTE' and j['real'][0] > j['real'][1]) or
                            (j['pick'] == 'VISITANTE' and j['real'][1] > j['real'][0]) or
                            (j['pick'] == 'EMPATE' and j['real'][0] == j['real'][1])
                            for j in ticket
                        )
                        lucro = aposta_por_bilhete * odd_bilhete - aposta_por_bilhete if bilhete_verde else -aposta_por_bilhete
                        capital += lucro
                        bilhetes_gerados += 1
                        if bilhete_verde:
                            bilhetes_verdes += 1
                        historico_bilhetes.append({
                            'Data': data_alvo.strftime('%d/%m/%Y'),
                            'Jogos': [j['confronto'] for j in ticket],
                            'Picks': [j['pick'] for j in ticket],
                            'Odd Bilhete': round(odd_bilhete, 2),
                            'Verde?': 'Sim' if bilhete_verde else 'Não',
                            'Lucro (R$)': round(lucro, 2),
                            'Capital Acum. (R$)': round(1000.0 + capital, 2)
                        })
                        jogos_ordenados = jogos_ordenados[4:]
                    else:
                        break

            progress = (i + 1) / total_datas
            progress_bar.progress(progress)
            status_text.text(f"Processando {data_alvo.strftime('%d/%m/%Y')} ({i+1}/{total_datas})")

        conn.close()
        progress_bar.empty()
        status_text.empty()

        if historico_bilhetes:
            st.subheader("📊 Resultado da Simulação (Confiança Pura)")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Bilhetes", bilhetes_gerados)
            col2.metric("Verdes", bilhetes_verdes)
            col3.metric("Vermelhos", bilhetes_gerados - bilhetes_verdes)
            col4.metric("Lucro Total (R$)", f"{capital:+.2f}")
            st.metric("ROI (%)", f"{(capital / (bilhetes_gerados * aposta_por_bilhete)) * 100:.2f}%")

            df_hist = pd.DataFrame(historico_bilhetes)
            st.dataframe(df_hist, width='stretch')

            fig = px.line(df_hist, y='Capital Acum. (R$)', title='Evolução do Capital')
            st.plotly_chart(fig, width='stretch')

            st.subheader("📆 Lucro Diário")
            df_hist['Data_dt'] = pd.to_datetime(df_hist['Data'], format='%d/%m/%Y')
            df_diario = df_hist.groupby('Data_dt')['Lucro (R$)'].sum().reset_index()
            df_diario.columns = ['Data', 'Lucro Líquido (R$)']
            fig_diario = px.bar(df_diario, x='Data', y='Lucro Líquido (R$)',
                                title='Lucro Diário',
                                color='Lucro Líquido (R$)',
                                color_continuous_scale=[(0, 'red'), (0.5, 'yellow'), (1, 'green')],
                                text=df_diario['Lucro Líquido (R$)'].apply(lambda x: f"R$ {x:+.2f}"))
            fig_diario.update_traces(textposition='outside')
            fig_diario.update_layout(coloraxis_showscale=False, template='plotly_dark')
            st.plotly_chart(fig_diario, width='stretch')
            st.dataframe(df_diario.style.format({'Lucro Líquido (R$)': 'R$ {:+.2f}'}), width='stretch', hide_index=True)

            st.subheader("📅 Lucro Mensal")
            df_hist['Mes'] = df_hist['Data_dt'].dt.to_period('M').astype(str)
            df_mensal = df_hist.groupby('Mes')['Lucro (R$)'].sum().reset_index()
            df_mensal.columns = ['Mês', 'Lucro Líquido (R$)']
            fig_mensal = px.bar(df_mensal, x='Mês', y='Lucro Líquido (R$)',
                                title='Lucro Mensal',
                                color='Lucro Líquido (R$)',
                                color_continuous_scale=[(0, 'red'), (0.5, 'yellow'), (1, 'green')],
                                text=df_mensal['Lucro Líquido (R$)'].apply(lambda x: f"R$ {x:+.2f}"))
            fig_mensal.update_traces(textposition='outside')
            fig_mensal.update_layout(coloraxis_showscale=False, template='plotly_dark')
            st.plotly_chart(fig_mensal, width='stretch')
            st.dataframe(df_mensal.style.format({'Lucro Líquido (R$)': 'R$ {:+.2f}'}), width='stretch', hide_index=True)
        else:
            st.info("Nenhum bilhete foi gerado. Verifique os filtros e a base de dados.")

with tab_diagnostico:
    st.header("🔍 Diagnóstico de Features por Liga")
    st.markdown("Verifique se os dados estão sendo coletados corretamente. Valores `None`, vazios ou zerados podem indicar problemas na extração.")

    conn = get_db_connection()
    ligas_df = pd.read_sql_query("SELECT DISTINCT liga FROM training_data ORDER BY liga", conn)
    conn.close()
    ligas = ligas_df['liga'].tolist()

    if not ligas:
        st.warning("Nenhum dado de treinamento encontrado. Execute o radar ou importe jogos primeiro.")
    else:
        liga_selecionada = st.selectbox("Selecione uma liga", ligas)
        if liga_selecionada:
            conn = get_db_connection()
            # Apenas colunas que existem na tabela
            query = """
                SELECT match_id, data_jogo, home_team, away_team, home_score, away_score, features
                FROM training_data
                WHERE liga = ?
                ORDER BY data_jogo DESC
                LIMIT 10
            """
            df_jogos = pd.read_sql_query(query, conn, params=(liga_selecionada,))
            conn.close()

            if df_jogos.empty:
                st.info("Nenhum jogo encontrado para esta liga.")
            else:
                st.subheader(f"Últimos {len(df_jogos)} jogos - {liga_selecionada}")

                for idx, row in df_jogos.iterrows():
                    with st.expander(f"{row['home_team']} vs {row['away_team']} ({row['data_jogo']}) - Placar: {row['home_score']}-{row['away_score']}"):
                        features = json.loads(row['features'])
                        
                        # Extrair tournament_id e season_id do JSON features (se existirem)
                        tournament_id = features.get('tournament_id', 'N/A')
                        season_id = features.get('season_id', 'N/A')
                        match_id = row['match_id']
                        
                        # Exibir IDs para teste manual
                        st.markdown(f"""
                        **🔑 IDs para teste manual:**  
                        - `tournament_id`: `{tournament_id}`  
                        - `season_id`: `{season_id}`  
                        - `match_id`: `{match_id}`
                        """)
                        
                        # Remover os campos de ID do features para não poluir a tabela principal (opcional)
                        features_display = {k: v for k, v in features.items() if k not in ['tournament_id', 'season_id']}
                        df_feats = pd.DataFrame(list(features_display.items()), columns=['Feature', 'Valor'])
                        
                        # Função para destacar valores problemáticos (None, vazio, zero)
                        def highlight_problem(val):
                            if val is None or val == '' or (isinstance(val, (int, float)) and val == 0):
                                return 'background-color: #ffcccc'
                            return ''
                        
                        styled_df = df_feats.style.map(highlight_problem, subset=['Valor'])
                        st.dataframe(styled_df, width='stretch')
                        
                        problemas = []
                        for k, v in features_display.items():
                            if v is None or v == '' or (isinstance(v, (int, float)) and v == 0):
                                problemas.append(k)
                        if problemas:
                            st.warning(f"⚠️ Features com problema (None, vazio ou zero): {', '.join(problemas[:10])}")
                        else:
                            st.success("✅ Todas as features preenchidas corretamente.")
                
                st.subheader("📊 Estatísticas médias das features para esta liga")
                conn = get_db_connection()
                df_all = pd.read_sql_query("SELECT features FROM training_data WHERE liga = ?", conn, params=(liga_selecionada,))
                conn.close()

                agg_features = {}
                count = 0
                for _, r in df_all.iterrows():
                    feats = json.loads(r['features'])
                    # Remover campos não numéricos para média
                    feats_numeric = {k: v for k, v in feats.items() if isinstance(v, (int, float))}
                    count += 1
                    for k, v in feats_numeric.items():
                        agg_features[k] = agg_features.get(k, 0) + v
                if count > 0:
                    medias = {k: v/count for k, v in agg_features.items()}
                    df_medias = pd.DataFrame(list(medias.items()), columns=['Feature', 'Média'])
                    st.dataframe(df_medias, width='stretch')
                    
                    zeradas = [k for k, v in medias.items() if v == 0]
                    if zeradas:
                        st.warning(f"⚠️ Features sempre zeradas (possível falta de dados): {', '.join(zeradas[:10])}")
                    else:
                        st.success("Nenhuma feature está sempre zerada.")
                else:
                    st.info("Não foi possível calcular médias.")
                    
with tab_monitor_ligas:
    st.header("📊 Desempenho dos Modelos por Liga")
    st.caption("Métricas de acurácia, log-loss, ROC-AUC e número de amostras (TimeSeriesSplit).")

    conn = get_db_connection()
    # Busca todos os modelos, exceto o GLOBAL (opcional)
    df_modelos = pd.read_sql_query("""
        SELECT liga, data_treinamento, num_amostras, acuracia, log_loss, roc_auc
        FROM modelos_ml
        WHERE liga != 'GLOBAL'
        ORDER BY acuracia DESC
    """, conn)
    conn.close()

    if df_modelos.empty:
        st.info("Nenhum modelo de liga treinado ainda. Execute o treinamento na aba 'Treinamento ML'.")
    else:
        # Formatação para exibição
        df_display = df_modelos.copy()
        df_display['acuracia'] = df_display['acuracia'].apply(lambda x: f"{x:.2%}")
        df_display['log_loss'] = df_display['log_loss'].apply(lambda x: f"{x:.4f}")
        df_display['roc_auc'] = df_display['roc_auc'].fillna(0).apply(lambda x: f"{x:.4f}")
        df_display['num_amostras'] = df_display['num_amostras'].astype(int)
        df_display['data_treinamento'] = pd.to_datetime(df_display['data_treinamento']).dt.strftime("%d/%m/%Y %H:%M")

        # Tabela
        st.dataframe(
            df_display,
            column_config={
                "liga": "Liga",
                "data_treinamento": "Último Treino",
                "num_amostras": "Amostras",
                "acuracia": "Acurácia",
                "log_loss": "Log-Loss",
                "roc_auc": "ROC-AUC"
            },
            width='stretch',
            hide_index=True
        )

        # Preparar dados para gráficos
        df_plot = df_modelos.copy()
        df_plot['acuracia_num'] = df_plot['acuracia']
        df_plot['roc_auc_num'] = df_plot['roc_auc'].fillna(0)

        # Gráfico de acurácia
        st.subheader("📈 Acurácia por Liga")
        fig_acc = px.bar(df_plot, x='liga', y='acuracia_num', 
                         text=df_plot['acuracia'].apply(lambda x: f"{x:.2%}"),
                         color='acuracia_num', color_continuous_scale='viridis',
                         title="Acurácia (quanto maior, melhor)")
        fig_acc.update_traces(textposition='outside')
        fig_acc.update_layout(xaxis_tickangle=-45, yaxis_title="Acurácia")
        st.plotly_chart(fig_acc, width='stretch')

        # Gráfico de ROC-AUC (se existirem valores positivos)
        if (df_plot['roc_auc_num'] > 0).any():
            st.subheader("📈 ROC-AUC (multiclasse) por Liga")
            fig_roc = px.bar(df_plot, x='liga', y='roc_auc_num',
                             text=df_plot['roc_auc'].apply(lambda x: f"{x:.4f}"),
                             color='roc_auc_num', color_continuous_scale='plasma',
                             title="ROC-AUC (quanto maior, melhor)")
            fig_roc.update_traces(textposition='outside')
            fig_roc.update_layout(xaxis_tickangle=-45, yaxis_title="ROC-AUC")
            st.plotly_chart(fig_roc, width='stretch')
        else:
            st.info("ROC-AUC ainda não disponível (re-treine os modelos para calcular).")

        # Métricas resumidas
        st.subheader("📊 Resumo Global")
        col1, col2, col3 = st.columns(3)
        col1.metric("Ligas Treinadas", len(df_modelos))
        col2.metric("Média de Acurácia", f"{df_plot['acuracia_num'].mean():.2%}")
        col3.metric("Total de Amostras (jogos)", f"{df_modelos['num_amostras'].sum():,.0f}")

        # Nota sobre validação
        st.caption("⚠️ As métricas são calculadas usando validação cronológica (TimeSeriesSplit). Quanto maior a acurácia e ROC-AUC, melhor o modelo.")

# ---------- ABA RELATÓRIO DIÁRIO ----------
with tab_relatorio:
    st.markdown("### 📆 Relatório Diário de Ganhos e Perdas")
    st.caption("Resultado líquido diário baseado nos bilhetes da tabela previsoes (ignora jogos anulados).")

    conn = get_db_connection()
    df_jogos = pd.read_sql_query(
        "SELECT timestamp, ticket_id, confronto, vencedor_previsto, odd_casa, odd_fora, odd_empate, "
        "status_resultado, anulado, data_jogo, hora_jogo FROM previsoes WHERE ticket_id IS NOT NULL AND ticket_id != ''",
        conn
    )
    conn.close()

    if df_jogos.empty:
        st.info("Nenhum bilhete encontrado.")
        st.stop()

    # Remove jogos anulados
    df_jogos = df_jogos[df_jogos["anulado"] != 1]

    # Converte datas
    df_jogos["timestamp_dt"] = pd.to_datetime(df_jogos["timestamp"], errors="coerce")
    df_jogos["data_jogo_dt"] = pd.to_datetime(df_jogos["data_jogo"], format="%d/%m/%Y", errors="coerce")
    # Usa data do jogo se disponível, senão usa data da criação do ticket
    df_jogos["data_efetiva"] = df_jogos["data_jogo_dt"].combine_first(df_jogos["timestamp_dt"].dt.floor("D"))
    df_jogos = df_jogos.dropna(subset=["data_efetiva"])

    if df_jogos.empty:
        st.info("Nenhum bilhete com data válida encontrado.")
        st.stop()

    # Agrupa por bilhete e calcula odd múltipla e lucro
    bilhetes = []
    for t_id, group in df_jogos.groupby("ticket_id"):
        statuses = group["status_resultado"].fillna("PENDENTE").str.strip().str.upper().tolist()
        if any("PENDENTE" in s for s in statuses):
            status_bilhete = "⏳ PENDENTE"
        elif any("RED" in s for s in statuses):
            status_bilhete = "RED ❌"
        else:
            status_bilhete = "GREEN ✅"

        # Recalcula odd do bilhete
        jogos_list = []
        for _, row in group.iterrows():
            jogos_list.append({
                "Confronto": row["confronto"],
                "Vencedor Escolhido": row["vencedor_previsto"],
                "Odd Casa": row["odd_casa"],
                "Odd Fora": row["odd_fora"],
                "Empate": row["odd_empate"]
            })
        odd_multipla = calcular_odd_multipla(jogos_list)

        if status_bilhete == "GREEN ✅":
            lucro = 1.0 * odd_multipla - 1.0
        elif status_bilhete == "RED ❌":
            lucro = -1.0
        else:
            lucro = 0.0

        data_bilhete = group["data_efetiva"].min()
        data_str = data_bilhete.strftime("%d/%m/%Y") if pd.notna(data_bilhete) else "Data inválida"

        bilhetes.append({
            "Data": data_str,
            "Ticket": t_id,
            "Status": status_bilhete,
            "Odd Múltipla": round(odd_multipla, 2),
            "Lucro (R$)": round(lucro, 2)
        })

    if not bilhetes:
        st.info("Nenhum bilhete finalizado encontrado.")
        st.stop()

    df_rel = pd.DataFrame(bilhetes)
    df_rel["Data_dt"] = pd.to_datetime(df_rel["Data"], format="%d/%m/%Y", errors="coerce")
    df_rel = df_rel.dropna(subset=["Data_dt"])
    df_rel = df_rel.sort_values("Data_dt")

    col1, col2 = st.columns(2)
    with col1:
        data_min = df_rel["Data_dt"].min().date()
        data_max = df_rel["Data_dt"].max().date()
        hoje = get_brt_time().date()
        if data_max < hoje:
            data_max = hoje
        intervalo = st.date_input(
            "📅 Intervalo de datas",
            [data_min, data_max],
            min_value=data_min,
            max_value=data_max,
            key="relatorio_intervalo"
        )
    with col2:
        status_filtro = st.multiselect(
            "✅ Status",
            ["GREEN ✅", "RED ❌", "⏳ PENDENTE"],
            default=["GREEN ✅", "RED ❌", "⏳ PENDENTE"],
            key="relatorio_status"
        )

    if len(intervalo) == 2:
        inicio, fim = intervalo
        mask = (df_rel["Data_dt"].dt.date >= inicio) & (df_rel["Data_dt"].dt.date <= fim)
        df_rel = df_rel[mask]
    df_rel = df_rel[df_rel["Status"].isin(status_filtro)]

    total_bilhetes = len(df_rel)
    lucro_total = df_rel["Lucro (R$)"].sum()
    greens = len(df_rel[df_rel["Status"] == "GREEN ✅"])
    reds = len(df_rel[df_rel["Status"] == "RED ❌"])
    pendentes = len(df_rel[df_rel["Status"] == "⏳ PENDENTE"])

    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    col_m1.metric("🎫 Bilhetes", total_bilhetes)
    col_m2.metric("✅ Greens", greens)
    col_m3.metric("❌ Reds", reds)
    col_m4.metric("💰 Lucro Líquido (R$)", f"{lucro_total:+.2f}")

    if pendentes > 0:
        st.info(f"📌 Existem {pendentes} bilhetes ainda pendentes (não contabilizados no lucro).")

    # Agregação diária
    df_diario = df_rel.groupby("Data_dt").agg(
        bilhetes=("Ticket", "count"),
        greens=("Status", lambda x: (x == "GREEN ✅").sum()),
        reds=("Status", lambda x: (x == "RED ❌").sum()),
        pendentes=("Status", lambda x: (x == "⏳ PENDENTE").sum()),
        lucro=("Lucro (R$)", "sum")
    ).reset_index().sort_values("Data_dt")

    st.subheader("📋 Tabela Diária")
    st.dataframe(
        df_diario.style.format({"lucro": "R$ {:+.2f}"}),
        width='stretch',
        hide_index=True
    )

    st.subheader("📊 Gráfico de Lucro Diário")
    fig = px.bar(
        df_diario,
        x="Data_dt",
        y="lucro",
        title="Lucro Líquido Diário",
        color="lucro",
        color_continuous_scale=[(0, 'red'), (0.5, 'yellow'), (1, 'green')],
        text=df_diario["lucro"].apply(lambda x: f"R$ {x:+.2f}")
    )
    fig.update_traces(textposition='outside')
    fig.update_layout(coloraxis_showscale=False, template='plotly_dark')
    st.plotly_chart(fig, width='stretch')
with tab_teste:
    st.header("🧪 Teste de Estatísticas com Fallback (AllSports → Sofascrape)")
    st.markdown("Teste a obtenção de estatísticas de uma partida usando AllSports e, se falhar, Sofascrape.")

    col1, col2 = st.columns([2, 1])
    with col1:
        match_id_input = st.text_input("Match ID (AllSports)", placeholder="Ex: 14023936")
    with col2:
        testar = st.button("🔍 Testar", type="primary", width='stretch')

    if testar and match_id_input:
        match_id = str(match_id_input).strip()
        
        # Buscar informações da partida na AllSports
        with st.status("🔍 Buscando informações da partida...", expanded=True) as status:
            status.write("1. Obtendo dados do jogo na AllSports...")
            match_info = obter_info_partida(match_id)
            
            if not match_info:
                st.error(f"Não foi possível obter dados do jogo {match_id}")
            else:
                st.success(f"✅ Jogo: {match_info['home_team']} vs {match_info['away_team']}")
                
                # 2. Tentar AllSports
                status.write("2. Tentando obter estatísticas via AllSports...")
                stats = obter_estatisticas_partida_all_sports(match_id)
                fonte = "AllSports"
                
                if not stats or all(v == 0 for v in stats.values()):
                    status.write("❌ AllSports sem dados (ou todos os valores zero).")
                    status.write("3. Tentando Sofascrape (SofaScore)...")
                    stats = buscar_estatisticas_sofascrape(match_id)
                    if stats and any(v != 0 for v in stats.values()):
                        fonte = "Sofascrape"
                        status.write("✅ Sofascrape retornou dados!")
                    else:
                        status.write("❌ Sofascrape também não retornou dados.")
                        fonte = "Nenhuma"
                
                if stats and any(v != 0 for v in stats.values()):
                    status.update(label=f"✅ Dados obtidos da {fonte}", state="complete")
                    
                    st.subheader(f"📊 Estatísticas obtidas - Fonte: {fonte}")
                    
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.markdown(f"**🏠 {match_info['home_team']}**")
                        st.metric("Posse de bola", f"{stats.get('posse_home', 0):.1f}%")
                        st.metric("xG (Expected Goals)", f"{stats.get('xg_home', 0):.2f}")
                        st.metric("Total de chutes", f"{stats.get('chutes_home', 0):.0f}")
                        st.metric("Chutes ao gol", f"{stats.get('chutes_gol_home', 0):.0f}")
                        st.metric("Escanteios", f"{stats.get('cantos_home', 0):.0f}")
                        st.metric("Faltas", f"{stats.get('faltas_home', 0):.0f}")
                        st.metric("Cartões amarelos", f"{stats.get('cartoes_home', 0):.0f}")
                    
                    with col_b:
                        st.markdown(f"**✈️ {match_info['away_team']}**")
                        st.metric("Posse de bola", f"{stats.get('posse_away', 0):.1f}%")
                        st.metric("xG (Expected Goals)", f"{stats.get('xg_away', 0):.2f}")
                        st.metric("Total de chutes", f"{stats.get('chutes_away', 0):.0f}")
                        st.metric("Chutes ao gol", f"{stats.get('chutes_gol_away', 0):.0f}")
                        st.metric("Escanteios", f"{stats.get('cantos_away', 0):.0f}")
                        st.metric("Faltas", f"{stats.get('faltas_away', 0):.0f}")
                        st.metric("Cartões amarelos", f"{stats.get('cartoes_away', 0):.0f}")
                    
                    with st.expander("📄 Ver JSON completo"):
                        st.json(stats)
                else:
                    status.update(label="❌ Nenhuma fonte retornou dados", state="error")
                    st.warning("Não foi possível obter estatísticas para esta partida em nenhuma fonte.")

    elif testar and not match_id_input:
        st.warning("Digite um Match ID válido")
