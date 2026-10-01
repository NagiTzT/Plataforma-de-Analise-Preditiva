# robo_auto.py - Versão profissional sincronizada com app.py (ML sem vazamento, features completas, paginação)

import collections
import contextvars
import hashlib
import json
import logging
import math
import os
import random
import re
import sqlite3
import sys
import threading
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from difflib import get_close_matches
from io import BytesIO

import joblib
import numpy as np
import pandas as pd
import pycountry
import requests
import xgboost as xgb
from dotenv import load_dotenv
from api_key_config import configured_keys
from playwright.sync_api import sync_playwright
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from ml_evolution import (
    MODEL_VERSION, balanced_sample_weights, ensure_evolution_tables, evaluate_evolution, get_active_confidence, get_active_model_identity,
    monitor_live_predictions, persist_evolution_result, record_skipped_evolution,
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
from radar_audit_scope import (
    ensure_radar_audit_schema,
    get_latest_radar_run_ids,
    radar_run_placeholders,
)
from radar_prediction_integrity import deduplicate_predictions
from operational_backtest import operational_snapshot_report
from allsports_api import (
    SCHEDULE_CACHE_DB_PATH,
    fetch_allsports_postmatch_resources,
    fetch_allsports_pregame_context,
    fetch_competition_events_for_date,
    fetch_football_events_for_date,
    fetch_recent_team_events_for_game,
    match_detail_url,
    match_resource_url,
    matches_odds_date_url,
    radar_window_brt,
    schedule_dates_for_brt_window,
    team_matches_url,
    tournament_seasons_url,
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
    get_postmortem_summary,
    get_pregame_features as get_sofascore_pregame_features,
    init_sofascore_db,
    load_prediction_snapshot_features,
    load_prediction_snapshot_metadata,
    monitor_pregame_context_sources,
    reclassify_stored_postmortems,
    save_prediction_snapshot,
)

# Evita encerramento por UnicodeEncodeError em terminais Windows/VS Code que
# iniciam com CP-1252, já que o robô usa emojis nas mensagens operacionais.
for _terminal_stream in (sys.stdout, sys.stderr):
    try:
        _terminal_stream.reconfigure(encoding='utf-8', errors='replace')
    except (AttributeError, ValueError, OSError):
        pass

PLAYWRIGHT_SEMAPHORE = threading.Semaphore(4)
sqlite3.register_adapter(datetime, lambda dt: dt.isoformat())
sqlite3.register_converter("timestamp", lambda v: datetime.fromisoformat(v.decode()))

PAIS_PARA_CODIGO = {}
CACHE_ACTIVE_SEASON = {}
_saved_league_mappings = set()
_saved_league_mappings_lock = threading.Lock()

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

def _carregar_segredo_streamlit(nome):
    """Permite ao robô reutilizar a configuração segura já usada pelo app."""
    try:
        import tomllib
        caminho = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".streamlit", "secrets.toml")
        with open(caminho, "rb") as arquivo:
            return str(tomllib.load(arquivo).get(nome, "")).strip()
    except (OSError, ValueError, TypeError):
        return ""

RAPIDAPI_HOST = os.getenv("RAPIDAPI_HOST", "allsportsapi2.p.rapidapi.com").strip()
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "").strip() or _carregar_segredo_streamlit("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip() or _carregar_segredo_streamlit("TELEGRAM_CHAT_ID")
RAPIDAPI_KEYS = configured_keys("RAPIDAPI_KEYS", "allsports")
RAPIDAPI_KEY = RAPIDAPI_KEYS[0] if RAPIDAPI_KEYS else ""
RAPIDAPI_DAILY_LIMIT = 100
# O plano permite 5 req/s. 0,21 s mantém uma pequena margem de segurança.
RAPIDAPI_MIN_INTERVAL_SECONDS = float(os.getenv("RAPIDAPI_MIN_INTERVAL_SECONDS", "0.21"))
MIN_ML_CONFIDENCE = max(50, int(os.getenv("MIN_ML_CONFIDENCE", "50")))
MIN_DRAW_CONFIDENCE = int(os.getenv("MIN_DRAW_CONFIDENCE", "50"))
MAX_TICKETS_PER_RUN = max(0, int(os.getenv("MAX_TICKETS_PER_RUN", "0")))
PERMITIR_EMPATES_BILHETE = os.getenv("PERMITIR_EMPATES_BILHETE", "0") == "1"
PUBLICAR_TODAS_PREVISOES = True
AUDIT_RESULT_GRACE_SECONDS = max(7200, int(os.getenv("AUDIT_RESULT_GRACE_SECONDS", "10800")))
AUDIT_MAX_ATTEMPTS = max(3, int(os.getenv("AUDIT_MAX_ATTEMPTS", "7")))
AUDIT_RECENT_DAYS = max(2, int(os.getenv("AUDIT_RECENT_DAYS", "10")))
AUDIT_POSTMORTEM_BACKFILL_LIMIT = max(
    0, int(os.getenv("AUDIT_POSTMORTEM_BACKFILL_LIMIT", "60"))
)
AUDIT_SCHEDULE_TIME = os.getenv("AUDIT_SCHEDULE_TIME", "17:00").strip() or "17:00"
USAR_MODELOS_POR_LIGA = os.getenv("USAR_MODELOS_POR_LIGA", "0") == "1"
ML_MIN_NEW_SAMPLES = max(20, int(os.getenv("ML_MIN_NEW_SAMPLES", "40")))
ML_CONTEXT_FEATURE_MIN_SAMPLES = max(
    50, int(os.getenv("ML_CONTEXT_FEATURE_MIN_SAMPLES", "200"))
)
HEADERS = {
    "x-rapidapi-host": RAPIDAPI_HOST,
    "x-rapidapi-key": RAPIDAPI_KEY,
    "Content-Type": "application/json"
}
DB_NAME = 'ia_sports_v5.db'

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
CACHE_ESTATISTICAS_SOFASCORE = {}
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
db_write_lock = threading.RLock()
MAX_WORKERS = 4   # conforme app.py, limitado para evitar sobrecarga

def get_brt_time():
    return datetime.now(timezone(timedelta(hours=-3)))

def log_api_error(endpoint, status_code, message=""):
    try:
        with open('api_errors.log', 'a', encoding='utf-8') as f:
            f.write(f"{get_brt_time().strftime('%Y-%m-%d %H:%M:%S')},{endpoint},{status_code},{message}\n")
    except Exception as e:
        logger.error(f"Erro ao logar erro de API: {e}")

def _notify_no_rapidapi_keys(endpoint):
    """Emite um único aviso por dia, mesmo com centenas de workers."""
    global _no_keys_notice_date
    usage_date = get_brt_time().strftime("%Y-%m-%d")
    with _no_keys_notice_lock:
        if _no_keys_notice_date == usage_date:
            return
        _no_keys_notice_date = usage_date
    message = "Todas as chaves atingiram o limite diário ou estão temporariamente bloqueadas"
    log_api_error(endpoint, "NO_KEYS", message)
    logger.warning("RAPIDAPI: %s. A coleta atual será ignorada; o robô continuará ativo.", message)
    print(f"⚠️ RAPIDAPI: {message}. A coleta atual será ignorada; o robô continuará ativo.")

def get_db_connection():
    conn = sqlite3.connect(DB_NAME, timeout=60, check_same_thread=False)
    conn.execute("PRAGMA busy_timeout = 30000")
    return conn

def init_db():
    with get_db_connection() as conn:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        cursor = conn.cursor()
        # Tabela cache_estatisticas_partida
        cursor.execute('''CREATE TABLE IF NOT EXISTS cache_estatisticas_partida (
            match_id TEXT PRIMARY KEY,
            stats_json TEXT,
            data_captura DATETIME
        )''')
        # Tabela previsoes
        cursor.execute('''CREATE TABLE IF NOT EXISTS previsoes (
            id INTEGER PRIMARY KEY AUTOINCREMENT, match_id TEXT UNIQUE, timestamp DATETIME, confronto TEXT, liga TEXT,
            odd_casa REAL, odd_fora REAL, odd_empate REAL, vencedor_previsto TEXT, confianca INTEGER,
            scout_report TEXT, status_resultado TEXT DEFAULT 'PENDENTE', placar_real TEXT DEFAULT '-',
            aprendizado_ia TEXT, ticket_id TEXT, telegram_msg_id TEXT, data_jogo TEXT, hora_jogo TEXT, lesoes_jogadores TEXT
        )''')
        
        # Colunas adicionais (se não existirem)
        colunas = [
            ('antecipado_detectado', 'INTEGER DEFAULT 0'),
            ('anulado', 'INTEGER DEFAULT 0'),
            ('telegram_enviado', 'INTEGER DEFAULT 0'),
            ('selecionado_radar', 'INTEGER DEFAULT 0'),
            ('ml_model_version', 'INTEGER DEFAULT 0'),
            ('ml_model_id', 'TEXT'),
            ('tournament_id', 'TEXT'),
            ('season_id', 'TEXT'),
            ('quase_acerto_notificado', 'INTEGER DEFAULT 0'),
             ('start_timestamp', 'INTEGER DEFAULT 0'),
             ('audit_attempts', 'INTEGER DEFAULT 0'),
             ('audit_next_at', 'INTEGER DEFAULT 0'),
             ('audit_last_error', 'TEXT'),
             ('draw_risk_score', 'REAL DEFAULT 0'),
             ('context_quality_score', 'REAL DEFAULT 0'),
             ('context_conflict_score', 'REAL DEFAULT 0'),
             ('analysis_version', 'TEXT'),
             ('postmortem_verdict', 'TEXT'),
             ('postmortem_process_score', 'REAL'),
             ('postmortem_coverage', 'REAL')
         ]
        for col, tipo in colunas:
            try:
                cursor.execute(f"ALTER TABLE previsoes ADD COLUMN {col} {tipo}")
            except sqlite3.OperationalError:
                pass
        
        # Tabela quarentena_liga
        cursor.execute('''CREATE TABLE IF NOT EXISTS quarentena_liga (
            id INTEGER PRIMARY KEY AUTOINCREMENT, match_id TEXT UNIQUE, timestamp DATETIME, confronto TEXT, liga TEXT,
            odd_casa REAL, odd_fora REAL, odd_empate REAL, vencedor_previsto TEXT, confianca INTEGER,
            scout_report TEXT, status_resultado TEXT DEFAULT 'PENDENTE', placar_real TEXT DEFAULT '-',
            aprendizado_ia TEXT, data_jogo TEXT, hora_jogo TEXT
        )''')
        for col in [('tournament_id','TEXT'),('season_id','TEXT')]:
            try:
                cursor.execute(f"ALTER TABLE quarentena_liga ADD COLUMN {col}")
            except:
                pass
        
        # Demais tabelas
        cursor.execute('''CREATE TABLE IF NOT EXISTS autopsias_liga (
            liga TEXT PRIMARY KEY, relatorio_geral TEXT, jogos_processados INTEGER DEFAULT 0, ultima_atualizacao DATETIME
        )''')
        cursor.execute('''CREATE TABLE IF NOT EXISTS aprendizado_global (
            id INTEGER PRIMARY KEY AUTOINCREMENT, data_analise DATETIME, relatorio TEXT, jogos_analisados INTEGER
        )''')
        cursor.execute('''CREATE TABLE IF NOT EXISTS aprendizado_liga (
            liga TEXT PRIMARY KEY, data_analise DATETIME, total_jogos INTEGER, taxa_acerto REAL,
            xG_casa_medio REAL, xG_fora_medio REAL, posse_casa_medio REAL, posse_fora_medio REAL, regras TEXT
        )''')
        cursor.execute('''CREATE TABLE IF NOT EXISTS aprendizado_time (
            time TEXT PRIMARY KEY, liga TEXT, data_analise DATETIME, total_jogos INTEGER, taxa_acerto REAL,
            xG_medio REAL, posse_media REAL, chutes_medio REAL, motivo_principal TEXT
        )''')
        cursor.execute('''CREATE TABLE IF NOT EXISTS modelos_ml (
            liga TEXT PRIMARY KEY, data_treinamento DATETIME, num_amostras INTEGER,
            modelo_blob BLOB, scaler_params TEXT, feature_order TEXT, acuracia REAL, log_loss REAL
        )''')
        cursor.execute('''CREATE TABLE IF NOT EXISTS training_weights (
            match_id TEXT PRIMARY KEY, peso REAL DEFAULT 1.0, data_ultima_atualizacao DATETIME
        )''')
        cursor.execute('''CREATE TABLE IF NOT EXISTS cache_xg_times (
            team_id TEXT, tournament_id TEXT, season_id TEXT, xg_medio REAL, gols_marcados_medio REAL,
            gols_sofridos_medio REAL, ultima_atualizacao DATETIME, PRIMARY KEY (team_id, tournament_id, season_id)
        )''')
        cursor.execute('''CREATE TABLE IF NOT EXISTS cache_jogos_liga (
            tournament_id TEXT, season_id TEXT, last_match_id TEXT PRIMARY KEY,
            empates_acumulados INTEGER DEFAULT 0, total_jogos_acumulados INTEGER DEFAULT 0, ultima_atualizacao DATETIME
        )''')
        cursor.execute('''CREATE TABLE IF NOT EXISTS mapeamento_ligas (
            liga_nome TEXT PRIMARY KEY, tournament_id TEXT, season_id TEXT, ultima_atualizacao DATETIME
        )''')
        cursor.execute('''CREATE TABLE IF NOT EXISTS training_data (
            match_id TEXT PRIMARY KEY, liga TEXT, data_jogo DATETIME, home_team TEXT, away_team TEXT,
            home_score INTEGER, away_score INTEGER, odd_casa REAL, odd_empate REAL, odd_fora REAL,
            features TEXT, usado_treinamento INTEGER DEFAULT 0, data_importacao DATETIME DEFAULT CURRENT_TIMESTAMP
        )''')
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_training_data_date ON training_data(data_jogo)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_training_data_home_date ON training_data(home_team, data_jogo)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_training_data_away_date ON training_data(away_team, data_jogo)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_training_data_liga_date ON training_data(liga, data_jogo)")
        cursor.execute('''CREATE TABLE IF NOT EXISTS ml_team_ratings (
            team_name TEXT PRIMARY KEY, elo REAL NOT NULL, updated_at DATETIME NOT NULL)''')
        cursor.execute('''CREATE TABLE IF NOT EXISTS elo_rating (
            team_id TEXT PRIMARY KEY, elo INTEGER, last_update DATETIME)''')
        cursor.execute('''CREATE TABLE IF NOT EXISTS rapidapi_key_usage (
            key_id TEXT, usage_date TEXT, request_count INTEGER DEFAULT 0,
            status TEXT DEFAULT 'active', blocked_until REAL DEFAULT 0,
            last_http_status TEXT, PRIMARY KEY (key_id, usage_date))''')
        cursor.execute('''CREATE TABLE IF NOT EXISTS drift_reference (
            liga TEXT PRIMARY KEY, feature_stats TEXT, data_referencia DATETIME
        )''')
        cursor.execute('''CREATE TABLE IF NOT EXISTS estatisticas_medias_liga (
            liga TEXT PRIMARY KEY, num_jogos INTEGER, xg_casa_medio REAL, xg_fora_medio REAL,
            posse_casa_medio REAL, posse_fora_medio REAL, chutes_casa_medio REAL, chutes_fora_medio REAL,
            ef_of_home_medio REAL, ef_df_home_medio REAL, ef_of_away_medio REAL, ef_df_away_medio REAL,
            empates_medio REAL, avg_cards_medio REAL
        )''')
        
        # Tabela de controle (para armazenar run_id, etc.)
        cursor.execute('''CREATE TABLE IF NOT EXISTS controle (
            chave TEXT PRIMARY KEY,
            valor TEXT
        )''')
        ensure_evolution_tables(conn, MIN_ML_CONFIDENCE)

        for table, col, tipo in [
            ('previsoes', 'unique_tournament_id', 'TEXT'),
            ('training_data', 'tournament_id', 'TEXT'), ('training_data', 'season_id', 'TEXT'),
            ('training_data', 'unique_tournament_id', 'TEXT'),
            ('training_weights', 'error_margin', 'REAL DEFAULT 0')]:
            try:
                cursor.execute(f"ALTER TABLE {table} ADD COLUMN {col} {tipo}")
            except sqlite3.OperationalError:
                pass
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_training_data_liga_used ON training_data(liga, usado_treinamento)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_previsoes_status ON previsoes(status_resultado, antecipado_detectado)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_previsoes_ticket ON previsoes(ticket_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_previsoes_radar_status ON previsoes(selecionado_radar, status_resultado)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_previsoes_model_status ON previsoes(ml_model_id, status_resultado)")
        cursor.execute("""CREATE INDEX IF NOT EXISTS idx_previsoes_audit
                          ON previsoes(status_resultado, start_timestamp, audit_next_at, audit_attempts)""")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_cache_stats_date ON cache_estatisticas_partida(data_captura)")
        ensure_radar_audit_schema(conn)

        for alter_sql in (
            "ALTER TABLE modelos_ml ADD COLUMN roc_auc REAL",
            "ALTER TABLE modelos_ml ADD COLUMN model_version INTEGER DEFAULT 1",
        ):
            try:
                cursor.execute(alter_sql)
            except sqlite3.OperationalError:
                pass

init_db()
init_soccer_context_db(DB_NAME)
init_sofascore_db(DB_NAME)

def obter_streaks(match_id):
    data = safe_api_get(match_resource_url(RAPIDAPI_HOST, match_id, "streaks"))
    if not data or 'streaks' not in data:
        return None, None
    streaks = {}
    for item in data['streaks']:
        team = item.get('team')
        if team == 'home':
            streaks['home'] = item.get('streak', '')
        elif team == 'away':
            streaks['away'] = item.get('streak', '')
    return streaks.get('home'), streaks.get('away')

def obter_codigo_pais(nome_pais):
    """
    Retorna o código ISO alpha-2 (ex: 'BR') para o nome do país fornecido.
    Utiliza um dicionário de cache para evitar consultas repetidas.
    """
    if not nome_pais:
        return None
    nome_pais = nome_pais.strip().lower()
    if nome_pais in PAIS_PARA_CODIGO:
        return PAIS_PARA_CODIGO[nome_pais]

    codigo = None
    try:
        country = pycountry.countries.get(name=nome_pais.title())
        if country:
            codigo = country.alpha_2
        else:
            # Busca alternativa: procura por nome comum ou código de 3 letras.
            for c in pycountry.countries:
                if c.name.lower() == nome_pais or getattr(c, 'common_name', '').lower() == nome_pais:
                    codigo = c.alpha_2
                    break
                if hasattr(c, 'alpha_3') and c.alpha_3.lower() == nome_pais:
                    codigo = c.alpha_2
                    break
    except Exception as e:
        print(f"Erro ao buscar código do país para '{nome_pais}': {e}")

    PAIS_PARA_CODIGO[nome_pais] = codigo
    return codigo

def obter_url_bandeira(nome_pais):
    """
    Retorna a URL da imagem da bandeira para o país fornecido.
    """
    codigo_pais = obter_codigo_pais(nome_pais)
    if codigo_pais:
        return f"https://allsportsapi2.p.rapidapi.com/api/country/{codigo_pais.lower()}/flag"
    return None

# Função para baixar a imagem da bandeira
def baixar_bandeira(codigo_pais):
    if not codigo_pais:
        return None
    url = f"https://allsportsapi2.p.rapidapi.com/api/country/{codigo_pais.lower()}/flag"
    try:
        # Também passa pelo controle persistente de cota e pelo rodízio de chaves.
        return safe_api_get(url, max_retries=1, timeout=5, return_bytes=True)
    except Exception as e:
        print(f"Erro ao baixar bandeira para {codigo_pais}: {e}")
    return None

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

# ----------------------------------------------------------------------
# TELEGRAM ENGINE
# ----------------------------------------------------------------------
def disparar_telegram(bilhetes, token, chat_id):
    """
    Envia bilhetes para o Telegram com emojis de bandeira ao lado dos times.
    """
    import requests
    import time

    # Dicionário de países -> emoji de bandeira (para os mais comuns)
    BANDEIRAS = {
        'brasil': '🇧🇷', 'brazil': '🇧🇷',
        'inglaterra': '🏴󠁧󠁢󠁥󠁮󠁧󠁿', 'england': '🏴󠁧󠁢󠁥󠁮󠁧󠁿',
        'espanha': '🇪🇸', 'spain': '🇪🇸',
        'alemanha': '🇩🇪', 'germany': '🇩🇪',
        'italia': '🇮🇹', 'italy': '🇮🇹',
        'frança': '🇫🇷', 'france': '🇫🇷',
        'portugal': '🇵🇹', 'holanda': '🇳🇱', 'netherlands': '🇳🇱',
        'argentina': '🇦🇷', 'argentina': '🇦🇷',
        'méxico': '🇲🇽', 'mexico': '🇲🇽',
        'estados unidos': '🇺🇸', 'usa': '🇺🇸',
        'japão': '🇯🇵', 'japan': '🇯🇵',
        'austrália': '🇦🇺', 'australia': '🇦🇺',
    }

    def obter_bandeira(pais):
        """Retorna emoji de bandeira para o nome do país (case insensitive)."""
        if not pais:
            return ''
        pais_lower = pais.strip().lower()
        # Tenta encontrar no dicionário
        for key, emoji in BANDEIRAS.items():
            if key in pais_lower:
                return emoji
        # Fallback: tenta extrair código de país de 2 letras se o nome for curto
        if len(pais_lower) == 2:
            # Converte código para emoji regional (ex: 'br' -> '🇧🇷')
            return ''.join(chr(127397 + ord(letra)) for letra in pais_lower.upper())
        return ''

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    sucessos = 0
    conn = get_db_connection()
    cursor = conn.cursor()

    for b in bilhetes:
        b["_telegram_enviado"] = False
        odd_multipla = 1.0
        odds_disponiveis = True
        jogos = b.get('Jogos', [])
        # Calcular odd múltipla e preparar jogos
        for j in jogos:
            casa_nome = str(j.get('Confronto', '')).split(' vs ')[0].strip().lower()
            pick = str(j.get('Vencedor Escolhido', '')).strip().lower()
            if 'empate' in pick:
                odd_jogo = float(j.get('Empate', 1.0))
            elif casa_nome in pick or pick in casa_nome:
                odd_jogo = float(j.get('Odd Casa', 1.0))
            else:
                odd_jogo = float(j.get('Odd Fora', 1.0))
            if odd_jogo <= 1.0:
                odd_jogo = max(float(j.get('Odd Casa', 1.0)), float(j.get('Odd Fora', 1.0)))
            if odd_jogo <= 1.0:
                odds_disponiveis = False
                odd_jogo = 0.0
            odd_multipla *= max(odd_jogo, 1.0)
            j['Odd_Calculada'] = odd_jogo

        # Construir mensagem
        prob_conjunta = float(b.get('Probabilidade Conjunta', 0.0) or np.prod([
            max(0.0, min(1.0, float(j.get('Confiança', 0)) / 100.0)) for j in b.get('Jogos', [])]))
        cabecalho_odd = (f"🔥 *Odd Múltipla: {odd_multipla:.2f}*" if odds_disponiveis
                         else "🧠 *Modelo contextual: odds ignoradas*")
        texto_msg = (f"⚡ *{b.get('Nome')}*\n{cabecalho_odd}\n"
                     f"📐 *Prob. conjunta estimada: {prob_conjunta:.2%}*\n━━━━━━━━━━━━━━━━━━\n")
        for j in jogos:
            confronto = j.get('Confronto', '')
            partes = confronto.split(' vs ')
            if len(partes) == 2:
                time_casa, time_fora = partes[0].strip(), partes[1].strip()
            else:
                time_casa, time_fora = confronto, ''
            # Extrair país da liga (ex: "Brasil - Campeonato Brasileiro")
            liga_exata = j.get('Liga_Exata', j.get('Liga', ''))
            pais = liga_exata.split(' - ')[0] if ' - ' in liga_exata else ''
            bandeira = obter_bandeira(pais)
            pick = j.get('Vencedor Escolhido', '').upper()
            odd = j.get('Odd_Calculada', 1.0)
            texto_odd = f" *(Odd: {odd:.2f})*" if float(odd or 0) > 1.0 else ""
            conf = j.get('Confiança', 0)
            dia = j.get('Dia_Str', '')
            hora = j.get('Hora BRT', '')
            liga = j.get('Liga_Exata', j.get('Liga', ''))

            texto_msg += (
                f"⚽ {bandeira} *{time_casa}* vs *{time_fora}*\n"
                f"⏰ {dia} {hora} BRT | 🌍 🏆 {liga}\n"
                f"🎯 *Pick:* `{pick}`{texto_odd} - 📊 Confiança: {conf}%\n\n"
            )

        ticket_id = str(b.get('Nome') or '')
        payload_hash = delivery_fingerprint(jogos)
        autorizado, motivo, message_id_existente = claim_ticket_delivery(
            DB_NAME, ticket_id, payload_hash
        )
        if not autorizado:
            print(f"Telegram ignorado para {ticket_id}: envio {motivo}.")
            if motivo == "sent":
                with db_write_lock:
                    cursor.execute("""UPDATE previsoes
                        SET telegram_msg_id=COALESCE(?,telegram_msg_id), telegram_enviado=1
                        WHERE ticket_id=?""", (message_id_existente, ticket_id))
                    conn.commit()
                b["_telegram_enviado"] = True
            continue

        # Tentativa de envio
        for _ in range(3):
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
                print(f"Erro ao enviar Telegram: {e}")
                # Nao repetir automaticamente: o Telegram pode ter aceitado a
                # mensagem antes de a resposta se perder, causando duplicata.
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
    todos_resolvidos = all(('PENDENTE' not in s) for s in statuses)
    todos_green = all(('GREEN' in s) for s in statuses) if todos_resolvidos else False

    odd_multipla = 1.0
    odds_disponiveis = True
    linhas_jogos = ""
    for j in jogos_nao_anulados:
        confronto, liga, pick, o_c, o_f, o_e, status, placar, _, d_j, h_j, conf, _ = j
        casa_nome = str(confronto).split(' vs ')[0].strip().lower()
        pick_lower = str(pick).strip().lower()
        if 'empate' in pick_lower:
            odd_jogo = float(o_e)
        elif casa_nome in pick_lower:
            odd_jogo = float(o_c)
        else:
            odd_jogo = float(o_f)
        if odd_jogo <= 1.0:
            odd_jogo = max(float(o_c), float(o_f))
        if odd_jogo <= 1.0:
            odds_disponiveis = False
            odd_jogo = 0.0
        odd_multipla *= max(odd_jogo, 1.0)
        icon = "✅" if "GREEN" in status else "❌" if "RED" in status else "⏳" if "PENDENTE" in status else "🚫" if "ANULADO" in status else "ℹ️"
        d_j_str = d_j if d_j else "--/--"
        h_j_str = h_j if h_j else "--:--"
        linhas_jogos += f"⚽ *{confronto}* ({placar})\n"
        linhas_jogos += f"⏰ {d_j_str} {h_j_str} BRT | 🌍 🏆 {liga}\n"
        texto_odd = f" *(Odd: {odd_jogo:.2f})*" if odd_jogo > 1.0 else ""
        linhas_jogos += f"{icon} *Pick:* `{pick}`{texto_odd} - 📊 Confiança: {conf}% - {status.replace('ARQUIVADO ', '')}\n\n"

    if todos_green:
        titulo = f"✅ GREEN | {ticket_id}"
    elif todos_resolvidos:
        titulo = f"🔴 RED | {ticket_id}"
    else:
        titulo = ticket_id

    cabecalho_odd = (f"🔥 *Odd Múltipla: {odd_multipla:.2f}*" if odds_disponiveis
                     else "🧠 *Modelo contextual: odds ignoradas*")
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

    elos = {}
    for home, away, hs, aws, _ in jogos:
        # Resultado para o time da casa
        if hs > aws:
            res_h = 1.0
            res_a = 0.0
        elif hs < aws:
            res_h = 0.0
            res_a = 1.0
        else:
            res_h = 0.5
            res_a = 0.5

        elo_h = elos.get(home, 1500)
        elo_a = elos.get(away, 1500)

        expected_h = 1 / (1 + 10 ** ((elo_a - elo_h) / 400))
        expected_a = 1 / (1 + 10 ** ((elo_h - elo_a) / 400))

        elos[home] = elo_h + 32 * (res_h - expected_h)
        elos[away] = elo_a + 32 * (res_a - expected_a)

    # Salvar na tabela elo_rating (cria se não existir)
    conn = get_db_connection()
    # Garantir que a tabela existe
    conn.execute("""CREATE TABLE IF NOT EXISTS elo_rating (
        team_id TEXT PRIMARY KEY,
        elo INTEGER,
        last_update DATETIME
    )""")
    for team, elo in elos.items():
        conn.execute("INSERT OR REPLACE INTO elo_rating (team_id, elo, last_update) VALUES (?, ?, ?)",
                     (team, int(elo), get_brt_time().isoformat()))
    conn.commit()
    conn.close()
    print(f"Elo recalculado para {len(elos)} times.")

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
    statuses = [j[6] for j in jogos_nao_anulados]
    if any('PENDENTE' in s for s in statuses) or any('RED' in s for s in statuses): return

    odd_multipla = 1.0
    for j in jogos_nao_anulados:
        confronto, pick, o_c, o_f, o_e, _, _, _ = j
        casa_nome = str(confronto).split(' vs ')[0].strip().lower()
        pick_lower = str(pick).strip().lower()
        if 'empate' in pick_lower:
            odd_jogo = float(o_e)
        elif casa_nome in pick_lower:
            odd_jogo = float(o_c)
        else:
            odd_jogo = float(o_f)
        if odd_jogo <= 1.0:
            odd_jogo = max(float(o_c), float(o_f))
        odd_multipla *= max(odd_jogo, 1.0)

    texto_msg = f"✅ <b>BINGO! GREEN NO BOLSO!</b> 💰\n\nO bilhete <b>{ticket_id}</b> bateu com uma Odd Múltipla de <b>{odd_multipla:.2f}</b>! 🔥"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": texto_msg, "parse_mode": "HTML"}
    msg_id_str = jogos[0][5]
    if msg_id_str and msg_id_str != 'None':
        payload["reply_to_message_id"] = int(msg_id_str)

    for _ in range(3):
        try:
            res = requests.post(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage", json=payload)
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
        except: time.sleep(1)

# ----------------------------------------------------------------------
# FUNÇÕES DE API E AUXILIARES
# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
# FUNÇÃO AUXILIAR PARA EXTRAIR ESTATÍSTICAS DA ALLSPORTS (SUA VERSÃO ATUAL)
# ----------------------------------------------------------------------
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

# Cache para estatísticas do SofaScore (evita chamadas repetidas)
SOFASCORE_CACHE = {}
SOFASCORE_CACHE_LOCK = threading.Lock()

def buscar_estatisticas_sofascore(match_id):
    """
    Busca estatísticas da partida no SofaScore usando o match_id da AllSports diretamente.
    Usa semáforo global para evitar concorrência excessiva do Playwright.
    """
    cache_key = str(match_id)
    if cache_key in CACHE_ESTATISTICAS_SOFASCORE:
        return CACHE_ESTATISTICAS_SOFASCORE[cache_key]

    cached = carregar_estatisticas_cache(match_id)
    if cached:
        CACHE_ESTATISTICAS_SOFASCORE[cache_key] = cached
        return cached

    stats = {}
    url = f"https://www.sofascore.com/api/v1/event/{match_id}/statistics"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept": "application/json",
        "Referer": "https://www.sofascore.com/"
    }
    try:
        with PLAYWRIGHT_SEMAPHORE:
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True)
                page = browser.new_page()
                page.set_extra_http_headers(headers)
                page.goto(url, timeout=15000)
                content = page.content()
                browser.close()
                start = content.find('{')
                end = content.rfind('}') + 1
                if start != -1 and end != 0:
                    json_str = content[start:end]
                    data = json.loads(json_str)
                    if 'statistics' in data:
                        stats = extrair_estatisticas_do_json_sofascore(data)
    except Exception as e:
        logger.error(f"Erro SofaScore para {match_id}: {e}")
        stats = {}

    salvar_estatisticas_cache(match_id, stats)
    CACHE_ESTATISTICAS_SOFASCORE[cache_key] = stats
    return stats

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

def buscar_estatisticas_sofascrape(match_id, home_team=None, away_team=None, match_date=None):
    cache_key = str(match_id)
    if cache_key in CACHE_ESTATISTICAS_SOFASCORE:
        return CACHE_ESTATISTICAS_SOFASCORE[cache_key]
    stats = {}
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            page.set_extra_http_headers({"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"})
            url = f"https://www.sofascore.com/api/v1/event/{match_id}/statistics"
            page.goto(url, timeout=10000)
            content = page.content()
            start = content.find('{')
            end = content.rfind('}') + 1
            if start != -1 and end != 0:
                data = json.loads(content[start:end])
                stats = extrair_estatisticas_sofascore(data)
            if not stats and home_team and away_team and match_date:
                dt = datetime.fromtimestamp(match_date).strftime("%Y-%m-%d")
                query = f"{home_team} {away_team} {dt}".replace(" ", "%20")
                search_url = f"https://www.sofascore.com/api/v1/search?q={query}"
                page.goto(search_url, timeout=10000)
                content = page.content()
                start = content.find('{')
                end = content.rfind('}') + 1
                if start != -1 and end != 0:
                    data = json.loads(content[start:end])
                    for res in data.get('results', []):
                        if res.get('type') == 'event':
                            event_id = res.get('id')
                            if event_id:
                                stats_url = f"https://www.sofascore.com/api/v1/event/{event_id}/statistics"
                                page.goto(stats_url, timeout=10000)
                                content2 = page.content()
                                start2 = content2.find('{')
                                end2 = content2.rfind('}') + 1
                                if start2 != -1 and end2 != 0:
                                    data2 = json.loads(content2[start2:end2])
                                    stats = extrair_estatisticas_sofascore(data2)
                                if stats:
                                    break
            browser.close()
    except Exception as e:
        logger.error(f"Erro Sofascrape {match_id}: {e}")
    CACHE_ESTATISTICAS_SOFASCORE[cache_key] = stats
    return stats

def buscar_estatisticas_por_id(event_id):
    """Busca estatísticas usando o event_id correto do SofaScore."""
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            url = f"https://www.sofascore.com/api/v1/event/{event_id}/statistics"
            page.goto(url)
            content = page.content()
            browser.close()
            start = content.find('{')
            end = content.rfind('}') + 1
            if start != -1 and end != 0:
                data = json.loads(content[start:end])
                return extrair_estatisticas_sofascore(data)
    except Exception as e:
        print(f"Erro ao buscar estatísticas por ID {event_id}: {e}")
    return {}
import json
import re
from playwright.sync_api import sync_playwright
from datetime import datetime

CACHE_ESTATISTICAS_SOFASCORE = {}

def extrair_estatisticas_sofascore(data):
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

# ----------------------------------------------------------------------
# CONSTRUÇÃO DA URL DO WHOSCORED A PARTIR DOS DADOS DO JOGO
# ----------------------------------------------------------------------
def construir_url_whoscored(match_info):
    """
    Constrói a URL do WhoScored com base nas informações da partida.
    match_info deve conter: home_team, away_team, data_jogo (timestamp), tournament_name
    """
    try:
        # Formatar data para YYYYMMDD
        data = datetime.fromtimestamp(match_info.get('startTimestamp', 0), tz=timezone.utc)
        data_str = data.strftime("%Y%m%d")
        
        # Limpar nomes dos times para formato da URL
        home = re.sub(r'[^a-zA-Z0-9]', '', match_info.get('home_team', '').lower())
        away = re.sub(r'[^a-zA-Z0-9]', '', match_info.get('away_team', '').lower())
        
        # Padrão comum: https://www.whoscored.com/Matches/{id}/...
        # Infelizmente, o WhoScored não tem um padrão fácil baseado em nomes.
        # Precisamos buscar ou usar uma abordagem alternativa.
        # Uma alternativa é usar o whoscraped com o match_id da AllSports.
        # O whoscraped geralmente precisa da URL completa.
        # Vamos tentar construir a URL usando o match_id (se for o mesmo)
        match_id = match_info.get('id')
        if match_id:
            return f"https://www.whoscored.com/Matches/{match_id}/Live"
        
        # Fallback: buscar via busca no site (mais complexo)
        return None
    except:
        return None

# ----------------------------------------------------------------------
# BUSCAR ESTATÍSTICAS DO WHOSCORED USANDO whoscraped
# ----------------------------------------------------------------------


# ----------------------------------------------------------------------
# BUSCAR ESTATÍSTICAS DO FBREF USANDO soccerdata
# ----------------------------------------------------------------------
def buscar_estatisticas_por_busca(home_team, away_team, timestamp):
    """
    Busca estatísticas no SofaScore pesquisando por nome dos times e data.
    Retorna dicionário com estatísticas.
    """
    dt = datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d")
    query = f"{home_team} {away_team} {dt}".replace(" ", "%20")
    search_url = f"https://www.sofascore.com/api/v1/search?q={query}"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept": "application/json",
        "Referer": "https://www.sofascore.com/"
    }
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            page.set_extra_http_headers(headers)
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
    except Exception as e:
        logger.error(f"Erro busca SofaScore: {e}")
    return {}
# ----------------------------------------------------------------------
# FUNÇÃO PRINCIPAL DE FALLBACK
# ----------------------------------------------------------------------
def obter_estatisticas_com_fallback(match_id, match_info=None):
    # 1. Verificar cache persistente
    cached = carregar_estatisticas_cache(match_id)
    if cached:
        print(f"[ESTATS] {match_id}: cache persistente OK")
        return cached

    # 2. AllSports
    stats = obter_estatisticas_partida_all_sports(match_id)
    if stats and any(v != 0 for v in stats.values()):
        salvar_estatisticas_cache(match_id, stats)
        print(f"[ESTATS] {match_id}: AllSports OK")
        return stats

    # 3. SofaScore (com semáforo)
    if not match_info:
        match_info = obter_info_partida(match_id)
    if match_info:
        with PLAYWRIGHT_SEMAPHORE:
            stats = buscar_estatisticas_sofascore(match_id)
        if stats and any(v != 0 for v in stats.values()):
            salvar_estatisticas_cache(match_id, stats)
            print(f"[ESTATS] {match_id}: SofaScore OK")
            return stats
        # Busca por nome/data (se necessário)
        home_team = match_info.get('home_team')
        away_team = match_info.get('away_team')
        start_ts = match_info.get('startTimestamp')
        if home_team and away_team and start_ts:
            with PLAYWRIGHT_SEMAPHORE:
                stats = buscar_estatisticas_por_busca(home_team, away_team, start_ts)
            if stats and any(v != 0 for v in stats.values()):
                salvar_estatisticas_cache(match_id, stats)
                print(f"[ESTATS] {match_id}: SofaScore via busca OK")
                return stats
    print(f"[ESTATS] {match_id}: FALHA - sem estatísticas")
    salvar_estatisticas_cache(match_id, {})  # cache de falha
    return {}

def listar_features_zeradas(features_dict):
    zeros = [k for k, v in features_dict.items() if v == 0 or v is None]
    if zeros:
        return f"Features zeradas: {', '.join(zeros[:15])}" + ("..." if len(zeros)>15 else "")
    return "Todas as features não-zero."

def extrair_fracional(frac_str):
    try:
        if not frac_str: return 0.0
        if '/' in str(frac_str):
            n, d = str(frac_str).split('/')
            return round((float(n) / float(d)) + 1.0, 2)
        return float(frac_str)
    except: return 0.0

# Cache para temporadas ativas
CACHE_ACTIVE_SEASON = {}

def get_active_season_id(tournament_id):
    """Busca a temporada ativa pela rota válida de temporadas da v2.0."""
    if not tournament_id:
        return None
    if tournament_id in CACHE_ACTIVE_SEASON:
        return CACHE_ACTIVE_SEASON[tournament_id]

    active_season_id = None

    seasons_url = tournament_seasons_url(RAPIDAPI_HOST, tournament_id)
    seasons_data = safe_api_get(seasons_url, max_retries=1, timeout=8)
    if seasons_data and 'seasons' in seasons_data:
        for s in seasons_data['seasons']:
            if s.get('isActive'):
                active_season_id = str(s.get('id'))
                break
        if not active_season_id and seasons_data['seasons']:
            # Pega a mais recente se não houver ativa
            active_season_id = str(max(seasons_data['seasons'], key=lambda x: x.get('id', 0)).get('id'))

    CACHE_ACTIVE_SEASON[tournament_id] = active_season_id
    return active_season_id

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
    if raw > 10_000_000_000:  # epoch em milissegundos
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
    global _key_rotation_cursor
    usage_date = get_brt_time().strftime("%Y-%m-%d"); now_epoch = time.time()
    with _key_rotation_lock:
        conn = get_db_connection()
        try:
            conn.execute("BEGIN IMMEDIATE")
            for offset in range(len(RAPIDAPI_KEYS)):
                idx = (_key_rotation_cursor + offset) % len(RAPIDAPI_KEYS); key = RAPIDAPI_KEYS[idx]; key_id = _rapidapi_key_id(key)
                conn.execute("""INSERT OR IGNORE INTO rapidapi_key_usage
                    (key_id, usage_date, request_count, status, blocked_until) VALUES (?,?,0,'active',0)""", (key_id, usage_date))
                count, status, blocked_until = conn.execute("""SELECT request_count, status, blocked_until
                    FROM rapidapi_key_usage WHERE key_id=? AND usage_date=?""", (key_id, usage_date)).fetchone()
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
                if count >= RAPIDAPI_DAILY_LIMIT or status in ('exhausted', 'invalid'): continue
                if status == 'blocked' and float(blocked_until or 0) > now_epoch: continue
                conn.execute("""UPDATE rapidapi_key_usage SET request_count=request_count+1,
                    status='active' WHERE key_id=? AND usage_date=?""", (key_id, usage_date))
                _key_rotation_cursor = idx; conn.commit(); return key, key_id, idx
            conn.commit(); return None, None, None
        finally:
            conn.close()

def _mark_rapidapi_key(key_id, idx, status, http_status, cooldown=0):
    global _key_rotation_cursor
    usage_date = get_brt_time().strftime("%Y-%m-%d")
    with _key_rotation_lock:
        with get_db_connection() as conn:
            conn.execute("""UPDATE rapidapi_key_usage SET status=?, blocked_until=?, last_http_status=?
                WHERE key_id=? AND usage_date=?""",
                (status, time.time()+cooldown if cooldown else 0, str(http_status), key_id, usage_date))
        if status in ('exhausted', 'invalid', 'blocked'): _key_rotation_cursor = (idx+1) % len(RAPIDAPI_KEYS)

def get_rapidapi_key_usage():
    usage_date = get_brt_time().strftime("%Y-%m-%d")
    with get_db_connection() as conn:
        rows = {r[0]: r[1:] for r in conn.execute("""SELECT key_id, request_count, status,
            blocked_until, last_http_status FROM rapidapi_key_usage WHERE usage_date=?""", (usage_date,))}
    result=[]
    for idx,key in enumerate(RAPIDAPI_KEYS,1):
        count,status,blocked,last_status=rows.get(_rapidapi_key_id(key),(0,'active',0,None))
        if count >= RAPIDAPI_DAILY_LIMIT:
            status = 'exhausted'
        result.append({'key':idx,'used':count,'remaining':max(0,RAPIDAPI_DAILY_LIMIT-count),'status':status,'last_http_status':last_status})
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

def safe_api_get(url, max_retries=3, timeout=(3.05, 20), return_bytes=False,
                 max_http_requests=None):
    """GET com cache TTL, deduplicação entre workers, rate limit e retries."""
    global _last_request_time
    with _api_cache_lock:
        _api_metrics["logical_calls"] += 1
    cached = _cached_api_response(url)
    if cached is not None:
        return cached

    with _get_api_url_lock(url):
        cached = _cached_api_response(url)
        if cached is not None:
            return cached
        failures = 0
        http_attempts = 0
        while failures < max_retries:
            if max_http_requests is not None and http_attempts >= max(0, int(max_http_requests)):
                return None
            api_key, key_id, key_idx = _reserve_rapidapi_key()
            if not api_key:
                _notify_no_rapidapi_keys(url)
                return None
            with _rate_limiter:
                with _rate_lock:
                    now = time.time()
                    elapsed = now - _last_request_time
                    if elapsed < RAPIDAPI_MIN_INTERVAL_SECONDS:
                        time.sleep(RAPIDAPI_MIN_INTERVAL_SECONDS - elapsed)
                    _last_request_time = time.time()
                try:
                    http_attempts += 1
                    with _api_cache_lock:
                        _api_metrics["http_requests"] += 1
                    request_headers = dict(HEADERS); request_headers["x-rapidapi-key"] = api_key
                    res = requests.get(url, headers=request_headers, timeout=timeout)
                    if res.status_code == 200:
                        _sync_rapidapi_response(key_id, key_idx, res)
                        data = res.content if return_bytes else res.json()
                        with _api_cache_lock:
                            _api_response_cache[url] = (time.monotonic(), data)
                        return data
                    if res.status_code == 204:
                        _sync_rapidapi_response(key_id, key_idx, res)
                        # Diferencia "sem eventos" de falha/sem chave para que
                        # o cache persistente não descarte uma coleta completa.
                        return {}
                    if res.status_code == 429:
                        remaining = res.headers.get('x-ratelimit-requests-remaining')
                        quota_exceeded = remaining == '0' or 'quota' in res.text.lower()
                        _sync_rapidapi_response(
                            key_id, key_idx, res,
                            status_override='exhausted' if quota_exceeded else 'blocked',
                            cooldown=0 if quota_exceeded else 2)
                        continue
                    if res.status_code in (401, 403):
                        _sync_rapidapi_response(key_id, key_idx, res, status_override='invalid')
                        continue
                    if res.status_code in (400, 404, 410):
                        _sync_rapidapi_response(key_id, key_idx, res)
                        log_api_error(url, res.status_code, res.text[:200])
                        return None
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

# ----------------------------------------------------------------------
# FUNÇÕES COM PAGINAÇÃO (NOVAS)
# ----------------------------------------------------------------------
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
        time.sleep(0.3)
    return todos_eventos[:max_jogos]

# ----------------------------------------------------------------------
# FUNÇÕES DE ANÁLISE DE JOGOS E ESTATÍSTICAS (ATUALIZADAS)
# ----------------------------------------------------------------------
# Cache para as temporadas ativas
CACHE_ACTIVE_SEASON = {}

def buscar_season_id_ativa(tournament_id):
    if not tournament_id:
        return None

    if tournament_id in CACHE_ACTIVE_SEASON:
        return CACHE_ACTIVE_SEASON[tournament_id]

    # Endpoint para listar as temporadas de um torneio (exemplo, pode variar)
    url = tournament_seasons_url(RAPIDAPI_HOST, tournament_id)
    data = safe_api_get(url, max_retries=2)

    if data and 'seasons' in data:
        # Procura a temporada ativa ou a mais recente
        for season in data['seasons']:
            if season.get('isActive'):
                active_id = str(season.get('id'))
                CACHE_ACTIVE_SEASON[tournament_id] = active_id
                return active_id
        # Se não encontrar uma ativa, pega a última (com maior ID)
        if data['seasons']:
            last_season = max(data['seasons'], key=lambda x: x.get('id', 0))
            active_id = str(last_season.get('id'))
            CACHE_ACTIVE_SEASON[tournament_id] = active_id
            return active_id

    CACHE_ACTIVE_SEASON[tournament_id] = None
    return None

def analisar_ultimos_jogos_pro(team_id, limit=5, tipo='geral'):
    eventos = obter_todos_ultimos_jogos(team_id, max_jogos=limit*2)
    if not eventos:
        return "Form N/A"
    try:
        v=e=d=count=0
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
            hs, ast = score
            if hs == ast:
                e += 1
            elif (hs > ast and is_home) or (ast > hs and not is_home):
                v += 1
            else:
                d += 1
            count += 1
            if count >= limit:
                break
        return f"{v}V-{e}E-{d}D"
    except:
        return "Form N/A"
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
def analisar_ultimos_jogos_por_torneio(team_id, tournament_id, limit=5):
    eventos = obter_todos_ultimos_jogos(team_id, max_jogos=limit*2)
    if not eventos:
        return None
    try:
        v=e=d=count=0
        for ev in eventos:
            if ev.get('status', {}).get('type') != 'finished':
                continue
            ev_tournament_id = str(ev.get('tournament', {}).get('id', ''))
            if ev_tournament_id != str(tournament_id):
                continue
            is_home = str(ev.get('homeTeam', {}).get('id')) == str(team_id)
            score = regulation_score(ev)
            if score is None:
                continue
            hs, ast = score
            if hs == ast:
                e += 1
            elif (hs > ast and is_home) or (ast > hs and not is_home):
                v += 1
            else:
                d += 1
            count += 1
            if count >= limit:
                break
        if count == 0:
            return None
        return v, e, d
    except:
        return None

def obter_media_ppg_adversarios(team_id, tournament_id, season_id, n=5):
    eventos = obter_todos_ultimos_jogos(team_id, max_jogos=n*2)
    ppgs = []
    count = 0
    for ev in eventos:
        if ev.get('status', {}).get('type') != 'finished':
            continue
        is_home = str(ev.get('homeTeam', {}).get('id')) == str(team_id)
        if is_home:
            adv_id = str(ev.get('awayTeam', {}).get('id', ''))
        else:
            adv_id = str(ev.get('homeTeam', {}).get('id', ''))
        if adv_id:
            ev_tournament_id = str(ev.get('tournament', {}).get('id', ''))
            ev_season_id = str(ev.get('season', {}).get('id', ''))
            ppg_adv = obter_ppg_time(adv_id, ev_tournament_id, ev_season_id)
            ppgs.append(ppg_adv)
        count += 1
        if count >= n:
            break
    if ppgs:
        return sum(ppgs) / len(ppgs)
    return 0.0

def obter_ppg_time(team_id, tournament_id, season_id):
    if not tournament_id or not season_id:
        return 0.0
    try:
        data = safe_api_get(tournament_standings_url(RAPIDAPI_HOST, tournament_id, season_id))
        if data and 'standings' in data:
            for row in data['standings'][0].get('rows', []):
                if str(row.get('team', {}).get('id')) == str(team_id):
                    games = int(row.get('games', 0))
                    points = int(row.get('points', 0))
                    if games > 0:
                        return points / games
                    return 0.0
    except:
        pass
    return 0.0

# Dicionário de cache global (colocar no início do arquivo)
CACHE_STANDINGS = {}

# Cache global para standings
CACHE_STANDINGS = {}

def obter_posicao_time(team_id, tournament_id, season_id):
    """
    Obtém posição do time na tabela.
    Retorna (0, 20) se a classificação não estiver disponível.
    """
    if not tournament_id:
        return 0, 20

    # Tenta com o season_id informado
    if season_id and str(season_id).strip():
        url = tournament_standings_url(RAPIDAPI_HOST, tournament_id, season_id)
        data = safe_api_get(url, max_retries=1, timeout=8)
        if data and 'standings' in data and data['standings']:
            rows = data['standings'][0].get('rows', [])
            total = len(rows)
            for row in rows:
                if str(row.get('team', {}).get('id')) == str(team_id):
                    return int(row.get('position', 0)), total

    # Fallback: tentar a temporada ativa. A antiga rota sem season_id foi
    # removida e retornava 404, portanto não é mais consultada.
    active_season_id = get_active_season_id(tournament_id)
    if active_season_id:
        return obter_posicao_time(team_id, tournament_id, active_season_id)

    # Sem dados de classificação
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

    # Contexto pré-jogo da segunda fonte. Mudança de W/D/L atualiza a linha
    # imediatamente; sem mudança, o retrato vence em 12 horas.
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

def importar_jogos_para_treinamento(dias=7, progress_bar=None, status_text=None, aplicar_filtros=True, aplicar_filtro_odds=False):
    logger.info(f"Importando jogos finalizados dos últimos {dias} dias...")
    agora = get_brt_time()
    # Formato de data correto: DD/MM/YYYY
    datas = [(agora - timedelta(days=i)).strftime("%d/%m/%Y") for i in range(dias)]
    datas = list(set(datas))
    
    if not datas:
        print("Nenhuma data para consultar.")
        return 0

    blacklist = ["u17","u19","u20","u21","u22","u23","u24","sub-","sub17","sub19","sub20","sub21","sub23",
                 "amateur","amador","amadores","youth","juniors","aspirantes","reserva","reservas","reserve",
                 "reserves","woman","women","feminino","femenino","femmes","frauen"," w ","ladies","girls",
                 "sub","junior"]
    todos_eventos = []
    for d in datas:
        # Divide a data em dia, mês, ano
        day, month, year = d.split('/')
        print(f"[IMPORT] Buscando agenda v2.0 de {day}/{month}/{year}")
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
            print(f"📅 Data {d}: {len(eventos)} eventos totais, {len(finalizados)} finalizados")
        time.sleep(0.3)

    eventos_unicos = {str(ev.get('id')): ev for ev in todos_eventos}.values()
    lista = list(eventos_unicos)
    total = len(lista)
    if progress_bar:
        progress_bar.progress(0)
    if status_text:
        status_text.text(f"Processando {total} jogos...")
    
    stats = {"sem_odds": 0, "odds_baixas": 0, "ja_existe": 0, "erro": 0, "sem_classificacao": 0, "novos": 0}
    processados = 0
    lock = threading.Lock()

    def listar_features_zeradas(features_dict):
        zeros = [k for k, v in features_dict.items() if v == 0 or v is None]
        if zeros:
            return f"⚠️ Features zeradas: {', '.join(zeros[:10])}{'...' if len(zeros)>10 else ''}"
        return "✅ Todas as features não-zero."

    def processar(ev):
        nonlocal processados, stats
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

        # Extrair features (usa fallback SofaScore se unique_tournament_id existir)
        features = extrair_features_basicas(match_id, home_id, away_id,
                                            tournament_id, season_id,
                                            odd_casa, odd_empate, odd_fora,
                                            unique_tournament_id)
        # Log de features zeradas
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

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(processar, ev): ev for ev in lista}
        for future in as_completed(futures):
            processados += 1
            if progress_bar:
                progress_bar.progress(processados / total)
            if status_text:
                status_text.text(f"Processados {processados}/{total} | Novos: {stats['novos']} | "
                                 f"Sem odds: {stats['sem_odds']} | "
                                 f"Sem classificação: {stats['sem_classificacao']}")

    print("\n" + "="*80)
    print(f"📊 RESUMO DA IMPORTAÇÃO:")
    print(f"   Novos jogos inseridos: {stats['novos']}")
    print(f"   Sem odds (ignorados): {stats['sem_odds']}")
    print(f"   Odds baixas (ignorados): {stats['odds_baixas']}")
    print(f"   Já existentes (ignorados): {stats['ja_existe']}")
    print(f"   Erros: {stats['erro']}")
    print(f"   Sem classificação: {stats['sem_classificacao']}")
    print("="*80)
    
    logger.info(f"Importação concluída. Novos: {stats['novos']}. Ignorados: sem odds={stats['sem_odds']}, "
                f"odds baixas={stats['odds_baixas']}, já existentes={stats['ja_existe']}, "
                f"erros={stats['erro']}, sem classificação={stats['sem_classificacao']}")
    return stats['novos']

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

def get_standings_from_sofascore(unique_tournament_id, season_id):
    cache_key = f"standings_{unique_tournament_id}_{season_id}"
    # Cache em memória
    if cache_key in CACHE_STANDINGS:
        return CACHE_STANDINGS[cache_key]
    
    # Cache persistente
    cached = carregar_standings_cache(unique_tournament_id, season_id)
    if cached:
        CACHE_STANDINGS[cache_key] = cached
        return cached

    if not unique_tournament_id or not season_id:
        return None

    url = f"https://www.sofascore.com/api/v1/unique-tournament/{unique_tournament_id}/season/{season_id}/standings/total"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept": "application/json",
        "Referer": "https://www.sofascore.com/"
    }

    try:
        with PLAYWRIGHT_SEMAPHORE:
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True)
                page = browser.new_page()
                page.set_extra_http_headers(headers)
                page.goto(url, timeout=10000)  # timeout reduzido
                content = page.content()
                browser.close()
                try:
                    data = json.loads(content)
                except:
                    import re
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
                    CACHE_STANDINGS[cache_key] = standings_map
                    salvar_standings_cache(unique_tournament_id, season_id, standings_map)
                    return standings_map
                return None
    except Exception as e:
        logger.error(f"Erro SofaScore standings {unique_tournament_id}/{season_id}: {e}")
        return None

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

    # 2. Fallback SofaScore (usa unique_tourn_id ou tourn_id como fallback)
    effective_unique = unique_tourn_id if unique_tourn_id else tourn_id
    if effective_unique and season_id:
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

                # Fuzzy match
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
        else:
            print("[CLASSIF] SofaScore retornou None")
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
    except:
        pass
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

def get_nivel_campeonato(liga_name):
    texto = liga_name.lower()
    if any(x in texto for x in ['champions', 'libertadores', 'uefa', 'copa do mundo', 'world cup']):
        return 5
    if any(x in texto for x in ['premier league', 'la liga', 'serie a', 'bundesliga', 'primeira liga', 'brasileirão',
                                'ligue 1', 'eredivisie', 'super lig', 'süper lig', 'mls']):
        return 4
    if any(x in texto for x in ['copa', 'cup', 'fa cup', 'dfb pokal', 'coppa italia', 'copa del rey']):
        return 3
    if any(x in texto for x in ['serie b', 'segunda', 'championship', '2. bundesliga', 'segunda división',
                                'serie c', 'liga 2', '2. liga']):
        return 2
    return 1

def detectar_fase_mata_mata(liga_name):
    flags = competition_flags(liga_name)
    return int(flags['is_knockout']), int(flags['is_volta'])

def get_zonas_flags(posicao, total_times=20):
    if posicao <= 0 or total_times <= 0:
        return 0, 0
    reb = 1 if posicao >= total_times - 3 else 0
    clas = 1 if posicao <= 4 else 0
    return reb, clas

def extrair_v_e_d(form_str):
    if not form_str:
        return 0,0,0
    m = re.search(r'(\d+)V\s*[-]?\s*(\d+)E\s*[-]?\s*(\d+)D', form_str, re.IGNORECASE)
    if m:
        return int(m.group(1)), int(m.group(2)), int(m.group(3))
    v=e=d=0
    mv = re.search(r'(\d+)V', form_str)
    if mv:
        v = int(mv.group(1))
    me = re.search(r'(\d+)E', form_str)
    if me:
        e = int(me.group(1))
    md = re.search(r'(\d+)D', form_str)
    if md:
        d = int(md.group(1))
    return v,e,d

def salvar_ids_liga(liga_nome, tournament_id, season_id):
    if not liga_nome or not tournament_id or not season_id:
        return
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
    if row:
        conn.close()
        return str(row[0]), str(row[1])
    cursor.execute("SELECT DISTINCT tournament_id, season_id FROM previsoes WHERE liga = ? AND season_id != ''", (liga,))
    rows = cursor.fetchall()
    conn.close()
    if rows:
        rows_sorted = sorted(rows, key=lambda x: int(x[1]) if x[1].isdigit() else 0, reverse=True)
        if modo == 'current':
            return rows_sorted[0]
        else:
            return rows_sorted[1] if len(rows_sorted)>1 else rows_sorted[0]
    return None, None

# ----------------------------------------------------------------------
# EXTRAÇÃO DE FEATURES (SEM VAZAMENTO, COMPLETA)
# ----------------------------------------------------------------------

def buscar_estatisticas_partida_sofascore(match_id):
    """
    Busca estatísticas da partida no SofaScore via API interna.
    Retorna dicionário com xg, posse, chutes, etc.
    """
    url = f"https://www.sofascore.com/api/v1/event/{match_id}/statistics"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept": "application/json",
        "Referer": "https://www.sofascore.com/"
    }
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            page.set_extra_http_headers(headers)
            page.goto(url, timeout=15000)
            content = page.content()
            browser.close()
            import re
            match = re.search(r'(\{.*\})', content, re.DOTALL)
            if match:
                data = json.loads(match.group(1))
                return extrair_estatisticas_do_json_sofascore(data)
    except Exception as e:
        logger.error(f"Erro SofaScore estatísticas {match_id}: {e}")
    return {}

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
    """
    Média de xG sofrido pelo time nos últimos n jogos (na mesma competição).
    """
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
    return 1.2  # valor padrão

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

def get_unique_tournament_and_season(event):
    unique_tournament = event.get('tournament', {}).get('uniqueTournament', {})
    unique_id = unique_tournament.get('id')
    if not unique_id:
        unique_id = event.get('tournament', {}).get('id')
    season_id = event.get('season', {}).get('id')
    return unique_id, season_id

# ----------------------------------------------------------------------
# FUNÇÕES DE TREINAMENTO E ML (SEM VAZAMENTO)
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

    home_gf = valor('form_home_10_gf', 1.30)
    home_ga = valor('form_home_10_ga', 1.30)
    away_gf = valor('form_away_10_gf', 1.30)
    away_ga = valor('form_away_10_ga', 1.30)
    ataque_home = home_gf - away_ga
    ataque_away = away_gf - home_ga
    gols_esperados_home = max(0.0, (home_gf + away_ga) / 2.0)
    gols_esperados_away = max(0.0, (away_gf + home_ga) / 2.0)
    intensidade = gols_esperados_home + gols_esperados_away
    # Mantém a semântica das features do campeão; os ajustes novos recebem
    # nomes próprios em ``add_venue_comparison``.
    ppg_gap = valor('form_home_10_ppg', 1.35) - valor('form_away_10_ppg', 1.35)
    mando_gap = valor('form_home_casa_5_ppg', 1.35) - valor('form_away_fora_5_ppg', 1.35)
    prior_liga = max(0.10, min(0.50, valor('liga_prior_empate', 0.27)))
    tendencia_empate = (
        valor('form_home_10_draw_rate', prior_liga)
        + valor('form_away_10_draw_rate', prior_liga)
    ) / 2.0
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
    n7_h, n14_h = stats(datas_home)
    n7_a, n14_a = stats(datas_away)
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
        # As APIs frequentemente usam grafias diferentes para o mesmo clube.
        # Sem este vínculo, o histórico, o Elo e a forma local eram consultados
        # com nomes inexistentes e ambos os times recebiam o prior neutro.
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
        # O nome da mesma competição muda entre provedores (por exemplo,
        # "Bulgaria Second League" vs "Bulgaria - Vtora Liga"). Usar texto
        # deixava o prior 1X2 neutro mesmo com dezenas de jogos disponíveis.
        # IDs estáveis têm precedência; texto é somente o último fallback.
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
            return conn.execute("""SELECT data_jogo, home_team, away_team, home_score, away_score, match_id FROM training_data
                WHERE data_jogo < ? AND (home_team IN ({slots}) OR away_team IN ({slots})) ORDER BY data_jogo DESC LIMIT 50""".format(slots=slots),
                (cutoff, *aliases, *aliases)).fetchall()
        recentes_h, recentes_a = jogos_recentes(home_aliases), jogos_recentes(away_aliases)
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
            # O radar Soccer nem sempre fornece IDs AllSports. Ainda assim a
            # temporada pode ser reconstruída, sem API, pela mesma liga numa
            # janela móvel. Limitar a 400 dias evita misturar épocas antigas.
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
            if venue == 'home' and not is_home: continue
            if venue == 'away' and is_home: continue
            gf, ga = (hs, aws) if is_home else (aws, hs); pontos = 3 if gf > ga else (1 if gf == ga else 0)
            historical_pair = opponent_strengths.get(str(recent_match_id), (1.35, 1.35))
            opponent_ppg = float(historical_pair[0 if is_home else 1])
            jogos.append((pontos, float(gf), float(ga), opponent_ppg))
            if len(jogos) >= limit: break
        return jogos
    elo_h = float(ratings.get(home_team, 1500.0)); elo_a = float(ratings.get(away_team, 1500.0))
    elo_prob = 1.0 / (1.0 + 10 ** ((elo_a - (elo_h + 70.0)) / 400.0))
    result = {
        'historical_identity_home_confidence': float(home_identity_confidence),
        'historical_identity_away_confidence': float(away_identity_confidence),
        'historical_identity_both_linked': float(
            home_identity_confidence > 0 and away_identity_confidence > 0
        ),
        'hist_ppg_home_team': ppg(h_n, h_pts, 1.35), 'hist_ppg_away_team': ppg(a_n, a_pts, 1.35),
        'hist_ppg_home_mandante': ppg(hh[0], hh[1], 1.55), 'hist_ppg_away_visitante': ppg(aa[0], aa[1], 1.15),
        'hist_log_jogos_home': float(np.log1p(h_n)), 'hist_log_jogos_away': float(np.log1p(a_n)),
        'liga_prior_casa': (float(lg[1] or 0) + 2.0) / (lg_n + 6.0),
        'liga_prior_empate': (float(lg[2] or 0) + 2.0) / (lg_n + 6.0),
        'liga_prior_fora': (float(lg[3] or 0) + 2.0) / (lg_n + 6.0),
        'elo_ml_home': elo_h, 'elo_ml_away': elo_a, 'elo_ml_diff': elo_h - elo_a, 'elo_ml_prob_home': elo_prob,
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
    # Consolida aliases antes de construir qualquer estado sequencial. A
    # resolução usa somente nomes e frequências, nunca placares futuros; assim
    # Barcelona SC/Barcelona Guayaquil, por exemplo, não criam históricos e
    # Elos independentes apenas por divergência entre provedores.
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
    elo_ratings = {}; rolling_times = collections.defaultdict(lambda: collections.deque(maxlen=10))
    rolling_casa = collections.defaultdict(lambda: collections.deque(maxlen=10)); rolling_fora = collections.defaultdict(lambda: collections.deque(maxlen=10))
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
            feats = {k: v for k, v in feats.items() if k in FEATURES_LEGADAS_SEGURAS}
        else:
            versoes_verificadas += 1
        feats = _sem_features_de_odds(feats)
        for feature_name in list(feats):
            if feature_name.startswith('calendario_descanso_'):
                feats.pop(feature_name, None)
        # O snapshot histórico do SofaScore é uma fonte pré-jogo independente.
        # Ele deve entrar depois da higienização das bases legadas; caso contrário,
        # jogos antigos (feature_version < 2) removeriam justamente essas variáveis.
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
        th = historico_times.get(row['home_team'], [0, 0.0]); ta = historico_times.get(row['away_team'], [0, 0.0])
        hh = historico_casa.get(row['home_team'], [0, 0.0]); aa = historico_fora.get(row['away_team'], [0, 0.0])
        league_history_key = competition_history_key(row)
        lg = historico_ligas.get(league_history_key, [0, 0, 0, 0])
        def ppg_hist(stats, prior): return (stats[1] + prior * 8.0) / (stats[0] + 8.0)
        opponent_h_pre = ppg_hist(ta, 1.35); opponent_a_pre = ppg_hist(th, 1.35)
        feats.update({
            'hist_ppg_home_team': ppg_hist(th, 1.35), 'hist_ppg_away_team': ppg_hist(ta, 1.35),
            'hist_ppg_home_mandante': ppg_hist(hh, 1.55), 'hist_ppg_away_visitante': ppg_hist(aa, 1.15),
            'hist_log_jogos_home': float(np.log1p(th[0])), 'hist_log_jogos_away': float(np.log1p(ta[0])),
            'liga_prior_casa': (lg[1] + 2.0) / (lg[0] + 6.0),
            'liga_prior_empate': (lg[2] + 2.0) / (lg[0] + 6.0),
            'liga_prior_fora': (lg[3] + 2.0) / (lg[0] + 6.0),
        })
        elo_h = float(elo_ratings.get(row['home_team'], 1500.0)); elo_a = float(elo_ratings.get(row['away_team'], 1500.0))
        elo_prob = 1.0 / (1.0 + 10 ** ((elo_a - (elo_h + 70.0)) / 400.0))
        feats.update({'elo_ml_home': elo_h, 'elo_ml_away': elo_a, 'elo_ml_diff': elo_h - elo_a, 'elo_ml_prob_home': elo_prob})
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
            if feature_name.startswith('form_sfi_'):
                supported = float(feats.get('context_sfi_available', 0) or 0) > 0
            elif feature_name.startswith('context_sfi_'):
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
        for store, key, pontos in ((historico_times, row['home_team'], pontos_h), (historico_times, row['away_team'], pontos_a),
                                   (historico_casa, row['home_team'], pontos_h), (historico_fora, row['away_team'], pontos_a)):
            stats = store.setdefault(key, [0, 0.0]); stats[0] += 1; stats[1] += pontos
        liga_stats = historico_ligas.setdefault(league_history_key, [0, 0, 0, 0]); liga_stats[0] += 1; liga_stats[resultado + 1] += 1
        hs, aws = max(0.0, float(row['home_score'])), max(0.0, float(row['away_score']))
        rolling_times[row['home_team']].append((pontos_h, hs, aws, opponent_h_pre)); rolling_times[row['away_team']].append((pontos_a, aws, hs, opponent_a_pre))
        rolling_casa[row['home_team']].append((pontos_h, hs, aws, opponent_h_pre)); rolling_fora[row['away_team']].append((pontos_a, aws, hs, opponent_a_pre))
        datas_times[row['home_team']].append(data_contexto); datas_times[row['away_team']].append(data_contexto)
        for team, pontos, gf, ga in ((row['home_team'], pontos_h, hs, aws),
                                     (row['away_team'], pontos_a, aws, hs)):
            stats_temp = temporada_atual.setdefault(team, [0, 0.0, 0.0, 0.0])
            stats_temp[0] += 1; stats_temp[1] += pontos; stats_temp[2] += gf; stats_temp[3] += ga
        score_h = 1.0 if resultado == 0 else (0.5 if resultado == 1 else 0.0); margem = max(1.0, float(np.log1p(abs(hs - aws))))
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
        # O diagnóstico pós-jogo já transforma a natureza do erro em peso_base.
        # Multiplicar novamente pela margem fazia um RED receber peso de até 8x,
        # inclusive quando a leitura foi correta e o resultado foi uma zebra.
        peso_final = float(row['peso_base']) * peso_temporal
        pesos.append(peso_final)

    usar_features_estendidas = versoes_verificadas >= max(200, int(len(registros) * 0.10))
    if not usar_features_estendidas:
        feature_keys = {k for k in feature_keys
                        if k in FEATURES_LEGADAS_SEGURAS or k.startswith('hist_')
                        or k.startswith('liga_prior_') or k.startswith('elo_ml_')
                        or k.startswith(('form_', 'context_', 'calendario_', 'temporada_',
                                         'sofa_pre_', 'sofa_roll_'))}
    feature_keys = {k for k in feature_keys if k not in ODDS_ML_FEATURES
                    and not k.lower().startswith(('odd_', 'odds_', 'prob_mercado_'))}
    # Fontes novas entram no XGBoost somente quando existe suporte real. Até lá,
    # elas são congeladas nos snapshots e usadas por um overlay conservador. Isso
    # evita treinar 41 mil zeros contra poucas dezenas de partidas enriquecidas.
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
    # Balanceamento leve (raiz quarta do inverso da frequência). A versão anterior
    # importava compute_class_weight, mas nunca o usava; com isso a classe empate
    # era subestimada. No holdout: recall de empate 2,63% -> 8,90%, com custo de
    # apenas 0,12 p.p. na acurácia; a raiz quadrada custava 1,16 p.p. e foi rejeitada.
    # O balanceamento depende dos rótulos e, portanto, só pode ser calculado
    # dentro de cada janela de fit. Fazê-lo aqui revelava a distribuição do
    # holdout futuro ao treino.
    global _ultimos_elos_treino, _ultimo_radar_mask_treino, _ultimos_registros_treino
    _ultimos_elos_treino = dict(elo_ratings)
    _ultimo_radar_mask_treino = np.asarray(radar_flags, dtype=bool)
    _ultimos_registros_treino = [dict(features) for features in registros]
    return X, y, feature_order, match_ids, pesos

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
            # ``usado_treinamento`` pertence ao campeão GLOBAL. Um modelo por
            # liga não pode consumir a amostra inédita usada no head-to-head.
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
            # Bancos antigos continuam treinando, mas não podem promover sem a
            # evidência de bilhetes que será acumulada pelos próximos radares.
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
    # Preserve new source fields in the pregame snapshot; do not silently
    # activate an unvalidated candidate in the incumbent's feature list.
    feats.update(load_free_statistics(DB_NAME, match_id,
        (info.get('startTimestamp') if info else start_timestamp)))
    vec = []
    for feature_index, k in enumerate(feature_order):          # REMOVIDO o slicing [:10]
        # Neutraliza dias de descanso nos modelos antigos sem produzir um
        # valor padronizado extremo; desafiantes novos nem recebem a feature.
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
    # Mantém a precisão da probabilidade calibrada. O int anterior convertia,
    # por exemplo, 49,99% em 49% e podia reprovar uma seleção na borda.
    conf = round(float(proba[idx]) * 100.0, 2)
    vencedor = ['MANDANTE', 'EMPATE', 'VISITANTE'][idx]
    # Congela a identidade da competição existente no instante do palpite.
    # Isso permite calibração hierárquica futura sem inferir o campeonato a
    # partir do placar ou de uma tabela preenchida depois do jogo.
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
    if feats.get('context_sfi_available', 0):
        fontes.append('forma Soccer atual')
    if feats.get('sofa_pre_available', 0):
        fontes.append('SofaScore pré-jogo')
    if feats.get('live_recent_available', 0):
        fontes.append('últimos 5 atualizados')
    if feats.get('sofa_roll_available', 0):
        fontes.append('desempenho detalhado anterior')
    fonte_texto = f" + {' + '.join(fontes)}" if fontes else ""
    relatorio = (f"🤖 ML sem odds{fonte_texto} "
                 f"({liga if liga != 'GLOBAL' else 'Global'}) | "
                 f"C:{proba[0]:.1%} E:{proba[1]:.1%} F:{proba[2]:.1%} | "
                 f"risco empate:{analise_contextual['draw_risk']:.0%} "
                 f"qualidade contexto:{analise_contextual['information_quality']:.0%}")
    return vencedor, conf, relatorio

def atualizar_peso_erro(match_id, errou=True):
    with db_write_lock:
        conn = get_db_connection(); cur = conn.cursor()
        cur.execute("SELECT peso FROM training_weights WHERE match_id=?", (match_id,))
        row = cur.fetchone()
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
# FUNÇÃO DE DECISÃO (EDGE) - usa ML ou heurística
# ----------------------------------------------------------------------
def calcular_edge_super_python_v5(h_form_str, a_form_str, h2h_str, class_str, desfalques_str, streaks_str,
                                 stats_home, stats_away, odd_casa, odd_fora, odd_empate,
                                 form_rating_str, arbitro_str, dist_gols_h, dist_gols_a, pr_str,
                                 perf_h, perf_a, macro_liga, win_prob, tatica_str,
                                 ef_home, ef_away, liga, match_id, home_id, away_id, tournament_id, season_id,
                                 home_team=None, away_team=None, start_timestamp=None,
                                 unique_tournament_id=''):
    """
    Versão que usa APENAS o modelo de Machine Learning.
    Retorna (vencedor, confianca, relatorio, None) se houver modelo.
    Caso contrário, retorna (None, None, None, None) – o jogo será ignorado.
    """
    pred_ml = prever_com_ml(match_id, home_id, away_id, tournament_id, season_id,
                  odd_casa, odd_empate, odd_fora, liga,
                  unique_tournament_id=unique_tournament_id,
                  home_team=home_team, away_team=away_team, start_timestamp=start_timestamp)
    if pred_ml is not None:
        vencedor, confianca, relatorio = pred_ml
        return vencedor, confianca, relatorio, None
    else:
        # Sem modelo ML disponível – não usa heurística
        return None, None, None, None

# ----------------------------------------------------------------------
# FUNÇÕES DE RADAR (ATUALIZADAS)
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
        # Usa o retrato realmente congelado no radar. Recalcular forma aqui,
        # depois do apito final, causaria vazamento e descartaria o contexto que
        # fundamentou a previsão.
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
        logger.info("Nenhum jogo novo para transferir.")
        return 0
    transferidos = 0
    for mid in jogos:
        transferir_jogo_para_treinamento(mid)
        transferidos += 1
        time.sleep(0.2)
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
# JOBS
# ----------------------------------------------------------------------
job_running = {k: False for k in ['auditoria','radar','envio','ml_retreino','ml_otimizacao','backtest']}

MENSAGENS_QUASE_ACERTO = [
    "🔔 *QUASE!* Faltou apenas 1 jogo para o green total. Bilhete de odd {odd:.2f} passou perto! Continue assim, estamos no caminho certo! 💪",
    "🎯 *Por um triz!* Acertamos {acertos}/{total} nesse bilhete (odd {odd:.2f}). A vitória está próxima! 🔥",
    "📈 *Evolução!* Ficamos a 1 jogo de um green espetacular (odd {odd:.2f}). O sistema está afinado, os resultados virão! 🚀",
    "⚡ *Quase lá!* Este bilhete de odd {odd:.2f} errou apenas 1. Ajustes finos e vamos buscar o próximo! 💰",
    "🎲 *Foi por pouco!* Odd {odd:.2f} quase no bolso. Persistência é a chave, sigamos o plano! 🧠",
    "📊 *Análise positiva:* Acertamos {acertos}/{total} picks. O bilhete de odd {odd:.2f} mostrou que estamos no caminho! 🛤️",
    "🔍 *Detalhe:* Faltou um jogo para o green total (odd {odd:.2f}). Vamos refinar e na próxima entra! 🏆",
    "💡 *Quase acerto de qualidade!* Odd {odd:.2f} com apenas 1 erro. O ROI positivo está próximo! 📈",
    "🧩 *Que pena!* Mas o desempenho foi bom: {acertos}/{total} certos. Bilhete de odd {odd:.2f} deixou saudades. Próximo é green! 🌟",
    "🚨 *Atenção:* Estamos muito perto! Apenas 1 jogo separou este bilhete (odd {odd:.2f}) do green. Continue confiando no processo! 🤖"
]

def enviar_mensagem_quase_acerto(ticket_id, odd_multipla, total_jogos, acertos, reply_to_msg_id=None):
    """Envia mensagem motivacional aleatória para um quase acerto"""
    idx = random.randint(0, len(MENSAGENS_QUASE_ACERTO) - 1)
    texto = MENSAGENS_QUASE_ACERTO[idx].format(odd=odd_multipla, acertos=acertos, total=total_jogos)
    texto = f" *Por Pouco* - {ticket_id}\n\n{texto}"
    
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": texto,
        "parse_mode": "Markdown"
    }
    if reply_to_msg_id:
        payload["reply_to_message_id"] = reply_to_msg_id
    
    try:
        res = requests.post(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage", json=payload, timeout=10)
        if res.status_code == 200:
            logger.info(f"Mensagem de quase acerto enviada para bilhete {ticket_id}")
        else:
            logger.warning(f"Falha ao enviar mensagem: {res.text}")
    except Exception as e:
        logger.error(f"Erro ao enviar mensagem de quase acerto: {e}")


def _backfill_start_timestamps_do_cache():
    """Preenche horários antigos usando a agenda local, sem consumir a RapidAPI."""
    if not os.path.exists(SCHEDULE_CACHE_DB_PATH):
        return 0
    eventos = {}
    try:
        with sqlite3.connect(SCHEDULE_CACHE_DB_PATH, timeout=30) as cache_conn:
            for (payload_json,) in cache_conn.execute("SELECT payload_json FROM schedule_cache"):
                try:
                    payload = json.loads(payload_json)
                except (TypeError, ValueError, json.JSONDecodeError):
                    continue
                for evento in payload.get("events", []):
                    match_id = str(evento.get("id") or "")
                    start_ts = safe_event_timestamp(evento.get("startTimestamp"))
                    if match_id and start_ts > 0:
                        eventos[match_id] = start_ts
    except (OSError, sqlite3.Error) as exc:
        logger.warning("Não foi possível ler o cache de agenda para auditoria: %s", exc)
        return 0
    if not eventos:
        return 0
    with get_db_connection() as conn:
        antes = conn.total_changes
        conn.executemany(
            """UPDATE previsoes SET start_timestamp=?
               WHERE match_id=? AND COALESCE(start_timestamp, 0)=0""",
            [(start_ts, match_id) for match_id, start_ts in eventos.items()],
        )
        return conn.total_changes - antes


def _adiar_auditoria(cursor, match_id, attempts, motivo, agora_epoch, consumir_tentativa=True):
    novas_tentativas = attempts + (1 if consumir_tentativa else 0)
    # 2h, 4h, 8h, 16h e depois 24h. Evita consultar o mesmo jogo a cada hora.
    horas = min(24, 2 ** max(1, novas_tentativas))
    cursor.execute(
        """UPDATE previsoes
           SET audit_attempts=?, audit_next_at=?, audit_last_error=?
           WHERE match_id=?""",
        (novas_tentativas, agora_epoch + horas * 3600, str(motivo)[:300], match_id),
    )


def _registrar_resultado_auditado(cursor, match_id, previsto, ticket_id, evento, tickets_afetados):
    """Aplica um evento retornado pelo lote do torneio; não faz chamadas HTTP."""
    prediction_identity = cursor.execute(
        "SELECT confronto,start_timestamp FROM previsoes WHERE match_id=?",
        (str(match_id),),
    ).fetchone()
    if (not prediction_identity or not event_matches_prediction(
        evento, match_id, prediction_identity[0], prediction_identity[1]
    )):
        return "evento_incorreto"
    status_evento = str(evento.get('status', {}).get('type') or '').lower()
    if status_evento in {'canceled', 'cancelled', 'postponed'}:
        cursor.execute("""UPDATE previsoes SET status_resultado='ANULADO', anulado=1,
                          audit_last_error=?, audit_next_at=0 WHERE match_id=?""",
                       (status_evento, match_id))
        if ticket_id:
            tickets_afetados.add(ticket_id)
        return "anulado"
    if not event_finished(evento):
        return status_evento or "desconhecido"

    score = regulation_score(evento)
    if score is None:
        return "placar_invalido"
    h, a = score
    snapshot = cursor.execute(
        "SELECT predicted_outcome FROM ml_prediction_snapshots WHERE match_id=?",
        (str(match_id),),
    ).fetchone()
    lado_congelado = resolve_pick_side(snapshot[0] if snapshot else None, previsto,
        (evento.get('homeTeam') or {}).get('name'), (evento.get('awayTeam') or {}).get('name'))
    if lado_congelado not in {"MANDANTE", "EMPATE", "VISITANTE"}:
        return "pick_ambiguo"
    if h > a and lado_congelado == "MANDANTE":
        resultado_previsao = "GREEN ✅"
    elif a > h and lado_congelado == "VISITANTE":
        resultado_previsao = "GREEN ✅"
    elif h == a and lado_congelado == "EMPATE":
        resultado_previsao = "GREEN ✅"
    else:
        resultado_previsao = "RED ❌"

    home_id = evento.get('homeTeam', {}).get('id')
    away_id = evento.get('awayTeam', {}).get('id')
    if home_id and away_id:
        # O peso definitivo será calculado pela autópsia: erro de leitura,
        # empate mal avaliado e zebra não podem receber o mesmo tratamento.
        cursor.execute("""INSERT INTO training_weights
            (match_id, peso, data_ultima_atualizacao, error_margin)
            VALUES (?, 1.0, ?, 0.0)
            ON CONFLICT(match_id) DO UPDATE SET
                data_ultima_atualizacao=excluded.data_ultima_atualizacao""",
            (match_id, get_brt_time().strftime("%Y-%m-%d %H:%M:%S")))

        res_h = 1.0 if h > a else (0.5 if h == a else 0.0)
        res_a = 1.0 - res_h
        elo_h_row = cursor.execute("SELECT elo FROM elo_rating WHERE team_id=?", (str(home_id),)).fetchone()
        elo_a_row = cursor.execute("SELECT elo FROM elo_rating WHERE team_id=?", (str(away_id),)).fetchone()
        elo_h = float(elo_h_row[0]) if elo_h_row else 1500.0
        elo_a = float(elo_a_row[0]) if elo_a_row else 1500.0
        expected_h = 1.0 / (1.0 + 10 ** ((elo_a - elo_h) / 400.0))
        expected_a = 1.0 - expected_h
        cursor.execute("INSERT OR REPLACE INTO elo_rating (team_id, elo, last_update) VALUES (?, ?, ?)",
                       (str(home_id), int(elo_h + 32 * (res_h - expected_h)), get_brt_time().isoformat()))
        cursor.execute("INSERT OR REPLACE INTO elo_rating (team_id, elo, last_update) VALUES (?, ?, ?)",
                       (str(away_id), int(elo_a + 32 * (res_a - expected_a)), get_brt_time().isoformat()))

    cursor.execute("""UPDATE previsoes SET status_resultado=?, placar_real=?,
                      audit_next_at=0, audit_last_error=NULL WHERE match_id=?""",
                   (resultado_previsao, f"{h}-{a}", match_id))
    if ticket_id:
        tickets_afetados.add(ticket_id)
    return "finished"


def _reconciliar_status_com_snapshot(cursor, radar_run_ids):
    """Corrige GREEN/RED pelo lado congelado, sem depender do nome do time."""
    run_ids = [str(value) for value in (radar_run_ids or []) if str(value)]
    if not run_ids:
        return 0, set()
    slots = ",".join("?" for _ in run_ids)
    rows = cursor.execute(
        f"""SELECT p.match_id, p.ticket_id, p.status_resultado,
                   s.predicted_outcome, m.actual_outcome
            FROM previsoes p
            JOIN ml_prediction_snapshots s ON s.match_id=p.match_id
            JOIN match_postmortems m ON m.match_id=p.match_id
            WHERE p.radar_run_id IN ({slots})
              AND p.anulado=0
              AND m.actual_outcome IN ('MANDANTE','EMPATE','VISITANTE')
              AND s.predicted_outcome IN ('MANDANTE','EMPATE','VISITANTE')""",
        run_ids,
    ).fetchall()
    alterados, tickets = 0, set()
    for match_id, ticket_id, status_atual, previsto, realizado in rows:
        status_correto = "GREEN ✅" if previsto == realizado else "RED ❌"
        if str(status_atual) == status_correto:
            continue
        cursor.execute(
            "UPDATE previsoes SET status_resultado=? WHERE match_id=?",
            (status_correto, str(match_id)),
        )
        cursor.execute(
            "UPDATE match_postmortems SET prediction_status=? WHERE match_id=?",
            (status_correto, str(match_id)),
        )
        alterados += 1
        if ticket_id:
            tickets.add(str(ticket_id))
    return alterados, tickets


def _backfill_autopsias_sofascore(limit=None):
    """Completa resultados recentes que foram auditados antes da autópsia v1."""
    limite = AUDIT_POSTMORTEM_BACKFILL_LIMIT if limit is None else max(0, int(limit))
    if limite <= 0:
        return []
    cutoff = (get_brt_time() - timedelta(days=AUDIT_RECENT_DAYS)).strftime("%Y-%m-%d %H:%M:%S")
    with get_db_connection() as conn:
        rows = conn.execute(
            """SELECT p.match_id, p.vencedor_previsto, p.status_resultado
               FROM previsoes p
               LEFT JOIN match_postmortems m ON m.match_id=p.match_id
               WHERE m.match_id IS NULL AND p.timestamp>=? AND p.anulado=0
                 AND (p.status_resultado='RED ❌' OR p.status_resultado LIKE 'GREEN ✅%')
               ORDER BY CASE WHEN p.status_resultado='RED ❌' THEN 0 ELSE 1 END,
                        p.timestamp DESC LIMIT ?""",
            (cutoff, limite),
        ).fetchall()
    diagnostics = []
    for match_id, predicted, status in rows:
        event = get_event_for_audit(DB_NAME, match_id)
        if not isinstance(event, dict):
            continue
        try:
            transferir_jogo_para_treinamento(str(match_id), event)
            diagnostics.append(audit_match_postmortem(
                DB_NAME, str(match_id), event, predicted, status,
                postmatch_payload_fallback=lambda mid, resources: (
                    fetch_allsports_postmatch_resources(
                        RAPIDAPI_HOST, mid, resources, safe_api_get,
                    )
                ),
            ))
        except Exception as exc:
            logger.warning("Backfill de autópsia falhou para %s: %s", match_id, exc)
    return diagnostics


def job_auditoria_e_deeplab():
    if job_running['auditoria']:
        return
    job_running['auditoria'] = True
    conn = None
    try:
        print("\n🔍 ====== JOB: AUDITORIA + QUASE ACERTOS + ELO =====")
        preenchidos = _backfill_start_timestamps_do_cache()
        if preenchidos:
            print(f"   🗃️ {preenchidos} horários recuperados do cache local (0 requisições).")
        conn = get_db_connection()
        cursor = conn.cursor()

        # 1. Atualizar resultados pendentes
        agora_epoch = int(get_brt_time().timestamp())
        limite_inicio = agora_epoch - AUDIT_RESULT_GRACE_SECONDS
        limite_registro = (get_brt_time() - timedelta(days=AUDIT_RECENT_DAYS)).strftime("%Y-%m-%d %H:%M:%S")
        radar_run_ids = get_latest_radar_run_ids(conn, limit=2)
        if radar_run_ids:
            run_slots = radar_run_placeholders(radar_run_ids)
            cursor.execute(f"""SELECT match_id, vencedor_previsto, ticket_id,
                                      COALESCE(audit_attempts, 0), tournament_id, start_timestamp
                               FROM previsoes
                               WHERE status_resultado='PENDENTE'
                                 AND antecipado_detectado=0
                                 AND radar_run_id IN ({run_slots})
                                 AND COALESCE(start_timestamp, 0)>0
                                 AND start_timestamp<=?
                                 AND COALESCE(audit_next_at, 0)<=?
                                 AND COALESCE(audit_attempts, 0)<?
                                 AND timestamp>=?
                               ORDER BY start_timestamp""",
                           (*radar_run_ids, limite_inicio, agora_epoch,
                            AUDIT_MAX_ATTEMPTS, limite_registro))
            pendentes = cursor.fetchall()
            total_pendentes = cursor.execute(
                f"""SELECT COUNT(*) FROM previsoes
                    WHERE status_resultado='PENDENTE' AND antecipado_detectado=0
                      AND radar_run_id IN ({run_slots})""",
                radar_run_ids,
            ).fetchone()[0]
            print("   Radares auditados: " + ", ".join(radar_run_ids))
        else:
            pendentes = []
            total_pendentes = 0
            print("   Auditoria: nenhum lote de radar registrado.")
        print(f"   Auditoria: {len(pendentes)} jogo(s) vencido(s) agora; "
              f"{max(0, total_pendentes - len(pendentes))} aguardando horário/backoff nos dois radares.")
        tickets_afetados = set()
        requisicoes_auditoria = 0
        resultados_sofascore = 0

        # Primeiro coleta todos os eventos. O cache SofaScore usa outra conexão
        # SQLite e não pode ser atualizado enquanto esta auditoria mantém uma
        # transação de escrita aberta no banco principal.
        eventos_coletados = []
        for match_id, previsto, ticket_id, attempts, _, _ in pendentes:
            erro_api = ""
            try:
                evento = get_event_for_audit(DB_NAME, match_id)
            except Exception as exc:
                evento = None
                erro_api = f"falha ao consultar/cachear SofaScore: {exc}"
            if isinstance(evento, dict):
                resultados_sofascore += 1
            else:
                try:
                    payload = safe_api_get(match_detail_url(RAPIDAPI_HOST, match_id))
                    requisicoes_auditoria += 1
                except Exception as exc:
                    payload = None
                    erro_api = f"falha SofaScore e RapidAPI: {exc}"
                else:
                    erro_api = "SofaScore sem cobertura e RapidAPI sem resposta/cota"
                evento = payload.get('event') if isinstance(payload, dict) else None
            eventos_coletados.append(
                (str(match_id), previsto, ticket_id, attempts, evento, erro_api)
            )

        # A gravação é curta, não faz HTTP e repete apenas em caso de contenção
        # temporária com outro processo (por exemplo, o app aberto).
        transferencias = []
        for tentativa in range(3):
            transferencias_tentativa = []
            tickets_tentativa = set()
            try:
                with db_write_lock:
                    for match_id, previsto, ticket_id, attempts, evento, erro_api in eventos_coletados:
                        if not isinstance(evento, dict):
                            _adiar_auditoria(cursor, match_id, attempts, erro_api,
                                            agora_epoch, consumir_tentativa=False)
                            continue
                        resultado = _registrar_resultado_auditado(
                            cursor, match_id, previsto, ticket_id, evento, tickets_tentativa)
                        if resultado == "finished":
                            status_row = cursor.execute(
                                "SELECT status_resultado FROM previsoes WHERE match_id=?", (match_id,)
                            ).fetchone()
                            transferencias_tentativa.append((
                                match_id, evento, previsto,
                                str(status_row[0]) if status_row else 'PENDENTE',
                            ))
                        elif resultado not in {"anulado"}:
                            _adiar_auditoria(cursor, match_id, attempts,
                                            f"status={resultado}", agora_epoch)
                    conn.commit()
                transferencias = transferencias_tentativa
                tickets_afetados.update(tickets_tentativa)
                break
            except sqlite3.OperationalError as exc:
                conn.rollback()
                if "locked" not in str(exc).lower() or tentativa == 2:
                    raise
                espera = 0.5 * (tentativa + 1)
                logger.warning("Banco ocupado na auditoria; nova tentativa em %.1fs.", espera)
                time.sleep(espera)

        # A transferência usa o snapshot pré-jogo. Em seguida, a autópsia final
        # lê estatísticas/mapa de chutes/pressão e classifica a natureza do erro.
        diagnosticos = []
        requisicoes_sofa_autopsia = 0
        for match_id, evento, previsto, status_previsao in transferencias:
            transferir_jogo_para_treinamento(match_id, evento)
            try:
                diagnostico = audit_match_postmortem(
                    DB_NAME, match_id, evento, previsto, status_previsao,
                    postmatch_payload_fallback=lambda mid, resources: (
                        fetch_allsports_postmatch_resources(
                            RAPIDAPI_HOST, mid, resources, safe_api_get,
                        )
                    ),
                )
                diagnosticos.append(diagnostico)
                requisicoes_sofa_autopsia += int(diagnostico.get('http_requests', 0) or 0)
            except Exception as exc:
                logger.warning("Autópsia SofaScore falhou para %s: %s", match_id, exc)
        print(f"   Resultados: {resultados_sofascore} via SofaScore; "
              f"{requisicoes_auditoria} fallback(s) na RapidAPI para {len(pendentes)} jogo(s).")
        if diagnosticos:
            resumo_diagnosticos = collections.Counter(d.get('verdict', 'INDEFINIDO') for d in diagnosticos)
            print(f"   Autópsias gravadas: {len(diagnosticos)} | "
                  + ", ".join(f"{nome}={total}" for nome, total in sorted(resumo_diagnosticos.items())))
            print(f"   SofaScore na autópsia: {requisicoes_sofa_autopsia} requisição(ões) nova(s); "
                  "cache persistente será reutilizado nas próximas auditorias.")
            requisicoes_allsports_autopsia = sum(
                int(d.get('allsports_fallback_http', 0) or 0) for d in diagnosticos
            )
            pulos_allsports_autopsia = sum(
                int(d.get('allsports_fallback_quota_skips', 0) or 0) for d in diagnosticos
            )
            if requisicoes_allsports_autopsia or pulos_allsports_autopsia:
                print(
                    "   AllSports fallback na autópsia: "
                    f"{requisicoes_allsports_autopsia} requisição(ões) nova(s); "
                    f"{pulos_allsports_autopsia} recurso(s) ignorado(s) pelo limite diário."
                )

        backfill = _backfill_autopsias_sofascore()
        if backfill:
            resumo_backfill = collections.Counter(d.get('verdict', 'INDEFINIDO') for d in backfill)
            print(f"   Backfill de autópsias antigas: {len(backfill)} | "
                  + ", ".join(f"{nome}={total}" for nome, total in sorted(resumo_backfill.items())))

        if radar_run_ids:
            run_slots = radar_run_placeholders(radar_run_ids)
            ids_escopo = [row[0] for row in cursor.execute(
                f"SELECT DISTINCT match_id FROM previsoes WHERE radar_run_id IN ({run_slots})",
                radar_run_ids,
            ).fetchall()]
            revisao = reclassify_stored_postmortems(DB_NAME, ids_escopo)
            if revisao.get('reclassified'):
                print(f"   Autópsias antigas reclassificadas sem API: "
                      f"{revisao['reclassified']} | {revisao['verdicts']}")
            with db_write_lock:
                corrigidos, tickets_corrigidos = _reconciliar_status_com_snapshot(
                    cursor, radar_run_ids,
                )
                conn.commit()
            tickets_afetados.update(tickets_corrigidos)
            if corrigidos:
                print(f"   Auditoria reconciliada: {corrigidos} GREEN/RED corrigido(s) "
                      "pelo lado congelado no snapshot.")

        monitor_fontes = monitor_pregame_context_sources(DB_NAME)
        allsports_monitor = monitor_fontes.get("allsports_goal_distribution", {})
        allsports_acc = allsports_monitor.get("accuracy")
        allsports_acc_text = (
            f"{float(allsports_acc) * 100:.1f}%" if allsports_acc is not None else "aguardando"
        )
        print(
            "   Monitor AllSports pré-jogo: "
            f"{allsports_monitor.get('snapshots', 0)} snapshot(s), "
            f"{allsports_monitor.get('resolved', 0)} resolvido(s), "
            f"acerto atual={allsports_acc_text}; "
            "promoção só após validação cronológica."
        )

        # 2. Garantir coluna quase_acerto_notificado
        try:
            cursor.execute("SELECT quase_acerto_notificado FROM previsoes LIMIT 1")
        except sqlite3.OperationalError:
            cursor.execute("ALTER TABLE previsoes ADD COLUMN quase_acerto_notificado INTEGER DEFAULT 0")
            conn.commit()

        # 3. Quase-acertos dos mesmos dois radares auditados
        quase_acertos = []
        if radar_run_ids:
            run_slots = radar_run_placeholders(radar_run_ids)
            cursor.execute(f"""
                SELECT ticket_id, COUNT(*) as total_jogos,
                       SUM(CASE WHEN status_resultado LIKE 'GREEN%' THEN 1 ELSE 0 END) as greens,
                       SUM(CASE WHEN status_resultado LIKE 'RED%' THEN 1 ELSE 0 END) as reds,
                       MAX(telegram_msg_id) as msg_id
                FROM previsoes
                WHERE ticket_id IS NOT NULL AND ticket_id != ''
                  AND anulado = 0
                  AND (quase_acerto_notificado = 0 OR quase_acerto_notificado IS NULL)
                  AND radar_run_id IN ({run_slots})
                GROUP BY ticket_id
                HAVING 
                    SUM(CASE WHEN status_resultado = 'PENDENTE' THEN 1 ELSE 0 END) = 0
                    AND reds = 1
                    AND greens = COUNT(*) - 1
            """, radar_run_ids)
            quase_acertos = cursor.fetchall()

            # Uma execução pode terminar depois que o último jogo de um bilhete
            # já havia sido conciliado por outro processo. Revarrer os vencedores
            # evita deixar GREEN completo sem edição/celebração no Telegram.
            cursor.execute(f"""
                SELECT ticket_id
                FROM previsoes
                WHERE ticket_id IS NOT NULL AND ticket_id != ''
                  AND radar_run_id IN ({run_slots})
                GROUP BY ticket_id
                HAVING SUM(CASE WHEN status_resultado='PENDENTE' THEN 1 ELSE 0 END)=0
                   AND SUM(CASE WHEN status_resultado LIKE 'RED%' THEN 1 ELSE 0 END)=0
                   AND SUM(CASE WHEN status_resultado LIKE 'GREEN%' THEN 1 ELSE 0 END)>0
            """, radar_run_ids)
            tickets_afetados.update(str(row[0]) for row in cursor.fetchall())

        # 4. Enviar notificações de quase acertos
        for ticket_id, total_jogos, greens, reds, msg_id in quase_acertos:
            cursor.execute("SELECT odd_casa, odd_fora, odd_empate, vencedor_previsto, confronto FROM previsoes WHERE ticket_id = ?", (ticket_id,))
            jogos = cursor.fetchall()
            odd_multipla = 1.0
            for oc, of, oe, pick, confronto in jogos:
                casa_nome = confronto.split(' vs ')[0].strip().lower()
                pick_low = pick.lower()
                if 'empate' in pick_low:
                    odd_jogo = float(oe)
                elif casa_nome in pick_low:
                    odd_jogo = float(oc)
                else:
                    odd_jogo = float(of)
                if odd_jogo <= 1.0:
                    odd_jogo = max(float(oc), float(of))
                odd_multipla *= odd_jogo
            reply_id = int(msg_id) if msg_id and msg_id != 'None' else None
            enviar_mensagem_quase_acerto(ticket_id, odd_multipla, total_jogos, greens, reply_id)
            cursor.execute("UPDATE previsoes SET quase_acerto_notificado = 1 WHERE ticket_id = ?", (ticket_id,))
            conn.commit()

        conn.close()
        conn = None

        # Phase 4B: os rótulos são gravados de forma cega no sidecar. Este job
        # recebe apenas volume/saúde; não lê placar, classe real ou desempenho.
        try:
            from phase4_labels import ingest_blind_outcomes
            phase4_labels = ingest_blind_outcomes(DB_NAME)
            print(
                "   Phase 4B cega: "
                f"{phase4_labels['resolved_new']} novo(s) rótulo(s) selado(s), "
                f"{phase4_labels['pending_candidates']} candidato(s) vencido(s), "
                f"{phase4_labels['http_requests']} requisição(ões), "
                f"status={phase4_labels['status']}."
            )
        except Exception as exc:
            logger.warning("Rotulagem cega Phase 4B falhou sem afetar a auditoria: %s", exc)

        # A força atual dos times é estado de entrada, não parte do artefato do
        # modelo. Ela deve avançar após a auditoria mesmo quando o desafiante é
        # rejeitado e o V5 permanece campeão.
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
        print(f"   Força cronológica atualizada: {len(ratings)} times (0 requisições).")

        # 5. Atualizar mensagens e celebrar greens
        for tid in tickets_afetados:
            atualizar_mensagem_telegram_por_bilhete(tid)
            verificar_e_celebrar_green(tid)

        monitor = monitor_live_predictions(DB_NAME, floor_percent=MIN_ML_CONFIDENCE)
        acc_real = monitor.get("accuracy")
        acc_text = f"{acc_real:.2%}" if acc_real is not None else "amostra insuficiente"
        print(f"   🔎 Validação individual real: {monitor['status']} | {acc_text} "
              f"| n={monitor['samples']} | corte={monitor['recommended_confidence']:.0f}%")
        print(f"✅ Auditoria concluída. {len(tickets_afetados)} bilhetes atualizados. {len(quase_acertos)} quase-acertos notificados.")
    except Exception as e:
        logger.error(f"Erro auditoria: {e}", exc_info=True)
    finally:
        if conn is not None:
            try:
                conn.rollback()
            except sqlite3.Error:
                pass
            try:
                conn.close()
            except sqlite3.Error:
                pass
        job_running['auditoria'] = False
        
def job_radar_e_analise():
    if job_running['radar']:
        return
    job_running['radar'] = True
    try:
        print("\n📡 ====== RADAR (apenas ML) =====")
        agora = get_brt_time()
        
        # Uma única definição BRT é compartilhada com o app. A data em comum
        # com o radar anterior é reaproveitada pelo cache persistente.
        start_time, end_time = radar_window_brt(agora)
        
        print(f"   Janela: {start_time.strftime('%d/%m/%Y %H:%M')} até {end_time.strftime('%d/%m/%Y %H:%M')}")
        
        datas_str = [day.strftime("%d/%m/%Y")
                     for day in schedule_dates_for_brt_window(start_time, end_time)]
        
        odds_payloads = []
        for d_str in datas_str:
            print(f"   Buscando odds atuais da AllSports para {d_str}...")
            odds = safe_api_get(matches_odds_date_url(RAPIDAPI_HOST, d_str))
            if odds:
                odds_payloads.append(odds)
            else:
                print(f"      ⚠️ Odds indisponíveis para {d_str}; essa data não será qualificada.")

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
                progress=lambda mensagem: print(f"      ⚠️ Agenda Soccer: {mensagem}"),
                allsports_fallback_loader=carregar_fallback_allsports,
            )
            # Phase 4: run the exact frozen champion on the complete valid
            # market pool before the production filter. Research-only mode
            # writes solely to the append-only sidecar and performs no HTTP.
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
            print("   ⚠️ Agenda Soccer indisponível. O robô continuará ativo.")
            return

        antes_blacklist = len(finais)
        research_before_blacklist = list(finais)
        finais = [
            jogo for jogo in finais
            if not any(
                termo in f"{jogo['Liga']} {jogo['Time Casa']} {jogo['Time Fora']}".lower()
                for termo in BLACKLIST_TERMS
            )
        ]
        print(
            "   Agenda Soccer: "
            f"{agenda_meta.get('events_in_window', 0)} jogos na janela; "
            f"{agenda_meta.get('http_requests', 0)} requisições novas e "
            f"{agenda_meta.get('cache_hits', 0)} páginas do cache."
        )
        from phase2_observations import observe_filter
        observe_filter(DB_NAME, research_before_blacklist, finais, 'blacklist_filter')
        print(
            "   Relação de odds: "
            f"{agenda_meta.get('linked_to_current_odds', 0)} jogo(s) com odds atuais "
            f"({agenda_meta.get('events_with_bet365_fid', 0)} evento(s) tinham fid, "
            f"{agenda_meta.get('fallback_linked', 0)} ligado(s) pelo fallback); "
            f"{antes_blacklist} passaram pelo filtro casa/fora > 1,99; "
            f"{len(finais)} após categorias bloqueadas."
        )
        print(
            "   Phase 4 pré-filtro: "
            f"{agenda_meta.get('phase4_prefilter_scored', 0)}/"
            f"{len(agenda_meta.get('research_candidates') or [])} candidato(s) "
            "pontuado(s) em shadow; "
            f"{agenda_meta.get('phase4_eligible', 0)} eligible e "
            f"{agenda_meta.get('phase4_non_eligible', 0)} non-eligible; "
            f"{agenda_meta.get('phase4_prefilter_failed', 0)} falha(s)."
        )
        print(
            f"   AllSports: {len(odds_payloads)}/{len(datas_str)} lote(s) diário(s) de odds."
        )
        fallback_meta = agenda_meta.get("fallback_meta") or {}
        if fallback_meta:
            print(
                "   Fallback de campeonatos sem fid: "
                f"{agenda_meta.get('fallback_linked', 0)} jogo(s) relacionado(s); "
                f"{fallback_meta.get('tournaments_selected', 0)} torneio(s) pesquisado(s); "
                f"~{fallback_meta.get('estimated_http_requests', 0)} requisição(ões) AllSports; "
                f"{fallback_meta.get('tournament_failures', 0)} consulta(s) não concluída(s)."
            )
        if agenda_meta.get('pages_missing'):
            print(
                "      ⚠️ Agenda Soccer parcial: faltaram "
                f"{agenda_meta['pages_missing']} página(s); os jogos coletados serão usados."
            )
        if agenda_meta.get('unmatched_examples'):
            print(
                "      Sem relação de odds atuais (amostra): "
                + "; ".join(agenda_meta['unmatched_examples'][:4])
            )
        
        if not finais:
            print("   Nenhum jogo encontrado na janela.")
            return
        
        # Remove qualquer jogo já registrado. Até os não selecionados são
        # auditados depois, portanto não precisam voltar ao radar.
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT match_id FROM previsoes")
        ids_salvos = {str(r[0]) for r in cursor.fetchall()}
        conn.close()
        
        antes = len(finais)
        research_before_existing = list(finais)
        finais = [j for j in finais if str(j['ID']) not in ids_salvos]
        observe_filter(DB_NAME, research_before_existing, finais, 'already_recorded_filter')
        print(f"   Jogos novos: {len(finais)} (de {antes})")
        
        if not finais:
            print("   Todos os jogos já foram processados.")
            return
        
        # Remove duplicatas pelo ID
        df_limpo = pd.DataFrame(finais).drop_duplicates(subset=['ID'])
        res_list = sorted(df_limpo.to_dict('records'), key=lambda x: x['Timestamp'])
        print(f"   Jogos únicos: {len(res_list)}")
        print(f"   [DEBUG] Total qualificado antes de remover já processados: {antes}")

        # Coleta jogo a jogo somente depois dos filtros de agenda/odds. São
        # snapshots pré-jogo cacheados; falha/403/404 não exclui a partida.
        sofa_meta = capture_pregame_contexts(
            DB_NAME, res_list,
            progress=lambda mensagem: print(f"      {mensagem}"),
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
        print(
            "   Contexto SofaScore: "
            f"{sofa_meta.get('available', 0)}/{len(res_list)} com forma pré-jogo; "
            f"{sofa_meta.get('http_requests', 0)} requisição(ões) nova(s), "
            f"{sofa_meta.get('cache_hits', 0) + sofa_meta.get('context_cache_hits', 0)} cache hit(s)."
        )
        print(
            "   Forma viva (últimos 5): "
            f"{sofa_meta.get('live_available', 0)}/{len(res_list)} com os dois times; "
            f"{sofa_meta.get('live_provider_sofascore', 0)} lado(s) via SofaScore, "
            f"{sofa_meta.get('live_provider_allsports', 0)} via AllSports; "
            f"{sofa_meta.get('live_fallback_calls', 0)} fallback(s) solicitado(s)."
        )
        print(
            "   Contexto alternativo AllSports: "
            f"{sofa_meta.get('allsports_form_fallback', 0)} forma(s), "
            f"{sofa_meta.get('allsports_streaks_fallback', 0)} streak(s), "
            f"{sofa_meta.get('allsports_pregame_http', 0)} requisição(ões), "
            f"{sofa_meta.get('allsports_pregame_cache_hits', 0)} cache hit(s)."
        )
        print(
            "   Distribuição sazonal AllSports: "
            f"{sofa_meta.get('allsports_season_http', 0)} requisição(ões), "
            f"{sofa_meta.get('allsports_season_cache_hits', 0)} cache hit(s)."
        )
        
        free_meta = enrich_free_statistics(DB_NAME, res_list)
        print(
            "   Fonte gratuita (dados em sombra, sem trocar o campeão): "
            f"{free_meta.get('linked_games', 0)}/{len(res_list)} vinculados; "
            f"{free_meta.get('requests', 0)} requisição(ões)."
        )
        metric_meta = enrich_recent_metrics_safely(
            DB_NAME, res_list,
            fallback=lambda mid, resources: fetch_allsports_postmatch_resources(
                RAPIDAPI_HOST, mid, resources, safe_api_get),
        )
        print(
            "   Estatísticas de jogos anteriores (até 40 consultas por radar por padrão): "
            f"{metric_meta.get('lookups', 0)} consultado(s), "
            f"{metric_meta.get('profile_cache_hits', 0)} no cache, "
            f"{metric_meta.get('measured_matches', 0)} com estatísticas, "
            f"{metric_meta.get('xg_matches', 0)} com xG; "
            f"{metric_meta.get('allsports_http', 0)} requisição(ões) AllSports."
        )
        # ========== ANÁLISE APENAS COM ML ========== 
        agregado_ind = []
        total_jogos = len(res_list)
        limiar_ml_ativo = (0.0 if PUBLICAR_TODAS_PREVISOES
                           else get_active_confidence(DB_NAME, MIN_ML_CONFIDENCE))
        ml_model_version, ml_model_id = get_active_model_identity(DB_NAME)
        if PUBLICAR_TODAS_PREVISOES:
            print("   Política ativa: confiança informativa; todas as previsões entram na montagem.")
        else:
            print(f"   Política ativa: mínimo {limiar_ml_ativo:.0f}% por seleção individual.")
        print(f"\n   Analisando {total_jogos} jogos usando ML (max_workers=3)...")
        
        def analisar_jogo_com_ml(j):
            """Consulta o modelo ML para cada jogo. Retorna dict com previsão ou None."""
            try:
                resultado = prever_com_ml(
                    match_id=j['ID'],
                    home_id=j['Home_ID'],
                    away_id=j['Away_ID'],
                    tournament_id=j['Tournament_ID'],
                    season_id=j['Season_ID'],
                    odd_casa=float(j['Odd Casa']),
                    odd_empate=float(j['Empate']),
                    odd_fora=float(j['Odd Fora']),
                    liga=j['Liga'],
                    unique_tournament_id=j.get('Unique_Tournament_ID', ''),
                    home_team=j.get('Time Casa'), away_team=j.get('Time Fora'),
                    start_timestamp=j.get('Timestamp'),
                    research_run_id=j.get('Research_Run_ID'),
                )
                if resultado is None:
                    return None
                vencedor, confianca, relatorio = resultado
                snapshot_meta = load_prediction_snapshot_metadata(DB_NAME, j['ID'])
                
                if vencedor == 'MANDANTE':
                    odd_pick = float(j['Odd Casa'])
                    vencedor_nome = j['Time Casa']
                elif vencedor == 'VISITANTE':
                    odd_pick = float(j['Odd Fora'])
                    vencedor_nome = j['Time Fora']
                else:
                    odd_pick = float(j['Empate'])
                    vencedor_nome = 'Empate'
                
                return {
                    "ID": j['ID'],
                    "Confronto": j['Confronto'],
                    "Liga": j['Liga'],
                    "Vencedor Escolhido": vencedor_nome,
                    "Confiança": confianca,
                    "Relatório Técnico": relatorio,
                    "Dia_Str": j['Dia'],
                    "Hora BRT": j['Hora'],
                    "Liga_Exata": j['Liga'],
                    "Odd Casa": float(j['Odd Casa']),
                    "Odd Fora": float(j['Odd Fora']),
                    "Empate": float(j['Empate']),
                    "Tournament_ID": j['Tournament_ID'],
                    "Season_ID": j['Season_ID'],
                    "Unique_Tournament_ID": j.get('Unique_Tournament_ID', ''),
                    "Timestamp": int(j.get('Timestamp', 0) or 0),
                    "Tipo_Previsao": "ML",
                    "Odd_Calculada": odd_pick,
                    "Pick": vencedor,
                    "Risco_Empate": float(snapshot_meta.get('draw_risk', 0.0)),
                    "Qualidade_Contexto": float(snapshot_meta.get('information_quality', 0.0)),
                    "Conflito_Contexto": float(snapshot_meta.get('selection_conflict', 0.0)),
                    "Divergencia_Fontes": float(snapshot_meta.get('source_disagreement', 0.0)),
                    "Confiabilidade_Amostra": float(snapshot_meta.get('sample_reliability', 0.0)),
                    "Margem_Probabilidade": float(snapshot_meta.get('probability_margin', 0.0)),
                    "Entropia_Normalizada": float(snapshot_meta.get('normalized_entropy', 1.0)),
                    "Contexto_Detalhado_Ambos": float(snapshot_meta.get('detailed_both_available', 0.0)),
                    "Forma_Viva_Ambos": float(snapshot_meta.get('live_recent_both_available', 0.0)),
                    "Jogos_Recentes_Min": float(snapshot_meta.get('recent_games_min', 0.0)),
                    "Volatilidade_Competicao": float(snapshot_meta.get('competition_volatility', 0.08)),
                    "Corroboracao_Empate": int(snapshot_meta.get('draw_corroboration', 0)),
                    "Analysis_Version": snapshot_meta.get('source_version', ''),
                }
            except Exception as e:
                logger.error(f"Erro no ML para jogo {j.get('ID','?')}: {e}")
                return None
        
        max_workers = min(3, total_jogos)
        resultados_temp = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(analisar_jogo_com_ml, j): j for j in res_list}
            concluidos = 0
            for future in as_completed(futures):
                concluidos += 1
                res = future.result()
                if res:
                    resultados_temp.append(res)
                if concluidos % 10 == 0 or concluidos == total_jogos:
                    print(f"      ML processou {concluidos}/{total_jogos} jogos...")
        
        agregado_ind = resultados_temp
        print(f"   Jogos com previsão do ML: {len(agregado_ind)}")
        
        if not agregado_ind:
            print("   Nenhuma previsão gerada (modelos não disponíveis).")
            from phase2_observations import finish_run
            finish_run(DB_NAME, res_list, [], [], [], None, ml_model_version, ml_model_id)
            return
        
        # Remove somente repetições do mesmo evento. O texto do confronto não
        # é uma identidade: os mesmos clubes podem se enfrentar duas vezes na
        # janela e categorias distintas podem compartilhar rótulos.
        jogos_unicos = deduplicate_predictions(agregado_ind)
        print(f"   Jogos após deduplicação: {len(jogos_unicos)}")
        
        # O corte é aplicado somente à decisão. Todos os demais jogos continuam
        # sendo salvos, auditados e aprendidos pelo próximo ciclo.
        jogos_qualificados = [
            jogo for jogo in jogos_unicos
            if float(jogo.get("Confiança", 0)) >= limiar_ml_ativo
            and (PUBLICAR_TODAS_PREVISOES or PERMITIR_EMPATES_BILHETE or jogo.get("Pick") != "EMPATE")
            and (PUBLICAR_TODAS_PREVISOES or jogo.get("Pick") != "EMPATE"
                 or float(jogo.get("Confiança", 0)) >= MIN_DRAW_CONFIDENCE)
        ]
        jogos_ordenados = sorted(jogos_qualificados, key=lambda x: -x['Confiança'])
        print(f"   Seleções individuais aprovadas: {len(jogos_ordenados)}/{len(jogos_unicos)}")
        
        radar_run_id = get_brt_time().strftime("%Y%m%d_%H%M%S")
        run_id = radar_run_id.rsplit("_", 1)[-1]
        agregado_bil = selecionar_grupos_bilhetes(
            jogos_ordenados, min_confianca=limiar_ml_ativo,
            max_bilhetes=MAX_TICKETS_PER_RUN,
            permitir_empate=(PUBLICAR_TODAS_PREVISOES or PERMITIR_EMPATES_BILHETE), ligas_unicas=True,
            relaxar_ligas=True)
        for idx, bilhete in enumerate(agregado_bil, 1):
            bilhete["Nome"] = f"🛡️ Ticket Quant ML {bilhete['Categoria']} #{idx} [#{run_id}]"
        from phase2_observations import finish_run
        finish_run(DB_NAME, res_list, jogos_unicos, jogos_qualificados, agregado_bil,
                   radar_run_id, ml_model_version, ml_model_id)
        for bilhete in agregado_bil:
            print(f"✅ {bilhete['Nome']} | prob. conjunta estimada: {bilhete['Probabilidade Conjunta']:.2%}")
            for pos, jogo in enumerate(bilhete["Jogos"], 1):
                print(f"   │ {pos}. {jogo['Confronto']} → {jogo['Vencedor Escolhido']} ({jogo['Confiança']}%)")
        
        print(f"\n🎫 Total de bilhetes gerados: {len(agregado_bil)}")
        sobra = len(jogos_ordenados) - (len(agregado_bil) * 4)
        if sobra:
            print(f"   ℹ️ {sobra} previsão(ões) de menor confiança ficaram sem bilhete "
                  "para não repetir partidas; serão mantidas para aprendizado.")
        
        # Salva todas as previsões; a publicação é limitada apenas a grupos 4/4.
        with db_write_lock:
            conn = get_db_connection()
            cursor = conn.cursor()
            timestamp_atual = get_brt_time().strftime("%Y-%m-%d %H:%M:%S")
            ids_qualificados = {str(jogo["ID"]) for jogo in jogos_qualificados}
            for jogo in jogos_unicos:
                try:
                    cursor.execute("""INSERT OR IGNORE INTO previsoes
                        (match_id, timestamp, confronto, liga, odd_casa, odd_fora, odd_empate,
                         vencedor_previsto, confianca, scout_report, data_jogo, hora_jogo,
                         telegram_enviado, selecionado_radar, tournament_id, season_id,
                         unique_tournament_id, ml_model_version, ml_model_id, start_timestamp,
                         draw_risk_score, context_quality_score, context_conflict_score, analysis_version,
                         radar_run_id)
                        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                        (jogo["ID"], timestamp_atual, jogo.get("Confronto"), jogo.get("Liga_Exata"),
                         jogo.get("Odd Casa", 0.0), jogo.get("Odd Fora", 0.0), jogo.get("Empate", 0.0),
                         jogo.get("Vencedor Escolhido"), jogo.get("Confiança", 0),
                         jogo.get("Relatório Técnico", ""), jogo.get("Dia_Str", ""), jogo.get("Hora BRT", ""),
                         1, int(str(jogo["ID"]) in ids_qualificados), jogo.get("Tournament_ID"),
                          jogo.get("Season_ID"), jogo.get("Unique_Tournament_ID"),
                          ml_model_version, ml_model_id, int(jogo.get("Timestamp", 0) or 0),
                          jogo.get("Risco_Empate", 0.0), jogo.get("Qualidade_Contexto", 0.0),
                          jogo.get("Conflito_Contexto", 0.0), jogo.get("Analysis_Version", ""),
                          radar_run_id))
                except Exception as e:
                    logger.error(f"Erro ao registrar previsão de aprendizado {jogo['ID']}: {e}")
            for bilhete in agregado_bil:
                for jogo in bilhete["Jogos"]:
                    try:
                        cursor.execute("""UPDATE previsoes SET ticket_id=?, data_jogo=?, hora_jogo=?,
                            confianca=?, telegram_enviado=0, selecionado_radar=1 WHERE match_id=?""",
                                       (bilhete["Nome"], jogo.get("Dia_Str", ""), jogo.get("Hora BRT", ""),
                                        jogo.get("Confiança", 0), jogo["ID"]))
                    except Exception as e:
                        logger.error(f"Erro ao salvar jogo {jogo['ID']}: {e}")
            conn.commit()
            conn.close()
        
        # Salva último run_id somente quando houve bilhete para acompanhar.
        if agregado_bil:
          with db_write_lock:
            conn_control = get_db_connection()
            conn_control.execute("CREATE TABLE IF NOT EXISTS controle (chave TEXT PRIMARY KEY, valor TEXT)")
            conn_control.execute("INSERT OR REPLACE INTO controle (chave, valor) VALUES (?, ?)", ('ultimo_run_id', run_id))
            conn_control.commit()
            conn_control.close()
        
        print(f"✅ Radar concluído. {len(jogos_unicos)} previsões registradas para aprendizado; "
              f"{len(agregado_bil)} bilhetes 4/4 liberados.")
        
    except Exception as e:
        import traceback
        logger.error("Erro radar: " + traceback.format_exc())
        traceback.print_exc()
    finally:
        job_running['radar'] = False
        def _unlock_radar():
            time.sleep(60)
            job_running['radar'] = False
        threading.Thread(target=_unlock_radar, daemon=True).start()

def job_enviar_telegram():
    if job_running['envio']: return
    job_running['envio'] = True
    conn = None
    try:
        print("\n📲 ====== ENVIO TELEGRAM ======")
        conn = get_db_connection()
        cursor = conn.cursor()
        hoje = get_brt_time().strftime("%Y-%m-%d")
        cursor.execute("SELECT ticket_id FROM previsoes WHERE ticket_id IS NOT NULL AND telegram_enviado=0 AND timestamp>=? GROUP BY ticket_id", (hoje+" 00:00:00",))
        tickets = cursor.fetchall()
        if not tickets:
            return
        bilhetes = []
        for (tid,) in tickets:
            cursor.execute("SELECT match_id, confronto, liga, vencedor_previsto, odd_casa, odd_fora, odd_empate, confianca, scout_report, data_jogo, hora_jogo FROM previsoes WHERE ticket_id=? AND telegram_enviado=0", (tid,))
            rows = cursor.fetchall()
            if not rows:
                continue
            jogos = []
            for row in rows:
                match_id, confronto, liga, vencedor, oc, of, oe, conf, scout, d_j, h_j = row
                casa_nome = str(confronto).split(' vs ')[0].strip().lower()
                pick_lower = str(vencedor).lower()
                odd_jogo = float(oe) if 'empate' in pick_lower else (float(oc) if casa_nome in pick_lower else float(of))
                if odd_jogo <= 1.0:
                    odd_jogo = max(float(oc), float(of))
                jogos.append({"ID": match_id, "Confronto": confronto, "Liga_Exata": liga, "Vencedor Escolhido": vencedor,
                              "Confiança": conf, "Relatório Técnico": scout, "Dia_Str": d_j, "Hora BRT": h_j,
                              "Odd Casa": oc, "Odd Fora": of, "Empate": oe, "Odd_Calculada": odd_jogo})
            bilhetes.append({"Nome": tid, "Jogos": jogos})
        if not bilhetes:
            return
        if bilhetes:
            sucessos = disparar_telegram(bilhetes, TELEGRAM_TOKEN, TELEGRAM_CHAT_ID)
            print(f"   📲 {sucessos} bilhetes enviados.")
            if sucessos:
                with db_write_lock:
                    for b in bilhetes:
                        if not b.get("_telegram_enviado"):
                            continue
                        cursor.execute(
                            "UPDATE previsoes SET telegram_enviado=1 WHERE ticket_id=?",
                            (b['Nome'],),
                        )
                    conn.commit()
        conn.close()
    except Exception as e:
        logger.error(f"Erro envio: {e}")
    finally:
        if conn is not None:
            try:
                conn.close()
            except sqlite3.Error:
                pass
        job_running['envio'] = False

def job_ml_retreino_semanal():
    if job_running['ml_retreino']: return
    job_running['ml_retreino'] = True
    try:
        print("\n🧬 ====== JOB: EVOLUÇÃO AUTOMÁTICA CAMPEÃO x DESAFIANTE ======")
        if USAR_MODELOS_POR_LIGA:
            conn = get_db_connection()
            df_resumo = pd.read_sql_query("""
                SELECT t.liga, COUNT(*) AS total,
                       COALESCE(MAX(m.num_amostras),0) AS usados,
                       COUNT(*) - COALESCE(MAX(m.num_amostras),0) AS disponiveis
                FROM training_data t
                LEFT JOIN modelos_ml m ON m.liga=t.liga
                GROUP BY t.liga HAVING disponiveis >= 30
            """, conn)
            conn.close()
        else:
            # O global teve validação melhor e é o padrão; evita centenas de treinos sem uso.
            df_resumo = pd.DataFrame({'liga': []})
        ligas_treinadas = 0
        for liga in df_resumo['liga']:
            print(f"   🎓 Re‑treinando {liga}... ", end='', flush=True)
            res = treinar_modelo_liga_sem_vazamento(liga, min_amostras=100, acuracia_minima=0.35, usar_otimizacao=False)
            if res and res[0]:
                print(f"OK (acc {res[3]:.2%})")
                ligas_treinadas += 1
            else:
                print("descartado.")
        print(f"   ✅ {ligas_treinadas} ligas atualizadas.")
        print("🌍 Validando desafiante GLOBAL em janelas cronológicas...")
        resultado = executar_evolucao_automatica(forcar=False)
        if resultado.get("promote"):
            final = resultado.get("final", {})
            print(f"   ✅ Desafiante promovido | individual={final.get('selected_accuracy', 0):.2%} "
                  f"| n={final.get('selected', 0)} | corte={final.get('threshold', .5):.0%}")
        else:
            print(f"   🛡️ Campeão preservado: {resultado.get('reason', 'sem melhora validada')}")
        print("   📈 Pesos de erro e resultados reais preservados pela auditoria.")
    except Exception as e:
        logger.error(f"Erro no retreino semanal: {e}")
    finally:
        job_running['ml_retreino'] = False

def job_ml_otimizacao_mensal():
    if job_running['ml_otimizacao']: return
    job_running['ml_otimizacao'] = True
    try:
        print("\n🚀 ====== JOB: OTIMIZAÇÃO MENSAL (dia 1, 00:20) ======")
        conn = get_db_connection()
        df_resumo = pd.read_sql_query("""
            SELECT t.liga, COUNT(*) AS total,
                   COALESCE(MAX(m.num_amostras),0) AS usados,
                   COUNT(*) - COALESCE(MAX(m.num_amostras),0) AS disponiveis
            FROM training_data t
            LEFT JOIN modelos_ml m ON m.liga=t.liga
            GROUP BY t.liga HAVING disponiveis >= 50
        """, conn)
        conn.close()
        ligas_otimizadas = 0
        for liga in df_resumo['liga']:
            print(f"   ⚙️ Otimizando hiperparâmetros para {liga}... ", end='', flush=True)
            res = treinar_modelo_liga_sem_vazamento(liga, min_amostras=100, acuracia_minima=0.35, usar_otimizacao=True)
            if res and res[0]:
                print(f"OK (acc {res[3]:.2%})")
                ligas_otimizadas += 1
            else:
                print("otimização falhou ou descartada.")
        print(f"   ✅ {ligas_otimizadas} ligas com novos hiperparâmetros.")
    except Exception as e:
        logger.error(f"Erro na otimização mensal: {e}")
    finally:
        job_running['ml_otimizacao'] = False

def _job_backtest_legado_inseguro():
    if job_running['backtest']: return
    job_running['backtest'] = True
    try:
        print("\n📈 ====== JOB: BACKTEST (cego) ======")
        conn = get_db_connection()
        modelos = pd.read_sql_query("SELECT liga FROM modelos_ml ORDER BY data_treinamento DESC", conn)
        conn.close()
        if modelos.empty:
            print("   Nenhum modelo treinado.")
            return
        print("   Ligas disponíveis:")
        for i, row in modelos.iterrows():
            print(f"   [{i+1}] {row['liga']}")
        escolha = input("   Escolha o número da liga (ou Enter para pular): ").strip()
        if not escolha or not escolha.isdigit():
            return
        idx = int(escolha) - 1
        if idx < 0 or idx >= len(modelos):
            return
        liga = modelos.iloc[idx]['liga']
        print(f"   Executando backtest para {liga}...")
        model, scaler, feature_order = carregar_modelo_liga(liga)
        if model is None:
            return

        conn = get_db_connection()
        df_jogos = pd.read_sql_query("""SELECT match_id, data_jogo, home_team, away_team, home_score, away_score,
                                         features, odd_casa, odd_empate, odd_fora, liga
                                  FROM training_data WHERE liga = ? ORDER BY data_jogo ASC""",
                                  conn, params=(liga,))
        conn.close()
        if len(df_jogos) == 0:
            print("   Nenhum jogo disponível nesta liga.")
            return

        capital = 1000.0
        banca_inicial = capital
        aposta_por_bilhete = 1.0
        jogos_analisados_hoje = []
        data_atual = None
        total_bilhetes = 0
        total_verdes = 0

        def processar_bilhetes(lista_jogos_dia, capital, aposta, total_bilhetes, total_verdes):
            if len(lista_jogos_dia) < 4:
                return capital, total_bilhetes, total_verdes
            ordenados = sorted(lista_jogos_dia, key=lambda x: -x['confianca'])
            while len(ordenados) >= 4:
                ticket, ligas_no_ticket, restantes = [], {}, []
                for jogo in ordenados:
                    if len(ticket) == 4:
                        restantes.append(jogo)
                        continue
                    if ligas_no_ticket.get(jogo['liga'], 0) < 1:
                        ticket.append(jogo)
                        ligas_no_ticket[jogo['liga']] = 1
                    else:
                        restantes.append(jogo)
                if len(ticket) == 4:
                    odd_bilhete = 1.0
                    for j in ticket:
                        odd_bilhete *= j['odd_pick']
                    bilhete_verde = all(
                        (j['pick'] == 'MANDANTE' and j['real'][0] > j['real'][1]) or
                        (j['pick'] == 'VISITANTE' and j['real'][1] > j['real'][0]) or
                        (j['pick'] == 'EMPATE' and j['real'][0] == j['real'][1])
                        for j in ticket
                    )
                    lucro = aposta * odd_bilhete - aposta if bilhete_verde else -aposta
                    capital += lucro
                    total_bilhetes += 1
                    if bilhete_verde:
                        total_verdes += 1
                    ordenados = restantes
                else:
                    if len(ordenados) >= 4:
                        ticket = ordenados[:4]
                        odd_bilhete = 1.0
                        for j in ticket:
                            odd_bilhete *= j['odd_pick']
                        bilhete_verde = all(
                            (j['pick'] == 'MANDANTE' and j['real'][0] > j['real'][1]) or
                            (j['pick'] == 'VISITANTE' and j['real'][1] > j['real'][0]) or
                            (j['pick'] == 'EMPATE' and j['real'][0] == j['real'][1])
                            for j in ticket
                        )
                        lucro = aposta * odd_bilhete - aposta if bilhete_verde else -aposta
                        capital += lucro
                        total_bilhetes += 1
                        if bilhete_verde:
                            total_verdes += 1
                        ordenados = ordenados[4:]
                    else:
                        break
            return capital, total_bilhetes, total_verdes

        for _, jogo in df_jogos.iterrows():
            data_jogo = pd.to_datetime(jogo['data_jogo']).date() if pd.notna(jogo['data_jogo']) else None
            if data_jogo is None:
                continue

            if data_atual is not None and data_jogo != data_atual:
                capital, total_bilhetes, total_verdes = processar_bilhetes(
                    jogos_analisados_hoje, capital, aposta_por_bilhete,
                    total_bilhetes, total_verdes
                )
                jogos_analisados_hoje = []

            data_atual = data_jogo

            odd_casa = jogo['odd_casa']
            odd_fora = jogo['odd_fora']

            texto = f"{jogo['liga']} {jogo['home_team']} {jogo['away_team']}".lower()
            if any(termo in texto for termo in BLACKLIST_TERMS):
                continue

            feats = json.loads(jogo['features'])
            vec = [feats.get(k, 0.0) for k in feature_order]
            X = np.array([vec]); X_s = scaler.transform(X)
            proba = model.predict_proba(X_s)[0]
            idx_pred = np.argmax(proba)
            pick = ['MANDANTE', 'EMPATE', 'VISITANTE'][idx_pred]
            confianca = round(float(proba[idx_pred]) * 100.0, 2)

            jogos_analisados_hoje.append({
                'match_id': jogo['match_id'],
                'confronto': f"{jogo['home_team']} vs {jogo['away_team']}",
                'liga': jogo['liga'],
                'pick': pick,
                'confianca': confianca,
                'odd_pick': odd_casa if pick == 'MANDANTE' else (odd_fora if pick == 'VISITANTE' else jogo['odd_empate']),
                'real': (jogo['home_score'], jogo['away_score'])
            })

        if jogos_analisados_hoje:
            capital, total_bilhetes, total_verdes = processar_bilhetes(
                jogos_analisados_hoje, capital, aposta_por_bilhete,
                total_bilhetes, total_verdes
            )

        print(f"   Resultado Acumulado: R$ {capital - banca_inicial:+.2f}")
        print(f"   ROI: {((capital - banca_inicial) / banca_inicial) * 100:.2f}%")
        print(f"   Bilhetes: {total_bilhetes} (Verdes: {total_verdes})")
    except Exception as e:
        logger.error(f"Erro no backtest: {e}")
    finally:
        job_running['backtest'] = False

# ----------------------------------------------------------------------
# FUNÇÕES AUXILIARES PARA OS JOBS (NÃO DEFINIDAS ANTERIORMENTE)
# ----------------------------------------------------------------------
# Cache para armazenar resultados das estatísticas
CACHE_ESTATISTICAS = {}

def buscar_estatisticas_temporada_pro(team_id, tourn_id, season_id):
    # Validação dos parâmetros
    if not team_id or not tourn_id or not season_id or str(season_id).strip() == '' or str(tourn_id).strip() == '':
        return "Stats Liga N/A", False

    # Verificação de cache
    cache_key = f"{team_id}_{tourn_id}_{season_id}"
    if cache_key in CACHE_ESTATISTICAS:
        return CACHE_ESTATISTICAS[cache_key]

    url = f"https://{RAPIDAPI_HOST}/api/team/{team_id}/tournament/{tourn_id}/season/{season_id}/statistics"
    
    # Usando a função safe_api_get que retorna None em caso de erro
    data = safe_api_get(url, max_retries=2, timeout=10)

    # Tratamento para quando não há dados (204) ou ocorre um erro
    if not data:
        CACHE_ESTATISTICAS[cache_key] = ("Stats Liga N/A", False)
        return "Stats Liga N/A", False

    try:
        st_data = data.get('statistics', {})
        # Se o dicionário de estatísticas estiver vazio, também consideramos como "sem dados"
        if not st_data:
            CACHE_ESTATISTICAS[cache_key] = ("Stats Liga N/A", False)
            return "Stats Liga N/A", False
            
        xg = st_data.get('expectedGoals')
        posse = st_data.get('averageBallPossession')
        chutes = st_data.get('shots')
        
        if xg is None and posse is None and chutes is None:
            CACHE_ESTATISTICAS[cache_key] = ("Stats Liga N/A", False)
            return "Stats Liga N/A", False
            
        result_str = f"Posse: {posse if posse else 'N/A'}% | Chutes: {chutes if chutes else 'N/A'} | xG: {xg if xg else 'N/A'}"
        CACHE_ESTATISTICAS[cache_key] = (result_str, True)
        return result_str, True
    except Exception as e:
        logger.error(f"Erro ao processar estatísticas para team {team_id}: {e}")
        CACHE_ESTATISTICAS[cache_key] = ("Stats Liga N/A", False)
        return "Stats Liga N/A", False

def buscar_form_rating(match_id):
    data = safe_api_get(match_resource_url(RAPIDAPI_HOST, match_id, "form"))
    try:
        return f"Form Rating: C:{data.get('home', {}).get('form', '-')} | F:{data.get('away', {}).get('form', '-')}"
    except:
        return "Form N/A"

def buscar_power_ranking_pro(tourn_id, season_id):
    # A v2.0 documenta power rankings somente para tênis de mesa. Não gastar
    # cota tentando uma rota de futebol que agora responde 404.
    return "Sem PR (indisponível para futebol na API v2.0)."

# Recomendo adicionar um dicionário de cache global no início do seu arquivo,
# perto de outros caches como 'CACHE_ESTATISTICAS'.
CACHE_PERFORMANCE = {}

def buscar_performance_grafico(tourn_id, season_id, team_id):
    # 1. Validação de segurança para evitar chamadas desnecessárias
    if not tourn_id or not season_id or not team_id or str(season_id).strip() == '':
        return "Perf N/A (IDs inválidos)"

    # 2. Verificação de cache para evitar chamadas repetidas
    cache_key = f"{team_id}_{tourn_id}_{season_id}"
    if cache_key in CACHE_PERFORMANCE:
        return CACHE_PERFORMANCE[cache_key]

    # 3. Chamada segura à API
    url = f"https://{RAPIDAPI_HOST}/api/tournament/{tourn_id}/season/{season_id}/team/{team_id}/performance"
    data = safe_api_get(url)

    # 4. Tratamento da resposta vazia (204) ou erro
    if not data:
        CACHE_PERFORMANCE[cache_key] = "Perf N/A (sem dados)"
        return "Perf N/A (sem dados)"

    # 5. Processamento padrão (se houver dados)
    try:
        result = f"Perf: {data.get('performance', {}).get('overall', 'N/A')}"
    except Exception:
        result = "Perf N/A (erro no processamento)"

    # 6. Armazena o resultado no cache e o retorna
    CACHE_PERFORMANCE[cache_key] = result
    return result

def buscar_win_probability_pro(match_id):
    data = safe_api_get(match_resource_url(RAPIDAPI_HOST, match_id, "win-probability"))
    try:
        if data:
            return f"WinProb: C{data.get('home',0)}% - E{data.get('draw',0)}% - F{data.get('away',0)}%"
    except:
        pass
    return "WinProb N/A"

def buscar_dados_jogadores_escalacao(match_id):
    data = safe_api_get(match_resource_url(RAPIDAPI_HOST, match_id, "lineups"))
    try:
        home_form = data.get('home', {}).get('formation', 'N/A')
        away_form = data.get('away', {}).get('formation', 'N/A')
        return f"Tática Casa: {home_form} | Tática Fora: {away_form}"
    except:
        return "Escalações N/A"

def buscar_streaks_odds_pro(match_id):
    data = safe_api_get(f"https://{RAPIDAPI_HOST}/api/match/{match_id}/streaks/odds")
    return "Streaks: " + " | ".join([f"{s.get('name','')}" for s in data.get('streaks',[])[:3]]) if data and data.get('streaks') else "Sem Streaks"

# Cache global para armazenar resultados de "goal-distributions"
CACHE_GOAL_DIST = {}

def buscar_distribuicao_gols_pro(team_id, tourn_id, season_id):
    # Validação: só prossegue se todos os IDs forem válidos
    if not team_id or not tourn_id or not season_id or str(season_id).strip() == '' or str(tourn_id).strip() == '':
        return "Minutos Gols N/A (IDs inválidos)"
    
    cache_key = f"{team_id}_{tourn_id}_{season_id}"
    if cache_key in CACHE_GOAL_DIST:
        return CACHE_GOAL_DIST[cache_key]
    
    url = f"https://{RAPIDAPI_HOST}/api/team/{team_id}/tournament/{tourn_id}/season/{season_id}/goal-distributions"
    data = safe_api_get(url)
    
    if not data:
        CACHE_GOAL_DIST[cache_key] = "Minutos Gols N/A (sem dados)"
        return CACHE_GOAL_DIST[cache_key]
    
    try:
        scored_list = data.get('scored', [])
        conceded_list = data.get('conceded', [])
        
        sc = max(scored_list, key=lambda x: x.get('value',0)) if scored_list else {}
        cc = max(conceded_list, key=lambda x: x.get('value',0)) if conceded_list else {}
        
        sc_interval = sc.get('interval', '-') if sc else '-'
        cc_interval = cc.get('interval', '-') if cc else '-'
        
        result = f"Marca(+): {sc_interval}m | Sofre(+): {cc_interval}m"
        CACHE_GOAL_DIST[cache_key] = result
        return result
    except Exception as e:
        logger.error(f"Erro ao processar goal-distributions para team {team_id}: {e}")
        CACHE_GOAL_DIST[cache_key] = "Minutos Gols N/A (erro)"
        return CACHE_GOAL_DIST[cache_key]

def buscar_eficiencia_time(team_id, tourn_id, season_id):
    # Implementação simplificada
    return 1.0, 1.0, True

# ----------------------------------------------------------------------
# SCHEDULER
# ----------------------------------------------------------------------
def scheduler_thread():
    print("🕒 Scheduler iniciado.")
    ultima_execucao = {k: None for k in job_running}
    while True:
        now = get_brt_time()
        hhmm = now.strftime("%H:%M")
        date = now.date()
        weekday = now.weekday()
        day = now.day
        # ❌ Coleta diária removida
        if hhmm == "15:20" and ultima_execucao['ml_retreino'] != date:
            threading.Thread(target=job_ml_retreino_semanal).start()
            ultima_execucao['ml_retreino'] = date
        if day == 1 and hhmm == "00:20" and ultima_execucao['ml_otimizacao'] != date:
            threading.Thread(target=job_ml_otimizacao_mensal).start()
            ultima_execucao['ml_otimizacao'] = date
        if hhmm == AUDIT_SCHEDULE_TIME and ultima_execucao['auditoria'] != date:
            threading.Thread(target=job_auditoria_e_deeplab).start()
            ultima_execucao['auditoria'] = date
        if hhmm == "18:00" and ultima_execucao['radar'] != date:
            threading.Thread(target=job_radar_e_analise).start()
            ultima_execucao['radar'] = date
        if hhmm == "21:00" and ultima_execucao['envio'] != hhmm:
            threading.Thread(target=job_enviar_telegram).start()
            ultima_execucao['envio'] = hhmm
        time.sleep(30)

# O código acima é mantido apenas para permitir auditoria de versões antigas;
# ele não é exposto porque treinava e testava no mesmo conjunto. Este é o
# relatório operacional realmente cego: usa somente o pick congelado antes do
# jogo e o desfecho apurado depois.
def job_backtest():
    if job_running['backtest']:
        return
    job_running['backtest'] = True
    try:
        print("\n📈 ====== BACKTEST OPERACIONAL SEM VAZAMENTO ======")
        report = operational_snapshot_report(DB_NAME)
        print(
            f"   Individuais: {report['correct_predictions']}/"
            f"{report['resolved_predictions']} "
            f"({report['individual_accuracy']:.2%})"
        )
        print(
            "   Recall: casa "
            f"{report['recall_by_outcome'].get('MANDANTE', 0):.2%}, "
            f"empate {report['recall_by_outcome'].get('EMPATE', 0):.2%}, "
            f"fora {report['recall_by_outcome'].get('VISITANTE', 0):.2%}"
        )
        print(
            f"   Bilhetes 4/4: {report['green_tickets']}/"
            f"{report['complete_tickets']} "
            f"({report['ticket_green_rate']:.2%})"
        )
        print(f"   Distribuição de acertos por bilhete: {report['ticket_hits_distribution']}")
    except Exception as exc:
        logger.exception("Erro no backtest operacional: %s", exc)
    finally:
        job_running['backtest'] = False


# ----------------------------------------------------------------------
# MENU INTERATIVO
# ----------------------------------------------------------------------
COMANDOS_INTERATIVOS = {
    '2': job_auditoria_e_deeplab, 'auditoria': job_auditoria_e_deeplab,
    '3': job_radar_e_analise, 'radar': job_radar_e_analise,
    '4': job_enviar_telegram, 'envio': job_enviar_telegram,
    '5': job_ml_retreino_semanal, 'retreino': job_ml_retreino_semanal,
    'evoluir': job_ml_retreino_semanal,
    '6': job_ml_otimizacao_mensal, 'otimizar': job_ml_otimizacao_mensal,
    '8': job_backtest, 'backtest': job_backtest,
}

def executar_comando_interativo(cmd):
    """Executa um comando em primeiro plano e mantém o terminal previsível."""
    cmd = str(cmd or '').strip().lower()
    if cmd in ('7', 'sair'):
        print("👋 Encerrando o robô com segurança.")
        return False
    func = COMANDOS_INTERATIVOS.get(cmd)
    if func is None:
        print("Comando inválido.")
        return True
    nome = getattr(func, '__name__', cmd)
    print(f"▶️ Executando {nome} em primeiro plano...")
    try:
        func()
    except KeyboardInterrupt:
        print("\n⚠️ Operação interrompida pelo usuário. O robô continua no menu.")
    except Exception:
        logger.exception("Falha não tratada no comando '%s'", cmd)
        print("❌ O comando falhou. Consulte o erro acima; o robô continua ativo.")
    else:
        print(f"✅ Comando '{cmd}' finalizado. Voltando ao menu.")
    return True

def menu_interativo():
    print("\n" + "="*50)
    print("        NEXUS QUANT - MENU DE CONTROLE")
    print("="*50)
    print("  2 ou 'auditoria' - Auditoria de resultados")
    print("  3 ou 'radar'     - Radar e análise")
    print("  4 ou 'envio'     - Enviar bilhetes ao Telegram")
    print("  5 ou 'evoluir'    - Evolução validada do ML (agora)")
    print("  6 ou 'otimizar'  - Otimização mensal (agora)")
    print("  8 ou 'backtest'  - Executar backtest (cego)")
    print("  7 ou 'sair'      - Encerrar o robô")
    print("="*50)
    while True:
        try:
            cmd = input("\n> ").strip().lower()
            if not executar_comando_interativo(cmd):
                return
        except KeyboardInterrupt:
            print("\n👋 Encerrando o robô com segurança.")
            return
        except EOFError:
            print("\n⚠️ A entrada do terminal foi fechada. Encerrando o menu com segurança.")
            return
        except Exception as e:
            logger.exception("Erro inesperado no menu: %s", e)

if __name__ == "__main__":
    print("\n🚀 NEXUS QUANT BOT INICIADO (v8 - ML de Duelo Pré-Jogo - Sincronizado com app.py)")
    if len(sys.argv) > 1:
        comando_cli = sys.argv[1].strip().lower()
        if comando_cli not in COMANDOS_INTERATIVOS and comando_cli not in ('7', 'sair'):
            print(f"❌ Comando de linha inválido: {comando_cli}")
            print("   Use: radar, auditoria, envio, evoluir, otimizar ou backtest")
            raise SystemExit(2)
        executar_comando_interativo(comando_cli)
    else:
        threading.Thread(target=scheduler_thread, daemon=True).start()
        menu_interativo()
