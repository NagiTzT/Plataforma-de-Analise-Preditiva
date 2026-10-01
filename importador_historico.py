# importador_historico.py
# Importa jogos finalizados dos últimos 5 ANOS (1825 dias) para a base de treinamento.
# Roda em paralelo com robo_auto.py / app.py.
# Pausa automaticamente entre 23:50 e 03:00 para não conflitar com o job ML diário.
# Pode ser interrompido e reiniciado – não duplica dados.

import requests, time, os, json, sqlite3, re, logging, sys
from datetime import datetime, timezone, timedelta
import threading
from football_results import regulation_score
from api_key_config import configured_keys

# ============ CONFIGURAÇÕES (idênticas ao robo_auto.py) ============
RAPIDAPI_KEY = next(iter(configured_keys("RAPIDAPI_KEYS", "allsports")), "")
RAPIDAPI_HOST = "allsportsapi2.p.rapidapi.com".strip()
HEADERS = {
    "x-rapidapi-host": RAPIDAPI_HOST,
    "x-rapidapi-key": RAPIDAPI_KEY,
    "Content-Type": "application/json"
}
DB_NAME = 'ia_sports_v5.db'

# Rate limiter conservador
_rate_limiter = threading.Semaphore(1)   # apenas 1 worker
_last_request_time = 0
_rate_lock = threading.Lock()

db_write_lock = threading.Lock()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("importador.log", encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def get_brt_time():
    return datetime.now(timezone(timedelta(hours=-3)))

def get_db_connection():
    conn = sqlite3.connect(DB_NAME, timeout=60, check_same_thread=False)
    conn.execute("PRAGMA busy_timeout = 30000")
    return conn

# ---------- Funções copiadas do robo_auto.py (sem dependências externas) ----------
def extrair_fracional(frac_str):
    try:
        if not frac_str: return 0.0
        if '/' in str(frac_str):
            n, d = str(frac_str).split('/')
            return round((float(n) / float(d)) + 1.0, 2)
        return float(frac_str)
    except: return 0.0

def safe_api_get(url, max_retries=3, timeout=15):
    global _last_request_time
    for attempt in range(max_retries):
        with _rate_limiter:
            with _rate_lock:
                now = time.time()
                elapsed = now - _last_request_time
                if elapsed < 0.35: time.sleep(0.35 - elapsed)
                _last_request_time = time.time()
            try:
                res = requests.get(url, headers=HEADERS, timeout=timeout)
                if res.status_code == 200: return res.json()
                elif res.status_code == 204: return {}
                elif res.status_code == 429:
                    time.sleep(3)
                    continue
                else:
                    time.sleep(1)
            except:
                time.sleep(2)
    return None
def filtrar_features_sem_vazamento(features_dict):
    """Remove as features que contêm informações do próprio jogo."""
    proibidas = [
        'xg_casa', 'xg_fora', 'posse_casa', 'posse_fora',
        'chutes_casa', 'chutes_fora', 'chutes_gol_casa', 'chutes_gol_fora',
        'escanteios_casa', 'escanteios_fora', 'faltas_casa', 'faltas_fora',
        'xg_diff', 'posse_diff', 'chutes_diff', 'chutes_gol_diff',
        'escanteios_diff', 'faltas_diff'
    ]
    return {k: v for k, v in features_dict.items() if k not in proibidas}

def analisar_ultimos_jogos_pro(team_id, limit=5, tipo='geral'):
    data = safe_api_get(f"https://{RAPIDAPI_HOST}/api/team/{team_id}/matches/previous/0")
    if not data: return "Form N/A"
    try:
        v=e=d=count=0
        for ev in data.get('events', []):
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

def extrair_v_e_d(form_str):
    v=e=d=0
    mv = re.search(r'(\d+)V', form_str)
    if mv: v = int(mv.group(1))
    me = re.search(r'(\d+)E', form_str)
    if me: e = int(me.group(1))
    md = re.search(r'(\d+)D', form_str)
    if md: d = int(md.group(1))
    return v, e, d

def obter_dias_descanso(team_id):
    data = safe_api_get(f"https://{RAPIDAPI_HOST}/api/team/{team_id}/matches/previous/0")
    if data and 'events' in data:
        for ev in data['events']:
            if ev.get('status', {}).get('type') == 'finished':
                last = datetime.fromtimestamp(ev['startTimestamp'], tz=timezone.utc)
                agora = datetime.now(timezone.utc)
                return (agora - last).days
    return 7

def extrair_features_basicas(match_id, home_id=None, away_id=None, tournament_id=None, season_id=None,
                             odd_casa=2.0, odd_empate=3.0, odd_fora=2.0):
    features = {
        'odd_casa': odd_casa, 'odd_empate': odd_empate, 'odd_fora': odd_fora,
        'prob_impl_casa': 1/odd_casa if odd_casa>1 else 0.5,
        'prob_impl_empate': 1/odd_empate if odd_empate>1 else 0.33,
        'prob_impl_fora': 1/odd_fora if odd_fora>1 else 0.5,
        'diff_odds': odd_casa - odd_fora,
        'ratio_odds': odd_casa / odd_fora if odd_fora>0 else 1.0,
    }
    if home_id:
        form_home = analisar_ultimos_jogos_pro(home_id, 5, 'geral')
        v_h,e_h,d_h = extrair_v_e_d(form_home)
        features['v_home_5'] = v_h; features['e_home_5'] = e_h; features['d_home_5'] = d_h
        features['dias_descanso_home'] = obter_dias_descanso(home_id)
    else:
        features['v_home_5']=2; features['e_home_5']=1; features['d_home_5']=2; features['dias_descanso_home']=7
    if away_id:
        form_away = analisar_ultimos_jogos_pro(away_id, 5, 'geral')
        v_a,e_a,d_a = extrair_v_e_d(form_away)
        features['v_away_5'] = v_a; features['e_away_5'] = e_a; features['d_away_5'] = d_a
        features['dias_descanso_away'] = obter_dias_descanso(away_id)
    else:
        features['v_away_5']=2; features['e_away_5']=1; features['d_away_5']=2; features['dias_descanso_away']=7
    stats = safe_api_get(f"https://{RAPIDAPI_HOST}/api/match/{match_id}/statistics", timeout=10)
    xg_c=xg_f=posse_c=posse_f=0.0
    if stats and 'statistics' in stats and len(stats['statistics'])>0:
        first = stats['statistics'][0]
        for group in first.get('groups', []):
            for item in group.get('statisticsItems', []):
                name = item.get('name')
                if name == 'Expected goals':
                    xg_c = float(item.get('home',0)); xg_f = float(item.get('away',0))
                elif name == 'Ball possession':
                    hv = item.get('home','0%'); av = item.get('away','0%')
                    posse_c = float(hv.replace('%','')) if isinstance(hv,str) else float(hv)
                    posse_f = float(av.replace('%','')) if isinstance(av,str) else float(av)
    features['xg_casa']=xg_c; features['xg_fora']=xg_f; features['posse_casa']=posse_c; features['posse_fora']=posse_f
    features['xg_diff']=xg_c-xg_f; features['posse_diff']=posse_c-posse_f
    return features

# ============ FUNÇÃO DE PAUSA NOTURNA ============
def deve_pausar():
    """Retorna True se o horário atual (BRT) está entre 23:50 e 03:00."""
    agora = get_brt_time()
    hora_minuto = agora.hour * 60 + agora.minute
    # 23:50 = 1430 minutos, 03:00 = 180 minutos (do dia seguinte)
    inicio_pausa = 23 * 60 + 50   # 1430
    fim_pausa = 3 * 60             # 180
    if hora_minuto >= inicio_pausa or hora_minuto < fim_pausa:
        return True
    return False

def aguardar_fim_pausa():
    """Dorme até o fim da janela de pausa (03:00)."""
    agora = get_brt_time()
    # Calcula próximo 03:00
    if agora.hour >= 3:
        # Já passou das 3h, o próximo 3h é amanhã
        proximo = agora.replace(hour=3, minute=0, second=0, microsecond=0) + timedelta(days=1)
    else:
        proximo = agora.replace(hour=3, minute=0, second=0, microsecond=0)
    segundos = (proximo - agora).total_seconds()
    print(f"⏸️  Pausa automática (23:50-03:00). Retomando às {proximo.strftime('%H:%M')}.")
    logger.info(f"Pausa automática iniciada. Retomando em {segundos/60:.0f} minutos.")
    time.sleep(segundos)

# ============ IMPORTAÇÃO DE UM DIA ============
def importar_um_dia(d_str):
    """Importa todos os jogos finalizados de uma data específica."""
    blacklist = ["u17","u19","u20","u21","u22","u23","u24","sub-","sub17","sub19","sub20","sub21","sub23",
                 "amateur","amador","amadores","youth","juniors","aspirantes","reserva","reservas","reserve","reserves",
                 "woman","women","feminino","femenino","femmes","frauen"," w ","ladies","girls","sub","junior"]

    url = f"https://{RAPIDAPI_HOST}/api/matches/{d_str}"
    data = safe_api_get(url, max_retries=3, timeout=25)
    if not data or 'events' not in data:
        return 0

    eventos = data['events']
    finalizados = [ev for ev in eventos if ev.get('status',{}).get('type')=='finished' and
                   not any(termo in (ev.get('tournament',{}).get('category',{}).get('name','')+' '+ev.get('tournament',{}).get('name','')).lower() for termo in blacklist)]

    if not finalizados:
        return 0

    novos = 0
    for ev in finalizados:
        match_id = str(ev.get('id'))
        # Verifica se já existe
        try:
            conn_check = get_db_connection()
            cur = conn_check.cursor()
            cur.execute("SELECT 1 FROM training_data WHERE match_id=?", (match_id,))
            existe = cur.fetchone() is not None
            conn_check.close()
            if existe:
                continue
        except:
            pass

        home_team = ev.get('homeTeam',{}).get('name','Desconhecido')
        away_team = ev.get('awayTeam',{}).get('name','Desconhecido')
        home_id = str(ev.get('homeTeam',{}).get('id',''))
        away_id = str(ev.get('awayTeam',{}).get('id',''))
        tournament_id = str(ev.get('tournament',{}).get('id',''))
        season_id = str(ev.get('season',{}).get('id',''))
        liga = f"{ev.get('tournament',{}).get('category',{}).get('name','Mundo')} - {ev.get('tournament',{}).get('name','Liga')}"
        start_ts = ev.get('startTimestamp',0)
        dt_jogo = datetime.fromtimestamp(start_ts, tz=timezone(timedelta(hours=-3)))
        score = regulation_score(ev)
        if score is None:
            continue
        hs, aws = score

        # Odds
        odd_casa = odd_empate = odd_fora = 0.0
        odds = safe_api_get(f"https://{RAPIDAPI_HOST}/api/match/{match_id}/odds", timeout=10)
        if odds and 'markets' in odds:
            for market in odds['markets']:
                if market.get('name') == '1X2':
                    for choice in market.get('choices',[]):
                        if choice.get('name')=='1': odd_casa = extrair_fracional(choice.get('fractionalValue'))
                        elif choice.get('name')=='X': odd_empate = extrair_fracional(choice.get('fractionalValue'))
                        elif choice.get('name')=='2': odd_fora = extrair_fracional(choice.get('fractionalValue'))
                    break
        if odd_casa == 0.0:
            odd_casa, odd_empate, odd_fora = 2.0, 3.0, 2.0

        features = extrair_features_basicas(match_id, home_id, away_id, tournament_id, season_id,
                                            odd_casa, odd_empate, odd_fora)
        try:
            with db_write_lock:
                conn = get_db_connection()
                cur = conn.cursor()
                cur.execute('''INSERT INTO training_data
                    (match_id, liga, data_jogo, home_team, away_team, home_score, away_score,
                     odd_casa, odd_empate, odd_fora, features)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?)''',
                    (match_id, liga, dt_jogo.strftime("%Y-%m-%d %H:%M:%S"),
                     home_team, away_team, hs, aws,
                     odd_casa, odd_empate, odd_fora, json.dumps(features)))
                cur.execute("INSERT OR IGNORE INTO training_weights (match_id, peso, data_ultima_atualizacao) VALUES (?,1.0,?)",
                            (match_id, dt_jogo.strftime("%Y-%m-%d %H:%M:%S")))
                conn.commit()
                conn.close()
            novos += 1
        except Exception as e:
            logger.error(f"Erro ao inserir {match_id}: {e}")

    return novos

# ============ MAIN ============
if __name__ == "__main__":
    DIAS_TOTAIS = 1825   # 5 anos
    print(f"🚀 Importador de Histórico ({DIAS_TOTAIS} dias ≈ 5 anos) iniciado.")
    logger.info(f"Iniciando importação dos últimos {DIAS_TOTAIS} dias...")
    agora = get_brt_time()
    total_importados = 0

    for dia_offset in range(DIAS_TOTAIS):
        # Verifica pausa noturna ANTES de cada dia
        if deve_pausar():
            aguardar_fim_pausa()

        data_alvo = (agora - timedelta(days=dia_offset)).strftime("%d/%m/%Y")
        print(f"[{dia_offset+1}/{DIAS_TOTAIS}] Processando {data_alvo}...", end=' ', flush=True)
        try:
            n = importar_um_dia(data_alvo)
            total_importados += n
            print(f"{n} jogos importados.")
            logger.info(f"Data {data_alvo}: {n} jogos importados.")
        except Exception as e:
            print(f"Erro: {e}")
            logger.error(f"Falha na data {data_alvo}: {e}")

        # Pausa maior a cada 10 dias para descanso da API
        if (dia_offset + 1) % 10 == 0:
            print("⏳ Pausa de 10 segundos (a cada 10 dias)...")
            time.sleep(10)
        else:
            time.sleep(0.8)   # pequena pausa entre dias para não sobrecarregar

    print(f"\n✅ Importação concluída! Total de novos jogos: {total_importados}")
    logger.info(f"Importação histórica finalizada. Total: {total_importados}")
