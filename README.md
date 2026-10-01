# Plataforma de Análise Preditiva

Projeto local de radar, auditoria e pesquisa prospectiva de futebol. Os dois pontos de entrada são `robo_auto.py` (terminal) e `app.py` (Streamlit).

## Preparação

1. Crie um ambiente Python 3.12 e instale `requirements.txt`.
2. Configure `.streamlit/secrets.toml` com `TELEGRAM_TOKEN` e `TELEGRAM_CHAT_ID` para usar o painel. O robô também aceita essas variáveis em `.env`.
3. Configure `api_keys.local.json` com listas `allsports` e `soccer`, ou as variáveis de ambiente `RAPIDAPI_KEYS` e `SOCCER_API_KEYS` (chaves separadas por vírgula). Os arquivos de configuração local não são versionados.
4. Para continuar com o histórico existente, mantenha `ia_sports_v5.db` e `phase2_observations.db` na raiz do projeto. Os caches `allsports_schedule_cache.db` e `free_football_data.db` também permanecem locais.

## Execução

```powershell
.\venv312\Scripts\python.exe robo_auto.py
.\venv312\Scripts\python.exe robo_auto.py radar
.\venv312\Scripts\python.exe -m streamlit run app.py
```

Para o acompanhamento prospectivo Fase 4B, consulte `phase4_labels.py` e `phase4_readiness.py`. Não execute um segundo radar na mesma data sem verificar primeiro a saúde do run já registrado.

## Dados e segurança

Bancos SQLite, relatórios, ambiente virtual, logs e segredos são intencionalmente excluídos do Git. O repositório contém código e testes, **não** um backup dos dados de treinamento ou do estado operacional. Guarde cópias dos bancos e de `api_keys.local.json` em local privado. Antes de publicar, confirme que o repositório remoto é privado.
