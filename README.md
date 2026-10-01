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

## Correção de coleta Fase 4B

Depois de um radar completo e válido, a coleta auxiliar pode executar:

```powershell
.\venv312\Scripts\python.exe phase4_context_collection.py --db ia_sports_v5.db --repair-metadata --max-requests 60 --max-games 12
```

O reparo de país utiliza exclusivamente o nome de liga já congelado e acrescenta um registro com o hash original; não substitui dados antigos. País não identificável continua ausente.

O coletor `phase4-covariates-v2` consulta somente jogos futuros do pool de pesquisa, alterna os grupos elegível/não elegível sem consultar resultados e prioriza completar pelo menos três partidas anteriores medidas de **cada** equipe. Usa históricos recentes, estatísticas locais medidas e a API AllSports como alternativa autorizada. Limita as tentativas HTTP, inclusive a rotação de chaves; não contorna bloqueios do fornecedor. O cache auxiliar `phase4_context.db` permanece local. Dados ausentes não viram zero; zero medido é válido.

A reserva diária é registrada antes do HTTP e impede nova coleta de rede no mesmo dia BRT, inclusive após interrupção. Execuções repetidas usam somente cache local; um reinício não renova artificialmente o orçamento.

Essas covariáveis são capturadas separadamente antes do início, sem modificar banco operacional, previsões, modelo ou bilhetes. O relatório distingue cobertura dos atributos originalmente congelados e cobertura auxiliar. A cobertura auxiliar **não libera retroativamente** o holdout original; usar novas entradas num challenger exige versão e holdout futuro próprios. Todos os mínimos pré-registrados permanecem inalterados. País recuperado e finalizações já presentes nos atributos congelados são correções de metadados/contagem, não reconstruções retrospectivas de estatísticas.

## Dados e segurança

Bancos SQLite, relatórios, ambiente virtual, logs e segredos são intencionalmente excluídos do Git. O repositório contém código e testes, **não** um backup dos dados de treinamento ou do estado operacional. Guarde cópias dos bancos e de `api_keys.local.json` em local privado. Antes de publicar, confirme que o repositório remoto é privado.
