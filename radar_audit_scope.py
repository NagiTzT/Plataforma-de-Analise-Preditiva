"""Identifica com segurança os lotes do radar usados pela auditoria."""


def ensure_radar_audit_schema(conn):
    """Adiciona o run_id e associa previsões antigas ao timestamp do radar."""
    columns = {row[1] for row in conn.execute("PRAGMA table_info(previsoes)")}
    if "radar_run_id" not in columns:
        conn.execute("ALTER TABLE previsoes ADD COLUMN radar_run_id TEXT")
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_previsoes_radar_run ON previsoes(radar_run_id, status_resultado)"
    )
    missing = conn.execute(
        """SELECT 1 FROM previsoes
           WHERE radar_run_id IS NULL OR TRIM(radar_run_id)='' LIMIT 1"""
    ).fetchone()
    if missing:
        conn.execute(
            """UPDATE previsoes
               SET radar_run_id='legacy:' || COALESCE(NULLIF(TRIM(timestamp), ''), 'row:' || id)
               WHERE radar_run_id IS NULL OR TRIM(radar_run_id)=''"""
        )


def get_latest_radar_run_ids(conn, limit=2):
    """Retorna os últimos lotes reais do radar, incluindo dados legados."""
    limit = max(1, int(limit))
    rows = conn.execute(
        """SELECT radar_run_id
           FROM previsoes
           WHERE radar_run_id IS NOT NULL AND TRIM(radar_run_id)!=''
           GROUP BY radar_run_id
           ORDER BY MAX(timestamp) DESC, MAX(id) DESC
           LIMIT ?""",
        (limit,),
    ).fetchall()
    return [str(row[0]) for row in rows]


def radar_run_placeholders(run_ids):
    """Gera placeholders somente para uma lista já validada e não vazia."""
    return ",".join("?" for _ in run_ids)
