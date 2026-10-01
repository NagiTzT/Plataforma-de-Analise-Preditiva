"""Idempotencia compartilhada para envios de bilhetes ao Telegram.

O app e o robo podem rodar em processos diferentes. Uma flag mantida apenas em
memoria nao impede que ambos leiam ``telegram_enviado=0`` e enviem a mesma
mensagem. Este modulo usa uma transacao SQLite como trava entre processos e
tambem bloqueia a mesma composicao quando ela recebe outro nome de bilhete.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time


def delivery_fingerprint(jogos):
    legs = sorted(
        (
            str(jogo.get("ID") or jogo.get("match_id") or ""),
            str(jogo.get("Vencedor Escolhido") or jogo.get("vencedor_previsto") or "").strip().upper(),
        )
        for jogo in (jogos or [])
    )
    raw = json.dumps(legs, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _connect(db_path):
    conn = sqlite3.connect(db_path, timeout=60, isolation_level=None)
    conn.execute("PRAGMA busy_timeout=60000")
    return conn


def _ensure_schema(conn):
    conn.execute(
        """CREATE TABLE IF NOT EXISTS telegram_delivery_claims (
               ticket_id TEXT PRIMARY KEY,
               payload_hash TEXT NOT NULL,
               status TEXT NOT NULL,
               claimed_at INTEGER NOT NULL,
               sent_at INTEGER,
               message_id TEXT,
               last_error TEXT
           )"""
    )
    conn.execute(
        """CREATE UNIQUE INDEX IF NOT EXISTS idx_telegram_delivery_payload
           ON telegram_delivery_claims(payload_hash)"""
    )


def claim_ticket_delivery(db_path, ticket_id, payload_hash, lease_seconds=900):
    """Tenta reservar um envio.

    Retorna ``(True, 'claimed', None)`` para o unico processo autorizado. Um
    estado UNKNOWN nunca e retomado automaticamente, pois um timeout pode ter
    acontecido depois de o Telegram aceitar a mensagem.
    """
    now = int(time.time())
    conn = _connect(db_path)
    try:
        conn.execute("BEGIN IMMEDIATE")
        _ensure_schema(conn)
        row = conn.execute(
            """SELECT ticket_id,status,claimed_at,message_id
               FROM telegram_delivery_claims
               WHERE ticket_id=? OR payload_hash=?
               ORDER BY CASE WHEN ticket_id=? THEN 0 ELSE 1 END LIMIT 1""",
            (str(ticket_id), str(payload_hash), str(ticket_id)),
        ).fetchone()
        if row:
            existing_ticket, status, claimed_at, message_id = row
            status = str(status or "").upper()
            stale = status == "SENDING" and now - int(claimed_at or 0) > int(lease_seconds)
            if status in {"SENT", "UNKNOWN"} or (status == "SENDING" and not stale):
                conn.commit()
                return False, status.lower(), message_id
            # FAILED ou uma reserva abandonada antes da chamada HTTP pode tentar novamente.
            conn.execute(
                """UPDATE telegram_delivery_claims
                   SET ticket_id=?,payload_hash=?,status='SENDING',claimed_at=?,
                       sent_at=NULL,message_id=NULL,last_error=NULL
                   WHERE ticket_id=?""",
                (str(ticket_id), str(payload_hash), now, str(existing_ticket)),
            )
        else:
            conn.execute(
                """INSERT INTO telegram_delivery_claims
                   (ticket_id,payload_hash,status,claimed_at)
                   VALUES (?,?,'SENDING',?)""",
                (str(ticket_id), str(payload_hash), now),
            )
        conn.commit()
        return True, "claimed", None
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def mark_ticket_sent(db_path, ticket_id, message_id):
    conn = _connect(db_path)
    try:
        _ensure_schema(conn)
        conn.execute(
            """UPDATE telegram_delivery_claims
               SET status='SENT',sent_at=?,message_id=?,last_error=NULL
               WHERE ticket_id=?""",
            (int(time.time()), str(message_id or ""), str(ticket_id)),
        )
    finally:
        conn.close()


def mark_ticket_failed(db_path, ticket_id, error, ambiguous=False):
    conn = _connect(db_path)
    try:
        _ensure_schema(conn)
        conn.execute(
            """UPDATE telegram_delivery_claims
               SET status=?,last_error=? WHERE ticket_id=?""",
            ("UNKNOWN" if ambiguous else "FAILED", str(error)[:1000], str(ticket_id)),
        )
    finally:
        conn.close()

