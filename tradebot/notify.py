"""Telegram delivery through the SQLite outbox. send() never raises."""
from __future__ import annotations

import os
from pathlib import Path

import httpx

from tradebot.store import Store

MAX_LEN = 4000
TIMEOUT = 10.0


def load_env(path: Path = Path(".env")) -> None:
    """Read KEY=VALUE lines into os.environ; existing variables win."""
    try:
        lines = Path(path).read_text().splitlines()
    except OSError:
        return
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if key and key not in os.environ:
            os.environ[key] = value


def _chunks(text: str) -> list[str]:
    parts: list[str] = []
    while len(text) > MAX_LEN:
        cut = text.rfind("\n", 0, MAX_LEN)
        if cut <= 0:
            cut = MAX_LEN
        parts.append(text[:cut])
        text = text[cut:].lstrip("\n")
    parts.append(text)
    return parts


def send(text: str, store: Store, http: httpx.Client | None = None) -> bool:
    """Queue `text`, then deliver the whole backlog in order. True only if all sent."""
    for part in _chunks(text):
        store.outbox_add(part)
    token = os.environ.get("TELEGRAM_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        store.log_event("telegram_not_configured", {})
        return False
    client = http or httpx.Client(timeout=TIMEOUT)
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    try:
        for msg_id, msg in store.outbox_pending():
            resp = client.post(url, data={"chat_id": chat_id, "text": msg}, timeout=TIMEOUT)
            if resp.status_code != 200:
                store.log_event("telegram_error", {"error": f"HTTP {resp.status_code}: {resp.text[:200]}"})
                return False
            store.outbox_mark_sent(msg_id)
    except Exception as exc:  # network trouble must never crash the bot
        # Never log str(exc) verbatim for httpx errors with URLs: the URL holds the token.
        store.log_event("telegram_error", {"error": f"{type(exc).__name__}: {str(exc).replace(token, '***')[:200]}"})
        return False
    finally:
        if http is None:
            client.close()
    return True
