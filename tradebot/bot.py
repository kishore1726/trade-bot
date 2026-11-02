"""Telegram command bot: the owner controls the paper bot from their phone.

handle_command is pure (store in, text out). poll_once/run_bot do the Telegram I/O.
Replies go straight to sendMessage (not the outbox): they are interactive, and a failed
reply is just logged. Only the chat in TELEGRAM_CHAT_ID is obeyed.
"""
from __future__ import annotations

import os
import time
from datetime import date

import httpx

from tradebot.calendar import previous_trading_day
from tradebot.config import Config
from tradebot.metrics import Metrics
from tradebot.notify import load_env
from tradebot.portfolio import Portfolio
from tradebot.reports import money, backtest_comparison
from tradebot.store import Store

POLL_TIMEOUT = 30
ERROR_SLEEP = 10
NO_REPORT = "No report yet — the first weekly report arrives after Friday's run."
HELP = ("Commands:\n"
        "/status - what the bot holds and its state\n"
        "/report - latest report\n"
        "/choose <name> - pick the strategy to paper trade\n"
        "/pause - stop new buys (stops and sells still run)\n"
        "/resume - allow new buys again\n"
        "/help - this list")


def _portfolio(store: Store, cfg: Config) -> Portfolio:
    raw = store.get("portfolio")
    return Portfolio.from_dict(raw) if raw else Portfolio(cfg.capital)


def _status(cfg: Config, store: Store) -> str:
    p = _portfolio(store, cfg)
    hist = store.get("equity_history") or {}
    last_eq = hist[max(hist)] if hist else None
    lines = [f"Strategy: {store.get('chosen_strategy') or 'none yet'}",
             f"Cash: {money(p.cash)}",
             f"Last equity: {money(last_eq) if last_eq is not None else 'not known yet'}"]
    if p.positions:
        lines.append("Positions:")
        lines += [f"  {x.ticker}: {x.qty} @ {money(x.entry_price)}, stop {money(x.stop)}"
                  for x in p.positions.values()]
    else:
        lines.append("Positions: none")
    lines.append(f"Paused: {'yes' if store.get('paused') else 'no'}")
    lines.append(f"Kill switch: {'TRIPPED' if store.get('kill_switch') else 'off'}")
    lines.append(f"Last processed day: {store.get('last_processed_day') or 'none yet'}")
    return "\n".join(lines)


def _report(store: Store) -> str:
    weekly = store.get("last_weekly_report")
    if weekly:
        return weekly
    summ = store.get("backtest_summary")
    if summ and summ.get("results"):
        results = {n: {k: Metrics.from_dict(v) for k, v in r.items()}
                   for n, r in summ["results"].items()}
        ranking = [n for n in summ.get("ranking") or results if n in results]
        return backtest_comparison(results, ranking)
    return NO_REPORT


def _pause(store: Store) -> str:
    with store.transaction():
        store.set("paused", True)
        store.log_event("pause", {})
    return "Paused. Stop-losses and sells still run; only new buys are paused. Send /resume to undo."


def _resume(cfg: Config, store: Store) -> str:
    with store.transaction():  # read and write under one lock: the daily run is another process
        raw = store.get("portfolio")
        if not raw and not store.get("chosen_strategy"):
            return "Nothing to resume yet: no strategy has been chosen. Send /choose <name> first."
        p = Portfolio.from_dict(raw) if raw else Portfolio(cfg.capital)
        hist = store.get("equity_history") or {}
        p.peak_equity = (float(hist[max(hist)]) if hist else
                         p.cash + sum(x.qty * x.entry_price for x in p.positions.values()))
        store.set("paused", False)
        store.set("kill_switch", False)
        store.set("portfolio", p.to_dict())
        store.log_event("resume", {"peak_equity": p.peak_equity})
    return "Resumed. The safety switch and pause are cleared; new buys resume tomorrow."


def _choose(arg: str, cfg: Config, store: Store, today: date) -> str:
    with store.transaction():  # read and write under one lock: the daily run is another process
        summ = store.get("backtest_summary")
        names = list((summ or {}).get("results") or {})
        if not names:
            return "No backtest results yet. Run the backtest first, then send /choose <name>."
        if not arg:
            return f"Which one? Send /choose <name> with one of: {', '.join(names)}."
        if arg not in names:
            return f"I don't know \"{arg}\". Choose one of: {', '.join(names)}."
        current = store.get("chosen_strategy")
        if current == arg:
            return f"{arg} is already your chosen strategy. Nothing changed."
        raw = store.get("portfolio")
        p = Portfolio.from_dict(raw) if raw else Portfolio(cfg.capital)
        if p.positions:
            return (f"Can't switch to {arg} while positions are open (they belong to "
                    f"{current or 'the current strategy'}). Wait until they are all sold, then try again.")
        switched = current is not None
        if switched:  # Ruling R20: a different strategy starts a fresh paper record
            store.log_event("strategy_switch", {"from": current, "to": arg,
                                                "trades": p.to_dict()["trades"]})
            p.trades.clear()
            p.peak_equity = p.cash
            store.set("pending_signals", [])
            store.set("equity_history", {})
        store.set("portfolio", p.to_dict())
        store.set("chosen_strategy", arg)
        store.set("last_processed_day", previous_trading_day(today, cfg).isoformat())
        store.log_event("choose", {"strategy": arg})
    if switched:
        return (f"Switched from {current} to {arg}. A fresh paper record has started "
                "(the old trades are saved in the log). Paper trading starts with the next trading day.")
    return f"Chosen: {arg}. Paper trading starts with the next trading day."


def handle_command(text: str, chat_id: int, cfg: Config, store: Store,
                   today: date | None = None) -> str:
    parts = (text or "").strip().split()
    if not parts or not parts[0].startswith("/"):
        return HELP
    cmd, arg = parts[0].split("@")[0].lower(), " ".join(parts[1:])
    today = today or date.today()
    if cmd == "/status":
        return _status(cfg, store)
    if cmd == "/report":
        return _report(store)
    if cmd == "/pause":
        return _pause(store)
    if cmd == "/resume":
        return _resume(cfg, store)
    if cmd == "/choose":
        return _choose(arg, cfg, store, today)
    return HELP


def _creds() -> tuple[str, str]:
    token, chat = os.environ.get("TELEGRAM_TOKEN"), os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat:
        raise SystemExit("Telegram is not set up: put TELEGRAM_TOKEN and TELEGRAM_CHAT_ID in .env.")
    return token, chat


def _reply(http, token: str, chat: str, text: str, store: Store) -> None:
    try:
        r = http.post(f"https://api.telegram.org/bot{token}/sendMessage",
                      data={"chat_id": chat, "text": text}, timeout=10)
        if r.status_code != 200:
            store.log_event("reply_error", {"error": f"HTTP {r.status_code}"})
    except Exception as exc:
        store.log_event("reply_error", {"error": type(exc).__name__})


def poll_once(cfg: Config, store: Store, http) -> int:
    """One getUpdates round. Returns the number of messages handled (own chat only).
    Network errors in getUpdates propagate to the caller."""
    token, chat = _creds()
    offset = store.get("tg_offset", 0)
    resp = http.get(f"https://api.telegram.org/bot{token}/getUpdates",
                    params={"offset": offset, "timeout": POLL_TIMEOUT}, timeout=POLL_TIMEOUT + 10)
    resp.raise_for_status()
    handled = 0
    for u in resp.json().get("result", []):
        store.set("tg_offset", u["update_id"] + 1)
        msg = u.get("message") or {}
        chat_id = (msg.get("chat") or {}).get("id")
        if chat_id is None or not msg.get("text"):
            continue
        if str(chat_id) != chat:
            store.log_event("ignored_chat", {"chat_id": chat_id})
            continue
        try:
            reply = handle_command(msg["text"], chat_id, cfg, store)
        except Exception as exc:
            store.log_event("bot_error", {"error": type(exc).__name__})
            reply = "Something went wrong with that command. Nothing was changed."
        _reply(http, token, chat, reply, store)
        handled += 1
    return handled


def run_bot(cfg: Config, store: Store, http=None, sleep=time.sleep) -> None:
    load_env()
    token, _ = _creds()
    client = http or httpx.Client()
    while True:
        try:
            poll_once(cfg, store, client)
        except Exception as exc:
            msg = str(exc).replace(token, "***")[:200]
            try:
                store.log_event("bot_error", {"error": f"{type(exc).__name__}: {msg}"})
            except Exception:
                pass
            sleep(ERROR_SLEEP)
