"""Evening paper-trading run: catch up every missed trading day, report on Telegram.

Runs unattended. Rules:
- Paper trading starts only after /choose (store key "chosen_strategy").
- Each day goes through backtest.step_day (Ruling R3), so paper == backtest by construction.
- Bad data or a failed download -> alert, stop, don't advance "last_processed_day".
- All days are computed in memory first; only then is each day committed in its own
  store transaction, so an unexpected error while computing leaves the stored state untouched.
  An error after days were saved (later commit, reports) says which days were processed and
  returns them.
- The kill switch is never cleared here (Ruling R11): only the owner's /resume does that.
"""
from __future__ import annotations

import copy
import hashlib
import traceback
from dataclasses import asdict, dataclass
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Callable

import pandas as pd

from tradebot import reports
from tradebot.backtest import DayResult, DayState, step_day
from tradebot.calendar import (HolidayFileError, is_trading_day, previous_trading_day,
                               trading_days_between)
from tradebot.config import Config
from tradebot.data import Bars, fetch_bars, validate_day
from tradebot.metrics import Metrics, compute_metrics
from tradebot.notify import send
from tradebot.portfolio import Portfolio
from tradebot.store import IST, Store
from tradebot.strategies import REGISTRY, Signal

# Today's bar is used only after this IST time (fetch_bars itself refuses bars before 16:00).
RUN_AFTER = time(16, 15)
WARMUP_DAYS = 450  # calendar days of history before the first processed day (SMA200 etc.)
MAX_PROBLEM_LINES = 10
# Ruling R23: a price-adjustment ratio this far from 1 is a split/bonus (shares change);
# anything closer is dividend-like (only prices change).
SPLIT_LOW, SPLIT_HIGH = 0.83, 1.2

ERROR_TEXT = ("Something went wrong in tonight's run. Nothing was traded. "
              "Details saved in the log file.")


def _log_path(cfg: Config) -> Path:
    return cfg.state_path.parent.parent / "logs" / "tradebot.log"


def _log_exception(cfg: Config, what: str) -> None:
    try:
        path = _log_path(cfg)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as f:
            f.write(f"--- {datetime.now(IST).isoformat(timespec='seconds')} {what}\n")
            f.write(traceback.format_exc())
    except Exception:
        pass  # logging must never turn into a second failure


def _ist(now: Callable[[], datetime] | None) -> datetime:
    t = (now or (lambda: datetime.now(IST)))()
    return t.astimezone(IST) if t.tzinfo else t.replace(tzinfo=IST)


def _dedup(sender, store: Store, now: Callable[[], datetime] | None):
    """Ruling R21: the same alert text goes out at most once per IST calendar day, so the hourly
    catch-up run cannot spam the owner. Only alerts (starting with the warning sign) are checked."""
    def send_once(text, st):
        if text.startswith(reports.alert("")):
            day = _ist(now).date().isoformat()
            digest = hashlib.sha1(text.encode()).hexdigest()
            seen = (store.get("last_alerts") or {}).get(day, [])
            if digest in seen:
                return False
            store.set("last_alerts", {day: seen + [digest]})
        return sender(text, st)
    return send_once


def _week_key(d: date) -> str:
    y, w, _ = d.isocalendar()
    return f"{y}-W{w:02d}"


def _last_day_of_week(d: date, cfg: Config) -> bool:
    """True if the next trading day falls in a different ISO week (Ruling R16)."""
    nxt = d + timedelta(days=1)
    while not is_trading_day(nxt, cfg):
        nxt += timedelta(days=1)
    return _week_key(nxt) != _week_key(d)


def _oos(store: Store, name: str) -> Metrics | None:
    summary = store.get("backtest_summary") or {}
    m = (summary.get("results") or {}).get(name, {}).get("out_of_sample")
    return Metrics.from_dict(m) if isinstance(m, dict) and m else None


def _below_benchmark(store: Store, name: str) -> bool:
    summary = store.get("backtest_summary") or {}
    m = (summary.get("results") or {}).get(name, {}).get("out_of_sample") or {}
    return bool(m.get("below_benchmark", False))


@dataclass
class _Day:
    result: DayResult | None  # None for an unlisted market holiday (Ruling R22)
    portfolio: dict | None = None
    pending: list[dict] | None = None
    kill_switch: bool = False
    holiday: date | None = None


def _paper_metrics(portfolio: Portfolio, history: dict[str, float], bars: Bars,
                   cfg: Config, start_capital: float | None = None) -> Metrics:
    """Paper record vs buy-and-hold benchmark since paper start (Ruling R17), measured from the
    cash the strategy started with (Ruling R25; config capital if unknown).
    Open positions are treated as sold at the last close for tax/costs, like backtests (R15)."""
    days = sorted(history)
    idx = pd.DatetimeIndex([pd.Timestamp(d) for d in days], name="date")
    equity = pd.Series([history[d] for d in days], index=idx, dtype=float)
    bench = bars[cfg.benchmark]
    bench = bench[(bench.index >= idx[0]) & (bench.index <= idx[-1])]
    last = idx[-1]
    liquid = copy.deepcopy(portfolio)
    open_at_end = []
    for t in list(liquid.positions):
        closes = bars[t]["close"]
        open_at_end.append(liquid.sell(t, float(closes[closes.index <= last].iloc[-1]),
                                       last.date(), "end of period", cfg))
    capital = float(start_capital) if start_capital else cfg.capital
    return compute_metrics(list(portfolio.trades), equity, bench["close"], capital, cfg,
                           benchmark_open=float(bench["open"].iloc[0]) if not bench.empty else None,
                           open_at_end=open_at_end, liquidated_equity=liquid.cash)


def _problems_text(d: date, problems: list[str]) -> str:
    shown = problems[:MAX_PROBLEM_LINES]
    more = len(problems) - len(shown)
    lines = [f"Market data for {d.isoformat()} looks wrong, so nothing was traded for that day "
             "(or any later day). I'll try again on the next run.", "Problems found:"]
    lines += [f"  {p}" for p in shown]
    if more:
        lines.append(f"  ...and {more} more")
    return reports.alert("\n".join(lines))


def run_daily(today: date, cfg: Config, store: Store, fetch=fetch_bars, sender=send,
              now: Callable[[], datetime] | None = None) -> list[date]:
    """Process every trading day after "last_processed_day" up to `today` (today only after
    16:15 IST). Returns the days processed. Never raises."""
    committed: list[date] = []  # days already saved, even if a later step fails
    sender = _dedup(sender, store, now)
    try:
        return _run(today, cfg, store, fetch, sender, now, committed)
    except HolidayFileError as exc:  # the owner can fix this: say exactly where
        _log_exception(cfg, "holiday file problem")
        try:
            sender(reports.alert(f"{exc} Nothing was traded until it is fixed."), store)
        except Exception:
            _log_exception(cfg, "sending the holiday-file alert failed")
        return list(committed)
    except Exception:
        _log_exception(cfg, "daily run failed")
        try:
            store.log_event("daily_error", {"day": today.isoformat()})
        except Exception:
            pass
        try:
            sender(reports.alert(_partial_text(committed) if committed else ERROR_TEXT), store)
        except Exception:
            _log_exception(cfg, "sending the error alert failed")
        return list(committed)


def _partial_text(days: list[date]) -> str:
    return (f"Tonight's run processed {', '.join(d.isoformat() for d in days)} but failed while "
            "preparing the report. Details saved in the log file.")


def _remind_once(store: Store, today: date, sender, text: str) -> None:
    if store.get("last_reminder") != today.isoformat():
        store.set("last_reminder", today.isoformat())
        sender(reports.alert(text), store)


def _holiday_file_check(today: date, cfg: Config, store: Store, sender) -> None:
    """Ruling R22: from 1 December, ask for next year's holiday list (once a day, R21)."""
    if today.month != 12:
        return
    nxt = today.year + 1
    if not (cfg.holidays_dir / f"nse_{nxt}.txt").exists():
        sender(reports.alert(
            f"The NSE holiday list for {nxt} is missing. Please add the file "
            f"holidays/nse_{nxt}.txt (one date per line, written as YYYY-MM-DD) copied from NSE's "
            "official holiday list. Without it I may wait or skip on market holidays."), store)


def _holiday_note(d: date) -> str:
    return f"Market was closed on {d.isoformat()} (holiday not in my list)."


def _apply_adjustments(store: Store, adj: dict[str, float]) -> list[str]:
    """Ruling R23: rescale held positions (and pending stops) to Yahoo's re-adjusted prices.

    Saved in its own transaction straight after the download, not with a day's save: the cache
    is already rebuilt, so if no day can be processed tonight the ratio would be lost for ever.
    Returns plain-English notes for the summary."""
    if not adj:
        return []
    notes: list[str] = []
    with store.transaction():
        raw = store.get("portfolio")
        p = Portfolio.from_dict(raw) if raw else None
        for t, r in sorted(adj.items()):
            pos = p.positions.get(t) if p else None
            if pos is None:
                continue
            split = r <= SPLIT_LOW or r >= SPLIT_HIGH
            q0 = pos.qty
            if split:
                pos.qty = max(1, round(q0 / r))
            pos.entry_price *= r
            pos.stop *= r
            store.log_event("price_adjustment", {"ticker": t, "ratio": r, "qty_before": q0,
                                                 "qty_after": pos.qty, "kind": "split" if split else "dividend"})
            head = f"{t}: Yahoo adjusted its past prices by a factor of {r:.4g}"
            if split:
                notes.append(f"{head} (looks like a share split or bonus issue). Your paper position "
                             f"was changed from {q0} to {pos.qty} shares, bought at "
                             f"{reports.money(pos.entry_price)}, stop {reports.money(pos.stop)}.")
            else:
                notes.append(f"{head} (looks like a dividend or a similar small change). The "
                             f"bought-at price and stop were scaled to match "
                             f"({reports.money(pos.entry_price)}, stop {reports.money(pos.stop)}); "
                             "the number of shares is unchanged.")
        pending = store.get("pending_signals") or []
        for sig in pending:
            if sig.get("ticker") in adj and sig.get("stop") is not None:
                sig["stop"] *= adj[sig["ticker"]]
        if p is not None and notes:
            store.set("portfolio", p.to_dict())
        store.set("pending_signals", pending)
    return notes


class _StateChanged(Exception):
    """An owner command changed the stored state while this run was computing (Ruling R24)."""


def _save_day(store: Store, day: _Day, history: dict[str, float]) -> None:
    """Write one processed day (caller holds the transaction)."""
    r = day.result
    history[r.d.isoformat()] = r.equity
    store.set("portfolio", day.portfolio)
    store.set("pending_signals", day.pending)
    store.set("kill_switch", day.kill_switch)
    store.set("equity_history", history)
    store.set("last_processed_day", r.d.isoformat())
    store.log_event("paper_day", {
        "day": r.d.isoformat(), "equity": r.equity, "buys": r.buys,
        "sells": [x.ticker for x in r.sells], "stops": [x.ticker for x in r.stop_exits],
        "skipped": [list(x) for x in r.skipped], "kill_tripped": r.kill_tripped_today})


def _unlisted_holiday(d: date, tickers: list[str], bars: Bars, cfg: Config) -> bool:
    """Ruling R22: no ticker has a bar on d, but the benchmark has a bar on a later day."""
    ts = pd.Timestamp(d)
    if any(x in bars and ts in bars[x].index for x in tickers):
        return False
    bench = bars.get(cfg.benchmark)
    return bench is not None and bool((bench.index > ts).any())


def _run(today, cfg, store, fetch, sender, now, committed: list[date]) -> list[date]:
    _holiday_file_check(today, cfg, store, sender)
    name = store.get("chosen_strategy")
    if not name:
        _remind_once(store, today, sender, "Paper trading has not started: no strategy chosen yet. "
                     + reports.CHOOSE_LINE)
        return []
    if name not in REGISTRY:
        _remind_once(store, today, sender,
                     f"The chosen strategy \"{name}\" is not known to this bot, so nothing was "
                     f"traded. Send /choose <name> with one of: {', '.join(sorted(REGISTRY))}.")
        return []

    last_raw = store.get("last_processed_day")
    last = date.fromisoformat(last_raw) if last_raw else previous_trading_day(today, cfg)
    days = trading_days_between(last + timedelta(days=1), today, cfg)
    t = _ist(now)
    days = [d for d in days if d < t.date() or (d == t.date() and t.time() >= RUN_AFTER)]
    if not days:
        return []

    # This first read only decides what to download; everything used for trading is read again
    # after the download (Ruling R24), since the owner may send a command meanwhile.
    raw = store.get("portfolio")
    held = list(Portfolio.from_dict(raw).positions) if raw else []
    paper_start = min([days[0]] + [date.fromisoformat(d) for d in store.get("equity_history") or {}])
    # Held tickers too, in case one was removed from the universe while still held.
    tickers = list(dict.fromkeys(cfg.universe + [cfg.benchmark] + held))
    start = min(days[0] - timedelta(days=WARMUP_DAYS), paper_start)
    adj: dict[str, float] = {}
    try:
        bars = fetch(tickers, start, days[-1], cfg, now=lambda: t,  # same clock as the cutoff
                     adjustments=adj)
    except Exception:
        _log_exception(cfg, "market data download failed")
        sender(reports.alert("Could not download market data tonight. Nothing was traded; "
                             "I'll try again on the next run."), store)
        return []
    notes = _apply_adjustments(store, adj)

    with store.transaction():  # one consistent snapshot of everything the run uses
        rev = store.get("state_rev", 0)
        same = (store.get("chosen_strategy") == name
                and store.get("last_processed_day") == (last_raw and last.isoformat()))
        raw = store.get("portfolio")
        portfolio = Portfolio.from_dict(raw) if raw else Portfolio(cfg.capital)
        history: dict[str, float] = dict(store.get("equity_history") or {})
        state = DayState(
            portfolio=portfolio,
            pending=[Signal(**s) for s in store.get("pending_signals") or []],
            paused=bool(store.get("paused", False)),
            kill_switch=bool(store.get("kill_switch", False)),
        )
    if not same or not set(portfolio.positions) <= set(tickers):
        # /choose changed the strategy or start day during the download: the next run redoes it.
        store.log_event("state_changed_during_run", {"day": days[0].isoformat()})
        return []

    # Compute every day in memory first (pure), so an error here changes nothing stored.
    strategy = REGISTRY[name]()
    done: list[_Day] = []
    bad: tuple[date, list[str]] | None = None
    for d in days:
        if _unlisted_holiday(d, tickers, bars, cfg):
            done.append(_Day(None, holiday=d))
            continue
        problems = [f"{x}: no data" for x in tickers if x not in bars] + validate_day(bars, d)
        if problems:
            bad = (d, problems)
            break
        r = step_day(state, d, bars, strategy, cfg)
        done.append(_Day(r, state.portfolio.to_dict(), [asdict(s) for s in state.pending],
                         state.kill_switch))

    saved: list[_Day] = []
    for day in done:
        try:
            with store.transaction():
                if store.get("state_rev", 0) != rev:
                    # Ruling R24: an owner command (/pause, /resume, /choose) landed after the
                    # state was read. Roll this day back and stop quietly; the next run redoes it.
                    raise _StateChanged
                if day.holiday is not None:  # nothing traded: only move past it
                    store.set("last_processed_day", day.holiday.isoformat())
                    store.log_event("unlisted_holiday", {"day": day.holiday.isoformat()})
                else:
                    _save_day(store, day, history)
        except _StateChanged:
            store.log_event("state_changed_during_run",
                            {"day": (day.holiday or day.result.d).isoformat()})
            break
        saved.append(day)
        if day.result is not None:
            committed.append(day.result.d)
    stopped = len(saved) < len(done)
    done = saved

    # Messages after the transactions (send writes the outbox itself). Each phase is guarded
    # separately so one failure doesn't suppress the others.
    failed = False

    def phase(what: str, fn: Callable[[], None]) -> None:
        nonlocal failed
        try:
            fn()
        except Exception:
            failed = True
            _log_exception(cfg, f"{what} failed")

    traded = [x for x in done if x.result is not None]
    notes += [_holiday_note(x.holiday) for x in done if x.holiday is not None]

    def kill_alerts() -> None:
        for day in traded:
            if day.result.kill_tripped_today:
                sender(reports.alert(
                    f"Safety switch tripped on {day.result.d.isoformat()}: the paper account fell "
                    f"{cfg.kill_switch_dd_pct * 100:.0f}% or more below its highest value. "
                    "New buys are paused; stop-losses and sells still run. "
                    "Send /resume when you're ready."), store)

    def summary() -> None:
        last_r = traded[-1].result
        # I3: exits and skipped orders from EVERY processed day, not just the last one.
        exits = [tr for x in traded for tr in x.result.sells + x.result.stop_exits]
        skipped = [(x.result.d, tk, why) for x in traded for tk, why in x.result.skipped]
        sender(reports.daily_summary(
            last_r.d, Portfolio.from_dict(traded[-1].portfolio), last_r.equity, last_r.new_signals,
            exits, [x.result.d for x in traded[:-1]], name,
            _below_benchmark(store, name), skipped=skipped, notes=notes), store)

    def weekly() -> None:
        week_ends = [x.result.d for x in traded if _last_day_of_week(x.result.d, cfg)]
        # A week whose last calendar trading day was an unlisted holiday ends on the trading
        # day before the first processed day of the next week.
        seen = sorted(date.fromisoformat(k) for k in history)
        for x in traded:
            prev = [h for h in seen if h < x.result.d]
            if prev and _week_key(prev[-1]) != _week_key(x.result.d):
                week_ends.append(prev[-1])
        week_ends.sort()
        if week_ends and store.get("last_weekly_week") != _week_key(week_ends[-1]):
            metrics = _paper_metrics(Portfolio.from_dict(traded[-1].portfolio), history, bars, cfg,
                                     store.get("paper_start_capital"))
            text = reports.weekly_report(metrics, name, _oos(store, name))
            sender(text, store)
            store.set("last_weekly_report", text)  # for the /report command (Ruling R19)
            # Only after send() was called (a False return is fine: it's queued in the outbox).
            store.set("last_weekly_week", _week_key(week_ends[-1]))

    if traded:
        phase("kill-switch alert", kill_alerts)
        phase("daily summary", summary)
        phase("weekly report", weekly)
    elif notes:
        phase("holiday note", lambda: sender("\n".join(notes), store))
    if bad is not None and not stopped:
        phase("bad-data alert", lambda: sender(_problems_text(*bad), store))
    if failed:
        if committed:
            sender(reports.alert(_partial_text(committed)), store)
        else:
            sender(reports.alert(ERROR_TEXT), store)
    return list(committed)
