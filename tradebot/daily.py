"""Evening paper-trading run: catch up every missed trading day, report on Telegram.

Runs unattended. Rules:
- Paper trading starts only after /choose (store key "chosen_strategy").
- Each day goes through backtest.step_day (Ruling R3), so paper == backtest by construction.
- Bad data or a failed download -> alert, stop, don't advance "last_processed_day".
  Exceptions: an unlisted market holiday is skipped (Ruling R22); a missing/stale ticker that is
  neither held nor the benchmark is left out for the day (R26); today's missing bars before
  20:00 IST just mean Yahoo is late (M8).
- Stored state is read after the download; a day's save is discarded if an owner command
  bumped "state_rev" meanwhile (R24). Only one run at a time (lock file, M7).
- All days are computed in memory first; only then is each day committed in its own
  store transaction, so an unexpected error while computing leaves the stored state untouched.
  An error after days were saved (later commit, reports) says which days were processed and
  returns them.
- The kill switch is never cleared here (Ruling R11): only the owner's /resume does that.
"""
from __future__ import annotations

import copy
import fcntl
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
from tradebot.costs import trade_cost
from tradebot.data import (BAD, MISSING, Bars, day_problems, fetch_bars, forget_adjustments,
                           pending_adjustments)
from tradebot.metrics import Metrics, compute_metrics
from tradebot.notify import send
from tradebot.portfolio import Portfolio
from tradebot.risk import size_position
from tradebot.store import IST, Store
from tradebot.strategies import REGISTRY, Signal

# Today's bar is used only after this IST time (fetch_bars itself refuses bars before 16:00).
RUN_AFTER = time(16, 15)
# M8: until this IST time a missing bar for today is taken as "Yahoo hasn't published it yet":
# nothing is traded for today and the hourly run retries. After it, Ruling R26 applies.
LATE_PUBLISH_UNTIL = time(20, 0)
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
    left_out: dict[str, date | None] | None = None  # Ruling R26: ticker -> last bar before


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
    lock = _lock(cfg)
    if lock is False:
        return []  # M7: another run-daily is still busy; it does the work
    try:
        return _run_daily(today, cfg, store, fetch, sender, now)
    finally:
        if lock:
            lock.close()


def _lock(cfg: Config):
    """An exclusive lock file next to the store: the open file, False if another run holds it,
    or None if locking itself failed (then the run goes ahead unlocked)."""
    try:
        cfg.state_path.parent.mkdir(parents=True, exist_ok=True)
        f = open(cfg.state_path.parent / "daily.lock", "w")
    except OSError:
        _log_exception(cfg, "could not open the run lock")
        return None
    try:
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        f.close()
        return False
    except OSError:
        _log_exception(cfg, "could not take the run lock")
        f.close()
        return None
    return f


def _run_daily(today, cfg, store, fetch, sender, now) -> list[date]:
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


def _apply_adjustments(store: Store, cfg: Config) -> list[str]:
    """Ruling R23: rescale held positions (and pending stops) to Yahoo's re-adjusted prices.

    The ratios come from the data ledger (data.ADJ_LEDGER), which fetch_bars writes before it
    rebuilds a cache, whoever downloaded (this run, an earlier failed run, `tradebot backtest`).
    Each ledger id is applied at most once: applied ids are saved in store key
    "applied_adjustments" in the same transaction, then removed from the ledger. Saved in its own
    transaction straight after the download, not with a day's save, so a night that processes no
    day still applies them. Returns plain-English notes for the summary."""
    ledger = pending_adjustments(cfg)
    if not ledger:
        return []
    notes: list[str] = []
    with store.transaction():
        done = set(store.get("applied_adjustments") or [])
        todo = [e for e in ledger if e["id"] not in done]
        raw = store.get("portfolio")
        p = Portfolio.from_dict(raw) if raw else None
        pending = store.get("pending_signals") or []
        for e in todo:
            t, r = e["ticker"], float(e["ratio"])
            for sig in pending:
                if sig.get("ticker") == t and sig.get("stop") is not None:
                    sig["stop"] *= r
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
        if p is not None and notes:
            store.set("portfolio", p.to_dict())
        store.set("pending_signals", pending)
        # Only ids still in the ledger can come back, so older ones need not be kept.
        store.set("applied_adjustments", sorted(e["id"] for e in ledger))
    forget_adjustments(cfg, [e["id"] for e in ledger])  # a crash here is harmless: ids are saved
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
    """Ruling R22: no ticker has a bar on d, but the benchmark has a bar on a later day.
    A stale bar (M9: Yahoo repeating the previous day) does not count as a bar here."""
    ts = pd.Timestamp(d)
    present = {x: bars[x] for x in tickers if x in bars and ts in bars[x].index}
    probs = day_problems(present, d)
    if any(probs.get(x, ("", ""))[0] != MISSING for x in present):
        return False
    bench = bars.get(cfg.benchmark)
    return bench is not None and bool((bench.index > ts).any())


def _left_out_text(left_out: dict[str, date | None]) -> str:
    lines = []
    for x, since in left_out.items():
        when = f"since {since.isoformat()}" if since else "at all"
        lines.append(f"{x} has had no data {when}. If it was renamed or left the index, remove "
                     "it from config.yaml under universe.")
    lines.append("I left it out for those days and traded the other stocks as normal."
                 if len(left_out) == 1 else
                 "I left them out for those days and traded the other stocks as normal.")
    return reports.alert("\n".join(lines))


def _critical_text(d: date, tickers: list[str], cfg: Config) -> str:
    """Ruling R26: a missing benchmark or held ticker still halts, with specific advice."""
    lines = []
    for x in tickers:
        if x == cfg.benchmark:
            lines.append(f"The benchmark {x} has no price for {d.isoformat()}. If this lasts more "
                         "than a day, check that \"benchmark\" in config.yaml is right.")
        else:
            lines.append(f"{x} has no price for {d.isoformat()} and the paper portfolio holds it. "
                         "If this lasts several days the stock may be suspended or renamed: check "
                         "it on the NSE website.")
    lines.append(f"Nothing was traded for {d.isoformat()} (or any later day). I'll try again on "
                 "the next run.")
    return reports.alert("\n".join(lines))


def _estimates(signals: list[Signal], portfolio: Portfolio, equity: float, bars: Bars, d: date,
               paused: bool, kill_switch: bool, cfg: Config) -> dict[str, tuple]:
    """I7: what each BUY for tomorrow would roughly be, using step_day's rules at today's close
    (SELLs first, then BUYs in order with size_position). Real fills use tomorrow's open."""
    ts = pd.Timestamp(d)

    def close(t: str) -> float | None:
        df = bars.get(t)
        c = df["close"][df.index <= ts] if df is not None else None
        return float(c.iloc[-1]) if c is not None and len(c) else None

    p = copy.deepcopy(portfolio)
    for s in signals:
        if s.action == "SELL" and s.ticker in p.positions and close(s.ticker) is not None:
            p.sell(s.ticker, close(s.ticker), d, "estimate", cfg)
    out: dict[str, tuple] = {}
    for s in (s for s in signals if s.action == "BUY"):
        price = close(s.ticker)
        if paused:
            out[s.ticker] = ("will", "paused - send /resume to allow buys")
        elif kill_switch:
            out[s.ticker] = ("will", "safety switch is on - send /resume to allow buys")
        elif s.ticker in p.positions:
            out[s.ticker] = ("will", "already held")
        elif len(p.positions) >= cfg.max_positions:
            out[s.ticker] = ("will", f"no free slot (already holding {cfg.max_positions} positions)")
        elif price is None or s.stop is None:
            out[s.ticker] = ("may", "no price or stop yet")
        else:
            entry = price * (1 + cfg.slippage_pct)
            qty, reason = size_position(equity, p.cash, entry, s.stop, len(p.positions), cfg)
            if qty == 0:
                out[s.ticker] = ("may", reason or "position size is zero")
                continue
            out[s.ticker] = (qty, qty * entry)
            p.cash -= qty * entry + trade_cost("BUY", qty * entry, cfg.costs)
            p.positions[s.ticker] = None  # takes a slot
    return out


def _late_text(d: date, tickers: list[str]) -> str:
    shown = ", ".join(tickers[:MAX_PROBLEM_LINES]) + (" and others" if len(tickers) > MAX_PROBLEM_LINES else "")
    return reports.alert(f"Today's prices ({d.isoformat()}) are not complete yet: nothing for {shown}. "
                         "Yahoo may not have published them yet. Nothing has been traded for today; "
                         "I'll retry within the hour.")


def _check_day(d: date, tickers: list[str], bars: Bars, held: set[str], cfg: Config,
               late: bool = False) -> tuple[str | None, dict[str, date | None]]:
    """(halt alert text or None, tickers to leave out for d with their last bar date).
    `late`: d is today and Yahoo may still publish missing bars (M8)."""
    ts = pd.Timestamp(d)
    empty = pd.DataFrame(columns=["open", "high", "low", "close", "volume"],
                         index=pd.DatetimeIndex([]), dtype=float)
    probs = day_problems({x: bars.get(x, empty) for x in tickers}, d)
    lines = [f"{x}: {msg}" for x, (_, msg) in probs.items()]
    if any(kind == BAD for kind, _ in probs.values()):
        return _problems_text(d, lines), {}
    missing = list(probs)
    if not missing:
        return None, {}
    if late:
        return _late_text(d, missing), {}
    critical = [x for x in missing if x in held or x == cfg.benchmark]
    if critical:
        return _critical_text(d, critical, cfg), {}
    if len(tickers) - len(missing) <= len(tickers) / 2:
        return _problems_text(d, lines), {}
    left_out = {}
    for x in missing:
        idx = bars[x].index[bars[x].index < ts] if x in bars else pd.DatetimeIndex([])
        left_out[x] = idx[-1].date() if len(idx) else None
    return None, left_out


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
    try:
        bars = fetch(tickers, start, days[-1], cfg, now=lambda: t)  # same clock as the cutoff
    except Exception:
        _log_exception(cfg, "market data download failed")
        # Caches rebuilt before the failure must still rescale the paper positions now.
        for note in _apply_adjustments(store, cfg):
            sender(note, store)
        sender(reports.alert("Could not download market data tonight. Nothing was traded; "
                             "I'll try again on the next run."), store)
        return []
    notes = _apply_adjustments(store, cfg)

    with store.transaction():  # one consistent snapshot of everything the run uses
        rev = store.get("state_rev", 0)
        same = (store.get("chosen_strategy") == name
                and store.get("last_processed_day") == last_raw)
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
    bad: str | None = None  # alert text when a day cannot be processed
    for d in days:
        if _unlisted_holiday(d, tickers, bars, cfg):
            done.append(_Day(None, holiday=d))
            continue
        late = d == t.date() and t.time() < LATE_PUBLISH_UNTIL
        bad, left_out = _check_day(d, tickers, bars, set(state.portfolio.positions), cfg, late)
        if bad:
            break
        day_bars = {k: v for k, v in bars.items() if k not in left_out}
        r = step_day(state, d, day_bars, strategy, cfg)
        done.append(_Day(r, state.portfolio.to_dict(), [asdict(s) for s in state.pending],
                         state.kill_switch, left_out=left_out))

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
        final = Portfolio.from_dict(traded[-1].portfolio)
        prev = [k for k in history if k < last_r.d.isoformat()]
        sender(reports.daily_summary(
            last_r.d, final, last_r.equity, last_r.new_signals,
            exits, [x.result.d for x in traded[:-1]], name,
            _below_benchmark(store, name), skipped=skipped, notes=notes,
            estimates=_estimates(last_r.new_signals, final, last_r.equity, bars, last_r.d,
                                 state.paused, traded[-1].kill_switch, cfg),
            previous_equity=history[max(prev)] if prev else None), store)

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
    left_out: dict[str, date | None] = {}
    for x in traded:
        for k, v in (x.left_out or {}).items():
            left_out.setdefault(k, v)
    if left_out:
        phase("left-out alert", lambda: sender(_left_out_text(left_out), store))
    if bad is not None and not stopped:
        phase("bad-data alert", lambda: sender(bad, store))
    if failed:
        if committed:
            sender(reports.alert(_partial_text(committed)), store)
        else:
            sender(reports.alert(ERROR_TEXT), store)
    return list(committed)
