"""Evening paper-trading run: catch up every missed trading day, report on Telegram.

Runs unattended. Rules:
- Paper trading starts only after /choose (store key "chosen_strategy").
- Each day goes through backtest.step_day (Ruling R3), so paper == backtest by construction.
- Bad data or a failed download -> alert, stop, don't advance "last_processed_day".
- All days are computed in memory first; only then is each day committed in its own
  store transaction, so an unexpected error leaves the stored state untouched.
- The kill switch is never cleared here (Ruling R11): only the owner's /resume does that.
"""
from __future__ import annotations

import copy
import traceback
from dataclasses import asdict, dataclass
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Callable

import pandas as pd

from tradebot import reports
from tradebot.backtest import DayResult, DayState, step_day
from tradebot.calendar import is_trading_day, previous_trading_day, trading_days_between
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
    result: DayResult
    portfolio: dict
    pending: list[dict]
    kill_switch: bool


def _paper_metrics(portfolio: Portfolio, history: dict[str, float], bars: Bars,
                   cfg: Config) -> Metrics:
    """Paper record vs buy-and-hold benchmark since paper start (Ruling R17).
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
    return compute_metrics(list(portfolio.trades), equity, bench["close"], cfg.capital, cfg,
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
    try:
        return _run(today, cfg, store, fetch, sender, now)
    except Exception:
        _log_exception(cfg, "daily run failed")
        try:
            store.log_event("daily_error", {"day": today.isoformat()})
        except Exception:
            pass
        sender(reports.alert(ERROR_TEXT), store)
        return []


def _run(today, cfg, store, fetch, sender, now) -> list[date]:
    name = store.get("chosen_strategy")
    if not name:
        if store.get("last_reminder") != today.isoformat():
            store.set("last_reminder", today.isoformat())
            sender(reports.alert("Paper trading has not started: no strategy chosen yet. "
                                 + reports.CHOOSE_LINE), store)
        return []

    last = store.get("last_processed_day")
    last = date.fromisoformat(last) if last else previous_trading_day(today, cfg)
    days = trading_days_between(last + timedelta(days=1), today, cfg)
    t = _ist(now)
    days = [d for d in days if d < t.date() or (d == t.date() and t.time() >= RUN_AFTER)]
    if not days:
        return []

    raw = store.get("portfolio")
    portfolio = Portfolio.from_dict(raw) if raw else Portfolio(cfg.capital)
    history: dict[str, float] = dict(store.get("equity_history") or {})
    paper_start = min([days[0]] + [date.fromisoformat(d) for d in history])
    # Held tickers too, in case one was removed from the universe while still held.
    tickers = list(dict.fromkeys(cfg.universe + [cfg.benchmark] + list(portfolio.positions)))
    start = min(days[0] - timedelta(days=WARMUP_DAYS), paper_start)
    try:
        bars = fetch(tickers, start, days[-1], cfg)
    except Exception:
        _log_exception(cfg, "market data download failed")
        sender(reports.alert("Could not download market data tonight. Nothing was traded; "
                             "I'll try again on the next run."), store)
        return []

    # Compute every day in memory first (pure), so an error here changes nothing stored.
    state = DayState(
        portfolio=portfolio,
        pending=[Signal(**s) for s in store.get("pending_signals") or []],
        paused=bool(store.get("paused", False)),
        kill_switch=bool(store.get("kill_switch", False)),
    )
    strategy = REGISTRY[name]()
    done: list[_Day] = []
    bad: tuple[date, list[str]] | None = None
    for d in days:
        problems = [f"{x}: no data" for x in tickers if x not in bars] + validate_day(bars, d)
        if problems:
            bad = (d, problems)
            break
        r = step_day(state, d, bars, strategy, cfg)
        done.append(_Day(r, state.portfolio.to_dict(), [asdict(s) for s in state.pending],
                         state.kill_switch))

    for day in done:
        r = day.result
        history[r.d.isoformat()] = r.equity
        with store.transaction():
            store.set("portfolio", day.portfolio)
            store.set("pending_signals", day.pending)
            store.set("kill_switch", day.kill_switch)
            store.set("equity_history", history)
            store.set("last_processed_day", r.d.isoformat())
            store.log_event("paper_day", {
                "day": r.d.isoformat(), "equity": r.equity, "buys": r.buys,
                "sells": [x.ticker for x in r.sells], "stops": [x.ticker for x in r.stop_exits],
                "skipped": [list(x) for x in r.skipped], "kill_tripped": r.kill_tripped_today})

    # Messages after the transactions (send writes the outbox itself).
    for day in done:
        if day.result.kill_tripped_today:
            sender(reports.alert(
                f"Safety switch tripped on {day.result.d.isoformat()}: the paper account fell "
                f"{cfg.kill_switch_dd_pct * 100:.0f}% or more below its highest value. "
                "New buys are paused; stop-losses and sells still run. "
                "Send /resume when you're ready."), store)
    if done:
        last_r = done[-1].result
        sender(reports.daily_summary(
            last_r.d, state.portfolio, last_r.equity, last_r.new_signals,
            last_r.sells + last_r.stop_exits, [x.result.d for x in done[:-1]], name,
            _below_benchmark(store, name), skipped=last_r.skipped), store)
        week_ends = [x.result.d for x in done if _last_day_of_week(x.result.d, cfg)]
        if week_ends and store.get("last_weekly_week") != _week_key(week_ends[-1]):
            store.set("last_weekly_week", _week_key(week_ends[-1]))
            metrics = _paper_metrics(state.portfolio, history, bars, cfg)
            sender(reports.weekly_report(metrics, name, _oos(store, name)), store)
    if bad is not None:
        sender(_problems_text(*bad), store)
    return [x.result.d for x in done]
