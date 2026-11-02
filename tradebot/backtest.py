"""Backtest engine: one shared day step (also used by the daily paper run), walk-forward, ranking.

No look-ahead: on day d the step only uses d's open (fills), d's open/low (stops) and closes
up to d (equity); strategies see rows <= d and their signals fill at the NEXT day's open.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Literal

import pandas as pd

from tradebot.config import Config
from tradebot.data import Bars
from tradebot.metrics import Metrics, compute_metrics
from tradebot.portfolio import Portfolio, Trade
from tradebot.risk import kill_switch_tripped, size_position
from tradebot.strategies import REGISTRY, Signal, Strategy

Segment = Literal["in_sample", "out_of_sample"]


@dataclass
class DayState:
    portfolio: Portfolio
    pending: list[Signal]
    paused: bool = False
    kill_switch: bool = False


@dataclass
class DayResult:
    d: date
    sells: list[Trade]
    buys: list[str]
    skipped: list[tuple[str, str]]
    stop_exits: list[Trade]
    equity: float
    kill_tripped_today: bool
    new_signals: list[Signal]


def _bar(bars: Bars, ticker: str, ts: pd.Timestamp) -> pd.Series | None:
    df = bars.get(ticker)
    if df is None or ts not in df.index:
        return None
    return df.loc[ts]


def _price_at_open(bars: Bars, ticker: str, ts: pd.Timestamp) -> float:
    """d's open, or the last close before d if the ticker has no bar on d."""
    row = _bar(bars, ticker, ts)
    if row is not None:
        return float(row["open"])
    closes = bars[ticker]["close"]
    return float(closes[closes.index < ts].iloc[-1])


def step_day(state: DayState, d: date, bars: Bars, strategy: Strategy, cfg: Config) -> DayResult:
    """Process one trading day in place: fills at open, stops, equity/kill switch, new signals."""
    p = state.portfolio
    ts = pd.Timestamp(d)
    sells: list[Trade] = []
    buys: list[str] = []
    skipped: list[tuple[str, str]] = []

    # 1. Fill yesterday's signals at today's open: SELLs first, then BUYs in signal order.
    for s in (s for s in state.pending if s.action == "SELL"):
        if s.ticker not in p.positions:
            skipped.append((s.ticker, "not held"))
            continue
        row = _bar(bars, s.ticker, ts)
        if row is None:
            skipped.append((s.ticker, "no price today"))
            continue
        sells.append(p.sell(s.ticker, float(row["open"]), d, "signal", cfg))

    for s in (s for s in state.pending if s.action == "BUY"):
        if state.paused:
            skipped.append((s.ticker, "paused"))
            continue
        if state.kill_switch:
            skipped.append((s.ticker, "kill switch active"))
            continue
        if s.ticker in p.positions:
            skipped.append((s.ticker, "already held"))
            continue
        row = _bar(bars, s.ticker, ts)
        if row is None:
            skipped.append((s.ticker, "no price today"))
            continue
        if s.stop is None:
            skipped.append((s.ticker, "no stop given"))
            continue
        open_price = float(row["open"])
        entry = open_price * (1 + cfg.slippage_pct)
        if entry <= s.stop:
            skipped.append((s.ticker, "gapped below stop"))
            continue
        sizing_equity = p.cash + sum(pos.qty * _price_at_open(bars, t, ts)
                                     for t, pos in p.positions.items())
        qty, reason = size_position(sizing_equity, p.cash, entry, s.stop, len(p.positions), cfg)
        if qty == 0:
            skipped.append((s.ticker, reason or "position size is zero"))
            continue
        p.buy(s.ticker, int(qty), open_price, d, s.stop, strategy.name, cfg)
        buys.append(s.ticker)

    # 2. Stops from today's bar.
    stop_exits = p.apply_stops(d, bars, cfg)

    # 3. Equity at close, peak, kill switch (stays on until the owner resumes / run ends).
    equity = p.equity(d, bars)
    p.update_peak(equity)
    tripped = False
    if not state.kill_switch and kill_switch_tripped(equity, p.peak_equity, cfg):
        state.kill_switch = tripped = True

    # 4. Signals at today's close become tomorrow's pending orders.
    state.pending = strategy.signals(bars, d, p.positions)
    return DayResult(d, sells, buys, skipped, stop_exits, equity, tripped, list(state.pending))


def _trading_days(bars: Bars, start: date, end: date) -> list[date]:
    lo, hi = pd.Timestamp(start), pd.Timestamp(end)
    days: set[pd.Timestamp] = set()
    for df in bars.values():
        days.update(df.index[(df.index >= lo) & (df.index <= hi)])
    return [t.date() for t in sorted(days)]


def run_backtest(strategy: Strategy, bars: Bars, start: date, end: date,
                 cfg: Config) -> tuple[Metrics, pd.Series, list[Trade]]:
    days = _trading_days(bars, start, end)
    if not days:
        raise ValueError(f"no market data between {start} and {end}")
    if cfg.benchmark not in bars:
        raise ValueError(f"no data for benchmark {cfg.benchmark}")
    state = DayState(Portfolio(cfg.capital), [])
    values = [step_day(state, d, bars, strategy, cfg).equity for d in days]
    equity = pd.Series(values, index=pd.DatetimeIndex([pd.Timestamp(d) for d in days], name="date"),
                       dtype=float)

    bench = bars[cfg.benchmark]
    bench = bench[(bench.index >= pd.Timestamp(days[0])) & (bench.index <= pd.Timestamp(days[-1]))]
    trades = list(state.portfolio.trades)
    bench_open = float(bench["open"].iloc[0]) if not bench.empty else None
    metrics = compute_metrics(trades, equity, bench["close"], cfg.capital, cfg,
                              benchmark_open=bench_open)
    return metrics, equity, trades


def _years_before(d: date, years: int) -> date:
    try:
        return d.replace(year=d.year - years)
    except ValueError:  # 29 Feb
        return d.replace(year=d.year - years, day=28)


def walk_forward(strategy_names: list[str], bars: Bars, cfg: Config,
                 holdout_years: int = 2) -> dict[str, dict[Segment, Metrics]]:
    """Out-of-sample = the last `holdout_years` years of data; in-sample = everything before.

    Each segment starts with fresh capital; strategies still see all earlier bars (warm-up).
    """
    all_days = _trading_days(bars, date.min, date.max)
    if not all_days:
        raise ValueError("no market data")
    first, last = all_days[0], all_days[-1]
    split = _years_before(last, holdout_years)  # in-sample ends on/before this date
    in_days = [d for d in all_days if d <= split]
    oos_days = [d for d in all_days if d > split]
    if not in_days or not oos_days:
        raise ValueError(f"need more than {holdout_years} years of data for walk-forward")
    out: dict[str, dict[Segment, Metrics]] = {}
    for name in strategy_names:
        out[name] = {
            "in_sample": run_backtest(REGISTRY[name](), bars, first, in_days[-1], cfg)[0],
            "out_of_sample": run_backtest(REGISTRY[name](), bars, oos_days[0], last, cfg)[0],
        }
    return out


def rank(results: dict[str, dict[Segment, Metrics]], cfg: Config) -> list[str]:
    """Best first, on out-of-sample metrics: drawdown within the kill-switch limit, then
    higher expectancy, then higher return over the benchmark (name breaks exact ties)."""
    def key(name: str):
        m = results[name]["out_of_sample"]
        return (m.max_drawdown > cfg.kill_switch_dd_pct, -m.expectancy,
                -(m.total_return - m.benchmark_return), name)
    return sorted(results, key=key)
