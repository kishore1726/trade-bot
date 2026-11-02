"""Trust metrics for a backtest or paper run, with estimated Indian tax and a benchmark."""
from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, fields

import pandas as pd

from tradebot.config import Config
from tradebot.costs import estimate_tax_ltcg, estimate_tax_stcg, trade_cost
from tradebot.portfolio import Trade


@dataclass
class Metrics:
    trades: int
    win_rate: float
    avg_win: float
    avg_loss: float  # negative (or 0 when there are no losing trades)
    expectancy: float  # mean net P&L per trade, rupees
    max_drawdown: float  # positive fraction, e.g. 0.12 = 12%
    total_return: float
    cagr: float
    benchmark_return: float
    benchmark_cagr: float
    total_costs: float  # charges on closed trades
    post_tax_return: float
    benchmark_post_tax_return: float
    below_benchmark: bool

    @classmethod
    def from_dict(cls, d: dict) -> "Metrics":
        return cls(**{f.name: d[f.name] for f in fields(cls)})


def _years(index: pd.Index) -> float:
    return (index[-1] - index[0]).days / 365.25 if len(index) > 1 else 0.0


def _cagr(final: float, start: float, years: float) -> float:
    if years <= 0:
        return 0.0
    if final <= 0:
        return -1.0
    return (final / start) ** (1 / years) - 1


def _max_drawdown(equity: pd.Series) -> float:
    if equity.empty:
        return 0.0
    return float(max((1 - equity / equity.cummax()).max(), 0.0))


def _benchmark(close: pd.Series, open_price: float | None, start: float, cfg: Config):
    """Buy-and-hold: whole shares at the first open (buy costs), valued at closes.

    Returns (return, cagr, post-tax return). Post-tax includes sale costs and tax
    (LTCG if held more than 365 days, else STCG).
    """
    if close.empty:
        raise ValueError("no benchmark prices for this period")
    price = float(open_price if open_price is not None else close.iloc[0])
    qty = math.floor(start / price)
    while qty > 0 and qty * price + trade_cost("BUY", qty * price, cfg.costs) > start:
        qty -= 1
    if qty == 0:
        return 0.0, 0.0, 0.0
    paid = qty * price + trade_cost("BUY", qty * price, cfg.costs)
    left = start - paid
    last = float(close.iloc[-1])
    final = left + qty * last
    ret = final / start - 1
    cagr = _cagr(final, start, _years(close.index))
    proceeds = qty * last - trade_cost("SELL", qty * last, cfg.costs)
    held_days = (close.index[-1] - close.index[0]).days
    tax_fn = estimate_tax_ltcg if held_days > 365 else estimate_tax_stcg
    tax = tax_fn(proceeds - paid, cfg.tax)
    post = (left + proceeds - tax) / start - 1
    return ret, cagr, post


def compute_metrics(trades: list[Trade], equity: pd.Series, benchmark_close: pd.Series,
                    start_capital: float, cfg: Config,
                    benchmark_open: float | None = None) -> Metrics:
    """`benchmark_open` is the benchmark's open on the first day (the buy price);
    when omitted the first benchmark close is used."""
    pnls = [t.pnl for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    n = len(pnls)

    final = float(equity.iloc[-1]) if not equity.empty else float(start_capital)
    total_return = final / start_capital - 1
    cagr = _cagr(final, start_capital, _years(equity.index))

    # Strategy trades are short-term: STCG on each calendar year's net realised gain.
    by_year: dict[int, float] = defaultdict(float)
    for t in trades:
        by_year[t.exit_date.year] += t.pnl
    tax = sum(estimate_tax_stcg(g, cfg.tax) for g in by_year.values())
    post_tax_return = (final - tax - start_capital) / start_capital

    b_ret, b_cagr, b_post = _benchmark(benchmark_close, benchmark_open, start_capital, cfg)

    return Metrics(
        trades=n,
        win_rate=len(wins) / n if n else 0.0,
        avg_win=float(sum(wins) / len(wins)) if wins else 0.0,
        avg_loss=float(sum(losses) / len(losses)) if losses else 0.0,
        expectancy=float(sum(pnls) / n) if n else 0.0,
        max_drawdown=_max_drawdown(equity),
        total_return=float(total_return),
        cagr=float(cagr),
        benchmark_return=float(b_ret),
        benchmark_cagr=float(b_cagr),
        total_costs=float(sum(t.costs for t in trades)),
        post_tax_return=float(post_tax_return),
        benchmark_post_tax_return=float(b_post),
        # Conservative: below if it trails the benchmark before OR after estimated tax.
        below_benchmark=bool(total_return < b_ret or post_tax_return < b_post),
    )
