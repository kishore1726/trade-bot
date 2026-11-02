"""Paper portfolio: positions, fills with slippage and costs, stop handling."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date

import pandas as pd

from tradebot.config import Config
from tradebot.costs import trade_cost
from tradebot.data import Bars


@dataclass
class Position:
    ticker: str
    qty: int
    entry_price: float  # slipped fill price
    entry_date: date
    stop: float
    strategy: str
    entry_cost: float = 0.0  # buy-side charges


@dataclass
class Trade:
    ticker: str
    qty: int
    entry_date: date
    entry_price: float
    exit_date: date
    exit_price: float
    costs: float  # entry + exit charges
    pnl: float  # fully net of costs
    reason: str  # "stop" | "signal"


class Portfolio:
    def __init__(self, cash: float):
        self.cash = float(cash)
        self.positions: dict[str, Position] = {}
        self.trades: list[Trade] = []
        self.peak_equity = float(cash)

    def buy(self, ticker: str, qty: int, open_price: float, d: date, stop: float,
            strategy: str, cfg: Config) -> None:
        if isinstance(qty, bool) or not isinstance(qty, int) or qty < 1:
            raise ValueError(f"quantity must be a whole number >= 1, got {qty!r}")
        if ticker in self.positions:
            raise ValueError(f"{ticker} is already held")
        fill = open_price * (1 + cfg.slippage_pct)
        value = qty * fill
        cost = trade_cost("BUY", value, cfg.costs)
        if value + cost > self.cash:
            raise ValueError(f"not enough cash to buy {qty} {ticker}")
        self.cash -= value + cost
        self.positions[ticker] = Position(ticker, qty, fill, d, stop, strategy, cost)

    def sell(self, ticker: str, price: float, d: date, reason: str, cfg: Config) -> Trade:
        pos = self.positions.pop(ticker)
        fill = price * (1 - cfg.slippage_pct)
        value = pos.qty * fill
        exit_cost = trade_cost("SELL", value, cfg.costs)
        self.cash += value - exit_cost
        trade = Trade(
            ticker, pos.qty, pos.entry_date, pos.entry_price, d, fill,
            pos.entry_cost + exit_cost,
            value - pos.qty * pos.entry_price - pos.entry_cost - exit_cost,
            reason,
        )
        self.trades.append(trade)
        return trade

    def apply_stops(self, d: date, bars: Bars, cfg: Config) -> list[Trade]:
        """open <= stop -> exit at open; elif low <= stop -> exit at stop."""
        ts = pd.Timestamp(d)
        out: list[Trade] = []
        for ticker in list(self.positions):
            df = bars.get(ticker)
            if df is None or ts not in df.index:
                continue
            row = df.loc[ts]
            stop = self.positions[ticker].stop
            if row["open"] <= stop:
                out.append(self.sell(ticker, float(row["open"]), d, "stop", cfg))
            elif row["low"] <= stop:
                out.append(self.sell(ticker, stop, d, "stop", cfg))
        return out

    def equity(self, d: date, bars: Bars) -> float:
        ts = pd.Timestamp(d)
        total = self.cash
        for ticker, pos in self.positions.items():
            closes = bars[ticker]["close"]
            total += pos.qty * float(closes[closes.index <= ts].iloc[-1])
        return total

    def update_peak(self, equity: float) -> None:
        self.peak_equity = max(self.peak_equity, equity)

    def to_dict(self) -> dict:
        def enc(obj) -> dict:
            return {k: (v.isoformat() if isinstance(v, date) else v) for k, v in asdict(obj).items()}
        return {
            "cash": self.cash,
            "peak_equity": self.peak_equity,
            "positions": [enc(p) for p in self.positions.values()],
            "trades": [enc(t) for t in self.trades],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Portfolio":
        p = cls(data["cash"])
        p.peak_equity = data["peak_equity"]
        for r in data["positions"]:
            r = {**r, "entry_date": date.fromisoformat(r["entry_date"])}
            p.positions[r["ticker"]] = Position(**r)
        for r in data["trades"]:
            r = {**r, "entry_date": date.fromisoformat(r["entry_date"]),
                 "exit_date": date.fromisoformat(r["exit_date"])}
            p.trades.append(Trade(**r))
        return p
