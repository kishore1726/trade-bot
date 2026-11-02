"""Strategy interface, shared indicators and helpers. Strategies only read rows <= d."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Literal, Protocol

import pandas as pd

from tradebot.data import Bars
from tradebot.portfolio import Position


@dataclass
class Signal:
    ticker: str
    action: Literal["BUY", "SELL"]
    stop: float | None
    reason: str  # plain English, shown on Telegram


class Strategy(Protocol):
    name: str
    lookback_days: int

    def signals(self, bars: Bars, d: date, positions: dict[str, Position]) -> list[Signal]: ...


def history(df: pd.DataFrame, d: date) -> pd.DataFrame:
    """Rows with index <= d (never look ahead)."""
    return df.loc[df.index <= pd.Timestamp(d)]


def usable(df: pd.DataFrame, d: date, lookback_days: int) -> pd.DataFrame | None:
    """History up to d, or None if there is no bar on d or too little history."""
    h = history(df, d)
    if len(h) < lookback_days or h.index[-1] != pd.Timestamp(d):
        return None
    return h


def sma(close: pd.Series, n: int) -> float:
    return float(close.iloc[-n:].mean())


def atr(h: pd.DataFrame, n: int = 20) -> float:
    """Simple mean of true range over the last n bars."""
    prev_close = h["close"].shift(1)
    tr = pd.concat([h["high"] - h["low"], (h["high"] - prev_close).abs(),
                    (h["low"] - prev_close).abs()], axis=1).max(axis=1)
    return float(tr.iloc[-n:].mean())


def rsi(close: pd.Series, n: int = 2) -> float:
    """Wilder-smoothed RSI of the last close."""
    delta = close.diff().dropna()
    gain = delta.clip(lower=0.0).to_numpy()
    loss = (-delta.clip(upper=0.0)).to_numpy()
    avg_g = gain[:n].mean()
    avg_l = loss[:n].mean()
    for g, l in zip(gain[n:], loss[n:]):
        avg_g = (avg_g * (n - 1) + g) / n
        avg_l = (avg_l * (n - 1) + l) / n
    if avg_l == 0:
        return 100.0 if avg_g > 0 else 50.0
    return float(100.0 - 100.0 / (1.0 + avg_g / avg_l))
