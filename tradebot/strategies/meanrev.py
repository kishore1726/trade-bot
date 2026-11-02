"""Mean reversion: buy sharp short-term dips in stocks that are in a long-term uptrend."""
from __future__ import annotations

from datetime import date

import pandas as pd

from tradebot.data import Bars
from tradebot.portfolio import Position
from tradebot.strategies.base import Signal, atr, rsi, sma, usable


class MeanReversion:
    name = "meanrev"
    trend_days = 200
    exit_sma_days = 5
    rsi_days = 2
    rsi_buy_below = 10.0
    max_hold_days = 10
    atr_days = 20
    atr_mult = 2.0
    lookback_days = 201

    def signals(self, bars: Bars, d: date, positions: dict[str, Position]) -> list[Signal]:
        sells: list[Signal] = []
        buys: list[tuple[float, Signal]] = []
        for t, df in bars.items():
            h = usable(df, d, self.lookback_days)
            if h is None:
                continue
            close = float(h["close"].iloc[-1])
            if t in positions:
                pos = positions[t]
                idx = h.index
                held = int(((idx > pd.Timestamp(pos.entry_date)) & (idx <= pd.Timestamp(d))).sum())
                if close > sma(h["close"], self.exit_sma_days):
                    sells.append(Signal(t, "SELL", None, "Price bounced back above the 5-day average"))
                elif held >= self.max_hold_days:
                    sells.append(Signal(t, "SELL", None, "Held for 10 trading days, time to exit"))
                continue
            if close > sma(h["close"], self.trend_days):
                r = rsi(h["close"], self.rsi_days)
                if r < self.rsi_buy_below:
                    stop = close - self.atr_mult * atr(h, self.atr_days)
                    buys.append((r, Signal(
                        t, "BUY", stop,
                        "Sharp short-term dip in a stock above its 200-day average")))
        buys.sort(key=lambda x: (x[0], x[1].ticker))
        return sells + [s for _, s in buys]
