"""Trend following: own the strongest stocks trading above their 100-day average."""
from __future__ import annotations

from datetime import date

from tradebot.data import Bars
from tradebot.portfolio import Position
from tradebot.strategies.base import Signal, atr, sma, usable


class TrendFollow:
    name = "trend"
    sma_days = 100
    momentum_days = 63
    top_n = 4
    atr_days = 20
    atr_mult = 2.0
    lookback_days = 101

    def signals(self, bars: Bars, d: date, positions: dict[str, Position]) -> list[Signal]:
        hist = {}
        for t, df in bars.items():
            h = usable(df, d, max(self.lookback_days, self.momentum_days + 1))
            if h is not None:
                hist[t] = h
        ret = {t: float(h["close"].iloc[-1] / h["close"].iloc[-1 - self.momentum_days] - 1)
               for t, h in hist.items()}
        top = sorted(ret, key=lambda t: (-ret[t], t))[: self.top_n]

        sells: list[Signal] = []
        buys: list[tuple[float, Signal]] = []
        for t, h in hist.items():
            close = float(h["close"].iloc[-1])
            avg = sma(h["close"], self.sma_days)
            if t in positions:
                if close < avg:
                    sells.append(Signal(t, "SELL", None, "Price fell below the 100-day average"))
            elif t in top and ret[t] > 0 and close > avg:
                stop = close - self.atr_mult * atr(h, self.atr_days)
                buys.append((ret[t], Signal(
                    t, "BUY", stop, "Price above 100-day average and among top 4 by 3-month gain")))
        buys.sort(key=lambda x: (-x[0], x[1].ticker))
        return sells + [s for _, s in buys]
