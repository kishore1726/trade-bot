from datetime import date

import numpy as np
import pandas as pd
import pytest

from tradebot.portfolio import Position
from tradebot.strategies import REGISTRY, MeanReversion, TrendFollow


def mk(closes, start="2024-01-01"):
    idx = pd.bdate_range(start, periods=len(closes))
    c = np.asarray(closes, dtype=float)
    return pd.DataFrame({"open": c, "high": c * 1.01, "low": c * 0.99, "close": c,
                         "volume": 1000}, index=idx)


def day(df, i):
    return df.index[i].date()


def pos(ticker, entry_date, strategy="x"):
    return Position(ticker, 10, 100.0, entry_date, 90.0, strategy)


def uptrend(n=260, slope=0.3):
    return 100 + slope * np.arange(n)


def dip_series():
    c = list(uptrend(260))
    c[-3:] = [c[-4] * 0.97, c[-4] * 0.93, c[-4] * 0.88]
    return c


def universe():
    bars = {"UP1": mk(uptrend(260, 0.5)), "UP2": mk(uptrend(260, 0.4)),
            "UP3": mk(uptrend(260, 0.3)), "UP4": mk(uptrend(260, 0.2)),
            "UP5": mk(uptrend(260, 0.1)), "FLAT": mk([100.0] * 260)}
    return bars


def test_registry_constructs():
    assert isinstance(REGISTRY["trend"](), TrendFollow)
    assert isinstance(REGISTRY["meanrev"](), MeanReversion)


def test_trend_buys_uptrend():
    bars = universe()
    d = day(bars["UP1"], -1)
    sigs = TrendFollow().signals(bars, d, {})
    assert [s.ticker for s in sigs] == ["UP1", "UP2", "UP3", "UP4"]
    assert all(s.action == "BUY" for s in sigs)
    assert len(sigs) == 4
    assert "FLAT" not in {s.ticker for s in sigs}
    assert sigs[0].reason


def test_trend_sells_below_sma():
    c = list(uptrend(260))
    c[-1] = 50.0
    bars = {"A": mk(c)}
    d = day(bars["A"], -1)
    sigs = TrendFollow().signals(bars, d, {"A": pos("A", day(bars["A"], -30), "trend")})
    assert len(sigs) == 1 and sigs[0].action == "SELL" and sigs[0].stop is None


def test_meanrev_buys_dip_in_uptrend():
    bars = {"A": mk(dip_series())}
    d = day(bars["A"], -1)
    sigs = MeanReversion().signals(bars, d, {})
    assert [(s.ticker, s.action) for s in sigs] == [("A", "BUY")]


def test_meanrev_time_exit_after_10_days():
    c = list(uptrend(260))
    # decline over the last 12 days so close stays below SMA5
    for i in range(12):
        c[-12 + i] = c[-13] - 2 * (i + 1)
    bars = {"A": mk(c)}
    df = bars["A"]
    held = {"A": pos("A", df.index[-11].date(), "meanrev")}  # 10 bars after entry
    sigs = MeanReversion().signals(bars, df.index[-1].date(), held)
    assert [(s.action, s.stop) for s in sigs] == [("SELL", None)]
    assert "10" in sigs[0].reason
    held = {"A": pos("A", df.index[-10].date(), "meanrev")}  # 9 bars
    assert MeanReversion().signals(bars, df.index[-1].date(), held) == []


def test_meanrev_sells_above_sma5():
    c = list(uptrend(260))
    c[-1] = c[-2] * 1.05
    bars = {"A": mk(c)}
    df = bars["A"]
    sigs = MeanReversion().signals(bars, df.index[-1].date(), {"A": pos("A", df.index[-3].date())})
    assert [s.action for s in sigs] == ["SELL"]


def test_buy_has_stop_below_close():
    bars = universe()
    d = day(bars["UP1"], -1)
    for s in TrendFollow().signals(bars, d, {}):
        assert s.stop is not None and s.stop < bars[s.ticker]["close"].iloc[-1]
    bars = {"A": mk(dip_series())}
    sigs = MeanReversion().signals(bars, day(bars["A"], -1), {})
    assert sigs and 0 < sigs[0].stop < bars["A"]["close"].iloc[-1]


def test_no_buy_when_held():
    bars = universe()
    d = day(bars["UP1"], -1)
    held = {t: pos(t, day(bars[t], -5), "trend") for t in ["UP1", "UP2"]}
    sigs = TrendFollow().signals(bars, d, held)
    assert all(s.action != "BUY" or s.ticker not in held for s in sigs)
    bars = {"A": mk(dip_series())}
    sigs = MeanReversion().signals(bars, day(bars["A"], -1), {"A": pos("A", day(bars["A"], -2))})
    assert all(s.action != "BUY" for s in sigs)


@pytest.mark.parametrize("cls", [TrendFollow, MeanReversion])
def test_no_lookahead(cls):
    full = {**universe(), "DIP": mk(dip_series())}
    d = day(full["UP1"], -1)
    cut = {t: df.iloc[:-1] for t, df in full.items()}
    # d is now the last row of `cut`; extend `full` with junk rows after d
    junk = {}
    for t, df in cut.items():
        extra = mk([1.0, 1e6, 1.0], start=str(df.index[-1].date() + pd.Timedelta(days=1)))
        junk[t] = pd.concat([df, extra])
    d = cut["UP1"].index[-1].date()
    held = {"UP1": pos("UP1", cut["UP1"].index[-15].date())}
    assert cls().signals(cut, d, held) == cls().signals(junk, d, held)


@pytest.mark.parametrize("cls", [TrendFollow, MeanReversion])
def test_insufficient_history_no_signal(cls):
    n = cls.lookback_days - 1
    bars = {"A": mk(uptrend(n))}
    d = day(bars["A"], -1)
    assert cls().signals(bars, d, {}) == []
    assert cls().signals(bars, d, {"A": pos("A", day(bars["A"], -5))}) == []


def test_no_bar_on_d_no_signal():
    bars = universe()
    d = day(bars["UP1"], -1)
    bars["UP1"] = bars["UP1"].iloc[:-1]
    sigs = TrendFollow().signals(bars, d, {})
    assert "UP1" not in {s.ticker for s in sigs}
