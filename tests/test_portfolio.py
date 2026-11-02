from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from tradebot.config import Config, CostConfig, TaxConfig
from tradebot.costs import trade_cost
from tradebot.portfolio import Portfolio

D0, D1 = date(2026, 1, 5), date(2026, 1, 6)


@pytest.fixture
def cfg():
    return Config(
        capital=20000, max_position_pct=0.25, max_positions=4, max_risk_pct=0.02,
        kill_switch_dd_pct=0.15, slippage_pct=0.001, costs=CostConfig(), tax=TaxConfig(),
        universe=["A.NS"], benchmark="A.NS", data_dir=Path("d"), state_path=Path("s"),
        holidays_dir=Path("h"),
    )


def bar(d, o, h, l, c):
    return pd.DataFrame(
        {"open": [o], "high": [h], "low": [l], "close": [c], "volume": [1000.0]},
        index=pd.DatetimeIndex([pd.Timestamp(d)]),
    )


def held(cfg, stop=95.0):
    p = Portfolio(20000.0)
    p.buy("A.NS", 10, 100.0, D0, stop, "s1", cfg)
    return p


def test_buy_reduces_cash_by_value_plus_cost(cfg):
    p = held(cfg)
    fill = 100.0 * 1.001
    value = 10 * fill
    assert p.cash == pytest.approx(20000 - value - trade_cost("BUY", value, cfg.costs))
    pos = p.positions["A.NS"]
    assert pos.entry_price == pytest.approx(fill) and pos.qty == 10
    assert pos.entry_cost == pytest.approx(trade_cost("BUY", value, cfg.costs))


def test_buy_guards(cfg):
    p = held(cfg)
    with pytest.raises(ValueError):
        p.buy("A.NS", 1, 100.0, D0, 95.0, "s1", cfg)  # already held
    for q in (0, -3, 1.5):
        with pytest.raises(ValueError):
            p.buy("B.NS", q, 100.0, D0, 95.0, "s1", cfg)
    with pytest.raises(ValueError):
        p.buy("B.NS", 1000, 100.0, D0, 95.0, "s1", cfg)  # no cash
    assert list(p.positions) == ["A.NS"]


def test_gap_down_exits_at_open(cfg):
    p = held(cfg)
    trades = p.apply_stops(D1, {"A.NS": bar(D1, 90, 92, 88, 91)}, cfg)
    assert len(trades) == 1
    t = trades[0]
    assert t.exit_price == pytest.approx(90 * 0.999) and t.reason == "stop"
    assert "A.NS" not in p.positions and p.trades == trades


def test_intraday_stop_exits_at_stop(cfg):
    p = held(cfg)
    trades = p.apply_stops(D1, {"A.NS": bar(D1, 98, 99, 94, 96)}, cfg)
    assert trades[0].exit_price == pytest.approx(95 * 0.999)


def test_no_stop_hit(cfg):
    p = held(cfg)
    assert p.apply_stops(D1, {"A.NS": bar(D1, 98, 99, 96, 97)}, cfg) == []
    assert "A.NS" in p.positions


def test_stop_applies_on_entry_day(cfg):
    p = held(cfg)
    assert len(p.apply_stops(D0, {"A.NS": bar(D0, 100, 101, 94, 96)}, cfg)) == 1


def test_missing_bar_skipped(cfg):
    p = held(cfg)
    assert p.apply_stops(D1, {}, cfg) == []
    assert "A.NS" in p.positions


def test_sell_pnl_net_and_cash(cfg):
    p = held(cfg)
    cash_before = p.cash
    t = p.sell("A.NS", 110.0, D1, "signal", cfg)
    exit_px = 110 * 0.999
    exit_val = 10 * exit_px
    exit_cost = trade_cost("SELL", exit_val, cfg.costs)
    pos_cost = trade_cost("BUY", 10 * 100.1, cfg.costs)
    assert t.costs == pytest.approx(pos_cost + exit_cost)
    assert t.pnl == pytest.approx(exit_val - 10 * 100.1 - pos_cost - exit_cost)
    assert p.cash == pytest.approx(cash_before + exit_val - exit_cost)
    assert t.reason == "signal" and p.positions == {}


def test_equity_and_peak(cfg):
    p = held(cfg)
    bars = {"A.NS": bar(D0, 100, 101, 99, 105)}
    assert p.equity(D0, bars) == pytest.approx(p.cash + 10 * 105)
    # no bar on D1 -> last close <= D1
    assert p.equity(D1, bars) == pytest.approx(p.cash + 10 * 105)
    p.update_peak(30000)
    p.update_peak(25000)
    assert p.peak_equity == 30000


def test_roundtrip_dict(cfg):
    p = held(cfg)
    p.apply_stops(D1, {"A.NS": bar(D1, 90, 92, 88, 91)}, cfg)
    p.buy("B.NS", 5, 200.0, D1, 190.0, "s2", cfg)
    p.update_peak(21000)
    q = Portfolio.from_dict(p.to_dict())
    assert q.cash == p.cash and q.positions == p.positions
    assert q.trades == p.trades and q.peak_equity == p.peak_equity
    import json
    assert json.loads(json.dumps(p.to_dict())) == p.to_dict()
