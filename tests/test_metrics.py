import json
from dataclasses import asdict, replace
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from tradebot.config import Config, CostConfig, TaxConfig
from tradebot.costs import trade_cost
from tradebot.metrics import Metrics, compute_metrics
from tradebot.portfolio import Trade


@pytest.fixture
def cfg():
    return Config(
        capital=20000, max_position_pct=0.25, max_positions=4, max_risk_pct=0.02,
        kill_switch_dd_pct=0.15, slippage_pct=0.001, costs=CostConfig(), tax=TaxConfig(),
        universe=["A.NS"], benchmark="NB.NS", data_dir=Path("d"), state_path=Path("s"),
        holidays_dir=Path("h"),
    )


def trade(pnl, exit_date=date(2024, 6, 3), costs=10.0, qty=10, entry=100.0):
    return Trade("A.NS", qty, date(2024, 6, 1), entry, exit_date, 100.0, costs, pnl, "signal")


def series(values, start="2024-01-01"):
    return pd.Series(values, index=pd.bdate_range(start, periods=len(values)), dtype=float)


def flat_bench(n=4, price=100.0):
    return series([price] * n)


def test_win_rate_and_expectancy(cfg):
    m = compute_metrics([trade(300), trade(-100), trade(-100)], series([20000, 20100]),
                        flat_bench(2), 20000, cfg)
    assert m.trades == 3
    assert m.win_rate == pytest.approx(1 / 3)
    assert m.avg_win == pytest.approx(300)
    assert m.avg_loss == pytest.approx(-100)
    assert m.expectancy == pytest.approx(33.33, abs=0.01)
    assert m.total_costs == pytest.approx(30.0)
    assert m.expectancy_pct == pytest.approx((0.3 - 0.1 - 0.1) / 3)  # position value 1000


def test_expectancy_pct_per_trade_position_value(cfg):
    m = compute_metrics([trade(100, qty=10, entry=100.0), trade(-100, qty=2, entry=250.0)],
                        series([20000, 20000]), flat_bench(2), 20000, cfg)
    assert m.expectancy == pytest.approx(0.0)
    assert m.expectancy_pct == pytest.approx((0.10 - 0.20) / 2)


def test_max_drawdown(cfg):
    m = compute_metrics([], series([100, 120, 90, 130]), flat_bench(), 100, cfg)
    assert m.max_drawdown == pytest.approx(0.25)


def test_empty_trades_gives_zeros(cfg):
    m = compute_metrics([], series([20000, 20000]), flat_bench(2), 20000, cfg)
    assert (m.trades, m.win_rate, m.avg_win, m.avg_loss, m.expectancy, m.expectancy_pct) == (
        0, 0.0, 0.0, 0.0, 0.0, 0.0)
    assert m.total_return == 0.0 and m.max_drawdown == 0.0 and m.post_tax_return == 0.0


def test_below_benchmark_flag(cfg):
    bench = series([100.0, 110.0, 120.0])  # benchmark +20% (less costs)
    weak = compute_metrics([], series([20000, 20500, 21000]), bench, 20000, cfg)
    strong = compute_metrics([], series([20000, 23000, 26000]), bench, 20000, cfg)
    assert weak.below_benchmark is True
    assert strong.below_benchmark is False
    assert weak.benchmark_return > weak.total_return
    assert strong.total_return > strong.benchmark_return


def test_benchmark_buy_and_hold_with_costs(cfg):
    bench = series([100.0, 105.0, 110.0])
    m = compute_metrics([], series([20000, 20000, 20000]), bench, 20000, cfg, benchmark_open=100.0)
    qty = 199  # 200 shares + buy costs would exceed 20000
    left = 20000 - qty * 100.0 - trade_cost("BUY", qty * 100.0, cfg.costs)
    assert m.benchmark_return == pytest.approx((left + qty * 110.0 - 20000) / 20000)
    proceeds = qty * 110.0 - trade_cost("SELL", qty * 110.0, cfg.costs)
    gain = proceeds - (20000 - left)
    tax = 0.20 * gain  # held < 1 year -> STCG
    assert m.benchmark_post_tax_return == pytest.approx((left + proceeds - tax - 20000) / 20000)


def test_benchmark_long_hold_uses_ltcg(cfg):
    idx = pd.DatetimeIndex([pd.Timestamp("2022-01-03"), pd.Timestamp("2024-01-03")])
    bench = pd.Series([100.0, 200.0], index=idx)
    eq = pd.Series([20000.0, 20000.0], index=idx)
    m = compute_metrics([], eq, bench, 20000, cfg, benchmark_open=100.0)
    # gain ~ 19,900 is under the 1.25L LTCG exemption -> no tax, only sale costs
    qty = 199
    left = 20000 - qty * 100.0 - trade_cost("BUY", qty * 100.0, cfg.costs)
    proceeds = qty * 200.0 - trade_cost("SELL", qty * 200.0, cfg.costs)
    assert m.benchmark_post_tax_return == pytest.approx((left + proceeds - 20000) / 20000)
    assert m.benchmark_cagr == pytest.approx(
        ((left + qty * 200.0) / 20000) ** (1 / (730 / 365.25)) - 1)


def test_strategy_tax_stcg_per_exit_year(cfg):
    trades = [trade(1000, date(2024, 3, 1)), trade(-400, date(2024, 5, 1)),
              trade(-300, date(2025, 2, 1))]
    eq = pd.Series([20000.0, 20300.0],
                   index=pd.DatetimeIndex([pd.Timestamp("2024-01-01"), pd.Timestamp("2025-12-31")]))
    m = compute_metrics(trades, eq, pd.Series([100.0, 100.0], index=eq.index), 20000, cfg)
    tax = 0.20 * 600  # 2024 net +600 taxed; 2025 net loss not taxed (and not carried)
    assert m.total_return == pytest.approx(300 / 20000)
    assert m.post_tax_return == pytest.approx((20300 - tax - 20000) / 20000)


def test_cagr(cfg):
    eq = pd.Series([100.0, 121.0],
                   index=pd.DatetimeIndex([pd.Timestamp("2020-01-01"), pd.Timestamp("2021-12-31")]))
    m = compute_metrics([], eq, pd.Series([1.0, 1.0], index=eq.index), 100, cfg)
    years = (date(2021, 12, 31) - date(2020, 1, 1)).days / 365.25
    assert m.cagr == pytest.approx(1.21 ** (1 / years) - 1)


def test_metrics_round_trip(cfg):
    m = compute_metrics([trade(300), trade(-100)], series([20000, 20200]), flat_bench(2), 20000, cfg)
    d = json.loads(json.dumps(asdict(m)))
    assert Metrics.from_dict(d) == m
    assert Metrics.from_dict(asdict(replace(m, trades=7))).trades == 7
    m3 = replace(m, kill_switch_trips=3, expectancy_pct=-0.0123)
    assert Metrics.from_dict(json.loads(json.dumps(asdict(m3)))) == m3
