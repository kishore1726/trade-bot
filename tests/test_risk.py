from dataclasses import replace
from pathlib import Path

import pytest

from tradebot.config import Config, CostConfig, TaxConfig
from tradebot.costs import trade_cost
from tradebot.risk import kill_switch_tripped, size_position


@pytest.fixture
def cfg():
    return Config(
        capital=20000, max_position_pct=0.25, max_positions=4, max_risk_pct=0.02,
        kill_switch_dd_pct=0.15, slippage_pct=0.001, costs=CostConfig(), tax=TaxConfig(),
        universe=["A.NS"], benchmark="A.NS", data_dir=Path("d"), state_path=Path("s"),
        holidays_dir=Path("h"),
    )


def test_size_risk_limited(cfg):
    assert size_position(20000, 20000, 100, 90, 0, cfg) == (40, None)


def test_size_position_limited(cfg):
    assert size_position(20000, 20000, 100, 99, 0, cfg) == (50, None)


def test_size_too_expensive(cfg):
    qty, reason = size_position(20000, 20000, 6000, 5800, 0, cfg)
    assert qty == 0 and "one share" in reason


def test_size_max_positions(cfg):
    qty, reason = size_position(20000, 20000, 100, 90, 4, cfg)
    assert qty == 0 and reason


def test_size_bad_stop(cfg):
    for stop in (100, 101):
        qty, reason = size_position(20000, 20000, 100, stop, 0, cfg)
        assert qty == 0 and reason


def test_size_cash_short_never_fractional_or_negative(cfg):
    # position cap would allow 50 shares but cash only covers 10
    qty, reason = size_position(20000, 1005, 100, 99, 0, cfg)
    assert isinstance(qty, int) and 0 < qty <= 10
    assert qty * 100 + trade_cost("BUY", qty * 100, cfg.costs) <= 1005
    # cash below one share, zero and negative cash
    for cash in (50, 0, -500):
        qty, reason = size_position(20000, cash, 100, 99, 0, cfg)
        assert qty == 0 and reason and "one share" in reason


def test_size_cash_limit_includes_costs(cfg):
    # cash exactly equals 50 shares of value: costs make 50 unaffordable
    qty, _ = size_position(20000, 5000, 100, 99, 0, cfg)
    assert qty < 50
    assert qty * 100 + trade_cost("BUY", qty * 100, cfg.costs) <= 5000


def test_kill_switch(cfg):
    assert kill_switch_tripped(17000, 20000, cfg) is True
    assert kill_switch_tripped(17001, 20000, cfg) is False
