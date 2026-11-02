import pytest

from tradebot.costs import CostConfig, TaxConfig, estimate_tax_ltcg, estimate_tax_stcg, trade_cost


def test_buy_5000():
    assert trade_cost("BUY", 5000, CostConfig()) == pytest.approx(11.83, abs=0.02)


def test_sell_includes_dp():
    cc = CostConfig()
    no_dp = CostConfig(dp_charge_per_sell=0.0)
    assert trade_cost("SELL", 5000, cc) - trade_cost("SELL", 5000, no_dp) == pytest.approx(20.0)
    # no stamp duty on sell
    no_stamp = CostConfig(dp_charge_per_sell=0.0, stamp_buy_pct=0.0)
    assert trade_cost("SELL", 5000, no_dp) == pytest.approx(trade_cost("SELL", 5000, no_stamp))


def test_brokerage_capped():
    zero = CostConfig(brokerage_min=0.0, brokerage_max=0.0)
    diff = trade_cost("BUY", 100000, CostConfig()) - trade_cost("BUY", 100000, zero)
    # brokerage 20 plus GST on it (0.18 * 20)
    assert diff == pytest.approx(20.0 * 1.18)


def test_stcg_zero_on_loss():
    assert estimate_tax_stcg(-100, TaxConfig()) == 0


def test_ltcg_exemption():
    assert estimate_tax_ltcg(130000, TaxConfig()) == pytest.approx(0.125 * 5000)
    assert estimate_tax_ltcg(100000, TaxConfig()) == 0
