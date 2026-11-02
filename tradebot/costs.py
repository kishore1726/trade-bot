"""Indian delivery-trade cost and tax model (CostConfig/TaxConfig live in config.py)."""
from typing import Literal

from tradebot.config import CostConfig, TaxConfig

__all__ = ["CostConfig", "TaxConfig", "trade_cost", "estimate_tax_stcg", "estimate_tax_ltcg"]

_CRORE = 1e7


def trade_cost(side: Literal["BUY", "SELL"], value: float, cc: CostConfig) -> float:
    """Total charges in rupees for one delivery trade of the given value."""
    brokerage = min(max(value * cc.brokerage_pct, cc.brokerage_min), cc.brokerage_max)
    stt = value * cc.stt_pct
    exchange = value * cc.exchange_pct
    sebi = value / _CRORE * cc.sebi_per_crore
    gst = cc.gst_pct * (brokerage + exchange + sebi)
    total = brokerage + stt + exchange + sebi + gst
    if side == "BUY":
        total += value * cc.stamp_buy_pct
    else:
        total += cc.dp_charge_per_sell
    return total


def estimate_tax_stcg(net_gain: float, tc: TaxConfig) -> float:
    return max(net_gain, 0.0) * tc.stcg_pct


def estimate_tax_ltcg(net_gain: float, tc: TaxConfig) -> float:
    return max(net_gain - tc.ltcg_exempt_per_year, 0.0) * tc.ltcg_pct
