"""Position sizing and the drawdown kill switch."""
from __future__ import annotations

import math

from tradebot.config import Config
from tradebot.costs import trade_cost

_EPS = 1e-9


def size_position(
    equity: float, cash: float, entry: float, stop: float, open_count: int, cfg: Config
) -> tuple[int, str | None]:
    """Whole-share quantity for a new long. `entry` is the already-slipped fill price.

    qty = floor(min(position cap, risk cap, cash after buy costs) / entry).
    Returns (0, reason) when no valid position can be opened; never fractional/negative.
    """
    if open_count >= cfg.max_positions:
        return 0, "already holding the maximum number of positions"
    if entry <= 0 or stop >= entry:
        return 0, "stop is not below the entry price"
    cap_value = equity * cfg.max_position_pct
    risk_value = equity * cfg.max_risk_pct / (entry - stop) * entry
    qty = max(math.floor(min(cap_value, risk_value) / entry + _EPS), 0)
    while qty >= 1 and qty * entry + trade_cost("BUY", qty * entry, cfg.costs) > cash:
        qty -= 1
    if qty < 1:
        return 0, "not enough room or cash to buy even one share within the risk limits"
    return qty, None


def kill_switch_tripped(equity: float, peak: float, cfg: Config) -> bool:
    return equity <= peak * (1 - cfg.kill_switch_dd_pct)
