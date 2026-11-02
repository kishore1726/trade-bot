"""Strategy registry."""
from tradebot.strategies.base import Signal, Strategy
from tradebot.strategies.meanrev import MeanReversion
from tradebot.strategies.trend import TrendFollow

REGISTRY: dict[str, type[Strategy]] = {"trend": TrendFollow, "meanrev": MeanReversion}

__all__ = ["Signal", "Strategy", "REGISTRY", "TrendFollow", "MeanReversion"]
