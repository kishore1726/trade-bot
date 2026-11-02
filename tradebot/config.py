from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class CostConfig:
    brokerage_pct: float = 0.001
    brokerage_max: float = 20.0
    brokerage_min: float = 5.0
    stt_pct: float = 0.001
    exchange_pct: float = 0.0000297
    sebi_per_crore: float = 10.0
    stamp_buy_pct: float = 0.00015
    gst_pct: float = 0.18
    dp_charge_per_sell: float = 20.0


@dataclass(frozen=True)
class TaxConfig:
    stcg_pct: float = 0.20
    ltcg_pct: float = 0.125
    ltcg_exempt_per_year: float = 125000


@dataclass(frozen=True)
class Config:
    capital: float
    max_position_pct: float
    max_positions: int
    max_risk_pct: float
    kill_switch_dd_pct: float
    slippage_pct: float
    costs: CostConfig
    tax: TaxConfig
    universe: list[str]
    benchmark: str
    data_dir: Path
    state_path: Path
    holidays_dir: Path


def _frac(name: str, v: float) -> float:
    if not 0 < v <= 1:
        raise ValueError(f"{name} must be a fraction in (0, 1], got {v}")
    return v


def load_config(path: Path = Path("config.yaml")) -> Config:
    path = Path(path)
    raw = yaml.safe_load(path.read_text()) or {}
    base = path.resolve().parent
    try:
        cfg = Config(
            capital=float(raw["capital"]),
            max_position_pct=_frac("max_position_pct", float(raw["max_position_pct"])),
            max_positions=int(raw["max_positions"]),
            max_risk_pct=_frac("max_risk_pct", float(raw["max_risk_pct"])),
            kill_switch_dd_pct=_frac("kill_switch_dd_pct", float(raw["kill_switch_dd_pct"])),
            slippage_pct=float(raw["slippage_pct"]),
            costs=CostConfig(**(raw.get("costs") or {})),
            tax=TaxConfig(**(raw.get("tax") or {})),
            universe=list(raw["universe"]),
            benchmark=str(raw["benchmark"]),
            data_dir=base / raw.get("data_dir", "data"),
            state_path=base / raw.get("state_path", "state/tradebot.sqlite"),
            holidays_dir=base / raw.get("holidays_dir", "holidays"),
        )
    except KeyError as e:
        raise ValueError(f"missing config setting: {e.args[0]}") from e
    except TypeError as e:
        raise ValueError(f"bad costs/tax settings: {e}") from e
    if cfg.capital <= 0 or cfg.max_positions < 1 or not 0 <= cfg.slippage_pct < 1:
        raise ValueError("capital, max_positions or slippage_pct out of range")
    return cfg
