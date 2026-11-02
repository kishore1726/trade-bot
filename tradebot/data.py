"""EOD market data: fetch (yfinance), CSV cache per ticker, validation.

Injectable source signature:
    source(ticker: str, start: date, end: date) -> pd.DataFrame
Both start and end are INCLUSIVE. The frame has columns open/high/low/close/volume
(lowercase) and a date index (any order, duplicates allowed, tz-naive or aware).
"""
from __future__ import annotations

from datetime import date, timedelta
from typing import Callable

import pandas as pd

from tradebot.calendar import trading_days_between
from tradebot.config import Config

Bars = dict[str, pd.DataFrame]
COLS = ["open", "high", "low", "close", "volume"]


def _normalise(df: pd.DataFrame) -> pd.DataFrame:
    """Lowercase columns, tz-naive midnight DatetimeIndex, sorted, unique (last wins)."""
    if df is None or len(df) == 0:
        return pd.DataFrame(columns=COLS, index=pd.DatetimeIndex([], name="date"), dtype=float)
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = df.columns.get_level_values(0)
    df = df.rename(columns=lambda c: str(c).lower())[COLS].astype(float)
    idx = pd.DatetimeIndex(pd.to_datetime(df.index))
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    df.index = idx.normalize()
    df.index.name = "date"
    df = df[~df.index.duplicated(keep="last")]
    return df.sort_index()


def yfinance_source(ticker: str, start: date, end: date) -> pd.DataFrame:
    import yfinance as yf

    df = yf.download(
        ticker,
        start=start.isoformat(),
        end=(end + timedelta(days=1)).isoformat(),  # yfinance end is exclusive
        auto_adjust=True,
        progress=False,
    )
    return _normalise(df)


def _read_cache(path) -> pd.DataFrame:
    if not path.exists():
        return _normalise(None)
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    return _normalise(df)


def fetch_bars(
    tickers: list[str],
    start: date,
    end: date,
    cfg: Config,
    source: Callable[[str, date, date], pd.DataFrame] | None = None,
) -> Bars:
    source = source or yfinance_source
    cfg.data_dir.mkdir(parents=True, exist_ok=True)
    needed = trading_days_between(start, end, cfg)
    out: Bars = {}
    for t in tickers:
        path = cfg.data_dir / f"{t}.csv"
        cached = _read_cache(path)
        if needed:
            parts = [cached]
            if cached.empty:
                parts.append(_normalise(source(t, start, end)))
            else:
                first, last = cached.index[0].date(), cached.index[-1].date()
                if first > needed[0]:
                    parts.append(_normalise(source(t, start, first)))
                if last < needed[-1]:
                    parts.append(_normalise(source(t, last, end)))
            if len(parts) > 1:
                merged = _normalise(pd.concat([p for p in parts if not p.empty])) if any(
                    not p.empty for p in parts
                ) else cached
                if not merged.equals(cached):
                    merged.to_csv(path)
                cached = merged
        mask = (cached.index >= pd.Timestamp(start)) & (cached.index <= pd.Timestamp(end))
        out[t] = cached[mask]
    return out


def validate_day(bars: Bars, d: date) -> list[str]:
    """Return problems found for trading day d (empty list = data OK)."""
    if not bars:
        return ["no market data at all"]
    problems: list[str] = []
    ts = pd.Timestamp(d)
    for t, df in bars.items():
        if df is None or df.empty:
            problems.append(f"{t}: no data")
            continue
        if df.index.duplicated().any():
            problems.append(f"{t}: duplicate dates in data")
        if ts not in df.index:
            problems.append(f"{t}: missing bar for {d}")
            continue
        row = df.loc[[ts]].iloc[-1]
        if row[["open", "high", "low", "close"]].isna().any():
            problems.append(f"{t}: NaN price on {d}")
            continue
        if row["low"] > min(row["open"], row["close"]) or row["high"] < max(row["open"], row["close"]):
            problems.append(f"{t}: inconsistent open/high/low/close on {d}")
    return problems
