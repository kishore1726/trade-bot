"""EOD market data: fetch (yfinance), CSV cache per ticker, validation.

Injectable source signature:
    source(ticker: str, start: date, end: date) -> pd.DataFrame
Both start and end are INCLUSIVE. The frame has columns open/high/low/close/volume
(lowercase) and a date index (any order, duplicates allowed, tz-naive or aware).
"""
from __future__ import annotations

from datetime import date, datetime, time, timedelta
from typing import Callable
from zoneinfo import ZoneInfo

import pandas as pd

from tradebot.calendar import trading_days_between
from tradebot.config import Config

Bars = dict[str, pd.DataFrame]
COLS = ["open", "high", "low", "close", "volume"]
# Ruling R8: interior gaps (trading days missing from the cache) are only detected and
# refetched within the last GAP_WINDOW trading days up to `end`. Older interior gaps are accepted.
GAP_WINDOW = 30
IST = ZoneInfo("Asia/Kolkata")
# Ruling R12: before this IST time today's bar may be partial (market still open / not settled),
# so it is never returned or cached.
DAY_BAR_READY = time(16, 0)


def _ist_now() -> datetime:
    return datetime.now(IST)


def _partial_day(now: datetime) -> date | None:
    """Today's IST date if its bar may still be partial (before 16:00 IST), else None."""
    now = now.astimezone(IST) if now.tzinfo else now.replace(tzinfo=IST)
    return now.date() if now.time() < DAY_BAR_READY else None


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
    # A bad bar (NaN or non-positive price) becomes a missing bar, so it is never cached
    # as if good and gets refetched on the next run.
    ohlc = df[["open", "high", "low", "close"]]
    df = df[ohlc.notna().all(axis=1) & (ohlc > 0).all(axis=1)]
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
    now: Callable[[], datetime] | None = None,
) -> Bars:
    """`now` returns the current time (IST if naive); default is the real clock."""
    source = source or yfinance_source
    partial = _partial_day((now or _ist_now)())
    if partial is not None:
        end = min(end, partial - timedelta(days=1))
    if end < start:
        return {t: _normalise(None) for t in tickers}
    cutoff = pd.Timestamp(end)

    def fetch(t: str, s: date, e: date) -> pd.DataFrame:
        df = _normalise(source(t, s, e))
        return df[df.index <= cutoff]  # never cache rows past the (clipped) end
    cfg.data_dir.mkdir(parents=True, exist_ok=True)
    needed = trading_days_between(start, end, cfg)
    out: Bars = {}
    for t in tickers:
        path = cfg.data_dir / f"{t}.csv"
        cached = _read_cache(path)
        dirty = False
        if partial is not None and not cached.empty and cached.index[-1] >= pd.Timestamp(partial):
            cached = cached[cached.index < pd.Timestamp(partial)]  # purge a partial bar cached earlier
            dirty = True
        if needed:
            parts = [cached]
            if cached.empty:
                parts.append(fetch(t, start, end))
            else:
                first, last = cached.index[0].date(), cached.index[-1].date()
                if first > needed[0]:
                    parts.append(fetch(t, start, first))
                if last < needed[-1]:
                    parts.append(fetch(t, last, end))
                have = {i.date() for i in cached.index}
                gaps = [d for d in needed[-GAP_WINDOW:] if d not in have and first < d < last]
                if gaps:
                    parts.append(fetch(t, gaps[0], gaps[-1]))
            if len(parts) > 1:
                merged = _normalise(pd.concat(parts))
                dirty = dirty or not merged.equals(cached)
                cached = merged
        if dirty:
            cached.to_csv(path)
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
