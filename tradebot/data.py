"""EOD market data: fetch (yfinance), CSV cache per ticker, validation.

Injectable source signature:
    source(ticker: str, start: date, end: date) -> pd.DataFrame
Both start and end are INCLUSIVE. The frame has columns open/high/low/close/volume
(lowercase) and a date index (any order, duplicates allowed, tz-naive or aware).
"""
from __future__ import annotations

import json
import os
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
# Ruling R23: Yahoo re-adjusts a ticker's whole price history after a split, bonus issue or
# dividend. If a re-downloaded bar we already cached differs by more than this, the cached
# history is stale: it is rebuilt from scratch and the ratio new/old close is reported.
ADJUST_TOLERANCE = 0.005
# Ruling R23-amended (C2 residual): every detected adjustment is written to this ledger in the
# data directory BEFORE the rebuilt cache replaces the old one, so the ratio survives a failed
# download later in the run and a `tradebot backtest` download. Its id names the stale cached
# bar, so detecting the same stale cache again (crash before the swap) adds nothing. The daily
# run applies each id once (store key "applied_adjustments") and then removes it here.
ADJ_LEDGER = "adjustments.json"


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


def _adjustment(cached: pd.DataFrame, parts: list[pd.DataFrame]) -> tuple[pd.Timestamp, float] | None:
    """(date, new_close / old_close) on the latest re-downloaded bar that moved more than the
    tolerance, or None."""
    worst: tuple[pd.Timestamp, float] | None = None
    for part in parts:
        common = part.index.intersection(cached.index)
        if common.empty:
            continue
        ratio = part.loc[common, "close"] / cached.loc[common, "close"]
        moved = ratio[(ratio - 1).abs() > ADJUST_TOLERANCE]
        if not moved.empty and (worst is None or moved.index[-1] > worst[0]):
            worst = (moved.index[-1], float(moved.iloc[-1]))
    return worst


def _write_atomic(path, write: Callable[[object], None]) -> None:
    """write(tmp_path) a temp file, then swap it in, so a crash never leaves half a file."""
    tmp = path.with_name(path.name + ".tmp")
    try:
        write(tmp)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def pending_adjustments(cfg: Config) -> list[dict]:
    """Recorded, not yet forgotten adjustments: [{"id", "ticker", "ratio"}], oldest first."""
    path = cfg.data_dir / ADJ_LEDGER
    return json.loads(path.read_text()) if path.exists() else []


def record_adjustment(cfg: Config, ticker: str, ratio: float, key: str) -> None:
    """Add one adjustment to the ledger; an id already there is left as it is."""
    entries = pending_adjustments(cfg)
    if any(e["id"] == key for e in entries):
        return
    entries.append({"id": key, "ticker": ticker, "ratio": ratio})
    cfg.data_dir.mkdir(parents=True, exist_ok=True)
    _write_atomic(cfg.data_dir / ADJ_LEDGER, lambda p: p.write_text(json.dumps(entries, indent=1)))


def forget_adjustments(cfg: Config, ids) -> None:
    ids = set(ids)
    entries = pending_adjustments(cfg)
    keep = [e for e in entries if e["id"] not in ids]
    if len(keep) != len(entries):
        _write_atomic(cfg.data_dir / ADJ_LEDGER, lambda p: p.write_text(json.dumps(keep, indent=1)))


def fetch_bars(
    tickers: list[str],
    start: date,
    end: date,
    cfg: Config,
    source: Callable[[str, date, date], pd.DataFrame] | None = None,
    now: Callable[[], datetime] | None = None,
    adjustments: dict[str, float] | None = None,
) -> Bars:
    """`now` returns the current time (IST if naive); default is the real clock.

    Every ticker whose cache had to be rebuilt because Yahoo re-adjusted its history is recorded
    in the adjustments ledger (see ADJ_LEDGER, Ruling R23). If `adjustments` is given, it also
    receives {ticker: new_close / old_close} for them."""
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
                moved = _adjustment(cached, parts[1:])
                if moved is not None:  # history re-adjusted: rebuild this ticker from scratch
                    at, ratio = moved
                    rebuilt = fetch(t, min(first, start), end)
                    if rebuilt.empty or rebuilt.index[-1] < cached.index[-1]:
                        # Incomplete rebuild download: keep the old cache untouched and record
                        # nothing; never glue new-basis bars to it. The next run tries again.
                        parts = [cached]
                    else:
                        record_adjustment(cfg, t, ratio,
                                          f"{t}|{at.date().isoformat()}|{float(cached.loc[at, 'close'])!r}")
                        parts = [_normalise(None), rebuilt]
                        cached = _normalise(None)
                        if adjustments is not None:
                            adjustments[t] = ratio
            if len(parts) > 1:
                merged = _normalise(pd.concat(parts))
                dirty = dirty or not merged.equals(cached)
                cached = merged
        if dirty:  # M6: atomic cache write
            _write_atomic(path, cached.to_csv)
        mask = (cached.index >= pd.Timestamp(start)) & (cached.index <= pd.Timestamp(end))
        out[t] = cached[mask]
    return out


MISSING, BAD = "missing", "bad"


def is_etf(ticker: str) -> bool:
    """ETFs (e.g. NIFTYBEES.NS) can legitimately show zero volume on Yahoo."""
    t = ticker.upper()
    return "BEES" in t or "ETF" in t


def day_problems(bars: Bars, d: date) -> dict[str, tuple[str, str]]:
    """{ticker: (kind, message)} for trading day d; tickers without a problem are left out.

    kind MISSING: there is no usable bar for d (the ticker can be left out for the day). A stale
    bar counts as missing (M9): OHLC and volume identical to the previous bar (Yahoo repeating
    an old bar), or zero volume for a stock that is not an ETF;
    kind BAD: a bar that cannot be trusted (never trade on it)."""
    out: dict[str, tuple[str, str]] = {}
    ts = pd.Timestamp(d)
    for t, df in bars.items():
        if df is None or df.empty:
            out[t] = (MISSING, "no data")
        elif df.index.duplicated().any():
            out[t] = (BAD, "duplicate dates in data")
        elif ts not in df.index:
            out[t] = (MISSING, f"missing bar for {d}")
        else:
            row = df.loc[ts]
            if row[["open", "high", "low", "close"]].isna().any():
                out[t] = (BAD, f"NaN price on {d}")
            elif row["low"] > min(row["open"], row["close"]) or row["high"] < max(row["open"], row["close"]):
                out[t] = (BAD, f"inconsistent open/high/low/close on {d}")
            elif row["volume"] == 0 and not is_etf(t):
                out[t] = (MISSING, f"stale bar on {d} (zero volume)")
            else:
                before = df[df.index < ts]
                if len(before) and before.iloc[-1][COLS].equals(row[COLS]):
                    out[t] = (MISSING, f"stale bar on {d} (same prices and volume as the day before)")
    return out


def validate_day(bars: Bars, d: date) -> list[str]:
    """Return problems found for trading day d (empty list = data OK)."""
    if not bars:
        return ["no market data at all"]
    return [f"{t}: {msg}" for t, (_, msg) in day_problems(bars, d).items()]
