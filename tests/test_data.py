from dataclasses import replace
from datetime import date

import numpy as np
import pandas as pd

from tradebot.config import load_config
from tradebot.data import fetch_bars, validate_day

D = date(2026, 11, 3)  # Tuesday
COLS = ["open", "high", "low", "close", "volume"]


def frame(dates, **over):
    idx = pd.DatetimeIndex(pd.to_datetime(dates))
    df = pd.DataFrame(
        {"open": 100.0, "high": 110.0, "low": 95.0, "close": 105.0, "volume": 1000.0},
        index=idx,
    )
    for k, v in over.items():
        df[k] = v
    return df


def cfg_in(tmp_path):
    return replace(load_config(), data_dir=tmp_path / "data")


def test_fetch_uses_cache(tmp_path):
    cfg = cfg_in(tmp_path)
    calls = []

    def src(t, s, e):
        calls.append((t, s, e))
        return frame(["2026-11-02", "2026-11-03", "2026-11-04"])

    a = fetch_bars(["AAA.NS"], date(2026, 11, 2), date(2026, 11, 4), cfg, source=src)
    b = fetch_bars(["AAA.NS"], date(2026, 11, 2), date(2026, 11, 4), cfg, source=src)
    assert len(calls) == 1
    assert len(a["AAA.NS"]) == 3
    pd.testing.assert_frame_equal(a["AAA.NS"], b["AAA.NS"], check_freq=False)
    assert (cfg.data_dir / "AAA.NS.csv").exists()


def test_fetch_cache_allows_nontrading_edges(tmp_path):
    cfg = cfg_in(tmp_path)
    calls = []

    def src(t, s, e):
        calls.append(1)
        return frame(["2026-11-05", "2026-11-06"])

    fetch_bars(["A.NS"], date(2026, 11, 5), date(2026, 11, 6), cfg, source=src)
    # Sat/Sun at the end and start edges: no new trading days needed
    fetch_bars(["A.NS"], date(2026, 11, 5), date(2026, 11, 8), cfg, source=src)
    assert len(calls) == 1


def test_fetch_extends_tail_only(tmp_path):
    cfg = cfg_in(tmp_path)
    calls = []

    def src(t, s, e):
        calls.append((s, e))
        return frame(["2026-11-02", "2026-11-03"]) if len(calls) == 1 else frame(["2026-11-04"])

    fetch_bars(["A.NS"], date(2026, 11, 2), date(2026, 11, 3), cfg, source=src)
    out = fetch_bars(["A.NS"], date(2026, 11, 2), date(2026, 11, 4), cfg, source=src)
    assert len(calls) == 2 and calls[1][0] >= date(2026, 11, 3)
    assert list(out["A.NS"].index.date) == [date(2026, 11, 2), date(2026, 11, 3), date(2026, 11, 4)]


def test_fetch_slices_to_range(tmp_path):
    cfg = cfg_in(tmp_path)
    src = lambda t, s, e: frame(["2026-11-02", "2026-11-03", "2026-11-04"])
    out = fetch_bars(["A.NS"], date(2026, 11, 3), date(2026, 11, 3), cfg, source=src)
    assert list(out["A.NS"].index.date) == [date(2026, 11, 3)]


def test_fetch_dedupes_and_sorts(tmp_path):
    cfg = cfg_in(tmp_path)
    src = lambda t, s, e: frame(["2026-11-04", "2026-11-02", "2026-11-03", "2026-11-02"])
    out = fetch_bars(["A.NS"], date(2026, 11, 2), date(2026, 11, 4), cfg, source=src)["A.NS"]
    assert out.index.is_unique and out.index.is_monotonic_increasing
    assert len(out) == 3
    assert list(out.columns) == COLS


def good():
    return {"A.NS": frame(["2026-11-02", "2026-11-03"]), "B.NS": frame(["2026-11-02", "2026-11-03"])}


def test_validate_ok():
    assert validate_day(good(), D) == []


def test_validate_missing_ticker_bar():
    bars = good()
    bars["B.NS"] = frame(["2026-11-02"])
    p = validate_day(bars, D)
    assert len(p) == 1 and "B.NS" in p[0]


def test_validate_nan():
    bars = good()
    bars["A.NS"].loc["2026-11-03", "close"] = np.nan
    p = validate_day(bars, D)
    assert len(p) == 1 and "A.NS" in p[0]


def test_validate_bad_ohlc():
    bars = good()
    bars["A.NS"].loc["2026-11-03", "low"] = 105.0
    bars["A.NS"].loc["2026-11-03", "open"] = 100.0
    assert any("A.NS" in x for x in validate_day(bars, D))


def test_validate_high_below_close():
    bars = good()
    bars["A.NS"].loc["2026-11-03", "high"] = 100.0
    assert validate_day(bars, D)


def test_validate_duplicate_dates():
    bars = good()
    bars["A.NS"] = pd.concat([bars["A.NS"], bars["A.NS"].iloc[[1]]])
    assert any("duplicate" in x for x in validate_day(bars, D))


def test_validate_empty_bars_and_empty_frame():
    assert validate_day({}, D)
    bars = good()
    bars["A.NS"] = bars["A.NS"].iloc[0:0]
    p = validate_day(bars, D)
    assert len(p) == 1 and "A.NS" in p[0]


def test_bad_bar_is_not_frozen_in_cache(tmp_path):
    cfg = cfg_in(tmp_path)
    n = []

    def src(t, s, e):
        n.append(1)
        df = frame(["2026-11-02", "2026-11-03"])
        if len(n) == 1:
            df.loc["2026-11-03", "close"] = np.nan
        return df

    first = fetch_bars(["A.NS"], date(2026, 11, 2), D, cfg, source=src)
    assert validate_day(first, D)
    second = fetch_bars(["A.NS"], date(2026, 11, 2), D, cfg, source=src)
    assert len(n) == 2
    assert validate_day(second, D) == []


def test_interior_gap_within_window_is_filled(tmp_path):
    cfg = cfg_in(tmp_path)
    days = ["2026-11-02", "2026-11-03", "2026-11-04", "2026-11-05", "2026-11-06"]
    calls = []

    def src(t, s, e):
        calls.append((s, e))
        if len(calls) == 1:
            return frame([d for d in days if d != "2026-11-03"])
        return frame(days)

    fetch_bars(["A.NS"], date(2026, 11, 2), date(2026, 11, 6), cfg, source=src)
    out = fetch_bars(["A.NS"], date(2026, 11, 2), date(2026, 11, 6), cfg, source=src)
    assert len(calls) == 2 and calls[1][0] <= date(2026, 11, 3) <= calls[1][1]
    assert len(out["A.NS"]) == 5
