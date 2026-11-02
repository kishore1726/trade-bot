from dataclasses import replace
from datetime import date, datetime

import numpy as np
import pandas as pd
import pytest

import tradebot.data as data_mod

from tradebot.config import load_config
from tradebot.data import fetch_bars, validate_day

D = date(2026, 11, 3)  # Tuesday
COLS = ["open", "high", "low", "close", "volume"]


def frame(dates, **over):
    idx = pd.DatetimeIndex(pd.to_datetime(dates))
    df = pd.DataFrame(
        {"open": 100.0, "high": 110.0, "low": 95.0, "close": 105.0,
         "volume": [1000.0 + i for i in range(len(idx))]},  # never stale (M9)
        index=idx,
    )
    for k, v in over.items():
        df[k] = v
    return df


@pytest.fixture(autouse=True)
def _after_close(monkeypatch):
    # Pin "now" after every test date's close so the R12 intraday clip never interferes.
    monkeypatch.setattr(data_mod, "_ist_now", lambda: datetime(2026, 12, 31, 20, 0, tzinfo=data_mod.IST))


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


def test_no_partial_bar_before_close(tmp_path):
    cfg = cfg_in(tmp_path)
    calls = []

    def src(t, s, e):
        calls.append((s, e))
        return frame(["2026-11-02", "2026-11-03", "2026-11-04"])  # source wrongly includes today

    at_noon = lambda: datetime(2026, 11, 4, 12, 30, tzinfo=data_mod.IST)  # noqa: E731
    out = fetch_bars(["A.NS"], date(2026, 11, 2), date(2026, 11, 4), cfg, source=src, now=at_noon)
    assert calls[0][1] == date(2026, 11, 3)  # end clipped to yesterday
    assert out["A.NS"].index[-1] == pd.Timestamp("2026-11-03")
    cached = pd.read_csv(cfg.data_dir / "A.NS.csv", index_col=0, parse_dates=True)
    assert cached.index[-1] == pd.Timestamp("2026-11-03")


def test_partial_bar_cached_earlier_is_purged(tmp_path):
    cfg = cfg_in(tmp_path)
    cfg.data_dir.mkdir(parents=True)
    frame(["2026-11-02", "2026-11-03", "2026-11-04"]).rename_axis("date").to_csv(
        cfg.data_dir / "A.NS.csv")
    at_noon = lambda: datetime(2026, 11, 4, 9, 0)  # naive = IST  # noqa: E731
    out = fetch_bars(["A.NS"], date(2026, 11, 2), date(2026, 11, 4), cfg,
                     source=lambda *a: frame([]), now=at_noon)
    assert out["A.NS"].index[-1] == pd.Timestamp("2026-11-03")
    cached = pd.read_csv(cfg.data_dir / "A.NS.csv", index_col=0, parse_dates=True)
    assert pd.Timestamp("2026-11-04") not in cached.index


def test_todays_bar_used_after_close(tmp_path):
    cfg = cfg_in(tmp_path)
    after = lambda: datetime(2026, 11, 4, 16, 5, tzinfo=data_mod.IST)  # noqa: E731
    out = fetch_bars(["A.NS"], date(2026, 11, 2), date(2026, 11, 4), cfg,
                     source=lambda *a: frame(["2026-11-02", "2026-11-03", "2026-11-04"]), now=after)
    assert out["A.NS"].index[-1] == pd.Timestamp("2026-11-04")


# --- C2 / Ruling R23 ----------------------------------------------------------------------

def test_split_rebuilds_cache_and_reports_ratio(tmp_path):
    """Reviewer probe: a 1:2 split re-adjusts Yahoo's whole history; the cached pre-split bars
    must not be glued to post-split bars (fake ~50% fall)."""
    cfg = cfg_in(tmp_path)
    old_days = list(pd.bdate_range("2026-10-01", "2026-10-30"))
    all_days = list(pd.bdate_range("2026-10-01", "2026-11-06"))
    calls = []

    def src(t, s, e):
        calls.append((s, e))
        if len(calls) == 1:
            return frame(old_days, open=2000.0, high=2000.0, low=2000.0, close=2000.0)
        df = frame(all_days, open=1000.0, high=1000.0, low=1000.0, close=1000.0)
        return df[(df.index >= pd.Timestamp(s)) & (df.index <= pd.Timestamp(e))]

    fetch_bars(["X.NS"], date(2026, 10, 1), date(2026, 10, 30), cfg, source=src)
    adj = {}
    out = fetch_bars(["X.NS"], date(2026, 10, 1), date(2026, 11, 6), cfg, source=src, adjustments=adj)
    assert adj == {"X.NS": pytest.approx(0.5)}
    assert (out["X.NS"]["close"] == 1000.0).all() and len(out["X.NS"]) == len(all_days)
    assert calls[-1] == (date(2026, 10, 1), date(2026, 11, 6))  # full rebuild
    cached = pd.read_csv(cfg.data_dir / "X.NS.csv", index_col=0, parse_dates=True)
    assert (cached["close"] == 1000.0).all()


def test_tiny_overlap_difference_is_not_an_adjustment(tmp_path):
    cfg = cfg_in(tmp_path)
    calls = []

    def src(t, s, e):
        calls.append((s, e))
        if len(calls) == 1:
            return frame(["2026-11-02", "2026-11-03"])
        return frame(["2026-11-03", "2026-11-04"], close=105.3)  # 0.29% off: rounding noise

    fetch_bars(["A.NS"], date(2026, 11, 2), date(2026, 11, 3), cfg, source=src)
    adj = {}
    out = fetch_bars(["A.NS"], date(2026, 11, 2), date(2026, 11, 4), cfg, source=src, adjustments=adj)
    assert adj == {} and len(calls) == 2 and len(out["A.NS"]) == 3



# --- M6: cache writes are atomic -----------------------------------------------------------

def test_failed_cache_write_keeps_old_cache(tmp_path, monkeypatch):
    cfg = cfg_in(tmp_path)
    fetch_bars(["A.NS"], date(2026, 11, 2), date(2026, 11, 3), cfg,
               source=lambda *a: frame(["2026-11-02", "2026-11-03"]))
    path = cfg.data_dir / "A.NS.csv"
    good_text = path.read_text()

    def broken(self, target, *a, **k):
        with open(target, "w") as fh:
            fh.write("date,open\n2026-11-02,1")  # half-written, then the disk fails
        raise OSError("disk full")
    monkeypatch.setattr(pd.DataFrame, "to_csv", broken)
    with pytest.raises(OSError):
        fetch_bars(["A.NS"], date(2026, 11, 2), date(2026, 11, 4), cfg,
                   source=lambda *a: frame(["2026-11-03", "2026-11-04"]))
    assert path.read_text() == good_text
    assert [p.name for p in cfg.data_dir.iterdir()] == ["A.NS.csv"]


# --- M9: stale bars ------------------------------------------------------------------------

def test_validate_repeated_bar_is_stale():
    bars = good()
    bars["B.NS"].loc["2026-11-03", "volume"] = bars["B.NS"].loc["2026-11-02", "volume"]
    p = validate_day(bars, D)
    assert len(p) == 1 and "B.NS" in p[0] and "stale" in p[0]


def test_validate_zero_volume_stock_is_stale_but_etf_is_fine():
    bars = {"INFY.NS": frame(["2026-11-02", "2026-11-03"]),
            "NIFTYBEES.NS": frame(["2026-11-02", "2026-11-03"])}
    for t in bars:
        bars[t].loc["2026-11-03", "volume"] = 0.0
    p = validate_day(bars, D)
    assert len(p) == 1 and "INFY.NS" in p[0] and "volume" in p[0]
