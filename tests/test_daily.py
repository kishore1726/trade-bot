from dataclasses import asdict
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from tradebot import daily
from tradebot.backtest import run_backtest
from tradebot.config import Config, CostConfig, TaxConfig
from tradebot.portfolio import Portfolio
from tradebot.store import IST, Store
from tradebot.strategies import REGISTRY, Signal

# Week of 2024-01-01: Mon 1 .. Fri 5.
MON, TUE, WED, THU, FRI = (date(2024, 1, d) for d in range(1, 6))
TICKERS = ["A.NS", "B.NS"]
BENCH = "NB.NS"


@pytest.fixture
def cfg(tmp_path):
    (tmp_path / "holidays").mkdir()
    return Config(
        capital=20000, max_position_pct=0.25, max_positions=4, max_risk_pct=0.02,
        kill_switch_dd_pct=0.15, slippage_pct=0.001, costs=CostConfig(), tax=TaxConfig(),
        universe=list(TICKERS), benchmark=BENCH, data_dir=tmp_path / "data",
        state_path=tmp_path / "state" / "tradebot.sqlite", holidays_dir=tmp_path / "holidays",
    )


@pytest.fixture
def store(cfg):
    s = Store(cfg.state_path)
    yield s
    s.close()


def after_close(d):
    return lambda: datetime(d.year, d.month, d.day, 17, 0, tzinfo=IST)


def frame(index, opens, closes=None, lows=None):
    closes = list(closes if closes is not None else opens)
    lows = list(lows if lows is not None else [min(o, c) for o, c in zip(opens, closes)])
    highs = [max(o, c) for o, c in zip(opens, closes)]
    return pd.DataFrame({"open": list(opens), "high": highs, "low": lows, "close": closes,
                         "volume": [1000.0] * len(index)}, index=pd.DatetimeIndex(index), dtype=float)


def flat_bars(start=date(2023, 12, 1), end=date(2024, 2, 29), price=100.0):
    idx = pd.bdate_range(start, end)
    return {t: frame(idx, [price] * len(idx)) for t in TICKERS + [BENCH]}


class FakeFetch:
    def __init__(self, bars, error=None):
        self.bars, self.error, self.calls = bars, error, []

    adjust: dict = {}  # what fetch_bars would report as price adjustments (C2 / R23)

    def __call__(self, tickers, start, end, cfg, now=None, adjustments=None):
        self.calls.append((list(tickers), start, end))
        self.now = now
        if adjustments is not None:
            adjustments.update(self.adjust)
        if self.error:
            raise self.error
        lo, hi = pd.Timestamp(start), pd.Timestamp(end)
        return {t: self.bars[t][(self.bars[t].index >= lo) & (self.bars[t].index <= hi)]
                for t in tickers if t in self.bars}


class Sender:
    def __init__(self):
        self.messages = []

    def __call__(self, text, store):
        self.messages.append(text)
        return True


class Scripted:
    """BUY A at the close of `buy_on` (stop given), SELL A at the close of `sell_on`."""
    name = "scripted"
    lookback_days = 1
    buy_on: set = set()
    sell_on: set = set()
    stop = 90.0

    def signals(self, bars, d, positions):
        out = []
        if d in self.buy_on and "A.NS" not in positions:
            out.append(Signal("A.NS", "BUY", self.stop, "test buy"))
        if d in self.sell_on and "A.NS" in positions:
            out.append(Signal("A.NS", "SELL", None, "test sell"))
        return out


class Momentum:
    """Stateless, data-driven: BUY on an up close (stop 3% below), SELL held on a down close."""
    name = "momentum"
    lookback_days = 2

    def signals(self, bars, d, positions):
        ts = pd.Timestamp(d)
        out = []
        for t in TICKERS:
            c = bars[t]["close"]
            c = c[c.index <= ts]
            if len(c) < 2:
                continue
            up = c.iloc[-1] > c.iloc[-2]
            if t in positions and not up:
                out.append(Signal(t, "SELL", None, "down close"))
            elif t not in positions and up:
                out.append(Signal(t, "BUY", float(c.iloc[-1]) * 0.97, "up close"))
        return out


@pytest.fixture
def scripted(monkeypatch):
    monkeypatch.setitem(REGISTRY, "scripted", Scripted)
    monkeypatch.setattr(Scripted, "buy_on", set())
    monkeypatch.setattr(Scripted, "sell_on", set())
    return Scripted


def start_paper(store, last_day, name="scripted"):
    store.set("chosen_strategy", name)
    store.set("last_processed_day", last_day.isoformat())


def run(today, cfg, store, fetch, sender, now=None):
    return daily.run_daily(today, cfg, store, fetch=fetch, sender=sender, now=now or after_close(today))


def test_requires_chosen_strategy(cfg, store):
    fetch, sender = FakeFetch(flat_bars()), Sender()
    assert run(TUE, cfg, store, fetch, sender) == []
    assert len(sender.messages) == 1 and "/choose" in sender.messages[0]
    assert fetch.calls == []
    # At most one reminder per day.
    assert run(TUE, cfg, store, fetch, sender) == []
    assert len(sender.messages) == 1
    assert run(WED, cfg, store, fetch, sender) == []
    assert len(sender.messages) == 2


def test_catch_up_processes_missed_days_in_order(cfg, store, scripted):
    start_paper(store, MON)
    fetch, sender = FakeFetch(flat_bars()), Sender()
    assert run(THU, cfg, store, fetch, sender) == [TUE, WED, THU]
    assert store.get("last_processed_day") == THU.isoformat()
    assert sorted(store.get("equity_history")) == [d.isoformat() for d in (TUE, WED, THU)]
    assert len(fetch.calls) == 1  # one fetch per run
    tickers, start, end = fetch.calls[0]
    assert sorted(tickers) == sorted(TICKERS + [BENCH]) and end == THU
    assert start <= TUE - timedelta(days=450)
    (summary,) = sender.messages
    assert f"Daily summary {THU.isoformat()}" in summary
    assert "Caught up 2 missed day(s)" in summary and TUE.isoformat() in summary


def test_before_close_excludes_today(cfg, store, scripted):
    start_paper(store, MON)
    fetch, sender = FakeFetch(flat_bars()), Sender()
    early = lambda: datetime(2024, 1, 4, 16, 14, tzinfo=IST)
    assert run(THU, cfg, store, fetch, sender, now=early) == [TUE, WED]
    assert store.get("last_processed_day") == WED.isoformat()


def test_idempotent_same_day(cfg, store, scripted):
    start_paper(store, MON)
    scripted.buy_on = {TUE}
    fetch, sender = FakeFetch(flat_bars()), Sender()
    assert run(WED, cfg, store, fetch, sender) == [TUE, WED]
    before = store.get("portfolio")
    assert len(before["positions"]) == 1
    n_msgs, n_fetch = len(sender.messages), len(fetch.calls)
    assert run(WED, cfg, store, fetch, sender) == []
    assert store.get("portfolio") == before
    assert len(sender.messages) == n_msgs and len(fetch.calls) == n_fetch


def test_bad_data_halts_and_alerts(cfg, store, scripted):
    start_paper(store, MON)
    bars = flat_bars()
    bars["B.NS"] = bars["B.NS"].drop(pd.Timestamp(WED))
    fetch, sender = FakeFetch(bars), Sender()
    assert run(THU, cfg, store, fetch, sender) == [TUE]
    assert store.get("last_processed_day") == TUE.isoformat()
    assert sorted(store.get("equity_history")) == [TUE.isoformat()]
    alerts = [m for m in sender.messages if "B.NS" in m]
    assert len(alerts) == 1 and WED.isoformat() in alerts[0]
    assert "Traceback" not in alerts[0]
    # Data fixed later: the next run resumes from Wed.
    fetch2 = FakeFetch(flat_bars())
    assert run(THU, cfg, store, fetch2, sender) == [WED, THU]


def test_missing_ticker_in_fetch_result_is_bad_data(cfg, store, scripted):
    start_paper(store, MON)
    bars = flat_bars()
    del bars["B.NS"]
    sender = Sender()
    assert run(TUE, cfg, store, FakeFetch(bars), sender) == []
    assert store.get("last_processed_day") == MON.isoformat()
    assert len(sender.messages) == 1 and "B.NS: no data" in sender.messages[0]


def test_fetch_failure_alerts_and_does_not_advance(cfg, store, scripted):
    start_paper(store, MON)
    fetch, sender = FakeFetch(flat_bars(), error=ConnectionError("down")), Sender()
    assert run(THU, cfg, store, fetch, sender) == []
    assert store.get("last_processed_day") == MON.isoformat()
    assert len(sender.messages) == 1 and "Nothing was traded" in sender.messages[0]


def test_unexpected_error_logged_alerted_state_unchanged(cfg, store, scripted, monkeypatch):
    start_paper(store, MON)
    store.set("portfolio", Portfolio(12345).to_dict())

    def boom(*a, **k):
        raise RuntimeError("kaboom")
    monkeypatch.setattr(daily, "step_day", boom)
    sender = Sender()
    assert run(THU, cfg, store, FakeFetch(flat_bars()), sender) == []
    assert store.get("last_processed_day") == MON.isoformat()
    assert store.get("portfolio") == Portfolio(12345).to_dict()
    (msg,) = sender.messages
    assert "Something went wrong" in msg and "kaboom" not in msg and "Traceback" not in msg
    log = (cfg.state_path.parent.parent / "logs" / "tradebot.log").read_text()
    assert "kaboom" in log and "Traceback" in log


def test_paused_skips_buys_but_runs_stops(cfg, store, scripted):
    start_paper(store, MON)
    bars = flat_bars()
    a = bars["A.NS"].copy()
    a.loc[pd.Timestamp(THU), "low"] = 80.0  # stop at 90 hit on Thu
    bars["A.NS"] = a
    scripted.buy_on = {TUE, WED}
    fetch, sender = FakeFetch(bars), Sender()
    assert run(WED, cfg, store, fetch, sender) == [TUE, WED]  # bought Wed open
    assert "A.NS" in {p["ticker"] for p in store.get("portfolio")["positions"]}
    store.set("paused", True)
    scripted.buy_on = {WED, THU}
    # Pending from Wed is empty (already held) -> sell via stop on Thu; Thu's BUY fills Fri: skipped.
    assert run(THU, cfg, store, fetch, sender) == [THU]
    p = store.get("portfolio")
    assert p["positions"] == [] and [t["reason"] for t in p["trades"]] == ["stop"]
    assert run(FRI, cfg, store, fetch, sender) == [FRI]
    assert store.get("portfolio")["positions"] == []
    (fri,) = [m for m in sender.messages if m.startswith(f"Daily summary {FRI.isoformat()}")]
    assert "A.NS: paused" in fri
    assert store.get("paused") is True


def test_kill_switch_trips_and_alerts(cfg, store, scripted):
    from dataclasses import replace
    cfg = replace(cfg, max_position_pct=0.5, max_risk_pct=1.0)
    start_paper(store, MON)
    store.set("pending_signals", [asdict(Signal("A.NS", "BUY", 1.0, "go"))])
    bars = flat_bars()
    a = bars["A.NS"].copy()
    for d in (WED, THU, FRI):
        a.loc[pd.Timestamp(d), ["open", "high", "low", "close"]] = 40.0
    bars["A.NS"] = a
    fetch, sender = FakeFetch(bars), Sender()
    assert run(THU, cfg, store, fetch, sender) == [TUE, WED, THU]
    assert store.get("kill_switch") is True
    alerts = [m for m in sender.messages if "/resume" in m]
    assert len(alerts) == 1 and "paused" in alerts[0].lower()
    # Never auto-resumes; no duplicate alert on later days.
    assert run(FRI, cfg, store, fetch, sender) == [FRI]
    assert store.get("kill_switch") is True
    assert len([m for m in sender.messages if "/resume" in m]) == 1


def random_bars(n_warm=40, n_days=30, seed=7, tickers=TICKERS):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2024-01-01", periods=n_warm + n_days)
    out = {}
    for i, t in enumerate(list(tickers) + [BENCH]):
        close = 100 * np.cumprod(1 + rng.normal(0, 0.02, len(idx)))
        opens = close * (1 + rng.normal(0, 0.005, len(idx)))
        lows = np.minimum(opens, close) * (1 - np.abs(rng.normal(0, 0.01, len(idx))))
        out[t] = frame(idx, opens, close, lows)
    return out, [d.date() for d in idx[n_warm:]]


def trade_key(t):
    t = t if isinstance(t, dict) else asdict(t)
    return (t["ticker"], str(t["entry_date"]), round(t["entry_price"], 6),
            str(t["exit_date"]), round(t["exit_price"], 6), t["qty"], t["reason"])


def test_matches_backtest(cfg, store, monkeypatch):
    monkeypatch.setitem(REGISTRY, "momentum", Momentum)
    bars, days = random_bars()
    m, equity, trades = run_backtest(Momentum(), bars, days[0], days[-1], cfg)
    assert len(trades) >= 3 and m.kill_switch_trips == 0  # non-vacuous, no R11 auto-resume
    start_paper(store, (pd.Timestamp(days[0]) - pd.offsets.BDay(1)).date(), name="momentum")
    fetch, sender = FakeFetch(bars), Sender()
    for d in days:
        assert run(d, cfg, store, fetch, sender) == [d]
    p = store.get("portfolio")
    assert [trade_key(t) for t in p["trades"]] == [trade_key(t) for t in trades]
    hist = store.get("equity_history")
    assert [hist[d.isoformat()] for d in days] == pytest.approx(list(equity))


def test_catch_up_same_as_day_by_day(cfg, tmp_path, monkeypatch):
    """Bars in a catch-up include later days; results must equal processing each day alone."""
    monkeypatch.setitem(REGISTRY, "momentum", Momentum)
    bars, days = random_bars(n_days=10, seed=3)
    start = (pd.Timestamp(days[0]) - pd.offsets.BDay(1)).date()
    s1, s2 = Store(tmp_path / "one.sqlite"), Store(tmp_path / "two.sqlite")
    try:
        for s in (s1, s2):
            start_paper(s, start, name="momentum")
        assert run(days[-1], cfg, s1, FakeFetch(bars), Sender()) == days
        for d in days:
            run(d, cfg, s2, FakeFetch(bars), Sender())
        for key in ("portfolio", "pending_signals", "equity_history", "last_processed_day"):
            assert s1.get(key) == s2.get(key), key
    finally:
        s1.close()
        s2.close()


def test_weekly_report_after_last_trading_day_of_week(cfg, store, scripted):
    start_paper(store, MON)
    fetch, sender = FakeFetch(flat_bars()), Sender()
    run(THU, cfg, store, fetch, sender)
    assert not any("Weekly report" in m for m in sender.messages)
    run(FRI, cfg, store, fetch, sender)
    weekly = [m for m in sender.messages if "Weekly report" in m]
    assert len(weekly) == 1
    assert store.get("last_weekly_week") == "2024-W01"
    # Catch-up across a week end also sends it (first run after), once.
    run(date(2024, 1, 15), cfg, store, fetch, sender)  # covers Fri 12th
    assert len([m for m in sender.messages if "Weekly report" in m]) == 2
    assert store.get("last_weekly_week") == "2024-W02"


def test_below_benchmark_label_from_backtest_summary(cfg, store, scripted):
    start_paper(store, MON)
    oos = {"below_benchmark": True}
    store.set("backtest_summary", {"results": {"scripted": {"in_sample": oos, "out_of_sample": oos}},
                                   "ranking": ["scripted"], "generated": "2024-01-01"})
    sender = Sender()
    run(TUE, cfg, store, FakeFetch(flat_bars()), sender)
    assert "BELOW BENCHMARK" in sender.messages[-1]


class FailingSender(Sender):
    """Records every message; raises when the text contains `fail_on`."""
    def __init__(self, fail_on):
        super().__init__()
        self.fail_on = fail_on

    def __call__(self, text, store):
        self.messages.append(text)
        if self.fail_on in text:
            raise RuntimeError("sender broke")
        return True


def test_failure_after_commit_reports_processed_days(cfg, store, scripted):
    start_paper(store, MON)
    sender = FailingSender("Daily summary")
    assert run(FRI, cfg, store, FakeFetch(flat_bars()), sender) == [TUE, WED, THU, FRI]
    assert store.get("last_processed_day") == FRI.isoformat()
    # Weekly report still sent even though the daily summary failed.
    assert any(m.startswith("Weekly report") for m in sender.messages)
    alerts = [m for m in sender.messages if "failed while preparing the report" in m]
    assert len(alerts) == 1 and FRI.isoformat() in alerts[0] and TUE.isoformat() in alerts[0]
    assert not any("Nothing was traded" in m for m in sender.messages)
    assert "sender broke" in (cfg.state_path.parent.parent / "logs" / "tradebot.log").read_text()


def test_weekly_failure_keeps_summary_and_week_unmarked(cfg, store, scripted, monkeypatch):
    start_paper(store, MON)

    def boom(*a, **k):
        raise RuntimeError("metrics broke")
    monkeypatch.setattr(daily, "_paper_metrics", boom)
    sender = Sender()
    assert run(FRI, cfg, store, FakeFetch(flat_bars()), sender) == [TUE, WED, THU, FRI]
    assert any(m.startswith(f"Daily summary {FRI.isoformat()}") for m in sender.messages)
    assert any("failed while preparing the report" in m for m in sender.messages)
    assert store.get("last_weekly_week") is None  # retried on a later run


def test_failure_in_later_commit_returns_committed_days(cfg, store, scripted, monkeypatch):
    start_paper(store, MON)
    real = store.transaction
    count = {"n": 0}

    def flaky():
        count["n"] += 1
        if count["n"] == 3:  # 1 = state snapshot after the download, 2 = TUE, 3 = WED
            raise RuntimeError("disk full")
        return real()
    monkeypatch.setattr(store, "transaction", flaky)
    sender = Sender()
    assert run(THU, cfg, store, FakeFetch(flat_bars()), sender) == [TUE]
    assert store.get("last_processed_day") == TUE.isoformat()
    (msg,) = sender.messages
    assert "failed while preparing the report" in msg and TUE.isoformat() in msg


def test_unknown_strategy_alerts_once_per_day(cfg, store):
    start_paper(store, MON, name="nosuch")
    fetch, sender = FakeFetch(flat_bars()), Sender()
    assert run(TUE, cfg, store, fetch, sender) == []
    assert run(TUE, cfg, store, fetch, sender) == []
    (msg,) = sender.messages
    assert "nosuch" in msg and "/choose" in msg and "Something went wrong" not in msg
    assert fetch.calls == [] and store.get("last_processed_day") == MON.isoformat()


def test_now_passed_to_fetch(cfg, store, scripted):
    start_paper(store, MON)
    fetch = FakeFetch(flat_bars())
    now = after_close(TUE)
    run(TUE, cfg, store, fetch, Sender(), now=now)
    assert fetch.now() == now()


@pytest.mark.parametrize("name", sorted(REGISTRY))
def test_real_strategy_catch_up_same_as_day_by_day(cfg, tmp_path, name):
    """Look-ahead check for the real strategies: a 15-day catch-up (bars include later days)
    must equal 15 single-day runs (bars end on the day)."""
    from dataclasses import replace
    tickers = [f"T{i}.NS" for i in range(8)]
    cfg = replace(cfg, universe=tickers)
    bars, days = random_bars(n_warm=260, n_days=15, seed=11, tickers=tickers)
    start = (pd.Timestamp(days[0]) - pd.offsets.BDay(1)).date()
    s1, s2 = Store(tmp_path / "one.sqlite"), Store(tmp_path / "two.sqlite")
    try:
        for s in (s1, s2):
            start_paper(s, start, name=name)
        assert run(days[-1], cfg, s1, FakeFetch(bars), Sender()) == days
        for d in days:
            assert run(d, cfg, s2, FakeFetch(bars), Sender()) == [d]
        for key in ("portfolio", "pending_signals", "equity_history", "last_processed_day"):
            assert s1.get(key) == s2.get(key), key
        p = s1.get("portfolio")
        assert p["positions"] or p["trades"] or s1.get("pending_signals")  # not vacuous
    finally:
        s1.close()
        s2.close()


def test_weekly_report_text_is_stored_for_report_command(cfg, store, scripted):
    start_paper(store, MON)
    fetch, sender = FakeFetch(flat_bars()), Sender()
    run(THU, cfg, store, fetch, sender)
    assert store.get("last_weekly_report") is None
    run(FRI, cfg, store, fetch, sender)
    weekly = [m for m in sender.messages if "Weekly report" in m]
    assert store.get("last_weekly_report") == weekly[0]


def test_alerts_deduplicated_per_ist_day(cfg, store, scripted):
    start_paper(store, MON)
    fetch, sender = FakeFetch(flat_bars(), error=ConnectionError("down")), Sender()
    run(THU, cfg, store, fetch, sender)
    run(THU, cfg, store, fetch, sender)
    assert len(sender.messages) == 1
    assert list(store.get("last_alerts")) == [THU.isoformat()]
    FRI = THU + timedelta(days=1)
    run(FRI, cfg, store, fetch, sender)
    assert len(sender.messages) == 2
    assert list(store.get("last_alerts")) == [FRI.isoformat()]


# --- C1 / Ruling R22: unlisted market holidays -------------------------------------------

def _closed_on(bars, d):
    return {t: df.drop(pd.Timestamp(d)) for t, df in bars.items()}


def test_unlisted_holiday_is_skipped_and_mentioned(cfg, store, scripted):
    bars = _closed_on(flat_bars(), WED)
    start_paper(store, MON)
    scripted.buy_on = {TUE}  # filled at the next real trading day's open (THU)
    s = Sender()
    assert run(THU, cfg, store, FakeFetch(bars), s) == [TUE, THU]
    assert store.get("last_processed_day") == THU.isoformat()
    assert WED.isoformat() not in store.get("equity_history")
    assert [e["payload"]["day"] for e in store.events("unlisted_holiday")] == [WED.isoformat()]
    assert store.get("portfolio")["positions"][0]["entry_date"] == THU.isoformat()
    assert f"Market was closed on {WED.isoformat()} (holiday not in my list)" in s.messages[-1]
    assert not any(m.startswith("⚠️") for m in s.messages)


def test_unlisted_holiday_does_not_halt_forever(cfg, store, scripted):
    """Reviewer probe: before the fix last_processed_day stayed on TUE for ever."""
    bars = _closed_on(flat_bars(), WED)
    start_paper(store, MON)
    s = Sender()
    assert run(WED, cfg, store, FakeFetch(bars), s) == [TUE]  # no later bar yet: wait
    assert store.get("last_processed_day") == TUE.isoformat()
    for today in (THU, FRI, date(2024, 1, 8), date(2024, 1, 9)):
        run(today, cfg, store, FakeFetch(bars), s)
    assert store.get("last_processed_day") == "2024-01-09"


def test_december_alert_for_missing_next_year_holiday_file(cfg, store):
    s = Sender()
    run(date(2024, 11, 29), cfg, store, FakeFetch({}), s)
    assert not any("nse_2025.txt" in m for m in s.messages)
    run(date(2024, 12, 2), cfg, store, FakeFetch({}), s)
    run(date(2024, 12, 2), cfg, store, FakeFetch({}), s)
    alerts = [m for m in s.messages if "nse_2025.txt" in m]
    assert len(alerts) == 1 and alerts[0].startswith("⚠️") and "holidays" in alerts[0]
    (cfg.holidays_dir / "nse_2025.txt").write_text("2025-01-26\n")
    run(date(2024, 12, 3), cfg, store, FakeFetch({}), s)
    assert len([m for m in s.messages if "nse_2025.txt" in m]) == 1


def test_bad_holiday_file_alert_names_file_and_line(cfg, store, scripted):
    (cfg.holidays_dir / "nse_2024.txt").write_text("# list\n2024-01-26\n2024-1-3x\n")
    start_paper(store, MON)
    s = Sender()
    assert run(TUE, cfg, store, FakeFetch(flat_bars()), s) == []
    (msg,) = s.messages
    assert "nse_2024.txt" in msg and "line 3" in msg and "Something went wrong" not in msg


def test_weekly_report_sent_when_friday_was_unlisted_holiday(cfg, store, scripted):
    bars = _closed_on(flat_bars(), FRI)
    start_paper(store, MON)
    s = Sender()
    run(THU, cfg, store, FakeFetch(bars), s)
    assert not any(m.startswith("Weekly report") for m in s.messages)
    run(date(2024, 1, 8), cfg, store, FakeFetch(bars), s)
    assert len([m for m in s.messages if m.startswith("Weekly report")]) == 1
    assert store.get("last_weekly_week") == "2024-W01"


# --- C2 / Ruling R23: splits, bonus issues, dividends ------------------------------------

def _hold_a(cfg, store, qty=10, price=100.0, stop=90.0):
    p = Portfolio(cfg.capital)
    p.buy("A.NS", qty, price, MON, stop, "scripted", cfg)
    store.set("portfolio", p.to_dict())
    return p.positions["A.NS"]


def test_split_rescales_held_position(cfg, store, scripted):
    start_paper(store, MON)
    before = _hold_a(cfg, store)
    bars = flat_bars(price=50.0)  # history re-adjusted by Yahoo after a 1:2 split
    fetch = FakeFetch(bars)
    fetch.adjust = {"A.NS": 0.5, "B.NS": 0.5}  # B is not held: nothing to rescale
    s = Sender()
    assert run(TUE, cfg, store, fetch, s, now=lambda: datetime(2024, 1, 2, 21, 0, tzinfo=IST)) == [TUE]
    (pos,) = store.get("portfolio")["positions"]
    assert pos["qty"] == 20 and pos["entry_price"] == pytest.approx(before.entry_price * 0.5)
    assert pos["stop"] == pytest.approx(45.0)
    assert store.get("portfolio")["trades"] == []  # no fake stop-out
    (ev,) = store.events("price_adjustment")
    assert ev["payload"]["ticker"] == "A.NS" and ev["payload"]["ratio"] == 0.5
    assert ev["payload"]["qty_before"] == 10 and ev["payload"]["qty_after"] == 20
    assert "A.NS" in s.messages[-1] and "split" in s.messages[-1]


def test_dividend_like_adjustment_scales_prices_only(cfg, store, scripted):
    start_paper(store, MON)
    before = _hold_a(cfg, store)
    fetch = FakeFetch(flat_bars(price=98.0))
    fetch.adjust = {"A.NS": 0.98}
    s = Sender()
    run(TUE, cfg, store, fetch, s, now=lambda: datetime(2024, 1, 2, 21, 0, tzinfo=IST))
    (pos,) = store.get("portfolio")["positions"]
    assert pos["qty"] == 10 and pos["entry_price"] == pytest.approx(before.entry_price * 0.98)
    assert pos["stop"] == pytest.approx(90.0 * 0.98)
    assert "dividend" in s.messages[-1]


def test_adjustment_saved_even_when_day_cannot_be_processed(cfg, store, scripted):
    """The cache is already rebuilt, so the ratio cannot be detected again on a later run."""
    start_paper(store, MON)
    _hold_a(cfg, store)
    bars = flat_bars(price=50.0)
    bars[BENCH] = bars[BENCH].drop(pd.Timestamp(TUE))
    fetch = FakeFetch(bars)
    fetch.adjust = {"A.NS": 0.5}
    assert run(TUE, cfg, store, fetch, Sender(), now=lambda: datetime(2024, 1, 2, 21, 0, tzinfo=IST)) == []
    assert store.get("portfolio")["positions"][0]["qty"] == 20


# --- I1 / Ruling R24: owner commands during a run are never undone ------------------------

class RacingFetch(FakeFetch):
    """Runs a bot command (other connection = other process) while the download is running."""
    def __init__(self, bars, command, cfg, other):
        super().__init__(bars)
        self.command, self.cfg, self.other, self.replies = command, cfg, other, []

    def __call__(self, *a, **k):
        from tradebot import bot
        self.replies.append(bot.handle_command(self.command, 1, self.cfg, self.other, today=TUE))
        return super().__call__(*a, **k)


def test_resume_during_download_is_kept(cfg, store, scripted):
    start_paper(store, MON)
    p = Portfolio(17000.0)
    p.peak_equity = 20000.0
    store.set("portfolio", p.to_dict())
    store.set("equity_history", {MON.isoformat(): 17000.0})
    store.set("kill_switch", True)
    other = Store(cfg.state_path)
    try:
        assert run(TUE, cfg, store, RacingFetch(flat_bars(), "/resume", cfg, other), Sender()) == [TUE]
    finally:
        other.close()
    assert store.get("kill_switch") is False
    assert store.get("portfolio")["peak_equity"] == 17000.0  # the /resume reset survived


def _command_mid_compute(monkeypatch, cfg, other, command):
    from tradebot import bot
    real = daily.step_day
    replies = []

    def step(*a, **k):
        if not replies:
            replies.append(bot.handle_command(command, 1, cfg, other, today=TUE))
        return real(*a, **k)
    monkeypatch.setattr(daily, "step_day", step)
    return replies


def test_resume_between_read_and_save_is_preserved(cfg, store, scripted, monkeypatch):
    start_paper(store, MON)
    store.set("kill_switch", True)
    other = Store(cfg.state_path)
    try:
        replies = _command_mid_compute(monkeypatch, cfg, other, "/resume")
        s = Sender()
        assert run(WED, cfg, store, FakeFetch(flat_bars()), s) == []
    finally:
        other.close()
    assert replies and store.get("kill_switch") is False
    assert store.get("last_processed_day") == MON.isoformat()  # rolled back: redone next run
    assert store.events("state_changed_during_run")
    assert s.messages == []  # quiet
    monkeypatch.undo()
    monkeypatch.setitem(REGISTRY, "scripted", Scripted)
    assert run(WED, cfg, store, FakeFetch(flat_bars()), Sender()) == [TUE, WED]
    assert store.get("kill_switch") is False


def test_choose_between_read_and_save_is_preserved(cfg, store, scripted, monkeypatch):
    monkeypatch.setitem(REGISTRY, "momentum", Momentum)
    start_paper(store, date(2023, 12, 29))
    store.set("backtest_summary", {"results": {"scripted": {}, "momentum": {}}})
    other = Store(cfg.state_path)
    try:
        _command_mid_compute(monkeypatch, cfg, other, "/choose momentum")
        assert run(WED, cfg, store, FakeFetch(flat_bars()), Sender()) == []
    finally:
        other.close()
    assert store.get("chosen_strategy") == "momentum"
    assert store.get("last_processed_day") == MON.isoformat()  # what /choose set
    assert store.get("equity_history") == {}


def test_commands_bump_state_rev(cfg, store):
    from tradebot import bot
    store.set("backtest_summary", {"results": {"scripted": {}}})
    revs = []
    for c in ("/status", "/pause", "/choose scripted", "/choose scripted", "/resume", "/help"):
        bot.handle_command(c, 1, cfg, store, today=TUE)
        revs.append(store.get("state_rev", 0))
    assert revs == [0, 1, 2, 2, 3, 3]
