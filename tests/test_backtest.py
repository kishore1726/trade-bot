from dataclasses import replace
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from tradebot.backtest import DayState, rank, run_backtest, step_day, walk_forward
from tradebot.config import Config, CostConfig, TaxConfig
from tradebot.costs import trade_cost
from tradebot.metrics import Metrics
from tradebot.portfolio import Portfolio
from tradebot.strategies import Signal


@pytest.fixture
def cfg():
    return Config(
        capital=20000, max_position_pct=0.25, max_positions=4, max_risk_pct=0.02,
        kill_switch_dd_pct=0.15, slippage_pct=0.001, costs=CostConfig(), tax=TaxConfig(),
        universe=["A.NS", "B.NS"], benchmark="NB.NS", data_dir=Path("d"), state_path=Path("s"),
        holidays_dir=Path("h"),
    )


DAYS = pd.bdate_range("2024-01-01", periods=8)


def D(i):
    return DAYS[i].date()


def mk(opens, closes=None, lows=None):
    closes = closes or opens
    lows = lows or [min(o, c) for o, c in zip(opens, closes)]
    highs = [max(o, c) for o, c in zip(opens, closes)]
    n = len(opens)
    return pd.DataFrame({"open": opens, "high": highs, "low": lows, "close": closes,
                         "volume": [1000.0] * n}, index=DAYS[:n], dtype=float)


class Stub:
    """Emits scheduled signals at the close of given days; records what it was shown."""
    name = "stub"
    lookback_days = 1

    def __init__(self, schedule):
        self.schedule = schedule
        self.seen = []

    def signals(self, bars, d, positions):
        self.seen.append(d)
        return list(self.schedule.get(d, []))


def bars_with(a, b=None):
    out = {"A.NS": a, "NB.NS": mk([100.0] * len(a))}
    if b is not None:
        out["B.NS"] = b
    return out


def test_backtest_fills_next_open(cfg):
    a = mk([100, 101, 102, 103, 104, 105], closes=[100, 101, 102, 103, 104, 105])
    strat = Stub({D(1): [Signal("A.NS", "BUY", 90.0, "go")],
                  D(3): [Signal("A.NS", "SELL", None, "exit")]})
    m, equity, trades = run_backtest(strat, bars_with(a), D(0), D(5), cfg)
    assert len(trades) == 1
    t = trades[0]
    assert t.entry_date == D(2) and t.entry_price == pytest.approx(102 * 1.001)
    assert t.exit_date == D(4) and t.exit_price == pytest.approx(104 * 0.999)
    assert list(equity.index.date) == [D(i) for i in range(6)]
    assert strat.seen == [D(i) for i in range(6)]
    assert m.trades == 1


def test_backtest_costs_reduce_equity(cfg):
    cfg = replace(cfg, slippage_pct=0.0)
    a = mk([100.0] * 5)
    strat = Stub({D(0): [Signal("A.NS", "BUY", 90.0, "go")],
                  D(2): [Signal("A.NS", "SELL", None, "exit")]})
    m, equity, trades = run_backtest(strat, bars_with(a), D(0), D(4), cfg)
    (t,) = trades
    expected = trade_cost("BUY", t.qty * 100.0, cfg.costs) + trade_cost("SELL", t.qty * 100.0, cfg.costs)
    assert t.costs == pytest.approx(expected)
    assert equity.iloc[-1] == pytest.approx(cfg.capital - expected)
    assert m.total_costs == pytest.approx(expected)
    assert m.total_return < 0


def test_backtest_kill_switch_blocks_new_buys(cfg):
    cfg = replace(cfg, max_position_pct=1.0, max_risk_pct=1.0)
    # A bought on D1 near full equity, then gaps down 30% (stop far below): drawdown > 15%.
    a = mk([100, 100, 70, 70, 70, 70])
    b = mk([50.0] * 6)
    buy_b = [Signal("B.NS", "BUY", 40.0, "go")]
    strat = Stub({D(0): [Signal("A.NS", "BUY", 1.0, "go")],
                  D(2): buy_b, D(3): buy_b, D(4): buy_b})
    m, equity, trades = run_backtest(strat, bars_with(a, b), D(0), D(5), cfg)
    assert equity.iloc[2] < cfg.capital * 0.85
    assert all(t.ticker != "B.NS" for t in trades)
    assert m.trades == 0  # A still held, never sold


def _state(cfg):
    return DayState(Portfolio(cfg.capital), [])


def test_step_day_reports_kill_and_skips(cfg):
    cfg = replace(cfg, max_position_pct=1.0, max_risk_pct=1.0)
    bars = bars_with(mk([100, 100, 70, 70]), mk([50.0] * 4))
    s = _state(cfg)
    strat = Stub({})
    s.pending = [Signal("A.NS", "BUY", 1.0, "go")]
    r1 = step_day(s, D(1), bars, strat, cfg)
    assert r1.buys == ["A.NS"] and not r1.kill_tripped_today
    r2 = step_day(s, D(2), bars, strat, cfg)
    assert r2.kill_tripped_today and s.kill_switch
    s.pending = [Signal("B.NS", "BUY", 40.0, "go")]
    r3 = step_day(s, D(3), bars, strat, cfg)
    assert r3.buys == [] and r3.skipped == [("B.NS", "kill switch active")]
    assert not r3.kill_tripped_today


def test_step_day_paused_gap_and_held_skips(cfg):
    bars = bars_with(mk([100, 100, 100]), mk([50, 39, 50]))
    s = _state(cfg)
    strat = Stub({D(0): [Signal("A.NS", "BUY", 95.0, "x")]})
    s.pending = [Signal("B.NS", "BUY", 40.0, "gap"), Signal("A.NS", "BUY", 95.0, "ok")]
    r = step_day(s, D(1), bars, strat, cfg)
    assert r.skipped == [("B.NS", "gapped below stop")]
    assert r.buys == ["A.NS"]
    assert r.new_signals == [] and s.pending == []
    s.pending = [Signal("A.NS", "BUY", 95.0, "again")]
    r = step_day(s, D(2), bars, strat, cfg)
    assert r.skipped == [("A.NS", "already held")]
    s.paused = True
    s.pending = [Signal("B.NS", "BUY", 40.0, "p")]
    r = step_day(s, D(2), bars, strat, cfg)
    assert r.skipped == [("B.NS", "paused")]


def test_step_day_sizes_on_open_equity(cfg):
    # A held (bought at 100), opens at 200 on D1: sizing equity = cash + qty*200.
    bars = bars_with(mk([100, 200], closes=[100, 200]), mk([50.0, 50.0]))
    s = _state(cfg)
    s.portfolio.buy("A.NS", 10, 100.0, D(0), 90.0, "stub", cfg)
    s.pending = [Signal("B.NS", "BUY", 49.0, "go")]
    cash_before = s.portfolio.cash
    r = step_day(s, D(1), bars, Stub({}), cfg)
    eq = cash_before + 10 * 200.0
    entry = 50 * 1.001
    expected = int(min(eq * 0.25, eq * 0.02 / (entry - 49.0) * entry) // entry)
    assert r.buys == ["B.NS"]
    assert s.portfolio.positions["B.NS"].qty == expected


def test_stop_exit_reported(cfg):
    bars = bars_with(mk([100, 100, 100], lows=[100, 100, 80]))
    s = _state(cfg)
    s.portfolio.buy("A.NS", 10, 100.0, D(0), 90.0, "stub", cfg)
    r = step_day(s, D(2), bars, Stub({}), cfg)
    assert [t.reason for t in r.stop_exits] == ["stop"]
    assert r.stop_exits[0].exit_price == pytest.approx(90 * 0.999)


def test_walk_forward_splits_and_warms_up(cfg, monkeypatch):
    days = pd.bdate_range("2020-01-01", "2023-12-29")
    flat = pd.DataFrame({"open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0,
                         "volume": 1000.0}, index=days)
    bars = {"A.NS": flat, "NB.NS": flat}
    made = []

    def factory():
        made.append(Stub({}))
        return made[-1]

    import tradebot.backtest as bt
    monkeypatch.setitem(bt.REGISTRY, "rec", factory)
    res = walk_forward(["rec"], bars, cfg, holdout_years=2)
    assert set(res["rec"]) == {"in_sample", "out_of_sample"}
    assert len(made) == 2
    runs = sorted((m.seen for m in made), key=lambda v: v[0])
    assert runs[0][0] == date(2020, 1, 1) and runs[0][-1] < date(2022, 1, 1)
    assert runs[1][0] >= date(2021, 12, 30) and runs[1][-1] == date(2023, 12, 29)
    assert runs[0][-1] < runs[1][0]


def _m(dd, exp, ret, bench=0.0):
    return Metrics(trades=1, win_rate=0.5, avg_win=1, avg_loss=-1, expectancy=exp,
                   max_drawdown=dd, total_return=ret, cagr=0, benchmark_return=bench,
                   benchmark_cagr=0, total_costs=0, post_tax_return=ret,
                   benchmark_post_tax_return=bench, below_benchmark=ret < bench)


def test_rank_order(cfg):
    ins = _m(0.1, 0, 0)
    results = {
        "deep": {"in_sample": ins, "out_of_sample": _m(0.30, 500.0, 2.0)},  # breaches 15% dd
        "lowexp": {"in_sample": ins, "out_of_sample": _m(0.10, 50.0, 0.9)},
        "beatbench": {"in_sample": ins, "out_of_sample": _m(0.12, 100.0, 0.5, 0.1)},
        "tie": {"in_sample": ins, "out_of_sample": _m(0.15, 100.0, 0.5, 0.3)},
    }
    assert rank(results, cfg) == ["beatbench", "tie", "lowexp", "deep"]



DAYS40 = pd.bdate_range("2024-01-01", periods=40)


def flat_frame(opens):
    return pd.DataFrame({"open": opens, "high": opens, "low": opens, "close": opens,
                         "volume": 1000.0}, index=DAYS40, dtype=float)


def resume_case():
    """A bought day 1 (fill), gaps 100->70 on day 2: kill switch trips. A sold day 4.
    B is signalled every day from day 2; B gaps 50->30 on day 26 (stop 40) -> exits at open."""
    n = len(DAYS40)
    bars = {"A.NS": flat_frame([100.0, 100.0] + [70.0] * (n - 2)),
            "B.NS": flat_frame([50.0] * 26 + [30.0] * (n - 26)),
            "NB.NS": flat_frame([100.0] * n)}
    d = [x.date() for x in DAYS40]
    sched = {d[0]: [Signal("A.NS", "BUY", 1.0, "go")]}
    for i in range(2, n):
        sched[d[i]] = [Signal("B.NS", "BUY", 40.0, "go")]
    sched[d[3]] = sched[d[3]] + [Signal("A.NS", "SELL", None, "exit")]
    return bars, sched, d


def test_backtest_resumes_after_20_days_and_counts_trips(cfg):
    cfg = replace(cfg, max_position_pct=1.0, max_risk_pct=1.0)
    bars, sched, d = resume_case()
    m, equity, trades = run_backtest(Stub(sched), bars, d[0], d[-1], cfg)
    # trip on day 2; BUY fills blocked on days 3..22; first fill allowed on day 23
    assert [(t.ticker, t.entry_date, t.exit_date) for t in trades] == [
        ("A.NS", d[1], d[4]), ("B.NS", d[23], d[26])]
    # B's 40% gap loss after the resume trips again, measured from the reset peak
    assert m.kill_switch_trips == 2


def test_step_day_never_auto_resumes(cfg):
    # The paper run (step_day alone) keeps the kill switch until the owner sends /resume.
    cfg = replace(cfg, max_position_pct=1.0, max_risk_pct=1.0)
    bars, sched, d = resume_case()
    s, strat = DayState(Portfolio(cfg.capital), []), Stub(sched)
    results = [step_day(s, day, bars, strat, cfg) for day in d]
    assert [i for i, r in enumerate(results) if r.kill_tripped_today] == [2]
    assert all(not r.buys for r in results[2:]) and s.kill_switch
