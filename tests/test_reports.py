from datetime import date

from tradebot import reports
from tradebot.metrics import Metrics
from tradebot.portfolio import Portfolio, Position, Trade
from tradebot.strategies.base import Signal


def _m(**kw):
    base = dict(trades=30, win_rate=0.4, avg_win=900.0, avg_loss=-400.0, expectancy=120.0,
                max_drawdown=0.12, total_return=0.25, cagr=0.1, benchmark_return=0.2,
                benchmark_cagr=0.08, total_costs=500.0, post_tax_return=0.2,
                benchmark_post_tax_return=0.17, below_benchmark=False, kill_switch_trips=1,
                expectancy_pct=0.01)
    base.update(kw)
    return Metrics(**base)


def _pf():
    pf = Portfolio.__new__(Portfolio)
    pf.cash = 12345.0
    pf.positions = {"INFY.NS": Position("INFY.NS", 5, 1500.0, date(2026, 10, 30), 1400.0, "trend")}
    return pf


def test_daily_summary_lists_signal_with_stop_and_reason():
    sigs = [Signal("TCS.NS", "BUY", 3200.0, "20-day high breakout"),
            Signal("INFY.NS", "SELL", None, "trend broke")]
    t = Trade("ITC.NS", 10, date(2026, 10, 1), 400.0, date(2026, 11, 2), 380.0, 30.0, -230.0, "stop")
    txt = reports.daily_summary(date(2026, 11, 2), _pf(), 20000.0, sigs, [t],
                                [date(2026, 10, 30)], "trend", False,
                                skipped=[(date(2026, 11, 2), "WIPRO.NS", "gapped below stop")])
    for s in ["TCS.NS", "3,200", "20-day high breakout", "trend broke", "₹12,345", "₹20,000",
              "Caught up 1 missed day", "2026-10-30", "ITC.NS", "-₹230", "gapped below stop",
              "Open positions: 1"]:
        assert s in txt, s
    assert "BELOW BENCHMARK" not in txt


def test_daily_summary_quiet_day_and_below_benchmark():
    txt = reports.daily_summary(date(2026, 11, 2), _pf(), 20000.0, [], [], [], "trend", True)
    assert "BELOW BENCHMARK" in txt and "Caught up" not in txt
    assert "No new orders" in txt


def test_reports_label_below_benchmark():
    m = _m(below_benchmark=True)
    assert "BELOW BENCHMARK" in reports.weekly_report(m, "trend")
    res = {"trend": {"in_sample": m, "out_of_sample": m}, "meanrev": {"in_sample": _m(), "out_of_sample": _m()}}
    txt = reports.backtest_comparison(res, ["meanrev", "trend"])
    assert txt.count("BELOW BENCHMARK") == 1
    assert txt.rstrip().endswith("Reply /choose <name> to start paper trading.")
    assert "today's Nifty 50 list" in txt
    assert txt.index("meanrev") < txt.index("trend")


def test_weekly_report_has_win_rate_avg_win_loss_drawdown():
    txt = reports.weekly_report(_m(), "trend")
    for s in ["40.0%", "₹900", "-₹400", "12.0%", "can mislead"]:
        assert s in txt, s
    assert "Paper vs backtest" not in txt
    txt2 = reports.weekly_report(_m(), "trend", _m(win_rate=0.5, expectancy=200.0, max_drawdown=0.1))
    assert "Paper vs backtest" in txt2 and "50.0%" in txt2


def test_alert_prefix():
    assert reports.alert("data is stale").startswith("⚠️ ")


def test_unknown_exit_reason_shown_raw():
    t = Trade("ITC.NS", 1, date(2026, 10, 1), 400.0, date(2026, 11, 2), 380.0, 3.0, -23.0, "time_exit")
    txt = reports.daily_summary(date(2026, 11, 2), _pf(), 1.0, [], [t], [], "trend", False)
    assert "(time_exit)" in txt and "strategy sell signal" not in txt
    t.reason = "signal"
    assert "strategy sell signal" in reports.daily_summary(date(2026, 11, 2), _pf(), 1.0, [], [t], [], "trend", False)


def test_weekly_zero_trades_and_comparison_caveat():
    assert "No closed trades yet" in reports.weekly_report(_m(trades=0, win_rate=0.0), "trend")
    assert "Win rate: 0.0%" not in reports.weekly_report(_m(trades=0, win_rate=0.0), "trend")
    res = {"trend": {"in_sample": _m(), "out_of_sample": _m()}}
    assert "can mislead" in reports.backtest_comparison(res, ["trend"])
    w = reports.weekly_report(_m(), "trend", _m(win_rate=0.5))
    assert "paper 40.0% / backtest 50.0%" in w
