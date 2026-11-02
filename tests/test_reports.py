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



def test_backtest_comparison_has_expectancy_costs_annualised_and_period():
    """I6: spec section 8 metrics in the comparison, plus the out-of-sample dates."""
    m = _m(expectancy=120.0, expectancy_pct=0.0123, total_costs=1234.0, cagr=0.105,
           benchmark_cagr=0.081)
    res = {"trend": {"in_sample": _m(), "out_of_sample": m}}
    seg = {"in_sample": {"start": "2016-01-01", "end": "2024-11-01"},
           "out_of_sample": {"start": "2024-11-04", "end": "2026-11-02"}}
    txt = reports.backtest_comparison(res, ["trend"], seg)
    for s in ["₹120", "1.2%", "₹1,234", "10.5%", "8.1%", "2024-11-04 to 2026-11-02"]:
        assert s in txt, s
    assert "2024-11-04" not in reports.backtest_comparison(res, ["trend"])  # old summaries


def test_daily_summary_buy_estimates_and_change():
    """I7: estimated shares and amount per BUY, orders that will be skipped, change in value."""
    sigs = [Signal("TCS.NS", "BUY", 3200.0, "breakout"), Signal("ITC.NS", "BUY", 400.0, "breakout"),
            Signal("LT.NS", "BUY", 3000.0, "breakout")]
    est = {"TCS.NS": (2, 6800.0), "ITC.NS": ("will", "no free slot (already holding 4 positions)"),
           "LT.NS": ("may", "not enough room or cash")}
    txt = reports.daily_summary(date(2026, 11, 2), _pf(), 20000.0, sigs, [], [], "trend", False,
                                estimates=est, previous_equity=19000.0)
    assert "BUY TCS.NS" in txt and "about 2 shares, ₹6,800 at today's close" in txt
    assert "ITC.NS" in txt and "will be skipped: no free slot (already holding 4 positions)" in txt
    assert "may be skipped: not enough room or cash" in txt
    assert "Change since the previous day: +₹1,000 (+5.3%)" in txt
    first = reports.daily_summary(date(2026, 11, 2), _pf(), 20000.0, [], [], [], "trend", False)
    assert "Change since" not in first



def test_zero_closed_trades_show_na():
    """M1: no closed trades -> "n/a", not 0.0% / Rs 0."""
    z = _m(trades=0, win_rate=0.0, avg_win=0.0, avg_loss=0.0, expectancy=0.0, expectancy_pct=0.0)
    w = reports.weekly_report(z, "trend", _m(trades=0))
    assert "Average win: n/a, average loss: n/a" in w
    assert "Average profit per trade: n/a" in w
    assert "win rate paper n/a / backtest n/a" in w and "profit per trade paper n/a / backtest n/a" in w
    c = reports.backtest_comparison({"trend": {"in_sample": z, "out_of_sample": z}}, ["trend"])
    assert "win rate n/a" in c and "Average win n/a, average loss n/a" in c
    assert "Average profit per trade: n/a" in c


def test_comparison_discloses_simulated_resume():
    """M3: Ruling R11 is disclosed."""
    c = reports.backtest_comparison({"trend": {"in_sample": _m(), "out_of_sample": _m()}}, ["trend"])
    assert "20 trading days" in c and "/resume" in c
