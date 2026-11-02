"""Plain-English message text for Telegram. Pure functions: no I/O."""
from __future__ import annotations

from datetime import date

from tradebot.metrics import Metrics
from tradebot.portfolio import Portfolio, Trade
from tradebot.strategies.base import Signal

BELOW = "BELOW BENCHMARK"
CHOOSE_LINE = "Reply /choose <name> to start paper trading."


def _rs(x: float) -> str:
    return f"{'-' if round(x) < 0 else ''}₹{abs(x):,.0f}"


def _pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def alert(text: str) -> str:
    return f"⚠️ {text}"


def daily_summary(d: date, portfolio: Portfolio, equity: float, signals: list[Signal],
                  exits: list[Trade], caught_up_days: list[date], strategy_name: str,
                  below_benchmark: bool, skipped: list[tuple[str, str]] = ()) -> str:
    lines = [f"Daily summary {d.isoformat()} - strategy: {strategy_name}"]
    if below_benchmark:
        lines.append(f"{BELOW}: this strategy has been lagging a simple Nifty buy-and-hold.")
    if caught_up_days:
        n = len(caught_up_days)
        lines.append(f"Caught up {n} missed day(s): " + ", ".join(x.isoformat() for x in caught_up_days))
    lines += ["", f"Cash: {_rs(portfolio.cash)}", f"Total value: {_rs(equity)}",
              f"Open positions: {len(portfolio.positions)}"]
    for p in portfolio.positions.values():
        lines.append(f"  {p.ticker}: {p.qty} shares, bought at {_rs(p.entry_price)}, stop {_rs(p.stop)}")
    if exits:
        lines += ["", "Closed today:"]
        for t in exits:
            why = "stop-loss hit" if t.reason == "stop" else "strategy sell signal"
            lines.append(f"  {t.ticker}: {t.qty} shares sold at {_rs(t.exit_price)} ({why}), "
                         f"profit/loss {_rs(t.pnl)} after costs")
    lines += ["", "Orders for tomorrow's open:"]
    if signals:
        for s in signals:
            if s.action == "BUY":
                stop = f", stop {_rs(s.stop)}" if s.stop is not None else ""
                lines.append(f"  BUY {s.ticker}{stop} - {s.reason}")
            else:
                lines.append(f"  SELL {s.ticker} - {s.reason}")
    else:
        lines.append("  No new orders.")
    if skipped:
        lines += ["", "Skipped orders:"]
        lines += [f"  {t}: {why}" for t, why in skipped]
    return "\n".join(lines)


def _vs_bench(m: Metrics) -> str:
    return f"return {_pct(m.total_return)} vs Nifty {_pct(m.benchmark_return)}"


def weekly_report(metrics: Metrics, strategy_name: str, backtest_oos: Metrics | None = None) -> str:
    m = metrics
    lines = [f"Weekly report - strategy: {strategy_name}"]
    if m.below_benchmark:
        lines.append(f"{BELOW}: lagging a simple Nifty buy-and-hold.")
    lines += [
        f"Closed trades: {m.trades}",
        f"Win rate: {_pct(m.win_rate)} (win rate alone can mislead: a few big wins can beat many small losses)",
        f"Average win: {_rs(m.avg_win)}, average loss: {_rs(m.avg_loss)}",
        f"Average profit per trade: {_rs(m.expectancy)}",
        f"Worst fall from a peak: {_pct(m.max_drawdown)}",
        f"Return: {_pct(m.total_return)} vs Nifty buy-and-hold {_pct(m.benchmark_return)}",
        f"After tax: {_pct(m.post_tax_return)} vs Nifty {_pct(m.benchmark_post_tax_return)}",
        f"Total trading costs: {_rs(m.total_costs)}",
    ]
    if backtest_oos is not None:
        b = backtest_oos
        lines.append(
            f"Paper vs backtest: win rate {_pct(m.win_rate)} vs {_pct(b.win_rate)}, "
            f"profit per trade {_rs(m.expectancy)} vs {_rs(b.expectancy)}, "
            f"worst fall {_pct(m.max_drawdown)} vs {_pct(b.max_drawdown)}")
    return "\n".join(lines)


def backtest_comparison(results: dict[str, dict[str, Metrics]], ranking: list[str]) -> str:
    lines = ["Backtest results (best first). Out-of-sample = recent period the strategy was not tuned on.", ""]
    for i, name in enumerate(ranking, 1):
        oos, ins = results[name]["out_of_sample"], results[name]["in_sample"]
        head = f"{i}. {name}"
        if oos.below_benchmark:
            head += f" - {BELOW}"
        lines += [
            head,
            f"  Trades: {oos.trades}, win rate {_pct(oos.win_rate)}",
            f"  Average win {_rs(oos.avg_win)}, average loss {_rs(oos.avg_loss)}",
            f"  Worst fall from a peak: {_pct(oos.max_drawdown)}",
            f"  {_vs_bench(oos).capitalize()}",
            f"  After tax: {_pct(oos.post_tax_return)} vs Nifty {_pct(oos.benchmark_post_tax_return)}",
            f"  Safety switch tripped: {oos.kill_switch_trips} time(s)",
            f"  Earlier (in-sample) period: {_vs_bench(ins)}",
            "",
        ]
    lines += ["Note: history uses today's Nifty 50 list, so past results look better than reality.", "",
              CHOOSE_LINE]
    return "\n".join(lines)
