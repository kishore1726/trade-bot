"""Command line: `tradebot <command>`. Every command ends with a plain message, never a traceback."""
from __future__ import annotations

import argparse
import plistlib
import subprocess
import sys
import traceback
import warnings
from dataclasses import asdict
from datetime import date, datetime
from pathlib import Path

from tradebot.backtest import rank, walk_forward
from tradebot.bot import run_bot
from tradebot.config import load_config
from tradebot.daily import run_daily
from tradebot.data import fetch_bars
from tradebot.notify import load_env, send
from tradebot.reports import backtest_comparison
from tradebot.store import IST, Store
from tradebot.strategies import REGISTRY

REPO = Path(__file__).resolve().parent.parent
LABELS = {"daily": "com.tradebot.daily", "bot": "com.tradebot.bot"}
COMMAND_ARG = {"daily": "run-daily", "bot": "bot"}
TEMPLATE_PYTHON = "/PATH/TO/trading-bot/.venv/bin/python"
TEMPLATE_REPO = "/PATH/TO/trading-bot"
HELLO = "Hello from your trading bot ✅"


def _log_traceback(what: str) -> Path:
    path = REPO / "logs" / "tradebot.log"
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as f:
            f.write(f"--- {datetime.now(IST).isoformat(timespec='seconds')} {what}\n")
            f.write(traceback.format_exc())
    except OSError:
        pass
    return path


def _setup():
    load_env(REPO / ".env")
    cfg = load_config(REPO / "config.yaml")
    cfg.state_path.parent.mkdir(parents=True, exist_ok=True)
    return cfg, Store(cfg.state_path)


def _today() -> date:
    return datetime.now(IST).date()


def cmd_backtest(cfg, store, start: date) -> int:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # yfinance prints pandas deprecation noise
            bars = fetch_bars(sorted(set(cfg.universe) | {cfg.benchmark}), start, _today(), cfg)
    except Exception:
        _log_traceback("backtest: market data download failed")
        print("Could not download market data. Check your internet connection and try again. "
              "Details are in logs/tradebot.log.", file=sys.stderr)
        return 1
    results = walk_forward(sorted(REGISTRY), bars, cfg)
    ranking = rank(results, cfg)
    store.set("backtest_summary", {
        "results": {n: {seg: asdict(m) for seg, m in segs.items()} for n, segs in results.items()},
        "ranking": ranking,
        "generated": _today().isoformat(),
    })
    text = backtest_comparison(results, ranking)
    print(text)
    send(text, store)
    return 0


def cmd_run_daily(cfg, store) -> int:
    run_daily(_today(), cfg, store)
    return 0


def cmd_bot(cfg, store) -> int:
    run_bot(cfg, store)
    return 0


def cmd_test_telegram(cfg, store) -> int:
    if send(HELLO, store):
        print("Delivered. Check Telegram - you should see the message.")
    else:
        print("The message was queued but NOT delivered. Check that .env has the right "
              "TELEGRAM_TOKEN and TELEGRAM_CHAT_ID, and that you are online. Queued messages "
              "are sent automatically once it works.")
    return 0


def build_plist(which: str, python: str, repo: Path) -> dict:
    plist = {
        "Label": LABELS[which],
        "ProgramArguments": [python, "-m", "tradebot.cli", COMMAND_ARG[which]],
        "WorkingDirectory": str(repo),
        "StandardOutPath": str(repo / "logs" / f"{which}.out.log"),
        "StandardErrorPath": str(repo / "logs" / f"{which}.err.log"),
        "RunAtLoad": True,
    }
    if which == "daily":
        plist["StartCalendarInterval"] = [{"Weekday": d, "Hour": 16, "Minute": 30} for d in range(1, 6)]
        plist["StartInterval"] = 3600  # catch-up after sleep; repeat runs are harmless
    else:
        plist["KeepAlive"] = True
    return plist


def _default_runner(cmd, **kw):
    return subprocess.run(cmd, check=False, **kw)


def _default_target() -> Path:
    return Path.home() / "Library" / "LaunchAgents"


def install_schedule(target: Path | None = None, runner=_default_runner,
                     python: str | None = None, repo: Path | None = None) -> int:
    target, repo, python = Path(target or _default_target()), Path(repo or REPO), python or sys.executable
    target.mkdir(parents=True, exist_ok=True)
    (repo / "logs").mkdir(parents=True, exist_ok=True)
    for which in LABELS:
        path = target / f"{LABELS[which]}.plist"
        path.write_bytes(plistlib.dumps(build_plist(which, python, repo)))
        runner(["launchctl", "unload", str(path)])  # harmless if not loaded yet (re-install)
        runner(["launchctl", "load", str(path)])
    print("Schedule installed. The bot now runs by itself: a daily run at 16:30 on weekdays "
          "(plus catch-up when your Mac wakes) and the Telegram command bot.")
    print("To stop everything later: tradebot uninstall-schedule")
    return 0


def uninstall_schedule(target: Path | None = None, runner=_default_runner) -> int:
    target = Path(target or _default_target())
    for which in LABELS:
        path = target / f"{LABELS[which]}.plist"
        if path.exists():
            runner(["launchctl", "unload", str(path)])
            path.unlink()
    print("Schedule removed. Nothing will run by itself any more.")
    return 0


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="tradebot", description="End-of-day NSE paper-trading bot.")
    sub = p.add_subparsers(dest="command", required=True)
    b = sub.add_parser("backtest", help="test both strategies on past data and send the comparison")
    b.add_argument("--start", type=date.fromisoformat, default=date(2016, 1, 1),
                   help="first date of history, YYYY-MM-DD (default 2016-01-01)")
    sub.add_parser("run-daily", help="run the end-of-day paper-trading step now")
    sub.add_parser("bot", help="run the Telegram command bot")
    sub.add_parser("test-telegram", help="send a test message to your Telegram")
    sub.add_parser("install-schedule", help="make everything run automatically on this Mac")
    sub.add_parser("uninstall-schedule", help="stop and remove the automatic runs")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "install-schedule":
            return install_schedule()
        if args.command == "uninstall-schedule":
            return uninstall_schedule()
        cfg, store = _setup()
        if args.command == "backtest":
            return cmd_backtest(cfg, store, args.start)
        handler = {"run-daily": cmd_run_daily, "bot": cmd_bot, "test-telegram": cmd_test_telegram}
        return handler[args.command](cfg, store)
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        log = _log_traceback(f"command {args.command} failed")
        print(f"Something went wrong ({type(exc).__name__}). Details are in {log}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
