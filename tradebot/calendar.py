from __future__ import annotations

from datetime import date, timedelta

from tradebot.config import Config


def _holidays(cfg: Config) -> set[date]:
    out: set[date] = set()
    if not cfg.holidays_dir.is_dir():
        return out
    for f in sorted(cfg.holidays_dir.glob("*.txt")):
        for line in f.read_text().splitlines():
            line = line.split("#", 1)[0].strip()
            if line:
                out.add(date.fromisoformat(line))
    return out


def is_trading_day(d: date, cfg: Config) -> bool:
    return d.weekday() < 5 and d not in _holidays(cfg)


def trading_days_between(start: date, end: date, cfg: Config) -> list[date]:
    hol = _holidays(cfg)
    out, d = [], start
    while d <= end:
        if d.weekday() < 5 and d not in hol:
            out.append(d)
        d += timedelta(days=1)
    return out


def previous_trading_day(d: date, cfg: Config) -> date:
    hol = _holidays(cfg)
    d -= timedelta(days=1)
    while d.weekday() >= 5 or d in hol:
        d -= timedelta(days=1)
    return d
