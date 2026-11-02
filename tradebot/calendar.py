from __future__ import annotations

from datetime import date, timedelta

from tradebot.config import Config


class HolidayFileError(ValueError):
    """A holiday file line is not a date. The message is plain English for the owner."""


def _holidays(cfg: Config) -> set[date]:
    out: set[date] = set()
    if not cfg.holidays_dir.is_dir():
        return out
    for f in sorted(cfg.holidays_dir.glob("*.txt")):
        for n, raw in enumerate(f.read_text().splitlines(), 1):
            line = raw.split("#", 1)[0].strip()
            if not line:
                continue
            try:
                out.add(date.fromisoformat(line))
            except ValueError:
                raise HolidayFileError(
                    f"The holiday file holidays/{f.name} has a problem on line {n}: \"{line}\" is "
                    "not a date written as YYYY-MM-DD (for example 2027-01-26). Please fix or "
                    "delete that line.") from None
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
