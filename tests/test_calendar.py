from dataclasses import replace
from datetime import date
from tradebot.calendar import is_trading_day, trading_days_between, previous_trading_day
from tradebot.config import load_config


def test_weekend_not_trading_day():
    assert is_trading_day(date(2026, 11, 7), load_config()) is False


def test_holiday_not_trading_day(tmp_path):
    (tmp_path / "x.txt").write_text("# c\n2026-11-04\n")
    cfg = replace(load_config(), holidays_dir=tmp_path)
    assert is_trading_day(date(2026, 11, 4), cfg) is False
    assert is_trading_day(date(2026, 11, 5), cfg) is True


def test_trading_days_between_skips_weekend():
    cfg = load_config()
    assert trading_days_between(date(2026, 11, 6), date(2026, 11, 9), cfg) == [date(2026, 11, 6), date(2026, 11, 9)]


def test_previous_trading_day_skips_holiday_and_weekend():
    cfg = load_config()
    assert previous_trading_day(date(2026, 11, 11), cfg) == date(2026, 11, 9)


def test_bad_holiday_line_names_file_and_line(tmp_path):
    import pytest
    from tradebot.calendar import HolidayFileError
    (tmp_path / "nse_2027.txt").write_text("# c\n2027-01-26\n26/01/2027\n")
    cfg = replace(load_config(), holidays_dir=tmp_path)
    with pytest.raises(HolidayFileError) as e:
        is_trading_day(date(2027, 1, 27), cfg)
    msg = str(e.value)
    assert "nse_2027.txt" in msg and "line 3" in msg and "26/01/2027" in msg
