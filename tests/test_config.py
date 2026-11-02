from pathlib import Path
import pytest
from tradebot.config import load_config


def test_load_config_defaults():
    cfg = load_config()
    assert cfg.capital == 20000
    assert cfg.max_positions == 4
    assert "NIFTYBEES.NS" in cfg.universe


def test_load_config_rejects_bad_pct(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("capital: 20000\nmax_position_pct: 1.5\nmax_positions: 4\nmax_risk_pct: 0.02\n"
                 "kill_switch_dd_pct: 0.15\nslippage_pct: 0.001\nbenchmark: X.NS\nuniverse: [X.NS]\n")
    with pytest.raises(ValueError):
        load_config(p)


def test_paths_relative_to_config_dir(tmp_path):
    cfg = load_config(Path("config.yaml").resolve())
    assert cfg.holidays_dir.name == "holidays"
    assert cfg.holidays_dir.is_absolute()
