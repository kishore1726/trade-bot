import plistlib
from pathlib import Path

import pytest

from tradebot import cli


@pytest.fixture
def calls(monkeypatch):
    log = []
    monkeypatch.setattr(cli, "_setup", lambda: ("CFG", "STORE"))
    monkeypatch.setattr(cli, "cmd_backtest", lambda cfg, store, start: log.append(("backtest", start)) or 0)
    monkeypatch.setattr(cli, "cmd_run_daily", lambda cfg, store: log.append(("run-daily",)) or 0)
    monkeypatch.setattr(cli, "cmd_bot", lambda cfg, store: log.append(("bot",)) or 0)
    monkeypatch.setattr(cli, "cmd_test_telegram", lambda cfg, store: log.append(("test-telegram",)) or 0)
    monkeypatch.setattr(cli, "install_schedule", lambda: log.append(("install",)) or 0)
    monkeypatch.setattr(cli, "uninstall_schedule", lambda: log.append(("uninstall",)) or 0)
    return log


def test_cli_parses_commands(calls):
    for argv in (["backtest"], ["backtest", "--start", "2018-05-01"], ["run-daily"], ["bot"],
                 ["test-telegram"], ["install-schedule"], ["uninstall-schedule"]):
        assert cli.main(argv) == 0
    from datetime import date
    assert calls == [("backtest", date(2016, 1, 1)), ("backtest", date(2018, 5, 1)), ("run-daily",),
                     ("bot",), ("test-telegram",), ("install",), ("uninstall",)]


def test_unexpected_error_is_one_line_and_logged(monkeypatch, tmp_path, capsys, calls):
    monkeypatch.setattr(cli, "REPO", tmp_path)
    monkeypatch.setattr(cli, "cmd_bot", lambda cfg, store: 1 / 0)
    assert cli.main(["bot"]) == 1
    out = capsys.readouterr()
    assert "Traceback" not in out.out + out.err
    assert "tradebot.log" in out.err
    assert "ZeroDivisionError" in (tmp_path / "logs" / "tradebot.log").read_text()


class FakeRunner:
    def __init__(self):
        self.calls = []

    def __call__(self, cmd, **kw):
        self.calls.append(cmd)


def test_install_schedule_writes_absolute_paths(tmp_path):
    target, repo, runner = tmp_path / "LA", tmp_path / "repo", FakeRunner()
    repo.mkdir()
    assert cli.install_schedule(target, runner, python="/v/bin/python", repo=repo) == 0
    daily = plistlib.loads((target / "com.tradebot.daily.plist").read_bytes())
    bot = plistlib.loads((target / "com.tradebot.bot.plist").read_bytes())
    assert daily["Label"] == "com.tradebot.daily" and bot["Label"] == "com.tradebot.bot"
    assert daily["ProgramArguments"] == ["/v/bin/python", "-m", "tradebot.cli", "run-daily"]
    assert bot["ProgramArguments"] == ["/v/bin/python", "-m", "tradebot.cli", "bot"]
    for p in (daily, bot):
        assert p["WorkingDirectory"] == str(repo)
        assert Path(p["StandardOutPath"]).parent == repo / "logs"
        assert Path(p["StandardErrorPath"]).parent == repo / "logs"
        assert p["RunAtLoad"] is True
    assert daily["StartInterval"] == 3600
    assert daily["StartCalendarInterval"] == [{"Weekday": d, "Hour": 16, "Minute": 30} for d in range(1, 6)]
    assert bot["KeepAlive"] is True
    loads = [c for c in runner.calls if c[:2] == ["launchctl", "load"]]
    assert len(loads) == 2 and all(str(target) in c[-1] for c in loads)
    assert (repo / "logs").is_dir()


def test_uninstall_unloads_and_removes(tmp_path):
    target, repo, runner = tmp_path / "LA", tmp_path / "repo", FakeRunner()
    repo.mkdir()
    cli.install_schedule(target, runner, python="/v/bin/python", repo=repo)
    runner.calls.clear()
    assert cli.uninstall_schedule(target, runner) == 0
    assert [c[:2] for c in runner.calls] == [["launchctl", "unload"]] * 2
    assert list(target.glob("*.plist")) == []
    assert cli.uninstall_schedule(target, runner) == 0  # nothing left: still fine


def test_committed_templates_match_generator():
    for name in ("daily", "bot"):
        want = cli.build_plist(name, cli.TEMPLATE_PYTHON, Path(cli.TEMPLATE_REPO))
        got = plistlib.loads((cli.REPO / "launchd" / f"com.tradebot.{name}.plist").read_bytes())
        assert got == want


def test_test_telegram_messages(monkeypatch, capsys):
    sent = []
    monkeypatch.setattr(cli, "send", lambda text, store: sent.append(text) or False)
    assert cli.cmd_test_telegram(None, None) == 0
    assert sent == ["Hello from your trading bot ✅"]
    assert ".env" in capsys.readouterr().out
    monkeypatch.setattr(cli, "send", lambda text, store: True)
    cli.cmd_test_telegram(None, None)
    out = capsys.readouterr().out
    assert "Delivered" in out and "NOT" not in out


def test_backtest_fetch_error_is_plain(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(cli, "REPO", tmp_path)
    def boom(*a, **k):
        raise RuntimeError("no network")
    monkeypatch.setattr(cli, "fetch_bars", boom)
    from types import SimpleNamespace
    cfg = SimpleNamespace(universe=["A"], benchmark="B")
    from datetime import date
    assert cli.cmd_backtest(cfg, None, date(2016, 1, 1)) == 1
    err = capsys.readouterr().err
    assert "download" in err and "Traceback" not in err
    assert "RuntimeError" in (tmp_path / "logs" / "tradebot.log").read_text()


def test_install_schedule_reports_launchctl_failure(tmp_path, monkeypatch, capsys):
    from types import SimpleNamespace
    monkeypatch.setattr(cli, "REPO", tmp_path)

    def runner(cmd, **kw):
        return SimpleNamespace(returncode=5 if cmd[1] == "load" else 3, stderr="Load failed: 5")
    repo = tmp_path / "repo"
    repo.mkdir()
    assert cli.install_schedule(tmp_path / "LA", runner, python="/v/bin/python", repo=repo) == 1
    cap = capsys.readouterr()
    assert "installed" not in cap.out and "Could not start" in cap.err
    assert "Load failed" in (tmp_path / "logs" / "tradebot.log").read_text()
