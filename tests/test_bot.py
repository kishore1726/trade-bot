from datetime import date

import httpx
import pytest

from tradebot import bot
from tradebot.config import Config, CostConfig, TaxConfig
from tradebot.portfolio import Portfolio
from tradebot.store import Store

CHAT = 42
TODAY = date(2024, 1, 3)  # Wednesday


@pytest.fixture
def cfg(tmp_path):
    (tmp_path / "holidays").mkdir()
    return Config(
        capital=20000, max_position_pct=0.25, max_positions=4, max_risk_pct=0.02,
        kill_switch_dd_pct=0.15, slippage_pct=0.001, costs=CostConfig(), tax=TaxConfig(),
        universe=["A.NS"], benchmark="NB.NS", data_dir=tmp_path / "data",
        state_path=tmp_path / "state" / "t.sqlite", holidays_dir=tmp_path / "holidays",
    )


@pytest.fixture
def store(cfg):
    s = Store(cfg.state_path)
    yield s
    s.close()


@pytest.fixture(autouse=True)
def env(monkeypatch):
    monkeypatch.setenv("TELEGRAM_TOKEN", "tok")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", str(CHAT))


def cmd(text, cfg, store, chat=CHAT):
    return bot.handle_command(text, chat, cfg, store, today=TODAY)


def metrics_dict(**kw):
    base = dict(trades=3, win_rate=0.5, avg_win=100.0, avg_loss=-50.0, max_drawdown=0.1,
                post_tax_return=0.05, benchmark_post_tax_return=0.03, kill_switch_trips=0,
                below_benchmark=False, expectancy=10.0, total_return=0.06,
                cagr=0.06, benchmark_return=0.04, benchmark_cagr=0.04, total_costs=50.0)
    base.update(kw)
    return base


def summary(*names):
    return {"results": {n: {"in_sample": metrics_dict(), "out_of_sample": metrics_dict()}
                        for n in names}, "ranking": list(names), "generated": "2024-01-01"}


def held(cfg, store):
    p = Portfolio(cfg.capital)
    p.buy("A.NS", 10, 100.0, date(2024, 1, 2), 90.0, "s", cfg)
    store.set("portfolio", p.to_dict())
    return p


def test_status_shows_positions(cfg, store):
    held(cfg, store)
    store.set("chosen_strategy", "momentum")
    store.set("equity_history", {"2024-01-02": 20100.0})
    store.set("last_processed_day", "2024-01-02")
    out = cmd("/status", cfg, store)
    for s in ["momentum", "A.NS", "10", "90", "20,100", "2024-01-02"]:
        assert s in out
    assert "paused" in out.lower() and "kill" in out.lower()


def test_status_empty(cfg, store):
    out = cmd("/status", cfg, store)
    assert "none yet" in out


def test_pause_resume_flags(cfg, store):
    out = cmd("/pause", cfg, store)
    assert store.get("paused") is True and store.get("kill_switch") is None
    assert "stop" in out.lower()
    assert store.events("pause")
    store.set("kill_switch", True)
    store.set("chosen_strategy", "momentum")
    out = cmd("/resume", cfg, store)
    assert store.get("paused") is False and store.get("kill_switch") is False
    assert "tomorrow" in out
    assert store.events("resume")


def test_resume_resets_peak(cfg, store):
    p = held(cfg, store)
    p.peak_equity = 99999.0
    store.set("portfolio", p.to_dict())
    store.set("equity_history", {"2024-01-01": 19000.0, "2024-01-02": 18000.0})
    cmd("/resume", cfg, store)
    assert Portfolio.from_dict(store.get("portfolio")).peak_equity == 18000.0


def test_resume_peak_fallback_without_history(cfg, store):
    p = held(cfg, store)
    p.peak_equity = 99999.0
    store.set("portfolio", p.to_dict())
    cmd("/resume", cfg, store)
    q = Portfolio.from_dict(store.get("portfolio"))
    assert q.peak_equity == pytest.approx(q.cash + 10 * p.positions["A.NS"].entry_price)


def test_choose_rejects_unknown(cfg, store):
    store.set("backtest_summary", summary("momentum"))
    out = cmd("/choose nonsense", cfg, store)
    assert "momentum" in out
    assert store.get("chosen_strategy") is None
    assert "/choose" in cmd("/choose", cfg, store)


def test_choose_without_backtest(cfg, store):
    assert "backtest" in cmd("/choose momentum", cfg, store).lower()
    assert store.get("chosen_strategy") is None


def test_choose_initialises_portfolio(cfg, store):
    store.set("backtest_summary", summary("momentum", "other"))
    out = cmd("/choose momentum", cfg, store)
    assert store.get("chosen_strategy") == "momentum"
    assert store.get("portfolio")["cash"] == 20000
    assert store.get("last_processed_day") == "2024-01-02"
    assert "momentum" in out


def test_choose_keeps_existing_portfolio(cfg, store):
    store.set("backtest_summary", summary("momentum"))
    p = Portfolio(5000)
    store.set("portfolio", p.to_dict())
    cmd("/choose momentum", cfg, store)
    assert store.get("portfolio")["cash"] == 5000


def test_choose_refused_with_open_positions(cfg, store):
    store.set("backtest_summary", summary("momentum", "other"))
    store.set("chosen_strategy", "momentum")
    held(cfg, store)
    out = cmd("/choose other", cfg, store)
    assert store.get("chosen_strategy") == "momentum"
    assert "open" in out.lower()
    same = cmd("/choose momentum", cfg, store)
    assert "already" in same.lower()
    assert store.get("last_processed_day") is None


def test_report_priority(cfg, store):
    assert cmd("/report", cfg, store).startswith("No report yet")
    store.set("backtest_summary", summary("momentum"))
    assert "Backtest results" in cmd("/report", cfg, store)
    store.set("last_weekly_report", "Weekly report X")
    assert cmd("/report", cfg, store) == "Weekly report X"


def test_help_and_unknown(cfg, store):
    assert "/status" in cmd("/help", cfg, store)
    assert "/status" in cmd("hello", cfg, store)
    assert "cash" in cmd("/status@MyBot extra", cfg, store).lower()
    assert "/pause" in cmd("/bogus", cfg, store)


class FakeHttp:
    def __init__(self, updates=None, fail=False):
        self.updates, self.fail = updates or [], fail
        self.sent, self.gets = [], []

    def get(self, url, params=None, timeout=None):
        self.gets.append((url, params))
        if self.fail:
            raise httpx.ConnectError("boom tok")
        return httpx.Response(200, json={"ok": True, "result": self.updates},
                              request=httpx.Request("GET", url))

    def post(self, url, data=None, timeout=None):
        self.sent.append(data)
        return httpx.Response(200, json={"ok": True}, request=httpx.Request("POST", url))


def upd(i, chat, text):
    return {"update_id": i, "message": {"chat": {"id": chat}, "text": text}}


def test_ignores_other_chat(cfg, store):
    http = FakeHttp([upd(5, 999, "/pause"), upd(6, CHAT, "/pause")])
    n = bot.poll_once(cfg, store, http)
    assert n == 1
    assert len(http.sent) == 1 and http.sent[0]["chat_id"] == str(CHAT)
    assert store.events("ignored_chat")
    assert store.get("tg_offset") == 7
    bot.poll_once(cfg, store, http)
    assert http.gets[-1][1]["offset"] == 7 and http.gets[-1][1]["timeout"] == 30


def test_poll_survives_handler_error(cfg, store, monkeypatch):
    monkeypatch.setattr(bot, "handle_command", lambda *a, **k: 1 / 0)
    http = FakeHttp([upd(1, CHAT, "/status")])
    assert bot.poll_once(cfg, store, http) == 1
    assert "went wrong" in http.sent[0]["text"]


def test_run_bot_network_error_sleeps_and_continues(cfg, store):
    http = FakeHttp(fail=True)
    sleeps = []

    def sleep(s):
        sleeps.append(s)
        if len(sleeps) == 2:
            raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        bot.run_bot(cfg, store, http=http, sleep=sleep)
    assert sleeps == [10, 10]
    assert store.events("bot_error")
    assert "tok" not in str(store.events("bot_error"))


def test_missing_credentials(cfg, store, monkeypatch):
    monkeypatch.delenv("TELEGRAM_TOKEN")
    monkeypatch.setattr(bot, "load_env", lambda *a, **k: None)
    with pytest.raises(SystemExit) as e:
        bot.run_bot(cfg, store, http=FakeHttp())
    assert "TELEGRAM" in str(e.value)


def test_resume_with_nothing_chosen_creates_nothing(cfg, store):
    store.set("paused", True)
    out = cmd("/resume", cfg, store)
    assert "Nothing to resume" in out
    assert store.get("portfolio") is None
    assert not store.events("resume")


def test_resume_preserves_latest_portfolio(cfg, store):
    p = Portfolio(7777)
    p.trades.clear()
    store.set("portfolio", p.to_dict())
    store.set("chosen_strategy", "momentum")
    cmd("/resume", cfg, store)
    q = Portfolio.from_dict(store.get("portfolio"))
    assert q.cash == 7777 and q.peak_equity == 7777


def test_commands_read_inside_transaction(cfg, store):
    depth = {"n": 0, "reads": []}
    orig_tx, orig_get = store.transaction, store.get

    import contextlib

    @contextlib.contextmanager
    def tx():
        with orig_tx():
            depth["n"] += 1
            try:
                yield
            finally:
                depth["n"] -= 1

    def get(key, default=None):
        depth["reads"].append((key, depth["n"]))
        return orig_get(key, default)

    store.transaction, store.get = tx, get
    store.set("backtest_summary", summary("momentum"))
    store.set("chosen_strategy", "momentum")
    store.set("portfolio", Portfolio(1000).to_dict())
    cmd("/resume", cfg, store)
    cmd("/choose momentum", cfg, store)
    assert depth["reads"] and all(n == 1 for k, n in depth["reads"]
                                  if k in ("portfolio", "equity_history", "chosen_strategy"))


def test_switch_strategy_starts_fresh_record(cfg, store):
    store.set("backtest_summary", summary("momentum", "other"))
    store.set("chosen_strategy", "momentum")
    p = Portfolio(15000)
    p.peak_equity = 20000
    from tradebot.portfolio import Trade
    p.trades.append(Trade("A.NS", 5, date(2024, 1, 1), 100.0, date(2024, 1, 2), 90.0, 10.0, -60.0, "stop"))
    store.set("portfolio", p.to_dict())
    store.set("pending_signals", [{"x": 1}])
    store.set("equity_history", {"2024-01-02": 15000.0})
    out = cmd("/choose other", cfg, store)
    assert "fresh" in out
    assert store.get("chosen_strategy") == "other"
    assert store.get("pending_signals") == []
    assert store.get("equity_history") == {}
    q = Portfolio.from_dict(store.get("portfolio"))
    assert q.trades == [] and q.cash == 15000 and q.peak_equity == 15000
    ev = store.events("strategy_switch")[0]["payload"]
    assert ev["from"] == "momentum" and ev["to"] == "other" and len(ev["trades"]) == 1
    assert ev["trades"][0]["ticker"] == "A.NS"
    assert store.get("last_processed_day") == "2024-01-02"
