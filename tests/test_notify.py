import httpx
import pytest

from tradebot import notify
from tradebot.store import Store


@pytest.fixture
def store(tmp_path):
    s = Store(tmp_path / "s.db")
    yield s
    s.close()


def _env(monkeypatch):
    monkeypatch.setenv("TELEGRAM_TOKEN", "TOK")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "42")


def _client(handler):
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_send_queues_on_network_error(store, monkeypatch):
    _env(monkeypatch)

    def boom(request):
        raise httpx.ConnectError("no network")

    assert notify.send("hello", store, _client(boom)) is False
    assert len(store.outbox_pending()) == 1
    err = store.events("telegram_error")
    assert len(err) == 1 and "Traceback" not in str(err[0]["payload"])


def test_send_flushes_backlog_in_order(store, monkeypatch):
    _env(monkeypatch)
    store.outbox_add("first")
    seen = []

    def ok(request):
        seen.append((str(request.url), request.content.decode()))
        return httpx.Response(200, json={"ok": True})

    assert notify.send("second", store, _client(ok)) is True
    assert store.outbox_pending() == []
    assert "botTOK/sendMessage" in seen[0][0]
    assert "first" in seen[0][1] and "second" in seen[1][1]
    assert "chat_id=42" in seen[0][1] or '"chat_id": "42"' in seen[0][1] or "42" in seen[0][1]


def test_flush_stops_at_first_failure(store, monkeypatch):
    _env(monkeypatch)
    calls = []

    def bad(request):
        calls.append(1)
        return httpx.Response(500, text="oops")

    assert notify.send("a", store, _client(bad)) is False
    notify.send("b", store, _client(bad))
    assert len(store.outbox_pending()) == 2
    assert len(calls) == 2  # one attempt per send, not per queued message


def test_not_configured_queues_and_logs(store, monkeypatch):
    monkeypatch.delenv("TELEGRAM_TOKEN", raising=False)
    monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)
    assert notify.send("hi", store, _client(lambda r: httpx.Response(200))) is False
    assert len(store.outbox_pending()) == 1
    assert len(store.events("telegram_not_configured")) == 1


def test_long_message_split(store, monkeypatch):
    _env(monkeypatch)
    text = "\n".join(["x" * 100] * 100)
    sent = []

    def ok(request):
        sent.append(request.content.decode())
        return httpx.Response(200, json={"ok": True})

    assert notify.send(text, store, _client(ok)) is True
    assert len(sent) >= 3


def test_load_env_does_not_override(tmp_path, monkeypatch):
    p = tmp_path / ".env"
    p.write_text("# c\nTELEGRAM_TOKEN=fromfile\nTELEGRAM_CHAT_ID = \"99\"\n\nbad line\n")
    monkeypatch.setenv("TELEGRAM_TOKEN", "existing")
    monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)
    notify.load_env(p)
    import os
    assert os.environ["TELEGRAM_TOKEN"] == "existing"
    assert os.environ["TELEGRAM_CHAT_ID"] == "99"
    notify.load_env(tmp_path / "missing.env")  # no error
