import pytest

from tradebot.store import Store


def test_get_set_roundtrip(tmp_path):
    s = Store(tmp_path / "sub" / "state.db")
    assert s.get("portfolio") is None
    assert s.get("portfolio", {"x": 1}) == {"x": 1}
    s.set("portfolio", {"cash": 20000.5, "pos": [1, 2], "d": "2026-11-02"})
    assert s.get("portfolio") == {"cash": 20000.5, "pos": [1, 2], "d": "2026-11-02"}
    s.set("paused", True)
    s.set("paused", False)
    assert s.get("paused") is False
    s.close()


def test_persists_across_instances(tmp_path):
    p = tmp_path / "state.db"
    a = Store(p)
    a.set("chosen_strategy", "trend")
    a.log_event("trade", {"ticker": "TCS"})
    a.outbox_add("hello")
    a.close()
    b = Store(p)
    assert b.get("chosen_strategy") == "trend"
    assert [t for _, t in b.outbox_pending()] == ["hello"]
    b.close()


def test_two_connections_see_each_other(tmp_path):
    p = tmp_path / "state.db"
    a, b = Store(p), Store(p)
    a.set("k", 1)
    assert b.get("k") == 1
    a.close()
    b.close()


def test_transaction_commits_on_success(tmp_path):
    s = Store(tmp_path / "state.db")
    with s.transaction():
        s.set("a", 1)
        s.log_event("x", {})
        s.outbox_add("m")
    assert s.get("a") == 1
    assert len(s.outbox_pending()) == 1
    s.close()


def test_transaction_rollback_on_error(tmp_path):
    p = tmp_path / "state.db"
    s = Store(p)
    s.set("a", 1)
    with pytest.raises(RuntimeError):
        with s.transaction():
            s.set("a", 2)
            s.set("b", 3)
            s.log_event("trade", {"n": 1})
            s.outbox_add("lost")
            raise RuntimeError("boom")
    assert s.get("a") == 1
    assert s.get("b") is None
    assert s.outbox_pending() == []
    assert s.events() == []
    # store still usable afterwards
    s.set("c", 4)
    c = Store(p)
    assert c.get("c") == 4
    c.close()
    s.close()


def test_outbox_order_and_mark_sent(tmp_path):
    s = Store(tmp_path / "state.db")
    s.outbox_add("one")
    s.outbox_add("two")
    s.outbox_add("three")
    pending = s.outbox_pending()
    assert [t for _, t in pending] == ["one", "two", "three"]
    s.outbox_mark_sent(pending[1][0])
    assert [t for _, t in s.outbox_pending()] == ["one", "three"]
    s.close()


def test_log_event_records_kind_payload_and_ist_timestamp(tmp_path):
    s = Store(tmp_path / "state.db")
    s.log_event("approval", {"who": "owner"})
    (ev,) = s.events()
    assert ev["kind"] == "approval" and ev["payload"] == {"who": "owner"}
    assert ev["ts"].endswith("+05:30")
    s.close()
