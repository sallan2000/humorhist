"""Tests for humorhist.buffer -- Phase 3.4 buffer-health monitor.

No network: a StubTelegram and a StubClient stand in for the real transports.
Covers depth computation, the silent/nudge/escalate levels, auto-draft
triggering, and the silent-when-healthy Telegram behaviour.
"""

from __future__ import annotations

from pathlib import Path

import humorhist.buffer as buf
import humorhist.db as db
from humorhist.llm import StubClient


def _fresh_db(tmp_path: Path):
    path = tmp_path / "test.sqlite"
    conn = db.connect(str(path))
    db.migrate(conn)
    return conn


def _seed_pool(conn, pid="pool-x"):
    conn.execute(
        "INSERT OR IGNORE INTO pool (id, title, status) VALUES (?, 'Topic', 'drafted')",
        (pid,),
    )


def _make_draft(conn, draft_id, status="pending", queued=False):
    _seed_pool(conn, f"pool-{draft_id}")
    conn.execute(
        "INSERT INTO drafts (id, pool_id, brief_json, angles_json, status, created_at) "
        "VALUES (?, ?, '{}', '{}', ?, '2026-01-01T00:00:00+00:00')",
        (draft_id, f"pool-{draft_id}", status),
    )
    if queued:
        conn.execute(
            "INSERT INTO queue (draft_id, scheduled_for, published) VALUES (?, NULL, 0)",
            (draft_id,),
        )
    conn.commit()


def test_buffer_depth_counts_unpublished_queue(tmp_path):
    conn = _fresh_db(tmp_path)
    _make_draft(conn, "d1", status="approved", queued=True)
    _make_draft(conn, "d2", status="approved", queued=True)
    # a published row must NOT count toward the buffer
    conn.execute("UPDATE queue SET published = 1 WHERE draft_id='d1'")
    conn.commit()
    assert buf.buffer_depth(conn) == 1


def test_buffer_health_levels(tmp_path):
    conn = _fresh_db(tmp_path)
    # healthy: >= 7 queued
    for i in range(7):
        _make_draft(conn, f"d{i}", status="approved", queued=True)
    assert buf.buffer_health(conn)["level"] == "silent"

    # drain to 5 -> nudge
    for i in range(2):
        conn.execute("DELETE FROM queue WHERE draft_id=?", (f"d{i}",))
    conn.commit()
    assert buf.buffer_health(conn)["level"] == "nudge"

    # drain to 2 -> escalate
    for i in range(3):
        conn.execute("DELETE FROM queue WHERE draft_id=?", (f"d{i + 2}",))
    conn.commit()
    assert buf.buffer_health(conn)["level"] == "escalate"


def test_buffer_auto_draft_flag_when_pending_low(tmp_path):
    conn = _fresh_db(tmp_path)
    _make_draft(conn, "d1", status="approved", queued=True)  # 1 day buffer, but pending high
    # many pending drafts -> auto_draft should be False (pending floor not hit)
    for i in range(6):
        _make_draft(conn, f"p{i}", status="pending")
    assert buf.buffer_health(conn)["auto_draft"] is False

    # no pending drafts -> auto_draft should be True
    conn.execute("DELETE FROM drafts WHERE status='pending'")
    conn.commit()
    assert buf.buffer_health(conn)["auto_draft"] is True


def test_run_buffer_check_silent_does_not_notify(tmp_path):
    conn = _fresh_db(tmp_path)
    for i in range(7):
        _make_draft(conn, f"d{i}", status="approved", queued=True)
    stub = _StubTelegram()
    result = buf.run_buffer_check(conn, client=None, auto_draft=False, chat_id="chat", telegram=stub)
    assert result["level"] == "silent"
    assert result["notified"] is False
    assert stub.sent == []


def test_run_buffer_check_nudges_when_low(tmp_path):
    conn = _fresh_db(tmp_path)
    _make_draft(conn, "d1", status="approved", queued=True)  # 1 day -> escalate
    stub = _StubTelegram()
    result = buf.run_buffer_check(conn, client=None, auto_draft=False, chat_id="chat", telegram=stub)
    assert result["level"] == "escalate"
    assert result["notified"] is True
    assert any("BUFFER" in m for m in stub.sent)


def test_run_buffer_check_auto_drafts_when_pending_low(tmp_path, monkeypatch):
    import humorhist.drafting as drafting

    conn = _fresh_db(tmp_path)
    # 1 day of buffer + zero pending -> should auto-draft
    _make_draft(conn, "d1", status="approved", queued=True)
    # seed a draftable pool candidate (status 'new', scored)
    conn.execute("INSERT OR IGNORE INTO pool (id, title, status, funny_score) VALUES ('cand', 'Candidate', 'new', 9.0)")
    conn.commit()

    captured = {}

    def fake_draft_candidates(conn2, client, count=3, min_score=7.0, rng=None, http_client=None):
        captured["count"] = count
        # simulate drafting one candidate
        conn2.execute(
            "INSERT INTO drafts (id, pool_id, brief_json, angles_json, status, created_at) "
            "VALUES ('nd1', 'cand', '{}', '{}', 'pending', '2026-01-01T00:00:00+00:00')"
        )
        conn2.commit()
        return {"selected": 1, "drafted": 1, "failed": 0, "draft_ids": ["nd1"], "failures": []}

    monkeypatch.setattr(drafting, "draft_candidates", fake_draft_candidates)
    stub = _StubTelegram()
    result = buf.run_buffer_check(conn, client=StubClient([{}]), auto_draft=True, chat_id="chat", telegram=stub)
    assert result["drafted"] == 1
    assert captured.get("count") == buf.AUTO_DRAFT_COUNT


class _StubTelegram:
    """Minimal Telegram stub recording sent messages."""

    def __init__(self):
        self.sent: list[str] = []

    def send_message(self, chat_id, text, reply_markup=None):
        self.sent.append(text)
        return {"message_id": len(self.sent), "text": text}
