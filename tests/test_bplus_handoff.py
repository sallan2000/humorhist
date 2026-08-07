"""Tests for the B+ handoff: approving a draft generates editable post copy.

Drives the real CLI review loop (cmd_review) and the Telegram approve callback
with simulated network (StubClient for the LLM, StubTelegram for the bot) to
confirm that an approve triggers fill_post_copy and writes post_copy onto the
queue row -- and that a missing LLM key degrades gracefully (approve still
succeeds, copy is simply absent).
"""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

import pytest

import humorhist.db as db
import humorhist.telegram as tg
from humorhist.llm import StubClient

BRIEF = {
    "verified_facts": ["France and Mexico went to war over a pastry shop"],
    "misconceptions": ["Nobody actually ate the pastry"],
    "sources": [],
}
ANGLES = {
    "angles": [{"angle_name": "Bureaucratic revenge"}],
    "suggested_hook": "A pastry shop was the casus belli",
}


def _fresh_db(tmp_path: Path):
    path = tmp_path / "test.sqlite"
    conn = db.connect(str(path))
    db.migrate(conn)
    return conn


def _seed_pending(conn, draft_id="d1", pool_id="pool-x"):
    conn.execute(
        "INSERT OR IGNORE INTO pool (id, title, year, status) VALUES (?, 'Pastry War', 1838, 'drafted')",
        (pool_id,),
    )
    conn.execute(
        """INSERT INTO drafts (id, pool_id, brief_json, angles_json, status, created_at)
           VALUES (?, ?, ?, ?, 'pending', '2026-01-01T00:00:00+00:00')""",
        (draft_id, pool_id, json.dumps(BRIEF), json.dumps(ANGLES)),
    )
    conn.commit()


def test_cli_approve_generates_post_copy(tmp_path, monkeypatch, capsys):
    from humorhist.cli import cmd_review
    from humorhist.copywriter import fill_post_copy

    real_fill = fill_post_copy

    def fake_fill(conn, client, draft_id=None, limit=None):
        return real_fill(
            conn,
            StubClient([{"post": "France invaded Mexico over a pastry shop."}]),
            draft_id=draft_id,
        )

    monkeypatch.setattr("humorhist.copywriter.fill_post_copy", fake_fill)
    import humorhist.llm as llm

    monkeypatch.setattr(llm, "default_client", lambda: StubClient([{"post": "x"}]))

    conn = _fresh_db(tmp_path)
    _seed_pending(conn)
    conn.close()

    monkeypatch.setattr("sys.stdin", io.StringIO("a\n\n\n"))
    args = argparse.Namespace(db=str(tmp_path / "test.sqlite"))
    rc = cmd_review(args)
    assert rc == 0

    conn = _fresh_db(tmp_path)
    row = conn.execute("SELECT post_copy FROM queue WHERE draft_id='d1'").fetchone()
    conn.close()
    assert row["post_copy"] == "France invaded Mexico over a pastry shop."


def test_cli_approve_skips_copy_when_no_client(tmp_path, monkeypatch, capsys):
    from humorhist.cli import cmd_review
    import humorhist.llm as llm

    # default_client() with no key raises LLMError; the loop must catch it and
    # still approve + enqueue, just without copy.
    monkeypatch.setattr(llm, "default_client", llm.NousClient)
    monkeypatch.delenv("HUMORHIST_LLM_API_KEY", raising=False)

    conn = _fresh_db(tmp_path)
    _seed_pending(conn)
    conn.close()

    monkeypatch.setattr("sys.stdin", io.StringIO("a\n\n\n"))
    args = argparse.Namespace(db=str(tmp_path / "test.sqlite"))
    rc = cmd_review(args)
    assert rc == 0

    conn = _fresh_db(tmp_path)
    row = conn.execute("SELECT post_copy FROM queue WHERE draft_id='d1'").fetchone()
    status = conn.execute("SELECT status FROM drafts WHERE id='d1'").fetchone()
    conn.close()
    assert status["status"] == "approved"
    assert row["post_copy"] is None


def test_telegram_approve_generates_post_copy(tmp_path, monkeypatch):
    from humorhist.copywriter import fill_post_copy
    import humorhist.llm as llm

    real_fill = fill_post_copy

    def fake_fill(conn, client, draft_id=None, limit=None):
        return real_fill(
            conn,
            StubClient([{"post": "A pastry shop started a war. Honestly."}]),
            draft_id=draft_id,
        )

    monkeypatch.setattr("humorhist.copywriter.fill_post_copy", fake_fill)
    monkeypatch.setattr(llm, "default_client", lambda: StubClient([{"post": "x"}]))

    conn = _fresh_db(tmp_path)
    _seed_pending(conn)

    stub = tg.StubTelegram()
    upd = {
        "update_id": 1,
        "callback_query": {
            "id": "cb1",
            "data": "approve:d1",
            "message": {"message_id": 10},
        },
    }
    res = tg.handle_callback(conn, stub, "chat", upd)
    assert res is not None
    assert res["decision"] == "approve"

    # Tap only approves + prompts for the joke; copy is generated once the
    # editor_line reply arrives (so the human joke can steer the copy).
    row = conn.execute("SELECT post_copy FROM queue WHERE draft_id='d1'").fetchone()
    assert row["post_copy"] is None

    # reply with the editor_line -> triggers fill_post_copy
    note_msg_id = res["note_message_id"]
    tg.handle_text(
        conn,
        stub,
        "chat",
        {"d1": {"note_message_id": note_msg_id, "stage": "editor_line", "decision": "approve"}},
        {"update_id": 2, "message": {"message_id": 20, "text": "The bear was a tax dodge.",
                                     "reply_to_message": {"message_id": note_msg_id}}},
    )

    row = conn.execute("SELECT post_copy FROM queue WHERE draft_id='d1'").fetchone()
    assert row["post_copy"] == "A pastry shop started a war. Honestly."


def test_telegram_approve_skips_copy_when_no_client(tmp_path, monkeypatch):
    import humorhist.llm as llm

    monkeypatch.setattr(llm, "default_client", llm.NousClient)
    monkeypatch.delenv("HUMORHIST_LLM_API_KEY", raising=False)

    conn = _fresh_db(tmp_path)
    _seed_pending(conn)

    stub = tg.StubTelegram()
    upd = {
        "update_id": 1,
        "callback_query": {
            "id": "cb1",
            "data": "approve:d1",
            "message": {"message_id": 10},
        },
    }
    res = tg.handle_callback(conn, stub, "chat", upd)
    assert res is not None
    status = conn.execute("SELECT status FROM drafts WHERE id='d1'").fetchone()
    copy = conn.execute("SELECT post_copy FROM queue WHERE draft_id='d1'").fetchone()
    assert status["status"] == "approved"
    assert copy["post_copy"] is None
