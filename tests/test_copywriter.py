"""Tests for humorhist.copywriter -- Phase 4 (B+) post-copy generation.

No network: a StubClient provides deterministic model output; DBs are fresh
file-backed via tmp_path. Covers the char-limit knob, over-limit trimming, the
no-key safe-skip, and idempotent fill.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

import humorhist.copywriter as cw
import humorhist.db as db
from humorhist.llm import StubClient


def _fresh_db(tmp_path: Path):
    conn = db.connect(str(tmp_path / "test.sqlite"))
    db.migrate(conn)
    return conn


def _make_draft(
    conn,
    draft_id: str,
    status: str = "approved",
    *,
    editor_line: str | None = None,
    with_brief: bool = True,
) -> None:
    conn.execute(
        "INSERT OR IGNORE INTO pool (id, title, year, status) VALUES (?, 'Pastry War', 1838, 'drafted')",
        ("pool-x",),
    )
    brief = (
        '{"verified_facts": ["France and Mexico went to war over a pastry shop"], '
        '"misconceptions": ["No, nobody actually ate the pastry"], "sources": []}'
        if with_brief
        else "{}"
    )
    angles = (
        '{"angles": [{"angle_name": "Bureaucratic revenge"}], "suggested_hook": "A pastry shop was the casus belli"}'
    )
    conn.execute(
        """INSERT INTO drafts (id, pool_id, brief_json, angles_json, status, created_at, editor_line)
           VALUES (?, 'pool-x', ?, ?, ?, '2026-01-01T00:00:00+00:00', ?)""",
        (draft_id, brief, angles, status, editor_line),
    )
    conn.commit()


def _enqueue(conn, draft_id: str) -> None:
    conn.execute(
        "INSERT INTO queue (draft_id, scheduled_for, published) VALUES (?, NULL, 0)",
        (draft_id,),
    )
    conn.commit()


# --- migration -----------------------------------------------------------------


def test_migrate_adds_copy_columns(tmp_path):
    conn = _fresh_db(tmp_path)
    cols = {r[1] for r in conn.execute("PRAGMA table_info(queue)")}
    assert "post_copy" in cols
    assert "post_copy_at" in cols


def test_migrate_is_idempotent(tmp_path):
    conn = _fresh_db(tmp_path)
    db.migrate(conn)  # second migrate must not error
    cols = {r[1] for r in conn.execute("PRAGMA table_info(queue)")}
    assert "post_copy" in cols


# --- char_limit knob -----------------------------------------------------------


def test_char_limit_default_is_280(monkeypatch):
    monkeypatch.delenv("HUMORHIST_CHAR_LIMIT", raising=False)
    assert cw.char_limit() == 280


def test_char_limit_reads_env(monkeypatch):
    monkeypatch.setenv("HUMORHIST_CHAR_LIMIT", "500")
    assert cw.char_limit() == 500


def test_char_limit_falls_back_on_garbage(monkeypatch):
    monkeypatch.setenv("HUMORHIST_CHAR_LIMIT", "not-a-number")
    assert cw.char_limit() == 280


# --- generate_post_copy --------------------------------------------------------


def test_generate_post_copy_uses_brief_and_trims(tmp_path, monkeypatch):
    monkeypatch.setenv("HUMORHIST_CHAR_LIMIT", "280")
    conn = _fresh_db(tmp_path)
    _make_draft(conn, "d1")
    draft = dict(conn.execute("SELECT * FROM drafts WHERE id='d1'").fetchone())
    pool = dict(db.get_pool_item(conn, draft["pool_id"]))

    # model returns a long post; we expect it hard-trimmed to <= limit
    long_post = "x" * 400
    client = StubClient([{"post": long_post}])
    copy = cw.generate_post_copy(client, draft, pool)
    assert len(copy) <= 280


def test_generate_post_copy_respects_custom_limit(tmp_path, monkeypatch):
    monkeypatch.setenv("HUMORHIST_CHAR_LIMIT", "280")
    conn = _fresh_db(tmp_path)
    _make_draft(conn, "d1")
    draft = dict(conn.execute("SELECT * FROM drafts WHERE id='d1'").fetchone())
    pool = dict(db.get_pool_item(conn, draft["pool_id"]))

    client = StubClient([{"post": "A perfectly sized pastry-war quip. "}])
    copy = cw.generate_post_copy(client, draft, pool, limit=50)
    assert len(copy) <= 50


def test_generate_post_copy_raises_on_empty_post(tmp_path):
    conn = _fresh_db(tmp_path)
    _make_draft(conn, "d1")
    draft = dict(conn.execute("SELECT * FROM drafts WHERE id='d1'").fetchone())
    pool = dict(db.get_pool_item(conn, draft["pool_id"]))

    client = StubClient([{"post": ""}])
    from humorhist.llm import LLMError

    with pytest.raises(LLMError):
        cw.generate_post_copy(client, draft, pool)


# --- fill_post_copy ------------------------------------------------------------


def test_fill_post_copy_writes_copy_and_timestamp(tmp_path, monkeypatch):
    monkeypatch.setenv("HUMORHIST_CHAR_LIMIT", "280")
    conn = _fresh_db(tmp_path)
    _make_draft(conn, "d1")
    _enqueue(conn, "d1")

    client = StubClient([{"post": "France invaded Mexico over a pastry shop. Naturally."}])
    n = cw.fill_post_copy(conn, client, draft_id="d1")
    assert n == 1
    row = conn.execute("SELECT post_copy, post_copy_at FROM queue WHERE draft_id='d1'").fetchone()
    assert row["post_copy"] == "France invaded Mexico over a pastry shop. Naturally."
    assert row["post_copy_at"] is not None


def test_fill_post_copy_is_idempotent(tmp_path, monkeypatch):
    monkeypatch.setenv("HUMORHIST_CHAR_LIMIT", "280")
    conn = _fresh_db(tmp_path)
    _make_draft(conn, "d1")
    _enqueue(conn, "d1")

    client = StubClient(
        [
            {"post": "First generation."},
            {"post": "Second generation (should never be used)."},
        ]
    )
    assert cw.fill_post_copy(conn, client, draft_id="d1") == 1
    # re-running must not generate again (stub would error if it did)
    assert cw.fill_post_copy(conn, client, draft_id="d1") == 0
    row = conn.execute("SELECT post_copy FROM queue WHERE draft_id='d1'").fetchone()
    assert row["post_copy"] == "First generation."


def test_fill_post_copy_skips_when_no_client(tmp_path):
    conn = _fresh_db(tmp_path)
    _make_draft(conn, "d1")
    _enqueue(conn, "d1")
    # client=None => safe no-op, no exception, no copy written
    assert cw.fill_post_copy(conn, None) == 0
    row = conn.execute("SELECT post_copy FROM queue WHERE draft_id='d1'").fetchone()
    assert row["post_copy"] is None


def test_fill_post_copy_scoped_to_draft_id(tmp_path, monkeypatch):
    monkeypatch.setenv("HUMORHIST_CHAR_LIMIT", "280")
    conn = _fresh_db(tmp_path)
    _make_draft(conn, "d1")
    _make_draft(conn, "d2")
    _enqueue(conn, "d1")
    _enqueue(conn, "d2")

    client = StubClient([{"post": "Only d1 should fill."}])
    assert cw.fill_post_copy(conn, client, draft_id="d1") == 1
    d2 = conn.execute("SELECT post_copy FROM queue WHERE draft_id='d2'").fetchone()
    assert d2["post_copy"] is None


# --- set_post_copy (manual edit) ----------------------------------------------


def test_set_post_copy_stores_editor_text(tmp_path):
    conn = _fresh_db(tmp_path)
    _make_draft(conn, "d1")
    _enqueue(conn, "d1")
    cw.set_post_copy(conn, "d1", "My hand-edited, funnier version.")
    row = conn.execute("SELECT post_copy FROM queue WHERE draft_id='d1'").fetchone()
    assert row["post_copy"] == "My hand-edited, funnier version."


def test_fill_post_copy_force_overwrites_existing(tmp_path, monkeypatch):
    monkeypatch.setenv("HUMORHIST_CHAR_LIMIT", "280")
    conn = _fresh_db(tmp_path)
    _make_draft(conn, "d1")
    _enqueue(conn, "d1")
    cw.set_post_copy(conn, "d1", "Original copy.")

    # non-forced fill must NOT overwrite
    assert cw.fill_post_copy(conn, StubClient([{"post": "ignored"}]), draft_id="d1") == 0
    assert conn.execute("SELECT post_copy FROM queue WHERE draft_id='d1'").fetchone()["post_copy"] == "Original copy."

    # forced fill (regen) MUST overwrite
    assert cw.fill_post_copy(conn, StubClient([{"post": "Regenerated copy."}]), draft_id="d1", force=True) == 1
    assert (
        conn.execute("SELECT post_copy FROM queue WHERE draft_id='d1'").fetchone()["post_copy"] == "Regenerated copy."
    )


@pytest.mark.parametrize("manual", [False, True])
def test_copy_respects_caller_transaction_and_rollback(tmp_path, manual):
    conn = _fresh_db(tmp_path)
    _make_draft(conn, "d1")
    _enqueue(conn, "d1")
    conn.execute("UPDATE pool SET title = 'uncommitted' WHERE id = 'pool-x'")
    if manual:
        cw.set_post_copy(conn, "d1", "Hand edited copy.")
    else:
        assert cw.fill_post_copy(conn, StubClient([{"post": "Generated copy."}])) == 1
    assert conn.in_transaction
    conn.rollback()
    assert conn.execute("SELECT title FROM pool").fetchone()[0] == "Pastry War"
    assert conn.execute("SELECT post_copy FROM queue").fetchone()[0] is None


@pytest.mark.parametrize("manual", [False, True])
def test_copy_commits_when_no_caller_transaction(tmp_path, manual):
    conn = _fresh_db(tmp_path)
    _make_draft(conn, "d1")
    _enqueue(conn, "d1")
    if manual:
        cw.set_post_copy(conn, "d1", "Hand edited copy.")
    else:
        assert cw.fill_post_copy(conn, StubClient([{"post": "Generated copy."}])) == 1
    assert not conn.in_transaction
    other = db.connect(str(tmp_path / "test.sqlite"))
    assert other.execute("SELECT post_copy FROM queue").fetchone()[0] is not None
    other.close()


@pytest.mark.parametrize("force", [False, True])
@pytest.mark.parametrize("scoped", [False, True])
@pytest.mark.parametrize("existing", [None, "published copy"])
def test_published_rows_are_not_generated_or_edited(tmp_path, force, scoped, existing):
    conn = _fresh_db(tmp_path)
    _make_draft(conn, "d1")
    _enqueue(conn, "d1")
    conn.execute("UPDATE queue SET published=1, post_copy=?", (existing,))
    conn.commit()

    class NoCalls:
        def complete_json(self, *args, **kwargs):
            pytest.fail("published row must not trigger generation")

    assert cw.fill_post_copy(conn, NoCalls(), draft_id="d1" if scoped else None, force=force) == 0
    cw.set_post_copy(conn, "d1", "edited")
    assert conn.execute("SELECT post_copy FROM queue").fetchone()[0] == existing


@pytest.mark.parametrize("change", ["published", "copy", "status"])
def test_fill_rechecks_row_after_generation(tmp_path, monkeypatch, change):
    conn = _fresh_db(tmp_path)
    _make_draft(conn, "d1")
    _enqueue(conn, "d1")
    other = db.connect(str(tmp_path / "test.sqlite"))

    def generate(*args):
        assert not conn.in_transaction
        if change == "published":
            other.execute("UPDATE queue SET published=1")
        elif change == "copy":
            other.execute("UPDATE queue SET post_copy='editor copy'")
        else:
            other.execute("UPDATE drafts SET status='rejected'")
        other.commit()
        return "Generated copy."

    monkeypatch.setattr(cw, "generate_post_copy", generate)
    assert cw.fill_post_copy(conn, StubClient([])) == 0
    assert not conn.in_transaction
    row = conn.execute("SELECT post_copy FROM queue").fetchone()
    assert row[0] == ("editor copy" if change == "copy" else None)
    other.close()


def test_batch_generates_before_any_write(tmp_path, monkeypatch):
    conn = _fresh_db(tmp_path)
    for draft_id in ("d1", "d2"):
        _make_draft(conn, draft_id)
        _enqueue(conn, draft_id)
    calls = []

    def generate(client, draft, pool, limit):
        assert not conn.in_transaction
        calls.append(draft["id"])
        return "Generated copy."

    monkeypatch.setattr(cw, "generate_post_copy", generate)
    assert cw.fill_post_copy(conn, StubClient([])) == 2
    assert set(calls) == {"d1", "d2"}


@pytest.mark.parametrize("caller_transaction", [False, True])
def test_later_batch_write_failure_rolls_back_only_batch(tmp_path, caller_transaction):
    conn = _fresh_db(tmp_path)
    for draft_id in ("d1", "d2"):
        _make_draft(conn, draft_id)
        _enqueue(conn, draft_id)
    conn.execute("""CREATE TRIGGER fail_second BEFORE UPDATE OF post_copy ON queue
                    WHEN NEW.draft_id = 'd2'
                    BEGIN SELECT RAISE(ABORT, 'injected write failure'); END""")
    conn.commit()
    if caller_transaction:
        conn.execute("UPDATE pool SET title='caller change' WHERE id='pool-x'")
    with pytest.raises(sqlite3.IntegrityError, match="injected write failure"):
        cw.fill_post_copy(conn, StubClient([{"post": "First generated copy."}, {"post": "Second generated copy."}]))
    assert conn.in_transaction == caller_transaction
    assert [r[0] for r in conn.execute("SELECT post_copy FROM queue ORDER BY id")] == [None, None]
    if caller_transaction:
        assert conn.execute("SELECT title FROM pool").fetchone()[0] == "caller change"
        conn.commit()
        other = db.connect(str(tmp_path / "test.sqlite"))
        assert other.execute("SELECT title FROM pool").fetchone()[0] == "caller change"
        assert [r[0] for r in other.execute("SELECT post_copy FROM queue ORDER BY id")] == [None, None]
        other.close()


@pytest.mark.parametrize("same_text", [False, True])
def test_force_regen_preserves_intervening_manual_edit(tmp_path, monkeypatch, same_text):
    conn = _fresh_db(tmp_path)
    _make_draft(conn, "d1")
    _enqueue(conn, "d1")
    assert cw.set_post_copy(conn, "d1", "Original copy.") == 1
    conn.execute("UPDATE queue SET post_copy_at='old timestamp'")
    conn.commit()
    other = db.connect(str(tmp_path / "test.sqlite"))

    def generate(*args):
        assert cw.set_post_copy(other, "d1", "Original copy." if same_text else "New editor copy.") == 1
        return "Regenerated copy."

    monkeypatch.setattr(cw, "generate_post_copy", generate)
    assert cw.fill_post_copy(conn, StubClient([]), force=True) == 0
    assert conn.execute("SELECT post_copy FROM queue").fetchone()[0] == (
        "Original copy." if same_text else "New editor copy."
    )
    other.close()


def test_set_post_copy_returns_updated_count(tmp_path):
    conn = _fresh_db(tmp_path)
    _make_draft(conn, "d1")
    _enqueue(conn, "d1")
    assert cw.set_post_copy(conn, "missing", "Copy") == 0
    assert cw.set_post_copy(conn, "d1", "Copy") == 1
    conn.execute("UPDATE queue SET published=1")
    conn.commit()
    assert cw.set_post_copy(conn, "d1", "Copy") == 0
