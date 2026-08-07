"""Tests for humorhist.db schema layer.

TDD: these are written before the implementation exists, so the first run
should fail with ModuleNotFoundError. A real file-backed DB via tmp_path is
used to test actual persistence (no in-memory DB).
"""

from __future__ import annotations

import sqlite3

import pytest

import humorhist.db as db

POOL_STATUSES = {"new", "drafted", "rejected", "used"}
DRAFTS_STATUSES = {"pending", "approved", "rejected"}


def _fresh_db(tmp_path):
    path = tmp_path / "test.sqlite"
    conn = db.connect(str(path))
    db.migrate(conn)
    return conn


def test_schema_roundtrip(tmp_path):
    conn = _fresh_db(tmp_path)
    item_id = db.make_id("source-a", "The Great Emu War")
    assert db.upsert_pool_item(
        conn,
        id=item_id,
        title="The Great Emu War",
        year=1932,
        date_hint="1932-11",
        summary="Soldiers vs emus, emus won.",
        source_url="https://example.com/emu",
        source_name="example.com",
    )
    row = db.get_pool_item(conn, item_id)
    assert row is not None
    assert row["title"] == "The Great Emu War"
    assert row["year"] == 1932
    assert row["status"] == "new"
    assert row["funny_score"] is None

    db.set_status(conn, "pool", item_id, "drafted")
    row2 = db.get_pool_item(conn, item_id)
    assert row2["status"] == "drafted"


def test_migrate_idempotent(tmp_path):
    conn = _fresh_db(tmp_path)
    item_id = db.make_id("s", "event")
    db.upsert_pool_item(
        conn,
        id=item_id,
        title="event",
        year=None,
        date_hint=None,
        summary=None,
        source_url=None,
        source_name=None,
    )
    # migrate again - must not error or duplicate data
    db.migrate(conn)
    db.migrate(conn)
    counts = db.counts(conn)
    assert counts["pool"] == 1
    assert counts["drafts"] == 0
    assert counts["queue"] == 0
    assert counts["posts"] == 0


def test_make_id_stable(tmp_path):
    a = db.make_id("source-x", "The event")
    b = db.make_id("source-x", "The event")
    c = db.make_id("source-y", "The event")
    assert a == b
    assert a != c
    assert len(a) == 16
    assert all(ch in "0123456789abcdef" for ch in a)


def test_upsert_returns_false_on_duplicate(tmp_path):
    conn = _fresh_db(tmp_path)
    item_id = db.make_id("src", "Dup event")
    first = db.upsert_pool_item(
        conn,
        id=item_id,
        title="Dup event",
        year=None,
        date_hint=None,
        summary=None,
        source_url=None,
        source_name=None,
    )
    assert first is True

    db.set_funny_score(conn, item_id, 0.9)
    # second upsert with a different title must NOT overwrite existing row
    second = db.upsert_pool_item(
        conn,
        id=item_id,
        title="Should not clobber",
        year=2000,
        date_hint=None,
        summary=None,
        source_url=None,
        source_name=None,
    )
    assert second is False
    row = db.get_pool_item(conn, item_id)
    assert row["title"] == "Dup event"  # unchanged
    assert row["funny_score"] == 0.9  # unchanged
    assert db.counts(conn)["pool"] == 1


def test_set_status_rejects_invalid(tmp_path):
    conn = _fresh_db(tmp_path)
    item_id = db.make_id("s", "e")
    db.upsert_pool_item(
        conn,
        id=item_id,
        title="e",
        year=None,
        date_hint=None,
        summary=None,
        source_url=None,
        source_name=None,
    )
    with pytest.raises(ValueError):
        db.set_status(conn, "pool", item_id, "not-a-real-status")
    with pytest.raises(ValueError):
        db.set_status(conn, "drafts", item_id, "bogus")
    with pytest.raises(ValueError):
        db.set_status(conn, "DROP TABLE pool", item_id, "new")
    with pytest.raises(ValueError):
        db.set_status(conn, "posts", item_id, "new")


def test_counts(tmp_path):
    conn = _fresh_db(tmp_path)
    item_id = db.make_id("s", "e")
    assert db.upsert_pool_item(
        conn,
        id=item_id,
        title="e",
        year=None,
        date_hint=None,
        summary=None,
        source_url=None,
        source_name=None,
    )
    counts = db.counts(conn)
    assert counts == {"pool": 1, "drafts": 0, "queue": 0, "posts": 0}


def test_migrate_adds_defer_and_note_columns(tmp_path):
    conn = _fresh_db(tmp_path)
    db.migrate(conn)  # idempotent re-run must keep the new columns
    draft_cols = {r[1] for r in conn.execute("PRAGMA table_info(drafts)")}
    pool_cols = {r[1] for r in conn.execute("PRAGMA table_info(pool)")}
    assert "defer_until" in draft_cols
    assert "note" in pool_cols


def test_add_suggested_pool_item_inserts_new(tmp_path):
    conn = _fresh_db(tmp_path)
    pid = db.add_suggested_pool_item(conn, title="The Dancing Plague of 1518", note="lean into mass hysteria")
    row = db.get_pool_item(conn, pid)
    assert row["title"] == "The Dancing Plague of 1518"
    assert row["status"] == "new"
    assert row["funny_score"] is None
    assert row["source_name"] == "editor-suggestion"
    assert row["note"] == "lean into mass hysteria"


def test_add_suggested_pool_item_is_idempotent(tmp_path):
    conn = _fresh_db(tmp_path)
    db.add_suggested_pool_item(conn, title="Same Topic", note="first")
    db.add_suggested_pool_item(conn, title="Same Topic", note="second")
    rows = conn.execute("SELECT count(*) n FROM pool WHERE title='Same Topic'").fetchone()["n"]
    assert rows == 1
    row = conn.execute("SELECT note, status FROM pool WHERE title='Same Topic'").fetchone()
    assert row["note"] == "second"
    assert row["status"] == "new"


def test_defer_draft_sets_defer_until(tmp_path):
    conn = _fresh_db(tmp_path)
    db.upsert_pool_item(conn, id="p1", title="t", year=None, date_hint=None, summary=None, source_url=None, source_name=None)
    conn.execute("INSERT INTO drafts (id, pool_id, status) VALUES ('d1','p1','pending')")
    conn.commit()
    db.defer_draft(conn, "d1", days=30)
    row = conn.execute("SELECT defer_until FROM drafts WHERE id='d1'").fetchone()
    assert row["defer_until"] is not None
    # cannot defer a non-pending draft
    db.set_status(conn, "drafts", "d1", "approved")
    with pytest.raises(ValueError):
        db.defer_draft(conn, "d1")
    # clear works
    db.clear_defer(conn, "d1")
    assert conn.execute("SELECT defer_until FROM drafts WHERE id='d1'").fetchone()["defer_until"] is None

