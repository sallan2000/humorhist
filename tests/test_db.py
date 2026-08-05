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
