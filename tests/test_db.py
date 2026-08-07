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


# --- duplicate-event prevention (normalized_title dedup) -------------------


def test_upsert_merges_same_event_different_source(tmp_path):
    """The same historical event from two sources collapses to ONE pool row."""
    conn = _fresh_db(tmp_path)
    a = db.make_id("wikipedia:List", "War of Jenkins' Ear")
    b = db.make_id("seed", "War of Jenkins' Ear")
    assert db.upsert_pool_item(conn, id=a, title="War of Jenkins' Ear",
                               year=1739, date_hint=None, summary="war",
                               source_url="u1", source_name="wikipedia:List")
    # second insert is the same event under a different source id -> merged
    assert db.upsert_pool_item(conn, id=b, title="War of Jenkins' Ear",
                               year=1739, date_hint=None, summary=None,
                               source_url=None, source_name="seed") is False
    assert db.counts(conn)["pool"] == 1


def test_upsert_merges_different_spelling(tmp_path):
    conn = _fresh_db(tmp_path)
    a = db.make_id("w", "Moroccan War of Succession")
    b = db.make_id("w", "moroccan war of succession ")  # casing + trailing space
    db.upsert_pool_item(conn, id=a, title="Moroccan War of Succession",
                        year=None, date_hint=None, summary=None,
                        source_url=None, source_name="w")
    db.upsert_pool_item(conn, id=b, title="moroccan war of succession ",
                        year=1727, date_hint=None, summary="war",
                        source_url="u", source_name="w")
    assert db.counts(conn)["pool"] == 1
    row = conn.execute("SELECT * FROM pool").fetchone()
    # the richer (later) insert filled the empty fields without clobbering
    assert row["year"] == 1727
    assert row["summary"] == "war"


def test_upsert_drops_junk_title(tmp_path):
    conn = _fresh_db(tmp_path)
    assert db.upsert_pool_item(conn, id="x", title="[http://junk]",
                               year=None, date_hint=None, summary=None,
                               source_url=None, source_name="w") is False
    assert db.counts(conn)["pool"] == 0


def test_upsert_tie_breaker_protects_drafted_row(tmp_path):
    """An already-drafted row must not be overwritten by a new 'new' twin."""
    conn = _fresh_db(tmp_path)
    drafted_id = db.make_id("w", "Emu War")
    new_id = db.make_id("seed", "Emu War")
    db.upsert_pool_item(conn, id=drafted_id, title="Emu War", year=1932,
                        date_hint=None, summary="emus", source_url="u",
                        source_name="w", funny_score=9.0)
    db.set_status(conn, "pool", drafted_id, "drafted")
    conn.execute("INSERT INTO drafts (id, pool_id, status) VALUES ('d1',?, 'pending')",
                 (drafted_id,))
    conn.commit()
    # newer twin arrives, scored lower
    assert db.upsert_pool_item(conn, id=new_id, title="Emu War", year=1932,
                               date_hint=None, summary=None, source_url=None,
                               source_name="seed", funny_score=4.0) is False
    assert db.counts(conn)["pool"] == 1
    row = conn.execute("SELECT * FROM pool").fetchone()
    # the drafted/survivor row keeps its higher score (not clobbered)
    assert row["id"] == drafted_id
    assert row["funny_score"] == 9.0
    # the draft still points at the survivor
    assert conn.execute("SELECT pool_id FROM drafts WHERE id='d1'").fetchone()["pool_id"] == drafted_id


def test_suggest_merges_into_existing_event(tmp_path):
    """/suggest for an event that already exists folds into that row, no twin."""
    conn = _fresh_db(tmp_path)
    db.upsert_pool_item(conn, id="w1", title="Pastry War",
                        year=1838, date_hint=None, summary="mexico",
                        source_url="u", source_name="wikipedia:List")
    db.set_status(conn, "pool", "w1", "drafted")
    suggested_id = db.add_suggested_pool_item(conn, title="pastry war", note="lean into bureaucracy")
    assert suggested_id == "w1"  # folded into the existing row, same id
    assert db.counts(conn)["pool"] == 1
    row = conn.execute("SELECT note, status FROM pool WHERE id='w1'").fetchone()
    assert row["note"] == "lean into bureaucracy"
    assert row["status"] == "new"  # bumped back into the pipeline


def test_dedupe_pool_collapses_existing_twins(tmp_path):
    """A DB already holding two rows for the same event gets collapsed."""
    conn = _fresh_db(tmp_path)
    conn.execute(
        "INSERT INTO pool (id, title, year, source_name, status, normalized_title) VALUES "
        "('a','War of Jenkins Ear',1739,'wikipedia:List','drafted','war of jenkins ear'),"
        "('b','War of Jenkins'' Ear',1739,'seed','new',NULL)"
    )
    conn.commit()
    # backfill the NULL one manually (as migrate would), then dedupe
    conn.execute("UPDATE pool SET normalized_title='war of jenkins ear' WHERE id='b'")
    conn.commit()
    removed = db.dedupe_pool(conn)
    assert removed == 1
    assert db.counts(conn)["pool"] == 1
    row = conn.execute("SELECT id, status FROM pool").fetchone()
    # the drafted row survives; the 'new' twin is removed
    assert row["id"] == "a"
    assert row["status"] == "drafted"


def test_migrate_runs_dedup_and_index(tmp_path):
    """migrate() collapses duplicate normalized titles; the UNIQUE index can then
    be added safely via the public helper (it must not fail once dups are gone)."""
    conn = _fresh_db(tmp_path)
    db.upsert_pool_item(conn, id="a", title="Tunguska Event", year=1908,
                        date_hint=None, summary=None, source_url=None, source_name="w")
    db.upsert_pool_item(conn, id="b", title="Tunguska event", year=1908,
                        date_hint=None, summary=None, source_url=None, source_name="seed")
    # re-run migrate (as on app start) must not error and must keep one row
    db.migrate(conn)
    assert db.counts(conn)["pool"] == 1
    # once duplicates are gone, the safety-net UNIQUE index can be created
    db._ensure_pool_norm_index(conn)
    idx = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='index' AND name='idx_pool_norm'"
    ).fetchone()
    assert idx is not None

