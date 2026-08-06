"""Tests for humorhist.review -- the Phase 3 human review gate.

TDD with a real file-backed DB via tmp_path (no network). The review state
machine must be transport-agnostic: the CLI and any future Telegram transport
both call apply_review(), so this is where the behaviour lives and is tested.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import humorhist.db as db
import humorhist.review as review


def _fresh_db(tmp_path: Path):
    path = tmp_path / "test.sqlite"
    conn = db.connect(str(path))
    db.migrate(conn)
    return conn


def _make_draft(conn, draft_id: str, status: str = "pending") -> None:
    pool_id = "pool-x"
    # drafts.pool_id REFERENCES pool(id) with foreign_keys ON, so the pool row
    # must exist.
    conn.execute(
        "INSERT OR IGNORE INTO pool (id, title, status) VALUES (?, 'Pool X', 'drafted')",
        (pool_id,),
    )
    conn.execute(
        """INSERT INTO drafts (id, pool_id, brief_json, angles_json, status, created_at)
           VALUES (?, 'pool-x', '{}', '{}', ?, '2026-01-01T00:00:00+00:00')""",
        (draft_id, status),
    )
    conn.commit()


# --- pending_drafts ----------------------------------------------------------


def test_pending_drafts_returns_only_pending(tmp_path):
    conn = _fresh_db(tmp_path)
    _make_draft(conn, "d1", "pending")
    _make_draft(conn, "d2", "pending")
    _make_draft(conn, "d3", "approved")
    _make_draft(conn, "d4", "rejected")

    pend = review.pending_drafts(conn)
    ids = {d["id"] for d in pend}
    assert ids == {"d1", "d2"}


def test_pending_drafts_ordered_by_created_at(tmp_path):
    conn = _fresh_db(tmp_path)
    _make_draft(conn, "older", "pending")
    conn.execute(
        "UPDATE drafts SET created_at = '2026-01-01T00:00:00+00:00' WHERE id='older'"
    )
    conn.execute(
        "INSERT INTO drafts (id, pool_id, brief_json, angles_json, status, created_at)"
        " VALUES ('newer', 'pool-x', '{}', '{}', 'pending', '2026-02-01T00:00:00+00:00')"
    )
    conn.commit()

    pend = review.pending_drafts(conn)
    assert [d["id"] for d in pend] == ["older", "newer"]


# --- apply_review ------------------------------------------------------------


def test_apply_review_approve_sets_status_and_timestamp(tmp_path):
    conn = _fresh_db(tmp_path)
    _make_draft(conn, "d1", "pending")

    review.apply_review(conn, "d1", decision="approve")

    row = conn.execute("SELECT * FROM drafts WHERE id='d1'").fetchone()
    assert row["status"] == "approved"
    assert row["reviewed_at"] is not None
    assert row["editor_line"] is None
    assert row["editor_notes"] is None


def test_apply_review_reject_sets_status_and_timestamp(tmp_path):
    conn = _fresh_db(tmp_path)
    _make_draft(conn, "d1", "pending")

    review.apply_review(conn, "d1", decision="reject")

    row = conn.execute("SELECT * FROM drafts WHERE id='d1'").fetchone()
    assert row["status"] == "rejected"
    assert row["reviewed_at"] is not None


def test_apply_review_records_editor_line_and_notes(tmp_path):
    conn = _fresh_db(tmp_path)
    _make_draft(conn, "d1", "pending")

    review.apply_review(
        conn,
        "d1",
        decision="approve",
        editor_line="Lead with the bears, not the bureaucracy.",
        notes="Tighten the third angle's payoff.",
    )

    row = conn.execute("SELECT * FROM drafts WHERE id='d1'").fetchone()
    assert row["status"] == "approved"
    assert row["editor_line"] == "Lead with the bears, not the bureaucracy."
    assert row["editor_notes"] == "Tighten the third angle's payoff."


def test_apply_review_rejects_bad_decision(tmp_path):
    conn = _fresh_db(tmp_path)
    _make_draft(conn, "d1", "pending")

    with pytest.raises(ValueError):
        review.apply_review(conn, "d1", decision="maybe")

    # status untouched
    assert conn.execute("SELECT status FROM drafts WHERE id='d1'").fetchone()["status"] == "pending"


def test_apply_review_on_missing_draft_raises(tmp_path):
    conn = _fresh_db(tmp_path)
    with pytest.raises(ValueError):
        review.apply_review(conn, "nope", decision="approve")


def test_apply_review_is_idempotent_on_same_decision(tmp_path):
    conn = _fresh_db(tmp_path)
    _make_draft(conn, "d1", "pending")

    review.apply_review(conn, "d1", decision="approve", notes="first pass")
    # re-approving must not raise and must preserve the reviewed_at
    review.apply_review(conn, "d1", decision="approve", notes="second pass")
    row = conn.execute("SELECT * FROM drafts WHERE id='d1'").fetchone()
    assert row["status"] == "approved"
    assert row["editor_notes"] == "second pass"


def test_apply_review_can_flip_approved_to_rejected(tmp_path):
    conn = _fresh_db(tmp_path)
    _make_draft(conn, "d1", "pending")
    review.apply_review(conn, "d1", decision="approve")
    review.apply_review(conn, "d1", decision="reject", notes="changed my mind")

    row = conn.execute("SELECT * FROM drafts WHERE id='d1'").fetchone()
    assert row["status"] == "rejected"
    assert row["editor_notes"] == "changed my mind"


def test_apply_review_rejects_unreviewable_status(tmp_path):
    conn = _fresh_db(tmp_path)
    # already beyond the review gate (e.g. used by Phase 4)
    _make_draft(conn, "d1", "used")
    with pytest.raises(ValueError):
        review.apply_review(conn, "d1", decision="approve")
