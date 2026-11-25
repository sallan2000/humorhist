"""Tests for humorhist.review -- the Phase 3 human review gate.

TDD with a real file-backed DB via tmp_path (no network). The review state
machine must be transport-agnostic: the CLI and any future Telegram transport
both call apply_review(), so this is where the behaviour lives and is tested.
"""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime, timedelta
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
    conn.execute("UPDATE drafts SET created_at = '2026-01-01T00:00:00+00:00' WHERE id='older'")
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


# --- enqueue_approved / queued_drafts (Phase 4 handoff) ----------------------


def test_enqueue_approved_moves_only_approved(tmp_path):
    conn = _fresh_db(tmp_path)
    _make_draft(conn, "d1", "approved")
    _make_draft(conn, "d2", "approved")
    _make_draft(conn, "d3", "pending")
    _make_draft(conn, "d4", "rejected")

    n = review.enqueue_approved(conn)
    assert n == 2
    q = {r["draft_id"] for r in conn.execute("SELECT draft_id FROM queue")}
    assert q == {"d1", "d2"}


def test_enqueue_approved_is_idempotent(tmp_path):
    conn = _fresh_db(tmp_path)
    _make_draft(conn, "d1", "approved")
    assert review.enqueue_approved(conn) == 1
    # running again must not insert a second row
    assert review.enqueue_approved(conn) == 0
    assert len(list(conn.execute("SELECT * FROM queue"))) == 1


def test_queued_drafts_lists_unpublished_oldest_first(tmp_path):
    conn = _fresh_db(tmp_path)
    _make_draft(conn, "d1", "approved")
    _make_draft(conn, "d2", "approved")
    review.enqueue_approved(conn)
    rows = review.queued_drafts(conn)
    assert [r["draft_id"] for r in rows] == ["d1", "d2"]
    assert rows[0]["title"] == "Pool X"
    assert rows[0]["published"] == 0


def test_apply_review_approve_auto_enqueues(tmp_path):
    conn = _fresh_db(tmp_path)
    _make_draft(conn, "d1", "pending")
    review.apply_review(conn, "d1", decision="approve")
    q = {r["draft_id"] for r in conn.execute("SELECT draft_id FROM queue")}
    assert q == {"d1"}


def test_apply_review_reject_removes_from_queue(tmp_path):
    conn = _fresh_db(tmp_path)
    _make_draft(conn, "d1", "pending")
    review.apply_review(conn, "d1", decision="approve")
    assert {r["draft_id"] for r in conn.execute("SELECT draft_id FROM queue")} == {"d1"}
    # flip to reject -> queue row must be cleaned up
    review.apply_review(conn, "d1", decision="reject")
    assert list(conn.execute("SELECT * FROM queue")) == []


def test_defer_draft_sets_defer_until_and_pends(tmp_path):
    conn = _fresh_db(tmp_path)
    _make_draft(conn, "d1", "pending")
    review.defer_draft(conn, "d1", days=30)
    row = conn.execute("SELECT defer_until, status FROM drafts WHERE id='d1'").fetchone()
    assert row["defer_until"] is not None
    assert row["status"] == "pending"  # still reviewable later
    # applying a review clears the defer
    review.apply_review(conn, "d1", decision="approve")
    assert conn.execute("SELECT defer_until FROM drafts WHERE id='d1'").fetchone()["defer_until"] is None


def test_pending_drafts_excludes_future_deferrals(tmp_path):
    conn = _fresh_db(tmp_path)
    _make_draft(conn, "normal", "pending")
    _make_draft(conn, "deferred", "pending")
    # defer one far into the future
    future = (datetime.now(UTC) + timedelta(days=30)).isoformat()
    conn.execute("UPDATE drafts SET defer_until = ? WHERE id='deferred'", (future,))
    conn.commit()
    pend = review.pending_drafts(conn)
    ids = [d["id"] for d in pend]
    assert ids == ["normal"]
    assert review.reviewed_summary(conn)["pending"]["count"] == 2


def test_only_deferred_drafts_return_after_expiry(tmp_path, monkeypatch):
    conn = _fresh_db(tmp_path)
    _make_draft(conn, "d1")
    start = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
    deadline = start + timedelta(days=1)
    conn.execute("UPDATE drafts SET defer_until = ? WHERE id = 'd1'", (deadline.isoformat(),))
    conn.commit()

    class Clock(datetime):
        current = start

        @classmethod
        def now(cls, tz=None):
            assert tz is UTC
            return cls.current

    monkeypatch.setattr(review, "datetime", Clock)
    assert review.pending_drafts(conn) == []
    assert [r["draft_id"] for r in review.deferred_drafts(conn)] == ["d1"]
    Clock.current = deadline
    assert [r["id"] for r in review.pending_drafts(conn)] == ["d1"]
    assert review.deferred_drafts(conn) == []
    row = conn.execute("SELECT status, defer_until FROM drafts WHERE id='d1'").fetchone()
    assert (row["status"], row["defer_until"]) == ("pending", deadline.isoformat())


# --- deferred_drafts / bring_forward (GAP 4) --------------------------------


def _defer(conn, draft_id: str, days: int = 30) -> None:
    when = (datetime.now(UTC) + timedelta(days=days)).isoformat()
    conn.execute("UPDATE drafts SET defer_until = ? WHERE id = ?", (when, draft_id))
    conn.commit()


def test_deferred_drafts_lists_only_future_deferred(tmp_path):
    conn = _fresh_db(tmp_path)
    _make_draft(conn, "d1", "pending")
    _make_draft(conn, "d2", "pending")
    # d1 deferred into the future; d2 not deferred
    _defer(conn, "d1")
    rows = review.deferred_drafts(conn)
    assert [r["draft_id"] for r in rows] == ["d1"]


def test_deferred_drafts_excludes_expired_and_nonpending(tmp_path):
    conn = _fresh_db(tmp_path)
    _make_draft(conn, "d1", "pending")
    _make_draft(conn, "d2", "pending")
    _make_draft(conn, "d3", "approved")
    # d1 deferred in the PAST (expired) -> excluded
    _defer(conn, "d1", days=-1)
    # d2 deferred in the future -> included
    _defer(conn, "d2")
    # d3 is approved -> not pending -> excluded even if deferred
    _defer(conn, "d3")
    rows = review.deferred_drafts(conn)
    assert [r["draft_id"] for r in rows] == ["d2"]


def test_bring_forward_one_clears_defer(tmp_path):
    conn = _fresh_db(tmp_path)
    _make_draft(conn, "d1", "pending")
    _defer(conn, "d1")
    assert review.pending_drafts(conn) == []
    n = review.bring_forward(conn, "d1")
    assert n == 1
    assert conn.execute("SELECT defer_until FROM drafts WHERE id='d1'").fetchone()["defer_until"] is None
    # now appears in the normal review surface
    assert [d["id"] for d in review.pending_drafts(conn)] == ["d1"]


def test_bring_forward_one_rejects_if_not_deferred(tmp_path):
    conn = _fresh_db(tmp_path)
    _make_draft(conn, "d1", "pending")
    # not deferred -> ValueError
    try:
        review.bring_forward(conn, "d1")
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for a non-deferred draft")


def test_bring_forward_all_clears_every_deferred(tmp_path):
    conn = _fresh_db(tmp_path)
    _make_draft(conn, "d1", "pending")
    _make_draft(conn, "d2", "pending")
    _make_draft(conn, "d3", "pending")
    _defer(conn, "d1")
    _defer(conn, "d2")
    # d3 not deferred
    assert [d["id"] for d in review.pending_drafts(conn)] == ["d3"]
    n = review.bring_forward(conn)  # no id -> all deferred
    assert n == 2
    remaining = {r["draft_id"] for r in review.deferred_drafts(conn)}
    assert remaining == set()
    assert [d["id"] for d in review.pending_drafts(conn)] == ["d1", "d2", "d3"]


# --- stuck_captures / set_editor_line (GAP 4b) -------------------------------


def _queue(conn, draft_id: str) -> None:
    conn.execute("INSERT INTO queue (draft_id, published) VALUES (?, 0)", (draft_id,))
    conn.commit()


def test_stuck_captures_finds_approved_queued_without_joke(tmp_path):
    conn = _fresh_db(tmp_path)
    _make_draft(conn, "d1", "pending")
    review.apply_review(conn, "d1", decision="approve")  # approves + enqueues
    # d1 is approved + queued but has no editor_line yet
    stuck = review.stuck_captures(conn)
    assert [s["draft_id"] for s in stuck] == ["d1"]


def test_stuck_captures_excludes_joke_present_and_pending(tmp_path):
    conn = _fresh_db(tmp_path)
    _make_draft(conn, "d1", "pending")
    _make_draft(conn, "d2", "pending")
    review.apply_review(conn, "d1", decision="approve", editor_line="already joked")
    # d2 stays pending (never approved) -> not a stuck capture
    stuck = review.stuck_captures(conn)
    assert stuck == []


def test_set_editor_line_fills_then_status_unchanged(tmp_path):
    conn = _fresh_db(tmp_path)
    _make_draft(conn, "d1", "pending")
    review.apply_review(conn, "d1", decision="approve")  # approved + queued
    review.set_editor_line(conn, "d1", "the bear was a tax dodge")
    row = conn.execute("SELECT editor_line, status FROM drafts WHERE id='d1'").fetchone()
    assert row["editor_line"] == "the bear was a tax dodge"
    assert row["status"] == "approved"  # status untouched
    # no longer a stuck capture
    assert review.stuck_captures(conn) == []


def test_set_editor_line_rejects_unknown(tmp_path):
    conn = _fresh_db(tmp_path)
    _make_draft(conn, "d1", "pending")
    try:
        review.set_editor_line(conn, "nope", "x")
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for unknown draft")


@pytest.mark.parametrize("operation", ["approve", "reject", "reopen"])
def test_review_queue_failure_rolls_back_draft_and_queue(tmp_path, operation):
    conn = _fresh_db(tmp_path)
    _make_draft(conn, "d1")
    if operation != "approve":
        review.apply_review(conn, "d1", decision="approve", notes="original")
    before = dict(conn.execute("SELECT * FROM drafts WHERE id='d1'").fetchone())
    queue_before = [dict(r) for r in conn.execute("SELECT * FROM queue")]
    verb = "INSERT" if operation == "approve" else "DELETE"
    conn.execute(f"CREATE TRIGGER fail_queue BEFORE {verb} ON queue BEGIN SELECT RAISE(ABORT, 'queue failure'); END")
    with pytest.raises(sqlite3.IntegrityError, match="queue failure"):
        if operation == "reopen":
            review.reopen_draft(conn, "d1")
        else:
            review.apply_review(conn, "d1", decision=operation, notes="changed")
    assert dict(conn.execute("SELECT * FROM drafts WHERE id='d1'").fetchone()) == before
    assert [dict(r) for r in conn.execute("SELECT * FROM queue")] == queue_before
    assert not conn.in_transaction


def test_enqueue_batch_failure_rolls_back_all_inserts(tmp_path):
    conn = _fresh_db(tmp_path)
    _make_draft(conn, "d1", "approved")
    _make_draft(conn, "d2", "approved")
    conn.execute("""CREATE TRIGGER fail_second BEFORE INSERT ON queue
                    WHEN NEW.draft_id = 'd2'
                    BEGIN SELECT RAISE(ABORT, 'queue failure'); END""")
    with pytest.raises(sqlite3.IntegrityError, match="queue failure"):
        review.enqueue_approved(conn)
    assert list(conn.execute("SELECT * FROM queue")) == []
    assert not conn.in_transaction


def test_review_respects_caller_transaction_and_source_link(tmp_path):
    conn = _fresh_db(tmp_path)
    _make_draft(conn, "d1")
    conn.execute("UPDATE pool SET source_url='https://example.com/story' WHERE id='pool-x'")
    review.apply_review(conn, "d1", decision="approve")
    assert conn.in_transaction
    assert conn.execute("SELECT source_link FROM queue").fetchone()[0] == "https://example.com/story"
    conn.rollback()
    assert conn.execute("SELECT status FROM drafts WHERE id='d1'").fetchone()[0] == "pending"
    assert list(conn.execute("SELECT * FROM queue")) == []


def test_failed_review_preserves_caller_transaction(tmp_path):
    conn = _fresh_db(tmp_path)
    _make_draft(conn, "d1")
    conn.execute("CREATE TRIGGER fail_queue BEFORE INSERT ON queue BEGIN SELECT RAISE(ABORT, 'queue failure'); END")
    conn.execute("UPDATE pool SET title='uncommitted' WHERE id='pool-x'")
    with pytest.raises(sqlite3.IntegrityError, match="queue failure"):
        review.apply_review(conn, "d1", decision="approve")
    assert conn.in_transaction
    assert conn.execute("SELECT title FROM pool").fetchone()[0] == "uncommitted"
    assert conn.execute("SELECT status FROM drafts").fetchone()[0] == "pending"
    conn.rollback()
    assert conn.execute("SELECT title FROM pool").fetchone()[0] == "Pool X"


def test_enqueue_never_generates_images_even_on_repeated_backfill(tmp_path, monkeypatch):
    from humorhist import imagegen

    conn = _fresh_db(tmp_path)
    _make_draft(conn, "d1", "approved")
    _make_draft(conn, "d2", "approved")
    calls = []
    monkeypatch.setattr(imagegen, "generate_image_for_queue", lambda *a, **k: calls.append(conn.in_transaction))
    conn.execute("BEGIN")
    assert review.enqueue_approved(conn, image_dir=str(tmp_path)) == 2
    assert calls == []
    conn.commit()
    assert review.enqueue_approved(conn, image_dir=str(tmp_path)) == 0
    assert review.enqueue_approved(conn, image_dir=str(tmp_path)) == 0
    _make_draft(conn, "d3", "approved")
    assert review.enqueue_approved(conn, image_dir=str(tmp_path)) == 1
    assert review.enqueue_approved(conn, image_dir=str(tmp_path)) == 0
    assert calls == []
    assert conn.execute("SELECT count(*) FROM queue WHERE image_path IS NULL").fetchone()[0] == 3


def test_publish_export_excludes_published_rows(tmp_path):
    conn = _fresh_db(tmp_path)
    _make_draft(conn, "d1", "approved")
    _make_draft(conn, "d2", "approved")
    review.enqueue_approved(conn)
    conn.execute("UPDATE queue SET post_copy = 'old copy', published = 1 WHERE draft_id = 'd1'")
    conn.execute("UPDATE queue SET post_copy = 'new copy' WHERE draft_id = 'd2'")
    conn.commit()
    rows = review.queued_drafts(conn)
    assert [r["draft_id"] for r in rows] == ["d2"]
    export = review.format_publish_export(conn, rows)
    assert "1 post(s)" in export and "new copy" in export
    assert "old copy" not in export


def test_link_backfill_respects_caller_transaction(tmp_path):
    conn = _fresh_db(tmp_path)
    _make_draft(conn, "d1", "approved")
    review.enqueue_approved(conn)
    conn.execute("UPDATE pool SET source_url = ? WHERE id = 'pool-x'", ("https://en.wikipedia.org/wiki/Example",))
    conn.commit()
    conn.execute("BEGIN")
    review.enqueue_approved(conn, image_dir=str(tmp_path))
    assert conn.execute("SELECT source_link FROM queue WHERE draft_id = 'd1'").fetchone()[0]
    assert conn.in_transaction
    conn.rollback()
    assert conn.execute("SELECT source_link FROM queue WHERE draft_id = 'd1'").fetchone()[0] is None
