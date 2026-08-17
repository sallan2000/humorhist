"""Tests for the Phase 4 publisher (humorhist.publish).

Covers: cadence guard, dry-run (no network, no DB mutation), the Mastodon
transport over httpx (respx-mocked), the resilient resolver's clean
unavailable path, and the end-to-end publish loop writing a posts row.
"""

from __future__ import annotations

import httpx
import pytest

import humorhist.db as db
import humorhist.publish as publish


def _fresh_db(tmp_path):
    path = tmp_path / "test.sqlite"
    conn = db.connect(str(path))
    db.migrate(conn)
    return conn


def _seed_queued(conn, *, draft_id="d1", post_copy="A very funny history fact.", image_path=None, source_link=None):
    """Insert a draft + an approved, unpublished queue row."""
    conn.execute("INSERT INTO pool (id, title, status) VALUES (?, ?, ?)", (f"p:{draft_id}", "Title", "used"))
    conn.execute(
        "INSERT INTO drafts (id, pool_id, status, editor_line) VALUES (?, ?, 'approved', 'joke')",
        (draft_id, f"p:{draft_id}"),
    )
    conn.execute(
        "INSERT INTO queue (draft_id, published, post_copy, post_copy_at, image_path, source_link) "
        "VALUES (?, 0, ?, datetime('now'), ?, ?)",
        (draft_id, post_copy, image_path, source_link),
    )
    conn.commit()


# --- cadence guard ---------------------------------------------------------

def test_remaining_quota_honours_daily_max(tmp_path):
    conn = _fresh_db(tmp_path)
    assert publish.remaining_quota(conn, daily_max=1) == 1
    db.add_post(conn, draft_id="x", transport="mastodon", url="u")
    assert publish.remaining_quota(conn, daily_max=1) == 0


def test_remaining_quota_is_per_transport(tmp_path):
    conn = _fresh_db(tmp_path)
    db.add_post(conn, draft_id="a", transport="mastodon")
    assert publish.remaining_quota(conn, transport="mastodon", daily_max=1) == 0
    # a different transport still has its full quota
    assert publish.remaining_quota(conn, transport="x", daily_max=1) == 1


def test_quota_zero_publishes_nothing(tmp_path):
    conn = _fresh_db(tmp_path)
    _seed_queued(conn)
    db.add_post(conn, draft_id="already", transport="mastodon")  # uses today's 1 quota
    results = publish.publish_due(conn, dry_run=False, daily_max=1, client=publish.MastodonTransport())
    assert results == []  # quota exhausted before touching the row


# --- dry run ---------------------------------------------------------------

def test_dry_run_mutates_nothing_and_sends_no_network(tmp_path):
    conn = _fresh_db(tmp_path)
    _seed_queued(conn, source_link="https://en.wikipedia.org/wiki/Fun")
    results = publish.publish_due(conn, dry_run=True, data_dir="data")
    assert len(results) == 1
    assert results[0]["status"] == "dry-run"
    assert "https://en.wikipedia.org/wiki/Fun" in results[0]["text"]
    # nothing recorded, queue still unpublished
    assert db.posts_today(conn) == 0
    row = conn.execute("SELECT published FROM queue WHERE draft_id='d1'").fetchone()
    assert row["published"] == 0


# --- resolver unavailable ---------------------------------------------------

def test_resilient_transport_raises_without_credentials(monkeypatch, tmp_path):
    monkeypatch.delenv("HUMORHIST_MASTODON_BASE_URL", raising=False)
    monkeypatch.delenv("HUMORHIST_MASTODON_TOKEN", raising=False)
    with pytest.raises(publish.PublishUnavailable):
        publish.resilient_transport(transport="mastodon")


def test_unknown_transport_raises(tmp_path):
    with pytest.raises(publish.PublishUnavailable):
        publish.resilient_transport(transport="nope")


def test_publish_reports_skipped_when_unconfigured(tmp_path):
    conn = _fresh_db(tmp_path)
    _seed_queued(conn)
    # no client injected and resolver will raise -> captured as skipped
    results = publish.publish_due(conn, dry_run=False, daily_max=5)
    assert results[0]["status"] == "skipped"
    assert "not configured" in results[0]["reason"]


# --- Mastodon transport (respx) ---------------------------------------------

BASE = "https://mastodon.example"


def _mastodon_client():
    return publish.MastodonTransport(base_url=BASE, token="TOK", max_retries=0)


def test_mastodon_send_posts_status(respx_mock):
    route = respx_mock.post(f"{BASE}/api/v1/statuses").mock(
        return_value=httpx.Response(200, json={"id": "123", "url": f"{BASE}/@me/123"})
    )
    client = _mastodon_client()
    out = client.send(text="hello world")
    assert out["id"] == "123"
    body = route.calls.last.request.content.decode()
    assert '"status":"hello world"' in body


def test_mastodon_send_uploads_media_when_image_present(tmp_path, respx_mock):
    img = tmp_path / "d1.png"
    img.write_bytes(b"\x89PNG fake")
    media_route = respx_mock.post(f"{BASE}/api/v1/media").mock(
        return_value=httpx.Response(200, json={"id": "M1"})
    )
    status_route = respx_mock.post(f"{BASE}/api/v1/statuses").mock(
        return_value=httpx.Response(200, json={"id": "S1", "url": f"{BASE}/@me/S1"})
    )
    client = _mastodon_client()
    out = client.send(text="with pic", image_path=str(img))
    assert out["id"] == "S1"
    assert media_route.call_count == 1
    assert status_route.call_count == 1
    # the status carried the media id
    body = status_route.calls.last.request.content.decode()
    assert '"media_ids":["M1"]' in body


def test_mastodon_send_missing_token_raises():
    client = publish.MastodonTransport(base_url="", token="")
    with pytest.raises(publish.PublishUnavailable):
        client.send(text="x")


# --- end to end -------------------------------------------------------------

def test_publish_end_to_end_writes_post_row(tmp_path, respx_mock):
    conn = _fresh_db(tmp_path)
    _seed_queued(conn, draft_id="d1", source_link="https://en.wikipedia.org/wiki/Fun")
    respx_mock.post(f"{BASE}/api/v1/statuses").mock(
        return_value=httpx.Response(200, json={"id": "999", "url": f"{BASE}/@me/999"})
    )
    results = publish.publish_due(
        conn, dry_run=False, daily_max=5, client=_mastodon_client(), data_dir="data"
    )
    assert results[0]["status"] == "published"
    assert results[0]["url"] == f"{BASE}/@me/999"
    # posts row recorded
    assert db.posts_today(conn) == 1
    prow = conn.execute("SELECT transport, post_id, url FROM posts WHERE draft_id='d1'").fetchone()
    assert prow["transport"] == "mastodon"
    assert prow["post_id"] == "999"
    # queue row flipped to published so it won't re-post
    qrow = conn.execute("SELECT published FROM queue WHERE draft_id='d1'").fetchone()
    assert qrow["published"] == 1


def test_publish_respects_limit(tmp_path, respx_mock):
    conn = _fresh_db(tmp_path)
    for i in range(3):
        _seed_queued(conn, draft_id=f"d{i}", post_copy=f"fact {i}")
    respx_mock.post(f"{BASE}/api/v1/statuses").mock(
        return_value=httpx.Response(200, json={"id": "x", "url": f"{BASE}/x"})
    )
    results = publish.publish_due(
        conn, dry_run=False, daily_max=5, limit=2, client=_mastodon_client(), data_dir="data"
    )
    assert len(results) == 2
    assert db.posts_today(conn) == 2
