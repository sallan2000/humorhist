"""Phase 4 publisher — turn approved queue rows into real social posts.

The pipeline up to here (harvest -> screen -> draft -> review -> edit copy +
image) produces a queue of approved, editor-finished posts. Nothing read that
queue and posted it — until now.

Design (mirrors the rest of the repo):
  * A small ``Transport`` Protocol. Only Mastodon is built (free API, low
    suspension risk). X is a documented opt-in second transport behind the same
    copy — see references/x-publisher-2026.md — and slots in without a rewrite.
  * ``MastodonTransport`` speaks the Mastodon API over httpx, so it is covered
    by ``respx`` in the tests with zero network.
  * ``resilient_transport()`` is the canonical resolver for unattended/CLI use:
    it raises ``PublishUnavailable`` (NOT a raw error) when the required
    credential is missing, so a dry/timer run degrades cleanly instead of
    dumping a traceback.
  * A daily cadence guard (``HUMORHIST_DAILY_MAX``, default 1) caps posts/day so
    an unattended run can never spam. The guard is per-transport.
  * ``publish_due`` is fully dry-runnable: ``--dry-run`` renders what would go
    out and returns the planned set WITHOUT touching the network or the DB flag.

The first real publish should be manual (never let the timer post #1). See the
status line in ``cmd_publish`` / ``status`` — the cadence guard only counts rows
already in ``posts``.
"""

from __future__ import annotations

import os
import sqlite3
import time
from pathlib import Path
from typing import Any, Protocol

import httpx

import humorhist.db as db

# Daily post ceiling. Text-only Mastodon posts are free, so 1/day is a sane
# default; raise for a paid/busier account. Per-transport (X has its own cost).
DEFAULT_DAILY_MAX = int(os.environ.get("HUMORHIST_DAILY_MAX", "1"))


class PublishUnavailable(RuntimeError):
    """Raised when the publisher cannot proceed (missing credential, etc.).

    Callers surface this as a clean message — never a traceback.
    """


class Transport(Protocol):
    """A place a finished post can be sent."""

    name: str

    def send(self, *, text: str, image_path: str | None = None) -> dict[str, Any]:
        """Post ``text`` (and optional ``image_path``). Return API result dict."""
        ...


class MastodonTransport:
    """Mastodon REST transport (statuses + media) over httpx.

    Instance API base comes from ``HUMORHIST_MASTODON_BASE_URL`` (e.g.
    https://mastodon.social) and the token from ``HUMORHIST_MASTODON_TOKEN``.
    Mirrors ``TelegramClient``: httpx POST, retry transient failures, raise a
    typed error on non-2xx.
    """

    name = "mastodon"

    def __init__(
        self,
        *,
        base_url: str | None = None,
        token: str | None = None,
        timeout: float = 30.0,
        max_retries: int = 2,
    ) -> None:
        self.base_url = (base_url or os.environ.get("HUMORHIST_MASTODON_BASE_URL", "")).rstrip("/")
        self.token = token or os.environ.get("HUMORHIST_MASTODON_TOKEN", "")
        self.timeout = timeout
        self.max_retries = max_retries

    def _post(self, path: str, *, json: dict | None = None, files: dict | None = None) -> dict:
        if not self.base_url or not self.token:
            raise PublishUnavailable(
                "Mastodon not configured — set HUMORHIST_MASTODON_BASE_URL and HUMORHIST_MASTODON_TOKEN"
            )
        url = f"{self.base_url}/api/v1/{path}"
        headers = {"Authorization": f"Bearer {self.token}"}
        last: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                with httpx.Client(timeout=self.timeout) as client:
                    resp = client.post(url, headers=headers, json=json, files=files)
                    resp.raise_for_status()
                    return resp.json()
            except Exception as exc:  # noqa: BLE001 - retry transient failures
                last = exc
                if attempt < self.max_retries:
                    time.sleep(2**attempt)
        raise PublishUnavailable(f"Mastodon {path} failed: {last}")

    def send(self, *, text: str, image_path: str | None = None) -> dict[str, Any]:
        media_id = None
        if image_path:
            media_id = self._upload_image(image_path)
        payload: dict[str, Any] = {"status": text, "visibility": "public"}
        if media_id:
            payload["media_ids"] = [media_id]
        return self._post("statuses", json=payload)

    def _upload_image(self, image_path: str) -> str:
        p = Path(image_path)
        if not p.is_file():
            raise PublishUnavailable(f"image file not found: {image_path}")
        data = p.read_bytes()
        # httpx multipart: (filename, bytes, content-type)
        files = {"file": (p.name, data, "image/png")}
        result = self._post("media", files=files)
        media_id = result.get("id")
        if not media_id:
            raise PublishUnavailable(f"Mastodon media upload returned no id: {result}")
        return media_id


def resilient_transport(*, transport: str = "mastodon") -> Transport:
    """Resolve a configured transport, or raise ``PublishUnavailable``.

    Only Mastodon is built. Add X here behind its own env vars when the X
    transport is implemented.
    """
    if transport == "mastodon":
        base = os.environ.get("HUMORHIST_MASTODON_BASE_URL")
        token = os.environ.get("HUMORHIST_MASTODON_TOKEN")
        if not (base and token):
            raise PublishUnavailable(
                "Mastodon not configured — set HUMORHIST_MASTODON_BASE_URL and HUMORHIST_MASTODON_TOKEN"
            )
        return MastodonTransport()
    raise PublishUnavailable(f"unknown transport: {transport!r} (supported: mastodon)")


def remaining_quota(conn: sqlite3.Connection, *, transport: str = "mastodon", daily_max: int | None = None) -> int:
    """How many posts are still allowed today for ``transport``."""
    if daily_max is None:
        daily_max = DEFAULT_DAILY_MAX
    used = db.posts_today(conn, transport=transport)
    return max(0, daily_max - used)


def build_post_text(row: sqlite3.Row) -> str:
    """Assemble the post body from a queued row.

    The editor-authored ``post_copy`` is the primary text (<= char limit). The
    'learn more' link is appended only when present — on Mastodon a link in the
    post body is free (unlike X, where it is the $0.20 write), so we include it.
    """
    text = (row["post_copy"] or "").strip()
    link = row["source_link"]
    if link:
        # Mastodon link in body is free; keep it on its own line.
        text = f"{text}\n\n{link}"
    return text


def publish_due(
    conn: sqlite3.Connection,
    *,
    transport: str = "mastodon",
    dry_run: bool = False,
    limit: int | None = None,
    daily_max: int | None = None,
    client: Transport | None = None,
    data_dir: str | None = None,
) -> list[dict]:
    """Publish queued, unpublished posts up to the daily quota.

    Returns a list of result dicts, one per post attempted, each carrying at
    least ``draft_id`` and ``status`` ('published' | 'skipped' | 'failed' |
    'dry-run'). With ``dry_run=True`` nothing is sent and the DB is unchanged.

    ``client`` lets callers/tests inject a transport; otherwise
    ``resilient_transport`` resolves one (raising ``PublishUnavailable`` when
    unconfigured — callers must catch it).
    """
    quota = remaining_quota(conn, transport=transport, daily_max=daily_max)
    rows = db.queued_unpublished(conn)
    if limit is not None:
        rows = rows[:limit]
    # Apply the quota last so a dry run still shows the full intended set.
    planned = rows[: quota if not dry_run else len(rows)]

    results: list[dict] = []
    for row in planned:
        draft_id = row["draft_id"]
        text = build_post_text(row)
        image_path = row["image_path"]
        # Resolve a concrete path: image_path is stored relative to the data dir.
        if image_path and data_dir:
            candidate = Path(data_dir) / image_path
            image_path = str(candidate) if candidate.is_file() else None

        if dry_run:
            results.append({
                "draft_id": draft_id,
                "status": "dry-run",
                "text": text,
                "has_image": bool(image_path),
            })
            continue

        try:
            if client is None:
                client = resilient_transport(transport=transport)
        except PublishUnavailable as exc:
            results.append({"draft_id": draft_id, "status": "skipped", "reason": str(exc)})
            continue

        try:
            result = client.send(text=text, image_path=image_path)
            url = result.get("url")
            post_id = result.get("id")
            db.add_post(conn, draft_id=draft_id, transport=transport, url=url, post_id=post_id)
            db.mark_queue_published(conn, draft_id)
            results.append({"draft_id": draft_id, "status": "published", "url": url})
        except PublishUnavailable as exc:
            results.append({"draft_id": draft_id, "status": "failed", "reason": str(exc)})

    return results
