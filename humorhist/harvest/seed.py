"""Seed event loader for the humorhist pool.

Reads the hand-curated ``data/seed_events.csv`` and upserts each row into the
pool table. The loader is fully idempotent: re-running it never duplicates
rows or overwrites existing scores/status, because ids are a stable sha1 of
``("seed", title)``.

CSV shape (header is exactly)::

    title,year,summary,source_url

Only stdlib ``csv`` is used.
"""

from __future__ import annotations

import csv
import sqlite3
from pathlib import Path

import humorhist.db as db

# Path to the packaged seed file, computed relative to this file so the loader
# works regardless of the current working directory.
# File lives at <repo_root>/humorhist/harvest/seed.py, so repo root is three
# levels up from __file__.
_DEFAULT_CSV = Path(__file__).resolve().parent.parent.parent / "data" / "seed_events.csv"


def load_seed(
    conn: sqlite3.Connection,
    csv_path: str | Path | None = None,
) -> dict:
    """Load seed events from a CSV into the pool table.

    Parameters
    ----------
    conn:
        An open, migrated sqlite3 connection (see :func:`humorhist.db.connect`
        and :func:`humorhist.db.migrate`).
    csv_path:
        Path to the seed CSV. If ``None``, the packaged ``data/seed_events.csv``
        at the repository root is used.

    Returns
    -------
    dict
        ``{"total_rows", "inserted", "skipped_duplicate", "skipped_invalid"}``.

    Behaviour
    ---------
    * Each pool id is ``make_id("seed", title)`` (stable, idempotent).
    * ``source_name`` is the literal ``"seed"``; ``date_hint`` is ``None``.
    * ``year`` is parsed to ``int``; empty/unparseable years and empty/missing
      titles cause the row to be counted as ``skipped_invalid`` and dropped
      without aborting the run.
    """
    path = Path(csv_path) if csv_path is not None else _DEFAULT_CSV

    summary: dict[str, int] = {
        "total_rows": 0,
        "inserted": 0,
        "skipped_duplicate": 0,
        "skipped_invalid": 0,
    }

    with path.open(encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            summary["total_rows"] += 1
            title = (row.get("title") or "").strip()
            if not title:
                summary["skipped_invalid"] += 1
                continue

            year_raw = (row.get("year") or "").strip()
            try:
                year = int(year_raw)
            except (ValueError, TypeError):
                summary["skipped_invalid"] += 1
                continue

            item_id = db.make_id("seed", title)
            inserted = db.upsert_pool_item(
                conn,
                id=item_id,
                title=title,
                year=year,
                date_hint=None,
                summary=(row.get("summary") or "").strip() or None,
                source_url=(row.get("source_url") or "").strip() or None,
                source_name="seed",
            )
            if inserted:
                summary["inserted"] += 1
            else:
                summary["skipped_duplicate"] += 1

    return summary
