"""Tests for humorhist.harvest.seed — the seed CSV loader.

TDD: the tests are written before the implementation, so the first run
should fail with ModuleNotFoundError / ImportError. A real file-backed DB
via tmp_path is used to test actual persistence (no in-memory DB).
"""

from __future__ import annotations

from pathlib import Path

import humorhist.db as db
from humorhist.harvest import seed


def _fresh_db(tmp_path: Path):
    path = tmp_path / "test.sqlite"
    conn = db.connect(str(path))
    db.migrate(conn)
    return conn


def _write_csv(path: Path, rows: list[str]) -> None:
    """Write a seed_events.csv-shaped file. `rows` are raw data lines;
    the header is added automatically."""
    header = "title,year,summary,source_url\n"
    path.write_text(header + "\n".join(rows) + "\n", encoding="utf-8")


def test_load_seed_inserts_rows(tmp_path):
    csv_path = tmp_path / "seed.csv"
    _write_csv(
        csv_path,
        [
            "The Emu War,1932,Australia lost to emus.,https://example.com/emu",
            "War of the Oaken Bucket,1325,Tiny war over a bucket.,https://example.com/bucket",
            "The Cadaver Synod,897,Pope tried a corpse.,https://example.com/synod",
        ],
    )
    conn = _fresh_db(tmp_path)
    summary = seed.load_seed(conn, csv_path)
    assert summary["total_rows"] == 3
    assert summary["inserted"] == 3
    assert summary["skipped_duplicate"] == 0
    assert summary["skipped_invalid"] == 0
    assert db.counts(conn)["pool"] == 3


def test_load_seed_idempotent(tmp_path):
    csv_path = tmp_path / "seed.csv"
    _write_csv(
        csv_path,
        [
            "The Emu War,1932,Australia lost to emus.,https://example.com/emu",
            "War of the Oaken Bucket,1325,Tiny war over a bucket.,https://example.com/bucket",
            "The Cadaver Synod,897,Pope tried a corpse.,https://example.com/synod",
        ],
    )
    conn = _fresh_db(tmp_path)
    first = seed.load_seed(conn, csv_path)
    assert first["inserted"] == 3
    assert db.counts(conn)["pool"] == 3

    second = seed.load_seed(conn, csv_path)
    assert second["inserted"] == 0
    assert second["skipped_duplicate"] == 3
    assert db.counts(conn)["pool"] == 3


def test_load_seed_skips_invalid(tmp_path):
    csv_path = tmp_path / "seed.csv"
    _write_csv(
        csv_path,
        [
            "Valid Event,2000,Has a year.,https://example.com/ok",
            ",2001,Missing title.,https://example.com/notitle",
            "Bad Year Event,notayear,Year not an int.,https://example.com/badyear",
        ],
    )
    conn = _fresh_db(tmp_path)
    summary = seed.load_seed(conn, csv_path)
    assert summary["total_rows"] == 3
    assert summary["inserted"] == 1
    assert summary["skipped_invalid"] == 2
    assert db.counts(conn)["pool"] == 1


def test_load_seed_sets_source_name_and_null_date_hint(tmp_path):
    csv_path = tmp_path / "seed.csv"
    _write_csv(
        csv_path,
        [
            "The Emu War,1932,Australia lost to emus.,https://example.com/emu",
        ],
    )
    conn = _fresh_db(tmp_path)
    summary = seed.load_seed(conn, csv_path)
    assert summary["inserted"] == 1

    item_id = db.make_id("seed", "The Emu War")
    row = db.get_pool_item(conn, item_id)
    assert row is not None
    assert row["source_name"] == "seed"
    assert row["date_hint"] is None
    assert row["year"] == 1932


def test_load_seed_real_file(tmp_path):
    """Load the REAL curated data/seed_events.csv and prove it is clean."""
    conn = _fresh_db(tmp_path)
    summary = seed.load_seed(conn)  # default path
    assert summary["inserted"] >= 100
    assert summary["skipped_invalid"] == 0
    assert summary["inserted"] == db.counts(conn)["pool"]
