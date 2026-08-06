"""Tests for humorhist.drafting -- candidate selection and draft assembly.

No network and no real LLM calls: ``fetch_wikipedia_extract`` is monkeypatched
and the LLM is always ``StubClient``.
"""

from __future__ import annotations

import json
import random

import pytest

import humorhist.db as db
import humorhist.drafting as drafting
from humorhist.factcheck import FactCheckError
from humorhist.llm import StubClient

# --- fixtures ---------------------------------------------------------------

EXTRACT = "In 1932 the Australian military attempted to cull emus and failed."


def _brief() -> dict:
    return {
        "verified_facts": [
            "In late 1932 Australia sanctioned a military cull of emus.",
            "Soldiers were armed with Lewis machine guns.",
            "The operation was withdrawn after several weeks.",
        ],
        "dates": {"event": "November 1932", "precision": "month"},
        "key_figures": ["Sir George Pearce", "Major G.P.W. Meredith"],
        "caveats": ["Kill counts vary by source."],
        "misconceptions": ["No formal war was declared on the emus."],
        "sources": [
            {"title": "Wikipedia: Emu War", "url": "https://en.wikipedia.org/wiki/Emu_War"}
        ],
    }


def _angle(name: str) -> dict:
    return {
        "angle_name": name,
        "setup": f"Setup for {name} that is long enough to read as a real setup.",
        "why_it_lands": f"The incongruity behind {name} is the gap between means and result.",
        "pitfalls": f"Do not punch down when using {name}; keep it on the absurdity.",
        "raw_material": [f"raw detail for {name}", "another concrete detail"],
    }


def _angles() -> dict:
    return {
        "angles": [
            _angle("MILITARY INCOMPETENCE"),
            _angle("BUREAUCRACY"),
            _angle("ONE RIDICULOUS DETAIL"),
            _angle("MODERN PARALLEL"),
        ],
        "strongest_single_detail": (
            "The army deployed machine guns and withdrew having failed to beat birds."
        ),
        "suggested_hook": (
            "In 1932 Australia sent soldiers with machine guns to cull emus."
        ),
    }


@pytest.fixture()
def conn(tmp_path):
    c = db.connect(str(tmp_path / "t.sqlite"))
    db.migrate(c)
    yield c
    c.close()


def _add(conn, pid, *, score=None, status="new", title=None):
    db.upsert_pool_item(
        conn,
        id=pid,
        title=title or f"Event {pid}",
        year=1932,
        date_hint=None,
        summary=f"Summary for {pid}",
        source_url=f"https://en.wikipedia.org/wiki/{pid}",
        source_name="wikipedia",
    )
    if score is not None:
        db.set_funny_score(conn, pid, score)
    if status != "new":
        db.set_status(conn, "pool", pid, status)
    return pid


def _stub_extract(monkeypatch, fn=None):
    monkeypatch.setattr(
        drafting,
        "fetch_wikipedia_extract",
        fn or (lambda title_or_url, client=None: EXTRACT),
    )


# --- select_candidates ------------------------------------------------------

def test_select_candidates_respects_min_score(conn):
    _add(conn, "a", score=3)
    _add(conn, "b", score=6)
    _add(conn, "c", score=9)
    rows = drafting.select_candidates(conn, count=5, min_score=7.0)
    assert [r["id"] for r in rows] == ["c"]


def test_select_candidates_excludes_non_new_status(conn):
    _add(conn, "drafted_one", score=9, status="drafted")
    _add(conn, "used_one", score=9.5, status="used")
    _add(conn, "fresh", score=8)
    rows = drafting.select_candidates(conn, count=5, min_score=7.0)
    assert [r["id"] for r in rows] == ["fresh"]


def test_select_candidates_excludes_unscored(conn):
    _add(conn, "unscored")
    _add(conn, "scored", score=8)
    rows = drafting.select_candidates(conn, count=5, min_score=7.0)
    assert [r["id"] for r in rows] == ["scored"]


def test_select_candidates_limit(conn):
    for i in range(10):
        _add(conn, f"p{i}", score=7.0 + i * 0.1)
    rows = drafting.select_candidates(conn, count=2, min_score=7.0)
    assert len(rows) == 2


def test_select_candidates_samples_from_window(conn):
    for i in range(40):
        _add(conn, f"p{i:02d}", score=7.0 + i * 0.05)

    seen: set[str] = set()
    for seed in range(20):
        rows = drafting.select_candidates(
            conn, count=2, min_score=7.0, rng=random.Random(seed)
        )
        assert len(rows) == 2
        seen.update(r["id"] for r in rows)
    # Sampling from a window must not always yield the same top-N.
    assert len(seen) > 2


# --- draft_one --------------------------------------------------------------

def test_draft_one_writes_draft_row(conn, monkeypatch):
    _stub_extract(monkeypatch)
    _add(conn, "emu", score=9)
    row = db.get_pool_item(conn, "emu")
    client = StubClient([_brief(), _angles()])

    draft_id = drafting.draft_one(conn, client, row)

    d = conn.execute("SELECT * FROM drafts WHERE id = ?", (draft_id,)).fetchone()
    assert d is not None
    assert d["pool_id"] == "emu"
    assert d["status"] == "pending"
    brief = json.loads(d["brief_json"])
    angles = json.loads(d["angles_json"])
    assert brief["verified_facts"]
    assert len(angles["angles"]) == 4
    assert db.get_pool_item(conn, "emu")["status"] == "drafted"


def test_draft_one_puts_source_url_in_factcheck_prompt(conn, monkeypatch):
    _stub_extract(monkeypatch)
    _add(conn, "emu", score=9)
    row = db.get_pool_item(conn, "emu")
    client = StubClient([_brief(), _angles()])

    drafting.draft_one(conn, client, row)

    assert "https://en.wikipedia.org/wiki/emu" in client.calls[0]["user"]


def test_draft_one_is_idempotent_by_pool_id(conn, monkeypatch):
    _stub_extract(monkeypatch)
    _add(conn, "emu", score=9)
    row = db.get_pool_item(conn, "emu")

    id1 = drafting.draft_one(conn, StubClient([_brief(), _angles()]), row)
    id2 = drafting.draft_one(conn, StubClient([_brief(), _angles()]), row)

    assert id1 == id2
    n = conn.execute("SELECT count(*) AS n FROM drafts").fetchone()["n"]
    assert n == 1


# --- draft_candidates -------------------------------------------------------

def test_draft_candidates_continues_on_failure(conn, monkeypatch):
    _add(conn, "bad", score=9.9)
    _add(conn, "ok1", score=9.0)
    _add(conn, "ok2", score=8.0)

    def fake_fetch(title_or_url, client=None):
        if "bad" in title_or_url:
            raise FactCheckError("no extract available for 'bad'")
        return EXTRACT

    _stub_extract(monkeypatch, fake_fetch)
    client = StubClient([_brief(), _angles(), _brief(), _angles()])

    result = drafting.draft_candidates(conn, client, count=3, min_score=7.0)

    assert result["selected"] == 3
    assert result["drafted"] == 2
    assert result["failed"] == 1
    assert [f["pool_id"] for f in result["failures"]] == ["bad"]


def test_draft_candidates_returns_summary_shape(conn, monkeypatch):
    _stub_extract(monkeypatch)
    _add(conn, "one", score=9)
    result = drafting.draft_candidates(
        conn, StubClient([_brief(), _angles()]), count=1, min_score=7.0
    )
    assert set(result) == {"selected", "drafted", "failed", "draft_ids", "failures"}
    assert result["drafted"] == 1
    assert len(result["draft_ids"]) == 1
