"""Tests for humorhist.cli -- argument parsing and the read-only commands.

Only ``status`` and ``show`` are exercised end-to-end; harvest/screen/draft
touch the network and are covered by their own modules' tests.
"""

from __future__ import annotations

import argparse
import json

import pytest

import humorhist.db as db
from humorhist.cli import build_parser, cmd_review, main


@pytest.fixture()
def dbpath(tmp_path):
    p = tmp_path / "cli.sqlite"
    conn = db.connect(str(p))
    db.migrate(conn)
    conn.close()
    return str(p)


BRIEF = {
    "verified_facts": ["Soldiers were issued Lewis machine guns."],
    "dates": {"event": "November 1932", "precision": "month"},
    "key_figures": ["Sir George Pearce"],
    "caveats": ["Kill counts are disputed."],
    "misconceptions": ["No war was formally declared."],
    "sources": [
        {"title": "Wikipedia: Emu War", "url": "https://en.wikipedia.org/wiki/Emu_War"}
    ],
}

ANGLES = {
    "angles": [
        {
            "angle_name": "MILITARY INCOMPETENCE",
            "setup": "An army turns up and the enemy refuses to cooperate.",
            "why_it_lands": "Mechanised warfare loses to a bird.",
            "pitfalls": "Do not mock the individual soldiers.",
            "raw_material": ["Lewis guns vs flightless birds"],
        },
        {
            "angle_name": "BUREAUCRACY",
            "setup": "A minister signs off machine guns for a farming complaint.",
            "why_it_lands": "The state treats pests as a military campaign.",
            "pitfalls": "Stay on the paperwork.",
            "raw_material": ["ministerial approval"],
        },
    ],
    "strongest_single_detail": "The military withdrew, beaten by emus.",
    "suggested_hook": "In 1932 Australia sent soldiers to cull emus.",
}


# --- parser -----------------------------------------------------------------

@pytest.mark.parametrize(
    "argv",
    [
        ["harvest"],
        ["harvest", "--seed-only"],
        ["screen"],
        ["screen", "--batch-size", "5"],
        ["draft"],
        ["draft", "--count", "2", "--min-score", "8"],
        ["status"],
        ["show"],
        ["show", "abc123"],
    ],
)
def test_build_parser_subcommands(argv):
    args = build_parser().parse_args(argv)
    assert args.command == argv[0]
    assert callable(args.func)


# --- status -----------------------------------------------------------------

def test_status_on_empty_db(dbpath, capsys):
    rc = main(["--db", dbpath, "status"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "humorhist status" in out
    assert "unscored : 0" in out
    # empty buffer (0 queued) must be flagged as critical under the new thresholds
    assert "BUFFER CRITICAL" in out


def test_status_reports_score_bands(dbpath, capsys):
    conn = db.connect(dbpath)
    for pid, score in [("a", 9), ("b", 7.5), ("c", 6), ("d", None)]:
        db.upsert_pool_item(
            conn,
            id=pid,
            title=f"T{pid}",
            year=1900,
            date_hint=None,
            summary=None,
            source_url=None,
            source_name=None,
        )
        if score is not None:
            db.set_funny_score(conn, pid, score)
    conn.close()

    rc = main(["--db", dbpath, "status"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "unscored : 1" in out
    assert ">= 8     : 1" in out
    assert "7 - 8    : 1" in out
    assert "5 - 7    : 1" in out
    assert "< 5      : 0" in out


# --- show -------------------------------------------------------------------

def test_show_with_no_drafts_returns_1(dbpath, capsys):
    rc = main(["--db", dbpath, "show"])
    out = capsys.readouterr().out
    assert rc == 1
    assert "No draft found." in out


def test_show_renders_a_draft(dbpath, capsys):
    conn = db.connect(dbpath)
    db.upsert_pool_item(
        conn,
        id="emu",
        title="The Emu War",
        year=1932,
        date_hint=None,
        summary="Australia lost to emus.",
        source_url="https://en.wikipedia.org/wiki/Emu_War",
        source_name="wikipedia",
    )
    conn.execute(
        """
        INSERT INTO drafts (id, pool_id, brief_json, angles_json, status, created_at)
        VALUES ('d1', 'emu', ?, ?, 'pending', '2026-01-01T00:00:00+00:00')
        """,
        (json.dumps(BRIEF), json.dumps(ANGLES)),
    )
    conn.commit()
    conn.close()

    rc = main(["--db", dbpath, "show"])
    out = capsys.readouterr().out

    assert rc == 0
    assert "The Emu War" in out
    assert "1932" in out
    assert "MILITARY INCOMPETENCE" in out
    assert "BUREAUCRACY" in out
    assert ANGLES["strongest_single_detail"] in out
    assert ANGLES["suggested_hook"] in out
    assert "Wikipedia: Emu War" in out
    assert "https://en.wikipedia.org/wiki/Emu_War" in out
    assert BRIEF["verified_facts"][0] in out
    assert BRIEF["misconceptions"][0] in out
    assert BRIEF["caveats"][0] in out


# --- review (Phase 3) --------------------------------------------------------


def _seed_pending_draft(dbpath, draft_id="d1", pool_id="emu"):
    conn = db.connect(dbpath)
    db.upsert_pool_item(
        conn,
        id=pool_id,
        title="The Emu War",
        year=1932,
        date_hint=None,
        summary="Australia lost to emus.",
        source_url="https://en.wikipedia.org/wiki/Emu_War",
        source_name="wikipedia",
    )
    conn.execute(
        """
        INSERT INTO drafts (id, pool_id, brief_json, angles_json, status, created_at)
        VALUES (?, ?, ?, ?, 'pending', '2026-01-01T00:00:00+00:00')
        """,
        (draft_id, pool_id, json.dumps(BRIEF), json.dumps(ANGLES)),
    )
    conn.commit()
    conn.close()


def _run_review(dbpath, inputs, monkeypatch, capsys):
    """Drive cmd_review with a sequence of simulated stdin lines."""
    import io

    from humorhist.cli import cmd_review

    conn = db.connect(dbpath)
    # cmd_review builds its own connection via _open_db; feed stdin instead.
    monkeypatch.setattr("sys.stdin", io.StringIO("\n".join(inputs) + "\n"))
    # argparse Namespace with the db path
    args = argparse.Namespace(db=dbpath)
    rc = cmd_review(args)
    out = capsys.readouterr().out
    conn.close()
    return rc, out


def test_review_approve_writes_status(dbpath, monkeypatch, capsys):
    _seed_pending_draft(dbpath)
    rc, out = _run_review(dbpath, ["a", "", ""], monkeypatch, capsys)
    assert rc == 0
    conn = db.connect(dbpath)
    row = conn.execute("SELECT status, reviewed_at FROM drafts WHERE id='d1'").fetchone()
    conn.close()
    assert row["status"] == "approved"
    assert row["reviewed_at"] is not None


def test_review_reject_writes_status(dbpath, monkeypatch, capsys):
    _seed_pending_draft(dbpath)
    rc, out = _run_review(dbpath, ["r", "", "too dark"], monkeypatch, capsys)
    assert rc == 0
    conn = db.connect(dbpath)
    row = conn.execute(
        "SELECT status, editor_line, editor_notes FROM drafts WHERE id='d1'"
    ).fetchone()
    conn.close()
    assert row["status"] == "rejected"
    assert row["editor_line"] is None
    assert row["editor_notes"] == "too dark"


def test_review_skip_leaves_pending(dbpath, monkeypatch, capsys):
    _seed_pending_draft(dbpath)
    rc, out = _run_review(dbpath, ["s"], monkeypatch, capsys)
    assert rc == 0
    conn = db.connect(dbpath)
    row = conn.execute("SELECT status FROM drafts WHERE id='d1'").fetchone()
    conn.close()
    assert row["status"] == "pending"


def test_review_empty_queue_exits_clean(dbpath, monkeypatch, capsys):
    rc, out = _run_review(dbpath, [], monkeypatch, capsys)
    assert rc == 0
    assert "no pending" in out.lower() or "nothing to review" in out.lower()


# --- telegram commands (Phase 3.3/3.4) ---------------------------------------


def test_telegram_review_errors_without_token(dbpath, monkeypatch, capsys):
    monkeypatch.delenv("HUMORHIST_TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.delenv("HUMORHIST_TELEGRAM_CHAT_ID", raising=False)
    rc = main(["--db", dbpath, "telegram-review", "--once"])
    out = capsys.readouterr().out
    assert rc == 2
    assert "HUMORHIST_TELEGRAM_BOT_TOKEN" in out


def test_notify_errors_without_token(dbpath, monkeypatch, capsys):
    monkeypatch.delenv("HUMORHIST_TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.delenv("HUMORHIST_TELEGRAM_CHAT_ID", raising=False)
    rc = main(["--db", dbpath, "notify"])
    out = capsys.readouterr().out
    assert rc == 2
    assert "HUMORHIST_TELEGRAM_BOT_TOKEN" in out


def test_telegram_status_errors_without_token(dbpath, monkeypatch, capsys):
    monkeypatch.delenv("HUMORHIST_TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.delenv("HUMORHIST_TELEGRAM_CHAT_ID", raising=False)
    rc = main(["--db", dbpath, "telegram-status"])
    out = capsys.readouterr().out
    assert rc == 2
    assert "HUMORHIST_TELEGRAM_BOT_TOKEN" in out


def test_queue_enqueue_moves_approved(dbpath, monkeypatch, capsys):
    import humorhist.db as db

    conn = db.connect(dbpath)
    conn.execute("INSERT OR IGNORE INTO pool (id, title) VALUES ('p1','X')")
    conn.execute(
        "INSERT INTO drafts (id, pool_id, brief_json, angles_json, status, created_at) "
        "VALUES ('d1','p1','{}','{}','approved','2026-01-01T00:00:00+00:00')"
    )
    conn.commit()
    # list (empty)
    rc = main(["--db", dbpath, "queue"])
    assert rc == 0
    assert "Queue is empty" in capsys.readouterr().out
    # enqueue
    rc = main(["--db", dbpath, "queue", "--enqueue"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "Enqueued 1" in out
    # now listed
    rc = main(["--db", dbpath, "queue"])
    out = capsys.readouterr().out
    assert "d1" in out
    # idempotent: second enqueue adds nothing
    rc = main(["--db", dbpath, "queue", "--enqueue"])
    out = capsys.readouterr().out
    assert "Enqueued 0" in out


# --- B+ copy subcommand (view / edit / regen) -------------------------------


def _seed_approved_with_queue(dbpath):
    conn = db.connect(dbpath)
    db.upsert_pool_item(
        conn,
        id="p1",
        title="The Emu War",
        year=1932,
        date_hint=None,
        summary=None,
        source_url=None,
        source_name=None,
    )
    conn.execute(
        "INSERT INTO drafts (id, pool_id, brief_json, angles_json, status, created_at) "
        "VALUES ('d1','p1',?,?,'approved','2026-01-01T00:00:00+00:00')",
        (json.dumps(BRIEF), json.dumps(ANGLES)),
    )
    conn.execute(
        "INSERT INTO queue (draft_id, scheduled_for, published, post_copy) "
        "VALUES ('d1', NULL, 0, 'France invaded Mexico over a pastry shop.')"
    )
    conn.commit()
    conn.close()


def test_copy_show_prints_copy_and_count(dbpath, capsys):
    from humorhist.cli import cmd_copy_show

    _seed_approved_with_queue(dbpath)
    rc = cmd_copy_show(argparse.Namespace(db=dbpath, draft_id="d1"))
    out = capsys.readouterr().out
    assert rc == 0
    assert "France invaded Mexico over a pastry shop." in out
    assert "280" in out  # char limit shown


def test_copy_show_rejects_non_approved(dbpath, capsys):
    from humorhist.cli import cmd_copy_show

    _seed_pending_draft(dbpath)  # pending, not approved
    rc = cmd_copy_show(argparse.Namespace(db=dbpath, draft_id="d1"))
    assert rc == 1
    assert "pending" in capsys.readouterr().out


def test_copy_edit_launches_editor_and_saves(dbpath, monkeypatch, capsys):
    from humorhist import cli as cli_mod

    _seed_approved_with_queue(dbpath)

    def fake_editor(initial):
        return "My hand-edited, funnier version."

    monkeypatch.setattr(cli_mod, "_launch_editor", fake_editor)
    rc = cli_mod.cmd_copy_edit(argparse.Namespace(db=dbpath, draft_id="d1"))
    assert rc == 0
    conn = db.connect(dbpath)
    row = conn.execute("SELECT post_copy FROM queue WHERE draft_id='d1'").fetchone()
    conn.close()
    assert row["post_copy"] == "My hand-edited, funnier version."


def test_copy_edit_falls_back_to_prompt_without_editor(dbpath, monkeypatch, capsys):
    from humorhist import cli as cli_mod

    _seed_approved_with_queue(dbpath)
    monkeypatch.setattr(cli_mod, "_launch_editor", lambda initial: None)
    import io

    monkeypatch.setattr("sys.stdin", io.StringIO("Typed-in replacement.\n\n"))
    rc = cli_mod.cmd_copy_edit(argparse.Namespace(db=dbpath, draft_id="d1"))
    assert rc == 0
    conn = db.connect(dbpath)
    row = conn.execute("SELECT post_copy FROM queue WHERE draft_id='d1'").fetchone()
    conn.close()
    assert row["post_copy"] == "Typed-in replacement."


def test_copy_regen_regenerates(dbpath, monkeypatch, capsys):
    from humorhist import cli as cli_mod
    from humorhist.copywriter import fill_post_copy
    from humorhist.llm import StubClient

    _seed_approved_with_queue(dbpath)
    real_fill = fill_post_copy

    def fake_fill(conn, client, draft_id=None, limit=None, **kwargs):
        return real_fill(
            conn,
            StubClient([{"post": "A fresh regeneration from the model."}]),
            draft_id=draft_id,
            **kwargs,
        )

    monkeypatch.setattr("humorhist.copywriter.fill_post_copy", fake_fill)
    import humorhist.llm as llm

    monkeypatch.setattr(llm, "default_client", lambda: StubClient([{"post": "x"}]))

    rc = cli_mod.cmd_copy_regen(argparse.Namespace(db=dbpath, draft_id="d1"))
    assert rc == 0
    conn = db.connect(dbpath)
    row = conn.execute("SELECT post_copy FROM queue WHERE draft_id='d1'").fetchone()
    conn.close()
    assert row["post_copy"] == "A fresh regeneration from the model."


def test_copy_parser_registered():
    from humorhist.cli import build_parser

    parser = build_parser()
    # 'copy show d1' parses into the show subcommand
    args = parser.parse_args(["copy", "show", "d1"])
    assert args.copy_command == "show"
    assert args.draft_id == "d1"
    assert callable(args.func)
