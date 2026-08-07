"""Command-line interface for humorhist.

Subcommands:
    harvest   populate the candidate pool from seed CSV + Wikipedia lists
    screen    LLM-score unscored pool candidates for funniness
    draft     fact-check + generate comic angles for top candidates
    status    show pool/draft/queue health
    show      print a single draft in full (for eyeballing angle quality)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import humorhist.db as db
import humorhist.render as render
from humorhist.buffer import ESCALATE_THRESHOLD, NUDGE_THRESHOLD
import humorhist.env as env  # noqa: F401  (loads local .env into os.environ)

env.load_env()

DEFAULT_DB = os.environ.get(
    "HUMORHIST_DB", str(Path.home() / "projects" / "humorhist" / "data" / "humorhist.sqlite")
)


def _open_db(path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    conn = db.connect(path)
    db.migrate(conn)
    return conn


def cmd_harvest(args: argparse.Namespace) -> int:
    from humorhist.harvest.seed import load_seed
    from humorhist.harvest.wikipedia_lists import harvest_wikipedia_lists

    conn = _open_db(args.db)

    if not args.wikipedia_only:
        print("Loading seed events...")
        print("  seed:", load_seed(conn))

    if not args.seed_only:
        print("Harvesting Wikipedia lists (this takes a moment)...")
        print("  wikipedia:", harvest_wikipedia_lists(conn))

    print("\nPool totals:", db.counts(conn))
    return 0


def cmd_screen(args: argparse.Namespace) -> int:
    from humorhist.harvest.screen import screen_pool
    from humorhist.llm import default_client

    conn = _open_db(args.db)
    client = default_client()
    print(f"Screening unscored candidates (batch={args.batch_size}, limit={args.limit})...")
    result = screen_pool(conn, client, batch_size=args.batch_size, limit=args.limit)
    print("Result:", result)

    cur = conn.execute(
        "SELECT count(*) AS n, round(avg(funny_score),2) AS avg FROM pool WHERE funny_score IS NOT NULL"
    )
    row = cur.fetchone()
    print(f"Scored so far: {row['n']} (avg {row['avg']})")
    return 0


def cmd_draft(args: argparse.Namespace) -> int:
    from humorhist.drafting import draft_candidates
    from humorhist.llm import default_client

    conn = _open_db(args.db)
    client = default_client()
    print(f"Drafting {args.count} candidates (min score {args.min_score})...")
    result = draft_candidates(
        conn, client, count=args.count, min_score=args.min_score
    )
    print("Result:", result)
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    conn = _open_db(args.db)
    c = db.counts(conn)
    print("=== humorhist status ===")
    print(f"DB: {args.db}\n")
    print("Row counts:", c)

    bands = conn.execute(
        """
        SELECT
          sum(CASE WHEN funny_score IS NULL THEN 1 ELSE 0 END) AS unscored,
          sum(CASE WHEN funny_score >= 8 THEN 1 ELSE 0 END) AS band8,
          sum(CASE WHEN funny_score >= 7 AND funny_score < 8 THEN 1 ELSE 0 END) AS band7,
          sum(CASE WHEN funny_score >= 5 AND funny_score < 7 THEN 1 ELSE 0 END) AS band5,
          sum(CASE WHEN funny_score < 5 THEN 1 ELSE 0 END) AS below5
        FROM pool
        """
    ).fetchone()
    print("\nPool by score band:")
    print(f"  unscored : {bands['unscored'] or 0}")
    print(f"  >= 8     : {bands['band8'] or 0}")
    print(f"  7 - 8    : {bands['band7'] or 0}")
    print(f"  5 - 7    : {bands['band5'] or 0}")
    print(f"  < 5      : {bands['below5'] or 0}")

    drafts = conn.execute(
        "SELECT status, count(*) AS n FROM drafts GROUP BY status"
    ).fetchall()
    print("\nDrafts by status:")
    if not drafts:
        print("  (none yet)")
    for d in drafts:
        print(f"  {d['status']:10s}: {d['n']}")

    queued = conn.execute(
        "SELECT count(*) AS n FROM queue WHERE published = 0"
    ).fetchone()["n"]
    print(f"\nApproved and queued (unpublished): {queued}")
    if queued < ESCALATE_THRESHOLD:
        print("  ** BUFFER CRITICAL ** draft more / review now")
    elif queued < NUDGE_THRESHOLD:
        print("  ** BUFFER LOW ** run a review session soon")
    else:
        print("  (healthy — a week+ of buffer)")
    return 0


def cmd_buffer(args: argparse.Namespace) -> int:
    """Phase 3.4: report buffer health and auto-draft if running low.

    With no flags it only reports. With --auto-draft it tops up candidates when
    pending drafts are scarce (needs an LLM key). With --notify it DMs the
    report to Telegram when the buffer is low (nudge/escalate).
    """
    import humorhist.buffer as buf

    conn = _open_db(args.db)
    telegram = None
    chat_id = args.chat_id or os.environ.get("HUMORHIST_TELEGRAM_CHAT_ID")
    if args.notify:
        if not os.environ.get("HUMORHIST_TELEGRAM_BOT_TOKEN"):
            print("error: --notify needs HUMORHIST_TELEGRAM_BOT_TOKEN")
            return 2
        if not chat_id:
            print("error: --notify needs --chat-id or HUMORHIST_TELEGRAM_CHAT_ID")
            return 2
        import humorhist.telegram as tg

        telegram = tg.TelegramClient()

    client = None
    if args.auto_draft:
        from humorhist.llm import default_client

        client = default_client()

    result = buf.run_buffer_check(
        conn,
        client=client if args.auto_draft else None,
        auto_draft=args.auto_draft,
        chat_id=chat_id,
        telegram=telegram,
    )
    will_draft = bool(args.auto_draft and client is not None)
    print(buf.health_message(result, will_draft=will_draft))
    if result.get("drafted"):
        print(f"Auto-drafted {result['drafted']} candidate(s).")
    if result.get("draft_error"):
        print(f"(auto-draft error: {result['draft_error']})")
    if result.get("notified"):
        print("(low-buffer alert sent to Telegram)")
    return 0


def render_draft(row, pool=None) -> str:
    """Back-compat alias for humorhist.render.render_draft.

    Kept so external callers (and the Telegram transport) can import the
    renderer from either module.
    """
    return render.render_draft(row, pool)


def _render_draft_with_pool(row, pool) -> str:
    return render.render_draft(row, pool)


def cmd_show(args: argparse.Namespace) -> int:
    conn = _open_db(args.db)
    if args.draft_id:
        row = conn.execute(
            "SELECT * FROM drafts WHERE id = ?", (args.draft_id,)
        ).fetchone()
    else:
        row = conn.execute(
            "SELECT * FROM drafts ORDER BY created_at DESC LIMIT 1"
        ).fetchone()

    if row is None:
        print("No draft found.")
        return 1

    pool = db.get_pool_item(conn, row["pool_id"])
    print(_render_draft_with_pool(row, pool))
    return 0


def cmd_review(args: argparse.Namespace) -> int:
    """Interactive Phase 3 review loop over pending drafts.

    For each pending draft: render it, then prompt for a decision
    (approve/reject/skip) and optional editor line + notes. Decisions are
    persisted via ``humorhist.review.apply_review``. 'skip' leaves the draft
    pending and moves on.
    """
    import humorhist.review as review

    conn = _open_db(args.db)
    pending = review.pending_drafts(conn)

    if not pending:
        print("Nothing to review — no pending drafts.")
        return 0

    print(f"{len(pending)} pending draft(s) to review.\n")

    acted = 0
    for row in pending:
        pool = db.get_pool_item(conn, row["pool_id"])
        print(_render_draft_with_pool(row, pool))
        print("-" * 70)

        decision = _prompt_choice(
            "Decision [a/r/s/l] (approve/reject/skip/later)",
            {"a": "approve", "r": "reject", "s": "skip", "l": "later"},
        )
        if decision == "skip":
            print("  skipped\n")
            continue
        if decision == "later":
            review.defer_draft(conn, row["id"])
            print("  -> deferred 30 days\n")
            continue

        editor_line = input("  Editor line (optional, Enter to skip): ").strip()
        notes = input("  Notes (optional, Enter to skip): ").strip()

        review.apply_review(
            conn,
            row["id"],
            decision=decision,
            editor_line=editor_line or None,
            notes=notes or None,
        )
        acted += 1
        print(f"  -> {decision}ed")

        # B+ handoff: on approve, generate initial post copy onto the queue row
        # so the editor can open + revise it before any publishing step. Skipped
        # silently when no LLM key is available (manual run, no Hermes session).
        if decision == "approve":
            from humorhist.copywriter import fill_post_copy
            from humorhist.llm import default_client

            try:
                n = fill_post_copy(conn, default_client(), draft_id=row["id"])
                if n:
                    print("  generated initial post copy (editable via `humorhist copy`)")
            except Exception as exc:  # noqa: BLE001 - don't fail the review on copy gen
                print(f"  (post copy not generated: {exc})")
        print()

    print(f"Review session done. {acted} draft(s) decided.")
    return 0


def cmd_suggest(args: argparse.Namespace) -> int:
    """Add an editor-suggested event/topic to the pool (plan 3.2 /suggest).

    Suggested items enter with status 'new' and a NULL score so they flow
    through the normal draft pipeline on the next harvest/draft pass.
    """
    import humorhist.db as db

    conn = _open_db(args.db)
    pool_id = db.add_suggested_pool_item(
        conn,
        title=args.topic,
        note=args.note,
        source_url=args.source_url,
        year=args.year,
    )
    print(f"Suggested '{args.topic}' added to the pool (id {pool_id[:8]}…).")
    print("It will be drafted in a future harvest/draft pass.")
    return 0


def _prompt_choice(prompt: str, mapping: dict[str, str]) -> str:
    """Loop until the user enters a key present in ``mapping``."""
    while True:
        ans = input(f"{prompt}: ").strip().lower()
        if ans in mapping:
            return mapping[ans]
        print(f"  enter one of: {', '.join(mapping)}")


def cmd_telegram_review(args: argparse.Namespace) -> int:
    """Run the Telegram review loop (Phase 3.3).

    Sends pending drafts to Telegram with Approve/Reject buttons and processes
    taps. With --once it processes queued updates and exits (one-shot); without
    it, it long-polls forever (run as a durable systemd --user unit).
    """
    import humorhist.telegram as tg

    chat_id = args.chat_id or os.environ.get("HUMORHIST_TELEGRAM_CHAT_ID")
    if not os.environ.get("HUMORHIST_TELEGRAM_BOT_TOKEN"):
        print("error: HUMORHIST_TELEGRAM_BOT_TOKEN is not set")
        return 2
    if not chat_id:
        print("error: need --chat-id or HUMORHIST_TELEGRAM_CHAT_ID")
        return 2

    conn = _open_db(args.db)
    client = tg.TelegramClient()
    if args.once:
        decided = tg.run_review_bot(conn, client, chat_id, once=True)
        print(f"Telegram review (once): {decided} decision(s) processed.")
        return 0
    print("Telegram review loop started (Ctrl-C to stop)...")
    tg.run_review_bot(conn, client, chat_id)
    return 0


def cmd_notify(args: argparse.Namespace) -> int:
    """Send a Telegram nudge with the current pending-draft count (Phase 3.4)."""
    import humorhist.telegram as tg

    chat_id = args.chat_id or os.environ.get("HUMORHIST_TELEGRAM_CHAT_ID")
    if not os.environ.get("HUMORHIST_TELEGRAM_BOT_TOKEN"):
        print("error: HUMORHIST_TELEGRAM_BOT_TOKEN is not set")
        return 2
    if not chat_id:
        print("error: need --chat-id or HUMORHIST_TELEGRAM_CHAT_ID")
        return 2

    conn = _open_db(args.db)
    client = tg.TelegramClient()
    n = tg.notify_new_drafts(conn, client, chat_id)
    print(f"Notified: {n} draft(s) awaiting review.")
    return 0


def cmd_telegram_status(args: argparse.Namespace) -> int:
    """Send a Telegram message listing approved/rejected/pending topics."""
    import humorhist.telegram as tg

    chat_id = args.chat_id or os.environ.get("HUMORHIST_TELEGRAM_CHAT_ID")
    if not os.environ.get("HUMORHIST_TELEGRAM_BOT_TOKEN"):
        print("error: HUMORHIST_TELEGRAM_BOT_TOKEN is not set")
        return 2
    if not chat_id:
        print("error: need --chat-id or HUMORHIST_TELEGRAM_CHAT_ID")
        return 2

    conn = _open_db(args.db)
    client = tg.TelegramClient()
    text = tg.send_reviewed_summary(conn, client, chat_id)
    print(text)
    return 0


def _copy_get_row(conn, draft_id: str):
    """Return (draft_row, queue_row) for a queued draft, or (None, None)."""
    draft = conn.execute(
        "SELECT d.*, p.title AS title, p.year AS year FROM drafts d "
        "LEFT JOIN pool p ON p.id = d.pool_id WHERE d.id = ?",
        (draft_id,),
    ).fetchone()
    if draft is None:
        return None, None
    q = conn.execute(
        "SELECT post_copy, post_copy_at FROM queue WHERE draft_id = ?", (draft_id,)
    ).fetchone()
    return dict(draft), dict(q) if q else None


def cmd_copy_show(args: argparse.Namespace) -> int:
    """B+ : show a queued draft's editable post copy + char count."""
    import humorhist.copywriter as cw

    conn = _open_db(args.db)
    draft, q = _copy_get_row(conn, args.draft_id)
    if draft is None:
        print(f"No such draft: {args.draft_id}")
        return 1
    if draft["status"] != "approved":
        print(f"Draft {args.draft_id} is '{draft['status']}', not approved/queued.")
        return 1
    copy = (q or {}).get("post_copy")
    limit = cw.char_limit()
    if not copy:
        print(f"{args.draft_id} — no post copy yet (generate with `copy regen`).")
        print(f"Limit: {limit} chars")
        return 0
    print(f"{args.draft_id} — {len(copy)}/{limit} chars")
    print("-" * 60)
    print(copy)
    print("-" * 60)
    return 0


def _launch_editor(initial: str) -> str | None:
    """Open $EDITOR (VISUAL/EDITOR, fallback nano) with `initial`; return edited text.

    Returns None if no usable editor is available (caller falls back to prompt).
    """
    import os
    import subprocess
    import tempfile

    editor = os.environ.get("VISUAL") or os.environ.get("EDITOR") or "nano"
    if not editor.strip():
        return None
    with tempfile.NamedTemporaryFile("w+", suffix=".txt", delete=False) as tf:
        tf.write(initial)
        path = tf.name
    try:
        rc = subprocess.run([editor, path], check=False).returncode
        if rc != 0:
            return None
        with open(path) as fh:
            return fh.read()
    finally:
        os.unlink(path)


def cmd_copy_edit(args: argparse.Namespace) -> int:
    """B+ : edit a queued draft's post copy in $EDITOR (or a typed prompt)."""
    import humorhist.copywriter as cw

    conn = _open_db(args.db)
    draft, q = _copy_get_row(conn, args.draft_id)
    if draft is None:
        print(f"No such draft: {args.draft_id}")
        return 1
    if draft["status"] != "approved":
        print(f"Draft {args.draft_id} is '{draft['status']}', not approved/queued.")
        return 1

    current = (q or {}).get("post_copy") or ""
    edited = _launch_editor(current)
    if edited is None:
        # no editor available: fall back to a typed prompt
        print("No $EDITOR available; enter the new copy (blank line to finish):")
        lines = []
        while True:
            line = input()
            if line == "":
                break
            lines.append(line)
        edited = "\n".join(lines)

    # confirm the result is within the active char limit (warn, don't block)
    limit = cw.char_limit()
    if len(edited) > limit:
        print(f"WARNING: copy is {len(edited)} chars, over the {limit} limit.")
    cw.set_post_copy(conn, args.draft_id, edited)
    print(f"Saved {len(edited)}/{limit} chars for {args.draft_id}.")
    return 0


def cmd_copy_regen(args: argparse.Namespace) -> int:
    """B+ : regenerate a queued draft's post copy via the LLM."""
    from humorhist.copywriter import fill_post_copy
    from humorhist.llm import default_client

    conn = _open_db(args.db)
    draft, q = _copy_get_row(conn, args.draft_id)
    if draft is None:
        print(f"No such draft: {args.draft_id}")
        return 1
    if draft["status"] != "approved":
        print(f"Draft {args.draft_id} is '{draft['status']}', not approved/queued.")
        return 1

    try:
        n = fill_post_copy(conn, default_client(), draft_id=args.draft_id, force=True)
    except Exception as exc:  # noqa: BLE001
        print(f"Regeneration failed: {exc}")
        return 2
    if n == 0:
        print(
            "Nothing generated. (If the draft already has copy, this is a no-op; "
            "edit it with `copy edit` or delete the copy first.)"
        )
        return 0
    row = conn.execute(
        "SELECT post_copy FROM queue WHERE draft_id = ?", (args.draft_id,)
    ).fetchone()
    limit = _char_limit()
    print(f"Regenerated ({len(row['post_copy'])}/{limit} chars):")
    print(row["post_copy"])
    return 0


def _char_limit() -> int:
    import humorhist.copywriter as cw

    return cw.char_limit()


def cmd_queue(args: argparse.Namespace) -> int:
    """Phase 4 handoff: list queued drafts, or move approved -> queue with --enqueue."""
    import humorhist.review as review

    conn = _open_db(args.db)
    if args.enqueue:
        n = review.enqueue_approved(conn, scheduled_for=args.scheduled_for)
        print(f"Enqueued {n} approved draft(s) into queue.")
        # B+ handoff: generate initial post copy for anything that just entered
        # the queue (and any other approved+queued row still lacking it).
        from humorhist.copywriter import fill_post_copy
        from humorhist.llm import default_client

        try:
            filled = fill_post_copy(conn, default_client())
            if filled:
                print(f"Generated post copy for {filled} draft(s) (editable via `humorhist copy`).")
        except Exception as exc:  # noqa: BLE001
            print(f"(post copy not generated: {exc})")
    rows = review.queued_drafts(conn)
    if not rows:
        print("Queue is empty.")
        return 0
    print(f"{len(rows)} draft(s) in queue:")
    for r in rows:
        sched = r["scheduled_for"] or "(unscheduled)"
        flag = " [published]" if r["published"] else ""
        print(f"  {r['draft_id']} — {r['title']} — scheduled {sched}{flag}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="humorhist", description=__doc__)
    p.add_argument("--db", default=DEFAULT_DB, help="path to the sqlite database")
    sub = p.add_subparsers(dest="command", required=True)

    h = sub.add_parser("harvest", help="populate the candidate pool")
    h.add_argument("--seed-only", action="store_true")
    h.add_argument("--wikipedia-only", action="store_true")
    h.set_defaults(func=cmd_harvest)

    s = sub.add_parser("screen", help="LLM-score unscored candidates")
    s.add_argument("--batch-size", type=int, default=20)
    s.add_argument("--limit", type=int, default=None)
    s.set_defaults(func=cmd_screen)

    d = sub.add_parser("draft", help="fact-check + generate angles")
    d.add_argument("--count", type=int, default=3)
    d.add_argument("--min-score", type=float, default=7.0)
    d.set_defaults(func=cmd_draft)

    st = sub.add_parser("status", help="pool/draft/queue health")
    st.set_defaults(func=cmd_status)

    bf = sub.add_parser("buffer", help="Phase 3.4: buffer health + auto-draft")
    bf.add_argument("--auto-draft", action="store_true", help="top up candidates when pending is low")
    bf.add_argument("--notify", action="store_true", help="DM a low-buffer alert to Telegram")
    bf.add_argument("--chat-id", default=None, help="Telegram chat id to DM")
    bf.set_defaults(func=cmd_buffer)

    sh = sub.add_parser("show", help="print a draft in full")
    sh.add_argument("draft_id", nargs="?", default=None)
    sh.set_defaults(func=cmd_show)

    rv = sub.add_parser("review", help="Phase 3: approve/reject pending drafts")
    rv.set_defaults(func=cmd_review)

    sg = sub.add_parser("suggest", help="add an editor-suggested event to the pool")
    sg.add_argument("topic", help="the event/topic to suggest")
    sg.add_argument("--note", default=None, help="optional context/steering note")
    sg.add_argument("--source-url", default=None, help="optional source URL")
    sg.add_argument("--year", type=int, default=None, help="optional year")
    sg.set_defaults(func=cmd_suggest)

    tr = sub.add_parser("telegram-review", help="Phase 3.3: Telegram review loop")
    tr.add_argument("--chat-id", default=None, help="Telegram chat id to DM")
    tr.add_argument("--once", action="store_true", help="process queued updates and exit")
    tr.set_defaults(func=cmd_telegram_review)

    nt = sub.add_parser("notify", help="Phase 3.4: Telegram nudge with pending count")
    nt.add_argument("--chat-id", default=None, help="Telegram chat id to DM")
    nt.set_defaults(func=cmd_notify)

    ts = sub.add_parser("telegram-status", help="DM reviewed/pending topic breakdown")
    ts.add_argument("--chat-id", default=None, help="Telegram chat id to DM")
    ts.set_defaults(func=cmd_telegram_status)

    q = sub.add_parser("queue", help="Phase 4: list queued drafts (--enqueue to fill)")
    q.add_argument("--enqueue", action="store_true", help="move approved drafts into queue")
    q.add_argument("--scheduled-for", default=None, help="ISO timestamp to schedule under")
    q.set_defaults(func=cmd_queue)

    cp = sub.add_parser("copy", help="B+ : view/edit/regenerate a draft's post copy")
    cp_sub = cp.add_subparsers(dest="copy_command", required=True)
    cps = cp_sub.add_parser("show", help="print the post copy + char count")
    cps.add_argument("draft_id")
    cps.set_defaults(func=cmd_copy_show)
    cpe = cp_sub.add_parser("edit", help="edit the post copy in $EDITOR")
    cpe.add_argument("draft_id")
    cpe.set_defaults(func=cmd_copy_edit)
    cpr = cp_sub.add_parser("regen", help="regenerate post copy via the LLM")
    cpr.add_argument("draft_id")
    cpr.set_defaults(func=cmd_copy_regen)

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
