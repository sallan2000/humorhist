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
    if queued < 3:
        print("  ** BUFFER LOW ** run a review session soon")
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
            "Decision [a/r/s] (approve/reject/skip)", {"a": "approve", "r": "reject", "s": "skip"}
        )
        if decision == "skip":
            print("  skipped\n")
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
        print(f"  -> {decision}ed\n")

    print(f"Review session done. {acted} draft(s) decided.")
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

    sh = sub.add_parser("show", help="print a draft in full")
    sh.add_argument("draft_id", nargs="?", default=None)
    sh.set_defaults(func=cmd_show)

    rv = sub.add_parser("review", help="Phase 3: approve/reject pending drafts")
    rv.set_defaults(func=cmd_review)

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

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
