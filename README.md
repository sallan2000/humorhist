# HumorHist

A pipeline that turns historical oddities into fully fact-checked "comic angle"
briefs for a humorous-history social account. A human still writes the actual
jokes — humorhist does the research and hands you angles, source links, and the
"watch out, the popular version is a myth" flags.

It is a command-line tool, not a service. You run stages in order:

    harvest  ->  screen  ->  draft  ->  (you review)  ->  copy  ->  publish*

    * publishing is not built yet (see "Project status" below).

-------------------------------------------------------------------------------
## What it actually does
-------------------------------------------------------------------------------

- harvest  Gather candidate historical events into a pool (741 curated +
           Wikipedia-list events out of the box).
- screen   Ask an LLM to score each event 0-10 on how absurd/funny it is, so
           drafting can later pick the best ones.
- draft    For the top-scored events: fetch the Wikipedia article, run a
           fact-check pass (separating the documented record from the popular
           myth), then generate several distinct comic angles with setups,
           "why it lands", pitfalls, and raw material you can use.
- show     Print a single draft in full so you can read and judge the angles.
- status   Pool/draft health at a glance.
- copy     Once you approve a draft it lands in the `queue` with a first-draft
           post already generated (≤280 chars by default). `copy show/edit/regen`
           let you read, rewrite, or regenerate that copy in place before any
           destination posting. (See "Phase 4: post copy" below.)

The output of `draft` is a structured brief: verified facts, dates, key
figures, caveats, misconceptions, sources, and 3-5 comic angles. You read it
and write the post yourself.

-------------------------------------------------------------------------------
## Requirements
-------------------------------------------------------------------------------

- Python 3.11+
- The only runtime dependency is `httpx` (tests use `pytest` + `respx`).

-------------------------------------------------------------------------------
## Install
-------------------------------------------------------------------------------

    cd humorhist
    python -m venv .venv && . .venv/bin/activate
    pip install -e ".[dev]"          # includes pytest/respx for running tests

-------------------------------------------------------------------------------
## Configure the LLM
-------------------------------------------------------------------------------

All LLM access is via environment variables (no secrets are stored in the
repo):

    HUMORHIST_LLM_API_KEY     REQUIRED. Your API key for a chat-completions API.
    HUMORHIST_LLM_BASE_URL    Optional. Default: https://inference-api.nousresearch.com/v1
    HUMORHIST_LLM_MODEL       Optional. Default: tencent/hy3:free
    HUMORHIST_CHAR_LIMIT      Optional. Max length of generated post copy
                              (default 280). Raise it (e.g. 280 -> 400) if you
                              later post to a platform with a larger limit.

The client speaks the OpenAI-compatible /chat/completions schema, so any
provider that exposes that endpoint works — point BASE_URL at it and set the
key. Example:

    export HUMORHIST_LLM_API_KEY="your-key-here"
    export HUMORHIST_LLM_MODEL="tencent/hy3:free"

-------------------------------------------------------------------------------
## The database
-------------------------------------------------------------------------------

Everything lives in one SQLite file. Default location:

    ~/projects/humorhist/data/humorhist.sqlite

Override per command with `--db /path/to/db.sqlite`. The SQLite file and
runtime logs are git-ignored; only source code is committed.

Two tables:
- pool   — candidate events (title, year, summary, funny_score, status).
- drafts — generated briefs (brief_json, angles_json, status).

-------------------------------------------------------------------------------
## Usage
-------------------------------------------------------------------------------

All commands accept `--db` to point at a specific database.

### 1. Harvest the candidate pool

    python -m humorhist.cli --db data/humorhist.sqlite harvest

Options:
    --seed-only          only load the curated data/seed_events.csv
    --wikipedia-only     only pull from the configured Wikipedia list pages

Idempotent: re-running never duplicates rows or overwrites existing scores.

### 2. Screen (score funniness)

    python -m humorhist.cli --db data/humorhist.sqlite screen

Options:
    --batch-size N       events per LLM call (default 20)
    --limit N            only score the first N unscored events

Only rows with a NULL score are scored, so re-running is safe.

### 3. Draft (fact-check + angles)

    python -m humorhist.cli --db data/humorhist.sqlite draft

Options:
    --count N            how many candidates to draft (default 3)
    --min-score F        only consider events scored >= F (default 7.0)

Candidates are sampled from a window 4x larger than --count so repeated runs
don't always pick the same few. Each draft is written with status "pending"
and the source pool row is marked "drafted".

### 4. Review a draft

    python -m humorhist.cli --db data/humorhist.sqlite show          # most recent
    python -m humorhist.cli --db data/humorhist.sqlite show <id>     # specific draft

Prints the verified facts, misconceptions (popular myth vs record), caveats,
sources, and every comic angle. This is the part you read before writing a post.

You can review interactively from the terminal:

    python -m humorhist.cli --db data/humorhist.sqlite review

It walks every `pending` draft, shows it, and prompts `[a/r/s]`
(approve/reject/skip) plus optional editor line + notes.

### 4b. Review from Telegram (optional)

Instead of the terminal, run a Telegram review loop and approve/reject from
your phone with inline buttons, and see at a glance which topics you've already
decided on.

**One-time setup**

1. Create a bot via @BotFather and copy the token it gives you.
2. Find its username (also from @BotFather) and **open that bot in Telegram /
   send it any message** — this registers your chat so the bot can DM you.
3. Get your numeric chat id. After messaging the bot, run:

       curl -s "https://api.telegram.org/bot<TOKEN>/getUpdates" \
         | python3 -c "import sys,json; r=json.load(sys.stdin); print(r['result'][0]['message']['chat']['id'] if r.get('result') else 'message the bot first')"

   (Only one process may poll `getUpdates` at a time — don't run this curl while
   the review loop is running, or it will show empty.)
4. Put the token and chat id in a local `.env` (gitignored — never commit it):

       HUMORHIST_TELEGRAM_BOT_TOKEN=123456:ABC...
       HUMORHIST_TELEGRAM_CHAT_ID=987654321

**Run it**

    python -m humorhist.cli --db data/humorhist.sqlite telegram-review

It long-polls forever — run it as a durable `systemd --user` unit (see
`scripts/telegram_review.py`) so it survives logout:

    systemctl --user status humorhist-telegram-review.service
    journalctl --user -u humorhist-telegram-review.service -f

The bot **idles** until you send a command; it does **not** push drafts on
startup.

**Commands**

    /reviewdraft   start reviewing pending drafts one by one (Approve/Reject + notes)
    /listapproved  list every draft you've greenlit; tap one to open its content
    /status        approved / rejected / pending breakdown
    /help          this list

**What you do in Telegram**

1. Send **/reviewdraft** to begin. The bot DM's a **📊 Review progress** block
   first: ✅ approved topics, ❌ rejected topics, and ⏳ how many are still
   pending — so you can see at a glance what's already been decided.
2. Then it sends **one draft at a time**. A draft may arrive as a few short
   messages (Telegram caps a message at 4096 chars, so long drafts are split);
   the **✅ Approve** / **❌ Reject** buttons are on the last message of that
   draft. Tap one.
3. Tapping a button saves the decision to the database immediately. The bot then
   asks for **optional notes** — type a short edit note and send it, or reply
   `/skip` to leave no note. (The note is attached to the draft as
   `editor_notes`; it does not change the decision.)
4. **Only after you've decided that draft does the next one arrive.** One at a
   time — no wall of drafts. When all are done you get "✅ All caught up". If new
   drafts are generated later, start another /reviewdraft to pick them up.

**Browse and annotate approved drafts**

Send **/listapproved** to see every draft you've greenlit. Tap a draft's
**👁** button to open its full content (rendered just like the review view). The
content message ends with an **✏️ Add notes** button — tap it to add more notes
later: the bot prompts, you reply (or `/skip`), and the note is saved on the
draft. Re-saving notes is idempotent and keeps the draft in the publish queue.

### 4c. Post copy (generate + edit before posting)

When you **approve** a draft (in the terminal review loop *or* from Telegram),
humorhist immediately generates a first-draft post — a short, on-brief caption
sized to `HUMORHIST_CHAR_LIMIT` (default 280) — and stores it on the `queue`
row. You then refine that copy in place; nothing is posted anywhere until the
(deferred) publisher exists.

**CLI**

    python -m humorhist.cli --db data/humorhist.sqlite copy show <id>     # copy + N/280
    python -m humorhist.cli --db data/humorhist.sqlite copy edit <id>     # $EDITOR, or typed prompt
    python -m humorhist.cli --db data/humorhist.sqlite copy regen <id>    # regenerate via LLM

`edit` opens your `$EDITOR` (falls back to a typed prompt if none is
available); it warns (does not block) if your edit exceeds the active limit.
`regen` overwrites the stored copy with a fresh LLM draft.

**Telegram**

    /listqueue     list queued drafts, each with its copy + char count + open button
    /viewcopy <id> open a draft's post copy with inline ✏️ Edit / 🔄 Regenerate

Tap **✏️ Edit** and reply with new copy (or `/cancel` to keep the current);
tap **🔄 Regenerate** to overwrite it with a fresh LLM draft. The stored copy is
updated on `queue.post_copy`.

**One-shot generation**

    python scripts/fill_approved_copy.py

Generates (or, with `--force`, regenerates) post copy for every approved,
unpublished queue row that lacks it. Borrows the Nous OAuth token like the
other scripts. Handy for backfilling copy on existing approved drafts.

-------------------------------------------------------------------------------

To just see the reviewed/pending breakdown without sending drafts:

    python -m humorhist.cli --db data/humorhist.sqlite telegram-status

This DM's the same 📊 Review progress block on demand.

**Nudge when new drafts are generated**

    python -m humorhist.cli --db data/humorhist.sqlite notify

`scripts/weekly_pipeline.py` and `scripts/regen_drafts.py` already call this
automatically when they finish drafting, so you get pinged whenever fresh drafts
are ready to review.

No webhook is used — the bot long-polls `getUpdates`, which matters because the
host sits behind Cloudflare/NAT and does not expose ports. (Only one process may
poll `getUpdates` at a time — don't run the `getUpdates` curl while the review
loop is running, or it will show empty.)

### 5. Health check

    python -m humorhist.cli --db data/humorhist.sqlite status

Shows pool size, score bands, drafts by status, and how full the un-published
buffer is.

-------------------------------------------------------------------------------
## Editorial note (taste filter)
-------------------------------------------------------------------------------

By deliberate decision, humorhist does NOT down-rank or refuse events involving
death, tragedy, or suffering. The screen prompt scores purely on absurdity, and
the fact-check pass does not emit sensitivity flags. That means grim material
(e.g. famine, fatal accidents) is eligible for drafting alongside the silly
stuff. If you want a different line, edit:

    humorhist/harvest/screen.py   -> SCREEN_SYSTEM_PROMPT
    humorhist/factcheck.py        -> FACTCHECK_SYSTEM_PROMPT

The comic-angle prompt that most affects output quality lives in:

    humorhist/brief.py            -> ANGLES_SYSTEM_PROMPT

-------------------------------------------------------------------------------
## Tests
-------------------------------------------------------------------------------

    pytest tests/

209 tests, no network calls (LLM, Wikipedia, and Telegram are stubbed).

-------------------------------------------------------------------------------
## Project status
-------------------------------------------------------------------------------

Built and working: Phases 1-3 — harvest, screen, draft, the review/show
commands, a Telegram review loop (approve/reject drafts from your phone via
inline buttons, plus a notify nudge), the **Phase 4 handoff**: approving a
draft auto-enqueues it (`queue`) **and generates a first-draft post (`queue.post_copy`)**
you can edit in place, and a weekly `systemd` timer discovers new
topics (harvest → screen → draft net-new) without creating duplicates.

The **B+ copy loop** is done (2026-08-06): on approve, post copy is generated
(≤280 chars, `HUMORHIST_CHAR_LIMIT`-driven) and is editable via the CLI
`copy show/edit/regen` commands and Telegram `/listqueue` + `/viewcopy`
(inline ✏️ Edit / 🔄 Regenerate). See "Phase 4: post copy" above.

Not yet built:
- Phase 4 publisher: turning a `queue` row into an actual posted item and
  writing `posts` (auto-post to Mastodon/X vs. just holding copy for you to
  paste — the `queue`/`posts` schema is ready for either). The copy-generation
  half is complete; only the destination is deferred.
- Cross-source topic dedup: dedup is by source page id, not by topic, so the
  same event appearing on two Wikipedia lists yields two pool rows.

Until the publisher exists, "publishing" means: read the approved draft with
`show`, write the post yourself, and it's already in `queue`.

-------------------------------------------------------------------------------
## Scheduling (weekly discovery)
-------------------------------------------------------------------------------

A `systemd --user` timer runs the discovery pipeline automatically:

    systemctl --user status humorhist-weekly.timer      # next run time
    systemctl --user start humorhist-weekly.service     # force a run now
    systemctl --user cat humorhist-weekly.timer         # view the schedule

It fires **weekly** (default Monday 00:00 local; edit `OnCalendar` in
`~/.config/systemd/user/humorhist-weekly.timer` to change). The job
(`scripts/weekly_pipeline.py`) does harvest → screen → draft net-new → best-effort
Telegram nudge. Every phase is idempotent, so re-running never duplicates:
harvest upserts by stable page id, screen scores only unscored rows, and the
draft step skips any pool item that already has a draft (so your reviewed/
approved work is never overwritten). `Linger=yes` keeps it alive after logout.

-------------------------------------------------------------------------------
## Layout
-------------------------------------------------------------------------------

    humorhist/
        cli.py              command-line entry point
        db.py               SQLite schema + access helpers
        llm.py              LLM client (OpenAI-compatible)
        render.py           shared plain-text draft renderer
        review.py           Phase 3 review state machine (transport-agnostic)
        telegram.py         Phase 3.3/3.4 Telegram transport (long-poll)
        copywriter.py       post-copy generation + in-place edit (B+ copy loop)
        harvest/
            seed.py         curated CSV loader
            wikipedia_lists.py   Wikipedia list-page harvester
            screen.py       LLM funniness pre-screen
        factcheck.py        fact-check pass -> verified brief
        brief.py            comic-angle generation
        drafting.py         orchestrates fact-check + angles, writes drafts
    data/
        seed_events.csv     hand-curated starter pool
    scripts/
        run_drafts.py       long-running drafting worker (token-refresh safe)
        screen_all.py       score every unscored row
        rescreen_60.py      one-off: re-score the originally-screened rows
        regen_drafts.py     regenerate existing drafts + draft fresh ones
        telegram_review.py  durable Telegram review-loop runner (systemd --user)
        weekly_pipeline.py  durable weekly discovery pipeline (systemd --user timer)
        fill_approved_copy.py  one-shot: generate post copy for approved+queued drafts
    tests/                  pytest suite (209 tests)
