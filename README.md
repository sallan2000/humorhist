# humorhist

A pipeline that turns historical oddities into fully fact-checked "comic angle"
briefs for a humorous-history social account. A human still writes the actual
jokes — humorhist does the research and hands you angles, source links, and the
"watch out, the popular version is a myth" flags.

It is a command-line tool, not a service. You run stages in order:

    harvest  ->  screen  ->  draft  ->  (you review)  ->  publish*

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

124 tests, no network calls (LLM and Wikipedia are stubbed).

-------------------------------------------------------------------------------
## Project status
-------------------------------------------------------------------------------

Built and working: Phases 1-2 — harvest, screen, draft, and the review/show
commands. The pipeline produces ready-to-read briefs with fact-checked angles.

Not yet built:
- Phase 3: a Telegram review loop (approve/reject drafts from your phone).
- Phase 4: publishing to the social account.

Until then, "publishing" means: read the draft with `show`, write the post
yourself, mark the draft approved in the DB.

-------------------------------------------------------------------------------
## Layout
-------------------------------------------------------------------------------

    humorhist/
        cli.py              command-line entry point
        db.py               SQLite schema + access helpers
        llm.py              LLM client (OpenAI-compatible)
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
    tests/                  pytest suite (124 tests)
