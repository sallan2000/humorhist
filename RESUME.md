# humorhist — resume notes

_Last updated: 2026-08-06 (Phase 3 landed on GitHub)_

## Where we got to
Phases 1, 2 and the Phase 3 review gate are **built, tested and committed**,
and the code is **pushed to GitHub**. Local `master` tracks `origin/main`.

- Repo: `https://github.com/sallan2000/humorhist` (public, default branch `main`)
- Local code: `/home/stevie/projects/humorhist` (git, 23 commits on `main`)
- Tests: **137 passing** (no network; `pytest tests/ -q`)
- Plan file: `/home/stevie/.hermes/plans/2026-08-05_213000-humorous-history-human-voiced.md`

## Git / remote setup (how to push)
- `origin` → `https://github.com/sallan2000/humorhist.git` (plain URL, no token in `.git/config`)
- Local branch `master` tracks `origin/main` (upstream set).
- Push: `git push` (maps `master → origin/main` via tracking).
- GitHub fine-grained PAT is stored in `~/.git-credentials` (mode 600, repo-scoped,
  NOT in `.git/config`, NOT committed). Re-auth happens automatically on push.
- NOTE: the PAT string was pasted into an earlier chat turn, so it is exposed in
  transcript history. Recommend rotating it at GitHub → Settings → Developer
  settings even though the repo is already up to date.
- If the remote has a commit you don't have (e.g. an edit made on github.com),
  `git push` is rejected — do `git pull --rebase origin main` then push again.
  Never force-push `main` unless you intend to clobber a commit.

## Working end-to-end, verified live
- `harvest` → **741 pool candidates** (105 curated seed + 636 from 5 Wikipedia list pages)
- `screen`  → **all 741 scored** by the LLM (whole-pool re-screen under the new
  prompt; 109 ≥7, 145 ≥5, pool avg 3.93). 0 failures.
- `draft`   → **9 drafts** total (4 regenerated from the original 3 + 5 fresh from
  the ≥7 pool). All `pending`.
- `status` / `show` / `review` CLI commands working.
- **Phase 3 review gate proven live**: a draft was approved via
  `python -m humorhist.cli --db data/humorhist.sqlite review` against the real DB.

## Editorial decision that changed the product
The mass-death / suffering **taste filter was deliberately removed** (your call).
- `SCREEN_SYSTEM_PROMPT` no longer penalises death/suffering; it now rewards
  absurd bureaucracy and rates purely on absurdity.
- `sensitivity_flags` dropped entirely from `factcheck.py` (it was never persisted
  to the DB, only emitted in the brief and printed by the CLI).
- Regression tests guard both removals (`test_screen_prompt_no_death_taste_penalty`,
  `test_system_prompt_has_no_sensitivity_flagging`).
- Result: formerly 0.0 items now score 5–9 (Sparrow Campaign 7.0, Liston 8.0,
  Pastry War 8.0, Karansebes 9.0). Grim-but-absurd events are now draftable.
- To re-introduce taste controls, edit those two prompts; the schema has no blocklist.

## Quality signals (the bit that matters)
The fact-check layer is the real value: on Acoustic Kitty it correctly flagged
that the famous "cat run over by a taxi" detail is **not supported** by the
source and is disputed lore — exactly the failure mode that would get the account
fact-checked into oblivion, caught automatically.

## Phase 3 — human review gate (built, committed, pushed)
- `humorhist/review.py` — transport-agnostic state machine:
  - `pending_drafts(conn)` → all `status='pending'` drafts, oldest first.
  - `apply_review(conn, draft_id, decision, editor_line, notes)` → validates
    decision (approve/reject), guards unknown id and non-reviewable status,
    writes `status`, stamps `reviewed_at`, stores optional `editor_line` +
    `editor_notes`. Idempotent; allows approve↔reject flips.
- `humorhist/cli.py` → `review` subcommand: walks pending drafts, renders each
  (shared `render_draft()`), prompts `[a/r/s]` then optional editor line + notes.
  'skip' leaves a draft `pending`.
- No schema migration needed — `drafts` already had `editor_line`, `editor_notes`,
  `reviewed_at`.
- 14 new tests (9 unit on `review.py`, 5 CLI on `cmd_review` via simulated stdin).

## What's left (Phases 3.3 → 4)
- **3.3 / 3.4 Telegram transport + notifications** (NOT built). Decision made:
  CLI-first, Telegram drops on top later. When built, use long-poll (`getUpdates`),
  NOT a webhook — this host is behind Cloudflare/NAT and does not expose ports.
  Needs a bot token from @BotFather. `render_draft()` is already shared so the
  Telegram presentation matches the CLI.
- **3.5 / Phase 4 — queue handoff + publisher** (NOT built). `queue` and `posts`
  tables already exist in the schema (stubbed). `enqueue_approved(conn)` and the
  actual publisher are the remaining work. Deliberately deferred from Phase 3.

## Known issues / decisions deferred
1. **API key.** Borrows the Nous OAuth token from `~/.hermes/auth.json` (expires
   hourly, refreshes only while Hermes runs). `scripts/run_drafts.py` re-reads it
   per item. For unattended operation this needs a real key in `HUMORHIST_LLM_API_KEY`.
2. **Model.** `DEFAULT_MODEL = "tencent/hy3:free"` in `humorhist/llm.py`. Works but
   slow (~50-130s per draft). Hermes-4 returns 404 on this endpoint.
3. **Per-row commits in db.py** (flagged by an earlier quality review, not yet
   actioned). Every mutating fn commits individually; SQLite fsyncs per commit.
   Not urgent — 636-row harvest ran in ~7s — but worth a batching helper before
   the pool grows to thousands. Reviewer found NO critical issues or security holes.
4. **Wikipedia source breadth.** 5 list pages harvested. `Lists_of_unusual_deaths`
   is an index page (27 items, 0 years) and could be swapped for its sub-lists.
   `List_of_practical_joke_topics` was dropped during a fixup (19 items) in favour
   of `List_of_April_Fools'_Day_jokes` — an agent content-sourcing choice to review.

## Durable background jobs
`Linger=yes` is set, so systemd user services survive logout:
    systemd-run --user --unit=humorhist-X --same-dir \
      /home/stevie/projects/humorhist/.venv/bin/python scripts/run_drafts.py --count 5
    journalctl --user -u humorhist-X -f
Note: Hermes subagents do NOT survive session end. Only systemd units do.

## Recent commits (top of log)
    e099ba1 feat(phase3): add CLI review gate (approve/reject drafts) + tests
    c2f1f55 Correct project name from 'humorhist' to 'HumorHist'   (made on github.com)
    a42eb84 docs: add user guide README
    4cf3d1e chore: stop tracking runtime logs; ignore data/*.log
    5e0b3a0 feat: regenerate 4 drafts + draft 5 fresh (no sensitivity flags)
    ... (full 23-commit history on origin/main)
