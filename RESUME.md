# humorhist — resume notes (2026-08-05)

## Where we got to
Phases 1 and 2 of the plan are **built, tested and committed**. 123 tests passing.
Plan: `/home/stevie/.hermes/plans/2026-08-05_213000-humorous-history-human-voiced.md`
Code: `/home/stevie/projects/humorhist` (git, 7 commits, all work committed)

## Working end-to-end, verified live
- `harvest` → **741 pool candidates** (105 curated seed + 636 from 5 Wikipedia list pages)
- `screen`  → **60 scored live** by the LLM, 0 failures, avg 5.85
- `draft`   → **3 real drafts generated** (Acoustic Kitty, Napoleon's rabbits, Cadaver Synod)
- `status` / `show` CLI commands working

## Quality signals (the bit that matters)
Taste filter is genuinely working — it scored the Great Sparrow Campaign (famine),
Vlad's envoys, and a fatal surgery at **0.0**, correctly refusing to treat mass death
as comedy. Molasses Flood and Dancing Plague at 1.0 (both killed people).
Top scorers were Emu War, Kettle War, Acoustic Kitty, Wojtek the bear — all 9.0.

The fact-check layer proved its worth immediately: on Acoustic Kitty it correctly
flagged that the famous "cat run over by a taxi" detail — the funniest part of the
story — is **not supported** by the source and is disputed lore. That is exactly the
failure mode that would get the account fact-checked into oblivion, caught automatically.

## THE DECISION FOR TOMORROW

**Caveat on the three existing drafts:** they were generated BEFORE the source-URL
bug fix (commit 55f2bf7). `_row_to_item()` emitted `source_url` but
`build_factcheck_prompt()` reads `url`, so the `SOURCE URL:` line was silently
missing from every fact-check prompt — no error, just less context. Now fixed and
regression-tested. The existing drafts are still worth reading, but if the angles
feel borderline, regenerate one before judging:
    systemd-run --user --unit=humorhist-redraft --same-dir \
      /home/stevie/projects/humorhist/.venv/bin/python scripts/run_drafts.py --count 1

Read the drafts and answer one question: **are the comic angles genuinely
useful to you as a writer?**

    cd ~/projects/humorhist
    .venv/bin/python -m humorhist.cli --db data/humorhist.sqlite show 8e07a219377a9273  # Acoustic Kitty
    .venv/bin/python -m humorhist.cli --db data/humorhist.sqlite show c2da55abb768d140  # Napoleon's rabbits
    .venv/bin/python -m humorhist.cli --db data/humorhist.sqlite show f90d305bb73309ef  # Cadaver Synod

If yes → build Phase 3 (Telegram review loop) and Phase 4 (publishing).
If no  → tune `ANGLES_SYSTEM_PROMPT` in `humorhist/brief.py` first. Everything
downstream is plumbing; that prompt is the product. Do not build further until
this passes.

## Known issues / decisions deferred
1. **API key.** Currently borrowing the Nous OAuth token from `~/.hermes/auth.json`,
   which expires hourly and only refreshes while Hermes runs. `scripts/run_drafts.py`
   re-reads it per item as a workaround. For unattended operation this needs a real
   API key in `HUMORHIST_LLM_API_KEY`.
2. **Model.** `DEFAULT_MODEL = "tencent/hy3:free"` in `humorhist/llm.py`. It works but
   is slow (~50-130s per draft). Hermes-4-70B / 405B return 404 on this endpoint.
   Worth trying a stronger model for angle quality — this is the one place where
   model quality directly affects the product.
3. **Pool only 60/741 screened.** Screening the rest takes ~1hr at current speed.
   Run: `.venv/bin/python -m humorhist.cli --db data/humorhist.sqlite screen`
4. **Wikipedia source breadth.** Currently 5 list pages, per-page yields:
       List_of_wars_of_succession        229 items, 213 with year
       List_of_hoaxes                    217 items,  29 with year
       List_of_Ig_Nobel_Prize_winners    110 items, 105 with year
       List_of_April_Fools'_Day_jokes     68 items,  49 with year
       Lists_of_unusual_deaths            27 items,   0 with year
   Note `Lists_of_unusual_deaths` is an index page (a list OF lists), hence only 27
   items and no years — could be replaced by the individual sub-lists it links to.
   AGENT DECISION TO REVIEW: `List_of_practical_joke_topics` was dropped during the
   1.3 fixup (only 19 items) and swapped for `List_of_April_Fools'_Day_jokes`.
   Reasonable, but it was an automated content-sourcing choice — override if you
   disagree. Adding more pages is the cheapest way to grow the pool.
5. Task 1.2 reviews both came back clean: spec PASS (all 9 requirements verified
   line-by-line, no scope creep) and quality APPROVED (no critical/important
   issues). Optional polish only: add `logging.info(summary)` when wiring
   harvesters into a runner, and consider a TypedDict return. Both harvesters now
   exist, so a shared upsert helper could be revisited — the reviewer advised
   waiting until the duplication was concrete, which it now is.
6. **Per-row commits in db.py** (flagged by the Task 1.1 quality review, not yet
   actioned). Every mutating function calls `conn.commit()` individually; SQLite
   fsyncs per commit, so bulk inserts are far slower than one wrapping transaction.
   Not urgent — the 636-row Wikipedia harvest still ran in ~7s — but worth fixing
   before the pool grows to thousands, or if a harvester ever feels sluggish.
   Fix: add an optional `commit=True` param to the mutating helpers, or expose a
   context manager so callers can batch. Reviewer found NO critical issues or
   security holes; the whitelist validation in set_status is sound.

## Durable background jobs
`Linger=yes` is set, so systemd user services survive logout:
    systemd-run --user --unit=humorhist-X --same-dir \
      /home/stevie/projects/humorhist/.venv/bin/python scripts/run_drafts.py --count 5
    journalctl --user -u humorhist-X -f
Note: Hermes subagents do NOT survive session end. Only systemd units do.

## Commits
    7f9bf69 feat: CLI, durable drafting worker, model default fix
    55f2bf7 feat: draft assembly orchestration and CLI, with tests
    52f4bdd feat: comic angle generation
    7db761f feat: fact-check pass with brief validation
    96d7721 fix: strip nested templates and comments before title derivation
    b62fa0d feat: LLM funny pre-screen for pool candidates
    a650c7e feat: wikipedia list harvester with redirect following and section-year fallback
