"""Offline scheduled-job reporting tests; all databases are temporary."""
import importlib
import sys

import pytest

import humorhist.db as db
import humorhist.drafting as drafting
import humorhist.env as env
import humorhist.harvest.screen as screen
import humorhist.harvest.seed as seed
import humorhist.harvest.wikipedia_lists as wiki
import humorhist.llm as llm
import humorhist.telegram as telegram


@pytest.fixture
def jobs(monkeypatch, tmp_path):
    monkeypatch.setattr(env, "load_env", lambda: None)
    weekly = importlib.import_module("scripts.weekly_pipeline")
    buffer = importlib.import_module("scripts.buffer_health")
    lines = []
    monkeypatch.setattr(weekly, "log", lines.append)
    monkeypatch.setattr(buffer, "log", lines.append)
    monkeypatch.setattr(weekly, "_llm", lambda: object())
    monkeypatch.setattr(llm, "screening_client", lambda **kw: object())
    monkeypatch.setattr(buffer, "_llm", lambda: object())
    monkeypatch.setattr(seed, "load_seed", lambda conn: {"inserted": 2})
    monkeypatch.setattr(wiki, "harvest_wikipedia_lists", lambda conn: {"inserted": 3, "pages_failed": 0})
    monkeypatch.setattr(screen, "screen_pool", lambda *a, **kw: {"scored": 0, "batches": 0, "failed_batches": 0})
    monkeypatch.setattr(drafting, "draft_candidates", lambda *a, **kw: {"selected": 0, "drafted": 0, "failed": 0})
    monkeypatch.setattr(sys, "argv", ["job", "--db", str(tmp_path / "test.sqlite")])
    return weekly, buffer, lines


def boom(*args, **kwargs):
    raise RuntimeError("secret-token-and-url")


def test_weekly_screen_and_draft_use_separate_factories(jobs, monkeypatch):
    weekly, _, _ = jobs
    screen_client, draft_client = object(), object()
    calls = []
    monkeypatch.setattr(llm, "screening_client", lambda **kw: calls.append(kw) or screen_client)
    monkeypatch.setattr(weekly, "_llm", lambda: draft_client)
    monkeypatch.setattr(screen, "screen_pool", lambda conn, client, **kw:
                        {"scored": 0, "batches": 0, "failed_batches": 0} if client is screen_client else boom())
    monkeypatch.setattr(drafting, "draft_candidates", lambda conn, client, **kw:
                        {"selected": 0, "drafted": 0, "failed": 0} if client is draft_client else boom())
    assert weekly.main() == 0
    assert calls == [{"timeout": 300.0}]


def test_weekly_screen_factory_failure_still_drafts(jobs, monkeypatch):
    weekly, _, lines = jobs
    monkeypatch.setattr(llm, "screening_client", boom)
    calls = []
    monkeypatch.setattr(drafting, "draft_candidates", lambda *a, **kw:
                        calls.append(1) or {"selected": 0, "drafted": 0, "failed": 0})
    assert weekly.main() == 1
    assert calls == [1] and "screen:RuntimeError" in lines[-1]
    assert "secret-token-and-url" not in "\n".join(lines)


def test_weekly_partial_results_and_phase_isolation(jobs, monkeypatch):
    weekly, _, lines = jobs
    calls = []
    monkeypatch.setattr(seed, "load_seed", boom)
    monkeypatch.setattr(wiki, "harvest_wikipedia_lists", lambda conn: {"inserted": 2, "pages_failed": 3})
    monkeypatch.setattr(screen, "screen_pool", lambda *a, **kw: {"scored": 4, "batches": 3, "failed_batches": 1})
    monkeypatch.setattr(drafting, "draft_candidates", lambda *a, **kw: calls.append(1) or {"selected": 2, "drafted": 1, "failed": 1, "failures": [{"error": "secret-token-and-url"}]})
    assert weekly.main() == 1
    text = "\n".join(lines)
    assert calls == [1]
    for expected in ("seed:RuntimeError", "wikipedia:pages_failed", "screen:failed_batches", "draft:failed", "pages_failed=3", "failed_batches=1", "drafted=1 failed=1"):
        assert expected in text
    assert "secret-token-and-url" not in text


def test_weekly_screen_exception_still_drafts_and_notifies(jobs, monkeypatch):
    weekly, _, lines = jobs
    monkeypatch.setattr(screen, "screen_pool", boom)
    monkeypatch.setenv("HUMORHIST_TELEGRAM_CHAT_ID", "chat")
    monkeypatch.setenv("HUMORHIST_TELEGRAM_BOT_TOKEN", "secret")
    def draft(conn, *args, **kwargs):
        conn.execute("INSERT INTO pool (id,title,status) VALUES ('p','title','drafted')")
        conn.execute("INSERT INTO drafts (id,pool_id,brief_json,angles_json,status,created_at) VALUES ('d','p','{}','{}','pending','2026')")
        return {"selected": 1, "drafted": 1, "failed": 0}
    monkeypatch.setattr(drafting, "draft_candidates", draft)
    monkeypatch.setattr(telegram, "TelegramClient", lambda: object())
    monkeypatch.setattr(telegram, "notify_new_drafts", boom)
    assert weekly.main() == 1
    text = "\n".join(lines)
    assert "screen:RuntimeError" in text and "notify:RuntimeError" in text
    assert "fresh=1" in text and "secret-token-and-url" not in text


def test_weekly_success_and_skips(jobs, monkeypatch):
    weekly, _, lines = jobs
    monkeypatch.setattr(sys, "argv", sys.argv + ["--no-harvest"])
    assert weekly.main() == 0
    assert "failures=none" in lines[-1]


def test_buffer_partial_draft_still_alerts_and_reports_counts(jobs, monkeypatch):
    _, buffer, lines = jobs
    path = sys.argv[2]
    conn = db.connect(path)
    db.migrate(conn)
    for i, status in enumerate(("pending", "pending", "approved", "approved")):
        conn.execute("INSERT INTO pool (id,title,status) VALUES (?, 'title', 'drafted')", (f'p{i}',))
        conn.execute("INSERT INTO drafts (id,pool_id,brief_json,angles_json,status,created_at,defer_until) VALUES (?,?,'{}','{}',?,'2026',?)", (f'd{i}', f'p{i}', status, '9999-01-01' if i == 1 else None))
    conn.execute("INSERT INTO queue (draft_id,scheduled_for,published) VALUES ('d2',NULL,0)")
    conn.commit()
    conn.close()
    monkeypatch.setattr(drafting, "draft_candidates", lambda *a, **kw: {"selected": 2, "drafted": 1, "failed": 1, "failures": [{"error": "secret-token-and-url"}]})
    sent = []
    class Transport:
        def send_message(self, chat, message):
            sent.append(message)
    monkeypatch.setattr(telegram, "TelegramClient", Transport)
    monkeypatch.setenv("HUMORHIST_TELEGRAM_CHAT_ID", "chat")
    monkeypatch.setenv("HUMORHIST_TELEGRAM_BOT_TOKEN", "secret")
    assert buffer.main() == 1
    text = "\n".join(lines)
    assert "ready=1 deferred=1 approved=2 queued=1" in text
    assert "draft:failed" in text and "draft:failed" in sent[0]
    assert "secret-token-and-url" not in text + sent[0]


@pytest.mark.parametrize("failure", ["draft", "client", "notify"])
def test_buffer_exceptions_fail_but_remain_isolated(jobs, monkeypatch, failure):
    _, buffer, lines = jobs
    sent = []
    class Transport:
        def send_message(self, chat, message):
            sent.append(message)
            if failure == "notify":
                boom()
    monkeypatch.setattr(telegram, "TelegramClient", Transport)
    monkeypatch.setenv("HUMORHIST_TELEGRAM_CHAT_ID", "chat")
    monkeypatch.setenv("HUMORHIST_TELEGRAM_BOT_TOKEN", "secret")
    if failure == "draft":
        monkeypatch.setattr(drafting, "draft_candidates", boom)
    if failure == "client":
        monkeypatch.setattr(buffer, "_llm", boom)
    assert buffer.main() == 1
    assert sent
    assert "secret-token-and-url" not in "\n".join(lines) + " ".join(sent)
    assert f"{failure if failure == 'notify' else 'draft'}:RuntimeError" in lines[-1]


def test_buffer_missing_credential_and_explicit_skips(jobs, monkeypatch):
    _, buffer, lines = jobs
    monkeypatch.setattr(buffer, "_llm", lambda: None)
    assert buffer.main() == 1
    assert "draft:LLMUnavailable" in lines[-1]
    lines.clear()
    monkeypatch.setattr(sys, "argv", sys.argv + ["--no-auto-draft", "--no-notify"])
    assert buffer.main() == 0
    assert "failures=none" in lines[-1]


@pytest.mark.parametrize("job_index", [0, 1])
def test_setup_failure_is_nonzero_and_redacted(jobs, monkeypatch, job_index):
    lines = jobs[2]
    monkeypatch.setattr(db, "connect", boom)
    assert jobs[job_index].main() == 1
    assert "RuntimeError" in "\n".join(lines)
    assert "secret-token-and-url" not in "\n".join(lines)


def test_weekly_suppresses_raw_wikipedia_error_logs(jobs, monkeypatch, caplog):
    import logging

    weekly, _, lines = jobs
    def harvest(conn):
        logging.getLogger("humorhist.harvest.wikipedia_lists").warning("secret-token-and-url")
        return {"inserted": 0, "pages_failed": 1}
    monkeypatch.setattr(wiki, "harvest_wikipedia_lists", harvest)
    assert weekly.main() == 1
    assert "pages_failed=1" in "\n".join(lines)
    assert "secret-token-and-url" not in caplog.text
    assert not logging.getLogger("humorhist.harvest.wikipedia_lists").disabled


def test_incomplete_screening_fails_without_blocking_draft(jobs, monkeypatch):
    weekly, _, lines = jobs
    conn = db.connect(sys.argv[2])
    db.migrate(conn)
    conn.execute("INSERT INTO pool (id,title,status) VALUES ('p','title','new')")
    conn.commit()
    conn.close()
    calls = []
    monkeypatch.setattr(drafting, "draft_candidates", lambda *a, **k: calls.append(1) or
                        {"selected": 0, "drafted": 0, "failed": 0})
    assert weekly.main() == 1
    assert calls == [1]
    assert "screen:incomplete" in lines[-1]


def test_weekly_uses_drafted_count_despite_concurrent_review(jobs, monkeypatch):
    weekly, _, lines = jobs
    sent = []
    monkeypatch.setenv("HUMORHIST_TELEGRAM_CHAT_ID", "chat")
    monkeypatch.setenv("HUMORHIST_TELEGRAM_BOT_TOKEN", "token")
    monkeypatch.setattr(telegram, "TelegramClient", lambda: object())
    monkeypatch.setattr(telegram, "notify_new_drafts", lambda *args: sent.append(True) or 1)
    # No net pending increase is needed for a newly drafted item to trigger a nudge.
    monkeypatch.setattr(drafting, "draft_candidates", lambda *a, **k:
                        {"selected": 1, "drafted": 1, "failed": 0})
    assert weekly.main() == 0
    assert sent == [True]
    assert "fresh=1" in lines[-1]


@pytest.mark.parametrize("missing", ["chat", "token", "both"])
def test_due_notifications_missing_config_fail(jobs, monkeypatch, missing):
    weekly, buffer, lines = jobs
    if missing != "chat" and missing != "both":
        monkeypatch.setenv("HUMORHIST_TELEGRAM_CHAT_ID", "chat")
    if missing != "token" and missing != "both":
        monkeypatch.setenv("HUMORHIST_TELEGRAM_BOT_TOKEN", "token")
    monkeypatch.setattr(drafting, "draft_candidates", lambda *a, **k:
                        {"selected": 1, "drafted": 1, "failed": 0})
    assert weekly.main() == 1
    assert "notify:NotConfigured" in lines[-1]
    assert buffer.main() == 1
    assert "notify:NotConfigured" in lines[-1]


def test_buffer_deferred_backlog_and_published_counts(jobs, monkeypatch):
    _, buffer, lines = jobs
    conn = db.connect(sys.argv[2])
    db.migrate(conn)
    for i in range(7):
        conn.execute("INSERT INTO pool (id,title,status) VALUES (?, 'title', 'drafted')", (f'p{i}',))
        conn.execute("INSERT INTO drafts (id,pool_id,status,defer_until) VALUES (?,?,?,?)",
                     (f'd{i}', f'p{i}', 'pending' if i < 5 else 'approved',
                      '9999-01-01' if i < 5 else None))
    conn.execute("INSERT INTO queue (draft_id,published) VALUES ('d5',1),('d6',0)")
    conn.commit()
    conn.close()
    calls = []
    monkeypatch.setattr(drafting, "draft_candidates", lambda *a, **k: calls.append(1) or
                        {"selected": 0, "drafted": 0, "failed": 0})
    monkeypatch.setattr(sys, "argv", sys.argv + ["--no-notify"])
    assert buffer.main() == 0
    assert calls == [1]
    assert any("ready=0 deferred=5 approved=1 queued=1 published=1" in line for line in lines)


@pytest.mark.parametrize("job_index", [0, 1])
def test_jobs_aggregate_redacted_draft_diagnostics(jobs, monkeypatch, job_index):
    job, lines = jobs[job_index], jobs[2]
    sent = []
    monkeypatch.setenv("HUMORHIST_TELEGRAM_CHAT_ID", "chat")
    monkeypatch.setenv("HUMORHIST_TELEGRAM_BOT_TOKEN", "token")
    class Transport:
        def send_message(self, chat, message):
            sent.append(message)
    monkeypatch.setattr(telegram, "TelegramClient", Transport)
    monkeypatch.setattr(drafting, "draft_candidates", lambda *a, **k: {
        "selected": 3, "drafted": 0, "failed": 3, "failures": [
            {"stage": "extract", "category": "validation", "error": "secret-token-and-url"},
            {"stage": "extract", "category": "validation"},
            {"stage": "secret-token-and-url", "category": "secret-token-and-url"}]})
    assert job.main() == 1
    text = "\n".join(lines + sent)
    assert "extract:validation=2" in text and "unknown:unexpected=1" in text
    assert "secret-token-and-url" not in text
    if job_index == 1:
        assert len(sent) == 1 and "2 during fetching Wikipedia extracts" in sent[0]
        assert "0 created, 3 failed out of 3 selected" in sent[0]


@pytest.mark.parametrize("mode", ["send", "missing", "disabled", "notify_error", "success"])
def test_healthy_buffer_alerts_only_on_failure(jobs, monkeypatch, mode):
    import humorhist.buffer as buf
    _, job, lines = jobs
    monkeypatch.setattr(buf, "buffer_health", lambda conn: {
        "auto_draft": True, "level": "silent", "depth": 7, "pending": 0})
    sent = []
    class Transport:
        def send_message(self, chat, message):
            sent.append(message)
            if mode == "notify_error":
                boom()
    monkeypatch.setattr(telegram, "TelegramClient", Transport)
    if mode != "missing":
        monkeypatch.setenv("HUMORHIST_TELEGRAM_CHAT_ID", "chat")
        monkeypatch.setenv("HUMORHIST_TELEGRAM_BOT_TOKEN", "token")
    if mode == "disabled":
        monkeypatch.setattr(sys, "argv", sys.argv + ["--no-notify"])
    if mode != "success":
        monkeypatch.setattr(drafting, "draft_candidates", boom)
    assert job.main() == (0 if mode == "success" else 1)
    assert len(sent) == (1 if mode in ("send", "notify_error") else 0)
    if mode == "missing":
        assert "notify:NotConfigured" in lines[-1]
    if mode == "notify_error":
        assert "notify:RuntimeError" in lines[-1]
    assert "secret-token-and-url" not in "\n".join(lines + sent)


@pytest.mark.parametrize("phase", ["setup", "health"])
def test_buffer_early_failure_attempts_one_redacted_alert(jobs, monkeypatch, phase):
    import humorhist.buffer as buf
    _, job, lines = jobs
    messages = []
    class Transport:
        def send_message(self, chat, message):
            messages.append(message)
    monkeypatch.setattr(telegram, "TelegramClient", Transport)
    monkeypatch.setenv("HUMORHIST_TELEGRAM_CHAT_ID", "chat")
    monkeypatch.setenv("HUMORHIST_TELEGRAM_BOT_TOKEN", "token")
    monkeypatch.setattr(db if phase == "setup" else buf,
                        "connect" if phase == "setup" else "buffer_health", boom)
    assert job.main() == 1
    assert len(messages) == 1 and f"{phase}:RuntimeError" in messages[0]
    assert "secret-token-and-url" not in "\n".join(lines + messages)


def test_weekly_screen_reasons_are_sanitized(jobs, monkeypatch):
    weekly, _, lines = jobs
    monkeypatch.setattr(screen, "screen_pool", lambda *a, **kw: {
        "scored": 0, "batches": 1, "failed_batches": 1, "skipped": 9,
        "failures": [{"category": "http_error", "status": 429, "error": "secret-token"}],
    })
    assert weekly.main() == 1
    text = "\n".join(lines)
    assert "http_error_http_429=1" in text and "skipped=9" in text
    assert "secret-token" not in text
