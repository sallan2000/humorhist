"""Screening entrypoint wiring, without credentials, services or real databases."""
import argparse
import importlib.util
import sys
from pathlib import Path

import pytest

import humorhist.env as env
import humorhist.harvest.screen as screen
import humorhist.llm as llm
import humorhist.telegram as telegram


def test_cli_screen_uses_screening_factory(monkeypatch):
    import humorhist.cli as cli
    sentinel = object()
    seen = []
    class Cursor:
        def fetchone(self): return {"n": 0, "avg": None}
    class Conn:
        def execute(self, *args): return Cursor()
        def close(self): pass
    monkeypatch.setattr(cli, "_open_db", lambda path: Conn())
    monkeypatch.setattr(llm, "screening_client", lambda: sentinel)
    monkeypatch.setattr(llm, "default_client", lambda: pytest.fail("draft factory used"))
    monkeypatch.setattr(screen, "screen_pool", lambda conn, client, **kw:
                        seen.append(client) or {"scored": 0, "failed_batches": 0, "unscored": 0})
    monkeypatch.setattr(screen, "screen_attempt_status", lambda conn: [])
    assert cli.cmd_screen(argparse.Namespace(db="unused", batch_size=20, limit=None, retry_failed=False)) == 0
    assert seen == [sentinel]


def test_telegram_screen_uses_screening_client(monkeypatch):
    sentinel = object()
    monkeypatch.setattr(llm, "screening_client", lambda: sentinel)
    monkeypatch.setattr(telegram, "_get_llm", lambda *a, **k: pytest.fail("draft factory used"))
    seen = []
    class Cursor:
        def fetchone(self):
            return (0,)
    class Worker:
        def execute(self, *args):
            return Cursor()
    monkeypatch.setattr(screen, "screen_pool", lambda conn, client, **kw:
                        seen.append((client, kw)) or {"scored": 0, "batches": 0, "failed_batches": 0})
    monkeypatch.setattr(telegram, "_telegram_job", lambda conn, client, chat, name, msg, work: work(Worker()))
    telegram.telegram_screen(None, telegram.StubTelegram(), "chat", batch_size=7, limit=3)
    assert seen == [(sentinel, {"batch_size": 7, "limit": 3})]


def test_telegram_screen_unavailable_is_redacted(monkeypatch):
    def unavailable():
        raise llm.LLMUnavailable("secret raw error")
    monkeypatch.setattr(llm, "screening_client", unavailable)
    stub = telegram.StubTelegram()
    assert telegram.telegram_screen(None, stub, "chat") is None
    assert "unavailable" in stub.sent[0]["text"].lower()
    assert "secret raw error" not in stub.sent[0]["text"]


@pytest.mark.parametrize("screen_model,expected", [("screen-model", "screen-model"), (None, "draft-model")])
def test_screen_all_loads_env_and_preserves_precedence(tmp_path, monkeypatch, screen_model, expected):
    config = tmp_path / ".env"
    config.write_text("HUMORHIST_LLM_MODEL=file-draft\nHUMORHIST_SCREEN_MODEL=file-screen\n")
    monkeypatch.setattr(env, "_ENV_CANDIDATES", (config,))
    monkeypatch.setenv("HUMORHIST_LLM_MODEL", "draft-model")
    if screen_model is None:
        # Test fallback with a config that doesn't provide a screening override.
        config.write_text("HUMORHIST_LLM_MODEL=file-draft\n")
        monkeypatch.delenv("HUMORHIST_SCREEN_MODEL", raising=False)
    else:
        monkeypatch.setenv("HUMORHIST_SCREEN_MODEL", screen_model)
    monkeypatch.setenv("HUMORHIST_LLM_API_KEY", "test-key")
    monkeypatch.delenv("HUMORHIST_LLM_BASE_URL", raising=False)
    path = Path(__file__).resolve().parents[1] / "scripts" / "screen_all.py"
    spec = importlib.util.spec_from_file_location("screen_all_test_import", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert llm.screening_client().model == expected
    assert llm.default_client().model == "draft-model"
    assert module.env is env


def test_screen_all_reads_screen_model_from_dotenv(tmp_path, monkeypatch):
    config = tmp_path / ".env"
    config.write_text("HUMORHIST_SCREEN_MODEL=file-screen\n")
    monkeypatch.setattr(env, "_ENV_CANDIDATES", (config,))
    monkeypatch.delenv("HUMORHIST_SCREEN_MODEL", raising=False)
    monkeypatch.setenv("HUMORHIST_LLM_API_KEY", "test-key")
    monkeypatch.delenv("HUMORHIST_LLM_BASE_URL", raising=False)
    path = Path(__file__).resolve().parents[1] / "scripts" / "screen_all.py"
    spec = importlib.util.spec_from_file_location("screen_all_test_dotenv", path)
    spec.loader.exec_module(importlib.util.module_from_spec(spec))
    assert llm.screening_client().model == "file-screen"


def test_screen_all_uses_screening_factory(monkeypatch):
    import humorhist.db as db
    from scripts import screen_all
    calls, received = [], []
    sentinel = object()
    class Cursor:
        def __init__(self, row): self.row = row
        def fetchone(self): return self.row
    class Conn:
        def execute(self, query):
            if "sqlite_master" in query:
                return Cursor(None)
            if "avg(" in query:
                return Cursor({"a": None})
            return Cursor({"n": 0})
    monkeypatch.setattr(db, "connect", lambda path: Conn())
    monkeypatch.setattr(db, "migrate", lambda conn: None)
    monkeypatch.setattr(screen_all, "log", lambda msg: None)
    monkeypatch.setattr(llm, "screening_client", lambda **kw: calls.append(kw) or sentinel)
    monkeypatch.setattr(llm, "resilient_client", lambda **kw: pytest.fail("draft factory used"))
    monkeypatch.setattr(screen, "screen_pool", lambda conn, client, **kw:
                        received.append(client) or {"scored": 0, "failed_batches": 0})
    monkeypatch.setattr(sys, "argv", ["screen_all"])
    assert screen_all.main() == 0
    assert calls == [{"timeout": 300.0}] and received == [sentinel]


def test_screen_all_unavailable_redacted_and_fails_even_with_empty_pool(tmp_path, monkeypatch, capsys):
    import humorhist.db as db
    from scripts import screen_all

    class Cursor:
        def __init__(self, row):
            self.row = row

        def fetchone(self):
            return self.row

    class EmptyConnection:
        def execute(self, query):
            if "sqlite_master" in query:
                return Cursor(None)
            if "avg(" in query:
                return Cursor({"a": None})
            return Cursor({"n": 0})

    def unavailable(**kwargs):
        raise llm.LLMUnavailable("private-credential-and-provider-detail")

    monkeypatch.setattr(db, "connect", lambda path: EmptyConnection())
    monkeypatch.setattr(db, "migrate", lambda conn: None)
    monkeypatch.setattr(llm, "screening_client", unavailable)
    monkeypatch.setattr(screen, "screen_pool", lambda *args, **kwargs: pytest.fail("screening invoked"))
    monkeypatch.setattr(screen_all, "LOG", tmp_path / "screen.log")
    monkeypatch.setattr(sys, "argv", ["screen_all"])

    assert screen_all.main() == 1
    output = capsys.readouterr().out
    log = (tmp_path / "screen.log").read_text()
    for text in (output, log):
        assert "screening client unavailable" in text
        assert "private-credential-and-provider-detail" not in text
