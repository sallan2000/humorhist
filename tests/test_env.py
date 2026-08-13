"""Tests for humorhist.env -- local .env loading into os.environ."""

from __future__ import annotations

import os
from pathlib import Path

import humorhist.env as env


def test_load_env_reads_pairs(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("HUMORHIST_TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.delenv("HUMORHIST_TELEGRAM_CHAT_ID", raising=False)
    p = tmp_path / ".env"
    p.write_text(
        "# comment line\nHUMORHIST_TELEGRAM_BOT_TOKEN=abc123\nHUMORHIST_TELEGRAM_CHAT_ID=999\n\n"  # blank line ignored
    )
    env.load_env(p)
    assert os.environ.get("HUMORHIST_TELEGRAM_BOT_TOKEN") == "abc123"
    assert os.environ.get("HUMORHIST_TELEGRAM_CHAT_ID") == "999"


def test_load_env_existing_var_wins(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("HUMORHIST_TELEGRAM_BOT_TOKEN", "already-set")
    p = tmp_path / ".env"
    p.write_text("HUMORHIST_TELEGRAM_BOT_TOKEN=file-value\n")
    env.load_env(p)
    # setdefault semantics: file does not override the real env var
    assert os.environ.get("HUMORHIST_TELEGRAM_BOT_TOKEN") == "already-set"


def test_load_env_skips_missing_file(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("HUMORHIST_TELEGRAM_BOT_TOKEN", raising=False)
    env.load_env(tmp_path / "does_not_exist.env")
    assert "HUMORHIST_TELEGRAM_BOT_TOKEN" not in os.environ
