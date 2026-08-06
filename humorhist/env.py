"""Minimal .env loader (stdlib only, no extra dependency).

Reads KEY=VALUE pairs from a local ``.env`` file (gitignored) into os.environ
so secrets like HUMORHIST_TELEGRAM_BOT_TOKEN never need to be exported by hand
or committed. Existing env vars win (we only setdefault). Safe to call multiple
times.
"""

from __future__ import annotations

import os
from pathlib import Path

_ENV_CANDIDATES = (Path.cwd() / ".env", Path(__file__).resolve().parent.parent / ".env")


def load_env(path: Path | None = None) -> None:
    """Load .env into os.environ (existing vars take precedence).

    `path` overrides the default search (repo-root .env, then cwd/.env).
    Lines beginning with # are comments; blank lines are skipped; inline
    # comments after a value are stripped.
    """
    candidates = [path] if path else list(_ENV_CANDIDATES)
    for cand in candidates:
        if not cand or not cand.is_file():
            continue
        for raw in cand.read_text().splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, _, val = line.partition("=")
            key, val = key.strip(), val.strip()
            # drop an inline trailing comment (only if value isn't quoted)
            if not (val.startswith('"') or val.startswith("'")):
                val = val.split(" #")[0].split(" #")[0].strip()
            os.environ.setdefault(key, val)
        return  # first existing file wins


if __name__ == "__main__":
    load_env()
    print("TELEGRAM token set:", bool(os.environ.get("HUMORHIST_TELEGRAM_BOT_TOKEN")))
    print("TELEGRAM chat id set:", bool(os.environ.get("HUMORHIST_TELEGRAM_CHAT_ID")))
