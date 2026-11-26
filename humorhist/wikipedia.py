"""Shared Wikimedia client identification and Retry-After parsing."""

import argparse
import fcntl
import math
import os
import tempfile
import threading
import time
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from pathlib import Path

USER_AGENT = "HumorHistBot/0.1 (https://github.com/sallan2000/humorhist) httpx/0.28"
MAX_RETRY_AFTER_SECONDS = 3600


def parse_retry_after(value: str | None, *, now: datetime | None = None) -> float | None:
    """Parse a bounded hint for diagnostics; never use a cap to retry early."""
    if not value or len(value) > 128:
        return None
    value = value.strip()
    if value.isascii() and value.isdigit():
        return min(int(value), MAX_RETRY_AFTER_SECONDS)
    try:
        date = parsedate_to_datetime(value)
        if date.tzinfo is None:
            return None
        seconds = (date - (now or datetime.now(UTC))).total_seconds()
        return min(max(0.0, seconds), MAX_RETRY_AFTER_SECONDS)
    except (TypeError, ValueError, OverflowError):
        return None


MIN_INTERVAL = 0.5
DEFAULT_COOLDOWN = 60.0
LOCK_WAIT = 5.0


class WikipediaCooldown(Exception):
    """A shared access restriction prevents another request for now."""


def _retry_delay(value: str | None) -> float | None:
    if not value or len(value) > 128:
        return None
    value = value.strip()
    if value.isascii() and value.isdigit():
        try:
            return float(int(value))
        except OverflowError:
            return math.inf
    try:
        date = parsedate_to_datetime(value)
        if date.tzinfo is None:
            return None
        return max(0.0, (date - datetime.now(UTC)).total_seconds())
    except (TypeError, ValueError, OverflowError):
        return None


def _directory():
    return Path.home() / ".cache" / "humorhist"


def _lock(directory):
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    fd = os.open(directory / "wikipedia-rate.lock", os.O_RDWR | os.O_CREAT, 0o600)
    lock = os.fdopen(fd, "r+")
    deadline = time.monotonic() + LOCK_WAIT
    while True:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return lock
        except BlockingIOError:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                lock.close()
                raise WikipediaCooldown(
                    "Wikipedia transport lock wait deadline exceeded; a request may be stuck"
                ) from None
            time.sleep(min(0.05, remaining))
        except BaseException:
            lock.close()
            raise


def _load(path):
    try:
        parts = path.read_text().split()
    except FileNotFoundError:
        return 0.0, 0.0, "none", 0.0
    except UnicodeError as exc:
        raise WikipediaCooldown("Wikipedia rate state invalid; inspect and review offline reset") from exc
    try:
        if len(parts) not in (2, 4):
            raise ValueError
        next_time, until = map(float, parts[:2])
        if len(parts) == 4:
            reason, minimum = parts[2], float(parts[3])
        else:
            # Old state did not record whether a finite deadline came from a
            # 403. Preserve that deadline and require review before resuming.
            reason, minimum, until = "legacy", until, math.inf
        if (math.isnan(next_time) or math.isnan(until) or math.isnan(minimum)
                or min(next_time, until, minimum) < 0 or not math.isfinite(next_time)
                or reason not in ("none", "inflight", "malformed", "403", "deadline", "error", "legacy")):
            raise ValueError
        return next_time, until, reason, minimum
    except ValueError as exc:
        raise WikipediaCooldown("Wikipedia rate state invalid; inspect and review offline reset") from exc


def _save(directory, next_time, until, reason, minimum=0.0):
    fd, name = tempfile.mkstemp(prefix=".wikipedia-rate-", dir=directory)
    try:
        with os.fdopen(fd, "w") as state:
            state.write(f"{next_time} {until} {reason} {minimum}")
            state.flush()
            os.fsync(state.fileno())
        os.replace(name, directory / "wikipedia-rate.state")
        dir_fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def request(client, url, **kwargs):
    """Serialize requests; preserve an in-flight sentinel across process death.

    HTTPX timeouts are inactivity limits, not total request deadlines. A slow
    peer may retain this lock indefinitely; other callers have a finite lock wait.
    """
    directory = _directory()
    with _lock(directory):
        next_time, until, reason, _ = _load(directory / "wikipedia-rate.state")
        now = time.time()
        if until > now:
            if reason == "403":
                raise WikipediaCooldown("Wikipedia HTTP 403 blocked pending offline review and reset")
            if math.isinf(until):
                raise WikipediaCooldown(
                    f"Wikipedia access blocked indefinitely ({reason}); inspect and review offline reset"
                )
            raise WikipediaCooldown(f"Wikipedia cooldown deadline {until:.3f} active ({reason}); retry later")
        delay = next_time - now
        if delay > LOCK_WAIT:
            raise WikipediaCooldown(f"Wikipedia pacing deadline {next_time:.3f} active; retry later")
        if delay > 0:
            threading.Event().wait(delay)
        started = time.time()
        _save(directory, started + MIN_INTERVAL, math.inf, "inflight")
        try:
            response = client.get(url, **kwargs)
        except Exception:
            now = time.time()
            _save(directory, now + MIN_INTERVAL, now + DEFAULT_COOLDOWN, "error")
            raise
        now = time.time()
        hint = response.headers.get("Retry-After")
        delay = _retry_delay(hint)
        if response.status_code == 403:
            minimum = (
                math.inf if hint is not None and delay is None
                else now + (delay if delay is not None else DEFAULT_COOLDOWN)
            )
            _save(directory, now + MIN_INTERVAL, math.inf, "403", minimum)
        elif hint is not None and delay is None:
            _save(directory, now + MIN_INTERVAL, math.inf, "malformed")
        else:
            if delay is None and response.status_code == 429:
                delay = DEFAULT_COOLDOWN
            _save(directory, now + MIN_INTERVAL, now + delay if delay is not None else 0.0, "deadline")
        return response


def main():
    parser = argparse.ArgumentParser(description="Inspect or carefully reset Wikimedia rate state offline")
    parser.add_argument("action", choices=("inspect", "reset"))
    parser.add_argument("--confirm-producers-stopped-and-permission-reviewed", action="store_true")
    args = parser.parse_args()
    directory = _directory()
    with _lock(directory):
        try:
            next_time, until, reason, minimum = _load(directory / "wikipedia-rate.state")
        except WikipediaCooldown:
            print("State is invalid; review it privately before any reset.")
            if args.action == "inspect":
                return
            next_time, until, reason, minimum = 0.0, math.inf, "invalid", math.inf
        print(f"reason={reason} next_request={next_time} blocked_until={until} minimum_reset_time={minimum}")
        if args.action == "reset":
            if not args.confirm_producers_stopped_and_permission_reviewed:
                parser.error("reset requires explicit confirmation; see README offline procedure")
            if minimum > time.time() or (math.isfinite(until) and until > time.time()):
                parser.error("known provider deadline has not expired; reset refused")
            _save(directory, max(next_time, time.time() + MIN_INTERVAL), 0.0, "none")
            print("State reset; restart producers only when permitted.")


if __name__ == "__main__":
    main()
