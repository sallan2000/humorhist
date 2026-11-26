"""LLM client abstraction for humorhist.

All LLM-touching code depends on the ``LLMClient`` protocol rather than a
concrete provider, so tests can inject a deterministic stub and never make a
network call. ``NousClient`` is the real implementation; ``StubClient`` is for
tests.

The provider is configured via environment variables:
    HUMORHIST_LLM_BASE_URL   default: https://inference-api.nousresearch.com/v1
    HUMORHIST_LLM_API_KEY    required for real calls
    HUMORHIST_LLM_MODEL      default: Hermes-4-405B
"""

from __future__ import annotations

import json
import os
import re
import stat
import time
from contextlib import ExitStack
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import urlsplit

import httpx

DEFAULT_BASE_URL = "https://inference-api.nousresearch.com/v1"
DEFAULT_MODEL = "tencent/hy3:free"


class LLMError(RuntimeError):
    """Raised when the LLM call fails or returns unusable output."""


class LLMCallError(LLMError):
    """Structured provider failure; message never contains provider data."""

    def __init__(self, category: str, status: int | None = None):
        self.category = category
        self.status = status
        super().__init__(f"LLM failure: {category}" + (f" (HTTP {status})" if status else ""))


class LLMUnavailable(RuntimeError):
    """Raised when no usable LLM credential is available at call time.

    Distinct from ``LLMError`` (a transient/permanent call failure) so callers
    — especially the Telegram bot — can show a clean "LLM unavailable" message
    instead of surfacing a raw traceback to the user's phone.
    """


class LLMClient(Protocol):
    """Minimal interface every LLM client must satisfy."""

    def complete_json(
        self,
        system: str,
        user: str,
        *,
        max_tokens: int = 4096,
        temperature: float = 0.7,
    ) -> Any:
        """Return parsed JSON from the model."""
        ...


def extract_json(text: str) -> Any:
    """Best-effort extraction of a JSON value from model output.

    Handles bare JSON, ```json fenced blocks, and leading/trailing prose.
    If no JSON object/array is found, returns the stripped raw text (this lets
    reasoning-off models that reply with plain prose be consumed by callers
    that accept a string). Raises LLMError only on empty input.
    """
    if not text:
        raise LLMError("empty response from model")
    text = text.strip()
    if not text:
        raise LLMError("empty response from model")

    # Strip markdown code fences if present.
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Fall back to the outermost JSON object or array in the text.
    for opener, closer in (("[", "]"), ("{", "}")):
        start = text.find(opener)
        end = text.rfind(closer)
        if start != -1 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                continue

    # No JSON found: return the raw prose (callers may accept a string).
    return text


class NousClient:
    """Real LLM client speaking the OpenAI-compatible chat completions API."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        timeout: float = 120.0,
        max_retries: int = 2,
    ) -> None:
        self.api_key = api_key or os.environ.get("HUMORHIST_LLM_API_KEY", "")
        self.base_url = (base_url or os.environ.get("HUMORHIST_LLM_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")
        self.model = model or os.environ.get("HUMORHIST_LLM_MODEL") or DEFAULT_MODEL
        self.timeout = timeout
        self.max_retries = max_retries

    def _http_client(self) -> httpx.Client:
        return httpx.Client(timeout=self.timeout)

    def complete_json(
        self,
        system: str,
        user: str,
        *,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        reasoning_off: bool = False,
        _single_attempt: bool = False,
    ) -> Any:
        if not self.api_key:
            raise LLMError("no API key: set HUMORHIST_LLM_API_KEY or pass api_key explicitly")

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        # Opt-in: disable the model's "reasoning" mode. tencent/hy3:free defaults
        # to reasoning, which burns the token budget before emitting `content`
        # (returns content:null, finish_reason:"length", truncated reasoning in
        # the `reasoning` field). For short, direct-output tasks this both fixes
        # that crash and yields cleaner copy. Off by default to leave the
        # drafting/screen stages' behaviour unchanged.
        if reasoning_off:
            payload["reasoning"] = {"enabled": False}
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        last_error: Exception | None = None
        retries = 0 if _single_attempt else self.max_retries
        for attempt in range(retries + 1):
            try:
                with self._http_client() as client:
                    resp = client.post(
                        f"{self.base_url}/chat/completions",
                        json=payload,
                        headers=headers,
                    )
                    resp.raise_for_status()
                    body = resp.json()
                msg = body["choices"][0]["message"]
                # Fall back to the reasoning field if content is empty/None.
                content = msg.get("content") or msg.get("reasoning")
                return extract_json(content)
            except Exception as exc:  # noqa: BLE001 - retry on any transient failure
                last_error = exc
                if attempt < retries:
                    time.sleep(2**attempt)

        if isinstance(last_error, httpx.HTTPStatusError):
            raise LLMCallError("http_error", last_error.response.status_code) from None
        if isinstance(last_error, httpx.TimeoutException):
            raise LLMCallError("timeout") from None
        if isinstance(last_error, httpx.RequestError):
            raise LLMCallError("network_error") from None
        raise LLMCallError("invalid_response") from None

    def complete_json_once(self, system: str, user: str, **kwargs: Any) -> Any:
        """Make exactly one request, including on ambiguous failures."""
        return self.complete_json(system, user, _single_attempt=True, **kwargs)


class StubClient:
    """Deterministic client for tests.

    Provide ``responses`` as a list; each call to complete_json pops the next
    one. A response may be a plain Python object (returned as-is), a string
    (parsed via extract_json), or an Exception instance (raised).
    """

    def __init__(self, responses: list[Any] | None = None) -> None:
        self.responses = list(responses or [])
        self.calls: list[dict[str, Any]] = []

    def preflight(self) -> str:
        """Offline test capability: canned responses never use a provider."""
        return "free"

    def complete_json(
        self,
        system: str,
        user: str,
        *,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        reasoning_off: bool = False,
    ) -> Any:
        self.calls.append(
            {
                "system": system,
                "user": user,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "reasoning_off": reasoning_off,
            }
        )
        if not self.responses:
            raise LLMError("StubClient exhausted: no more canned responses")

        nxt = self.responses.pop(0)
        if isinstance(nxt, Exception):
            raise nxt
        if isinstance(nxt, str):
            return extract_json(nxt)
        return nxt

    def complete_json_once(self, system: str, user: str, **kwargs: Any) -> Any:
        return self.complete_json(system, user, **kwargs)


def default_client() -> LLMClient:
    """Return the configured real client."""
    return NousClient()


def _screen_base_url(base_url: str) -> None:
    """Enforce the Nous HTTPS origin; never include the supplied URL in errors."""
    try:
        parts = urlsplit(base_url)
        valid = (
            parts.scheme == "https"
            and parts.hostname == "inference-api.nousresearch.com"
            and parts.port in (None, 443)
            and parts.username is None
            and parts.password is None
            and not parts.query
            and not parts.fragment
            and not any(c in base_url for c in "\r\n\t\\")
        )
    except ValueError:
        valid = False
    if not valid:
        raise LLMUnavailable("invalid screening provider endpoint") from None


def _screen_keyfile(path: str) -> str:
    """Read a private file via trusted, non-symlink absolute-path ancestors.

    Every directory is owned by root or the effective UID and is not group- or
    world-writable, including sticky directories. Descriptor-relative traversal
    avoids replacing an ancestor with a symlink between checking and opening.
    """
    try:
        parts = path.split("/")
        if not path.startswith("/") or any(part in ("", ".", "..") for part in parts[1:]):
            raise ValueError("invalid credential file")
        with ExitStack() as stack:
            directory = os.open("/", os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
            stack.callback(os.close, directory)
            uid = os.geteuid()
            for part in parts[1:-1]:
                info = os.fstat(directory)
                if info.st_uid not in (0, uid) or info.st_mode & 0o022:
                    raise ValueError("invalid credential directory")
                directory = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=directory)
                stack.callback(os.close, directory)
            info = os.fstat(directory)
            if info.st_uid not in (0, uid) or info.st_mode & 0o022:
                raise ValueError("invalid credential directory")
            fd = os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory)
            keyfile = stack.enter_context(os.fdopen(fd, "rb"))
            info = os.fstat(keyfile.fileno())
            if not stat.S_ISREG(info.st_mode) or info.st_uid != uid or info.st_mode & 0o077:
                raise ValueError("invalid credential file")
            raw = keyfile.read(16385)
            if len(raw) > 16384:
                raise ValueError("invalid credential file")
            key = raw.decode("utf-8").strip()
            if not key:
                raise ValueError("invalid credential file")
            return key
    except (OSError, ValueError, UnicodeError):
        raise LLMUnavailable("invalid or unavailable screening credential file") from None


class ScreeningClient(NousClient):
    """Nous-only client with an exact-model anonymous catalog cost gate.

    The model is immutable after construction. Each pool run refreshes the gate;
    direct completions also gate themselves. Catalog zero prices do not guarantee
    account billing or access. There is no paid override.
    """

    def __setattr__(self, name: str, value: Any) -> None:
        if name == "_model" and hasattr(self, "_model"):
            raise AttributeError("screening model is immutable")
        super().__setattr__(name, value)

    def __delattr__(self, name: str) -> None:
        if name in {"model", "_model"}:
            raise AttributeError("screening model is immutable")
        super().__delattr__(name)

    @property
    def model(self) -> str:
        return self._model

    @model.setter
    def model(self, value: str) -> None:
        if hasattr(self, "_model"):
            raise AttributeError("screening model is immutable")
        self._model = value

    def preflight(self) -> str:
        from humorhist.screen_catalog import CatalogResult, check_screen_model

        try:
            result = check_screen_model(self.model)
        except Exception:
            # Fail closed at the catalog boundary without exposing provider details.
            return "unavailable"
        categories = {"invalid_model", "unavailable", "invalid_catalog", "model_missing", "unsafe_pricing"}
        if (isinstance(result, CatalogResult) and result.allowed is True
                and result.category == "free" and result.model_id == self.model):
            return "free"
        if isinstance(result, CatalogResult) and result.allowed is False and result.category in categories:
            return result.category
        return "unavailable"

    def _http_client(self) -> httpx.Client:
        # Avoid environment proxy configuration and redirects crossing origins.
        return httpx.Client(timeout=self.timeout, follow_redirects=False, trust_env=False)

    def complete_json(self, system: str, user: str, **kwargs: Any) -> Any:
        _screen_base_url(self.base_url)
        category = ScreeningClient.preflight(self)
        if category != "free":
            raise LLMCallError("preflight_" + category)
        kwargs["_single_attempt"] = True
        return super().complete_json(system, user, **kwargs)


def screening_client(timeout: float = 120.0) -> ScreeningClient:
    """Create a noninteractive, static-credential Nous screening client.

    A nonempty HUMORHIST_LLM_API_KEY takes precedence over
    HUMORHIST_LLM_API_KEY_FILE. The file must be owner-only, regular, and
    owned by the effective UID. Its path must be absolute, without symlinks,
    and all directories must be owned by root or the effective UID and not
    group- or world-writable (including sticky directories such as /tmp).
    OAuth is never consulted. Only HTTPS Nous on port 443 is allowed.
    HUMORHIST_SCREEN_MODEL falls back to the existing drafting model setting.
    """
    base_url = os.environ.get("HUMORHIST_LLM_BASE_URL") or DEFAULT_BASE_URL
    _screen_base_url(base_url)
    key = os.environ.get("HUMORHIST_LLM_API_KEY")
    if not key:
        path = os.environ.get("HUMORHIST_LLM_API_KEY_FILE")
        if not path:
            raise LLMUnavailable("no static screening credential available")
        key = _screen_keyfile(path)
    model = os.environ.get("HUMORHIST_SCREEN_MODEL") or os.environ.get("HUMORHIST_LLM_MODEL") or DEFAULT_MODEL
    return ScreeningClient(api_key=key, base_url=base_url, model=model, timeout=timeout, max_retries=0)


def nous_auth_token() -> str | None:
    """Best-effort read of the Nous OAuth access token from ~/.hermes/auth.json.

    This token only refreshes while a Hermes session is active and expires
    hourly; it is a convenience fallback, not a durable credential. Returns
    ``None`` if the file is absent or malformed.
    """
    try:
        path = Path.home() / ".hermes" / "auth.json"
        data = json.loads(path.read_text())
        return data["providers"]["nous"]["access_token"]
    except Exception:  # noqa: BLE001 - token is optional
        return None


def resilient_client(
    timeout: float = 120.0,
    max_retries: int = 2,
) -> NousClient:
    """Return an LLM client that works *unattended*.

    Preference order:
      1. ``HUMORHIST_LLM_API_KEY`` (persistent, survives logout — preferred for
         the always-on Telegram bot / daily timers).
      2. The Nous OAuth token from ``~/.hermes/auth.json`` (only valid while a
         Hermes session is live).

    Raises ``LLMUnavailable`` if neither is present, so callers can show a clean
    "LLM unavailable" message to the user instead of a raw traceback.
    """
    static_key = os.environ.get("HUMORHIST_LLM_API_KEY")
    if static_key:
        return NousClient(api_key=static_key, timeout=timeout, max_retries=max_retries)
    token = nous_auth_token()
    if token:
        return NousClient(api_key=token, timeout=timeout, max_retries=max_retries)
    raise LLMUnavailable(
        "no LLM credential available — set HUMORHIST_LLM_API_KEY for unattended "
        "use, or keep a Hermes session open for the Nous OAuth token"
    )
