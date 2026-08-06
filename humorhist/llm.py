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
import time
from typing import Any, Protocol

import httpx

DEFAULT_BASE_URL = "https://inference-api.nousresearch.com/v1"
DEFAULT_MODEL = "tencent/hy3:free"


class LLMError(RuntimeError):
    """Raised when the LLM call fails or returns unusable output."""


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
        self.base_url = (
            base_url or os.environ.get("HUMORHIST_LLM_BASE_URL") or DEFAULT_BASE_URL
        ).rstrip("/")
        self.model = model or os.environ.get("HUMORHIST_LLM_MODEL") or DEFAULT_MODEL
        self.timeout = timeout
        self.max_retries = max_retries

    def complete_json(
        self,
        system: str,
        user: str,
        *,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        reasoning_off: bool = False,
    ) -> Any:
        if not self.api_key:
            raise LLMError(
                "no API key: set HUMORHIST_LLM_API_KEY or pass api_key explicitly"
            )

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
        for attempt in range(self.max_retries + 1):
            try:
                with httpx.Client(timeout=self.timeout) as client:
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
                if attempt < self.max_retries:
                    time.sleep(2 ** attempt)

        raise LLMError(f"LLM call failed after {self.max_retries + 1} attempts: {last_error}")


class StubClient:
    """Deterministic client for tests.

    Provide ``responses`` as a list; each call to complete_json pops the next
    one. A response may be a plain Python object (returned as-is), a string
    (parsed via extract_json), or an Exception instance (raised).
    """

    def __init__(self, responses: list[Any] | None = None) -> None:
        self.responses = list(responses or [])
        self.calls: list[dict[str, Any]] = []

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


def default_client() -> LLMClient:
    """Return the configured real client."""
    return NousClient()
