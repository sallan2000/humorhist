"""Tests for the LLM client abstraction."""

from __future__ import annotations

import json

import httpx
import pytest
import respx

from humorhist.llm import (
    DEFAULT_BASE_URL,
    LLMError,
    LLMUnavailable,
    NousClient,
    StubClient,
    extract_json,
    nous_auth_token,
    resilient_client,
)


# --- extract_json -----------------------------------------------------------


def test_extract_json_bare_object():
    assert extract_json('{"a": 1}') == {"a": 1}


def test_extract_json_bare_array():
    assert extract_json('[{"a": 1}, {"b": 2}]') == [{"a": 1}, {"b": 2}]


def test_extract_json_fenced():
    text = 'Here you go:\n```json\n{"score": 7}\n```\nHope that helps!'
    assert extract_json(text) == {"score": 7}


def test_extract_json_fenced_without_language():
    text = '```\n[1, 2, 3]\n```'
    assert extract_json(text) == [1, 2, 3]


def test_extract_json_with_surrounding_prose():
    text = 'Sure thing. {"verdict": "ok"} Let me know if you need more.'
    assert extract_json(text) == {"verdict": "ok"}


def test_extract_json_raises_on_empty():
    with pytest.raises(LLMError):
        extract_json("   ")


def test_extract_json_returns_raw_text_on_garbage():
    # No JSON present: return the stripped prose so reasoning-off models that
    # reply with plain text (not a JSON wrapper) are still consumable.
    assert extract_json("there is no json here at all") == "there is no json here at all"


# --- StubClient -------------------------------------------------------------


def test_stub_returns_canned_objects_in_order():
    stub = StubClient([{"first": True}, {"second": True}])
    assert stub.complete_json("s", "u") == {"first": True}
    assert stub.complete_json("s", "u") == {"second": True}


def test_stub_parses_string_responses():
    stub = StubClient(['```json\n{"parsed": 1}\n```'])
    assert stub.complete_json("s", "u") == {"parsed": 1}


def test_stub_raises_exception_responses():
    stub = StubClient([LLMError("boom")])
    with pytest.raises(LLMError, match="boom"):
        stub.complete_json("s", "u")


def test_stub_records_calls():
    stub = StubClient([{"ok": 1}])
    stub.complete_json("SYS", "USER", temperature=0.2)
    assert stub.calls[0]["system"] == "SYS"
    assert stub.calls[0]["user"] == "USER"
    assert stub.calls[0]["temperature"] == 0.2


def test_stub_raises_when_exhausted():
    stub = StubClient([])
    with pytest.raises(LLMError, match="exhausted"):
        stub.complete_json("s", "u")


# --- NousClient -------------------------------------------------------------


def _chat_response(content: str) -> httpx.Response:
    return httpx.Response(
        200, json={"choices": [{"message": {"content": content}}]}
    )


@respx.mock
def test_nous_client_parses_response():
    route = respx.post(f"{DEFAULT_BASE_URL}/chat/completions").mock(
        return_value=_chat_response('{"score": 9}')
    )
    client = NousClient(api_key="test-key")
    assert client.complete_json("sys", "user") == {"score": 9}
    assert route.called
    sent = json.loads(route.calls[0].request.content)
    assert sent["messages"][0]["content"] == "sys"
    assert sent["messages"][1]["content"] == "user"


@respx.mock
def test_nous_client_sends_auth_header():
    route = respx.post(f"{DEFAULT_BASE_URL}/chat/completions").mock(
        return_value=_chat_response("{}")
    )
    NousClient(api_key="secret123").complete_json("s", "u")
    assert route.calls[0].request.headers["Authorization"] == "Bearer secret123"


def test_nous_client_requires_api_key(monkeypatch):
    monkeypatch.delenv("HUMORHIST_LLM_API_KEY", raising=False)
    with pytest.raises(LLMError, match="no API key"):
        NousClient(api_key="").complete_json("s", "u")


@respx.mock
def test_nous_client_retries_then_succeeds():
    route = respx.post(f"{DEFAULT_BASE_URL}/chat/completions").mock(
        side_effect=[
            httpx.Response(500),
            _chat_response('{"recovered": true}'),
        ]
    )
    client = NousClient(api_key="k", max_retries=2)
    assert client.complete_json("s", "u") == {"recovered": True}
    assert route.call_count == 2


@respx.mock
def test_nous_client_raises_after_exhausting_retries():
    respx.post(f"{DEFAULT_BASE_URL}/chat/completions").mock(
        return_value=httpx.Response(500)
    )
    client = NousClient(api_key="k", max_retries=1)
    with pytest.raises(LLMError, match="failed after 2 attempts"):
        client.complete_json("s", "u")


def test_nous_client_reads_env(monkeypatch):
    monkeypatch.setenv("HUMORHIST_LLM_API_KEY", "env-key")
    monkeypatch.setenv("HUMORHIST_LLM_MODEL", "env-model")
    client = NousClient()
    assert client.api_key == "env-key"
    assert client.model == "env-model"


# --- resilient_client (unattended credential resolution) -------------------


def test_resilient_client_prefers_static_api_key(monkeypatch, tmp_path):
    monkeypatch.setenv("HUMORHIST_LLM_API_KEY", "static-key")
    # with a static key set, a missing/garbage auth.json must not break resolution
    monkeypatch.setattr("humorhist.llm.Path.home", lambda: tmp_path)
    client = resilient_client()
    assert isinstance(client, NousClient)
    assert client.api_key == "static-key"


def test_resilient_client_falls_back_to_nous_auth_token(monkeypatch, tmp_path):
    monkeypatch.delenv("HUMORHIST_LLM_API_KEY", raising=False)
    hermes = tmp_path / ".hermes"
    hermes.mkdir()
    (hermes / "auth.json").write_text(
        json.dumps({"providers": {"nous": {"access_token": "oauth-token"}}})
    )
    monkeypatch.setattr("humorhist.llm.Path.home", lambda: tmp_path)
    client = resilient_client()
    assert client.api_key == "oauth-token"


def test_resilient_client_raises_llm_unavailable_with_no_creds(monkeypatch, tmp_path):
    monkeypatch.delenv("HUMORHIST_LLM_API_KEY", raising=False)
    # no .hermes/auth.json present
    monkeypatch.setattr("humorhist.llm.Path.home", lambda: tmp_path)
    with pytest.raises(LLMUnavailable):
        resilient_client()


def test_nous_auth_token_reads_file(monkeypatch, tmp_path):
    hermes = tmp_path / ".hermes"
    hermes.mkdir()
    (hermes / "auth.json").write_text(
        json.dumps({"providers": {"nous": {"access_token": "abc"}}})
    )
    monkeypatch.setattr("humorhist.llm.Path.home", lambda: tmp_path)
    assert nous_auth_token() == "abc"


def test_nous_auth_token_returns_none_when_missing(monkeypatch, tmp_path):
    monkeypatch.setattr("humorhist.llm.Path.home", lambda: tmp_path)
    assert nous_auth_token() is None
