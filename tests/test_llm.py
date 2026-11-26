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
    text = "```\n[1, 2, 3]\n```"
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
    return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})


@respx.mock
def test_nous_client_parses_response():
    route = respx.post(f"{DEFAULT_BASE_URL}/chat/completions").mock(return_value=_chat_response('{"score": 9}'))
    client = NousClient(api_key="test-key")
    assert client.complete_json("sys", "user") == {"score": 9}
    assert route.called
    sent = json.loads(route.calls[0].request.content)
    assert sent["messages"][0]["content"] == "sys"
    assert sent["messages"][1]["content"] == "user"


@respx.mock
def test_nous_client_sends_auth_header():
    route = respx.post(f"{DEFAULT_BASE_URL}/chat/completions").mock(return_value=_chat_response("{}"))
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
    respx.post(f"{DEFAULT_BASE_URL}/chat/completions").mock(return_value=httpx.Response(500))
    client = NousClient(api_key="k", max_retries=1)
    with pytest.raises(LLMError, match="http_error"):
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
    (hermes / "auth.json").write_text(json.dumps({"providers": {"nous": {"access_token": "oauth-token"}}}))
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
    (hermes / "auth.json").write_text(json.dumps({"providers": {"nous": {"access_token": "abc"}}}))
    monkeypatch.setattr("humorhist.llm.Path.home", lambda: tmp_path)
    assert nous_auth_token() == "abc"


def test_nous_auth_token_returns_none_when_missing(monkeypatch, tmp_path):
    monkeypatch.setattr("humorhist.llm.Path.home", lambda: tmp_path)
    assert nous_auth_token() is None


@pytest.fixture
def screen_env(monkeypatch, tmp_path):
    import os
    import stat

    for name in ("HUMORHIST_LLM_API_KEY_FILE", "HUMORHIST_LLM_BASE_URL", "HUMORHIST_SCREEN_MODEL", "HUMORHIST_LLM_MODEL"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr("humorhist.llm.nous_auth_token", lambda: pytest.fail("OAuth must not be read"))
    # Test roots can live below /tmp or another group-writable directory.
    # Simulate trusted ancestors outside tmp_path only; test-created
    # descendants retain their real ownership and permissions.
    ancestors = {(info.st_dev, info.st_ino) for path in tmp_path.parents if (info := path.stat())}
    original_fstat = os.fstat

    def trusted_test_ancestor(fd):
        info = original_fstat(fd)
        if (info.st_dev, info.st_ino) in ancestors:
            fields = list(info)
            fields[stat.ST_MODE] = info.st_mode & ~0o022
            return os.stat_result(fields)
        return info

    monkeypatch.setattr("humorhist.llm.os.fstat", trusted_test_ancestor)


def test_screen_requires_static_credential(screen_env):
    from humorhist.llm import screening_client

    with pytest.raises(LLMUnavailable):
        screening_client()


def test_screen_model_and_draft_defaults(screen_env, monkeypatch):
    from humorhist.llm import DEFAULT_MODEL, default_client, screening_client

    monkeypatch.setenv("HUMORHIST_LLM_API_KEY", "test-secret")
    assert screening_client().model == default_client().model == DEFAULT_MODEL
    monkeypatch.setenv("HUMORHIST_LLM_MODEL", "draft-model")
    assert screening_client().model == default_client().model == "draft-model"
    monkeypatch.setenv("HUMORHIST_SCREEN_MODEL", "screen-model")
    assert screening_client().model == "screen-model"
    assert default_client().model == "draft-model"


def test_screen_keyfile_and_env_precedence(screen_env, monkeypatch, tmp_path):
    from humorhist.llm import screening_client

    keyfile = tmp_path / "private-key"
    keyfile.write_text("file-secret\n")
    keyfile.chmod(0o600)
    monkeypatch.setenv("HUMORHIST_LLM_API_KEY_FILE", str(keyfile))
    assert screening_client().api_key == "file-secret"
    monkeypatch.setenv("HUMORHIST_LLM_API_KEY", "env-secret")
    keyfile.chmod(0o644)
    assert screening_client().api_key == "env-secret"


@pytest.mark.parametrize(
    "kind", ["symlink_parent", "group_writable", "world_writable", "sticky_writable", "foreign_owner", "relative"]
)
def test_screen_rejects_untrusted_keyfile_ancestors(screen_env, monkeypatch, tmp_path, kind):
    import os
    import stat
    import traceback

    from humorhist.llm import screening_client

    parent = tmp_path / "sensitive-parent"
    parent.mkdir(mode=0o700)
    keyfile = parent / "sensitive-file"
    keyfile.write_text("sensitive-content")
    keyfile.chmod(0o600)
    if kind == "symlink_parent":
        link = tmp_path / "sensitive-link"
        link.symlink_to(parent, target_is_directory=True)
        keyfile = link / keyfile.name
    elif kind == "group_writable":
        parent.chmod(0o720)
    elif kind == "world_writable":
        parent.chmod(0o702)
    elif kind == "sticky_writable":
        parent.chmod(0o1777)
    elif kind == "foreign_owner":
        parent_info = parent.stat()
        original_fstat = os.fstat

        def foreign_parent(fd):
            info = original_fstat(fd)
            if (info.st_dev, info.st_ino) == (parent_info.st_dev, parent_info.st_ino):
                fields = list(info)
                fields[stat.ST_UID] = 123456789
                return os.stat_result(fields)
            return info

        monkeypatch.setattr("humorhist.llm.os.fstat", foreign_parent)
    else:
        monkeypatch.chdir(tmp_path)
        keyfile = keyfile.relative_to(tmp_path)
    monkeypatch.setenv("HUMORHIST_LLM_API_KEY_FILE", str(keyfile))
    with pytest.raises(LLMUnavailable) as exc:
        screening_client()
    output = "".join(traceback.format_exception(exc.value))
    assert "sensitive" not in output


@pytest.mark.parametrize("kind", ["permissive", "symlink", "directory", "missing", "empty", "wrong_owner"])
def test_screen_rejects_bad_keyfiles_without_leaks(screen_env, monkeypatch, tmp_path, kind):
    from humorhist.llm import screening_client

    keyfile = tmp_path / "sensitive-path"
    if kind == "directory":
        keyfile.mkdir()
    elif kind != "missing":
        keyfile.write_text("sensitive-secret" if kind != "empty" else "  \n")
        keyfile.chmod(0o644 if kind == "permissive" else 0o600)
    if kind == "symlink":
        link = tmp_path / "link"
        link.symlink_to(keyfile)
        keyfile = link
    if kind == "wrong_owner":
        monkeypatch.setattr("humorhist.llm.os.geteuid", lambda: -1)
    monkeypatch.setenv("HUMORHIST_LLM_API_KEY_FILE", str(keyfile))
    with pytest.raises(LLMUnavailable) as exc:
        screening_client()
    assert "sensitive" not in str(exc.value)
    assert str(keyfile) not in str(exc.value)
    assert exc.value.__cause__ is None


@pytest.mark.parametrize("url", [
    "http://inference-api.nousresearch.com/v1",
    "https://evil.example/v1",
    "https://inference-api.nousresearch.com.evil.example/v1",
    "https://user:secret@inference-api.nousresearch.com/v1",
    "https://inference-api.nousresearch.com:444/v1",
    "https://inference-api.nousresearch.com/v1?secret=value",
])
def test_screen_rejects_unsafe_endpoint(screen_env, monkeypatch, url):
    from humorhist.llm import screening_client

    monkeypatch.setenv("HUMORHIST_LLM_API_KEY", "secret-key")
    monkeypatch.setenv("HUMORHIST_LLM_BASE_URL", url)
    with pytest.raises(LLMUnavailable) as exc:
        screening_client()
    assert "secret" not in str(exc.value)
    assert url not in str(exc.value)


def test_screen_catalog_exception_fails_closed_without_leaking(screen_env, monkeypatch, capsys):
    import traceback

    from humorhist.llm import LLMCallError, screening_client

    monkeypatch.setenv("HUMORHIST_LLM_API_KEY", "test-private-key")

    def broken_catalog(_model):
        raise RuntimeError("test-private-key sensitive-provider-detail")

    monkeypatch.setattr("humorhist.screen_catalog.check_screen_model", broken_catalog)
    client = screening_client()
    assert client.preflight() == "unavailable"
    with respx.mock(assert_all_called=False) as mock:
        post = mock.post(f"{DEFAULT_BASE_URL}/chat/completions").mock()
        with pytest.raises(LLMCallError) as exc:
            client.complete_json("s", "u")
        assert exc.value.category == "preflight_unavailable"
        assert post.call_count == 0
    traceback.print_exception(exc.value)
    captured = capsys.readouterr()
    assert "test-private-key" not in captured.out + captured.err
    assert "sensitive-provider-detail" not in captured.out + captured.err


def test_direct_screening_fails_closed_without_post(screen_env, monkeypatch):
    import respx

    from humorhist.llm import LLMCallError, screening_client

    monkeypatch.setenv("HUMORHIST_LLM_API_KEY", "private-key")
    monkeypatch.setenv("HUMORHIST_SCREEN_MODEL", "exact")
    with respx.mock(assert_all_called=False) as mock:
        catalog = mock.get("https://inference-api.nousresearch.com/v1/models").mock(
            return_value=httpx.Response(200, json={"data": [{"id": "exact", "pricing": {"prompt": "0", "completion": "1"}}]}))
        post = mock.post("https://inference-api.nousresearch.com/v1/chat/completions").mock()
        with pytest.raises(LLMCallError) as exc:
            screening_client().complete_json("s", "u")
        assert exc.value.category == "preflight_unsafe_pricing"
        assert catalog.call_count == 1 and post.call_count == 0
        assert "authorization" not in catalog.calls[0].request.headers
        assert "private-key" not in str(exc.value)


@pytest.mark.parametrize("status", [301, 307, 400, 429, 500])
def test_screen_one_request_no_redirect_no_retry(screen_env, monkeypatch, status):
    from humorhist.llm import LLMCallError, screening_client

    monkeypatch.setenv("HUMORHIST_LLM_API_KEY", "secret-key")
    seen = []

    def handler(request):
        seen.append(request)
        if request.method == "GET":
            assert "authorization" not in request.headers
            return httpx.Response(200, stream=httpx.ByteStream(b'{"data":[{"id":"tencent/hy3:free","pricing":{"prompt":"0","completion":"0"}}]}'))
        return httpx.Response(status, headers={"Location": "https://evil.example/steal"}, text="secret-key")

    real_client = httpx.Client
    monkeypatch.setattr("humorhist.llm.httpx.Client", lambda **kw: real_client(transport=httpx.MockTransport(handler), **kw))
    client = screening_client()
    with pytest.raises(LLMCallError) as exc:
        client.complete_json_once("s", "u")
    assert exc.value.status == status
    assert "secret-key" not in str(exc.value)
    assert len(seen) == 2


def test_screen_success_and_endpoint_revalidation(screen_env, monkeypatch):
    from humorhist.llm import screening_client

    monkeypatch.setenv("HUMORHIST_LLM_API_KEY", "secret-key")
    seen = []

    def handler(request):
        seen.append(request)
        if request.method == "GET":
            assert "authorization" not in request.headers
            return httpx.Response(200, stream=httpx.ByteStream(b'{"data":[{"id":"tencent/hy3:free","pricing":{"prompt":"0","completion":"0"}}]}'))
        return _chat_response('{"ok": true}')

    real_client = httpx.Client
    monkeypatch.setattr("humorhist.llm.httpx.Client", lambda **kw: real_client(transport=httpx.MockTransport(handler), **kw))
    client = screening_client()
    assert client.complete_json_once("s", "u") == {"ok": True}
    assert len(seen) == 2
    client.base_url = "https://evil.example"
    with pytest.raises(LLMUnavailable):
        client.complete_json_once("s", "u")
    assert len(seen) == 2


def test_screen_timeout_is_single_attempt_and_does_not_print_secrets(screen_env, monkeypatch, capsys):
    import traceback

    from humorhist.llm import LLMCallError, screening_client

    monkeypatch.setenv("HUMORHIST_LLM_API_KEY", "sensitive-token")
    monkeypatch.setenv("HUMORHIST_LLM_BASE_URL", "https://inference-api.nousresearch.com:443/v1")
    seen = []

    def handler(request):
        seen.append(request)
        if request.method == "GET":
            assert "authorization" not in request.headers
            return httpx.Response(200, stream=httpx.ByteStream(b'{"data":[{"id":"tencent/hy3:free","pricing":{"prompt":"0","completion":"0"}}]}'))
        raise httpx.ReadTimeout("sensitive-token", request=request)

    real_client = httpx.Client
    monkeypatch.setattr("humorhist.llm.httpx.Client", lambda **kw: real_client(transport=httpx.MockTransport(handler), **kw))
    client = screening_client()
    client.max_retries = 10
    with pytest.raises(LLMCallError) as exc:
        client.complete_json("s", "u")
    assert exc.value.category == "timeout"
    assert len(seen) == 2
    traceback.print_exception(exc.value)
    assert "sensitive-token" not in capsys.readouterr().err
