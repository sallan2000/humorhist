import gzip
import json
from types import SimpleNamespace

import httpx
import pytest
import respx

from humorhist import screen_catalog as catalog

MODEL = "tencent/hy3:free"
URL = "https://inference-api.nousresearch.com/v1/models"
ZERO = {"prompt": "0", "completion": "0", "input_cache_read": "0"}


def response(pricing=ZERO, model=MODEL):
    return {"data": [{"id": model, "pricing": pricing}]}


@respx.mock
def test_anonymous_fixed_endpoint_and_isolated_client(monkeypatch):
    for name in ("HUMORHIST_LLM_API_KEY", "OPENAI_API_KEY", "HTTPS_PROXY", "HTTP_PROXY", "ALL_PROXY"):
        monkeypatch.setenv(name, "secret-hostile-value")
    route = respx.get(URL).mock(return_value=httpx.Response(200, json=response()))
    original = httpx.Client
    options = []

    def client(*args, **kwargs):
        options.append(kwargs.copy())
        return original(*args, **kwargs)

    monkeypatch.setattr(catalog.httpx, "Client", client)
    result = catalog.check_screen_model(MODEL)
    assert result.allowed and result.category == "free" and result.model_id == MODEL
    assert result.prices == ZERO
    assert len(route.calls) == 1
    request = route.calls[0].request
    assert request.method == "GET" and request.url == URL
    assert "authorization" not in request.headers and "proxy-authorization" not in request.headers
    assert request.headers["accept-encoding"] == "identity"
    assert options == [{"trust_env": False, "follow_redirects": False, "timeout": 5.0}]


@pytest.mark.parametrize("price", ["0", "0.000", "0e-999", "+0.0", 0, 0.0])
@respx.mock
def test_zero_values(price):
    respx.get(URL).mock(return_value=httpx.Response(200, json=response({"prompt": price, "completion": price})))
    assert catalog.check_screen_model(MODEL).allowed


@pytest.mark.parametrize("price", [None, True, False, "", " ", "0_0", "-0", "-1", "0.00001", "NaN", "Infinity", "-Infinity", [], {}, 1])
@respx.mock
def test_unsafe_values_fail_closed(price):
    respx.get(URL).mock(return_value=httpx.Response(200, json=response({"prompt": price, "completion": "0"})))
    result = catalog.check_screen_model(MODEL)
    assert not result.allowed and result.category == "unsafe_pricing" and result.prices == {}


@pytest.mark.parametrize("pricing", [None, {}, {"prompt": "0"}, {"completion": "0"}, [], "free", {"prompt": "0", "completion": "0", "other": "0.001"}, {"prompt": "0", "completion": "0", "other": None}])
@respx.mock
def test_missing_or_unknown_pricing(pricing):
    respx.get(URL).mock(return_value=httpx.Response(200, json=response(pricing)))
    assert catalog.check_screen_model(MODEL).category == "unsafe_pricing"


@respx.mock
def test_unknown_zero_price_is_checked_but_not_echoed():
    pricing = {**ZERO, "SECRET-UNTRUSTED-FIELD": "0e20"}
    respx.get(URL).mock(return_value=httpx.Response(200, json=response(pricing)))
    result = catalog.check_screen_model(MODEL)
    assert result.allowed and result.prices == ZERO
    assert "SECRET" not in repr(result)


@pytest.mark.parametrize("data,category", [
    ([{"id": MODEL + "x", "pricing": ZERO}], "model_missing"),
    ([{"id": MODEL.upper(), "pricing": ZERO}], "model_missing"),
    ([{"id": MODEL, "pricing": ZERO}, {"id": MODEL, "pricing": {"prompt": "1", "completion": "0"}}], "invalid_catalog"),
    ([{"id": MODEL, "pricing": ZERO}, {"id": MODEL, "pricing": ZERO}], "invalid_catalog"),
    ([None], "invalid_catalog"),
    ([{}], "invalid_catalog"),
    ([{"id": "", "pricing": ZERO}], "invalid_catalog"),
    ([{"id": "  \t", "pricing": ZERO}], "invalid_catalog"),
    ([{"id": MODEL, "pricing": ZERO}, {"id": ""}], "invalid_catalog"),
    ([{"id": MODEL, "pricing": ZERO}, {"id": " \n"}], "invalid_catalog"),
    ([{"id": MODEL, "pricing": ZERO}, {}], "invalid_catalog"),
    ([{"id": " " + MODEL + " ", "pricing": ZERO}], "model_missing"),
])
@respx.mock
def test_exact_match_and_ambiguous_catalog(data, category):
    respx.get(URL).mock(return_value=httpx.Response(200, json={"data": data}))
    assert catalog.check_screen_model(MODEL).category == category


@pytest.mark.parametrize("body", [b"not json secret", b"[]", b'{"data":{}}', b'{"data":[],"data":[]}', b'{"data":[],"x":NaN}', b'\xff'])
@respx.mock
def test_malformed_and_duplicate_json(body):
    respx.get(URL).mock(return_value=httpx.Response(200, content=body))
    result = catalog.check_screen_model(MODEL)
    assert result.category == "invalid_catalog"
    assert "secret" not in repr(result)


@pytest.mark.parametrize("status", [301, 302, 307, 308, 401, 429, 500])
@respx.mock
def test_statuses_never_redirect_or_retry(status):
    route = respx.get(URL).mock(return_value=httpx.Response(status, headers={"Location": "https://secret.invalid/"}, text="secret"))
    result = catalog.check_screen_model(MODEL)
    assert result.category == "unavailable" and not result.allowed
    assert len(route.calls) == 1 and "secret" not in repr(result)


@respx.mock
def test_oversized_stream_and_compression():
    route = respx.get(URL).mock(return_value=httpx.Response(200, content=b"x" * (catalog._MAX_BYTES + 1)))
    assert catalog.check_screen_model(MODEL).category == "unavailable"
    route.mock(return_value=httpx.Response(200, headers={"content-encoding": "gzip"}, content=gzip.compress(b"secret")))
    assert catalog.check_screen_model(MODEL).category == "invalid_catalog"


@respx.mock
def test_active_deadline_during_stream(monkeypatch):
    class Stream(httpx.SyncByteStream):
        def __iter__(self):
            yield b'{"data":'
            clock[0] = 6.0
            yield b'[]}'

    clock = [0.0]
    monkeypatch.setattr(catalog, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    respx.get(URL).mock(return_value=httpx.Response(200, stream=Stream()))
    assert catalog.check_screen_model(MODEL).category == "unavailable"


@respx.mock
def test_network_exception_redacted_without_retry():
    route = respx.get(URL).mock(side_effect=httpx.ConnectError("SECRET https://private.invalid"))
    result = catalog.check_screen_model(MODEL)
    assert result.category == "unavailable" and len(route.calls) == 1
    assert "SECRET" not in repr(result) and "private.invalid" not in repr(result)


@pytest.mark.parametrize("model", ["", "foo?token=secret", "foo\nsecret", "é", "x" * 201, None])
def test_invalid_model_not_echoed(model):
    result = catalog.check_screen_model(model)
    assert result.category == "invalid_model" and result.model_id is None
    assert "secret" not in repr(result)


@respx.mock
def test_paid_override_environment_is_ignored(monkeypatch):
    monkeypatch.setenv("HUMORHIST_ALLOW_PAID_MODELS", "1")
    respx.get(URL).mock(return_value=httpx.Response(200, json=response({"prompt": "0.01", "completion": "0"})))
    assert not catalog.check_screen_model(MODEL).allowed


@pytest.mark.parametrize("numeric", ["1e-9999", "-0", "-0.0"])
@respx.mock
def test_decimal_json_numbers_preserve_precision(numeric):
    body = json.dumps(response({"prompt": "0", "completion": "0"})).replace('"prompt": "0"', f'"prompt": {numeric}')
    respx.get(URL).mock(return_value=httpx.Response(200, content=body))
    assert catalog.check_screen_model(MODEL).category == "unsafe_pricing"
