"""Tests for the real FAL image client (humorhist.imagegen.FalClient).

The generation flow is: POST to the queue endpoint (returns a status_url),
poll status_url until "completed", then GET the resulting image URL and return
its bytes. `respx` mocks each HTTP leg so no network is touched. The
StubImageClient path is covered by the pipeline tests; here we guard the
production transport that runs at publish time.
"""

from __future__ import annotations

import httpx
import pytest
import respx

import humorhist.imagegen as ig

FAL = "https://queue.fal.run"


def _client(api_key: str = "FAKEYEY") -> ig.FalClient:
    return ig.FalClient(api_key=api_key, timeout=10.0, max_retries=0)


def test_generate_happy_path():
    model = "fal-ai/flux/dev"
    with respx.mock(assert_all_called=False) as rt:
        # 1) submit
        rt.post(f"{FAL}/{model}").mock(
            return_value=httpx.Response(
                200, json={"status_url": "https://queue.fal.run/status/abc"}
            )
        )
        # 2) poll -> completed with an image url
        rt.get("https://queue.fal.run/status/abc").mock(
            return_value=httpx.Response(
                200,
                json={
                    "status": "completed",
                    "images": [{"url": "https://v3.fal.media/img1.png"}],
                },
            )
        )
        # 3) download the image bytes
        rt.get("https://v3.fal.media/img1.png").mock(
            return_value=httpx.Response(200, content=b"PNG-BYTES")
        )
        client = _client()
        data = client.generate("a duck in a wig")
    assert data == b"PNG-BYTES"


def test_generate_leading_slash_model_stripped():
    with respx.mock(assert_all_called=False) as rt:
        submit = rt.post(f"{FAL}/fal-ai/flux/dev").mock(
            return_value=httpx.Response(
                200, json={"status_url": "https://queue.fal.run/status/x"}
            )
        )
        rt.get("https://queue.fal.run/status/x").mock(
            return_value=httpx.Response(
                200, json={"status": "completed", "images": [{"url": "https://v3.fal.media/u"}]}
            )
        )
        rt.get("https://v3.fal.media/u").mock(return_value=httpx.Response(200, content=b"BBB"))
        # model supplied with a leading slash -> client strips it
        client = ig.FalClient(api_key="K", model="/fal-ai/flux/dev", timeout=5.0)
        assert client.generate("x") == b"BBB"
    assert submit.called


def test_generate_no_status_url_raises():
    with respx.mock(assert_all_called=False) as rt:
        rt.post(f"{FAL}/fal-ai/flux/dev").mock(
            return_value=httpx.Response(200, json={"something": "else"})
        )
        client = _client()
        try:
            client.generate("prompt")
        except ig.ImageError as exc:
            assert "no status_url" in str(exc)
        else:
            raise AssertionError("expected ImageError when status_url missing")


def test_generate_poll_error_raises():
    with respx.mock(assert_all_called=False) as rt:
        rt.post(f"{FAL}/fal-ai/flux/dev").mock(
            return_value=httpx.Response(
                200, json={"status_url": "https://queue.fal.run/status/e"}
            )
        )
        rt.get("https://queue.fal.run/status/e").mock(
            return_value=httpx.Response(200, json={"status": "error", "reason": "boom"})
        )
        client = _client()
        try:
            client.generate("p")
        except ig.ImageError as exc:
            assert "errored" in str(exc)
        else:
            raise AssertionError("expected ImageError on FAL job error")


def test_generate_completed_without_images_raises():
    with respx.mock(assert_all_called=False) as rt:
        rt.post(f"{FAL}/fal-ai/flux/dev").mock(
            return_value=httpx.Response(
                200, json={"status_url": "https://queue.fal.run/status/empty"}
            )
        )
        rt.get("https://queue.fal.run/status/empty").mock(
            return_value=httpx.Response(200, json={"status": "completed", "images": []})
        )
        client = _client()
        try:
            client.generate("p")
        except ig.ImageError:
            pass
        else:
            raise AssertionError("expected ImageError when completed but no images")


def test_generate_missing_api_key_raises(monkeypatch):
    monkeypatch.delenv("HUMORHIST_IMAGE_API_KEY", raising=False)
    client = ig.FalClient(api_key="", timeout=1.0)
    try:
        client.generate("p")
    except ig.ImageError as exc:
        assert "no API key" in str(exc)
    else:
        raise AssertionError("expected ImageError for missing key")


def test_generate_does_not_retry_paid_submission():
    with respx.mock(assert_all_called=False) as rt:
        route = rt.post(f"{FAL}/fal-ai/flux/dev").mock(return_value=httpx.Response(500))
        client = ig.FalClient(api_key="K", timeout=5.0, max_retries=2)
        with pytest.raises(ig.ImageError, match="not retried"):
            client.generate("p")
        assert route.call_count == 1
