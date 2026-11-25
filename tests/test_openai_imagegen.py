"""OpenAI image transport tests: all HTTP requests are mocked."""
import base64
import json
import struct
import zlib

import httpx
import pytest
import respx

from humorhist import imagegen as ig

URL = "https://api.openai.com/v1/images/generations"


def _chunk(kind, data):
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))


PNG = (b"\x89PNG\r\n\x1a\n"
       + _chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 6, 0, 0, 0))
       + _chunk(b"IDAT", zlib.compress(b"\x00\xff\x00\x00\xff"))
       + _chunk(b"IEND", b""))


def test_openai_request_and_decode():
    client = ig.OpenAIImageClient("test-key", "gpt-image-2.5-flare", "low")
    with respx.mock as mock:
        route = mock.post(URL).mock(return_value=httpx.Response(200, json={"data": [{"b64_json": base64.b64encode(PNG).decode()}]}))
        assert client.generate("a bear") == PNG
        request = route.calls[0].request
        assert request.headers["Authorization"] == "Bearer test-key"
        assert json.loads(request.content) == {"model": "gpt-image-2.5-flare", "quality": "low", "size": "1024x1024", "output_format": "png", "n": 1, "prompt": "a bear"}


@pytest.mark.parametrize("body", [{}, {"data": []}, {"data": [{"b64_json": "!!!"}]}, {"data": [{"b64_json": ""}]}])
def test_bad_response_is_not_retried(body):
    with respx.mock as mock:
        route = mock.post(URL).mock(return_value=httpx.Response(200, json=body))
        with pytest.raises(ig.ImageError, match="invalid or empty"):
            ig.OpenAIImageClient("test-key", "gpt-image-2.5-flare", "low").generate("a bear")
        assert route.call_count == 1


@pytest.mark.parametrize("image", [b"not a png", PNG[:-1], PNG[:40] + bytes([PNG[40] ^ 1]) + PNG[41:],
                                   PNG[:33] + _chunk(b"IDAT", zlib.compress(b"\x00")) + _chunk(b"IEND", b"")])
def test_invalid_png_rejected_without_retry(image):
    with respx.mock as mock:
        route = mock.post(URL).mock(return_value=httpx.Response(200, json={"data": [{"b64_json": base64.b64encode(image).decode()}]}))
        with pytest.raises(ig.ImageError, match="invalid or empty"):
            ig.OpenAIImageClient("test-key", "gpt-image-2.5-flare", "low").generate("a bear")
        assert route.call_count == 1


@pytest.mark.parametrize("response", [httpx.Response(429), httpx.Response(500), httpx.ReadTimeout("timeout")])
def test_failures_not_retried(response):
    with respx.mock as mock:
        route = mock.post(URL).mock(side_effect=response if isinstance(response, Exception) else None, return_value=None if isinstance(response, Exception) else response)
        with pytest.raises(ig.ImageError, match="not retried"):
            ig.OpenAIImageClient("test-key", "gpt-image-2.5-flare", "low").generate("a bear")
        assert route.call_count == 1


def test_provider_selection_and_separate_credentials(monkeypatch):
    for name in ("HUMORHIST_IMAGE_PROVIDER", "HUMORHIST_IMAGE_API_KEY", "OPENAI_API_KEY", "HUMORHIST_IMAGE_MODEL", "HUMORHIST_IMAGE_QUALITY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai")
    with pytest.raises(ig.ImageUnavailable):
        ig.resilient_image_client()  # OpenAI credential alone does not enable it
    monkeypatch.setenv("HUMORHIST_IMAGE_API_KEY", "test-fal")
    assert isinstance(ig.resilient_image_client(), ig.FalClient)
    monkeypatch.setenv("HUMORHIST_IMAGE_PROVIDER", "openai")
    c = ig.resilient_image_client()
    assert isinstance(c, ig.OpenAIImageClient)
    assert (c.api_key, c.model, c.quality) == ("test-openai", "gpt-image-2.5-flare", "low")
    monkeypatch.delenv("OPENAI_API_KEY")
    with pytest.raises(ig.ImageUnavailable):
        ig.resilient_image_client()  # never fall back to FAL key
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai")
    monkeypatch.setenv("HUMORHIST_IMAGE_QUALITY", "auto")
    with pytest.raises(ig.ImageUnavailable):
        ig.resilient_image_client()
    monkeypatch.setenv("HUMORHIST_IMAGE_PROVIDER", "typo")
    with pytest.raises(ig.ImageUnavailable):
        ig.resilient_image_client()
