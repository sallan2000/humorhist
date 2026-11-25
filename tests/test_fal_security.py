"""FAL URL trust boundaries; all requests are intercepted by respx."""
import httpx
import pytest
import respx

from humorhist.imagegen import FalClient, ImageError

BASE = "https://queue.fal.run"
SUBMIT = BASE + "/fal-ai/flux/dev"
STATUS = BASE + "/fal-ai/flux/requests/abc/status"
RESULT = BASE + "/fal-ai/flux/requests/abc"
IMAGE = "https://v3.fal.media/files/image.png?token=signed"


def client():
    return FalClient(api_key="secret", timeout=1)


@pytest.mark.parametrize("bad", [
    "https://evil.test/status", "http://queue.fal.run/status",
    "https://queue.fal.run.evil.test/status", "https://evil.test@queue.fal.run/status",
    "https://queue.fal.run:444/status", "https://127.0.0.1/status",
    "https://queue.fal.run\\@evil.test/status", "//queue.fal.run/status",
    "https://queue.fal.run/status\n",
])
@pytest.mark.parametrize("field", ["status_url", "response_url"])
def test_untrusted_authenticated_urls_rejected_before_get(bad, field):
    body = {"status_url": STATUS, "response_url": RESULT}
    body[field] = bad
    with respx.mock(assert_all_called=False) as router:
        router.post(SUBMIT).mock(return_value=httpx.Response(200, json=body))
        with pytest.raises(ImageError):
            client().generate("prompt")
        assert len(router.calls) == 1


@pytest.mark.parametrize("bad", [
    "http://v3.fal.media/files/x", "https://v3.fal.media.evil.test/x",
    "https://u@v3.fal.media/x", "https://v3.fal.media:8443/x",
    "https://169.254.169.254/latest/meta-data", "https://evil.test/x",
])
def test_untrusted_image_urls_never_fetched(bad):
    with respx.mock(assert_all_called=False) as router:
        router.post(SUBMIT).mock(return_value=httpx.Response(200, json={"status_url": STATUS}))
        router.get(STATUS).mock(return_value=httpx.Response(200, json={"status": "completed", "images": [{"url": bad}]}))
        with pytest.raises(ImageError):
            client().generate("prompt")
        assert len(router.calls) == 2


@pytest.mark.parametrize("image_url", [IMAGE, "https://v3b.fal.media/files/image.png"])
def test_real_queue_protocol_and_unauthenticated_download(image_url):
    with respx.mock(assert_all_called=False) as router:
        router.post(SUBMIT).mock(return_value=httpx.Response(200, json={"status_url": STATUS, "response_url": RESULT}))
        router.get(STATUS).mock(return_value=httpx.Response(200, json={"status": "COMPLETED"}))
        router.get(RESULT).mock(return_value=httpx.Response(200, json={"images": [{"url": image_url}]}))
        download = router.get(image_url).mock(return_value=httpx.Response(200, content=b"image"))
        assert client().generate("prompt") == b"image"
        assert all(c.request.headers.get("authorization") == "Key secret" for c in router.calls[:3])
        assert "authorization" not in download.calls[0].request.headers


@pytest.mark.parametrize("redirect_leg", ["submit", "status", "result", "image"])
def test_redirects_never_followed(redirect_leg):
    with respx.mock(assert_all_called=False) as router:
        redirect = httpx.Response(302, headers={"Location": "https://evil.test/steal"})
        router.post(SUBMIT).mock(return_value=redirect if redirect_leg == "submit" else httpx.Response(200, json={"status_url": STATUS, "response_url": RESULT}))
        router.get(STATUS).mock(return_value=redirect if redirect_leg == "status" else httpx.Response(200, json={"status": "COMPLETED"}))
        router.get(RESULT).mock(return_value=redirect if redirect_leg == "result" else httpx.Response(200, json={"images": [{"url": IMAGE}]}))
        router.get(IMAGE).mock(return_value=redirect if redirect_leg == "image" else httpx.Response(200, content=b"ok"))
        with pytest.raises(ImageError):
            client().generate("prompt")
        assert all(c.request.url.host != "evil.test" for c in router.calls)


def test_oversized_download_rejected():
    with respx.mock(assert_all_called=False) as router:
        router.post(SUBMIT).mock(return_value=httpx.Response(200, json={"status_url": STATUS}))
        router.get(STATUS).mock(return_value=httpx.Response(200, json={"status": "completed", "images": [{"url": IMAGE}]}))
        router.get(IMAGE).mock(return_value=httpx.Response(200, content=b"x" * (20 * 1024 * 1024 + 1)))
        with pytest.raises(ImageError, match="download limit"):
            client().generate("prompt")
