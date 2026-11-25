"""Phase 3 image generation: a generated picture that represents the story.

On an explicit image command, generate an image representing the story. This module:

  - distills a draft + pool into a concise image prompt via the LLM, and
  - calls the configured image provider to render that prompt, saving the bytes to
    ``data/images/<draft_id>.png``.

Both steps are best-effort. Generation needs credentials and is never run by
approval or enqueue; CLI and Telegram image commands request it explicitly.

Image providers: ``FalClient`` uses ``HUMORHIST_IMAGE_API_KEY`` and the opt-in
``OpenAIImageClient`` uses ``OPENAI_API_KEY``. ``resilient_image_client`` raises
``ImageUnavailable`` when no credential is present, so callers can show a clean
message (or silently skip) instead of surfacing a traceback. A ``StubImageClient``
lets tests run with zero network.

The visual style is configurable via ``HUMORHIST_IMAGE_STYLE`` (default
"editorial-historical"):
  - "editorial-historical": a believable period scene, painterly, lightly comic.
  - "editorial-cartoon": a clean editorial-cartoon / line-art gag.
  - "meme": a bold, high-contrast visual-gag format.

Generation always targets the chosen style; if the model/file step fails the
caller receives ``ImageError`` and can proceed without the image (the post copy
and the rest of the pipeline are unaffected).
"""

from __future__ import annotations

import base64
import binascii
import os
import re
import struct
import time
import zlib
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import urlsplit

import httpx

# Where generated images live, relative to the data directory handled by the
# caller (they pass an absolute output dir). Kept as a single subfolder name so
# it can't escape the data dir.
IMAGE_DIR_NAME = "images"

# Map of supported styles to a short style brief appended to every prompt. The
# default ("editorial-historical") matches the product's "believable but
# funny" history-humor voice.
STYLE_BRIEFS: dict[str, str] = {
    "editorial-historical": (
        "Painterly editorial-historical illustration, period-accurate setting "
        "with a sly, lightly comic tone. Warm muted palette, soft natural light, "
        "no modern objects, no text in the image."
    ),
    "editorial-cartoon": (
        "Clean editorial-cartoon / line-art style, bold ink outlines, 3-4 flat "
        "colours, witty and readable at small size. No photographic detail, no "
        "text in the image unless it is part of a speech bubble."
    ),
    "meme": (
        "Bold high-contrast visual-gag format: simple iconic composition, thick "
        "outline, saturated colours, instantly readable. No text in the image."
    ),
}

DEFAULT_STYLE = "editorial-historical"


def image_style() -> str:
    """Return the active image style name (validated against STYLE_BRIEFS)."""
    name = (os.environ.get("HUMORHIST_IMAGE_STYLE") or DEFAULT_STYLE).strip().lower()
    return name if name in STYLE_BRIEFS else DEFAULT_STYLE


class ImageError(RuntimeError):
    """Raised when image generation fails or returns unusable output."""


class ImageUnavailable(RuntimeError):
    """Raised when no usable image credential is available at call time.

    Distinct from ``ImageError`` (a transient/permanent generation failure) so
    callers — especially the Telegram bot — can skip the image silently instead
    of surfacing a raw traceback to the user's phone.
    """


class ImageGenClient(Protocol):
    """Minimal interface an image provider must satisfy."""

    def generate(self, prompt: str) -> bytes:
        """Return raw image bytes (PNG) for ``prompt``."""
        ...


# --------------------------------------------------------------------------- #
# Prompt distillation (LLM)                                                    #
# --------------------------------------------------------------------------- #

IMAGE_PROMPT_SYSTEM = (
    "You are the art director for a history-humor account. You turn a "
    "fact-checked brief, a chosen comic angle, and the editor's one-line joke "
    "into ONE image-generation prompt that represents the story.\n"
    "Rules:\n"
    "  - Use only the VERIFIED FACTS. Never depict anything listed under "
    "MISCONCEPTIONS.\n"
    "  - If an EDITOR LINE (joke) is given, let it shape the scene's comic "
    "beat without putting words in the image.\n"
    "  - Be concrete and visual: subject, setting, action, composition, mood.\n"
    "  - Do NOT request text/lettering in the image.\n"
    "  - Keep it to ~3-5 sentences, no preamble.\n"
    'Return a JSON object: {"prompt": "<the image prompt>"}.\n'
)


def _build_prompt_user(draft: dict, pool: dict | None) -> str:
    import json

    brief = json.loads(draft.get("brief_json") or "{}")
    angles = json.loads(draft.get("angles_json") or "{}")

    facts = brief.get("verified_facts", [])
    misconceptions = brief.get("misconceptions", [])
    angle_name = (angles.get("angles") or [{}])[0].get("angle_name", "")
    hook = angles.get("suggested_hook", "")
    title = pool["title"] if pool else "(unknown)"
    year = pool["year"] if pool else ""

    editor_line = draft.get("editor_line") or "(none)"

    parts = [
        f"TOPIC: {title} ({year})",
        f"EDITOR LINE (comic steer): {editor_line}",
        f"LEAD ANGLE: {angle_name}",
        f"SUGGESTED HOOK: {hook}",
        "",
        "VERIFIED FACTS:",
    ]
    parts += [f"  - {f}" for f in facts] or ["  - (none provided)"]
    if misconceptions:
        parts.append("")
        parts.append("MISCONCEPTIONS TO AVOID (do NOT depict these):")
        parts += [f"  - {m}" for m in misconceptions]
    return "\n".join(parts)


def generate_image_prompt(
    client: Any,
    draft: dict,
    pool: dict | None = None,
) -> str:
    """Distill a draft + pool into a single image prompt string.

    ``client`` is an ``LLMClient`` (see ``humorhist.llm``). Raises ``ImageError``
    on a bad model response or any transport error.
    """

    user = _build_prompt_user(draft, pool)
    style = image_style()
    system = (
        IMAGE_PROMPT_SYSTEM
        + f"\nStyle: {STYLE_BRIEFS[style]}\n"
        + "Append this style direction to the prompt you write."
    )

    last_err: Exception | None = None
    for _ in range(3):
        try:
            result = client.complete_json(system, user, max_tokens=1024, reasoning_off=True)
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            continue
        if isinstance(result, dict) and result.get("prompt"):
            text = result["prompt"].strip()
        elif isinstance(result, str) and result.strip():
            text = result.strip()
        else:
            last_err = ValueError(f"model returned no image prompt: {result!r}")
            continue
        if len(text) < 12:
            last_err = ValueError(f"degenerate image prompt rejected: {text!r}")
            continue
        return text

    raise ImageError(f"image prompt generation failed: {last_err}")


# --------------------------------------------------------------------------- #
# Image providers                                                             #
# --------------------------------------------------------------------------- #


def _fal_url(url: Any, hosts: set[str]) -> str:
    """Reject untrusted origins before httpx sees a provider-supplied URL."""
    if not isinstance(url, str) or not url or any(ord(c) <= 32 or ord(c) == 127 for c in url):
        raise ImageError("FAL: invalid response URL")
    try:
        parts = urlsplit(url)
        if (parts.scheme != "https" or parts.hostname not in hosts
                or parts.username is not None or parts.password is not None
                or parts.port not in (None, 443) or parts.fragment
                or "\\" in url):
            raise ValueError("untrusted origin")
    except ValueError:
        raise ImageError("FAL: untrusted response URL") from None
    return url


class FalClient:
    """Real image client speaking the FAL FLUX text-to-image API over httpx.

    Uses the FAL REST queue: POST the request, poll ``status_url`` until done,
    then GET the result URL and download the bytes. The API key comes from
    ``HUMORHIST_IMAGE_API_KEY`` (or the constructor).
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        timeout: float = 180.0,
        max_retries: int = 2,
    ) -> None:
        self.api_key = api_key or os.environ.get("HUMORHIST_IMAGE_API_KEY", "")
        self.model = model or os.environ.get("HUMORHIST_IMAGE_MODEL", "fal-ai/flux/dev")
        self.timeout = timeout
        self.max_retries = max_retries

    def generate(self, prompt: str) -> bytes:
        if not self.api_key:
            raise ImageError("no API key: set HUMORHIST_IMAGE_API_KEY or pass api_key explicitly")

        base = "https://queue.fal.run"
        model = self.model.lstrip('/')
        if not re.fullmatch(r"[A-Za-z0-9_-]+(?:/[A-Za-z0-9_.-]+)*", model):
            raise ImageError("FAL: invalid model identifier")
        url = f"{base}/{model}"
        headers = {
            "Authorization": f"Key {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "prompt": prompt,
            "num_images": 1,
            "image_size": "square_hd",
        }

        # A timeout or server error may occur after the job was accepted. Never
        # resubmit a paid generation automatically, even when polling fails.
        try:
            with httpx.Client(timeout=self.timeout, follow_redirects=False, trust_env=False) as client:
                resp = client.post(url, json=payload, headers=headers)
                resp.raise_for_status()
                body = resp.json()
                status_url = body.get("status_url") or body.get("url")
                if not status_url:
                    raise ImageError("FAL: no status_url in response")
                status_url = _fal_url(status_url, {"queue.fal.run"})
                response_url = body.get("response_url")
                if response_url is not None:
                    response_url = _fal_url(response_url, {"queue.fal.run"})
                image_url = self._wait_for_result(client, headers, status_url, response_url)
                # The locally installed fal-client identifies these as its CDN
                # and fallback hosts. Other FAL delivery hosts may exist.
                image_url = _fal_url(image_url, {"v3.fal.media", "v3b.fal.media", "fal.media"})
                image = bytearray()
                with client.stream("GET", image_url, headers={"Accept-Encoding": "identity"},
                                   follow_redirects=False) as img_resp:
                    img_resp.raise_for_status()
                    if img_resp.headers.get("content-encoding", "identity").lower() != "identity":
                        raise ImageError("FAL: compressed image response rejected")
                    for chunk in img_resp.iter_raw(chunk_size=65536):
                        image.extend(chunk)
                        if len(image) > 20 * 1024 * 1024:
                            raise ImageError("FAL: image exceeds download limit")
                if not image:
                    raise ImageError("FAL: empty image")
                return bytes(image)
        except ImageError:
            raise
        except Exception as exc:
            raise ImageError(f"FAL image request failed ({type(exc).__name__}); not retried") from None

    @staticmethod
    def _wait_for_result(client: httpx.Client, headers: dict, status_url: str,
                         response_url: str | None = None) -> str:
        """Poll ``status_url`` until the FAL job is complete; return the image URL."""
        for _ in range(60):  # up to ~5 min at 5s cadence
            resp = client.get(status_url, headers=headers)
            resp.raise_for_status()
            body = resp.json()
            if str(body.get("status", "")).lower() == "completed":
                if response_url:
                    result = client.get(response_url, headers=headers, follow_redirects=False)
                    result.raise_for_status()
                    body = result.json()
                images = body.get("images") or []
                if not images:
                    raise ImageError("FAL: completed but no images")
                first = images[0]
                return first.get("url") or first.get("image_url")
            if str(body.get("status", "")).lower() == "error":
                raise ImageError("FAL job errored")
            time.sleep(5)
        raise ImageError("FAL: timed out waiting for image result")


def _validate_png(data: bytes) -> None:
    """Reject incomplete/corrupt PNGs before they become persistent artifacts."""
    if not data.startswith(b"\x89PNG\r\n\x1a\n"):
        raise ValueError("invalid PNG signature")
    pos, ihdr, image_data, ended = 8, None, bytearray(), False
    palette = False
    idat_ended = False
    while pos + 12 <= len(data):
        length = int.from_bytes(data[pos:pos + 4], "big")
        end = pos + 12 + length
        if end > len(data):
            raise ValueError("truncated PNG chunk")
        kind = data[pos + 4:pos + 8]
        chunk = data[pos + 8:pos + 8 + length]
        checksum = int.from_bytes(data[pos + 8 + length:end], "big")
        if zlib.crc32(kind + chunk) != checksum:
            raise ValueError("invalid PNG checksum")
        if ihdr is None:
            if kind != b"IHDR" or length != 13:
                raise ValueError("missing PNG header")
            ihdr = struct.unpack(">IIBBBBB", chunk)
        elif kind == b"IHDR":
            raise ValueError("duplicate PNG header")
        if kind == b"PLTE":
            if palette or image_data or not length or length > 768 or length % 3:
                raise ValueError("invalid PNG palette")
            palette = True
        if kind == b"IDAT":
            if idat_ended:
                raise ValueError("non-consecutive PNG image data")
            image_data.extend(chunk)
        elif image_data:
            idat_ended = True
        if kind not in (b"IHDR", b"PLTE", b"IDAT", b"IEND") and not (kind[0] & 0x20):
            raise ValueError("unknown critical PNG chunk")
        if kind == b"IEND":
            if length or end != len(data):
                raise ValueError("invalid PNG end")
            ended = True
            break
        pos = end
    if not ended or not image_data or ihdr is None:
        raise ValueError("incomplete PNG")
    width, height, depth, color, compression, filtering, interlace = ihdr
    depths = {0: (1, 2, 4, 8, 16), 2: (8, 16), 3: (1, 2, 4, 8), 4: (8, 16), 6: (8, 16)}
    channels = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}
    if (not width or not height or color not in depths or depth not in depths[color]
            or compression or filtering or interlace not in (0, 1)
            or (color == 3 and not palette) or (color in (0, 4) and palette)):
        raise ValueError("unsupported PNG header")
    passes = [(0, 0, 1, 1)] if not interlace else [
        (0, 0, 8, 8), (4, 0, 8, 8), (0, 4, 4, 8), (2, 0, 4, 4),
        (0, 2, 2, 4), (1, 0, 2, 2), (0, 1, 1, 2),
    ]
    rows = []
    for x, y, dx, dy in passes:
        w = max(0, (width - x + dx - 1) // dx)
        h = max(0, (height - y + dy - 1) // dy)
        if w and h:
            rows.append(((w * depth * channels[color] + 7) // 8 + 1, h))
    expected = sum(size * count for size, count in rows)
    if expected > 64 * 1024 * 1024:
        raise ValueError("PNG too large")
    decoder = zlib.decompressobj()
    raw = decoder.decompress(bytes(image_data), expected + 1)
    if len(raw) != expected or not decoder.eof or decoder.unused_data:
        raise ValueError("invalid PNG image data")
    offset = 0
    for size, count in rows:
        if any(raw[offset + i * size] > 4 for i in range(count)):
            raise ValueError("invalid PNG row filter")
        offset += size * count


class OpenAIImageClient:
    """OpenAI Images API; one request, one image, no automatic retry."""

    def __init__(self, api_key: str, model: str, quality: str, timeout: float = 180.0) -> None:
        if not api_key:
            raise ImageUnavailable("set OPENAI_API_KEY for the OpenAI image provider")
        if model not in {"gpt-image-1.5", "gpt-image-2", "gpt-image-2.5-flare", "gpt-image-2.5-sunburst"}:
            raise ImageUnavailable("unsupported OpenAI image model")
        if quality not in {"low", "medium", "high"}:
            raise ImageUnavailable("OpenAI image quality must be low, medium, or high")
        self.api_key, self.model, self.quality, self.timeout = api_key, model, quality, timeout

    def generate(self, prompt: str) -> bytes:
        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(
                    "https://api.openai.com/v1/images/generations",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    json={"model": self.model, "quality": self.quality, "size": "1024x1024",
                          "output_format": "png", "n": 1, "prompt": prompt},
                )
                response.raise_for_status()
                body = response.json()
            encoded = body["data"][0]["b64_json"]
            if not isinstance(encoded, str):
                raise ValueError("invalid image encoding")
            data = base64.b64decode(encoded, validate=True)
            if not data:
                raise ValueError("empty image")
            _validate_png(data)
            return data
        except httpx.HTTPStatusError as exc:
            raise ImageError(f"OpenAI image request failed (HTTP {exc.response.status_code}); not retried") from exc
        except (KeyError, IndexError, TypeError, ValueError, binascii.Error, zlib.error) as exc:
            raise ImageError("OpenAI returned an invalid or empty image response; not retried") from exc
        except httpx.HTTPError as exc:
            raise ImageError(f"OpenAI image request failed ({type(exc).__name__}); not retried") from exc


class StubImageClient:
    """Deterministic image client for tests.

    Provide ``responses`` as a list; each call to ``generate`` pops the next one.
    A response may be bytes (returned as-is) or an Exception instance (raised).
    """

    def __init__(self, responses: list[Any] | None = None) -> None:
        self.responses = list(responses or [])
        self.prompts: list[str] = []

    def generate(self, prompt: str) -> bytes:
        self.prompts.append(prompt)
        if not self.responses:
            raise ImageError("StubImageClient exhausted: no more canned responses")
        nxt = self.responses.pop(0)
        if isinstance(nxt, Exception):
            raise nxt
        return nxt


def default_image_client() -> ImageGenClient:
    """Return the configured real image client."""
    return resilient_image_client()


def resilient_image_client(
    timeout: float = 180.0,
    max_retries: int = 2,
) -> ImageGenClient:
    """Return an image client that works unattended.

    Selects FAL by default, or OpenAI only when explicitly configured. The
    legacy max_retries argument is accepted but paid submissions are not retried.
    """
    provider = (os.environ.get("HUMORHIST_IMAGE_PROVIDER") or "fal").strip().lower()
    if provider == "openai":
        return OpenAIImageClient(
            api_key=os.environ.get("OPENAI_API_KEY", "").strip(),
            model=os.environ.get("HUMORHIST_IMAGE_MODEL", "gpt-image-2.5-flare"),
            quality=os.environ.get("HUMORHIST_IMAGE_QUALITY", "low"),
            timeout=timeout,
        )
    if provider != "fal":
        raise ImageUnavailable(f"unknown image provider: {provider!r}")
    key = os.environ.get("HUMORHIST_IMAGE_API_KEY")
    if not key:
        raise ImageUnavailable("no image credential available — set HUMORHIST_IMAGE_API_KEY to enable story images")
    return FalClient(api_key=key, timeout=timeout, max_retries=max_retries)


# --------------------------------------------------------------------------- #
# End-to-end: prompt -> image -> save                                         #
# --------------------------------------------------------------------------- #


def _sanitize_filename(name: str) -> str:
    """Keep only safe characters for a filename; collapse the rest to '_'."""
    return re.sub(r"[^A-Za-z0-9._-]", "_", name)[:120]


def generate_image(
    llm_client: Any,
    image_client: ImageGenClient,
    draft: dict,
    pool: dict | None = None,
    *,
    out_dir: str | Path,
    draft_id: str | None = None,
) -> tuple[str, str]:
    """Generate an image for a draft and save it to ``out_dir``.

    Builds the image prompt via ``llm_client``, renders it via ``image_client``,
    and writes the bytes to ``out_dir/<draft_id>.png`` (creating ``out_dir`` if
    needed). Returns ``(image_path, image_prompt)`` on success.

    Raises ``ImageError`` / ``ImageUnavailable`` on failure; callers should treat
    those as best-effort and continue without the image.
    """
    draft_id = draft_id or draft["id"]
    prompt = generate_image_prompt(llm_client, draft, pool)
    data = image_client.generate(prompt)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # draft_id is a sha1 hex (from db.make_id) so it is already filename-safe,
    # but sanitize defensively.
    path = out_dir / f"{_sanitize_filename(draft_id)}.png"
    path.write_bytes(data)
    return str(path), prompt


def generate_image_for_queue(
    conn: Any,
    draft_id: str,
    *,
    out_dir: str | Path,
    llm_client: Any | None = None,
) -> tuple[str, str] | None:
    """Best-effort story image for a queued draft, persisted on the queue row.

    Distills the prompt from the draft + pool via the LLM, renders it via the
    configured provider, saves the PNG to ``out_dir/<draft_id>.png``, and writes
    ``image_prompt`` / ``image_path`` onto the queue row via ``db.set_image``.

    Returns ``(image_path, image_prompt)`` on success, or ``None`` if image
    generation is unavailable or fails for any reason (missing credential,
    exhausted balance, transport error, bad model response). Callers use this at
    the *publish/enqueue* step so a missing image never blocks the pipeline.

    ``llm_client`` is looked up lazily via ``humorhist.llm.default_client`` when
    not supplied (so it is only constructed if an image client is actually
    available).
    """
    import humorhist.db as db

    img_client = None
    try:
        from humorhist.imagegen import resilient_image_client

        img_client = resilient_image_client()
    except ImageUnavailable as exc:
        print(f"[image] unavailable: {exc}")
        img_client = None
    except Exception as exc:  # noqa: BLE001 - setup must not block publishing
        print(f"[image] setup failed: {type(exc).__name__}")
        img_client = None
    if img_client is None:
        return None

    try:
        if llm_client is None:
            from humorhist.llm import default_client

            llm_client = default_client()
        draft_row = dict(conn.execute("SELECT * FROM drafts WHERE id = ?", (draft_id,)).fetchone())
        pool_row = db.get_pool_item(conn, draft_row["pool_id"])
        path, prompt = generate_image(
            llm_client,
            img_client,
            draft_row,
            dict(pool_row) if pool_row else None,
            out_dir=out_dir,
            draft_id=draft_id,
        )
        db.set_image(conn, draft_id, image_prompt=prompt, image_path=path)
        return path, prompt
    except Exception as exc:  # noqa: BLE001 - best-effort: never block publish
        print(f"[image] not generated for {draft_id}: {exc}")
        return None
