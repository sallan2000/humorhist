"""Phase 3 image generation: a generated picture that represents the story.

On approve, alongside the editable post copy (B+), we generate an image that
represents the历史事件 (the story). This module:

  - distills a draft + pool into a concise image prompt via the LLM, and
  - calls an image model (FAL FLUX) to render that prompt, saving the bytes to
    ``data/images/<draft_id>.png``.

Both steps are best-effort. Generation needs credentials, so it is deliberately
kept OUT of ``review.apply_review`` (which is pure and transport-agnostic). The
approve paths (CLI review loop and the Telegram review bot) call
``generate_image`` right after ``apply_review`` returns — the same pattern as
``copywriter.fill_post_copy``.

Image providers: a ``FalClient`` speaks the FAL FLUX API over httpx. The key
comes from ``HUMORHIST_IMAGE_API_KEY``. ``resilient_image_client`` raises
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

import os
import re
import time
from pathlib import Path
from typing import Any, Protocol

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
    "Return a JSON object: {\"prompt\": \"<the image prompt>\"}.\n"
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
    from humorhist.llm import extract_json

    user = _build_prompt_user(draft, pool)
    style = image_style()
    system = (
        IMAGE_PROMPT_SYSTEM
        + f"\nStyle: {STYLE_BRIEFS[style]}\n"
        + "Append this style direction to the prompt you write."
    )

    last_err: Exception | None = None
    for attempt in range(3):
        try:
            result = client.complete_json(
                system, user, max_tokens=1024, reasoning_off=True
            )
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

    from humorhist.llm import LLMError

    raise ImageError(f"image prompt generation failed: {last_err}")


# --------------------------------------------------------------------------- #
# Image providers                                                             #
# --------------------------------------------------------------------------- #


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
        self.model = model or os.environ.get(
            "HUMORHIST_IMAGE_MODEL", "fal-ai/flux/dev"
        )
        self.timeout = timeout
        self.max_retries = max_retries

    def generate(self, prompt: str) -> bytes:
        if not self.api_key:
            raise ImageError(
                "no API key: set HUMORHIST_IMAGE_API_KEY or pass api_key explicitly"
            )

        base = "https://queue.fal.run"
        url = f"{base}/{self.model.lstrip('/')}"
        headers = {
            "Authorization": f"Key {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "prompt": prompt,
            "num_images": 1,
            "image_size": "square_hd",
        }

        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                with httpx.Client(timeout=self.timeout) as client:
                    # Submit the generation request.
                    resp = client.post(url, json=payload, headers=headers)
                    resp.raise_for_status()
                    body = resp.json()
                    status_url = body.get("status_url") or body.get("url")
                    if not status_url:
                        raise ImageError(f"FAL: no status_url in response: {body}")
                    # Poll until the result is ready.
                    image_url = self._wait_for_result(client, headers, status_url)
                    # Download the actual image bytes.
                    img_resp = client.get(image_url)
                    img_resp.raise_for_status()
                    return img_resp.content
            except Exception as exc:  # noqa: BLE001 - retry transient failures
                last_error = exc
                if attempt < self.max_retries:
                    time.sleep(2 ** attempt)

        raise ImageError(f"image generation failed after retries: {last_error}")

    @staticmethod
    def _wait_for_result(
        client: httpx.Client, headers: dict, status_url: str
    ) -> str:
        """Poll ``status_url`` until the FAL job is complete; return the image URL."""
        for _ in range(60):  # up to ~5 min at 5s cadence
            resp = client.get(status_url, headers=headers)
            resp.raise_for_status()
            body = resp.json()
            if body.get("status") == "completed":
                images = body.get("images") or []
                if not images:
                    raise ImageError(f"FAL: completed but no images: {body}")
                first = images[0]
                return first.get("url") or first.get("image_url")
            if body.get("status") == "error":
                raise ImageError(f"FAL job errored: {body}")
            time.sleep(5)
        raise ImageError("FAL: timed out waiting for image result")


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


def default_image_client() -> FalClient:
    """Return the configured real image client."""
    return FalClient()


def resilient_image_client(
    timeout: float = 180.0,
    max_retries: int = 2,
) -> FalClient:
    """Return an image client that works unattended.

    Reads ``HUMORHIST_IMAGE_API_KEY``. Raises ``ImageUnavailable`` if absent, so
    callers can skip image generation cleanly (post copy + pipeline unaffected).
    """
    key = os.environ.get("HUMORHIST_IMAGE_API_KEY")
    if not key:
        raise ImageUnavailable(
            "no image credential available — set HUMORHIST_IMAGE_API_KEY to enable "
            "story images"
        )
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
