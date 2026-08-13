"""Fact-check pass: turn a pool candidate into a validated research brief.

The funniest historical anecdotes are disproportionately the ones that are
exaggerated or outright false. Before a human writes a joke about an event,
this module verifies it against an encyclopedia extract and produces a
structured brief that explicitly separates the documented record from the
popular version of the story.

All LLM access goes through the ``LLMClient`` protocol (see ``humorhist.llm``)
so tests inject ``StubClient`` and never touch the network.
"""

from __future__ import annotations

import urllib.parse
from typing import Any

import httpx

from humorhist.llm import LLMClient, LLMError

WIKI_SUMMARY_API = "https://en.wikipedia.org/api/rest_v1/page/summary/"
USER_AGENT = "humorhist/0.1 (personal history project; contact: stevie@local)"

#: Allowed values for ``dates.precision``.
PRECISIONS = ("day", "month", "year", "approx", "unknown")

#: Every top-level key a brief must carry.
REQUIRED_KEYS = (
    "verified_facts",
    "dates",
    "key_figures",
    "caveats",
    "misconceptions",
    "sources",
)

_LIST_KEYS = (
    "verified_facts",
    "key_figures",
    "caveats",
    "misconceptions",
)

# Maximum complete_json attempts (initial attempt + one corrective retry).
_MAX_ATTEMPTS = 2


class FactCheckError(RuntimeError):
    """Raised when fetching an extract or producing a valid brief fails."""


FACTCHECK_SYSTEM_PROMPT = """\
You are a rigorous fact-checker for a historical-humour project. A human \
comedian will write a joke from your brief, so publishing a debunked myth \
would be reputationally fatal. Your job is to separate the documented record \
from the story people like to tell.

Verify the event against the supplied encyclopedia extract and widely-known, \
well-established history. Never invent facts that are not supported by the \
extract or by well-established history. If you do not know something, say so \
in "caveats" rather than guessing.

Field rules:
- "verified_facts": 3-8 SPECIFIC, checkable statements. Names, dates, places, \
numbers, outcomes. No vague generalities such as "it was controversial".
- "dates": {"event": "<date or range as documented>", "precision": one of \
"day", "month", "year", "approx", "unknown"}.
- "key_figures": the people who actually matter to the event. May be empty.
- "misconceptions": CRITICAL. Where the popular, commonly-repeated version of \
this story differs from the documented record, state the popular claim and \
then state what the record actually supports. This is what stops the joke \
from repeating a myth.
- "caveats": anything disputed, uncertain, self-reported, or where sources \
conflict. If the single most entertaining element of the story is the part \
that is NOT well documented, say so EXPLICITLY here. This is the most \
important field in the whole brief.
- "sources": at least one {"title": ..., "url": ...} entry.

Output STRICT JSON only, matching exactly this schema, with no commentary:
{
  "verified_facts": ["..."],
  "dates": {"event": "...", "precision": "day|month|year|approx|unknown"},
  "key_figures": ["..."],
  "caveats": ["..."],
  "misconceptions": ["..."],
  "sources": [{"title": "...", "url": "..."}]
}
"""


def _title_from(title_or_url: str) -> str:
    """Return an API-ready page title from a bare title or a wiki URL."""
    value = (title_or_url or "").strip()
    if not value:
        raise FactCheckError("empty Wikipedia title or URL")

    if value.startswith("http://") or value.startswith("https://"):
        parsed = urllib.parse.urlparse(value)
        path = parsed.path
        marker = "/wiki/"
        value = (
            path.split(marker, 1)[1]
            if marker in path
            else path.rsplit("/", 1)[-1]
        )
        value = urllib.parse.unquote(value)
        if not value:
            raise FactCheckError(f"could not extract page title from URL: {title_or_url!r}")

    return value.strip().replace(" ", "_")


def fetch_wikipedia_extract(title_or_url: str, client: httpx.Client | None = None) -> str:
    """Fetch the plain-text intro extract for an English Wikipedia article.

    Accepts a bare page title ("Emu War") or a full en.wikipedia.org URL.
    Raises ``FactCheckError`` on any transport error, non-200 status, or an
    empty/missing extract.
    """
    title = _title_from(title_or_url)
    url = WIKI_SUMMARY_API + urllib.parse.quote(title, safe="")
    headers = {"User-Agent": USER_AGENT, "Accept": "application/json"}

    owns_client = client is None
    http = client or httpx.Client(timeout=30.0)
    try:
        resp = http.get(url, headers=headers, follow_redirects=True)
    except httpx.HTTPError as exc:
        raise FactCheckError(f"failed to fetch Wikipedia extract for {title!r}: {exc}") from exc
    finally:
        if owns_client:
            http.close()

    if resp.status_code != 200:
        raise FactCheckError(f"Wikipedia returned HTTP {resp.status_code} for {title!r}")

    try:
        body = resp.json()
    except ValueError as exc:
        raise FactCheckError(f"non-JSON response from Wikipedia for {title!r}") from exc

    extract = (body or {}).get("extract") if isinstance(body, dict) else None
    if not isinstance(extract, str) or not extract.strip():
        raise FactCheckError(f"no extract available for {title!r}")

    return extract.strip()


def build_factcheck_prompt(item: dict, extract: str) -> str:
    """Render the user prompt for one candidate event."""
    title = str(item.get("title") or "").strip()
    year = item.get("year")
    summary = (item.get("summary") or "").strip()
    url = (item.get("url") or "").strip()

    lines = [f"EVENT TITLE: {title}"]
    if year is not None:
        lines.append(f"CLAIMED YEAR: {year}")
    if summary:
        lines.append(f"POOL SUMMARY: {summary}")
    if url:
        lines.append(f"SOURCE URL: {url}")
    lines.append("")
    lines.append("ENCYCLOPEDIA EXTRACT:")
    lines.append(extract.strip())
    lines.append("")
    lines.append("Produce the research brief as STRICT JSON matching the required schema.")
    return "\n".join(lines)


def _as_list_of_str(value: Any, key: str, *, required: bool = False) -> list[str]:
    """Coerce ``value`` to a list of non-empty strings."""
    if isinstance(value, str):
        value = [value] if value.strip() else []
    if not isinstance(value, list):
        raise FactCheckError(f"{key} must be a list, got {type(value).__name__}")

    out: list[str] = []
    for entry in value:
        if not isinstance(entry, str) or not entry.strip():
            raise FactCheckError(f"{key} must contain only non-empty strings")
        out.append(entry.strip())

    if required and not out:
        raise FactCheckError(f"{key} must be a non-empty list of statements")
    return out


def validate_brief(brief: Any) -> dict:
    """Validate and normalise a research brief.

    Raises ``FactCheckError`` naming the offending field on any schema
    violation. Returns the normalised brief.
    """
    if not isinstance(brief, dict):
        raise FactCheckError(f"brief must be a JSON object, got {type(brief).__name__}")

    for key in REQUIRED_KEYS:
        if key not in brief:
            raise FactCheckError(f"missing required key: {key}")

    out: dict[str, Any] = {}
    out["verified_facts"] = _as_list_of_str(brief["verified_facts"], "verified_facts", required=True)

    dates = brief["dates"]
    if not isinstance(dates, dict):
        raise FactCheckError(f"dates must be an object, got {type(dates).__name__}")
    if "event" not in dates:
        raise FactCheckError("dates is missing 'event'")
    if "precision" not in dates:
        raise FactCheckError("dates is missing 'precision'")
    event = dates["event"]
    if not isinstance(event, str) or not event.strip():
        raise FactCheckError("dates.event must be a non-empty string")
    precision = dates["precision"]
    if precision not in PRECISIONS:
        raise FactCheckError(f"dates.precision must be one of {', '.join(PRECISIONS)}; got {precision!r}")
    out["dates"] = {"event": event.strip(), "precision": precision}

    for key in ("key_figures", "caveats", "misconceptions"):
        out[key] = _as_list_of_str(brief[key], key)

    sources = brief["sources"]
    if not isinstance(sources, list):
        raise FactCheckError(f"sources must be a list, got {type(sources).__name__}")
    if not sources:
        raise FactCheckError("sources must have at least one entry")
    clean_sources: list[dict[str, str]] = []
    for entry in sources:
        if not isinstance(entry, dict):
            raise FactCheckError("each sources entry must be an object")
        title = entry.get("title")
        url = entry.get("url")
        if not isinstance(title, str) or not title.strip():
            raise FactCheckError("each sources entry needs a non-empty title")
        if not isinstance(url, str) or not url.strip():
            raise FactCheckError("each sources entry needs a non-empty url")
        clean_sources.append({"title": title.strip(), "url": url.strip()})
    out["sources"] = clean_sources

    return out


def factcheck(client: LLMClient, item: dict, extract: str) -> dict:
    """Fact-check one event with a single LLM call, retrying once on invalid output.

    If the first response fails validation, the call is repeated once with the
    specific validation error appended to the user prompt so the model can
    correct itself. If the retry also fails, ``FactCheckError`` is raised.
    """
    base_prompt = build_factcheck_prompt(item, extract)
    prompt = base_prompt
    last_error: str = "unknown error"

    for attempt in range(_MAX_ATTEMPTS):
        try:
            raw = client.complete_json(FACTCHECK_SYSTEM_PROMPT, prompt)
        except LLMError as exc:
            last_error = str(exc)
        else:
            try:
                return validate_brief(raw)
            except FactCheckError as exc:
                last_error = str(exc)

        if attempt < _MAX_ATTEMPTS - 1:
            prompt = (
                f"{base_prompt}\n\n"
                "IMPORTANT: your previous response was rejected by schema "
                f"validation with this error:\n{last_error}\n"
                "Fix exactly that problem and return the corrected STRICT JSON "
                "brief, with no commentary."
            )

    raise FactCheckError(f"fact-check failed after {_MAX_ATTEMPTS} attempts: {last_error}")
