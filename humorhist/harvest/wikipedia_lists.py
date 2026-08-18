"""Wikipedia "list of unusual things" harvester for the humorhist pool.

This harvester pulls wikitext from Wikipedia "list" pages via the public
MediaWiki API (action=parse) and extracts candidate historical events that are
likely to make good humorous-history material (odd deaths, hoaxes, wars of
succession, practical jokes, Ig Nobel winners, ...).

Only stdlib ``re`` and ``httpx`` are used -- no BeautifulSoup / mwparserfromhell.
All parsing is line-oriented regex over wiki list markup.

Harvesting is fully idempotent: pool ids are a stable sha1 of
``("wikipedia:<page>", title)`` and rows are upserted with INSERT OR IGNORE.
"""

from __future__ import annotations

import logging
import re
import time

import httpx

import humorhist.db as db

logger = logging.getLogger(__name__)

# Wikipedia "list of" pages to harvest by default.
DEFAULT_PAGES: list[str] = [
    "List_of_wars_of_succession",
    "Lists_of_unusual_deaths",
    "List_of_Ig_Nobel_Prize_winners",
    "List_of_hoaxes",
    "List_of_April_Fools'_Day_jokes",
]

API_URL = "https://en.wikipedia.org/w/api.php"

# Wikimedia rate-limits default/absent User-Agents aggressively; always set one.
USER_AGENT = "humorhist/0.1 (personal history project; contact: stevie@local)"

# Namespaces whose links we drop entirely (they are not candidate titles).
_LINK_NAMESPACES = {"file", "image", "category", "media", "filepath", "commons"}

# A polite delay between page fetches (seconds). Injectable / skippable.
DEFAULT_SLEEP_SECONDS = 0.5

# How many #REDIRECT hops fetch_page_wikitext will follow before giving up.
MAX_REDIRECT_HOPS = 2

_REDIRECT_RE = re.compile(r"^\s*#REDIRECT\s*\[\[\s*([^\[\]|#]+)", re.IGNORECASE)


def _extract_redirect_target(wikitext: str) -> str | None:
    """Return the target page of a ``#REDIRECT [[Target]]`` page, else None."""
    m = _REDIRECT_RE.match(wikitext or "")
    if not m:
        return None
    target = m.group(1).strip()
    return target or None


class HarvestError(Exception):
    """Raised when a Wikipedia API fetch fails or a page is missing/malformed."""


# --------------------------------------------------------------------------- #
# HTTP layer
# --------------------------------------------------------------------------- #


def fetch_page_wikitext(title: str, client: httpx.Client | None = None) -> str:
    """Return the wikitext for a Wikipedia page via the MediaWiki parse API.

    Calls ``https://en.wikipedia.org/w/api.php`` with
    ``action=parse``, ``prop=wikitext``, ``format=json``, ``formatversion=2``.
    A single level of ``#REDIRECT`` is followed automatically (``action=parse``
    does not follow redirects on its own).

    Raises
    ------
    HarvestError
        If the API returns an error object, the page is missing, the response
        is non-JSON, or no ``wikitext`` field is present.
    """
    params = {
        "action": "parse",
        "page": title,
        "prop": "wikitext",
        "format": "json",
        "formatversion": 2,
    }
    headers = {"User-Agent": USER_AGENT}

    own_client = client is None
    if own_client:
        client = httpx.Client(headers=headers, timeout=30.0)
    try:
        wikitext = _fetch_raw(title, client, params, headers)
        # Follow up to MAX_REDIRECT_HOPS levels of #REDIRECT.
        hops = 0
        while True:
            target = _extract_redirect_target(wikitext)
            if target is None:
                break
            hops += 1
            if hops > MAX_REDIRECT_HOPS:
                raise HarvestError(
                    f"too many redirects following Wikipedia page {title!r} (possible redirect loop at {target!r})"
                )
            wikitext = _fetch_raw(target, client, params, headers)
        return wikitext
    finally:
        if own_client:
            client.close()


def _fetch_raw(
    title: str,
    client: httpx.Client,
    params: dict,
    headers: dict,
) -> str:
    """Perform one parse API call and return the wikitext string (or raise)."""
    req_params = dict(params)
    req_params["page"] = title
    resp = client.get(API_URL, params=req_params, headers=headers)
    try:
        data = resp.json()
    except Exception as exc:  # noqa: BLE001 - surface as HarvestError
        raise HarvestError(f"non-JSON response from Wikipedia API for {title!r}: {exc}") from exc

    if isinstance(data, dict) and "error" in data:
        code = data["error"].get("code", "error")
        info = data["error"].get("info", "unknown error")
        raise HarvestError(f"Wikipedia API error for {title!r}: {code}: {info}")

    if (
        not isinstance(data, dict)
        or "parse" not in data
        or not isinstance(data["parse"], dict)
        or "wikitext" not in data["parse"]
    ):
        raise HarvestError(f"Wikipedia page {title!r} returned no wikitext (missing or malformed)")

    return data["parse"]["wikitext"]


# --------------------------------------------------------------------------- #
# Markup cleaning
# --------------------------------------------------------------------------- #


def _strip_noise(text: str) -> str:
    """Remove comments, refs, HTML tags, templates and bold/italic markers.

    Wiki links are deliberately left intact so title derivation can still find
    the first link. Template removal is innermost-first and repeated until the
    text stops changing, so nested ``{{a|{{b}}|c}}`` and multi-pipe templates
    are fully removed (including unterminated/truncated residue).
    """
    text = text or ""
    # HTML comments <!-- ... --> (also tolerate an unterminated one)
    text = re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)
    text = re.sub(r"<!--.*$", "", text, flags=re.DOTALL)
    # <ref>...</ref> and <ref .../> (self-closing and paired)
    text = re.sub(r"<ref\b[^>]*>.*?</ref>", "", text, flags=re.DOTALL)
    text = re.sub(r"<ref\b[^>]*?/?>", "", text)
    # Other stray HTML tags (e.g. <br>, <nowiki>) -- drop the tag, keep content
    text = re.sub(r"</?[a-zA-Z][^>]*?>", "", text)
    # Templates {{...}}: innermost-first until stable.
    while True:
        new = re.sub(r"\{\{[^{}]*\}\}", "", text)
        if new == text:
            break
        text = new
    # Unbalanced / truncated template residue.
    text = re.sub(r"\{\{.*?(?:\}\}|$)", "", text, flags=re.DOTALL)
    text = text.replace("{{", "").replace("}}", "")
    # Italic/bold markers
    text = text.replace("'''", "").replace("''", "")
    return text


def _clean_markup(text: str) -> str:
    """Strip common wiki/markup noise from a single line of wikitext."""
    text = _strip_noise(text)

    # Wiki links: [[Target|Display]] -> Display, [[Target]] -> Target.
    # Drop file/image/category style links entirely.
    def _link(m: re.Match) -> str:
        inner = m.group(1)
        target = inner.split("|", 1)[0].strip()
        ns = target.split(":", 1)[0].strip().lower()
        if ns in _LINK_NAMESPACES:
            return ""
        display = inner.split("|", 1)[1].strip() if "|" in inner else target
        return display

    text = re.sub(r"\[\[([^\[\]]+)\]\]", _link, text)
    # External links [url display] -> display (drop the URL)
    text = re.sub(
        r"\[https?://[^\s\[\]]+(?:\s+([^\]]+))?\]",
        lambda m: m.group(1) or "",
        text,
    )
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


# --------------------------------------------------------------------------- #
# Field extraction
# --------------------------------------------------------------------------- #


def _extract_first_link(text: str) -> str | None:
    """Return the display text of the first wiki link in ``text``, else None."""
    m = re.search(r"\[\[([^\[\]]+)\]\]", text)
    if not m:
        return None
    inner = m.group(1)
    target = inner.split("|", 1)[0].strip()
    ns = target.split(":", 1)[0].strip().lower()
    if ns in _LINK_NAMESPACES:
        return None
    return inner.split("|", 1)[1].strip() if "|" in inner else target


def _first_link_target(text: str) -> str | None:
    """Return the URL-safe article slug of the first wiki link in ``text``.

    Wikipedia link targets may carry a namespace prefix (``File:``,
    ``Category:``, ...) or an anchor (``Article#section``); this strips the
    namespace, keeps the article title, and converts spaces/entities to the
    underscore form used in article URLs. Returns ``None`` when there is no
    (non-namespace) link, so callers can fall back to the parent list page.
    """
    m = re.search(r"\[\[([^\[\]]+)\]\]", text)
    if not m:
        return None
    inner = m.group(1)
    target = inner.split("|", 1)[0].strip()
    # Drop namespace prefixes (File:, Category:, ...) — they are not articles.
    if target.split(":", 1)[0].strip().lower() in _LINK_NAMESPACES:
        return None
    # Strip any section anchor; article slugs never include '#'.
    target = target.split("#", 1)[0].strip()
    if not target:
        return None
    slug = target.replace(" ", "_")
    return slug


def _extract_year(text: str) -> int | None:
    """Extract a year from the first 60 chars of cleaned text.

    * 3-4 digit years are taken directly (e.g. ``1932``, ``1325``).
    * A 2-digit year is only accepted when an AD/BC(E) era marker is present
      (e.g. ``AD 79``, ``79 AD``), to avoid false positives.
    * A BC/BCE marker negates the year (e.g. ``500 BC`` -> ``-500``).
    * Returns ``None`` when no plausible year is found.
    """
    head = text[:60]
    is_bc = bool(re.search(r"\b(?:B\.?C\.?(?:E)?|BCE)\b", head, flags=re.IGNORECASE))
    is_ad = bool(re.search(r"\bAD\b", head, flags=re.IGNORECASE))

    # Prefer a 3-4 digit year anywhere in the first 60 chars.
    m = re.search(r"\b(\d{3,4})\b", head)
    if m:
        year = int(m.group(1))
        return -year if is_bc else year

    # Fall back to a 1-2 digit year only when an era marker gives it meaning.
    if is_ad or is_bc:
        m2 = re.search(r"\b(\d{1,2})\b", head)
        if m2:
            year = int(m2.group(1))
            return -year if is_bc else year

    return None


def _derive_title(content: str) -> str:
    """Derive a title from a raw list-item line.

    Uses the first wiki link's display text if present; otherwise the first
    clause up to the first comma/dash/full-stop/colon, truncated to 120 chars.
    """
    # Strip all non-link markup FIRST so clause splitting can never land
    # inside a template or an HTML comment.
    content = _strip_noise(content)
    link = _extract_first_link(content)
    if link:
        title = _clean_markup(link)
    else:
        first_clause = re.split(r"[,.\-–—:]", content, maxsplit=1)[0]
        title = _clean_markup(first_clause)
    return title.strip()[:120]


# --------------------------------------------------------------------------- #
# Parsing
# --------------------------------------------------------------------------- #


_HEADING_RE = re.compile(r"^\s*(={2,6})\s*(.*?)\s*\1\s*$")


def _heading_year(line: str) -> int | None:
    """Return the year for a ``== 1991 ==`` style heading, else None.

    Returns None for non-heading lines and for headings whose text is not a
    bare 3-4 digit year (e.g. ``== Physics ==``).
    """
    m = _HEADING_RE.match(line)
    if not m:
        return None
    text = _clean_markup(m.group(2))
    return int(text) if re.fullmatch(r"\d{3,4}", text) else None


def parse_list_items(wikitext: str, source_page: str) -> list[dict]:
    """Extract candidate events from a page's wikitext.

    Only ``*`` / ``**`` list lines are considered (deeper nesting is ignored),
    and template/header noise is stripped. Each returned dict has keys:
    ``title``, ``year``, ``summary``, ``source_url``, ``source_name``.

    Section headings that are a bare year (``== 1991 ==``) set a fallback year
    used by subsequent items that carry no year of their own.
    """
    items: list[dict] = []
    section_year: int | None = None
    for line in wikitext.splitlines():
        if not line.startswith("*"):
            if _HEADING_RE.match(line):
                section_year = _heading_year(line)
            continue
        # Count leading list markers; ignore anything deeper than "**".
        stars = len(line) - len(line.lstrip("*"))
        if stars > 2:
            continue

        content = line[stars:].strip()
        cleaned = _clean_markup(content)
        # Drop navigation noise / empty entries.
        if len(cleaned) < 25:
            continue

        title = _derive_title(content)
        if not title:
            continue

        year = _extract_year(cleaned)
        if year is None:
            year = section_year
        summary = cleaned[:500]

        # Prefer the *specific article* behind the event as the "learn more"
        # link. When the list line links a Wikipedia article, the first link's
        # target is the real subject (Wikipedia resolves redirects), which is far
        # more useful than the parent list page for a reader. Link-less entries
        # fall back to the list page, preserving prior behaviour.
        slug = _first_link_target(content)
        source_url = (
            f"https://en.wikipedia.org/wiki/{slug}" if slug else f"https://en.wikipedia.org/wiki/{source_page}"
        )

        items.append(
            {
                "title": title,
                "year": year,
                "summary": summary,
                "source_url": source_url,
                "source_name": f"wikipedia:{source_page}",
            }
        )
    return items


# --------------------------------------------------------------------------- #
# Harvest orchestration
# --------------------------------------------------------------------------- #


def harvest_wikipedia_lists(
    conn,
    pages: list[str] | None = None,
    client: httpx.Client | None = None,
    sleep_seconds: float = DEFAULT_SLEEP_SECONDS,
) -> dict:
    """Harvest candidate events from Wikipedia list pages into the pool.

    Parameters
    ----------
    conn:
        An open, migrated sqlite3 connection.
    pages:
        Page titles to harvest. Defaults to :data:`DEFAULT_PAGES`.
    client:
        An optional :class:`httpx.Client` (e.g. for testing). If ``None``, a
        client with the required User-Agent is created and closed here.
    sleep_seconds:
        Delay between page fetches (politeness). Pass ``0`` to disable (tests).

    Returns
    -------
    dict
        ``pages_fetched``, ``candidates_found``, ``inserted``,
        ``skipped_duplicate``, ``skipped_invalid``, ``pages_failed``.
    """
    pages = list(pages) if pages is not None else list(DEFAULT_PAGES)

    summary: dict[str, int] = {
        "pages_fetched": 0,
        "candidates_found": 0,
        "inserted": 0,
        "skipped_duplicate": 0,
        "skipped_invalid": 0,
        "pages_failed": 0,
    }

    own_client = client is None
    if own_client:
        client = httpx.Client(headers={"User-Agent": USER_AGENT}, timeout=30.0)
    try:
        for idx, page in enumerate(pages):
            try:
                wikitext = fetch_page_wikitext(page, client=client)
            except HarvestError as exc:
                logger.warning("Skipping %s after fetch failure: %s", page, exc)
                summary["pages_failed"] += 1
                continue
            except Exception as exc:  # noqa: BLE001 - one bad page must not abort
                logger.warning("Skipping %s after unexpected error: %s", page, exc)
                summary["pages_failed"] += 1
                continue

            summary["pages_fetched"] += 1
            items = parse_list_items(wikitext, page)
            summary["candidates_found"] += len(items)

            for item in items:
                source_name = item["source_name"]
                title = item["title"]
                item_id = db.make_id(source_name, title)
                inserted = db.upsert_pool_item(
                    conn,
                    id=item_id,
                    title=title,
                    year=item["year"],
                    date_hint=None,
                    summary=item["summary"],
                    source_url=item["source_url"],
                    source_name=source_name,
                )
                if inserted:
                    summary["inserted"] += 1
                else:
                    summary["skipped_duplicate"] += 1

            # Be polite: pause between fetches (skippable for tests).
            if sleep_seconds and idx < len(pages) - 1:
                time.sleep(sleep_seconds)
    finally:
        if own_client:
            client.close()

    return summary
