"""Tests for humorhist.harvest.wikipedia_lists -- the Wikipedia list harvester.

TDD: these are written before the implementation exists, so the first run
should error out on import. Real httpx calls are NEVER made -- every network
request is intercepted by respx. The final, real acceptance check against the
live API lives outside pytest (it is run manually after the suite passes).
"""

from __future__ import annotations

from pathlib import Path

import httpx
import respx

import humorhist.db as db
from humorhist.harvest import wikipedia_lists
from humorhist.harvest.wikipedia_lists import (
    DEFAULT_PAGES,
    HarvestError,
    fetch_page_wikitext,
    harvest_wikipedia_lists,
    parse_list_items,
)

API_URL = "https://en.wikipedia.org/w/api.php"


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #

# A realistic multi-line wikitext sample with varied markup (10+ list items).
WIKITEXT_SAMPLE = """== Wars and odd deaths ==
* [[Emu War]] (1932) fought in [[Australia]]; soldiers lost to birds, a ridiculous campaign that ended in defeat.
* '''Operation Anthropoid''' — the 1942 assassination of [[Reinhard Heydrich]] in [[Prague]] by resistance fighters.
* AD 79: [[Pliny the Elder]] died observing the [[Eruption of Mount Vesuvius in 79 AD|Vesuvius eruption]] near [[Pompeii]].
* [[Tenochtitlan]] ([[1325]]) was founded by the [[Mexica]] people in the [[Valley of Mexico]] on an island.
* 500 BC: [[Xerxes I]] ruled the [[Persian Empire]] and invaded [[Greece]] with a huge army of soldiers.
* [[Ig Nobel Prize]]: in 1991, [[David Schmidt]] studied the [[physics]] of [[wet T-shirts]] in a humorous study.
* [[Hoax]]: the [[Piltdown Man]], found in 1912, fooled scientists for decades before exposure as a fake.
* [[The Cadaver Synod]]: in 897, [[Pope Stephen VI]] put a dead pope on trial in [[Rome]] for political reasons.
* The [[Dancing Plague of 1518]] saw hundreds dance uncontrollably in [[Strasbourg]] for no clear reason at all.
* [[Caligula]]: the Roman emperor once appointed his horse [[Incitatus]] to the senate as a joke appointment.
* {{some template}} This line is mostly template noise but has enough words to pass the length check and mentions [[Rome]].
* A perfectly valid event with no link but a long enough description to be kept as a candidate item for the pool.
* xy short
"""

# A simpler wikitext sample used by the harvest integration tests. Two pages,
# each with four real items (one trailing short-noise line per page).
PAGE_ONE = """* [[Emu War]] (1932) fought in [[Australia]]; soldiers lost to birds, a ridiculous campaign that ended in defeat.
* '''Operation Anthropoid''' — the 1942 assassination of [[Reinhard Heydrich]] in [[Prague]] by resistance fighters.
* AD 79: [[Pliny the Elder]] died observing the [[Vesuvius]] eruption near [[Pompeii]] according to historical records.
* [[Tenochtitlan]] ([[1325]]) was founded by the [[Mexica]] people in the [[Valley of Mexico]] on an island.
"""

PAGE_TWO = """* 500 BC: [[Xerxes I]] ruled the [[Persian Empire]] and invaded [[Greece]] with a massive army of soldiers.
* [[Ig Nobel Prize]]: in 1991, [[David Schmidt]] studied the [[physics]] of [[wet T-shirts]] in a humorous study.
* [[Hoax]]: the [[Piltdown Man]], found in 1912, fooled scientists for decades before exposure as a fake.
* [[The Cadaver Synod]]: in 897, [[Pope Stephen VI]] put a dead pope on trial in [[Rome]] for political reasons.
* short noise
"""


def _fresh_db(tmp_path: Path):
    path = tmp_path / "test.sqlite"
    conn = db.connect(str(path))
    db.migrate(conn)
    return conn


# Sentinel value in the pages_text dict marking a page that should fail to fetch.
_ERROR_MARKER = "__ERROR__"


def _page_router(pages_text: dict[str, str]):
    """Return a respx side_effect that serves wikitext keyed by the ?page= param.

    A page mapped to ``_ERROR_MARKER`` is reported as a missing/error API
    response so the harvester's per-page failure handling can be exercised.
    """

    def _cb(request: httpx.Request) -> httpx.Response:
        page = request.url.params.get("page")
        if page in pages_text:
            if pages_text[page] == _ERROR_MARKER:
                return httpx.Response(
                    200,
                    json={"error": {"code": "missingtitle", "info": f"Page {page!r} does not exist"}},
                )
            return httpx.Response(
                200,
                json={"parse": {"title": page, "wikitext": pages_text[page]}},
            )
        # Anything unexpected is reported missing.
        return httpx.Response(
            200,
            json={"error": {"code": "missingtitle", "info": f"Page {page!r} does not exist"}},
        )

    return _cb


# --------------------------------------------------------------------------- #
# fetch_page_wikitext
# --------------------------------------------------------------------------- #


def test_fetch_page_wikitext_returns_text(respx_mock):
    respx_mock.get(API_URL).mock(
        return_value=httpx.Response(200, json={"parse": {"title": "Foo", "wikitext": "HELLO WIKITEXT"}})
    )
    text = fetch_page_wikitext("Foo")
    assert text == "HELLO WIKITEXT"


def test_fetch_page_wikitext_raises_on_missing_page(respx_mock):
    respx_mock.get(API_URL).mock(
        return_value=httpx.Response(
            200,
            json={"error": {"code": "missingtitle", "info": "The page does not exist"}},
        )
    )
    try:
        fetch_page_wikitext("NoSuchPage")
        assert False, "expected HarvestError"
    except HarvestError:
        pass


def test_fetch_page_wikitext_sets_user_agent(respx_mock):
    route = respx_mock.get(API_URL).mock(
        return_value=httpx.Response(200, json={"parse": {"title": "Foo", "wikitext": "X"}})
    )
    fetch_page_wikitext("Foo")
    assert route.called
    request = route.calls.last.request
    assert request.headers["User-Agent"].startswith("humorhist/")


# --------------------------------------------------------------------------- #
# parse_list_items -- markup stripping & field derivation
# --------------------------------------------------------------------------- #


def test_parse_strips_wiki_markup():
    wt = (
        "* [[Link|Display]] was a '''bold''' event <ref>junk</ref> "
        "{{template}} that happened in the year 1500 with enough words to be "
        "kept as a valid candidate item in the pool table for testing."
    )
    items = parse_list_items(wt, "List_of_foo")
    assert len(items) == 1
    summary = items[0]["summary"]
    assert "[[" not in summary
    assert "]]" not in summary
    assert "''' " not in summary and "''" not in summary
    assert "<ref" not in summary
    assert "{{" not in summary
    # Display text is preserved from the link.
    assert "Display" in summary
    # The year 1500 should have been captured.
    assert items[0]["year"] == 1500


def test_parse_extracts_year():
    wt = (
        "* 1932 — The Emu War happened in Australia and was rather silly.\n"
        "* AD 79: Pliny died near Vesuvius according to the historical record.\n"
        "* 500 BC: Xerxes ruled Persia and invaded Greece with many soldiers.\n"
        "* 1325: Tenochtitlan was founded by the Mexica people in Mexico.\n"
    )
    items = parse_list_items(wt, "List_of_foo")
    years = {it["year"] for it in items}
    # order-independent checks
    assert 1932 in years
    assert 79 in years
    assert 1325 in years
    # BC must be stored as a negative integer.
    assert -500 in years
    # exactly four items parsed
    assert len(items) == 4


def test_parse_year_none_when_absent():
    wt = (
        "* The Dancing Plague saw people dance uncontrollably in Strasbourg "
        "for no clear reason that anyone could explain at the time.\n"
    )
    items = parse_list_items(wt, "List_of_foo")
    assert len(items) == 1
    assert items[0]["year"] is None
    # still a valid candidate, not discarded
    assert items[0]["title"]


def test_parse_discards_short_noise():
    wt = (
        "* This is a perfectly valid and sufficiently long candidate line that should be kept.\n"
        "* xy short\n"
        "* ok\n"
        "* [[Real Event]] that is long enough to be retained as a good candidate item here.\n"
    )
    items = parse_list_items(wt, "List_of_foo")
    # only the two long lines survive
    assert len(items) == 2
    for it in items:
        assert len(it["summary"]) >= 25


def test_parse_full_sample_counts_and_sources():
    items = parse_list_items(WIKITEXT_SAMPLE, "List_of_unusual_deaths")
    # 13 raw lines, 1 short-noise line dropped -> 12 candidates
    assert len(items) == 12
    for it in items:
        assert it["source_name"] == "wikipedia:List_of_unusual_deaths"
        assert it["source_url"] == "https://en.wikipedia.org/wiki/List_of_unusual_deaths"
        assert 0 < len(it["title"]) <= 120
        assert 0 < len(it["summary"]) <= 500


# --------------------------------------------------------------------------- #
# harvest_wikipedia_lists -- integration with the pool
# --------------------------------------------------------------------------- #


def test_harvest_inserts_into_pool(respx_mock):
    respx_mock.get(API_URL).mock(side_effect=_page_router({"Page1": PAGE_ONE, "Page2": PAGE_TWO}))
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        conn = _fresh_db(Path(d))
        summary = harvest_wikipedia_lists(conn, pages=["Page1", "Page2"], sleep_seconds=0)
        assert summary["pages_fetched"] == 2
        assert summary["candidates_found"] == 8  # 4 + 4
        assert summary["inserted"] == 8
        assert summary["skipped_duplicate"] == 0
        assert summary["skipped_invalid"] == 0
        assert summary["pages_failed"] == 0
        assert db.counts(conn)["pool"] == 8
        # rows carry the correct source_name
        rows = conn.execute(
            "SELECT source_name FROM pool WHERE source_name='wikipedia:Page1'"
        ).fetchall()
        assert len(rows) == 4


def test_harvest_continues_on_page_failure(respx_mock):
    respx_mock.get(API_URL).mock(
        side_effect=_page_router({"BadPage": _ERROR_MARKER, "GoodPage": PAGE_ONE})
    )
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        conn = _fresh_db(Path(d))
        summary = harvest_wikipedia_lists(
            conn, pages=["BadPage", "GoodPage"], sleep_seconds=0
        )
        # the failed page is counted as a failure, not as fetched
        assert summary["pages_failed"] == 1
        assert summary["pages_fetched"] == 1
        # the good page's candidates still landed
        assert summary["candidates_found"] == 4
        assert summary["inserted"] == 4
        assert db.counts(conn)["pool"] == 4


def test_harvest_idempotent(respx_mock):
    respx_mock.get(API_URL).mock(side_effect=_page_router({"Page1": PAGE_ONE, "Page2": PAGE_TWO}))
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        conn = _fresh_db(Path(d))
        first = harvest_wikipedia_lists(conn, pages=["Page1", "Page2"], sleep_seconds=0)
        assert first["inserted"] == 8
        second = harvest_wikipedia_lists(conn, pages=["Page1", "Page2"], sleep_seconds=0)
        assert second["inserted"] == 0
        assert second["skipped_duplicate"] == 8
        assert db.counts(conn)["pool"] == 8


def test_default_pages_is_sane():
    assert isinstance(DEFAULT_PAGES, list)
    assert "List_of_wars_of_succession" in DEFAULT_PAGES
    assert "Lists_of_unusual_deaths" in DEFAULT_PAGES
    assert "List_of_Ig_Nobel_Prize_winners" in DEFAULT_PAGES
    assert "List_of_hoaxes" in DEFAULT_PAGES
    assert len(DEFAULT_PAGES) >= 5


# --------------------------------------------------------------------------- #
# Redirect following
# --------------------------------------------------------------------------- #

REAL_WIKITEXT = (
    "* [[Emu War]] (1932) fought in [[Australia]]; soldiers lost to birds "
    "in a ridiculous campaign.\n"
)


def test_follows_redirect(respx_mock):
    respx_mock.get(API_URL).mock(
        side_effect=_page_router(
            {"PageA": "#REDIRECT [[Page B]]\n", "Page B": REAL_WIKITEXT}
        )
    )
    assert fetch_page_wikitext("PageA") == REAL_WIKITEXT


def test_redirect_loop_raises(respx_mock):
    respx_mock.get(API_URL).mock(
        side_effect=_page_router(
            {"PageA": "#REDIRECT [[PageB]]\n", "PageB": "#REDIRECT [[PageA]]\n"}
        )
    )
    try:
        fetch_page_wikitext("PageA")
        assert False, "expected HarvestError"
    except HarvestError:
        pass


def test_extract_redirect_target():
    f = wikipedia_lists._extract_redirect_target
    assert f("#REDIRECT [[Lists of unusual deaths]]\n{{x}}") == "Lists of unusual deaths"
    assert f("  #redirect[[Foo|bar]]") == "Foo"
    assert f("* not a redirect line") is None


# --------------------------------------------------------------------------- #
# Section-year fallback
# --------------------------------------------------------------------------- #


def test_section_year_fallback():
    wt = (
        "== 1991 ==\n"
        "* [[David Schmidt]] won for studying the physics of wet t-shirts, a fine study.\n"
        "===1992===\n"
        "* [[Someone Else]] won for an equally silly but well documented investigation.\n"
    )
    items = parse_list_items(wt, "List_of_foo")
    assert len(items) == 2
    assert items[0]["year"] == 1991
    assert items[1]["year"] == 1992


def test_inline_year_beats_section_year():
    wt = (
        "== 1991 ==\n"
        "* In 1975 someone did a thing that was documented at considerable length here.\n"
    )
    items = parse_list_items(wt, "List_of_foo")
    assert items[0]["year"] == 1975


def test_non_year_heading_ignored():
    wt = (
        "== Physics ==\n"
        "* Someone did a thing with no year at all but plenty of words to keep it.\n"
    )
    items = parse_list_items(wt, "List_of_foo")
    assert len(items) == 1
    assert items[0]["year"] is None
