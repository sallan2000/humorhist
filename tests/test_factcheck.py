"""Tests for the fact-check pass. No real network or LLM calls, ever."""

from __future__ import annotations

import copy

import httpx
import pytest
import respx

from humorhist.factcheck import (
    FACTCHECK_SYSTEM_PROMPT,
    FactCheckError,
    build_factcheck_prompt,
    factcheck,
    fetch_wikipedia_extract,
    validate_brief,
)
from humorhist.llm import StubClient

SUMMARY_URL = "https://en.wikipedia.org/api/rest_v1/page/summary/Emu_War"

GOOD_BRIEF = {
    "verified_facts": [
        "The Australian military deployed soldiers with Lewis guns in Nov 1932.",
        "Major G.P.W. Meredith commanded the operation in Western Australia.",
        "Around 986 emus were confirmed killed by the end of the operation.",
    ],
    "dates": {"event": "November 1932", "precision": "month"},
    "key_figures": ["Major G.P.W. Meredith", "Sir George Pearce"],
    "caveats": ["Kill counts were self-reported and are disputed."],
    "misconceptions": ["Popularly told as 'the emus won a war'; no war was declared."],
    "sources": [{"title": "Emu War", "url": "https://en.wikipedia.org/wiki/Emu_War"}],
    "sensitivity_flags": ["animal death"],
}

ITEM = {
    "id": "pool-1",
    "title": "Emu War",
    "year": 1932,
    "summary": "Australian soldiers deployed against emus.",
    "url": "https://en.wikipedia.org/wiki/Emu_War",
}

EXTRACT = "The Emu War was a wildlife management operation in Western Australia."


def good_brief() -> dict:
    return copy.deepcopy(GOOD_BRIEF)


# --- fetch_wikipedia_extract ------------------------------------------------


@respx.mock
def test_fetch_extract_from_title():
    respx.get(SUMMARY_URL).mock(
        return_value=httpx.Response(200, json={"extract": EXTRACT})
    )
    assert fetch_wikipedia_extract("Emu War") == EXTRACT


@respx.mock
def test_fetch_extract_from_url():
    route = respx.get(SUMMARY_URL).mock(
        return_value=httpx.Response(200, json={"extract": EXTRACT})
    )
    assert fetch_wikipedia_extract("https://en.wikipedia.org/wiki/Emu_War") == EXTRACT
    assert route.called
    assert route.calls[0].request.url.path == "/api/rest_v1/page/summary/Emu_War"


@respx.mock
def test_fetch_extract_raises_on_404():
    respx.get(SUMMARY_URL).mock(return_value=httpx.Response(404, json={}))
    with pytest.raises(FactCheckError):
        fetch_wikipedia_extract("Emu War")


@respx.mock
def test_fetch_extract_sends_user_agent():
    route = respx.get(SUMMARY_URL).mock(
        return_value=httpx.Response(200, json={"extract": EXTRACT})
    )
    fetch_wikipedia_extract("Emu War")
    ua = route.calls[0].request.headers["User-Agent"]
    assert ua == "humorhist/0.1 (personal history project; contact: stevie@local)"


# --- validate_brief ---------------------------------------------------------


def test_validate_brief_accepts_good():
    out = validate_brief(good_brief())
    assert out["verified_facts"] == GOOD_BRIEF["verified_facts"]
    assert out["sources"][0]["url"] == "https://en.wikipedia.org/wiki/Emu_War"


def test_validate_brief_missing_sources_raises():
    brief = good_brief()
    brief["sources"] = []
    with pytest.raises(FactCheckError, match="sources"):
        validate_brief(brief)


@pytest.mark.parametrize(
    "key",
    [
        "verified_facts",
        "dates",
        "key_figures",
        "caveats",
        "misconceptions",
        "sources",
        "sensitivity_flags",
    ],
)
def test_validate_brief_missing_key_raises(key):
    brief = good_brief()
    del brief[key]
    with pytest.raises(FactCheckError, match=key):
        validate_brief(brief)


def test_validate_brief_bad_precision_raises():
    brief = good_brief()
    brief["dates"]["precision"] = "epoch"
    with pytest.raises(FactCheckError, match="precision"):
        validate_brief(brief)


def test_validate_brief_coerces_string_to_list():
    brief = good_brief()
    brief["caveats"] = "Kill counts are disputed."
    assert validate_brief(brief)["caveats"] == ["Kill counts are disputed."]


def test_validate_brief_normalises_empty_sensitivity_flags():
    brief = good_brief()
    brief["sensitivity_flags"] = []
    assert validate_brief(brief)["sensitivity_flags"] == ["none"]


def test_validate_brief_rejects_empty_verified_facts():
    brief = good_brief()
    brief["verified_facts"] = []
    with pytest.raises(FactCheckError, match="verified_facts"):
        validate_brief(brief)


# --- build_factcheck_prompt -------------------------------------------------


def test_build_prompt_includes_extract_and_title():
    prompt = build_factcheck_prompt(ITEM, EXTRACT)
    assert "Emu War" in prompt
    assert EXTRACT in prompt


# --- factcheck --------------------------------------------------------------


def test_factcheck_succeeds_first_try():
    stub = StubClient(responses=[good_brief()])
    brief = factcheck(stub, ITEM, EXTRACT)
    assert brief["dates"]["precision"] == "month"
    assert len(stub.calls) == 1


def test_factcheck_retries_on_invalid_then_succeeds():
    bad = good_brief()
    bad["sources"] = []
    stub = StubClient(responses=[bad, good_brief()])

    brief = factcheck(stub, ITEM, EXTRACT)

    assert brief["sources"]
    assert len(stub.calls) == 2
    assert "sources" in stub.calls[1]["user"]
    # the second prompt must carry the specific validation error text
    assert "previous" in stub.calls[1]["user"].lower()


def test_factcheck_raises_after_two_failures():
    bad = good_brief()
    bad["sources"] = []
    stub = StubClient(responses=[copy.deepcopy(bad), copy.deepcopy(bad)])
    with pytest.raises(FactCheckError):
        factcheck(stub, ITEM, EXTRACT)
    assert len(stub.calls) == 2


def test_system_prompt_mentions_misconceptions_and_caveats():
    assert "misconceptions" in FACTCHECK_SYSTEM_PROMPT
    assert "caveats" in FACTCHECK_SYSTEM_PROMPT
