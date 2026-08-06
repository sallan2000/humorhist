"""Tests for humorhist.brief -- comic angle generation.

TDD with ``StubClient``: no network calls are ever made. The prompt is the
core deliverable of the whole product, so we also assert that the disputed-
claim warnings (caveats + misconceptions) actually reach the model prompt.
"""

from __future__ import annotations

import pytest

from humorhist.brief import (
    ANGLES_SYSTEM_PROMPT,
    AnglesError,
    build_angles_prompt,
    generate_angles,
    validate_angles,
)
from humorhist.llm import LLMError, StubClient

# --- fixtures ---------------------------------------------------------------

EMU_ITEM = {
    "title": "The Emu War",
    "year": 1932,
    "summary": (
        "Australia deployed soldiers with machine guns against ~20,000 emus "
        "and effectively lost."
    ),
}

EMU_BRIEF = {
    "verified_facts": [
        "In late 1932 the Australian government sanctioned a military cull of "
        "emus in Western Australia using soldiers armed with Lewis machine guns.",
        "An estimated ~20,000 emus were roaming the Campion district during harvest.",
        "After several weeks the gunners had killed only a few thousand emus; "
        "the operation was withdrawn.",
        "Defence Minister Sir George Pearce approved deploying the soldiers and guns.",
    ],
    "dates": [
        "November 1932 -- first culling operation begins",
        "December 1932 -- operation called off",
    ],
    "key_figures": [
        "Sir George Pearce (Minister for Defence)",
        "Major G.P.W. Meredith (commanded the soldiers)",
    ],
    "caveats": [
        "Emu kill counts vary by source; 'only 986 killed' is commonly cited but disputed.",
        "Some accounts say the military was 'recalled', others that it ran out of "
        "ammunition and funding.",
    ],
    "misconceptions": [
        "It is a myth that a formal 'war' was declared on the emus -- it was a "
        "culling operation, not a declared war.",
        "It is false that the government issued a surrender to the emus.",
    ],
    "sources": [
        "Wikipedia: Emu War",
        "The Sydney Morning Herald archival reports, 1932",
    ],
}


def _good_payload() -> dict:
    """A fully valid 4-angle payload built from the Emu War fixture."""
    return {
        "angles": [
            {
                "angle_name": "MILITARY INCOMPETENCE",
                "setup": (
                    "Frame it as a modern army that turned up to a fight and the "
                    "enemy refused to cooperate."
                ),
                "why_it_lands": (
                    "The incongruity is mechanised warfare losing to a bird that "
                    "doesn't understand it is supposed to lose."
                ),
                "pitfalls": (
                    "Don't mock the soldiers as individuals; mock the absurd gap "
                    "between the arsenal and the outcome."
                ),
                "raw_material": [
                    "Lewis machine guns deployed against flightless birds",
                    "~20,000 emus vs a handful of soldiers",
                    "operation withdrawn after weeks, not won",
                ],
            },
            {
                "angle_name": "BUREAUCRACY",
                "setup": (
                    "Present the paper trail: a Defence Minister signing off on "
                    "machine guns to solve a farming complaint."
                ),
                "why_it_lands": (
                    "The incongruity is the state treating a pest problem as a "
                    "military campaign requiring ministerial approval."
                ),
                "pitfalls": (
                    "Keep it on the absurdity of the paperwork, not on the "
                    "farmers' hardship."
                ),
                "raw_material": [
                    "Sir George Pearce, Minister for Defence, approved the deployment",
                    "soldiers and Lewis guns dispatched to a wheat district",
                ],
            },
            {
                "angle_name": "ONE RIDICULOUS DETAIL",
                "setup": (
                    "Open on the single strangest fact: the birds kept scattering "
                    "and reformed, like a tactical retreat nobody ordered."
                ),
                "why_it_lands": (
                    "The incongruity is that the emus behaved like a disciplined "
                    "opposing army without any of them knowing it."
                ),
                "pitfalls": (
                    "Avoid attributing human strategy or malice to the birds; the "
                    "funny is in the accident, not a conspiracy."
                ),
                "raw_material": [
                    "emus split into small groups to evade fire",
                    "gunners reported the birds 'split and dispersed' under fire",
                ],
            },
            {
                "angle_name": "MODERN PARALLEL",
                "setup": (
                    "Compare to any modern over-engineered solution that fails "
                    "against a simpler problem."
                ),
                "why_it_lands": (
                    "The incongruity is timeless: expensive hardware, trivial "
                    "adversary, humbling result."
                ),
                "pitfalls": (
                    "Name the parallel specifically; a vague 'like today' lands "
                    "flat. Do not punch down at anyone actually struggling."
                ),
                "raw_material": [
                    "1932: Lewis guns vs emus",
                    "the bill for soldiers, guns and ammunition to lose",
                ],
            },
        ],
        "strongest_single_detail": (
            "The Australian military deployed Lewis machine guns and, after "
            "weeks of effort, withdrew having failed to subdue a flock of "
            "flightless birds."
        ),
        "suggested_hook": (
            "In 1932 the Australian government sent soldiers with machine guns "
            "into Western Australia to cull emus."
        ),
    }


# --- build_angles_prompt ----------------------------------------------------

def test_build_angles_prompt_includes_facts_and_caveats():
    prompt = build_angles_prompt(EMU_ITEM, EMU_BRIEF)
    # The writer needs the verified facts, the disputed-claim warnings
    # (caveats) AND the debunked-claim warnings (misconceptions) in front
    # of the model.
    for fact in EMU_BRIEF["verified_facts"]:
        assert fact in prompt
    for caveat in EMU_BRIEF["caveats"]:
        assert caveat in prompt
    for misconception in EMU_BRIEF["misconceptions"]:
        assert misconception in prompt
    # And the event itself.
    assert EMU_ITEM["title"] in prompt
    assert str(EMU_ITEM["year"]) in prompt


def test_build_angles_prompt_includes_item_summary():
    prompt = build_angles_prompt(EMU_ITEM, EMU_BRIEF)
    assert EMU_ITEM["summary"] in prompt


# --- validate_angles --------------------------------------------------------

def test_validate_angles_accepts_good():
    payload = validate_angles(_good_payload())
    # raw_material stays a list; returns the same payload object normalised.
    assert isinstance(payload, dict)
    assert len(payload["angles"]) == 4


def test_validate_angles_rejects_too_few():
    payload = _good_payload()
    payload["angles"] = payload["angles"][:2]  # only 2 angles
    with pytest.raises(AnglesError) as exc:
        validate_angles(payload)
    assert "2" in str(exc.value)


def test_validate_angles_rejects_too_many():
    payload = _good_payload()
    # duplicate the list to get 8 angles
    payload["angles"] = payload["angles"] * 2
    with pytest.raises(AnglesError) as exc:
        validate_angles(payload)
    assert "8" in str(exc.value)


def test_validate_angles_rejects_duplicate_names():
    payload = _good_payload()
    payload["angles"][1]["angle_name"] = "bureaucracy"
    payload["angles"][2]["angle_name"] = "BUREAUCRACY"  # case-insensitive dup
    with pytest.raises(AnglesError):
        validate_angles(payload)


@pytest.mark.parametrize("field", [
    "angle_name", "setup", "why_it_lands", "pitfalls", "raw_material",
])
def test_validate_angles_rejects_missing_field(field):
    payload = _good_payload()
    del payload["angles"][0][field]
    with pytest.raises(AnglesError) as exc:
        validate_angles(payload)
    assert field in str(exc.value)


def test_validate_angles_coerces_raw_material_string():
    payload = _good_payload()
    payload["angles"][0]["raw_material"] = "a single bare string of detail"
    validate_angles(payload)
    assert payload["angles"][0]["raw_material"] == [
        "a single bare string of detail"
    ]


def test_validate_angles_rejects_empty_hook():
    payload = _good_payload()
    payload["suggested_hook"] = "   "
    with pytest.raises(AnglesError):
        validate_angles(payload)


# --- generate_angles --------------------------------------------------------

def test_generate_angles_retries_then_succeeds():
    bad = _good_payload()
    bad["angles"] = bad["angles"][:2]  # invalid: only 2 angles
    stub = StubClient([bad, _good_payload()])
    result = generate_angles(stub, EMU_ITEM, EMU_BRIEF)
    assert len(stub.calls) == 2
    # The second user prompt must carry the validation error so the model
    # can self-correct.
    assert "found 2" in stub.calls[1]["user"]
    assert len(result["angles"]) == 4


def test_generate_angles_raises_after_two_failures():
    bad1 = _good_payload()
    bad1["angles"] = bad1["angles"][:2]
    bad2 = _good_payload()
    bad2["angles"] = bad2["angles"][:2]
    stub = StubClient([bad1, bad2])
    with pytest.raises(AnglesError):
        generate_angles(stub, EMU_ITEM, EMU_BRIEF)
    assert len(stub.calls) == 2


def test_generate_angles_single_call_on_success():
    stub = StubClient([_good_payload()])
    result = generate_angles(stub, EMU_ITEM, EMU_BRIEF)
    assert len(stub.calls) == 1
    assert len(result["angles"]) == 4


def test_angles_system_prompt_is_meaningful():
    # The system prompt is the core deliverable; it must at minimum contain
    # the key instructions the product thesis depends on.
    assert "research assistant" in ANGLES_SYSTEM_PROMPT
    assert "NOT writing the final joke" in ANGLES_SYSTEM_PROMPT
    assert "DISTINCT" in ANGLES_SYSTEM_PROMPT
    assert "caveats" in ANGLES_SYSTEM_PROMPT
    assert "misconceptions" in ANGLES_SYSTEM_PROMPT
    assert "STRICT JSON" in ANGLES_SYSTEM_PROMPT
