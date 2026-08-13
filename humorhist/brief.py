"""Comic angle generation for humorhist.

This is the core creative-assist module of the whole product. The thesis is
that an LLM cannot write good comedy, but it CAN do the research, find where
the comic potential lies, and hand a human comedian the raw material. This
module produces that material: a set of DISTINCT comic angles, each with the
specific, concrete details a writer needs.

It consumes a verified brief produced by ``humorhist.factcheck`` (keys:
``verified_facts``, ``dates``, ``key_figures``, ``caveats``,
``misconceptions``, ``sources``, ``sensitivity_flags``) and the pool ``item``
(keys such as ``title``, ``year``, ``summary``).

All LLM-touching code goes through the ``LLMClient`` protocol (see
``humorhist.llm``) so tests can inject ``StubClient`` and never hit the
network.
"""

from __future__ import annotations

import json
import sqlite3
from typing import Any

from humorhist.llm import LLMClient, LLMError

# One LLM call, then a single validation retry. Same contract as the
# pre-screen: initial attempt + one retry.
_MAX_ATTEMPTS = 2

# Required keys inside each angle object, in schema order.
_ANGLE_FIELDS = ("angle_name", "setup", "why_it_lands", "pitfalls", "raw_material")

# Top-level required keys outside the angles list.
_TOP_FIELDS = ("strongest_single_detail", "suggested_hook")


class AnglesError(ValueError):
    """Raised when an angles payload violates the required schema."""


# --- The core deliverable: the system prompt --------------------------------

ANGLES_SYSTEM_PROMPT = """\
You are a comedy writer's research assistant. You are NOT writing the final \
joke -- a human comedian will do that. Your job is to identify where the comic \
potential lies in the historical event below and hand them the raw material.

The finished product is a human-written joke built on a real, verified event. \
You are the scout, not the performer. Do not try to be funny in your output. \
Be specific, be useful, and be honest about what is and is not solid ground.

PRODUCE 3-5 DISTINCT ANGLES.
Vary them STRUCTURALLY, not superficially. Five versions of the same joke are \
useless. Aim for different comic mechanisms, for example:
- absurd bureaucracy (forms, ministers, approvals for something silly)
- the gap between intention and outcome (grand plan, humbling result)
- a single ridiculous concrete detail (one fact that carries a whole post)
- a modern-day parallel (the same absurd pattern repeating now)
- the dignity of institutions collapsing (authority rendered silly, not cruel)
Do NOT produce five angles that are the same joke with different wording.

For EACH angle provide exactly these five fields:
- "angle_name": a short UPPERCASE label, e.g. "MILITARY INCOMPETENCE". Angle \
names MUST be distinct from one another (case-insensitive).
- "setup": how to frame / stage the story for this angle. 1-3 sentences.
- "why_it_lands": name the SPECIFIC incongruity -- the exact collision of \
expectations that makes this funny. Do not say "it's ironic"; say what collides.
- "pitfalls": how this angle could fall flat or be in poor taste, and how the \
writer should avoid that. Specific, actionable cautions.
- "raw_material": an array of SPECIFIC concrete details the writer can use -- \
quotes, numbers, names, dates, turns of phrase. Specificity is what makes \
history funny; vague observations ("it was a strange time") are useless to a \
writer. If you only have one concrete detail, give that one; never pad with \
generics.

HARD RULES -- these are non-negotiable:
- NEVER propose an angle whose humour requires punching down, or that mocks \
victims of violence, poverty, illness, or bigotry. Comedy at the expense of the \
powerless is off the table. Punch up, or punch at absurd systems and \
institutions, never at the vulnerable.
- If the funniest available framing depends on a disputed or debunked claim, \
you MUST flag that explicitly in that angle's "pitfalls" field. The brief gives \
you "caveats" (uncertain / partially disputed claims) and "misconceptions" \
(debunked claims). If an angle leans on one of those, say so plainly in pitfalls \
and tell the writer the claim is not safe to state as fact.
- "suggested_hook" (a top-level field, see schema) must be FACTUAL and NOT a \
joke. It is a runway for the human writer -- a verified opening line they can \
build on. No punchline, no wink.
- "strongest_single_detail" (top-level) should be the single most arresting \
concrete fact from the verified brief -- the one fact most likely to carry a \
post on its own.

OUTPUT FORMAT:
Return STRICT JSON matching this schema, with no commentary, prose, or code \
fences outside the JSON:
{
  "angles": [
    {
      "angle_name": str,
      "setup": str,
      "why_it_lands": str,
      "pitfalls": str,
      "raw_material": [str, ...]
    }
  ],
  "strongest_single_detail": str,
  "suggested_hook": str
}
"angles" MUST contain between 3 and 5 angle objects. All string fields must be \
non-empty. "raw_material" must be a non-empty array of strings. Output only the \
JSON.
"""


# --- Prompt construction ----------------------------------------------------


def build_angles_prompt(item: dict, brief: dict, steering_note: str | None = None) -> str:
    """Render the item + verified brief into the user prompt for the model.

    The disputed-claim warnings (``caveats`` and ``misconceptions``) are
    surfaced explicitly so the model cannot miss them -- a writer must never
    accidentally build a joke on a debunked claim.

    When ``steering_note`` is provided (e.g. an editor's /notes on a pending
    draft), it is appended as explicit steering so regeneration can pivot the
    angles toward what the human wants.
    """
    lines: list[str] = []

    title = item.get("title", "")
    year = item.get("year")
    summary = (item.get("summary") or "").strip()

    lines.append("HISTORICAL EVENT")
    if title:
        lines.append(f"Title: {title}")
    if year is not None:
        lines.append(f"Year: {year}")
    if summary:
        lines.append(f"Summary: {summary}")
    lines.append("")

    def _section(header: str, items: Any) -> None:
        lines.append(header)
        if not items:
            lines.append("  (none provided)")
        else:
            for entry in items:
                entry = str(entry).strip()
                if entry:
                    lines.append(f"  - {entry}")
        lines.append("")

    _section("VERIFIED FACTS:", brief.get("verified_facts"))
    _section("KEY DATES:", brief.get("dates"))
    _section("KEY FIGURES:", brief.get("key_figures"))
    _section(
        "CAVEATS (claims that are uncertain or partially disputed -- do not state as fact):",
        brief.get("caveats"),
    )
    _section(
        "MISCONCEPTIONS / DEBUNKED CLAIMS (do NOT build humour on these):",
        brief.get("misconceptions"),
    )
    _section("SOURCES:", brief.get("sources"))
    _section("SENSITIVITY FLAGS:", brief.get("sensitivity_flags"))

    if steering_note:
        lines.append("EDITOR STEERING (the human editor wants angles that lean into this):")
        lines.append(f"  {steering_note.strip()}")
        lines.append("")

    lines.append("Produce the comic angles payload now, following the system instructions exactly.")
    return "\n".join(lines)


# --- Validation -------------------------------------------------------------


def validate_angles(payload: Any) -> dict:
    """Validate and normalise an angles payload.

    Raises :class:`AnglesError` on any schema violation, naming the specific
    problem (including the actual angle count found). Returns the normalised
    payload (a bare-string ``raw_material`` is coerced to a one-element list).
    """
    if not isinstance(payload, dict):
        raise AnglesError(f"angles payload must be a JSON object, got {type(payload).__name__}")

    angles = payload.get("angles")
    if not isinstance(angles, list):
        raise AnglesError("'angles' must be a list of angle objects")
    count = len(angles)
    if count < 3:
        raise AnglesError(f"'angles' must contain 3-5 items, found {count}")
    if count > 5:
        raise AnglesError(f"'angles' must contain 3-5 items, found {count}")

    seen_names: set[str] = set()
    for idx, angle in enumerate(angles):
        label = f"angle #{idx + 1}"
        if not isinstance(angle, dict):
            raise AnglesError(f"{label} must be an object")

        for field in _ANGLE_FIELDS:
            if field not in angle:
                raise AnglesError(f"{label} is missing required field '{field}'")
            value = angle[field]

            if field == "raw_material":
                # Coerce a bare string into a one-element list.
                if isinstance(value, str):
                    angle[field] = [value]
                    value = angle[field]
                if not isinstance(value, list):
                    raise AnglesError(f"{label} 'raw_material' must be a list of strings")
                if len(value) == 0:
                    raise AnglesError(f"{label} 'raw_material' must be a non-empty list")
                for item in value:
                    if not isinstance(item, str) or not item.strip():
                        raise AnglesError(f"{label} 'raw_material' entries must be non-empty strings")
            else:
                if not isinstance(value, str) or not value.strip():
                    raise AnglesError(f"{label} field '{field}' must be a non-empty string")

        name = str(angle["angle_name"]).strip().lower()
        if name in seen_names:
            raise AnglesError(
                f"duplicate angle_name '{angle['angle_name']}' is not allowed; "
                "angle names must be distinct (case-insensitive)"
            )
        seen_names.add(name)

    for field in _TOP_FIELDS:
        value = payload.get(field)
        if not isinstance(value, str) or not value.strip():
            raise AnglesError(f"'{field}' must be a non-empty string")

    return payload


# --- Generation (one LLM call, validated, with one retry) -------------------


def generate_angles(
    client: LLMClient,
    item: dict,
    brief: dict,
    *,
    max_tokens: int = 4096,
    temperature: float = 0.7,
    steering_note: str | None = None,
) -> dict:
    """Generate validated comic angles for ``item`` from ``brief``.

    Makes a single LLM call, validates the result, and -- if validation fails --
    retries ONCE with the specific validation error appended so the model can
    self-correct. Raises :class:`AnglesError` if both attempts fail.

    ``steering_note`` (an editor's note) is forwarded into the prompt so the
    angles pivot toward what the human wants (used by regeneration).
    """
    prompt = build_angles_prompt(item, brief, steering_note=steering_note)
    last_error: Exception | None = None

    for _ in range(_MAX_ATTEMPTS):
        try:
            raw = client.complete_json(
                ANGLES_SYSTEM_PROMPT,
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return validate_angles(raw)
        except (AnglesError, LLMError) as exc:
            last_error = exc
            prompt = (
                prompt
                + "\n\n---\nYour previous response was REJECTED during validation:\n"
                + str(exc)
                + "\n\nFix the problem and return STRICT JSON matching the required "
                "schema, with no text outside the JSON."
            )

    if last_error is None:  # pragma: no cover - defensive
        last_error = AnglesError("generate_angles failed with no error captured")
    raise last_error


def regenerate_angles(
    conn: sqlite3.Connection,
    client: LLMClient,
    draft_id: str,
    *,
    steering_note: str | None = None,
    http_client=None,
) -> dict:
    """Re-generate the comic angles for an existing draft, steering on a note.

    Used by the Telegram ``/notes`` flow on a *pending* draft: instead of only
    storing the note, it re-runs angle generation with the note as steering and
    writes the fresh ``angles_json`` back onto the draft (and stores the note as
    ``editor_notes``). The draft stays ``pending`` -- no review decision is made.

    Reuses the draft's existing ``brief_json``; only the angles are regenerated.
    Raises ValueError if the draft is unknown or not pending.
    """

    import humorhist.db as db

    row = conn.execute(
        "SELECT id, pool_id, brief_json, angles_json, status FROM drafts WHERE id = ?",
        (draft_id,),
    ).fetchone()
    if row is None:
        raise ValueError(f"no draft with id {draft_id!r}")
    if row["status"] != "pending":
        raise ValueError(f"draft {draft_id!r} has status {row['status']!r}; only pending drafts can regenerate angles")

    item = _row_to_item(db.get_pool_item(conn, row["pool_id"]))
    brief = json.loads(row["brief_json"] or "{}")
    angles = generate_angles(client, item, brief, steering_note=steering_note)

    conn.execute(
        "UPDATE drafts SET angles_json = ?, editor_notes = ? WHERE id = ?",
        (json.dumps(angles, ensure_ascii=False), (steering_note or "").strip() or None, draft_id),
    )
    conn.commit()
    return angles


def _row_to_item(row: sqlite3.Row | None) -> dict:
    """Adapt a pool row into the item dict the brief/drafting code expects."""
    if row is None:
        return {}
    return {
        "id": row["id"],
        "title": row["title"],
        "year": row["year"],
        "summary": row["summary"],
        "source_url": row["source_url"],
        "url": row["source_url"],  # alias the fact-check prompt looks for
        "source_name": row["source_name"],
        "funny_score": row["funny_score"],
    }
