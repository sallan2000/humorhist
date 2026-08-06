"""Shared plain-text rendering of a draft (brief + angles).

Both the CLI review loop and the Telegram transport call ``render_draft`` so
the presentation of a draft is identical everywhere it is shown.
"""

from __future__ import annotations

import json


def render_draft(row, pool=None) -> str:
    """Render a draft as plain text.

    ``row`` is a drafts-row mapping (must contain ``brief_json`` / ``angles_json``).
    ``pool`` is an optional pool-row mapping supplying the title/year; when omitted
    the title renders as '(unknown)'.
    """
    title = pool["title"] if pool else "(unknown)"
    year = pool["year"] if pool else ""
    brief = json.loads(row["brief_json"] or "{}")
    angles = json.loads(row["angles_json"] or "{}")

    out: list[str] = []
    out.append("=" * 70)
    out.append(f"DRAFT {row['id']} — {title} ({year})")
    out.append(f"status: {row['status']}")
    out.append("=" * 70)

    out.append("\n--- VERIFIED FACTS ---")
    for f in brief.get("verified_facts", []):
        out.append(f"  • {f}")

    if brief.get("misconceptions"):
        out.append("\n--- MISCONCEPTIONS (popular version vs record) ---")
        for m in brief["misconceptions"]:
            out.append(f"  ! {m}")

    if brief.get("caveats"):
        out.append("\n--- CAVEATS ---")
        for c in brief["caveats"]:
            out.append(f"  ? {c}")

    out.append("\n--- COMIC ANGLES ---")
    for i, a in enumerate(angles.get("angles", []), 1):
        out.append(f"\n  {i}. {a.get('angle_name', '?')}")
        out.append(f"     setup   : {a.get('setup', '')}")
        out.append(f"     lands   : {a.get('why_it_lands', '')}")
        out.append(f"     pitfalls: {a.get('pitfalls', '')}")
        for rm in a.get("raw_material", []):
            out.append(f"     raw     : {rm}")

    if angles.get("strongest_single_detail"):
        out.append(f"\n--- STRONGEST DETAIL ---\n  {angles['strongest_single_detail']}")
    if angles.get("suggested_hook"):
        out.append(f"\n--- SUGGESTED HOOK (factual, not a joke) ---\n  {angles['suggested_hook']}")

    out.append("\n--- SOURCES ---")
    for s in brief.get("sources", []):
        out.append(f"  {s.get('title', '')} — {s.get('url', '')}")
    out.append("")
    return "\n".join(out)
