"""Tests for humorhist.imagegen -- story image generation (Phase 3, A+B).

No network: the LLM is a StubClient and the image model is a StubImageClient.
We assert the prompt is distilled from the draft, the image bytes are written to
disk, and the prompt + path come back, plus the resilience/error paths.
"""

from __future__ import annotations

import pytest
from pathlib import Path

import humorhist.db as db
from humorhist.imagegen import (
    ImageError,
    ImageUnavailable,
    StubImageClient,
    generate_image,
    generate_image_prompt,
    image_style,
    resilient_image_client,
)
from humorhist.llm import StubClient

import humorhist.imagegen as ig


def _fresh_db(tmp_path):
    conn = db.connect(str(tmp_path / "test.sqlite"))
    db.migrate(conn)
    return conn


def _seed(conn, draft_id="d1"):
    conn.execute(
        "INSERT OR IGNORE INTO pool (id, title, status) VALUES ('pool-x', 'The Tax-Dodge Bear', 'drafted')"
    )
    conn.execute(
        """INSERT INTO drafts (id, pool_id, brief_json, angles_json, status, created_at)
           VALUES (?, 'pool-x',
                   '{"verified_facts": ["bears filed taxes"], "misconceptions": ["bears are lazy"]}',
                   '{"angles": [{"angle_name": "comedy of paperwork"}], "suggested_hook": "the bear filed"}',
                   'pending', '2026-01-01T00:00:00+00:00')""",
        (draft_id,),
    )
    # generate_image_for_queue reads/writes via the queue row (db.set_image/get_image)
    conn.execute("INSERT OR IGNORE INTO queue (draft_id, published) VALUES (?, 0)", (draft_id,))
    conn.commit()


def _draft_row(conn, draft_id="d1"):
    return dict(conn.execute("SELECT * FROM drafts WHERE id=?", (draft_id,)).fetchone())


def _pool_row(conn):
    return dict(conn.execute("SELECT * FROM pool WHERE id='pool-x'").fetchone())


def test_image_style_default_and_override(monkeypatch):
    monkeypatch.delenv("HUMORHIST_IMAGE_STYLE", raising=False)
    assert image_style() == "editorial-historical"
    monkeypatch.setenv("HUMORHIST_IMAGE_STYLE", "meme")
    assert image_style() == "meme"
    monkeypatch.setenv("HUMORHIST_IMAGE_STYLE", "bogus")
    assert image_style() == "editorial-historical"  # invalid -> default


def test_generate_image_prompt_distills_draft(tmp_path, monkeypatch):
    conn = _fresh_db(tmp_path)
    _seed(conn)
    monkeypatch.setenv("HUMORHIST_IMAGE_STYLE", "editorial-historical")
    llm = StubClient([{"prompt": "a wry period scene of the tax-dodge bear, paperwork, no text"}])
    prompt = generate_image_prompt(llm, _draft_row(conn), _pool_row(conn))
    assert "tax-dodge bear" in prompt.lower()
    # the model was asked with the verified facts (not the misconception)
    sys = llm.calls[0]["system"]
    assert "verified facts" in sys.lower()


def test_generate_image_writes_file_and_returns_prompt_path(tmp_path, monkeypatch):
    conn = _fresh_db(tmp_path)
    _seed(conn)
    monkeypatch.setenv("HUMORHIST_IMAGE_STYLE", "meme")
    llm = StubClient([{"prompt": "bold meme of a bear doing taxes"}])
    img_client = StubImageClient([b"\x89PNG fake"])
    out_dir = tmp_path / "images"
    path, prompt = generate_image(
        llm, img_client, _draft_row(conn), _pool_row(conn),
        out_dir=out_dir, draft_id="d1",
    )
    assert path.endswith("d1.png")
    assert prompt == "bold meme of a bear doing taxes"
    assert (out_dir / "d1.png").is_file()
    assert (out_dir / "d1.png").read_bytes() == b"\x89PNG fake"


def test_generate_image_prompt_failure_raises_image_error(tmp_path):
    conn = _fresh_db(tmp_path)
    _seed(conn)
    llm = StubClient([{"prompt": "x"}])  # <12 chars -> degenerate, retried -> fails
    img_client = StubImageClient([b"png"])
    with pytest.raises(ImageError):
        generate_image(llm, img_client, _draft_row(conn), _pool_row(conn), out_dir=tmp_path, draft_id="d1")


def test_resilient_image_client_requires_key(monkeypatch):
    monkeypatch.delenv("HUMORHIST_IMAGE_API_KEY", raising=False)
    with pytest.raises(ImageUnavailable):
        resilient_image_client()


def test_stub_image_client_exhaustion_raises(monkeypatch):
    from humorhist.imagegen import ImageError

    img_client = StubImageClient([])
    with pytest.raises(ImageError):
        img_client.generate("a prompt")


def test_generate_image_for_queue_persists_and_returns(tmp_path, monkeypatch):
    """generate_image_for_queue generates, writes the PNG, persists on the queue
    row, and returns (path, prompt); returns None when no image client."""
    from humorhist.imagegen import ImageUnavailable
    from humorhist.llm import StubClient

    conn = _fresh_db(tmp_path)
    _seed(conn)

    # No image client -> returns None, nothing persisted, no crash.
    monkeypatch.setattr("humorhist.imagegen.resilient_image_client",
                        lambda: (_ for _ in ()).throw(ImageUnavailable("no key")))
    assert ig.generate_image_for_queue(conn, "d1", out_dir=tmp_path / "images") is None
    none_info = db.get_image(conn, "d1")
    assert none_info is not None
    assert none_info["image_path"] is None

    # With a client -> generates + persists + returns.
    monkeypatch.setattr("humorhist.llm.default_client",
                        lambda: StubClient([{"prompt": "a wry period scene"}]))
    monkeypatch.setattr("humorhist.imagegen.resilient_image_client",
                        lambda: StubImageClient([b"\x89PNG generated"]))
    out_dir = tmp_path / "images"
    res = ig.generate_image_for_queue(conn, "d1", out_dir=out_dir)
    assert res is not None
    path, prompt = res
    assert path.endswith("d1.png") and Path(path).is_file()
    info = db.get_image(conn, "d1")
    assert info is not None
    assert info["image_path"] == path and info["image_prompt"] == "a wry period scene"
