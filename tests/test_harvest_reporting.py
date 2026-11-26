"""Offline tests for honest, safe harvest reporting."""

import httpx
import pytest

from humorhist import db
from humorhist import telegram as tg
from humorhist.harvest.wikipedia_lists import API_URL, HarvestError, fetch_page_wikitext, harvest_wikipedia_lists


def test_http_status_checked_before_json(respx_mock):
    respx_mock.get(API_URL).mock(return_value=httpx.Response(403, text="secret response body"))
    with pytest.raises(HarvestError) as exc:
        fetch_page_wikitext("Some page")
    assert exc.value.status_code == 403
    assert "HTTP 403" in str(exc.value)
    assert "secret" not in str(exc.value)


def test_harvest_classifies_http_and_network_without_leaking(respx_mock, tmp_path, caplog):
    responses = [httpx.Response(403, text="secret-body"), httpx.ConnectError("secret-url")]
    respx_mock.get(API_URL).mock(side_effect=responses)
    conn = db.connect(str(tmp_path / "db.sqlite"))
    db.migrate(conn)
    try:
        result = harvest_wikipedia_lists(conn, pages=["one", "two"], sleep_seconds=0)
    finally:
        conn.close()
    assert result["pages_failed"] == 2
    assert result["http_403"] == 1 and result["network_failed"] == 1
    assert "secret" not in caplog.text


def test_telegram_harvest_source_failures_and_no_secrets(monkeypatch, tmp_path):
    conn = db.connect(str(tmp_path / "db.sqlite"))
    db.migrate(conn)
    def fail(_conn):
        raise RuntimeError("secret-token")
    monkeypatch.setattr("humorhist.harvest.seed.load_seed", fail)
    monkeypatch.setattr("humorhist.harvest.wikipedia_lists.harvest_wikipedia_lists", lambda _: {
        "inserted": 0, "skipped_duplicate": 0, "skipped_invalid": 0,
        "pages_fetched": 0, "pages_failed": 5, "http_403": 5,
    })
    client = tg.StubTelegram()
    try:
        future = tg.telegram_harvest(conn, client, "123")
        future.result(5)
    finally:
        conn.close()
    message = client.sent[-1]["text"]
    assert "Harvest incomplete" in message and "Seed list: failed" in message
    assert "access denied (HTTP 403): 5" in message
    assert "does not create drafts" in message and "secret-token" not in message
