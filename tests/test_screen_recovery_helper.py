"""Exercise the documented offline helper against disposable SQLite databases."""

import json
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

HELPER = Path(__file__).resolve().parents[1] / 'scripts' / 'recover_screen_reservation.py'
IDS = ['legacy-id', '"legacy-id"', '  café  ', 'cafe\u0301', 'line\n\x1b[31m\x7f\x00']


def setup_db(path):
    with sqlite3.connect(path) as conn:
        conn.execute('CREATE TABLE screen_attempts (pool_id TEXT COLLATE NOCASE, state TEXT)')
        conn.executemany('INSERT INTO screen_attempts VALUES (?, ?)', [(v, 'pending') for v in IDS])
        conn.execute("INSERT INTO screen_attempts VALUES ('LEGACY-ID', 'pending')")
        conn.execute("INSERT INTO screen_attempts VALUES ('failed', 'failed')")


def rows(path):
    with sqlite3.connect(path) as conn:
        return conn.execute('SELECT pool_id, state FROM screen_attempts ORDER BY rowid').fetchall()


def run_helper(path, encoded, confirm='DELETE'):
    return subprocess.run(
        [sys.executable, str(HELPER)], input=f'{path}\n{encoded}\n{confirm}\n',
        text=True, capture_output=True, check=False,
    )


@pytest.mark.parametrize('ident', IDS)
def test_delete_exact_decoded_id_only(tmp_path, ident):
    path = tmp_path / 'test.sqlite'
    setup_db(path)
    before = rows(path)
    encoded = json.dumps(ident, ensure_ascii=True).replace('\x7f', r'\u007f')
    result = run_helper(path, encoded)
    assert result.returncode == 0, result.stderr
    assert f'Reviewed id_json: {encoded}' in result.stdout
    assert '\x1b' not in result.stdout and '\x7f' not in result.stdout
    assert rows(path) == [row for row in before if row[0] != ident]


@pytest.mark.parametrize('encoded', ['legacy-id', '"unterminated', 'null', '1', '[]', '{}', 'true', '"\\ud800"'])
def test_invalid_input_never_deletes(tmp_path, encoded):
    path = tmp_path / 'test.sqlite'
    setup_db(path)
    before = rows(path)
    assert run_helper(path, encoded).returncode != 0
    assert rows(path) == before


@pytest.mark.parametrize(('encoded', 'confirm'), [('"legacy-id"', 'no'), ('"failed"', 'DELETE'), ('"missing"', 'DELETE')])
def test_unconfirmed_or_nonpending_never_deletes(tmp_path, encoded, confirm):
    path = tmp_path / 'test.sqlite'
    setup_db(path)
    before = rows(path)
    assert run_helper(path, encoded, confirm).returncode != 0
    assert rows(path) == before


def test_duplicate_id_refuses_delete(tmp_path):
    path = tmp_path / 'test.sqlite'
    setup_db(path)
    with sqlite3.connect(path) as conn:
        conn.execute("INSERT INTO screen_attempts VALUES ('legacy-id', 'pending')")
    before = rows(path)
    assert run_helper(path, '"legacy-id"').returncode != 0
    assert rows(path) == before


def snapshot_directory(directory):
    return {
        entry.name: (entry.read_bytes(), entry.stat().st_mtime_ns)
        for entry in directory.iterdir() if entry.is_file()
    }


def test_cancel_preserves_hot_rollback_journal(tmp_path):
    path = tmp_path / 'hot.sqlite'
    setup_db(path)
    with sqlite3.connect(path) as conn:
        conn.execute('CREATE TABLE padding (value TEXT)')
        conn.executemany('INSERT INTO padding VALUES (?)', [('x' * 3000,)] * 200)
    before_rows = rows(path)
    crash = subprocess.run(
        [sys.executable, '-c', '''
import os, sqlite3, sys
conn = sqlite3.connect(sys.argv[1])
conn.execute("PRAGMA journal_mode=DELETE")
conn.execute("PRAGMA cache_size=5")
conn.execute("BEGIN IMMEDIATE")
conn.execute("UPDATE padding SET value = 'changed'")
os._exit(0)
''', str(path)], check=False, capture_output=True,
    )
    assert crash.returncode == 0, crash.stderr
    journal = tmp_path / 'hot.sqlite-journal'
    assert journal.exists() and journal.read_bytes()[:8] == bytes.fromhex('d9d505f920a163d7')
    before = snapshot_directory(tmp_path)
    result = run_helper(path, '"legacy-id"', 'no')
    assert result.returncode != 0 and 'Cancelled' in result.stdout
    assert snapshot_directory(tmp_path) == before
    # A confirmed run recovers the journal, then deletes only the selected ID.
    result = run_helper(path, '"legacy-id"')
    assert result.returncode == 0, result.stderr
    assert rows(path) == [row for row in before_rows if row[0] != 'legacy-id']
    with sqlite3.connect(path) as conn:
        assert conn.execute("SELECT count(*) FROM padding WHERE value = 'changed'").fetchone()[0] == 0


def test_cancel_preserves_wal_artifacts(tmp_path):
    path = tmp_path / 'wal.sqlite'
    setup_db(path)
    before_rows = rows(path)
    with sqlite3.connect(path) as conn:
        assert conn.execute('PRAGMA journal_mode=WAL').fetchone()[0] == 'wal'
        conn.execute("INSERT INTO screen_attempts VALUES ('wal-only', 'pending')")
        conn.commit()
        assert (tmp_path / 'wal.sqlite-wal').stat().st_size > 0
        assert (tmp_path / 'wal.sqlite-shm').exists()
        before = snapshot_directory(tmp_path)
        result = run_helper(path, '"legacy-id"', 'no')
        assert result.returncode != 0 and 'Cancelled' in result.stdout
        assert snapshot_directory(tmp_path) == before
        result = run_helper(path, '"legacy-id"')
        assert result.returncode == 0, result.stderr
        assert rows(path) == [row for row in before_rows if row[0] != 'legacy-id'] + [('wal-only', 'pending')]


def test_cancel_does_not_require_existing_path(tmp_path):
    result = run_helper(tmp_path / 'missing.sqlite', '"legacy-id"', 'no')
    assert result.returncode != 0 and 'Cancelled' in result.stdout
    assert not list(tmp_path.iterdir())
