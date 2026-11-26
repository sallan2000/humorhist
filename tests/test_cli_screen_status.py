"""Screen status treats persisted diagnostics as untrusted data."""
import hashlib
import json
import sqlite3

import humorhist.db as db
from humorhist.cli import main


def test_status_empty_legacy_and_corrupt(tmp_path, capsys):
    path = str(tmp_path / 'test.sqlite')
    conn = db.connect(path)
    db.migrate(conn)
    assert main(['--db', path, 'screen-status']) == 0
    assert 'Blocked unscored: 0' in capsys.readouterr().out
    conn.execute("INSERT INTO pool (id,title,status) VALUES ('pool000','secret-title','new')")
    conn.execute('CREATE TABLE screen_attempts (pool_id TEXT, state TEXT)')
    conn.execute("INSERT INTO screen_attempts VALUES ('pool000','pending')")
    conn.commit()
    assert main(['--db', path, 'screen-status']) == 0
    out = capsys.readouterr().out
    assert 'pending  id_json="pool000"' in out and 'reason=unknown' in out and 'timestamp=unknown' in out
    assert 'secret-title' not in out
    conn.execute('ALTER TABLE screen_attempts ADD COLUMN category TEXT')
    conn.execute('ALTER TABLE screen_attempts ADD COLUMN status INTEGER')
    conn.execute('ALTER TABLE screen_attempts ADD COLUMN completed_at TEXT')
    conn.execute("UPDATE screen_attempts SET state='secret-state',category='secret-category',"
                 "completed_at='secret-timestamp'")
    conn.commit()
    assert main(['--db', path, 'screen-status']) == 0
    out = capsys.readouterr().out
    assert 'secret-' not in out and 'unknown' in out
    conn.execute("UPDATE screen_attempts SET state='failed',category='http_error',status=429,"
                 "completed_at='2026-11-26T12:00:00Z'")
    conn.commit()
    assert main(['--db', path, 'screen-status']) == 0
    out = capsys.readouterr().out
    assert 'http_error_http_429' in out and '2026-11-26T12:00:00+00:00' in out
    conn.close()


def test_status_identifiers_are_lossless_and_terminal_safe(tmp_path, capsys):
    path = str(tmp_path / 'test.sqlite')
    conn = db.connect(path)
    db.migrate(conn)
    ids = ['pool1000', 'legacy-id', '"legacy-id"', '  café  ', 'line\nINJECT\x1b[31m\x7f']
    conn.execute('CREATE TABLE screen_attempts (pool_id TEXT, state TEXT)')
    for ident in ids:
        conn.execute("INSERT INTO pool (id,title,status) VALUES (?,'secret-title','new')", (ident,))
        conn.execute("INSERT INTO screen_attempts VALUES (?, 'pending')", (ident,))
    conn.commit()
    assert main(['--db', path, 'screen-status']) == 0
    out = capsys.readouterr().out
    for ident in ids:
        encoded = json.dumps(ident, ensure_ascii=True).replace('\x7f', r'\u007f')
        assert f'id_json={encoded}  reason=' in out and json.loads(encoded) == ident
    assert '\nINJECT' not in out and '\x1b' not in out and '\x7f' not in out
    assert 'secret-title' not in out
    conn.close()


def test_read_only_legacy_database_unchanged(tmp_path, capsys):
    path = tmp_path / 'legacy?#.sqlite'
    with sqlite3.connect(path) as conn:
        conn.execute('CREATE TABLE screen_attempts (pool_id TEXT, state TEXT)')
        conn.execute('INSERT INTO screen_attempts VALUES (?, ?)', ('unsafe\nID', 'pending'))
        schema_before = conn.execute('SELECT sql FROM sqlite_master ORDER BY name').fetchall()
        rows_before = conn.execute('SELECT * FROM screen_attempts').fetchall()
    before = hashlib.sha256(path.read_bytes()).digest()
    assert main(['--db', str(path), 'screen-status']) == 0
    out = capsys.readouterr().out
    assert json.dumps('unsafe\nID') in out
    assert 'Blocked unscored: unknown (legacy schema)' in out
    assert hashlib.sha256(path.read_bytes()).digest() == before
    with sqlite3.connect(path) as conn:
        assert conn.execute('SELECT sql FROM sqlite_master ORDER BY name').fetchall() == schema_before
        assert conn.execute('SELECT * FROM screen_attempts').fetchall() == rows_before


def test_existing_current_database_unchanged(tmp_path, capsys):
    path = tmp_path / 'current.sqlite'
    conn = db.connect(str(path))
    db.migrate(conn)
    conn.execute("INSERT INTO pool (id,title,status) VALUES ('pool000','title','new')")
    conn.execute('CREATE TABLE screen_attempts (pool_id TEXT, state TEXT)')
    conn.execute("INSERT INTO screen_attempts VALUES ('pool000', 'pending')")
    conn.commit()
    dump_before = list(conn.iterdump())
    conn.close()
    before = hashlib.sha256(path.read_bytes()).digest()
    assert main(['--db', str(path), 'screen-status']) == 0
    assert 'Blocked unscored: 1' in capsys.readouterr().out
    assert hashlib.sha256(path.read_bytes()).digest() == before
    with sqlite3.connect(path) as conn:
        assert list(conn.iterdump()) == dump_before


def test_missing_db_does_not_create_anything(tmp_path, capsys):
    path = tmp_path / 'absent' / 'test.sqlite'
    assert main(['--db', str(path), 'screen-status']) == 1
    assert 'Cannot open database read-only' in capsys.readouterr().err
    assert not path.parent.exists()


def test_empty_database_is_not_migrated(tmp_path, capsys):
    path = tmp_path / 'empty.sqlite'
    sqlite3.connect(path).close()
    before = hashlib.sha256(path.read_bytes()).digest()
    assert main(['--db', str(path), 'screen-status']) == 0
    assert '0 screening reservations' in capsys.readouterr().out
    assert hashlib.sha256(path.read_bytes()).digest() == before
    with sqlite3.connect(path) as conn:
        assert conn.execute('SELECT * FROM sqlite_master').fetchall() == []
