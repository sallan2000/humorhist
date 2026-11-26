"""Offline, one-at-a-time recovery; follow the safety steps in README first."""

import json
import sqlite3
from pathlib import Path


def main() -> int:
    path = input('Database path: ')
    encoded = input('Copy full id_json value from screen-status: ')
    try:
        pool_id = json.loads(encoded)
    except (ValueError, RecursionError):
        print('Invalid JSON; no deletion.')
        return 1
    if not isinstance(pool_id, str):
        print('ID must be a JSON string; no deletion.')
        return 1
    # Unpaired surrogates cannot be bound as SQLite text.
    try:
        pool_id.encode('utf-8')
    except UnicodeEncodeError:
        print('ID is not valid UTF-8 text; no deletion.')
        return 1
    print('Reviewed id_json:', json.dumps(pool_id, ensure_ascii=True).replace('\x7f', r'\u007f'))
    if input('Type DELETE to confirm this exact ID: ') != 'DELETE':
        print('Cancelled; no deletion.')
        return 1
    path = Path(path).expanduser().resolve(strict=True)
    with sqlite3.connect(path.as_uri() + '?mode=rw', uri=True) as conn:
        conn.execute('BEGIN IMMEDIATE')
        rows = conn.execute(
            "SELECT state FROM screen_attempts WHERE typeof(pool_id) = 'text' AND pool_id COLLATE BINARY = ?",
            (pool_id,),
        ).fetchall()
        if len(rows) != 1 or rows[0][0] != 'pending':
            print('Expected exactly one pending reservation; no deletion.')
            return 1
        print('Current state: pending')
        result = conn.execute(
            "DELETE FROM screen_attempts WHERE typeof(pool_id) = 'text' AND pool_id COLLATE BINARY = ? "
            "AND state COLLATE BINARY = 'pending'", (pool_id,),
        )
        if result.rowcount != 1:
            raise RuntimeError('Unexpected deletion count; transaction rolled back')
    print('Deleted: 1')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
