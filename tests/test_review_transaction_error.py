import sqlite3

import pytest

from humorhist.review import _transaction


def test_transaction_preserves_original_trigger_rollback_error():
    conn = sqlite3.connect(":memory:")
    conn.executescript("""
        CREATE TABLE example (value INTEGER);
        CREATE TRIGGER fail BEFORE INSERT ON example
        BEGIN SELECT RAISE(ROLLBACK, 'original failure'); END;
    """)
    with pytest.raises(sqlite3.IntegrityError, match="original failure"):
        with _transaction(conn):
            conn.execute("INSERT INTO example VALUES (1)")
    assert not conn.in_transaction
