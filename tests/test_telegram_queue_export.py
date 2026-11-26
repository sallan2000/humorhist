"""Per-item manual export remains local, versioned, and reversible."""
import pytest

from humorhist import db
from humorhist import telegram as tg


@pytest.fixture
def conn(tmp_path):
    c = db.connect(str(tmp_path / 'queue.sqlite'))
    db.migrate(c)
    for did in ('one', 'two'):
        c.execute('INSERT INTO pool (id,title,status) VALUES (?,?,?)', (did, 'Title ' + did, 'drafted'))
        c.execute("INSERT INTO drafts (id,pool_id,short_code,brief_json,angles_json,status) "
                  "VALUES (?,?,?,'{}','{}','approved')", (did, did, did.upper()))
        c.execute('INSERT INTO queue (draft_id,published,post_copy,source_link) VALUES (?,0,?,?)',
                  (did, 'Copy ' + did, 'https://example.org/' + did))
    c.commit()
    yield c
    c.close()


def tap(conn, client, data):
    tg.handle_callback(conn, client, '123', {'callback_query': {'id': 'tap', 'data': data}})


def buttons(client):
    return [b for m in client.sent for row in (m.get('reply_markup') or {}).get('inline_keyboard', []) for b in row]


def find(client, prefix):
    return next(b['callback_data'] for b in buttons(client) if b['callback_data'].startswith(prefix))


def test_each_item_export_and_manual_mark_are_together_and_read_only(conn):
    client = tg.StubTelegram()
    tg.send_queue_list(conn, client, '123')
    for did in ('one', 'two'):
        assert any(any(b['callback_data'].startswith('export:' + did + ':') for b in row) and
                   any(b['callback_data'].startswith('posted:' + did + ':') for b in row)
                   for m in client.sent for row in (m.get('reply_markup') or {}).get('inline_keyboard', []))
    before = list(conn.iterdump())
    out = tg.StubTelegram()
    tap(conn, out, find(client, 'export:one:'))
    text = '\n'.join(m['text'] for m in out.sent)
    assert 'Copy one' in text and 'https://example.org/one' in text
    assert 'Copy two' not in text and 'Nothing was sent automatically' in text
    assert list(conn.iterdump()) == before
    assert not any('photo' in message for message in out.sent)
    tap(conn, out, find(out, 'posted:one:'))
    assert conn.execute("SELECT published FROM queue WHERE draft_id='one'").fetchone()[0] == 0
    confirm = find(out, 'postyes:one:')
    tap(conn, out, confirm)
    assert conn.execute("SELECT published FROM queue WHERE draft_id='one'").fetchone()[0] == 1
    tap(conn, out, confirm)
    assert 'expired' in out.sent[-1]['text'].lower()
    tap(conn, out, find(out, 'unpost:one:'))
    assert conn.execute("SELECT published FROM queue WHERE draft_id='one'").fetchone()[0] == 0


def test_long_export_has_button_on_last_chunk(conn):
    copy = '🌍' * 5000
    conn.execute("UPDATE queue SET post_copy=? WHERE draft_id='one'", (copy,))
    conn.commit()
    listing = tg.StubTelegram()
    tg.send_queue_list(conn, listing, '123')
    out = tg.StubTelegram()
    tap(conn, out, find(listing, 'export:one:'))
    assert len(out.sent) > 1
    assert all(tg._units(m['text']) <= 4096 for m in out.sent)
    assert copy in ''.join(m['text'] for m in out.sent)
    assert not any(m.get('reply_markup') for m in out.sent[:-1])
    assert find(out, 'posted:one:') in str(out.sent[-1])
    assert 'Title one (#ONE)' in out.sent[-1]['text']


def test_short_export_does_not_expose_image_path(conn):
    conn.execute("UPDATE queue SET image_path='/private/secret/image.png' WHERE draft_id='one'")
    conn.commit()
    listing = tg.StubTelegram()
    tg.send_queue_list(conn, listing, '123')
    out = tg.StubTelegram()
    tap(conn, out, find(listing, 'export:one:'))
    assert '/private/secret' not in repr(out.sent)
    assert 'image path recorded (availability not checked)' in out.sent[-1]['text']
    assert 'https://example.org/one' in out.sent[-1]['text']
    assert not any('document' in m for m in out.sent)


@pytest.mark.parametrize('image_path', ['/private/secret/path.png', None])
def test_long_url_export_document_is_complete_and_private(conn, image_path):
    link = 'https://example.org/' + '🌍' * 2000
    copy_link = 'https://copy.example/' + 'a' * 4000
    conn.execute("UPDATE queue SET image_path=?, source_link=?, post_copy=? WHERE draft_id='one'",
                 (image_path, link, 'Copy ' + copy_link))
    conn.commit()
    listing = tg.StubTelegram()
    tg.send_queue_list(conn, listing, '123')
    before = list(conn.iterdump())
    out = tg.StubTelegram()
    tap(conn, out, find(listing, 'export:one:'))
    assert list(conn.iterdump()) == before
    documents = [m for m in out.sent if 'document' in m]
    assert len(documents) == 1
    document = documents[0]['document'].decode('utf-8')
    assert link in document and copy_link in document
    assert '/private/secret/path.png' not in repr(out.sent)
    assert ('image path recorded (availability not checked)' if image_path else 'image none (optional)') in document
    assert tg._units(documents[0]['caption']) <= 1024
    messages = [m for m in out.sent if 'text' in m]
    assert all(tg._units(m['text']) <= 4096 for m in messages)
    assert 'Full URL in attached text document' in ''.join(m['text'] for m in messages)
    assert 'retrieve the full URL' not in repr(out.sent)
    assert find(out, 'posted:one:') in str(messages[-1])


def test_document_transport_multipart_and_redacted_failures(monkeypatch):
    import httpx

    requests = []
    def respond(request):
        requests.append(request)
        return httpx.Response(200, json={'ok': True, 'result': {'message_id': 8}})
    real_client = httpx.Client
    monkeypatch.setattr(tg.httpx, 'Client', lambda **kw: real_client(transport=httpx.MockTransport(respond), **kw))
    client = tg.TelegramClient(token='secret-token', max_retries=2)
    assert client.send_document('123', 'link 🌍'.encode(), 'Export')['message_id'] == 8
    assert len(requests) == 1
    body = requests[0].content
    assert b'name="chat_id"' in body and b'123' in body
    assert b'filename="queue-export.txt"' in body and 'link 🌍'.encode() in body

    def fail(request):
        requests.append(request)
        return httpx.Response(400, text='secret-token /private/secret/path.png')
    monkeypatch.setattr(tg.httpx, 'Client', lambda **kw: real_client(transport=httpx.MockTransport(fail), **kw))
    with pytest.raises(tg.TelegramError) as exc:
        client.send_document('123', b'payload', 'Export')
    assert len(requests) == 2
    assert 'secret-token' not in str(exc.value) and '/private/secret' not in str(exc.value)


def test_failed_document_does_not_offer_incomplete_export(conn):
    conn.execute("UPDATE queue SET source_link=? WHERE draft_id='one'", ('https://x/' + 'x' * 4000,))
    conn.commit()
    listing = tg.StubTelegram()
    tg.send_queue_list(conn, listing, '123')
    class Failing(tg.StubTelegram):
        def send_document(self, chat_id, document, caption):
            raise tg.TelegramError('upload failed')
    out = Failing()
    tap(conn, out, find(listing, 'export:one:'))
    assert 'Could not confirm' in out.sent[-1]['text']
    assert 'Check this chat' in out.sent[-1]['text']
    assert not buttons(out)


def test_url_shorter_than_limit_split_by_prefix_is_attached(conn):
    link = 'https://example.org/' + 'x' * 3550
    conn.execute("UPDATE queue SET post_copy=? WHERE draft_id='one'", ('x' * 3500 + link,))
    conn.commit()
    listing = tg.StubTelegram()
    tg.send_queue_list(conn, listing, '123')
    out = tg.StubTelegram()
    tap(conn, out, find(listing, 'export:one:'))
    docs = [m for m in out.sent if 'document' in m]
    assert len(docs) == 1 and link in docs[0]['document'].decode('utf-8')
    assert 'Full URL in attached text document' in ''.join(m.get('text', '') for m in out.sent)


def test_export_mark_button_expires_after_copy_changes(conn):
    listing = tg.StubTelegram()
    tg.send_queue_list(conn, listing, '123')
    out = tg.StubTelegram()
    tap(conn, out, find(listing, 'export:one:'))
    mark = find(out, 'posted:one:')
    conn.execute("UPDATE queue SET post_copy='new version' WHERE draft_id='one'")
    conn.commit()
    tap(conn, out, mark)
    assert 'expired' in out.sent[-1]['text'].lower()
    assert conn.execute("SELECT published FROM queue WHERE draft_id='one'").fetchone()[0] == 0


def test_queue_pagination_keeps_each_items_controls_with_its_text(conn):
    for i in range(15):
        did = f'extra{i:02}'
        conn.execute('INSERT INTO pool (id,title,status) VALUES (?,?,?)', (did, did, 'drafted'))
        conn.execute("INSERT INTO drafts (id,pool_id,short_code,brief_json,angles_json,status) "
                     "VALUES (?,?,?,'{}','{}','approved')", (did, did, did))
        conn.execute("INSERT INTO queue (draft_id,published,post_copy) VALUES (?,0,'copy')", (did,))
    conn.commit()
    listing = tg.StubTelegram()
    assert tg.send_queue_list(conn, listing, '123') == 17
    assert len(listing.sent) > 1
    for message in listing.sent:
        assert tg._units(message['text']) <= 4096
        for row in message['reply_markup']['inline_keyboard']:
            for button in row:
                assert len(button['callback_data'].encode()) <= 64
                if button['callback_data'].startswith('export:'):
                    did = button['callback_data'].split(':')[1]
                    assert f'#{did.upper() if did in ("one", "two") else did}' in message['text']
                    assert any(b['callback_data'].startswith(f'posted:{did}:') for b in row)


def test_old_stale_removed_and_missing_copy_exports_fail_closed(conn):
    listing = tg.StubTelegram()
    tg.send_queue_list(conn, listing, '123')
    old = find(listing, 'export:one:')
    conn.execute("UPDATE queue SET post_copy='changed' WHERE draft_id='one'")
    conn.commit()
    for callback in (old, 'export:one', 'export:one:legacy'):
        out = tg.StubTelegram()
        tap(conn, out, callback)
        assert 'expired' in out.sent[-1]['text'].lower()
        assert not buttons(out)
    listing = tg.StubTelegram()
    tg.send_queue_list(conn, listing, '123')
    fresh = find(listing, 'export:one:')
    conn.execute("DELETE FROM queue WHERE draft_id='one'")
    conn.commit()
    out = tg.StubTelegram()
    tap(conn, out, fresh)
    assert 'expired' in out.sent[-1]['text'].lower()
    conn.execute("UPDATE queue SET post_copy=NULL WHERE draft_id='two'")
    conn.commit()
    listing = tg.StubTelegram()
    tg.send_queue_list(conn, listing, '123')
    out = tg.StubTelegram()
    tap(conn, out, find(listing, 'export:two:'))
    assert 'no post copy' in out.sent[-1]['text'].lower()
    assert not buttons(out)
