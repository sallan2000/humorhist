"""Exercise guided Telegram paths with real bot dispatch and no network."""
import threading

import pytest

import humorhist.db as db
import humorhist.telegram as tg


@pytest.fixture
def conn(tmp_path):
    c = db.connect(str(tmp_path / 'guided.sqlite'))
    db.migrate(c)
    for n in range(1, 4):
        c.execute('INSERT INTO pool (id,title,status) VALUES (?,?,?)', (f'p{n}', f'Topic {n}', 'drafted'))
        c.execute("INSERT INTO drafts (id,pool_id,brief_json,angles_json,status,created_at) VALUES (?,?,'{}','{}','pending',?)", (f'd{n}', f'p{n}', f'2026-01-0{n}'))
    c.commit()
    return c


class Script(tg.StubTelegram):
    def __init__(self, steps, chat_id=123):
        super().__init__()
        self.chat_id = chat_id
        self.answer_texts = []
        self.steps = list(steps)
        self.i = 0
        self.saved = {}

    def answer_callback_query(self, callback_query_id, text=""):
        self.answer_texts.append(text)
        return super().answer_callback_query(callback_query_id, text)

    def get_updates(self, offset=0, timeout=0):
        if not self.steps:
            return []
        self.i += 1
        action, value = self.steps.pop(0)
        chat = {'id': self.chat_id, 'type': 'private'}
        if action == 'text':
            return [{'update_id': self.i, 'message': {'chat': chat, 'from': {'id': self.chat_id}, 'text': value}}]
        if action == 'reply':
            prompt = next(m for m in reversed(self.sent) if value[0] in m['text'])
            return [{'update_id': self.i, 'message': {'chat': chat, 'from': {'id': self.chat_id}, 'text': value[1], 'reply_to_message': {'message_id': prompt['message_id']}}}]
        if action == 'raw':
            data, mid = value, 1
        elif action == 'replay':
            data, mid = self.saved[value]
        else:
            for m in reversed(self.sent):
                matches = [b['callback_data'] for row in m.get('reply_markup', {}).get('inline_keyboard', []) for b in row if b['callback_data'].startswith(value)]
                if matches:
                    data, mid = matches[0], m['message_id']
                    break
            else:
                raise AssertionError(f'No button {value}: {self.sent}')
            if action == 'save':
                self.saved[value] = (data, mid)
        return [{'update_id': self.i, 'callback_query': {'id': str(self.i), 'data': data, 'from': {'id': self.chat_id}, 'message': {'chat': chat, 'message_id': mid}}}]


def test_guided_review_full_flow_and_replays(conn, monkeypatch):
    monkeypatch.setattr(tg, '_get_llm', lambda *a, **k: None)
    steps = [
        ('text', '/start'), ('tap', 'menu:review'), ('tap', 'review:normal'),
        ('tap', 'approve:d1:'), ('save', 'confirm:approve:d1:'), ('replay', 'confirm:approve:d1:'),
        ('reply', ('one-line joke', 'A joke')), ('save', 'skipnotes:d1'),
        ('replay', 'skipnotes:d1'), ('tap', 'nextfast:d1'),
        ('save', 'fastreject:d2:'), ('replay', 'fastreject:d2:'),
        ('tap', 'review:normal'), ('tap', 'reject:d3:'),
    ]
    stub = Script(steps)
    assert tg.run_review_bot(conn, stub, '123', max_iterations=len(steps) + 1) == 2
    rows = [tuple(r) for r in conn.execute('SELECT id,status,editor_line,editor_notes FROM drafts ORDER BY id')]
    assert rows == [('d1', 'approved', 'A joke', None), ('d2', 'rejected', None, None), ('d3', 'pending', None, None)]
    assert any('Notes prompt expired' in m['text'] for m in stub.sent)
    assert any('Confirm: reject' in m['text'] and 'Topic 3' in m['text'] for m in stub.sent)
    assert any('no confirmation' in str(m.get('reply_markup')) for m in stub.sent)
    assert not stub.steps


def test_find_navigation_is_read_only_and_steps_are_explicit(conn, monkeypatch):
    calls = []
    for name in ('harvest', 'screen', 'draft'):
        monkeypatch.setattr(tg, 'telegram_' + name, lambda *a, _name=name, **k: calls.append(_name))
    before = list(conn.iterdump())
    steps = [('text', '/start'), ('tap', 'menu:find'), ('tap', 'menu:home'), ('tap', 'menu:queue')]
    stub = Script(steps)
    tg.run_review_bot(conn, stub, '123', max_iterations=len(steps) + 1)
    assert calls == []
    assert list(conn.iterdump()) == before
    assert any('may incur charges' in m['text'] for m in stub.sent)
    for step in ('harvest', 'screen', 'draft'):
        s = Script([('text', '/start'), ('tap', 'menu:find'), ('tap', 'find:' + step)])
        tg.run_review_bot(conn, s, '123', max_iterations=4)
    assert calls == ['harvest', 'screen', 'draft']


def test_skip_joke_and_stale_prompt_cannot_skip_new_notes(conn, monkeypatch):
    monkeypatch.setattr(tg, '_get_llm', lambda *a, **k: None)
    steps = [('text', '/start'), ('tap', 'menu:review'), ('tap', 'review:normal'),
             ('tap', 'approve:d1:'), ('tap', 'confirm:approve:d1:'),
             ('save', 'skipjoke:d1'), ('replay', 'skipjoke:d1'),
             ('reply', ('Optionally reply', 'Note retained')), ('tap', 'next:d1')]
    stub = Script(steps)
    tg.run_review_bot(conn, stub, '123', max_iterations=len(steps) + 1)
    row = conn.execute("SELECT status,editor_line,editor_notes FROM drafts WHERE id='d1'").fetchone()
    assert tuple(row) == ('approved', None, 'Note retained')
    assert any('DRAFT Topic 2' in m['text'] for m in stub.sent)


def test_fast_next_stays_fast_and_old_normal_next_is_not_fast(conn):
    steps = [('text', '/reviewdraft'), ('save', 'next:d1'), ('tap', 'menu:review'),
             ('tap', 'review:fast'), ('tap', 'nextfast:d3'), ('replay', 'next:d1')]
    stub = Script(steps)
    tg.run_review_bot(conn, stub, '123', max_iterations=len(steps) + 1)
    cards = [m for m in stub.sent if m.get('text', '').startswith('===')]
    assert 'fastapprove:d1:' in str(cards[-2]['reply_markup'])
    assert 'approve:d2:' in str(cards[-1]['reply_markup'])
    assert 'fastapprove:' not in str(cards[-1]['reply_markup'])
    assert all(r[0] == 'pending' for r in conn.execute('SELECT status FROM drafts'))


@pytest.mark.parametrize("step", ["screen", "draft"])
def test_find_intents_one_use_legacy_and_fresh_menu(conn, monkeypatch, step):
    calls = []
    monkeypatch.setattr(tg, 'telegram_' + step, lambda *a, **k: calls.append(step))
    steps = [('tap', 'menu:find'), ('save', 'find:' + step),
             ('replay', 'find:' + step), ('raw', 'find:' + step),
             ('tap', 'menu:find'), ('tap', 'find:' + step)]
    stub = Script(steps)
    tg.run_review_bot(conn, stub, '123', max_iterations=len(steps) + 1)
    assert calls == [step, step]
    assert sum('expired' in x.lower() for x in stub.answer_texts) >= 2


def test_find_intent_bound_to_chat_and_action(conn, monkeypatch):
    calls = []
    monkeypatch.setattr(tg, 'telegram_screen', lambda *a, **k: calls.append('screen'))
    stub = tg.StubTelegram()
    tg._find_menu(stub, '123')
    data = stub.sent[-1]['reply_markup']['inline_keyboard'][1][0]['callback_data']
    nonce = data.rsplit(':', 1)[1]
    for chat, raw in [('456', data), ('123', 'find:draft:' + nonce)]:
        s = Script([('raw', raw)], chat_id=int(chat))
        tg.run_review_bot(conn, s, chat, max_iterations=2)
        assert any('expired' in text.lower() for text in s.answer_texts)
    assert tg._intents[nonce][:3] == ('123', '', 'screen')
    assert calls == []
    s = Script([('raw', data)])
    tg.run_review_bot(conn, s, '123', max_iterations=2)
    assert calls == ['screen']


def test_find_ack_failure_never_dispatches_and_consumes(conn, monkeypatch):
    calls = []
    monkeypatch.setattr(tg, 'telegram_screen', lambda *a, **k: calls.append(1))
    class FailAck(Script):
        def answer_callback_query(self, callback_query_id, text=''):
            if callback_query_id == '2':
                raise RuntimeError('ack failed')
            return super().answer_callback_query(callback_query_id, text)
    stub = FailAck([('tap', 'menu:find'), ('save', 'find:screen')])
    with pytest.raises(RuntimeError, match='ack failed'):
        tg.run_review_bot(conn, stub, '123', max_iterations=3)
    data, _ = stub.saved['find:screen']
    replay = Script([('raw', data)])
    tg.run_review_bot(conn, replay, '123', max_iterations=2)
    assert calls == []


@pytest.mark.parametrize('finish', ['skip_button', 'skip_text', 'notes'])
def test_fast_approval_prompt_and_completion_stay_fast(conn, monkeypatch, finish):
    calls = []
    monkeypatch.setattr(tg, '_get_llm', lambda *a, **k: object())
    import humorhist.copywriter as copywriter
    monkeypatch.setattr(copywriter, 'fill_post_copy', lambda *a, **k: calls.append(k['draft_id']))
    joke = ('tap', 'skipjoke:d1') if finish == 'skip_button' else ('reply', ('one-line joke', 'A joke'))
    notes = ('reply', ('Optionally reply', 'Notes')) if finish == 'notes' else ('reply', ('Optionally reply', '/skip'))
    steps = [('tap', 'menu:review'), ('tap', 'review:fast'), ('tap', 'fastapprove:d1:'), joke, notes, ('tap', 'nextfast:d1')]
    stub = Script(steps)
    tg.run_review_bot(conn, stub, '123', max_iterations=len(steps) + 1)
    prompt = next(m for m in stub.sent if 'one-line joke' in m['text'])
    assert 'Replying or skipping automatically' in prompt['text'] and 'may incur charges' in prompt['text']
    assert calls == ['d1']
    cards = [m for m in stub.sent if m.get('text', '').startswith('===')]
    assert 'fastapprove:d2:' in str(cards[-1]['reply_markup'])


def test_fast_defer_and_skip_notes_button_preserve_mode(conn, monkeypatch):
    monkeypatch.setattr(tg, '_get_llm', lambda *a, **k: None)
    steps = [('tap', 'menu:review'), ('tap', 'review:fast'), ('tap', 'fastlater:d1:'),
             ('tap', 'nextfast:d1'), ('tap', 'fastapprove:d2:'), ('tap', 'skipjoke:d2'),
             ('tap', 'skipnotes:d2'), ('tap', 'nextfast:d2')]
    stub = Script(steps)
    tg.run_review_bot(conn, stub, '123', max_iterations=len(steps) + 1)
    assert 'fastapprove:d3:' in str([m for m in stub.sent if m.get('text', '').startswith('===')][-1]['reply_markup'])


def test_new_find_menu_revokes_older_unused_buttons(conn, monkeypatch):
    calls = []
    monkeypatch.setattr(tg, 'telegram_screen', lambda *a, **k: calls.append(1))
    # Extract an unused button from an older menu.
    menu = tg.StubTelegram()
    tg._find_menu(menu, '123')
    old = menu.sent[-1]['reply_markup']['inline_keyboard'][1][0]['callback_data']
    steps = [('tap', 'menu:find'), ('raw', old), ('tap', 'find:screen')]
    stub = Script(steps)
    tg.run_review_bot(conn, stub, '123', max_iterations=4)
    assert calls == [1]
    assert any('expired' in text.lower() for text in stub.answer_texts)


def test_replay_after_real_background_screen_finishes(conn, monkeypatch):
    import humorhist.harvest.screen as screen
    calls = []
    completed = threading.Event()
    monkeypatch.setattr(tg, '_get_llm', lambda *a, **k: object())
    def fake_screen(worker, llm, **kwargs):
        calls.append('screen')
        return {'scored': 0, 'batches': 0, 'failed_batches': 0}
    monkeypatch.setattr(screen, 'screen_pool', fake_screen)
    class WaitForCompletion(Script):
        def send_message(self, chat_id, text, reply_markup=None):
            result = super().send_message(chat_id, text, reply_markup)
            if 'scored=' in text and 'failed_batches=' in text:
                completed.set()
            return result
        def get_updates(self, offset=0, timeout=0):
            if self.steps and self.steps[0][0] == 'replay':
                assert completed.wait(5), 'screen did not finish'
            return super().get_updates(offset, timeout)
    stub = WaitForCompletion([('tap', 'menu:find'), ('save', 'find:screen'), ('replay', 'find:screen')])
    tg.run_review_bot(conn, stub, '123', max_iterations=4)
    assert calls == ['screen']
    assert any('expired' in text.lower() for text in stub.answer_texts)
