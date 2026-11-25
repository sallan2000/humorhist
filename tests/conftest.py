"""Keep test credentials and network access isolated from the host."""

import socket

import pytest


@pytest.fixture(autouse=True)
def offline_test_environment(monkeypatch, tmp_path):
    """Reject IP sockets and DNS, including attempts caught by application code.

    HTTP mock transports bypass sockets altogether. Unix sockets remain available
    for local event-loop plumbing. The isolated HOME prevents OAuth fallback from
    reading a developer's real ~/.hermes/auth.json; credential tests can still
    populate their own temporary home or set a test credential explicitly.
    """
    monkeypatch.setenv("HOME", str(tmp_path))
    for name in (
        "HUMORHIST_LLM_API_KEY",
        "HUMORHIST_IMAGE_API_KEY",
        "OPENAI_API_KEY",
        "HUMORHIST_TELEGRAM_BOT_TOKEN",
        "HUMORHIST_TELEGRAM_CHAT_ID",
    ):
        monkeypatch.delenv(name, raising=False)

    attempts = []
    original_socket = socket.socket

    def deny(operation):
        attempts.append(operation)
        raise OSError(f"network access forbidden in tests: {operation}")

    class OfflineSocket(original_socket):
        def __init__(self, family=socket.AF_INET, *args, **kwargs):
            if family in (socket.AF_INET, socket.AF_INET6):
                deny(f"socket family {family}")
            super().__init__(family, *args, **kwargs)

    monkeypatch.setattr(socket, "socket", OfflineSocket)
    for name in ("getaddrinfo", "gethostbyname", "gethostbyname_ex", "gethostbyaddr", "getnameinfo"):
        monkeypatch.setattr(socket, name, lambda *args, _name=name, **kwargs: deny(_name))

    yield

    if attempts:
        pytest.fail("Network access attempted during test: " + ", ".join(attempts), pytrace=False)
