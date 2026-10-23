"""Tests for the datatraffic client factory."""

import pytest

from sayteacher.transport import RedisSettings
from sayteacher.transport import make_client
from sayteacher.transport import redis_settings


def test_redis_settings_reads_env(monkeypatch):
    monkeypatch.setenv("REDIS_URL", "redis://example:6380")
    monkeypatch.setenv("CRYPTO_KEY", "ab" * 32)
    monkeypatch.setenv("REDIS_PASSWORD", "secret")

    s = redis_settings()
    assert isinstance(s, RedisSettings)
    assert s.redis_url == "redis://example:6380"
    assert s.key_hex == "ab" * 32
    assert s.password == "secret"


def test_redis_settings_defaults(monkeypatch):
    monkeypatch.setenv("CRYPTO_KEY", "cd" * 32)
    monkeypatch.delenv("REDIS_URL", raising=False)
    monkeypatch.delenv("REDIS_PASSWORD", raising=False)

    s = redis_settings()
    assert s.redis_url == "redis://localhost:6379"
    assert s.key_hex == "cd" * 32
    assert s.password == ""


def test_redis_settings_requires_crypto_key(monkeypatch):
    monkeypatch.delenv("CRYPTO_KEY", raising=False)

    with pytest.raises(KeyError, match="CRYPTO_KEY"):
        redis_settings()


def test_make_client_returns_async_client(monkeypatch):
    """make_client returns an AsyncClient watching all given projects (0.0.4 API)."""
    from unittest.mock import MagicMock
    from unittest.mock import patch

    monkeypatch.setenv("REDIS_URL", "redis://localhost:6379")
    monkeypatch.setenv("CRYPTO_KEY", "ef" * 32)
    monkeypatch.delenv("REDIS_PASSWORD", raising=False)

    with patch("pydatatraffic.api.AsyncClient") as mock_cls:
        client = make_client(("lesson-dialogue", "ready-state"), "test-client")
        assert isinstance(client, MagicMock)
        # Verify it was constructed with the right parameters.
        _, kwargs = mock_cls.call_args
        assert kwargs["redis_url"] == "redis://localhost:6379"
        assert kwargs["key_hex"] == "ef" * 32
        assert kwargs["client_name"] == "test-client"
        assert kwargs["watch_projects"] == ("lesson-dialogue", "ready-state")
        # 0.0.4 dropped the send_to parameter: project is passed per send() call.
        assert "send_to" not in kwargs
