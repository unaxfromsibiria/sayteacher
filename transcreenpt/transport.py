import os
from dataclasses import dataclass


@dataclass(frozen=True)
class RedisSettings:
    """Connection parameters for the datatraffic Redis backend."""

    redis_url: str
    key_hex: str
    password: str


def redis_settings() -> RedisSettings:
    """Read datatraffic connection settings from environment variables.

    Required: CRYPTO_KEY (64-char hex AES-256-GCM key).
    Optional: REDIS_URL (default redis://localhost:6379), REDIS_PASSWORD (default empty).
    """
    return RedisSettings(
        redis_url=os.environ.get("REDIS_URL", "redis://localhost:6379"),
        key_hex=os.environ["CRYPTO_KEY"],
        password=os.environ.get("REDIS_PASSWORD", ""),
    )


def make_client(projects: tuple[str, ...], client_name: str = ""):
    """Create an AsyncClient watching all given projects (pydatatraffic 0.0.4 API).

    The returned client must be used as an async context manager::

        async with make_client(("out-audio",)) as c:
            frame = await c.recv()
            c.send("trans-text", "007", "translated text", b"")
    """
    from pydatatraffic.api import AsyncClient

    settings = redis_settings()
    return AsyncClient(
        redis_url=settings.redis_url,
        key_hex=settings.key_hex,
        client_name=client_name,
        password=settings.password,
        watch_projects=projects,
    )
