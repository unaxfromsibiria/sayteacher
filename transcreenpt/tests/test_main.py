import asyncio
import os
import signal
import time
from unittest import mock

import main
import pytest
from connection import Connection


class _FakeApi:
    """Stands in for RecognitionService so data_processing runs without real models."""

    def __init__(self, parallel: int = 1) -> None:
        self.outboxes: dict = {}
        self.parallel = parallel
        self.reset_calls = 0

    async def reset_session(self, name: str) -> None:
        self.reset_calls += 1


class _FakeConn:
    """Stands in for Connection so a session can run and be torn down without a server."""

    instances: list["_FakeConn"] = []

    def __init__(self, host: str, port: int, password: str, transport) -> None:
        self.host = host
        self.port = port
        self.password = password
        self.transport = transport
        self.closed = False
        _FakeConn.instances.append(self)

    async def connect(self, attempts: int = 8, timeout: float = 3.0, stop=None) -> None:
        # Succeed immediately so data_processing enters the session loop.
        pass

    def start_pump(self):
        return asyncio.create_task(asyncio.sleep(3600))

    async def next_frame(self, timeout: float | None = None):
        await asyncio.Event().wait()  # block until the task is cancelled at shutdown

    async def send(self, msg_id: bytes, text: str, audio: bytes) -> bool:
        return True

    async def send_loop(self, outbox, name: str, stop_event: "asyncio.Event") -> None:
        while not stop_event.is_set():
            await asyncio.sleep(0.05)

    async def close(self) -> None:
        self.closed = True


@pytest.mark.asyncio
async def test_signal_handlers_install_and_remove():
    # install registers both SIGINT and SIGTERM; remove restores defaults cleanly enough to re-install.
    stop = asyncio.Event()
    installed = main._install_signal_handlers(stop)
    assert {sig for _, sig in installed} == {signal.SIGINT, signal.SIGTERM}
    main._remove_signal_handlers(installed)

    # Re-installing after removal must not raise (handlers were restored).
    stop2 = asyncio.Event()
    installed2 = main._install_signal_handlers(stop2)
    assert {sig for _, sig in installed2} == {signal.SIGINT, signal.SIGTERM}
    main._remove_signal_handlers(installed2)


@pytest.mark.asyncio
async def test_sigterm_sets_stop():
    # Sending SIGTERM to ourselves must set the stop event through the registered handler.
    stop = asyncio.Event()
    installed = main._install_signal_handlers(stop)
    try:
        os.kill(os.getpid(), signal.SIGTERM)
        await asyncio.wait_for(stop.wait(), timeout=2.0)
        assert stop.is_set()
    finally:
        main._remove_signal_handlers(installed)


@pytest.mark.asyncio
async def test_connect_bails_when_stop_pre_set():
    # A pre-set stop makes connect give up before even attempting, so no retry sleep happens.
    stop = asyncio.Event()
    stop.set()
    conn = Connection("127.0.0.1", 1, "", mock.Mock())
    t0 = time.monotonic()
    with pytest.raises(ConnectionError) as ei:
        await conn.connect(attempts=8, timeout=3.0, stop=stop)
    elapsed = time.monotonic() - t0
    assert "no attempt" in str(ei.value)  # bailed before the first open_connection
    assert elapsed < 0.5


@pytest.mark.asyncio
async def test_data_processing_stops_on_shutdown():
    # Setting `stop` mid-session tears the session down promptly and resets state once.
    _FakeConn.instances.clear()
    api = _FakeApi()
    stop = asyncio.Event()

    with mock.patch.object(main, "Connection", _FakeConn):
        task = asyncio.create_task(
            main.data_processing(api, "127.0.0.1:16050", "127.0.0.1", 16050, "", stop)
        )
        await asyncio.sleep(0.1)  # let it enter the session loop
        assert not task.done()

        stop.set()
        await asyncio.wait_for(task, timeout=3.0)

    conn = _FakeConn.instances[-1]
    assert conn.closed is True
    assert api.reset_calls == 1


@pytest.mark.asyncio
async def test_data_processing_creates_parallel_processor_tasks():
    # With parallel=2 the session must spawn two processor coroutines on the same frame queue.
    _FakeConn.instances.clear()
    api = _FakeApi(parallel=2)
    stop = asyncio.Event()

    calls: list[str] = []

    async def fake_processor(frame_q, api_, name, stop_event):
        calls.append(name)
        await asyncio.Event().wait()  # run until cancelled at shutdown

    with mock.patch.object(main, "Connection", _FakeConn), \
            mock.patch.object(main, "_processor_task", new=fake_processor):
        task = asyncio.create_task(
            main.data_processing(api, "127.0.0.1:16050", "127.0.0.1", 16050, "", stop)
        )
        await asyncio.sleep(0.1)  # let it enter the session loop and spawn processors
        assert len(calls) == 2

        stop.set()
        await asyncio.wait_for(task, timeout=3.0)
