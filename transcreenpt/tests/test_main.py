import asyncio
import os
import signal
from unittest import mock

import main
import pytest


class _FakeApi:
    """Stands in for RecognitionService so data_processing runs without real models."""

    def __init__(self, parallel: int = 1) -> None:
        self.outboxes: dict = {}
        self.parallel = parallel
        self.reset_calls = 0

    async def reset_session(self, name: str) -> None:
        self.reset_calls += 1


class _FakeFrame:
    """Stands in for a datatraffic ReceivedFile."""

    def __init__(self, name: str = "0", data: bytes = b"\x00" * 64) -> None:
        self.name = name
        self.data = data


class _FakeConn:
    """Stands in for Connection so a session can run and be torn down without Redis."""

    instances: list["_FakeConn"] = []

    def __init__(self, client=None) -> None:
        self.client = client
        self.closed = False
        self._frames: asyncio.Queue = asyncio.Queue()
        _FakeConn.instances.append(self)

    async def recv_audio(self, timeout: float | None = None):
        # Block until a frame is pushed or the task is cancelled at shutdown.
        return await self._frames.get()

    async def send_group(self, group_id: int, text: str) -> None:
        pass

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
async def test_data_processing_stops_on_shutdown():
    # Setting `stop` mid-session tears the session down promptly and resets state once.
    _FakeConn.instances.clear()
    api = _FakeApi()
    stop = asyncio.Event()

    with mock.patch.object(main, "Connection", _FakeConn):
        task = asyncio.create_task(
            main.data_processing(api, "transcreenpt", stop)
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
            main.data_processing(api, "transcreenpt", stop)
        )
        await asyncio.sleep(0.1)  # let it enter the session loop and spawn processors
        assert len(calls) == 2

        stop.set()
        await asyncio.wait_for(task, timeout=3.0)


@pytest.mark.asyncio
async def test_reader_task_processes_frames():
    # The reader task receives frames from conn.recv_audio and enqueues them into frame_q.
    _FakeConn.instances.clear()
    frame_q: asyncio.Queue = asyncio.Queue(maxsize=10)

    with mock.patch.object(main, "Connection", _FakeConn):
        conn = _FakeConn()
        # Push a frame so the reader can pick it up.
        await conn._frames.put(_FakeFrame("5", b"\x01" * 32))

        stop_event = asyncio.Event()
        reader_task = asyncio.create_task(main._reader_task(conn, frame_q, stop_event))
        # Wait for the frame to be enqueued.
        frame = await asyncio.wait_for(frame_q.get(), timeout=1.0)
        assert frame.name == "5"
        assert bytes(frame.data) == b"\x01" * 32

        stop_event.set()
        reader_task.cancel()
        try:
            await reader_task
        except asyncio.CancelledError:
            pass


@pytest.mark.asyncio
async def test_send_loop_sends_groups():
    # The send loop drains the outbox and calls conn.send_group for each (group_id, text).
    _FakeConn.instances.clear()
    sent: list[tuple[int, str]] = []

    class _RecordingConn(_FakeConn):
        async def send_group(self, group_id: int, text: str) -> None:
            sent.append((group_id, text))

    conn = _RecordingConn()
    outbox: asyncio.Queue = asyncio.Queue()
    await outbox.put((1, "hello"))
    await outbox.put((2, "world"))

    stop_event = asyncio.Event()
    sender_task = asyncio.create_task(main._send_loop(conn, outbox, "test", stop_event))
    # Wait for both items to be sent.
    await asyncio.sleep(0.1)
    assert (1, "hello") in sent
    assert (2, "world") in sent

    stop_event.set()
    sender_task.cancel()
    try:
        await sender_task
    except asyncio.CancelledError:
        pass
