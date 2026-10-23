"""Tests for the Connection wrapper over one datatraffic AsyncClient (0.0.4 API)."""

import asyncio
from collections import deque

import pytest

from sayteacher.common import MessageKindEnum
from sayteacher.connection import LESSON_PROJECT
from sayteacher.connection import READY_PROJECT
from sayteacher.connection import Connection


class FakeClient:
    """Minimal fake matching AsyncClient 0.0.4's send/recv/close/__aenter__ interface."""

    def __init__(self) -> None:
        self.sent = []  # list of (project, name, note, data) tuples
        self._inbox = deque()  # frames to be received
        self.closed = False
        self.entered = False

    async def __aenter__(self):
        self.entered = True
        return self

    def send(self, project: str, name: str, note: str = "", data: bytes = b"") -> None:
        self.sent.append((project, name, note, data))

    async def recv(self):
        """Block until a frame is available; raise if none and closed."""
        while not self._inbox:
            if self.closed:
                raise ConnectionError("client closed")
            await asyncio.sleep(0.01)
        return self._inbox.popleft()

    def put_frame(self, project: str, name: str, note: str = "", data: bytes = b"") -> None:
        """Test helper: enqueue a frame for recv()."""
        from pydatatraffic.api import ReceivedFile
        self._inbox.append(ReceivedFile(project=project, name=name, note=note, data=data))

    async def close(self) -> None:
        self.closed = True


def _make_conn() -> tuple[Connection, FakeClient]:
    client = FakeClient()
    conn = Connection(client=client)
    return conn, client


def test_drain_outbox_sends_typed_frames():
    """Outbox items are sent on lesson-dialogue with kind byte mapped to frame name."""

    async def scenario():
        conn, client = _make_conn()
        outbox: asyncio.Queue = asyncio.Queue()
        outbox.put_nowait((int(MessageKindEnum.NOTE), "hi!", b""))
        audio_payload = b"\x00\x00\x80?\x00\x00\x00@\x00\x00\x00@\x00\x00\x00@"
        outbox.put_nowait((int(MessageKindEnum.DIALOGUE), "you: hi\nteacher: hello\n", audio_payload))

        result = await conn.drain_outbox(outbox, "student")
        assert result == (True, False)
        assert outbox.empty()

        # Verify frames were sent on the lesson project with correct names.
        assert client.sent[0] == (LESSON_PROJECT, "n", "hi!", b"")
        assert client.sent[1][0] == LESSON_PROJECT
        assert client.sent[1][1] == "d"
        assert client.sent[1][2] == "you: hi\nteacher: hello\n"
        assert client.sent[1][3] == audio_payload

    asyncio.run(scenario())


def test_drain_outbox_reports_quit_marker():
    """A 'q' frame in the outbox must surface as quit_delivered=True."""

    async def scenario():
        conn, client = _make_conn()
        outbox: asyncio.Queue = asyncio.Queue()
        outbox.put_nowait((int(MessageKindEnum.DIALOGUE), "you: bye\nteacher: goodbye\n", b""))
        outbox.put_nowait((int(MessageKindEnum.QUIT), "", b""))

        send_ok, quit_delivered = await conn.drain_outbox(outbox, "student")
        assert (send_ok, quit_delivered) == (True, True)
        # Both frames sent: dialogue then quit.
        assert client.sent[0][1] == "d"
        assert client.sent[1][1] == "q"

    asyncio.run(scenario())


def test_drain_outbox_empty_returns_true_without_sending():
    """An empty outbox returns (True, False) and sends nothing."""

    async def scenario():
        conn, client = _make_conn()
        result = await conn.drain_outbox(asyncio.Queue(), "student")
        assert result == (True, False)
        assert client.sent == []

    asyncio.run(scenario())


def test_pump_routes_frames_by_project():
    """Frames are routed to the matching inbox by their project field."""

    async def scenario():
        conn, client = _make_conn()
        client.put_frame(LESSON_PROJECT, "a", "", b"\x01\x02")
        client.put_frame(READY_PROJECT, "greeting", "lesson text", b"")
        client.put_frame("other-project", "x", "", b"")

        lesson = await asyncio.wait_for(conn.recv_lesson(), timeout=1.0)
        assert (lesson.project, lesson.name, lesson.data) == (LESSON_PROJECT, "a", b"\x01\x02")

        ready = await asyncio.wait_for(conn.recv_ready(), timeout=1.0)
        assert (ready.project, ready.name, ready.note) == (READY_PROJECT, "greeting", "lesson text")

        # The unknown-project frame must not leak into either inbox.
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(conn.recv_lesson(), timeout=0.1)
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(conn.recv_ready(), timeout=0.1)

    asyncio.run(scenario())


def test_close_stops_pump_and_closes_client():
    """close() must cancel the pump task and close the client."""

    async def scenario():
        conn, client = _make_conn()
        # Start the pump via a recv that will time out (no frames available).
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(conn.recv_ready(), timeout=0.1)
        assert client.entered is True
        pump_task = conn._pump_task
        assert not pump_task.done()

        await conn.close()
        assert client.closed is True
        assert pump_task.cancelled()

    asyncio.run(scenario())


def test_close_is_idempotent():
    """Calling close() twice must not raise."""

    async def scenario():
        conn, _ = _make_conn()
        await conn.close()
        await conn.close()  # second call must be a no-op

    asyncio.run(scenario())


def test_connection_accepts_injected_client():
    """DI: passing a client explicitly skips make_client()."""
    client = FakeClient()
    conn = Connection(client=client)
    assert conn.client is client
