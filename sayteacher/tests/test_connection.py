import asyncio
import logging
import struct
from pathlib import Path

import pytest

from sayteacher.common import MessageKindEnum
from sayteacher.connection import Connection
from sayteacher.connection import connect_addr
from sayteacher.transport import Transport

LIB_PATH = Path(__file__).resolve().parent.parent / "lib" / "libmicroagent_transport.so"


class _FakeWriter:
    def __init__(self) -> None:
        self.chunks = bytearray()
        self.closed = False

    def write(self, data: bytes) -> None:
        self.chunks += data

    async def drain(self) -> bool:
        return True

    def close(self) -> None:
        self.closed = True

    async def wait_closed(self) -> None:
        pass

    @property
    def buffer(self) -> bytes:
        return bytes(self.chunks)


class _FakeReader:
    def __init__(self, data: bytes) -> None:
        self._buf = bytearray(data)

    async def read(self, n: int) -> bytes:
        out = self._buf[:n]
        del self._buf[:n]
        return bytes(out)


def _frame(payload: bytes) -> bytes:
    return len(payload).to_bytes(4, "big") + payload


def test_connect_addr_parses_host_and_port():
    host, port = connect_addr("127.0.0.1:16050")
    assert (host, port) == ("127.0.0.1", 16050)


def test_connect_addr_rejects_missing_port():
    try:
        connect_addr("localhost")
    except ValueError as exc:
        assert "host:port" in str(exc)
    else:  # pragma: no cover - guard against accidental pass-through.
        raise AssertionError("expected ValueError for addr without a port")


def test_write_raw_is_big_endian_length_prefixed():
    writer = _FakeWriter()
    Connection._write_raw(writer, b"payload-bytes")
    assert writer.buffer == len(b"payload-bytes").to_bytes(4, "big") + b"payload-bytes"


def test_read_whole_roundtrips_a_written_frame():
    reader = _FakeReader(_frame(b"hello from frame"))
    assert asyncio.run(Connection._read_whole(reader)) == b"hello from frame"


def test_read_whole_handles_multiple_chunk_reads():
    # The declared payload length is 10 but each underlying read returns at most 3 bytes,
    # so the helper must keep reading until it has exactly the full payload.
    payload = b"x" * 10

    class _ChunkedReader(_FakeReader):
        async def read(self, n: int) -> bytes:
            return await super().read(min(n, 3))

    reader = _ChunkedReader(_frame(payload))
    assert asyncio.run(Connection._read_whole(reader)) == payload


def test_read_whole_raises_when_peer_closes_mid_frame():
    # Declared length is 8 but only 4 bytes are available before EOF.
    reader = _FakeReader((8).to_bytes(4, "big") + b"only-4")
    with pytest.raises(ConnectionError, match="peer closed mid-frame"):
        asyncio.run(Connection._read_whole(reader))


def test_send_encodes_and_writes_one_frame():
    transport = Transport(str(LIB_PATH))
    conn = Connection("h", 1, "", transport)

    writer = _FakeWriter()
    conn.writer = writer
    uid = bytes(range(16))
    assert asyncio.run(conn.send(uid, "record x length: 4", b"abcd")) is True

    raw = writer.buffer[4:]  # strip the 4-byte frame length prefix
    msg = transport.decode(raw)
    assert msg.id == uid
    assert msg.text == "record x length: 4"
    assert msg.audio == b"abcd"


def test_send_returns_false_when_socket_write_fails(caplog):
    class _BrokenWriter(_FakeWriter):
        async def drain(self) -> bool:
            raise OSError("broken pipe")

    transport = Transport(str(LIB_PATH))
    conn = Connection("h", 1, "", transport)
    conn.writer = _BrokenWriter()

    with caplog.at_level(logging.ERROR):
        assert asyncio.run(conn.send(bytes(16), "x", b"")) is False


def test_drain_outbox_sends_typed_frames():
    transport = Transport(str(LIB_PATH))
    conn = Connection("h", 1, "", transport)

    writer = _FakeWriter()
    conn.writer = writer
    outbox: asyncio.Queue = asyncio.Queue()
    outbox.put_nowait((int(MessageKindEnum.NOTE), "hi!", b""))
    audio_payload = struct.pack("<f", 0.25) * 4
    outbox.put_nowait((int(MessageKindEnum.DIALOGUE), "you: hi\nteacher: hello\n", audio_payload))

    assert asyncio.run(conn.drain_outbox(outbox, "student")) == (True, False)
    assert outbox.empty()

    # Two frames back-to-back; the second must carry the 'd' kind as its first audio byte.
    data = writer.buffer
    first_len = int.from_bytes(data[:4], "big")
    rest = data[4 + first_len:]

    msg2 = transport.decode(rest[4:])
    assert bytes(msg2.audio)[0] == ord("d")
    assert bytes(msg2.audio)[1:] == audio_payload


def test_drain_outbox_reports_quit_marker():
    """A 'q' frame in the outbox must surface as quit_delivered=True after its drain."""
    transport = Transport(str(LIB_PATH))
    conn = Connection("h", 1, "", transport)

    writer = _FakeWriter()
    conn.writer = writer
    outbox: asyncio.Queue = asyncio.Queue()
    # Final answer first (delivered before quit), then the quit marker.
    outbox.put_nowait((int(MessageKindEnum.DIALOGUE), "you: bye\nteacher: goodbye\n", b""))
    outbox.put_nowait((int(MessageKindEnum.QUIT), "", b""))

    send_ok, quit_delivered = asyncio.run(conn.drain_outbox(outbox, "student"))
    assert (send_ok, quit_delivered) == (True, True)


def test_drain_outbox_empty_returns_true_without_writing():
    transport = Transport(str(LIB_PATH))
    conn = Connection("h", 1, "", transport)

    writer = _FakeWriter()
    conn.writer = writer
    assert asyncio.run(conn.drain_outbox(asyncio.Queue(), "student")) == (True, False)
    assert writer.buffer == b""


def test_abort_closes_writer_quietly():
    async def scenario(writer: _FakeWriter) -> bool:
        await Connection._abort(writer)
        return writer.closed

    writer = _FakeWriter()
    assert asyncio.run(scenario(writer)) is True
    # A None writer must be a no-op, not an exception.
    assert asyncio.run(Connection._abort(None)) is None
