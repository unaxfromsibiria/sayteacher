import asyncio
import logging

try:  # `python -m sayteacher.connection` (package mode)
    from .common import CLOSED
    from .common import MessageKindEnum
    from .transport import Transport
except ImportError:  # flat run, e.g. `cd sayteacher && python main.py` or pytest
    from common import CLOSED
    from common import MessageKindEnum
    from transport import Transport

log = logging.getLogger("sayteacher")


def connect_addr(addr: str) -> tuple[str, int]:
    """Split a host:port string into (host, port); raise ValueError on bad input."""
    try:
        host, port_s = addr.rsplit(":", 1)
        return host, int(port_s)
    except (ValueError, IndexError):
        raise ValueError(f"sayteacher: --connect must be host:port (got {addr!r})") from None


class Connection:
    """One authenticated TCP session with the microagent server.

    Owns the socket pair and the frame pump task that reads whole length-prefixed frames
    into `inbox`, so callers can poll it with short timeouts without desyncing stream
    framing. Teacher answers queued by the LLM worker are drained from a per-student
    outbox before they leave this class."""

    def __init__(self, host: str, port: int, password: str, transport: Transport) -> None:
        self.host = host
        self.port = port
        self.password = password
        self.transport = transport

        self.reader: asyncio.StreamReader | None = None
        self.writer: asyncio.StreamWriter | None = None

        # Raw frames from the pump; CLOSED is enqueued once when the peer goes away.
        self.inbox: "asyncio.Queue" = asyncio.Queue()
        self._pump_task: "asyncio.Task | None" = None

    async def connect(self, attempts: int = 8, timeout: float = 3.0) -> None:
        """Connect and complete the greeting/auth handshake.

        Retries on connection failures and the server's 'busy' marker (sent while a previous
        session is still torn down); raises ConnectionError once retries are exhausted."""
        last_err = "no attempt"
        for attempt in range(attempts):
            try:
                reader, writer = await asyncio.wait_for(asyncio.open_connection(self.host, self.port), timeout)
            except OSError as e:
                last_err = f"connect failed: {e}"
                await asyncio.sleep(0.5)
                continue

            try:
                raw = await self._read_frame(reader, timeout=timeout)
                msg = self.transport.decode(raw)
            except (ConnectionError, ValueError, RuntimeError):
                await Connection._abort(writer)
                last_err = f"server rejected us with a non-message frame (busy), attempt={attempt + 1}"
                log.debug("%s", last_err)
                await asyncio.sleep(0.5)
                continue

            token = self.transport.auth_token(msg.id, self.password)
            reply_frame = self.transport.encode(msg.id, token, b"")
            Connection._write_raw(writer, reply_frame)
            try:
                await writer.drain()
            except OSError as e:
                last_err = f"auth send failed: {e}"
                continue

            log.info("handshake ok (greeting id=%s)", msg.id.hex())
            self.reader = reader
            self.writer = writer
            return

        raise ConnectionError(f"sayteacher handshake failed after retries; last={last_err}")

    def start_pump(self) -> "asyncio.Task":
        """Spawn the background frame pump feeding `inbox`."""
        assert self.reader is not None, "call connect() before start_pump()"
        self._pump_task = asyncio.create_task(self._frame_pump())
        return self._pump_task

    async def next_frame(self, timeout: float | None = None):
        """Next raw frame from `inbox`; raises asyncio.TimeoutError when it stays empty."""
        if timeout is not None:
            return await asyncio.wait_for(self.inbox.get(), timeout=timeout)
        return await self.inbox.get()

    def decode(self, raw: bytes):
        """Decode one raw frame into a Message; raises on malformed frames."""
        return self.transport.decode(raw)

    def last_error(self) -> int:
        """FFI error code from the most recent failing decode (see constants in Rust crate)."""
        return self.transport.last_error()

    def write_frame(self, data: bytes) -> None:
        """Write one length-prefixed frame; the caller drains afterwards."""
        assert self.writer is not None, "call connect() before writing"
        Connection._write_raw(self.writer, data)

    async def send(self, msg_id: bytes, text: str, audio: bytes) -> bool:
        """Encode and write one message frame; False when the socket write failed."""
        self.write_frame(self.transport.encode(msg_id, text, audio))
        try:
            await self.writer.drain()
        except OSError as e:
            log.error("send to server failed: %s", e)
            return False
        return True

    async def drain_outbox(self, outbox: "asyncio.Queue", name: str) -> tuple[bool, bool]:
        """Send pending teacher answers as typed frames.

        Each item is (kind byte, text, audio payload); the kind becomes the first byte of the
        frame's audio field so the client UI can route it to notes vs dialogue.
        Returns (send_ok, quit_delivered): False in send_ok means the socket write failed and
        the caller should stop looping; quit_delivered is True once a QUIT marker was sent —
        the session must be closed right after that."""
        sent = 0
        quit_delivered = False
        while True:
            try:
                kind, reply_text, reply_audio = outbox.get_nowait()
            except asyncio.QueueEmpty:
                break

            log.info(
                "sending answer for %s: kind=%r text_len=%d audio_bytes=%d",
                name, chr(kind), len(reply_text), len(reply_audio),
            )
            frame_audio = bytes([kind]) + bytes(reply_audio)
            self.write_frame(self.transport.encode(bytes(16), reply_text, frame_audio))
            sent += 1
            if kind == int(MessageKindEnum.QUIT):
                quit_delivered = True

        if not sent:
            return True, False

        try:
            await self.writer.drain()
        except OSError as e:
            log.error("send to server failed (llm answer): %s", e)
            return False, quit_delivered
        return True, quit_delivered

    async def close(self) -> None:
        """Stop the pump and shut down the socket; safe to call twice."""
        task, self._pump_task = self._pump_task, None
        if task is not None:
            task.cancel()
            try:
                await task
            except BaseException:  # noqa: BLE001 - pump may raise on close
                pass

        writer, self.writer = self.writer, None
        await Connection._abort(writer)

    async def _frame_pump(self) -> None:
        """Read whole frames into `inbox` so callers can poll with short timeouts.

        A frame read is never cancelled mid-way (it runs to completion or a real error),
        which keeps the stream framing in sync while another task awaits `inbox.get()`.
        CLOSED is enqueued once when the peer goes away."""
        while True:
            try:
                self.inbox.put_nowait(await self._read_frame(self.reader))
            except (ConnectionError, OSError):
                break
        self.inbox.put_nowait(CLOSED)

    async def _read_frame(self, reader: asyncio.StreamReader, timeout: float | None = None) -> bytes:
        if timeout is not None:
            return await asyncio.wait_for(self._read_whole(reader), timeout=timeout)
        return await self._read_whole(reader)

    @staticmethod
    async def _read_n(reader: asyncio.StreamReader, n: int) -> bytes:
        buf = b""
        while len(buf) < n:
            chunk = await reader.read(n - len(buf))
            if not chunk:
                raise ConnectionError("peer closed mid-frame")
            buf += chunk
        return buf

    @classmethod
    async def _read_whole(cls, reader: asyncio.StreamReader) -> bytes:
        head = await cls._read_n(reader, 4)
        length = int.from_bytes(head, "big")
        payload = b""
        while len(payload) < length:
            chunk = await reader.read(length - len(payload))
            if not chunk:
                raise ConnectionError("peer closed mid-frame")
            payload += chunk
        return payload

    @staticmethod
    def _write_raw(writer: asyncio.StreamWriter, data: bytes) -> None:
        writer.write(len(data).to_bytes(4, "big"))
        writer.write(data)

    @staticmethod
    async def _abort(writer: asyncio.StreamWriter | None) -> None:
        if writer is None:
            return
        try:
            writer.close()
            await writer.wait_closed()
        except Exception:  # noqa: BLE001 - best-effort teardown of a failed attempt
            pass
