import asyncio
import logging

try:
    from .common import CLOSED
    from .common import MessageKindEnum
    from .transport import Transport
except ImportError:
    from common import CLOSED
    from common import MessageKindEnum
    from transport import Transport

log = logging.getLogger("transcreenpt")


def connect_addr(addr: str) -> tuple[str, int]:
    """Split a host:port string into (host, port); raise ValueError on bad input."""
    try:
        host, port_s = addr.rsplit(":", 1)
        return host, int(port_s)
    except (ValueError, IndexError):
        raise ValueError(f"transcreenpt: --connect must be host:port (got {addr!r})") from None


class Connection:
    """One TCP session with the outvoiceagent server (no greeting/auth handshake).

    Owns the socket pair and the frame pump task that reads whole length-prefixed frames
    into `inbox`, so callers can poll it with short timeouts without desyncing stream
    framing. Translation replies queued by the recognition worker are sent by a dedicated
    sender task that awaits the per-student outbox (see `send_loop`)."""

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

    async def connect(
        self, attempts: int = 8, timeout: float = 3.0, stop: "asyncio.Event | None" = None
    ) -> None:
        """Connect to the server.

        outvoiceagent does not perform a greeting/auth handshake and sends the first audio
        frame only after the capture delay, so we simply open the TCP connection without
        waiting for an immediate greeting. Retries on connection failures; raises ConnectionError
        once retries are exhausted (or immediately when `stop` is set)."""
        last_err = "no attempt"
        for attempt in range(attempts):
            if stop is not None and stop.is_set():
                break
            try:
                connect = asyncio.open_connection(self.host, self.port)
                reader, writer = await asyncio.wait_for(connect, timeout)
            except OSError as e:
                last_err = f"connect failed: {e}"
                if stop is not None and stop.is_set():
                    break
                await asyncio.sleep(0.5)
                continue

            log.info("connected to %s:%d", self.host, self.port)
            self.reader = reader
            self.writer = writer
            return

        raise ConnectionError(f"transcreenpt handshake failed after retries; last={last_err}")

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

    async def send_loop(self, outbox: "asyncio.Queue", name: str, stop_event: "asyncio.Event") -> None:
        """Send translation replies as typed frames until the session ends.

        Awaits `outbox.get()` so a reply leaves the moment recognition queues it, instead of
        waiting for the next read-loop iteration. Each item is (kind byte, text, audio
        payload); the kind becomes the first byte of the frame's audio field so the server can
        route it. Sets `stop_event` and returns when a QUIT marker is delivered or a send
        fails — the caller must close the session right after that."""
        while True:
            kind, reply_text, reply_audio = await outbox.get()
            log.info(
                "sending answer for %s: kind=%r text_len=%d audio_bytes=%d",
                name, chr(kind), len(reply_text), len(reply_audio),
            )
            frame_audio = bytes([kind]) + bytes(reply_audio)
            self.write_frame(self.transport.encode(bytes(16), reply_text, frame_audio))
            try:
                await self.writer.drain()
            except OSError as e:
                log.error("send to server failed (translation reply): %s", e)
                stop_event.set()
                return
            if kind == int(MessageKindEnum.QUIT):
                stop_event.set()
                return

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
