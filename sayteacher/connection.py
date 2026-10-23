import asyncio
import logging

try:
    from .common import MessageKindEnum
except ImportError:
    from common import MessageKindEnum

log = logging.getLogger("sayteacher")

LESSON_PROJECT = "lesson-dialogue"
READY_PROJECT = "ready-state"


class Connection:
    """Wraps one datatraffic AsyncClient watching both projects, routing frames by project.
    Accepts an injectable client for testing; when omitted, a real AsyncClient is built
    from environment variables via transport.make_client(). The client's background loops
    and the pump task start lazily on first use."""

    def __init__(self, client=None) -> None:
        if client is None:
            try:
                from .transport import make_client
            except ImportError:
                from transport import make_client
            client = make_client((LESSON_PROJECT, READY_PROJECT), "sayteacher")
        self.client = client
        self._lesson_inbox: asyncio.Queue = asyncio.Queue()
        self._ready_inbox: asyncio.Queue = asyncio.Queue()
        self._pump_task: asyncio.Task | None = None

    async def _ensure_started(self) -> None:
        """Start the client's background loops and the pump task on first use."""
        if self._pump_task is not None and not self._pump_task.done():
            return
        await self.client.__aenter__()
        self._pump_task = asyncio.create_task(self._pump())

    async def _pump(self) -> None:
        """Route received frames into per-project inboxes until cancelled or the client errors."""
        while True:
            try:
                frame = await self.client.recv()
            except asyncio.CancelledError:
                raise
            except Exception as err:
                log.warning("datatraffic recv failed, stopping pump: %s", err)
                return
            if frame.project == LESSON_PROJECT:
                self._lesson_inbox.put_nowait(frame)
            elif frame.project == READY_PROJECT:
                self._ready_inbox.put_nowait(frame)
            else:
                log.debug("frame from unknown project %r dropped", frame.project)

    async def recv_lesson(self):
        """Wait for the next frame on the lesson-dialogue project."""
        await self._ensure_started()
        return await self._lesson_inbox.get()

    async def recv_ready(self):
        """Wait for the next frame on the ready-state project."""
        await self._ensure_started()
        return await self._ready_inbox.get()

    async def drain_outbox(self, outbox: asyncio.Queue, name: str) -> tuple[bool, bool]:
        """Send pending teacher answers as datatraffic frames on the lesson-dialogue project.

        Each item is (kind byte, text, audio payload). The kind byte maps to a frame name:
        'a' → "a", 'd' → "d", 'n' → "n", 'q' → "q".
        Returns (send_ok, quit_delivered): send_ok is always True with datatraffic (no socket),
        quit_delivered is True once a QUIT marker was sent — the session must end after that."""
        await self._ensure_started()
        sent = 0
        quit_delivered = False
        while True:
            try:
                kind, reply_text, reply_audio = outbox.get_nowait()
            except asyncio.QueueEmpty:
                break

            frame_name = chr(kind)
            log.info(
                "sending answer for %s: name=%r text_len=%d audio_bytes=%d",
                name, frame_name, len(reply_text), len(reply_audio),
            )
            self.client.send(LESSON_PROJECT, frame_name, reply_text, bytes(reply_audio))
            sent += 1
            if kind == int(MessageKindEnum.QUIT):
                quit_delivered = True

        return True, quit_delivered

    async def close(self) -> None:
        """Stop the pump and close the client; safe to call twice."""
        if self._pump_task is not None and not self._pump_task.done():
            self._pump_task.cancel()
            try:
                await self._pump_task
            except BaseException as err:
                log.warning("After closing %s", err)
        self._pump_task = None
        await self.client.close()
