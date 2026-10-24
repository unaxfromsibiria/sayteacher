import asyncio
import logging

log = logging.getLogger("transcreenpt")

AUDIO_PROJECT = "out-audio"
TEXT_PROJECT = "trans-text"


class Connection:
    """Wraps one datatraffic AsyncClient watching out-audio, publishing to trans-text.

    Accepts an injectable client for testing; when omitted, a real AsyncClient is built
    from environment variables via transport.make_client(). The client's background loops
    and the pump task start lazily on first use."""

    def __init__(self, client=None) -> None:
        if client is None:
            try:
                from .transport import make_client
            except ImportError:
                from transport import make_client
            client = make_client((AUDIO_PROJECT,), "transcreenpt")
        self.client = client
        self._audio_inbox: asyncio.Queue = asyncio.Queue()
        self._pump_task: asyncio.Task | None = None

    async def _ensure_started(self) -> None:
        """Start the client's background loops and the pump task on first use."""
        if self._pump_task is not None and not self._pump_task.done():
            return
        await self.client.__aenter__()
        self._pump_task = asyncio.create_task(self._pump())

    async def _pump(self) -> None:
        """Route received frames into the audio inbox until cancelled or the client errors."""
        while True:
            try:
                frame = await self.client.recv()
            except asyncio.CancelledError:
                raise
            except Exception as err:
                log.warning("datatraffic recv failed, stopping pump: %s", err)
                return
            self._audio_inbox.put_nowait(frame)

    async def recv_audio(self, timeout: float | None = None):
        """Wait for the next audio frame from outvoiceagent.

        Returns a ReceivedFile with .name (counter string), .data (f32 LE mono samples).
        Raises asyncio.TimeoutError when `timeout` expires without a frame."""
        await self._ensure_started()
        if timeout is not None:
            return await asyncio.wait_for(self._audio_inbox.get(), timeout=timeout)
        return await self._audio_inbox.get()

    async def send_group(self, group_id: int, text: str) -> None:
        """Send a translation result for the given group to trans-text.

        The frame name is the zero-padded group counter (e.g. "007"); an empty note clears
        the client display."""
        await self._ensure_started()
        name = f"{group_id:03d}"
        log.info("sending translation for group %s (%d chars)", name, len(text))
        self.client.send(TEXT_PROJECT, name, text, b"")

    async def close(self) -> None:
        """Stop the pump and close the client; safe to call twice."""
        if self._pump_task is not None and not self._pump_task.done():
            self._pump_task.cancel()
            try:
                await self._pump_task
            except BaseException:  # noqa: BLE001 - pump may raise on close
                pass
        self._pump_task = None
        await self.client.close()
