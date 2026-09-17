import asyncio
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import click

LIB_PATH = Path(__file__).resolve().parent / "lib" / "libmicroagent_transport.so"

if not LIB_PATH.exists():
    sys.exit(
        f"sayteacher: {LIB_PATH} not found; build first with "
        "'cargo build --release -p microagent-transport'"
    )

try:  # `python -m sayteacher.main` (package mode)
    from .common import CLOSED
    from .common import IDLE_TIMEOUT_SECONDS
    from .common import strip_type_prefix
    from .connection import Connection
    from .connection import connect_addr
    from .service import LlmAnswerService
    from .service import RecognitionService
    from .service import TtsService
    from .transport import Transport
except ImportError:  # run flat, e.g. `cd sayteacher && python main.py` or pytest
    from common import CLOSED
    from common import IDLE_TIMEOUT_SECONDS
    from common import strip_type_prefix
    from connection import Connection
    from connection import connect_addr
    from service import LlmAnswerService
    from service import RecognitionService
    from service import TtsService
    from transport import Transport

log = logging.getLogger("sayteacher")


class SessionFinished(Exception):
    """Raised by dialogue_data_processing once the LLM's quit marker has been delivered.

    The outer reconnect loop treats this as a clean end of session (not a transient error),
    so it stops re-dialing after the final answer left the socket."""


def _is_lesson(msg) -> bool:
    """Lesson marker matches Rust's tools.rs heuristic: empty audio + non-empty text."""
    return not msg.audio and bool(msg.text)


async def _poll_frame(conn: Connection, timeout: float):
    """Next raw frame from `conn.inbox`; returns (None, True) when the poll timed out.

    A TimeoutError is swallowed here so callers can accumulate idle time without try/except."""
    try:
        return await conn.next_frame(timeout=timeout), False
    except asyncio.TimeoutError:
        return None, True


def _idle_check(idle_seconds: float) -> float | bool:
    """Add 1s to the idle counter; return False when IDLE_TIMEOUT_SECONDS is exceeded."""
    if idle_seconds + 1.0 >= IDLE_TIMEOUT_SECONDS:
        log.warning("no frame from server in %.0fs, giving up", IDLE_TIMEOUT_SECONDS)
        return False
    return idle_seconds + 1.0


async def echo_session(host: str, port: int, password: str) -> None:
    """Test client mirroring Rust tools.rs: receive frames and echo them back."""
    conn = Connection(host, port, password, Transport(str(LIB_PATH)))
    try:
        await conn.connect()
    except ConnectionError as e:
        log.error("handshake failed: %s", e)
        return

    idle_seconds = 0.0
    conn.start_pump()
    try:
        while True:
            item, timed_out = await _poll_frame(conn, timeout=1.0)
            if timed_out:
                result = _idle_check(idle_seconds)
                if result is False:
                    break
                idle_seconds = result
                continue

            idle_seconds = 0.0
            if item is CLOSED:
                log.info("peer closed connection")
                break

            try:
                msg = conn.decode(item)
            except Exception as e:
                log.warning(
                    "undecodable frame (%d bytes): %s (ffi_last_error=%d)",
                    len(item), e, conn.last_error(),
                )
                continue

            text_preview = (msg.text[:32] + "\u2026") if len(msg.text) > 33 else msg.text
            audio_len = len(msg.audio)
            log.info(
                "received id=%s text=%r audio_bytes=%d",
                msg.id.hex(), text_preview, audio_len,
            )

            reply_text = f"record {msg.id.hex()} length: {audio_len}"
            if not await conn.send(msg.id, reply_text, msg.audio):
                break
            log.info("echoed back (text=%r)", reply_text)
    finally:
        await conn.close()
        log.info("disconnected")


async def dialogue_data_processing(
    api: RecognitionService, name: str, host: str, port: int, password: str,
) -> None:
    """Connect once and route decoded frames into the recognition service under `name`.

    The connected student is identified by its connection (host:port), so every frame we
    handle here — lessons included — is stored/queued for that named student. Lessons
    (empty audio + non-empty text) fill `api.lessons[name]`; audio clips go into the
    per-student queue as SpeechClip objects. Teacher "write" answers produced by the
    LLM worker are drained from `api.outboxes[name]` and sent back as typed frames."""
    conn = Connection(host, port, password, Transport(str(LIB_PATH)))
    try:
        await conn.connect()
    except ConnectionError as e:
        log.error("handshake failed: %s", e)
        return

    outbox = api.outboxes.setdefault(name, asyncio.Queue())
    idle_seconds = 0.0
    conn.start_pump()
    try:
        while True:
            send_ok, quit_delivered = await conn.drain_outbox(outbox, name)
            if not send_ok:
                break
            if quit_delivered:
                # The LLM's final answer left the socket in this drain; end our side of the session.
                log.info("quit marker delivered to %s, closing connection", name)
                await conn.send(bytes(16), "lesson finished", b"")
                raise SessionFinished

            item, timed_out = await _poll_frame(conn, timeout=1.0)
            if timed_out:
                result = _idle_check(idle_seconds)
                if result is False and outbox.empty():
                    break
                if result is not False:
                    idle_seconds = result
                continue

            idle_seconds = 0.0
            if item is CLOSED:
                log.info("peer closed connection")
                break

            try:
                msg = conn.decode(item)
            except Exception as e:
                # Decode failed inside the library: nothing to hand to the service.
                log.warning(
                    "undecodable frame (%d bytes): %s (ffi_last_error=%d)",
                    len(item), e, conn.last_error(),
                )
                continue

            if _is_lesson(msg):
                # Route under the connected student's name: stored in api.lessons[name].
                await api.send(name, msg.id, msg.text, b"")
                log.info("Lesson for %s: %s", name, msg.text[:80])
                # Echo confirmation like the test client so the server stays responsive.
                dt = datetime.now().time().isoformat()[:5]
                if not await conn.send(msg.id, f"Lesson started at {dt}. Notes from teacher:", b""):
                    break
                continue

            audio_payload = strip_type_prefix(bytes(msg.audio))
            if not audio_payload:
                log.debug("empty frame id=%s text=%r", msg.id.hex(), msg.text)
                continue

            clip_seconds = (len(audio_payload) // 4) / 16_000.0
            await api.send(name, msg.id, msg.text, audio_payload)
            log.info(
                "queued speech for %s id=%s %d bytes ~%.2fs",
                name, msg.id.hex(), len(audio_payload), clip_seconds,
            )

            if not await conn.send(msg.id, "", b""):
                break
    finally:
        await conn.close()
        # Every exit path (lesson finished, idle timeout, peer closed, send failure) resets the
        # per-session state so a new lesson after reconnect starts with an empty dialogue.
        try:
            await api.reset_session(name)
        except Exception as e:  # noqa: BLE001 - cleanup must not mask the session result
            log.warning("reset session state failed for %s: %s", name, e)
        log.info("disconnected")


@click.command(name="sayteacher")
@click.option(
    "--connect", "addr", default="127.0.0.1:16050", show_default=True, help="host:port of the microagent server."
)
@click.option(
    "--password", default="", show_default=True, help="shared secret used for auth; empty matches a server with no password set."
)
@click.option(
    "--debug", is_flag=True, default=False, show_default=True,
    help="log at DEBUG level.",
)
@click.option(
    "--check", is_flag=True, help="one short round-trip and exit instead of looping forever (smoke test / CI)."
)
@click.option(
    "--model", "model_id", default="bond005/whisper-podlodka-turbo", show_default=True,
    help="Whisper model id for recognition; empty disables it (clips are drained, dialogue is not filled).",
)
@click.option(
    "--lang", default="English", show_default=True,
    help="Language used in the LLM prompt template.",
)
@click.option(
    "--llm-model", "llm_model", default="", show_default=True,
    help="Ollama model name for teacher answers; empty disables the LLM (dialogue not filled with teacher entries).",
)
@click.option(
    "--llm-api", "llm_api", default="", show_default=True,
    help="Ollama API base url (e.g. http://127.0.0.1:11434); empty disables the LLM.",
)
@click.option(
    "--speaker", default="en_101", show_default=True,
    help=(
        "Silero TTS voice id (en_37, 39, 76) or model bundle id (v3_en / v5_ru). Language is derived from --lang; "
        "empty disables speech output."
    ),
)
def main(
    addr: str, password: str, debug: bool, check: bool, model_id: str, lang: str,
    llm_model: str, llm_api: str, speaker: str,
) -> None:
    """Echo test client for the microagent server (mirrors Rust tools.rs behaviour)."""
    if debug:
        level = logging.DEBUG
    else:
        level = getattr(logging, os.environ.get("LOGLEVEL", "INFO").upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    try:
        host, port = connect_addr(addr)
    except ValueError as e:
        sys.exit(str(e))
    student_name = f"{host}:{port}"

    async def run() -> int:
        # Check mode is a smoke test / CI hook: one echo round-trip and exit.
        if check:
            await echo_session(host, port, password)
            return 0

        api = RecognitionService(student_name, model=model_id, lang=lang)
        llm_service = LlmAnswerService(api_url=llm_api, model=llm_model)
        tts_service = TtsService(lang=lang, speaker=speaker)
        process_task: asyncio.Task | None = None
        try:
            while True:
                if process_task is None or process_task.done():
                    process_task = asyncio.create_task(
                        api.process(student_name, llm_api=llm_service, tts=tts_service),
                    )

                try:
                    await dialogue_data_processing(api, student_name, host, port, password)
                except SessionFinished:
                    # The LLM ended the dialogue; its final answer was delivered. Stop re-dialing.
                    log.info("session finished by quit protocol, stopping reconnect loop")
                    break

                # Connection dropped without a clean quit: clear stale dialogue/queues so the
                # next session starts fresh (task 41).
                await api.reset_session(student_name)
                await asyncio.sleep(0.1)
        finally:
            # The dialogue session ended; drop stale state (dialogue entries, pending
            # queues/timers) so a future run starts fresh instead of carrying leftovers.
            await api.reset_session(student_name)
            if process_task is not None and not process_task.done():
                process_task.cancel()
                try:
                    await process_task
                except BaseException:  # noqa: BLE001
                    pass
    try:
        sys.exit(asyncio.run(run()))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
