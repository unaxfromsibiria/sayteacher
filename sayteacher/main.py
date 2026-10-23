"""SayTeacher main entry point: dialogue processing over datatraffic (Redis pub/sub)."""

import asyncio
import logging
import os
import sys

import click

try:  # `python -m sayteacher.main` (package mode)
    from .common import IDLE_TIMEOUT_SECONDS
    from .connection import READY_PROJECT
    from .connection import Connection
    from .service import LlmAnswerService
    from .service import RecognitionService
    from .service import TtsService
except ImportError:  # run flat, e.g. `cd sayteacher && python main.py` or pytest
    from common import IDLE_TIMEOUT_SECONDS
    from connection import READY_PROJECT
    from connection import Connection
    from service import LlmAnswerService
    from service import RecognitionService
    from service import TtsService

log = logging.getLogger("sayteacher")

STUDENT_NAME = "student"


class SessionFinished(Exception):
    """Raised by dialogue_data_processing once the LLM's quit marker has been delivered.

    The outer reconnect loop treats this as a clean end of lesson (not a transient error): it
    resets and stays alive to receive the next one, rather than stopping the process."""


def _idle_check(idle_seconds: float) -> float | bool:
    """Add 1s to the idle counter; return False when IDLE_TIMEOUT_SECONDS is exceeded."""
    if idle_seconds + 1.0 >= IDLE_TIMEOUT_SECONDS:
        log.warning("no frame from server in %.0fs, giving up", IDLE_TIMEOUT_SECONDS)
        return False
    return idle_seconds + 1.0


async def echo_session(conn: Connection) -> None:
    """Test client mirroring Rust tools.rs: receive frames and echo them back."""
    idle_seconds = 0.0
    try:
        while True:
            try:
                frame = await asyncio.wait_for(conn.recv_lesson(), timeout=1.0)
            except asyncio.TimeoutError:
                result = _idle_check(idle_seconds)
                if result is False:
                    break
                idle_seconds = result
                continue

            idle_seconds = 0.0
            text_preview = (frame.note[:32] + "\u2026") if len(frame.note) > 33 else frame.note
            log.info(
                "received name=%r note=%r data_bytes=%d",
                frame.name, text_preview, len(frame.data),
            )

            reply_text = f"record {frame.name} length: {len(frame.data)}"
            conn.client.send("lesson-dialogue", frame.name, reply_text, frame.data)
            log.info("echoed back (name=%r)", frame.name)
    finally:
        await conn.close()
        log.info("disconnected")


def _apply_lesson(api: RecognitionService, name: str, conn: Connection, note: str) -> None:
    """Store an incoming lesson for `name` and ack it so the sender stops re-sending."""
    api.lessons[name] = note
    log.info("Lesson for %s: %s", name, note[:80])
    # Ack so microagent stops re-sending the lesson (it would otherwise ping it forever).
    conn.client.send(READY_PROJECT, "lesson-set", "", b"")


async def dialogue_data_processing(
    api: RecognitionService, name: str, conn: Connection,
) -> None:
    """Route datatraffic frames into the recognition service under `name`.

    Background tasks handle heartbeat (ready-state) and greeting/lesson delivery.
    The main loop polls lesson-dialogue for student audio clips and drains the
    outbox of teacher answers produced by the LLM worker."""
    outbox = api.outboxes.setdefault(name, asyncio.Queue())

    async def _heartbeat() -> None:
        """Send a heartbeat frame on ready-state every 10 seconds."""
        while True:
            await asyncio.sleep(10)
            conn.client.send("ready-state", "hb", "", b"")

    async def _greeting_listener() -> None:
        """Listen for greeting/lesson frames on ready-state; store them and ack receipt."""
        while True:
            frame = await conn.recv_ready()
            if frame.name == "greeting":
                _apply_lesson(api, name, conn, frame.note)

    hb_task = asyncio.create_task(_heartbeat())
    greeting_task = asyncio.create_task(_greeting_listener())
    idle_seconds = 0.0

    try:
        while True:
            send_ok, quit_delivered = await conn.drain_outbox(outbox, name)
            if not send_ok:
                break
            if quit_delivered:
                log.info("quit marker delivered to %s, ending session", name)
                raise SessionFinished

            try:
                frame = await asyncio.wait_for(conn.recv_lesson(), timeout=1.0)
            except asyncio.TimeoutError:
                result = _idle_check(idle_seconds)
                if result is False and outbox.empty():
                    break
                if result is not False:
                    idle_seconds = result
                continue

            idle_seconds = 0.0

            if frame.name == "a":
                audio_payload = frame.data
                if not audio_payload:
                    log.debug("empty audio frame note=%r", frame.note)
                    continue

                clip_seconds = (len(audio_payload) // 4) / 16_000.0
                await api.send(name, b"", frame.note, audio_payload)
                log.info(
                    "queued speech for %s %d bytes ~%.2fs",
                    name, len(audio_payload), clip_seconds,
                )
            else:
                log.debug("unexpected frame on lesson-dialogue: name=%r note=%r", frame.name, frame.note[:40])
    finally:
        hb_task.cancel()
        greeting_task.cancel()
        await asyncio.gather(hb_task, greeting_task, return_exceptions=True)
        # Every exit path resets per-session state so a new lesson starts fresh.
        try:
            await api.reset_session(name)
        except Exception as e:  # noqa: BLE001 - cleanup must not mask the session result
            log.warning("reset session state failed for %s: %s", name, e)
        log.info("disconnected")


@click.command(name="sayteacher")
@click.option(
    "--debug", is_flag=True, default=False, show_default=True, help="log at DEBUG level.",
)
@click.option(
    "--check", is_flag=True, help="one short round-trip and exit instead of looping forever (smoke test / CI)."
)
@click.option(
    "--model", "model_id", default="distil-medium.en", show_default=True,
    help="Whisper model id for recognition; empty disables it (clips are drained, dialogue is not filled).",
)
@click.option(
    "--lang", default="English", show_default=True,
    help="Language used in the LLM prompt template.",
)
@click.option(
    "--llm-model", "llm_model", default="", show_default=True,
    help="Model name for teacher answers, empty disables the LLM.",
)
@click.option(
    "--llm-api", "llm_api", default="", show_default=True,
    help="OpenAI-compatible API base URL, empty disables the LLM.",
)
@click.option(
    "--speaker", default="en_101", show_default=True,
    help=(
        "Silero TTS voice id (en_37, 39, 76) or model bundle id (v3_en / v5_ru). Language is derived from --lang; "
        "empty disables speech output."
    ),
)
def main(
    debug: bool, check: bool, model_id: str, lang: str,
    llm_model: str, llm_api: str, speaker: str,
) -> None:
    """Dialogue processing service over datatraffic (Redis pub/sub + AES-256-GCM)."""
    if debug:
        level = logging.DEBUG
    else:
        level = getattr(logging, os.environ.get("LOGLEVEL", "INFO").upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    async def run() -> int:
        # Check mode is a smoke test / CI hook: one echo round-trip and exit.
        if check:
            conn = Connection()
            await echo_session(conn)
            return 0

        api = RecognitionService(STUDENT_NAME, model=model_id, lang=lang)
        llm_service = LlmAnswerService(api_url=llm_api, model=llm_model)
        tts_service = TtsService(lang=lang, speaker=speaker)
        process_task: asyncio.Task | None = None
        try:
            while True:
                if process_task is None or process_task.done():
                    process_task = asyncio.create_task(
                        api.process(STUDENT_NAME, llm_api=llm_service, tts=tts_service),
                    )

                conn = Connection()
                try:
                    await dialogue_data_processing(api, STUDENT_NAME, conn)
                except SessionFinished:
                    # Lesson ended by quit protocol: stay alive and wait for the next lesson
                    # instead of exiting. Drop the finished worker so a fresh one starts below;
                    # the next iteration opens a new Connection with heartbeat + greeting listener.
                    log.info("session finished by quit protocol; staying up to receive the next lesson")
                    if process_task is not None and not process_task.done():
                        process_task.cancel()
                        try:
                            await process_task
                        except BaseException:  # noqa: BLE001 - expected on cancel
                            pass
                    await api.reset_session(STUDENT_NAME)
                    continue
                finally:
                    await conn.close()

                # Connection dropped without a clean quit: clear stale state.
                await api.reset_session(STUDENT_NAME)
                await asyncio.sleep(0.1)
        finally:
            await api.reset_session(STUDENT_NAME)
            if process_task is not None and not process_task.done():
                process_task.cancel()
                try:
                    await process_task
                except BaseException:  # noqa: BLE001
                    pass
            api.shutdown()
            await llm_service.shutdown()

    try:
        sys.exit(asyncio.run(run()))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
