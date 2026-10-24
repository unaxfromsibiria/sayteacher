import asyncio
import logging
import os
import signal
import sys

import click

try:
    from .common import IDLE_TIMEOUT_SECONDS
    from .connection import Connection
    from .service import RecognitionService
except ImportError:
    from common import IDLE_TIMEOUT_SECONDS
    from connection import Connection
    from service import RecognitionService

log = logging.getLogger("transcreenpt")

BUFFER_SIZE = 128


def _idle_check(idle_seconds: float) -> float | bool:
    if idle_seconds + 1.0 >= IDLE_TIMEOUT_SECONDS:
        log.warning("no frame from server in %.0fs, giving up", IDLE_TIMEOUT_SECONDS)
        return False
    return idle_seconds + 1.0


async def _reader_task(conn: Connection, frame_q: asyncio.Queue, stop_event: asyncio.Event) -> None:
    """Receive audio frames from datatraffic and enqueue them for processing."""
    idle_seconds = 0.0
    while not stop_event.is_set():
        try:
            frame = await conn.recv_audio(timeout=1.0)
        except asyncio.TimeoutError:
            result = _idle_check(idle_seconds)
            if result is False:
                stop_event.set()
                break
            idle_seconds = result
            continue
        except Exception as e:
            log.warning("recv_audio failed: %s", e)
            stop_event.set()
            break
        idle_seconds = 0.0
        try:
            frame_q.put_nowait(frame)
        except asyncio.QueueFull:
            log.warning("frame queue full, dropping frame %s", frame.name)


async def _processor_task(frame_q: asyncio.Queue, api: RecognitionService, name: str, stop_event: asyncio.Event) -> None:
    while not stop_event.is_set():
        try:
            frame = await asyncio.wait_for(frame_q.get(), timeout=0.5)
        except asyncio.TimeoutError:
            continue
        except asyncio.CancelledError:
            break
        try:
            await api.process_segment(name, frame.name, bytes(frame.data))
            log.info("queued output for %s frame=%s %d bytes", name, frame.name, len(frame.data))
        except Exception as e:
            log.warning("processing failed for %s: %s", name, e)
        finally:
            frame_q.task_done()


async def _send_loop(conn: Connection, outbox: asyncio.Queue, name: str, stop_event: asyncio.Event) -> None:
    """Send translation results as datatraffic frames until the session ends."""
    while not stop_event.is_set():
        try:
            group_id, text = await asyncio.wait_for(outbox.get(), timeout=0.5)
        except asyncio.TimeoutError:
            continue
        except asyncio.CancelledError:
            break
        try:
            await conn.send_group(group_id, text)
        except Exception as e:
            log.error("send_group failed for %s group=%d: %s", name, group_id, e)
            stop_event.set()
            break


async def data_processing(api: RecognitionService, name: str, stop: asyncio.Event) -> None:
    conn = Connection()
    outbox = api.outboxes.setdefault(name, asyncio.Queue())
    stop_event = asyncio.Event()
    frame_q = asyncio.Queue(maxsize=BUFFER_SIZE)

    reader_task = asyncio.create_task(_reader_task(conn, frame_q, stop_event))
    processor_tasks = [
        asyncio.create_task(_processor_task(frame_q, api, name, stop_event))
        for _ in range(api.parallel)
    ]
    sender_task = asyncio.create_task(_send_loop(conn, outbox, name, stop_event))
    shutdown_waiter = asyncio.ensure_future(stop.wait())

    tasks = [reader_task, *processor_tasks, sender_task]
    try:
        await asyncio.wait(
            [*tasks, shutdown_waiter],
            return_when=asyncio.FIRST_COMPLETED,
        )
    finally:
        stop_event.set()
        for t in tasks:
            if not t.done():
                t.cancel()
        if not shutdown_waiter.done():
            shutdown_waiter.cancel()
        await asyncio.gather(
            *tasks, shutdown_waiter,
            return_exceptions=True,
        )
        await conn.close()
        try:
            await api.reset_session(name)
        except Exception as e:
            log.warning("reset session state failed for %s: %s", name, e)
        log.info("disconnected")


def _install_signal_handlers(stop: asyncio.Event) -> list[tuple[str, int]]:
    """Set `stop` on SIGINT/SIGTERM so the run loop can shut down cleanly."""
    loop = asyncio.get_running_loop()
    installed: list[tuple[str, int]] = []
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, stop.set)
            installed.append(("loop", sig))
        except NotImplementedError:
            signal.signal(sig, lambda *_args, _stop=stop: _stop.set())
            installed.append(("sig", sig))
    return installed


def _remove_signal_handlers(installed: list[tuple[str, int]]) -> None:
    """Restore default handling for the signals registered by _install_signal_handlers."""
    loop = asyncio.get_running_loop()
    for kind, sig in installed:
        if kind == "loop":
            try:
                loop.remove_signal_handler(sig)
            except (NotImplementedError, RuntimeError):
                pass
        else:
            signal.signal(sig, signal.SIG_DFL)


@click.command(name="transcreenpt")
@click.option("--debug", is_flag=True, default=False, show_default=True, help="log at DEBUG level.")
@click.option("--fromlang", default="english", show_default=True, help="Source language for ASR/translation.")
@click.option("--tolang", default="english", show_default=True, help="Target language for translation.")
@click.option("--model", "model_id", default="distil-medium.en", show_default=True, help="faster-whisper model id for recognition.")
@click.option("--translate-model", "translate_model_id", default="facebook/nllb-200-distilled-600M", show_default=True, help="NLLB model id used to translate recognized text into the target language; empty disables translation.")
@click.option("--audio-out", "audio_out", default="", show_default=True, help="Directory to save audio segments as WAV before recognition.")
@click.option("--num-beams", "num_beams", type=int, default=1, show_default=True, help="Decoding beam width: 1 (greedy) is fastest; higher improves accuracy but slows recognition.")
@click.option("--parallel", type=int, default=1, show_default=True, help="Worker processes for recognition; each loads its own model instance (needs enough VRAM).")
@click.option("--no-segment-validation", is_flag=True, default=False, show_default=True, help="Skip recognized-text quality checks (ratio/frequency/diversity/growth).")
def main(debug: bool, fromlang: str, tolang: str, model_id: str, translate_model_id: str, audio_out: str, num_beams: int, parallel: int, no_segment_validation: bool) -> None:
    if debug:
        level = logging.DEBUG
    else:
        level = getattr(logging, os.environ.get("LOGLEVEL", "INFO").upper(), logging.INFO)
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt="%H:%M:%S")

    student_name = "transcreenpt"

    async def run() -> int:
        api = RecognitionService(
            student_name,
            model=model_id,
            translate_model=translate_model_id,
            from_lang=fromlang,
            to_lang=tolang,
            num_beams=num_beams,
            audio_out=audio_out,
            parallel=parallel,
            validate=not no_segment_validation,
        )
        stop = asyncio.Event()
        handlers = _install_signal_handlers(stop)
        try:
            while not stop.is_set():
                await data_processing(api, student_name, stop)
                if stop.is_set():
                    break
                # small delay to avoid a tight loop on rapid reconnects
                await asyncio.sleep(0.1)
        finally:
            _remove_signal_handlers(handlers)
            await api.reset_session(student_name)
            api.shutdown()

    try:
        sys.exit(asyncio.run(run()))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
