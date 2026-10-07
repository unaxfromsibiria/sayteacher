import asyncio
import logging
import os
import signal
import sys
from pathlib import Path

import click

LIB_PATH = Path(__file__).resolve().parent / "lib" / "libmicroagent_transport.so"
BUFFER_SIZE = 128
if not LIB_PATH.exists():
    sys.exit(f"transcreenpt: {LIB_PATH} not found; build first with 'cargo build --release -p microagent-transport'")

try:
    from .common import CLOSED
    from .common import IDLE_TIMEOUT_SECONDS
    from .common import MessageKindEnum
    from .common import strip_type_prefix
    from .connection import Connection
    from .connection import connect_addr
    from .service import RecognitionService
    from .transport import Transport
except ImportError:
    from common import CLOSED
    from common import IDLE_TIMEOUT_SECONDS
    from common import MessageKindEnum
    from common import strip_type_prefix
    from connection import Connection
    from connection import connect_addr
    from service import RecognitionService
    from transport import Transport

log = logging.getLogger("transcreenpt")


def _is_output(msg) -> bool:
    audio = bytes(msg.audio)
    return bool(audio) and MessageKindEnum.from_byte(audio[0]) is MessageKindEnum.OUTPUT


def _is_ping(msg) -> bool:
    audio = bytes(msg.audio)
    return bool(audio) and MessageKindEnum.from_byte(audio[0]) is MessageKindEnum.PING


async def _poll_frame(conn: Connection, timeout: float):
    try:
        return await conn.next_frame(timeout=timeout), False
    except asyncio.TimeoutError:
        return None, True


def _idle_check(idle_seconds: float) -> float | bool:
    if idle_seconds + 1.0 >= IDLE_TIMEOUT_SECONDS:
        log.warning("no frame from server in %.0fs, giving up", IDLE_TIMEOUT_SECONDS)
        return False
    return idle_seconds + 1.0


async def _reader_task(conn: Connection, frame_q: asyncio.Queue, ack_q: asyncio.Queue, stop_event: asyncio.Event) -> None:
    idle_seconds = 0.0
    while not stop_event.is_set():
        item, timed_out = await _poll_frame(conn, timeout=1.0)
        if timed_out:
            result = _idle_check(idle_seconds)
            if result is False:
                stop_event.set()
                break
            idle_seconds = result
            continue
        idle_seconds = 0.0
        if item is CLOSED:
            log.info("peer closed connection")
            stop_event.set()
            break
        try:
            msg = conn.decode(item)
        except Exception as e:
            log.warning("undecodable frame (%d bytes): %s (ffi_last_error=%d)", len(item), e, conn.last_error())
            continue
        if _is_ping(msg):
            # Echo the ping back so the server knows we are alive; any frame already reset idle_seconds.
            log.debug("ping from server id=%s", msg.id.hex())
            if not await conn.send(msg.id, "", bytes([MessageKindEnum.PING])):
                stop_event.set()
                break
            continue
        if not _is_output(msg):
            log.debug("skipping non-output message id=%s", msg.id.hex())
            continue
        audio_payload = strip_type_prefix(bytes(msg.audio))
        if not audio_payload:
            log.debug("empty output frame id=%s", msg.id.hex())
            continue
        try:
            frame_q.put_nowait((msg, audio_payload))
        except asyncio.QueueFull:
            log.warning("frame queue full, dropping frame for %s", msg.id.hex())
        try:
            ack_q.put_nowait(msg.id)
        except asyncio.QueueFull:
            log.warning("ack queue full, dropping ack for %s", msg.id.hex())


async def _processor_task(frame_q: asyncio.Queue, api: RecognitionService, name: str, stop_event: asyncio.Event) -> None:
    while not stop_event.is_set():
        try:
            msg, audio_payload = await asyncio.wait_for(frame_q.get(), timeout=0.5)
        except asyncio.TimeoutError:
            continue
        except asyncio.CancelledError:
            break
        try:
            await api.process_segment(name, msg.id, audio_payload)
            log.info("queued output for %s id=%s %d bytes", name, msg.id.hex(), len(audio_payload))
        except Exception as e:
            log.warning("processing failed for %s: %s", name, e)
        finally:
            frame_q.task_done()


async def _ack_sender_task(conn: Connection, ack_q: asyncio.Queue, stop_event: asyncio.Event) -> None:
    while not stop_event.is_set():
        try:
            msg_id = await asyncio.wait_for(ack_q.get(), timeout=0.5)
        except asyncio.TimeoutError:
            continue
        except asyncio.CancelledError:
            break
        ok = await conn.send(msg_id, "", b"")
        if not ok:
            stop_event.set()
            break
        ack_q.task_done()


async def echo_session(host: str, port: int, password: str) -> None:
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
                log.warning("undecodable frame (%d bytes): %s (ffi_last_error=%d)", len(item), e, conn.last_error())
                continue
            text_preview = (msg.text[:32] + "…") if len(msg.text) > 33 else msg.text
            audio_len = len(msg.audio)
            log.info("received id=%s text=%r audio_bytes=%d", msg.id.hex(), text_preview, audio_len)
            reply_text = f"record {msg.id.hex()} length: {audio_len}"
            if not await conn.send(msg.id, reply_text, msg.audio):
                break
            log.info("echoed back (text=%r)", reply_text)
    finally:
        await conn.close()
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


async def data_processing(
    api: RecognitionService, name: str, host: str, port: int, password: str, stop: asyncio.Event
) -> None:
    conn = Connection(host, port, password, Transport(str(LIB_PATH)))
    try:
        await conn.connect(stop=stop)
    except ConnectionError as e:
        log.error("connect failed: %s", e)
        try:
            await api.reset_session(name)
        except Exception as e2:
            log.warning("reset session state failed for %s: %s", name, e2)
        return

    outbox = api.outboxes.setdefault(name, asyncio.Queue())
    conn.start_pump()
    stop_event = asyncio.Event()
    frame_q = asyncio.Queue(maxsize=BUFFER_SIZE)
    ack_q = asyncio.Queue(maxsize=BUFFER_SIZE)

    sender_task = asyncio.create_task(conn.send_loop(outbox, name, stop_event))
    reader_task = asyncio.create_task(_reader_task(conn, frame_q, ack_q, stop_event))
    # One coroutine per worker process: while one waits on ASR the others keep feeding frames.
    processor_tasks = [
        asyncio.create_task(_processor_task(frame_q, api, name, stop_event))
        for _ in range(api.parallel)
    ]
    ack_task = asyncio.create_task(_ack_sender_task(conn, ack_q, stop_event))
    # External shutdown (SIGINT/SIGTERM): ends this session as soon as `stop` is set.
    shutdown_waiter = asyncio.ensure_future(stop.wait())

    tasks = [reader_task, *processor_tasks, ack_task, sender_task]
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


@click.command(name="transcreenpt")
@click.option("--connect", "addr", default="127.0.0.1:16050", show_default=True, help="host:port of the outvoiceagent server.")
@click.option("--password", default="", show_default=True, help="shared secret used for auth; empty matches a server with no password set.")
@click.option("--debug", is_flag=True, default=False, show_default=True, help="log at DEBUG level.")
@click.option("--fromlang", default="english", show_default=True, help="Source language for ASR/translation.")
@click.option("--tolang", default="english", show_default=True, help="Target language for translation.")
@click.option("--model", "model_id", default="distil-medium.en", show_default=True, help="faster-whisper model id for recognition.")
@click.option("--translate-model", "translate_model_id", default="facebook/nllb-200-distilled-600M", show_default=True, help="NLLB model id used to translate recognized text into the target language; empty disables translation.")
@click.option("--audio-out", "audio_out", default="", show_default=True, help="Directory to save audio segments as WAV before recognition.")
@click.option("--num-beams", "num_beams", type=int, default=1, show_default=True, help="Decoding beam width: 1 (greedy) is fastest; higher improves accuracy but slows recognition.")
@click.option("--parallel", type=int, default=1, show_default=True, help="Worker processes for recognition; each loads its own model instance (needs enough VRAM).")
@click.option("--no-segment-validation", is_flag=True, default=False, show_default=True, help="Skip recognized-text quality checks (ratio/frequency/diversity/growth).")
def main(addr: str, password: str, debug: bool, fromlang: str, tolang: str, model_id: str, translate_model_id: str, audio_out: str, num_beams: int, parallel: int, no_segment_validation: bool) -> None:
    if debug:
        level = logging.DEBUG
    else:
        level = getattr(logging, os.environ.get("LOGLEVEL", "INFO").upper(), logging.INFO)
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt="%H:%M:%S")
    try:
        host, port = connect_addr(addr)
    except ValueError as e:
        sys.exit(str(e))
    student_name = f"{host}:{port}"

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
                await data_processing(api, student_name, host, port, password, stop)
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
