import asyncio
import io
import json
import logging
import os
import struct
import time
import wave
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime

import numpy as np
from openai import AsyncOpenAI

try:  # `python -m sayteacher.service` (package mode)
    from .common import DEFAULT_BUNDLE_BY_LANG
    from .common import QUIT_MARKER_DELAY_SECONDS
    from .common import QUIT_WORD
    from .common import SPEAKER_LANG_HINT
    from .common import TARGET_SAMPLE_RATE
    from .common import DialogueEntry
    from .common import MessageKindEnum
    from .common import silero_language
    from .prompt import _duration_minutes
    from .prompt import build_prompt
    from .silero_models import known_bundle_ids
except ImportError:  # flat run / pytest bootstrap
    from common import DEFAULT_BUNDLE_BY_LANG
    from common import QUIT_MARKER_DELAY_SECONDS
    from common import QUIT_WORD
    from common import SPEAKER_LANG_HINT
    from common import TARGET_SAMPLE_RATE
    from common import DialogueEntry
    from common import MessageKindEnum
    from common import silero_language
    from prompt import _duration_minutes
    from prompt import build_prompt
    from silero_models import known_bundle_ids

log = logging.getLogger("sayteacher")

# Model bundle ids Silero supports; a static snapshot of the official models.yml shipped
# in silero_models.py (no runtime file read / OmegaConf needed). We only use membership to
# tell --speaker voice ids apart from bundle ids — silero_tts() downloads on demand.
_KNOWN_BUNDLE_IDS: frozenset[str] = known_bundle_ids()

# Per-worker-process cache of loaded ASR models, keyed by (model, device, compute_type).
_WHISPER_CACHE: dict[tuple, object] = {}

# Per-worker-process cache of loaded TTS models, keyed by (bundle, voice, language).
_TTS_CACHE: dict[tuple, object] = {}


def _worker_transcribe(model: str, lang: str | None, samples: np.ndarray) -> str:
    """Runs in a worker process: load/cache the faster-whisper model and transcribe samples.

    Module-level so it is picklable for ProcessPoolExecutor; only settings and the numpy
    samples cross the process boundary, never the service instance."""
    try:
        import torch
        from faster_whisper import WhisperModel
    except Exception as e:
        log.error("failed to import ASR deps for %s: %s", model, e)
        return ""

    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float16"
    key = (model, device, compute_type)
    pipe = _WHISPER_CACHE.get(key)
    if pipe is None:
        try:
            pipe = WhisperModel(model, device=device, compute_type=compute_type)
        except Exception as e:
            log.error("failed to load ASR model %s: %s", model, e)
            return ""
        _WHISPER_CACHE[key] = pipe
        log.info("ASR model loaded %s (device=%s compute=%s)", model, device, compute_type)

    kwargs: dict = {"beam_size": 1}
    if lang:
        kwargs["language"] = lang

    try:
        segments, info = pipe.transcribe(samples, **kwargs)
        return " ".join(seg.text.strip() for seg in segments).strip()
    except Exception as e:
        log.warning("transcribe error: %s", e)
        return ""


def _worker_tts(bundle: str, voice: str, language: str, text: str) -> bytes:
    """Runs in a worker process: load/cache the Silero TTS model and synthesize speech.

    Returns 16 kHz mono f32 LE bytes; b"" on empty input or failure."""
    if not text:
        return b""

    try:
        import torch
        from silero import silero_tts
        from torchaudio.transforms import Resample
    except Exception as e:
        log.error("failed to import TTS deps: %s", e)
        return b""

    key = (bundle, voice, language)
    model = _TTS_CACHE.get(key)
    if model is None:
        try:
            loaded = silero_tts(language=language, speaker=bundle)
            if isinstance(loaded, tuple) and loaded:
                model = loaded[0]
            else:  # pragma: no cover - unexpected silero version layout
                raise RuntimeError(f"unexpected silero_tts result type: {type(loaded)!r}")
        except Exception as e:
            log.error("failed to load TTS model %s: %s", bundle, e)
            return b""

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        _TTS_CACHE[key] = model
        log.info("tts: loaded %s (voice=%s) on %s", bundle, voice or "(default)", device.type)

    try:
        voices = getattr(model, "speakers", None)
        if isinstance(voices, (list, tuple)) and voices:
            speaker = str(voice) if voice else sorted(str(v) for v in voices)[0]
            audio = model.apply_tts(text=text, speaker=speaker, sample_rate=24_000)
            source_rate = 24_000
        else:
            audio = model.apply_tts(text=text, sample_rate=16_000)
            source_rate = 16_000

        samples = audio.detach().float().cpu()
        if samples.dim() == 2:
            samples = samples[0]
        if source_rate != TARGET_SAMPLE_RATE and samples.numel():
            samples = Resample(source_rate, TARGET_SAMPLE_RATE)(samples)
        samples = samples.flatten().contiguous()
        return bytes(samples.numpy().astype("<f4").tobytes())
    except Exception as e:
        log.warning("tts: generation failed for %r: %s", text[:60], e)
        return b""


def _drain(queue) -> int:
    """Remove and count all items currently in `queue`; returns how many were left behind."""
    count = 0
    while True:
        try:
            queue.get_nowait()
            count += 1
        except asyncio.QueueEmpty:
            return count


def _dialogue_tail_text(dialogue: list[DialogueEntry]) -> str:
    """Last `limit` turns as 'you:'/'teacher:' lines for the client UI dialogue field."""
    if dialogue and len(dialogue) > 1:
        *_, s_say, t_say = dialogue
        return f"you: {s_say.text}\nteacher: {t_say.text}\n"
    return ""


def _parse_json_object(text: str) -> dict | None:
    """Parse the first balanced JSON object {...} in a possibly wrapped LLM answer.

    Models sometimes wrap the reply in markdown code fences (```json ... ```), which breaks
    json.loads on the raw text. We scan from the first '{', skipping braces inside string
    literals, and parse once the matching '}' closes it. Returns None when no object parses.
    """
    start = text.find("{")
    while start != -1:
        depth = 0
        in_string = False
        escaped = False
        for i in range(start, len(text)):
            ch = text[i]
            if in_string:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"':
                    in_string = False
                continue

            if ch == '"':
                in_string = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        parsed = json.loads(text[start:i + 1])
                    except (json.JSONDecodeError, TypeError):
                        break
                    return parsed if isinstance(parsed, dict) else None
        start = text.find("{", start + 1)
    return None


class TtsService:
    """Holds Silero TTS settings; actual synthesis runs in a worker process via _worker_tts."""

    def __init__(self, lang: str, speaker: str) -> None:
        self.speaker = (speaker or "").strip()
        self.language_code = silero_language(lang)
        if self.enabled and self.language_code == "auto":
            log.warning("tts: unknown --lang %r; Silero language will be derived from the speaker id", lang)

        self._voice = "" if self.speaker in _KNOWN_BUNDLE_IDS else self.speaker

    @property
    def enabled(self) -> bool:
        return bool(self.speaker) or self.language_code != "auto"

    def _resolve_language(self) -> str:
        """Language key for silero_tts; falls back from a bundle id, else 'en'."""
        if self.language_code != "auto":
            return self.language_code
        return SPEAKER_LANG_HINT.get(self.speaker, "en")

    def worker_params(self) -> tuple[str, str, str]:
        """Return (bundle, voice, language) for _worker_tts; resolved once at call time."""
        language = self._resolve_language()
        if self.speaker and self.speaker in _KNOWN_BUNDLE_IDS:
            bundle = self.speaker
        else:
            bundle = DEFAULT_BUNDLE_BY_LANG.get(language, "v3_en")
        return bundle, self._voice, language


class LlmAnswerService:
    """Call an OpenAI-compatible API once; returns a JSON object with "say" / "write" fields.

    The client is created at init and reused across requests; close it via shutdown()."""

    def __init__(self, api_url: str, model: str) -> None:
        self.model = model
        if api_url:
            # A placeholder key keeps construction from raising when OPENAI_API_KEY is unset
            # (e.g. the LLM is disabled by an empty --llm-model); real requests use the env key.
            self._client = AsyncOpenAI(
                base_url=api_url or None,
                api_key=os.environ.get("OPENAI_API_KEY") or "sk-placeholder",
            )
        else:
            self._client = None

    @property
    def enabled(self) -> bool:
        return self._client is not None and bool(self.model)

    async def shutdown(self) -> None:
        """Close the OpenAI client; safe to call when the LLM is disabled."""
        if self._client is not None:
            await self._client.close()

    async def request(self, dialogue: list[DialogueEntry], prompt: str) -> tuple[str, str, int | None, bool]:
        """Return (say, write, expected_duration, to_quit).

        Empty strings mean the corresponding field was missing or unparseable;
        expected_duration is None unless the LLM returned a number; to_quit means the
        LLM decided the dialogue should end once its final answer has been delivered.
        """
        if not self.enabled:
            log.debug("llm disabled (model=%r)", self.model)
            return "", "", None, False

        # extra_body disables the chat-template thinking phase so short dialogue replies skip
        # reasoning tokens; `reasoning` stays as a budget in case thinking is re-enabled.
        t0 = time.perf_counter()
        response = await self._client.responses.create(
            model=self.model, input=prompt, reasoning={"effort": "low"},
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        request_elapsed = time.perf_counter() - t0
        text = (response.output_text or "").strip()
        parsed = _parse_json_object(text)
        if parsed is None:
            log.warning("llm answer contains no JSON object: %r", text[:120])
            return "", "", None, False

        if not isinstance(parsed, dict):
            log.warning("llm answer is not a JSON object")
            return "", "", None, False

        say = str(parsed.get("say", "")).strip()
        write = str(parsed.get("write", "")).strip()
        to_quit = "true" in str(parsed.get("quit", "")).lower()

        duration: float | None = None
        if parsed.get("expected_duration"):
            try:
                duration = max(float(parsed["expected_duration"]), 0)
            except Exception as err:
                log.warning("expected duration format: %s", err)

        elapsed_minutes = _duration_minutes(dialogue)
        remaining = max(duration - elapsed_minutes, 0) if duration is not None else None
        log.info(
            "to say: %r to write: %r expected_duration=%s elapsed=%d min remaining=%s quit=%s llm_took=%.2fs",
            say, write, duration, elapsed_minutes, remaining, to_quit, request_elapsed,
        )
        return say, write, duration, to_quit


class SpeechClip:
    """One audio fragment received from the microagent transport.

    Stores raw little-endian float32 samples (mono, `sample_rate` Hz) plus a helper to
    get them as an ndarray — the exact shape librosa/whisper consume in wisper-data.txt.
    No heavy deps are needed until the recognition step is implemented.
    """

    def __init__(self, uid: bytes, data: bytes, sample_rate: int = TARGET_SAMPLE_RATE) -> None:
        if len(data) % 4 != 0:
            raise ValueError(f"audio length {len(data)} is not a multiple of 4 (f32 LE)")

        self.uid = uid
        self.data = data
        self.sample_rate = sample_rate

    @property
    def num_samples(self) -> int:
        return len(self.data) // 4

    @property
    def duration_seconds(self) -> float:
        if self.sample_rate == 0:
            return 0.0
        return self.num_samples / self.sample_rate

    def to_numpy(self):
        """Decode into a 1-D float32 ndarray, ready for librosa/whisper."""
        return np.frombuffer(self.data, dtype="<f4")

    def wav_bytes(self) -> bytes:
        """Encode the clip as an in-memory PCM16 mono WAV file (librosa.load compatible)."""
        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)  # 16 bit — speech quality is fine for recognition
            w.setframerate(self.sample_rate)

            i = 0
            while i + 4 <= len(self.data):
                s = struct.unpack_from("<f", self.data, i)[0]
                s = max(-1.0, min(1.0, s))
                w.writeframes(struct.pack("<h", int(s * 32767)))
                i += 4

        return buf.getvalue()


class RecognitionService:
    """Holds per-student queues of SpeechClip and the lesson texts seen so far.

    A student is the named peer that connects through `--connect` (host:port); one
    sayteacher process serves one such student by default ("student")."""

    quit_msg: asyncio.Queue

    def __init__(self, *names: str, model: str = "", lang: str = "English") -> None:
        self.model_id = model  # empty means recognition is off (no dialogue filling)
        self.lang = lang

        self.students: dict[str, asyncio.Queue[SpeechClip]] = {}
        for name in names:
            self.students[name] = asyncio.Queue()

        # (kind byte, frame text, audio payload) triples queued by the process worker;
        # main.py prefixes the kind byte and sends them as frames back to the client.
        self.outboxes: dict[str, "asyncio.Queue[tuple[int, str, bytes]]"] = {}
        for name in names:
            self.outboxes[name] = asyncio.Queue()

        # struct of dialogues
        # {name: [DialogueEntry(now(), True, "student said"), DialogueEntry(...)]}
        self.dialogues: dict[str, list[DialogueEntry]] = {}
        # Lesson texts keyed by the student who received them (not by session id).
        self.lessons: dict[str, str] = {}
        self.quit_msg = asyncio.Queue()
        # One-shot timer task started when _expected_duration is first seen; puts QUIT_WORD
        # into quit_msg once it fires so the next LLM reply carries a "time to leave" hint.
        self._duration_timer_task: asyncio.Task | None = None
        # Once set (hint consumed from quit_msg or the LLM answered quit=true), every further
        # build_prompt call carries the same quit=true.
        self.quit_hint = False
        # One-shot delayed QUIT-marker task; created at most once per student session.
        self._quit_marker_task: asyncio.Task | None = None

        # The ASR/TTS models are created inside worker processes; the main process only
        # holds settings and dispatches numpy samples / text to the pool.
        self._executor = ProcessPoolExecutor(max_workers=1)

    def shutdown(self) -> None:
        """Shut down the worker process pool, cancelling pending tasks."""
        self._executor.shutdown(wait=False, cancel_futures=True)

    def _start_duration_timer(self, expected_minutes: float) -> None:
        """Arm the one-shot end-of-lesson timer; started only on the first duration value."""
        if self._duration_timer_task is not None and not self._duration_timer_task.done():
            return

        async def _timer() -> None:
            await asyncio.sleep(expected_minutes * 60.0)
            log.info("session reached its expected lesson time (%s min); requesting dialogue end", expected_minutes)
            self.quit_msg.put_nowait(QUIT_WORD)

        self._duration_timer_task = asyncio.create_task(_timer())

    async def reset_session(self, name: str) -> None:
        """Clear all per-session state for a student after the session ended or was dropped.

        Called by main when the quit protocol finished or the connection broke; the next
        reconnection starts with an empty dialogue and fresh queues so stale entries from
        the previous lesson do not leak into the new one. Pending timer tasks are cancelled
        first, then any items they may have queued are drained."""
        if name not in self.students:
            return

        for attr_name in ("_quit_marker_task", "_duration_timer_task"):
            task = getattr(self, attr_name)
            if task is not None and not task.done():
                task.cancel()
                try:
                    await task
                except BaseException:  # noqa: BLE001 - expected on cancel
                    pass
            setattr(self, attr_name, None)

        # In-place clear: a worker may hold a reference to the list from setdefault().
        self.dialogues.setdefault(name, []).clear()
        queue = self.students[name]
        outbox = self.outboxes.get(name)
        dropped = _drain(queue) + (_drain(outbox) if outbox is not None else 0)
        # quit_hint and pending quit words are one-shot per lesson; drop them.
        self.quit_hint = False
        _drain(self.quit_msg)
        log.info("%s: session state reset (%d pending items dropped)", name, dropped)

    async def process(self, name: str, llm_api: LlmAnswerService | None = None, tts: TtsService | None = None):
        """Consume the student's clip queue and fill `self.dialogues[name]`.

        Intended to run as a task in the event loop. When no model is configured
        (`model_id` empty) clips are drained so `send()` never blocks, but nothing is
        recognized or stored — recognition simply does not happen for this student.

        After each recognized utterance an LLM answer request is made (if enabled);
        the "say" text joins the dialogue as a teacher entry. The worker then queues two
        outbox items per turn: a 'd' frame carrying recent you:/teacher: lines plus TTS audio
        of "say", and, when present, an extra 'n' frame with the LLM's "write" text.
        """
        if name not in self.students:
            raise KeyError(f"unknown student {name!r}; registered={list(self.students)}")

        queue = self.students[name]
        outbox = self.outboxes.setdefault(name, asyncio.Queue())
        if not self.model_id:
            log.info("student %s has no model configured, draining clips without recognition", name)
            while True:
                await queue.get()
                continue

        async def llm_reply(dialogue: list[DialogueEntry]) -> tuple[str | None, str, bool]:
            """Ask the LLM; store "say" as a teacher entry. Return ("write", "say", to_quit)."""
            if not (llm_api and getattr(llm_api, "enabled", False)):
                return None, "", False

            lesson = self.lessons.get(name, "")
            # Consume the one-shot hint from quit_msg (timer or LLM); `empty()` is the safe check —
            # get_nowait() on an empty queue raises QueueEmpty and would kill this reply path.
            if not self.quit_hint:
                if not self.quit_msg.empty():
                    try:
                        self.quit_msg.get_nowait()
                        self.quit_hint = True
                    except asyncio.QueueEmpty:  # pragma: no cover - race between the two checks
                        pass

            prompt = build_prompt(lesson, dialogue, self.lang, quit=self.quit_hint)
            try:
                say, write, expected_duration, to_quit = await llm_api.request(dialogue, prompt)
            except Exception as e:  # noqa: BLE001 - one failed reply must not kill the worker
                log.warning("llm request failed for %s: %s", name, e)
                return None, "", False
            else:
                log.debug(prompt)

            if expected_duration is not None:
                self._start_duration_timer(float(expected_duration))

            if say:
                dialogue.append(DialogueEntry(time=datetime.now(), is_student=False, text=say))

            # The LLM itself decided the lesson is over -> keep quit=true for all further prompts.
            self.quit_hint = self.quit_hint or to_quit
            return write or None, say, to_quit

        loop = asyncio.get_running_loop()
        while True:
            clip = await queue.get()
            try:
                # Map --lang to a short code (en/ru/de/es/fr) whisper understands; skip the
                # hint entirely when we don't recognize the value, letting the model guess.
                lang_code = silero_language(self.lang)
                if lang_code == "auto":
                    lang_code = ""

                t0 = time.perf_counter()
                text = await loop.run_in_executor(
                    self._executor, _worker_transcribe, self.model_id, lang_code or None, clip.to_numpy(),
                )
                recognition_elapsed = time.perf_counter() - t0
                if not text:
                    log.debug("empty recognition for %s (uid=%s)", name, clip.uid.hex())
                    continue

                dialogue = self.dialogues.setdefault(name, [])
                dialogue.append(DialogueEntry(time=datetime.now(), is_student=True, text=text))
                log.info(
                    "recognized %s uid=%s took=%.2fs: %r",
                    name, clip.uid.hex(), recognition_elapsed, text[:80],
                )

                write, say, to_quit = await llm_reply(dialogue)
                if not (write or say):
                    continue

                # TTS of the "say" text; runs in a worker process so synthesis does not block.
                tts_audio = b""
                if say and tts is not None:
                    try:
                        bundle, voice, language = tts.worker_params()
                        tts_audio = await loop.run_in_executor(
                            self._executor, _worker_tts, bundle, voice, language, say,
                        )
                    except Exception as e:  # noqa: BLE001 - one bad reply must not kill the worker
                        log.warning("tts generation failed for %s: %s", name, e)

                # Optional additional frame (type 'n') with the LLM's "write" text, if any.
                if write:
                    await outbox.put((int(MessageKindEnum.NOTE), write, b""))
                # Primary frame (type 'd'): recent dialogue turns + TTS audio of "say".
                # The client UI routes this to its dialogue field.
                await outbox.put((int(MessageKindEnum.DIALOGUE), _dialogue_tail_text(dialogue), tts_audio))

                # Quit request: queued only after the final answer above, so it is delivered last and
                # the connection closes right after those frames leave the socket. A short delay lets
                # the client start TTS playback of "say" before the marker (and thus close) arrives.
                if to_quit and self._quit_marker_task is None:
                    log.info(
                        "llm asked to end the dialogue for %s; quit marker in %.0fs",
                        name, QUIT_MARKER_DELAY_SECONDS,
                    )

                    async def _delayed_quit() -> None:
                        await asyncio.sleep(QUIT_MARKER_DELAY_SECONDS)
                        outbox.put_nowait((int(MessageKindEnum.QUIT), "", b""))

                    self._quit_marker_task = asyncio.create_task(_delayed_quit())

            except Exception as e:
                log.warning("recognition failed for %s: %s", name, e)

    async def send(self, name: str, uid: bytes, text: str, data: bytes) -> None:
        """Route one decoded transport frame to a named student; call only after a successful decode.

        Lesson frames (empty audio + non-empty text) are stored in `self.lessons[name]`.
        Audio frames become a SpeechClip enqueued for that student's queue. Unknown names
        raise KeyError so wiring mistakes surface immediately.
        """
        if name not in self.students:
            raise KeyError(f"unknown student {name!r}; registered={list(self.students)}")

        if not data and text:
            self.lessons[name] = text
            return

        clip = SpeechClip(uid, data)
        await self.students[name].put(clip)
