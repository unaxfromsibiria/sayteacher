import asyncio
import inspect
import io
import json
import logging
import os
import struct
import wave
from datetime import datetime
from typing import Callable

import numpy as np
import torch
from ollama import AsyncClient
from silero import silero_tts
from torchaudio.transforms import Resample

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


# Loaded silero models, one per speaker id; avoids re-reading the .pt cache per service.
_MODEL_CACHE: dict[str, object] = {}

# Model bundle ids Silero supports; a static snapshot of the official models.yml shipped
# in silero_models.py (no runtime file read / OmegaConf needed). We only use membership to
# tell --speaker voice ids apart from bundle ids — silero_tts() downloads on demand.
_KNOWN_BUNDLE_IDS: frozenset[str] = known_bundle_ids()


class TtsService:
    """Synthesize teacher speech (Silero v3 model) into 16 kHz mono f32 LE bytes.

    The Rust client resamples playback back to its output device rate from this rate;
    the transport audio field is little-endian float32, matching SpeechClip encoding.
    """

    def __init__(self, lang: str, speaker: str) -> None:
        self.speaker = (speaker or "").strip()
        self.language_code = silero_language(lang)
        if self.enabled and self.language_code == "auto":
            log.warning("tts: unknown --lang %r; Silero language will be derived from the speaker id", lang)

        self._voice = "" if self.speaker in _KNOWN_BUNDLE_IDS else self.speaker

        self._model = None

    @property
    def enabled(self) -> bool:
        return bool(self.speaker) or self.language_code != "auto"

    def _resolve_language(self) -> str:
        """Language key for silero_tts; falls back from a bundle id, else 'en'."""
        if self.language_code != "auto":
            return self.language_code
        return SPEAKER_LANG_HINT.get(self.speaker, "en")

    def _load_model(self):
        if not self.enabled or self._model is not None:
            return self._model

        language = self._resolve_language()
        # --speaker may be a bundle id (v3_en, v5_ru ...) or a voice id (en_117, xenia ...).
        # silero_tts(language=..., speaker=<bundle>) needs the *model* to download/load and
        # asserts it belongs to `language`, so we distinguish by bundle-id membership.
        if self.speaker and self.speaker in _KNOWN_BUNDLE_IDS:
            bundle = self.speaker
        else:
            bundle = DEFAULT_BUNDLE_BY_LANG.get(language, "v3_en")

        cache_key = f"{bundle}::{self.speaker}"  # same bundle+voice -> one cached model
        if cache_key in _MODEL_CACHE:
            self._model = _MODEL_CACHE[cache_key]
            return self._model

        loaded = silero_tts(language=language, speaker=bundle)
        # jit-style models unpack into (model, symbols, sample_rate, example, apply_tts);
        # packaged v3/v5 ones only return (model, example). Normalize to the model object.
        if isinstance(loaded, tuple) and loaded:
            self._model = loaded[0]
        else:  # pragma: no cover - unexpected silero version layout
            raise RuntimeError(f"unexpected silero_tts result type: {type(loaded)!r}")

        device = torch_device()
        self._model.to(device)  # packaged v3 mutates in place and returns None; prefer CUDA.
        log.info("tts: loaded %s (voice=%s) on %s", bundle, self._voice or "(default)", device.type)
        _MODEL_CACHE[cache_key] = self._model
        return self._model

    def generate(self, text: str) -> bytes:
        """One synthesis call; empty text or a failed load/synthesis returns b''."""
        if not text or not self.enabled:
            return b""
        model = self._load_model()
        if model is None:
            log.warning("tts: no model for speaker=%r lang=%s", self.speaker, self.language_code)
            return b""

        # Empty voice -> _synthesize picks the first available one (bundle case). Non-empty
        # voice id (en_117, xenia ...) is passed through verbatim to apply_tts().
        try:
            audio, source_rate = _synthesize(model, text, self._voice)
        except Exception as e:  # noqa: BLE001 - synthesis failure must not kill the worker
            log.warning("tts: generation failed for %r: %s", text[:60], e)
            return b""

        try:
            samples = audio.detach().float().cpu()
            if samples.dim() == 2:
                samples = samples[0]
            # The client expects 16 kHz mono f32 LE; resample from the model's native rate.
            if source_rate != TARGET_SAMPLE_RATE and samples.numel():
                samples = Resample(source_rate, TARGET_SAMPLE_RATE)(samples)
            samples = samples.flatten().contiguous()
        except Exception as e:  # noqa: BLE001 - resampling failure degrades to raw samples
            log.warning("tts: resample to %d Hz failed: %s", TARGET_SAMPLE_RATE, e)

        return bytes(samples.numpy().astype("<f4").tobytes())


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


def torch_device():
    """CUDA when available so synthesis runs on the GPU, CPU otherwise."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _synthesize(model, text: str, voice: str) -> tuple[object, int]:
    """Run one synthesis; returns (audio tensor, source sample rate).

    `voice` is the speaker id for multi-voice packaged models. When it's empty we fall back
    to the first available voice — this is what happens when --speaker named a bundle
    like v3_en rather than a specific voice such as en_117."""
    voices = getattr(model, "speakers", None)
    if isinstance(voices, (list, tuple)) and voices:
        speaker = str(voice) if voice else sorted(str(v) for v in voices)[0]
        return model.apply_tts(text=text, speaker=speaker, sample_rate=24_000), 24_000

    # jit-style single-voice models synthesize with a fixed rate.
    return model.apply_tts(text=text, sample_rate=16_000), 16_000


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


class LlmAnswerService:
    """One call to the ollama API returning a JSON object with "say" / "write" fields."""

    def __init__(self, api_url: str, model: str) -> None:
        self.api_url = api_url
        self.model = model

    @property
    def enabled(self) -> bool:
        return bool(self.api_url and self.model)

    async def request(self, dialogue: list[DialogueEntry], prompt: str) -> tuple[str, str, int | None, bool]:
        """Return (say, write, expected_duration, to_quit).

        Empty strings mean the corresponding field was missing or unparseable;
        expected_duration is None unless the LLM returned a number; to_quit means the
        LLM decided the dialogue should end once its final answer has been delivered.
        """
        if not self.enabled:
            log.debug("llm disabled (api_url=%r model=%r)", self.api_url, self.model)
            return "", "", None, False

        client = AsyncClient(host=self.api_url)
        try:
            response = await client.generate(model=self.model, think=False, prompt=prompt)
        finally:
            # ollama <=0.6.x exposes async close(); newer releases may use aclose(). A failed
            # cleanup must not mask the real request result or error.
            for _name in ("aclose", "close"):
                closer = getattr(client, _name, None)
                if closer is None:
                    continue
                try:
                    result = closer()
                    if inspect.isawaitable(result):
                        await result
                except Exception as e:  # noqa: BLE001 - cleanup only
                    log.debug("llm client %s failed: %r", _name, e)
                break

        text = str(response.response or "").strip()
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
            "to say: %r to write: %r expected_duration=%s elapsed=%d min remaining=%s quit=%s",
            say, write, duration, elapsed_minutes, remaining, to_quit,
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
    sayteacher process serves one such student by default ("student").
    """

    pipeline: Callable
    quit_msg: asyncio.Queue

    def __init__(self, *names: str, model: str = "", lang: str = "English") -> None:
        self.model_id = model  # empty means recognition is off (no dialogue filling)
        self.lang = lang
        if model:
            from transformers import pipeline
            self.pipeline = pipeline

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
        # build_prompt call carries the same quit=true — task 36.
        self.quit_hint = False
        # One-shot delayed QUIT-marker task; created at most once per student session.
        self._quit_marker_task: asyncio.Task | None = None

    def _start_duration_timer(self, expected_minutes: float) -> None:
        """Arm the one-shot end-of-lesson timer; started only on the first duration value."""
        if self._duration_timer_task is not None and not self._duration_timer_task.done():
            return

        async def _timer() -> None:
            await asyncio.sleep(expected_minutes * 60.0)
            log.info("session reached its expected lesson time (%s min); requesting dialogue end", expected_minutes)
            self.quit_msg.put_nowait(QUIT_WORD)

        self._duration_timer_task = asyncio.create_task(_timer())

    def _cache_ready(self) -> bool:
        """True when every file of `self.model_id` is already present in the local HF cache.

        Uses huggingface_hub's native cache scan (offline, no network). A repo counts as
        ready only if it is listed and has zero leftover `.incomplete` marker files — that
        means the download finished, so we can load fully from disk.
        """
        if not self.model_id:
            return False

        try:
            from huggingface_hub import scan_cache_dir
        except ImportError:  # pragma: no cover - dep is required, but be graceful
            log.warning("asr: huggingface_hub missing; cannot verify local cache")
            return False

        try:
            info = scan_cache_dir()
        except Exception as e:  # noqa: BLE001 - a broken cache must not block startup
            log.debug("hf cache scan failed: %s", e)
            return False

        for repo in info.repos:
            if repo.repo_id != self.model_id:
                continue
            incomplete = [f for f in info.incomplete_files if str(f.file_path).startswith(repo.repo_path)]
            ready = repo.nb_files > 0 and not incomplete
            log.info(
                "hf cache %s: files=%d incomplete=%d -> %s",
                repo.repo_id, repo.nb_files, len(incomplete), "ready" if ready else "incomplete",
            )
            return ready

        log.warning("hf model %r not in local cache; first run will download it (needs network)", self.model_id)
        return False

    def _build_asr_pipeline(self):
        """Create the Whisper transcription pipeline for `self.model_id`.

        Kept as a separate method so tests can monkeypatch it and avoid loading a real
        model / touching CUDA. When the whole repo is already cached locally we load offline
        (no HEAD/GET requests to huggingface.co), which also enables an offline mode; otherwise
        we let transformers fetch what's missing on the first run.
        """
        pipeline = getattr(self, "pipeline", None)
        if pipeline is None:  # not set at init time or mocked out without the import
            from transformers import pipeline as pipeline

        # The explicit task is mandatory in offline mode (transformers 5.x cannot infer it
        # from a bare model id), and `local_files_only` leaks into model_kwargs there, so we
        # gate offlineness through HF_HUB_OFFLINE instead — same effect, no stray kwarg.
        if self._cache_ready():
            os.environ["HF_HUB_OFFLINE"] = "1"
            log.info("asr: loading %s from local cache (offline)", self.model_id)

        return pipeline(
            task="automatic-speech-recognition",
            model=self.model_id,
            device_map="auto",
            dtype="auto",
        )

    @staticmethod
    def _transcribe(pipeline, samples, lang: str | None = "en"):
        """Transcribe one float32 sample buffer through the ASR pipeline.

        `lang` is a Whisper language code ('en', 'ru', ...) or None to let Whisper guess.
        no_speech_threshold and logprob_threshold suppress low-confidence / non-speech clips;
        condition_on_prev_tokens=False keeps each clip independent (no cross-contamination).

        temperature is a tuple of fallback temperatures (per the podlodka-turbo doc) — without
        it transformers 5.x raises "'>' not supported between NoneType and float" inside
        _retrieve_avg_logprobs whenever logprob_threshold is set.
        """
        kwargs: dict = {
            "task": "transcribe",
            "no_speech_threshold": 0.6,
            "logprob_threshold": -1.0,
            "temperature": (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
            "condition_on_prev_tokens": False,
        }
        if lang is not None:
            kwargs["language"] = lang

        result = pipeline(
            samples,
            generate_kwargs=kwargs,
            return_timestamps=False,
        )
        text = ""
        if isinstance(result, dict) and "text" in result:
            text = str(result["text"]).strip()
        elif isinstance(result, list) and result:
            first = result[0]
            text = str(first.get("text", "")).strip() if isinstance(first, dict) else str(first).strip()
        return text

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

        pipeline = None
        try:
            pipeline = self._build_asr_pipeline()
        except Exception as e:
            log.error("failed to load ASR model %s for %s: %s", self.model_id, name, e)

        if pipeline is None:
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

        while True:
            clip = await queue.get()
            try:
                # Map --lang to a short code (en/ru/de/es/fr) whisper understands; skip the
                # hint entirely when we don't recognize the value, letting the model guess.
                lang_code = silero_language(self.lang)
                if lang_code == "auto":
                    lang_code = ""

                def _transcribe_clip() -> str:
                    return self._transcribe(pipeline, clip.to_numpy(), lang=lang_code or None)

                text = await asyncio.to_thread(_transcribe_clip)
                if not text:
                    log.debug("empty recognition for %s (uid=%s)", name, clip.uid.hex())
                    continue

                dialogue = self.dialogues.setdefault(name, [])
                dialogue.append(DialogueEntry(time=datetime.now(), is_student=True, text=text))
                log.info("recognized %s uid=%s: %r", name, clip.uid.hex(), text[:80])

                write, say, to_quit = await llm_reply(dialogue)
                if not (write or say):
                    continue

                # TTS of the "say" text; runs off-loop so synthesis does not block the event loop.
                tts_audio = b""
                if say and tts is not None:
                    try:
                        tts_audio = await asyncio.to_thread(tts.generate, say)
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
