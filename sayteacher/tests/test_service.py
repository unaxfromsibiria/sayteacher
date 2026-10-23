import asyncio
import pickle
import struct
import uuid as _uuid
import wave
from concurrent.futures import Future as _CFFuture
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from io import BytesIO
from unittest import mock

import numpy as np
import pytest

from sayteacher.common import QUIT_WORD
from sayteacher.common import DialogueEntry
from sayteacher.service import _TTS_CACHE
from sayteacher.service import _WHISPER_CACHE
from sayteacher.service import RecognitionService
from sayteacher.service import SpeechClip
from sayteacher.service import TtsService
from sayteacher.service import _worker_transcribe
from sayteacher.service import _worker_tts


def _clip(n_samples: int, value: float = 0.25) -> bytes:
    return b"".join(struct.pack("<f", value) for _ in range(n_samples))


# ---------------------------------------------------------------------------
# SpeechClip tests (unchanged behaviour)
# ---------------------------------------------------------------------------

def test_speech_clip_rejects_non_f32_length():
    with pytest.raises(ValueError, match="multiple of 4"):
        SpeechClip(bytes(16), b"\x01\x02\x03")


def test_speech_clip_duration_and_numpy_shape():
    clip = SpeechClip(bytes(16), _clip(16_000))
    assert clip.num_samples == 16_000
    assert clip.duration_seconds == pytest.approx(1.0)

    arr = clip.to_numpy()
    assert len(arr) == 16_000
    assert list(arr[:2]) == [pytest.approx(0.25), pytest.approx(0.25)]


def test_speech_clip_wav_roundtrip():
    samples = [0.5, -0.5, 0.25, -0.75]
    raw = b"".join(struct.pack("<f", s) for s in samples)
    clip = SpeechClip(bytes(16), raw, sample_rate=8_000)

    with wave.open(BytesIO(clip.wav_bytes()), "rb") as w:
        assert w.getnchannels() == 1
        assert w.getsampwidth() == 2
        assert w.getframerate() == 8_000
        assert w.getnframes() == len(samples)

        frames = w.readframes(len(samples))
        pcm = [s / 32767.0 for s in struct.unpack(f"<{len(samples)}h", frames)]
        # 16-bit quantization error <= 1/65536; use a small tolerance instead.
        for got, want in zip(pcm, samples):
            assert abs(got - want) < 2.0 / 65536


# ---------------------------------------------------------------------------
# send() tests (unchanged behaviour)
# ---------------------------------------------------------------------------

def test_send_lesson_is_keyed_by_student_name():
    api = RecognitionService("127.0.0.1:16050")
    uid = _uuid.uuid4().bytes

    asyncio.run(api.send("127.0.0.1:16050", uid, "раз два три", b""))

    assert api.lessons["127.0.0.1:16050"] == "раз два три"
    assert uid.hex() not in api.lessons
    assert api.students["127.0.0.1:16050"].qsize() == 0
    api.shutdown()


def test_send_audio_is_queued_for_named_student():
    api = RecognitionService("a", "b")
    uid = _uuid.uuid4().bytes

    asyncio.run(api.send("b", uid, "", _clip(1_600)))

    clip = api.students["b"].get_nowait()
    assert isinstance(clip, SpeechClip)
    assert clip.uid == uid
    assert clip.num_samples == 1_600
    assert api.students["a"].qsize() == 0
    api.shutdown()


def test_send_unknown_student_raises():
    api = RecognitionService("known")
    with pytest.raises(KeyError):
        asyncio.run(api.send("unknown", bytes(16), "", _clip(4)))
    api.shutdown()


# ---------------------------------------------------------------------------
# Helpers for async tests
# ---------------------------------------------------------------------------

async def _wait_until(predicate, timeout: float = 2.0) -> bool:
    """Poll a predicate on the event loop until true or the deadline passes."""
    import time

    end = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() >= end:
            return False
        await asyncio.sleep(0.01)
    return True


async def _stop(worker) -> None:
    worker.cancel()
    try:
        await worker
    except (asyncio.CancelledError, StopIteration):
        pass


# ---------------------------------------------------------------------------
# Fake executor for integration tests of process()
# ---------------------------------------------------------------------------

class _FakeExecutor:
    """Stands in for ProcessPoolExecutor so process() runs without real models."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple]] = []
        self._transcribe_replies: list[str] = ["привет", "как дела?"]
        self._tts_result = b"\x01\x02\x03\x04"

    def submit(self, fn, *args):
        name = getattr(fn, "__name__", str(fn))
        self.calls.append((name, args))
        fut = _CFFuture()
        if name == "_worker_transcribe":
            idx = sum(1 for c in self.calls if c[0] == "_worker_transcribe") - 1
            text = self._transcribe_replies[idx] if idx < len(self._transcribe_replies) else ""
            fut.set_result(text)
        elif name == "_worker_tts":
            fut.set_result(self._tts_result)
        else:
            fut.set_result("")
        return fut

    def shutdown(self, wait=False, cancel_futures=True) -> None:
        pass


# ---------------------------------------------------------------------------
# process() integration tests (using fake executor)
# ---------------------------------------------------------------------------

def test_process_unknown_student_raises():
    api = RecognitionService()
    with pytest.raises(KeyError):
        asyncio.run(api.process("nobody"))
    api.shutdown()


def test_process_without_model_drains_clips_and_fills_nothing():
    api = RecognitionService("s", model="")

    async def scenario():
        worker = asyncio.create_task(api.process("s"))
        await api.send("s", bytes(16), "", _clip(3_200))
        drained = await _wait_until(lambda: api.students["s"].qsize() == 0)
        assert drained, "worker did not drain the clip without a model"
        await _stop(worker)

    asyncio.run(scenario())
    assert api.dialogues.get("s", []) == []
    api.shutdown()


def test_process_recognizes_clips_into_dialogue():
    api = RecognitionService("s", model="some/model")
    fake = _FakeExecutor()
    api._executor = fake

    async def scenario():
        worker = asyncio.create_task(api.process("s"))
        for i in range(2):
            await api.send("s", bytes.fromhex(f"{i:032x}"), "", _clip(1_600))

        done = await _wait_until(lambda: len(api.dialogues.get("s", [])) == 2)
        assert done, f"expected 2 dialogue entries, got {api.dialogues.get('s')}"
        await _stop(worker)

    asyncio.run(scenario())

    entries = api.dialogues["s"]
    assert [e.text for e in entries] == ["привет", "как дела?"]
    assert all(e.is_student is True for e in entries)
    # both ASR calls went through the executor
    asr_calls = [c for c in fake.calls if c[0] == "_worker_transcribe"]
    assert len(asr_calls) == 2


def test_process_empty_recognition_is_not_stored():
    api = RecognitionService("s", model="some/model")
    fake = _FakeExecutor()
    fake._transcribe_replies = [""]
    api._executor = fake

    async def scenario():
        worker = asyncio.create_task(api.process("s"))
        await api.send("s", bytes(16), "", _clip(800))
        drained = await _wait_until(lambda: api.students["s"].qsize() == 0)
        assert drained, "worker did not consume the clip"
        await _stop(worker)

    asyncio.run(scenario())
    assert api.dialogues.get("s", []) == []


def test_process_survives_worker_errors():
    """A worker that raises must not kill the process loop."""
    api = RecognitionService("s", model="some/model")

    class _ErrorExecutor(_FakeExecutor):
        def submit(self, fn, *args):
            name = getattr(fn, "__name__", str(fn))
            self.calls.append((name, args))
            fut = _CFFuture()
            if name == "_worker_transcribe":
                idx = sum(1 for c in self.calls if c[0] == "_worker_transcribe") - 1
                if idx == 0:
                    fut.set_exception(RuntimeError("cuda oom"))
                else:
                    fut.set_result("второе")
            else:
                fut.set_result("")
            return fut

    fake = _ErrorExecutor()
    api._executor = fake

    async def scenario():
        worker = asyncio.create_task(api.process("s"))
        await api.send("s", bytes(16), "", _clip(800))
        await api.send("s", b"\x02" * 16, "", _clip(800))

        done = await _wait_until(lambda: len(api.dialogues.get("s", [])) == 1)
        assert done, f"worker stopped after an error: {api.dialogues.get('s')}"
        await _stop(worker)

    asyncio.run(scenario())
    assert [e.text for e in api.dialogues["s"]] == ["второе"]


# ---------------------------------------------------------------------------
# reset_session tests (unchanged behaviour)
# ---------------------------------------------------------------------------

def test_reset_session_clears_dialogue_and_queues():
    api = RecognitionService("s", model="some/model")
    api.dialogues["s"] = [DialogueEntry(time=datetime.now(), is_student=True, text="old turn")]
    api.lessons["s"] = "lesson text"  # lesson survives: it arrives again on reconnect
    api.students["s"].put_nowait(SpeechClip(bytes(16), _clip(800)))
    api.outboxes["s"].put_nowait((100, "stale answer", b""))
    api.quit_msg.put_nowait(QUIT_WORD)
    api.quit_hint = True

    asyncio.run(api.reset_session("s"))

    assert api.dialogues.get("s") == []  # cleared in place, key stays for a live worker reference
    assert api.lessons["s"] == "lesson text"
    assert api.students["s"].qsize() == 0
    assert api.outboxes["s"].qsize() == 0
    assert api.quit_msg.empty()
    assert api.quit_hint is False
    api.shutdown()


def test_reset_session_cancels_pending_timer_tasks():
    async def scenario():
        api = RecognitionService("s", model="some/model")
        api._start_duration_timer(60.0)  # one-shot task that would put QUIT_WORD after an hour
        assert api._duration_timer_task is not None and not api._duration_timer_task.done()

        await api.reset_session("s")

        assert api._duration_timer_task is None
        assert api.quit_msg.empty()
        api.shutdown()
        return True

    assert asyncio.run(scenario())


def test_reset_session_unknown_student_is_a_noop():
    api = RecognitionService("s", model="some/model")
    asyncio.run(api.reset_session("unknown"))  # must not raise
    api.shutdown()


# ---------------------------------------------------------------------------
# ProcessPoolExecutor / shutdown tests
# ---------------------------------------------------------------------------

def test_recognition_service_uses_process_pool():
    svc = RecognitionService("t", model="")
    assert isinstance(svc._executor, ProcessPoolExecutor)
    svc.shutdown()


def test_recognition_service_shutdown_stops_pool():
    svc = RecognitionService("t", model="")
    assert not svc._executor._shutdown_thread
    svc.shutdown()
    assert svc._executor._shutdown_thread


# ---------------------------------------------------------------------------
# _worker_transcribe unit tests
# ---------------------------------------------------------------------------

def test_worker_transcribe_is_picklable():
    """The worker must be a module-level function so ProcessPoolExecutor can ship it."""
    data = pickle.dumps(_worker_transcribe)
    assert pickle.loads(data) is _worker_transcribe


def test_worker_transcribe_caches_model_per_process():
    """The model is loaded once per worker process and reused for later segments."""
    samples = np.frombuffer(_clip(10), dtype="<f4")
    _WHISPER_CACHE.clear()

    def _transcribe(*args, **kwargs):
        seg = mock.Mock()
        seg.text = " hello world "
        return (iter([seg]), mock.Mock())

    fake_model = mock.Mock()
    fake_model.transcribe.side_effect = _transcribe
    try:
        with mock.patch("faster_whisper.WhisperModel", return_value=fake_model) as loader:
            text1 = _worker_transcribe("model", None, samples)
            text2 = _worker_transcribe("model", None, samples)
        assert text1 == "hello world"
        assert text2 == "hello world"
        # loaded once, then served from the per-process cache
        assert loader.call_count == 1
        assert fake_model.transcribe.call_count == 2
    finally:
        _WHISPER_CACHE.clear()


def test_worker_transcribe_passes_language():
    """The lang parameter is forwarded to transcribe as language= when non-empty."""
    samples = np.frombuffer(_clip(10), dtype="<f4")
    _WHISPER_CACHE.clear()

    def _transcribe(*args, **kwargs):
        return (iter([]), mock.Mock())

    fake_model = mock.Mock()
    fake_model.transcribe.side_effect = _transcribe
    try:
        with mock.patch("faster_whisper.WhisperModel", return_value=fake_model):
            _worker_transcribe("model", "ru", samples)
        _, kwargs = fake_model.transcribe.call_args
        assert kwargs["language"] == "ru"
    finally:
        _WHISPER_CACHE.clear()


def test_worker_transcribe_no_language_when_empty():
    """When lang is None or empty, the language kwarg is not passed."""
    samples = np.frombuffer(_clip(10), dtype="<f4")
    _WHISPER_CACHE.clear()

    def _transcribe(*args, **kwargs):
        return (iter([]), mock.Mock())

    fake_model = mock.Mock()
    fake_model.transcribe.side_effect = _transcribe
    try:
        with mock.patch("faster_whisper.WhisperModel", return_value=fake_model):
            _worker_transcribe("model", None, samples)
        _, kwargs = fake_model.transcribe.call_args
        assert "language" not in kwargs
    finally:
        _WHISPER_CACHE.clear()


# ---------------------------------------------------------------------------
# _worker_tts unit tests
# ---------------------------------------------------------------------------

def test_worker_tts_is_picklable():
    """The worker must be a module-level function so ProcessPoolExecutor can ship it."""
    data = pickle.dumps(_worker_tts)
    assert pickle.loads(data) is _worker_tts


def test_worker_tts_empty_text_returns_empty_bytes():
    assert _worker_tts("v3_en", "en_101", "en", "") == b""


def test_worker_tts_synthesizes_and_resamples():
    """Mock silero_tts; verify synthesis is called and output is f32 LE bytes."""
    _TTS_CACHE.clear()

    class _FakeTensor:
        def __init__(self):
            self._data = np.array([0.5, -0.5, 0.25, -0.75], dtype="<f4")

        def dim(self):
            return 1

        def numel(self):
            return len(self._data)

        def detach(self):
            return self

        def float(self):
            return self

        def cpu(self):
            return self

        def flatten(self):
            return self

        def contiguous(self):
            return self

        def numpy(self):
            return self._data

    fake_tensor = _FakeTensor()
    fake_model = mock.Mock()
    fake_model.speakers = ["en_101", "en_37"]
    fake_model.apply_tts.return_value = fake_tensor
    fake_model.to.return_value = None

    # Identity resample: pass through the tensor unchanged.
    class _IdentityResample:
        def __init__(self, src_rate, dst_rate):
            pass

        def __call__(self, samples):
            return samples

    try:
        with mock.patch("silero.silero_tts", return_value=(fake_model,)) as loader, \
             mock.patch("torchaudio.transforms.Resample", _IdentityResample):
            result = _worker_tts("v3_en", "en_101", "en", "hello")
        assert loader.call_count == 1
        # apply_tts called with the specific voice
        fake_model.apply_tts.assert_called_once()
        call_kwargs = fake_model.apply_tts.call_args.kwargs
        assert call_kwargs["speaker"] == "en_101"
        assert call_kwargs["sample_rate"] == 24_000
        # result is f32 LE bytes (resampled from 24k to 16k)
        assert len(result) > 0
        assert len(result) % 4 == 0
    finally:
        _TTS_CACHE.clear()


def test_worker_tts_caches_model_per_process():
    """The TTS model is loaded once per worker process."""
    _TTS_CACHE.clear()

    class _FakeTensor:
        def dim(self):
            return 1

        def numel(self):
            return 4

        def detach(self):
            return self

        def float(self):
            return self

        def cpu(self):
            return self

        def flatten(self):
            return self

        def contiguous(self):
            return self

        def numpy(self):
            return np.array([0.1, 0.2, 0.3, 0.4], dtype="<f4")

    fake_model = mock.Mock()
    fake_model.speakers = ["en_101"]
    fake_model.apply_tts.return_value = _FakeTensor()
    fake_model.to.return_value = None

    try:
        with mock.patch("silero.silero_tts", return_value=(fake_model,)) as loader:
            _worker_tts("v3_en", "en_101", "en", "hello")
            _worker_tts("v3_en", "en_101", "en", "world")
        assert loader.call_count == 1
        assert fake_model.apply_tts.call_count == 2
    finally:
        _TTS_CACHE.clear()


# ---------------------------------------------------------------------------
# TtsService.worker_params tests
# ---------------------------------------------------------------------------

def test_tts_service_worker_params_voice_id():
    """When --speaker is a voice id (not a bundle), the default bundle for the lang is used."""
    tts = TtsService(lang="English", speaker="en_101")
    bundle, voice, language = tts.worker_params()
    assert bundle == "v3_en"
    assert voice == "en_101"
    assert language == "en"


def test_tts_service_worker_params_bundle_id():
    """When --speaker is a known bundle id, it is used directly and voice is empty."""
    tts = TtsService(lang="English", speaker="v3_en")
    bundle, voice, language = tts.worker_params()
    assert bundle == "v3_en"
    assert voice == ""
    assert language == "en"


def test_tts_service_worker_params_russian():
    tts = TtsService(lang="Russian", speaker="xenia")
    bundle, voice, language = tts.worker_params()
    assert bundle == "v5_ru"
    assert voice == "xenia"
    assert language == "ru"
