import asyncio
import struct
import uuid as _uuid
import wave
from datetime import datetime
from io import BytesIO
from types import MethodType

import pytest

from sayteacher.common import QUIT_WORD
from sayteacher.common import DialogueEntry
from sayteacher.service import RecognitionService
from sayteacher.service import SpeechClip


def _clip(n_samples: int, value: float = 0.25) -> bytes:
    return b"".join(struct.pack("<f", value) for _ in range(n_samples))


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


def test_send_lesson_is_keyed_by_student_name():
    api = RecognitionService("127.0.0.1:16050")
    uid = _uuid.uuid4().bytes

    asyncio.run(api.send("127.0.0.1:16050", uid, "раз два три", b""))

    # Lesson stored under the student name (the connection), not the session id.
    assert api.lessons["127.0.0.1:16050"] == "раз два три"
    assert uid.hex() not in api.lessons
    assert api.students["127.0.0.1:16050"].qsize() == 0


def test_send_audio_is_queued_for_named_student():
    api = RecognitionService("a", "b")
    uid = _uuid.uuid4().bytes

    asyncio.run(api.send("b", uid, "", _clip(1_600)))

    # Only the named student receives the clip.
    clip = api.students["b"].get_nowait()
    assert isinstance(clip, SpeechClip)
    assert clip.uid == uid
    assert clip.num_samples == 1_600
    assert api.students["a"].qsize() == 0


def test_send_unknown_student_raises():
    api = RecognitionService("known")
    with pytest.raises(KeyError):
        asyncio.run(api.send("unknown", bytes(16), "", _clip(4)))


async def _wait_until(predicate, timeout: float = 1.0) -> bool:
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


class FakePipeline:
    """Stand-in for a transformers pipeline; returns fixed text and records calls."""

    def __init__(self, replies) -> None:
        self.replies = list(replies)
        self.calls = []

    def __call__(self, samples, generate_kwargs=None, return_timestamps=None):
        self.calls.append(samples)
        if not self.replies:
            raise AssertionError("FakePipeline exhausted")
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return {"text": reply}


def test_process_unknown_student_raises():
    api = RecognitionService()
    with pytest.raises(KeyError):
        asyncio.run(api.process("nobody"))


def test_process_without_model_drains_clips_and_fills_nothing():
    api = RecognitionService("s", model="")
    built = []

    def should_not_build(self):
        built.append(1)
        return None

    # patching is enough to prove the lazy builder never runs for an empty model id.
    api._build_asr_pipeline = MethodType(should_not_build, api)

    async def scenario():
        worker = asyncio.create_task(api.process("s"))
        await api.send("s", bytes(16), "", _clip(3_200))
        # send() must not block while the worker drains.
        drained = await _wait_until(lambda: api.students["s"].qsize() == 0)
        assert drained, "worker did not drain the clip without a model"
        await _stop(worker)

    asyncio.run(scenario())

    assert built == []
    assert api.dialogues.get("s", []) == []


def test_process_recognizes_clips_into_dialogue(monkeypatch):
    api = RecognitionService("s", model="some/model")
    pipeline = FakePipeline(["привет", "как дела?"])

    monkeypatch.setattr(api, "_build_asr_pipeline", lambda: pipeline)

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
    # every entry is a DialogueEntry namedtuple with the student flag set.
    assert all(e.is_student is True for e in entries)
    assert len(pipeline.calls) == 2


def test_process_survives_pipeline_errors(monkeypatch):
    api = RecognitionService("s", model="some/model")
    pipeline = FakePipeline([RuntimeError("cuda oom"), "второе"])

    monkeypatch.setattr(api, "_build_asr_pipeline", lambda: pipeline)

    async def scenario():
        worker = asyncio.create_task(api.process("s"))
        await api.send("s", bytes(16), "", _clip(800))
        await api.send("s", b"\x02" * 16, "", _clip(800))

        # the first clip raises inside the pipeline but the worker keeps going and
        # still recognizes the second one — a failure must not kill the task.
        done = await _wait_until(lambda: len(api.dialogues.get("s", [])) == 1)
        assert done, f"worker stopped after an error: {api.dialogues.get('s')}"
        await _stop(worker)

    asyncio.run(scenario())

    assert [e.text for e in api.dialogues["s"]] == ["второе"]


def test_process_empty_recognition_is_not_stored(monkeypatch):
    api = RecognitionService("s", model="some/model")

    monkeypatch.setattr(api, "_build_asr_pipeline", lambda: FakePipeline([""]))

    async def scenario():
        worker = asyncio.create_task(api.process("s"))
        await api.send("s", bytes(16), "", _clip(800))
        drained = await _wait_until(lambda: api.students["s"].qsize() == 0)
        assert drained, "worker did not consume the clip"
        await _stop(worker)

    asyncio.run(scenario())

    assert api.dialogues.get("s", []) == []


def test_process_build_failure_still_drains(monkeypatch):
    api = RecognitionService("s", model="some/model")

    def boom():
        raise RuntimeError("model load failed")

    monkeypatch.setattr(api, "_build_asr_pipeline", boom)

    async def scenario():
        worker = asyncio.create_task(api.process("s"))
        await api.send("s", bytes(16), "", _clip(800))
        drained = await _wait_until(lambda: api.students["s"].qsize() == 0)
        assert drained, "worker did not drain after a build failure"
        await _stop(worker)

    asyncio.run(scenario())

    assert api.dialogues.get("s", []) == []


def test_transcribe_handles_list_and_dict_results():
    class _Fixed:
        def __init__(self, out):
            self._out = out

        def __call__(self, *args, **kwargs):
            return self._out

    samples = _clip(16)
    assert RecognitionService._transcribe(_Fixed({"text": "  привет  "}), samples).strip() == "привет"
    assert RecognitionService._transcribe(_Fixed([{"text": "в списке"}]), samples) == "в списке"


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


def test_reset_session_cancels_pending_timer_tasks():
    async def scenario():
        api = RecognitionService("s", model="some/model")
        api._start_duration_timer(60.0)  # one-shot task that would put QUIT_WORD after an hour
        assert api._duration_timer_task is not None and not api._duration_timer_task.done()

        await api.reset_session("s")

        assert api._duration_timer_task is None
        assert api.quit_msg.empty()
        return True

    assert asyncio.run(scenario())


def test_reset_session_unknown_student_is_a_noop():
    api = RecognitionService("s", model="some/model")
    asyncio.run(api.reset_session("unknown"))  # must not raise

