import asyncio
import os
import pickle
import struct
import tempfile
from concurrent.futures import Future as _CFFuture
from concurrent.futures import ProcessPoolExecutor
from unittest import mock

import numpy as np
import pytest
from service import _WHISPER_CACHE
from service import AudioBuffer
from service import RecognitionService
from service import _worker_transcribe
from service import _worker_translate
from service import count_words
from service import growth_ok
from service import nllb_lang_code
from service import ratio_ok
from service import split_sentences
from service import top_word_freq
from service import validate_recognition
from service import word_diversity
from service import word_diversity_ok
from service import word_freq_ok
from service import word_ratio


def _fake_audio(n_samples=400):
    return b"".join(struct.pack("<f", 0.1) for _ in range(n_samples))


# Example from the log: the model loops on one word (words=226).
REPEAT_LOOP_TEXT = "И вы либо показываете мне, что там находится, или я вернусь с " + "правой " * 213 + "прав"

# A recognized sentence known to pass validate_recognition; used by the fake executor below.
RECOGNIZED_TEXT = "please review the documentation carefully before submitting your final report today"
TRANSLATED_TEXT = "переведённый текст результата распознавания"


class _FakeExecutor:
    """Stands in for ProcessPoolExecutor so process_segment runs without real models."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple]] = []
        self.results = {
            "_worker_transcribe": RECOGNIZED_TEXT,
            "_worker_translate": TRANSLATED_TEXT,
        }

    def submit(self, fn, *args):
        self.calls.append((fn.__name__, args))
        fut = _CFFuture()
        fut.set_result(self.results.get(fn.__name__, ""))
        return fut

    def shutdown(self, wait=False, cancel_futures=True) -> None:
        pass


class _SeqExecutor(_FakeExecutor):
    """Returns distinct recognized text per ASR call and echoes the translated input."""

    def __init__(self, texts) -> None:
        super().__init__()
        self._texts = list(texts)
        self._i = 0

    def submit(self, fn, *args):
        self.calls.append((fn.__name__, args))
        fut = _CFFuture()
        if fn.__name__ == "_worker_transcribe":
            text = self._texts[self._i] if self._i < len(self._texts) else ""
            self._i += 1
            fut.set_result(text)
        elif fn.__name__ == "_worker_translate":
            # echo the input so tests can verify what was translated.
            fut.set_result(f"TR[{args[2]}]")
        else:
            fut.set_result("")
        return fut


def test_word_ratio_sensible_text():
    text = "please review the documentation carefully before submitting your final report today"
    assert ratio_ok(word_ratio(text))


def test_word_ratio_letter_run_rejected():
    # model sometimes outputs long runs of a single letter: very low words/char ratio
    text = "aaaaaa" * 5
    assert not ratio_ok(word_ratio(text))


def test_word_ratio_english_high_ratio_passes():
    # function-word-heavy English sits above the old upper bound (0.203) but is valid text
    text = "Well, okay, I found the name Zool for you."
    assert ratio_ok(word_ratio(text))


def test_validate_recognition_english_high_ratio_passes():
    # these real segments were rejected at ratio ~0.21 with bounds [0.098..0.203]
    texts = (
        "Well, okay, I found the name Zool for you.",
        "You, I got hundreds of people dying to abuse me. I know you're a big celebrity now.",
    )
    for text in texts:
        ok, reason = validate_recognition(text, [])
        assert ok and not reason, f"{text!r}: {reason}"


def test_word_ratio_short_token_spam_rejected():
    # many distinct short tokens -> high word/char ratio above RATIO_MAX
    text = " ".join(chr(ord("a") + i) for i in range(26))  # a b c ... z, all distinct
    assert not ratio_ok(word_ratio(text))


def test_ratio_check_skipped_for_short_text():
    # short texts have a noisy ratio; the check is skipped
    text = "hi there how are you"  # 5 words, 20 chars
    ok, reason = validate_recognition(text, [])
    assert ok and not reason


def test_ratio_check_applied_for_long_text():
    # long texts are still checked; a letter run is rejected
    text = "aaaaaa" * 10  # 1 word, 60 chars
    ok, reason = validate_recognition(text, [])
    assert not ok and "ratio" in reason


def test_count_words_basic():
    assert count_words("") == 0
    assert count_words("one two   three") == 3


def test_growth_ok_no_history():
    assert growth_ok(12, []) is True


def test_growth_ok_within_limit():
    # latest adds ~5 words on top of previous total 6: within the limit
    assert growth_ok(5, [6]) is True


def test_growth_rejects_spike_over_previous_total():
    # repeated phrase inflates the newest recognition far above recorded earlier totals
    assert growth_ok(100, [8]) is False


def test_growth_ok_several_times_average():
    # several times the average is now allowed
    assert growth_ok(20, [2]) is True  # 10x the average


def test_growth_rejects_far_above_average():
    # far above the average is rejected
    assert growth_ok(100, [2]) is False  # 50x the average


def test_growth_ok_after_short_segments():
    # previous segments were short (silence/pause); a normal ~41-word phrase must not be
    # rejected just because the mean previous increment was tiny.
    assert growth_ok(41, [0, 2]) is True


def test_growth_still_rejects_spike_after_short_segments():
    # a genuine spike far above the floored baseline is still rejected.
    assert growth_ok(60, [0, 2]) is False  # > MIN_GROWTH_BASE * GROWTH_LIMIT_FACTOR (50)


def test_validate_recognition_pass_and_fail():
    text = "please review the documentation carefully before submitting your final report today"
    ok, reason = validate_recognition(text, [])
    assert ok and not reason

    bad_ratio_text = "aaaaaa" * 10  # 60 chars: long enough for the ratio check to apply
    ok2, reason2 = validate_recognition(bad_ratio_text, [])
    assert not ok2 and "ratio" in reason2


def test_repeat_loop_example_rejected():
    # the ratio check passes for this text (a normal-length word repeated), so the
    # word-frequency check is what catches the loop
    assert ratio_ok(word_ratio(REPEAT_LOOP_TEXT))
    ok, reason = validate_recognition(REPEAT_LOOP_TEXT, [])
    assert not ok and "frequency" in reason


def test_top_word_freq_basic():
    assert top_word_freq("") == 0.0
    assert top_word_freq("one two three") == pytest.approx(1 / 3)
    assert top_word_freq("a a a b") == pytest.approx(0.75)


def test_top_word_freq_strips_punctuation():
    # "мне," and "мне" are the same word after stripping edge punctuation
    assert top_word_freq("мне, мне мне") == pytest.approx(1.0)


def test_word_freq_ok_short_text_skipped():
    # short texts are skipped even if one word dominates
    assert word_freq_ok("да " * 9)  # 9 words, all the same


def test_word_freq_ok_rejects_dominant_word():
    text = "слово " * 30  # 30 words, all the same
    assert not word_freq_ok(text)


def test_word_freq_ok_normal_text():
    text = "please review the documentation carefully before submitting your final report today"
    assert word_freq_ok(text)


def test_word_diversity_metric():
    # all-unique text scores high, a looped short phrase scores low.
    assert word_diversity("one two three four") == pytest.approx(1.0)
    assert word_diversity("я не могу сказать что " * 30) < 0.1


def test_word_diversity_ok_short_text_skipped():
    # short texts are skipped even if a phrase repeats.
    assert word_diversity_ok("да нет да нет да") is True


def test_diversity_rejects_repeated_phrase():
    # whisper loops on a short phrase; each word alone stays under MAX_WORD_FREQ, but the
    # low unique/total ratio catches it (the log had words=165).
    text = "я не могу сказать что " * 30  # ~150 words from only 5 distinct ones
    ok, reason = validate_recognition(text, [])
    assert not ok and "diversity" in reason


def test_diversity_passes_normal_text():
    # a normal ~41-word dialogue has high word diversity and must pass.
    text = ("Welcome aboard I don't know where they get these guest conductors who think the audience "
            "wants to hear the same story every single night without any variation at all and honestly "
            "the whole thing feels a bit rehearsed to be completely frank with you")
    assert count_words(text) >= 41
    assert word_diversity_ok(text) is True


def test_audio_buffer_word_counts_per_group():
    buf = AudioBuffer(size=3)
    buf.add(_fake_audio())
    # cumulative totals recorded per element of the current group
    buf.set_word_count(1, 5)
    assert buf.word_count_at(1) == 5

    buf.add(_fake_audio())
    buf.set_word_count(2, 8)  # total words over two merged segments
    assert buf.word_count_at(1) == 5
    assert buf.word_count_at(2) == 8

    # group rolls over: previous cumulative totals no longer apply to the new group
    buf.add(_fake_audio())
    buf.add(_fake_audio())
    assert buf.group_id == 2
    assert buf.word_count_at(1) is None


def test_clear_resets_word_counts():
    buf = AudioBuffer(size=3)
    buf.add(_fake_audio())
    buf.set_word_count(1, 7)
    buf.clear()
    assert buf.word_count_at(1) is None


def test_audio_buffer_group_stable():
    buf = AudioBuffer(size=3)
    buf.add(_fake_audio())
    assert buf.group_id == 1
    assert buf.collect() == _fake_audio()

    buf.add(_fake_audio())
    assert buf.group_id == 1

    buf.add(_fake_audio())
    assert buf.group_id == 1

    # fourth addition -> new group
    buf.add(_fake_audio())
    assert buf.group_id == 2
    # only one slot filled after reset
    parts = [b for b in buf._slots if b is not None]
    assert len(parts) == 1


def test_audio_buffer_clear():
    buf = AudioBuffer(size=3)
    buf.add(_fake_audio())
    buf.clear()
    assert buf.group_id == 0
    assert buf.collect() == b""


def test_recognition_service_save_wav():
    svc = RecognitionService("test", model="", audio_out="")
    buf = AudioBuffer(size=3)
    buf.add(_fake_audio())
    with tempfile.TemporaryDirectory() as td:
        merged = buf.collect()
        svc.audio_out = td
        svc._save_wav_for_buffer(buf, merged)
        path = buf.file_name(td)
        assert path is not None
        assert os.path.isfile(path)


def test_audio_buffer_file_name():
    buf = AudioBuffer(size=3)
    buf.add(_fake_audio())
    with tempfile.TemporaryDirectory() as td:
        path = buf.file_name(td)
        assert path is not None
        assert os.path.basename(path) == "audio_001_segment_1_3.wav"


def test_audio_buffer_latest():
    # latest returns the most recently added segment, not the merged buffer.
    buf = AudioBuffer(size=3)
    assert buf.latest() == b""  # empty buffer

    seg_a = _fake_audio(400)
    buf.add(seg_a)
    assert buf.latest() == seg_a

    seg_b = _fake_audio(500)
    buf.add(seg_b)
    assert buf.latest() == seg_b
    assert buf.latest() != buf.collect()  # collect is both segments concatenated


def test_audio_buffer_add_and_collect_text():
    # recognized text is stored per slot and joined in fill order for translation.
    buf = AudioBuffer(size=3)
    assert buf.collect_text() == ""  # empty buffer

    buf.add(_fake_audio())
    buf.add_text(1, "hello world")
    buf.add(_fake_audio())
    buf.add_text(2, "how are you")
    assert buf.collect_text() == "hello world how are you"

    buf.clear_slots()
    assert buf.collect_text() == ""  # cleared along with the audio slots


def test_audio_buffer_clear_resets_text():
    # clear() also drops stored segment text.
    buf = AudioBuffer(size=3)
    buf.add(_fake_audio())
    buf.add_text(1, "some recognized words")
    buf.clear()
    assert buf.collect_text() == ""


def test_collect_text_join_and_out_of_bounds():
    # stored slots join in fill order; an index beyond size is ignored without corrupting state.
    buf = AudioBuffer(size=3)
    buf.add(_fake_audio())
    buf.add_text(1, "first")
    buf.add(_fake_audio())
    buf.add_text(2, "second")
    assert buf.collect_text() == "first second"

    # index beyond size: a no-op that leaves the existing slots untouched.
    buf.add_text(99, "out of bounds")
    assert buf.collect_text() == "first second"
    assert buf._text_slots[0] == "first"
    assert buf._text_slots[1] == "second"

    # clear resets the text slots too.
    buf.clear_slots()
    assert buf.collect_text() == ""


def test_split_sentences_basic():
    # split on terminal punctuation, keeping the delimiter attached to each sentence.
    assert split_sentences("And may I see this storage facility? No.") == [
        "And may I see this storage facility?", "No.",
    ]


def test_split_sentences_single_and_empty():
    # empty input -> []; no terminal punctuation -> the whole text is one sentence.
    assert split_sentences("") == []
    assert split_sentences("hello world how are you") == ["hello world how are you"]


def test_split_sentences_multiple_boundaries():
    assert split_sentences("First one. Second two! Third three? Fourth four.") == [
        "First one.", "Second two!", "Third three?", "Fourth four.",
    ]


@pytest.mark.asyncio
async def test_recognition_service_process_without_model():
    svc = RecognitionService("test", model="", from_lang="english", to_lang="english")
    await svc.process_segment("test", "0", _fake_audio(10))
    outbox = svc.outboxes["test"]
    group_id, text = await asyncio.wait_for(outbox.get(), timeout=1)
    assert group_id == 1
    assert text == ""


def test_worker_transcribe_passes_beam_size():
    # the beam width is forwarded to faster-whisper transcribe as beam_size.
    samples = np.frombuffer(_fake_audio(10), dtype="<f4")
    _WHISPER_CACHE.clear()

    def _transcribe(*args, **kwargs):
        return (iter([]), mock.Mock())

    fake_model = mock.Mock()
    fake_model.transcribe.side_effect = _transcribe
    try:
        with mock.patch("faster_whisper.WhisperModel", return_value=fake_model):
            _worker_transcribe("model", 4, samples)
        _, kwargs = fake_model.transcribe.call_args
        assert kwargs["beam_size"] == 4
    finally:
        _WHISPER_CACHE.clear()


def test_num_beams_clamped_to_one():
    # greedy (1) is the minimum beam width; negative values are clamped up.
    svc = RecognitionService("t", model="", num_beams=-2)
    assert svc.num_beams == 1


@pytest.mark.asyncio
async def test_recognition_service_reset():
    svc = RecognitionService("t", model="")
    await svc.process_segment("t", "0", _fake_audio())
    await svc.reset_session("t")
    buf = svc._buffers["t"]
    assert buf.group_id == 0
    assert buf.collect() == b""


@pytest.mark.asyncio
async def test_reset_clears_stale_audio_files():
    with tempfile.TemporaryDirectory() as td:
        svc = RecognitionService("t", model="", audio_out=td)
        # stale files from a previous session plus one unrelated file
        for f in ("audio_001_segment_1_3.wav", "audio_004_segment_3_3.wav", "other.txt"):
            open(os.path.join(td, f), "wb").write(b"stale")
        await svc.reset_session("t")
        names = sorted(os.listdir(td))
        assert names == ["other.txt"]


@pytest.mark.asyncio
async def test_reset_without_audio_dir_keeps_state_clean():
    svc = RecognitionService("t", model="")
    await svc.process_segment("t", "0", _fake_audio())
    outbox = svc.outboxes["t"]
    assert not outbox.empty()
    await svc.reset_session("t")
    # buffer and outbox both back to first-connection state
    assert svc._buffers["t"].group_id == 0
    assert outbox.empty()


def test_recognition_service_uses_process_pool():
    # recognition runs in a process pool so the model can use multiple VRAM instances
    svc = RecognitionService("t", model="", parallel=2)
    assert isinstance(svc._executor, ProcessPoolExecutor)
    assert svc._executor._max_workers == 2
    svc.shutdown()


def test_recognition_service_parallel_clamped_to_one():
    svc = RecognitionService("t", model="", parallel=0)
    assert svc.parallel == 1
    svc.shutdown()
    svc2 = RecognitionService("t", model="", parallel=-3)
    assert svc2.parallel == 1
    svc2.shutdown()


def test_worker_transcribe_is_picklable():
    # the worker must be a module-level function so ProcessPoolExecutor can ship it to a worker
    data = pickle.dumps(_worker_transcribe)
    assert pickle.loads(data) is _worker_transcribe


def test_worker_transcribe_caches_model_per_process():
    # the model is loaded once per worker process and reused for later segments
    samples = np.frombuffer(_fake_audio(10), dtype="<f4")
    _WHISPER_CACHE.clear()

    def _transcribe(*args, **kwargs):
        seg = mock.Mock()
        seg.text = " hello world "
        return (iter([seg]), mock.Mock())

    fake_model = mock.Mock()
    fake_model.transcribe.side_effect = _transcribe
    try:
        with mock.patch("faster_whisper.WhisperModel", return_value=fake_model) as loader:
            text1 = _worker_transcribe("model", 1, samples)
            text2 = _worker_transcribe("model", 1, samples)
        assert text1 == "hello world"
        assert text2 == "hello world"
        # loaded once, then served from the per-process cache
        assert loader.call_count == 1
        assert fake_model.transcribe.call_count == 2
    finally:
        _WHISPER_CACHE.clear()


def test_recognition_service_shutdown_stops_pool():
    svc = RecognitionService("t", model="", parallel=1)
    assert not svc._executor._shutdown_thread
    svc.shutdown()
    assert svc._executor._shutdown_thread


def test_worker_translate_is_picklable():
    # the worker must be a module-level function so ProcessPoolExecutor can ship it to a worker
    data = pickle.dumps(_worker_translate)
    assert pickle.loads(data) is _worker_translate


def test_nllb_lang_code_known_and_unknown():
    assert nllb_lang_code("russian") == "rus_Cyrl"
    assert nllb_lang_code("english") == "eng_Latn"
    # unknown names pass through unchanged (already an NLLB code)
    assert nllb_lang_code("rus_Cyrl") == "rus_Cyrl"


@pytest.mark.asyncio
async def test_process_segment_translates_when_configured():
    svc = RecognitionService(
        "test", model="asr-model", from_lang="english", to_lang="russian",
        translate_model="nllb-model",
    )
    fake = _FakeExecutor()
    svc._executor = fake
    await svc.process_segment("test", "0", _fake_audio(400))
    outbox = svc.outboxes["test"]
    group_id, text = await asyncio.wait_for(outbox.get(), timeout=1)
    assert group_id == 1
    # translation replaced the recognized text before sending
    assert text == TRANSLATED_TEXT
    names = [c[0] for c in fake.calls]
    assert "_worker_transcribe" in names and "_worker_translate" in names


@pytest.mark.asyncio
async def test_process_segment_no_translate_when_model_empty():
    svc = RecognitionService(
        "test", model="asr-model", from_lang="english", to_lang="russian",
        translate_model="",
    )
    fake = _FakeExecutor()
    svc._executor = fake
    await svc.process_segment("test", "0", _fake_audio(400))
    outbox = svc.outboxes["test"]
    group_id, text = await asyncio.wait_for(outbox.get(), timeout=1)
    assert group_id == 1
    # no translation: the recognized text is sent as-is
    assert text == RECOGNIZED_TEXT
    names = [c[0] for c in fake.calls]
    assert "_worker_translate" not in names


def test_should_translate_rule():
    # translation runs only with a translator model and a non-English target
    def svc(to_lang: str, translate_model: str = "nllb") -> RecognitionService:
        return RecognitionService("t", model="m", to_lang=to_lang, translate_model=translate_model)

    assert svc("russian")._should_translate() is True
    assert svc("russian", "")._should_translate() is False  # no translator model
    assert svc("english")._should_translate() is False      # English target: no-op
    assert svc("")._should_translate() is False             # empty target: no-op


@pytest.mark.asyncio
async def test_process_segment_no_translate_when_target_english():
    svc = RecognitionService(
        "test", model="asr-model", from_lang="russian", to_lang="english",
        translate_model="nllb-model",
    )
    fake = _FakeExecutor()
    svc._executor = fake
    await svc.process_segment("test", "0", _fake_audio(400))
    outbox = svc.outboxes["test"]
    group_id, text = await asyncio.wait_for(outbox.get(), timeout=1)
    assert group_id == 1
    # English target: the recognized text is sent as-is
    assert text == RECOGNIZED_TEXT
    names = [c[0] for c in fake.calls]
    assert "_worker_translate" not in names


@pytest.mark.asyncio
async def test_process_segment_no_translate_when_target_empty():
    svc = RecognitionService(
        "test", model="asr-model", from_lang="russian", to_lang="",
        translate_model="nllb-model",
    )
    fake = _FakeExecutor()
    svc._executor = fake
    await svc.process_segment("test", "0", _fake_audio(400))
    outbox = svc.outboxes["test"]
    group_id, text = await asyncio.wait_for(outbox.get(), timeout=1)
    assert group_id == 1
    # empty target: the recognized text is sent as-is
    assert text == RECOGNIZED_TEXT
    names = [c[0] for c in fake.calls]
    assert "_worker_translate" not in names


@pytest.mark.asyncio
async def test_process_segment_validation_disabled():
    # with validate=False a segment that would fail the quality checks is still sent.
    svc = RecognitionService("test", model="asr-model", from_lang="english", to_lang="english")
    svc.validate = False
    fake = _FakeExecutor()
    fake.results["_worker_transcribe"] = "aaaaaa" * 10  # letter run: fails the ratio check
    svc._executor = fake
    await svc.process_segment("test", "0", _fake_audio(400))
    outbox = svc.outboxes["test"]
    group_id, text = await asyncio.wait_for(outbox.get(), timeout=1)
    assert group_id == 1
    # the (bad) recognized text is sent as-is because validation was skipped.
    assert text == "aaaaaa" * 10


@pytest.mark.asyncio
async def test_process_segment_validation_enabled_rejects():
    # with validate=True (the default) the same bad segment is dropped, no output produced.
    svc = RecognitionService("test", model="asr-model", from_lang="english", to_lang="english")
    assert svc.validate is True
    fake = _FakeExecutor()
    fake.results["_worker_transcribe"] = "aaaaaa" * 10
    svc._executor = fake
    await svc.process_segment("test", "0", _fake_audio(400))
    outbox = svc.outboxes["test"]
    assert outbox.empty()


@pytest.mark.asyncio
async def test_process_segment_recognizes_latest_and_translates_accumulated():
    # ASR receives only the newest segment, while translation (and the outbox) carry
    # the full accumulated group text.
    svc = RecognitionService(
        "test", model="asr-model", from_lang="english", to_lang="russian",
        translate_model="nllb-model",
    )
    fake = _SeqExecutor([
        "the teacher explained the new lesson clearly",
        "students asked several questions about it later",
    ])
    svc._executor = fake

    await svc.process_segment("test", "0", _fake_audio(400))
    first = await asyncio.wait_for(svc.outboxes["test"].get(), timeout=1)

    await svc.process_segment("test", "0", _fake_audio(500))
    second = await asyncio.wait_for(svc.outboxes["test"].get(), timeout=1)

    # ASR calls: the second must receive only segment B (500 samples), not A+B (900).
    transcribe_calls = [c for c in fake.calls if c[0] == "_worker_transcribe"]
    assert len(transcribe_calls) == 2
    first_samples, second_samples = transcribe_calls[0][1][2], transcribe_calls[1][1][2]
    assert first_samples.size == 400
    assert second_samples.size == 500

    # Step 1 carries only segment A; step 2 accumulates both segments (translated).
    _, first_msg = first
    assert "the teacher explained the new lesson clearly" in first_msg
    assert "students asked several questions about it later" not in first_msg
    _, msg_text = second
    assert "the teacher explained the new lesson clearly" in msg_text
    assert "students asked several questions about it later" in msg_text


@pytest.mark.asyncio
async def test_process_segment_empty_recognition_keeps_slot_aligned():
    # a silent segment (empty recognition) still records an empty entry for its audio slot.
    svc = RecognitionService("test", model="asr-model", from_lang="english", to_lang="english")
    fake = _SeqExecutor(["", "the teacher explained the new lesson clearly"])
    svc._executor = fake

    await svc.process_segment("test", "0", _fake_audio(400))  # silence -> empty text
    await svc.process_segment("test", "0", _fake_audio(500))  # valid

    buf = svc._buffers["test"]
    assert buf.filled_count() == 2
    assert buf._text_slots[0] == ""  # explicit empty entry keeps the slot aligned with its audio
    assert buf._text_slots[1] == "the teacher explained the new lesson clearly"
    assert buf.collect_text() == "the teacher explained the new lesson clearly"


@pytest.mark.asyncio
async def test_process_segment_rejected_keeps_slot_aligned():
    # a segment rejected by validation records an empty entry so text slots stay in lockstep.
    svc = RecognitionService("test", model="asr-model", from_lang="english", to_lang="english")
    fake = _SeqExecutor(["aaaaaa" * 10, "the teacher explained the new lesson clearly"])
    svc._executor = fake

    await svc.process_segment("test", "0", _fake_audio(400))  # letter run -> rejected
    await svc.process_segment("test", "0", _fake_audio(500))  # valid

    buf = svc._buffers["test"]
    assert buf.filled_count() == 2
    assert buf._text_slots[0] == ""  # rejected segment: empty entry keeps the slot aligned
    assert buf._text_slots[1] == "the teacher explained the new lesson clearly"


@pytest.mark.asyncio
async def test_process_segment_translates_per_sentence_with_cache():
    # NLLB only translates a single sentence reliably, so the group text is split into
    # sentences and each translated; already-translated sentences are cached per group.
    svc = RecognitionService(
        "test", model="asr-model", from_lang="english", to_lang="russian",
        translate_model="nllb-model",
    )
    fake = _SeqExecutor([
        "And may I see this storage facility?",
        "No. And why not, Mr. Beckman?",
    ])
    svc._executor = fake

    await svc.process_segment("test", "0", _fake_audio(400))
    first_msg = (await asyncio.wait_for(svc.outboxes["test"].get(), timeout=1))[1]
    assert first_msg == "TR[And may I see this storage facility?]"

    await svc.process_segment("test", "0", _fake_audio(500))
    second_msg = (await asyncio.wait_for(svc.outboxes["test"].get(), timeout=1))[1]
    # every sentence of the accumulated group is present, in order.
    assert second_msg == "TR[And may I see this storage facility?] TR[No.] TR[And why not, Mr.] TR[Beckman?]"

    # caching: the first sentence is translated only once across both steps (step 2 reuses it).
    translate_inputs = [c[1][2] for c in fake.calls if c[0] == "_worker_translate"]
    assert translate_inputs.count("And may I see this storage facility?") == 1


@pytest.mark.asyncio
async def test_translation_cache_resets_on_new_group():
    # the per-group sentence cache is dropped when the buffer rolls over to a new group.
    svc = RecognitionService(
        "test", model="asr-model", from_lang="english", to_lang="russian",
        translate_model="nllb-model",
    )
    fake = _SeqExecutor([
        "First sentence here.",
        "Second sentence here.",
        "Third sentence here.",
        "Fourth sentence here.",  # fourth add rolls the size-3 buffer into a new group
    ])
    svc._executor = fake

    for _ in range(4):
        await svc.process_segment("test", "0", _fake_audio(400))
        await asyncio.wait_for(svc.outboxes["test"].get(), timeout=1)

    buf = svc._buffers["test"]
    assert buf.group_id == 2
    # previous group's sentences were cleared on rollover; only the new group's sentence remains.
    for old in ("First sentence here.", "Second sentence here.", "Third sentence here."):
        assert old not in buf._translations
    assert set(buf._translations) == {"Fourth sentence here."}
