import asyncio
from concurrent.futures import Future as _CFFuture


def _clip(n_samples: int) -> bytes:
    import struct

    return b"".join(struct.pack("<f", 0.25) for _ in range(n_samples))


async def _wait_until(predicate, timeout: float = 1.0) -> bool:
    loop = asyncio.get_event_loop()
    end = loop.time() + timeout
    while not predicate():
        if loop.time() >= end:
            return False
        await asyncio.sleep(0.01)
    return True


async def _stop(worker):
    worker.cancel()
    try:
        await worker
    except (asyncio.CancelledError, StopIteration):
        pass


class FakeTts:
    """Duck-typed stand-in for TtsService; records text and returns canned audio."""

    def __init__(self) -> None:
        self.calls = []

    def generate(self, text: str) -> bytes:
        self.calls.append(text)
        return b"audio-bytes"


class _FakeExecutor:
    """Stands in for ProcessPoolExecutor; returns canned results for transcribe and tts."""

    def __init__(self, transcribe_replies=None, tts_result=b"", tts_exc=None) -> None:
        self._transcribe = list(transcribe_replies or ["привет"])
        self._i = 0
        self._tts_result = tts_result
        self._tts_exc = tts_exc
        self.calls: list[tuple[str, tuple]] = []

    def submit(self, fn, *args):
        name = getattr(fn, "__name__", str(fn))
        self.calls.append((name, args))
        fut = _CFFuture()
        if name == "_worker_transcribe":
            text = self._transcribe[self._i] if self._i < len(self._transcribe) else ""
            self._i += 1
            fut.set_result(text)
        elif name == "_worker_tts":
            if self._tts_exc:
                fut.set_exception(self._tts_exc)
            else:
                fut.set_result(self._tts_result)
        else:
            fut.set_result("")
        return fut

    def shutdown(self, wait=False, cancel_futures=True) -> None:
        pass


def test_silero_language_mapping_from_names_and_codes():
    from sayteacher.common import silero_language

    assert silero_language("English") == "en"
    assert silero_language("english") == "en"
    assert silero_language("Russian") == "ru"
    assert silero_language("German") == "de"
    assert silero_language("Spanish") == "es"
    assert silero_language("French") == "fr"
    # Short codes pass through as-is.
    assert silero_language("en") == "en"
    assert silero_language("ru") == "ru"


def test_silero_language_mapping_uses_iso3166_countries():
    from sayteacher.common import silero_language

    # Country names resolved via iso3166 alpha2 (see requirements.txt).
    assert silero_language("United States of America") == "en"
    assert silero_language("Germany") == "de"


def test_silero_language_unknown_returns_auto():
    from sayteacher.common import silero_language

    assert silero_language("") == "auto"
    assert silero_language("Klingon") == "auto"


def test_tts_enabled_and_disabled_states():
    from sayteacher.service import TtsService

    # No language and no speaker: nothing to synthesize.
    assert TtsService(lang="", speaker="").enabled is False
    assert TtsService(lang="Klingon", speaker="").enabled is False

    # A known --lang enables it even without a named speaker; so does any speaker id.
    assert TtsService(lang="English", speaker="").enabled is True
    assert TtsService(lang="Klingon", speaker="v3_en").enabled is True


def test_tts_worker_params_resolves_bundle_and_voice():
    from sayteacher.service import TtsService

    # English with no specific voice -> default bundle, empty voice.
    tts = TtsService(lang="English", speaker="")
    assert tts.worker_params() == ("v3_en", "", "en")

    # Russian with a named voice.
    tts = TtsService(lang="Russian", speaker="ru_12")
    bundle, voice, language = tts.worker_params()
    assert language == "ru"
    assert voice == "ru_12"
    assert bundle != ""  # some valid Russian bundle


def test_tts_worker_params_bundle_id_as_speaker():
    from sayteacher.service import TtsService

    # When --speaker is a known bundle id, it becomes the bundle and voice stays empty.
    tts = TtsService(lang="English", speaker="v3_en")
    assert tts.worker_params() == ("v3_en", "", "en")


def test_process_puts_write_and_tts_audio_into_outbox(monkeypatch):
    import sayteacher.service as svc_mod
    from sayteacher.service import LlmAnswerService
    from sayteacher.service import RecognitionService

    api = RecognitionService("s", model="some/model")
    # Fake executor: transcribe returns "привет", tts returns audio bytes.
    api._executor = _FakeExecutor(transcribe_replies=["привет"], tts_result=b"audio-bytes")

    from test_llm import FakeOpenAIClient

    monkeypatch.setattr(
        svc_mod, "AsyncOpenAI",
        lambda base_url=None, **kw: FakeOpenAIClient(base_url=base_url, payload={"say": "teacher hi", "write": "hi!"}),
    )
    llm = LlmAnswerService("http://x", "m")

    from sayteacher.service import TtsService
    tts = TtsService(lang="English", speaker="v3_en")

    async def scenario():
        worker = asyncio.create_task(api.process("s", llm_api=llm, tts=tts))
        await api.send("s", b"\x01" * 16, "", _clip(800))
        done = await _wait_until(lambda: api.outboxes["s"].qsize() >= 2)
        assert done, "outbox was never filled with both items"
        items = []
        while not api.outboxes["s"].empty():
            items.append(api.outboxes["s"].get_nowait())
        await _stop(worker)

        note_kind, write_text, note_audio = items[0]
        dlg_kind, dialogue_tail, tts_audio = items[1]
        return (note_kind, write_text, note_audio, dlg_kind, dialogue_tail, tts_audio)

    result = asyncio.run(scenario())

    # The outbox carries both the LLM "write" text and synthesized speech of "say".
    assert (result[0], result[1], result[2]) == (ord("n"), "hi!", b"")
    assert result[4]  # dialogue tail is filled too
    assert result[5] == b"audio-bytes"


def test_process_tts_failure_still_sends_write(monkeypatch):
    import sayteacher.service as svc_mod
    from sayteacher.service import LlmAnswerService
    from sayteacher.service import RecognitionService

    api = RecognitionService("s", model="some/model")
    # Fake executor: transcribe works, tts raises.
    api._executor = _FakeExecutor(
        transcribe_replies=["привет"],
        tts_exc=RuntimeError("cuda oom during synthesis"),
    )

    from test_llm import FakeOpenAIClient

    monkeypatch.setattr(
        svc_mod, "AsyncOpenAI",
        lambda base_url=None, **kw: FakeOpenAIClient(base_url=base_url, payload={"say": "teacher hi", "write": "hi!"}),
    )
    llm = LlmAnswerService("http://x", "m")

    from sayteacher.service import TtsService
    tts = TtsService(lang="English", speaker="v3_en")

    async def scenario():
        worker = asyncio.create_task(api.process("s", llm_api=llm, tts=tts))
        await api.send("s", b"\x01" * 16, "", _clip(800))

        done = await _wait_until(lambda: api.outboxes["s"].qsize() >= 2)
        assert done, "outbox was never filled with both items"
        items = []
        while not api.outboxes["s"].empty():
            items.append(api.outboxes["s"].get_nowait())
        await _stop(worker)

        note_kind, write_text, note_audio = items[0]
        dlg_kind, dialogue_tail, tts_audio = items[1]
        return (note_kind, write_text, note_audio, dlg_kind, tts_audio)

    result = asyncio.run(scenario())
    # A failing TTS must not drop the text answer; kind byte is int('n').
    assert (result[0], result[1], result[2]) == (ord("n"), "hi!", b"")
    # The worker survived and is still alive after the error; dialogue was filled too.
    assert api.dialogues["s"]


def test_voice_id_selects_that_speaker():
    """--speaker en_117 must resolve to v3_en bundle with voice=en_117."""
    from sayteacher.service import TtsService

    tts = TtsService(lang="English", speaker="en_117")
    # Voice resolution must keep en_117 (it's not a bundle id).
    assert tts._voice == "en_117"
    bundle, voice, language = tts.worker_params()
    assert bundle == "v3_en"
    assert voice == "en_117"
    assert language == "en"


def test_bundle_id_falls_back_to_first_voice():
    """--speaker v3_en (bundle) must still work; voice stays empty so worker picks first."""
    from sayteacher.service import TtsService

    tts = TtsService(lang="English", speaker="v3_en")
    assert tts._voice == ""
    bundle, voice, language = tts.worker_params()
    assert bundle == "v3_en"
    assert voice == ""
    assert language == "en"


def test_unknown_speaker_still_loads_language_default():
    """An unknown voice (e.g. made-up id) still loads the language's default bundle."""
    from sayteacher.service import TtsService

    tts = TtsService(lang="English", speaker="en_999")  # not a bundle, treated as a voice
    assert tts._voice == "en_999"
    bundle, voice, language = tts.worker_params()
    assert bundle == "v3_en"
    assert voice == "en_999"
    assert language == "en"
