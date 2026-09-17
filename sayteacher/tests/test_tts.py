import asyncio

from sayteacher.service import TtsService


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
    # No language and no speaker: nothing to synthesize.
    assert TtsService(lang="", speaker="").enabled is False
    assert TtsService(lang="Klingon", speaker="").enabled is False

    # A known --lang enables it even without a named speaker; so does any speaker id.
    assert TtsService(lang="English", speaker="").enabled is True
    assert TtsService(lang="Klingon", speaker="v3_en").enabled is True

    # generate() short-circuits to empty audio when there is no text / not enabled.
    assert TtsService(lang="", speaker="").generate("hello") == b""


def test_process_puts_write_and_tts_audio_into_outbox(monkeypatch):
    import sayteacher.service as svc_mod
    from sayteacher.service import LlmAnswerService
    from sayteacher.service import RecognitionService

    api = RecognitionService("s", model="some/model")

    class FakePipeline:
        def __call__(self, samples, generate_kwargs=None, return_timestamps=None):
            return {"text": "привет"}

    monkeypatch.setattr(api, "_build_asr_pipeline", lambda: FakePipeline())

    class FakeOllamaClient:
        async def generate(self, model, think, prompt):
            import json

            return type("R", (), {"response": json.dumps({"say": "teacher hi", "write": "hi!"})})()

        async def close(self):
            return None

    monkeypatch.setattr(svc_mod, "AsyncClient", lambda host=None, **kw: FakeOllamaClient())
    llm = LlmAnswerService("http://x", "m")
    tts = FakeTts()

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
    assert tts.calls == ["teacher hi"]


def test_process_tts_failure_still_sends_write(monkeypatch):
    import sayteacher.service as svc_mod
    from sayteacher.service import LlmAnswerService
    from sayteacher.service import RecognitionService

    api = RecognitionService("s", model="some/model")

    class FakePipeline:
        def __call__(self, samples, generate_kwargs=None, return_timestamps=None):
            return {"text": "привет"}

    monkeypatch.setattr(api, "_build_asr_pipeline", lambda: FakePipeline())

    class FakeOllamaClient:
        async def generate(self, model, think, prompt):
            import json

            return type("R", (), {"response": json.dumps({"say": "teacher hi", "write": "hi!"})})()

        async def close(self):
            return None

    monkeypatch.setattr(svc_mod, "AsyncClient", lambda host=None, **kw: FakeOllamaClient())
    llm = LlmAnswerService("http://x", "m")

    class BrokenTts(FakeTts):
        def generate(self, text: str) -> bytes:
            raise RuntimeError("cuda oom during synthesis")

    async def scenario():
        worker = asyncio.create_task(api.process("s", llm_api=llm, tts=BrokenTts()))
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


def test_voice_id_selects_that_speaker(monkeypatch):
    """--speaker en_117 must load the v3_en bundle and call apply_tts with voice=en_117."""
    import torch

    calls = {}

    class FakeModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.speakers = ["en_0", "en_5", "en_117"]

        def to(self, _device):
            return None

        def apply_tts(self, text=None, speaker=None, sample_rate=None, **kw):
            calls["speaker"] = speaker
            calls["sample_rate"] = sample_rate
            return torch.zeros(640)  # tiny tensor so resample path is exercised too

    def fake_silero(language, speaker):
        assert language == "en"
        assert speaker == "v3_en", f"expected v3_en bundle but got {speaker!r}"
        return FakeModel(), "example text"

    import sayteacher.service as svc_mod
    monkeypatch.setattr(svc_mod, "silero_tts", fake_silero)

    tts = TtsService(lang="English", speaker="en_117")
    # Voice resolution must keep en_117 (it's not a bundle id).
    assert tts._voice == "en_117"
    audio = tts.generate("hello world")
    assert len(audio) > 0, "expected non-empty f32 bytes"
    assert calls["speaker"] == "en_117", f"apply_tts got speaker={calls['speaker']!r}"


def test_bundle_id_falls_back_to_first_voice(monkeypatch):
    """--speaker v3_en (bundle) must still work, picking the model's first voice."""
    import torch

    calls = {}

    class FakeModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.speakers = ["en_0", "en_5"]

        def to(self, _device):
            return None

        def apply_tts(self, text=None, speaker=None, sample_rate=None, **kw):
            calls["speaker"] = speaker
            return torch.zeros(640)

    def fake_silero(language, speaker):
        assert language == "en"
        assert speaker == "v3_en", f"expected v3_en but got {speaker!r}"
        return FakeModel(), "example text"

    import sayteacher.service as svc_mod
    monkeypatch.setattr(svc_mod, "silero_tts", fake_silero)

    tts = TtsService(lang="English", speaker="v3_en")
    assert tts._voice == ""
    tts.generate("hello world")
    assert calls["speaker"] == "en_0", f"apply_tts got speaker={calls['speaker']!r}"


def test_unknown_speaker_still_loads_language_default(monkeypatch):
    """An unknown voice (e.g. made-up id) still loads the language's default bundle."""
    import torch

    calls = {}

    class FakeModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.speakers = ["en_0"]

        def to(self, _device):
            return None

        def apply_tts(self, text=None, speaker=None, sample_rate=None, **kw):
            calls["speaker"] = speaker
            return torch.zeros(64)

    def fake_silero(language, speaker):
        assert language == "en"
        assert speaker == "v3_en", f"expected v3_en but got {speaker!r}"
        return FakeModel(), "example text"

    import sayteacher.service as svc_mod
    monkeypatch.setattr(svc_mod, "silero_tts", fake_silero)

    tts = TtsService(lang="English", speaker="en_999")  # not a bundle, treated as a voice
    assert tts._voice == "en_999"
    tts.generate("hello")
    assert calls["speaker"] == "en_999", f"apply_tts got speaker={calls['speaker']!r}"
