import asyncio
import json
import struct
from concurrent.futures import Future as _CFFuture


def _clip(n_samples: int) -> bytes:
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


class _FakeResponses:
    """Stands in for AsyncOpenAI.responses; create() returns a canned output_text."""

    def __init__(self, client):
        self._client = client

    async def create(self, model, input, reasoning=None, extra_body=None):
        if self._client.exc:
            raise self._client.exc
        self._client.last_create_kwargs = {
            "model": model,
            "input": input,
            "reasoning": reasoning,
            "extra_body": extra_body,
        }
        return type("R", (), {"output_text": self._client.output_text})()


class FakeOpenAIClient:
    """Stands in for openai.AsyncOpenAI; responses.create returns a canned output_text."""

    def __init__(self, base_url=None, api_key="", payload=None, exc=None):
        self._base_url = base_url
        if payload is None:
            payload = {"say": "hi there", "write": "hello!"}
        # The real API always yields a string; encode non-str payloads as JSON, use str verbatim.
        self.output_text = payload if isinstance(payload, str) else json.dumps(payload)
        self.exc = exc
        self.last_create_kwargs: dict = {}
        self.responses = _FakeResponses(self)

    async def close(self):
        pass


def _patch_openai(monkeypatch, payload=None, exc=None):
    """Patch AsyncOpenAI in the service module's namespace so LlmAnswerService sees it."""
    import sayteacher.service as svc_mod

    def ctor(base_url=None, **kw):
        return FakeOpenAIClient(base_url=base_url, payload=payload, exc=exc)

    monkeypatch.setattr(svc_mod, "AsyncOpenAI", ctor)


class _FakeExecutor:
    """Stands in for ProcessPoolExecutor; returns canned transcribe results."""

    def __init__(self, replies=None) -> None:
        self._replies = list(replies or ["привет"])
        self._i = 0
        self.calls: list[tuple[str, tuple]] = []

    def submit(self, fn, *args):
        name = getattr(fn, "__name__", str(fn))
        self.calls.append((name, args))
        fut = _CFFuture()
        if name == "_worker_transcribe":
            text = self._replies[self._i] if self._i < len(self._replies) else ""
            self._i += 1
            fut.set_result(text)
        elif name == "_worker_tts":
            fut.set_result(b"")
        else:
            fut.set_result("")
        return fut

    def shutdown(self, wait=False, cancel_futures=True) -> None:
        pass


def _setup_api(model="some/model", replies=None):
    """Create a RecognitionService with a fake executor for ASR."""
    from sayteacher.service import RecognitionService

    api = RecognitionService("s", model=model)
    api._executor = _FakeExecutor(replies=replies or ["привет"])
    return api


def test_llm_request_survives_close_error(monkeypatch):
    """close() is no longer called per request; a broken client must not affect request()."""
    import sayteacher.service as svc_mod
    from sayteacher.service import LlmAnswerService

    class BrokenCloseClient(FakeOpenAIClient):
        async def close(self):
            raise RuntimeError("socket already closed")

    monkeypatch.setattr(svc_mod, "AsyncOpenAI", lambda base_url=None, **kw: BrokenCloseClient(base_url=base_url))

    svc = LlmAnswerService(api_url="http://x", model="m")
    say, write, duration, to_quit = asyncio.run(svc.request([], "p"))
    assert (say, write, duration, to_quit) == ("hi there", "hello!", None, False)


def test_llm_disabled_when_api_or_model_empty():
    from sayteacher.service import LlmAnswerService

    assert LlmAnswerService("", "").enabled is False
    assert LlmAnswerService("http://x", "").enabled is False
    assert LlmAnswerService("", "model").enabled is False
    assert LlmAnswerService("http://x", "model").enabled is True


def test_llm_request_returns_say_and_write(monkeypatch):
    from sayteacher.service import LlmAnswerService

    _patch_openai(monkeypatch)
    svc = LlmAnswerService(api_url="http://127.0.0.1:11434", model="llama")
    say, write, duration, to_quit = asyncio.run(svc.request([], "prompt"))
    assert (say, write, duration, to_quit) == ("hi there", "hello!", None, False)


def test_llm_request_passes_extra_body_disable_thinking(monkeypatch):
    """request() must send extra_body disabling the chat-template thinking phase."""
    import sayteacher.service as svc_mod
    from sayteacher.service import LlmAnswerService

    created = []

    def ctor(base_url=None, **kw):
        client = FakeOpenAIClient(base_url=base_url)
        created.append(client)
        return client

    monkeypatch.setattr(svc_mod, "AsyncOpenAI", ctor)
    svc = LlmAnswerService(api_url="http://x", model="m")
    asyncio.run(svc.request([], "prompt"))

    assert len(created) == 1
    assert created[0].last_create_kwargs["extra_body"] == {
        "chat_template_kwargs": {"enable_thinking": False},
    }


def test_llm_request_returns_to_quit(monkeypatch):
    from sayteacher.service import LlmAnswerService

    _patch_openai(monkeypatch, payload={"say": "bye now", "write": "goodbye!", "quit": True})
    svc = LlmAnswerService(api_url="http://x", model="m")
    assert asyncio.run(svc.request([], "prompt")) == ("bye now", "goodbye!", None, True)

    _patch_openai(monkeypatch, payload={"say": "hi", "write": "hello", "quit": False})
    svc = LlmAnswerService(api_url="http://x", model="m")
    assert asyncio.run(svc.request([], "prompt")) == ("hi", "hello", None, False)


def test_llm_request_parses_expected_duration(monkeypatch):
    from datetime import datetime
    from datetime import timedelta

    from sayteacher.common import DialogueEntry
    from sayteacher.service import LlmAnswerService

    _patch_openai(monkeypatch, payload={"say": "s", "write": "w", "expected_duration": 30})
    svc = LlmAnswerService(api_url="http://x", model="m")
    dialogue = [DialogueEntry(time=datetime.now() - timedelta(minutes=5), is_student=True, text="hi")]
    say, write, duration, to_quit = asyncio.run(svc.request(dialogue, "prompt"))
    assert (say, write, duration, to_quit) == ("s", "w", 30, False)


def test_llm_request_ignores_non_numeric_expected_duration(monkeypatch):
    from sayteacher.service import LlmAnswerService

    _patch_openai(monkeypatch, payload={"say": "s", "write": "w", "expected_duration": "30 min"})
    svc = LlmAnswerService(api_url="http://x", model="m")
    assert asyncio.run(svc.request([], "prompt")) == ("s", "w", None, False)


def test_llm_request_disabled_returns_empty():
    from sayteacher.service import LlmAnswerService

    svc = LlmAnswerService(api_url="", model="")
    assert asyncio.run(svc.request([], "p")) == ("", "", None, False)


def test_llm_request_handles_missing_fields(monkeypatch):
    from sayteacher.service import LlmAnswerService

    _patch_openai(monkeypatch, payload={"unexpected": 1})
    svc = LlmAnswerService(api_url="http://x", model="m")
    assert asyncio.run(svc.request([], "p")) == ("", "", None, False)


def test_llm_request_handles_non_dict_json(monkeypatch):
    from sayteacher.service import LlmAnswerService

    _patch_openai(monkeypatch, payload=["list"])
    svc = LlmAnswerService(api_url="http://x", model="m")
    assert asyncio.run(svc.request([], "p")) == ("", "", None, False)


def test_llm_request_strips_markdown_code_fence(monkeypatch):
    from sayteacher.service import LlmAnswerService

    # Models often wrap the answer in ```json ... ``` fences; only the {...} object parses.
    fenced = '```json\n{\n  "say": "I\'m doing very well, thank you! How about you?",\n  "write": "how are you?"\n}\n```'

    import sayteacher.service as svc_mod

    monkeypatch.setattr(
        svc_mod, "AsyncOpenAI", lambda base_url=None, **kw: FakeOpenAIClient(base_url=base_url, payload=fenced),
    )
    svc = LlmAnswerService(api_url="http://x", model="m")
    say, write, duration, to_quit = asyncio.run(svc.request([], "p"))
    assert (say, write) == ("I'm doing very well, thank you! How about you?", "how are you?")
    assert duration is None
    assert to_quit is False


def test_parse_json_object_braces_in_strings():
    from sayteacher.service import _parse_json_object

    # Braces inside string literals must not break the balanced-scan.
    text = 'noise {"say": "a {b} c", "write": "x"} trailing'
    assert _parse_json_object(text) == {"say": "a {b} c", "write": "x"}

    # No object at all -> None, not an exception.
    assert _parse_json_object("no braces here") is None


def test_llm_request_handles_api_error(monkeypatch):
    from sayteacher.service import LlmAnswerService

    _patch_openai(monkeypatch, exc=RuntimeError("network down"))
    svc = LlmAnswerService(api_url="http://x", model="m")
    # Exception propagates to caller (process() catches it).
    try:
        asyncio.run(svc.request([], "p"))
        raise AssertionError("expected RuntimeError")
    except RuntimeError as e:
        assert "network down" in str(e)


def test_process_calls_llm_and_fills_dialogue_with_teacher(monkeypatch):
    from sayteacher.service import LlmAnswerService

    api = _setup_api(replies=["привет"])
    _patch_openai(monkeypatch, payload={"say": "teacher said hi", "write": "hi!"})
    llm = LlmAnswerService("http://x", "m")

    async def scenario():
        worker = asyncio.create_task(api.process("s", llm_api=llm))
        await api.send("s", b"\x01" * 16, "", _clip(800))
        done = await _wait_until(lambda: len(api.dialogues.get("s", [])) == 2)
        assert done, f"dialogue not filled: {api.dialogues.get('s')}"
        sent = None if api.outboxes["s"].empty() else api.outboxes["s"].get_nowait()
        await _stop(worker)
        return sent

    item = asyncio.run(scenario())
    _, write_text, tts_audio = (item or ("", b""))

    entries = api.dialogues["s"]
    assert [e.text for e in entries] == ["привет", "teacher said hi"]
    assert entries[0].is_student is True
    assert entries[1].is_student is False
    # TTS is not wired in this test, so the outbox message has no audio bytes.
    assert (write_text, tts_audio) == ("hi!", b"")


def test_process_without_llm_config_keeps_dialogue():
    api = _setup_api(replies=["привет"])

    async def scenario():
        worker = asyncio.create_task(api.process("s"))
        await api.send("s", b"\x01" * 16, "", _clip(800))
        done = await _wait_until(lambda: len(api.dialogues.get("s", [])) == 1)
        assert done
        drained = await _wait_until(lambda: api.outboxes["s"].empty())
        assert drained
        await _stop(worker)

    asyncio.run(scenario())
    assert [e.text for e in api.dialogues["s"]] == ["привет"]


def test_process_llm_exception_does_not_kill_worker(monkeypatch):
    from sayteacher.service import LlmAnswerService

    api = _setup_api(replies=["a", "b"])
    _patch_openai(monkeypatch, exc=RuntimeError("network down"))
    llm = LlmAnswerService("http://x", "m")

    async def scenario():
        worker = asyncio.create_task(api.process("s", llm_api=llm))
        await api.send("s", b"\x01" * 16, "", _clip(400))
        await api.send("s", b"\x02" * 16, "", _clip(400))
        done = await _wait_until(lambda: len(api.dialogues.get("s", [])) == 2)
        assert done, f"worker stopped after llm error: {api.dialogues.get('s')}"
        await _stop(worker)

    asyncio.run(scenario())


def test_build_prompt_uses_lesson_and_lang(monkeypatch):
    from sayteacher.service import LlmAnswerService

    api = _setup_api(model="some/model", replies=["x"])
    # Override lang: recreate with Russian.
    from sayteacher.service import RecognitionService
    api = RecognitionService("s", model="some/model", lang="Russian")
    api._executor = _FakeExecutor(replies=["x"])
    asyncio.run(api.send("s", b"\x01" * 16, "lesson text", b""))

    prompts = []

    class Recorder(LlmAnswerService):
        async def request(self, dialogue, prompt):
            prompts.append(prompt)
            return ("said", "", None, False)

    llm = Recorder("http://x", "m")

    async def scenario():
        worker = asyncio.create_task(api.process("s", llm_api=llm))
        await api.send("s", b"\x02" * 16, "", _clip(400))
        done = await _wait_until(lambda: bool(prompts))
        assert done
        await _stop(worker)

    asyncio.run(scenario())
    assert "Language: Russian" in prompts[0]
    assert "lesson text" in prompts[0]


def test_process_queues_quit_marker_after_final_answer(monkeypatch):
    import sayteacher.service as svc_mod
    from sayteacher.service import LlmAnswerService

    # Fast-forward the "client starts playback" delay so the marker lands in ~0.1s instead of 10s.
    monkeypatch.setattr(svc_mod, "QUIT_MARKER_DELAY_SECONDS", svc_mod.QUIT_MARKER_DELAY_SECONDS / 100)

    api = _setup_api(replies=["привет"])
    _patch_openai(monkeypatch, payload={"say": "goodbye now", "write": "bye!", "quit": True})
    llm = LlmAnswerService("http://x", "m")

    async def scenario():
        worker = asyncio.create_task(api.process("s", llm_api=llm))
        await api.send("s", b"\x01" * 16, "", _clip(800))
        done = await _wait_until(lambda: len(api.dialogues.get("s", [])) == 2)
        assert done
        # NOTE + DIALOGUE (final answer) first; the QUIT marker must be queued only after them.
        drained = await _wait_until(lambda: api.outboxes["s"].qsize() >= 3)
        assert drained, f"outbox={api.outboxes['s'].qsize()} items"
        kinds = []
        while not api.outboxes["s"].empty():
            kinds.append(api.outboxes["s"].get_nowait()[0])
        await _stop(worker)
        return kinds

    kinds = asyncio.run(scenario())
    assert kinds == [ord("n"), ord("d"), ord("q")]


def test_process_starts_duration_timer_once_and_puts_quit_word(monkeypatch):
    from sayteacher.common import QUIT_WORD
    from sayteacher.service import LlmAnswerService

    api = _setup_api(replies=["hi"])

    # 1/15 minute == 4 seconds of real sleep so the test finishes quickly.
    _patch_openai(monkeypatch, payload={"say": "t", "write": "w", "expected_duration": 1 / 15})
    llm = LlmAnswerService("http://x", "m")

    async def scenario():
        worker = asyncio.create_task(api.process("s", llm_api=llm))
        await api.send("s", b"\x01" * 16, "", _clip(800))
        armed = await _wait_until(lambda: api._duration_timer_task is not None)
        assert armed, "timer task was never created on first expected_duration"

        # Second arming must be a no-op — the timer runs exactly once.
        original = api._duration_timer_task
        api._start_duration_timer(99)
        assert api._duration_timer_task is original

        fired = await _wait_until(lambda: not api.quit_msg.empty(), timeout=8.0)
        assert fired, "quit word did not reach the queue after the timer elapsed"
        token = api.quit_msg.get_nowait()
        await _stop(worker)
        return token

    # The timer puts QUIT_WORD into quit_msg once it fires; llm_reply consumes it as a hint.
    assert asyncio.run(scenario()) == QUIT_WORD


def test_process_no_timer_without_expected_duration(monkeypatch):
    from sayteacher.service import LlmAnswerService

    api = _setup_api(replies=["hi"])
    _patch_openai(monkeypatch, payload={"say": "t", "write": "w"})  # no expected_duration
    llm = LlmAnswerService("http://x", "m")

    async def scenario():
        worker = asyncio.create_task(api.process("s", llm_api=llm))
        await api.send("s", b"\x01" * 16, "", _clip(800))
        done = await _wait_until(lambda: len(api.dialogues.get("s", [])) == 2)
        assert done
        await asyncio.sleep(0.05)
        state = api._duration_timer_task is not None and not api.quit_msg.empty()
        await _stop(worker)
        return state

    # No duration -> no timer task, quit queue stays empty.
    assert asyncio.run(scenario()) is False


def test_quit_hint_stays_true_for_all_further_prompts(monkeypatch):
    """Once quit has happened (LLM answered quit=true) build_prompt must always see it."""
    from sayteacher.service import LlmAnswerService

    api = _setup_api(replies=["x", "y"])

    # Shorten the quit-marker delay so no task is left pending at scenario end.
    import sayteacher.service as svc_mod
    monkeypatch.setattr(svc_mod, "QUIT_MARKER_DELAY_SECONDS", 0.05)

    prompts_seen = []

    class Recorder(LlmAnswerService):
        async def request(self, dialogue, prompt):
            prompts_seen.append(prompt)
            return ("t", "w", None, True)  # quit=true on every answer -> hint must latch

    llm = Recorder("http://x", "m")

    async def scenario():
        worker = asyncio.create_task(api.process("s", llm_api=llm))
        await api.send("s", b"\x01" * 16, "", _clip(400))
        ok1 = await _wait_until(lambda: len(prompts_seen) >= 1)
        assert ok1
        # Second clip -> the prompt for this reply must also carry quit=true.
        await api.send("s", b"\x02" * 16, "", _clip(400))
        ok2 = await _wait_until(lambda: len(prompts_seen) >= 2)
        assert ok2
        await _stop(worker)
        return list(prompts_seen), api.quit_hint

    prompts, hint = asyncio.run(scenario())
    assert hint is True
    for p in prompts[1:]:
        assert "The lesson time is over" in p


def test_llm_request_quit_field_missing_defaults_false(monkeypatch):
    from sayteacher.service import LlmAnswerService

    _patch_openai(monkeypatch, payload={"say": "s", "write": "w"})
    svc = LlmAnswerService(api_url="http://x", model="m")
    assert asyncio.run(svc.request([], "p")) == ("s", "w", None, False)
