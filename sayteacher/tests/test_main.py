"""Tests for main.py helpers."""

import pytest

from sayteacher.common import IDLE_TIMEOUT_SECONDS
from sayteacher.connection import READY_PROJECT
from sayteacher.connection import Connection
from sayteacher.main import STUDENT_NAME
from sayteacher.main import SessionFinished
from sayteacher.main import _apply_lesson
from sayteacher.main import _idle_check


class _RecordingClient:
    """Minimal stand-in for AsyncClient that records every send() call."""

    def __init__(self) -> None:
        self.sent = []  # list of (project, name, note, data) tuples

    def send(self, project: str, name: str, note: str = "", data: bytes = b"") -> None:
        self.sent.append((project, name, note, data))


class _FakeApi:
    """Just enough of RecognitionService for _apply_lesson (it only touches .lessons)."""

    def __init__(self) -> None:
        self.lessons = {}


def test_session_finished_is_a_clean_exit_signal():
    """The quit protocol raises SessionFinished; the reconnect loop treats it as a clean end of
    lesson and stays alive to receive the next one (it no longer stops the process)."""
    with pytest.raises(SessionFinished):
        raise SessionFinished()


def test_idle_check_increments_counter():
    assert _idle_check(0.0) == 1.0
    assert _idle_check(59.0) == 60.0


def test_idle_check_returns_false_at_timeout():
    result = _idle_check(IDLE_TIMEOUT_SECONDS - 0.5)
    assert result is False


def test_student_name_is_fixed():
    """SayTeacher serves a single student; the name is a fixed constant."""
    assert STUDENT_NAME == "student"


def test_greeting_stores_and_acks_lesson():
    """Receiving a lesson stores it and acks with 'lesson-set' so microagent stops re-sending."""
    client = _RecordingClient()
    conn = Connection(client=client)

    api = _FakeApi()
    _apply_lesson(api, "student", conn, "### Small talk")

    assert api.lessons["student"] == "### Small talk"
    assert (READY_PROJECT, "lesson-set", "", b"") in client.sent
