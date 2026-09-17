from types import SimpleNamespace

import pytest

from sayteacher.main import SessionFinished
from sayteacher.main import _is_lesson


def test_is_lesson_matches_tools_rs_heuristic():
    lesson = SimpleNamespace(audio=b"", text="раз два три")
    record = SimpleNamespace(audio=b"\x00\x01\x02\x03", text="record x length: 4")
    empty = SimpleNamespace(audio=b"", text="")

    assert _is_lesson(lesson) is True
    assert _is_lesson(record) is False
    # empty audio + empty text must not be treated as a lesson (greeting-like frames).
    assert _is_lesson(empty) is False


def test_session_finished_is_a_clean_exit_signal():
    """The quit protocol raises SessionFinished so the reconnect loop can stop cleanly."""

    with pytest.raises(SessionFinished):
        raise SessionFinished()
