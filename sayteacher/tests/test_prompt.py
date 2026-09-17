import time
from datetime import datetime

from sayteacher.common import DialogueEntry
from sayteacher.prompt import build_prompt


def _dialogue() -> list[DialogueEntry]:
    now = datetime.now()
    return [
        DialogueEntry(time=now, is_student=True, text="hello?"),
        DialogueEntry(time=now, is_student=False, text="hi!"),
        DialogueEntry(time=now, is_student=True, text="how are you?"),
    ]


def test_prompt_contains_language_lesson_and_dialogue_lines():
    dialogue = _dialogue()
    prompt = build_prompt("Lesson A", dialogue, "English")

    assert "Language: English" in prompt
    assert "## Current lesson\n\nLesson A" in prompt
    # flattened format without timestamps
    assert "- student: hello?" in prompt
    assert "- teacher: hi!" in prompt
    assert "- student: how are you?" in prompt
    # the raw timestamp values must not leak into the dialogue section.
    assert str(dialogue[0].time) not in prompt


def test_prompt_empty_lesson_is_allowed():
    prompt = build_prompt("", _dialogue())
    # default language is English; no exception for an empty lesson text.
    assert "Language: English" in prompt
