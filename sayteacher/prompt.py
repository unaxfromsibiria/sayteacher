from datetime import datetime

try:  # `python -m sayteacher.prompt` (package mode)
    from .common import DialogueEntry
except ImportError:  # flat run / pytest bootstrap
    from common import DialogueEntry

MAIN_TPL = """# You are a language teacher

As a teacher, you engage in a dialogue with the student about the tasks and the context of the lesson.
The student’s incorrect words in the dialogue or sudden words in another language are signal recognition errors on the microphone, don’t focus on this response.
If it asks for examples and poses questions about words, use the response format ("write").

Language: {}

## Current lesson

{}

## Dialogue

The student and the teacher already have the following dialogue:

{}

## Task

Continue the dialogue with the teacher’s response.

- Take into account the context of the lesson
- Take into account that the response will be converted into speech
- Take into account that you can include some text in the response
- Speech recognition of a student’s speech may introduce inaccuracies into the dialogue text
- Don’t keep track of the time, the notification will appear in "System event"

## Answer format

Please make answers as JSON with 4 fields (exclude emojis, emoticons, special characters in the answer):

1. "say": text to be said in response (will be added to the dialogue) what the student will hear
2. "write": extra notes (wording, grammar, examples) leave it blank or skip if there’s nothing to show
3. "expected_duration": extract only number value of duration from the lesson instruction (like line Expected duration: about 10 minutes -> 10)
4. "quit": (true/false) set to true if you have said goodbye and it is time to leave when the lesson is over, or if the student is behaving badly. If student ask to end the lesson, send it too

Fields "write" and "quit" are optional.

## System event

{}

"""


def _dialogue_lines(dialogue: list[DialogueEntry]) -> str:
    """Flatten dialogue into plain `- student:` / `- teacher:` lines without timestamps."""
    lines = []
    for entry in dialogue:
        speaker = "student" if entry.is_student else "teacher"
        lines.append(f"- {speaker}: {entry.text}")
    return "\n".join(lines)


def _duration_minutes(dialogue: list[DialogueEntry]) -> int:
    """Whole minutes since the first entry (dialogue start); 0 when empty."""
    if not dialogue:
        return 0

    start = dialogue[0].time
    if isinstance(start, datetime):
        elapsed = (datetime.now() - start).total_seconds() / 60.0
        return max(int(elapsed), 0)
    return 0


def build_prompt(lesson: str, dialogue: list[DialogueEntry], lang: str = "English", quit: bool = False) -> str:
    """Fill MAIN_TPL with language, lesson text, flattened dialogue and session duration."""
    return MAIN_TPL.format(
        lang,
        lesson or "",
        _dialogue_lines(dialogue),
        (
            "The lesson time is over! You can say goodbye, "
            "tell student about the lesson time add quit: true in answer"
        ) if quit else "No new events"
    )
