from collections import namedtuple
from enum import IntEnum

from iso3166 import countries

# One recognized utterance in a student's dialogue history.
DialogueEntry = namedtuple("DialogueEntry", ["time", "is_student", "text"])

TARGET_SAMPLE_RATE = 16_000  # whisper input rate; the server already sends at this rate
IDLE_TIMEOUT_SECONDS = 180.0  # Idle limit before giving up on a silent connection (task: 60s was too short).
QUIT_WORD = "quit"  # Token that ends a dialogue session
QUIT_MARKER_DELAY_SECONDS = 10.0  # Wait before queueing the 'q' frame so the client can start TTS playback first.

# Sentinel enqueued by the frame pump once when the peer goes away; compare with `is`.
CLOSED = "__closed__"


# Silero language keys (see silero_models.py). --lang accepts full names or these codes.
LANGUAGE_CODES = ("en", "de", "es", "fr", "ru")

# ISO 3166 alpha-2 -> Silero language key, for country-name values in --lang.
_ISO_ALPHA2_TO_CODE = {
    "EN": "en",
    "US": "en",
    "GB": "en",
    "DE": "de",
    "ES": "es",
    "FR": "fr",
    "RU": "ru",
}

# English language names have no ISO country; direct lookup only.
_LANG_NAME_TO_CODE = {
    "english": "en",
    "russian": "ru",
    "german": "de",
    "spanish": "es",
    "french": "fr",
}

# Speaker-id prefix -> Silero language key, used when --lang is unknown ('auto').
SPEAKER_LANG_HINT = {
    "v3_en": "en",
    "v3_de": "de",
    "v3_es": "es",
    "v3_fr": "fr",
    "v5_ru": "ru",
    "v4_uz": "uz",
    "v4_cyrillic": "cyrillic",
}

# Per-language default model bundle, used when --speaker names a voice (e.g. en_117)
# rather than a bundle id: the loader still needs a concrete bundle to download/load.
DEFAULT_BUNDLE_BY_LANG = {
    "en": "v3_en",
    "de": "v3_de",
    "es": "v3_es",
    "fr": "v3_fr",
    "ru": "v5_ru",
}


def silero_language(lang: str) -> str:
    """Map a --lang value to the Silero tts language key; 'auto' means unknown.

    Accepts Silero codes (en/ru/de/es/fr), English names (English, Russian...) and ISO
    3166 country names/codes (resolved through the iso3166 package).
    """
    low = (lang or "").strip().lower()
    if low in LANGUAGE_CODES:
        return low

    code = _LANG_NAME_TO_CODE.get(low)
    if code is not None:
        return code

    try:
        country = countries[lang.strip()]
    except (KeyError, ValueError):
        return "auto"
    return _ISO_ALPHA2_TO_CODE.get(country.alpha2.upper(), "auto")


class MessageKindEnum(IntEnum):
    """First-audio-byte marker, mirroring Rust's `MessageKindEnum` in microagent-transport.

    The values are the ASCII codes of the letters so both sides build/parse plain bytes.
    A frame is `[kind][data]`; `data` may be empty for text-only frames."""

    AUDIO = 0x61       # 'a' — raw audio only; any text alongside it is informational (e.g. TTS of "say")
    DIALOGUE = 0x64    # 'd' — recent you:/teacher: lines in `text`, optional audio
    NOTE = 0x6E        # 'n' — a note / plain text message
    OUTPUT = 0x6F      # 'o' — output-device capture stream from outvoiceagent (system audio)
    QUIT = 0x71        # 'q' — end-of-session marker; the client closes after delivery

    @classmethod
    def from_byte(cls, b: int):
        """Map a single byte to a kind, or None when it is not an a/d/n marker."""
        try:
            return cls(b)
        except ValueError:
            return None


def strip_type_prefix(audio: bytes) -> bytes:
    """Return the payload after the one-byte type marker, when present.

    Frames from the Rust server are prefixed with `a` (mic recording), so this removes that
    single byte before samples are decoded. Text-only frames and legacy untyped audio pass
    through unchanged because their first byte is not an a/d/n marker."""
    if audio and MessageKindEnum.from_byte(audio[0]) is not None:
        return audio[1:]
    return audio


def kind_of(audio: bytes):
    """The frame kind from the first byte, or None when absent/unknown."""
    if not audio:
        return None
    return MessageKindEnum.from_byte(audio[0])
