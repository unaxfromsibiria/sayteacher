from enum import IntEnum

TARGET_SAMPLE_RATE = 16_000  # whisper input rate; the server already sends at this rate
IDLE_TIMEOUT_SECONDS = 180.0  # Idle limit before giving up on a silent connection (task: 60s was too short).

# Sentinel enqueued by the frame pump once when the peer goes away; compare with `is`.
CLOSED = "__closed__"


class MessageKindEnum(IntEnum):
    """First-audio-byte marker, mirroring Rust's `MessageKind` in microagent-transport.

    The values are the ASCII codes of the letters so both sides build/parse plain bytes.
    A frame is `[kind][data]`; `data` may be empty for text-only frames."""

    AUDIO = 0x61       # 'a' — raw audio only; any text alongside it is informational (e.g. TTS of "say")
    DIALOGUE = 0x64    # 'd' — recent you:/teacher: lines in `text`, optional audio
    NOTE = 0x6E        # 'n' — a note / plain text message
    OUTPUT = 0x6F      # 'o' — output-device capture stream from outvoiceagent (system audio)
    PING = 0x70        # 'p' — keep-alive during silence; both sides echo it back
    QUIT = 0x71        # 'q' — end-of-session marker; the client closes after delivery
    TRANSLATE = 0x74   # 't' — translation result

    @classmethod
    def from_byte(cls, b: int):
        """Map a single byte to a kind, or None when it is not a known marker."""
        try:
            return cls(b)
        except ValueError:
            return None


def strip_type_prefix(audio: bytes) -> bytes:
    """Return the payload after the one-byte type marker, when present.

    Frames from outvoiceagent are prefixed with `o` (output-device capture), so this removes
    that single byte before samples are decoded. Text-only frames and untyped audio pass
    through unchanged because their first byte is not a known marker."""
    if audio and MessageKindEnum.from_byte(audio[0]) is not None:
        return audio[1:]
    return audio
