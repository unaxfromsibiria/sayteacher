import hashlib
import uuid as _uuid
from pathlib import Path

import pytest

from sayteacher.transport import Transport

LIB_PATH = Path(__file__).resolve().parent.parent / "lib" / "libmicroagent_transport.so"


@pytest.fixture()
def transport():
    return Transport(str(LIB_PATH))


def test_roundtrip_empty(transport):
    msg = transport.decode(transport.encode(bytes(16), "", b""))
    assert msg.id == bytes(16)
    assert msg.text == ""
    assert msg.audio == b""


def test_roundtrip_with_text_and_audio(transport):
    uid = bytes(range(16))
    text = "hello world"
    audio = bytes(range(128))
    decoded = transport.decode(transport.encode(uid, text, audio))
    assert decoded.id == uid
    assert decoded.text == text
    assert decoded.audio == audio


def test_encode_rejects_wrong_length_uuid(transport):
    with pytest.raises(ValueError, match="exactly 16 bytes"):
        transport.encode(b"\x00\x01\x02", "x", b"")


def test_decode_empty_frame_raises(transport):
    with pytest.raises(ValueError, match="empty frame"):
        transport.decode(b"")


def test_decode_too_short_reports_ffi_error(transport):
    # 16 bytes is a valid uuid but missing the two u32 length fields -> TooShort.
    with pytest.raises(ValueError):
        transport.decode(bytes(16))
    assert transport.last_error() == 2  # FFI_ERR_TOO_SHORT


def test_auth_token_matches_stdlib_sha256(transport):
    uid = bytes(range(16))
    password = "secret"
    expected = hashlib.sha256((str(_uuid.UUID(bytes=uid)) + password).encode()).hexdigest()
    assert transport.auth_token(uid, password) == expected


def test_auth_token_is_deterministic_and_depends_on_inputs(transport):
    uid = bytes(range(16))
    other_id = bytes(range(15, 31))
    token = transport.auth_token(uid, "secret")
    assert len(token) == 64
    assert token.islower()
    assert all(c in "0123456789abcdef" for c in token)
    # same inputs -> same token; different id or password -> different token.
    assert transport.auth_token(uid, "secret") == token
    assert transport.auth_token(other_id, "secret") != token
    assert transport.auth_token(uid, "other") != token


def test_verify_auth_accepts_valid_rejects_invalid(transport):
    uid = bytes(range(16))
    good = transport.auth_token(uid, "secret")
    assert transport.verify_auth(uid, good, "secret") is True
    # wrong password for the same id must not verify.
    assert transport.verify_auth(uid, transport.auth_token(uid, "wrong"), "secret") is False


def test_verify_auth_empty_password_is_open_handshake(transport):
    uid = bytes(range(16))
    assert transport.verify_auth(uid, "", "") is True
