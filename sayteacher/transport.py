import ctypes
from dataclasses import dataclass


@dataclass(frozen=True)
class Message:
    id: bytes
    text: str
    audio: bytes


_U8P = ctypes.POINTER(ctypes.c_uint8)
_CSTRP = ctypes.POINTER(ctypes.c_char)


def _buf(data: bytes | None):
    """Return a ctypes buffer that stays alive for the caller's lifetime. Pass `None` or
    empty bytes to get an empty (null) buffer."""
    if not data:
        return None
    b = ctypes.create_string_buffer(bytes(data), len(data))
    return b


class Transport:
    def __init__(self, lib_path) -> None:
        self.lib = ctypes.CDLL(str(lib_path))

        class _Buffer(ctypes.Structure):
            _fields_ = [
                ("ptr", ctypes.c_void_p),
                ("len", ctypes.c_size_t),
                ("cap", ctypes.c_size_t),
            ]

        class _MsgC(ctypes.Structure):
            _fields_ = [
                ("id", ctypes.c_uint8 * 16),
                ("text", ctypes.c_void_p),
                ("text_len", ctypes.c_size_t),
                ("text_cap", ctypes.c_size_t),
                ("audio", ctypes.c_void_p),
                ("audio_len", ctypes.c_size_t),
                ("audio_cap", ctypes.c_size_t),
            ]

        self._Buffer = _Buffer
        self._MsgC = _MsgC

        self.lib.ffi_buffer_free.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t]
        self.lib.ffi_buffer_free.restype = None

        self.lib.ffi_message_free.argtypes = [ctypes.POINTER(_MsgC)]
        self.lib.ffi_message_free.restype = None

        self.lib.ffi_encode.argtypes = [_U8P, _U8P, ctypes.c_size_t, _U8P, ctypes.c_size_t]
        self.lib.ffi_encode.restype = _Buffer

        self.lib.ffi_decode.argtypes = [_U8P, ctypes.c_size_t]
        self.lib.ffi_decode.restype = ctypes.POINTER(_MsgC)

        self.lib.ffi_auth_token.argtypes = [_U8P, _CSTRP]
        self.lib.ffi_auth_token.restype = _Buffer

        self.lib.ffi_verify_auth.argtypes = [_U8P, _U8P, ctypes.c_size_t, _CSTRP]
        self.lib.ffi_verify_auth.restype = ctypes.c_bool

        self.lib.ffi_last_error.argtypes = []
        self.lib.ffi_last_error.restype = ctypes.c_int32

    def last_error(self) -> int:
        """FFI error code from the most recent failing call (see constants in Rust crate)."""
        return self.lib.ffi_last_error()

    @staticmethod
    def _ptr_of(buf):
        if buf is None:
            return ctypes.cast(ctypes.c_void_p(0), _U8P)
        return ctypes.cast(ctypes.c_void_p(ctypes.addressof(buf)), _U8P)

    def encode(self, id16: bytes, text: str, audio: bytes) -> bytes:
        if len(id16) != 16:
            raise ValueError(f"uuid must be exactly 16 bytes, got {len(id16)}")

        # Hold these for the duration of the FFI call so the C side never sees freed memory.
        id_buf = _buf(id16)
        text_buf = _buf(text.encode("utf-8"))
        audio_buf = _buf(audio)
        try:
            out = self.lib.ffi_encode(
                self._ptr_of(id_buf),
                self._ptr_of(text_buf),
                len(text_buf) if text_buf else 0,
                self._ptr_of(audio_buf),
                len(audio_buf) if audio_buf else 0,
            )
        finally:
            id_buf = text_buf = audio_buf = None

        result = ctypes.string_at(out.ptr, out.len) if out.len else b""
        self.lib.ffi_buffer_free(out.ptr, out.len, out.cap)
        return bytes(result)

    def decode(self, data: bytes) -> Message:
        n = len(data)
        if not n:
            raise ValueError("empty frame")

        src_buf = _buf(data)
        try:
            ptr = self.lib.ffi_decode(self._ptr_of(src_buf), n)
        finally:
            src_buf = None

        raw_id = bytes(ptr.contents.id)

        text = b""
        if ptr.contents.text_len:
            buf = ctypes.create_string_buffer(ptr.contents.text_len)
            ctypes.memmove(buf, ptr.contents.text, ptr.contents.text_len)
            text = bytes(buf)[:ptr.contents.text_len]

        audio = b""
        if ptr.contents.audio_len:
            buf = ctypes.create_string_buffer(ptr.contents.audio_len)
            ctypes.memmove(buf, ptr.contents.audio, ptr.contents.audio_len)
            audio = bytes(buf)[:ptr.contents.audio_len]

        self.lib.ffi_message_free(ptr)
        return Message(id=raw_id, text=text.decode("utf-8"), audio=audio)

    def auth_token(self, id16: bytes, password: str) -> str:
        if len(id16) != 16:
            raise ValueError(f"uuid must be exactly 16 bytes, got {len(id16)}")

        id_buf = _buf(id16)
        pass_c = ctypes.create_string_buffer(password.encode("utf-8"))
        try:
            buf = self.lib.ffi_auth_token(self._ptr_of(id_buf), pass_c)
        finally:
            id_buf = None
            del pass_c

        out = bytes(ctypes.string_at(buf.ptr, buf.len)) if buf.len else b""
        self.lib.ffi_buffer_free(buf.ptr, buf.len, buf.cap)
        return out.decode("utf-8")

    def verify_auth(self, id16: bytes, token: str, password: str) -> bool:
        if len(id16) != 16:
            raise ValueError(f"uuid must be exactly 16 bytes, got {len(id16)}")

        tok_b = token.encode("utf-8")
        id_buf = _buf(id16)
        tok_buf = _buf(tok_b)
        pass_c = ctypes.create_string_buffer(password.encode("utf-8"))
        try:
            return bool(self.lib.ffi_verify_auth(
                self._ptr_of(id_buf),
                self._ptr_of(tok_buf),
                len(tok_buf) if tok_buf else 0,
                pass_c,
            ))
        finally:
            id_buf = tok_buf = None
            del pass_c
