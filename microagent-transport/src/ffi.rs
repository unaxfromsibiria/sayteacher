//! Minimal, safe C ABI so this crate can be used from Python via ctypes or any other
//! FFI host. Only pointer + integer arguments are exposed; no Rust types cross the boundary.
//!
//! Typical flow: `ffi_encode` -> write buffer to socket OR read from socket -> `ffi_decode`.

use std::ptr;
use std::slice;
use std::str;

use bytes::Bytes;
use uuid::Uuid;

use super::{
    Message, TransportError,
    FFI_ERR_BAD_INPUT, FFI_ERR_TOO_SHORT, FFI_ERR_BAD_UTF8, FFI_ERR_BAD_UUID, FFI_OK,
};

/// Opaque byte buffer allocated by the crate. Free with `ffi_buffer_free`.
#[repr(C)]
pub struct FfiBuffer {
    pub ptr: *mut u8,
    pub len: usize,
    pub cap: usize,
}

/// Decoded transport message in a flat, C-friendly layout. Free with `ffi_message_free`;
/// nested buffers are released automatically by that call.
#[repr(C)]
pub struct FfiMessage {
    pub id: [u8; 16],
    pub text: *mut u8,
    pub text_len: usize,
    pub text_cap: usize,
    pub audio: *mut u8,
    pub audio_len: usize,
    pub audio_cap: usize,
}

const EMPTY_BUF: FfiBuffer = FfiBuffer { ptr: std::ptr::null_mut(), len: 0, cap: 0 };

/// Release a buffer previously returned by `ffi_encode` / `ffi_auth_token`, or the
/// `text`/`audio` field of an `FfiMessage`. Safe on null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ffi_buffer_free(ptr: *mut u8, len: usize, cap: usize) {
    if ptr.is_null() || len == 0 {
        return;
    }
    drop(unsafe { Vec::from_raw_parts(ptr, len, cap.max(len)) });
}

/// Release a message returned by `ffi_decode`, including its nested text/audio buffers.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ffi_message_free(msg: *mut FfiMessage) {
    if msg.is_null() {
        return;
    }
    let m = unsafe { &*msg };
    if !m.text.is_null() && m.text_len > 0 {
        drop(unsafe {
            Vec::from_raw_parts(m.text, m.text_len, m.text_cap.max(m.text_len))
        });
    }
    if !m.audio.is_null() && m.audio_len > 0 {
        drop(unsafe {
            Vec::from_raw_parts(m.audio, m.audio_len, m.audio_cap.max(m.audio_len))
        });
    }
    drop(unsafe { Box::from_raw(msg) });
}

/// Last error code for the most recent failing FFI call. `FFI_OK` after a success.
static LAST_ERROR: std::sync::atomic::AtomicI32 = std::sync::atomic::AtomicI32::new(0);

#[inline]
fn set_error(code: i32) {
    LAST_ERROR.store(code, std::sync::atomic::Ordering::Relaxed);
}

/// Read the last FFI error code (see constants in crate root). Useful after a null return.
#[unsafe(no_mangle)]
pub extern "C" fn ffi_last_error() -> i32 {
    LAST_ERROR.load(std::sync::atomic::Ordering::Relaxed)
}

fn map_err(e: &TransportError) -> i32 {
    match e {
        TransportError::TooShort => FFI_ERR_TOO_SHORT,
        TransportError::BadUuid => FFI_ERR_BAD_UUID,
        TransportError::BadUtf8 => FFI_ERR_BAD_UTF8,
    }
}

fn vec_to_buffer(v: Vec<u8>) -> FfiBuffer {
    let (ptr, len, cap) = v.into_raw_parts();
    FfiBuffer { ptr, len, cap }
}

/// Serialize a message into the flat byte layout described in `Message::encode`.
/// Returns an empty buffer on error; check with `ffi_last_error` for details.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ffi_encode(
    id: *const u8,
    text: *const u8,
    text_len: usize,
    audio: *const u8,
    audio_len: usize,
) -> FfiBuffer {
    if id.is_null() || (text.is_null() && text_len > 0) || (audio.is_null() && audio_len > 0) {
        set_error(FFI_ERR_BAD_INPUT);
        return EMPTY_BUF;
    }

    let uuid = match Uuid::from_slice(unsafe { slice::from_raw_parts(id, 16) }) {
        Ok(u) => u,
        Err(_) => {
            set_error(FFI_ERR_BAD_UUID);
            return EMPTY_BUF;
        }
    };

    let text_bytes = if text_len == 0 {
        Vec::new()
    } else {
        unsafe { slice::from_raw_parts(text, text_len) }.to_vec()
    };
    let Ok(text_str) = str::from_utf8(&text_bytes) else {
        set_error(FFI_ERR_BAD_UTF8);
        return EMPTY_BUF;
    };

    let audio_slice: &[u8] = if audio_len == 0 {
        &[]
    } else {
        unsafe { slice::from_raw_parts(audio, audio_len) }
    };

    let msg = Message {
        id: uuid,
        text: text_str.to_owned(),
        audio: Bytes::copy_from_slice(audio_slice),
    };

    set_error(FFI_OK);
    vec_to_buffer(msg.encode().to_vec())
}

/// Deserialize a flat buffer produced by `ffi_encode`. Returns null on error; call
/// `ffi_last_error` for details. On success the caller must free via `ffi_message_free`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ffi_decode(data: *const u8, data_len: usize) -> *mut FfiMessage {
    if data.is_null() && data_len > 0 {
        set_error(FFI_ERR_BAD_INPUT);
        return ptr::null_mut();
    }

    let slice = unsafe { slice::from_raw_parts(data, data_len) };
    match Message::decode(slice) {
        Ok(m) => {
            let text: Vec<u8> = m.text.into_bytes();
            let (text_ptr, text_len, text_cap) = text.into_raw_parts();

            let audio_vec = m.audio.to_vec();
            let (audio_ptr, audio_len, audio_cap) = audio_vec.into_raw_parts();

            set_error(FFI_OK);
            Box::into_raw(Box::new(FfiMessage {
                id: *m.id.as_bytes(),
                text: text_ptr,
                text_len,
                text_cap,
                audio: audio_ptr,
                audio_len,
                audio_cap,
            }))
        }
        Err(e) => {
            set_error(map_err(&e));
            ptr::null_mut()
        }
    }
}

/// Compute hex(sha256(id_string + password)) as a byte buffer (no NUL terminator).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ffi_auth_token(id: *const u8, password: *const i8) -> FfiBuffer {
    if id.is_null() || password.is_null() {
        set_error(FFI_ERR_BAD_INPUT);
        return EMPTY_BUF;
    }

    let Ok(uuid) = Uuid::from_slice(unsafe { slice::from_raw_parts(id, 16) }) else {
        set_error(FFI_ERR_BAD_UUID);
        return EMPTY_BUF;
    };
    // password is a NUL-terminated C string.
    let Ok(cstr) = (unsafe { std::ffi::CStr::from_ptr(password).to_str() }) else {
        set_error(FFI_ERR_BAD_UTF8);
        return EMPTY_BUF;
    };

    set_error(FFI_OK);
    vec_to_buffer(Message::auth_token_for(&uuid, cstr).into_bytes())
}

/// Verify that `token` (as a byte buffer) matches `sha256(id + password)` case-insensitively.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ffi_verify_auth(
    id: *const u8,
    token: *const u8,
    token_len: usize,
    password: *const i8,
) -> bool {
    if id.is_null() || password.is_null() || (token.is_null() && token_len > 0) {
        set_error(FFI_ERR_BAD_INPUT);
        return false;
    }

    let Ok(uuid) = Uuid::from_slice(unsafe { slice::from_raw_parts(id, 16) }) else {
        set_error(FFI_ERR_BAD_UUID);
        return false;
    };
    let Ok(token_str) = std::str::from_utf8(unsafe { slice::from_raw_parts(token, token_len) }) else {
        set_error(FFI_ERR_BAD_UTF8);
        return false;
    };
    let Ok(pass_cstr) = (unsafe { std::ffi::CStr::from_ptr(password).to_str() }) else {
        set_error(FFI_ERR_BAD_UTF8);
        return false;
    };

    // verify_auth takes &self and only reads `self.text`, so the id field is irrelevant.
    let msg = Message {
        id: uuid,
        text: token_str.to_owned(),
        audio: Bytes::new(),
    };

    set_error(FFI_OK);
    msg.verify_auth(&uuid, pass_cstr)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ffi_roundtrip() {
        let id_bytes = [0u8; 16].map(|b| b.wrapping_add(7));
        let text: Vec<u8> = b"hello".to_vec();
        let audio: Vec<u8> = (0..32).collect();

        let buf = unsafe {
            ffi_encode(id_bytes.as_ptr(), text.as_ptr(), text.len(), audio.as_ptr(), audio.len())
        };
        assert!(!buf.ptr.is_null());
        assert_eq!(ffi_last_error(), FFI_OK);

        let msg = unsafe { ffi_decode(buf.ptr, buf.len) };
        assert!(!msg.is_null());
        let m = unsafe { &*msg };
        assert_eq!(m.id, id_bytes);
        assert_eq!(unsafe { std::str::from_utf8(slice::from_raw_parts(m.text, m.text_len)) }, Ok("hello"));
        assert_eq!(unsafe { slice::from_raw_parts(m.audio, m.audio_len) }, audio.as_slice());

        unsafe { ffi_message_free(msg); }
        unsafe { ffi_buffer_free(buf.ptr, buf.len, buf.cap); }
    }

    #[test]
    fn ffi_auth_token_matches_rust_api() {
        let id = Uuid::new_v4();
        let bytes = id.as_bytes();
        let password: Vec<i8> = b"secret\0".iter().copied().map(|b| b as i8).collect();

        let buf = unsafe { ffi_auth_token(bytes.as_ptr(), password.as_ptr()) };
        assert!(!buf.ptr.is_null());
        let token_c = unsafe { std::str::from_utf8(slice::from_raw_parts(buf.ptr, buf.len)).unwrap() }.to_owned();
        let expected = Message::auth_token_for(&id, "secret");
        assert_eq!(token_c, expected);

        unsafe { ffi_buffer_free(buf.ptr, buf.len, buf.cap) }
    }

    #[test]
    fn ffi_verify_auth_accepts_and_rejects() {
        let id = Uuid::new_v4();
        let bytes = id.as_bytes();
        let token = Message::auth_token_for(&id, "secret");
        let pass: Vec<i8> = b"secret\0".iter().copied().map(|b| b as i8).collect();
        let wrong_pass: Vec<i8> = b"wrong\0".iter().copied().map(|b| b as i8).collect();

        assert!(unsafe { ffi_verify_auth(bytes.as_ptr(), token.as_ptr(), token.len(), pass.as_ptr()) });
        assert!(!unsafe { ffi_verify_auth(bytes.as_ptr(), token.as_ptr(), token.len(), wrong_pass.as_ptr()) });
    }
}
