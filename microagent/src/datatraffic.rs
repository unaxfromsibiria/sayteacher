//! Safe Rust bindings for `libdatatraffic` (prebuilt cdylib).
//!
//! This is a self-contained module with no external dependencies: it declares the C ABI of
//! `libdatatraffic.so` and wraps it in a small safe interface. The shared library itself is
//! not compiled here — install it into your project with `scripts/install_rust.sh` (see
//! README.md, "Using libdatatraffic from Rust"), which places it under `vendor/datatraffic/`
//! and configures the link search path + rpath via `.cargo/config.toml`.
//!
//! Usage:
//! ```ignore
//! mod datatraffic; // this file at src/datatraffic.rs
//! use datatraffic::{Client, Config};
//!
//! let cfg = Config {
//!     redis_url: "redis://localhost:6379".into(),
//!     key_hex: std::env::var("CRYPTO_KEY")?,
//!     watch_projects: vec!["demo".into()],
//!     ..Default::default()
//! };
//! let mut client = Client::new(&cfg)?;
//! client.send("clip.wav", "encoded audio", &payload)?;
//! if let Some(file) = client.recv(1000) { /* file.name / file.note / file.data */ }
//! ```
//!
//! All entry points must be called sequentially from a single thread: the library drives an
//! async client on one global runtime (the same contract as the C ABI and the Python bindings).

use std::ffi::c_void;

// --- Error codes (mirror the `libdatatraffic` crate root) ----------------------

/// FFI call succeeded.
pub const OK: i32 = 0;
/// A pointer/length argument was invalid or a text field is not valid UTF-8.
pub const ERR_BAD_INPUT: i32 = 1;
/// Failed to create the traffic client (bad settings or Redis connection).
pub const ERR_CLIENT_CREATE: i32 = 2;
/// The outbox send failed because the channel was closed / client already freed.
pub const ERR_SEND: i32 = 3;

/// Error returned by FFI entry points; `code` is one of the constants above.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FfiError(pub i32);

impl std::fmt::Display for FfiError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let msg = match self.0 {
            OK => "ok",
            ERR_BAD_INPUT => "bad input (null or invalid argument)",
            ERR_CLIENT_CREATE => "failed to create traffic client",
            ERR_SEND => "send failed (outbox closed)",
            _ => "unknown ffi error",
        };
        write!(f, "ffi error {}: {}", self.0, msg)
    }
}

impl std::error::Error for FfiError {}

// --- C ABI ----------------------------------------------------------------------

/// Opaque byte buffer allocated by the library; released via `ffi_buffer_free`.
#[repr(C)]
struct FfiBuffer {
    ptr: *mut u8,
    len: usize,
    cap: usize,
}

/// A received data file exposed across the FFI boundary.
#[repr(C)]
struct FfiDataFile {
    name: FfiBuffer,
    note: FfiBuffer,
    data: FfiBuffer,
}

#[link(name = "datatraffic")]
unsafe extern "C" {
    fn ffi_last_error() -> i32;
    // Part of the C ABI (used by the Python bindings); not needed by the safe interface.
    #[allow(dead_code)]
    fn ffi_buffer_free(ptr: *mut u8, len: usize, cap: usize);
    fn ffi_client_new(
        redis_url: *const u8,
        redis_url_len: usize,
        key_hex: *const u8,
        key_hex_len: usize,
        client_name: *const u8,
        client_name_len: usize,
        password: *const u8,
        password_len: usize,
        watch_projects: *const u8,
        watch_projects_len: usize,
        send_to: *const u8,
        send_to_len: usize,
    ) -> *mut c_void;
    fn ffi_client_send(
        handle: *mut c_void,
        name: *const u8,
        name_len: usize,
        note: *const u8,
        note_len: usize,
        data: *const u8,
        data_len: usize,
    ) -> i32;
    fn ffi_client_recv(handle: *mut c_void, timeout_ms: u32) -> *mut FfiDataFile;
    fn ffi_datafile_free(ptr: *mut FfiDataFile);
    fn ffi_client_free(handle: *mut c_void);
}

// --- Safe interface ---------------------------------------------------------------

/// Connection settings. Empty fields mean "not provided" and fall back to the environment
/// (`REDIS_URL`, `CRYPTO_KEY`) or defaults — the same resolution as the Python/CLI APIs.
#[derive(Default, Clone)]
pub struct Config {
    /// Redis connection URL; empty -> env `REDIS_URL`.
    pub redis_url: String,
    /// AES-256-GCM key, 32 bytes as 64 hex chars; empty -> env `CRYPTO_KEY`.
    pub key_hex: String,
    /// Client identity; empty -> random name.
    pub client_name: String,
    /// Redis password; empty -> no auth (or env `REDIS_PASSWORD`).
    pub password: String,
    /// Projects to watch for incoming data files.
    pub watch_projects: Vec<String>,
    /// Project this client publishes to; empty -> "default".
    pub send_to: String,
}

/// A data file received from the wire.
#[derive(Debug, Clone)]
pub struct DataFile {
    pub name: String,
    pub note: String,
    pub data: Vec<u8>,
}

/// Synchronous wrapper over the traffic client lifecycle.
///
/// Not `Send`/`Sync`: call all methods sequentially from one thread (the library drives an
/// async client on a single global runtime). The handle is released automatically on drop.
#[must_use]
pub struct Client {
    handle: *mut c_void,
}

impl Client {
    /// Connect to Redis, register presence and start the background send/receive tasks.
    pub fn new(cfg: &Config) -> Result<Self, FfiError> {
        let watch = cfg.watch_projects.join(";");
        let (ru_p, ru_l) = bytes(&cfg.redis_url);
        let (kh_p, kh_l) = bytes(&cfg.key_hex);
        let (cn_p, cn_l) = bytes(&cfg.client_name);
        let (pw_p, pw_l) = bytes(&cfg.password);
        let (wp_p, wp_l) = bytes(&watch);
        let (st_p, st_l) = bytes(&cfg.send_to);
        unsafe {
            let handle = ffi_client_new(
                ru_p, ru_l, kh_p, kh_l, cn_p, cn_l, pw_p, pw_l, wp_p, wp_l, st_p, st_l,
            );
            if handle.is_null() {
                Err(FfiError(ffi_last_error()))
            } else {
                Ok(Self { handle })
            }
        }
    }

    /// Send a data file through the outbox.
    pub fn send(&self, name: &str, note: &str, data: &[u8]) -> Result<(), FfiError> {
        let (n_p, n_l) = bytes(name);
        let (no_p, no_l) = bytes(note);
        let (d_p, d_l) = raw(data);
        let code = unsafe { ffi_client_send(self.handle, n_p, n_l, no_p, no_l, d_p, d_l) };
        if code == OK {
            Ok(())
        } else {
            Err(FfiError(code))
        }
    }

    /// Receive a file, waiting up to `timeout_ms`. Returns `None` on timeout (not an error).
    pub fn recv(&mut self, timeout_ms: u32) -> Option<DataFile> {
        let ptr = unsafe { ffi_client_recv(self.handle, timeout_ms) };
        if ptr.is_null() {
            return None;
        }
        let f = unsafe { &*ptr };
        let file = DataFile {
            name: buf_to_string(&f.name),
            note: buf_to_string(&f.note),
            data: buf_to_vec(&f.data),
        };
        unsafe { ffi_datafile_free(ptr) };
        Some(file)
    }

    /// Release the client handle (idempotent; also runs on drop).
    pub fn close(&mut self) {
        if !self.handle.is_null() {
            unsafe { ffi_client_free(self.handle) };
            self.handle = std::ptr::null_mut();
        }
    }
}

impl Drop for Client {
    fn drop(&mut self) {
        self.close();
    }
}

// --- Helpers -----------------------------------------------------------------------

/// (ptr, len) for a string slice; empty slices pass null/0 ("not provided").
fn bytes(s: &str) -> (*const u8, usize) {
    if s.is_empty() {
        (std::ptr::null(), 0)
    } else {
        (s.as_ptr(), s.len())
    }
}

/// (ptr, len) for a byte slice; empty slices pass null/0.
fn raw(b: &[u8]) -> (*const u8, usize) {
    if b.is_empty() {
        (std::ptr::null(), 0)
    } else {
        (b.as_ptr(), b.len())
    }
}

/// Copy an `FfiBuffer` into a String (lossy UTF-8). Null/empty buffers yield "".
fn buf_to_string(buf: &FfiBuffer) -> String {
    if buf.ptr.is_null() || buf.len == 0 {
        return String::new();
    }
    let slice = unsafe { std::slice::from_raw_parts(buf.ptr, buf.len) };
    String::from_utf8_lossy(slice).into_owned()
}

/// Copy an `FfiBuffer` into a Vec<u8>. Null/empty buffers yield an empty vec.
fn buf_to_vec(buf: &FfiBuffer) -> Vec<u8> {
    if buf.ptr.is_null() || buf.len == 0 {
        return Vec::new();
    }
    let slice = unsafe { std::slice::from_raw_parts(buf.ptr, buf.len) };
    slice.to_vec()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn empty_buf() -> FfiBuffer {
        FfiBuffer { ptr: std::ptr::null_mut(), len: 0, cap: 0 }
    }

    #[test]
    fn null_empty_buffer_yields_empty_string() {
        assert_eq!(buf_to_string(&empty_buf()), "");
    }

    #[test]
    fn null_empty_buffer_yields_empty_vec() {
        assert!(buf_to_vec(&empty_buf()).is_empty());
    }

    #[test]
    fn nonempty_buffer_copies_content() {
        let bytes: &[u8] = b"hello";
        let buf = FfiBuffer { ptr: bytes.as_ptr() as *mut u8, len: bytes.len(), cap: bytes.len() };
        assert_eq!(buf_to_string(&buf), "hello");
        assert_eq!(buf_to_vec(&buf), vec![104, 101, 108, 108, 111]);
    }
}
