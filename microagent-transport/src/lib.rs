pub mod ffi;
pub mod transport;

pub use bytes::{Bytes, BytesMut};
pub use uuid::Uuid;
pub use transport::{Message, MessageKind, TransportError};

/// Error codes returned by the C ABI (see `ffi`). Kept here so both Rust and C
/// consumers can refer to the same constants.
pub const FFI_OK: i32 = 0;
pub const FFI_ERR_BAD_INPUT: i32 = 1;
pub const FFI_ERR_TOO_SHORT: i32 = 2;
pub const FFI_ERR_BAD_UUID: i32 = 3;
pub const FFI_ERR_BAD_UTF8: i32 = 4;
