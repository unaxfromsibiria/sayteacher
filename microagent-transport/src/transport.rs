use bytes::{BufMut, Bytes, BytesMut};
use sha2::{Digest, Sha256};
use uuid::Uuid;

/// First-audio-byte marker that tells the receiver what a frame is about.
/// The values are ASCII letters so both Rust and Python sides can build/parse them as plain u8.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MessageKind {
    /// Raw audio only; any text alongside it is informational (e.g. TTS of a "say" reply).
    Audio,
    /// A dialogue fragment: recent "you:" / "teacher:" lines in `text`, optional audio.
    Dialogue,
    /// A note or plain text message; audio may be present but is not speech content.
    Note,
    /// End-of-session marker from the teacher side; delivered after its final answer frame.
    Quit,
    /// Output-device capture stream from outvoiceagent (system audio playback).
    Output,
    /// Translation result for transcreenpt service.
    Translate,
    /// Keep-alive frame sent during silence so neither side times the connection out; echoed back by the peer.
    Ping,
}

impl MessageKind {
    pub fn from_byte(b: u8) -> Option<Self> {
        match b as char {
            'a' => Some(Self::Audio),
            'd' => Some(Self::Dialogue),
            'n' => Some(Self::Note),
            'q' => Some(Self::Quit),
            'o' => Some(Self::Output),
            't' => Some(Self::Translate),
            'p' => Some(Self::Ping),
            _ => None,
        }
    }

    pub fn byte(self) -> u8 {
        match self {
            Self::Audio => b'a',
            Self::Dialogue => b'd',
            Self::Note => b'n',
            Self::Quit => b'q',
            Self::Output => b'o',
            Self::Translate => b't',
            Self::Ping => b'p',
        }
    }
}

/// Transport message exchanged between server and client.
#[derive(Debug)]
pub struct Message {
    pub id: Uuid,
    pub text: String,
    pub audio: Bytes,
}

impl Message {
    /// Serialize into a flat byte sequence:
    /// `uuid(16) + text_len(u32 be) + text + audio_len(u32 be) + audio`.
    pub fn encode(&self) -> Bytes {
        let mut buf = BytesMut::new();
        buf.extend_from_slice(self.id.as_bytes());

        let t = self.text.as_bytes();
        buf.put_u32(t.len() as u32);
        buf.extend_from_slice(t);

        let a = &self.audio;
        buf.put_u32(a.len() as u32);
        buf.extend_from_slice(a);

        buf.freeze()
    }

    /// Deserialize from bytes produced by `encode`.
    pub fn decode(data: &[u8]) -> Result<Message, TransportError> {
        if data.len() < 16 + 4 + 4 {
            return Err(TransportError::TooShort);
        }

        let mut cursor = 0;

        // uuid (fixed 16 bytes)
        if data.len() - cursor < 16 {
            return Err(TransportError::TooShort);
        }
        let id_bytes: [u8; 16] = data[cursor..cursor + 16].try_into().map_err(|_| TransportError::BadUuid)?;
        let id = Uuid::from_slice(&id_bytes).map_err(|_| TransportError::BadUuid)?;
        cursor += 16;

        // text_len (u32 be) + text bytes
        if data.len() - cursor < 4 {
            return Err(TransportError::TooShort);
        }
        let text_len = u32::from_be_bytes([data[cursor], data[cursor+1], data[cursor+2], data[cursor+3]]) as usize;
        cursor += 4;
        if data.len() - cursor < text_len {
            return Err(TransportError::TooShort);
        }
        let text = String::from_utf8(data[cursor..cursor + text_len].to_vec()).map_err(|_| TransportError::BadUtf8)?;
        cursor += text_len;

        // audio_len (u32 be) + audio bytes
        if data.len() - cursor < 4 {
            return Err(TransportError::TooShort);
        }
        let audio_len = u32::from_be_bytes([data[cursor], data[cursor+1], data[cursor+2], data[cursor+3]]) as usize;
        cursor += 4;
        if data.len() - cursor < audio_len {
            return Err(TransportError::TooShort);
        }
        let audio = Bytes::copy_from_slice(&data[cursor..cursor + audio_len]);

        Ok(Message { id, text, audio })
    }

    /// The frame type from the first audio byte, or None when there is no audio at all.
    pub fn kind(&self) -> Option<MessageKind> {
        self.audio.first().and_then(|b| MessageKind::from_byte(*b))
    }

    /// Audio payload after the optional one-byte type marker (the "data" part of `[a|d|n][data]`).
    pub fn audio_body(&self) -> Bytes {
        match self.kind() {
            Some(_) => self.audio.slice(1..),
            None => self.audio.clone(),
        }
    }

    /// Build an audio field that starts with `kind` followed by the raw payload.
    pub fn audio_with_kind(kind: MessageKind, data: &[u8]) -> Bytes {
        let mut buf = BytesMut::new();
        buf.put_u8(kind.byte());
        buf.extend_from_slice(data);
        buf.freeze()
    }

    /// Greeting the server sends on connect: carries the session id used for auth.
    pub fn greeting(id: Uuid) -> Self {
        Message { id, text: String::new(), audio: Bytes::new() }
    }

    /// hex(sha256("<id>" + password)) — the token a client must echo back in `text`.
    pub fn auth_token_for(id: &Uuid, password: &str) -> String {
        let mut hasher = Sha256::new();
        hasher.update(id.to_string().as_bytes());
        hasher.update(password.as_bytes());
        hex::encode(hasher.finalize())
    }

    /// True when `text` is the valid auth token for this id and password.
    /// An empty text + empty password counts as a handshake between unauthenticated peers,
    /// so local echo clients keep working with no shared secret configured.
    pub fn verify_auth(&self, id: &Uuid, password: &str) -> bool {
        if self.text.is_empty() && password.is_empty() {
            return true;
        }
        const LEN: usize = 64; // sha256 hex digest length
        let expected = Self::auth_token_for(id, password);
        self.text.len() == LEN && self.text.eq_ignore_ascii_case(expected.as_str())
    }
}

#[derive(Debug)]
pub enum TransportError {
    TooShort,
    BadUuid,
    BadUtf8,
}

impl std::fmt::Display for TransportError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::TooShort => write!(f, "not enough data"),
            Self::BadUuid => write!(f, "invalid uuid bytes"),
            Self::BadUtf8 => write!(f, "text is not valid UTF-8"),
        }
    }
}

impl std::error::Error for TransportError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kind_and_audio_body_strip_the_type_byte() {
        let data: Vec<u8> = (0..12).collect();
        for kind in [MessageKind::Audio, MessageKind::Dialogue, MessageKind::Note, MessageKind::Output] {
            let msg = Message {
                id: Uuid::nil(),
                text: String::new(),
                audio: Message::audio_with_kind(kind, &data),
            };
            assert_eq!(msg.kind(), Some(kind));
            assert_eq!(&*msg.audio_body(), data.as_slice());
        }

        // No type byte at all -> kind is None and the body is the full payload.
        let plain = Message { id: Uuid::nil(), text: String::new(), audio: Bytes::from(data.clone()) };
        assert_eq!(plain.kind(), None);
        assert_eq!(&*plain.audio_body(), data.as_slice());

        // Empty audio -> no kind, empty body.
        let empty = Message { id: Uuid::nil(), text: String::new(), audio: Bytes::new() };
        assert_eq!(empty.kind(), None);
        assert!(empty.audio_body().is_empty());
    }

    #[test]
    fn ping_kind_roundtrips() {
        // Python mirrors this exact code (p) in its MessageKind enum.
        assert_eq!(MessageKind::Ping.byte(), b'p');
        assert_eq!(MessageKind::from_byte(b'p'), Some(MessageKind::Ping));

        let msg = Message {
            id: Uuid::new_v4(),
            text: String::new(),
            audio: Message::audio_with_kind(MessageKind::Ping, &[]),
        };
        assert_eq!(msg.kind(), Some(MessageKind::Ping));
        let decoded = Message::decode(&msg.encode()).unwrap();
        assert_eq!(decoded.id, msg.id);
        assert_eq!(decoded.kind(), Some(MessageKind::Ping));
    }

    #[test]
    fn kind_byte_values_are_ascii_markers() {
        // Python mirrors these exact codes (a/n/d) in its MessageKind enum.
        assert_eq!(MessageKind::Audio.byte(), b'a');
        assert_eq!(MessageKind::Note.byte(), b'n');
        assert_eq!(MessageKind::Dialogue.byte(), b'd');

        let mut lone = BytesMut::new();
        lone.put_u8(b'x');
        let msg = Message { id: Uuid::nil(), text: String::new(), audio: lone.freeze() };
        assert_eq!(msg.kind(), None);
    }

    #[test]
    fn roundtrip_empty() {
        let msg = Message { id: Uuid::new_v4(), text: String::new(), audio: Bytes::new() };
        let encoded = msg.encode();
        let decoded = Message::decode(&encoded).unwrap();
        assert_eq!(decoded.id, msg.id);
        assert!(decoded.text.is_empty());
        assert!(decoded.audio.is_empty());
    }

    #[test]
    fn roundtrip_with_text_and_audio() {
        let text = "hello world";
        let audio: Vec<u8> = (0..128).collect();
        let msg = Message { id: Uuid::new_v4(), text: text.into(), audio: Bytes::from(audio) };
        let encoded = msg.encode();
        let decoded = Message::decode(&encoded).unwrap();
        assert_eq!(decoded.id, msg.id);
        assert_eq!(decoded.text, "hello world");
        assert_eq!(decoded.audio.as_ref(), (0..128u8).collect::<Vec<_>>().as_slice());
    }

    #[test]
    fn decode_too_short() {
        assert!(Message::decode(&[0u8; 4]).is_err());
    }

    #[test]
    fn auth_token_is_64_hex_chars_and_deterministic() {
        let id = Uuid::new_v4();
        let t1 = Message::auth_token_for(&id, "secret");
        let t2 = Message::auth_token_for(&id, "secret");
        assert_eq!(t1.len(), 64);
        assert_eq!(t1, t2);
    }

    #[test]
    fn auth_token_depends_on_password_and_id() {
        let id = Uuid::new_v4();
        let other = Uuid::new_v4();
        assert_ne!(Message::auth_token_for(&id, "a"), Message::auth_token_for(&id, "b"));
        assert_ne!(Message::auth_token_for(&id, "a"), Message::auth_token_for(&other, "a"));
    }

    #[test]
    fn verify_auth_accepts_valid_rejects_invalid() {
        let id = Uuid::new_v4();
        let token = Message::auth_token_for(&id, "secret");
        let ok_msg = Message { id, text: token.clone(), audio: Bytes::new() };
        assert!(ok_msg.verify_auth(&id, "secret"));

        let wrong_pass = Message { id, text: Message::auth_token_for(&id, "wrong"), audio: Bytes::new() };
        assert!(!wrong_pass.verify_auth(&id, "secret"));

        let empty = Message { id, text: String::new(), audio: Bytes::new() };
        assert!(!empty.verify_auth(&id, "secret"));
    }
}
