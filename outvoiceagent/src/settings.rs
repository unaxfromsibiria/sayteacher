use std::fmt;
use std::str::FromStr;

use super::datatraffic::Config as TrafficConfig;

/// datatraffic project that carries captured audio segments (f32 LE mono @ 16 kHz).
/// outvoiceagent publishes here; transcreenpt watches it.
pub const AUDIO_PROJECT: &str = "out-audio";
/// datatraffic project that carries translation groups back to the UI overlay.
/// transcreenpt publishes one file per group (name = zero-padded counter, note = text);
/// outvoiceagent watches it and routes each file to its UI group.
pub const TEXT_PROJECT: &str = "trans-text";

/// Required signal level for a sound to be considered non-silent.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SignalLevel {
    Low,
    Middle,
    High,
}

impl fmt::Display for SignalLevel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let txt = match self {
            SignalLevel::Low => "low",
            SignalLevel::Middle => "middle",
            SignalLevel::High => "high",
        };
        write!(f, "{}", txt)
    }
}

impl FromStr for SignalLevel {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().trim() {
            "low" => Ok(SignalLevel::Low),
            "" | "middle" => Ok(SignalLevel::Middle),
            "high" => Ok(SignalLevel::High),
            _ => Err(format!("invalid --level value: '{}'", s)),
        }
    }
}

impl SignalLevel {
    /// Peak-amplitude cutoff: samples below this count as silence. Higher for
    /// noisier conditions, so only loud voices pass and background noise is dropped.
    pub fn threshold(self) -> f32 {
        match self {
            SignalLevel::Low => 0.001,
            SignalLevel::Middle => 0.005,
            SignalLevel::High => 0.01,
        }
    }
}

/// Runtime settings loaded from CLI args at startup.
#[derive(Debug, Clone)]
pub struct Settings {
    /// Capture interval in seconds (`--delay`), fractional allowed.
    pub delay: f64,
    /// Required signal level for non-silent detection (`--level low|middle|high`).
    pub level: SignalLevel,
    /// When true the program runs as a test client instead of a server (`--connect`).
    pub client: bool,
    /// When set (non-empty), the program runs as a check client that saves received
    /// audio segments to WAV files in this directory (`--check /tmp/`). Implies client mode.
    pub check_dir: String,
    /// Name of capture device; empty = auto-select (monitor on Linux, default output on Windows).
    pub device: String,
    /// Redis connection URL for datatraffic (`--redis-url` / env `REDIS_URL`).
    pub redis_url: String,
    /// AES-256-GCM key as 64 hex chars (`--crypto-key` / env `CRYPTO_KEY`).
    pub crypto_key: String,
    /// Redis password (`--redis-password` / env `REDIS_PASSWORD`); empty = no auth.
    pub redis_password: String,
    /// Client identity registered with datatraffic (`--client-name`).
    pub client_name: String,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            delay: 6.,
            level: SignalLevel::Middle,
            client: false,
            check_dir: String::new(),
            device: String::new(),
            redis_url: std::env::var("REDIS_URL").unwrap_or_else(|_| "redis://localhost:6379".to_string()),
            crypto_key: std::env::var("CRYPTO_KEY").unwrap_or_default(),
            redis_password: std::env::var("REDIS_PASSWORD").unwrap_or_default(),
            client_name: "outvoiceagent".to_string(),
        }
    }
}

impl Settings {
    /// True when the program should run in check (WAV-capturing) client mode.
    pub fn check_mode(&self) -> bool {
        !self.check_dir.is_empty()
    }

    /// datatraffic config for server mode: watch translation groups, publish captured audio.
    pub fn server_config(&self) -> TrafficConfig {
        TrafficConfig {
            redis_url: self.redis_url.clone(),
            key_hex: self.crypto_key.clone(),
            client_name: self.client_name.clone(),
            password: self.redis_password.clone(),
            watch_projects: vec![TEXT_PROJECT.to_string()],
            send_to: AUDIO_PROJECT.to_string(),
        }
    }

    /// datatraffic config for test/check client mode: watch captured audio, publish nothing.
    pub fn client_config(&self) -> TrafficConfig {
        TrafficConfig {
            redis_url: self.redis_url.clone(),
            key_hex: self.crypto_key.clone(),
            client_name: format!("{}-client", self.client_name),
            password: self.redis_password.clone(),
            watch_projects: vec![AUDIO_PROJECT.to_string()],
            send_to: String::new(),
        }
    }

    /// Full CLI help text for `-h` / `--help`.
    pub fn help_text() -> String {
        "outvoiceagent — captures system audio output and streams it over datatraffic (Redis).

usage: outvoiceagent [options]

Runs as a server by default; pass --connect to run the test client instead.
Connection settings come from --redis-url/--crypto-key/--redis-password or the
REDIS_URL/CRYPTO_KEY/REDIS_PASSWORD environment variables. CRYPTO_KEY is required.

Options:
  -h, --help              Show this help and exit.
      --delay <SEC>       Capture interval in seconds; fractional allowed [default: 6]
      --level <LVL>       Signal level low|middle|high for the silence threshold [default: middle]
      --connect           Run as a test client that prints received audio segments
      --check <DIR>       Check mode: save received audio segments to out_<id>.wav in DIR
      --device <NAME>     Capture device name (Linux monitor, Windows loopback); auto = 'auto'
      --redis-url <URL>   Redis URL [default: env REDIS_URL or redis://localhost:6379]
      --crypto-key <KEY>  AES-256-GCM key, 64 hex chars [default: env CRYPTO_KEY]
      --redis-password <P> Redis password [default: env REDIS_PASSWORD or empty]
      --client-name <N>   datatraffic client identity [default: outvoiceagent]

Examples:
  outvoiceagent                            # server: capture loopback, publish every 6s to out-audio
  outvoiceagent --connect                  # test client that prints received audio segments
  outvoiceagent --check /tmp/              # check mode: save received segments as WAV files
"
        .to_string()
    }

    /// Parse known arguments from the command line.
    pub fn from_args() -> Self {
        let mut settings = Self::default();
        let args: Vec<String> = std::env::args().collect();
        let mut i = 1;
        while i < args.len() {
            match args[i].as_str() {
                "-h" | "--help" => {
                    println!("{}", Self::help_text());
                    std::process::exit(0);
                },
                "--delay" => {
                    if i + 1 >= args.len() {
                        log::error!("--delay requires a value");
                        std::process::exit(2);
                    }
                    match args[i + 1].parse::<f64>() {
                        Ok(v) if v.is_finite() && v > 0.0 => settings.delay = v,
                        _ => {
                            log::error!("invalid --delay value: {}", args[i + 1]);
                            std::process::exit(2);
                        }
                    }
                    i += 2;
                },
                "--level" => {
                    if i + 1 >= args.len() {
                        log::error!("--level requires a value");
                        std::process::exit(2);
                    }
                    match args[i + 1].parse::<SignalLevel>() {
                        Ok(v) => settings.level = v,
                        Err(e) => {
                            log::error!("{}", e);
                            std::process::exit(2);
                        }
                    }
                    i += 2;
                },
                "--connect" => {
                    // Switching into client (test) mode.
                    settings.client = true;
                    i += 1;
                },
                "--check" => {
                    if i + 1 >= args.len() {
                        log::error!("--check requires a directory value");
                        std::process::exit(2);
                    }
                    // Check mode implies client (test) mode: listen and save WAV files.
                    settings.check_dir = args[i + 1].clone();
                    settings.client = true;
                    i += 2;
                },
                "--device" => {
                    if i + 1 >= args.len() {
                        log::error!("--device requires a value");
                        std::process::exit(2);
                    }
                    settings.device = args[i + 1].clone();
                    i += 2;
                },
                "--redis-url" => {
                    if i + 1 >= args.len() {
                        log::error!("--redis-url requires a value");
                        std::process::exit(2);
                    }
                    settings.redis_url = args[i + 1].clone();
                    i += 2;
                },
                "--crypto-key" => {
                    if i + 1 >= args.len() {
                        log::error!("--crypto-key requires a value");
                        std::process::exit(2);
                    }
                    settings.crypto_key = args[i + 1].clone();
                    i += 2;
                },
                "--redis-password" => {
                    if i + 1 >= args.len() {
                        log::error!("--redis-password requires a value");
                        std::process::exit(2);
                    }
                    settings.redis_password = args[i + 1].clone();
                    i += 2;
                },
                "--client-name" => {
                    if i + 1 >= args.len() {
                        log::error!("--client-name requires a value");
                        std::process::exit(2);
                    }
                    settings.client_name = args[i + 1].clone();
                    i += 2;
                },
                other => {
                    log::warn!("ignoring unknown argument: {}", other);
                    i += 1;
                }
            }
        }
        settings
    }
}

#[cfg(test)]
mod tests {
    use super::{Settings, SignalLevel};

    #[test]
    fn defaults_are_sane() {
        let s = Settings::default();
        assert!(s.delay > 0.0);
        assert_eq!(s.level, SignalLevel::Middle);
        assert!(!s.client);
        assert!(!s.check_mode());
        // Project names are fixed and distinct.
        assert_ne!(super::AUDIO_PROJECT, super::TEXT_PROJECT);
    }

    #[test]
    fn server_config_watches_text_and_publishes_audio() {
        let s = Settings {
            redis_url: "redis://example:6379".into(),
            crypto_key: "ab".repeat(32),
            client_name: "srv".into(),
            ..Settings::default()
        };
        let cfg = s.server_config();
        assert_eq!(cfg.watch_projects, vec![super::TEXT_PROJECT.to_string()]);
        assert_eq!(cfg.send_to, super::AUDIO_PROJECT);
        assert_eq!(cfg.redis_url, "redis://example:6379");
        assert_eq!(cfg.client_name, "srv");
    }

    #[test]
    fn client_config_watches_audio_and_publishes_nothing() {
        let s = Settings {
            client_name: "cli".into(),
            ..Settings::default()
        };
        let cfg = s.client_config();
        assert_eq!(cfg.watch_projects, vec![super::AUDIO_PROJECT.to_string()]);
        assert!(cfg.send_to.is_empty());
        assert_eq!(cfg.client_name, "cli-client");
    }

    #[test]
    fn level_parses_each_variant() {
        assert_eq!("low".parse::<SignalLevel>().unwrap(), SignalLevel::Low);
        assert_eq!("middle".parse::<SignalLevel>().unwrap(), SignalLevel::Middle);
        assert_eq!("high".parse::<SignalLevel>().unwrap(), SignalLevel::High);
    }

    #[test]
    fn level_rejects_unknown_value() {
        let err = "extreme".parse::<SignalLevel>().expect_err("expected error");
        assert!(err.contains("invalid --level"));
    }

    #[test]
    fn thresholds_increase_with_level() {
        let low = SignalLevel::Low.threshold();
        let mid = SignalLevel::Middle.threshold();
        let high = SignalLevel::High.threshold();
        assert!(low < mid);
        assert!(mid < high);
    }

    #[test]
    fn help_text_lists_all_options() {
        let h = Settings::help_text();
        for flag in [
            "--delay", "--level", "--connect", "--check", "--device",
            "--redis-url", "--crypto-key", "usage",
        ] {
            assert!(h.contains(flag), "missing '{}' in help text", flag);
        }
    }
}
