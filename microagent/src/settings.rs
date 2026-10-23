use std::fmt;
use std::str::FromStr;

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
            SignalLevel::Low => 0.01,
            SignalLevel::Middle => 0.02,
            SignalLevel::High => 0.045,
        }
    }
}

/// Runtime settings loaded from CLI args at startup.
#[derive(Debug, Clone)]
pub struct Settings {
    /// Recording duration in seconds per cycle (`--delay`), fractional allowed.
    pub delay: f64,
    /// Required signal level for non-silent detection (`--level low|middle|high`).
    pub level: SignalLevel,
    /// True when the program should run as an echo test client (`--client` was passed).
    pub client: bool,
    /// Client identity prefix used to name datatraffic clients.
    pub client_name: String,
    /// Redis connection URL; empty falls back to env `REDIS_URL`.
    pub redis_url: String,
    /// AES-256-GCM key (64 hex chars); empty falls back to env `CRYPTO_KEY`.
    pub crypto_key: String,
    /// Redis password; empty means no auth.
    pub redis_password: String,
    /// Content of the text file passed with `--lesson`, sent as greeting to sayteacher.
    pub lesson_data: String,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            delay: 3.5,
            level: SignalLevel::Middle,
            client: false,
            client_name: "microagent".to_string(),
            redis_url: String::new(),
            crypto_key: String::new(),
            redis_password: String::new(),
            lesson_data: String::new(),
        }
    }
}

impl Settings {
    /// Full CLI help text for `--help` / `-h`.
    pub fn help_text() -> String {
        "microagent — datatraffic audio server + UI, or an echo test client.

usage: microagent [options]
       Run as a server by default; pass --client to run the echo test client instead.

Options:
  -h, --help              Show this help and exit.
      --delay <SEC>       Recording interval in seconds; fractional allowed [default: 3.5]
      --level <LVL>       Signal level low|middle|high [default: middle]
      --client            Run as an echo test client (no UI)
      --lesson <FILE>     Lesson file read at startup and sent to sayteacher as greeting
      --redis-url <URL>   Redis connection URL [env: REDIS_URL]
      --crypto-key <KEY>  AES-256-GCM key, 64 hex chars [env: CRYPTO_KEY]
      --redis-password <P> Redis password (empty = no auth)
      --client-name <N>   Client identity prefix [default: microagent]

Examples:
  microagent                                   # server, delay 3.5s
  microagent --level high --lesson lesson.txt  # server with lesson file
  microagent --client                          # echo test client
"
        .to_string()
    }

    /// Parse command line args into Settings.
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
                "--client" => {
                    settings.client = true;
                    i += 1;
                },
                "--lesson" => {
                    if i + 1 >= args.len() {
                        log::error!("--lesson requires a file path value");
                        std::process::exit(2);
                    }
                    let path = args[i + 1].as_str();
                    match std::fs::read_to_string(path) {
                        Ok(content) => {
                            settings.lesson_data = content;
                            log::info!("lesson loaded from {} ({} chars)", path, settings.lesson_data.len());
                        }
                        Err(e) => {
                            log::error!("cannot read lesson file {}: {}", path, e);
                            std::process::exit(1);
                        }
                    }
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
                },
            }
        }
        settings
    }
}

#[cfg(test)]
mod tests {
    use super::SignalLevel;

    #[test]
    fn level_parses_each_variant() {
        assert_eq!("low".parse::<SignalLevel>().unwrap(), SignalLevel::Low);
        assert_eq!("middle".parse::<SignalLevel>().unwrap(), SignalLevel::Middle);
        assert_eq!("high".parse::<SignalLevel>().unwrap(), SignalLevel::High);
        // Case-insensitive and trimmed.
        assert_eq!("HIGH ".parse::<SignalLevel>().unwrap(), SignalLevel::High);
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
    fn level_display_roundtrip() {
        for lvl in [SignalLevel::Low, SignalLevel::Middle, SignalLevel::High] {
            let s = lvl.to_string();
            assert_eq!(s.parse::<SignalLevel>().unwrap(), lvl);
        }
    }

    #[test]
    fn help_text_lists_all_options() {
        let h = super::Settings::help_text();
        for flag in [
            "--delay", "--level", "--client", "--lesson",
            "--redis-url", "--crypto-key", "--redis-password", "--client-name", "usage",
        ] {
            assert!(h.contains(flag), "missing '{}' in help text", flag);
        }
    }

    #[test]
    fn defaults_are_sane() {
        let s = super::Settings::default();
        assert_eq!(s.delay, 3.5);
        assert!(!s.client);
        assert_eq!(s.client_name, "microagent");
        assert!(s.redis_url.is_empty());
        assert!(s.crypto_key.is_empty());
    }
}
