use std::fmt;
use std::net::ToSocketAddrs;
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
    /// TCP host to listen on (server mode). Default 127.0.0.1.
    pub listen_host: String,
    /// TCP port to listen on (default 16050).
    pub listen_port: u16,
    /// When true the program runs as a test client instead of a server (`--connect`).
    pub client: bool,
    /// Host of the outvoiceagent server in client mode. Default 127.0.0.1.
    pub connect_host: String,
    /// Port of the server in client mode (default 16050).
    pub connect_port: u16,
    /// When set (non-empty), the program runs as a check client that saves received
    /// output frames to WAV files in this directory (`--check /tmp/`).
    pub check_dir: String,
    /// Name of capture device; empty = auto-select (monitor on Linux, default output on Windows).
    pub device: String,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            delay: 6.,
            level: SignalLevel::Middle,
            listen_host: "127.0.0.1".to_string(),
            listen_port: 16050,
            client: false,
            connect_host: "127.0.0.1".to_string(),
            connect_port: 16050,
            check_dir: String::new(),
            device: String::new(),
        }
    }
}

impl Settings {
    /// True when the program should run in check (WAV-capturing) client mode.
    pub fn check_mode(&self) -> bool {
        !self.check_dir.is_empty()
    }

    /// Full socket address string used for `TcpListener::bind`.
    pub fn listen_addr(&self) -> String {
        format!("{}:{}", self.listen_host, self.listen_port)
    }

    /// `HOST:PORT` string the client connects to.
    pub fn connect_addr_str(&self) -> String {
        format!("{}:{}", self.connect_host, self.connect_port)
    }

    /// Full CLI help text for `-h` / `--help`.
    pub fn help_text() -> String {
        "outvoiceagent — TCP server that captures system audio output and streams it to a client.

usage: outvoiceagent [options]

Runs as a server by default; pass --connect to run the test client instead.

Options:
  -h, --help            Show this help and exit.
      --delay <SEC>     Capture interval in seconds; fractional allowed [default: 6]
      --level <LVL>     Signal level low|middle|high for the silence threshold [default: middle]
      --listen <ADDR>   Host:port to listen on (server) [default: 127.0.0.1:16050]
      --connect <ADDR>  Server host:port for the test client mode [default: 127.0.0.1:16050]
      --check <DIR>     Check mode: save received output frames to out_<uuid>.wav in DIR
      --device <NAME>   Capture device name (Linux monitor, Windows loopback); auto = 'auto'

Examples:
  outvoiceagent                            # server: capture loopback, send every 3.5s
  outvoiceagent --listen 0.0.0.0:16050     # listen on all interfaces
  outvoiceagent --connect 127.0.0.1:16050  # test client that prints received frames
  outvoiceagent --check /tmp/               # check mode: save output frames as WAV files
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
                "--listen" => {
                    if i + 1 >= args.len() {
                        log::error!("--listen requires a HOST:PORT value");
                        std::process::exit(2);
                    }
                    let parsed = parse_addr(&args[i + 1]);
                    settings.listen_host = format!("{}", parsed.ip());
                    settings.listen_port = parsed.port();
                    i += 2;
                },
                "--connect" => {
                    if i + 1 >= args.len() {
                        log::error!("--connect requires a HOST:PORT value");
                        std::process::exit(2);
                    }
                    let parsed = parse_addr(&args[i + 1]);
                    settings.connect_host = format!("{}", parsed.ip());
                    settings.connect_port = parsed.port();
                    // Switching into client (test) mode.
                    settings.client = true;
                    i += 2;
                },
                "--check" => {
                    if i + 1 >= args.len() {
                        log::error!("--check requires a directory value");
                        std::process::exit(2);
                    }
                    // Check mode implies client (test) mode: connect and save WAV files.
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
                other => {
                    log::warn!("ignoring unknown argument: {}", other);
                    i += 1;
                }
            }
        }
        settings
    }
}

/// Resolve a `HOST:PORT` string to a socket address, logging and exiting on failure.
fn parse_addr(value: &str) -> std::net::SocketAddr {
    match value.to_socket_addrs().and_then(|mut it| {
        it.next().ok_or_else(|| std::io::Error::new(std::io::ErrorKind::InvalidInput, "empty socket address"))
    }) {
        Ok(a) => a,
        Err(e) => {
            log::error!("invalid address value '{}': {}", value, e);
            std::process::exit(2);
        }
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
        assert_eq!(s.listen_addr(), "127.0.0.1:16050");
        assert_eq!(s.connect_addr_str(), "127.0.0.1:16050");
        assert!(!s.client);
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
        for flag in ["--delay", "--level", "--listen", "--connect", "usage"] {
            assert!(h.contains(flag), "missing '{}' in help text", flag);
        }
    }
}
