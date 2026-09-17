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
    /// TCP host to listen on (`--listen HOST:PORT`, default 127.0.0.1).
    pub listen_host: String,
    /// TCP port to listen on (default 16050).
    pub listen_port: u16,
    /// True when the program should run as a client (`--connect` was passed).
    pub client: bool,
    /// TCP server host to connect to in client mode (`--connect HOST:PORT`).
    pub connect_host: String,
    /// TCP server port to connect to (default 16050).
    pub connect_port: u16,
    /// Required signal level for non-silent detection (`--level low|middle|high`).
    pub level: SignalLevel,
    /// Shared secret used for the auth handshake; empty disables token matching.
    pub password: String,
    /// Content of the text file passed with `--lesson`, sent to each new client.
    pub lesson_data: String,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            delay: 3.5,
            level: SignalLevel::Middle,
            listen_host: "127.0.0.1".to_string(),
            listen_port: 16050,
            client: false,
            connect_host: "127.0.0.1".to_string(),
            connect_port: 16050,
            password: String::new(),
            lesson_data: String::new(),
        }
    }
}

impl Settings {
    /// Full socket address string used for `TcpListener::bind`.
    pub fn listen_addr(&self) -> String {
        format!("{}:{}", self.listen_host, self.listen_port)
    }

    /// Human-readable `HOST:PORT` string the client connects to.
    pub fn connect_addr_str(&self) -> String {
        format!("{}:{}", self.connect_host, self.connect_port)
    }

    /// Full CLI help text for `--help` / `-h`.
    pub fn help_text() -> String {
        "microagent — TCP audio server + UI, or an echo test client.

usage: microagent [options]
       Run as a server by default; pass --connect to run the client instead.

Options:
  -h, --help            Show this help and exit.
      --delay <SEC>     Recording interval in seconds; fractional allowed [default: 3.5]
      --level <LVL>     Signal level low|middle|high [default: middle]
      --listen <ADDR>   Host:port to listen on (server) [default: 127.0.0.1:16050]
      --connect <ADDR>  Host:port of the server; switches to client mode [default: 127.0.0.1:16050]
      --password <PWD>  Shared secret for the auth handshake (empty disables it)
      --lesson <FILE>   Lesson file read at startup and sent to each new client

Examples:
  microagent                                   # server on 127.0.0.1:16050, delay 3.5s
  microagent --listen 0.0.0.0:16050 --level high
  microagent --connect 192.168.1.20:16050      # echo test client
"
        .to_string()
    }

    /// Parse `--delay` and `--listen` from command line args.
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
                    let parsed = match args[i + 1].to_socket_addrs().and_then(|mut it| {
                        it.next().ok_or(std::io::Error::new(std::io::ErrorKind::InvalidInput, "empty socket address"))
                    }) {
                        Ok(a) => a,
                        Err(e) => {
                            log::error!("invalid --listen value {}: {}", args[i + 1], e);
                            std::process::exit(2);
                        },
                    };
                    settings.listen_host = format!("{}", parsed.ip());
                    settings.listen_port = parsed.port();
                    i += 2;
                },
                "--connect" => {
                    if i + 1 >= args.len() {
                        log::error!("--connect requires a HOST:PORT value");
                        std::process::exit(2);
                    }
                    let parsed = match args[i + 1].to_socket_addrs().and_then(|mut it| {
                        it.next().ok_or(std::io::Error::new(std::io::ErrorKind::InvalidInput, "empty socket address"))
                    }) {
                        Ok(a) => a,
                        Err(e) => {
                            log::error!("invalid --connect value {}: {}", args[i + 1], e);
                            std::process::exit(2);
                        },
                    };
                    settings.connect_host = format!("{}", parsed.ip());
                    settings.connect_port = parsed.port();
                    // Passing --connect switches the program into client (echo) mode.
                    settings.client = true;
                    i += 2;
                },
                "--password" => {
                    if i + 1 >= args.len() {
                        log::error!("--password requires a value");
                        std::process::exit(2);
                    }
                    settings.password = args[i + 1].clone();
                    i += 2;
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
        for flag in ["--delay", "--level", "--listen", "--connect", "--password", "--lesson", "usage"] {
            assert!(h.contains(flag), "missing '{}' in help text", flag);
        }
    }
}
