FROM nvidia/cuda:13.3.1-runtime-ubuntu26.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

RUN apt-get update
RUN apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        ca-certificates \
        pkg-config \
        libssl-dev \
        python3 python3-dev python3-pip python3-venv \
        libasound2t64 \
    && rm -rf /var/lib/apt/lists/*

# Rust toolchain (kept on PATH via the default home profile; rustup is in /usr/local).
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y \
    && . "$HOME/.cargo/env" && rustc --version

WORKDIR /app

COPY Cargo.toml Cargo.lock ./
COPY microagent/ ./microagent/
COPY microagent-transport/ ./microagent-transport/

RUN . "$HOME/.cargo/env" && cargo build --release -p microagent-transport

COPY setup.cfg ./
COPY sayteacher/ ./sayteacher/
RUN ln -sfn ../target/release sayteacher/lib && test -e sayteacher/lib/libmicroagent_transport.so

RUN python3 -m pip install --no-cache-dir --break-system-packages -r sayteacher/requirements.txt \
    && ln -sf "$(command -v python3)" /usr/local/bin/python

# Non-sensitive defaults baked in so a bare `docker run` works; sensitive values (PASSWORD,
# LLM_API) stay purely in compose's environment section. LOGLEVEL gates main.py logging.
ENV CONNECT="127.0.0.1:16050" \
    MODEL="bond005/whisper-podlodka-turbo" \
    LANGUAGE="English" \
    SPEAKER="en_101" \
    LOGLEVEL="INFO"

# Shell-form entrypoint so ${VAR} is expanded by /bin/sh. Empty defaults (${VAR:-}) keep a
# bare `docker run` from passing an unset value to click; compose overrides any of these
# via its environment section (e.g. CONNECT, PASSWORD, LLM_API).
ENTRYPOINT ["sh", "-c", "python -m sayteacher.main \
    --connect \"${CONNECT:-127.0.0.1:16050}\" \
    --password \"${PASSWORD:-}\" \
    --model \"${MODEL:-bond005/whisper-podlodka-turbo}\" \
    --lang \"${LANGUAGE:-English}\" \
    --llm-model \"${LLM_MODEL:-}\" \
    --llm-api \"${LLM_API:-}\" \
    --speaker \"${SPEAKER:-en_101}\""]
