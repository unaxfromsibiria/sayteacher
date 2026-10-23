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

WORKDIR /app

# pydatatraffic wheel is prebuilt by vendor/datatraffic (see Makefile: build-pydatatraffic)
# and staged into ./dist/. It bundles the compiled libdatatraffic shared library.
COPY dist/ ./dist/
RUN python3 -m pip install --no-cache-dir --break-system-packages \
        ./dist/pydatatraffic-*.whl

COPY setup.cfg ./
COPY sayteacher/ ./sayteacher/
RUN python3 -m pip install --no-cache-dir --break-system-packages \
        -r sayteacher/requirements.txt \
    && ln -sf "$(command -v python3)" /usr/local/bin/python

# Datatraffic connection (REDIS_URL/CRYPTO_KEY/REDIS_PASSWORD) is read from the environment
# by sayteacher.transport; CRYPTO_KEY is required and must be provided at run time.
ENV REDIS_URL="redis://localhost:6379" \
    MODEL="distil-medium.en" \
    LANGUAGE="English" \
    SPEAKER="en_101" \
    LOGLEVEL="INFO"

ENTRYPOINT ["sh", "-c", "python -m sayteacher.main \
    --model \"${MODEL:-distil-medium.en}\" \
    --lang \"${LANGUAGE:-English}\" \
    --llm-model \"${LLM_MODEL:-}\" \
    --llm-api \"${LLM_API:-}\" \
    --speaker \"${SPEAKER:-en_101}\""]
