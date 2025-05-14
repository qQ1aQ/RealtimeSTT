FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04 AS gpu

WORKDIR /app

# Install necessary system dependencies
RUN apt-get update -y && \
    apt-get install -y \
    python3 \
    python3-dev \
    python3-pip \
    libcudnn8 \
    libcudnn8-dev \
    libcublas-12-4 \
    portaudio19-dev \
    libasound2-dev \
    libsndfile1 \
    build-essential \
    --no-install-recommends && \
    rm -rf /var/lib/apt/lists/*

RUN pip3 install torch==2.3.0 torchaudio==2.3.0

COPY requirements-gpu.txt /app/requirements-gpu.txt
RUN pip3 install --no-cache-dir -r /app/requirements-gpu.txt soundfile

# Downgrade ctranslate2 to be compatible with cuDNN 8.x
RUN pip3 install --no-cache-dir ctranslate2==4.4.0

# Use -p to create parent directories if they don't exist
RUN mkdir -p example_browserclient
COPY example_browserclient/server.py /app/example_browserclient/server.py

# Copies the RealtimeSTT package and the Silero assets
COPY RealtimeSTT /app/RealtimeSTT
COPY silero_assets /app/silero_assets


# Expose the internal port the server runs on
EXPOSE 9001

ENV PYTHONPATH="${PYTHONPATH}:/app"

CMD ["python3", "example_browserclient/server.py"]

# --------------------------------------------

FROM ubuntu:22.04 AS cpu

WORKDIR /app

# Install necessary system dependencies
RUN apt-get update -y && \
    apt-get install -y \
    python3 \
    python3-dev \
    python3-pip \
    portaudio19-dev \
    libasound2-dev \
    libsndfile1 \
    build-essential \
    --no-install-recommends && \
    rm -rf /var/lib/apt/lists/*

RUN pip3 install torch==2.3.0 torchaudio==2.3.0

COPY requirements.txt /app/requirements.txt
RUN pip3 install --no-cache-dir -r /app/requirements.txt soundfile

# Downgrade ctranslate2 (relevant for CPU if faster-whisper with ctranslate2 is used on CPU)
# This might not be strictly necessary for CPU-only if a different STT engine version is used,
# but good for consistency if requirements.txt also pulls a newer ctranslate2.
RUN pip3 install --no-cache-dir ctranslate2==4.4.0

# Expose the internal port the server runs on
EXPOSE 9001

ENV PYTHONPATH="${PYTHONPATH}:/app"

CMD ["python3", "example_browserclient/server.py"]