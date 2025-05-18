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
    git \  # <--- ADD GIT HERE
    --no-install-recommends && \
    rm -rf /var/lib/apt/lists/*

RUN pip3 install torch==2.3.0 torchaudio==2.3.0

COPY requirements-gpu.txt /app/requirements-gpu.txt
RUN pip3 install --no-cache-dir -r /app/requirements-gpu.txt soundfile
RUN pip3 install --no-cache-dir ctranslate2==4.4.0 # Keep pinned for now

# Remove old copied directories if they were from previous builds layers (defensive)
RUN rm -rf /app/RealtimeSTT
RUN rm -rf /app/silero_assets

# Clone the RealtimeSTT repository. This creates /app/RealtimeSTT directory containing the library and assets.
RUN git clone https://github.com/qQ1aQ/RealtimeSTT.git /app/RealtimeSTT
# Now you have /app/RealtimeSTT/RealtimeSTT (the package) and /app/RealtimeSTT/silero_assets

# Your server.py COPY command needs to be relative to this new structure if it's part of the repo,
# or if you're using your own server.py, it needs to be able to find the RealtimeSTT package.
# Your current setup:
# COPY example_browserclient/server.py /app/example_browserclient/server.py
# This server.py is *your* version. It needs to find the RealtimeSTT package.
# The `RealtimeSTT` package from the clone is at `/app/RealtimeSTT/RealtimeSTT/`.
# And `silero_assets` is at `/app/RealtimeSTT/silero_assets/`.
# The library code (e.g., audio_recorder.py) is in `/app/RealtimeSTT/RealtimeSTT/audio_recorder.py`.
# It looks for `silero_assets` using `os.path.join(os.path.dirname(__file__), '..', 'silero_assets', SILERO_MODEL_NAME)`
# So from `/app/RealtimeSTT/RealtimeSTT/audio_recorder.py`, `os.path.dirname(__file__)` is `/app/RealtimeSTT/RealtimeSTT`.
# `..` goes to `/app/RealtimeSTT`.
# So it correctly finds `/app/RealtimeSTT/silero_assets`. This is good.

# For your server.py at `/app/example_browserclient/server.py` to successfully do `from RealtimeSTT import ...`
# it expects `RealtimeSTT` to be a top-level package.
# The cloned structure at `/app/RealtimeSTT/` has the `RealtimeSTT` *package* inside it.
# So the import path needed for Python is essentially `/app/RealtimeSTT` on the PYTHONPATH.

# Your current `ENV PYTHONPATH="${PYTHONPATH}:/app"` should be sufficient if Python's import mechanism
# correctly sees `/app/RealtimeSTT` as the location of the main project, and then can import the
# `RealtimeSTT` package which is a directory inside it.
# So `from RealtimeSTT import AudioToTextRecorder` should work.

# Re-confirming COPY lines from your Dockerfile:
# RUN mkdir -p example_browserclient # This is not needed if we COPY the directory
COPY example_browserclient/server.py /app/example_browserclient/server.py # This provides *your* modified server.py

# Expose the internal port the server runs on
EXPOSE 7860 # Your server.py listens on 7860 now

# ENV PYTHONPATH="${PYTHONPATH}:/app" # This allows imports from /app
# Your `server.py` is in `/app/example_browserclient/`
# It tries `from RealtimeSTT import AudioToTextRecorder`
# Python will look for `/app/RealtimeSTT/...` due to PYTHONPATH.
# The clone created `/app/RealtimeSTT/RealtimeSTT/__init__.py`.
# This means Python can find the package.

CMD ["python3", "example_browserclient/server.py"]
