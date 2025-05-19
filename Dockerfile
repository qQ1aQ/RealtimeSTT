FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04 AS gpu

WORKDIR /app

# Install necessary system dependencies including git
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
    git \
    --no-install-recommends && \
    rm -rf /var/lib/apt/lists/*

RUN pip3 install torch==2.3.0 torchaudio==2.3.0

# This requirements-gpu.txt MUST contain 'onnxruntime-gpu'
COPY requirements-gpu.txt /app/requirements-gpu.txt
RUN pip3 install --no-cache-dir -r /app/requirements-gpu.txt soundfile

# Pin ctranslate2 for current cuDNN compatibility with faster-whisper
RUN pip3 install --no-cache-dir ctranslate2==4.4.0

# Remove any old versions if they exist from previous layers to ensure clean clone
RUN rm -rf /app/RealtimeSTT
RUN rm -rf /app/silero_assets # This ensures /app/silero_assets is clean before a potential move

# Clone the latest RealtimeSTT repository
# This will place 'RealtimeSTT' package and 'silero_assets' inside /app/RealtimeSTT/
RUN git clone https://github.com/qQ1aQ/RealtimeSTT.git /app/RealtimeSTT

# Move silero_assets to /app/silero_assets, where the library seems to expect it.
RUN mv /app/RealtimeSTT/silero_assets /app/silero_assets

# Copy your custom server.py (assuming it's in 'example_browserclient' in your build context)
# Adjust the source path if your local 'server.py' is elsewhere.
COPY example_browserclient/server.py /app/example_browserclient/server.py

# Expose the port your server.py listens on
EXPOSE 7860

# Add /app to PYTHONPATH.
# The 'RealtimeSTT' package is now at /app/RealtimeSTT/RealtimeSTT/
# The import `from RealtimeSTT.RealtimeSTT import AudioToTextRecorder` works
# because /app is on PYTHONPATH, so it finds the /app/RealtimeSTT directory (cloned repo root),
# and then imports the /app/RealtimeSTT/RealtimeSTT package from there.
ENV PYTHONPATH="${PYTHONPATH}:/app"

CMD ["python3", "example_browserclient/server.py"]
