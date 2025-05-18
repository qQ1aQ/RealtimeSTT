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
RUN rm -rf /app/silero_assets # This is now part of the cloned repo

# Clone the latest RealtimeSTT repository
# This will place the 'RealtimeSTT' package and 'silero_assets' inside /app/RealtimeSTT/
# e.g. /app/RealtimeSTT/RealtimeSTT/ (package) and /app/RealtimeSTT/silero_assets/
RUN git clone https://github.com/qQ1aQ/RealtimeSTT.git /app/RealtimeSTT

# Copy your custom server.py (assuming it's in 'example_browserclient' in your build context)
# Adjust the source path if your local 'server.py' is elsewhere.
COPY example_browserclient/server.py /app/example_browserclient/server.py

# Expose the port your server.py listens on
EXPOSE 7860

# Add /app to PYTHONPATH.
# The 'RealtimeSTT' package is now at /app/RealtimeSTT/RealtimeSTT/
# For 'from RealtimeSTT import ...' to work from your server.py,
# Python needs to find the directory *containing* the 'RealtimeSTT' package.
# The cloned repository root is /app/RealtimeSTT. Inside this is the RealtimeSTT package directory.
# So, /app (which is on PYTHONPATH) allows Python to see the /app/RealtimeSTT directory, which is the project root.
# The import `from RealtimeSTT import ...` refers to the package directory `/app/RealtimeSTT/RealtimeSTT/`.
ENV PYTHONPATH="${PYTHONPATH}:/app"

CMD ["python3", "example_browserclient/server.py"]
