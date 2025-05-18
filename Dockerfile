FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04 AS gpu

WORKDIR /app

# Install necessary system dependencies for Python, build tools, PyTorch, audio
RUN apt-get update -y && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
    python3 \
    python3-dev \
    python3-pip \
    git \
    # portaudio for microphone input (though use_microphone=False, RealtimeSTT might link against it)
    portaudio19-dev \
    libasound2-dev \
    libsndfile1 \
    # build-essential for some pip packages that might compile C extensions
    build-essential \
    # ffmpeg is a common dependency for audio processing, good to have.
    ffmpeg \
    --no-install-recommends && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Pip install PyTorch and Torchaudio with specific CUDA support first
RUN pip3 install torch==2.3.0+cu121 torchaudio==2.3.0+cu121 --index-url https://download.pytorch.org/whl/cu121

# Copy requirements first to leverage Docker cache
COPY requirements-gpu.txt /app/requirements-gpu.txt
RUN pip3 install --no-cache-dir -r /app/requirements-gpu.txt

# Clone RealtimeSTT repository
# Adding rm -rf and ls commands to ensure a clean clone and to debug contents.
# This should definitely change the layer and bust the cache.
RUN echo "Preparing to clone RealtimeSTT..." && \
    rm -rf /app/RealtimeSTT && \
    echo "Cloning RealtimeSTT repository..." && \
    git clone --depth 1 https://github.com/qQ1aQ/RealtimeSTT.git /app/RealtimeSTT && \
    echo "Listing contents of /app/RealtimeSTT after clone:" && \
    ls -la /app/RealtimeSTT && \
    echo "Checking for /app/RealtimeSTT/silero_assets:" && \
    (ls -la /app/RealtimeSTT/silero_assets && echo "silero_assets directory found.") || echo "silero_assets directory NOT found in /app/RealtimeSTT immediately after clone."

# Copy silero_assets from the cloned repo to /app/silero_assets
# This makes it accessible as "./silero_assets" from the CWD /app when the server runs,
# which is what RealtimeSTT's default configuration expects.
COPY --chown=root:root /app/RealtimeSTT/silero_assets /app/silero_assets

# Copy your application code that uses RealtimeSTT
COPY example_browserclient/server.py /app/example_browserclient/server.py

# Expose the port the server runs on (server.py is configured for 7860)
EXPOSE 7860

# Set PYTHONPATH so Python can find modules in /app
ENV PYTHONPATH="/app:${PYTHONPATH}"

# Command to run your application
CMD ["python3", "example_browserclient/server.py"]
