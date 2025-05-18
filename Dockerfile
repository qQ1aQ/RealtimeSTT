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
# Ensure this matches the CUDA version of the base image (12.4 for base, torch 2.3 uses 12.1 compatible index)
# Using the +cu121 versions for PyTorch 2.3.0
RUN pip3 install torch==2.3.0+cu121 torchaudio==2.3.0+cu121 --index-url https://download.pytorch.org/whl/cu121

# Copy requirements first to leverage Docker cache
# Make sure your requirements-gpu.txt includes:
# faster-whisper
# onnxruntime-gpu  (Ensure this is specifically onnxruntime-gpu, not just onnxruntime)
# websockets
# numpy
# scipy
# (and any other direct dependencies RealtimeSTT or your server.py might have)
COPY requirements-gpu.txt /app/requirements-gpu.txt
RUN pip3 install --no-cache-dir -r /app/requirements-gpu.txt

# Clone RealtimeSTT repository
# Using a shallow clone to save space and time if you don't need history
RUN git clone --depth 1 https://github.com/qQ1aQ/RealtimeSTT.git /app/RealtimeSTT

# Copy silero_assets from the cloned repo to /app/silero_assets
# This makes it accessible as "./silero_assets" from the CWD /app when the server runs,
# which is what RealtimeSTT's default configuration expects.
COPY --chown=root:root /app/RealtimeSTT/silero_assets /app/silero_assets

# Copy your application code that uses RealtimeSTT
# Assuming your server.py is in a directory named 'example_browserclient'
# relative to your Dockerfile's context.
COPY example_browserclient/server.py /app/example_browserclient/server.py
# If you have an HTML file or other assets for the client, copy them too:
# COPY example_browserclient/realtime1.html /app/example_browserclient/realtime1.html

# Expose the port the server runs on (server.py is configured for 7860)
EXPOSE 7860

# Set PYTHONPATH so Python can find modules in /app (like RealtimeSTT.RealtimeSTT)
# and also the top-level RealtimeSTT if it were structured differently.
ENV PYTHONPATH="/app:${PYTHONPATH}"

# Command to run your application
# The default WORKDIR is /app
CMD ["python3", "example_browserclient/server.py"]
```

**Key changes and considerations for this Dockerfile:**
1.  **`git clone --depth 1`**: Fetches only the latest commit, making the clone faster and smaller.
2.  **`COPY --chown=root:root /app/RealtimeSTT/silero_assets /app/silero_assets`**: This is the crucial line that makes `silero_assets` available at `/app/silero_assets/`. Since the `WORKDIR` is `/app`, when `AudioToTextRecorder` looks for `"silero_assets"`, it will find `/app/silero_assets/`. The `--chown` flag is good practice to ensure consistent ownership; adjust if your base image user is different and permissions become an issue (though `root` is typical for running services in many Docker setups).
3.  **`requirements-gpu.txt`**: **Crucially important:** ensure `onnxruntime-gpu` is listed in this file, not just `onnxruntime`. If you only have `onnxruntime`, the ONNX VAD model will run on CPU regardless of other settings.
4.  **PyTorch Installation**: Explicitly installing `torch==2.3.0+cu121` and `torchaudio==2.3.0+cu121` from the `cu121` index. Your base image `nvidia/cuda:12.4.1-runtime-ubuntu22.04` should be compatible.
5.  **`apt-get clean` and `rm -rf /var/lib/apt/lists/*`**: Added to reduce image size.
6.  **`ffmpeg`**: Added as a common audio dependency. `RealtimeSTT` or its dependencies might use it.
7.  **No `CPU` stage**: For now, focusing only on the `gpu` stage as that's our primary goal.

Make sure your `requirements-gpu.txt` looks something like this (add any other specific versions or packages you need):
```txt
# requirements-gpu.txt
faster-whisper
onnxruntime-gpu # IMPORTANT for GPU VAD
websockets
numpy
scipy
# Add any other libraries like colorama if your server still uses them,
# or any specific versions you require.
# Example:
# pydub
# soundfile
