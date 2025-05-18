FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04 AS gpu

# Declare ARG CACHE_BUSTER_ARG *after* FROM for this build stage
# Provide a default for local builds if not passed, or if GHA somehow doesn't pass it.
ARG CACHE_BUSTER_ARG="default_build_$(date +%s)"

# Using the ARG in a LABEL might help influence caching behavior in some builder versions.
LABEL cache_buster_label="${CACHE_BUSTER_ARG}"

WORKDIR /app

# Install necessary system dependencies
RUN apt-get update -y && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
    python3 \
    python3-dev \
    python3-pip \
    git \
    portaudio19-dev \
    libasound2-dev \
    libsndfile1 \
    build-essential \
    ffmpeg \
    --no-install-recommends && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Pip install PyTorch and Torchaudio
RUN pip3 install torch==2.3.0+cu121 torchaudio==2.3.0+cu121 --index-url https://download.pytorch.org/whl/cu121

# Copy requirements
COPY requirements-gpu.txt /app/requirements-gpu.txt
RUN pip3 install --no-cache-dir -r /app/requirements-gpu.txt

# Clone RealtimeSTT repository
# This RUN command now explicitly creates and removes a file using CACHE_BUSTER_ARG
# in its name, and echoes the ARG's value. The goal is to make this layer's
# command string and operations undeniably unique if CACHE_BUSTER_ARG changes.
RUN echo "--- BEGINNING CLONE STEP ---" && \
    echo "CURRENT CACHE_BUSTER_ARG value reported: '${CACHE_BUSTER_ARG}'" && \
    echo "Creating temporary cache busting file: /tmp/bust_file_${CACHE_BUSTER_ARG}.txt" && \
    touch /tmp/bust_file_${CACHE_BUSTER_ARG}.txt && \
    ls -l /tmp/bust_file_${CACHE_BUSTER_ARG}.txt && \
    echo "Removing temporary cache busting file: /tmp/bust_file_${CACHE_BUSTER_ARG}.txt" && \
    rm /tmp/bust_file_${CACHE_BUSTER_ARG}.txt && \
    echo "Temporary cache busting file removed." && \
    echo "Now proceeding with git clone operations..." && \
    echo "Removing pre-existing /app/RealtimeSTT (if any)..." && \
    rm -rf /app/RealtimeSTT && \
    echo "Cloning RealtimeSTT repository..." && \
    git clone --depth 1 https://github.com/qQ1aQ/RealtimeSTT.git /app/RealtimeSTT && \
    echo "Listing contents of /app/RealtimeSTT after clone:" && \
    ls -la /app/RealtimeSTT && \
    echo "Checking for /app/RealtimeSTT/silero_assets:" && \
    (ls -ld /app/RealtimeSTT/silero_assets && echo "SUCCESS: silero_assets directory found in /app/RealtimeSTT.") || (echo "FAILURE: silero_assets directory NOT found in /app/RealtimeSTT after clone." && exit 1) && \
    echo "--- CLONE STEP COMPLETED SUCCESSFULLY ---"

# Copy silero_assets from the cloned repo to /app/silero_assets
COPY --chown=root:root /app/RealtimeSTT/silero_assets /app/silero_assets

# Copy your application code
COPY example_browserclient/server.py /app/example_browserclient/server.py

# Expose the port
EXPOSE 7860

# Set PYTHONPATH
ENV PYTHONPATH="/app:${PYTHONPATH}"

# Command to run your application
CMD ["python3", "example_browserclient/server.py"]
