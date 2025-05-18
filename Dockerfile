FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04 AS gpu

# Declare ARG CACHE_BUSTER_ARG *after* FROM for this build stage
# This makes it available to RUN, COPY, etc. within this stage and helps in cache invalidation
ARG CACHE_BUSTER_ARG

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
# Using ARG to ensure this step is re-run if CACHE_BUSTER_ARG changes.
RUN echo "Executing clone step. Cache Buster Value: ${CACHE_BUSTER_ARG:-default_value_if_not_set}" && \
    echo "Attempting to remove pre-existing /app/RealtimeSTT (if any)" && \
    rm -rf /app/RealtimeSTT && \
    echo "Cloning RealtimeSTT repository..." && \
    git clone --depth 1 https://github.com/qQ1aQ/RealtimeSTT.git /app/RealtimeSTT && \
    echo "Listing contents of /app/RealtimeSTT after clone:" && \
    ls -la /app/RealtimeSTT && \
    echo "Checking for /app/RealtimeSTT/silero_assets:" && \
    (ls -ld /app/RealtimeSTT/silero_assets && echo "SUCCESS: silero_assets directory found.") || (echo "FAILURE: silero_assets directory NOT found in /app/RealtimeSTT immediately after clone." && exit 1)

# Copy silero_assets from the cloned repo to /app/silero_assets
COPY --chown=root:root /app/RealtimeSTT/silero_assets /app/silero_assets

# Copy your application code
COPY example_browserclient/server.py /app/example_browserclient/server.py

# Expose the port
EXPOSE 7860

# Set PYTHONPATH
ENV PYTHONPATH="/app:${PYTHONPATH}" # The warning about $PYTHONPATH here is minor and not related to the COPY failure

# Command to run your application
CMD ["python3", "example_browserclient/server.py"]
```

**Key Changes in this Dockerfile:**

1.  **Correct `ARG` Placement**: `ARG CACHE_BUSTER_ARG` is now *after* `FROM nvidia/cuda...`. This is crucial.
2.  **Safer `ARG` Expansion**: In the `echo` command, I've used `${CACHE_BUSTER_ARG:-default_value_if_not_set}`. This bash-like expansion provides a default if the ARG isn't set, making the echo more robust, though the main goal is ensuring the command string changes if `CACHE_BUSTER_ARG` is passed and changes.
3.  Emphasized success/failure messages in the clone step.

**Crucial GitHub Actions Workflow Step:**
Please ensure your `docker/build-push-action@v5` step in your GitHub Actions workflow file correctly passes the `build-args`. This is essential for the `ARG` to receive a new value and bust the cache.

```yaml
      - name: Build and push Docker image
        uses: docker/build-push-action@v5
        with:
          context: .
          file: ./Dockerfile
          push: true # Or false, depending on your needs
          tags: yourusername/realtime-stt:latest # Replace with your tags
          target: gpu # Ensure your target is 'gpu'
          # THIS IS THE CRITICAL PART FOR CACHE BUSTING THE ARG
          build-args: |
            CACHE_BUSTER_ARG=${{ github.sha }}
          # To be absolutely sure about caching for now (will slow down builds):
          # no-cache: true # <-- You can try this if ARG still fails.
