FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04 as gpu

WORKDIR /app

# Install necessary system dependencies, including libsndfile1
RUN apt-get update -y && \
   apt-get install -y python3 python3-pip libcudnn8 libcudnn8-dev libcublas-12-4 portaudio19-dev libsndfile1 --no-install-recommends && \
   rm -rf /var/lib/apt/lists/* # Clean up apt cache

RUN pip3 install torch==2.3.0 torchaudio==2.3.0

COPY requirements-gpu.txt /app/requirements-gpu.txt
# Ensure python-soundfile is installed
RUN pip3 install --no-cache-dir -r /app/requirements-gpu.txt python-soundfile

RUN mkdir -p example_browserclient # Use -p to create parent directories if they don't exist
COPY example_browserclient/server.py /app/example_browserclient/server.py
COPY RealtimeSTT /app/RealtimeSTT # Copies the RealtimeSTT package

EXPOSE 9001 # Expose the internal port the server runs on

ENV PYTHONPATH="${PYTHONPATH}:/app"
# Removed redundant RUN export

CMD ["python3", "example_browserclient/server.py"]

# --------------------------------------------

FROM ubuntu:22.04 as cpu

WORKDIR /app

# Install necessary system dependencies, including libsndfile1
RUN apt-get update -y && \
   apt-get install -y python3 python3-pip portaudio19-dev libsndfile1 --no-install-recommends && \
   rm -rf /var/lib/apt/lists/* # Clean up apt cache

RUN pip3 install torch==2.3.0 torchaudio==2.3.0

COPY requirements.txt /app/requirements.txt
# Ensure python-soundfile is installed
RUN pip3 install --no-cache-dir -r /app/requirements.txt python-soundfile

EXPOSE 9001 # Expose the internal port the server runs on

ENV PYTHONPATH="${PYTHONPATH}:/app"
# Removed redundant RUN export

CMD ["python3", "example_browserclient/server.py"]
