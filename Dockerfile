FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04 AS gpu

WORKDIR /app

# Install necessary system dependencies, including python3-dev for PyAudio
RUN apt-get update -y && \
   apt-get install -y python3 python3-dev python3-pip libcudnn8 libcudnn8-dev libcublas-12-4 portaudio19-dev libsndfile1 --no-install-recommends && \
   rm -rf /var/lib/apt/lists/*

RUN pip3 install torch==2.3.0 torchaudio==2.3.0

COPY requirements-gpu.txt /app/requirements-gpu.txt
# Ensure soundfile is installed (PyAudio is a dependency from requirements.txt/RealtimeSTT)
RUN pip3 install --no-cache-dir -r /app/requirements-gpu.txt soundfile

# Use -p to create parent directories if they don't exist
RUN mkdir -p example_browserclient
COPY example_browserclient/server.py /app/example_browserclient/server.py
# Copies the RealtimeSTT package
COPY RealtimeSTT /app/RealtimeSTT

# Expose the internal port the server runs on
EXPOSE 9001

ENV PYTHONPATH="${PYTHONPATH}:/app"

CMD ["python3", "example_browserclient/server.py"]

# --------------------------------------------

FROM ubuntu:22.04 AS cpu

WORKDIR /app

# Install necessary system dependencies, including python3-dev for PyAudio
RUN apt-get update -y && \
   apt-get install -y python3 python3-dev python3-pip portaudio19-dev libsndfile1 --no-install-recommends && \
   rm -rf /var/lib/apt/lists/*

RUN pip3 install torch==2.3.0 torchaudio==2.3.0

COPY requirements.txt /app/requirements.txt
# Ensure soundfile is installed (PyAudio is a dependency from requirements.txt/RealtimeSTT)
RUN pip3 install --no-cache-dir -r /app/requirements.txt soundfile

# Expose the internal port the server runs on
EXPOSE 9001

ENV PYTHONPATH="${PYTHONPATH}:/app"

CMD ["python3", "example_browserclient/server.py"]
