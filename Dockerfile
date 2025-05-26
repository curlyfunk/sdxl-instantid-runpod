FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=on \
    SHELL=/bin/bash

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

RUN apt update && \
    apt upgrade -y && \
    apt install -y \
      python3-pip \
      git \
      wget \
      curl \
      ffmpeg \
      libgl1 \
      libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Install dependencies
RUN pip install --upgrade pip && \
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124 && \
    pip install diffusers transformers accelerate opencv-python runpod

# Clone InstantID repo
RUN git clone https://github.com/InstantX/InstantID.git && \
    cd InstantID && \
    git lfs install && \
    git lfs pull

# Download SDXL-Turbo
RUN mkdir -p /workspace/models && \
    curl -L -o /workspace/models/sdxl-turbo.safetensors https://huggingface.co/stabilityai/sdxl-turbo/resolve/main/sdxl-turbo.safetensors

COPY --chmod=755 handler.py /workspace/InstantID/handler.py
COPY --chmod=755 start.sh /start.sh

ENTRYPOINT ["/start.sh"]
