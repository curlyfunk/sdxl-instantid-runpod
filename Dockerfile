FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=on \
    SHELL=/bin/bash

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# Системни зависимости
RUN apt update && \
    apt upgrade -y && \
    apt install -y \
      python3-pip \
      git \
      git-lfs \
      wget \
      curl \
      ffmpeg \
      libgl1 \
      libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

# Git LFS + клониране на InstantID
RUN git lfs install && \
    git clone https://github.com/InstantX/InstantID.git && \
    cd InstantID && git lfs pull

WORKDIR /workspace

# Инсталиране на Python зависимости
RUN pip install --upgrade pip && \
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124 && \
    pip install diffusers transformers accelerate opencv-python runpod

# Изтегляне на SDXL-Turbo модел (placeholder, ако се ползва локално)
RUN mkdir -p /workspace/models && \
    curl -L -o /workspace/models/sdxl-turbo.safetensors https://huggingface.co/stabilityai/sdxl-turbo/resolve/main/sdxl-turbo.safetensors || true

# Копиране на скриптове
COPY --chmod=755 handler.py /workspace/InstantID/handler.py
COPY --chmod=755 start.sh /start.sh

# Стартиране
ENTRYPOINT ["/start.sh"]
