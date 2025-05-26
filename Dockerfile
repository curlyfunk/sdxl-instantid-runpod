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
      wget \
      curl \
      ffmpeg \
      libgl1 \
      libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Python зависимости
RUN pip install --upgrade pip && \
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124 && \
    pip install diffusers transformers accelerate opencv-python runpod

# Клониране на InstantID (без LFS)
RUN git clone https://github.com/InstantX/InstantID.git || true

# Пренасочване на кеш към volume
ENV HF_HOME=/runpod-volume/.cache/huggingface
ENV TRANSFORMERS_CACHE=$HF_HOME
ENV DIFFUSERS_CACHE=$HF_HOME

# Копиране на скриптове
COPY --chmod=755 handler.py /workspace/InstantID/handler.py
COPY --chmod=755 start.sh /start.sh

ENTRYPOINT ["/start.sh"]
