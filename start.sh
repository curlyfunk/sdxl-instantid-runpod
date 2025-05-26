#!/bin/bash

# Създаване на кеш директория в прикачения volume
mkdir -p /runpod-volume/.cache/huggingface

# Пренасочване към директорията с handler.py
cd /workspace/InstantID

# Стартира handler-а
python3 handler.py
