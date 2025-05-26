#!/bin/bash
mkdir -p /runpod-volume/.cache/huggingface
cd /workspace/InstantID
python3 handler.py
