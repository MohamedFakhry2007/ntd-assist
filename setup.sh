#!/bin/bash
pip install --upgrade -r requirements.txt
# Pre-cache the CPU-fallback model to avoid download timeout at runtime
python -c "
from huggingface_hub import snapshot_download
snapshot_download('Qwen/Qwen2-VL-2B-Instruct')
"