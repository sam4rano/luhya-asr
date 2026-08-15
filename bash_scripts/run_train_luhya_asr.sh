#!/usr/bin/env bash

nvidia-smi
echo $CUDA_VISIBLE_DEVICES

export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
NUMBA_CACHE_DIR='/tmp/numba_cache'
LIBROSA_CACHE_DIR="/tmp/librosa_cache"

echo "Current working directory: $(pwd)"

python3 scripts/train_model.py --config config_files/ASR_train_config_luhya.yaml
