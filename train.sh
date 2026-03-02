#!/usr/bin/env bash
set -e
set -o pipefail

MODEL_NAME="qwen3_0.6b"
LOG_DIR="output/${MODEL_NAME}/logs"
mkdir -p $LOG_DIR

CUDA_VISIBLE_DEVICES=0 python train.py \
    --config nanoflash/config/${MODEL_NAME}.yaml \
    train.batch_size=2 \
    train.max_steps=500 \
    2>&1 | tee "${LOG_DIR}/$(date +%Y%m%d_%H%M%S).log"