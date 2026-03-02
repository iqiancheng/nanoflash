#!/usr/bin/env bash
set -e
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

CUDA_VISIBLE_DEVICES=0 python train.py \
    --config nanoflash/config/qwen3_0.6b.yaml \
    train.batch_size=2 \
    train.max_steps=500 \
    2>&1 | tee "output/qwen3_0.6b/logs/$(date +%Y%m%d_%H%M%S).log"