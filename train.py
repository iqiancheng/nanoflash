#!/usr/bin/env python3
"""
Nanoflash training entrypoint.

Usage:
    python -m nanoflash.train --config nanoflash/config/qwen3_1.7b.yaml
    python -m nanoflash.train --config config.yaml train.batch_size=16
"""
import sys

from nanoflash.pipeline.run import run

if __name__ == "__main__":
    run()
    sys.exit(0)
