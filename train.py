#!/usr/bin/env python3
"""
Nanoflash training entrypoint.

Usage:
    python -m train --config nanoflash/config/qwen3_0.6b.yaml
    python -m train --config config.yaml train.batch_size=16
"""
import sys

from nanoflash.pipeline.run import run

if __name__ == "__main__":
    run()
    sys.exit(0)
