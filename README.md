# Nanoflash

Minimal training framework driven by a single YAML config. Uses OmegaConf, Hydra instantiate, and a simple train pipeline. Supports Qwen/Qwen3-1.7B SFT on Nvidia GPU.

## Quick Start

```bash
# From project root (ensure PYTHONPATH includes project root)
python -m train --config nanoflash/config/qwen3_0.6b.yaml

# Override config via CLI
python -m train --config nanoflash/config/qwen3_0.6b.yaml train.batch_size=8 train.max_steps=500
```

## Requirements

- PyTorch (with CUDA for GPU)
- transformers
- datasets
- omegaconf
- hydra-core

```bash
pip install torch transformers datasets omegaconf hydra-core
```

## Config

All components are configured via YAML with `class_path` pointing to the class or factory:

- `model`: Causal LM (e.g. Qwen3)
- `data`: Dataset (e.g. Alpaca-style)
- `optimizer`, `lr_scheduler`, `loss`, `collate_fn`
- `checkpoint`, `logging`

Use `${key.subkey}` for interpolation (e.g. `${model.model_name_or_path}`).

## Project Path

To load components from an external directory, set `project` in config:

```yaml
project: /path/to/your/code
```

This prepends the path to `sys.path` so `class_path` can import from that tree.

## Adding Components

- **New loss/dataset**: Implement the class or factory, then set `class_path` in YAML. No code change in nanoflash.
- **New recipe**: Implement a class with `__init__`, `setup`, `train`, `cleanup`; register in `nanoflash.pipeline.recipe.RECIPES`; set `recipe: your_name` in YAML.

See [design.md](docs/design.md) for full design. [Usage](docs/usage.md) for usage guide.
