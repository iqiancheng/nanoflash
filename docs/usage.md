# Nanoflash Training Framework — Usage Guide

## 1. Running training

**Entrypoint**

```bash
python -m train --config <path/to/config.yaml>
```

From repo root, either:

```bash
python -m train --config nanoflash/config/qwen3_0.6b.yaml
# or
python train.py --config nanoflash/config/qwen3_0.6b.yaml
```

**Override config from CLI**

Any key can be overridden with `key=value` (use dot notation for nested keys). CLI overrides take precedence over the YAML file.

```bash
python -m train --config nanoflash/config/qwen3_0.6b.yaml \
  train.batch_size=4 \
  train.max_steps=2000 \
  train.output_dir=./runs/exp1
```

Use `key=null` or `key=None` to unset a value (e.g. disable a section).

---

## 2. Config overview

Training is driven by a single YAML config. Main top-level keys:

| Key | Purpose |
|-----|--------|
| `train` | Batch size, max steps, device, dtype, seed, `output_dir`, gradient accumulation, workers. |
| `model` | `class_path` + kwargs (e.g. `model_name_or_path`, `torch_dtype`). |
| `data` | Dataset: `class_path` + kwargs; may include nested `tokenizer` config. |
| `data_val` | Optional validation dataset (same shape as `data`). |
| `optimizer` | `class_path` + kwargs (e.g. `torch.optim.AdamW`, `lr`, `weight_decay`). |
| `lr_scheduler` | Optional. `class_path` + kwargs. |
| `loss` | Optional. `class_path` + kwargs (default: `CrossEntropyLoss` with `ignore_index=-100`). |
| `collate_fn` | Optional. `class_path` for the collate callable. |
| `checkpoint` | Optional. Save/load; `checkpoint_dir`, `save_every_n_steps`, etc. |
| `logging` | Optional. Logger `class_path` + kwargs. |
| `project` | Optional. Path to a directory to prepend to `sys.path` so `class_path` can import from your code. |

Every component (model, data, optimizer, lr_scheduler, loss, etc.) uses the same pattern: **`class_path`** (dotted path to class or factory) plus other keys as constructor/factory keyword arguments. The loader rewrites `class_path` to `_target_` and builds the object with Hydra’s `instantiate`.

You can use **interpolation** in YAML, e.g. `${train.output_dir}/ckpt` or `${model.model_name_or_path}`.

---

## 3. Configuring the dataset

### 3.1 Data section in YAML

Example (Alpaca-style from HuggingFace):

```yaml
data:
  class_path: nanoflash.components.dataset.load_alpaca_dataset
  source: yahma/alpaca-cleaned
  max_length: 2048
  split: train
  train_on_input: true
  tokenizer:
    class_path: nanoflash.components.tokenizer.load_tokenizer
    model_name_or_path: ${model.model_name_or_path}
    max_length: 2048
```

- **`class_path`**: Dotted path to the dataset builder (function or class).
- **`source`**: Dataset name or path (HuggingFace hub name or local path).
- **`max_length`**, **`split`**, **`train_on_input`**: Passed as kwargs to the builder.
- **`tokenizer`**: Nested config; the pipeline builds it and passes it to the dataset builder (when the builder accepts a `tokenizer` argument).

For a **local JSON/Parquet** dataset, point `source` to the file or directory and ensure your dataset `class_path` supports that (e.g. `data_path: ./data/train.json`).

### 3.2 What one dataset sample must look like

After loading and tokenization, each item must be a dict with exactly these keys (used by the default collate and train loop):

| Key | Type | Description |
|-----|------|-------------|
| `input_ids` | 1D tensor (long) | Token ids, shape `(seq_len,)`. |
| `attention_mask` | 1D tensor (long) | 1 for real tokens, 0 for padding. |
| `labels` | 1D tensor (long) | Same length as `input_ids`. Use `-100` for positions you do not want to contribute to the loss (e.g. prompt). |

The train loop does causal LM: it uses `input_ids` / `attention_mask` as model input and computes loss on `labels` with a shift (predict next token). So `labels` are usually the same as `input_ids` except masked positions set to `-100`.

**Alpaca-style raw format** (for `load_alpaca_dataset`):

Each raw example should have:

- `instruction`: string
- `input`: string (optional; can be empty)
- `output`: string

The built-in loader formats them as `Instruction: ...\nInput: ...\nResponse: ` + `output`, then tokenizes and sets `labels` to `-100` on the prompt part if `train_on_input` is false.

**Custom dataset**

Implement a function or class that returns a HuggingFace `Dataset` (or compatible) whose items are dicts with `input_ids`, `attention_mask`, and `labels`. Set `data.class_path` to that callable and pass any extra args (e.g. `data_path`, `max_length`) in the same YAML section. If your loader needs a tokenizer, add a `tokenizer` nested config under `data`; the recipe will build it and pass it in.

---

## 4. Changing a component (e.g. LR scheduler)

All components are built from config via `class_path` + kwargs. To swap one (e.g. use a different LR scheduler), change the YAML for that section; no framework code change is required.

### 4.1 Use another built-in or library scheduler

Example: cosine with warmup (current example):

```yaml
lr_scheduler:
  class_path: transformers.get_cosine_schedule_with_warmup
  num_warmup_steps: 100
```

The recipe calls the scheduler factory with:

- First positional argument: the **optimizer**
- Keyword argument: **`num_training_steps`** = `train.max_steps`

So the target must accept the optimizer and `num_training_steps`. For `get_cosine_schedule_with_warmup` you only need to set `num_warmup_steps` (and optionally other kwargs) in the config.

Linear schedule example:

```yaml
lr_scheduler:
  class_path: transformers.get_linear_schedule_with_warmup
  num_warmup_steps: 100
```

### 4.2 Using a DeepSpeed LR scheduler

Config:

```yaml
lr_scheduler:
  class_path: deepspeed.ops.lr_schedules.WarmupLR
  warmup_max_lr: 2.0e-5
  warmup_num_steps: 100
  total_num_steps: 1000
```

---

## 5. Output directory layout

`train.output_dir` is the root for all outputs. Typical structure:

```
<output_dir>/
├── ckpt/                          # if checkpoint is configured
│   ├── step_500/
│   │   ├── state.pt               # full resume: model_state, opt_state, step
│   │   ├── config.json            # HF model config
│   │   ├── model.safetensors      # HF format (or pytorch_model.bin)
│   │   └── tokenizer files        # if tokenizer was set on checkpointer
│   ├── step_1000/
│   └── ...
└── tensorboard/                   # if logging is configured (default)
    └── events.out.tfevents.*
```

- **Resume**: The checkpointer loads the latest `step_*` under `checkpoint_dir`, then `state.pt` for model state, optimizer state, and step count.
- **HF export**: Each `step_*/` also contains HuggingFace model (and optionally tokenizer) so you can load with `from_pretrained(step_dir)`.

---

## 6. Common tasks

### 6.1 Custom model or code in another repo

Set `project` to the root directory of that repo (absolute or relative). The runner prepends it to `sys.path`, so you can use `class_path: your_package.models.YourModel` and similar in the config.

### 6.2 Custom collate

Implement a callable `collate_fn(batch: List[dict]) -> dict` that returns a batch dict of tensors (e.g. `input_ids`, `attention_mask`, `labels`). Register it in config:

```yaml
collate_fn:
  class_path: your_package.data.your_collate_fn
```

The recipe resolves this to a callable and passes it to the DataLoader.

### 6.3 Disable checkpointing or logging

Omit the section or set it to null so the recipe does not build it:

```bash
python -m train --config config.yaml checkpoint=null
```

Or in YAML, remove the `checkpoint:` block to disable saving.

### 6.4 Validation

Add a `data_val` section with the same shape as `data` (same `class_path` pattern and tokenizer if needed). The recipe builds a validation DataLoader; whether it is used in the loop depends on the recipe (the default SFT recipe currently only builds it; you can extend the recipe to run validation and log).

### 6.5 Gradient accumulation

Set in `train`:

```yaml
train:
  batch_size: 4
  gradient_accumulation_steps: 4   # effective batch = 16
```

### 6.6 Dtype and device

In `train`:

```yaml
train:
  device: cuda
  dtype: bf16   # or fp16, fp32
```

Model dtype is set via `model.torch_dtype` (e.g. `bf16`).

---

## 7. Summary

| Need | Action |
|------|--------|
| Run training | `python -m train --config <config.yaml>` |
| Override config | Add `key=value` (or `key.nested=value`) after `--config`. |
| New dataset | Implement a builder that returns a dataset with `input_ids`, `attention_mask`, `labels` per item; set `data.class_path` and kwargs. |
| New LR scheduler | Set `lr_scheduler.class_path` to a callable that accepts `(optimizer, num_training_steps=...)` (and optional kwargs from config). |
| Custom code path | Set `project` to the root directory containing your packages. |
| Outputs | Check `train.output_dir`: `ckpt/step_*/` for checkpoints and HF export, `tensorboard/` for logs. |

For design and extension points (e.g. new recipe, new component), see `design.md`.
