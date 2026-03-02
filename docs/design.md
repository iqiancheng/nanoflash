# Nanoflash Training Framework — Design Document (`design.md`)

## 1. Overview

**Nanoflash** is a minimal training framework that:

- Drives the whole run from a **single YAML config** plus optional CLI overrides.
- Uses **OmegaConf** for config loading/merge/interpolation and **Hydra’s `instantiate`** to build model, dataset, optimizer, etc. from config (no hardcoded classes in the train script).
- Exposes a small **train pipeline**: parse config → resolve component paths → (optional) extend `sys.path` → build recipe → setup → train → cleanup.
- Keeps **file and config naming** in normal English (idiomatic for native speakers).

This doc is the single source of truth for implementation; agents should implement exactly to this design.

---

## 2. Core Stack

| Component        | Role |
|-----------------|------|
| **OmegaConf**   | Load YAML, merge CLI key=value, resolve `${...}` interpolations, recursive config traversal. |
| **Hydra `instantiate`** | Build objects from config: `_target_` = dotted path to class or factory; rest of the node = kwargs. |
| **YAML config** | One main config file per run; may reference another YAML for data index. |
| **Train pipeline** | One entry script that: loads config → prepares env → creates “recipe” → runs setup / train / cleanup. |

No Hydra’s composable configs or multirun; no TorchTune. Just OmegaConf + Hydra instantiate + one entrypoint.

---

## 3. Directory Layout (Normal Naming)

Use short, common English words; no abbreviations unless very standard (e.g. `cfg`, `fn`).

```
nanoflash/
├── config/                    # Example or default YAML configs
│   └── train_example.yaml
├── core/                      # Config and instantiation
│   ├── config.py              # Load YAML + merge CLI, resolve interpolations
│   └── instantiate.py         # class_path → _target_, then Hydra instantiate
├── pipeline/                  # Train flow
│   ├── recipe.py              # Base recipe interface + one default recipe
│   └── run.py                 # Entry: parse → create recipe → setup/train/cleanup
├── train.py                   # CLI entrypoint (e.g. python train.py --config path/to/config.yaml)
└── README.md
```

- **User code** (model, dataset, loss, etc.) can live **outside** this tree; the config and `project` (see below) point to it.

---

## 4. Config Schema (YAML Shape)

### 4.1 Top-Level Keys

Use normal English; same style across files and YAML.

| Key | Purpose |
|-----|--------|
| `project` | Optional. Absolute or relative path to a directory to prepend to `sys.path` so `class_path` can import from that tree. |
| `recipe` | Optional. Name of the recipe to run (default: `default`). Used to select the recipe class. |
| `train` | Training hyperparameters (batch size, steps, device, dtype, seed, log dir, etc.). |
| `model` | One node with `class_path` + constructor kwargs (e.g. `model_name_or_path`, `dtype`). |
| `data` | Train dataset: `class_path` + kwargs (e.g. `data_path`, `max_length`). |
| `data_val` | Optional. Validation dataset, same shape as `data`. |
| `optimizer` | `class_path` (e.g. `torch.optim.AdamW`) + kwargs (e.g. `lr`, `weight_decay`). |
| `lr_scheduler` | Optional. `class_path` + kwargs. |
| `loss` | Optional. `class_path` + kwargs. |
| `collate_fn` | Optional. `class_path` + kwargs for the collate callable. |
| `checkpoint` | Optional. Save/load: `class_path` + kwargs (e.g. `checkpoint_dir`, `save_every_n_steps`). Saves to `checkpoint_dir/step_{step}/` both state.pt (resume) and HF format (model, tokenizer). |
| `logging` | Optional. Logger: `class_path` + kwargs (e.g. `log_dir`, `log_every_n_steps` default 1). |

All “component” nodes (model, data, optimizer, loss, etc.) use the same pattern: **`class_path`** (dotted path to class or factory) + other keys as kwargs.

### 4.2 Naming Conventions in YAML

- Use **snake_case** for keys: `batch_size`, `max_steps`, `model_name_or_path`.
- Use **dotted paths** for `class_path`: `torch.optim.AdamW`, `mypackage.models.MyModel`, `mypackage.data.get_train_dataset`.
- No `_target_` in the YAML; the loader converts `class_path` → `_target_` before calling Hydra (see below).

### 4.3 Example YAML (Minimal)

```yaml
train:
  batch_size: 8
  max_steps: 10000
  device: cuda
  dtype: bf16
  seed: 42
  output_dir: ./output

model:
  class_path: mypackage.models.ExampleModel
  model_name_or_path: /path/to/ckpt
  dtype: bf16

data:
  class_path: mypackage.data.get_train_dataset
  data_path: ./data/train.json
  max_length: 2048

optimizer:
  class_path: torch.optim.AdamW
  lr: 1.0e-5
  weight_decay: 0.01

loss:
  class_path: torch.nn.CrossEntropyLoss

checkpoint:
  class_path: nanoflash.core.checkpoint.Checkpointer
  checkpoint_dir: ${train.output_dir}/ckpt
  save_every_n_steps: 1000
```

---

## 5. Pipeline Flow (Step-by-Step)

1. **Parse**
   - Parse CLI: required `--config` (path to YAML), optional overrides as `key=value` (e.g. `train.batch_size=16`).
   - Load YAML with OmegaConf; merge overrides (CLI overrides YAML).
   - Result: one `DictConfig` `cfg`.

2. **Prepare environment**
   - If `cfg.project` is set and the path exists: `sys.path.insert(0, cfg.project)`.
   - So any `class_path` under that project can be imported later.

3. **Resolve class_path**
   - Walk the whole `cfg` recursively.
   - For every key `class_path`, rename it to `_target_` (value unchanged). This makes the config compatible with Hydra’s `instantiate`.

4. **Create recipe**
   - From `cfg.recipe` (default `default`) resolve the recipe class (e.g. from a small registry: `{"default": SFTRecipe}`).
   - Instantiate: `recipe = RecipeClass(cfg)`.

5. **Setup**
   - Call `recipe.setup(cfg)`.
   - Inside `setup`, the recipe uses a single **instantiate helper** (see below) to build:
     - model from `cfg.model`
     - train (and optionally val) dataset from `cfg.data` / `cfg.data_val`
     - collate from `cfg.collate_fn` if present
     - optimizer from `cfg.optimizer` (with model parameters)
     - loss from `cfg.loss` if present
     - lr_scheduler from `cfg.lr_scheduler` if present
     - checkpoint from `cfg.checkpoint` if present
     - logger from `cfg.logging` if present
   - No hardcoded class names; everything comes from config via the instantiate helper.

6. **Train**
   - Call `recipe.train()`. The recipe uses the objects built in setup (model, dataloader, optimizer, loss, scheduler, checkpoint, logging).

7. **Cleanup**
   - Call `recipe.cleanup()` (e.g. destroy process group, flush logs).

---

## 6. Instantiate Helper (Core)

- **Location**: `core/instantiate.py`.
- **API**: one function, e.g. `build(cfg_node, *args, **kwargs) -> Any`.
- **Behavior**:
  - If `cfg_node` is `None`, return `None`.
  - If the node is a dict-like (OmegaConf/DictConfig) and has key `_target_` (or after resolution, `class_path` → `_target_`):
    - Copy the config, resolve interpolations, then call `hydra.utils.instantiate(config_copy, *args, **kwargs)`.
  - For Hydra, the node must be (or be converted to) a structure that Hydra’s `instantiate` understands: `_target_` = dotted path string; other keys = kwargs. So after `class_path` → `_target_`, the rest of the design is standard Hydra.
- **Recursion**: Hydra’s `instantiate` already handles nested configs; ensure that any nested component also uses `_target_` (after the global `class_path` → `_target_` pass), so that nested objects (e.g. dataset’s tokenizer) are built by Hydra recursively.
- No need to support `torch_compile` or other extras in the first version; keep the helper minimal.

---

## 7. Config Loading (Core)

- **Location**: `core/config.py`.
- **Responsibilities**:
  - Parse CLI: `--config` required; remaining args in the form `key=value` (support dot keys, e.g. `train.batch_size=16`).
  - Load base config: `OmegaConf.load(cfg_path)`.
  - Merge overrides: e.g. build an OmegaConf from the key=value list and merge with the base config (override wins).
  - Return a single `DictConfig` with all interpolations resolved (`OmegaConf.resolve(cfg)`).
- Do not use Hydra’s application-based API; just OmegaConf load + merge + resolve.

---

## 8. Recipe Interface and Default Recipe

- **Location**: `pipeline/recipe.py`.
- **Interface** (normal English method names):
  - `__init__(self, cfg: DictConfig)`
  - `setup(self, cfg: DictConfig) -> None`
  - `train(self) -> None`
  - `cleanup(self) -> None`

- **Default recipe** (e.g. class name `SFTRecipe`):
  - In `__init__`: store `cfg`, set device/dtype from `cfg.train`, init distributed if needed.
  - In `setup`: using the **instantiate helper** only (no hardcoded classes), build in a fixed order:
    1. Checkpointer (if `cfg.checkpoint`), then load checkpoint if resuming.
    2. Model from `cfg.model`; if checkpoint was loaded, load model state.
    3. Loss from `cfg.loss` (optional).
    4. Optimizer from `cfg.optimizer` with model parameters.
    5. LR scheduler from `cfg.lr_scheduler` (optional), with optimizer and step count from `cfg.train`.
    6. Train dataset from `cfg.data`, then DataLoader (batch size, workers from `cfg.train`; collate from `cfg.collate_fn` if present).
    7. Validation dataset/dataloader from `cfg.data_val` if present.
    8. Logger from `cfg.logging` if present.
  - In `train`: loop over dataloader, forward, backward, optimizer step, scheduler step; log and checkpoint according to `cfg.train` and `cfg.checkpoint`.
  - In `cleanup`: e.g. destroy process group, close logger.

- **Registry**: In the same file (or a tiny `pipeline/registry.py`), a dict mapping recipe name → class, e.g. `RECIPES = {"default": SFTRecipe}`. The entrypoint uses `cfg.recipe` to pick the class, then instantiates it.

---

## 9. Entrypoint

- **Location**: `train.py` at repo root.
- **Behavior**:
  - Parse config (using `core.config`).
  - Apply `project` to `sys.path` if set.
  - Run **resolve class_path** over the full config (function in `core/instantiate.py`).
  - Get recipe class from registry by `cfg.recipe`; instantiate `recipe = RecipeClass(cfg)`.
  - Call `recipe.setup(cfg)`, then `recipe.train()`, then `recipe.cleanup()`.
- No decorators or magic; plain function calls and one clear sequence.

---

## 10. File and Field Naming Summary

- **Files / dirs**: `config`, `core`, `pipeline`, `train.py`; files like `config.py`, `instantiate.py`, `recipe.py`, `run.py`. Names read like normal English.
- **Config keys**: `project`, `recipe`, `train`, `model`, `data`, `data_val`, `optimizer`, `lr_scheduler`, `loss`, `collate_fn`, `checkpoint`, `logging`. All snake_case.
- **Train subkeys**: `batch_size`, `max_steps`, `device`, `dtype`, `seed`, `output_dir`, etc. — standard, unabbreviated.
- **Component key**: always `class_path` in YAML (converted to `_target_` only internally before Hydra).

---

## 11. How to Extend (For Implementers)

- **New component (e.g. new loss or dataloader)**  
  - Implement the class or factory in user code (inside or outside the repo).  
  - In YAML, add or override the corresponding section with `class_path: package.module.ClassName` and kwargs.  
  - If the class lives outside the repo, set `project` to that repo’s root.  
  - No change to nanoflash code if the recipe already builds that section via the instantiate helper.

- **New recipe**  
  - Implement a class that satisfies the recipe interface (`__init__`, `setup`, `train`, `cleanup`).  
  - Register it in the recipe registry under a name (e.g. `RECIPES["custom"] = CustomRecipe`).  
  - Set `recipe: custom` in the YAML.

- **New strategy (e.g. different distributed or device flow)**  
  - Either implement as a different recipe that uses the same config shape but different setup/train logic, or introduce a small strategy registry and have the default recipe choose a strategy from config (e.g. `train.strategy: fsdp`). Prefer the minimal approach: one default recipe, optional second recipe for a different flow; avoid deep inheritance unless needed.

---

## 13. Implementation Checklist for Agents

- [ ] Create `nanoflash/` layout: `config/`, `core/`, `pipeline/`, `train.py`, `README.md`.
- [ ] Implement `core/config.py`: CLI `--config` + key=value merge with OmegaConf, return resolved `DictConfig`.
- [ ] Implement `core/instantiate.py`: recursive `class_path` → `_target_`; `build(cfg_node, *args, **kwargs)` using `hydra.utils.instantiate`.
- [ ] Implement `pipeline/recipe.py`: recipe interface + `SFTRecipe` (setup order as in section 8) + recipe registry.
- [ ] Implement `train.py`: load config → project path → resolve class_path → get recipe → setup → train → cleanup.
- [ ] Add `config/train_example.yaml` matching the schema and example in section 4.3.
- [ ] README: how to run (`python train.py --config ...`), how to override (`key=value`), meaning of `project` and `class_path`, and how to add a new component or recipe (point to this design).
