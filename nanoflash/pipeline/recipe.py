"""
Recipe interface and SFT recipe.
"""
import random
from typing import Optional

import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from nanoflash.components.collate import causal_lm_collate
from nanoflash.core.instantiate import build, resolve_target


class SFTRecipe:
    """
    Default SFT recipe: model, data, optimizer, train loop.
    All components built from config via build().
    """

    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg
        train_cfg = cfg.get("train", {})
        self.device = torch.device(train_cfg.get("device", "cuda"))
        dtype_str = train_cfg.get("dtype", "bf16")
        self.dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}.get(
            dtype_str, torch.bfloat16
        )
        self.batch_size = train_cfg.get("batch_size", 8)
        self.max_steps = train_cfg.get("max_steps", 10000)
        self.seed = train_cfg.get("seed", 42)
        self.output_dir = train_cfg.get("output_dir", "./output")
        self.log_every_n_steps = train_cfg.get("log_every_n_steps", 10)
        self.gradient_accumulation_steps = train_cfg.get("gradient_accumulation_steps", 1)
        self.num_workers = train_cfg.get("num_workers", 0)

        self._model = None
        self._optimizer = None
        self._lr_scheduler = None
        self._loss_fn = None
        self._train_loader = None
        self._val_loader = None
        self._checkpointer = None
        self._logger = None
        self._step = 0

    def setup(self, cfg: DictConfig) -> None:
        torch.manual_seed(self.seed)
        random.seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

        if cfg.get("checkpoint"):
            self._checkpointer = build(cfg.checkpoint)
            if cfg.get("data", {}).get("tokenizer") and getattr(self._checkpointer, "hf_output_dir", None):
                tokenizer = build(cfg.data.tokenizer)
                self._checkpointer.set_tokenizer(tokenizer)
        else:
            self._checkpointer = None

        self._model = build(cfg.model)
        if self._model is None:
            raise ValueError("model is required in config")
        if getattr(self._model, "hf_device_map", None) is None:
            self._model.to(self.device)

        ckpt = {}
        if self._checkpointer is not None:
            self._checkpointer.set_model(self._model)
            ckpt = self._checkpointer.load()
            if ckpt.get("model_state") is not None:
                self._model.load_state_dict(ckpt["model_state"], strict=True)
            self._step = ckpt.get("step", 0)

        if cfg.get("loss"):
            self._loss_fn = build(cfg.loss)
        else:
            self._loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100)

        self._optimizer = build(cfg.optimizer, self._model.parameters())
        if self._optimizer is None:
            raise ValueError("optimizer is required in config")

        if self._checkpointer is not None:
            self._checkpointer.set_optimizer(self._optimizer)
            if ckpt.get("opt_state") is not None:
                self._optimizer.load_state_dict(ckpt["opt_state"])

        if cfg.get("lr_scheduler"):
            self._lr_scheduler = build(
                cfg.lr_scheduler,
                self._optimizer,
                num_training_steps=self.max_steps,
            )
        else:
            self._lr_scheduler = None

        collate_fn = resolve_target(cfg.collate_fn) if cfg.get("collate_fn") else None
        if collate_fn is None:
            collate_fn = causal_lm_collate

        train_ds = build(cfg.data)
        if train_ds is None:
            raise ValueError("data is required in config")
        self._train_loader = DataLoader(
            train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
        )

        if cfg.get("data_val"):
            val_ds = build(cfg.data_val)
            self._val_loader = DataLoader(
                val_ds,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=0,
                collate_fn=collate_fn,
            )
        else:
            self._val_loader = None

        if cfg.get("logging"):
            self._logger = build(cfg.logging)
        else:
            from nanoflash.components.logging import TrainingLogger
            self._logger = TrainingLogger(
                log_dir=f"{self.output_dir}/logs",
                log_every_n_steps=self.log_every_n_steps,
            )

    def train(self) -> None:
        self._model.train()
        accum_steps = self.gradient_accumulation_steps
        train_iter = iter(self._train_loader)
        accum_loss = 0.0

        while self._step < self.max_steps:
            accum_loss = 0.0
            self._optimizer.zero_grad()
            for _ in range(accum_steps):
                try:
                    batch = next(train_iter)
                except StopIteration:
                    train_iter = iter(self._train_loader)
                    batch = next(train_iter)

                device = next(self._model.parameters()).device if getattr(self._model, "hf_device_map", None) else self.device
                batch = {k: v.to(device) for k, v in batch.items()}
                model_inputs = {"input_ids": batch["input_ids"], "attention_mask": batch["attention_mask"], "use_cache": False}
                outputs = self._model(**model_inputs)
                shift_logits = outputs.logits[..., :-1, :].contiguous().view(-1, outputs.logits.size(-1))
                shift_labels = batch["labels"][..., 1:].contiguous().view(-1).to(shift_logits.device)
                loss = self._loss_fn(shift_logits, shift_labels)
                (loss / accum_steps).backward()
                accum_loss += loss.item() / accum_steps

            torch.nn.utils.clip_grad_norm_(self._model.parameters(), 1.0)
            self._optimizer.step()
            if self._lr_scheduler is not None:
                self._lr_scheduler.step()

            self._step += 1

            if self._logger and self._step % self.log_every_n_steps == 0:
                self._logger.log(self._step, {"loss": accum_loss})

            if self._checkpointer and self._checkpointer.should_save(self._step):
                self._checkpointer.save(self._step)

        if self._checkpointer and self._step > 0:
            self._checkpointer.save(self._step)

    def cleanup(self) -> None:
        pass


RECIPES = {"default": SFTRecipe, "sft": SFTRecipe}
