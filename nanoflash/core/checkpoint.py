"""
Checkpointer for save/load model and training state.
"""
from pathlib import Path
from typing import Any, Optional

import torch


class Checkpointer:
    """
    Save and load model + optimizer + step count.
    Uses HuggingFace safetensors for model, torch for optimizer/step.
    """

    def __init__(
        self,
        checkpoint_dir: str,
        save_every_n_steps: int = 1000,
        hf_output_dir: Optional[str] = None,
        model: Optional[torch.nn.Module] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        **kwargs,
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.save_every_n_steps = save_every_n_steps
        self.hf_output_dir = Path(hf_output_dir) if hf_output_dir else None
        self._model = model
        self._optimizer = optimizer
        self._tokenizer = None

    def set_model(self, model: torch.nn.Module) -> None:
        self._model = model

    def set_optimizer(self, optimizer: torch.optim.Optimizer) -> None:
        self._optimizer = optimizer

    def set_tokenizer(self, tokenizer) -> None:
        self._tokenizer = tokenizer

    def _latest_step_dir(self) -> Optional[Path]:
        """Find step_* dir with max step. Fallback to root for legacy checkpoint.pt."""
        if not self.checkpoint_dir.exists():
            return None
        step_dirs = [d for d in self.checkpoint_dir.iterdir() if d.is_dir() and d.name.startswith("step_")]
        if step_dirs:
            steps = []
            for d in step_dirs:
                try:
                    steps.append((int(d.name.split("_")[1]), d))
                except (IndexError, ValueError):
                    pass
            if steps:
                return max(steps, key=lambda x: x[0])[1]
        if (self.checkpoint_dir / "checkpoint.pt").exists():
            return self.checkpoint_dir
        return None

    def load(self) -> dict[str, Any]:
        """Load checkpoint if exists. Returns dict with model_state, opt_state, step."""
        result = {"model_state": None, "opt_state": None, "step": 0}
        load_dir = self._latest_step_dir()
        if load_dir is None:
            return result
        ckpt_path = load_dir / "checkpoint.pt"
        if not ckpt_path.exists():
            return result
        data = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        result["model_state"] = data.get("model_state")
        result["opt_state"] = data.get("opt_state")
        result["step"] = data.get("step", 0)
        self._step = result["step"]
        return result

    def save(self, step: int) -> None:
        """Save model, optimizer, step to step_{step}/. Optionally save HF format."""
        step_dir = self.checkpoint_dir / f"step_{step}"
        step_dir.mkdir(parents=True, exist_ok=True)
        data = {
            "step": step,
            "model_state": self._model.state_dict() if self._model else None,
            "opt_state": self._optimizer.state_dict() if self._optimizer else None,
        }
        torch.save(data, step_dir / "checkpoint.pt")

        if self.hf_output_dir and self._model and hasattr(self._model, "save_pretrained"):
            hf_step_dir = self.hf_output_dir / f"step_{step}"
            hf_step_dir.mkdir(parents=True, exist_ok=True)
            self._model.save_pretrained(str(hf_step_dir), safe_serialization=True)
            if self._tokenizer is not None and hasattr(self._tokenizer, "save_pretrained"):
                self._tokenizer.save_pretrained(str(hf_step_dir))

    def should_save(self, step: int) -> bool:
        return step > 0 and step % self.save_every_n_steps == 0
