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
        model: Optional[torch.nn.Module] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        **kwargs,
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.save_every_n_steps = save_every_n_steps
        self._model = model
        self._optimizer = optimizer
        self._step = 0

    def set_model(self, model: torch.nn.Module) -> None:
        self._model = model

    def set_optimizer(self, optimizer: torch.optim.Optimizer) -> None:
        self._optimizer = optimizer

    def load(self) -> dict[str, Any]:
        """Load checkpoint if exists. Returns dict with model_state, opt_state, step."""
        result = {"model_state": None, "opt_state": None, "step": 0}
        ckpt_path = self.checkpoint_dir / "checkpoint.pt"
        if not ckpt_path.exists():
            return result
        data = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        result["model_state"] = data.get("model_state")
        result["opt_state"] = data.get("opt_state")
        result["step"] = data.get("step", 0)
        self._step = result["step"]
        return result

    def save(self, step: int) -> None:
        """Save model, optimizer, step."""
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        data = {
            "step": step,
            "model_state": self._model.state_dict() if self._model else None,
            "opt_state": self._optimizer.state_dict() if self._optimizer else None,
        }
        torch.save(data, self.checkpoint_dir / "checkpoint.pt")

    def should_save(self, step: int) -> bool:
        return step > 0 and step % self.save_every_n_steps == 0
