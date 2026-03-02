"""
Training metric logger.
"""
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from torch.utils.tensorboard import SummaryWriter


class TrainingLogger:
    """Log metrics to console and TensorBoard events."""

    def __init__(self, log_dir: Optional[str] = None, log_every_n_steps: int = 1):
        self.log_dir = Path(log_dir) if log_dir else None
        self.log_every_n_steps = log_every_n_steps
        self._writer: Optional[SummaryWriter] = None
        if self.log_dir:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            runs_dir = self.log_dir / "runs"
            runs_dir.mkdir(parents=True, exist_ok=True)
            run_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            run_path = runs_dir / run_name
            self._writer = SummaryWriter(log_dir=str(run_path))

    def log(self, step: int, metrics: dict[str, Any]) -> None:
        if step > 0 and step % self.log_every_n_steps != 0:
            return
        parts = [f"step={step}"]
        for k, v in metrics.items():
            if isinstance(v, float):
                parts.append(f"{k}={v:.4f}")
            else:
                parts.append(f"{k}={v}")
        line = " ".join(parts)
        print(line, flush=True)
        if self._writer:
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    self._writer.add_scalar(k, v, step)

    def close(self) -> None:
        if self._writer:
            self._writer.close()
            self._writer = None
