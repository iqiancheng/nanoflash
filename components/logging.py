"""
Training metric logger.
"""
from pathlib import Path
from typing import Any, Optional


class TrainingLogger:
    """Log metrics to console and optionally to a file."""

    def __init__(self, log_dir: Optional[str] = None, log_every_n_steps: int = 10):
        self.log_dir = Path(log_dir) if log_dir else None
        self.log_every_n_steps = log_every_n_steps
        if self.log_dir:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            self.file_path = self.log_dir / "train.log"
        else:
            self.file_path = None

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
        print(line)
        if self.file_path:
            with open(self.file_path, "a") as f:
                f.write(line + "\n")
