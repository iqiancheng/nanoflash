"""
Load YAML config and merge CLI key=value overrides.
"""
import argparse
from pathlib import Path

from omegaconf import DictConfig, OmegaConf


def load_config() -> DictConfig:
    """
    Parse CLI: required --config, optional key=value overrides.
    Load YAML, merge overrides (CLI wins), resolve interpolations.
    """
    parser = argparse.ArgumentParser(description="Nanoflash training")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    args, unknown = parser.parse_known_args()

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")

    base_cfg = OmegaConf.load(cfg_path)

    dotlist = []
    for arg in unknown:
        if "=" not in arg:
            raise ValueError(f"Override must be key=value, got: {arg}")
        k, v = arg.split("=", 1)
        if v == "None":
            v = "null"
        dotlist.append(f"{k}={v}")

    if dotlist:
        override_cfg = OmegaConf.from_dotlist(dotlist)
        base_cfg = OmegaConf.merge(base_cfg, override_cfg)

    OmegaConf.resolve(base_cfg)
    return base_cfg
