"""
Main run logic: load config, prepare env, create recipe, setup, train, cleanup.
"""
import os
import sys
from pathlib import Path

from omegaconf import DictConfig

from nanoflash.core.config import load_config
from nanoflash.core.instantiate import resolve_class_path
from nanoflash.pipeline.recipe import RECIPES


def run() -> None:
    cfg = load_config()

    if cfg.get("project"):
        project = Path(cfg.project).resolve()
        if project.exists() and str(project) not in sys.path:
            sys.path.insert(0, str(project))

    resolve_class_path(cfg)

    recipe_name = cfg.get("recipe", "default")
    if recipe_name not in RECIPES:
        raise ValueError(f"Unknown recipe: {recipe_name}. Available: {list(RECIPES.keys())}")

    recipe_cls = RECIPES[recipe_name]
    recipe = recipe_cls(cfg)
    recipe.setup(cfg)
    recipe.train()
    recipe.cleanup()


if __name__ == "__main__":
    run()
