"""
Convert class_path -> _target_, then build objects via Hydra instantiate.
"""
import importlib
import copy
import os
import sys
from typing import Any

from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf


def resolve_target(cfg_node: Any) -> Any:
    """
    Resolve _target_/class_path to the callable without instantiating.
    Use for collate_fn and similar pass-through callables.
    """
    if cfg_node is None:
        return None
    if not OmegaConf.is_dict(cfg_node):
        return cfg_node
    target = cfg_node.get("_target_") or cfg_node.get("class_path")
    if target is None:
        return cfg_node
    if isinstance(target, str):
        module_path, name = target.rsplit(".", 1)
        mod = importlib.import_module(module_path)
        return getattr(mod, name)
    return target


def resolve_class_path(cfg: Any) -> None:
    """Recursively rename class_path to _target_ for Hydra compatibility."""
    if OmegaConf.is_dict(cfg):
        for key, value in list(cfg.items()):
            if key == "class_path":
                cfg["_target_"] = cfg.pop("class_path")
            else:
                resolve_class_path(value)
    elif OmegaConf.is_list(cfg):
        for item in cfg:
            resolve_class_path(item)


def build(cfg_node: Any, *args: Any, **kwargs: Any) -> Any:
    """
    Instantiate object from config node.

    If cfg_node has _target_ (after class_path resolution), use Hydra instantiate.
    Otherwise return None.
    """
    if cfg_node is None:
        return None
    if not OmegaConf.is_dict(cfg_node):
        return cfg_node
    if "_target_" not in cfg_node:
        return cfg_node

    if os.getcwd() not in sys.path:
        sys.path.insert(0, os.getcwd())

    cfg_copy = copy.deepcopy(cfg_node)
    cfg_copy._set_flag(
        flags=["allow_objects", "struct", "readonly"],
        values=[True, False, False],
    )
    OmegaConf.resolve(cfg_copy)

    obj = OmegaConf.to_object(cfg_copy)
    return instantiate(obj, *args, **kwargs)
