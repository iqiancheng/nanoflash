"""
Causal LM loader for HuggingFace models (Qwen, Qwen3, etc.).
"""
from typing import Optional

import torch
from transformers import AutoModelForCausalLM


def load_causal_lm(
    model_name_or_path: str,
    torch_dtype: Optional[torch.dtype] = None,
    device_map: Optional[str] = None,
    trust_remote_code: bool = True,
    gradient_checkpointing: bool = False,
    **kwargs,
):
    """
    Load Qwen/Qwen3 causal LM from HuggingFace.

    Args:
        model_name_or_path: HF model id or local path (e.g. Qwen/Qwen3-1.7B)
        torch_dtype: bf16, fp16, fp32 or None (auto)
        device_map: "auto" for single GPU, or None to load on default device
        gradient_checkpointing: enable activation checkpointing to reduce memory
    """
    if torch_dtype is None or isinstance(torch_dtype, str):
        dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
        torch_dtype = dtype_map.get(str(torch_dtype), torch.bfloat16)

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch_dtype,
        device_map=device_map,
        trust_remote_code=trust_remote_code,
        **kwargs,
    )
    if gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    return model
