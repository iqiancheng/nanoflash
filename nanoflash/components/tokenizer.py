"""
Tokenizer loader for HuggingFace models.
"""
from typing import Optional

from transformers import AutoTokenizer


def load_tokenizer(
    model_name_or_path: str,
    max_length: Optional[int] = None,
    trust_remote_code: bool = True,
    **kwargs,
):
    """Load HuggingFace tokenizer for Qwen/Qwen3 models."""
    tok = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=trust_remote_code,
        **kwargs,
    )
    if max_length is not None:
        tok.model_max_length = max_length
    return tok
