"""
Collate function for causal LM batches.
"""
from typing import Any, List

import torch


def causal_lm_collate(batch: List[dict[str, Any]]) -> dict[str, torch.Tensor]:
    """Stack batch tensors. Expects each item to have input_ids, attention_mask, labels."""
    input_ids = torch.stack([b["input_ids"] for b in batch])
    attention_mask = torch.stack([b["attention_mask"] for b in batch])
    labels = torch.stack([b["labels"] for b in batch])
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}
