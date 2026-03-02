"""
Alpaca-style SFT dataset loader.
"""
import json
import urllib.request
from functools import partial
from typing import Any

from datasets import load_dataset


def _load_dataset_fallback(source: str, split: str, **kwargs):
    """Fallback when load_dataset fails with fsspec glob error (datasets<2.19 + fsspec)."""
    r = urllib.request.urlopen(
        f"https://datasets-server.huggingface.co/parquet?dataset={source}"
    )
    data = json.loads(r.read().decode())
    urls = [f["url"] for f in data.get("parquet_files", []) if f.get("split") == split]
    if not urls:
        urls = [f["url"] for f in data.get("parquet_files", [])]
    if not urls:
        raise ValueError(
            f"Could not load {source} via fallback. Upgrade datasets>=2.19.1 and huggingface_hub>=0.21.2 to fix."
        )
    ds_full = load_dataset("parquet", data_files=urls)
    return ds_full[split] if split in ds_full else list(ds_full.values())[0]


_tokenizer_ref = None


def _tokenize_alpaca(example, max_length: int, train_on_input: bool):
    tokenizer = _tokenizer_ref
    instruction = example.get("instruction", "")
    inp = example.get("input", "")
    output = example.get("output", "")
    if inp:
        prompt = f"Instruction: {instruction}\nInput: {inp}\nResponse: "
    else:
        prompt = f"Instruction: {instruction}\nResponse: "
    full_text = prompt + output
    tok = tokenizer(
        full_text,
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_tensors=None,
    )
    input_ids = tok["input_ids"]
    labels = input_ids.copy()
    if not train_on_input:
        prompt_tok = tokenizer(
            prompt,
            truncation=True,
            max_length=max_length,
            return_tensors=None,
        )
        prompt_len = len(prompt_tok["input_ids"])
        for i in range(min(prompt_len, len(labels))):
            labels[i] = -100
    return {"input_ids": input_ids, "attention_mask": tok["attention_mask"], "labels": labels}


def load_alpaca_dataset(
    tokenizer,
    source: str = "yahma/alpaca-cleaned",
    max_length: int = 2048,
    split: str = "train",
    train_on_input: bool = True,
    **kwargs,
):
    """
    Load Alpaca-style dataset and tokenize for causal LM.

    Expects tokenizer with __call__ that accepts text and returns input_ids, attention_mask.
    Returns dataset with keys: input_ids, attention_mask, labels.
    """
    try:
        ds = load_dataset(source, split=split, **kwargs)
    except ValueError as e:
        if "Invalid pattern" in str(e) and "**" in str(e):
            ds = _load_dataset_fallback(source, split, **kwargs)
        else:
            raise

    global _tokenizer_ref
    _tokenizer_ref = tokenizer
    ds = ds.map(
        partial(_tokenize_alpaca, max_length=max_length, train_on_input=train_on_input),
        remove_columns=ds.column_names,
        num_proc=1,
    )
    ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    return ds
