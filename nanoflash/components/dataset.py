"""
Alpaca-style SFT dataset loader.
"""
from typing import Any

from datasets import load_dataset


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
    ds = load_dataset(source, split=split, **kwargs)

    def tokenize(example):
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

    ds = ds.map(tokenize, remove_columns=ds.column_names, num_proc=1)
    ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    return ds
