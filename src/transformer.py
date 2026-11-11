"""Hugging Face encoder + classification head, trained with our own loop."""
from __future__ import annotations

import torch
import torch.nn as nn


class HFClassifier(nn.Module):
    """Thin wrapper so every model has forward(input_ids, attention_mask) -> logits."""

    def __init__(self, hf_model: nn.Module):
        super().__init__()
        self.hf = hf_model

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return self.hf(input_ids=input_ids, attention_mask=attention_mask).logits


def load_model_and_tokenizer(name: str, num_labels: int):
    """Separated so tests can swap in a tiny offline model."""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForSequenceClassification.from_pretrained(name, num_labels=num_labels)
    return HFClassifier(model), tok


def encode(tokenizer, texts, max_len: int) -> list[list[int]]:
    enc = tokenizer(list(texts), truncation=True, max_length=max_len, add_special_tokens=True)
    return enc["input_ids"]
