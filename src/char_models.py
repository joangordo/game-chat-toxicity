"""Character-level models written from scratch in PyTorch."""
from __future__ import annotations

from collections import Counter

import torch
import torch.nn as nn

PAD, UNK = 0, 1


class CharVocab:
    """Characters seen at least `min_freq` times in training; case is kept
    because shouting in capitals is a signal in chat."""

    def __init__(self, texts, min_freq: int = 2, max_len: int = 128):
        counts = Counter(ch for t in texts for ch in t)
        self.itos = ["<pad>", "<unk>"] + sorted(c for c, n in counts.items() if n >= min_freq)
        self.stoi = {c: i for i, c in enumerate(self.itos)}
        self.max_len = max_len

    def __len__(self):
        return len(self.itos)

    def encode(self, text: str) -> list[int]:
        ids = [self.stoi.get(ch, UNK) for ch in text[: self.max_len]]
        return ids or [UNK]

    def to_dict(self) -> dict:
        return {"itos": self.itos, "max_len": self.max_len}

    @classmethod
    def from_dict(cls, d: dict) -> "CharVocab":
        v = cls([], max_len=d["max_len"])
        v.itos = d["itos"]
        v.stoi = {c: i for i, c in enumerate(v.itos)}
        return v


class CharCNN(nn.Module):
    """Embedding -> parallel 1D convolutions (several widths) -> masked max-pool."""

    def __init__(self, vocab_size: int, num_labels: int, emb_dim: int = 64,
                 n_filters: int = 128, kernel_sizes=(3, 4, 5), dropout: float = 0.3):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=PAD)
        self.convs = nn.ModuleList([nn.Conv1d(emb_dim, n_filters, k, padding=k // 2) for k in kernel_sizes])
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(n_filters * len(kernel_sizes), num_labels)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        x = self.emb(input_ids).transpose(1, 2)            # (B, E, L)
        mask = attention_mask.unsqueeze(1).bool()          # (B, 1, L)
        pooled = []
        for conv in self.convs:
            h = torch.relu(conv(x))[..., : input_ids.size(1)]
            h = h.masked_fill(~mask, float("-inf"))
            pooled.append(h.max(dim=2).values)
        return self.out(self.drop(torch.cat(pooled, dim=1)))


class CharBiGRU(nn.Module):
    """Embedding -> bidirectional GRU -> masked mean + max pooling."""

    def __init__(self, vocab_size: int, num_labels: int, emb_dim: int = 64,
                 hidden: int = 128, layers: int = 1, dropout: float = 0.3):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=PAD)
        self.gru = nn.GRU(emb_dim, hidden, num_layers=layers, batch_first=True, bidirectional=True,
                          dropout=dropout if layers > 1 else 0.0)
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(4 * hidden, num_labels)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        lengths = attention_mask.sum(1).clamp(min=1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(self.emb(input_ids), lengths,
                                                   batch_first=True, enforce_sorted=False)
        out, _ = self.gru(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True, total_length=input_ids.size(1))
        m = attention_mask.unsqueeze(-1).float()
        mean = (out * m).sum(1) / m.sum(1).clamp(min=1)
        mx = out.masked_fill(m == 0, float("-inf")).max(1).values
        return self.out(self.drop(torch.cat([mean, mx], dim=1)))
