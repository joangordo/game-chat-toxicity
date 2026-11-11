"""Generic PyTorch training loop shared by the char models and the transformers."""
from __future__ import annotations

import math
import random
import time
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from . import config
from .metrics import macro_f1


# --- Reproducibility and device --------------------------------------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def amp_settings(device: torch.device, mode: str = "auto"):
    """Returns (autocast dtype or None, use GradScaler).
    bf16 where supported (RTX 30xx and newer), fp16 + scaler otherwise, off on CPU."""
    if device.type != "cuda" or mode == "off":
        return None, False
    if mode in ("auto", "bf16") and torch.cuda.is_bf16_supported():
        return torch.bfloat16, False
    return torch.float16, True


# --- Data ------------------------------------------------------------------------------------
class EncodedDataset(Dataset):
    """Pre-encoded token id lists + optional labels."""

    def __init__(self, ids: list[list[int]], labels=None):
        self.ids = ids
        self.labels = None if labels is None else np.asarray(labels, dtype=np.int64)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        return self.ids[i], (-1 if self.labels is None else int(self.labels[i]))


def make_collate(pad_id: int):
    def collate(batch):
        seqs, labels = zip(*batch)
        L = max(len(s) for s in seqs)
        x = torch.full((len(seqs), L), pad_id, dtype=torch.long)
        mask = torch.zeros((len(seqs), L), dtype=torch.long)
        for i, s in enumerate(seqs):
            x[i, : len(s)] = torch.tensor(s, dtype=torch.long)
            mask[i, : len(s)] = 1
        return {"input_ids": x, "attention_mask": mask, "labels": torch.tensor(labels)}
    return collate


def make_loader(ds: Dataset, pad_id: int, batch_size: int, shuffle: bool, seed: int = 0) -> DataLoader:
    g = torch.Generator()
    g.manual_seed(seed)
    # num_workers=0: safe on Windows, and the data are small and pre-encoded.
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, collate_fn=make_collate(pad_id),
                      num_workers=0, generator=g)


# --- Losses ------------------------------------------------------------------------------------
def class_weights(labels, power: float = 0.5, k: int = config.NUM_LABELS) -> torch.Tensor:
    """(1 / frequency) ** power, normalised to mean 1. power=0.5 softens the
    extreme weights that pure inverse frequency would give to 24 examples."""
    counts = np.bincount(np.asarray(labels, dtype=int), minlength=k).astype(float)
    w = (counts.sum() / np.maximum(counts, 1)) ** power
    return torch.tensor(w / w.mean(), dtype=torch.float)


class FocalLoss(nn.Module):
    """Cross-entropy down-weighting easy examples: (1 - p_t) ** gamma * CE."""

    def __init__(self, weight: torch.Tensor | None = None, gamma: float = 2.0):
        super().__init__()
        self.register_buffer("weight", weight if weight is not None else None)
        self.gamma = gamma

    def forward(self, logits, target):
        logp = F.log_softmax(logits.float(), dim=-1)
        logp_t = logp.gather(1, target[:, None]).squeeze(1)
        loss = -((1 - logp_t.exp()) ** self.gamma) * logp_t
        if self.weight is not None:
            loss = loss * self.weight[target]
        return loss.mean()


def build_loss(kind: str, weights: torch.Tensor | None, gamma: float = 2.0) -> nn.Module:
    if kind == "ce":
        return nn.CrossEntropyLoss(weight=weights)
    if kind == "focal":
        return FocalLoss(weights, gamma)
    raise ValueError(f"Unknown loss: {kind}")


def linear_warmup_schedule(optimizer, total_steps: int, warmup_frac: float):
    warm = max(1, int(total_steps * warmup_frac))

    def f(step):
        if step < warm:
            return (step + 1) / warm
        return max(0.0, (total_steps - step) / max(1, total_steps - warm))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, f)


# --- Loop -------------------------------------------------------------------------------------
@torch.no_grad()
def predict_proba(model: nn.Module, loader: DataLoader, device, amp_dtype=None) -> np.ndarray:
    model.eval()
    out = []
    ctx = torch.autocast(device.type, dtype=amp_dtype) if amp_dtype else nullcontext()
    for batch in loader:
        with ctx:
            logits = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
        out.append(torch.softmax(logits.float(), dim=-1).cpu().numpy())
    return np.concatenate(out)


def train(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, val_labels,
          loss_fn: nn.Module, lr: float, weight_decay: float, epochs: int, patience: int,
          warmup_frac: float, grad_clip: float, device, amp: str = "auto",
          grad_accum: int = 1, log=print) -> dict:
    """Train with early stopping on validation macro-F1. Restores the best weights."""
    model.to(device)
    loss_fn.to(device)
    amp_dtype, use_scaler = amp_settings(device, amp)
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
    decay = [p for n, p in model.named_parameters() if p.ndim > 1 and "emb" not in n.lower()]
    no_decay = [p for n, p in model.named_parameters() if not (p.ndim > 1 and "emb" not in n.lower())]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": weight_decay},
                             {"params": no_decay, "weight_decay": 0.0}], lr=lr)
    steps = math.ceil(len(train_loader) / grad_accum) * epochs
    sched = linear_warmup_schedule(opt, steps, warmup_frac)

    best, best_state, bad, history = -1.0, None, 0, []
    for epoch in range(1, epochs + 1):
        model.train()
        t0, total, n = time.time(), 0.0, 0
        opt.zero_grad(set_to_none=True)
        for step, batch in enumerate(train_loader, 1):
            ctx = torch.autocast(device.type, dtype=amp_dtype) if amp_dtype else nullcontext()
            with ctx:
                logits = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
                loss = loss_fn(logits.float(), batch["labels"].to(device)) / grad_accum
            scaler.scale(loss).backward()
            if step % grad_accum == 0 or step == len(train_loader):
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                sched.step()
            total += loss.item() * grad_accum * len(batch["labels"])
            n += len(batch["labels"])
            if not math.isfinite(loss.item()):
                raise FloatingPointError("Loss became NaN/inf. Try --amp off or a lower learning rate.")
        probs = predict_proba(model, val_loader, device, amp_dtype)
        f1 = macro_f1(val_labels, probs.argmax(1))
        history.append({"epoch": epoch, "train_loss": total / n, "val_macro_f1": f1,
                        "seconds": round(time.time() - t0, 1)})
        log(f"  epoch {epoch}: loss {total / n:.4f}  val macro-F1 {f1:.4f}  ({time.time() - t0:.0f}s)")
        if f1 > best:
            best, bad = f1, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                log(f"  early stopping (no improvement for {patience} epochs)")
                break
    model.load_state_dict(best_state)
    return {"best_val_macro_f1": best, "history": history, "amp": str(amp_dtype)}
