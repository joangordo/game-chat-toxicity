"""Run directories: config, metrics and per-split probability files."""
from __future__ import annotations

import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from . import config


def run_dir(name: str) -> Path:
    d = config.RUNS / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_json(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def environment() -> dict:
    env = {"python": platform.python_version(), "platform": platform.platform(),
           "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    try:
        import torch
        env["torch"] = torch.__version__
        env["cuda"] = torch.version.cuda
        env["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except ImportError:
        pass
    return env


def save_predictions(d: Path, split: str, ids, probs: np.ndarray) -> None:
    """Multiclass: probs (n, K). Binary-only models: probs (n,) = P(toxic)."""
    probs = np.asarray(probs)
    df = pd.DataFrame({"id": np.asarray(ids)})
    if probs.ndim == 1:
        df["p_toxic"] = probs
    else:
        for k in range(probs.shape[1]):
            df[f"p{k}"] = probs[:, k]
    df.to_parquet(d / f"preds_{split}.parquet", index=False)


def load_predictions(d: Path, split: str) -> pd.DataFrame | None:
    p = d / f"preds_{split}.parquet"
    return pd.read_parquet(p) if p.exists() else None


def list_runs() -> list[Path]:
    return sorted(p.parent for p in config.RUNS.glob("*/run_config.json"))


def predict_and_save_all(d: Path, predict_fn) -> None:
    """Run `predict_fn(list_of_texts) -> probs` on val, test and (if present)
    the CONDA transfer set, and save the predictions in the run directory."""
    from .data import load_processed

    splits = load_processed()
    for split in ("val", "test"):
        df = splits[split]
        save_predictions(d, split, df["id"], predict_fn(df["text"].tolist()))
    conda_path = config.PROCESSED / "conda_labelled.parquet"
    if conda_path.exists():
        conda = pd.read_parquet(conda_path)
        save_predictions(d, "conda", conda["id"], predict_fn(conda["text"].tolist()))
