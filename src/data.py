"""Load GameTox and CONDA into a common format: id, text, label (+ extras)."""
from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import pandas as pd

from . import config


def normalize_text(s: str) -> str:
    """Normalisation used ONLY to detect duplicates (models see the raw text)."""
    s = unicodedata.normalize("NFKC", str(s)).lower().strip()
    return re.sub(r"\s+", " ", s)


def _find(raw_dir: Path, split: str, kind: str) -> Path:
    hits = sorted(raw_dir.rglob(f"{split}_index_{kind}.csv"))
    if not hits:
        raise FileNotFoundError(f"No '{split}_index_{kind}.csv' under {raw_dir}. "
                                "See data/README.md for where to download GameTox.")
    return hits[0]


def load_gametox(raw_dir: Path = config.RAW_GAMETOX) -> dict[str, pd.DataFrame]:
    """Official splits. Each file pair: *_index_text.csv (index, message) and
    *_index_label.csv (index, label). Test labels may be missing."""
    out = {}
    for split in config.SPLITS:
        text = pd.read_csv(_find(raw_dir, split, "text"))
        try:
            lab = pd.read_csv(_find(raw_dir, split, "label"))
            df = text.merge(lab, on="index", how="left", validate="one_to_one")
        except FileNotFoundError:
            df = text.assign(label=pd.NA)
        df = df.rename(columns={"index": "id", "message": "text"})
        df["text"] = df["text"].fillna("").astype(str)
        df["label"] = pd.to_numeric(df["label"], errors="coerce").astype("Int64")
        bad = df["label"].dropna()
        if not bad.isin(config.LABELS).all():
            raise ValueError(f"Unexpected labels in {split}: {sorted(bad.unique())}")
        df["split"] = split
        out[split] = df[["id", "text", "label", "split"]].reset_index(drop=True)
    return out


def load_conda(raw_dir: Path = config.RAW_CONDA) -> pd.DataFrame:
    """Labelled CONDA utterances (train + valid; the official test has no labels)."""
    frames = []
    for name in ("train", "valid"):
        p = raw_dir / f"CONDA_{name}.csv"
        if not p.exists():
            raise FileNotFoundError(f"{p} not found. Run scripts/00_download_data.py.")
        frames.append(pd.read_csv(p).assign(conda_split=name))
    df = pd.concat(frames, ignore_index=True)
    df = df.rename(columns={"Id": "id", "utterance": "text", "matchId": "match_id"})
    df["text"] = df["text"].fillna("").astype(str)
    df["label_binary"] = df["intentClass"].isin(config.CONDA_TOXIC_INTENTS).astype(int)
    return df[["id", "text", "match_id", "intentClass", "label_binary", "conda_split"]]


def to_binary(labels) -> pd.Series:
    """Any toxic class -> 1."""
    return (pd.Series(labels).astype(int) > 0).astype(int)


def add_novelty_flag(splits: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Mark val/test messages whose normalised text also appears in train.
    'Novel' messages measure generalisation rather than memorisation."""
    seen = set(splits["train"]["text"].map(normalize_text))
    for s in ("val", "test"):
        splits[s] = splits[s].assign(in_train=splits[s]["text"].map(normalize_text).isin(seen))
    splits["train"] = splits["train"].assign(in_train=True)
    return splits


def save_processed(splits: dict[str, pd.DataFrame], conda: pd.DataFrame | None) -> None:
    config.PROCESSED.mkdir(parents=True, exist_ok=True)
    for s, df in splits.items():
        df.to_parquet(config.PROCESSED / f"gametox_{s}.parquet", index=False)
    if conda is not None:
        conda.to_parquet(config.PROCESSED / "conda_labelled.parquet", index=False)


def load_processed() -> dict[str, pd.DataFrame]:
    out = {}
    for s in config.SPLITS:
        p = config.PROCESSED / f"gametox_{s}.parquet"
        if not p.exists():
            raise FileNotFoundError(f"{p} missing. Run scripts/01_audit_data.py first.")
        out[s] = pd.read_parquet(p)
        out[s]["label"] = out[s]["label"].astype("Int64")
    return out


def load_processed_conda() -> pd.DataFrame:
    return pd.read_parquet(config.PROCESSED / "conda_labelled.parquet")
