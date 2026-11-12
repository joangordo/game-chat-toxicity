"""Collect run predictions and align them with the gold labels."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config
from .runs import list_runs, load_json, load_predictions


def collect_runs(include_smoke: bool = False) -> list[dict]:
    runs = []
    for d in list_runs():
        cfg = load_json(d / "run_config.json")
        if not include_smoke and d.name.endswith("_smoke"):
            continue
        base = cfg.get("name", d.name)
        loss = cfg.get("loss")
        model = f"{base}_{loss}" if loss and loss != "ce" else base
        runs.append({"dir": d, "run": d.name, "model": model,
                     "family": cfg.get("family", "?"), "type": cfg.get("type", "multiclass"),
                     "seed": cfg.get("seed"), "loss": cfg.get("loss"), "cfg": cfg})
    return runs


def aligned(run: dict, split: str, gold: pd.DataFrame) -> pd.DataFrame | None:
    """Predictions merged onto the gold rows (same order as `gold`)."""
    preds = load_predictions(run["dir"], split)
    if preds is None:
        return None
    m = gold.merge(preds, on="id", how="left", validate="one_to_one")
    pcols = [c for c in m.columns if c.startswith("p") and c[1:].isdigit()]
    if pcols:
        if m[pcols].isna().any().any():
            raise ValueError(f"{run['run']}: missing predictions for some {split} rows")
        probs = m[[f"p{k}" for k in range(config.NUM_LABELS)]].to_numpy()
        m["pred"] = probs.argmax(1)
        m["p_toxic_score"] = 1 - probs[:, 0]
    else:
        m["p_toxic_score"] = m["p_toxic"]
    return m


def probs_of(m: pd.DataFrame) -> np.ndarray:
    return m[[f"p{k}" for k in range(config.NUM_LABELS)]].to_numpy()
