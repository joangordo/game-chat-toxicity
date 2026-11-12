"""Shared command-line handling for the training scripts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_train_args(description: str) -> tuple[dict, argparse.Namespace]:
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--config", required=True, help="JSON file in configs/")
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--loss", choices=["ce", "focal"], default=None, help="Override the config")
    ap.add_argument("--amp", choices=["auto", "bf16", "fp16", "off"], default="auto")
    ap.add_argument("--batch-size", type=int, default=None, help="Override the config")
    ap.add_argument("--grad-accum", type=int, default=None, help="Override the config")
    ap.add_argument("--epochs", type=int, default=None, help="Override the config")
    ap.add_argument("--limit", type=int, default=None, help="Use only N training messages (smoke test)")
    ap.add_argument("--save-model", action="store_true",
                    help="Keep the weights (for the demo). Saved as a separate '_export' run "
                         "that is excluded from the evaluation")
    args = ap.parse_args()
    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    for key in ("loss", "batch_size", "grad_accum", "epochs"):
        val = getattr(args, key)
        if val is not None:
            cfg[key] = val
    cfg["seed"] = args.seed
    cfg["amp_mode"] = args.amp
    suffix = f"_{cfg['loss']}" if cfg["loss"] != "ce" else ""
    cfg["run_name"] = (f"{cfg['name']}{suffix}_s{args.seed}" + ("_smoke" if args.limit else "")
                       + ("_export" if args.save_model else ""))
    return cfg, args
