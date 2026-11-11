"""
05 - Fine-tune a multilingual transformer with our own PyTorch loop (Phase 3).

The first run downloads the model from Hugging Face (about 1 GB).

Examples (from the project root):
    python scripts/05_train_transformer.py --config configs/xlmr_base.json --seed 13
    python scripts/05_train_transformer.py --config configs/mdeberta_base.json --seed 13
    python scripts/05_train_transformer.py --config configs/xlmr_base.json --limit 2000 --epochs 1   # smoke test
    # GPU with 4 GB: smaller batches, same effective batch size, fp16
    python scripts/05_train_transformer.py --config configs/xlmr_base.json --batch-size 8 --grad-accum 4 --amp fp16
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from src import config
from src.cli import parse_train_args
from src.data import load_processed
from src.runs import environment, predict_and_save_all, run_dir, save_json
from src.training import (EncodedDataset, amp_settings, build_loss, class_weights, get_device,
                          make_loader, predict_proba, set_seed, train)
from src.transformer import encode, load_model_and_tokenizer


def main() -> None:
    cfg, args = parse_train_args(__doc__)
    set_seed(cfg["seed"])
    device = get_device()
    print(f"Run {cfg['run_name']} on {device}"
          + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))

    model, tok = load_model_and_tokenizer(cfg["hf_model"], config.NUM_LABELS)
    d = load_processed()
    tr = d["train"].sample(n=args.limit, random_state=cfg["seed"]) if args.limit else d["train"]
    va = d["val"]
    y_tr = tr["label"].astype(int).to_numpy()
    y_va = va["label"].astype(int).to_numpy()
    pad = tok.pad_token_id

    train_dl = make_loader(EncodedDataset(encode(tok, tr["text"], cfg["max_tokens"]), y_tr), pad,
                           cfg["batch_size"], shuffle=True, seed=cfg["seed"])
    val_dl = make_loader(EncodedDataset(encode(tok, va["text"], cfg["max_tokens"])), pad,
                         cfg["eval_batch_size"], shuffle=False)
    loss_fn = build_loss(cfg["loss"], class_weights(y_tr, cfg["class_weight_power"]), cfg["focal_gamma"])
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params / 1e6:.0f}M. Train: {len(tr):,}. Batch {cfg['batch_size']} x accum {cfg['grad_accum']}.")

    t0 = time.time()
    result = train(model, train_dl, val_dl, y_va, loss_fn, cfg["lr"], cfg["weight_decay"], cfg["epochs"],
                   cfg["patience"], cfg["warmup_frac"], cfg["grad_clip"], device, amp=args.amp,
                   grad_accum=cfg["grad_accum"])
    amp_dtype, _ = amp_settings(device, args.amp)
    if device.type == "cuda":
        result["peak_gpu_memory_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)

    def predict(texts):
        dl = make_loader(EncodedDataset(encode(tok, texts, cfg["max_tokens"])), pad,
                         cfg["eval_batch_size"], shuffle=False)
        return predict_proba(model, dl, device, amp_dtype)

    rd = run_dir(cfg["run_name"])
    t1 = time.time()
    predict_and_save_all(rd, predict)
    if args.save_model:  # ~1 GB per model: only when asked (e.g. for the demo)
        model.hf.save_pretrained(rd / "model")
        tok.save_pretrained(rd / "model")
    save_json(rd / "run_config.json", {**cfg, "type": "multiclass", "n_params": n_params,
                                       "train_seconds": round(t1 - t0, 1),
                                       "predict_seconds": round(time.time() - t1, 1),
                                       "n_train": len(tr), **result, "env": environment()})
    print(f"Best val macro-F1 {result['best_val_macro_f1']:.4f}. Saved to {rd}")


if __name__ == "__main__":
    main()
