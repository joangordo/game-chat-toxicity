"""
04 - Character-level model in PyTorch, from scratch (Phase 2).

Examples (from the project root):
    python scripts/04_train_char_model.py --config configs/char_cnn.json --seed 13
    python scripts/04_train_char_model.py --config configs/char_bigru.json --seed 42 --loss focal
    python scripts/04_train_char_model.py --config configs/char_cnn.json --limit 2000 --epochs 1   # smoke test
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from src.char_models import PAD, CharBiGRU, CharCNN, CharVocab
from src.cli import parse_train_args
from src.data import load_processed
from src.runs import environment, predict_and_save_all, run_dir, save_json
from src.training import (EncodedDataset, amp_settings, build_loss, class_weights, get_device,
                          make_loader, predict_proba, set_seed, train)
from src import config


def build_model(cfg: dict, vocab_size: int) -> torch.nn.Module:
    if cfg["arch"] == "cnn":
        return CharCNN(vocab_size, config.NUM_LABELS, cfg["emb_dim"], cfg["n_filters"],
                       tuple(cfg["kernel_sizes"]), cfg["dropout"])
    if cfg["arch"] == "bigru":
        return CharBiGRU(vocab_size, config.NUM_LABELS, cfg["emb_dim"], cfg["hidden"],
                         cfg["layers"], cfg["dropout"])
    raise ValueError(cfg["arch"])


def main() -> None:
    cfg, args = parse_train_args(__doc__)
    set_seed(cfg["seed"])
    device = get_device()
    print(f"Run {cfg['run_name']} on {device}")

    d = load_processed()
    tr = d["train"].sample(n=args.limit, random_state=cfg["seed"]) if args.limit else d["train"]
    va = d["val"]
    vocab = CharVocab(tr["text"], min_freq=2, max_len=cfg["max_chars"])
    y_tr = tr["label"].astype(int).to_numpy()
    y_va = va["label"].astype(int).to_numpy()

    train_dl = make_loader(EncodedDataset([vocab.encode(t) for t in tr["text"]], y_tr), PAD,
                           cfg["batch_size"], shuffle=True, seed=cfg["seed"])
    val_dl = make_loader(EncodedDataset([vocab.encode(t) for t in va["text"]]), PAD, 512, shuffle=False)

    model = build_model(cfg, len(vocab))
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Vocabulary: {len(vocab)} characters. Parameters: {n_params:,}")
    loss_fn = build_loss(cfg["loss"], class_weights(y_tr, cfg["class_weight_power"]), cfg["focal_gamma"])

    t0 = time.time()
    result = train(model, train_dl, val_dl, y_va, loss_fn, cfg["lr"], cfg["weight_decay"], cfg["epochs"],
                   cfg["patience"], cfg["warmup_frac"], cfg["grad_clip"], device, amp=args.amp)
    amp_dtype, _ = amp_settings(device, args.amp)

    def predict(texts):
        dl = make_loader(EncodedDataset([vocab.encode(t) for t in texts]), PAD, 512, shuffle=False)
        return predict_proba(model, dl, device, amp_dtype)

    rd = run_dir(cfg["run_name"])
    predict_and_save_all(rd, predict)
    # Char models are small (a few MB), so the weights are always kept.
    torch.save({"state_dict": model.state_dict(), "vocab": vocab.to_dict(), "config": cfg},
               rd / "model.pt")
    save_json(rd / "run_config.json", {**cfg, "type": "multiclass", "n_params": n_params,
                                       "train_seconds": round(time.time() - t0, 1),
                                       "n_train": len(tr), **result, "env": environment()})
    print(f"Best val macro-F1 {result['best_val_macro_f1']:.4f}. Saved to {rd}")


if __name__ == "__main__":
    main()
