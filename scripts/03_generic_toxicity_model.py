"""
03 - Off-the-shelf toxicity model, no training on game chat (Phase 1).

Measures the domain gap: how well does a general-purpose toxicity classifier
(trained on Wikipedia / social media comments) work on game chat?
The comparison is binary (toxic vs non-toxic) because these models do not use
the six GameTox classes.

Examples (from the project root):
    python scripts/03_generic_toxicity_model.py
    python scripts/03_generic_toxicity_model.py --model unitary/multilingual-toxic-xlm-roberta
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from src.runs import environment, predict_and_save_all, run_dir, save_json
from src.training import get_device


def toxic_index(id2label: dict) -> int:
    """Pick the overall 'toxic' output; fall back to the first output."""
    names = {int(k): str(v).lower() for k, v in id2label.items()}
    for wanted in ("toxic", "toxicity", "toxic_comment"):
        for k, v in names.items():
            if v == wanted:
                return k
    for k, v in names.items():
        if "toxic" in v and "non" not in v and "not" not in v and "severe" not in v:
            return k
    return 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="unitary/toxic-bert")
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--batch-size", type=int, default=128)
    args = ap.parse_args()

    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    device = get_device()
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model).to(device).eval()
    id2label = model.config.id2label
    k = toxic_index(id2label)
    multilabel = model.config.problem_type == "multi_label_classification" or model.config.num_labels == 1
    print(f"Model outputs: {id2label}\nUsing output {k} ({id2label[k]}), "
          f"{'sigmoid (multi-label)' if multilabel else 'softmax'}.")

    @torch.no_grad()
    def predict(texts):
        scores = []
        for i in range(0, len(texts), args.batch_size):
            enc = tok(texts[i:i + args.batch_size], truncation=True, max_length=args.max_tokens,
                      padding=True, return_tensors="pt").to(device)
            logits = model(**enc).logits.float()
            p = torch.sigmoid(logits) if multilabel else torch.softmax(logits, -1)
            scores.append(p[:, k].cpu().numpy())
        return np.concatenate(scores)

    name = "generic_" + args.model.split("/")[-1].replace("-", "_")
    rd = run_dir(name)
    predict_and_save_all(rd, predict)
    save_json(rd / "run_config.json", {"name": name, "family": "generic", "type": "binary",
                                       "hf_model": args.model, "output_used": id2label[k],
                                       "seed": None, "env": environment()})
    print(f"Saved to {rd}")


if __name__ == "__main__":
    main()
