"""
09 - Prepare the demo model and, optionally, publish it.

1. Train the model to publish with its weights kept (separate '_export' run,
   excluded from the evaluation):
       python scripts/05_train_transformer.py --config configs/xlmr_base.json --loss focal --seed 13 --save-model
2. Prepare it (temperature, thresholds, label names, model card):
       python scripts/09_prepare_demo.py --run xlmr_base_focal_s13_export
3. Publish (after `hf auth login` with a WRITE token):
       python scripts/09_prepare_demo.py --run xlmr_base_focal_s13_export \
           --push-model joangordo/game-chat-toxicity-xlmr --push-space joangordo/game-chat-toxicity
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src import config
from src.data import load_processed
from src.evaluation import aligned, probs_of
from src.metrics import (apply_temperature, binary_metrics, fit_temperature, macro_f1,
                         threshold_for_precision, threshold_for_recall)
from src.runs import load_json

MODEL_CARD = """---
language: multilingual
library_name: transformers
pipeline_tag: text-classification
base_model: FacebookAI/xlm-roberta-base
tags: [toxicity, content-moderation, gaming]
---

# Game chat toxicity classifier (XLM-RoBERTa base)

Fine-tuned on GameTox (World of Tanks chat) for six classes: {labels}.
Training code, evaluation and limitations: https://github.com/joangordo/game-chat-toxicity

| Metric (this checkpoint) | Value |
|---|---:|
| Validation macro-F1 | {val_f1:.3f} |
| Test macro-F1 | {test_f1:.3f} |
| Test binary AUROC (toxic vs non-toxic) | {test_auroc:.3f} |

The test set has only 8 Threats and 3 Extremism messages, so macro-F1 is very noisy
(bootstrap SD ≈ 0.035). Use `demo_config.json` for the calibration temperature ({temperature:.2f})
and the moderation thresholds chosen on validation.

**Limitations.** Misses most implicit toxicity; weaker in Russian; trained on one game
(transfer to Dota 2 drops AUROC from ~0.95 to ~0.83). Not suitable for automatic moderation
decisions without human review.

**Data.** GameTox: Naseem et al., NAACL 2025. Use subject to the dataset's terms.
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="Run folder name ending in _export")
    ap.add_argument("--push-model", default=None, help="Hugging Face model repo, e.g. user/name")
    ap.add_argument("--push-space", default=None, help="Hugging Face Space repo, e.g. user/name")
    args = ap.parse_args()

    rd = config.RUNS / args.run
    model_dir = rd / "model"
    if not model_dir.exists():
        sys.exit(f"STOP: {model_dir} not found. Train with --save-model first (see the docstring).")
    cfg = load_json(rd / "run_config.json")

    d = load_processed()
    run = {"dir": rd, "run": args.run, "type": "multiclass"}
    mv, mt = aligned(run, "val", d["val"]), aligned(run, "test", d["test"])
    y_val, y_test = d["val"]["label"].astype(int).to_numpy(), d["test"]["label"].astype(int).to_numpy()
    pv = probs_of(mv)
    T = fit_temperature(y_val, pv)
    s_val = 1 - apply_temperature(pv, T)[:, 0]
    yb_val = (y_val > 0).astype(int)
    demo_cfg = {"labels": {str(k): v for k, v in config.LABELS.items()}, "temperature": T,
                "max_tokens": cfg["max_tokens"],
                "threshold_auto": threshold_for_precision(yb_val, s_val, 0.95),
                "threshold_review": threshold_for_recall(yb_val, s_val, 0.90),
                "source_run": args.run}
    (model_dir / "demo_config.json").write_text(json.dumps(demo_cfg, indent=2), encoding="utf-8")

    # Readable label names in the published config
    from transformers import AutoConfig
    mc = AutoConfig.from_pretrained(model_dir)
    mc.id2label = {k: v for k, v in config.LABELS.items()}
    mc.label2id = {v: k for k, v in config.LABELS.items()}
    mc.save_pretrained(model_dir)

    metrics = {"val_f1": macro_f1(y_val, mv["pred"]), "test_f1": macro_f1(y_test, mt["pred"]),
               "test_auroc": binary_metrics((y_test > 0).astype(int), mt["p_toxic_score"])["auroc"]}
    (model_dir / "README.md").write_text(MODEL_CARD.format(
        labels=", ".join(config.LABELS.values()), temperature=T, **metrics), encoding="utf-8")
    print(f"Temperature {T:.2f}; thresholds auto={demo_cfg['threshold_auto']}, "
          f"review={demo_cfg['threshold_review']:.3f}")
    print(f"Val macro-F1 {metrics['val_f1']:.3f}, test macro-F1 {metrics['test_f1']:.3f}, "
          f"test AUROC {metrics['test_auroc']:.3f}")

    demo_src = config.ROOT / "demo"
    try:
        from importlib.metadata import version
        sdk = version("gradio")
    except Exception:
        sdk = "6.0.0"
    space_readme = (demo_src / "README.md").read_text(encoding="utf-8").replace("{sdk_version}", sdk) \
        .replace("{model_repo}", args.push_model or "joangordo/game-chat-toxicity-xlmr")

    if args.push_model or args.push_space:
        from huggingface_hub import HfApi
        api = HfApi()
        if args.push_model:
            api.create_repo(args.push_model, repo_type="model", exist_ok=True)
            api.upload_folder(folder_path=str(model_dir), repo_id=args.push_model, repo_type="model",
                              commit_message="Upload fine-tuned model")
            print(f"Model published: https://huggingface.co/{args.push_model}")
        if args.push_space:
            stage = rd / "space"
            if stage.exists():
                shutil.rmtree(stage)
            stage.mkdir()
            for f in ("app.py", "requirements.txt"):
                shutil.copy(demo_src / f, stage / f)
            app = (stage / "app.py").read_text(encoding="utf-8")
            if args.push_model:
                app = app.replace("joangordo/game-chat-toxicity-xlmr", args.push_model)
            (stage / "app.py").write_text(app, encoding="utf-8")
            (stage / "README.md").write_text(space_readme, encoding="utf-8")
            api.create_repo(args.push_space, repo_type="space", space_sdk="gradio", exist_ok=True)
            api.upload_folder(folder_path=str(stage), repo_id=args.push_space, repo_type="space",
                              commit_message="Upload demo")
            print(f"Space published: https://huggingface.co/spaces/{args.push_space}")
    else:
        print(f"\nReady. Try it locally:\n  cd demo\n  set MODEL_ID={model_dir}\n  python app.py")


if __name__ == "__main__":
    main()
