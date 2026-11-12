"""
Run the whole experiment plan, or one phase of it. Runs that already exist
in runs/ are skipped, so you can stop and resume at any time.

    python run_experiments.py --phase data
    python run_experiments.py --phase baselines
    python run_experiments.py --phase char
    python run_experiments.py --phase transformers
    python run_experiments.py --phase evaluate
    python run_experiments.py --phase all

Extra options are passed to the training scripts, e.g. for a 4 GB GPU:
    python run_experiments.py --phase transformers --extra "--batch-size 8 --grad-accum 4 --amp fp16"
"""
import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from src import config  # noqa: E402

GENERIC_MODELS = ["unitary/toxic-bert", "unitary/multilingual-toxic-xlm-roberta"]
CHAR_PLAN = [("configs/char_cnn.json", "ce"), ("configs/char_bigru.json", "ce"),
             ("configs/char_cnn.json", "focal")]
TRANSFORMER_PLAN = [("configs/xlmr_base.json", "ce"), ("configs/xlmr_base_w075.json", "ce"),
                    ("configs/xlmr_base.json", "focal"), ("configs/mdeberta_base.json", "ce")]


def run(cmd: list[str]) -> None:
    print(f"\n>>> {' '.join(cmd)}", flush=True)
    if subprocess.run([sys.executable, *cmd], cwd=ROOT).returncode != 0:
        sys.exit(f"Stopped: {' '.join(cmd)} failed.")


def done(cfg_path: str, loss: str, seed: int) -> bool:
    name = json.loads((ROOT / cfg_path).read_text())["name"]
    suffix = f"_{loss}" if loss != "ce" else ""
    return (config.RUNS / f"{name}{suffix}_s{seed}" / "run_config.json").exists()


def train_plan(script: str, plan, seeds, extra) -> None:
    for cfg, loss in plan:
        for seed in seeds:
            if done(cfg, loss, seed):
                print(f"skip {cfg} {loss} seed {seed} (already done)")
                continue
            run([script, "--config", cfg, "--loss", loss, "--seed", str(seed), *extra])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True,
                    choices=["data", "baselines", "char", "transformers", "evaluate", "all"])
    ap.add_argument("--seeds", type=int, nargs="+", default=config.SEEDS)
    ap.add_argument("--extra", default="", help="Extra arguments for the training scripts")
    args = ap.parse_args()
    extra = shlex.split(args.extra)
    phases = ["data", "baselines", "char", "transformers", "evaluate"] if args.phase == "all" else [args.phase]

    for phase in phases:
        print(f"\n{'=' * 70}\nPHASE: {phase}\n{'=' * 70}")
        if phase == "data":
            run(["scripts/00_download_data.py"])
            run(["scripts/01_audit_data.py"])
        elif phase == "baselines":
            if not (config.RUNS / "tfidf_lr" / "run_config.json").exists():
                run(["scripts/02_baselines.py"])
            for m in GENERIC_MODELS:
                name = "generic_" + m.split("/")[-1].replace("-", "_")
                if not (config.RUNS / name / "run_config.json").exists():
                    run(["scripts/03_generic_toxicity_model.py", "--model", m])
        elif phase == "char":
            train_plan("scripts/04_train_char_model.py", CHAR_PLAN, args.seeds, extra)
        elif phase == "transformers":
            train_plan("scripts/05_train_transformer.py", TRANSFORMER_PLAN, args.seeds, extra)
        elif phase == "evaluate":
            run(["scripts/06_evaluate.py"])
            run(["scripts/07_transfer_conda.py"])
            run(["scripts/08_build_report.py"])
    print("\nDone.")


if __name__ == "__main__":
    main()
