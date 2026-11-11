"""
07 - Cross-game transfer: World of Tanks -> Dota 2 (Phase 5).

Models trained only on GameTox are applied, without any retraining, to the
labelled CONDA messages (binary: explicit or implicit toxicity vs the rest).
The decision threshold is the one chosen on GameTox validation: no CONDA
label is used for tuning. An 'oracle' threshold tuned on CONDA itself is
reported separately to tell ranking quality apart from threshold shift.
CIs resample whole matches (cluster bootstrap), because messages from the
same match are not independent.

Run from the project root:
    python scripts/07_transfer_conda.py
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src import config
from src.data import load_processed, load_processed_conda
from src.evaluation import aligned, collect_runs
from src.metrics import _boot_indices, binary_metrics


def best_f1_threshold(y, score) -> float:
    from sklearn.metrics import precision_recall_curve
    p, r, t = precision_recall_curve(y, score)
    f1 = 2 * p * r / np.clip(p + r, 1e-12, None)
    return float(t[np.nanargmax(f1[:-1])])


def cluster_ci_auroc(y, s, groups, n_boot, seed=0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    vals = []
    for idx in _boot_indices(len(y), n_boot, rng, groups):
        if 0 < y[idx].mean() < 1:
            vals.append(roc_auc_score(y[idx], s[idx]))
    return tuple(np.quantile(vals, [0.025, 0.975]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--include-smoke", action="store_true")
    ap.add_argument("--n-boot", type=int, default=500)
    args = ap.parse_args()
    try:
        conda = load_processed_conda()
    except FileNotFoundError:
        sys.exit("STOP: CONDA not processed. Run 00_download_data.py and 01_audit_data.py.")
    d = load_processed()
    val, test = d["val"], d["test"]
    yb_val = (val["label"].astype(int) > 0).astype(int).to_numpy()
    has_test = test["label"].notna().all()
    y_c = conda["label_binary"].to_numpy()
    groups = conda["match_id"].to_numpy()

    rows = []
    for r in collect_runs(args.include_smoke):
        mc, mv = aligned(r, "conda", conda), aligned(r, "val", val)
        if mc is None or mv is None or r["model"] == "majority":
            continue
        s = mc["p_toxic_score"].to_numpy()
        thr = best_f1_threshold(yb_val, mv["p_toxic_score"])
        b = binary_metrics(y_c, s, thr)
        oracle = binary_metrics(y_c, s, best_f1_threshold(y_c, s))
        lo, hi = cluster_ci_auroc(y_c, s, groups, args.n_boot)
        row = {"run": r["run"], "model": r["model"], "seed": r["seed"],
               "conda_auroc": b["auroc"], "conda_auroc_ci_low": lo, "conda_auroc_ci_high": hi,
               "conda_f1_gametox_threshold": b["f1"], "conda_f1_oracle_threshold": oracle["f1"],
               "recall_explicit": (s[conda["intentClass"] == "E"] >= thr).mean(),
               "recall_implicit": (s[conda["intentClass"] == "I"] >= thr).mean(),
               "false_positive_rate": (s[y_c == 0] >= thr).mean()}
        if has_test:
            mt = aligned(r, "test", test)
            row["gametox_test_auroc"] = binary_metrics((test["label"].astype(int) > 0).astype(int),
                                                       mt["p_toxic_score"])["auroc"]
        rows.append(row)
    if not rows:
        sys.exit("STOP: no runs with CONDA predictions.")
    res = pd.DataFrame(rows)
    res.to_csv(config.RESULTS / "transfer_conda.csv", index=False)
    by_model = res.groupby("model").mean(numeric_only=True).drop(columns="seed", errors="ignore") \
        .sort_values("conda_auroc", ascending=False)

    lines = ["# Cross-game transfer: World of Tanks → Dota 2 (CONDA)", "",
             f"{len(conda):,} labelled Dota 2 messages from {conda['match_id'].nunique():,} matches, "
             f"{y_c.mean():.1%} toxic (explicit or implicit). No CONDA label is used for training or "
             "thresholds; the oracle column is only a reference.", "",
             "AUROC CIs resample whole matches (cluster bootstrap).", "",
             by_model.to_markdown(floatfmt=".3f"), "",
             "`gametox_test_auroc` is the same binary task in the original game: the drop to "
             "`conda_auroc` measures how much is lost when moving to another game. A large gap between "
             "the GameTox-threshold F1 and the oracle F1 means the scores still rank messages well but "
             "the threshold does not transfer.", ""]
    (config.RESULTS / "transfer_report.md").write_text("\n".join(lines), encoding="utf-8")
    pd.set_option("display.width", 180)
    print(by_model.to_string(float_format="%.3f"))
    print(f"\nSaved: {config.RESULTS / 'transfer_report.md'}")


if __name__ == "__main__":
    main()
