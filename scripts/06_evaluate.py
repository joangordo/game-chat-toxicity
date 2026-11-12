"""
06 - Evaluation with uncertainty (Phase 4).

For every run in runs/:
  * Macro-F1 on the official test set with a 95% bootstrap CI, and on the
    'novel' subset (test messages whose text never appears in train).
  * Variability across seeds vs variability from the finite test set.
  * Binary toxic / non-toxic AUROC and F1 (threshold chosen on validation),
    which also allows comparing the off-the-shelf generic model.
  * Paired bootstrap tests between models on the same test messages.
For the best model (chosen on VALIDATION, never on test):
  * Per-class and per-language results, calibration with temperature scaling,
    and a two-threshold moderation policy (auto-action / human review).

Run from the project root:
    python scripts/06_evaluate.py
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import config
from src.data import load_processed
from src.evaluation import aligned, collect_runs, probs_of
from src.metrics import (apply_temperature, binary_metrics, bootstrap_macro_f1, expected_calibration_error,
                         fit_temperature, macro_f1, paired_bootstrap, per_class_f1,
                         threshold_for_precision, threshold_for_recall)
from src.runs import save_json

LANG_GROUPS = ["en", "ru", "pl", "de", "cs", "short"]


def best_f1_threshold(y, score) -> float:
    from sklearn.metrics import precision_recall_curve
    p, r, t = precision_recall_curve(y, score)
    f1 = 2 * p * r / np.clip(p + r, 1e-12, None)
    return float(t[np.nanargmax(f1[:-1])])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--include-smoke", action="store_true")
    ap.add_argument("--n-boot", type=int, default=config.N_BOOT)
    ap.add_argument("--detail-run", default=None,
                    help="Run for the per-class/language/calibration/policy section "
                         "(default: best on validation)")
    args = ap.parse_args()
    config.FIGURES.mkdir(parents=True, exist_ok=True)

    d = load_processed()
    val, test = d["val"], d["test"]
    if test["label"].isna().any():
        sys.exit("STOP: test labels not available; nothing to evaluate on test.")
    y_val = val["label"].astype(int).to_numpy()
    y_test = test["label"].astype(int).to_numpy()
    yb_val, yb_test = (y_val > 0).astype(int), (y_test > 0).astype(int)
    novel = ~test["in_train"].to_numpy()

    runs = collect_runs(args.include_smoke)
    if not runs:
        sys.exit("STOP: no runs found. Train some models first.")

    rows, cache = [], {}
    for r in runs:
        mv, mt = aligned(r, "val", val), aligned(r, "test", test)
        if mv is None or mt is None:
            continue
        cache[r["run"]] = (mv, mt)
        thr = best_f1_threshold(yb_val, mv["p_toxic_score"])
        b = binary_metrics(yb_test, mt["p_toxic_score"], thr)
        row = {"run": r["run"], "model": r["model"], "family": r["family"], "type": r["type"],
               "seed": r["seed"], "bin_auroc": b["auroc"], "bin_f1": b["f1"],
               "bin_precision": b["precision"], "bin_recall": b["recall"]}
        if r["type"] == "multiclass":
            full = bootstrap_macro_f1(y_test, mt["pred"], n_boot=args.n_boot)
            nov = bootstrap_macro_f1(y_test[novel], mt["pred"].to_numpy()[novel], n_boot=args.n_boot)
            pcf = per_class_f1(y_test, mt["pred"])
            frequent = float(np.nanmean(pcf[:4]))  # without Threats and Extremism (≈ 10 test messages)
            row.update(val_macro_f1=macro_f1(y_val, mv["pred"]), test_macro_f1=full["macro_f1"],
                       test_ci_low=full["ci_low"], test_ci_high=full["ci_high"], test_boot_sd=full["boot_sd"],
                       novel_macro_f1=nov["macro_f1"], novel_ci_low=nov["ci_low"], novel_ci_high=nov["ci_high"],
                       test_macro_f1_4_frequent=frequent)
            for k, f in enumerate(pcf):
                row[f"f1_class{k}"] = f
        rows.append(row)
    res = pd.DataFrame(rows)
    res.to_csv(config.RESULTS / "run_metrics.csv", index=False)
    mc = res[res["type"] == "multiclass"].copy()

    # --- Seeds vs test-set noise ---------------------------------------------------------
    agg = (mc.groupby("model").agg(n_seeds=("run", "size"), val_mean=("val_macro_f1", "mean"),
                                   test_mean=("test_macro_f1", "mean"), test_seed_sd=("test_macro_f1", "std"),
                                   test_boot_sd=("test_boot_sd", "mean"), novel_mean=("novel_macro_f1", "mean"),
                                   bin_auroc=("bin_auroc", "mean"))
           .sort_values("val_mean", ascending=False).reset_index())
    agg.to_csv(config.RESULTS / "model_comparison.csv", index=False)

    # --- Selected run per model (best validation), paired tests ----------------------------
    selected = mc.loc[mc.groupby("model")["val_macro_f1"].idxmax()].sort_values("val_macro_f1", ascending=False)
    best_run = selected.iloc[0]["run"]
    detail_run = args.detail_run or best_run
    if detail_run not in cache:
        sys.exit(f"STOP: --detail-run {detail_run} not found among runs.")
    paired = []
    ref = "tfidf_lr" if "tfidf_lr" in cache else None
    pairs = [(a, ref) for a in selected["run"] if ref and a != ref]
    if len(selected) >= 2:
        pairs.append((selected.iloc[0]["run"], selected.iloc[1]["run"]))
    for a, b in dict.fromkeys(pairs):
        pb = paired_bootstrap(y_test, cache[a][1]["pred"], cache[b][1]["pred"], n_boot=args.n_boot)
        paired.append({"model_a": a, "model_b": b, **pb})
    paired = pd.DataFrame(paired)

    # --- Best model: per class, per language, calibration, moderation policy --------------
    mv, mt = cache[detail_run]
    pc = pd.DataFrame({"class": [config.LABELS[k] for k in config.LABELS],
                       "test_support": np.bincount(y_test, minlength=config.NUM_LABELS),
                       "f1": per_class_f1(y_test, mt["pred"])})
    lang = test["lang"].where(test["lang"].isin(LANG_GROUPS), "other")
    lang_rows = []
    for g in [*LANG_GROUPS, "other"]:
        m = (lang == g).to_numpy()
        if m.sum() >= 50:
            yb, sb = yb_test[m], mt["p_toxic_score"].to_numpy()[m]
            lang_rows.append({"language": g, "n": int(m.sum()), "toxic_share": yb.mean(),
                              "macro_f1": macro_f1(y_test[m], mt["pred"].to_numpy()[m]),
                              "binary_auroc": binary_metrics(yb, sb)["auroc"] if 0 < yb.mean() < 1 else np.nan})
    lang_df = pd.DataFrame(lang_rows)

    pv, pt = probs_of(mv), probs_of(mt)
    T = fit_temperature(y_val, pv)
    pt_cal = apply_temperature(pt, T)
    calib = {"temperature": T, "ece_before": expected_calibration_error(y_test, pt),
             "ece_after": expected_calibration_error(y_test, pt_cal)}

    sv = 1 - apply_temperature(pv, T)[:, 0]
    st = 1 - pt_cal[:, 0]
    t_auto = threshold_for_precision(yb_val, sv, 0.95)
    t_review = threshold_for_recall(yb_val, sv, 0.90)
    policy = []
    if t_auto is not None:
        auto = st >= t_auto
        review = (st >= t_review) & ~auto
        policy = [{"band": "auto-action", "threshold": t_auto, "share_of_messages": auto.mean(),
                   "precision": yb_test[auto].mean() if auto.any() else np.nan,
                   "share_of_toxic_caught": (auto & (yb_test == 1)).sum() / yb_test.sum()},
                  {"band": "human review", "threshold": t_review, "share_of_messages": review.mean(),
                   "precision": yb_test[review].mean() if review.any() else np.nan,
                   "share_of_toxic_caught": (review & (yb_test == 1)).sum() / yb_test.sum()},
                  {"band": "missed", "threshold": np.nan, "share_of_messages": (~auto & ~review).mean(),
                   "precision": np.nan,
                   "share_of_toxic_caught": (~auto & ~review & (yb_test == 1)).sum() / yb_test.sum()}]
    policy = pd.DataFrame(policy)

    # --- Figures -------------------------------------------------------------------------
    sel = selected.sort_values("val_macro_f1")
    fig, ax = plt.subplots(figsize=(7.5, 0.5 * len(sel) + 1.8))
    y = np.arange(len(sel))
    ax.errorbar(sel["test_macro_f1"], y, xerr=[sel["test_macro_f1"] - sel["test_ci_low"],
                                              sel["test_ci_high"] - sel["test_macro_f1"]],
                fmt="o", color="#4C72B0", capsize=3, label="official test (95% bootstrap CI)")
    ax.scatter(sel["novel_macro_f1"], y + 0.2, marker="s", color="#C44E52", s=20, label="novel test messages")
    for i, mname in enumerate(sel["model"]):
        seeds = mc.loc[mc["model"] == mname, "test_macro_f1"]
        ax.scatter(seeds, np.full(len(seeds), i - 0.2), marker="|", color="grey", s=60)
    ax.set_yticks(y, sel["model"])
    ax.set_xlabel("Macro-F1 (grey ticks: other seeds)")
    ax.set_title("Test macro-F1 by model (run selected on validation)")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(config.FIGURES / "06_model_comparison.png", dpi=150)

    fig, ax = plt.subplots(figsize=(5, 4.5))
    bins = np.linspace(0, 1, 11)
    for probs, lab in ((pt, "before"), (pt_cal, f"after (T = {T:.2f})")):
        conf, corr = probs.max(1), (probs.argmax(1) == y_test)
        idx = np.clip(np.digitize(conf, bins[1:-1]), 0, 9)
        xs = [conf[idx == b].mean() for b in range(10) if (idx == b).sum() >= 20]
        ys = [corr[idx == b].mean() for b in range(10) if (idx == b).sum() >= 20]
        ax.plot(xs, ys, "o-", label=lab)
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set_xlabel("Predicted confidence")
    ax.set_ylabel("Observed accuracy")
    ax.set_title(f"Calibration: {detail_run}")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(config.FIGURES / "06_calibration.png", dpi=150)

    # --- Report --------------------------------------------------------------------------
    def ci(r, a, b, c):
        return f"{r[a]:.3f} [{r[b]:.3f}, {r[c]:.3f}]"

    lines = ["# Evaluation", "",
             "Runs are selected on **validation** macro-F1; test is only used to report. "
             "CIs are 95% percentile bootstrap over test messages.", "",
             "## Models (selected run per model)", "",
             "| Model | Val macro-F1 | Test macro-F1 [95% CI] | Novel messages [95% CI] | Mean F1, 4 frequent classes | Binary AUROC |",
             "|---|---:|---|---|---:|---:|",
             *[f"| {r['model']} | {r['val_macro_f1']:.3f} | {ci(r, 'test_macro_f1', 'test_ci_low', 'test_ci_high')} "
               f"| {ci(r, 'novel_macro_f1', 'novel_ci_low', 'novel_ci_high')} | {r['test_macro_f1_4_frequent']:.3f} "
               f"| {r['bin_auroc']:.3f} |"
               for _, r in selected.iterrows()], "",
             "![Comparison](figures/06_model_comparison.png)", "",
             "## Seeds vs test-set noise", "",
             "`test_seed_sd`: variation from retraining with another seed. `test_boot_sd`: variation from "
             "the finite test set. If either is comparable to the gap between two models, the ranking is not reliable.", "",
             agg.to_markdown(index=False, floatfmt=".3f"), "",
             "## Paired bootstrap comparisons (same test messages)", "",
             paired.to_markdown(index=False, floatfmt=".4f") if len(paired) else "_Not enough models._", "",
             "## Binary toxic vs non-toxic (all runs, including the generic model)", "",
             res.groupby("model")[["bin_auroc", "bin_f1", "bin_precision", "bin_recall"]].mean()
             .sort_values("bin_auroc", ascending=False).to_markdown(floatfmt=".3f"), "",
             f"## Detailed analysis: `{detail_run}`"
             + (" (best on validation)" if detail_run == best_run else f" (best on validation: `{best_run}`)"), "",
             "### Per class", "", pc.to_markdown(index=False, floatfmt=".3f"), "",
             "Classes with a handful of test examples make macro-F1 very noisy: one message more or "
             "less correct moves the score by several points.", "",
             "### Per language (langid, groups with ≥ 50 test messages)", "",
             lang_df.to_markdown(index=False, floatfmt=".3f") if len(lang_df) else "_n/a_", "",
             "### Calibration", "",
             f"Temperature fitted on validation: T = {T:.2f}. Expected calibration error on test: "
             f"{calib['ece_before']:.3f} before, {calib['ece_after']:.3f} after.", "",
             "![Calibration](figures/06_calibration.png)", "",
             "### Moderation policy (thresholds chosen on validation)", "",
             "Auto-action: precision ≥ 95% on validation. Human review: the remaining band needed to "
             "reach 90% recall of toxic messages.", "",
             policy.to_markdown(index=False, floatfmt=".3f") if len(policy) else
             "_95% precision is not reachable on validation for this model._", ""]
    (config.RESULTS / "eval_report.md").write_text("\n".join(lines), encoding="utf-8")
    save_json(config.RESULTS / "eval_summary.json", {"best_run": best_run, "detail_run": detail_run, **calib})

    pd.set_option("display.width", 180)
    print(selected[["model", "run", "val_macro_f1", "test_macro_f1", "test_ci_low", "test_ci_high",
                    "novel_macro_f1", "bin_auroc"]].to_string(index=False, float_format="%.3f"))
    print(f"\nBest on validation: {best_run}")
    print(f"Saved: {config.RESULTS / 'eval_report.md'}")


if __name__ == "__main__":
    main()
