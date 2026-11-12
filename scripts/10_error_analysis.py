"""
10 - Error analysis on the test set for one run.

Quantitative: confusion matrix and error rates by train overlap, language,
message length and label conflicts. Qualitative: sampled examples of each
error type, with offensive words masked.

Outputs
  results/error_analysis.md            masked, safe to publish after review
  results/figures/10_confusion.png
  results/error_samples_UNMASKED.csv   for reading locally only (git-ignored)

Run from the project root:
    python scripts/10_error_analysis.py                 # run of the detailed analysis in 06
    python scripts/10_error_analysis.py --run xlmr_base_focal_s13
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import config
from src.data import load_processed, normalize_text
from src.evaluation import aligned, collect_runs, probs_of
from src.masking import load_lexicon, mask
from src.metrics import confusion

SHORT = {0: "Non-toxic", 1: "Insult", 2: "Offensive", 3: "Hate", 4: "Threat", 5: "Extremism"}


def default_run() -> str:
    p = config.RESULTS / "eval_summary.json"
    if p.exists():
        s = json.loads(p.read_text(encoding="utf-8"))
        return s.get("detail_run") or s["best_run"]
    sys.exit("STOP: run scripts/06_evaluate.py first or pass --run.")


def error_rate_table(df: pd.DataFrame, by: str) -> pd.DataFrame:
    g = df.groupby(by, observed=True)
    return pd.DataFrame({"messages": g.size(), "error_rate": g["wrong"].mean(),
                         "toxic_share": g["toxic"].mean(),
                         "missed_toxic_rate": g.apply(lambda x: (x["wrong"] & x["toxic"]).sum()
                                                      / max(x["toxic"].sum(), 1), include_groups=False)})


def examples_table(df: pd.DataFrame, lex: set, n: int, seed: int, sort_by: str | None = None) -> str:
    if df.empty:
        return "_No cases._"
    sample = df.sort_values(sort_by, ascending=False).head(n) if sort_by else \
        df.sample(min(n, len(df)), random_state=seed)
    rows = ["| Message (masked) | True | Predicted | Confidence | Lang |", "|---|---|---|---:|---|"]
    for _, r in sample.iterrows():
        msg = mask(r["text"], lex).replace("|", "/").replace("\n", " ")[:90]
        rows.append(f"| `{msg}` | {SHORT[r['label']]} | {SHORT[r['pred']]} | {r['confidence']:.2f} | {r['lang']} |")
    return "\n".join(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=None)
    ap.add_argument("--n", type=int, default=8, help="Examples per category")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    run_name = args.run or default_run()
    runs = {r["run"]: r for r in collect_runs(include_smoke=True)}
    if run_name not in runs:
        sys.exit(f"STOP: run {run_name} not found.")

    d = load_processed()
    test = d["test"]
    m = aligned(runs[run_name], "test", test)
    probs = probs_of(m)
    df = m.assign(label=m["label"].astype(int), pred=m["pred"].astype(int),
                  confidence=probs.max(1), p_true=probs[np.arange(len(m)), m["label"].astype(int)])
    df["wrong"] = df["label"] != df["pred"]
    df["toxic"] = df["label"] > 0
    train = d["train"].assign(norm=d["train"]["text"].map(normalize_text))
    n_labels = train.groupby("norm")["label"].nunique()
    df["conflicting_in_train"] = df["text"].map(normalize_text).map(n_labels).fillna(0) > 1
    df["lang_group"] = df["lang"].where(df["lang"].isin(["en", "ru", "de", "pl", "short"]), "other")
    df["length"] = pd.cut(df["text"].str.len(), [0, 5, 15, 40, 10_000],
                          labels=["1-5 chars", "6-15", "16-40", ">40"])
    df["seen_in_train"] = np.where(df["in_train"], "text seen in train", "novel text")
    lex = load_lexicon()

    # --- Confusion matrix ---------------------------------------------------------------
    conf = confusion(df["label"], df["pred"])
    conf_df = pd.DataFrame(conf, index=[f"true {SHORT[k]}" for k in SHORT],
                           columns=[f"pred {SHORT[k]}" for k in SHORT])
    fig, ax = plt.subplots(figsize=(6.5, 5.2))
    norm = conf / np.maximum(conf.sum(1, keepdims=True), 1)
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    for i in range(conf.shape[0]):
        for j in range(conf.shape[1]):
            ax.text(j, i, conf[i, j], ha="center", va="center", fontsize=8,
                    color="white" if norm[i, j] > 0.5 else "black")
    ax.set_xticks(range(6), [SHORT[k] for k in SHORT], rotation=35, ha="right")
    ax.set_yticks(range(6), [SHORT[k] for k in SHORT])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion matrix (counts; colour = row share)\n{run_name}", fontsize=10)
    fig.tight_layout()
    fig.savefig(config.FIGURES / "10_confusion.png", dpi=150)

    # --- Error categories --------------------------------------------------------------
    cats = {
        "missed": df[df["toxic"] & (df["pred"] == 0)],
        "false_alarm": df[~df["toxic"] & (df["pred"] > 0)],
        "wrong_toxic_class": df[df["toxic"] & (df["pred"] > 0) & df["wrong"]],
        "non_english": df[df["wrong"] & df["lang_group"].isin(["ru", "de", "pl", "other"])],
        "confident": df[df["wrong"]],
        "conflicting": df[df["wrong"] & df["conflicting_in_train"]],
    }
    unmasked = pd.concat([c.assign(category=k) for k, c in cats.items()])
    unmasked[["category", "id", "text", "label", "pred", "confidence", "lang", "in_train"]] \
        .to_csv(config.RESULTS / "error_samples_UNMASKED.csv", index=False, encoding="utf-8")

    tables = {by: error_rate_table(df, by) for by in ["seen_in_train", "lang_group", "length", "conflicting_in_train"]}
    lines = [f"# Error analysis: `{run_name}` (test set)", "",
             f"{df['wrong'].sum():,} errors out of {len(df):,} messages ({df['wrong'].mean():.1%}). "
             "Offensive words are masked automatically; review before publishing.", "",
             "## Confusion matrix", "", conf_df.to_markdown(), "",
             "![Confusion](figures/10_confusion.png)", "",
             "## Where the errors are", ""]
    for by, t in tables.items():
        lines += [f"### By {by.replace('_', ' ')}", "", t.to_markdown(floatfmt=".3f"), ""]
    titles = {
        "missed": ("Missed toxicity (toxic predicted as non-toxic)", None),
        "false_alarm": ("False alarms (non-toxic predicted as toxic), most confident first", "confidence"),
        "wrong_toxic_class": ("Right that it is toxic, wrong class", None),
        "non_english": ("Errors in other languages", None),
        "confident": ("Most confident errors (candidates for label noise)", "confidence"),
        "conflicting": ("Errors on texts that appear in train with different labels", None),
    }
    for k, (title, sort_by) in titles.items():
        lines += [f"## {title}", "", f"{len(cats[k]):,} cases.", "",
                  examples_table(cats[k], lex, args.n, args.seed, sort_by), ""]
    (config.RESULTS / "error_analysis.md").write_text("\n".join(lines), encoding="utf-8")

    pd.set_option("display.width", 160)
    print(conf_df.to_string())
    for by, t in tables.items():
        print(f"\nBy {by}:\n{t.round(3).to_string()}")
    print(f"\nSaved: {config.RESULTS / 'error_analysis.md'} (masked)")
    print(f"Saved: {config.RESULTS / 'error_samples_UNMASKED.csv'} (local only, not uploaded)")


if __name__ == "__main__":
    main()
