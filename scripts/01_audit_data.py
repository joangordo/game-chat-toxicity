"""
01 - Data audit (Phase 0). No model involved.

  * Size and class balance per split.
  * Exact duplicates within each split (after light normalisation).
  * Same text with different labels (annotation conflicts).
  * Overlap between splits: share of val/test messages whose text is in train.
    These messages can be solved by memorisation; we keep them (the official
    benchmark includes them) but also report results on the 'novel' subset.
  * Message length and language mix.

Writes data/processed/*.parquet (used by every later script) and
results/audit_report.md.

Run from the project root:
    python scripts/01_audit_data.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import config
from src.data import add_novelty_flag, load_conda, load_gametox, normalize_text, save_processed


def detect_language(texts: pd.Series) -> pd.Series:
    """langid on messages with at least 3 letters; short chat is unreliable,
    so very short messages are marked 'short'."""
    import langid
    letters = texts.str.count(r"[^\W\d_]")
    out = pd.Series("short", index=texts.index)
    mask = letters >= 3
    out[mask] = [langid.classify(t)[0] for t in texts[mask]]
    return out


def main() -> None:
    config.RESULTS.mkdir(parents=True, exist_ok=True)
    config.FIGURES.mkdir(parents=True, exist_ok=True)
    splits = load_gametox()
    labelled = {s: d for s, d in splits.items() if d["label"].notna().all()}
    if "test" not in labelled:
        print("Note: test labels not found; test metrics will be skipped later.")

    for s, d in splits.items():
        d["norm"] = d["text"].map(normalize_text)
        d["n_chars"] = d["text"].str.len()
    splits = add_novelty_flag(splits)
    print("Detecting languages (takes a minute)...")
    for s, d in splits.items():
        d["lang"] = detect_language(d["text"])

    try:
        conda = load_conda()
    except FileNotFoundError:
        conda = None
        print("CONDA not found; transfer experiments will be skipped.")

    save_processed({s: d.drop(columns=["norm"]) for s, d in splits.items()}, conda)

    # --- Tables ---------------------------------------------------------------------------
    rows = []
    for s, d in splits.items():
        r = {"split": s, "messages": len(d), "unique_texts": d["norm"].nunique(),
             "duplicate_rows_share": 1 - d["norm"].nunique() / len(d),
             "median_chars": d["n_chars"].median(), "p95_chars": d["n_chars"].quantile(0.95)}
        if s != "train":
            r["share_text_in_train"] = d["in_train"].mean()
        rows.append(r)
    overview = pd.DataFrame(rows)

    balance = pd.DataFrame({s: d["label"].value_counts().reindex(config.LABELS).fillna(0).astype(int)
                            for s, d in labelled.items()})
    balance.index = [f"{k}: {v}" for k, v in config.LABELS.items()]

    overlap = pd.DataFrame({s: splits[s].groupby("label")["in_train"].mean().reindex(config.LABELS)
                            for s in ("val", "test") if s in labelled})
    overlap.index = [f"{k}: {v}" for k, v in config.LABELS.items()]

    tr = splits["train"]
    conflicts = tr.groupby("norm")["label"].nunique()
    conflict_texts = conflicts[conflicts > 1].index
    conflict_rows = tr[tr["norm"].isin(conflict_texts)]
    top_dups = tr["norm"].value_counts().head(15)

    langs = pd.concat([d["lang"] for d in splits.values()]).value_counts(normalize=True).head(12)

    # --- Figures ---------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.hist(tr["n_chars"].clip(upper=150), bins=75, color="#4C72B0")
    ax.set_xlabel("Characters per message (capped at 150)")
    ax.set_ylabel("Messages")
    ax.set_title("Message length, training split")
    fig.tight_layout()
    fig.savefig(config.FIGURES / "01_length.png", dpi=150)

    if not overlap.empty:
        fig, ax = plt.subplots(figsize=(7, 3.6))
        (100 * overlap).plot.barh(ax=ax)
        ax.set_xlabel("% of messages whose exact text appears in train")
        ax.set_title("Train overlap by class")
        fig.tight_layout()
        fig.savefig(config.FIGURES / "01_train_overlap.png", dpi=150)

    # --- Report ----------------------------------------------------------------------------
    lines = ["# Data audit (Phase 0)", "",
             "## Overview", "", overview.to_markdown(index=False, floatfmt=".3f"), "",
             "## Class balance", "", balance.to_markdown(), "",
             "## Overlap with the training split", "",
             "Share of messages whose normalised text also appears in train. "
             "Short, repeated chat ('gg', 'lol') explains much of it; these messages can be "
             "classified by memorisation, so results are also reported on the novel subset.", "",
             overlap.to_markdown(floatfmt=".3f") if not overlap.empty else "_No labelled val/test._", "",
             "![Overlap](figures/01_train_overlap.png)", "",
             "## Annotation conflicts in train", "",
             f"{len(conflict_texts):,} distinct texts appear with more than one label "
             f"({len(conflict_rows):,} rows, {len(conflict_rows) / len(tr):.1%} of train).", "",
             "## Most repeated messages in train", "",
             "| Message | Count |", "|---|---:|",
             *[f"| `{t[:40]}` | {n:,} |" for t, n in top_dups.items()], "",
             "## Languages (langid; messages with ≥ 3 letters, otherwise 'short')", "",
             langs.to_frame("share").to_markdown(floatfmt=".3f"), "",
             "![Length](figures/01_length.png)", ""]
    if conda is not None:
        lines += ["## CONDA (transfer set)", "",
                  f"{len(conda):,} labelled utterances from {conda['match_id'].nunique():,} Dota 2 matches; "
                  f"{conda['label_binary'].mean():.1%} toxic (explicit or implicit).", ""]
    (config.RESULTS / "audit_report.md").write_text("\n".join(lines), encoding="utf-8")

    pd.set_option("display.width", 160)
    print(overview.to_string(index=False))
    print("\n", balance.to_string())
    if not overlap.empty:
        print("\nShare in train by class:\n", overlap.round(3).to_string())
    print(f"\nTexts with conflicting labels in train: {len(conflict_texts):,}")
    print(f"\nSaved: {config.RESULTS / 'audit_report.md'}")


if __name__ == "__main__":
    main()
