"""
11 - Summary figure for the README: what happens when the game changes.

Reads results/transfer_conda.csv (written by 07_transfer_conda.py), so it needs
no data or GPU. Writes a light and a dark version; GitHub shows the one that
matches the reader's theme.

Run from the project root:
    python scripts/11_readme_figure.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src import config

MODELS = {  # model id -> (label, trained on game chat?)
    "xlmr_base_focal": ("XLM-R fine-tuned (published model)", True),
    "char_bigru": ("Character BiGRU", True),
    "tfidf_lr": ("TF-IDF + logistic regression", True),
    "generic_toxic_bert": ("Generic toxic-bert", False),
    "generic_multilingual_toxic_xlm_roberta": ("Generic multilingual XLM-R", False),
}
THEMES = {
    "light": dict(bg="#ffffff", ink="#1f2328", ink2="#59636e", grid="#e1e0d9",
                  game="#2a78d6", generic="#eb6834"),
    "dark": dict(bg="#0d1117", ink="#f0f6fc", ink2="#9198a1", grid="#2c2c2a",
                 game="#3987e5", generic="#d95926"),
}


def spread(values: list[float], min_gap: float) -> list[float]:
    """Move label positions apart (keeping their order) so none are closer than min_gap."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    pos = [values[i] for i in order]
    for k in range(1, len(pos)):
        pos[k] = max(pos[k], pos[k - 1] + min_gap)
    shift = (sum(values[i] for i in order) - sum(pos)) / len(pos)  # re-centre the block
    out = [0.0] * len(values)
    for k, i in enumerate(order):
        out[i] = pos[k] + shift
    return out


def main() -> None:
    t = pd.read_csv(config.RESULTS / "transfer_conda.csv")
    m = t.groupby("model")[["gametox_test_auroc", "conda_auroc"]].mean().loc[list(MODELS)]

    for mode, th in THEMES.items():
        fig, ax = plt.subplots(figsize=(9, 5.2), dpi=160)
        fig.patch.set_facecolor(th["bg"])
        ax.set_facecolor(th["bg"])
        x0, x1 = 0, 1
        right = spread(m["conda_auroc"].tolist(), 0.017)
        for (mid, row), yr in zip(m.iterrows(), right):
            label, game = MODELS[mid]
            c = th["game"] if game else th["generic"]
            ax.plot([x0, x1], [row["gametox_test_auroc"], row["conda_auroc"]], color=c,
                    lw=2.2 if mid == "xlmr_base_focal" else 1.6, solid_capstyle="round")
            ax.scatter([x0, x1], [row["gametox_test_auroc"], row["conda_auroc"]], s=36, color=c,
                       edgecolor=th["bg"], linewidth=1.5, zorder=3)
            ax.text(x1 + 0.04, yr, f"{label}   {row['gametox_test_auroc']:.2f} \u2192 {row['conda_auroc']:.2f}",
                    va="center", ha="left",
                    fontsize=9.5, color=th["ink"],
                    fontweight="bold" if mid == "xlmr_base_focal" else "normal")

        ax.set_xlim(-0.3, 2.3)
        ax.set_ylim(0.73, 0.975)
        ax.set_xticks([x0, x1], ["World of Tanks\n(game-chat models trained here)",
                                 "Dota 2\n(a game no model saw)"], color=th["ink"], fontsize=10)
        ax.tick_params(axis="x", length=0, pad=8)
        ax.set_yticks([0.75, 0.80, 0.85, 0.90, 0.95])
        ax.tick_params(axis="y", colors=th["ink2"], labelsize=9, length=0)
        ax.set_ylabel("AUROC, toxic vs non-toxic", color=th["ink2"], fontsize=10)
        ax.grid(axis="y", color=th["grid"], linewidth=0.6)
        ax.set_axisbelow(True)
        for s in ax.spines.values():
            s.set_visible(False)
        fig.text(0.02, 0.965, "Trained on one game, tested on another", fontsize=13.5,
                 fontweight="bold", color=th["ink"], va="top")
        fig.text(0.02, 0.905, "Models trained on game chat lose most of their advantage over generic "
                 "toxicity models when the game changes", fontsize=10, color=th["ink2"], va="top")
        handles = [plt.Line2D([], [], color=th["game"], lw=2, marker="o", ms=5),
                   plt.Line2D([], [], color=th["generic"], lw=2, marker="o", ms=5)]
        leg = ax.legend(handles, ["trained on World of Tanks chat", "generic, never saw game chat"],
                        loc="lower left", frameon=False, fontsize=9, labelcolor=th["ink2"])
        fig.subplots_adjust(left=0.09, right=0.98, top=0.82, bottom=0.14)
        out = config.FIGURES / f"readme_transfer_{mode}.png"
        fig.savefig(out, facecolor=th["bg"])
        plt.close(fig)
        print(f"Saved {out}")


if __name__ == "__main__":
    main()
