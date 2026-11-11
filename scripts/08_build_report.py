"""
08 - Assemble results/REPORT.md from the audit, evaluation and transfer reports.

Run from the project root:
    python scripts/08_build_report.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config

SECTIONS = [("Data audit", "audit_report.md"), ("Evaluation", "eval_report.md"),
            ("Cross-game transfer", "transfer_report.md")]


def main() -> None:
    parts = ["# Toxicity detection in game chat — full report", ""]
    for i, (title, fname) in enumerate(SECTIONS, 1):
        p = config.RESULTS / fname
        if not p.exists():
            print(f"Skipping missing {fname}")
            continue
        body = "\n".join(("##" + l) if l.startswith("#") else l
                         for l in p.read_text(encoding="utf-8").splitlines())
        parts += [f"## {i}. {title}", "", body, ""]
    out = config.RESULTS / "REPORT.md"
    out.write_text("\n".join(parts), encoding="utf-8")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
