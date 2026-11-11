"""Project-wide configuration. Experiment hyperparameters live in configs/*.json."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_GAMETOX = ROOT / "data" / "raw" / "gametox"
RAW_CONDA = ROOT / "data" / "raw" / "conda"
PROCESSED = ROOT / "data" / "processed"
RUNS = ROOT / "runs"
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"

LABELS = {0: "Non-toxic", 1: "Insults & Flaming", 2: "Other Offensive",
          3: "Hate & Harassment", 4: "Threats", 5: "Extremism"}
NUM_LABELS = len(LABELS)
SPLITS = ("train", "val", "test")

# CONDA utterance intents: E = explicit toxicity, I = implicit toxicity,
# A = action, O = other. Mapped to binary toxic / non-toxic for transfer.
CONDA_TOXIC_INTENTS = {"E", "I"}

SEEDS = [13, 42, 2027]
N_BOOT = 2000
ALPHA = 0.05
