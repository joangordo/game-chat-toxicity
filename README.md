# Toxicity detection in game chat

**Does a general-purpose toxicity model work on game chat? And does a model trained
on one game work on another?**

Six-class toxicity detection on World of Tanks chat (GameTox), from a TF-IDF baseline
to character-level networks written from scratch in PyTorch and fine-tuned multilingual
transformers trained with a custom loop. Evaluation focuses on uncertainty: bootstrap
confidence intervals, seed variability, paired model comparisons, calibration and a
moderation policy. Models are then applied, without retraining, to Dota 2 chat (CONDA).

> Results: _to be completed after the experiments._

> Content warning: the data contain insults and hate speech. Examples are masked.

## Approach

| Phase | Script | What it does |
|---|---|---|
| 0. Data audit | `01_audit_data.py` | Duplicates, label conflicts, train/test text overlap, languages |
| 1. Baselines | `02_baselines.py`, `03_generic_toxicity_model.py` | Majority class, TF-IDF + logistic regression, off-the-shelf toxicity model |
| 2. PyTorch from scratch | `04_train_char_model.py` | Character CNN and BiGRU, own training loop |
| 3. Transformers | `05_train_transformer.py` | XLM-RoBERTa and mDeBERTa fine-tuned with the same loop |
| 4. Evaluation | `06_evaluate.py` | Bootstrap CIs, novel-message subset, seeds, paired tests, calibration, thresholds |
| 5. Transfer | `07_transfer_conda.py` | World of Tanks → Dota 2, cluster bootstrap by match |

## How to run

1. Install PyTorch for your GPU from [pytorch.org](https://pytorch.org), then
   `pip install -r requirements.txt`.
2. `python -m pytest -q` (unit tests, no data or GPU needed).
3. `python run_experiments.py --phase all`, or phase by phase:
   `data`, `baselines`, `char`, `transformers`, `evaluate`.
   Existing runs are skipped, so the pipeline can be stopped and resumed.

Each run writes its configuration, environment, training history and predictions to
`runs/<name>/`. The final report is `results/REPORT.md`.

Hardware used: NVIDIA RTX 5060 Laptop (8 GB) for transformers, bf16 mixed precision.
On a 4 GB GPU: `--batch-size 8 --grad-accum 4 --amp fp16`.

## Repository structure

```
configs/   hyperparameters of each model (JSON)
scripts/   one script per step, numbered in execution order
src/       data loading, models, training loop, metrics
tests/     unit tests
runs/      one folder per trained model (predictions and configs; weights not versioned)
results/   reports and figures
```

## Data

See [`data/README.md`](data/README.md). GameTox (Naseem et al., NAACL 2025) and
CONDA (Weld et al., Findings of ACL 2021) are not redistributed here.

## License

Code: MIT. The datasets keep their original licences and terms of use.
