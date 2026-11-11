# Data

Neither dataset is included in this repository. `scripts/00_download_data.py`
downloads them into `data/raw/`.

## GameTox (training and in-domain evaluation)

About 53,000 World of Tanks chat messages labelled into six classes
(0 Non-toxic, 1 Insults & Flaming, 2 Other Offensive, 3 Hate & Harassment,
4 Threats, 5 Extremism). Released by the organisers of the EEUCA 2026 shared task;
download links: https://github.com/therealthapa/eeuca-toxicity

Expected files (anywhere under `data/raw/gametox/`, subfolders are fine):
`train_index_text.csv`, `train_index_label.csv`, `val_index_text.csv`,
`val_index_label.csv`, `test_index_text.csv`, `test_index_label.csv`.

Naseem, U., Shiwakoti, S., Shah, S. B., Thapa, S., & Zhang, Q. (2025). GameTox: A
Comprehensive Dataset and Analysis for Enhanced Toxicity Detection in Online Gaming
Communities. NAACL 2025 (Short Papers), 440-447.

## CONDA (cross-game transfer)

About 45,000 Dota 2 chat messages; the labelled train and validation files are used.
Utterance intents E (explicit) and I (implicit) are mapped to toxic, A and O to non-toxic.
Source: https://github.com/usydnlp/CONDA

Weld, H., Huang, G., Lee, J., et al. (2021). CONDA: a CONtextual Dual-Annotated dataset
for in-game toxicity understanding and detection. Findings of ACL-IJCNLP 2021, 2406-2416.

## Content warning

Both datasets contain insults, slurs and hate speech. Examples shown in this
repository are masked.
