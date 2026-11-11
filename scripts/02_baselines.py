"""
02 - Baselines (Phase 1).

  * Majority class: the floor every model has to beat.
  * TF-IDF (character 2-5-grams + word 1-2-grams) + logistic regression with
    balanced class weights. Character n-grams are robust to the misspellings,
    leetspeak and mixed languages of game chat. C is chosen on validation.

Run from the project root:
    python scripts/02_baselines.py
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from src import config
from src.data import load_processed
from src.metrics import macro_f1
from src.runs import environment, predict_and_save_all, run_dir, save_json, save_predictions

C_GRID = [1.0, 4.0, 16.0, 64.0]


class TfidfFeatures:
    def __init__(self):
        self.char = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2,
                                    sublinear_tf=True, max_features=300_000)
        self.word = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2, sublinear_tf=True,
                                    token_pattern=r"(?u)\b\w+\b", max_features=100_000)

    def fit_transform(self, texts):
        return hstack([self.char.fit_transform(texts), self.word.fit_transform(texts)]).tocsr()

    def transform(self, texts):
        return hstack([self.char.transform(texts), self.word.transform(texts)]).tocsr()


class TfidfModel:
    """Features + classifier, saved together so later scripts can reuse them."""

    def __init__(self, feats, clf):
        self.feats, self.clf = feats, clf

    def predict_proba(self, texts):
        return self.clf.predict_proba(self.feats.transform(list(texts)))


def main() -> None:
    d = load_processed()
    tr, va, te = d["train"], d["val"], d["test"]
    y_tr, y_va = tr["label"].astype(int).to_numpy(), va["label"].astype(int).to_numpy()
    has_test_labels = te["label"].notna().all()

    # --- Majority class ------------------------------------------------------------------
    maj = int(np.bincount(y_tr).argmax())
    rd = run_dir("majority")
    def majority_proba(texts):
        probs = np.zeros((len(texts), config.NUM_LABELS))
        probs[:, maj] = 1.0
        return probs

    predict_and_save_all(rd, majority_proba)
    save_json(rd / "run_config.json", {"name": "majority", "family": "baseline", "type": "multiclass",
                                       "seed": None, "env": environment()})
    print(f"Majority: val macro-F1 = {macro_f1(y_va, np.full(len(y_va), maj)):.4f}")

    # --- TF-IDF + logistic regression -----------------------------------------------------
    feats = TfidfFeatures()
    X_tr = feats.fit_transform(tr["text"].tolist())
    X_va = feats.transform(va["text"].tolist())
    print(f"TF-IDF features: {X_tr.shape[1]:,}")
    search = []
    best = (-1, None, None)
    for C in C_GRID:
        t0 = time.time()
        clf = LogisticRegression(C=C, class_weight="balanced", max_iter=3000, random_state=0)
        clf.fit(X_tr, y_tr)
        f1 = macro_f1(y_va, clf.predict(X_va))
        search.append({"C": C, "val_macro_f1": f1, "seconds": round(time.time() - t0, 1)})
        print(f"  C={C:<4} val macro-F1 = {f1:.4f}  ({time.time() - t0:.0f}s)")
        if f1 > best[0]:
            best = (f1, C, clf)

    model = TfidfModel(feats, best[2])
    rd = run_dir("tfidf_lr")
    joblib.dump(model, rd / "model.joblib")
    predict_and_save_all(rd, model.predict_proba)
    save_json(rd / "run_config.json", {"name": "tfidf_lr", "family": "baseline", "type": "multiclass",
                                       "seed": 0, "C": best[1], "search": search,
                                       "best_val_macro_f1": best[0], "env": environment()})
    msg = f"TF-IDF + LR: best C = {best[1]}, val macro-F1 = {best[0]:.4f}"
    if has_test_labels:
        y_te = te["label"].astype(int).to_numpy()
        msg += f", test macro-F1 = {macro_f1(y_te, model.predict_proba(te['text']).argmax(1)):.4f}"
    print(msg)


if __name__ == "__main__":
    main()
