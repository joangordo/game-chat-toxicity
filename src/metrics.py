"""Metrics with uncertainty: macro-F1, bootstrap CIs, paired comparisons, calibration."""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar

from . import config


# --- F1 from confusion counts ---------------------------------------------------------
def _macro_f1_from_conf(conf: np.ndarray) -> np.ndarray:
    """conf: (..., K, K) with rows = true, cols = predicted.

    Classes absent from both truth and prediction are excluded, which matches
    scikit-learn's default macro-F1 (labels = union of y_true and y_pred).
    """
    tp = np.diagonal(conf, axis1=-2, axis2=-1)
    fp = conf.sum(axis=-2) - tp
    fn = conf.sum(axis=-1) - tp
    denom = 2 * tp + fp + fn
    present = denom > 0
    f1 = np.where(present, 2 * tp / np.where(present, denom, 1), 0.0)
    return f1.sum(axis=-1) / np.maximum(present.sum(axis=-1), 1)


def confusion(y_true: np.ndarray, y_pred: np.ndarray, k: int = config.NUM_LABELS) -> np.ndarray:
    return np.bincount(np.asarray(y_true) * k + np.asarray(y_pred), minlength=k * k).reshape(k, k)


def macro_f1(y_true, y_pred, k: int = config.NUM_LABELS) -> float:
    return float(_macro_f1_from_conf(confusion(y_true, y_pred, k)))


def per_class_f1(y_true, y_pred, k: int = config.NUM_LABELS) -> np.ndarray:
    conf = confusion(y_true, y_pred, k)
    tp = np.diag(conf)
    denom = 2 * tp + (conf.sum(0) - tp) + (conf.sum(1) - tp)
    return np.where(denom > 0, 2 * tp / np.where(denom > 0, denom, 1), np.nan)


# --- Bootstrap ----------------------------------------------------------------------------
def _boot_indices(n: int, n_boot: int, rng: np.random.Generator,
                  groups: np.ndarray | None = None):
    """Yield arrays of resampled row indices (iid rows, or whole groups/clusters)."""
    if groups is None:
        for _ in range(n_boot):
            yield rng.integers(0, n, n)
        return
    uniq, inv = np.unique(groups, return_inverse=True)
    members = [np.flatnonzero(inv == g) for g in range(len(uniq))]
    for _ in range(n_boot):
        pick = rng.integers(0, len(uniq), len(uniq))
        yield np.concatenate([members[g] for g in pick])


def bootstrap_macro_f1(y_true, y_pred, k: int = config.NUM_LABELS, n_boot: int = config.N_BOOT,
                       alpha: float = config.ALPHA, groups=None, seed: int = 0) -> dict:
    """Point estimate and percentile CI. With `groups`, resamples clusters
    (e.g. whole matches) instead of single messages."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    rng = np.random.default_rng(seed)
    stats = np.array([macro_f1(y_true[i], y_pred[i], k)
                      for i in _boot_indices(len(y_true), n_boot, rng, groups)])
    lo, hi = np.quantile(stats, [alpha / 2, 1 - alpha / 2])
    return {"macro_f1": macro_f1(y_true, y_pred, k), "ci_low": float(lo), "ci_high": float(hi),
            "boot_sd": float(stats.std(ddof=1))}


def paired_bootstrap(y_true, pred_a, pred_b, k: int = config.NUM_LABELS,
                     n_boot: int = config.N_BOOT, alpha: float = config.ALPHA, seed: int = 0) -> dict:
    """Macro-F1(A) − Macro-F1(B) on the same resampled messages.
    Two-sided p-value from the bootstrap distribution of the difference."""
    y_true, pred_a, pred_b = map(np.asarray, (y_true, pred_a, pred_b))
    rng = np.random.default_rng(seed)
    diffs = np.array([macro_f1(y_true[i], pred_a[i], k) - macro_f1(y_true[i], pred_b[i], k)
                      for i in _boot_indices(len(y_true), n_boot, rng)])
    lo, hi = np.quantile(diffs, [alpha / 2, 1 - alpha / 2])
    p = 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())
    return {"diff": macro_f1(y_true, pred_a, k) - macro_f1(y_true, pred_b, k),
            "ci_low": float(lo), "ci_high": float(hi), "p_value": float(min(p, 1.0))}


# --- Binary metrics -----------------------------------------------------------------------
def binary_metrics(y_true, score, threshold: float = 0.5) -> dict:
    from sklearn.metrics import average_precision_score, roc_auc_score
    y_true, score = np.asarray(y_true), np.asarray(score)
    pred = (score >= threshold).astype(int)
    tp = int(((pred == 1) & (y_true == 1)).sum())
    fp = int(((pred == 1) & (y_true == 0)).sum())
    fn = int(((pred == 0) & (y_true == 1)).sum())
    prec = tp / (tp + fp) if tp + fp else np.nan
    rec = tp / (tp + fn) if tp + fn else np.nan
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else np.nan
    return {"auroc": float(roc_auc_score(y_true, score)),
            "avg_precision": float(average_precision_score(y_true, score)),
            "precision": prec, "recall": rec, "f1": f1, "threshold": threshold}


def threshold_for_precision(y_true, score, target: float) -> float | None:
    """Lowest threshold whose precision is at least `target` (None if unreachable)."""
    from sklearn.metrics import precision_recall_curve
    prec, _, thr = precision_recall_curve(y_true, score)
    ok = np.flatnonzero(prec[:-1] >= target)
    return float(thr[ok[0]]) if len(ok) else None


def threshold_for_recall(y_true, score, target: float) -> float:
    """Highest threshold whose recall is at least `target`."""
    from sklearn.metrics import precision_recall_curve
    _, rec, thr = precision_recall_curve(y_true, score)
    ok = np.flatnonzero(rec[:-1] >= target)
    return float(thr[ok[-1]]) if len(ok) else float(thr[0])


# --- Calibration --------------------------------------------------------------------------
def expected_calibration_error(y_true, probs: np.ndarray, n_bins: int = 15) -> float:
    """ECE of the top-class confidence."""
    probs = np.asarray(probs)
    conf, pred = probs.max(1), probs.argmax(1)
    correct = (pred == np.asarray(y_true)).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(conf, bins[1:-1]), 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.any():
            ece += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(ece)


def apply_temperature(probs: np.ndarray, t: float) -> np.ndarray:
    logits = np.log(np.clip(probs, 1e-12, 1.0)) / t
    logits -= logits.max(1, keepdims=True)
    e = np.exp(logits)
    return e / e.sum(1, keepdims=True)


def fit_temperature(y_true, probs: np.ndarray) -> float:
    """Temperature that minimises negative log-likelihood (fit on validation only)."""
    y_true = np.asarray(y_true)

    def nll(t):
        p = apply_temperature(probs, t)
        return -np.log(np.clip(p[np.arange(len(y_true)), y_true], 1e-12, 1)).mean()

    return float(minimize_scalar(nll, bounds=(0.05, 10), method="bounded").x)
