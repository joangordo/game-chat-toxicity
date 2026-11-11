import numpy as np
import pytest
from sklearn.metrics import f1_score

from src.metrics import (apply_temperature, bootstrap_macro_f1, expected_calibration_error, fit_temperature,
                         macro_f1, paired_bootstrap, per_class_f1, threshold_for_precision)


@pytest.mark.parametrize("seed", range(5))
def test_macro_f1_matches_sklearn(seed):
    rng = np.random.default_rng(seed)
    y = rng.choice(6, 500, p=[.8, .12, .05, .02, .007, .003])
    p = np.where(rng.random(500) < 0.7, y, rng.integers(0, 6, 500))
    assert macro_f1(y, p) == pytest.approx(f1_score(y, p, average="macro"))


def test_per_class_nan_for_absent_class():
    f = per_class_f1(np.array([0, 0, 1]), np.array([0, 1, 1]))
    assert np.isnan(f[5]) and f[0] == pytest.approx(2 / 3)


def test_bootstrap_ci_contains_point_and_shrinks_with_n():
    rng = np.random.default_rng(1)
    def run(n):
        y = rng.integers(0, 3, n)
        p = np.where(rng.random(n) < 0.7, y, rng.integers(0, 3, n))
        r = bootstrap_macro_f1(y, p, k=3, n_boot=400)
        return r, r["ci_high"] - r["ci_low"]
    (r1, w1), (_, w2) = run(300), run(5000)
    assert r1["ci_low"] <= r1["macro_f1"] <= r1["ci_high"] and w2 < w1


def test_cluster_bootstrap_is_wider_with_correlated_clusters():
    rng = np.random.default_rng(2)
    groups = np.repeat(np.arange(100), 30)
    quality = rng.random(100) < 0.5            # whole clusters easy or hard
    y = rng.integers(0, 2, len(groups))
    correct = np.where(quality[groups], rng.random(len(groups)) < 0.95, rng.random(len(groups)) < 0.55)
    p = np.where(correct, y, 1 - y)
    iid = bootstrap_macro_f1(y, p, k=2, n_boot=300)
    clu = bootstrap_macro_f1(y, p, k=2, n_boot=300, groups=groups)
    assert (clu["ci_high"] - clu["ci_low"]) > 1.5 * (iid["ci_high"] - iid["ci_low"])


def test_paired_bootstrap_detects_better_model_and_not_identical_one():
    rng = np.random.default_rng(3)
    y = rng.integers(0, 3, 3000)
    good = np.where(rng.random(3000) < 0.8, y, rng.integers(0, 3, 3000))
    bad = np.where(rng.random(3000) < 0.6, y, rng.integers(0, 3, 3000))
    assert paired_bootstrap(y, good, bad, k=3, n_boot=300)["p_value"] < 0.01
    same = paired_bootstrap(y, good, good, k=3, n_boot=100)
    assert same["diff"] == 0 and same["ci_low"] == 0


def test_temperature_scaling_fixes_overconfidence():
    rng = np.random.default_rng(4)
    n, k = 4000, 3
    logits = rng.normal(size=(n, k)) * 3
    true_p = np.exp(logits) / np.exp(logits).sum(1, keepdims=True)
    y = np.array([rng.choice(k, p=pp) for pp in true_p])
    over = apply_temperature(true_p, 0.4)       # sharpen = overconfident
    t = fit_temperature(y, over)
    assert t == pytest.approx(1 / 0.4, rel=0.15)
    assert expected_calibration_error(y, apply_temperature(over, t)) < expected_calibration_error(y, over)


def test_threshold_for_precision():
    y = np.array([0, 0, 1, 1, 1])
    s = np.array([0.1, 0.6, 0.5, 0.8, 0.9])
    t = threshold_for_precision(y, s, 1.0)
    assert t == pytest.approx(0.8)
    assert threshold_for_precision(np.array([0, 1]), np.array([0.9, 0.1]), 1.0) is None  # unreachable
