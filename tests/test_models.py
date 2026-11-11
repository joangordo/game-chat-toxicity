import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn

from src.char_models import PAD, CharBiGRU, CharCNN, CharVocab
from src.data import add_novelty_flag, normalize_text
from src.training import (EncodedDataset, FocalLoss, class_weights, get_device, make_loader,
                          predict_proba, set_seed, train)


def test_vocab_roundtrip_and_unknowns():
    v = CharVocab(["gg wp", "gg ez", "noob"], min_freq=2, max_len=4)
    assert v.encode("gg wp!!")[:2] == [v.stoi["g"], v.stoi["g"]]
    assert len(v.encode("gg wp!!")) == 4 and 1 in v.encode("zzz")
    assert CharVocab.from_dict(v.to_dict()).stoi == v.stoi


@pytest.mark.parametrize("cls", [CharCNN, CharBiGRU])
def test_models_ignore_padding(cls):
    torch.manual_seed(0)
    m = cls(vocab_size=20, num_labels=6).eval()
    x = torch.tensor([[5, 6, 7, PAD, PAD]])
    mask = (x != PAD).long()
    short = m(x[:, :3], mask[:, :3])
    padded = m(x, mask)
    assert short.shape == (1, 6)
    assert torch.allclose(short, padded, atol=1e-5)


def test_focal_equals_ce_when_gamma_zero():
    logits, y = torch.randn(16, 6), torch.randint(0, 6, (16,))
    assert torch.isclose(FocalLoss(gamma=0.0)(logits, y), nn.CrossEntropyLoss()(logits, y), atol=1e-6)


def test_class_weights_favour_rare_classes():
    w = class_weights([0] * 90 + [1] * 9 + [2], power=0.5, k=3)
    assert w[2] > w[1] > w[0] and w.mean() == pytest.approx(1.0)


def test_training_loop_learns_a_simple_rule():
    """Label = 1 if the message contains the character 7, else 0."""
    set_seed(0)
    rng = np.random.default_rng(0)
    seqs = [list(rng.integers(2, 12, rng.integers(3, 10))) for _ in range(1500)]
    y = np.array([int(7 in s) for s in seqs])
    tr = make_loader(EncodedDataset(seqs[:1200], y[:1200]), PAD, 64, shuffle=True)
    va = make_loader(EncodedDataset(seqs[1200:]), PAD, 256, shuffle=False)
    model = CharCNN(vocab_size=12, num_labels=2, emb_dim=16, n_filters=16)
    res = train(model, tr, va, y[1200:], nn.CrossEntropyLoss(), 5e-3, 0.0, 8, 3, 0.1, 1.0,
                get_device(), amp="off", log=lambda *_: None)
    assert res["best_val_macro_f1"] > 0.95
    probs = predict_proba(model, va, get_device())
    assert probs.shape == (300, 2) and np.allclose(probs.sum(1), 1, atol=1e-5)


def test_normalize_and_novelty():
    assert normalize_text("  GG\tWP ") == "gg wp"
    splits = {"train": pd.DataFrame({"text": ["gg", "lol"]}),
              "val": pd.DataFrame({"text": ["GG ", "new"]}),
              "test": pd.DataFrame({"text": ["lol", "x"]})}
    out = add_novelty_flag(splits)
    assert out["val"]["in_train"].tolist() == [True, False]
    assert out["test"]["in_train"].tolist() == [True, False]


def test_toxic_index_picks_overall_toxicity():
    import importlib.util
    from pathlib import Path
    spec = importlib.util.spec_from_file_location(
        "s03", Path(__file__).resolve().parents[1] / "scripts" / "03_generic_toxicity_model.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    assert m.toxic_index({0: "severe_toxic", 1: "toxic", 2: "insult"}) == 1
    assert m.toxic_index({0: "non-toxic", 1: "toxicity"}) == 1
    assert m.toxic_index({0: "LABEL_0"}) == 0
