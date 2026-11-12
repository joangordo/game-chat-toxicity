import importlib
import json
import sys
from pathlib import Path

import pytest

from src.masking import is_offensive, mask


def test_mask_hides_offensive_words_and_keeps_game_terms():
    lex = {"assmonkey"}
    out = mask("cry more b1tch, Pz kill artys, fuuuuck you assmonkey", lex)
    assert "b1tch" not in out and "fuuuuck" not in out and "assmonkey" not in out
    assert "kill artys" in out and out.startswith("cry more b")


def test_mask_other_languages():
    assert not is_offensive("gg", set())
    assert mask("w dupe ty cwelu", set()) == "w d*** ty c****"


def test_export_runs_are_excluded(tmp_path, monkeypatch):
    from src import config, evaluation
    monkeypatch.setattr(config, "RUNS", tmp_path)
    for name in ("a_s13", "a_s13_export", "a_s13_smoke"):
        (tmp_path / name).mkdir()
        (tmp_path / name / "run_config.json").write_text(json.dumps({"name": "a", "loss": "ce"}))
    names = [r["run"] for r in evaluation.collect_runs(include_smoke=True)]
    assert names == ["a_s13", "a_s13_smoke"]


def test_cli_names_export_runs(monkeypatch, tmp_path):
    cfg = tmp_path / "c.json"
    cfg.write_text(json.dumps({"name": "xlmr_base", "loss": "ce"}))
    from src.cli import parse_train_args
    monkeypatch.setattr(sys, "argv", ["x", "--config", str(cfg), "--loss", "focal", "--save-model"])
    c, _ = parse_train_args("t")
    assert c["run_name"] == "xlmr_base_focal_s13_export"


@pytest.mark.skipif(importlib.util.find_spec("gradio") is None, reason="gradio not installed")
def test_demo_classify_with_tiny_model(tmp_path, monkeypatch):
    from transformers import BertConfig, BertForSequenceClassification, BertTokenizerFast
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"] + list("abcdefghijklmnopqrstuvwxyz")
    (tmp_path / "vocab.txt").write_text("\n".join(vocab))
    BertTokenizerFast(vocab_file=str(tmp_path / "vocab.txt")).save_pretrained(tmp_path)
    BertForSequenceClassification(BertConfig(vocab_size=len(vocab), hidden_size=16, num_hidden_layers=1,
                                             num_attention_heads=2, intermediate_size=32,
                                             num_labels=6)).save_pretrained(tmp_path)
    (tmp_path / "demo_config.json").write_text(json.dumps({
        "labels": {str(i): f"c{i}" for i in range(6)}, "temperature": 1.0, "max_tokens": 16,
        "threshold_auto": 0.99, "threshold_review": 0.5}))
    monkeypatch.setenv("MODEL_ID", str(tmp_path))
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "demo"))
    app = importlib.import_module("app")
    probs, summary = app.classify("gg wp")
    assert len(probs) == 6 and abs(sum(probs.values()) - 1) < 1e-5 and "P(toxic)" in summary
    assert app.classify("  ") == ({}, "Write a chat message above.")
