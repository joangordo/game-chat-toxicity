---
title: Game Chat Toxicity
emoji: 🎮
colorFrom: indigo
colorTo: red
sdk: gradio
sdk_version: {sdk_version}
app_file: app.py
pinned: false
license: mit
models:
  - {model_repo}
---

# Game chat toxicity classifier

Interactive demo of the model trained in
[github.com/joangordo/game-chat-toxicity](https://github.com/joangordo/game-chat-toxicity):
XLM-RoBERTa base fine-tuned on World of Tanks chat (GameTox) to classify messages into six
toxicity classes, with calibrated probabilities and a two-threshold moderation policy.

Research demo only. See the repository for the evaluation, its uncertainty and its limitations.
