r"""
Game chat toxicity demo (Gradio).

Locally:   set MODEL_ID=..\runs\xlmr_base_focal_s13_export\model  and  python app.py
On Spaces: MODEL_ID defaults to the model published on the Hugging Face Hub.
"""
import json
import os
from pathlib import Path

import gradio as gr
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_ID = os.environ.get("MODEL_ID", "joangordo/game-chat-toxicity-xlmr")


def load_demo_config(model_id: str) -> dict:
    local = Path(model_id) / "demo_config.json"
    if local.exists():
        return json.loads(local.read_text(encoding="utf-8"))
    from huggingface_hub import hf_hub_download
    return json.loads(Path(hf_hub_download(model_id, "demo_config.json")).read_text(encoding="utf-8"))


CFG = load_demo_config(MODEL_ID)
TOKENIZER = AutoTokenizer.from_pretrained(MODEL_ID)
MODEL = AutoModelForSequenceClassification.from_pretrained(MODEL_ID).eval()
LABELS = [CFG["labels"][str(i)] for i in range(len(CFG["labels"]))]
TEMPERATURE = CFG["temperature"]


@torch.no_grad()
def classify(text: str):
    text = (text or "").strip()
    if not text:
        return {}, "Write a chat message above."
    enc = TOKENIZER(text, truncation=True, max_length=CFG["max_tokens"], return_tensors="pt")
    probs = torch.softmax(MODEL(**enc).logits[0] / TEMPERATURE, dim=-1).tolist()
    p_toxic = 1 - probs[0]
    t_auto, t_review = CFG.get("threshold_auto"), CFG["threshold_review"]
    if t_auto is not None and p_toxic >= t_auto:
        band = "🔴 **Automatic action** (very high confidence)"
    elif p_toxic >= t_review:
        band = "🟠 **Send to human review**"
    else:
        band = "🟢 **No action**"
    summary = f"P(toxic) = **{p_toxic:.1%}** → {band}"
    return {lab: float(p) for lab, p in zip(LABELS, probs)}, summary


EXAMPLES = [
    "gg wp everyone",
    "nice shot, well played",
    "report our arty, afk the whole game",
    "noob team as always, uninstall the game",
    "you are the worst player i have ever seen",
    "i will find you after the match",
]

DESCRIPTION = f"""
# Game chat toxicity classifier
XLM-RoBERTa base fine-tuned on **World of Tanks** chat (GameTox, 6 classes).
Probabilities are calibrated (temperature {TEMPERATURE:.2f}); the review threshold catches ~90% of toxic
messages on validation data.

**Research demo, not a moderation tool.** It misses most implicit toxicity (sarcasm, indirect insults),
is weaker outside English, and rare classes (Threats, Extremism) were learned from 24–60 examples.
Inputs are not stored. Code and evaluation: [github.com/joangordo/game-chat-toxicity](https://github.com/joangordo/game-chat-toxicity)
"""

with gr.Blocks(title="Game chat toxicity") as demo:
    gr.Markdown(DESCRIPTION)
    with gr.Row():
        with gr.Column():
            inp = gr.Textbox(label="Chat message", placeholder="Type a game chat message...", lines=2)
            btn = gr.Button("Classify", variant="primary")
            gr.Examples(EXAMPLES, inputs=inp)
        with gr.Column():
            summary = gr.Markdown()
            out = gr.Label(label="Class probabilities", num_top_classes=6)
    btn.click(classify, inputs=inp, outputs=[out, summary])
    inp.submit(classify, inputs=inp, outputs=[out, summary])

if __name__ == "__main__":
    demo.launch()
