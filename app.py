import os, time
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import gradio as gr
from transformers import AutoTokenizer, AutoModel
from huggingface_hub import hf_hub_download
from captum.attr import LayerIntegratedGradients

# --- 1. CONFIG ---
M_NAME, MAX_LEN, ENCODER_DIM, HIDDEN_DIM, NUM_LABELS = "xlm-roberta-base", 128, 768, 256, 2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HF_REPO_ID = "RextonRZ/malay-english-toxic-detector"
REPORTS_URL = "https://raw.githubusercontent.com/RextonRZ/malay-english-toxic-detector/main/reports"

tokenizer = AutoTokenizer.from_pretrained(M_NAME)


class ClassificationHead(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=256, num_labels=2, dropout_rate=0.3):
        super().__init__()
        self.norm = nn.LayerNorm(input_dim)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_dim, num_labels)

    def forward(self, x):
        x = self.norm(x)
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        return self.fc2(x)


class ToxicClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(M_NAME)
        self.head = ClassificationHead(ENCODER_DIM, HIDDEN_DIM, NUM_LABELS, 0.3)

    def forward(self, ids, mask):
        return self.head(self.encoder(input_ids=ids, attention_mask=mask).last_hidden_state[:, 0, :])


# --- 2. LOAD WEIGHTS FROM HUGGING FACE (few-shot, deployed model) ---
model = ToxicClassifier().to(DEVICE)
try:
    ckpt_p = hf_hub_download(repo_id=HF_REPO_ID, filename="best_model.pt")
    checkpoint = torch.load(ckpt_p, map_location=DEVICE)
    state = checkpoint["model_state"] if isinstance(checkpoint, dict) and "model_state" in checkpoint else checkpoint
    model.load_state_dict(state)
    model.eval()
    print(f"Loaded weights from {HF_REPO_ID}")
except Exception as e:
    print(f"FALLBACK baseline mode (untrained!). Trace: {e}")


# --- 3. INFERENCE + INTEGRATED GRADIENTS ATTRIBUTION ---
def _forward_logits(inp_ids, attn_mask):
    return model(inp_ids, attn_mask)


lig = LayerIntegratedGradients(_forward_logits, model.encoder.embeddings.word_embeddings)


def _merge_subwords(tokens, scores):
    words, word_scores = [], []
    specials = {tokenizer.pad_token, tokenizer.bos_token, tokenizer.eos_token}
    for tok, sc in zip(tokens, scores):
        if tok in specials:
            continue
        if tok.startswith("\u2581") or not words:
            words.append(tok.replace("\u2581", ""))
            word_scores.append(float(sc))
        else:
            words[-1] += tok.replace("\u2581", "")
            word_scores[-1] += float(sc)
    return words, word_scores


def predict_toxic_sentiment(text, threshold):
    if not text or not text.strip():
        return "Empty input", "0.00%", {}, [], "Please enter a comment."
    t_start = time.time()
    try:
        enc = tokenizer(text, padding="max_length", truncation=True,
                        max_length=MAX_LEN, return_tensors="pt")
        input_ids = enc["input_ids"].to(DEVICE)
        attention_mask = enc["attention_mask"].to(DEVICE)

        with torch.no_grad():
            probs = torch.softmax(model(input_ids, attention_mask), dim=1)[0].cpu().numpy()
        non_toxic_conf, toxic_conf = float(probs[0]), float(probs[1])
        is_toxic = toxic_conf >= threshold

        baseline = torch.full_like(input_ids, tokenizer.pad_token_id)
        attributions = lig.attribute(
            inputs=input_ids, baselines=baseline,
            additional_forward_args=(attention_mask,), target=1, n_steps=32,
        )
        token_scores = attributions.sum(dim=-1).squeeze(0).detach().cpu().numpy()
        raw_tokens = tokenizer.convert_ids_to_tokens(enc["input_ids"][0])
        words, word_scores = _merge_subwords(raw_tokens, token_scores)

        highlights = []
        max_abs = max((abs(s) for s in word_scores), default=1.0) or 1.0
        for w, s in zip(words, word_scores):
            weight = (s / max_abs) if is_toxic else 0.0
            # darken green: push safe-leaning (negative) weights stronger toward -1
            if weight < 0:
                weight *= 1.6
            weight = float(np.clip(weight, -1.0, 1.0))
            highlights.append((w + " ", weight))

        ms = (time.time() - t_start) * 1000
        return ("Toxic" if is_toxic else "Not toxic",
                f"{(toxic_conf if is_toxic else non_toxic_conf)*100:.1f}%",
                {"Not toxic": non_toxic_conf, "Toxic": toxic_conf},
                highlights,
                f"Analyzed in {ms:.0f} ms · threshold {threshold:.2f}")
    except Exception as e:
        return "Error", "0.00%", {}, [], f"Something went wrong: {str(e)}"


# --- 4. GRADIO INTERFACE ---
my_gray = gr.themes.Color(
    c50="#f8f8f8",
    c100="#ebebeb",
    c200="#d9d9d9",
    c300="#c0c0c0",
    c400="#9a9a9a",
    c500="#6e6e6e",
    c600="#555555",
    c700="#3f3f3f",
    c800="#2a2a2a",
    c900="#1a1a1a",
    c950="#0f0f0f",
)

custom_theme = gr.themes.Soft(
    primary_hue=my_gray,
    secondary_hue=my_gray,
    neutral_hue=my_gray,
    font=[gr.themes.GoogleFont("IBM Plex Sans"), "sans-serif"],
)

with gr.Blocks(theme=custom_theme, title="Manglish Toxic Detector") as demo:
    gr.Markdown(
        "# Multilingual Toxic Comment Detector\n"
        "Detects toxic language in English, Malay, and code-mixed Manglish, "
        "using a fine-tuned XLM-RoBERTa model. Highlighted words show which terms "
        "influenced the prediction."
    )

    with gr.Tabs():
        with gr.TabItem("Classifier"):
            with gr.Row():
                with gr.Column(scale=3):
                    inp = gr.Textbox(lines=4, placeholder="Type a comment in English, Malay, or Manglish...", label="Comment")
                    sl = gr.Slider(0.1, 0.9, value=0.5, step=0.05, label="Detection threshold",
                                   info="Lower flags more comments (stricter); higher flags fewer (more lenient).")
                    with gr.Row():
                        clr = gr.Button("Clear", variant="secondary")
                        sub = gr.Button("Analyze", variant="primary")
                with gr.Column(scale=2):
                    v_out = gr.Textbox(label="Result")
                    c_out = gr.Textbox(label="Confidence")
                    g_out = gr.Label(label="Probability")

            output_hl = gr.HighlightedText(
                label="Influential words",
                combine_adjacent=False, show_legend=True,
                color_map={"toxic": "#c0392b", "safe": "#1e7d34"},  # red, dark green
            )
            logs = gr.Textbox(label="Details", interactive=False)

            gr.Markdown("#### Try an example")
            examples = {
                "English": [("Clean", "Let's keep the discussion respectful and on topic."),
                            ("Toxic", "Haha you fat nigger")],
                "Malay":   [("Clean", "Sila bincang dengan baik tanpa gaduh di sini."),
                            ("Toxic", "Awak ni menyusahkan orang betul!")],
                "Manglish":[("Clean", "We can discuss sikit-sikit and settle safely."),
                            ("Toxic", "Why you always act macam ni, so bodoh lah.")],
            }
            with gr.Row():
                for lang, sample_pair in examples.items():
                    with gr.Column():
                        gr.Markdown(f"**{lang}**")
                        for btn_name, text_val in sample_pair:
                            btn = gr.Button(btn_name, size="sm")
                            btn.click(lambda t=text_val: t, inputs=None, outputs=inp)

        with gr.TabItem("Model Performance"):
            gr.Markdown(
                "Per-language evaluation results, comparing the zero-shot model "
                "(trained on English + Malay only) against the few-shot model "
                "(with code-mixed data added to training)."
            )
            frames = []
            for model_name, sub_dir in [("Few-shot", "fewshot"), ("Zero-shot", "zeroshot")]:
                try:
                    d = pd.read_csv(f"{REPORTS_URL}/{sub_dir}/metrics_{sub_dir}.csv")
                    d.insert(0, "Model", model_name)
                    frames.append(d)
                except Exception:
                    pass
            metrics_df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame({"Status": ["No metrics available"]})
            gr.Dataframe(metrics_df)

    sub.click(predict_toxic_sentiment, [inp, sl], [v_out, c_out, g_out, output_hl, logs])
    clr.click(lambda: ("", 0.5, "", "", {"Not toxic": 0.0, "Toxic": 0.0}, [], ""),
              outputs=[inp, sl, v_out, c_out, g_out, output_hl, logs])

demo.launch()