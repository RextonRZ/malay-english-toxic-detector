# Toxic Comment Detector for English, Malay and Manglish

A multilingual toxic comment classifier for English, Malay, and code-mixed English & Malay (Manglish) text using XLM-RoBERTa. The model is fine-tuned on monolingual English and Malay data combined with a small amount of code-mixed data, and evaluated separately on each language to measure cross-lingual performance.

---

## Problem Statement

Online platforms receive massive volumes of user-generated content daily, much of which contains toxic comments such as hate speech, threats, and offensive language. Manual moderation cannot keep pace with this scale. Malaysian social media is particularly challenging because users frequently mix English and Malay within a single comment (Manglish), making detection harder for monolingual models.

This project aims to build an intelligent text classification system that detects toxic comments across three input types:
- Pure English
- Pure Malay
- Code-mixed Manglish (English-Malay mixed within a sentence)

---

## Approach

The system uses **XLM-RoBERTa**, a multilingual transformer model pretrained on 100 languages including English and Malay. Because XLM-RoBERTa learns a shared multilingual representation space during pretraining, it generalizes across languages and handles code-mixed text well.

**Training data:** Monolingual English and Malay (from the Mendeley bilingual hate speech dataset), plus a small portion of the custom code-mixed set folded in for **few-shot fine-tuning**.

**Test data:** Three separate test sets evaluated independently:
- English test set
- Malay test set
- Code-mixed Manglish test set (held-out portion of the custom set, never seen during training)

This setup measures both in-distribution performance (English, Malay) and generalization to code-mixed input (Manglish). A small amount of code-mixed training data is used to make the model practically relevant to real Malaysian social media phrasing, while a disjoint held-out portion of the same set provides a fair test measure.

---

## Tech Stack

| Component | Tool |
|---|---|
| Model | XLM-RoBERTa-base + custom classification head |
| Framework | PyTorch + HuggingFace Transformers |
| Training | Custom PyTorch training loop (AdamW, linear warmup, early stopping) |
| Compute | Google Colab (GPU) |
| Model Hosting | Hugging Face Hub |
| Explainability | Integrated Gradients (Captum) |
| Frontend | Gradio |
| Data Source | Mendeley bilingual Malay-English hate speech dataset + custom code-mixed set |

---

## Datasets

### Primary Dataset
**Bilingual Malay-English Social Media Dataset for Binary Hate Speech Detection**
Published by Jun Chen Tan & Lee-Yeng Ong on Mendeley Data (v3, September 2025).
Source: https://data.mendeley.com/datasets/mgv2n2vcb9/3

Contains 26,985 bilingual posts curated from five original sources:
- HateXplain (English)
- HateM (Malay)
- Toxicity-Small (Malay)
- Snapshot-Twitter-2022 (Malay)
- Supervised-Twitter (Malay)

### Custom Code-Mixed Set
A manually curated set of 500 code-mixed Malay-English (Manglish) sentences collected from social media such as Reddit Malaysia, TikTok, and Facebook, and labelled by the team (274 toxic / 226 non-toxic). It is split into a training portion (folded into training/validation) and a disjoint held-out test portion used only for evaluation.

---

## Project Structure

```
malay-english-toxic-detector/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── notebooks/
│   └── malay_english_toxic_detector.ipynb         ← main pipeline notebook
│
├── src/
│   ├── predict.py                                 ← inference function
│   └── app.py                                     ← Gradio web interface
│
├── data/
│   ├── raw/
│   │   ├── bilingual_hatespeech_ms_en_v2.csv      ← original Mendeley data
│   │   └── Code-Mixed Test Set ... .xlsx          ← custom 500-row Manglish set
│   └── processed/
│       ├── train.csv                              ← combined en+ms+codemixed training data
│       ├── val.csv                                ← validation data (incl. codemixed)
│       ├── test.csv                               ← combined test data
│       ├── test_english.csv                       ← English-only test split
│       ├── test_malay.csv                         ← Malay-only test split
│       └── test_codemixed.csv                     ← held-out Manglish test set
│
└── reports/
    ├── figures/                                   ← confusion matrices, plots
    ├── metrics.csv                                ← per-language evaluation results
    └── error_analysis.xlsx                        ← per-sample error analysis
```

> **Note:** The fine-tuned model weights are hosted on the [Hugging Face Hub](https://huggingface.co/RextonRZ/malay-english-toxic-detector) rather than committed to this repository, as the checkpoint is ~1 GB.

---

## Pipeline Overview

The notebook covers the full pipeline in five parts:

**Part A: Dataset Preparation**
- Data loading from the Mendeley dataset
- Exploratory Data Analysis (language, label, source, placeholder tags, text length)
- Language-based filtering (English vs Malay subsets)
- Text cleaning (URLs, mentions, anonymization tags removed; emoticons converted to words)
- Dataset standardization to unified `text, label, language` format
- Stratified 70/15/15 train/validation/test split
- Integration of the custom code-mixed set (split 300 train / 75 val / 125 held-out test)

**Part B: Tokenization & Encoding**
- XLM-RoBERTa subword tokenizer setup
- Token length analysis and configuration (max_length=128)

**Part C: Model Training**
- Fine-tune XLM-RoBERTa-base + custom head on combined English + Malay + code-mixed data
- Custom training loop with AdamW, linear warmup, mixed precision, and early stopping
- Best checkpoint (highest validation F1) uploaded to the Hugging Face Hub

**Part D: Evaluation**
- Best model loaded directly from the Hugging Face Hub
- Per-language metrics (Accuracy, Precision, Recall, F1, MCC, AUC-ROC)
- Confusion matrices for English, Malay, and code-mixed test sets
- Per-sample error analysis exported to Excel

**Part E: Deployment**
- Gradio web interface (`src/app.py`) with live classification, confidence scores, and Integrated Gradients word-level attribution

---

## How to Run

### 1. Clone the repository

```bash
git clone https://github.com/RextonRZ/malay-english-toxic-detector.git
cd malay-english-toxic-detector
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the pipeline notebook

Open `notebooks/malay_english_toxic_detector.ipynb` in Google Colab (recommended for GPU access) or Jupyter.

The notebook runs the full pipeline end-to-end:
- Data preparation only needs to run **once** (the processed splits are already in `data/processed/`)
- Tokenization, Training, and Evaluation load directly from the processed files
- Evaluation loads the fine-tuned weights from the Hugging Face Hub, so it can run without retraining

### 4. Launch the Gradio demo

```bash
python src/app.py
```

This loads the model from the Hugging Face Hub and opens a web interface where you can input comments and receive toxic/non-toxic predictions with confidence scores and word-level attribution.

---

## Evaluation Methodology

The trained model is evaluated on separate test sets to measure performance across input types:

| Test Set | Language | Source | Purpose |
|---|---|---|---|
| `test_english.csv` | English | Mendeley test split | In-distribution English performance |
| `test_malay.csv` | Malay | Mendeley test split | In-distribution Malay performance |
| `test_codemixed.csv` | Manglish | Custom team-curated (held-out) | Code-mixed generalization |

Reported metrics: Accuracy, Precision, Recall, F1, MCC, AUC-ROC (per-language).

The gap between the English/Malay F1 and the code-mixed F1 indicates how much performance changes on mixed-language input. Comparing the model with and without code-mixed training data shows the effect of few-shot fine-tuning on Manglish performance.

---

## Limitations & Future Work

- The custom code-mixed set is relatively small (500 sentences) and manually curated by the team, with only ~300 sentences available for training and 125 for testing.
- On code-mixed input the model shows high precision but lower recall, meaning it under-detects some toxic Manglish content — reflecting the limited code-mixed training data.
- Future work could incorporate a larger code-mixed training set (manual annotation or synthetic generation) to further improve Manglish recall.

---

## License

This project is for academic purposes (WID3011 Group Assignment, Universiti Malaya).
Dataset usage follows the license of the Mendeley source dataset. Trained model weights and code-mixed set are released for research use only.

---

## Acknowledgments
- Mendeley Data — Jun Chen Tan & Lee-Yeng Ong for the bilingual hate speech dataset
- HuggingFace — for the XLM-RoBERTa model and Transformers library
- Captum — for the Integrated Gradients attribution implementation
