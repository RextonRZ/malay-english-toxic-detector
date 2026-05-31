# Toxic Comment Detector for English, Malay and Manglish

A multilingual toxic comment classifier for English, Malay, and code-mixed English & Malay (Manglish) text using XLM-RoBERTa. The model is fine-tuned on monolingual English and Malay data, and the project compares a **zero-shot** configuration (monolingual training only) against a **few-shot** configuration (a small amount of code-mixed data folded into training) to measure the effect of code-mixed fine-tuning on Manglish detection.

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

To measure how much code-mixed training data helps, two models are trained and compared on the **same** held-out test sets:

- **Zero-shot:** fine-tuned on monolingual English + Malay only; Manglish is never seen during training.
- **Few-shot:** fine-tuned on English + Malay plus a small portion of code-mixed data folded into training and validation.

Both are evaluated on an identical held-out code-mixed test set, so the difference in Manglish performance isolates the effect of the code-mixed training data.

---

## Tech Stack

| Component | Tool |
|---|---|
| Model | XLM-RoBERTa-base + custom classification head |
| Framework | PyTorch + HuggingFace Transformers |
| Training | Custom PyTorch training loop (AdamW, linear warmup, mixed precision, early stopping) |
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
A manually curated set of 500 code-mixed Malay-English (Manglish) sentences collected from social media such as Reddit Malaysia, TikTok, and Facebook, and labelled by the team (274 toxic / 226 non-toxic). It is split 300 train / 75 validation / 125 held-out test. The training and validation portions are used only in the few-shot configuration; the 125-sentence held-out test is identical across both configurations.

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
│   ├── processed/                                 ← FEW-SHOT splits (code-mixed folded in)
│   │   ├── train.csv
│   │   ├── val.csv
│   │   ├── test.csv
│   │   ├── test_english.csv
│   │   ├── test_malay.csv
│   │   └── test_codemixed.csv
│   └── processed_zeroshot/                         ← ZERO-SHOT splits (monolingual train/val)
│       ├── train.csv
│       ├── val.csv
│       ├── test.csv
│       ├── test_english.csv
│       ├── test_malay.csv
│       └── test_codemixed.csv                      ← identical held-out test as few-shot
│
└── reports/
    ├── fewshot/                                    ← few-shot results
    │   ├── metrics_fewshot.csv
    │   ├── cm_combined.png / cm_english.png / cm_malay.png / cm_manglish.png
    │   └── error_analysis_fewshot.xlsx
    └── zeroshot/                                   ← zero-shot results
        ├── metrics_zeroshot.csv
        ├── cm_combined.png / cm_english.png / cm_malay.png / cm_manglish.png
        └── error_analysis_zeroshot.xlsx
```

> **Note:** The fine-tuned model weights are hosted on the [Hugging Face Hub](https://huggingface.co/RextonRZ/malay-english-toxic-detector) rather than committed to this repository, as each checkpoint is ~1 GB. Two checkpoints are stored: `best_model.pt` (few-shot, used by the deployed demo) and `best_model_zeroshot.pt` (zero-shot, for comparison).

---

## Pipeline Overview

The notebook covers the full pipeline in five parts. An `EXPERIMENT` flag (`"fewshot"` or `"zeroshot"`) selects which dataset version to train and evaluate, so the same pipeline runs both configurations.

**Part A: Dataset Preparation**
- Data loading from the Mendeley dataset
- Exploratory Data Analysis (language, label, source, placeholder tags, text length)
- Language-based filtering (English vs Malay subsets)
- Text cleaning (URLs, mentions, anonymization tags removed; emoticons converted to words)
- Dataset standardization to unified `text, label, language` format
- Stratified 70/15/15 train/validation/test split
- Code-mixed integration: saves both a few-shot version (`data/processed/`) and a zero-shot version (`data/processed_zeroshot/`)

**Part B: Tokenization & Encoding**
- XLM-RoBERTa subword tokenizer setup
- Token length analysis and configuration (max_length=128)

**Part C: Model Training**
- Fine-tune XLM-RoBERTa-base + custom head on the selected dataset version
- Custom training loop with AdamW, linear warmup, mixed precision, and early stopping
- Best checkpoint (highest validation F1) uploaded to the Hugging Face Hub

**Part D: Evaluation**
- Best model loaded directly from the Hugging Face Hub
- Per-language metrics (Accuracy, Precision, Recall, F1, MCC, AUC-ROC)
- Confusion matrices for English, Malay, and code-mixed test sets
- Per-sample error analysis exported to Excel
- Results saved per experiment under `reports/<experiment>/`

**Part E: Deployment**
- Gradio web interface (`src/app.py`) with live classification, confidence scores, and Integrated Gradients word-level attribution
- Always loads the few-shot model (`best_model.pt`) as the deployed version

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

- Data preparation in Part A only needs to run once; the processed splits for both configurations are already in `data/processed/` and `data/processed_zeroshot/`.
- Set the `EXPERIMENT` flag to `"fewshot"` or `"zeroshot"` to choose which configuration to train and evaluate.
- Evaluation loads the fine-tuned weights from the Hugging Face Hub, so it can run without retraining.

### 4. Launch the Gradio demo

```bash
python src/app.py
```

This loads the few-shot model from the Hugging Face Hub and opens a web interface where you can input comments and receive toxic/non-toxic predictions with confidence scores and word-level attribution.

---

## Evaluation Methodology

Each trained model is evaluated on separate test sets to measure performance across input types:

| Test Set | Language | Source | Purpose |
|---|---|---|---|
| `test_english.csv` | English | Mendeley test split | In-distribution English performance |
| `test_malay.csv` | Malay | Mendeley test split | In-distribution Malay performance |
| `test_codemixed.csv` | Manglish | Custom team-curated (held-out) | Code-mixed generalization |

Reported metrics: Accuracy, Precision, Recall, F1, MCC, AUC-ROC (per-language).

The gap between the English/Malay F1 and the code-mixed F1 indicates how much performance changes on mixed-language input. Comparing the zero-shot and few-shot models on the identical code-mixed test set shows the effect of code-mixed fine-tuning on Manglish detection.

---

## Limitations & Future Work

- The custom code-mixed set is relatively small (500 sentences), with only ~300 sentences available for training and 125 for testing.
- On code-mixed input the model shows high precision but lower recall, meaning it under-detects some toxic Manglish content — reflecting the limited code-mixed training data.
- Future work could incorporate a larger code-mixed training set (manual annotation or synthetic generation) to further improve Manglish recall.

---

## License

This project is for academic purposes (WID3011 Group Assignment, Universiti Malaya).
Dataset usage follows the license of the Mendeley source dataset. Trained model weights and code-mixed set are released for research use only.

---

## Acknowledgments
- Mendeley Data — Jun Chen Tan & Lee-Yeng Ong for the bilingual hate speech dataset
- Custom Code-Mixed Dataset — manually collected and labelled by our team from Reddit Malaysia, TikTok, and Facebook
- HuggingFace — for the XLM-RoBERTa model and Transformers library
- Captum — for the Integrated Gradients attribution implementation
