# Multilingual Toxic Comment Detector for English, Malay and Manglish

A multilingual toxic comment classifier for English, Malay, and code-mixed English & Malay text using XLM-RoBERTa. The model is fine-tuned on monolingual English and Malay data and evaluated on code-mixed Manglish to test cross-lingual generalization.

---

## Problem Statement

Online platforms receive massive volumes of user-generated content daily, much of which contains toxic comments such as hate speech, threats, and offensive language. Manual moderation cannot keep pace with this scale. Malaysian social media is particularly challenging because users frequently mix English and Malay within a single comment (Manglish), making detection harder for monolingual models.

This project aims to build an intelligent text classification system that detects toxic comments across three input types:
- Pure English
- Pure Malay
- Code-mixed Manglish (English-Malay mixed within a sentence)

---

## Approach

The system uses **XLM-RoBERTa**, a multilingual transformer model pretrained on 100 languages including English and Malay. Because XLM-RoBERTa learns a shared multilingual representation space during pretraining, it can handle code-mixed text even without being explicitly trained on it.

**Training data:** Monolingual English and Malay only (from the Mendeley bilingual hate speech dataset)

**Test data:** Three separate test sets evaluated independently:
- English test set
- Malay test set
- Code-mixed Manglish test set (custom-built by the team) for **zero-shot evaluation**

This setup tests whether XLM-RoBERTa's pretrained multilingual representations enable cross-lingual generalization to code-mixed input.

---

## Tech Stack

| Component | Tool |
|---|---|
| Model | XLM-RoBERTa-base |
| Framework | PyTorch + HuggingFace Transformers |
| Training | HuggingFace Trainer API |
| Compute | Google Colab (GPU) |
| Frontend | Gradio |
| Data Source | Mendeley bilingual Malay-English hate speech dataset |

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

### Custom Code-Mixed Test Set
A manually curated test set of code-mixed Malay-English (Manglish) sentences collected from social media like Reddit Malaysia, Tiktok & Facebook and labeled by the team. Used **only for zero-shot evaluation** — never seen during training.

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
│   │   └── bilingual_hatespeech_ms_en_v2.csv      ← original Mendeley data
│   └── processed/
│       ├── train.csv                              ← combined en+ms training data
│       ├── val.csv                                ← validation data
│       ├── test.csv                               ← combined test data
│       ├── test_english.csv                       ← English-only test split
│       ├── test_malay.csv                         ← Malay-only test split
│       └── test_codemixed.csv                     ← custom Manglish test set
│
├── models/
│   └── xlm_roberta_toxic/                         ← fine-tuned model checkpoints
│
└── reports/
    ├── figures/                                   ← confusion matrices, plots
    └── metrics.md                                 ← per-language evaluation results
```

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

**Part B: Tokenization & Encoding**
- XLM-RoBERTa subword tokenizer setup
- Token length configuration (max_length=128)

**Part C: Model Training**
- Fine-tune XLM-RoBERTa-base on combined English + Malay data
- Use HuggingFace Trainer with stratified validation

**Part D: Evaluation**
- Per-language metrics (Accuracy, Precision, Recall, F1)
- Confusion matrices for English, Malay, and Code-mixed test sets
- Zero-shot evaluation of code-mixed generalization

**Part E: Deployment**
- Gradio web interface (`src/app.py`) for live demo

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
- Subsequent sections (Tokenization, Training, Evaluation) load directly from the processed files

### 4. Launch the Gradio demo

After the model has been trained and saved to `models/xlm_roberta_toxic/`:

```bash
python src/app.py
```

This opens a local web interface where you can input comments and receive toxic/non-toxic predictions with confidence scores.

---

## Evaluation Methodology

The trained model is evaluated on three separate test sets to measure performance across input types:

| Test Set | Language | Source | Purpose |
|---|---|---|---|
| `test_english.csv` | English | Mendeley test split | In-distribution English performance |
| `test_malay.csv` | Malay | Mendeley test split | In-distribution Malay performance |
| `test_codemixed.csv` | Manglish | Custom team-curated | **Zero-shot** code-mixed generalization |

Reported metrics: Accuracy, Precision, Recall, F1-score (per-language).

A drop in F1 on the code-mixed test set indicates how much performance is lost when the model encounters mixed-language input it was never trained on.

---

## Limitations & Future Work

- The custom code-mixed test set is relatively small (~500+ sentences) and manually curated by the team.
- Future work could incorporate code-mixed training data (manual annotation or synthetic generation) to improve Manglish performance beyond zero-shot baseline.

---

## License

This project is for academic purposes (WID3011 Group Assignment, Universiti Malaya).
Dataset usage follows the license of the Mendeley source dataset. Trained model weights and code-mixed test set are released for research use only.

---

## Acknowledgments
- Mendeley Data — Jun Chen Tan & Lee-Yeng Ong for the bilingual hate speech dataset
- HuggingFace — for the XLM-RoBERTa model and Transformers library
