# NLP - Named Entity Recognition & POS Tagging

> MSc Data Science Project | University of Surrey | COMM061 Natural Language Processing

## Overview

This repository contains both the **individual experimentation** and **group deployment** components of an NLP coursework project focused on **Named Entity Recognition (NER)** and **Part-of-Speech (POS) tagging** using the `surrey-nlp/PLOD-CW` dataset from Hugging Face.

The individual component benchmarks 4 experimental pipelines (SVM, RNN, TF-IDF, Count Vectorization) across preprocessing strategies and optimizers. The group component deploys the best-performing model (GRU + TF-IDF) as a Flask web service with monitoring and a CI/CD pipeline.

---

## Repository Structure

```
nlp-ner-classification/
├── src/
│   ├── individual/
│   │   ├── Visualization.ipynb          # EDA - POS/NER distributions, heatmaps
│   │   ├── Experiment_1.ipynb           # Tokenization+TF-IDF+SVM vs Normalization+TF-IDF+SVM
│   │   ├── Experiment_2.ipynb           # TF-IDF+SVM vs Count Vectorization+SVM
│   │   ├── Experiment_3.ipynb           # SVM+TF-IDF vs RNN+TF-IDF
│   │   └── Experiment_4.ipynb           # RNN Adam vs RNN SGD optimizer comparison
│   └── group/
│       ├── app.py                       # Flask deployment endpoint
│       ├── testing.ipynb                # Endpoint testing notebook
│       └── data.json                    # Sample test payload
├── docs/
│   ├── Individual_Report.pdf
│   └── Group_Report.pdf
├── logs/
│   └── model_predictions.log           # Inference monitoring log
├── requirements.txt
├── ARCHITECTURE.md
└── README.md
```

---

## Individual Component - Experimental Results

### Dataset: `surrey-nlp/PLOD-CW`

| Feature | Detail |
|---------|--------|
| Source | Hugging Face (`surrey-nlp/PLOD-CW`) |
| Task | NER tagging + POS tagging |
| Tags | B-O, B-LF, I-LF, B-AC |
| Format | Tokens, POS tags, NER tags per entry |

### Experiment Results

| Experiment | Pipeline | Accuracy | Precision | Recall | F-Score |
|------------|----------|----------|-----------|--------|---------|
| 1 | Tokenization + TF-IDF + SVM | 83.66% | 76.70% | 83.66% | 77.72% |
| 1 | Normalization + TF-IDF + SVM | 84.31% | 80.27% | 84.31% | 78.16% |
| 2 | TF-IDF Vectorization + SVM | 87.58% | — | — | — |
| 2 | Count Vectorization + SVM | 86.93% | — | — | — |
| 3 | SVM + TF-IDF | 87.58% | — | — | — |
| 3 | RNN + TF-IDF | 87.58% | — | — | — |
| 4 | RNN + Adam optimizer | 87.58% | — | — | — |
| 4 | RNN + SGD optimizer | 87.58% | — | — | — |

### Key Findings

- **TF-IDF marginally outperforms Count Vectorization** for NER classification (87.58% vs 86.93%)
- **SVM and RNN achieve identical accuracy** - SVM preferred for lower compute cost
- **Adam and SGD optimizers produce equivalent results** - architecture/features are the limiting factor
- **Text normalization provides marginal gain** - SVM is robust to text variations without it
- All models excel on dominant `B-O` class but struggle with minority classes (`B-AC`, `B-LF`, `I-LF`)

---

### Model: GRU + TF-IDF Vectorization + Grid Search

Chosen for its effectiveness on sequential data, efficiency over LSTM, and grid search-optimised hyperparameters.

### Flask Web Service

**Endpoint:** `POST /predict`

**Request:**
```json
{ "text": "your input text here" }
```

**Response:**
```json
{ "prediction": "B-O" }
```

### Performance (ApacheBench)

| Metric | Value |
|--------|-------|
| Requests per second | 78.26 req/sec |
| Average latency | 127.78 ms |
| Total requests | 100 in 1.278 seconds |

### CI/CD Pipeline (GitLab CI)

Three-stage pipeline: **Build → Test → Deploy**

- Docker base image: Python 3.9
- Test stage: `pytest` automated tests
- Deploy stage: Flask app launched, endpoint verified via `curl`

---

## Setup & Usage

### Install dependencies
```bash
pip install -r requirements.txt
```

### Run Flask service locally
```bash
python src/group/app.py
# Runs on http://0.0.0.0:5001
```

### Test the endpoint
```bash
curl -X POST -H "Content-Type: application/json" \
  -d '{"text": "your text here"}' \
  http://localhost:5001/predict
```

Or run `src/group/testing.ipynb` in Jupyter.

### Run individual experiments
Open any notebook in `src/individual/` in Jupyter and run all cells.

---

## Academic Context

- **Institution:** University of Surrey
- **Programme:** MSc Data Science
- **Module:** COMM061 - Natural Language Processing
- **Dataset:** `surrey-nlp/PLOD-CW` (Hugging Face)
- **Individual Author:** Jeganathan Duraisamy (URN: 6835871)
