# Architecture: NLP NER Classification & Deployment

## Overview

This project is structured in two parts: an **individual experimentation** phase that benchmarks multiple NLP pipelines for Named Entity Recognition, and a **group deployment** phase that productionises the best model as a Flask REST API with monitoring and CI/CD.

---

## Individual Component Architecture

### Data Pipeline

```
Hugging Face (surrey-nlp/PLOD-CW)
        │
        ▼
┌─────────────────────┐
│   Data Loading      │  pandas DataFrame
│   EDA & Viz         │  POS/NER distributions
│   (Visualization.   │  Heatmaps, histograms
│    ipynb)           │  Scatter + box plots
└────────┬────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────┐
│              Preprocessing                          │
│  Option A: Tokenization (built into TF-IDF)         │
│  Option B: Normalization (lowercase + strip punct)  │
└────────┬────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────┐
│              Vectorization                          │
│  Option A: TF-IDF Vectorization                     │
│  Option B: Count Vectorization                      │
└────────┬────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────┐
│              Classification                         │
│  Option A: SVM (linear kernel)                      │
│  Option B: RNN (Embedding → RNN → Dense)            │
│            Optimizers: Adam / SGD                   │
└────────┬────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────┐
│              Evaluation                             │
│  Accuracy, Precision, Recall, F-Score               │
│  Confusion matrices, Learning curves                │
└─────────────────────────────────────────────────────┘
```

### Experiment Design

| Experiment | Variable | Options Compared |
|------------|----------|-----------------|
| 1 | Preprocessing | Tokenization vs Normalization (both + TF-IDF + SVM) |
| 2 | Vectorization | TF-IDF vs Count Vectorization (both + SVM) |
| 3 | Model | SVM vs RNN (both + TF-IDF) |
| 4 | Optimizer | Adam vs SGD (both RNN + TF-IDF) |

### RNN Architecture (Experiments 3 & 4)

```
Input (TF-IDF vectorized, padded sequences)
        │
        ▼
Embedding Layer
        │
        ▼
Simple RNN Layer
        │
        ▼
Dense Layer (softmax, 4 classes)
        │
        ▼
Output: B-O | B-LF | I-LF | B-AC
```

Training: Adam / SGD optimizer, sparse categorical cross-entropy loss, early stopping to prevent overfitting.

---

## Group Deployment Architecture

```
Client (HTTP)
     │
     ▼ POST /predict
┌─────────────────────────────────┐
│         Flask App               │
│         app.py                  │
│  - Input validation             │
│  - TF-IDF transform             │
│  - Sequence reshape             │
│  - GRU model inference          │
│  - Label decode                 │
│  - Logging                      │
└──────────┬──────────────────────┘
           │
     ┌─────┴──────┐
     │            │
     ▼            ▼
best_model   tfidf_vectorizer.pkl
  .keras     label_encoder.pkl
```

### Inference Pipeline (per request)

```
1. Receive POST JSON: {"text": "input text"}
2. TF-IDF transform: tfidf_vectorizer.transform([text]).toarray()
3. Reshape: np.expand_dims(text_vectorized, axis=1)
4. Predict: model.predict(text_vectorized_reshaped)
5. Decode: label_encoder.inverse_transform(np.argmax(predictions))
6. Log: timestamp + input + prediction → model_predictions.log
7. Return: {"prediction": "B-O"}
```

### CI/CD Pipeline

```
Code Push (GitLab)
     │
     ▼
┌─────────────┐
│  BUILD      │  echo "Building..."
│  Stage      │  pip install dependencies
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  TEST       │  pip install pytest
│  Stage      │  pytest
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  DEPLOY     │  nohup python app.py &
│  Stage      │  sleep 5
│             │  curl POST /predict (smoke test)
└─────────────┘
```

### Monitoring

Predictions logged to `model_predictions.log`:

```
2024-05-23 22:42:34 Input: da, Prediction: I-LF
2024-05-23 22:46:55 Input: Your input text here, Prediction: B-O
2024-05-23 22:47:12 Input: da, Prediction: I-LF
```

Each entry includes: timestamp, input text, predicted NER label.

---

## Technology Stack

| Component | Technology |
|-----------|-----------|
| Dataset | Hugging Face `surrey-nlp/PLOD-CW` |
| Data manipulation | pandas |
| Visualisation | matplotlib, seaborn |
| Vectorization | scikit-learn TfidfVectorizer, CountVectorizer |
| Classical ML | scikit-learn SVM (linear kernel) |
| Deep Learning | TensorFlow / Keras (RNN, GRU) |
| Hyperparameter tuning | Grid Search (scikit-learn) |
| Deployment | Flask (Python) |
| Serialisation | joblib (vectorizer, label encoder), .keras (model) |
| Monitoring | Python logging module |
| CI/CD | GitLab CI (.gitlab-ci.yml) |
| Testing | pytest, requests |
| Performance testing | ApacheBench |

---

## Design Decisions

1. **SVM chosen over RNN for production** — identical accuracy (87.58%) but lower compute cost and simpler deployment
2. **TF-IDF over Count Vectorization** — marginal accuracy gain (87.58% vs 86.93%), better handling of term significance
3. **Flask over FastAPI/TF Serving** — simpler setup for moderate traffic, easier integration with sklearn/Keras
4. **Model loaded once at startup** — avoids reload overhead per request, stored in memory for fast inference
5. **Early stopping on RNN** — prevents overfitting, convergence monitored via validation loss
6. **Grid search for GRU** — exhaustive hyperparameter search ensures reproducible, optimal configuration
