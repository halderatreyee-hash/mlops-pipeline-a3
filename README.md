#  MLOps Assignment 3 — End-to-End MLOps Pipeline
### IIT Jodhpur | PGD AI Program | Group Project 18

[![CI](https://github.com/halderatreyee-hash/mlops-pipeline-a3/actions/workflows/ci.yml/badge.svg)](https://github.com/halderatreyee-hash/mlops-pipeline-a3/actions/workflows/ci.yml)
[![Inference](https://github.com/halderatreyee-hash/mlops-pipeline-a3/actions/workflows/inference.yml/badge.svg)](https://github.com/halderatreyee-hash/mlops-pipeline-a3/actions/workflows/inference.yml)

---

##  Project Overview

This project demonstrates a **complete, production-grade End-to-End MLOps pipeline** for sentiment analysis on IMDB movie reviews. The pipeline covers every stage — from raw data ingestion and cleaning, model selection and fine-tuning, experiment tracking, containerized inference with Docker, to fully automated CI/CD workflows via GitHub Actions.

**Task:** Binary Sentiment Classification (Positive / Negative) on IMDB Movie Reviews  
**Base Model:** `distilbert-base-uncased` (HuggingFace)  
**Best Model Accuracy:** 91.70% (run-v2, learning rate = 5e-5)  
**Training Platform:** Kaggle GPU (T4 x2)  
**Experiment Tracking:** Weights & Biases (W&B)

---

##  Project Links

| Resource | Link |
|----------|------|
|  GitHub Repository | https://github.com/halderatreyee-hash/mlops-pipeline-a3 |
|  HuggingFace Model | https://huggingface.co/Atreyee-Halder/mlops-imdb-sentiment |
|  Docker Hub Image | https://hub.docker.com/r/g25ait2065/mlops-a3-inference |
|  W&B Dashboard | https://wandb.ai/g25ait2023-iit-jodhpur/mlops-assignment3 |
|  Kaggle Notebook V1 | https://www.kaggle.com/code/ahalderg25ait2023/mlops-group-project |
|  Kaggle Notebook V2 | https://www.kaggle.com/code/ahalderg25ait2023/mlops-group-project-v2 |

---

##  Team Members & Contributions

| Member | GitHub Username | Tasks Completed |
|--------|----------------|-----------------|
| **Member 1** — Atreyee Halder | `halderatreyee-hash` | Task 1 (Repo Setup), Task 2 (Data Prep), Task 3 (Model Load), Task 4 (Training + W&B), Task 8 (W&B Dashboard) |
| **Member 2** — Mahadev Vishwakarma | `g25ait2131` | Task 5 (HuggingFace Model Push), PR Review & Approval |
| **Member 3** — Md. Umar Sharief  | `g25ait2065` | Task 6 (Dockerfile + Inference Script + Docker Hub Push) |
| **Member 4** - Manmeet Singh | `g25ait2139` | Task 7 (GitHub Actions CI + Inference Workflows) |

---

##  End-to-End Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                        END-TO-END MLOps PIPELINE                                │
│                    IIT Jodhpur | PGD AI | Group Assignment 3                    │
└─────────────────────────────────────────────────────────────────────────────────┘

  STAGE 1             STAGE 2             STAGE 3             STAGE 4
┌───────────┐      ┌───────────┐      ┌───────────┐      ┌───────────┐
│   DATA    │      │  KAGGLE   │      │HUGGINGFACE│      │  DOCKER   │
│PREPARATION│─────▶│ TRAINING  │─────▶│    HUB    │─────▶│   IMAGE   │
│  (Task 2) │      │ (Task 4)  │      │  (Task 5) │      │  (Task 6) │
└───────────┘      └───────────┘      └───────────┘      └───────────┘
      │                  │                  │                   │
      ▼                  ▼                  ▼                   ▼
 IMDB Dataset       run-v1 (3e-5)    Atreyee-Halder/     g25ait2065/
 50K reviews        run-v2 (5e-5)    mlops-imdb-         mlops-a3-
 Cleaned &          W&B Tracked      sentiment           inference: latest
 Preprocessed       Best: 91.70%     (Public)            (Public)

  STAGE 5             STAGE 6             STAGE 7             STAGE 8
┌───────────┐      ┌───────────┐      ┌───────────┐      ┌───────────┐
│  GITHUB   │      │    CI     │      │ INFERENCE │      │   W&B     │
│   REPO    │─────▶│ WORKFLOW  │─────▶│ WORKFLOW  │─────▶│DASHBOARD  │
│  (Task 1) │      │ (Task 7.1)│      │ (Task 7.2)│      │ (Task 8)  │
└───────────┘      └───────────┘      └───────────┘      └───────────┘
      │                  │                  │                   │
      ▼                  ▼                  ▼                   ▼
 develop→main       flake8 lint        Manual trigger      Both runs
 via PR review      on every push      workflow_dispatch   Public &
 (Branch Protected) to develop         Prediction:      Comparable
```

---

##  Task-by-Task Completion

---

###  Task 1 — GitHub Repository Setup  — Member 1

- Created public GitHub repository: `mlops-pipeline-a3`
- Initialized with `README.md`, `.gitignore`, `LICENSE`
- Created `develop` branch; protected `main` branch (requires ≥1 PR review before merge)
- Member 1 = Admin (repo owner); Members 2, 3, 4 added as Collaborators with Write access
- Added GitHub Secrets: `HF_TOKEN`, `WANDB_API_KEY`, `DOCKERHUB_USERNAME`, `DOCKERHUB_TOKEN`

**Project Folder Structure:**
```
mlops-pipeline-a3/
├── .github/
│   └── workflows/
│       ├── ci.yml           ← Lint check on every push
│       └── inference.yml    ← Manual inference trigger
├── src/
│   ├── prepare_data.py      ← Data cleaning script
│   ├── train.py             ← Model loading script
│   └── inference.py         ← Inference script
├── Dockerfile               ← Container definition
├── requirements.txt         ← Python dependencies
├── id2label.json            ← Label mapping
├── push_model.py            ← HF push script
└── README.md                ← This file
```

---

### Task 2 — Data Preparation & Normalization  — Member 1

**Dataset:** IMDB Movie Reviews (50,000 samples)  
**Source:** HuggingFace Datasets Hub (`load_dataset("imdb")`)  
**Script:** `src/prepare_data.py`

**Data Cleaning Steps & Justification:**

| Step | What was done | Why |
|------|--------------|-----|
| Remove HTML tags | `re.sub(r'<.*?>', '', text)` | IMDB reviews contain raw HTML like `<br/>` from web scraping |
| Remove extra spaces | `re.sub(r'\s+', ' ', text)` | Normalises whitespace for consistent tokenization |
| Lowercase | `text.lower().strip()` | Reduces vocabulary size; DistilBERT is case-insensitive |
| Remove duplicates | `df.drop_duplicates(subset="text")` | Prevents data leakage and biased training |
| Remove nulls | `df.dropna()` | Eliminates empty rows that would crash the trainer |

**Final Dataset After Cleaning:**
- Train: ~25,000 samples
- Test: ~25,000 samples
- Class distribution: Balanced (50% positive, 50% negative) 

**Label Mapping (`id2label.json`):**
```json
{"0": "negative", "1": "positive"}
```

---

###  Task 3 — Select & Load Model from HuggingFace — Member 1

**Model Chosen:** `distilbert-base-uncased`  
**HuggingFace Model Card:** https://huggingface.co/distilbert-base-uncased

**Why DistilBERT?**  
DistilBERT is a distilled (compressed) version of BERT that retains 97% of BERT's language understanding while being 40% smaller and 60% faster. It is under 200MB, making it ideal for Kaggle's free GPU tier. Its bidirectional attention makes it excellent for sentiment classification tasks. The model card confirms it was pre-trained on BookCorpus and English Wikipedia, making it well-suited for understanding movie review language. It supports the Hugging Face Trainer API directly, allowing seamless fine-tuning with minimal code.

**Model Loading (`src/train.py`):**
```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import json

with open("id2label.json", "r") as f:
    id2label = json.load(f)

label2id = {v: k for k, v in id2label.items()}
model_name = "distilbert-base-uncased"
num_labels = len(id2label)

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=num_labels,
    id2label=id2label,
    label2id=label2id
)
```

---

###  Task 4 — Train Multiple Versions on Kaggle & Track with W&B  — Member 1

**Platform:** Kaggle Notebooks (GPU T4 x2)  
**Tracking:** Weights & Biases — https://wandb.ai/g25ait2023-iit-jodhpur/mlops-assignment3  
**Kaggle V1:** https://www.kaggle.com/code/ahalderg25ait2023/mlops-group-project  
**Kaggle V2:** https://www.kaggle.com/code/ahalderg25ait2023/mlops-group-project-v2

**Two Experiments Run — Differing in Learning Rate:**

| Hyperparameter | run-v1 | run-v2 (Best) |
|----------------|--------|-----------------|
| Learning Rate | 3e-5 | **5e-5** |
| Epochs | 3 | 3 |
| Batch Size | 16 | 16 |
| Max Token Length | 256 | 256 |
| Warmup Steps | 500 | 500 |
| Weight Decay | 0.01 | 0.01 |
| Platform | Kaggle GPU T4 x2 | Kaggle GPU T4 x2 |

**Results Comparison:**

| Metric | run-v1 (lr=3e-5) | run-v2 (lr=5e-5) | Winner |
|--------|-----------------|-----------------|--------|
| Accuracy | 91.54% | **91.70%** |  run-v2 |
| F1 Score | 91.53% | **91.70%** |  run-v2 |
| Validation Loss | 0.7264 | 0.7424 | run-v1 |
| Runtime | 22m 45s | 24m 2s | — |

**Why run-v2 won:** Higher learning rate (5e-5) helped the model converge to a better accuracy and F1 score within 3 epochs, despite slightly higher validation loss.

**W&B Logging includes:** training loss, validation loss, accuracy, F1 score, all hyperparameters, runtime, and HuggingFace model URL in run summary.

**Secrets stored in Kaggle Secrets (never hardcoded):** `WANDB_API_KEY`, `HF_TOKEN.`

---

###  Task 5 — Push Trained Model to HuggingFace Hub  — Member 2

**Model pushed:** `run-v2` (best model — 91.70% accuracy)  
**HuggingFace URL:** https://huggingface.co/Atreyee-Halder/mlops-imdb-sentiment  
**Visibility:** Public 

**What was pushed:**
- Fine-tuned model weights (~268MB)
- Tokenizer (distilbert vocabulary)
- Model card (README.md) with training details, metrics, and usage

**HuggingFace Model URL logged to W&B run summary** 

**Usage:**
```python
from transformers import pipeline

classifier = pipeline(
    'text-classification',
    model='Atreyee-Halder/mlops-imdb-sentiment',
    token='your_hf_token'
)

result = classifier("This movie was absolutely amazing!")
# Output: [{'label': 'positive', 'score': 0.9967}]
```

---

### Task 6 — Create a Dockerfile — Member 3

**Docker Hub Image:** `g25ait2065/mlops-a3-inference:latest`  
**Docker Hub URL:** https://hub.docker.com/r/g25ait2065/mlops-a3-inference  
**Visibility:** Public 

#### `Dockerfile`
```dockerfile
FROM python:3.11-slim

ARG HF_MODEL_NAME=Atreyee-Halder/mlops-imdb-sentiment
ENV HF_MODEL_NAME=$HF_MODEL_NAME

WORKDIR /app
COPY requirements.txt.
RUN pip install --no-cache-dir -r requirements.txt

COPY src/inference.py.
CMD ["python", "inference.py"]
```

**Dockerfile Design Choices:**
- `python:3.11-slim` — lightweight base image (~50MB vs ~900MB full); only inference dependencies installed
- `ARG HF_MODEL_NAME` — accepts model name at build time with a sensible default; allows reuse with any HF model
- `ENV HF_MODEL_NAME` — makes ARG available at runtime inside the container
- `--no-cache-dir` — keeps image size minimal by not caching pip downloads

#### `src/inference.py`
```python
import os
from transformers import pipeline

HF_TOKEN = os.environ.get('HF_TOKEN')
INPUT_TEXT = os.environ.get('INPUT_TEXT', 'This movie was absolutely amazing!')
HF_MODEL_NAME = os.environ.get('HF_MODEL_NAME', 'Atreyee-Halder/mlops-imdb-sentiment')

print(f"Loading model: {HF_MODEL_NAME}")
print(f"Input text: {INPUT_TEXT}")

classifier = pipeline('text-classification', model=HF_MODEL_NAME, token=HF_TOKEN)
result = classifier(INPUT_TEXT)

print(f"Prediction: {result[0]['label']}")
print(f"Confidence: {result[0]['score']:.4f}")
```

**Build, Test & Push Commands:**
```bash
# Build
docker build -t g25ait2065/mlops-a3-inference.

# Test locally
docker run --rm \
  -e HF_TOKEN=your_hf_token \
  -e INPUT_TEXT='This movie was amazing!' \
  g25ait2065/mlops-a3-inference:latest

# Push to Docker Hub
docker push g25ait2065/mlops-a3-inference
```

**Local Test Output (Confirmed):**
```
Loading model: Atreyee-Halder/mlops-imdb-sentiment
Input text: This movie was amazing!
Prediction: positive
Confidence: 0.9966
```

---

### Task 7 — GitHub Actions CI/CD — Member 4

Two automated workflows were implemented and verified to be working:

#### 7.1 CI Workflow — `.github/workflows/ci.yml.` 

```yaml
name: CI
on:
  push:
    branches: [develop]
  pull_request:
    branches: [main]
jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - name: Install dependencies
        run: pip install flake8
      - name: Run flake8 lint check
        run: flake8 src/ --max-line-length=120 --extend-ignore=E203,W503
```

- **Trigger:** Every push to `develop` and every PR to `main.`
- **Action:** Runs `flake8` Python linter on `src/` directory
- **Status:**  Passing — verified green on GitHub Actions

#### 7.2 Inference Workflow — `.github/workflows/inference.yml` 

```yaml
name: Inference
on:
  push:
    branches: [develop]
  workflow_dispatch:
    inputs:
      input_text:
        description: "Text to classify."
        required: true
        default: "This movie was absolutely amazing!"
        type: string
jobs:
  infer:
    name: Run Model Inference
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - name: Install Dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt
      - name: Run Inference
        env:
          HF_TOKEN: ${{ secrets.HF_TOKEN }}
          HF_MODEL_NAME: Atreyee-Halder/mlops-imdb-sentiment
          INPUT_TEXT: ${{ github.event.inputs.input_text || 'This movie was absolutely amazing!' }}
        run: |
          python src/inference.py
```

- **Trigger:** Manual (`workflow_dispatch`) with custom input text
- **Status:**  Passing — confirmed output below

**GitHub Actions Inference Output (Confirmed):**
```
Loading model: Atreyee-Halder/mlops-imdb-sentiment
Input text: This movie was absolutely amazing!
Prediction: positive
Confidence: 0.9967
```

#### 7.3 GitHub Secrets 

All secrets stored securely in GitHub → Settings → Secrets → Actions:

| Secret Name | Purpose |
|-------------|---------|
| `HF_TOKEN` | HuggingFace API authentication |
| `WANDB_API_KEY` | W&B experiment tracking |
| `DOCKERHUB_USERNAME` | Docker Hub login |
| `DOCKERHUB_TOKEN` | Docker Hub push authentication |

**No API tokens are hardcoded anywhere in the repository** 

---

### Task 8 — Show All Experiments on W&B  — Member 1

- W&B Project: `mlops-assignment3` (entity: `g25ait2023-iit-jodhpur`)
- Both `run-v1` and `run-v2` appear in the dashboard
- Project visibility: **Public** 
- Runs Comparison table shows Accuracy, F1, and Loss side-by-side
- W&B Dashboard: https://wandb.ai/g25ait2023-iit-jodhpur/mlops-assignment3

---

##  Dependencies & Libraries

### `requirements.txt`
```
transformers
torch
huggingface_hub
datasets
pandas
scikit-learn
wandb
flake8
```

### Full Dependency Details

| Package | Purpose |
|---------|---------|
| `transformers` | HuggingFace model loading, tokenization, Trainer API, fine-tuning |
| `torch` | PyTorch deep learning framework — model training and inference |
| `huggingface_hub` | Push/pull models and tokenizers to HuggingFace Hub |
| `datasets` | Load IMDB dataset directly from HuggingFace Datasets Hub |
| `pandas` | Data manipulation, cleaning, deduplication |
| `scikit-learn` | Evaluation metrics — accuracy_score, f1_score |
| `wandb` | Experiment tracking, hyperparameter logging, dashboard |
| `flake8` | Python code linting — used in CI workflow |

### Infrastructure & Tools

| Tool | Version/Tier | Purpose |
|------|-------------|---------|
| Python | 3.11 | Runtime environment |
| Docker | Desktop (free) | Container build and run |
| Docker Hub | Free tier | Public Docker image registry |
| GitHub Actions | Free tier | CI/CD automation |
| HuggingFace Hub | Free tier | Model registry and hosting |
| Weights & Biases | Free tier | Experiment tracking |
| Kaggle | Free GPU (T4 x2) | Model training environment |
| Git | Latest | Version control |

---

##  Git Workflow

```
main (protected — requires 1 PR review before merge)
  ↑
  │ ← Pull Request #1 (Final Merge)
  │   Created by:  halderatreyee-hash (Member 1)
  │   Reviewed by: g25ait2131 (Member 2)  Approved
  │   Merged by:   halderatreyee-hash (Member 1)
  │   CI Checks:   3/3 passed 
  │
  │ ← Pull Request #2 (Member 2 - Task 5)
  │   Branch:      feat/member2-model-push
  │   Created by:  g25ait2131 (Member 2)
  │   Reviewed by: halderatreyee-hash (Member 1)  Approved
  │   Merged by:   halderatreyee-hash (Member 1)
  │   CI Checks:   1/1 passed 
  │
develop branch (all development work — 16 commits)
  ├── Member 1 (halderatreyee-hash) → Task 1, 2, 3, 4, 8
  ├── Member 2 (g25ait2131)         → Task 5
  ├── Member 3 (g25ait2065)         → Task 6
  └── Member 4 (g25ait2139)         → Task 7
```


##  Model Performance Summary

| Run | Learning Rate | Accuracy | F1 Score | Val Loss | Runtime | Status |
|-----|-------------|----------|----------|----------|---------|--------|
| run-v1 | 3e-5 | 91.54% | 91.53% | 0.7264 | 22m 45s | Baseline |
| **run-v2** | **5e-5** | **91.70%** | **91.70%** | 0.7424 | 24m 2s | **Best — Deployed** |

---

##  How to Run Inference

### Option 1 — Docker (Tested & Verified )
```bash
docker pull g25ait2065/mlops-a3-inference:latest

docker run --rm \
  -e HF_TOKEN=your_hf_token \
  -e INPUT_TEXT='This movie was absolutely amazing!' \
  g25ait2065/mlops-a3-inference:latest
```

**Output:**
```
Loading model: Atreyee-Halder/mlops-imdb-sentiment
Input text: This movie was absolutely amazing!
Prediction: positive
Confidence: 0.9966
```

### Option 2 — GitHub Actions (Tested & Verified )
1. Go to [Actions → Inference](https://github.com/halderatreyee-hash/mlops-pipeline-a3/actions/workflows/inference.yml)
2. Click **"Run workflow"**
3. Enter your text in the input field
4. Click green **"Run workflow"** button
5. View prediction output in the logs

**Output:**
```
Loading model: Atreyee-Halder/mlops-imdb-sentiment
Input text: This movie was absolutely amazing!
Prediction: positive
Confidence: 0.9967
```

---

##  Pipeline Completion Checklist

| Task | Description | Status |
|------|-------------|--------|
| Task 1 | GitHub repo setup with branch protection & collaborators | Complete |
| Task 2 | Data preparation and cleaning script (IMDB, 50K samples) | Complete |
| Task 3 | DistilBERT model selected, loaded with id2label mapping | Complete |
| Task 4 | 2 training runs (run-v1, run-v2) on Kaggle + W&B tracking | Complete |
| Task 5 | Best model (run-v2) pushed to HuggingFace Hub (Public) | Complete |
| Task 6 | Dockerfile + inference script + Docker Hub push (Public) |Complete |
| Task 7 | GitHub Actions — CI (flake8) + Inference workflows | Complete |
| Task 8 | W&B public dashboard — both runs visible & comparable| Complete |
| Final PR | develop → main with review, approval & merge |Complete |

---

##  Setup Instructions

### Clone & Install
```bash
git clone https://github.com/halderatreyee-hash/mlops-pipeline-a3.git
cd mlops-pipeline-a3
pip install -r requirements.txt
```

### Run Data Preparation
```bash
python src/prepare_data.py
```

### Run Inference Locally
```bash
export HF_TOKEN=your_hf_token
export INPUT_TEXT='This movie was great!'
python src/inference.py
```

### Build Docker Image
```bash
docker build -t mlops-a3-inference.
docker run --rm -e HF_TOKEN=your_token -e INPUT_TEXT='Great film!' mlops-a3-inference
```

---

*IIT Jodhpur | PGD AI Program | MLOps Group Project Assignment 3*  
* Submitted by Group Members: halderatreyee-hash, g25ait2131, g25ait2065, g25ait2139*
