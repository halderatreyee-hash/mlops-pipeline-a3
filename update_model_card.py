from huggingface_hub import HfApi
import os

api = HfApi()

model_card_content = """---
license: mit
language:
- en
tags:
- text-classification
- sentiment-analysis
- distilbert
---

 MLOps IMDB Sentiment Analysis Model
---‐----------------------------------
## Model Description
Fine-tuned distilbert-base-uncased for binary sentiment classification on IMDB movie reviews.

## Training Details
- *Base Model:* distilbert-base-uncased
- *Dataset:* IMDB Movie Reviews (50,000 samples)
- *Task:* Binary Text Classification
- *Platform:* Kaggle GPU T4 x2

## Performance (run-v2 - Best Model)
| Metric | Score |
|--------|-------|
| Accuracy | 91.70% |
| F1 Score | 91.70% |
| Validation Loss | 0.7424 |

## Hyperparameters
| Parameter | Value |
|-----------|-------|
| Learning Rate | 5e-5 |
| Epochs | 3 |
| Batch Size | 16 |
| Max Length | 256 |

## Experiment Comparison
| Run | Learning Rate | Accuracy | F1 |
|-----|-------------|----------|-----|
| run-v1 | 3e-5 | 91.54% | 91.53% |
| run-v2 | 5e-5 | *91.70%* | *91.70%* |

## Usage
from transformers import pipeline
classifier = pipeline('text-classification', model='Atreyee-Halder/mlops-imdb-sentiment')
result = classifier("This movie was absolutely amazing!")
print(result)

## Labels
- 0 = negative
- 1 = positive

## Project Links
- GitHub: https://github.com/halderatreyee-hash/mlops-pipeline-a3
- W&B: https://wandb.ai/g25ait2023-iit-jodhpur/mlops-assignment3
"""

api.upload_file(
    path_or_fileobj=model_card_content.encode(),
    path_in_repo="README.md",
    repo_id="Atreyee-Halder/mlops-imdb-sentiment",
    token=os.environ.get('HF_TOKEN'),
    commit_message="Update model card with V2 training details"
)

print("Model card updated successfully!")