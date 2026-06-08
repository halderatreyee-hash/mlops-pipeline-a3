# Task 5: Push best trained model (run-v2) to HuggingFace Hub
# Member 2 - g25ait2131 (Mahadev Vishwakarma)
# Model: Atreyee-Halder/mlops-imdb-sentiment
# Best run: run-v2 | Accuracy: 91.70% | F1: 91.70%
# Experiment Tracking: https://wandb.ai/g25ait2023-iit-jodhpur/mlops-assignment3

from transformers import AutoModelForSequenceClassification, AutoTokenizer
from huggingface_hub import login
import os

# Login to HuggingFace using environment token
login(token=os.environ.get('HF_TOKEN'))

# Load best trained model (run-v2) from local directory
model = AutoModelForSequenceClassification.from_pretrained('./best-model')
tokenizer = AutoTokenizer.from_pretrained('./best-model')

# Push model and tokenizer to HuggingFace Hub
model.push_to_hub('Atreyee-Halder/mlops-imdb-sentiment')
tokenizer.push_to_hub('Atreyee-Halder/mlops-imdb-sentiment')

print("Model pushed successfully!")
print("URL: https://huggingface.co/Atreyee-Halder/mlops-imdb-sentiment")
