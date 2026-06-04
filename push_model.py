from transformers import AutoModelForSequenceClassification, AutoTokenizer
from huggingface_hub import login
import os

login(token=os.environ.get('HF_TOKEN'))

model = AutoModelForSequenceClassification.from_pretrained('./best-model')
tokenizer = AutoTokenizer.from_pretrained('./best-model')

model.push_to_hub('Atreyee-Halder/mlops-imdb-sentiment')
tokenizer.push_to_hub('Atreyee-Halder/mlops-imdb-sentiment')

print("Model pushed successfully!")
print("URL: https://huggingface.co/Atreyee-Halder/mlops-imdb-sentiment")