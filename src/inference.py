import os
from transformers import pipeline

HF_TOKEN = os.environ.get('HF_TOKEN')
INPUT_TEXT = os.environ.get('INPUT_TEXT', 'This movie was absolutely amazing!')
HF_MODEL_NAME = os.environ.get('HF_MODEL_NAME', 'Atreyee-Halder/mlops-imdb-sentiment')

print(f"Loading model: {HF_MODEL_NAME}")
print(f"Input text: {INPUT_TEXT}")

classifier = pipeline(
    'text-classification',
    model=HF_MODEL_NAME,
    token=HF_TOKEN
)

result = classifier(INPUT_TEXT)

print(f"Prediction: {result[0]['label']}")
print(f"Confidence: {result[0]['score']:.4f}")
