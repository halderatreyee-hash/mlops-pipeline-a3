import json
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load id2label mapping
with open("id2label.json", "r") as f:
    id2label = json.load(f)

label2id = {v: k for k, v in id2label.items()}

model_name = "distilbert-base-uncased"
num_labels = len(id2label)

print(f"Loading tokenizer for {model_name}...")
tokenizer = AutoTokenizer.from_pretrained(model_name)

print(f"Loading model with {num_labels} output labels...")
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=num_labels,
    id2label=id2label,
    label2id=label2id
)

print(f"Model loaded successfully!")
print(f"Model name: {model_name}")
print(f"Number of labels: {num_labels}")
print(f"Labels: {id2label}")