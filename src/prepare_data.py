import pandas as pd
import re
import json
from datasets import load_dataset

# Load IMDB dataset
print("Loading IMDB dataset...")
dataset = load_dataset("imdb")

def clean_text(text):
    text = re.sub(r'<.*?>', '', text)      # remove HTML tags
    text = re.sub(r'\s+', ' ', text)       # remove extra spaces
    text = text.lower().strip()            # lowercase
    return text

# Clean train and test splits
for split in ["train", "test"]:
    texts = [clean_text(t) for t in dataset[split]["text"]]
    labels = dataset[split]["label"]
    df = pd.DataFrame({"text": texts, "label": labels})
    df.drop_duplicates(subset="text", inplace=True)
    df.dropna(inplace=True)
    print(f"{split}: {len(df)} samples after cleaning")
    print(f"Label distribution:\n{df['label'].value_counts()}\n")

# Save id2label mapping
id2label = {"0": "negative", "1": "positive"}
with open("id2label.json", "w") as f:
    json.dump(id2label, f)

print("id2label.json saved!")
print("Data preparation complete!")