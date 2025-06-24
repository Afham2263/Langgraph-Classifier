from datasets import load_dataset
from transformers import AutoTokenizer
import torch
import os

SAVE_DIR = "./data"
os.makedirs(SAVE_DIR, exist_ok=True)

print("Loading IMDb dataset...")
dataset = load_dataset("imdb")

# Reduce dataset size for faster debugging
print("🔪 Truncating dataset to 2000 train and 500 test samples...")
dataset["train"] = dataset["train"].select(range(2000))
dataset["test"] = dataset["test"].select(range(500))

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

def tokenize(example):
    return tokenizer(example["text"], truncation=True, padding="max_length", max_length=256)

print("Tokenizing dataset...")
dataset = dataset.map(tokenize, batched=True)
dataset = dataset.rename_column("label", "labels")
dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

# Save the train and test sets to disk as .pt
torch.save(dataset["train"], os.path.join(SAVE_DIR, "train.pt"))
torch.save(dataset["test"], os.path.join(SAVE_DIR, "test.pt"))

print("✅ Done! Tokenized data saved to ./data/")
