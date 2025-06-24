from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
from peft import get_peft_model, LoraConfig, TaskType
import torch
import os

# === Load tokenized data ===
print("🔄 Loading tokenized dataset from ./data...")
train_dataset = torch.load("./data/train.pt", weights_only=False)
eval_dataset = torch.load("./data/test.pt", weights_only=False)

# ↓↓↓ Trim dataset for faster test training ↓↓↓
train_dataset = train_dataset.select(range(1000))
eval_dataset = eval_dataset.select(range(200))

# === Load tokenizer and base model ===
print("📦 Loading tokenizer and base model...")
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
base_model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased", num_labels=2
)

# === Apply LoRA config ===
print("⚙️ Applying LoRA...")
peft_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    inference_mode=False,
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    target_modules=["q_lin", "v_lin"]
)
model = get_peft_model(base_model, peft_config)

# === Move model to CUDA if available ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# === Training arguments ===
print("🛠️ Setting training args...")
training_args = TrainingArguments(
    output_dir="./model",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=1,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    report_to="none",
    save_total_limit=2
)

# === Data collator handles dynamic padding ===
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# === Define Trainer ===
print("🚂 Starting training with Trainer...")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

trainer.train()

# === Save model and tokenizer ===
print("💾 Saving model and tokenizer to ./model...")
model.save_pretrained("./model")
tokenizer.save_pretrained("./model")

print("✅ Done training!")
