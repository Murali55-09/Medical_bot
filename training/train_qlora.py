import os
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model

# -------------------------------------------------
# 1️⃣ HuggingFace cache (G drive)
# -------------------------------------------------
os.environ["HF_HOME"] = "G:/hf_cache"
os.environ["HF_DATASETS_CACHE"] = "G:/hf_cache/datasets"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

# -------------------------------------------------
# 2️⃣ Model Name (NON-GATED)
# -------------------------------------------------
model_name = "Qwen/Qwen2-1.5B-Instruct"

# -------------------------------------------------
# 3️⃣ QLoRA 4-bit Configuration
# -------------------------------------------------
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

# -------------------------------------------------
# 4️⃣ Tokenizer
# -------------------------------------------------
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# -------------------------------------------------
# 5️⃣ Load Base Model (4-bit)
# -------------------------------------------------
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
    low_cpu_mem_usage=True,
)

# -------------------------------------------------
# 6️⃣ Apply LoRA
# -------------------------------------------------
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# -------------------------------------------------
# 7️⃣ Load Dataset
# -------------------------------------------------
dataset = load_dataset(
    "json",
    data_files="data/finetune/pathology_train_v2.jsonl"
)

# -------------------------------------------------
# 8️⃣ Format + Tokenize
# -------------------------------------------------
def format_data(example):
    prompt = (
        f"### Instruction:\n{example['instruction']}\n\n"
        f"### Question:\n{example['input']}\n\n"
        f"### Answer:\n{example['output']}"
    )

    tokenized = tokenizer(
        prompt,
        truncation=True,
        padding="max_length",
        max_length=512,
    )

    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

tokenized_data = dataset.map(
    format_data,
    remove_columns=dataset["train"].column_names
)

# -------------------------------------------------
# 9️⃣ Training Arguments (Stable Settings)
# -------------------------------------------------
training_args = TrainingArguments(
    output_dir="models/pathology-qwen-qlora",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    num_train_epochs=4,
    learning_rate=1e-4,
    fp16=True,
    logging_steps=10,
    save_strategy="epoch",
    report_to="none"
)

# -------------------------------------------------
# 🔟 Trainer
# -------------------------------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_data["train"],
)

# -------------------------------------------------
# 1️⃣1️⃣ Train
# -------------------------------------------------
trainer.train()

# -------------------------------------------------
# 1️⃣2️⃣ Save Adapters + Tokenizer
# -------------------------------------------------
save_dir = "models/pathology-qwen-qlora"
model.save_pretrained(save_dir)
tokenizer.save_pretrained(save_dir)

print("✅ Training complete. Model saved to:", save_dir)