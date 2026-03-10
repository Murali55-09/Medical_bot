import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig
from peft import PeftModel

# --------------------------------------------
# ✅ USE QWEN BASE MODEL
# --------------------------------------------
BASE_MODEL = "Qwen/Qwen2-1.5B-Instruct"
ADAPTER_PATH = "models/pathology-qwen-qlora"

print("Loading Qwen model for evaluation...")

# Same 4-bit config used during training
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_config,
    device_map="auto"
)

# Load tokenizer from BASE MODEL (IMPORTANT)
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# Load LoRA adapters
model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)

# Build pipeline
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer
)

# --------------------------------------------
# Evaluation questions
# --------------------------------------------
test_questions = [
    "What is necrosis?",
    "What is thrombosis?",
    "Describe necrosis.",
    "Describe myocardial infarction.",
    "Explain the mechanism of ischemic injury.",
    "Explain septic shock.",
    "Differentiate acute and chronic inflammation.",
    "Differentiate benign and malignant tumors.",
    "What is cricket?",
    "Explain Python programming."
]

def ask(question):
    prompt = f"""You are a medical pathology assistant.
Answer clearly and accurately.

Question:
{question}

Answer:
"""

    result = pipe(
        prompt,
        max_new_tokens=150,
        do_sample=False,
        repetition_penalty=1.2,
        no_repeat_ngram_size=3,
        return_full_text=False
    )

    return result[0]["generated_text"]


print("\nRunning evaluation...\n")

with open("training/eval_results.txt", "w", encoding="utf-8") as f:
    for q in test_questions:
        ans = ask(q)
        output = f"\nQUESTION: {q}\nANSWER: {ans}\n{'-'*60}\n"
        print(output)
        f.write(output)

print("Evaluation complete ✅")
print("Results saved to training/eval_results.txt")