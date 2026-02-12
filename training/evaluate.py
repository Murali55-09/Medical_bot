import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig
from peft import PeftModel

BASE_MODEL = "EleutherAI/gpt-neo-1.3B"
ADAPTER_PATH = "models/pathology-qlora-v1"

print("Loading model for evaluation...")

# Same quantization config used during training
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

# Load tokenizer + adapters
tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH)
tokenizer.pad_token = tokenizer.eos_token
model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)

# Build pipeline
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer
)

# ---------------------------------------------------
# Evaluation questions
# ---------------------------------------------------
questions = [
    "What is necrosis?",
    "What is apoptosis?",
    "What is thrombosis?",
    "What is inflammation?",
    "What is cirrhosis?",
    "What is trading?",
    "What is cricket?",
    "Explain Python programming."
]

def ask(question):
    prompt = f"""You are a medical pathology assistant.

Question:
{question}

Answer:
"""
    result = pipe(
        prompt,
        max_new_tokens=80,
        do_sample=False,
        repetition_penalty=1.2,
        return_full_text=False
    )
    return result[0]["generated_text"]

print("\nRunning evaluation...\n")

with open("training/eval_results.txt", "w", encoding="utf-8") as f:
    for q in questions:
        ans = ask(q)
        output = f"\nQUESTION: {q}\nANSWER: {ans}\n{'-'*50}\n"
        print(output)
        f.write(output)

print("Evaluation complete ✅")
print("Results saved to training/eval_results.txt")
