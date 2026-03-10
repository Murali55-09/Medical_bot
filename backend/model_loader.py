import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig
from peft import PeftModel

BASE_MODEL = "Qwen/Qwen2-1.5B-Instruct"  # Base model name (NON-GATED)

# Get the project root directory (parent of backend)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ADAPTER_PATH = os.path.join(PROJECT_ROOT, "models", "pathology-qwen-qlora")

def load_llm_pipeline():
    print(f"Adapter path: {ADAPTER_PATH}")
    print("Loading base model...")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto"
    )

    print("Loading tokenizer + adapters...")
    tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH)
    tokenizer.pad_token = tokenizer.eos_token

    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer
    )

    print("Model loaded successfully")
    return pipe

