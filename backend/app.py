from unittest import result

from fastapi import FastAPI
from backend.model_loader import load_llm_pipeline
from backend.schemas import QuestionRequest
from classifier.simple_classifier import is_pathology_question

app = FastAPI(title="Medical Pathology LLM API")

# Load model ONCE at startup
pipe = load_llm_pipeline()

@app.get("/")
def root():
    return {"message": "Medical Pathology LLM API is running"}

@app.post("/ask")
def ask_question(request: QuestionRequest):

    question = request.question

    # 🔹 Domain control layer
    if not is_pathology_question(question):
        return {
            "answer": "I can only answer pathology-related questions."
        }

    prompt = f"""You are a medical pathology assistant.

Question:
{question}

Answer:
"""

    result = pipe(
    prompt,
    max_new_tokens=60,
    do_sample=False,
    repetition_penalty=1.3,
    return_full_text=False
    )

    answer = result[0]["generated_text"].split("Question:")[0].strip()

    return {"answer": answer}
