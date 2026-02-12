from fastapi import FastAPI
from backend.model_loader import load_llm_pipeline
from backend.schemas import QuestionRequest

app = FastAPI(title="Medical Pathology LLM API")

# Load model ONCE at startup
pipe = load_llm_pipeline()

@app.get("/")
def root():
    return {"message": "Medical Pathology LLM API is running"}

@app.post("/ask")
def ask_question(request: QuestionRequest):
    prompt = f"""You are a medical pathology assistant.

Question:
{request.question}

Answer:
"""

    result = pipe(
        prompt,
        max_new_tokens=80,
        do_sample=False,
        repetition_penalty=1.2,
        return_full_text=False
    )

    answer = result[0]["generated_text"]

    return {"answer": answer}
