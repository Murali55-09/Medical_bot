import openai
import json

# Set your OpenAI API key
import os
openai.api_key = os.getenv("OPENAI_API_KEY")
# Define the prompt for generating pathology-related Q&A pairs
PROMPT = "Generate 10 pathology-related questions and their answers. Format each as a JSON object with 'instruction', 'input', and 'output'."

# Number of examples to generate
NUM_EXAMPLES = 50

# Output file
OUTPUT_FILE = "pathology_synthetic_data.jsonl"

def generate_data():
    """Generate synthetic pathology-related Q&A pairs using GPT-4."""
    data = []
    for _ in range(NUM_EXAMPLES // 10):
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a medical pathology expert."},
                {"role": "user", "content": PROMPT}
            ],
            max_tokens=1000
        )
        # Parse the response
        generated_text = response['choices'][0]['message']['content']
        examples = json.loads(generated_text)
        data.extend(examples)

    # Save to JSONL file
    with open(OUTPUT_FILE, "w") as f:
        for entry in data:
            f.write(json.dumps(entry) + "\n")

    print(f"Generated {len(data)} examples and saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    generate_data()