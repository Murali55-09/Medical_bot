import requests
from bs4 import BeautifulSoup
import json

# Define the target URL (example: open-access medical journal or forum)
URL = "https://example.com/pathology-questions"

# Output file
OUTPUT_FILE = "pathology_scraped_data.jsonl"

def scrape_data():
    """Scrape pathology-related Q&A from a public website."""
    response = requests.get(URL)
    if response.status_code != 200:
        print(f"Failed to fetch data from {URL}")
        return

    soup = BeautifulSoup(response.text, "html.parser")

    # Example: Extract questions and answers (modify selectors as needed)
    questions = soup.select(".question-class")
    answers = soup.select(".answer-class")

    data = []
    for question, answer in zip(questions, answers):
        entry = {
            "instruction": "Answer the pathology question clearly and accurately.",
            "input": question.text.strip(),
            "output": answer.text.strip()
        }
        data.append(entry)

    # Save to JSONL file
    with open(OUTPUT_FILE, "w") as f:
        for entry in data:
            f.write(json.dumps(entry) + "\n")

    print(f"Scraped {len(data)} examples and saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    scrape_data()