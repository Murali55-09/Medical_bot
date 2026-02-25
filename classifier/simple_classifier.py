import re

# Basic pathology-related keywords
PATHOLOGY_KEYWORDS = [
    "necrosis",
    "apoptosis",
    "thrombosis",
    "inflammation",
    "cirrhosis",
    "tumor",
    "neoplasia",
    "biopsy",
    "lesion",
    "infection",
    "disease",
    "cell",
    "tissue",
    "organ",
    "cancer",
    "pathology",
    "histology",
    "diagnosis",
    "etiology",
    "symptom"
]

def is_pathology_question(question: str) -> bool:
    question = question.lower()

    for keyword in PATHOLOGY_KEYWORDS:
        if re.search(rf"\b{keyword}\b", question):
            return True

    return False