import fitz  # PyMuPDF
import re
import nltk
import os
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer, util
from fuzzywuzzy import fuzz

nltk.download('punkt')
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
model = SentenceTransformer('all-MiniLM-L6-v2')

# === Extract Data from Resume ===
def extract_text(path):
    try:
        with fitz.open(path) as doc:
            return " ".join(page.get_text() for page in doc)
    except Exception as e:
        print(f"Error reading {path}: {e}")
        return ""

# === Entity Extraction ===
def extract_entities(text):
    text_lower = text.lower()

    # Graduation Year
    grad_years = re.findall(r'\bjun(?:e|\.|)\s*(20\d{2})', text_lower)

    # Majors
    major_phrases = re.findall(r'bachelor.*?in ([a-z\s&]+)', text_lower)
    flat_majors = []
    for phrase in major_phrases:
        parts = re.split(r'and|&|,', phrase)
        flat_majors.extend([p.strip() for p in parts if p.strip()])

    # Coursework
    courses = []
    coursework_sections = re.findall(r'relevant coursework:?([\s\S]{0,300})', text_lower)
    for section in coursework_sections:
        raw_courses = re.split(r',|\n|\u2022|-', section)
        courses += [c.strip() for c in raw_courses if 2 < len(c.strip()) < 50]

    # Clubs / Orgs
    org_matches = re.findall(r'\b([A-Z][a-z]+(?:\s[A-Z][a-z]+)+)\b', text)
    org_blacklist = {"Bachelor of Arts", "High School", "College Park", "New York", "Evanston"}
    clubs = [org.strip() for org in org_matches if org not in org_blacklist]

    # Dynamic Keywords (skills, tools, misc terms)
    tokens = re.findall(r'\b[a-zA-Z][a-zA-Z\-]+\b', text_lower)
    words = [t.strip() for t in tokens if 2 < len(t) < 30]
    all_terms = list(set(words))

    return {
        "grad_year": grad_years,
        "majors": flat_majors,
        "courses": courses,
        "clubs": clubs,
        "terms": all_terms,
        "full_text": text
    }

# === Compare and Score ===
def compare_and_score(data1, data2, weights):
    score = 0.0
    report = {}
    for key in weights:
        shared = list(set(data1.get(key, [])) & set(data2.get(key, [])))
        report[key] = shared
        score += weights[key] * len(shared)
    return score, report

# === Semantic Similarity ===
def semantic_score(text1, text2):
    embeddings = model.encode([text1, text2], convert_to_tensor=True)
    return util.pytorch_cos_sim(embeddings[0], embeddings[1]).item()

# === Main Comparison Function ===
def compare_resumes(path1, path2):
    text1 = extract_text(path1)
    text2 = extract_text(path2)

    data1 = extract_entities(text1)
    data2 = extract_entities(text2)

    weights = {
        "grad_year": 3.0,
        "majors": 2.5,
        "courses": 1.75,
        "clubs": 2.0,
        "terms": 1.0
    }

    category_score, match_report = compare_and_score(data1, data2, weights)
    sem_score = semantic_score(data1["full_text"], data2["full_text"])

    total_score = 0.5 * sem_score + 0.5 * (category_score / 10.0)  # normalize for blended score

    print("\n=== Resume Similarity Report ===\n")
    print(f"Semantic Similarity: {sem_score:.4f}")
    print(f"Category Match Score: {category_score:.2f}")
    print(f"Final Combined Similarity Score: {total_score:.4f}\n")

    for key, val in match_report.items():
        if val:
            print(f"- Shared {key.title()}: {', '.join(val)}")

    return total_score

# === Example Call ===
# compare_resumes("Paari Dhanasekaran Resume.pdf", "Jain_Rishabh_Resume.pdf")

if __name__ == "__main__":
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    resume1 = os.path.join(base_dir, "resumes", "resume_john.pdf")
    resume2 = os.path.join(base_dir, "resumes", "resume_steve.pdf")
    compare_resumes(resume1, resume2)