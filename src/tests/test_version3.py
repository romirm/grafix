import fitz  # PyMuPDF
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sentence_transformers import SentenceTransformer, util
from fuzzywuzzy import fuzz

nltk.download('punkt')
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
model = SentenceTransformer('all-MiniLM-L6-v2')

# === Utility Functions ===

def extract_text(path):
    try:
        with fitz.open(path) as doc:
            return " ".join(page.get_text() for page in doc)
    except Exception as e:
        print(f"Error reading {path}: {e}")
        return ""

def preprocess(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text.lower())
    tokens = word_tokenize(text)
    return [t for t in tokens if t not in stop_words and len(t) > 1]

def extract_entities(text):
    text_lower = text.lower()

    # Extract majors from degree lines
    major_phrases = re.findall(r'bachelor.*?in ([a-z\s&]+)', text_lower)
    flat_majors = []
    for phrase in major_phrases:
        parts = re.split(r'and|&|,', phrase)
        flat_majors.extend([p.strip() for p in parts if p.strip()])

    # Extract graduation year
    grad_years = re.findall(r'jun[ea]*\.*\s*(20\d{2})', text_lower)

    # === Dynamic Club Detection ===
    # Get capitalized multi-word org names, skipping all-caps (like acronyms) and common words
    org_matches = re.findall(r'\b([A-Z][a-z]+(?:\s[A-Z][a-z]+)+)\b', text)
    org_blacklist = {"Bachelor of Arts", "High School", "College Park", "Evanston", "New York"}  # filter non-club stuff
    clubs = [org for org in org_matches if org not in org_blacklist]

    # Extract common tools/languages/interests (you can expand these as needed)
    skills = re.findall(r'\b(excel|factset|stata|pitchbook|html|python|sql)\b', text_lower)
    interests = re.findall(r'\b(guitar|piano|basketball|falcons|birdwatching|biking|hiking|documentaries)\b', text_lower)

    return {
        "grad_year": grad_years,
        "majors": flat_majors,
        "clubs": clubs,
        "skills": skills,
        "interests": interests,
        "full_text": text
    }

def compare_and_score(data1, data2, weights):
    score = 0.0
    report = {}
    for key in weights:
        shared = list(set(data1.get(key, [])) & set(data2.get(key, [])))
        report[key] = shared
        score += weights[key] * len(shared)
    return score, report

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
        "clubs": 2.0,
        "skills": 1.5,
        "interests": 1.0
    }

    category_score, match_report = compare_and_score(data1, data2, weights)
    sem_score = semantic_score(data1["full_text"], data2["full_text"])

    total_score = 0.5 * sem_score + 0.5 * (category_score / 10.0)  # normalize weights
    print("\n=== Resume Similarity Report ===\n")
    print(f"Semantic Similarity: {sem_score:.4f}")
    print(f"Category Match Score: {category_score:.2f}")
    print(f"Final Combined Similarity Score: {total_score:.4f}\n")

    for key, val in match_report.items():
        if val:
            print(f"- Shared {key.title()}: {', '.join(val)}")
    return total_score

if __name__ == "__main__":
    compare_resumes("resumes/resume_paari.pdf", "resumes/resume_rishabh.pdf")
