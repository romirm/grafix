import fitz  # PyMuPDF
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sentence_transformers import SentenceTransformer, util
from fuzzywuzzy import fuzz
import json
import itertools
import requests
from io import BytesIO

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
    org_matches = re.findall(r'\b([A-Z][a-z]+(?:\s[A-Z][a-z]+)+)\b', text)
    org_blacklist = {"Bachelor of Arts", "High School", "College Park", "Evanston", "New York"}
    clubs = [org for org in org_matches if org not in org_blacklist]

    # Extract common terms dynamically
    tokens = re.findall(r'\b[a-zA-Z][a-zA-Z\-]+\b', text_lower)
    words = [t.strip() for t in tokens if 2 < len(t) < 30]
    all_terms = list(set(words))

    return {
        "grad_year": grad_years,
        "majors": flat_majors,
        "clubs": clubs,
        "terms": all_terms,
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

def compare_resumes(path1, path2):
    text1 = extract_text(path1)
    text2 = extract_text(path2)

    data1 = extract_entities(text1)
    data2 = extract_entities(text2)

    weights = {
        "grad_year": 3.0,
        "majors": 2.5,
        "clubs": 2.0,
        "terms": 1.0
    }

    category_score, match_report = compare_and_score(data1, data2, weights)
    sem_score = semantic_score(data1["full_text"], data2["full_text"])

    total_score = 0.5 * sem_score + 0.5 * (category_score / 10.0)
    print("\n=== Resume Similarity Report ===\n")
    print(f"Semantic Similarity: {sem_score:.4f}")
    print(f"Category Match Score: {category_score:.2f}")
    print(f"Final Combined Similarity Score: {total_score:.4f}\n")

    for key, val in match_report.items():
        if val:
            print(f"- Shared {key.title()}: {', '.join(val)}")
    return total_score

# === Graph Similarity Computation ===
with open('ktp_members.json', 'r', encoding='utf-8') as f:
    raw_data = json.load(f)

members = []
for uid, data in raw_data.items():
    members.append({
        "id": uid,
        "name": data.get("name", ""),
        "profile_pic": data.get("profile_pic_link", ""),
        "resume_link": data.get("resume_link", "")
    })

nodes = [{
    "name": m["name"],
    "image": m["profile_pic"],
    "shape": "circularImage"
} for m in members]

with open('nodes.json', 'w', encoding='utf-8') as f:
    json.dump(nodes, f, indent=2)
print("Nodes saved")

def extract_pdf_text(url, name):
    try:
        response = requests.get(url)
        response.raise_for_status()
        with fitz.open(stream=response.content, filetype="pdf") as doc:
            return " ".join(page.get_text() for page in doc)
    except Exception as e:
        print(f"Failed to read for {name}, {e}")
        return ""

resume_texts = [extract_pdf_text(m["resume_link"], m["name"]) for m in members]
names = [m["name"] for m in members]

model = SentenceTransformer('all-mpnet-base-v2')
processed_texts = [text.lower().strip() for text in resume_texts]
resume_embeddings = model.encode(processed_texts, convert_to_tensor=True)
similarity_matrix = util.pytorch_cos_sim(resume_embeddings, resume_embeddings).cpu().numpy()

edges = []
for i, j in itertools.combinations(range(len(members)), 2):
    weight = 100 * similarity_matrix[i][j]
    if weight > 0 and weight < 100:
        edges.append({
            "from": names[i],
            "to": names[j],
            "weight": round(float(weight), 6),
        })

with open('edges.json', 'w', encoding='utf-8') as f:
    json.dump(edges, f, indent=2)
print("Edges saved")
