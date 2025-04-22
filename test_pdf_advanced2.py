import fitz  # PyMuPDF
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer, util
from sklearn.feature_extraction.text import TfidfVectorizer
from fuzzywuzzy import fuzz

nltk.download('punkt')
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

model = SentenceTransformer("all-MiniLM-L6-v2")

def extract_pdf_text(path):
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

def extract_keywords(text, top_k=20):
    vectorizer = TfidfVectorizer(stop_words='english', max_features=top_k)
    X = vectorizer.fit_transform([text])
    return vectorizer.get_feature_names_out()

def extract_education_info(text):
    grad_year = re.findall(r'expected\s+jun\.*\s*(20\d{2})', text.lower())
    majors = re.findall(r'(computer science|mathematics|finance|electrical engineering|data science)', text.lower())
    return {"grad_year": grad_year, "majors": majors}

def extract_affiliations(text):
    # Extract capitalized multi-word orgs, e.g., Northwestern Financial Technologies
    return re.findall(r'\b(?:[A-Z][a-z]+(?:\s[A-Z][a-z]+)+)', text)

def compare_and_score(keywords1, keywords2, weights):
    score = 0.0
    matches = {}
    for cat, weight in weights.items():
        shared = list(set(keywords1.get(cat, [])) & set(keywords2.get(cat, [])))
        matches[cat] = shared
        score += weight * len(shared)
    return score, matches

# --- Resume Paths ---
pdf1 = "resume1.pdf"
pdf2 = "resume2.pdf"

text1 = extract_pdf_text(pdf1)
text2 = extract_pdf_text(pdf2)

tokens1 = preprocess(text1)
tokens2 = preprocess(text2)

# --- Semantic Similarity ---
embeddings = model.encode([text1, text2], convert_to_tensor=True)
semantic_score = util.pytorch_cos_sim(embeddings[0], embeddings[1]).item()

# --- Automatic Feature Extraction ---
tfidf_keywords1 = extract_keywords(text1)
tfidf_keywords2 = extract_keywords(text2)

education1 = extract_education_info(text1)
education2 = extract_education_info(text2)

affiliations1 = extract_affiliations(text1)
affiliations2 = extract_affiliations(text2)

# --- Weighted Score ---
weights = {
    "tfidf": 1.0,
    "grad_year": 3.0,
    "majors": 2.5,
    "affiliations": 2.0,
}

keywords1 = {
    "tfidf": tfidf_keywords1,
    "grad_year": education1["grad_year"],
    "majors": education1["majors"],
    "affiliations": affiliations1,
}

keywords2 = {
    "tfidf": tfidf_keywords2,
    "grad_year": education2["grad_year"],
    "majors": education2["majors"],
    "affiliations": affiliations2,
}

weighted_score, category_matches = compare_and_score(keywords1, keywords2, weights)

# --- Additional Overlap Analysis ---
common_words = set(tokens1) & set(tokens2)
fuzzy_overlap = [w1 for w1 in tokens1 for w2 in tokens2 if fuzz.partial_ratio(w1, w2) > 90]

# --- Output ---
print(f"=== Weighted Resume Score ===\n{weighted_score:.2f}\n")

print("=== Shared Tokens ===")
print(", ".join(sorted(common_words)))

print("\n=== Fuzzy Matched Tokens ===")
print(", ".join(sorted(set(fuzzy_overlap))))

print("\n=== Inferred Category Matches ===")
for cat, matches in category_matches.items():
    print(f"- {cat.title()}: {', '.join(matches) if matches else 'None'}")
