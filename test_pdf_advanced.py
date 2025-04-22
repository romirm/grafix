import fitz  # PyMuPDF
import re
import nltk
import torch
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer, util
from fuzzywuzzy import fuzz

# Download NLTK data
nltk.download('punkt')
nltk.download('stopwords')

stop_words = set(stopwords.words('english'))

# Step 1: Extract text from PDF
def extract_pdf_text(path):
    try:
        with fitz.open(path) as doc:
            return " ".join(page.get_text() for page in doc)
    except Exception as e:
        print(f"Error reading {path}: {e}")
        return ""

# Step 2: Preprocess text
def preprocess(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text.lower())
    tokens = word_tokenize(text)
    return [t for t in tokens if t not in stop_words and len(t) > 1]

# Step 3: Define resume-specific keywords to compare
KEY_CATEGORIES = {
    "skills": ["python", "c++", "aws", "react", "docker", "sql", "tensorflow", "git", "javascript", "firebase"],
    "interests": ["trading", "machine learning", "backend", "finance", "infrastructure", "data", "systems"],
    "experience_tags": ["netflix", "citadel", "amazon", "google", "northwestern financial technologies", "ktp", "mercor", "openai"]
}

# Step 4: Extract matching terms by category
def extract_matches(tokens, category_terms):
    tokens_set = set(tokens)
    matches = [term for term in category_terms if term in tokens_set]
    return matches

# Step 5: Resume file paths
pdf1 = "resume1.pdf"
pdf2 = "resume2.pdf"

text1 = extract_pdf_text(pdf1)
text2 = extract_pdf_text(pdf2)

tokens1 = preprocess(text1)
tokens2 = preprocess(text2)

# Step 6: Semantic similarity
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode([text1, text2], convert_to_tensor=True)
semantic_score = util.pytorch_cos_sim(embeddings[0], embeddings[1]).item()

# Step 7: Compute overlaps and fuzzy matches
common_words = set(tokens1) & set(tokens2)
fuzzy_overlap = [w1 for w1 in tokens1 for w2 in tokens2 if fuzz.partial_ratio(w1, w2) > 90]

# Step 8: Extract resume-specific category matches
category_matches = {}
for cat, terms in KEY_CATEGORIES.items():
    matches1 = extract_matches(tokens1, terms)
    matches2 = extract_matches(tokens2, terms)
    category_matches[cat] = list(set(matches1) & set(matches2))

# Output
print(f"=== Semantic Similarity Score ===\n{semantic_score:.4f}")
print("\n=== Shared Tokens ===")
print(", ".join(sorted(common_words)))
print("\n=== Fuzzy Matched Tokens ===")
print(", ".join(sorted(set(fuzzy_overlap))))
print("\n=== Resume Category Matches ===")
for cat, matches in category_matches.items():
    print(f"- {cat.title()}: {', '.join(matches) if matches else 'None'}")
