
import nltk
import re
import fitz  # PyMuPDF
import torch
import requests
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sentence_transformers import SentenceTransformer, util

# Download required NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Initialize stopwords
stop_words = set(stopwords.words('english'))

# Function to preprocess text
def preprocess(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text.lower())
    tokens = word_tokenize(text)
    return [t for t in tokens if t not in stop_words and len(t) > 1]

# Function to extract text from PDF
def extract_pdf_text(path):
    try:
        with fitz.open(path) as doc:
            return " ".join(page.get_text() for page in doc)
    except Exception as e:
        print(f"Error reading {path}: {e}")
        return ""

# --- Load two resumes ---
pdf_path_1 = "resume1.pdf"
pdf_path_2 = "resume2.pdf"

resume1 = extract_pdf_text(pdf_path_1)
resume2 = extract_pdf_text(pdf_path_2)

# --- Preprocess for overlapping term analysis ---
tokens1 = preprocess(resume1)
tokens2 = preprocess(resume2)
overlap = set(tokens1) & set(tokens2)

# --- Semantic similarity using Sentence-BERT ---
model = SentenceTransformer('all-mpnet-base-v2')
embeddings = model.encode([resume1, resume2], convert_to_tensor=True)
similarity_score = util.pytorch_cos_sim(embeddings[0], embeddings[1]).item()

# --- Output ---
print("=== Resume 1 Path ===", pdf_path_1)
print("=== Resume 2 Path ===", pdf_path_2)
print("\n=== Overlapping Terms ===")
print(", ".join(sorted(overlap)))
print(f"\n=== Semantic Similarity Score ===\n{similarity_score:.4f}")
