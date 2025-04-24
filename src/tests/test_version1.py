
import nltk
import re
import torch
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sentence_transformers import SentenceTransformer, util

# Download required NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Initialize stopwords
stop_words = set(stopwords.words('english'))

# Sample resumes
resume1 = """
Software engineer with experience in backend development using Python, Flask, and PostgreSQL.
Previously worked at Google and contributed to scalable microservices architecture.
"""

resume2 = """
Backend developer skilled in Python, Django, and relational databases like MySQL and PostgreSQL.
Built systems at Meta and focused on clean API design and scalable services.
"""

# Step 1: Preprocess and extract overlapping terms
def preprocess(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text.lower())
    tokens = word_tokenize(text)
    return [t for t in tokens if t not in stop_words and len(t) > 1]

tokens1 = preprocess(resume1)
tokens2 = preprocess(resume2)

overlap = set(tokens1) & set(tokens2)

# Step 2: Semantic similarity
model = SentenceTransformer('all-mpnet-base-v2')
embeddings = model.encode([resume1, resume2], convert_to_tensor=True)
similarity_score = util.pytorch_cos_sim(embeddings[0], embeddings[1]).item()

# Output
print("=== Resume 1 ===\n", resume1)
print("\n=== Resume 2 ===\n", resume2)
print("\n=== Overlapping Terms ===")
print(", ".join(sorted(overlap)))
print(f"\n=== Semantic Similarity Score ===\n{similarity_score:.4f}")
