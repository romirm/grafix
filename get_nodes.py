import json
import itertools
import requests
import fitz  # PyMuPDF
from io import BytesIO
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- Load the KTP member JSON ---
with open('ktp_members.json', 'r', encoding='utf-8') as f:
    raw_data = json.load(f)

# --- Parse valid members ---
members = []
for uid, data in raw_data.items():
    if not data.get("signed_up") or "resume_link" not in data:
        continue
    members.append({
        "id": uid,
        "name": data.get("name", "Unknown"),
        "profile_pic": data.get("profile_pic_link", ""),
        "resume_link": data["resume_link"]
    })

# --- Step 1: Create nodes ---
nodes = [{
    "id": m["id"],
    "label": m["name"],
    "image": m["profile_pic"],
    "shape": "circularImage"
} for m in members]

with open('nodes.json', 'w', encoding='utf-8') as f:
    json.dump(nodes, f, indent=2)
print("✅ Nodes saved to nodes.json")

# --- Step 2: Download and extract resume text ---
def extract_pdf_text(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        with fitz.open(stream=response.content, filetype="pdf") as doc:
            return " ".join(page.get_text() for page in doc)
    except Exception as e:
        print(f"⚠️ Failed to extract text from {url[:60]}...: {e}")
        return ""

resume_texts = [extract_pdf_text(m["resume_link"]) for m in members]
ids = [m["id"] for m in members]

# --- Step 3: Vectorize + compute cosine similarity ---
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(resume_texts)
similarity_matrix = cosine_similarity(tfidf_matrix)

# --- Step 4: Create edges based on similarity ---
edges = []
for i, j in itertools.combinations(range(len(members)), 2):
    weight = similarity_matrix[i][j]
    if weight > 0.15:  # tune threshold for graph density
        edges.append({
            "from": ids[i],
            "to": ids[j],
            "value": round(weight, 3),
            "label": f"{round(weight, 2)}"
        })

# --- Step 5: Save graph.json ---
graph = {
    "nodes": nodes,
    "edges": edges
}

with open('graph.json', 'w', encoding='utf-8') as f:
    json.dump(graph, f, indent=2)

print("✅ Graph saved to graph.json")
