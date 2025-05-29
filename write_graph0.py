import json
import itertools
import requests
import fitz  # PyMuPDF
from io import BytesIO
from sentence_transformers import SentenceTransformer, util

# --- Step 1: Load member data from the JSON file ---
with open('ktp_members.json', 'r', encoding='utf-8') as f:
    raw_data = json.load(f)

# --- Step 2: Parse members into a list ---
members = []
for uid, data in raw_data.items():
    members.append({
        "id": uid,
        "name": data.get("name", ""),
        "profile_pic": data.get("profile_pic_link", ""),
        "resume_link": data.get("resume_link", "")
    })

# --- Step 3: Create graph nodes ---
nodes = [{
    "name": m["name"],
    "image": m["profile_pic"],
    "shape": "circularImage"
} for m in members]

with open('nodes.json', 'w', encoding='utf-8') as f:
    json.dump(nodes, f, indent=2)
print("Nodes saved")

# --- Step 4: Download and extract resume text ---
def extract_pdf_text(url, name):
    try:
        response = requests.get(url)
        response.raise_for_status()
        with fitz.open(stream=response.content, filetype="pdf") as doc:
            return " ".join(page.get_text() for page in doc)
    except Exception as e:
        return ""

resume_texts = [extract_pdf_text(m["resume_link"], m["name"]) for m in members]
names = [m["name"] for m in members]

# --- Step 5: Use Sentence-BERT for semantic embeddings ---
model = SentenceTransformer('all-mpnet-base-v2')
processed_texts = [text.lower().strip() for text in resume_texts]
resume_embeddings = model.encode(processed_texts, convert_to_tensor=True)

# --- Step 6: Compute cosine similarity matrix ---
similarity_matrix = util.pytorch_cos_sim(resume_embeddings, resume_embeddings).cpu().numpy()

# --- Step 7: Create graph edges from similarity matrix ---
edges = []
for i, j in itertools.combinations(range(len(members)), 2):
    weight = 100 * similarity_matrix[i][j]
    if weight > 0 and weight < 1:
        edges.append({
            "from": names[i],
            "to": names[j],
            "weight": round(float(weight), 6),
        })

with open('edges.json', 'w', encoding='utf-8') as f:
    json.dump(edges, f, indent=2)
print("Edges saved")
[]