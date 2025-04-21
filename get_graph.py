import json
import itertools
import requests
import fitz  # PyMuPDF
from io import BytesIO
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load member data from the json file
with open('ktp_members.json', 'r', encoding='utf-8') as f:
    raw_data = json.load(f)

# Parsing members
members = []
for uid, data in raw_data.items():
    members.append({
        "id": uid,
        "name": data.get("name", ""), # safer with .get in case it's missing
        "profile_pic": data.get("profile_pic_link", ""),
        "resume_link": data.get("resume_link", "") 
    })


# Creating Nodes
nodes = [{
    "name": m["name"],
    "image": m["profile_pic"],
    "shape": "circularImage"
} for m in members]

# Creating Nodes JSON
with open('nodes.json', 'w', encoding='utf-8') as f:
    json.dump(nodes, f, indent=2)
print("Nodes saved")


# Downloading and extracting 
def extract_pdf_text(url, name):
    try:
        response = requests.get(url)
        response.raise_for_status()
        with fitz.open(stream=response.content, filetype="pdf") as doc:
            return " ".join(page.get_text() for page in doc)
    except Exception as e:
        print(f"Failed to read for {name}, {e}") # Tells which people have error with resume parsing
        return ""

# Extracts resume information as text and tells who has resume errors
resume_texts = [extract_pdf_text(m["resume_link"], m["name"]) for m in members]
names = [m["name"] for m in members]



# --- Step 3: Vectorize + compute cosine similarity ---
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(resume_texts)
similarity_matrix = cosine_similarity(tfidf_matrix)

# --- Step 4: Create edges based on similarity ---
edges = []
for i, j in itertools.combinations(range(len(members)), 2):
    weight = similarity_matrix[i][j]
    if weight > 0.10:  # tuning the threshold for graph density
        edges.append({
            "from": names[i],
            "to": names[j],
            "weight": round(weight, 6),
        })



with open('edges.json', 'w', encoding='utf-8') as f:
    json.dump(edges, f, indent=2)
print("Edges saved")
