import fitz  # PyMuPDF
import google.generativeai as genai
import json
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY not found. Make sure your .env file is correct.")

import google.generativeai as genai
genai.configure(api_key=api_key)
model = genai.GenerativeModel(model_name="models/gemini-1.5-pro-latest")


# === Step 1: Extract text from PDF ===
def extract_text(path):
    try:
        with fitz.open(path) as doc:
            return " ".join(page.get_text() for page in doc)
    except Exception as e:
        print(f"Error reading {path}: {e}")
        return ""

# === Step 2: Call Gemini API to extract structured info ===
def extract_resume_entities(text):
    prompt = f"""
    You are a resume analysis tool. Given the following resume text, extract key information in JSON format.
    The fields must include:
    - full_name (string)
    - graduation_year (string or int)
    - majors (list of strings)
    - experiences (list of org names, especially internships and clubs)
    - skills (list of tools/languages)
    - interests (list of topics or hobbies)

    Resume:
    {text}
    """
    response = model.generate_content(prompt)
    try:
        return json.loads(response.text)
    except json.JSONDecodeError:
        print("Gemini output could not be parsed. Raw output:")
        print(response.text)
        return {}

# === Step 3: Search Index Compatibility ===
def profile_contains_term(profile, term):
    term_lower = term.lower()
    return any(
        term_lower in str(value).lower()
        for field in ["experiences", "majors", "interests", "skills"]
        for value in profile.get(field, [])
    )

# === Step 4: Vectorized Similarity ===
def resume_similarity(profile1, profile2):
    prompt = f"""
    Given these two resumes, return a similarity score from 0 to 1. 
    Give more weight to shared internships, clubs, graduation year, and majors. 
    
    Resume A: {json.dumps(profile1)}
    Resume B: {json.dumps(profile2)}
    
    Respond only with a float between 0 and 1.
    """
    response = model.generate_content(prompt)
    try:
        return float(response.text.strip())
    except ValueError:
        print("Could not parse similarity score. Raw response:")
        print(response.text)
        return 0.0

# === Main Function ===
def compare_resumes(path1, path2):
    text1 = extract_text(path1)
    text2 = extract_text(path2)

    profile1 = extract_resume_entities(text1)
    profile2 = extract_resume_entities(text2)

    sim = resume_similarity(profile1, profile2)

    print("\n=== Resume Similarity Report ===\n")
    print(f"Final Similarity Score: {sim:.4f}")
    print(f"Common Experiences: {set(profile1.get('experiences', [])) & set(profile2.get('experiences', []))}")
    print(f"Common Majors: {set(profile1.get('majors', [])) & set(profile2.get('majors', []))}")
    print(f"Common Interests: {set(profile1.get('interests', [])) & set(profile2.get('interests', []))}")
    return sim

if __name__ == "__main__":
    compare_resumes("resume_paari.pdf", "resume_rishabh.pdf")
