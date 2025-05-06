import fitz  # PyMuPDF
import google.generativeai as genai
import json
from dotenv import load_dotenv
import os
import re
from difflib import SequenceMatcher

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
    - skills (list of tools/languages — deduplicated)
    - interests (list of topics or hobbies, leave empty if none are listed)

    IMPORTANT:
    - Only include each key once in the final JSON.
    - Do not repeat fields like "skills".
    - Ensure the output is valid JSON, not Markdown or wrapped in triple backticks.
    - Only output the json, no other labels.

    Resume:
    {text}
    """
    response = model.generate_content(prompt)
    raw_text = response.text.strip()

    # Remove Markdown wrapping like ```json and ```
    if raw_text.startswith("```json"):
        raw_text = raw_text[7:]
    if raw_text.startswith("```"):
        raw_text = raw_text[3:]
    if raw_text.endswith("```"):
        raw_text = raw_text[:-3]

    try:
        return json.loads(raw_text)
    except json.JSONDecodeError as e:
        print("Gemini output could not be parsed. Raw output:")
        print(raw_text)
        raise ValueError(f"Failed to parse JSON: {e}")

# === Step 3: Search Index Compatibility ===
def profile_contains_term(profile, term):
    term_lower = term.lower()
    return any(
        term_lower in str(value).lower()
        for field in ["experiences", "majors", "interests", "skills"]
        for value in profile.get(field, [])
    )

# === Step 4: Check for similarities in sets ===
def fuzzy_match(set1, set2, threshold=0.6):
    matches = set()
    for a in set1:
        for b in set2:
            if SequenceMatcher(None, a.lower(), b.lower()).ratio() > threshold:
                matches.add((a, b))
    return matches

# === Step 4: Vectorized Similarity ===
def resume_similarity(profile1, profile2):
    # Convert string "N/A" or empty lists to empty sets
    majors1 = set(profile1.get("majors", []))
    majors2 = set(profile2.get("majors", []))
    skills1 = set(profile1.get("skills", []))
    skills2 = set(profile2.get("skills", []))
    exp1 = set(profile1.get("experiences", []))
    exp2 = set(profile2.get("experiences", []))
    interests1 = set(profile1.get("interests", [])) if isinstance(profile1.get("interests"), list) else set()
    interests2 = set(profile2.get("interests", [])) if isinstance(profile2.get("interests"), list) else set()

    # Set-based intersections
    common_majors = majors1 & majors2
    common_skills = skills1 & skills2
    common_interests = interests1 & interests2

    fuzzy_experiences = fuzzy_match(exp1, exp2)
    common_experience_count = len(fuzzy_experiences)

    score = (
        1.0 * len(common_majors) +
        1.0 * len(common_skills) +
        1.5 * common_experience_count +
        0.5 * len(common_interests)
    )
    max_score = (
        1.0 * max(len(majors1), len(majors2), 1) +
        1.0 * max(len(skills1), len(skills2), 1) +
        1.5 * max(len(exp1), len(exp2), 1) +
        0.5 * max(len(interests1), len(interests2), 1)
    )

    similarity_score = round(score / max_score, 4)

    # Optional: Print debug info
    print("=== Resume Similarity Report ===")
    print("Common Majors:", common_majors)
    print("Common Skills:", common_skills)
    print("Fuzzy Experience Matches:", fuzzy_experiences)
    print("Common Interests:", common_interests)
    print("Final Similarity Score:", similarity_score)

    return similarity_score

# === Main Function ===
def compare_resumes(path1, path2):
    text1 = extract_text(path1)
    text2 = extract_text(path2)

    profile1 = extract_resume_entities(text1)
    profile2 = extract_resume_entities(text2)

    # Save JSON outputs
    with open("parsed_resume_1.json", "w") as f:
        json.dump(profile1, f, indent=2)
    with open("parsed_resume_2.json", "w") as f:
        json.dump(profile2, f, indent=2)

    print("\n=== Parsed Resume 1 ===")
    print(json.dumps(profile1, indent=2))
    print("\n=== Parsed Resume 2 ===")
    print(json.dumps(profile2, indent=2))

    sim = resume_similarity(profile1, profile2)

    print("\n=== Resume Similarity Report ===\n")
    print(f"Final Similarity Score: {sim:.4f}")
    print(f"Experiences: {set(profile1.get('experiences', [])) & set(profile2.get('experiences', []))}")
    print(f"Majors: {set(profile1.get('majors', [])) & set(profile2.get('majors', []))}")
    print(f"Interests: {set(profile1.get('interests', [])) & set(profile2.get('interests', []))}")
    return sim



if __name__ == "__main__":
    compare_resumes("resumes/resume_steve.pdf", "resumes/resume_paari.pdf")
