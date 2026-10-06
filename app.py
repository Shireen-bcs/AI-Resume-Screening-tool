
import streamlit as st
import fitz  # PyMuPDF
import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="AI Resume Screening Tool", layout="centered")

st.title("📄 AI-Powered Resume Screening Tool - Yhills")
st.write("Upload a job description and multiple resumes (PDF), and get ranked matches.")

# Upload job description
jd_file = st.file_uploader("Upload Job Description (TXT or PDF)", type=["txt", "pdf"])

# Upload multiple resumes
resumes = st.file_uploader("Upload Resumes (PDF)", type="pdf", accept_multiple_files=True)

def extract_text(file):
    if file.name.endswith(".txt"):
        return file.read().decode("utf-8")
    elif file.name.endswith(".pdf"):
        doc = fitz.open(stream=file.read(), filetype="pdf")
        text = ""
        for page in doc:
            text += page.get_text()
        return text
    return ""

if jd_file and resumes:
    jd_text = extract_text(jd_file)
    resume_data = []

    for res_file in resumes:
        res_text = extract_text(res_file)
        resume_data.append((res_file.name, res_text))

    texts = [jd_text] + [r[1] for r in resume_data]
    names = [r[0] for r in resume_data]

    # TF-IDF and cosine similarity
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(texts)
    similarity = cosine_similarity(vectors[0:1], vectors[1:]).flatten()

    result_df = pd.DataFrame({
        "Resume Name": names,
        "Matching Score (%)": [round(score * 100, 2) for score in similarity]
    }).sort_values(by="Matching Score (%)", ascending=False)

    st.success("Ranking Completed ✅")
    st.dataframe(result_df)

    csv = result_df.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Download Results as CSV", csv, "ranked_resumes.csv", "text/csv")
