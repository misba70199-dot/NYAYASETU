"""
NyayaSetu — AI-Based Legal Triage and Justice Navigation System
Pipeline: TF-IDF classifier -> category-scoped statute retrieval -> Groq-generated grounded answer
"""

import os
import json

import pandas as pd
import requests
import streamlit as st
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics.pairwise import cosine_similarity

# Anchor paths to this script's folder, not the current working directory —
# `streamlit run` can be launched from a different cwd, which otherwise breaks
# relative paths like "Dataset_1_finalized.xlsx" even when the file sits right
# next to app.py.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "NyayaSetu_v2_500_balanced_verified_sources.xlsx")
AUGMENT_PATH = os.path.join(BASE_DIR, "NyayaSetu_targeted_augmentation_100.xlsx")
CORPUS_PATH = os.path.join(BASE_DIR, "statute_corpus_v1.json")
GROQ_MODEL = "openai/gpt-oss-120b"

# Loads GROQ_API_KEY from a .env file sitting next to this script (see .env.example)
load_dotenv(os.path.join(BASE_DIR, ".env"))

st.set_page_config(page_title="NyayaSetu - Legal Triage", page_icon="⚖️", layout="centered")


# ---------- Cached setup (runs once per app session) ----------

@st.cache_resource
def load_classifier():
    """Train the TF-IDF + Linear SVM category classifier on the labeled dataset.

    Linear SVM scored 84% held-out accuracy in testing (vs ~76% for Logistic
    Regression) on this dataset. LinearSVC has no predict_proba, so it's wrapped
    in CalibratedClassifierCV to get real probabilities for the confidence score
    shown in the UI. The base + targeted-augmentation sets are merged and used
    in full here (unlike the notebook's held-out evaluation split) since this is
    the deployed model, not an accuracy test.
    """
    if not os.path.isfile(DATA_PATH):
        raise FileNotFoundError(
            f"Training dataset not found: {DATA_PATH}. "
            "Place the Excel file beside app.py or update DATA_PATH."
        )
    if not os.path.isfile(AUGMENT_PATH):
        raise FileNotFoundError(
            f"Augmentation dataset not found: {AUGMENT_PATH}. "
            "Place the Excel file beside app.py or update AUGMENT_PATH."
        )

    df = pd.read_excel(DATA_PATH, sheet_name="Legal_Problems_500")
    aug_df = pd.read_excel(AUGMENT_PATH, sheet_name="Targeted_100")

    X = pd.concat(
        [df["example_user_problem"], aug_df["synthetic_query"]], ignore_index=True
    ).astype(str)
    y = pd.concat([df["category"], aug_df["category"]], ignore_index=True).astype(str)

    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True)
    X_vec = vectorizer.fit_transform(X)

    base_clf = LinearSVC(C=1)
    clf = CalibratedClassifierCV(base_clf, method="sigmoid", cv=5)
    clf.fit(X_vec, y)
    return vectorizer, clf


@st.cache_resource
def load_corpus():
    """Load and vectorize the statute corpus used for retrieval."""
    if not os.path.isfile(CORPUS_PATH):
        raise FileNotFoundError(
            f"Statute corpus not found: {CORPUS_PATH}. "
            "Place the JSON file beside app.py or update CORPUS_PATH."
        )
    with open(CORPUS_PATH, encoding="utf-8") as f:
        corpus = json.load(f)
    texts = [f"{c['section_title']} {c['text']}" for c in corpus]
    corpus_vectorizer = TfidfVectorizer(stop_words="english")
    corpus_vectors = corpus_vectorizer.fit_transform(texts)
    return corpus, corpus_vectorizer, corpus_vectors


# ---------- Pipeline steps ----------

def classify_category(query, vectorizer, clf):
    vec = vectorizer.transform([query])
    probs = clf.predict_proba(vec)[0]
    idx = probs.argmax()
    return clf.classes_[idx], probs[idx]


def search_corpus(query, predicted_category, corpus, corpus_vectorizer, corpus_vectors, top_k=3):
    """Category-scoped retrieval: trust the classifier's category, don't filter by
    cosine score (legal text vocabulary rarely overlaps lexically with plain-language
    queries, so a raw similarity threshold discards correct matches and can let
    unrelated higher-scoring chunks from other categories leak in)."""
    query_vec = corpus_vectorizer.transform([query])
    scores = cosine_similarity(query_vec, corpus_vectors)[0]
    ranked = sorted(zip(corpus, scores), key=lambda x: x[1], reverse=True)
    matches = [c for c, _ in ranked if c["category"] == predicted_category]
    return matches[:top_k]


def call_groq(query, category, confidence, chunks, api_key):
    if chunks:
        context = "\n\n".join(
            f"[{c['chunk_id']}] {c['act']}, {c['section']} - {c['section_title']}\n{c['text']}"
            for c in chunks
        )
    else:
        context = "No specific statutory text was retrieved for this category."

    system_prompt = (
        "You are a legal-information triage assistant for India, not a lawyer. "
        "Answer using ONLY the retrieved statutory text provided. Do not invent "
        "section numbers, Acts, or advice beyond what is given. Cite the Act and "
        "Section for every claim you make. Clearly state this is legal-information "
        "triage, not legal advice, and that the user should consult NALSA/their "
        "Legal Services Authority or a lawyer for their specific case."
    )
    user_prompt = (
        f"User's problem: {query}\n\n"
        f"Predicted category: {category} (confidence: {confidence:.0%})\n\n"
        f"Retrieved statutory text:\n{context}\n\n"
        "Based only on the above, explain in plain language: "
        "1) what legal process/forum applies, 2) what documents they likely need, "
        "3) which Act and Section this falls under. Keep it under 200 words."
    )

    response = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={
            "model": GROQ_MODEL,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.2,
        },
        timeout=30,
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]


# ---------- UI ----------

st.title("⚖️ NyayaSetu")
st.caption("AI-Based Legal Triage and Justice Navigation System — legal-information triage, not legal advice.")

api_key = os.environ.get("GROQ_API_KEY")
query = st.text_area("Describe your legal problem in plain language:", height=100,
                      placeholder="e.g. I bought a phone online and it arrived broken, the seller won't refund me.")

if st.button("Get Triage Guidance", type="primary"):
    if not query.strip():
        st.warning("Please describe your problem first.")
    elif not api_key:
        st.error("Application configuration error: API key is not configured.")
    else:
        vectorizer, clf = load_classifier()
        corpus, corpus_vectorizer, corpus_vectors = load_corpus()

        category, confidence = classify_category(query, vectorizer, clf)
        chunks = search_corpus(query, category, corpus, corpus_vectorizer, corpus_vectors)

        st.subheader("Predicted Category")
        st.write(f"**{category}**  ·  confidence: {confidence:.0%}")
        if confidence < 0.40:
            st.warning(
                "⚠️ Category confidence is low — the system is not fully sure this is "
                "the right category. Please verify manually or rephrase with more detail."
            )

        with st.spinner("Generating grounded guidance from retrieved statutes..."):
            try:
                answer = call_groq(query, category, confidence, chunks, api_key)
                st.subheader("Guidance")
                st.write(answer)
            except requests.exceptions.HTTPError as e:
                detail = e.response.text
                try:
                    body = e.response.json()
                    if isinstance(body, dict):
                        detail = body.get("error", {}).get("message", detail) if isinstance(
                            body.get("error"), dict
                        ) else body.get("error", detail)
                except ValueError:
                    pass
                st.error(f"Groq API returned an error (status {e.response.status_code}): {detail}")
            except requests.exceptions.RequestException as e:
                st.error(f"Groq API call failed: {e}")

        st.subheader("Retrieved Legal Sources")
        if chunks:
            for c in chunks:
                with st.expander(f"{c['act']} — {c['section']}"):
                    st.write(f"**{c['section_title']}**")
                    st.write(c["text"])
                    st.markdown(f"[View on India Code]({c['source_url']})")
        else:
            st.info("No matching statutory chunk was found in the corpus for this category yet.")

        st.divider()
        st.caption(
            "⚠️ This tool provides legal-information triage only, not legal advice. "
            "For guidance on your specific case, contact NALSA (helpline 15100) or "
            "your nearest District/State Legal Services Authority."
        )
