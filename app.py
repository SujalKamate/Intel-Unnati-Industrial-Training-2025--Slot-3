import streamlit as st
import numpy as np
import re
import cv2
import pickle
import whisper
import datetime
import pandas as pd

from PIL import Image
import pytesseract
from st_audiorec import st_audiorec
from sklearn.feature_extraction.text import TfidfVectorizer
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

from rag_core import generate_answer_from_chunks  # ❗ DO NOT CHANGE



if "rag_result" not in st.session_state:
    st.session_state.rag_result = None

if "feedback" not in st.session_state:
    st.session_state.feedback = []



st.set_page_config(
    page_title="NCERT Hybrid RAG",
    layout="wide"
)

st.title("📘 NCERT Hybrid RAG System")

@st.cache_resource
def load_sparse_index():
    with open("sparse_index.pkl", "rb") as f:
        return pickle.load(f)

@st.cache_resource
def load_doc_embeddings():
    with open("doc_embeddings.pkl", "rb") as f:
        return pickle.load(f)

vectorizer, tfidf_matrix, all_docs, all_metas = load_sparse_index()
doc_embeddings = load_doc_embeddings()

embeddings = HuggingFaceEmbeddings(
    model_name="intfloat/multilingual-e5-large"
)

vectorstore = Chroma(
    collection_name="ncert_multilingual",
    embedding_function=embeddings,
    persist_directory="./chroma_ncert_db"
)


LANGUAGE_MAP = {
    "English": "eng",
    "Hindi": "hin",
    "Malayalam": "mal",
    "Marathi": "mar",
    "Urdu": "urd"
}

query_mode = st.radio(
    "Choose input method",
    ["✍️ Text", "📷 OCR", "🎤 Voice"],
    horizontal=True
)

final_query = None


if query_mode == "✍️ Text":
    q = st.text_area("Enter your question")
    if q.strip():
        final_query = q.strip()


elif query_mode == "📷 OCR":
    lang = st.selectbox("OCR Language", list(LANGUAGE_MAP.keys()))
    img = st.file_uploader("Upload image", type=["png", "jpg", "jpeg"])

    if img:
        image = Image.open(img)
        st.image(image, use_container_width=True)

        gray = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2GRAY)
        raw = pytesseract.image_to_string(
            gray,
            lang=LANGUAGE_MAP[lang],
            config="--psm 6"
        )

        cleaned = re.sub(r"\s+", " ", raw).strip()
        st.text_area("Extracted text", cleaned)

        if cleaned:
            final_query = cleaned

elif query_mode == "🎤 Voice":
    st.info("Record your question")

    wav_audio_data = st_audiorec()
    if wav_audio_data is not None:
        with open("voice_query.wav", "wb") as f:
            f.write(wav_audio_data)

        @st.cache_resource
        def load_whisper():
            return whisper.load_model("base")

        with st.spinner("Transcribing..."):
            model = load_whisper()
            out = model.transcribe("voice_query.wav", fp16=False)

        final_query = out["text"].strip()
        st.success(final_query)

def hybrid_search(query, alpha=0.7, k=10):
    dense = vectorstore.similarity_search_with_score(query, k=k)
    dense_scores = {d.page_content: 1 / (1 + s) for d, s in dense}

    q_vec = vectorizer.transform([query])
    sparse_raw = (tfidf_matrix @ q_vec.T).toarray().ravel()
    top_idx = np.argsort(-sparse_raw)[:k]
    sparse_scores = {all_docs[i]: sparse_raw[i] for i in top_idx}

    combined = {}
    for t, s in dense_scores.items():
        combined[t] = combined.get(t, 0) + alpha * s
    for t, s in sparse_scores.items():
        combined[t] = combined.get(t, 0) + (1 - alpha) * s

    ranked = sorted(combined.items(), key=lambda x: x[1], reverse=True)
    return [
        {"text": t, "metadata": all_metas[all_docs.index(t)]}
        for t, _ in ranked[:k]
    ]


if final_query:
    st.divider()
    alpha = st.slider("Dense vs Sparse (α)", 0.0, 1.0, 0.7)
    if st.button("🔎 Retrieve & Answer"):

        chunks = hybrid_search(final_query, alpha=alpha)

        with st.spinner("Generating answer..."):
            st.session_state.rag_result = generate_answer_from_chunks(
                question=final_query,
                chunks=chunks
            )

        st.rerun()


if st.session_state.rag_result:
    res = st.session_state.rag_result

    st.subheader("📘 Answer")
    st.markdown(res["answer"])

    if res.get("summary"):
        st.subheader("📝 Summary")
        st.info(res["summary"])

    if res.get("sources"):
        st.subheader("📂 Sources")
        for s in res["sources"]:
            st.write(f"- {s}")


st.divider()
st.subheader("🧠 Was this helpful?")

col1, col2 = st.columns(2)

with col1:
    if st.button("👍 Helpful"):
        st.session_state.feedback.append({
            "question": final_query,
            "answer": res["answer"],
            "rating": "positive",
            "timestamp": datetime.datetime.now().isoformat()
        })
        st.success("Thanks!")

with col2:
    if st.button("👎 Not Helpful"):
        st.session_state.feedback.append({
            "question": final_query,
            "answer": res["answer"],
            "rating": "negative",
            "timestamp": datetime.datetime.now().isoformat()
        })
        st.success("Thanks!")


if st.session_state.feedback:
    df = pd.DataFrame(st.session_state.feedback)
    st.sidebar.download_button(
        "⬇️ Download Feedback",
        df.to_csv(index=False),
        "feedback.csv"
    )