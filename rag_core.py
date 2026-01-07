# rag_core.py

import subprocess
import json
from langdetect import detect
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# =========================
# Vector DB & Embeddings
# =========================
embeddings = HuggingFaceEmbeddings(
    model_name="intfloat/multilingual-e5-large"
)

vectorstore = Chroma(
    collection_name="ncert_multilingual",
    embedding_function=embeddings,
    persist_directory="./chroma_ncert_db"
)

# =========================
# Conversation Memory
# =========================
conversation_memory = {
    "summaries": []
}



def generate_answer_from_chunks(question, chunks):
    """
    question: str
    chunks: List[{"text": str, "metadata": dict}]
    """

    answer = mistral_answer_with_citations(
        question=question,
        retrieved_docs=chunks,
        memory=conversation_memory
    )

    return {
        "answer": answer,
        "summary": mistral_summarise_answer(answer, detect_language(question)),
        "sources": extract_stable_sources(chunks)
    }







# =========================
# Language Detection
# =========================
def detect_language(text):
    try:
        return detect(text)
    except:
        return "en"

# =========================
# Query Expansion (SAFE)
# =========================
def mistral_query_expansion_ncert_multilingual(query, n_variants=3):
    prompt = f"""
Generate {n_variants} NCERT-style alternative queries
in the SAME LANGUAGE as the question.

Rules:
- Do NOT translate
- Do NOT answer
- Use textbook terminology only

Question:
{query}

Return ONLY a JSON list.
"""

    result = subprocess.run(
        ["ollama", "run", "mistral"],
        input=prompt.encode("utf-8"),
        capture_output=True
    )

    try:
        return json.loads(result.stdout.decode("utf-8").strip())
    except:
        return [query]

# =========================
# Retrieval for ANSWER (uses expansion)
# =========================
def retrieve_for_answer(query, grade, language, k=5):
    expanded = mistral_query_expansion_ncert_multilingual(query)
    docs = []

    normalized_grade = f"Class{grade}"

    filters = {
        "$and": [
            {"class": normalized_grade},
            {"language": language}
        ]
    }

    for q in expanded:
        results = vectorstore.similarity_search(
            str(q),
            k=k,
            filter=filters
        )
        docs.extend(results)

    unique = {d.page_content: d for d in docs}

    return [
        {"text": d.page_content, "metadata": d.metadata}
        for d in unique.values()
    ]

# =========================
# Retrieval for SOURCES (NO expansion)
# =========================
def retrieve_for_sources(query, grade, language, k=3):
    normalized_grade = f"Class{grade}"

    results = vectorstore.similarity_search(
        query,
        k=k,
        filter={
            "$and": [
                {"class": normalized_grade},
                {"language": language}
            ]
        }
    )

    return [
        {"text": d.page_content, "metadata": d.metadata}
        for d in results
    ]
# =========================
# Answer Generation (STRICT)
# =========================
def mistral_answer_with_citations(question, retrieved_docs, memory):
    lang = detect_language(question)
    language_name = {
        "hi": "Hindi (हिंदी)",
        "en": "English",
        "ml": "Malayalam (മലയാളം)",
        "mr": "Marathi (मराठी)",
        "ur": "Urdu (اردو)"
    }.get(lang, "the same language as the question")

    context = "\n\n".join(d["text"] for d in retrieved_docs[:5])
    previous = "\n".join(memory["summaries"][-3:])

    prompt = f"""
You are an NCERT-based tutor.

STRICT RULES (DO NOT VIOLATE):
- Answer ONLY in the SAME LANGUAGE as the question

- NEVER switch languages
- Use ONLY the provided NCERT context
- Do NOT add outside knowledge
- Cite ONLY the provided context

- Do NOT mention Physics, Biology, Chemistry as separate books

- Answer should be in 400 to 600 words and also as per  user requirements 

- If possible make sure the groundings are from ncert textbook only greater than 60 percent

- If the answer is not found, say EXACTLY:
    "I don't know the answer to this question based on the provided context."

Conversation so far (for context):
{previous if previous else "None"}

NCERT Context:
{context}

Question:
{question}

Answer (same language as question):
"""

    result = subprocess.run(
        ["ollama", "run", "mistral"],
        input=prompt.encode("utf-8"),
        capture_output=True
    )

    return result.stdout.decode("utf-8").strip()

# =========================
# Out-of-Scope Check
# =========================
def is_out_of_scope(answer):
    triggers = [
        "i don't know",
        "not found in the provided context",
        "cannot be answered based on"
    ]
    return any(t in answer.lower() for t in triggers)

# =========================
# Summarisation
# =========================
def mistral_summarise_answer(answer, lang):
    prompt = f"""
Summarise the following answer.

STRICT RULES:
- Use the SAME LANGUAGE as the answer

- Keep it short and NCERT-style 

- Only give summary in one line  in same language of answer


Answer:
{answer}
"""
    result = subprocess.run(
        ["ollama", "run", "mistral"],
        input=prompt.encode("utf-8"),
        capture_output=True
    )
    return result.stdout.decode("utf-8").strip()

# =========================
# Stable Source Extraction (1–3 only)
# =========================
def extract_stable_sources(source_docs, max_sources=3):
    sources = []
    for d in source_docs:
        src = d["metadata"].get("source")
        if src and src not in sources:
            sources.append(src)
        if len(sources) == max_sources:
            break
    return sources


def is_weak_retrieval(retrieved_docs, min_chunks=2, min_chars=300):
    if len(retrieved_docs) < min_chunks:
        return True

    total_chars = sum(len(d["text"]) for d in retrieved_docs)
    return total_chars < min_chars










# =========================
# FINAL API (used by app.py)
# =========================
def handle_student_query(question, grade):
    lang = detect_language(question)

    # 🔹 Retrieval (grade + language only)
    answer_docs = retrieve_for_answer(
        query=question,
        grade=grade,
        language=lang
    )

    answer = mistral_answer_with_citations(
        question,
        answer_docs,
        conversation_memory
    )

    # 🔹 Fallback (still show sources from checked chunks)
    if is_out_of_scope(answer) and is_weak_retrieval(answer_docs):
        return {
        "answer": answer,
        "summary": None,
        "sources": extract_stable_sources(answer_docs)
    }

    summary = mistral_summarise_answer(answer, lang)
    conversation_memory["summaries"].append(summary)

    return {
        "answer": answer,
        "summary": summary,
        "sources": extract_stable_sources(answer_docs)
    }