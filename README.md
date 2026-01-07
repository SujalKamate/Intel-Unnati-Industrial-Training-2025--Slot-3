📘NCERT Hybrid Retrieval‑Augmented Generation (RAG) System

Intel® Unnati Industrial Training Program 2025
A curriculum‑aligned, multimodal NCERT question‑answering system built using Hybrid Retrieval + RAG, supporting text, OCR, and voice queries, with grade‑specific filtering (Class 5–10) and strict answer grounding.



📋TL;DR (Short Overview)



| Item             | Description                                                      |
| ---------------- | ---------------------------------------------------------------- |
| What is it?      | A Streamlit‑based Hybrid RAG system for NCERT textbooks          |
| Why?             | To provide accurate, grade‑specific, curriculum‑grounded answers |
| Core Idea        | Dense + Sparse Hybrid Retrieval + Controlled Generation          |
| Input Modes      | Text, Image (OCR), Voice (Whisper)                               |
| Grades Supported | Class 5 to Class 10                                              |
| Vector Store     | ChromaDB                                                         |
| Lines of Code    | ~3,000+ (Python)                                                 |
| License          | Academic / Educational                                           |






🎯Problem Statement → Implementation Mapping



| Requirement            | Implementation                       | Location                                           |
| ---------------------- | ------------------------------------ | -------------------------------------------------- |
| Curriculum‑aligned QA  | NCERT‑only context grounding         | `rag_core.py`                                      |
| Grade‑specific answers | Metadata filtering (Class 5–10)      | `app.py`, `rag_core.py`                            |
| Hybrid Retrieval       | Dense embeddings + TF‑IDF            | `build_doc_embeddings.py`, `build_sparse_index.py` |
| RAG Answer Generation  | Controlled prompts, no hallucination | `rag_core.py`                                      |
| OCR support            | Tesseract + OpenCV                   | `app.py`                                           |
| Voice input            | Whisper speech‑to‑text               | `app.py`                                           |
| Re‑ranking             | MMR (diversity + relevance)          | `app.py`                                           |
| Evaluation             | Metric + LLM‑based evaluation        | `evaluate_answers.py`, `llm_evaluate.py`           |



▶️End-to-End Workflow

<img width="2013" height="1179" alt="image" src="https://github.com/user-attachments/assets/2609d4ed-4660-4fe3-949a-6828cc470b13" />




🏗️ System Architecture

<img width="2179" height="991" alt="image" src="https://github.com/user-attachments/assets/4b3627f3-6aae-4770-91d5-fda7ca4a988a" />



🚀 Key Features

📚 NCERT‑only, curriculum‑grounded answers

🎯 Grade‑specific filtering (Class 5–10)

🔎 Hybrid Retrieval (Dense + Sparse)

🧠 MMR‑based re‑ranking

🖼️ OCR support for textbook images

🎤 Voice input using Whisper

🌍 Multilingual question support

📝 Answer summarization

📂 Source citation

📊 Automated + LLM‑based evaluation





## 📁 Project Structure

```text
Intel-AI-Unnati-NCERT-RAG/
│
├── app.py
│   └── Streamlit frontend
│       - Text, OCR, and Voice-based query input
│       - Hybrid dense + sparse retrieval
│       - MMR re-ranking
│       - Grade-specific filtering (Class 5–10)
│       - Displays answers, summaries, sources, and feedback
│
├── rag_core.py
│   └── Core RAG backend
│       - NCERT-only answer generation
│       - Query expansion
│       - Language detection
│       - Controlled prompt design
│       - Lightweight conversation memory (summaries only)
│
├── build_doc_embeddings.py
│   └── Offline preprocessing script
│       - Generates dense embeddings for NCERT chunks
│       - Stores embeddings for fast reuse
│
├── build_sparse_index.py
│   └── Offline preprocessing script
│       - Builds TF-IDF sparse index
│       - Enables keyword-based retrieval
│
├── doc_embeddings.pkl        (ignored in Git)
├── sparse_index.pkl          (ignored in Git)
│
├── evaluate_answers.py
│   └── Automated evaluation script
│       - Quantitative evaluation of generated answers
│
├── llm_evaluate.py
│   └── LLM-based qualitative evaluation
│       - Assesses relevance, faithfulness, and completeness
│
├── MiniOne.ipynb
│   └── End-to-end experimentation notebook
│       - Data ingestion and cleaning
│       - OCR extraction
│       - Chunking and embedding generation
│       - ChromaDB storage
│
├── base.ipynb
│   └── Initial baseline experiments
│
├── BaseProject.ipynb
│   └── Consolidated end-to-end workflow notebook
│
├── query_expansion_and_ans_generation.ipynb
│   └── Experiments with query expansion and answer generation
│
├── requirements.txt
│   └── Project dependencies
│
├── README.md
│   └── Project documentation
│
├── .gitignore
│   └── Git ignore rules for large data and generated artifacts
│
├── chroma_ncert_db/           (ignored in Git)
│   └── Persistent Chroma vector database
│
├── NcertData/                 (ignored in Git)
│   ├── Class5/
│   ├── Class6/
│   ├── Class7/
│   ├── Class8/
│   ├── Class9/
│   └── Class10/
│
├── voice_query.wav            (ignored in Git)
├── voice_query.json           (ignored in Git)
│
├── __pycache__/               (ignored in Git)
├── .venv/                     (ignored in Git)
└── .python-version
```


























