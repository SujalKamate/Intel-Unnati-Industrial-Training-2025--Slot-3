# 📘 **NCERT HYBRID RETRIEVAL‑AUGMENTED GENERATION (RAG) SYSTEM**
### **Intel® Unnati Industrial Training Program 2025**



Intel® Unnati Industrial Training Program 2025
A curriculum‑aligned, multimodal NCERT question‑answering system built using Hybrid Retrieval + RAG, supporting text, OCR, and voice queries, with grade‑specific filtering (Class 5–10) and strict answer grounding.

Projet Overview

This project was developed as part of the Intel® Unnati Industrial Training Program with the objective of building a multilingual, grade‑specific NCERT doubt‑solving system using a Retrieval‑Augmented Generation (RAG) approach. The system is designed to answer student queries strictly from NCERT textbooks for Classes 5 to 10, ensuring factual correctness and preventing hallucinations. The workflow begins with multi‑modal user input, where questions can be provided via text, OCR‑extracted images, or voice recordings. Voice queries are transcribed using Whisper, while OCR inputs are processed using OpenCV and Tesseract. Once the query is obtained, the language is detected and expanded into NCERT‑style reformulations to improve retrieval recall. A hybrid retrieval mechanism is then employed, combining dense semantic search using multilingual‑e5 embeddings stored in ChromaDB with sparse keyword‑based TF‑IDF retrieval. These results are further refined using Maximal Marginal Relevance (MMR) re‑ranking to ensure diverse and relevant context selection. Grade‑specific filtering is strictly enforced so that students only receive answers aligned with their academic level. The final response is generated using the Mistral language model served locally via Ollama, with strict prompting rules that limit the model to the retrieved NCERT context only. Along with the answer, the system provides a concise summary and source references for transparency. To validate system quality, both automated and LLM‑based evaluation pipelines are implemented, along with a user feedback mechanism. The entire solution is designed to be locally deployable, reproducible, and Intel‑friendly, demonstrating a complete, well‑engineered educational RAG system rather than a generic chatbot.


<img width="1536" height="1024" alt="image" src="https://github.com/user-attachments/assets/48ca51ea-da4d-4c74-a997-86d644ac7349" />

<img width="1536" height="1024" alt="image" src="https://github.com/user-attachments/assets/ea0339fd-6b0a-4c42-b04b-c5ecc8f08ccf" />


RAG Pipeline – Phase‑wise Breakdown (Table)


| Phase                   | Component         | Tools / Models Used              | What Happens                                                       |
| ----------------------- | ----------------- | -------------------------------- | ------------------------------------------------------------------ |
| **1. Data Ingestion**   | Document Loading  | DirectoryLoader, PyPDFLoader     | NCERT PDFs (Class 5–10) are loaded from structured directories     |
|                         | Text Cleaning     | OPEA Architecture                | Noise removal, normalization, NCERT‑aligned preprocessing          |
|                         | Text Chunking     | RecursiveCharacterTextSplitter   | Chunks of **450 chars** with **90 overlap** for context continuity |
| **2. Embedding Phase**  | Embedding Model   | `intfloat/multilingual-e5-large` | Multilingual semantic embeddings (1024‑dim vectors)                |
|                         | Vector Storage    | ChromaDB                         | Persistent storage with metadata (grade, language, source)         |
| **3. Retrieval Phase**  | Dense Retrieval   | Chroma Similarity Search         | Semantic matching of relevant chunks                               |
|                         | Sparse Retrieval  | TF‑IDF                           | Keyword‑based lexical matching                                     |
|                         | Hybrid Retrieval  | Dense + Sparse (α‑weighted)      | Combines semantic + keyword relevance                              |
|                         | Re‑Ranking        | MMR (Maximal Marginal Relevance) | Removes redundancy, ensures diverse context                        |
| **4. Generation Phase** | Query Expansion   | Mistral (via Ollama)             | Generates NCERT‑style reformulated queries                         |
|                         | Answer Generation | Mistral (Grounded Prompt)        | Generates answers strictly from retrieved NCERT context            |
| **5. Frontend**         | User Interface    | Streamlit                        | Text, OCR, Voice input + Answer, Summary, Sources output           |


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

1.📚 NCERT‑only, curriculum‑grounded answers

2.🎯 Grade‑specific filtering (Class 5–10)

3.🔎 Hybrid Retrieval (Dense + Sparse)

4.🧠 MMR‑based re‑ranking

5.🖼️ OCR support for textbook images

6.🎤 Voice input using Whisper

7.🌍 Multilingual question support

8.📝 Answer summarization

9.📂 Source citation

10.📊 Automated + LLM‑based evaluation





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




🚀 Quick Start (< 2 minutes)

Follow the steps below to set up and run the NCERT Hybrid RAG system locally.

1️⃣ Clone the Repository


                    git clone https://github.com/<your-username>/<your-repo-name>.git
                    cd <your-repo-name>

2️⃣ Initialize Python Environment (Python 3.13.5)
This project uses Python 3.13.5.

Initialize the project using uv:

                        uv init --python 3.13.5

Create a virtual environment:

                         uv venv

Activate the virtual environment:

Linux / macOS

             source .venv/bin/activate

Windows (PowerShell)

             .venv\Scripts\Activate.ps1
Windows (CMD)

              .venv\Scripts\activate



3️⃣ Install Dependencies

            pip install -r requirements.txt

4️⃣ Prepare NCERT Dataset (Required)

The NCERT dataset (~7–8 GB) is not included in the repository due to size limits.
Download the dataset from the provided Google Drive link
Extract it into the project root in the following structure:

             NcertData/
             ├── Class5/
             ├── Class6/
             ├── Class7/
             ├── Class8/
             ├── Class9/
             └── Class10/


5️⃣ Download Prebuilt ChromaDB (Recommended)

To save time, a prebuilt Chroma vector database is provided separately.
Download chroma_ncert_db/ from the Google Drive link
Place it directly in the project root:

               chroma_ncert_db/

⚠️ This folder is not tracked in Git due to size constraints.



6️⃣ (Optional) Build Indexes Manually

If you do NOT use the prebuilt ChromaDB, generate indexes locally:

            python build_doc_embeddings.py
            python build_sparse_index.py

This step:
Creates dense embeddings
Builds sparse TF‑IDF index
Stores reusable artifacts locally


7️⃣ Install & Configure Ollama (LLM Backend)
This project uses Mistral via Ollama for answer generation.

Install Ollama
Download and install Ollama from:
👉 https://ollama.com

Verify installation:

           ollama --version

Pull the Mistral Model:

           ollama pull mistral

Run Mistral Locally:

            ollama run mistral
            
⚠️ Keep Ollama running in the background while using the app.


8️⃣ Run the Streamlit Application

              streamlit run app.py
 
 The application will open automatically in your browser.


 
 🎯 Supported Input Modes
Once the app is running, you can ask questions using:

1.✍️ Text Input

2.📷 OCR Input (Upload textbook images)

3.🎤 Voice Input (Microphone-based queries)



📄 Output Provided
For each query, the system generates:

1.✅ NCERT‑grounded answer

2.📝 Short summary

3.📂 Source citations

4.🎓 Grade‑specific response (Class 5–10)


🧪 Evaluation (Optional)

Run automated evaluation:

            python evaluate_answers.py

Run LLM-based qualitative evaluation:

            python llm_evaluate.py



♻️ Notes
1.Large datasets and vector databases are excluded from Git

2.All results are reproducible using provided scripts

3.Answers are generated strictly from NCERT context only

4.Ollama must be running locally for answer generation





## 📦 Dataset & Prebuilt Artifacts (Google Drive)

Due to GitHub storage limitations, large files required to run this project are **not included** in the repository.

The following resources are provided via Google Drive:

- 📚 **NCERT Dataset (Class 5–10)** (~7–8 GB)
- 🧠 **Prebuilt Chroma Vector Database (`chroma_ncert_db`)**
- 📊 **Sparse Index (`sparse_index.pkl`)**
- 🧬 **Dense Embeddings (`doc_embeddings.pkl`)**
- 🎤 Sample voice input files (for testing)

👉 **Download from Google Drive:**  
https://drive.google.com/drive/folders/1L5sEtZPXwRtxqNKCctFVpgmYm7N-Uk1Y?usp=sharing

---

### 📂 After Download

Extract the contents into the **project root directory** so that the structure looks like this:

```text
NcertData/
├── Class5/
├── Class6/
├── Class7/
├── Class8/
├── Class9/
└── Class10/

chroma_ncert_db/
doc_embeddings.pkl
sparse_index.pkl







