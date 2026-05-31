
# 🚀 TEYZIX CORE - Production Grade RAG System (AI-3)

## 📌 Project Overview
This project is a Production-Grade Retrieval-Augmented Generation (RAG) system built for enterprise document QA.  
It supports hybrid retrieval (dense + sparse), reranking, query rewriting, and evaluation using RAGAS.

---

## ⚙️ Features

### 🔍 Retrieval System
- Hybrid Retrieval (FAISS + BM25)
- Reciprocal Rank Fusion (RRF)
- Cross-encoder reranking

### 🧠 Intelligence Layer
- Query rewriting (LLM-based)
- Intent handling (small talk support)
- Context filtering & scoring

### 🛡️ Security Layer
- Prompt injection guardrails
- Unsafe query detection

### 🤖 Generation Layer
- Gemini LLM integration
- Context-grounded answers
- Hallucination reduction via strict prompting

### 📊 Evaluation System
- RAGAS evaluation (faithfulness, relevancy, recall)
- Automated benchmark pipeline

### ⚡ Performance
- Latency tracking (retrieval + generation)
- Optimized top-k context selection

---

## 🏗️ Architecture

Pipeline:

1. Document Ingestion (`ingest.py`)
2. Chunking + Embedding (FAISS)
3. BM25 Sparse Index
4. Hybrid Retrieval (RRF)
5. Reranking (CrossEncoder)
6. Query Rewriting
7. Prompt Builder
8. LLM Generation (Gemini)
9. Evaluation (RAGAS)

---

## 📂 Project Structure

```

app/
├── ingestion/
├── retrieval/
│    ├── dense_retriever.py
│    ├── sparse_retriever.py
│    ├── hybrid_retriever.py
│    └── reranker.py
├── llm/
├── query/
├── security/
├── evaluation/
├── chatbot.py
└── main.py

vectorstore/
├── faiss_index/
├── bm25.pkl

````

---

## 🚀 How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
````

### 2. Run ingestion

```bash
python app/ingestion/ingest.py
```

### 3. Start API

```bash
uvicorn app.main:app --reload
```

### 4. Open Swagger UI

```
http://127.0.0.1:8000/docs
```

---

## 📊 Evaluation

Run evaluation pipeline:

```bash
python run_eval.py
```

Outputs:

* Faithfulness score
* Answer relevancy
* Context recall
* JSON report

---

## 📌 Key Design Decisions

* Hybrid retrieval improves recall (dense + sparse)
* RRF used for stable ranking fusion
* CrossEncoder reranking improves precision
* Query rewriting improves ambiguity handling
* Strict prompt reduces hallucination

---

## ⚠️ Notes

* Gemini API quota is required
* FAISS index must be generated before chat
* BM25 index stored in pickle file

---

## 👨‍💻 Author


Ayesha Abbas