# Advanced Multi-PDF Retrieval-Augmented Generation (RAG) System

An end-to-end **Advanced Retrieval-Augmented Generation (RAG)** pipeline built completely in **Python** without using LangChain or LlamaIndex. The project demonstrates how modern RAG systems retrieve, rerank, compress, and generate context-aware responses from multiple PDF documents using hybrid retrieval techniques and Large Language Models.

---

## Features

- Multi-PDF document indexing
- Recursive parent-child chunking
- SentenceTransformer embeddings
- FAISS vector search
- BM25 lexical retrieval
- Hybrid retrieval (FAISS + BM25)
- Reciprocal Rank Fusion (RRF)
- CrossEncoder reranking
- Parent-Child Retrieval
- Metadata Filtering
- Rule-based Query Routing
- LLM-based Query Expansion
- HyDE (Hypothetical Document Embeddings)
- Embedding-based Context Compression
- LLM-based Context Compression
- Multi-document Retrieval
- Ollama (Llama 3) based answer generation

---

# Pipeline

```text
                    User Question
                          │
                          ▼
                 Query Routing
                          │
                          ▼
              LLM Query Expansion
                          │
                          ▼
                     HyDE Generation
                          │
                          ▼
          Hybrid Retrieval (FAISS + BM25)
                          │
                          ▼
            Reciprocal Rank Fusion (RRF)
                          │
                          ▼
               CrossEncoder Re-ranking
                          │
                          ▼
               Parent Chunk Retrieval
                          │
                          ▼
          Embedding Context Compression
                          │
                          ▼
            LLM Context Compression
                          │
                          ▼
                  Llama 3 Generation
                          │
                          ▼
                     Final Answer
```

---

# Project Structure

```
RAG_Implementation/
│
├── PDFs/
│   ├── resume.pdf
│   ├── PDL_REPORT.pdf
│   ├── Python_Lab.pdf
│   └── Curriculum.pdf
│
├── RAG_multi_pdf.py
├── requirements.txt
├── README.md
```

---

# Technologies Used

### Programming Language

- Python

### Embedding Model

- Sentence Transformers
- all-MiniLM-L6-v2

### Vector Database

- FAISS

### Lexical Retrieval

- BM25

### Reranking

- CrossEncoder
- BAAI/bge-reranker-base

### Large Language Model

- Ollama
- Llama 3

### Libraries

- SentenceTransformers
- FAISS
- rank_bm25
- PyMuPDF
- NumPy
- Requests
- RecursiveCharacterTextSplitter

---

# Retrieval Techniques Implemented

| Feature | Status |
|----------|---------|
| Semantic Search | ✅ |
| Hybrid Retrieval | ✅ |
| Multi Query Retrieval | ✅ |
| Reciprocal Rank Fusion | ✅ |
| CrossEncoder Re-ranking | ✅ |
| Parent Child Retrieval | ✅ |
| Metadata Filtering | ✅ |
| Query Routing | ✅ |
| Query Expansion | ✅ |
| HyDE | ✅ |
| Embedding Compression | ✅ |
| LLM Compression | ✅ |

---

# Parent-Child Retrieval

Instead of sending only the retrieved child chunk to the LLM:

```
Query
   ↓
Child Chunk
   ↓
LLM
```

The system first retrieves highly relevant child chunks and then expands them into their larger parent chunks.

```
Query
   ↓
Child Retrieval
   ↓
Parent Retrieval
   ↓
Context Compression
   ↓
LLM
```

This preserves more surrounding context while maintaining retrieval precision.

---

# Hybrid Retrieval

The system combines both semantic and lexical retrieval.

```
FAISS Search
        \
         \
          ---> RRF ---> CrossEncoder
         /
BM25 Search
```

This improves recall while reducing retrieval errors.

---

# Context Compression

The retrieved parent chunks are compressed before passing them to the LLM.

Two-stage compression is implemented:

- Embedding Similarity Compression
- LLM Context Compression

This significantly reduces token usage while preserving relevant information.

---

# Query Expansion

Every user query is expanded using Llama 3 into multiple semantically similar search queries.

Example

```
Question:

What is the student's CGPA?

Expanded Queries

Current CGPA

Student cumulative grade point average

Academic performance

Current grade point average
```

---

# HyDE

The project implements **Hypothetical Document Embeddings (HyDE)**.

Instead of embedding only the user query:

```
Question
      ↓
Embedding
```

the system first generates a hypothetical answer and embeds it.

```
Question
      ↓
Hypothetical Document
      ↓
Embedding
```

This improves semantic retrieval for abstract and descriptive queries.

---

# Example Questions

- What is the current CGPA?
- Name the members of the PDL project.
- What is the project title?
- Summarize the PDL report.
- What skills does the candidate have?
- Which technologies were used in the project?
- Explain ESP32 mentioned in the report.
- What certifications does the candidate have?

---

# Future Improvements

- Self-RAG
- CRAG (Corrective RAG)
- Adaptive RAG
- Graph RAG
- RAPTOR
- Agentic RAG
- Embedding Cache
- FAISS Index Persistence
- BM25 Index Persistence
- Semantic Document Router
- Confidence-based Generation
- Streaming Responses

---

# Key Learning Outcomes

This project demonstrates the implementation of modern Retrieval-Augmented Generation systems from scratch without relying on high-level orchestration frameworks.

Topics covered include:

- Document Parsing
- Recursive Chunking
- Parent-Child Retrieval
- Vector Search
- Hybrid Retrieval
- Query Expansion
- HyDE
- Reciprocal Rank Fusion
- CrossEncoder Re-ranking
- Context Compression
- Prompt Engineering
- Large Language Models
- Retrieval Pipeline Design

---

# Author

**Madhav Kumar**

B.Tech Electronics & Communication Engineering

Indian Institute of Information Technology, Una

GitHub: https://github.com/maddycode5