# RAG Milestone 1 — RAG Foundation & Implementation

## Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure API keys
```bash
cp .env.example .env
# Edit .env and add your keys
```

### 3. Download sample documents (60 Wikipedia articles, no API key needed)
```bash
python src/data_fetcher.py
```

### 4. Run the full pipeline
```bash
python main.py
```

---

## Project Structure
```
rag-milestone1/
├── data/                        # Raw .txt / .pdf documents
├── vectorstores/                # ChromaDB persisted stores
├── src/
│   ├── data_fetcher.py          # Downloads Wikipedia docs
│   ├── document_processor.py   # Chunking (fixed/recursive/semantic)
│   ├── vector_store.py         # ChromaDB + Pinecone setup
│   ├── rag_pipeline.py         # Top-K and MMR retrieval + generation
│   └── evaluator.py            # Precision/recall + generation eval
├── main.py                      # Full pipeline runner
├── requirements.txt
├── .env.example
└── README.md
```

---

## What Gets Generated
| File | Contents |
|------|----------|
| `milestone1_chunking_report.json` | Comparison of 3 chunking strategies |
| `milestone1_benchmark.json` | ChromaDB retrieval speed per strategy |
| `milestone1_eval_metrics.json` | Precision@k and generation samples |

---

## API Keys
| Key | Required for |
|-----|-------------|
| `OPENAI_API_KEY` | LLM generation (gpt-3.5-turbo) |
| `PINECONE_API_KEY` | Cloud vector store (optional) |

Without API keys, Steps 1-3 (loading, chunking, ChromaDB) still work fully.

---

## Milestone 1 Checklist
- [x] Fixed, recursive, semantic chunking
- [x] Metadata extraction (chunk_id, source, timestamps)
- [x] ChromaDB local vector store
- [x] Pinecone cloud vector store (optional)
- [x] HuggingFace + OpenAI embedding support
- [x] Top-K and MMR retrieval
- [x] Source citations in responses
- [x] Retrieval benchmarking
- [x] Precision/Recall evaluation
- [x] 50+ documents via data_fetcher.py
