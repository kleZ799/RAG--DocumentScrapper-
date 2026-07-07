"""
Milestone 1 - Main Runner
Run this after:
  1. pip install -r requirements.txt
  2. Copy .env.example to .env and fill in your API keys
  3. python src/data_fetcher.py  (downloads 60 Wikipedia docs)
"""

import os
import json
from dotenv import load_dotenv

load_dotenv()

from src.document_processor import DocumentProcessor
from src.vector_store import VectorStoreManager
from src.rag_pipeline import RAGPipeline
from src.evaluator import RAGEvaluator

# ── Config ────────────────────────────────────────────────────────
DATA_DIR = "./data"
USE_OPENAI_EMBEDDINGS = False  # set True if you have OPENAI_API_KEY

BENCHMARK_QUERIES = [
    "What is attention mechanism in transformers?",
    "How does BERT differ from GPT?",
    "Explain backpropagation in neural networks",
    "What is reinforcement learning?",
    "How are word embeddings created?",
]

TEST_QA_PAIRS = [
    {
        "question": "What is a neural network?",
        "expected_answer": "A neural network is a computational model inspired by the brain.",
    },
    {
        "question": "What is transfer learning?",
        "expected_answer": "Transfer learning reuses a pretrained model on a new task.",
    },
]

# ── Step 1: Load documents ────────────────────────────────────────
print("\n" + "="*50)
print("STEP 1: Loading Documents")
print("="*50)
processor = DocumentProcessor()
raw_docs = processor.load_documents(DATA_DIR)

if not raw_docs:
    print("\n⚠️  No documents found. Run: python src/data_fetcher.py")
    exit(1)

# ── Step 2: Compare chunking strategies ──────────────────────────
print("\n" + "="*50)
print("STEP 2: Chunking Strategy Comparison")
print("="*50)
comparison = processor.compare_strategies(raw_docs)
print("\n📊 Summary:")
for strategy, stats in comparison.items():
    print(f"  {strategy:10s}: {stats['total_chunks']} chunks, avg {stats['avg_size']} chars")

with open("milestone1_chunking_report.json", "w") as f:
    json.dump(comparison, f, indent=2)
print("\n💾 Saved: milestone1_chunking_report.json")

# ── Step 3: Build vector stores ───────────────────────────────────
print("\n" + "="*50)
print("STEP 3: Building Vector Stores (ChromaDB)")
print("="*50)
manager = VectorStoreManager(use_openai=USE_OPENAI_EMBEDDINGS)

bench_results = {}
best_vs = None
for strategy in ["fixed", "recursive", "semantic"]:
    chunks = processor.chunk(raw_docs, strategy=strategy)
    vs = manager.create_chroma(chunks, collection_name=f"rag_{strategy}")
    bench = manager.benchmark(vs, BENCHMARK_QUERIES)
    bench_results[strategy] = bench
    print(f"   ⚡ {strategy}: avg {bench['avg_latency_ms']}ms")
    if strategy == "recursive":
        best_vs = vs  # use recursive for the pipeline

with open("milestone1_benchmark.json", "w") as f:
    json.dump(bench_results, f, indent=2)
print("\n💾 Saved: milestone1_benchmark.json")

# ── Step 4: Pinecone (optional) ───────────────────────────────────
if os.environ.get("PINECONE_API_KEY"):
    print("\n" + "="*50)
    print("STEP 4: Building Pinecone Vector Store")
    print("="*50)
    best_chunks = processor.chunk(raw_docs, strategy="recursive")
    manager.create_pinecone(best_chunks, index_name="rag-milestone1")
else:
    print("\n⏭️  Skipping Pinecone (no PINECONE_API_KEY in .env)")

# ── Step 5: RAG pipeline ──────────────────────────────────────────
if not os.environ.get("OPENAI_API_KEY"):
    print("\n⏭️  Skipping RAG generation (no OPENAI_API_KEY in .env)")
    print("    Retrieval is working ✅ — add key to test generation.")
else:
    print("\n" + "="*50)
    print("STEP 5: RAG Pipeline — Top-K vs MMR")
    print("="*50)
    test_q = "How does the attention mechanism work in transformer models?"

    for mode in ["topk", "mmr"]:
        pipeline = RAGPipeline(best_vs, retrieval_mode=mode)
        result = pipeline.query(test_q)
        print(f"\n🔍 [{mode.upper()}]")
        print(f"   Answer  : {result['answer'][:300]}...")
        print(f"   Sources : {result['num_sources']} chunks retrieved")

    # ── Step 6: Evaluate ──────────────────────────────────────────
    print("\n" + "="*50)
    print("STEP 6: Evaluation")
    print("="*50)
    evaluator = RAGEvaluator()

    # Build simple qa pairs with chunk_ids from the store
    sample_docs = best_vs.similarity_search("neural network", k=3)
    eval_pairs = [
        {
            "question": "What is a neural network?",
            "expected_doc_id": sample_docs[0].metadata.get("chunk_id", "")
        }
    ] if sample_docs else []

    if eval_pairs:
        metrics = evaluator.evaluate_retrieval(best_vs, eval_pairs, k=5)
        gen_results = evaluator.evaluate_generation(
            RAGPipeline(best_vs, "topk"), TEST_QA_PAIRS
        )
        evaluator.print_report(metrics, gen_results)

        with open("milestone1_eval_metrics.json", "w") as f:
            json.dump({"retrieval": metrics, "generation": gen_results}, f, indent=2)
        print("\n💾 Saved: milestone1_eval_metrics.json")

print("\n🎉 Milestone 1 Complete!")
print("   Check: milestone1_chunking_report.json")
print("          milestone1_benchmark.json")
print("          milestone1_eval_metrics.json")
print("          vectorstores/")