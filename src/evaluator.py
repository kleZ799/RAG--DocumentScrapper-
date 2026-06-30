"""
Milestone 1 - RAG Evaluator
Basic retrieval precision/recall and generation quality checks.
"""

from typing import List, Dict


class RAGEvaluator:
    """
    Lightweight retrieval evaluator using ground-truth Q&A pairs.
    For full LLM-based scoring, use RAGAS (Milestone 2).
    """

    def evaluate_retrieval(
        self,
        vectorstore,
        qa_pairs: List[Dict],
        k: int = 5
    ) -> Dict:
        """
        qa_pairs: list of dicts with keys:
          - question (str)
          - expected_doc_id (str) — chunk_id of the expected document
        """
        hits = 0
        total = len(qa_pairs)
        misses = []

        for pair in qa_pairs:
            docs = vectorstore.similarity_search(pair["question"], k=k)
            retrieved_ids = [d.metadata.get("chunk_id", "") for d in docs]
            if pair["expected_doc_id"] in retrieved_ids:
                hits += 1
            else:
                misses.append(pair["question"])

        return {
            "precision_at_k": round(hits / total, 3) if total else 0,
            "hit_rate": round(hits / total, 3) if total else 0,
            "k": k,
            "total_queries": total,
            "hits": hits,
            "misses": misses,
        }

    def evaluate_generation(
        self,
        rag_pipeline,
        qa_pairs: List[Dict]
    ) -> List[Dict]:
        """
        qa_pairs: list of dicts with keys:
          - question (str)
          - expected_answer (str)
        """
        results = []
        for pair in qa_pairs:
            output = rag_pipeline.query(pair["question"])
            results.append({
                "question": pair["question"],
                "expected": pair.get("expected_answer", ""),
                "generated": output["answer"],
                "num_sources": output["num_sources"],
                "sources": output["sources"],
            })
        return results

    def print_report(self, retrieval_metrics: Dict, generation_results: List[Dict]):
        print("\n" + "="*50)
        print("📊 MILESTONE 1 EVALUATION REPORT")
        print("="*50)
        print(f"Retrieval Hit Rate  : {retrieval_metrics['hit_rate']}")
        print(f"Precision@{retrieval_metrics['k']}        : {retrieval_metrics['precision_at_k']}")
        print(f"Queries tested      : {retrieval_metrics['total_queries']}")
        print(f"Hits / Misses       : {retrieval_metrics['hits']} / {len(retrieval_metrics['misses'])}")

        if generation_results:
            print(f"\nGeneration samples  : {len(generation_results)}")
            for r in generation_results[:3]:
                print(f"\n  Q: {r['question']}")
                print(f"  A: {r['generated'][:200]}...")
        print("="*50)
