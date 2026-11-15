"""Evaluation framework for EchoNote retrieval strategies:
- SQLite FTS5 Keyword Retrieval
- Vector / Semantic Retrieval (nomic-embed-text)
- Hybrid Retrieval (Reciprocal Rank Fusion RRF)

Calculates:
- Recall@1, Recall@3, Recall@5
- Mean Reciprocal Rank (MRR)
- nDCG@1, nDCG@3, nDCG@5
- Query latency & scaling
- Failure & fallback behavior under mismatched vectors and absent models
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from journal_app.database import JournalStore
from journal_app.services import embed_text

DATASET_PATH = REPO_ROOT / "evaluations" / "datasets" / "labeled_retrieval_queries.json"
RESULTS_DIR = REPO_ROOT / "evaluations" / "results"


def compute_dcg(relevance_scores: list[float], k: int) -> float:
    """Discounted Cumulative Gain at K: sum((2^rel - 1) / log2(i + 1))."""
    dcg = 0.0
    for i, rel in enumerate(relevance_scores[:k], start=1):
        dcg += (2.0**rel - 1.0) / math.log2(i + 1)
    return dcg


def compute_idcg(ideal_relevances: list[float], k: int) -> float:
    """Ideal Discounted Cumulative Gain at K."""
    sorted_ideal = sorted(ideal_relevances, reverse=True)
    return compute_dcg(sorted_ideal, k)


def compute_ndcg(retrieved_ids: list[int], gold_relevance: dict[str, int], k: int) -> float:
    """Calculate normalized Discounted Cumulative Gain at rank K."""
    if not gold_relevance:
        return 1.0

    retrieved_rels = [float(gold_relevance.get(str(doc_id), 0)) for doc_id in retrieved_ids[:k]]
    actual_dcg = compute_dcg(retrieved_rels, k)
    ideal_dcg = compute_idcg([float(v) for v in gold_relevance.values()], k)

    if ideal_dcg == 0.0:
        return 1.0 if actual_dcg == 0.0 else 0.0
    return round(actual_dcg / ideal_dcg, 4)


def compute_recall_at_k(retrieved_ids: list[int], gold_relevance: dict[str, int], k: int) -> float:
    """Proportion of relevant gold documents (rel >= 2) retrieved within top K."""
    relevant_gold = {int(k) for k, rel in gold_relevance.items() if rel >= 2}
    if not relevant_gold:
        return 1.0

    retrieved_at_k = set(retrieved_ids[:k])
    hits = len(retrieved_at_k & relevant_gold)
    return round(hits / len(relevant_gold), 4)


def compute_mrr(retrieved_ids: list[int], gold_relevance: dict[str, int]) -> float:
    """Mean Reciprocal Rank: 1 / rank of first relevant item (rel >= 2)."""
    relevant_gold = {int(k) for k, rel in gold_relevance.items() if rel >= 2}
    if not relevant_gold:
        return 1.0

    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in relevant_gold:
            return round(1.0 / rank, 4)
    return 0.0


def setup_evaluation_store(corpus: list[dict[str, Any]], store_path: Path) -> dict[int, list[float]]:
    """Populate isolated evaluation store and generate/cache embeddings."""
    store = JournalStore(store_path)
    embedding_cache: dict[int, list[float]] = {}

    print(f"Ingesting {len(corpus)} corpus documents into benchmark store...")
    for item in corpus:
        text_for_embedding = f"{item['title']}. {item['summary']}. {' '.join(item['topics'])}"
        emb_data = embed_text(text_for_embedding, model_name="nomic-embed-text")
        vector = emb_data["vector"] if emb_data else None

        record = {
            "title": item["title"],
            "summary": item["summary"],
            "transcript": item["transcript"],
            "topics": item["topics"],
            "mood": item["mood"],
            "action_items": item["action_items"],
            "embedding": vector,
            "embedding_model": "nomic-embed-text" if vector else None,
            "embedding_dim": len(vector) if vector else None,
        }
        eid = store.add(record)
        # Map item id from dataset to inserted entry id
        if vector:
            embedding_cache[item["id"]] = vector

    return embedding_cache


def evaluate_retrieval_strategies(
    dataset_path: Path = DATASET_PATH,
) -> dict[str, Any]:
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    corpus = data["corpus"]
    queries = data["queries"]

    tmp_dir = tempfile.mkdtemp()
    db_path = Path(tmp_dir) / "retrieval_eval.sqlite3"

    try:
        store = JournalStore(db_path)
        corpus_vectors = setup_evaluation_store(corpus, db_path)

        # ID mapping: map corpus item id to store id
        store_entries = store.list(limit=100)
        title_to_store_id = {e["title"]: e["id"] for e in store_entries}
        corpus_id_to_store_id = {c["id"]: title_to_store_id[c["title"]] for c in corpus}

        # Strategies
        strategies = ["fts5_keyword", "semantic_vector", "hybrid_rrf"]
        results_by_strategy: dict[str, list[dict[str, Any]]] = {s: [] for s in strategies}

        for q in queries:
            q_id = q["query_id"]
            q_text = q["query_text"]
            # Translate gold relevance keys to actual store IDs
            gold_rel_mapped = {
                str(corpus_id_to_store_id[int(cid)]): grade
                for cid, grade in q["relevance"].items()
                if int(cid) in corpus_id_to_store_id
            }

            # Embed query for semantic/hybrid strategies
            t_emb_0 = time.perf_counter()
            q_emb_data = embed_text(q_text, model_name="nomic-embed-text")
            q_vec = q_emb_data["vector"] if q_emb_data else None

            # 1. FTS5 Keyword
            t0 = time.perf_counter()
            fts_res = store.search(q_text, limit=10)
            fts_lat = time.perf_counter() - t0
            fts_ids = [e["id"] for e in fts_res]

            # 2. Semantic Search
            t1 = time.perf_counter()
            sem_res = store.semantic_search(q_vec, model_name="nomic-embed-text", limit=10) if q_vec else []
            sem_lat = time.perf_counter() - t1
            sem_ids = [e["id"] for e in sem_res]

            # 3. Hybrid Search (RRF)
            t2 = time.perf_counter()
            hyb_res = store.hybrid_search(q_text, vector=q_vec, model_name="nomic-embed-text", limit=10)
            hyb_lat = time.perf_counter() - t2
            hyb_ids = [e["id"] for e in hyb_res]

            runs = [
                ("fts5_keyword", fts_ids, fts_lat),
                ("semantic_vector", sem_ids, sem_lat),
                ("hybrid_rrf", hyb_ids, hyb_lat),
            ]

            for s_name, retrieved, lat in runs:
                results_by_strategy[s_name].append({
                    "query_id": q_id,
                    "query_type": q["query_type"],
                    "latency_ms": round(lat * 1000, 3),
                    "recall_at_1": compute_recall_at_k(retrieved, gold_rel_mapped, 1),
                    "recall_at_3": compute_recall_at_k(retrieved, gold_rel_mapped, 3),
                    "recall_at_5": compute_recall_at_k(retrieved, gold_rel_mapped, 5),
                    "mrr": compute_mrr(retrieved, gold_rel_mapped),
                    "ndcg_at_1": compute_ndcg(retrieved, gold_rel_mapped, 1),
                    "ndcg_at_3": compute_ndcg(retrieved, gold_rel_mapped, 3),
                    "ndcg_at_5": compute_ndcg(retrieved, gold_rel_mapped, 5),
                })

        # Aggregate summary
        summary: dict[str, Any] = {"strategies": {}}
        for s_name, metrics in results_by_strategy.items():
            n = len(metrics)
            summary["strategies"][s_name] = {
                "queries_evaluated": n,
                "avg_latency_ms": round(sum(m["latency_ms"] for m in metrics) / n, 3),
                "mean_recall_at_1": round(sum(m["recall_at_1"] for m in metrics) / n, 4),
                "mean_recall_at_3": round(sum(m["recall_at_3"] for m in metrics) / n, 4),
                "mean_recall_at_5": round(sum(m["recall_at_5"] for m in metrics) / n, 4),
                "mean_mrr": round(sum(m["mrr"] for m in metrics) / n, 4),
                "mean_ndcg_at_1": round(sum(m["ndcg_at_1"] for m in metrics) / n, 4),
                "mean_ndcg_at_3": round(sum(m["ndcg_at_3"] for m in metrics) / n, 4),
                "mean_ndcg_at_5": round(sum(m["ndcg_at_5"] for m in metrics) / n, 4),
            }

        # Robustness & Failure Mode Tests
        print("\nEvaluating retrieval robustness and fallback behavior...")
        robustness: dict[str, Any] = {}

        # Test 1: Incompatible embedding dimensions (384-dim query vs 768-dim entries)
        incompat_vec = [0.1] * 384
        incompat_res = store.semantic_search(incompat_vec, limit=10)
        robustness["incompatible_dimensions"] = {
            "query_dim": 384,
            "expected_behavior": "Zero corrupt comparisons; safe dimension rejection",
            "results_count": len(incompat_res),
            "passed": len(incompat_res) == 0,
        }

        # Test 2: Missing Ollama model fallback
        # When embedding is None, hybrid_search must fall back directly to FTS5
        fallback_res = store.hybrid_search("WAL mode", vector=None, limit=10)
        robustness["missing_embedding_model_fallback"] = {
            "vector_provided": False,
            "expected_behavior": "Automatic fallback to keyword FTS5 search",
            "results_count": len(fallback_res),
            "top_match_title": fallback_res[0]["title"] if fallback_res else None,
            "passed": len(fallback_res) > 0 and "WAL" in fallback_res[0]["title"],
        }

        summary["robustness_tests"] = robustness

        # Save machine-readable outputs
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        out_json = RESULTS_DIR / "retrieval_eval_results.json"
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump({"summary": summary, "per_query_results": results_by_strategy}, f, indent=2)
        print(f"\n[OK] Retrieval evaluation results saved to: {out_json}")

        out_csv = RESULTS_DIR / "retrieval_eval_summary.csv"
        save_retrieval_csv(summary["strategies"], out_csv)
        print(f"[OK] Summary CSV saved to: {out_csv}")

        # Print comparison table
        print_retrieval_table(summary)

        return summary
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def save_retrieval_csv(strat_summary: dict[str, Any], csv_path: Path) -> None:
    rows = []
    for s_name, data in strat_summary.items():
        rows.append({"strategy": s_name, **data})
    if rows:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)


def print_retrieval_table(summary: dict[str, Any]) -> None:
    print("\n" + "=" * 90)
    print("RETRIEVAL EVALUATION RESULTS: FTS5 vs SEMANTIC vs HYBRID (RRF)")
    print("=" * 90)
    fmt = "{:<20} | {:<10} | {:<10} | {:<10} | {:<10} | {:<10} | {:<10}"
    print(fmt.format("Strategy", "Latency(ms)", "Recall@1", "Recall@3", "Recall@5", "MRR", "nDCG@5"))
    print("-" * 90)

    for s_name, d in summary["strategies"].items():
        print(fmt.format(
            s_name,
            f"{d['avg_latency_ms']:.2f}",
            str(d["mean_recall_at_1"]),
            str(d["mean_recall_at_3"]),
            str(d["mean_recall_at_5"]),
            str(d["mean_mrr"]),
            str(d["mean_ndcg_at_5"]),
        ))
    print("=" * 90)

    print("\nRobustness & Fallback Verification:")
    for test_name, res in summary.get("robustness_tests", {}).items():
        status = "PASSED" if res.get("passed") else "FAILED"
        print(f"  [{status}] {test_name}: {res.get('expected_behavior')}")
    print("=" * 90)


def main() -> None:
    evaluate_retrieval_strategies()


if __name__ == "__main__":
    main()
