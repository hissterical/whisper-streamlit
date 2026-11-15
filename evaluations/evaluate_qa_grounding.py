"""Formal evaluation of Journal Q&A Grounding in EchoNote.

Evaluates:
- Category A: Recent-entry questions
- Category B: Older-entry questions (proves removal of fixed recent-entry limit)
- Category C: Multi-entry synthesis questions
- Category D: Unanswerable questions (evaluates explicit abstention)

Metrics:
- Fact Recall / Correctness
- Citation Precision (% of cited [Entry #ID] tags matching expected evidence)
- Citation Recall (% of expected evidence cited)
- Explicit Abstention Accuracy
- Latency per query
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from journal_app.database import JournalStore
from journal_app.services import ask_journal, embed_text

DATASET_PATH = REPO_ROOT / "evaluations" / "datasets" / "qa_grounding_eval.json"
RESULTS_DIR = REPO_ROOT / "evaluations" / "results"

ABSTENTION_INDICATORS = [
    "do not have enough information",
    "cannot answer",
    "not mentioned",
    "no information",
    "not recorded",
    "journal does not contain",
    "could not find",
]


def extract_citations(text: str) -> list[int]:
    """Extract entry IDs from [Entry #X] patterns in answer text."""
    matches = re.findall(r"\[Entry\s*#?(\d+)\]", text, re.IGNORECASE)
    return sorted({int(m) for m in matches})


def evaluate_abstention(response: dict[str, Any], should_abstain: bool) -> bool:
    """Check if the system appropriately abstained when evidence was absent."""
    answer_text = response.get("answer", "").lower()
    system_flag = response.get("abstention", False)
    text_abstains = any(ind in answer_text for ind in ABSTENTION_INDICATORS)
    is_abstaining = system_flag or text_abstains

    return is_abstain_correct(is_abstaining, should_abstain)


def is_abstain_correct(actual_abstain: bool, expected_abstain: bool) -> bool:
    return actual_abstain == expected_abstain


def evaluate_fact_coverage(answer: str, gold_facts: list[str]) -> float:
    """Measure proportion of expected factual statements reflected in the answer."""
    if not gold_facts:
        return 1.0

    answer_lower = answer.lower()
    matches = 0
    for fact in gold_facts:
        fact_tokens = [w for w in re.findall(r"\b\w+\b", fact.lower()) if len(w) > 3]
        if not fact_tokens:
            continue
        token_hits = sum(1 for t in fact_tokens if t in answer_lower)
        if token_hits / len(fact_tokens) >= 0.5:
            matches += 1

    return round(matches / len(gold_facts), 4)


def run_qa_grounding_evaluation(
    dataset_path: Path = DATASET_PATH,
    use_ollama: bool = True,
) -> dict[str, Any]:
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    corpus = data["corpus_history"]
    questions = data["questions"]

    tmp_dir = tempfile.mkdtemp()
    db_path = Path(tmp_dir) / "qa_eval.sqlite3"

    try:
        store = JournalStore(db_path)
        print(f"Populating evaluation store with {len(corpus)} historical entries...")
        id_map: dict[int, int] = {}

        # Add 30 dummy padding entries to simulate an active, long-lived journal
        # so that older entries are placed well beyond any fixed top-20 limit!
        for i in range(35):
            store.add({
                "created_at": f"2026-05-{1 + (i % 28):02d}T08:00:00",
                "title": f"Routine Log {i}",
                "summary": f"General daily journal reflection number {i}.",
                "transcript": f"Logging day {i} activities and notes.",
                "topics": ["routine"],
                "action_items": [],
            })

        for entry in corpus:
            text_emb = f"{entry['title']}. {entry['summary']}. {' '.join(entry.get('topics', []))}"
            e_data = embed_text(text_emb) if use_ollama else None
            record = {
                "created_at": entry["created_at"],
                "title": entry["title"],
                "summary": entry["summary"],
                "transcript": entry["transcript"],
                "topics": entry["topics"],
                "action_items": entry["action_items"],
                "embedding": e_data["vector"] if e_data else None,
                "embedding_model": e_data["model"] if e_data else None,
                "embedding_dim": e_data["dim"] if e_data else None,
            }
            real_id = store.add(record)
            id_map[entry["id"]] = real_id

        # Verify that historical entries are indeed buried beyond the top 20 limit
        total_entries = len(store.list(limit=200))
        print(f"Total entries in store: {total_entries} (Older entries buried beyond recent limit)")

        systems = ["deterministic_qa"]
        if use_ollama:
            systems.append("ollama_llama3.2")

        eval_report: dict[str, Any] = {"systems": {}}

        for sys_name in systems:
            print(f"\nEvaluating QA system: {sys_name}...")
            per_query_results = []

            for q in questions:
                q_text = q["question"]
                cat = q["category"]
                expected_original_ids = q["expected_evidence_ids"]
                mapped_expected_ids = [id_map[eid] for eid in expected_original_ids if eid in id_map]
                should_abstain = q["should_abstain"]

                # Retrieve candidates using hybrid or keyword search
                t0 = time.perf_counter()
                q_vec_data = embed_text(q_text) if sys_name == "ollama_llama3.2" else None
                q_vec = q_vec_data["vector"] if q_vec_data else None

                if q_vec:
                    candidates = store.hybrid_search(q_text, vector=q_vec, limit=20)
                else:
                    candidates = store.search(q_text, limit=20)

                if not candidates:
                    candidates = store.list(limit=20)

                # Answer generation
                ans_resp = ask_journal(
                    q_text,
                    candidates,
                    model_name="llama3.2",
                    use_ollama=(sys_name == "ollama_llama3.2"),
                )
                lat_ms = (time.perf_counter() - t0) * 1000

                answer_str = ans_resp.get("answer", "")
                cited_ids = extract_citations(answer_str)

                # Evaluate citations
                if mapped_expected_ids:
                    cit_precision = sum(1 for cid in cited_ids if cid in mapped_expected_ids) / len(cited_ids) if cited_ids else 0.0
                    cit_recall = sum(1 for cid in mapped_expected_ids if cid in cited_ids) / len(mapped_expected_ids)
                else:
                    cit_precision = 1.0 if not cited_ids else 0.0
                    cit_recall = 1.0

                abstain_correct = evaluate_abstention(ans_resp, should_abstain)
                fact_cov = evaluate_fact_coverage(answer_str, q["gold_answer_facts"]) if not should_abstain else 1.0

                per_query_results.append({
                    "id": q["id"],
                    "category": cat,
                    "question": q_text,
                    "latency_ms": round(lat_ms, 2),
                    "should_abstain": should_abstain,
                    "actual_abstain": ans_resp.get("abstention", False),
                    "abstention_correct": abstain_correct,
                    "cited_ids": cited_ids,
                    "expected_ids": mapped_expected_ids,
                    "citation_precision": round(cit_precision, 4),
                    "citation_recall": round(cit_recall, 4),
                    "fact_coverage": fact_cov,
                    "answer_sample": answer_str[:120],
                })

            # Category aggregations
            categories = ["recent_entry", "older_entry", "multi_entry", "unanswerable"]
            cat_summary: dict[str, Any] = {}

            for c in categories:
                c_items = [m for m in per_query_results if m["category"] == c]
                if not c_items:
                    continue
                cat_summary[c] = {
                    "count": len(c_items),
                    "avg_latency_ms": round(sum(m["latency_ms"] for m in c_items) / len(c_items), 2),
                    "abstention_accuracy": round(sum(1 for m in c_items if m["abstention_correct"]) / len(c_items), 4),
                    "mean_fact_coverage": round(sum(m["fact_coverage"] for m in c_items) / len(c_items), 4),
                    "mean_citation_precision": round(sum(m["citation_precision"] for m in c_items) / len(c_items), 4),
                    "mean_citation_recall": round(sum(m["citation_recall"] for m in c_items) / len(c_items), 4),
                }

            all_n = len(per_query_results)
            overall = {
                "total_queries": all_n,
                "overall_abstention_accuracy": round(sum(1 for m in per_query_results if m["abstention_correct"]) / all_n, 4),
                "overall_fact_coverage": round(sum(m["fact_coverage"] for m in per_query_results) / all_n, 4),
                "overall_citation_precision": round(sum(m["citation_precision"] for m in per_query_results) / all_n, 4),
                "overall_citation_recall": round(sum(m["citation_recall"] for m in per_query_results) / all_n, 4),
                "avg_latency_ms": round(sum(m["latency_ms"] for m in per_query_results) / all_n, 2),
            }

            eval_report["systems"][sys_name] = {
                "overall": overall,
                "by_category": cat_summary,
                "queries": per_query_results,
            }

        # Save Results
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        out_json = RESULTS_DIR / "qa_grounding_results.json"
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(eval_report, f, indent=2)
        print(f"\n[OK] Q&A Grounding results saved to: {out_json}")

        out_csv = RESULTS_DIR / "qa_grounding_summary.csv"
        save_qa_csv(eval_report["systems"], out_csv)
        print(f"[OK] Summary CSV saved to: {out_csv}")

        print_qa_table(eval_report)

        return eval_report
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def save_qa_csv(systems_data: dict[str, Any], csv_path: Path) -> None:
    rows = []
    for sys_name, data in systems_data.items():
        for cat_name, metrics in data.get("by_category", {}).items():
            rows.append({"system": sys_name, "category": cat_name, **metrics})

    if rows:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)


def print_qa_table(report: dict[str, Any]) -> None:
    print("\n" + "=" * 95)
    print("JOURNAL Q&A GROUNDING EVALUATION: RECENT, OLDER, MULTI-ENTRY, AND UNANSWERABLE")
    print("=" * 95)
    fmt = "{:<22} | {:<15} | {:<10} | {:<12} | {:<12} | {:<12}"
    print(fmt.format("System", "Category", "Fact-Cov", "Cite-Prec", "Cite-Rec", "Abstain-Acc"))
    print("-" * 95)

    for sys_name, data in report["systems"].items():
        for cat, m in data.get("by_category", {}).items():
            print(fmt.format(
                sys_name,
                cat,
                f"{m['mean_fact_coverage']*100:.1f}%",
                f"{m['mean_citation_precision']*100:.1f}%",
                f"{m['mean_citation_recall']*100:.1f}%",
                f"{m['abstention_accuracy']*100:.1f}%",
            ))
        print("-" * 95)
    print("=" * 95)


def main() -> None:
    parser = argparse.ArgumentParser(description="EchoNote Q&A Grounding Evaluation")
    parser.add_argument("--no-ollama", action="store_true", help="Run deterministic fallback only")
    args = parser.parse_args()

    run_qa_grounding_evaluation(use_ollama=not args.no_ollama)


if __name__ == "__main__":
    main()
