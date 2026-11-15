"""Formal evaluation framework for Summary, Topics, and Action Items in EchoNote.

Compares:
- Deterministic Fallback
- Ollama LLM (llama3.2)

Metrics:
- Summary ROUGE-1, ROUGE-2, ROUGE-L (Precision, Recall, F1)
- Key Fact Coverage (% of gold reference facts captured in summary)
- Unsupported Claim / Hallucination Rate (% of generated claims not grounded in transcript)
- Topic Precision, Recall, F1
- Action Item Precision, Recall, F1

Evaluates against:
- Human-annotated research notes (human_annotated_notes.json)
- Synthetic test fixtures (synthetic_test_fixtures.json)
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from journal_app.services import _fallback_analysis, analyze_transcript

DATASETS_DIR = REPO_ROOT / "evaluations" / "datasets"
RESULTS_DIR = REPO_ROOT / "evaluations" / "results"

STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "because", "as", "what",
    "which", "this", "that", "these", "those", "then", "just", "so", "than",
    "such", "both", "through", "about", "for", "is", "of", "while", "during",
    "to", "from", "in", "out", "on", "off", "again", "further", "then", "once",
    "here", "there", "when", "where", "why", "how", "all", "any", "both",
    "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not",
    "only", "own", "same", "so", "than", "too", "very", "can", "will", "just",
    "don", "should", "now", "it", "its", "was", "were", "are", "been", "our",
    "we", "i", "my", "me", "you", "your", "he", "she", "they", "them",
}


def tokenize(text: str) -> list[str]:
    """Clean and tokenize text into lowercase word tokens."""
    return [w for w in re.findall(r"\b[a-zA-Z0-9'-]+\b", text.lower()) if w]


def get_ngrams(tokens: list[str], n: int) -> Counter:
    """Generate n-gram counts from token list."""
    if len(tokens) < n:
        return Counter()
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def compute_rouge_n(hyp_tokens: list[str], ref_tokens: list[str], n: int) -> dict[str, float]:
    """Calculate ROUGE-N precision, recall, and F1."""
    if not hyp_tokens or not ref_tokens:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    hyp_ngrams = get_ngrams(hyp_tokens, n)
    ref_ngrams = get_ngrams(ref_tokens, n)

    if not hyp_ngrams or not ref_ngrams:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    overlap = sum((hyp_ngrams & ref_ngrams).values())
    total_hyp = sum(hyp_ngrams.values())
    total_ref = sum(ref_ngrams.values())

    prec = overlap / total_hyp if total_hyp > 0 else 0.0
    rec = overlap / total_ref if total_ref > 0 else 0.0
    f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

    return {"precision": round(prec, 4), "recall": round(rec, 4), "f1": round(f1, 4)}


def compute_rouge_l(hyp_tokens: list[str], ref_tokens: list[str]) -> dict[str, float]:
    """Calculate ROUGE-L based on Longest Common Subsequence (LCS)."""
    m, n = len(hyp_tokens), len(ref_tokens)
    if m == 0 or n == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m):
        for j in range(n):
            if hyp_tokens[i] == ref_tokens[j]:
                dp[i + 1][j + 1] = dp[i][j] + 1
            else:
                dp[i + 1][j + 1] = max(dp[i + 1][j], dp[i][j + 1])

    lcs_len = dp[m][n]
    prec = lcs_len / m if m > 0 else 0.0
    rec = lcs_len / n if n > 0 else 0.0
    f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

    return {"precision": round(prec, 4), "recall": round(rec, 4), "f1": round(f1, 4)}


def evaluate_key_fact_coverage(summary: str, key_facts: list[str]) -> float:
    """Measure proportion of atomic reference facts captured in the summary."""
    if not key_facts:
        return 1.0

    summary_tokens = set(tokenize(summary))
    captured = 0
    for fact in key_facts:
        fact_tokens = [w for w in tokenize(fact) if w not in STOPWORDS]
        if not fact_tokens:
            continue
        # If at least 50% of the key fact's informative tokens appear in summary
        matches = sum(1 for w in fact_tokens if w in summary_tokens)
        if matches / len(fact_tokens) >= 0.5:
            captured += 1

    return round(captured / len(key_facts), 4)


def evaluate_unsupported_claims(summary: str, transcript: str) -> dict[str, Any]:
    """Detect ungrounded claims or hallucinations not supported by the transcript."""
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", summary) if s.strip()]
    if not sentences:
        return {"unsupported_count": 0, "total_sentences": 0, "hallucination_rate": 0.0}

    transcript_tokens = set(tokenize(transcript))
    unsupported_sentences = []

    for sentence in sentences:
        tokens = [w for w in tokenize(sentence) if w not in STOPWORDS and len(w) > 3]
        if not tokens:
            continue
        # Count words not in transcript
        unsupported_tokens = [w for w in tokens if w not in transcript_tokens]
        # If more than 50% of key substantive words never appeared in the transcript
        if len(unsupported_tokens) / len(tokens) > 0.5:
            unsupported_sentences.append(sentence)

    rate = len(unsupported_sentences) / len(sentences) if sentences else 0.0
    return {
        "unsupported_count": len(unsupported_sentences),
        "total_sentences": len(sentences),
        "hallucination_rate": round(rate, 4),
        "unsupported_examples": unsupported_sentences[:2],
    }


def evaluate_set_overlap(predicted: list[str], reference: list[str]) -> dict[str, float]:
    """Calculate Precision, Recall, and F1 over lists of topics or action items."""
    pred_clean = {re.sub(r"[^a-z0-9]", "", p.lower()) for p in predicted if p and p.strip()}
    ref_clean = {re.sub(r"[^a-z0-9]", "", r.lower()) for r in reference if r and r.strip()}

    if not pred_clean and not ref_clean:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    if not pred_clean or not ref_clean:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    # Match tokens with exact match or substring containment (e.g. "sqlite" matches "sqlite3")
    matches = 0
    for p in pred_clean:
        if any(p == r or (len(p) > 3 and (p in r or r in p)) for r in ref_clean):
            matches += 1

    prec = matches / len(pred_clean) if pred_clean else 0.0
    rec = matches / len(ref_clean) if ref_clean else 0.0
    f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

    return {"precision": round(prec, 4), "recall": round(rec, 4), "f1": round(f1, 4)}


def run_extraction_evaluation(
    human_dataset_path: Path = DATASETS_DIR / "human_annotated_notes.json",
    synthetic_dataset_path: Path = DATASETS_DIR / "synthetic_test_fixtures.json",
    use_ollama: bool = True,
) -> dict[str, Any]:
    """Run extraction benchmark comparing Deterministic Fallback vs Ollama."""
    with open(human_dataset_path, "r", encoding="utf-8") as f:
        human_notes = json.load(f)

    with open(synthetic_dataset_path, "r", encoding="utf-8") as f:
        synthetic_notes = json.load(f)

    all_notes = human_notes + synthetic_notes

    models_to_test = ["deterministic_fallback"]
    if use_ollama:
        models_to_test.append("ollama_llama3.2")

    report: dict[str, Any] = {
        "datasets": {
            "human_notes_count": len(human_notes),
            "synthetic_fixtures_count": len(synthetic_notes),
            "total_count": len(all_notes),
        },
        "results_by_system": {},
    }

    for system_name in models_to_test:
        print(f"\nEvaluating system: {system_name}...")
        metrics_by_dataset: dict[str, list[dict[str, Any]]] = {
            "human_annotated": [],
            "synthetic_fixture": [],
        }

        for note in all_notes:
            transcript = note["transcript"]
            dtype = note.get("dataset_type", "human_annotated")

            if system_name == "deterministic_fallback":
                pred = _fallback_analysis(transcript)
            else:
                pred = analyze_transcript(transcript, use_ollama=True, model_name="llama3.2")

            # Tokenize summaries
            hyp_tokens = tokenize(pred.get("summary", ""))
            ref_tokens = tokenize(note["reference_summary"])

            r1 = compute_rouge_n(hyp_tokens, ref_tokens, 1)
            r2 = compute_rouge_n(hyp_tokens, ref_tokens, 2)
            rl = compute_rouge_l(hyp_tokens, ref_tokens)

            fact_cov = evaluate_key_fact_coverage(pred.get("summary", ""), note.get("key_facts", []))
            halluc = evaluate_unsupported_claims(pred.get("summary", ""), transcript)

            topics_eval = evaluate_set_overlap(pred.get("topics", []), note.get("reference_topics", []))
            actions_eval = evaluate_set_overlap(pred.get("action_items", []), note.get("reference_action_items", []))

            metrics_by_dataset[dtype].append({
                "note_id": note["id"],
                "rouge_1": r1,
                "rouge_2": r2,
                "rouge_l": rl,
                "key_fact_coverage": fact_cov,
                "hallucination_rate": halluc["hallucination_rate"],
                "topic_eval": topics_eval,
                "action_eval": actions_eval,
            })

        # Compute aggregate averages for this system
        def aggregate(metric_list: list[dict[str, Any]]) -> dict[str, float]:
            if not metric_list:
                return {}
            n = len(metric_list)
            return {
                "count": n,
                "rouge_1_f1": round(sum(m["rouge_1"]["f1"] for m in metric_list) / n, 4),
                "rouge_2_f1": round(sum(m["rouge_2"]["f1"] for m in metric_list) / n, 4),
                "rouge_l_f1": round(sum(m["rouge_l"]["f1"] for m in metric_list) / n, 4),
                "key_fact_coverage": round(sum(m["key_fact_coverage"] for m in metric_list) / n, 4),
                "hallucination_rate": round(sum(m["hallucination_rate"] for m in metric_list) / n, 4),
                "topic_precision": round(sum(m["topic_eval"]["precision"] for m in metric_list) / n, 4),
                "topic_recall": round(sum(m["topic_eval"]["recall"] for m in metric_list) / n, 4),
                "topic_f1": round(sum(m["topic_eval"]["f1"] for m in metric_list) / n, 4),
                "action_precision": round(sum(m["action_eval"]["precision"] for m in metric_list) / n, 4),
                "action_recall": round(sum(m["action_eval"]["recall"] for m in metric_list) / n, 4),
                "action_f1": round(sum(m["action_eval"]["f1"] for m in metric_list) / n, 4),
            }

        all_entries = metrics_by_dataset["human_annotated"] + metrics_by_dataset["synthetic_fixture"]
        report["results_by_system"][system_name] = {
            "human_annotated": aggregate(metrics_by_dataset["human_annotated"]),
            "synthetic_fixtures": aggregate(metrics_by_dataset["synthetic_fixture"]),
            "overall": aggregate(all_entries),
        }

    # Save Results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_json = RESULTS_DIR / "extraction_eval_results.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\n[OK] Machine-readable extraction results saved to: {out_json}")

    # Output CSV summary
    out_csv = RESULTS_DIR / "extraction_eval_summary.csv"
    save_extraction_csv(report, out_csv)
    print(f"[OK] Summary CSV saved to: {out_csv}")

    # Print clean comparison table
    print_comparison_table(report)

    return report


def save_extraction_csv(report: dict[str, Any], csv_path: Path) -> None:
    rows = []
    for sys_name, splits in report["results_by_system"].items():
        for split_name, metrics in splits.items():
            row = {"system": sys_name, "split": split_name, **metrics}
            rows.append(row)

    if rows:
        fieldnames = list(rows[0].keys())
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)


def print_comparison_table(report: dict[str, Any]) -> None:
    print("\n" + "=" * 90)
    print("EXTRACTION EVALUATION RESULTS: HUMAN-ANNOTATED BENCHMARK (ROUGE, TOPICS, ACTIONS, GROUNDING)")
    print("=" * 90)
    fmt = "{:<25} | {:<10} | {:<10} | {:<10} | {:<10} | {:<10} | {:<10}"
    print(fmt.format("System", "ROUGE-L", "Fact Cov", "Halluc%", "Topic-F1", "Action-P", "Action-F1"))
    print("-" * 90)

    for sys_name, splits in report["results_by_system"].items():
        h = splits.get("human_annotated", {})
        print(fmt.format(
            sys_name,
            str(h.get("rouge_l_f1", 0.0)),
            f"{h.get('key_fact_coverage', 0.0)*100:.1f}%",
            f"{h.get('hallucination_rate', 0.0)*100:.1f}%",
            str(h.get("topic_f1", 0.0)),
            str(h.get("action_precision", 0.0)),
            str(h.get("action_f1", 0.0)),
        ))
    print("=" * 90)


def main() -> None:
    parser = argparse.ArgumentParser(description="EchoNote Extraction Evaluation")
    parser.add_argument("--no-ollama", action="store_true", help="Run deterministic fallback only")
    args = parser.parse_args()

    run_extraction_evaluation(use_ollama=not args.no_ollama)


if __name__ == "__main__":
    main()
