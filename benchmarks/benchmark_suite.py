"""Comprehensive reproducible benchmark suite for EchoNote.

Measures:
1. Whisper model cold loading (isolated subprocess)
2. Cold vs warm transcription
3. Ollama analysis (LLM vs deterministic fallback)
4. Embedding generation (Ollama nomic-embed-text)
5. SQLite persistence & FTS5 trigger indexing
6. Keyword retrieval (FTS5)
7. Semantic retrieval (Cosine similarity over vector store)
8. Journal Question Answering (Retrieval + Evidence + Grounding)
9. Complete Audio-to-Journal pipeline
10. Combined memory pipeline session

Outputs machine-readable JSON and CSV with statistics:
mean, median, std_dev, p90, p95, min, max, CPU seconds, peak RSS bytes.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Callable

import psutil
import torch

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from journal_app.database import JournalStore
from journal_app.services import (
    _fallback_analysis,
    analyze_transcript,
    ask_journal,
    embed_text,
    transcribe,
)

RESULTS_DIR = REPO_ROOT / "benchmarks" / "results"


def get_system_environment() -> dict[str, Any]:
    """Capture comprehensive hardware, OS, and software platform metadata."""
    cpu_info = platform.processor() or "Unknown CPU"
    try:
        total_ram_gb = round(psutil.virtual_memory().total / (1024**3), 2)
    except Exception:
        total_ram_gb = None

    return {
        "os_name": platform.system(),
        "os_release": platform.release(),
        "os_version": platform.version(),
        "machine": platform.machine(),
        "processor": cpu_info,
        "physical_cores": psutil.cpu_count(logical=False),
        "logical_cores": psutil.cpu_count(logical=True),
        "total_ram_gb": total_ram_gb,
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def compute_statistics(measurements: list[float]) -> dict[str, float]:
    """Compute central tendency, dispersion, and percentiles."""
    if not measurements:
        return {"count": 0, "mean": 0.0, "median": 0.0, "std_dev": 0.0, "p90": 0.0, "p95": 0.0, "min": 0.0, "max": 0.0}

    sorted_vals = sorted(measurements)
    n = len(sorted_vals)
    mean_val = statistics.mean(sorted_vals)
    median_val = statistics.median(sorted_vals)
    std_val = statistics.stdev(sorted_vals) if n > 1 else 0.0

    def percentile(p: float) -> float:
        k = (n - 1) * (p / 100.0)
        f = math.floor(k)
        c = math.ceil(k)
        if f == c:
            return sorted_vals[int(k)]
        d0 = sorted_vals[int(f)] * (c - k)
        d1 = sorted_vals[int(c)] * (k - f)
        return d0 + d1

    return {
        "count": n,
        "mean": round(mean_val, 5),
        "median": round(median_val, 5),
        "std_dev": round(std_val, 5),
        "p90": round(percentile(90), 5),
        "p95": round(percentile(95), 5),
        "min": round(sorted_vals[0], 5),
        "max": round(sorted_vals[-1], 5),
    }


def measure_execution(func: Callable, *args: Any, **kwargs: Any) -> tuple[Any, dict[str, float]]:
    """Execute a function while recording wall time, process CPU time, and memory."""
    proc = psutil.Process()
    mem_before = proc.memory_info().rss
    t_start = time.perf_counter()
    cpu_start = time.process_time()

    result = func(*args, **kwargs)

    t_end = time.perf_counter()
    cpu_end = time.process_time()
    mem_after = proc.memory_info().rss

    wall_sec = t_end - t_start
    cpu_sec = cpu_end - cpu_start
    rss_peak = max(mem_after, mem_before)

    return result, {
        "wall_seconds": wall_sec,
        "cpu_seconds": cpu_sec,
        "peak_rss_bytes": rss_peak,
        "rss_delta_bytes": mem_after - mem_before,
    }


def benchmark_whisper_cold_load(model_name: str = "base.en") -> dict[str, Any]:
    """Measure cold loading of Whisper in a completely clean subprocess."""
    script = f"""
import time, psutil, whisper
proc = psutil.Process()
mem_init = proc.memory_info().rss
t0 = time.perf_counter()
c0 = time.process_time()
model = whisper.load_model('{model_name}')
t1 = time.perf_counter()
c1 = time.process_time()
mem_final = proc.memory_info().rss
import json
print(json.dumps({{
    'wall_seconds': round(t1 - t0, 4),
    'cpu_seconds': round(c1 - c0, 4),
    'initial_rss_bytes': mem_init,
    'final_rss_bytes': mem_final,
    'model_rss_bytes': mem_final - mem_init
}}))
"""
    cmd = [sys.executable, "-c", script]
    res = subprocess.run(cmd, capture_output=True, text=True, cwd=str(REPO_ROOT))
    if res.returncode != 0:
        return {"error": res.stderr.strip()}
    for line in res.stdout.strip().splitlines():
        if line.startswith("{") and line.endswith("}"):
            try:
                return json.loads(line)
            except Exception:
                pass
    return {"raw_output": res.stdout}


def benchmark_whisper_transcription(
    audio_path: Path,
    model_name: str = "base.en",
    warm_iterations: int = 3,
) -> dict[str, Any]:
    """Benchmark cold transcription (subprocess) vs warm transcription."""
    results: dict[str, Any] = {"model": model_name, "audio_file": audio_path.name}

    # 1. Cold transcription in isolated process
    cold_script = f"""
import time, psutil
proc = psutil.Process()
from journal_app.services import transcribe
t0 = time.perf_counter()
c0 = time.process_time()
text = transcribe(r'{str(audio_path)}', model_name='{model_name}')
t1 = time.perf_counter()
c1 = time.process_time()
import json
print(json.dumps({{
    'wall_seconds': round(t1 - t0, 4),
    'cpu_seconds': round(c1 - c0, 4),
    'peak_rss_bytes': proc.memory_info().rss,
    'text_len': len(text)
}}))
"""
    res = subprocess.run([sys.executable, "-c", cold_script], capture_output=True, text=True, cwd=str(REPO_ROOT))
    if res.returncode == 0:
        for line in res.stdout.strip().splitlines():
            if line.startswith("{") and line.endswith("}"):
                try:
                    results["cold_transcription"] = json.loads(line)
                    break
                except Exception:
                    pass
    else:
        results["cold_transcription"] = {"error": res.stderr.strip()}

    # 2. Warm transcription in this process
    # Run one warmup to prime model weights in cache
    try:
        _ = transcribe(audio_path, model_name=model_name)
        warm_times = []
        cpu_times = []
        for _ in range(warm_iterations):
            _, metrics = measure_execution(transcribe, audio_path, model_name=model_name)
            warm_times.append(metrics["wall_seconds"])
            cpu_times.append(metrics["cpu_seconds"])
        results["warm_transcription"] = {
            "wall_seconds": compute_statistics(warm_times),
            "cpu_seconds": compute_statistics(cpu_times),
        }
    except Exception as err:
        results["warm_transcription"] = {"error": str(err)}

    return results


def benchmark_extraction(
    sample_text: str,
    iterations: int = 5,
    test_ollama: bool = True,
) -> dict[str, Any]:
    """Benchmark deterministic fallback extraction vs Ollama LLM extraction."""
    results: dict[str, Any] = {}

    # Deterministic fallback
    fb_walls = []
    fb_cpus = []
    for _ in range(iterations * 10):
        _, m = measure_execution(_fallback_analysis, sample_text)
        fb_walls.append(m["wall_seconds"])
        fb_cpus.append(m["cpu_seconds"])
    results["deterministic_fallback"] = {
        "wall_seconds": compute_statistics(fb_walls),
        "cpu_seconds": compute_statistics(fb_cpus),
    }

    # Ollama analysis
    if test_ollama:
        ollama_walls = []
        ollama_cpus = []
        ollama_success = True
        for i in range(iterations):
            try:
                res, m = measure_execution(analyze_transcript, sample_text, True, "llama3.2")
                ollama_walls.append(m["wall_seconds"])
                ollama_cpus.append(m["cpu_seconds"])
            except Exception as err:
                ollama_success = False
                results["ollama_analysis"] = {"error": str(err)}
                break
        if ollama_success and ollama_walls:
            results["ollama_analysis"] = {
                "wall_seconds": compute_statistics(ollama_walls),
                "cpu_seconds": compute_statistics(ollama_cpus),
            }

    return results


def benchmark_embeddings(sample_texts: list[str], iterations: int = 5) -> dict[str, Any]:
    """Benchmark embedding generation latency using nomic-embed-text."""
    walls = []
    dims = []
    for i in range(iterations):
        for text in sample_texts:
            t0 = time.perf_counter()
            data = embed_text(text, model_name="nomic-embed-text")
            elapsed = time.perf_counter() - t0
            if data and data.get("vector"):
                walls.append(elapsed)
                dims.append(data["dim"])
    if not walls:
        return {"available": False, "reason": "Ollama nomic-embed-text unreachable or not generating vectors"}
    return {
        "available": True,
        "model": "nomic-embed-text",
        "dim": dims[0] if dims else 0,
        "wall_seconds": compute_statistics(walls),
    }


def benchmark_sqlite_persistence(num_records: int = 500) -> dict[str, Any]:
    """Benchmark SQLite persistence and FTS5 indexing insert throughput."""
    tmp_dir = tempfile.mkdtemp()
    db_path = Path(tmp_dir) / "bench_store.sqlite3"
    try:
        store = JournalStore(db_path)
        insert_times = []
        for i in range(num_records):
            entry = {
                "title": f"Thought {i}: Architectural Planning",
                "summary": f"Discussion regarding microservice architectures and databases for session {i}.",
                "transcript": f"In this session {i}, we analyzed the trade-offs between monolithic and microservice designs.",
                "topics": ["architecture", "scaling", f"topic_{i % 10}"],
                "mood": "focused",
                "action_items": [f"Benchmark query performance for cluster {i % 5}"],
                "embedding": [0.1 * (i % 5), 0.2 * (i % 3)],
                "embedding_model": "test-embed-2d",
            }
            t0 = time.perf_counter()
            store.add(entry)
            insert_times.append(time.perf_counter() - t0)

        total_time = sum(insert_times)
        throughput = num_records / total_time if total_time > 0 else 0
        return {
            "record_count": num_records,
            "total_wall_seconds": round(total_time, 4),
            "throughput_records_per_sec": round(throughput, 2),
            "insert_wall_seconds": compute_statistics(insert_times),
            "db_size_bytes": db_path.stat().st_size,
        }
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def benchmark_retrieval(num_records: int = 500, query_reps: int = 20) -> dict[str, Any]:
    """Benchmark FTS5 keyword retrieval, semantic retrieval, and hybrid retrieval."""
    tmp_dir = tempfile.mkdtemp()
    db_path = Path(tmp_dir) / "bench_retrieval.sqlite3"
    try:
        store = JournalStore(db_path)
        # Populate dataset
        for i in range(num_records):
            angle = (2.0 * math.pi * i) / num_records
            vec = [math.cos(angle), math.sin(angle)]
            store.add({
                "title": f"Entry {i} on Deep Learning and Transformers",
                "summary": f"Studying attention mechanisms, query key value projections in iteration {i}.",
                "transcript": f"Transformers revolutionized NLP and audio processing. Record {i}.",
                "topics": ["ai", "transformers", f"topic_{i % 5}"],
                "mood": "reflective",
                "action_items": [f"Read attention is all you need paper part {i % 3}"],
                "embedding": vec,
                "embedding_model": "unit-circle-2d",
            })

        # Benchmark Keyword (FTS5) Search
        fts_times = []
        for _ in range(query_reps):
            t0 = time.perf_counter()
            res = store.search("Transformers attention mechanisms", limit=20)
            fts_times.append(time.perf_counter() - t0)

        # Benchmark Semantic Search
        sem_times = []
        query_vec = [1.0, 0.0]
        for _ in range(query_reps):
            t0 = time.perf_counter()
            res = store.semantic_search(query_vec, limit=20)
            sem_times.append(time.perf_counter() - t0)

        # Benchmark Hybrid Search
        hybrid_times = []
        for _ in range(query_reps):
            t0 = time.perf_counter()
            res = store.hybrid_search("Transformers", vector=query_vec, limit=20)
            hybrid_times.append(time.perf_counter() - t0)

        return {
            "collection_size": num_records,
            "keyword_fts5_wall_seconds": compute_statistics(fts_times),
            "semantic_search_wall_seconds": compute_statistics(sem_times),
            "hybrid_search_wall_seconds": compute_statistics(hybrid_times),
        }
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def benchmark_qa_grounding(num_entries: int = 30, test_ollama: bool = True) -> dict[str, Any]:
    """Benchmark Question Answering latency and grounding behavior."""
    entries = [
        {
            "id": i + 1,
            "created_at": f"2026-11-{10 + (i % 15):02d}T10:00:00",
            "title": f"Research Lab Meeting {i}",
            "summary": f"Discussed automated code refactoring and AST transformation in meeting {i}.",
            "topics": ["research", "ast"],
            "action_items": [f"Prepare slides for meeting {i}"],
        }
        for i in range(num_entries)
    ]

    question = "What was discussed in meeting 3?"
    results: dict[str, Any] = {}

    # Deterministic fallback QA
    fb_times = []
    for _ in range(10):
        t0 = time.perf_counter()
        resp = ask_journal(question, entries, use_ollama=False)
        fb_times.append(time.perf_counter() - t0)
    results["deterministic_qa"] = {
        "wall_seconds": compute_statistics(fb_times),
        "sample_answer": resp["answer"],
        "abstention": resp["abstention"],
    }

    # Ollama QA
    if test_ollama:
        try:
            ollama_times = []
            for _ in range(3):
                t0 = time.perf_counter()
                resp = ask_journal(question, entries, model_name="llama3.2", use_ollama=True)
                ollama_times.append(time.perf_counter() - t0)
            results["ollama_qa"] = {
                "wall_seconds": compute_statistics(ollama_times),
                "sample_answer": resp["answer"],
                "abstention": resp["abstention"],
                "evidence_count": len(resp["evidence"]),
            }
        except Exception as err:
            results["ollama_qa"] = {"error": str(err)}

    return results


def benchmark_complete_pipeline(
    audio_path: Path,
    use_ollama: bool = True,
    iterations: int = 3,
) -> dict[str, Any]:
    """Benchmark end-to-end audio-to-journal pipeline."""
    tmp_dir = tempfile.mkdtemp()
    db_path = Path(tmp_dir) / "bench_pipeline.sqlite3"
    try:
        store = JournalStore(db_path)
        e2e_walls = []
        e2e_cpus = []

        # Prime whisper model once
        _ = transcribe(audio_path, "base.en")

        for _ in range(iterations):
            proc = psutil.Process()
            t0 = time.perf_counter()
            c0 = time.process_time()

            # 1. Transcribe audio
            transcript = transcribe(audio_path, "base.en")
            # 2. Extract title, summary, topics, actions
            analysis = analyze_transcript(transcript, use_ollama=use_ollama, model_name="llama3.2")
            # 3. Embed text
            embed_prompt = f"{analysis['title']}. {analysis['summary']}. {' '.join(analysis['topics'])}"
            embed_info = embed_text(embed_prompt, model_name="nomic-embed-text") if use_ollama else None
            # 4. Store in SQLite
            record = {
                **analysis,
                "transcript": transcript,
                "source": "upload",
                "embedding": embed_info["vector"] if embed_info else None,
                "embedding_model": embed_info["model"] if embed_info else None,
                "embedding_dim": embed_info["dim"] if embed_info else None,
            }
            eid = store.add(record)

            e2e_walls.append(time.perf_counter() - t0)
            e2e_cpus.append(time.process_time() - c0)

        return {
            "iterations": iterations,
            "use_ollama": use_ollama,
            "wall_seconds": compute_statistics(e2e_walls),
            "cpu_seconds": compute_statistics(e2e_cpus),
            "final_entry_id": eid,
        }
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def run_full_benchmark_suite(
    audio_file: Path | None = None,
    run_ollama: bool = True,
) -> dict[str, Any]:
    """Run full benchmark suite and export machine-readable results."""
    print("=" * 60)
    print("Starting EchoNote Full Pipeline Benchmark Suite")
    print("=" * 60)

    env = get_system_environment()
    print(f"System: {env['processor']} | RAM: {env['total_ram_gb']} GB | OS: {env['os_name']} {env['os_release']}")
    print(f"PyTorch: {env['torch_version']} | CUDA: {env['cuda_available']} ({env['device_name']})")

    results: dict[str, Any] = {"environment": env, "benchmarks": {}}

    # 1. Whisper Cold Load
    print("\n[1/8] Benchmarking Whisper Cold Loading (base.en)...")
    cold_load = benchmark_whisper_cold_load("base.en")
    results["benchmarks"]["whisper_cold_load"] = cold_load
    print(f"  Cold Load Wall Time: {cold_load.get('wall_seconds')} s | RSS delta: {cold_load.get('model_rss_bytes', 0) / (1024**2):.1f} MB")

    # 2. Whisper Transcription (Cold & Warm)
    if audio_file and audio_file.exists():
        print(f"\n[2/8] Benchmarking Whisper Transcription ({audio_file.name})...")
        trans_res = benchmark_whisper_transcription(audio_file, "base.en", warm_iterations=3)
        results["benchmarks"]["whisper_transcription"] = trans_res
        if "cold_transcription" in trans_res and "wall_seconds" in trans_res["cold_transcription"]:
            print(f"  Cold Transcription: {trans_res['cold_transcription']['wall_seconds']} s")
        if "warm_transcription" in trans_res and "wall_seconds" in trans_res["warm_transcription"]:
            print(f"  Warm Transcription Median: {trans_res['warm_transcription']['wall_seconds']['median']} s")
    else:
        print("\n[2/8] Skipping audio transcription benchmark (no audio file specified)")

    # 3. Extraction: Deterministic vs LLM
    print("\n[3/8] Benchmarking Journal Extraction (Deterministic vs Ollama)...")
    sample_text = (
        "Today was productive. I completed the database indexing optimization and fixed the vector search bug. "
        "Tomorrow I need to finish writing the empirical evaluation section and email the draft to the team."
    )
    extract_res = benchmark_extraction(sample_text, iterations=3, test_ollama=run_ollama)
    results["benchmarks"]["extraction"] = extract_res
    print(f"  Deterministic Fallback Mean: {extract_res['deterministic_fallback']['wall_seconds']['mean']*1000:.3f} ms")
    if "ollama_analysis" in extract_res and "wall_seconds" in extract_res["ollama_analysis"]:
        print(f"  Ollama Analysis Median: {extract_res['ollama_analysis']['wall_seconds']['median']:.3f} s")

    # 4. Embeddings
    print("\n[4/8] Benchmarking Embedding Generation (nomic-embed-text)...")
    embed_samples = [
        "Productive coding session on SQLite full-text search.",
        "Reflecting on machine learning architectures and tokenizers.",
        "Team meeting discussing sprint priorities and deadlines.",
    ]
    embed_res = benchmark_embeddings(embed_samples, iterations=3) if run_ollama else {"available": False}
    results["benchmarks"]["embeddings"] = embed_res
    if embed_res.get("available"):
        print(f"  Embedding Latency Median: {embed_res['wall_seconds']['median']:.3f} s (dim: {embed_res['dim']})")

    # 5. SQLite Persistence & Indexing
    print("\n[5/8] Benchmarking SQLite Persistence & FTS5 Indexing (500 entries)...")
    db_res = benchmark_sqlite_persistence(500)
    results["benchmarks"]["sqlite_persistence"] = db_res
    print(f"  Persistence Throughput: {db_res['throughput_records_per_sec']} records/sec | Single Insert Median: {db_res['insert_wall_seconds']['median']*1000:.3f} ms")

    # 6. Keyword, Semantic, and Hybrid Retrieval
    print("\n[6/8] Benchmarking Retrieval Strategies (500 entries)...")
    ret_res = benchmark_retrieval(500, query_reps=20)
    results["benchmarks"]["retrieval"] = ret_res
    print(f"  FTS5 Keyword Median: {ret_res['keyword_fts5_wall_seconds']['median']*1000:.3f} ms")
    print(f"  Semantic Search Median: {ret_res['semantic_search_wall_seconds']['median']*1000:.3f} ms")
    print(f"  Hybrid Search Median: {ret_res['hybrid_search_wall_seconds']['median']*1000:.3f} ms")

    # 7. Q&A Grounding
    print("\n[7/8] Benchmarking Grounded Journal Q&A...")
    qa_res = benchmark_qa_grounding(num_entries=25, test_ollama=run_ollama)
    results["benchmarks"]["journal_qa"] = qa_res
    print(f"  Deterministic QA Latency: {qa_res['deterministic_qa']['wall_seconds']['mean']*1000:.3f} ms")
    if "ollama_qa" in qa_res and "wall_seconds" in qa_res["ollama_qa"]:
        print(f"  Ollama Grounded QA Median: {qa_res['ollama_qa']['wall_seconds']['median']:.3f} s")

    # 8. Complete Pipeline
    if audio_file and audio_file.exists():
        print("\n[8/8] Benchmarking Complete End-to-End Audio Pipeline...")
        pipeline_res = benchmark_complete_pipeline(audio_file, use_ollama=run_ollama, iterations=2)
        results["benchmarks"]["complete_pipeline"] = pipeline_res
        print(f"  End-to-End Pipeline Median: {pipeline_res['wall_seconds']['median']:.3f} s")

    # Save Results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    json_path = RESULTS_DIR / "benchmark_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n[OK] Machine-readable results saved to: {json_path}")

    # Save Flat CSV Summary
    csv_path = RESULTS_DIR / "benchmark_summary.csv"
    save_csv_summary(results, csv_path)
    print(f"[OK] Benchmark CSV summary saved to: {csv_path}")

    return results


def save_csv_summary(results: dict[str, Any], csv_path: Path) -> None:
    """Flatten key statistics into a tabular CSV report."""
    rows = []
    benchmarks = results.get("benchmarks", {})

    for bench_name, data in benchmarks.items():
        if not isinstance(data, dict):
            continue
        for key, val in data.items():
            if isinstance(val, dict) and "mean" in val:
                rows.append({
                    "benchmark": bench_name,
                    "metric": key,
                    "count": val.get("count"),
                    "mean_sec": val.get("mean"),
                    "median_sec": val.get("median"),
                    "std_dev_sec": val.get("std_dev"),
                    "p90_sec": val.get("p90"),
                    "p95_sec": val.get("p95"),
                    "min_sec": val.get("min"),
                    "max_sec": val.get("max"),
                })
            elif key == "throughput_records_per_sec":
                rows.append({
                    "benchmark": bench_name,
                    "metric": "throughput_records_per_sec",
                    "count": data.get("record_count"),
                    "mean_sec": val,
                    "median_sec": val,
                    "std_dev_sec": 0.0,
                    "p90_sec": val,
                    "p95_sec": val,
                    "min_sec": val,
                    "max_sec": val,
                })

    if rows:
        fieldnames = ["benchmark", "metric", "count", "mean_sec", "median_sec", "std_dev_sec", "p90_sec", "p95_sec", "min_sec", "max_sec"]
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="EchoNote Full Pipeline Benchmark Suite")
    parser.add_argument("--audio", type=Path, default=REPO_ROOT / "McDonalds_LNG_061019.wav", help="Audio file path")
    parser.add_argument("--no-ollama", action="store_true", help="Skip live Ollama calls")
    args = parser.parse_args()

    audio_target = args.audio if args.audio.exists() else None
    run_full_benchmark_suite(audio_file=audio_target, run_ollama=not args.no_ollama)


if __name__ == "__main__":
    main()
