"""Reproducible lightweight-AI measurements for the EchoNote paper.

Examples:
    uv run python benchmarks/benchmark.py
    uv run python benchmarks/benchmark.py --audio sample.wav --reference "known transcript"
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from journal_app.services import analyze_transcript, transcribe


def word_error_rate(reference: str, hypothesis: str) -> float:
    expected = reference.lower().split()
    actual = hypothesis.lower().split()
    matrix = list(range(len(actual) + 1))
    for row, expected_word in enumerate(expected, 1):
        previous = matrix[0]
        matrix[0] = row
        for column, actual_word in enumerate(actual, 1):
            current = matrix[column]
            matrix[column] = min(
                matrix[column] + 1,
                matrix[column - 1] + 1,
                previous + (expected_word != actual_word),
            )
            previous = current
    return matrix[-1] / max(len(expected), 1)


def measure(function, *args):
    start = time.perf_counter()
    cpu_start = time.process_time()
    result = function(*args)
    elapsed = time.perf_counter() - start
    cpu = time.process_time() - cpu_start
    try:
        import psutil
        memory = psutil.Process().memory_info().rss
    except ImportError:
        memory = None
    return result, {"wall_seconds": round(elapsed, 4), "cpu_seconds": round(cpu, 4), "peak_rss_bytes": memory}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", type=Path)
    parser.add_argument("--reference")
    args = parser.parse_args()
    report: dict = {"configuration": {"audio": str(args.audio) if args.audio else None, "model": "base.en"}}
    if args.audio:
        transcript, metrics = measure(transcribe, args.audio)
        report["transcription"] = metrics
        report["word_error_rate"] = word_error_rate(args.reference, transcript) if args.reference else None
    sample = "I need to review the project plan and call Sam tomorrow."
    _, report["local_extraction"] = measure(analyze_transcript, sample, False)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
