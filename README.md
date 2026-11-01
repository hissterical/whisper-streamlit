# EchoNote

EchoNote is a polished, local-first lightweight AI workspace with two intentionally separate flows:

- **Analyze** is a private scratchpad. It records or uploads audio, transcribes it, and produces a summary without saving anything to journal memory.
- **Journal** is the durable memory space. Its own record/upload controls analyze audio and save the resulting thought to SQLite for search, filtering, and questions.

## Run locally

Requirements: Python 3.13+, [`uv`](https://docs.astral.sh/uv/), and optionally [Ollama](https://ollama.com/).

```bash
uv sync
uv run uvicorn main:app --reload
```

Open http://127.0.0.1:8000. For richer summaries and Ask your journal, run `ollama serve` and `ollama pull llama3.2`. Without Ollama, deterministic local analysis is used.

The Analyze page calls `POST /api/analyze` and never writes an entry. The Journal capture dialog calls `POST /api/journal/entries`, which writes the analyzed thought to the local journal.

## Whisper cache behavior

Whisper is **not loaded or downloaded when the server starts**. It is loaded lazily on the first Analyze request, then cached process-wide for subsequent requests. Whisper's normal persistent local model cache is used, so restarting the server does not redownload an already-cached model. Audio files are stored by content hash under `data/audio_cache`; journal entries are stored in `data/echonote.sqlite3`.

## Architecture

`main.py` exposes FastAPI health, non-persistent analysis, persistent journal capture, entry search/filter, and journal Q&A endpoints and serves the static frontend. `frontend/` contains the responsive HTML/CSS/JavaScript app shell. `journal_app/services.py` owns Whisper/Ollama processing and fallbacks; `journal_app/database.py` owns SQLite persistence. User-provided content is rendered with DOM text escaping rather than injected HTML.

The Journal data layer uses SQLite FTS5 for exact full-text retrieval. When Ollama has the `nomic-embed-text` model available, saved entries also receive local embeddings and semantic search can be requested with the meaning-search field. If that model is unavailable, the application explicitly falls back to FTS5. The graph endpoint derives topic co-occurrence links, and the calendar endpoint derives date-based journal activity.

For research measurements, run:

```bash
uv run python benchmarks/benchmark.py
uv run python benchmarks/benchmark.py --audio path/to/audio.wav --reference "known transcript"
```

The benchmark reports transcription wall time, process CPU time, optional resident memory (when `psutil` is installed), local extraction latency, and word error rate when a reference transcript is supplied. For a paper, report the hardware, Whisper model, audio duration, Ollama model, number of repetitions, mean, standard deviation, and peak memory rather than relying on one run.

## Research framing

EchoNote is a lightweight, privacy-preserving generative-AI system for local spoken-thought organization. The UI is the human-facing experimental instrument: it provides controlled Analyze and Journal workflows. The FastAPI layer is the reproducible system boundary: it makes transcription, persistence, search, graph, calendar, and question-answering behavior testable independently of the browser. SQLite provides a low-overhead local data layer; Whisper provides speech recognition; Ollama provides optional local generation and embeddings.

The central experimental question is whether a small local stack can provide useful voice-note organization with acceptable latency, memory use, and privacy compared with a cloud pipeline. Suitable evaluation dimensions are transcription word error rate, summary/metadata usefulness, end-to-end latency, CPU time, peak memory, storage size, and search relevance. The current benchmark measures WER and system performance; human evaluation of summaries and metadata should be added as a separate annotated study.

## Validation

```bash
uv lock
uv run python -m compileall main.py journal_app
uv run python -m unittest discover
```
