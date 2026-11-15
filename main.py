from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path
from typing import Annotated, Any

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from journal_app.database import JournalStore
from journal_app.models import (
    AnalyzeResponse,
    AskRequest,
    AskResponse,
    EntriesListResponse,
    JournalEntryResponse,
)
from journal_app.services import (
    analyze_transcript,
    ask_journal,
    embed_text,
    get_embedding_vector,
    transcribe,
)
from journal_app.validation import validate_audio_bytes

LOGGER = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parent
STATIC = ROOT / "frontend"
STORE = JournalStore(os.getenv("ECHONOTE_DB", "data/echonote.sqlite3"))
UPLOAD_DIR = ROOT / "data" / "audio_cache"

app = FastAPI(title="EchoNote", version="1.1")
if STATIC.exists():
    app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.get("/", include_in_schema=False)
def home() -> FileResponse:
    index_path = STATIC / "index.html"
    if not index_path.exists():
        raise HTTPException(404, "Frontend static files not found.")
    return FileResponse(index_path)


@app.get("/api/health")
def health() -> dict[str, Any]:
    # Check Whisper cache status
    from journal_app.services import _WHISPER_LOCK, _load_whisper_model

    whisper_cached = False
    with _WHISPER_LOCK:
        whisper_cached = _load_whisper_model.cache_info().currsize > 0

    # Probe Ollama availability
    ollama_ok = False
    try:
        import httpx
        res = httpx.get("http://127.0.0.1:11434/api/tags", timeout=httpx.Timeout(1.0, connect=0.1))
        ollama_ok = res.status_code == 200
    except Exception:
        ollama_ok = False

    return {
        "status": "ok",
        "whisper": "cached" if whisper_cached else "lazy",
        "ollama_available": ollama_ok,
    }


@app.get("/api/entries", response_model=EntriesListResponse)
def entries(
    search: Annotated[str, Query(max_length=200)] = "",
    topic: Annotated[str, Query(max_length=80)] = "",
    mood: Annotated[str, Query(max_length=80)] = "",
    has_actions: bool = False,
    semantic: Annotated[str, Query(max_length=500)] = "",
    mode: Annotated[str, Query(pattern="^(auto|keyword|semantic|hybrid)$")] = "auto",
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
) -> dict[str, Any]:
    search_clean = search.strip()
    semantic_clean = semantic.strip()

    effective_mode = mode
    results: list[dict[str, Any]] = []

    if mode == "auto":
        if semantic_clean:
            effective_mode = "semantic"
        else:
            effective_mode = "keyword"

    if effective_mode in {"semantic", "hybrid"}:
        query_text = semantic_clean or search_clean
        embed_data = embed_text(query_text) if query_text else None

        if embed_data and effective_mode == "hybrid":
            results = STORE.hybrid_search(
                query=search_clean,
                vector=embed_data["vector"],
                model_name=embed_data["model"],
                limit=limit,
                topic=topic,
                mood=mood,
                has_actions=has_actions,
            )
            actual_mode = "hybrid"
        elif embed_data:
            results = STORE.semantic_search(
                vector=embed_data["vector"],
                model_name=embed_data["model"],
                limit=limit,
            )
            actual_mode = "semantic"
        else:
            # Fall back to keyword search if embedding generation is unavailable
            results = STORE.search(
                query=query_text,
                limit=limit,
                topic=topic,
                mood=mood,
                has_actions=has_actions,
            )
            actual_mode = "keyword"
    else:
        results = STORE.search(
            query=search_clean,
            limit=limit,
            topic=topic,
            mood=mood,
            has_actions=has_actions,
        )
        actual_mode = "keyword"

    return {
        "entries": results,
        "facets": STORE.facets(),
        "search_mode": actual_mode,
        "total": len(results),
    }


def _read_and_validate_audio(audio: UploadFile) -> tuple[bytes, str, str]:
    content = audio.file.read()
    validation = validate_audio_bytes(content, filename=audio.filename)
    if not validation.is_valid:
        error_msg = validation.error or "Invalid audio file"
        if "exceeds maximum size" in error_msg:
            raise HTTPException(413, error_msg)
        if "Unsupported audio file extension" in error_msg:
            raise HTTPException(415, error_msg)
        raise HTTPException(400, error_msg)

    ext = Path(audio.filename or f"audio.{validation.format}").suffix.lower()
    if not ext:
        ext = f".{validation.format}"
    digest = hashlib.sha256(content).hexdigest()
    return content, digest, ext


@app.post("/api/analyze", response_model=AnalyzeResponse)
def analyze(
    audio: Annotated[UploadFile, File(...)],
) -> dict[str, Any]:
    """Analyze audio without persisting it in the journal. Always cleans up temporary audio."""
    content, digest, ext = _read_and_validate_audio(audio)
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    temp_path = UPLOAD_DIR / f"temp_{digest}{ext}"

    try:
        temp_path.write_bytes(content)
        transcript = transcribe(temp_path)
        if not transcript:
            raise HTTPException(422, "No speech was detected in the audio file.")

        analysis = analyze_transcript(transcript)
        return {**analysis, "transcript": transcript}
    except HTTPException:
        raise
    except Exception as error:
        LOGGER.exception("Audio analysis failed: %s", error)
        raise HTTPException(500, f"Could not analyze audio: {error}") from error
    finally:
        if temp_path.exists():
            try:
                temp_path.unlink()
            except OSError as err:
                LOGGER.warning("Could not delete temporary audio file %s: %s", temp_path, err)


@app.post("/api/journal/entries", response_model=JournalEntryResponse)
def create_journal_entry(
    audio: Annotated[UploadFile, File(...)],
    source: str = Query("upload", pattern="^(upload|microphone)$"),
) -> dict[str, Any]:
    """Analyze audio and persist the resulting thought into durable journal storage."""
    content, digest, ext = _read_and_validate_audio(audio)
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    target_path = UPLOAD_DIR / f"{digest}{ext}"
    is_new_file = not target_path.exists()

    try:
        if is_new_file:
            target_path.write_bytes(content)

        transcript = transcribe(target_path)
        if not transcript:
            raise HTTPException(422, "No speech was detected in the audio file.")

        analysis = analyze_transcript(transcript)

        # Generate embedding vector
        embed_prompt = f"{analysis['title']}. {analysis['summary']}. {' '.join(analysis['topics'])}"
        embed_info = embed_text(embed_prompt)

        try:
            relative_audio_path = str(target_path.relative_to(ROOT))
        except ValueError:
            relative_audio_path = str(target_path)
        record_payload = {
            **analysis,
            "transcript": transcript,
            "source": source,
            "audio_path": relative_audio_path,
            "embedding": embed_info["vector"] if embed_info else None,
            "embedding_model": embed_info["model"] if embed_info else None,
            "embedding_dim": embed_info["dim"] if embed_info else None,
        }

        entry_id = STORE.add(record_payload)
        saved_entry = STORE.get_entry(entry_id)
        if not saved_entry:
            raise HTTPException(500, "Failed to retrieve newly persisted journal entry.")
        return saved_entry

    except Exception:
        # If persistence fails for any reason and this was a newly created file, clean it up
        if is_new_file and target_path.exists():
            try:
                target_path.unlink()
            except OSError:
                pass
        raise


@app.get("/api/graph")
def graph() -> dict[str, list[dict[str, Any]]]:
    return STORE.graph()


@app.get("/api/calendar")
def calendar(year: int | None = None, month: int | None = None) -> dict[str, list[dict[str, Any]]]:
    return {"entries": STORE.calendar(year, month)}


@app.post("/api/ask", response_model=AskResponse)
def ask(request: AskRequest) -> dict[str, Any]:
    question_clean = request.question.strip()

    # Retrieve entries relevant to the question across the whole journal, rather than fixed recent entries
    query_vec = get_embedding_vector(question_clean)
    if query_vec:
        relevant_entries = STORE.hybrid_search(
            query=question_clean,
            vector=query_vec,
            limit=25,
        )
    else:
        # Fall back to keyword search across the journal
        relevant_entries = STORE.search(query=question_clean, limit=25)

    if not relevant_entries:
        # If search returned nothing, fall back to the most recent entries
        relevant_entries = STORE.list(limit=25)

    return ask_journal(question_clean, relevant_entries)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=False)
