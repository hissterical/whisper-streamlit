from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from journal_app.database import JournalStore
from journal_app.services import analyze_transcript, ask_journal, embed_text, transcribe

LOGGER = logging.getLogger(__name__)
ROOT = Path(__file__).parent
STATIC = ROOT / "frontend"
STORE = JournalStore(os.getenv("ECHONOTE_DB", "data/echonote.sqlite3"))
UPLOAD_DIR = ROOT / "data" / "audio_cache"
ALLOWED_EXTENSIONS = {".wav", ".mp3", ".m4a", ".webm", ".ogg", ".flac", ".mp4"}
MAX_UPLOAD_BYTES = 100 * 1024 * 1024

app = FastAPI(title="EchoNote", version="1.0")
app.mount("/static", StaticFiles(directory=STATIC), name="static")


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000)


@app.get("/", include_in_schema=False)
def home() -> FileResponse:
    return FileResponse(STATIC / "index.html")


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok", "whisper": "lazy"}


@app.get("/api/entries")
def entries(
    search: str = Query("", max_length=200),
    topic: str = Query("", max_length=80),
    mood: str = Query("", max_length=80),
    has_actions: bool = False,
    semantic: str = Query("", max_length=500),
) -> dict:
    vector = None
    if semantic.strip():
        vector = embed_text(semantic)
        results = STORE.semantic_search(vector) if vector else STORE.search(semantic, topic=topic, mood=mood, has_actions=has_actions)
    else:
        results = STORE.search(search, topic=topic, mood=mood, has_actions=has_actions)
    return {"entries": results, "facets": STORE.facets(), "search_mode": "semantic" if semantic and vector else "keyword"}


async def process_audio(audio: UploadFile, keep_audio: bool = False) -> dict:
    suffix = Path(audio.filename or ".wav").suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(415, "Unsupported audio type.")
    content = await audio.read(MAX_UPLOAD_BYTES + 1)
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "Audio file is too large (100 MB maximum).")
    if not content:
        raise HTTPException(400, "The audio file is empty.")
    digest = hashlib.sha256(content).hexdigest()
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    path = UPLOAD_DIR / f"{digest}{suffix}"
    if not path.exists():
        path.write_bytes(content)
    try:
        transcript = transcribe(path)
        if not transcript:
            raise HTTPException(422, "No speech was detected.")
        return analyze_transcript(transcript) | {"transcript": transcript}
    except HTTPException:
        raise
    except (ImportError, OSError, RuntimeError, ValueError) as error:
        LOGGER.exception("Audio analysis failed")
        raise HTTPException(500, f"Could not analyze audio: {error}") from error
    finally:
        if not keep_audio and path.exists():
            path.unlink()


@app.post("/api/analyze")
async def analyze(
    audio: Annotated[UploadFile, File(...)],
) -> dict:
    """Analyze audio without persisting it in the journal."""
    result = await process_audio(audio)
    return result


@app.post("/api/journal/entries")
async def create_journal_entry(
    audio: Annotated[UploadFile, File(...)],
    source: str = Query("upload", pattern="^(upload|microphone)$"),
) -> dict:
    """Analyze audio and persist the resulting thought in the journal."""
    result = await process_audio(audio, keep_audio=True)
    result["source"] = source
    result["embedding"] = embed_text(f"{result['title']}. {result['summary']}. {' '.join(result['topics'])}")
    result["id"] = STORE.add(result)
    return result


@app.get("/api/graph")
def graph() -> dict:
    return STORE.graph()


@app.get("/api/calendar")
def calendar(year: int | None = None, month: int | None = None) -> dict[str, list[dict]]:
    return {"entries": STORE.calendar(year, month)}


@app.post("/api/ask")
def ask(request: AskRequest) -> dict[str, str]:
    return {"answer": ask_journal(request.question.strip(), STORE.list())}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=False)
