from __future__ import annotations

import json
import logging
import re
from functools import lru_cache
from threading import Lock
from pathlib import Path
from typing import Any

from httpx import HTTPError
from utils.langchain_llms import LangchainLLama

LOGGER = logging.getLogger(__name__)
_WHISPER_LOCK = Lock()


def _fallback_analysis(transcript: str) -> dict[str, Any]:
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", transcript) if s.strip()]
    summary = " ".join(sentences[:3]) or transcript
    words = re.findall(r"[A-Za-z][A-Za-z'-]{3,}", transcript.lower())
    stopwords = {"this", "that", "with", "have", "from", "about", "there", "would", "could", "should"}
    topics: list[str] = []
    for word in words:
        if word not in stopwords and word not in topics:
            topics.append(word)
        if len(topics) == 4:
            break
    actions = [
        sentence for sentence in sentences
        if re.search(r"\b(todo|need to|should|will|follow up|action)\b", sentence, re.I)
    ]
    return {
        "title": (sentences[0][:58] if sentences else "Untitled thought").rstrip("."),
        "summary": summary,
        "topics": topics or ["reflection"],
        "mood": "reflective",
        "action_items": actions[:5],
    }


@lru_cache(maxsize=1)
def _load_whisper_model(model_name: str):
    import whisper
    LOGGER.info("Loading Whisper model %s on first analysis", model_name)
    return whisper.load_model(model_name)


def transcribe(audio_path: str | Path, model_name: str = "base.en") -> str:
    # The lock avoids duplicate model loads when two first requests arrive together.
    with _WHISPER_LOCK:
        model = _load_whisper_model(model_name)
    return str(model.transcribe(str(audio_path))["text"]).strip()


def embed_text(text: str, model_name: str = "nomic-embed-text") -> list[float] | None:
    """Create a local embedding through Ollama when an embedding model is available."""
    import httpx

    try:
        response = httpx.post(
            "http://127.0.0.1:11434/api/embeddings",
            json={"model": model_name, "prompt": text},
            timeout=30,
        )
        response.raise_for_status()
        embedding = response.json().get("embedding")
        if not isinstance(embedding, list) or not all(isinstance(value, (int, float)) for value in embedding):
            raise ValueError("Ollama returned an invalid embedding")
        return [float(value) for value in embedding]
    except (httpx.HTTPError, OSError, TypeError, ValueError) as error:
        LOGGER.info("Semantic embedding unavailable; keeping keyword search: %s", error)
        return None


def analyze_transcript(transcript: str, use_ollama: bool = True) -> dict[str, Any]:
    if not use_ollama:
        LOGGER.info("Using deterministic journal fallback because Ollama is disabled")
        return _fallback_analysis(transcript)
    prompt = """Analyze this personal journal transcript. Return ONLY valid JSON with exactly:
{"title":"short title","summary":"2-4 sentence summary","topics":["topic"],"mood":"one word",
"action_items":["concrete next step"]}. Do not invent details.
Transcript:
""" + transcript
    try:
        raw = str(LangchainLLama(model="llama3.2").get_response(prompt))
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not match:
            raise ValueError("Ollama returned no JSON")
        result = json.loads(match.group())
        if not isinstance(result, dict):
            raise ValueError("Ollama returned a JSON value instead of an object")
        fallback = _fallback_analysis(transcript)
        for key in fallback:
            if not result.get(key):
                result[key] = fallback[key]
        return result
    except (ConnectionError, HTTPError, OSError, TimeoutError, TypeError, ValueError, json.JSONDecodeError) as error:
        LOGGER.warning("Using deterministic journal fallback: %s", error)
        return _fallback_analysis(transcript)


def ask_journal(question: str, entries: list[dict[str, Any]]) -> str:
    context = "\n\n".join(f"{e['title']}: {e['summary']}" for e in entries[:20])
    if not context:
        return "Your journal is empty. Record a thought first."
    prompt = f"""Answer the question using only these journal notes. If the notes do not answer it, say so.
Question: {question}
Notes:
{context}"""
    try:
        return str(LangchainLLama(model="llama3.2").get_response(prompt)).strip()
    except (ConnectionError, HTTPError, OSError, TimeoutError) as error:
        LOGGER.warning("Ask journal unavailable; Ollama request failed: %s", error)
        return "Ask your journal needs Ollama running locally (`ollama serve` and `ollama pull llama3.2`)."
