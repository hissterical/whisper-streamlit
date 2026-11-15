from __future__ import annotations

import json
import logging
import re
from functools import lru_cache
from pathlib import Path
from threading import Lock
from typing import Any

from journal_app.validation import parse_and_validate_llm_json, validate_vector
from utils.langchain_llms import LangchainLLama

LOGGER = logging.getLogger(__name__)
_WHISPER_LOCK = Lock()


def _fallback_analysis(transcript: str) -> dict[str, Any]:
    """Deterministic, lightweight extraction when Ollama is unavailable or disabled."""
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", transcript) if s.strip()]
    summary = " ".join(sentences[:3]) or transcript
    words = re.findall(r"[A-Za-z][A-Za-z'-]{3,}", transcript.lower())
    stopwords = {
        "this", "that", "with", "have", "from", "about", "there", "would",
        "could", "should", "their", "which", "these", "those", "being",
        "where", "going", "doing", "think", "thought",
    }
    topics: list[str] = []
    for word in words:
        if word not in stopwords and word not in topics:
            topics.append(word)
        if len(topics) == 4:
            break
    actions = [
        sentence for sentence in sentences
        if re.search(r"\b(todo|need to|should|will|follow up|action|plan to|must)\b", sentence, re.I)
    ]
    return {
        "title": (sentences[0][:58] if sentences else "Untitled thought").rstrip("."),
        "summary": summary,
        "topics": topics or ["reflection"],
        "mood": "reflective",
        "action_items": actions[:5],
    }


@lru_cache(maxsize=2)
def _load_whisper_model(model_name: str):
    import whisper

    LOGGER.info("Loading Whisper model '%s'", model_name)
    return whisper.load_model(model_name)


def transcribe(audio_path: str | Path, model_name: str = "base.en") -> str:
    """Transcribe an audio file with Whisper under process-wide lock."""
    path = Path(audio_path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    if path.stat().st_size == 0:
        raise ValueError("Audio file is 0 bytes")

    with _WHISPER_LOCK:
        model = _load_whisper_model(model_name)
    result = model.transcribe(str(path))
    return str(result.get("text", "")).strip()


def embed_text(text: str, model_name: str = "nomic-embed-text") -> dict[str, Any] | None:
    """Create a local embedding through Ollama. Returns dict with vector, model, and dim, or None."""
    import httpx

    text_clean = text.strip()
    if not text_clean:
        return None

    try:
        response = httpx.post(
            "http://127.0.0.1:11434/api/embeddings",
            json={"model": model_name, "prompt": text_clean},
            timeout=httpx.Timeout(60.0, connect=1.5),
        )
        response.raise_for_status()
        raw_vec = response.json().get("embedding")
        clean_vec = validate_vector(raw_vec)
        if not clean_vec:
            LOGGER.warning("Ollama returned an invalid or unparseable embedding vector")
            return None
        return {
            "vector": clean_vec,
            "model": model_name,
            "dim": len(clean_vec),
        }
    except Exception as error:
        LOGGER.info("Semantic embedding unavailable (%s); keeping keyword search fallback", error)
        return None


def get_embedding_vector(text: str, model_name: str = "nomic-embed-text") -> list[float] | None:
    """Convenience helper that extracts just the validated float vector or None."""
    res = embed_text(text, model_name=model_name)
    return res["vector"] if res else None


def analyze_transcript(
    transcript: str,
    use_ollama: bool = True,
    model_name: str = "llama3.2",
) -> dict[str, Any]:
    """Analyze a transcript to extract title, summary, topics, mood, and action items."""
    transcript_clean = transcript.strip()
    if not transcript_clean:
        return _fallback_analysis("")

    if not use_ollama:
        LOGGER.info("Using deterministic journal fallback because Ollama is disabled")
        return _fallback_analysis(transcript_clean)

    prompt = (
        "You are an expert personal journal analyzer. Analyze this spoken journal transcript.\n"
        "Return ONLY a valid JSON object with the exact keys: 'title', 'summary', 'topics', 'mood', 'action_items'.\n"
        "- title: concise title (under 10 words, no trailing period)\n"
        "- summary: clear 2-4 sentence summary of what was discussed\n"
        "- topics: array of 1-5 lowercase single-word or short-phrase topic tags\n"
        "- mood: single lowercase word representing emotional tone (e.g. reflective, focused, energized, anxious, calm)\n"
        "- action_items: array of concrete next steps or commitments mentioned (empty array if none)\n"
        "Do not invent facts not present in the transcript.\n\n"
        f"Transcript:\n{transcript_clean}\n"
    )

    try:
        raw_response = str(LangchainLLama(model=model_name).get_response(prompt))
        return parse_and_validate_llm_json(raw_response, _fallback_analysis, transcript_clean)
    except Exception as error:
        LOGGER.warning("Ollama LLM analysis failed (%s); reverting to deterministic fallback", error)
        return _fallback_analysis(transcript_clean)


def ask_journal(
    question: str,
    entries: list[dict[str, Any]],
    model_name: str = "llama3.2",
    use_ollama: bool = True,
) -> dict[str, Any]:
    """Retrieval-grounded Q&A over journal entries with explicit evidence citations and abstention."""
    q_clean = question.strip()
    if not q_clean:
        return {
            "answer": "Please ask a question about your journal entries.",
            "abstention": True,
            "evidence": [],
        }

    if not entries:
        return {
            "answer": "Your journal is empty. Record or upload a thought first.",
            "abstention": True,
            "evidence": [],
        }

    # Format evidence blocks with explicit entry ID and created_at timestamps
    context_blocks: list[str] = []
    evidence_list: list[dict[str, Any]] = []

    # Sort or prioritize entries: take up to 25 relevant candidates
    candidates = entries[:25]
    for e in candidates:
        eid = e.get("id", 0)
        c_at = e.get("created_at", "unknown date")
        title = e.get("title", "Untitled")
        summary = e.get("summary", "")
        actions = ", ".join(e.get("action_items", []))
        block = f"[Entry #{eid} | Date: {c_at[:10]}]\nTitle: {title}\nSummary: {summary}"
        if actions:
            block += f"\nAction Items: {actions}"
        context_blocks.append(block)
        evidence_list.append({
            "entry_id": eid,
            "created_at": c_at,
            "title": title,
            "summary": summary,
            "relevance_score": e.get("relevance_score") or e.get("rrf_score"),
        })

    context_str = "\n\n".join(context_blocks)

    if not use_ollama:
        # Deterministic keyword-matching response when Ollama is unavailable
        matches = [e for e in candidates if any(w.lower() in (e["title"] + " " + e["summary"]).lower() for w in q_clean.split())]
        if not matches:
            return {
                "answer": f"I could not find information in your journal entries to answer '{q_clean}'.",
                "abstention": True,
                "evidence": evidence_list[:5],
            }
        top_match = matches[0]
        return {
            "answer": f"Based on [Entry #{top_match['id']}] ({top_match['title']}): {top_match['summary']}",
            "abstention": False,
            "evidence": [e for e in evidence_list if e["entry_id"] in {m["id"] for m in matches[:3]}],
        }

    prompt = (
        "You are an assistant answering questions strictly based on a user's personal voice journal.\n"
        "Guidelines:\n"
        "1. Answer using ONLY information found in the Notes below.\n"
        "2. Ground each claim by citing the entry identifier (e.g. [Entry #3]).\n"
        "3. If the journal notes do not contain enough evidence to answer the question, explicitly state: "
        "'I do not have enough information in your journal to answer this question.' Do not guess or extrapolate.\n\n"
        f"Journal Notes:\n{context_str}\n\n"
        f"Question: {q_clean}\n\n"
        "Answer:"
    )

    try:
        response_text = str(LangchainLLama(model=model_name).get_response(prompt)).strip()
        abstain_phrases = [
            "do not have enough information",
            "not mentioned in the notes",
            "cannot answer",
            "no information",
            "notes do not contain",
            "not mentioned in the journal",
        ]
        is_abstention = any(p in response_text.lower() for p in abstain_phrases)
        return {
            "answer": response_text,
            "abstention": is_abstention,
            "evidence": evidence_list[:5] if not is_abstention else [],
        }
    except Exception as error:
        LOGGER.warning("Ollama Q&A failed (%s); providing search-based fallback", error)
        # Search-based fallback
        q_words = [w.lower() for w in q_clean.split() if len(w) > 3]
        matches = [e for e in candidates if any(w in (e["title"] + " " + e["summary"]).lower() for w in q_words)]
        if not matches:
            return {
                "answer": f"Ollama is currently unreachable, and no direct keyword matches for '{q_clean}' were found in your notes.",
                "abstention": True,
                "evidence": [],
            }
        top = matches[0]
        return {
            "answer": f"Based on [Entry #{top['id']}] ({top['title']}): {top['summary']}",
            "abstention": False,
            "evidence": [e for e in evidence_list if e["entry_id"] == top["id"]],
        }
