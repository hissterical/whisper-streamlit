from __future__ import annotations

from typing import Any
from pydantic import BaseModel, Field, field_validator


class AnalysisResult(BaseModel):
    title: str = Field(..., min_length=1, max_length=200, description="Short title of the thought")
    summary: str = Field(..., min_length=1, description="Concise summary of the transcript")
    topics: list[str] = Field(default_factory=list, description="Categorical topics or tags")
    mood: str = Field(default="reflective", min_length=1, max_length=50, description="Dominant emotional tone")
    action_items: list[str] = Field(default_factory=list, description="Concrete next steps or action items")

    @field_validator("title", mode="before")
    @classmethod
    def sanitize_title(cls, v: Any) -> str:
        if not isinstance(v, str):
            v = str(v or "Untitled thought")
        clean = v.strip().rstrip(".")
        return clean[:200] if clean else "Untitled thought"

    @field_validator("summary", mode="before")
    @classmethod
    def sanitize_summary(cls, v: Any) -> str:
        if not isinstance(v, str):
            v = str(v or "")
        clean = v.strip()
        return clean if clean else "No summary available."

    @field_validator("topics", mode="before")
    @classmethod
    def sanitize_topics(cls, v: Any) -> list[str]:
        if not isinstance(v, list):
            if isinstance(v, str):
                v = [v]
            else:
                return ["reflection"]
        cleaned: list[str] = []
        for item in v:
            if item is None:
                continue
            text = str(item).strip().lower()
            if text and text not in cleaned and len(text) <= 50:
                cleaned.append(text)
        return cleaned[:10] if cleaned else ["reflection"]

    @field_validator("mood", mode="before")
    @classmethod
    def sanitize_mood(cls, v: Any) -> str:
        if not isinstance(v, str):
            v = str(v or "reflective")
        clean = v.strip().lower()
        return clean[:50] if clean else "reflective"

    @field_validator("action_items", mode="before")
    @classmethod
    def sanitize_action_items(cls, v: Any) -> list[str]:
        if not isinstance(v, list):
            if isinstance(v, str):
                v = [v]
            else:
                return []
        cleaned: list[str] = []
        for item in v:
            if item is None:
                continue
            text = str(item).strip()
            if text and text not in cleaned and len(text) <= 300:
                cleaned.append(text)
        return cleaned[:10]


class AnalyzeResponse(AnalysisResult):
    transcript: str = Field(..., description="Full transcribed speech text")


class EmbeddingData(BaseModel):
    vector: list[float] = Field(..., description="Normalized numerical embedding vector")
    model: str = Field(..., description="Embedding model identifier")
    dim: int = Field(..., description="Embedding vector dimensionality")


class JournalEntryCreate(AnalysisResult):
    transcript: str = Field(..., description="Full transcribed speech text")
    source: str = Field(default="upload", description="Input source: upload or microphone")
    audio_path: str | None = Field(default=None, description="Relative path to stored audio file")
    embedding: list[float] | None = Field(default=None, description="Embedding vector")
    embedding_model: str | None = Field(default=None, description="Embedding model name")
    embedding_dim: int | None = Field(default=None, description="Embedding dimension")
    created_at: str | None = Field(default=None, description="ISO-8601 creation timestamp")


class JournalEntryResponse(BaseModel):
    id: int
    created_at: str
    title: str
    summary: str
    transcript: str
    topics: list[str]
    mood: str
    action_items: list[str]
    source: str
    audio_path: str | None = None
    embedding_model: str | None = None
    embedding_dim: int | None = None
    has_embedding: bool = False


class EntriesListResponse(BaseModel):
    entries: list[JournalEntryResponse]
    facets: dict[str, list[str]]
    search_mode: str = Field(description="Search mode: keyword, semantic, or hybrid")
    total: int


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000, description="User question about journal entries")


class EvidenceCitation(BaseModel):
    entry_id: int
    created_at: str
    title: str
    summary: str
    relevance_score: float | None = None


class AskResponse(BaseModel):
    answer: str = Field(..., description="Grounded answer to question or abstention statement")
    abstention: bool = Field(default=False, description="Whether the system abstained due to lack of evidence")
    evidence: list[EvidenceCitation] = Field(default_factory=list, description="Retrieved entry citations")


class AudioValidationResult(BaseModel):
    is_valid: bool
    format: str | None = None
    byte_count: int
    error: str | None = None
