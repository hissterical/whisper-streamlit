from __future__ import annotations

import json
import logging
import math
import re
from pathlib import Path
from typing import Any, Callable

from journal_app.models import AnalysisResult, AudioValidationResult

LOGGER = logging.getLogger(__name__)

MAGIC_SIGNATURES: list[tuple[bytes, str, int]] = [
    (b"RIFF", "wav", 0),  # Followed by WAVE at offset 8
    (b"ID3", "mp3", 0),
    (b"OggS", "ogg", 0),
    (b"fLaC", "flac", 0),
    (b"\x1a\x45\xdf\xa3", "webm", 0),
]

ALLOWED_EXTENSIONS = {".wav", ".mp3", ".m4a", ".webm", ".ogg", ".flac", ".mp4"}
MIN_AUDIO_BYTES = 100
MAX_AUDIO_BYTES = 100 * 1024 * 1024  # 100 MB


def validate_audio_bytes(content: bytes, filename: str | None = None) -> AudioValidationResult:
    byte_count = len(content)
    if byte_count == 0:
        return AudioValidationResult(is_valid=False, byte_count=0, error="Audio payload is completely empty.")

    suffix = Path(filename).suffix.lower() if filename else ""
    if suffix and suffix not in ALLOWED_EXTENSIONS:
        return AudioValidationResult(
            is_valid=False,
            byte_count=byte_count,
            error=f"Unsupported audio file extension '{suffix}'. Allowed: {sorted(ALLOWED_EXTENSIONS)}",
        )

    if byte_count < MIN_AUDIO_BYTES:
        return AudioValidationResult(
            is_valid=False,
            byte_count=byte_count,
            error=f"Audio payload too small ({byte_count} bytes, minimum {MIN_AUDIO_BYTES} bytes).",
        )
    if byte_count > MAX_AUDIO_BYTES:
        return AudioValidationResult(
            is_valid=False,
            byte_count=byte_count,
            error=f"Audio payload exceeds maximum size ({byte_count} > {MAX_AUDIO_BYTES} bytes).",
        )

    # Magic byte inspection
    detected_format: str | None = None
    if content.startswith(b"RIFF") and len(content) >= 12 and content[8:12] == b"WAVE":
        detected_format = "wav"
    elif content.startswith(b"ID3") or (len(content) >= 2 and content[0] == 0xFF and (content[1] & 0xE0) == 0xE0):
        detected_format = "mp3"
    elif content.startswith(b"OggS"):
        detected_format = "ogg"
    elif content.startswith(b"fLaC"):
        detected_format = "flac"
    elif content.startswith(b"\x1a\x45\xdf\xa3"):
        detected_format = "webm"
    elif len(content) >= 8 and content[4:8] == b"ftyp":
        detected_format = "m4a"

    if not detected_format and suffix:
        # Fallback to extension check if magic bytes are ambiguous (e.g. raw AAC or unusual headers)
        detected_format = suffix.lstrip(".")

    if not detected_format:
        return AudioValidationResult(
            is_valid=False,
            byte_count=byte_count,
            error="Audio header format could not be verified.",
        )

    return AudioValidationResult(is_valid=True, format=detected_format, byte_count=byte_count)


def repair_json_string(text: str) -> str:
    """Attempt basic repairs on malformed LLM JSON strings."""
    # Remove markdown code block fences
    fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    if fence_match:
        text = fence_match.group(1)

    # Extract JSON between outermost curly braces
    brace_match = re.search(r"\{[\s\S]*\}", text)
    if brace_match:
        text = brace_match.group(0)

    # Replace smart/curly quotes with standard double quotes
    text = text.replace("“", '"').replace("”", '"').replace("’", "'").replace("‘", "'")

    # Remove trailing commas before closing braces/brackets
    text = re.sub(r",\s*([\]}])", r"\1", text)
    return text


def parse_and_validate_llm_json(
    raw_response: str,
    fallback_generator: Callable[[str], dict[str, Any]],
    transcript: str,
) -> dict[str, Any]:
    """Parse Ollama response, validate with Pydantic, and repair/fallback on any defect."""
    fallback_data = fallback_generator(transcript)
    if not raw_response or not raw_response.strip():
        LOGGER.warning("Empty LLM response received; using deterministic fallback")
        return fallback_data

    cleaned_json = repair_json_string(raw_response)
    parsed: dict[str, Any] | None = None

    try:
        loaded = json.loads(cleaned_json)
        if isinstance(loaded, dict):
            parsed = loaded
        else:
            LOGGER.warning("Parsed JSON was not a dictionary (type=%s)", type(loaded).__name__)
    except (json.JSONDecodeError, ValueError) as err:
        LOGGER.warning("Failed to decode JSON from LLM: %s. Raw: %s", err, raw_response[:200])

    if parsed is None:
        return fallback_data

    # Merge missing fields with fallback values prior to Pydantic validation
    for key, value in fallback_data.items():
        if key not in parsed or parsed[key] is None or parsed[key] == "":
            parsed[key] = value

    try:
        validated = AnalysisResult.model_validate(parsed)
        return validated.model_dump()
    except Exception as validation_err:
        LOGGER.warning("Pydantic validation failed on LLM output: %s; falling back", validation_err)
        return fallback_data


def validate_vector(vector: Any, expected_dim: int | None = None) -> list[float] | None:
    """Validate vector array, ensuring finite numerical values, non-zero norm, and dimension match."""
    if not isinstance(vector, list) or len(vector) == 0:
        return None

    if expected_dim is not None and len(vector) != expected_dim:
        LOGGER.warning("Vector dimension mismatch: expected %d, got %d", expected_dim, len(vector))
        return None

    cleaned: list[float] = []
    norm_sq = 0.0
    for idx, item in enumerate(vector):
        if not isinstance(item, (int, float)):
            LOGGER.warning("Non-numeric vector element at index %d: %s", idx, item)
            return None
        val = float(item)
        if not math.isfinite(val):
            LOGGER.warning("Non-finite vector element at index %d: %s", idx, val)
            return None
        cleaned.append(val)
        norm_sq += val * val

    if norm_sq <= 1e-12:
        LOGGER.warning("Vector norm is zero or negligible: norm_sq=%s", norm_sq)
        return None

    return cleaned
