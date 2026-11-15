import io
import json
import math
import shutil
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi import HTTPException
from fastapi.datastructures import UploadFile

from journal_app.database import JournalStore
from journal_app.models import AnalysisResult
from journal_app.services import _fallback_analysis, ask_journal
from journal_app.validation import (
    parse_and_validate_llm_json,
    repair_json_string,
    validate_audio_bytes,
    validate_vector,
)


class TestAudioValidation(unittest.TestCase):
    def test_valid_wav_header(self) -> None:
        # RIFF header + size (4 bytes) + WAVE + fmt  chunk
        wav_header = b"RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x44\xac\x00\x00" + b"\x00" * 80
        res = validate_audio_bytes(wav_header, "speech.wav")
        self.assertTrue(res.is_valid)
        self.assertEqual(res.format, "wav")

    def test_valid_mp3_header(self) -> None:
        # ID3 header
        mp3_header = b"ID3\x04\x00\x00\x00\x00\x00\x00" + b"\x00" * 100
        res = validate_audio_bytes(mp3_header, "speech.mp3")
        self.assertTrue(res.is_valid)
        self.assertEqual(res.format, "mp3")

    def test_valid_webm_header(self) -> None:
        # EBML signature \x1a\x45\xdf\xa3
        webm_header = b"\x1a\x45\xdf\xa3" + b"\x00" * 100
        res = validate_audio_bytes(webm_header, "recording.webm")
        self.assertTrue(res.is_valid)
        self.assertEqual(res.format, "webm")

    def test_empty_audio_rejected(self) -> None:
        res = validate_audio_bytes(b"", "empty.wav")
        self.assertFalse(res.is_valid)
        self.assertIn("empty", res.error.lower())

    def test_too_small_audio_rejected(self) -> None:
        res = validate_audio_bytes(b"RIFFWAVE", "tiny.wav")
        self.assertFalse(res.is_valid)
        self.assertIn("too small", res.error.lower())

    def test_unsupported_extension_rejected(self) -> None:
        res = validate_audio_bytes(b"RIFF\x24\x00\x00\x00WAVE" + b"\x00" * 100, "malicious.exe")
        self.assertFalse(res.is_valid)
        self.assertIn("unsupported", res.error.lower())

    def test_unrecognized_binary_rejected(self) -> None:
        random_bytes = b"\x00\x11\x22\x33" * 30
        res = validate_audio_bytes(random_bytes, None)
        self.assertFalse(res.is_valid)


class TestSchemaValidationAndRepair(unittest.TestCase):
    def test_repair_markdown_code_fences(self) -> None:
        text = '```json\n{"title": "Morning Walk", "summary": "Walked in park.", "topics": ["health"], "mood": "calm", "action_items": []}\n```'
        repaired = repair_json_string(text)
        data = json.loads(repaired)
        self.assertEqual(data["title"], "Morning Walk")

    def test_repair_trailing_commas_and_curly_quotes(self) -> None:
        text = '{\n  “title”: “Deep Work”, \n  “summary”: “Worked on code.”, \n  “topics”: [“focus”, ], \n  “mood”: “productive”, \n  “action_items”: [“Write docs”, ], \n}'
        data = parse_and_validate_llm_json(text, _fallback_analysis, "Worked on code.")
        self.assertEqual(data["title"], "Deep Work")
        self.assertEqual(data["topics"], ["focus"])
        self.assertEqual(data["action_items"], ["Write docs"])

    def test_fallback_on_unparseable_json(self) -> None:
        raw_garbage = "I am an AI and I could not parse this note. Here is random text."
        result = parse_and_validate_llm_json(raw_garbage, _fallback_analysis, "I need to call Alice tomorrow.")
        self.assertTrue(result["title"])
        self.assertIn("Alice", result["summary"])
        self.assertIn("alice", result["action_items"][0].lower())

    def test_pydantic_model_sanitizes_bad_fields(self) -> None:
        model = AnalysisResult(
            title="A very long title that should be valid",
            summary="A short summary.",
            topics=["WORK", "work", None, 123],  # Deduplicate and normalize
            mood="ENERGIZED",
            action_items=["Buy milk", None, 456],
        )
        self.assertEqual(model.mood, "energized")
        self.assertIn("work", model.topics)
        self.assertIn("123", model.topics)
        self.assertEqual(len(model.topics), 2)  # WORK and work deduplicated to 'work'


class TestVectorValidationAndCompatibility(unittest.TestCase):
    def test_validate_vector_success(self) -> None:
        vec = [0.1, 0.2, 0.3, 0.4]
        res = validate_vector(vec, expected_dim=4)
        self.assertEqual(res, vec)

    def test_validate_vector_dimension_mismatch(self) -> None:
        vec = [0.1, 0.2, 0.3]
        res = validate_vector(vec, expected_dim=4)
        self.assertIsNone(res)

    def test_validate_vector_rejects_nan_and_inf(self) -> None:
        self.assertIsNone(validate_vector([0.1, float("nan"), 0.3]))
        self.assertIsNone(validate_vector([0.1, float("inf"), 0.3]))
        self.assertIsNone(validate_vector([0.0, 0.0, 0.0]))  # zero norm


class TestDatabaseMigrationAndVectorSafety(unittest.TestCase):
    def setUp(self) -> None:
        self.test_dir = tempfile.mkdtemp()
        self.db_path = Path(self.test_dir) / "test_migration.sqlite3"

    def tearDown(self) -> None:
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_backward_compatibility_migration(self) -> None:
        # 1. Create a legacy database matching the original schema (no embedding_model, embedding_dim, audio_path)
        conn = sqlite3.connect(self.db_path)
        conn.execute("""CREATE TABLE entries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL, title TEXT NOT NULL,
            summary TEXT NOT NULL, transcript TEXT NOT NULL,
            topics TEXT NOT NULL, mood TEXT NOT NULL,
            action_items TEXT NOT NULL, source TEXT NOT NULL,
            embedding TEXT
        )""")
        conn.execute("""INSERT INTO entries VALUES (
            1, '2026-11-15T12:00:00', 'Old Entry', 'Summary of old entry',
            'Transcript of old entry', '["legacy"]', 'nostalgic', '[]', 'upload', NULL
        )""")
        conn.commit()
        conn.close()

        # 2. Instantiate JournalStore on this legacy database: auto-migration must succeed
        store = JournalStore(self.db_path)
        entries = store.list()
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["title"], "Old Entry")
        self.assertIsNone(entries[0]["embedding_model"])
        self.assertIsNone(entries[0]["embedding_dim"])
        self.assertFalse(entries[0]["has_embedding"])

        # 3. Verify that new columns can be written and queried
        new_id = store.add({
            "title": "New Entry with Vector",
            "summary": "Summary with vector",
            "transcript": "Transcript with vector",
            "topics": ["ai"],
            "mood": "focused",
            "action_items": ["Run benchmark"],
            "embedding": [0.6, 0.8],
            "embedding_model": "test-embed-2d",
        })
        new_entry = store.get_entry(new_id)
        self.assertEqual(new_entry["embedding_dim"], 2)
        self.assertEqual(new_entry["embedding_model"], "test-embed-2d")
        self.assertTrue(new_entry["has_embedding"])

    def test_prevent_cross_dimensional_vector_comparisons(self) -> None:
        store = JournalStore(self.db_path)

        # Insert 3-dimensional entry
        id_3d = store.add({
            "title": "3D Note",
            "summary": "Three dimensional note",
            "transcript": "3d",
            "topics": ["geometry"],
            "embedding": [1.0, 0.0, 0.0],
            "embedding_model": "model-3d",
        })

        # Insert 2-dimensional entry
        id_2d = store.add({
            "title": "2D Note",
            "summary": "Two dimensional note",
            "transcript": "2d",
            "topics": ["plane"],
            "embedding": [1.0, 0.0],
            "embedding_model": "model-2d",
        })

        # Search with a 3-dimensional vector [1.0, 0.0, 0.0]
        results_3d = store.semantic_search([1.0, 0.0, 0.0])
        # Only the 3D note should match! The 2D note must NEVER match or corrupt similarity!
        self.assertEqual(len(results_3d), 1)
        self.assertEqual(results_3d[0]["id"], id_3d)
        self.assertAlmostEqual(results_3d[0]["relevance_score"], 1.0, places=3)

        # Search with a 2-dimensional vector [1.0, 0.0]
        results_2d = store.semantic_search([1.0, 0.0])
        self.assertEqual(len(results_2d), 1)
        self.assertEqual(results_2d[0]["id"], id_2d)

    def test_keyword_search_works_without_embeddings(self) -> None:
        store = JournalStore(self.db_path)
        store.add({
            "title": "Trip to Paris",
            "summary": "Visited the Louvre and walked along the Seine.",
            "transcript": "Paris was beautiful.",
            "topics": ["travel", "france"],
            "mood": "delighted",
            "action_items": ["Share photos"],
        })
        store.add({
            "title": "Budget Meeting",
            "summary": "Quarterly financial review and revenue projections.",
            "transcript": "Numbers look solid.",
            "topics": ["finance", "work"],
            "mood": "serious",
            "action_items": ["Update spreadsheet"],
        })

        res_paris = store.search("Paris")
        self.assertEqual(len(res_paris), 1)
        self.assertEqual(res_paris[0]["title"], "Trip to Paris")

        res_action = store.search("", has_actions=True)
        self.assertEqual(len(res_action), 2)

        res_topic = store.search("", topic="finance")
        self.assertEqual(len(res_topic), 1)
        self.assertEqual(res_topic[0]["title"], "Budget Meeting")

    def test_hybrid_search_rrf_scoring(self) -> None:
        store = JournalStore(self.db_path)
        # Entry 1: matches keyword and vector
        id1 = store.add({
            "title": "Machine Learning Research",
            "summary": "Studying neural networks and transformers.",
            "transcript": "Deep learning research.",
            "topics": ["ai"],
            "embedding": [1.0, 0.0],
            "embedding_model": "model-2d",
        })
        # Entry 2: matches keyword only
        id2 = store.add({
            "title": "Machine Maintenance",
            "summary": "Repairing espresso machine.",
            "transcript": "Fixed the espresso machine.",
            "topics": ["coffee"],
            "embedding": [0.0, 1.0],
            "embedding_model": "model-2d",
        })

        # Hybrid search with query "Machine" and vector [1.0, 0.0]
        hybrid_results = store.hybrid_search(
            query="Machine",
            vector=[1.0, 0.0],
            limit=5,
        )
        self.assertGreaterEqual(len(hybrid_results), 2)
        # Entry 1 should be ranked first because it ranks high in both keyword and vector search!
        self.assertEqual(hybrid_results[0]["id"], id1)
        self.assertIn("rrf_score", hybrid_results[0])
        self.assertGreater(hybrid_results[0]["rrf_score"], hybrid_results[1]["rrf_score"])

    def test_entry_deletion_cleans_up_audio_file(self) -> None:
        store = JournalStore(self.db_path)
        audio_file = Path(self.test_dir) / "sample_recording.wav"
        audio_file.write_bytes(b"RIFF\x24\x00\x00\x00WAVE" + b"\x00" * 100)
        self.assertTrue(audio_file.exists())

        eid = store.add({
            "title": "Note with Audio",
            "summary": "Has audio file on disk.",
            "transcript": "Audio note.",
            "audio_path": str(audio_file),
        })
        self.assertTrue(audio_file.exists())

        deleted = store.delete(eid)
        self.assertTrue(deleted)
        self.assertFalse(audio_file.exists())  # Audio file must be removed!


class TestGroundedJournalQA(unittest.TestCase):
    def test_ask_journal_abstains_when_empty(self) -> None:
        res = ask_journal("Where did I leave my keys?", [])
        self.assertTrue(res["abstention"])
        self.assertEqual(len(res["evidence"]), 0)

    def test_ask_journal_fallback_with_evidence(self) -> None:
        entries = [
            {
                "id": 1,
                "created_at": "2026-11-10T10:00:00",
                "title": "Doctor Appointment",
                "summary": "Dr. Smith prescribed vitamins on Tuesday.",
                "topics": ["health"],
                "action_items": ["Take vitamins"],
            },
            {
                "id": 2,
                "created_at": "2026-11-12T15:00:00",
                "title": "Grocery Shopping",
                "summary": "Bought apples and oat milk.",
                "topics": ["groceries"],
                "action_items": [],
            },
        ]
        res = ask_journal("What did the doctor prescribe?", entries, use_ollama=False)
        self.assertFalse(res["abstention"])
        self.assertIn("[Entry #1]", res["answer"])
        self.assertIn("vitamins", res["answer"].lower())
        self.assertEqual(len(res["evidence"]), 1)
        self.assertEqual(res["evidence"][0]["entry_id"], 1)

    def test_ask_journal_fallback_abstains_on_unknown(self) -> None:
        entries = [
            {
                "id": 1,
                "created_at": "2026-11-10T10:00:00",
                "title": "Doctor Appointment",
                "summary": "Dr. Smith prescribed vitamins on Tuesday.",
                "topics": ["health"],
                "action_items": ["Take vitamins"],
            }
        ]
        res = ask_journal("What is the capital of Mars?", entries, use_ollama=False)
        self.assertTrue(res["abstention"])


if __name__ == "__main__":
    unittest.main()
