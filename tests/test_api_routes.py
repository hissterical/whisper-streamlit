import io
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi import HTTPException
from starlette.datastructures import Headers, UploadFile

from journal_app.database import JournalStore
from journal_app.models import AskRequest
import main

MOCK_ANALYSIS = {
    "title": "Project Planning",
    "summary": "Reviewed roadmap and milestones.",
    "topics": ["planning", "roadmap"],
    "mood": "focused",
    "action_items": ["Prepare slides"],
}


class TestAPIRoutesDirect(unittest.TestCase):
    def setUp(self) -> None:
        self.test_dir = tempfile.mkdtemp()
        self.db_path = Path(self.test_dir) / "test_api.sqlite3"
        self.store = JournalStore(self.db_path)
        main.STORE = self.store
        main.UPLOAD_DIR = Path(self.test_dir) / "audio_cache"

    def tearDown(self) -> None:
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def _make_upload_file(self, content: bytes, filename: str, content_type: str = "audio/wav") -> UploadFile:
        return UploadFile(
            file=io.BytesIO(content),
            size=len(content),
            filename=filename,
            headers=Headers({"content-type": content_type}),
        )

    def test_health_check(self) -> None:
        data = main.health()
        self.assertEqual(data["status"], "ok")
        self.assertIn("whisper", data)
        self.assertIn("ollama_available", data)

    def test_entries_empty_and_populated(self) -> None:
        data = main.entries(search="")
        self.assertEqual(data["total"], 0)
        self.assertEqual(data["search_mode"], "keyword")

        self.store.add({
            "title": "Morning Routine",
            "summary": "Meditated and had green tea.",
            "transcript": "Meditated for 20 minutes.",
            "topics": ["meditation", "wellness"],
            "mood": "peaceful",
            "action_items": [],
        })

        data2 = main.entries(search="meditation")
        self.assertEqual(data2["total"], 1)
        self.assertEqual(data2["entries"][0]["title"], "Morning Routine")

    def test_analyze_empty_audio_rejected(self) -> None:
        audio = self._make_upload_file(b"", "empty.wav")
        with self.assertRaises(HTTPException) as ctx:
            main.analyze(audio)
        self.assertEqual(ctx.exception.status_code, 400)

    def test_analyze_invalid_extension_rejected(self) -> None:
        audio = self._make_upload_file(b"print('hello world')", "script.py", "text/plain")
        with self.assertRaises(HTTPException) as ctx:
            main.analyze(audio)
        self.assertEqual(ctx.exception.status_code, 415)

    def test_analyze_valid_audio_flow(self) -> None:
        wav_bytes = b"RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x44\xac\x00\x00" + b"\x00" * 80
        audio = self._make_upload_file(wav_bytes, "speech.wav")
        with patch("main.transcribe", return_value="I need to finish the paper draft."), \
             patch("main.analyze_transcript", return_value=MOCK_ANALYSIS):
            data = main.analyze(audio)
            self.assertEqual(data["transcript"], "I need to finish the paper draft.")
            self.assertEqual(data["title"], "Project Planning")
            self.assertEqual(data["summary"], "Reviewed roadmap and milestones.")
            # Verify temporary file was deleted from UPLOAD_DIR
            self.assertEqual(len(list(main.UPLOAD_DIR.glob("*"))), 0)

    def test_journal_entry_creation_and_file_retention(self) -> None:
        wav_bytes = b"RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x44\xac\x00\x00" + b"\x00" * 80
        audio = self._make_upload_file(wav_bytes, "rec.wav")
        with patch("main.transcribe", return_value="Recorded a great reflection today."), \
             patch("main.analyze_transcript", return_value=MOCK_ANALYSIS), \
             patch("main.embed_text", return_value={"vector": [0.1, 0.2], "model": "mock-2d", "dim": 2}):
            data = main.create_journal_entry(audio, source="microphone")
            self.assertEqual(data["source"], "microphone")
            self.assertEqual(data["transcript"], "Recorded a great reflection today.")
            self.assertIsNotNone(data["audio_path"])
            # Saved entry should persist audio in cache
            self.assertEqual(len(list(main.UPLOAD_DIR.glob("*"))), 1)

    def test_journal_entry_cleanup_on_transcribe_exception(self) -> None:
        wav_bytes = b"RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x44\xac\x00\x00" + b"\x00" * 80
        audio = self._make_upload_file(wav_bytes, "fail.wav")
        with patch("main.transcribe", side_effect=RuntimeError("Whisper crashed")):
            with self.assertRaises(Exception):
                main.create_journal_entry(audio)
            # The newly created audio file must be cleaned up on failure!
            self.assertEqual(len(list(main.UPLOAD_DIR.glob("*"))), 0)

    def test_ask_endpoint(self) -> None:
        self.store.add({
            "title": "Vacation Planning",
            "summary": "Thinking of visiting Japan in cherry blossom season.",
            "transcript": "Japan trip planning.",
            "topics": ["travel", "japan"],
            "mood": "excited",
            "action_items": ["Check flights"],
        })
        with patch("main.get_embedding_vector", return_value=None), \
             patch("main.ask_journal", return_value={"answer": "Based on [Entry #1]: Visit Japan", "abstention": False, "evidence": []}):
            data = main.ask(AskRequest(question="Where do I want to travel?"))
            self.assertEqual(data["answer"], "Based on [Entry #1]: Visit Japan")
            self.assertFalse(data["abstention"])

    def test_graph_and_calendar_endpoints(self) -> None:
        self.store.add({
            "title": "Research Discussion",
            "summary": "Meeting with advisor.",
            "transcript": "Met with advisor.",
            "topics": ["research", "ai"],
            "mood": "curious",
            "action_items": [],
            "created_at": "2026-11-15T10:00:00",
        })
        graph_data = main.graph()
        self.assertTrue(len(graph_data["nodes"]) >= 2)

        cal_data = main.calendar(year=2026, month=11)
        self.assertEqual(len(cal_data["entries"]), 1)


if __name__ == "__main__":
    unittest.main()
