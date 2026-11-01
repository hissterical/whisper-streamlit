import shutil
import unittest
from pathlib import Path

from journal_app.database import JournalStore
from journal_app.services import analyze_transcript


class JournalTests(unittest.TestCase):
    def test_round_trip_and_search(self) -> None:
        folder = Path("data") / "test_journal"
        folder.mkdir(parents=True, exist_ok=True)
        try:
            store = JournalStore(folder / "journal.sqlite3")
            store.add({
                "title": "Plan a hike",
                "summary": "Plan a hike this weekend.",
                "transcript": "I should plan a hike this weekend.",
                "topics": ["hiking"],
                "mood": "energized",
                "action_items": ["Plan a hike"],
            })
            results = store.search("hiking")
            self.assertEqual(len(results), 1)
            self.assertEqual(results[0]["topics"], ["hiking"])
            self.assertEqual(len(store.graph()["nodes"]), 1)
            self.assertEqual(store.calendar()[0]["title"], "Plan a hike")
        finally:
            shutil.rmtree(folder, ignore_errors=True)

    def test_fallback_analysis_has_journal_fields(self) -> None:
        result = analyze_transcript("I feel good about the project. I need to call Sam.", use_ollama=False)
        self.assertTrue(result["title"])
        self.assertTrue(result["summary"])
        self.assertIn("action_items", result)


if __name__ == "__main__":
    unittest.main()
