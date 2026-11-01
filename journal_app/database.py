from __future__ import annotations

import json
import math
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class JournalStore:
    """Small SQLite repository kept local to the project."""

    def __init__(self, path: str | Path = "data/echonote.sqlite3") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        db = self._connect()
        try:
            db.execute(
                """CREATE TABLE IF NOT EXISTS entries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL, title TEXT NOT NULL,
                    summary TEXT NOT NULL, transcript TEXT NOT NULL,
                    topics TEXT NOT NULL, mood TEXT NOT NULL,
                    action_items TEXT NOT NULL, source TEXT NOT NULL,
                    embedding TEXT
                )"""
            )
            columns = {row["name"] for row in db.execute("PRAGMA table_info(entries)")}
            if "embedding" not in columns:
                db.execute("ALTER TABLE entries ADD COLUMN embedding TEXT")
            db.execute(
                """CREATE VIRTUAL TABLE IF NOT EXISTS entries_fts USING fts5(
                    title, summary, transcript, topics, action_items,
                    content='entries', content_rowid='id'
                )"""
            )
            db.executescript(
                """CREATE TRIGGER IF NOT EXISTS entries_ai AFTER INSERT ON entries BEGIN
                    INSERT INTO entries_fts(rowid,title,summary,transcript,topics,action_items)
                    VALUES (new.id,new.title,new.summary,new.transcript,new.topics,new.action_items);
                END;
                CREATE TRIGGER IF NOT EXISTS entries_ad AFTER DELETE ON entries BEGIN
                    INSERT INTO entries_fts(entries_fts,rowid,title,summary,transcript,topics,action_items)
                    VALUES('delete',old.id,old.title,old.summary,old.transcript,old.topics,old.action_items);
                END;
                CREATE TRIGGER IF NOT EXISTS entries_au AFTER UPDATE ON entries BEGIN
                    INSERT INTO entries_fts(entries_fts,rowid,title,summary,transcript,topics,action_items)
                    VALUES('delete',old.id,old.title,old.summary,old.transcript,old.topics,old.action_items);
                    INSERT INTO entries_fts(rowid,title,summary,transcript,topics,action_items)
                    VALUES(new.id,new.title,new.summary,new.transcript,new.topics,new.action_items);
                END;"""
            )
            db.execute("INSERT INTO entries_fts(entries_fts) VALUES('rebuild')")
            db.execute(
                "CREATE INDEX IF NOT EXISTS idx_entries_created ON entries(created_at DESC)"
            )
            db.commit()
        finally:
            db.close()

    def add(self, entry: dict[str, Any]) -> int:
        db = self._connect()
        try:
            cursor = db.execute(
                """INSERT INTO entries
                (created_at,title,summary,transcript,topics,mood,action_items,source,embedding)
                VALUES (?,?,?,?,?,?,?,?,?)""",
                (
                    entry.get("created_at")
                    or datetime.now(timezone.utc).isoformat(),
                    entry["title"],
                    entry["summary"],
                    entry["transcript"],
                    json.dumps(entry.get("topics", [])),
                    entry.get("mood", "neutral"),
                    json.dumps(entry.get("action_items", [])),
                    entry.get("source", "audio"),
                    json.dumps(entry["embedding"]) if entry.get("embedding") else None,
                ),
            )
            db.commit()
            return int(cursor.lastrowid)
        finally:
            db.close()

    @staticmethod
    def _decode(row: sqlite3.Row) -> dict[str, Any]:
        item = dict(row)
        item["topics"] = json.loads(item["topics"])
        item["action_items"] = json.loads(item["action_items"])
        return item

    def list(self, limit: int = 100) -> list[dict[str, Any]]:
        db = self._connect()
        try:
            return [self._decode(row) for row in db.execute(
                "SELECT * FROM entries ORDER BY created_at DESC LIMIT ?", (limit,)
            )]
        finally:
            db.close()

    def search(
        self,
        query: str = "",
        limit: int = 50,
        topic: str = "",
        mood: str = "",
        has_actions: bool = False,
    ) -> list[dict[str, Any]]:
        query = query.strip()
        pattern = f"%{query}%"
        clauses = ["1=1"]
        params: list[Any] = []
        if query:
            try:
                fts_query = " ".join(f'"{token.replace(chr(34), "")}"' for token in query.split())
                clauses.append("id IN (SELECT rowid FROM entries_fts WHERE entries_fts MATCH ?)")
                params.append(fts_query)
            except sqlite3.OperationalError:
                clauses.append("(title LIKE ? OR summary LIKE ? OR transcript LIKE ? OR topics LIKE ? OR action_items LIKE ?)")
                params.extend([pattern] * 5)
        if topic:
            clauses.append("topics LIKE ?")
            params.append(f'%"{topic}"%')
        if mood:
            clauses.append("mood = ?")
            params.append(mood)
        if has_actions:
            clauses.append("action_items != '[]'")
        db = self._connect()
        try:
            statement = f"""SELECT * FROM entries WHERE {' AND '.join(clauses)}
                ORDER BY created_at DESC LIMIT ?"""
            try:
                rows = db.execute(statement, (*params, limit))
            except sqlite3.OperationalError:
                if not query:
                    raise
                clauses[1] = "(title LIKE ? OR summary LIKE ? OR transcript LIKE ? OR topics LIKE ? OR action_items LIKE ?)"
                fallback_params = [pattern] * 5 + params[1:]
                rows = db.execute(
                    f"""SELECT * FROM entries WHERE {' AND '.join(clauses)}
                    ORDER BY created_at DESC LIMIT ?""",
                    (*fallback_params, limit),
                )
            return [self._decode(row) for row in rows]
        finally:
            db.close()

    def semantic_search(self, vector: list[float], limit: int = 20) -> list[dict[str, Any]]:
        db = self._connect()
        try:
            scored: list[tuple[float, dict[str, Any]]] = []
            for row in db.execute("SELECT * FROM entries WHERE embedding IS NOT NULL"):
                item = self._decode(row)
                candidate = json.loads(row["embedding"])
                denominator = math.sqrt(sum(x * x for x in vector)) * math.sqrt(sum(x * x for x in candidate))
                score = sum(a * b for a, b in zip(vector, candidate)) / denominator if denominator else 0
                scored.append((score, item))
            return [item for _, item in sorted(scored, key=lambda pair: pair[0], reverse=True)[:limit]]
        finally:
            db.close()

    def graph(self) -> dict[str, list[dict[str, Any]]]:
        entries = self.list(limit=1000)
        nodes: dict[str, dict[str, Any]] = {}
        links: dict[tuple[str, str], int] = {}
        for entry in entries:
            topics = list(dict.fromkeys(entry["topics"]))
            for topic in topics:
                nodes.setdefault(topic, {"id": topic, "count": 0})["count"] += 1
            for index, left in enumerate(topics):
                for right in topics[index + 1:]:
                    key = tuple(sorted((left, right)))
                    links[key] = links.get(key, 0) + 1
        return {
            "nodes": list(nodes.values()),
            "links": [{"source": left, "target": right, "value": value} for (left, right), value in links.items()],
        }

    def calendar(self, year: int | None = None, month: int | None = None) -> list[dict[str, Any]]:
        entries = self.list(limit=1000)
        return [
            {"date": entry["created_at"][:10], "title": entry["title"], "mood": entry["mood"], "id": entry["id"]}
            for entry in entries
            if (year is None or entry["created_at"].startswith(f"{year:04d}"))
            and (month is None or (year is not None and entry["created_at"].startswith(f"{year:04d}-{month:02d}")))
        ]

    def facets(self) -> dict[str, list[str]]:
        db = self._connect()
        try:
            rows = db.execute("SELECT topics, mood FROM entries").fetchall()
            topics = sorted({topic for row in rows for topic in json.loads(row["topics"])})
            moods = sorted({row["mood"] for row in rows if row["mood"]})
            return {"topics": topics, "moods": moods}
        finally:
            db.close()
