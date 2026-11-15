from __future__ import annotations

import json
import logging
import math
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from journal_app.validation import validate_vector

LOGGER = logging.getLogger(__name__)


class JournalStore:
    """SQLite repository for local thoughts with FTS5, vector search, and hybrid retrieval."""

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
            if "embedding_model" not in columns:
                db.execute("ALTER TABLE entries ADD COLUMN embedding_model TEXT")
            if "embedding_dim" not in columns:
                db.execute("ALTER TABLE entries ADD COLUMN embedding_dim INTEGER")
            if "audio_path" not in columns:
                db.execute("ALTER TABLE entries ADD COLUMN audio_path TEXT")

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
            db.execute(
                "CREATE INDEX IF NOT EXISTS idx_entries_model ON entries(embedding_model)"
            )
            db.commit()
        finally:
            db.close()

    def add(self, entry: dict[str, Any]) -> int:
        db = self._connect()
        try:
            embedding = entry.get("embedding")
            embedding_model = entry.get("embedding_model")
            embedding_dim = entry.get("embedding_dim")

            validated_vec: list[float] | None = None
            if embedding:
                validated_vec = validate_vector(embedding)
                if validated_vec:
                    embedding_dim = len(validated_vec)
                else:
                    LOGGER.warning("Discarding invalid or corrupted embedding before inserting entry")
                    embedding = None
                    embedding_model = None
                    embedding_dim = None

            topics = entry.get("topics", [])
            if not isinstance(topics, list):
                topics = [str(topics)] if topics else []

            action_items = entry.get("action_items", [])
            if not isinstance(action_items, list):
                action_items = [str(action_items)] if action_items else []

            cursor = db.execute(
                """INSERT INTO entries
                (created_at,title,summary,transcript,topics,mood,action_items,source,embedding,embedding_model,embedding_dim,audio_path)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    entry.get("created_at")
                    or datetime.now(timezone.utc).isoformat(),
                    str(entry["title"]),
                    str(entry["summary"]),
                    str(entry["transcript"]),
                    json.dumps([str(t) for t in topics]),
                    str(entry.get("mood", "reflective")),
                    json.dumps([str(a) for a in action_items]),
                    str(entry.get("source", "upload")),
                    json.dumps(validated_vec) if validated_vec else None,
                    str(embedding_model) if embedding_model else None,
                    int(embedding_dim) if embedding_dim else None,
                    str(entry.get("audio_path")) if entry.get("audio_path") else None,
                ),
            )
            db.commit()
            return int(cursor.lastrowid)
        finally:
            db.close()

    @staticmethod
    def _decode(row: sqlite3.Row) -> dict[str, Any]:
        item = dict(row)
        try:
            item["topics"] = json.loads(item["topics"]) if item.get("topics") else []
        except Exception:
            item["topics"] = []
        try:
            item["action_items"] = json.loads(item["action_items"]) if item.get("action_items") else []
        except Exception:
            item["action_items"] = []

        item["has_embedding"] = bool(item.get("embedding"))
        # Exclude raw embedding floats from standard listing to save serialization overhead
        # unless explicitly requested
        item.pop("embedding", None)
        return item

    def get_entry(self, entry_id: int) -> dict[str, Any] | None:
        db = self._connect()
        try:
            row = db.execute("SELECT * FROM entries WHERE id = ?", (entry_id,)).fetchone()
            return self._decode(row) if row else None
        finally:
            db.close()

    def delete(self, entry_id: int) -> bool:
        db = self._connect()
        try:
            row = db.execute("SELECT audio_path FROM entries WHERE id = ?", (entry_id,)).fetchone()
            if not row:
                return False
            audio_path = row["audio_path"]
            if audio_path:
                full_path = Path(audio_path)
                if not full_path.is_absolute():
                    full_path = self.path.parent / audio_path
                try:
                    if full_path.exists():
                        full_path.unlink()
                except OSError as err:
                    LOGGER.warning("Could not delete audio file %s: %s", full_path, err)

            db.execute("DELETE FROM entries WHERE id = ?", (entry_id,))
            db.commit()
            return True
        finally:
            db.close()

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
                # Sanitize tokens for SQLite FTS5 MATCH
                safe_tokens = [t.replace('"', "").replace("'", "") for t in query.split() if t.strip()]
                if safe_tokens:
                    fts_query = " ".join(f'"{token}"' for token in safe_tokens)
                    clauses.append("id IN (SELECT rowid FROM entries_fts WHERE entries_fts MATCH ?)")
                    params.append(fts_query)
                else:
                    clauses.append("(title LIKE ? OR summary LIKE ? OR transcript LIKE ? OR topics LIKE ? OR action_items LIKE ?)")
                    params.extend([pattern] * 5)
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
            clauses.append("action_items != '[]' AND action_items != '' AND action_items IS NOT NULL")
        db = self._connect()
        try:
            statement = f"""SELECT * FROM entries WHERE {' AND '.join(clauses)}
                ORDER BY created_at DESC LIMIT ?"""
            try:
                rows = db.execute(statement, (*params, limit)).fetchall()
            except sqlite3.OperationalError:
                if not query:
                    raise
                # Fallback from FTS5 to LIKE if FTS index had query syntax error
                clauses = ["1=1", "(title LIKE ? OR summary LIKE ? OR transcript LIKE ? OR topics LIKE ? OR action_items LIKE ?)"]
                fallback_params = [pattern] * 5
                if topic:
                    clauses.append("topics LIKE ?")
                    fallback_params.append(f'%"{topic}"%')
                if mood:
                    clauses.append("mood = ?")
                    fallback_params.append(mood)
                if has_actions:
                    clauses.append("action_items != '[]' AND action_items != '' AND action_items IS NOT NULL")
                rows = db.execute(
                    f"""SELECT * FROM entries WHERE {' AND '.join(clauses)}
                    ORDER BY created_at DESC LIMIT ?""",
                    (*fallback_params, limit),
                ).fetchall()
            return [self._decode(row) for row in rows]
        finally:
            db.close()

    def semantic_search(
        self,
        vector: list[float],
        model_name: str | None = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        clean_vector = validate_vector(vector)
        if not clean_vector:
            LOGGER.warning("semantic_search received invalid query vector; returning empty results")
            return []

        query_dim = len(clean_vector)
        query_norm = math.sqrt(sum(x * x for x in clean_vector))
        if query_norm <= 1e-12:
            return []

        db = self._connect()
        try:
            scored: list[tuple[float, dict[str, Any]]] = []
            # Match only entries with identical vector dimension
            query_sql = "SELECT * FROM entries WHERE embedding IS NOT NULL AND (embedding_dim = ? OR embedding_dim IS NULL)"
            params: list[Any] = [query_dim]
            if model_name:
                query_sql += " AND (embedding_model = ? OR embedding_model IS NULL)"
                params.append(model_name)

            for row in db.execute(query_sql, tuple(params)):
                try:
                    candidate = json.loads(row["embedding"])
                except Exception:
                    continue

                if not isinstance(candidate, list) or len(candidate) != query_dim:
                    # Reject mismatched vector dimensions to prevent silent dimensional corruption
                    continue

                cand_norm_sq = 0.0
                dot_product = 0.0
                valid = True
                for a, b in zip(clean_vector, candidate):
                    if not isinstance(b, (int, float)) or not math.isfinite(b):
                        valid = False
                        break
                    fb = float(b)
                    dot_product += a * fb
                    cand_norm_sq += fb * fb

                if not valid or cand_norm_sq <= 1e-12:
                    continue

                score = dot_product / (query_norm * math.sqrt(cand_norm_sq))
                if score > 1e-6:
                    item = self._decode(row)
                    item["relevance_score"] = round(float(score), 4)
                    scored.append((score, item))

            return [item for _, item in sorted(scored, key=lambda pair: pair[0], reverse=True)[:limit]]
        finally:
            db.close()

    def hybrid_search(
        self,
        query: str = "",
        vector: list[float] | None = None,
        model_name: str | None = None,
        limit: int = 20,
        topic: str = "",
        mood: str = "",
        has_actions: bool = False,
        rrf_k: int = 60,
    ) -> list[dict[str, Any]]:
        """Combine FTS5 keyword retrieval and semantic search via Reciprocal Rank Fusion (RRF)."""
        clean_vector = validate_vector(vector) if vector else None
        clean_query = query.strip()

        # If only query or only vector is available, route directly
        if not clean_vector and clean_query:
            return self.search(clean_query, limit=limit, topic=topic, mood=mood, has_actions=has_actions)
        if clean_vector and not clean_query:
            return self.semantic_search(clean_vector, model_name=model_name, limit=limit)
        if not clean_vector and not clean_query:
            return self.list(limit=limit)

        # Both query and vector are available: run both and apply RRF
        fts_candidates = self.search(clean_query, limit=limit * 2, topic=topic, mood=mood, has_actions=has_actions)
        semantic_candidates = self.semantic_search(clean_vector, model_name=model_name, limit=limit * 2)

        rrf_scores: dict[int, float] = {}
        entries_by_id: dict[int, dict[str, Any]] = {}

        for rank, entry in enumerate(fts_candidates, start=1):
            eid = entry["id"]
            rrf_scores[eid] = rrf_scores.get(eid, 0.0) + (1.0 / (rrf_k + rank))
            entries_by_id[eid] = entry

        for rank, entry in enumerate(semantic_candidates, start=1):
            eid = entry["id"]
            rrf_scores[eid] = rrf_scores.get(eid, 0.0) + (1.0 / (rrf_k + rank))
            if eid not in entries_by_id:
                entries_by_id[eid] = entry

        sorted_eids = sorted(rrf_scores.keys(), key=lambda eid: rrf_scores[eid], reverse=True)
        results: list[dict[str, Any]] = []
        for eid in sorted_eids[:limit]:
            entry = entries_by_id[eid]
            entry["rrf_score"] = round(rrf_scores[eid], 6)
            results.append(entry)

        return results

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
            topics = set()
            for row in rows:
                if row["topics"]:
                    try:
                        for topic in json.loads(row["topics"]):
                            topics.add(str(topic))
                    except Exception:
                        pass
            moods = sorted({row["mood"] for row in rows if row["mood"]})
            return {"topics": sorted(topics), "moods": moods}
        finally:
            db.close()
