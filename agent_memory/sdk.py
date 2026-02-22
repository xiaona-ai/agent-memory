"""Python SDK for agent-memory — use as a library, not just CLI.

Usage:
    from agent_memory import Memory

    mem = Memory("/path/to/project")  # or Memory() for cwd
    mem.init()
    mem.add("learned something important", tags=["insight"])
    results = mem.search("important")
    all_mems = mem.list(limit=50)
    mem.tag(memory_id, add=["key"], remove=["old"])
    mem.delete(memory_id)
    md = mem.export(fmt="md")  # or "json"
"""

import json
import math
import os
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


STORE_DIR = ".agent-memory"
MEMORIES_FILE = "memories.jsonl"
CONFIG_FILE = "config.json"

DEFAULT_CONFIG = {
    "store_path": STORE_DIR,
    "default_export_format": "md",
    "max_results": 10,
}


class Memory:
    """Lightweight memory store for AI agents.

    Args:
        path: Root directory containing (or to contain) .agent-memory/.
              Defaults to current working directory.
        config: Optional config overrides (merged with defaults).
    """

    def __init__(self, path: Optional[str] = None, config: Optional[dict] = None):
        self._root = Path(path or os.getcwd())
        self._config = {**DEFAULT_CONFIG, **(config or {})}
        self._store_dir: Optional[Path] = None

    @property
    def store(self) -> Path:
        """Resolved store directory."""
        if self._store_dir is not None:
            return self._store_dir
        sp = self._config.get("store_path", STORE_DIR)
        p = Path(sp)
        if p.is_absolute():
            self._store_dir = p
        else:
            self._store_dir = self._root / sp
        return self._store_dir

    @property
    def _memories_path(self) -> Path:
        return self.store / MEMORIES_FILE

    def init(self) -> Path:
        """Initialize the memory store. Returns the store directory."""
        self.store.mkdir(parents=True, exist_ok=True)
        cfg_path = self.store / CONFIG_FILE
        if not cfg_path.exists():
            cfg_path.write_text(json.dumps(self._config, indent=2))
        if not self._memories_path.exists():
            self._memories_path.touch()
        return self.store

    def _ensure_store(self):
        if not self.store.is_dir():
            raise FileNotFoundError(
                f"Memory store not found at {self.store}. Call .init() first."
            )

    def _load_all(self) -> list:
        self._ensure_store()
        p = self._memories_path
        if not p.exists():
            return []
        entries = []
        for line in p.read_text().splitlines():
            line = line.strip()
            if line:
                entries.append(json.loads(line))
        return entries

    def _save_all(self, entries: list):
        with open(self._memories_path, "w") as f:
            for e in entries:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")

    def add(
        self,
        text: str,
        tags: Optional[list] = None,
        metadata: Optional[dict] = None,
    ) -> dict:
        """Add a memory. Returns the created entry."""
        self._ensure_store()
        entry = {
            "id": uuid.uuid4().hex[:12],
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "text": text,
            "tags": tags or [],
            "metadata": metadata or {},
        }
        with open(self._memories_path, "a") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        return entry

    def list(self, limit: int = 20) -> list:
        """List recent memories (most recent last)."""
        return self._load_all()[-limit:]

    def get(self, memory_id: str) -> Optional[dict]:
        """Get a single memory by ID."""
        for e in self._load_all():
            if e["id"] == memory_id:
                return e
        return None

    def search(
        self,
        query: str,
        limit: Optional[int] = None,
        tag: Optional[str] = None,
    ) -> list:
        """TF-IDF keyword search with optional tag filter."""
        if limit is None:
            limit = self._config.get("max_results", 10)
        entries = self._load_all()

        # Tag filter
        if tag:
            entries = [e for e in entries if tag in e.get("tags", [])]

        if not entries:
            return []

        query_tokens = set(self._tokenize(query))
        if not query_tokens:
            return entries[-limit:]

        # TF-IDF scoring
        docs = []
        for e in entries:
            tokens = self._tokenize(e["text"] + " " + " ".join(e.get("tags", [])))
            docs.append(tokens)
        N = len(docs)
        df: dict = {}
        for tokens in docs:
            for t in set(tokens):
                df[t] = df.get(t, 0) + 1

        scored = []
        for i, tokens in enumerate(docs):
            tf: dict = {}
            for t in tokens:
                tf[t] = tf.get(t, 0) + 1
            score = sum(
                tf[qt] * math.log((N + 1) / (df[qt] + 1))
                for qt in query_tokens
                if qt in tf and qt in df
            )
            if score > 0:
                scored.append((score, entries[i]))
        scored.sort(key=lambda x: -x[0])
        return [e for _, e in scored[:limit]]

    def delete(self, memory_id: str) -> bool:
        """Delete a memory by ID. Returns True if found."""
        entries = self._load_all()
        new = [e for e in entries if e["id"] != memory_id]
        if len(new) == len(entries):
            return False
        self._save_all(new)
        return True

    def tag(
        self,
        memory_id: str,
        add: Optional[list] = None,
        remove: Optional[list] = None,
    ) -> Optional[dict]:
        """Add/remove tags. Returns updated entry or None if not found."""
        entries = self._load_all()
        found = None
        for e in entries:
            if e["id"] == memory_id:
                tags = set(e.get("tags", []))
                if add:
                    tags.update(add)
                if remove:
                    tags -= set(remove)
                e["tags"] = sorted(tags)
                found = e
                break
        if not found:
            return None
        self._save_all(entries)
        return found

    def export(self, fmt: Optional[str] = None) -> str:
        """Export memories as markdown or JSON string."""
        if fmt is None:
            fmt = self._config.get("default_export_format", "md")
        entries = self._load_all()
        if fmt == "json":
            return json.dumps(entries, indent=2, ensure_ascii=False)
        # Markdown
        lines = ["# Agent Memory Export", ""]
        for e in entries:
            ts = e.get("timestamp", "")[:19].replace("T", " ")
            tags = ", ".join(e.get("tags", []))
            lines.append(f"## {e['id']} ({ts})")
            if tags:
                lines.append(f"**Tags:** {tags}")
            lines.append("")
            lines.append(e["text"])
            lines.append("")
        return "\n".join(lines)

    def count(self) -> int:
        """Return total number of memories."""
        return len(self._load_all())

    def clear(self) -> int:
        """Delete all memories. Returns count of deleted entries."""
        entries = self._load_all()
        n = len(entries)
        self._save_all([])
        return n

    @staticmethod
    def _tokenize(text: str) -> list:
        return re.findall(r"\w+", text.lower())

    def __repr__(self) -> str:
        return f"Memory(path={self._root!r}, store={self.store!r})"

    def __len__(self) -> int:
        return self.count()
