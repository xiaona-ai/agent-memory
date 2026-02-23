"""Core storage and search logic."""
import json
import math
import os
import re
import uuid
from datetime import datetime, timezone
from math import exp
from pathlib import Path
from typing import List, Optional

from .config import STORE_DIR, CONFIG_FILE, load_config, create_default_config

MEMORIES_FILE = "memories.jsonl"


def _root() -> Path:
    """Resolve store directory from config or walk up to find .agent-memory/."""
    config = load_config()
    store_path = config.get("store_path", STORE_DIR)
    # If absolute path, use directly
    sp = Path(store_path)
    if sp.is_absolute():
        return sp
    # Walk up to find existing store
    p = Path.cwd()
    while p != p.parent:
        if (p / store_path).is_dir():
            return p / store_path
        p = p.parent
    return Path.cwd() / store_path


def init_store() -> Path:
    d = Path.cwd() / STORE_DIR
    d.mkdir(exist_ok=True)
    cfg = d / CONFIG_FILE
    if not cfg.exists():
        create_default_config(d)
    mem = d / MEMORIES_FILE
    if not mem.exists():
        mem.touch()
    print(f"Initialized agent-memory in {d}")
    return d


def add_memory(text: str, tags=None, metadata=None, importance: int = 3) -> dict:
    d = _root()
    if not d.is_dir():
        raise FileNotFoundError("Not initialized. Run `agent-memory init` first.")
    importance = max(1, min(5, int(importance)))
    entry = {
        "id": uuid.uuid4().hex[:12],
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "text": text,
        "tags": tags or [],
        "metadata": metadata or {},
        "importance": importance,
    }
    with open(d / MEMORIES_FILE, "a") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(f"Added memory {entry['id']}")
    return entry


def _load_all() -> List[dict]:
    d = _root()
    p = d / MEMORIES_FILE
    if not p.exists():
        return []
    entries = []
    for line in p.read_text().splitlines():
        line = line.strip()
        if line:
            entries.append(json.loads(line))
    return entries


def list_memories(n: int = 20) -> List[dict]:
    return _load_all()[-n:]


def _tokenize(text: str) -> List[str]:
    return re.findall(r"\w+", text.lower())


def search_memories(query: str, limit: Optional[int] = None) -> List[dict]:
    """TF-IDF keyword search with time-decay and importance scoring.

    final_score = tfidf_score * time_factor * (importance / 3.0)
    time_factor = exp(-lambda * days_old)
    """
    config = load_config()
    if limit is None:
        limit = config.get("max_results", 10)
    decay_lambda = config.get("time_decay_lambda", 0.01)
    entries = _load_all()
    if not entries:
        return []
    query_tokens = set(_tokenize(query))
    if not query_tokens:
        return entries[-limit:]

    now = datetime.now(timezone.utc)

    # Build document frequency
    docs = []
    for e in entries:
        tokens = _tokenize(e["text"] + " " + " ".join(e.get("tags", [])))
        docs.append(tokens)
    N = len(docs)
    df = {}
    for tokens in docs:
        for t in set(tokens):
            df[t] = df.get(t, 0) + 1

    # Score each doc
    scored = []
    for i, tokens in enumerate(docs):
        tf = {}
        for t in tokens:
            tf[t] = tf.get(t, 0) + 1
        tfidf_score = 0.0
        for qt in query_tokens:
            if qt in tf and qt in df:
                tfidf_score += tf[qt] * math.log((N + 1) / (df[qt] + 0.5))
        if tfidf_score > 0:
            # Time decay
            ts = entries[i].get("timestamp", "")
            try:
                entry_time = datetime.fromisoformat(ts)
                days_old = (now - entry_time).total_seconds() / 86400.0
            except (ValueError, TypeError):
                days_old = 0.0
            time_factor = exp(-decay_lambda * days_old) if decay_lambda > 0 else 1.0

            # Importance
            importance = entries[i].get("importance", 3)
            importance_factor = importance / 3.0

            final_score = tfidf_score * time_factor * importance_factor
            scored.append((final_score, entries[i]))
    scored.sort(key=lambda x: -x[0])
    return [e for _, e in scored[:limit]]


def delete_memory(memory_id: str) -> bool:
    """Delete a memory by ID. Returns True if found and deleted."""
    d = _root()
    entries = _load_all()
    new_entries = [e for e in entries if e["id"] != memory_id]
    if len(new_entries) == len(entries):
        return False
    with open(d / MEMORIES_FILE, "w") as f:
        for e in new_entries:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    return True


def tag_memory(memory_id: str, add_tags: List[str] = None, remove_tags: List[str] = None) -> Optional[dict]:
    """Add or remove tags from a memory. Returns updated entry or None if not found."""
    d = _root()
    entries = _load_all()
    found = None
    for e in entries:
        if e["id"] == memory_id:
            tags = set(e.get("tags", []))
            if add_tags:
                tags.update(add_tags)
            if remove_tags:
                tags -= set(remove_tags)
            e["tags"] = sorted(tags)
            found = e
            break
    if not found:
        return None
    with open(d / MEMORIES_FILE, "w") as f:
        for e in entries:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    return found


def export_memories(fmt: Optional[str] = None) -> str:
    """Export memories. fmt defaults to config default_export_format."""
    if fmt is None:
        config = load_config()
        fmt = config.get("default_export_format", "md")
    if fmt == "json":
        return export_json()
    return export_markdown()


def export_markdown() -> str:
    entries = _load_all()
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


def export_json() -> str:
    return json.dumps(_load_all(), indent=2, ensure_ascii=False)
