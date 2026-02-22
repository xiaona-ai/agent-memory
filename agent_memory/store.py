"""Core storage and search logic."""
import json
import math
import os
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path

STORE_DIR = ".agent-memory"
MEMORIES_FILE = "memories.jsonl"
CONFIG_FILE = "config.json"


def _root() -> Path:
    """Walk up to find .agent-memory/, fallback to cwd."""
    p = Path.cwd()
    while p != p.parent:
        if (p / STORE_DIR).is_dir():
            return p / STORE_DIR
        p = p.parent
    return Path.cwd() / STORE_DIR


def init_store() -> Path:
    d = Path.cwd() / STORE_DIR
    d.mkdir(exist_ok=True)
    cfg = d / CONFIG_FILE
    if not cfg.exists():
        cfg.write_text(json.dumps({"version": "0.1.0"}, indent=2) + "\n")
    mem = d / MEMORIES_FILE
    if not mem.exists():
        mem.touch()
    print(f"Initialized agent-memory in {d}")
    return d


def add_memory(text: str, tags=None, metadata=None) -> dict:
    d = _root()
    if not d.is_dir():
        raise FileNotFoundError("Not initialized. Run `agent-memory init` first.")
    entry = {
        "id": uuid.uuid4().hex[:12],
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "text": text,
        "tags": tags or [],
        "metadata": metadata or {},
    }
    with open(d / MEMORIES_FILE, "a") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(f"Added memory {entry['id']}")
    return entry


def _load_all() -> list[dict]:
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


def list_memories(n: int = 20) -> list[dict]:
    return _load_all()[-n:]


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def search_memories(query: str, limit: int = 10) -> list[dict]:
    """Simple TF-IDF keyword search."""
    entries = _load_all()
    if not entries:
        return []
    query_tokens = set(_tokenize(query))
    if not query_tokens:
        return entries[-limit:]

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
        score = 0.0
        for qt in query_tokens:
            if qt in tf and qt in df:
                score += tf[qt] * math.log((N + 1) / (df[qt] + 1))
        if score > 0:
            scored.append((score, entries[i]))
    scored.sort(key=lambda x: -x[0])
    return [e for _, e in scored[:limit]]


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
