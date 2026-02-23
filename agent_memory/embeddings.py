"""Optional vector embeddings via OpenAI-compatible API.

Requires no extra dependencies — uses urllib from stdlib.
Configure via .agent-memory/config.json:

{
    "embedding": {
        "api_base": "https://api.openai.com/v1",
        "api_key": "sk-...",
        "model": "text-embedding-3-small"
    }
}

Or set environment variables:
    AGENT_MEMORY_EMBEDDING_API_BASE
    AGENT_MEMORY_EMBEDDING_API_KEY
    AGENT_MEMORY_EMBEDDING_MODEL
"""

import json
import math
import os
import urllib.request
from pathlib import Path
from typing import List, Optional

VECTORS_FILE = "vectors.jsonl"


def _get_embedding_config(config: dict) -> Optional[dict]:
    """Extract embedding config from config dict + env vars."""
    emb = config.get("embedding", {})
    api_base = os.environ.get("AGENT_MEMORY_EMBEDDING_API_BASE", emb.get("api_base", ""))
    api_key = os.environ.get("AGENT_MEMORY_EMBEDDING_API_KEY", emb.get("api_key", ""))
    model = os.environ.get("AGENT_MEMORY_EMBEDDING_MODEL", emb.get("model", "text-embedding-3-small"))

    if not api_base or not api_key:
        return None
    return {"api_base": api_base.rstrip("/"), "api_key": api_key, "model": model}


def get_embeddings(texts: List[str], config: dict) -> Optional[List[List[float]]]:
    """Get embeddings for a list of texts. Returns None if not configured."""
    emb_config = _get_embedding_config(config)
    if not emb_config:
        return None

    url = f"{emb_config['api_base']}/embeddings"
    payload = json.dumps({
        "input": texts,
        "model": emb_config["model"],
    }).encode("utf-8")

    req = urllib.request.Request(
        url,
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {emb_config['api_key']}",
        },
    )

    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read())
        # Sort by index to ensure order matches input
        sorted_data = sorted(data["data"], key=lambda x: x["index"])
        return [item["embedding"] for item in sorted_data]
    except Exception:
        return None


def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Pure Python cosine similarity. No numpy needed."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


class VectorStore:
    """Manages vector embeddings alongside the JSONL memory store."""

    def __init__(self, store_dir: Path, config: dict):
        self._store_dir = store_dir
        self._config = config
        self._vectors_path = store_dir / VECTORS_FILE

    @property
    def enabled(self) -> bool:
        return _get_embedding_config(self._config) is not None

    def _load_vectors(self) -> dict:
        """Load id -> vector mapping."""
        if not self._vectors_path.exists():
            return {}
        vectors = {}
        for line in self._vectors_path.read_text().splitlines():
            line = line.strip()
            if line:
                entry = json.loads(line)
                vectors[entry["id"]] = entry["vector"]
        return vectors

    def _append_vector(self, memory_id: str, vector: List[float]):
        with open(self._vectors_path, "a") as f:
            f.write(json.dumps({"id": memory_id, "vector": vector}) + "\n")

    def _save_vectors(self, vectors: dict):
        with open(self._vectors_path, "w") as f:
            for mid, vec in vectors.items():
                f.write(json.dumps({"id": mid, "vector": vec}) + "\n")

    def embed_and_store(self, memory_id: str, text: str) -> bool:
        """Embed text and store vector. Returns True on success."""
        result = get_embeddings([text], self._config)
        if result is None:
            return False
        self._append_vector(memory_id, result[0])
        return True

    def embed_batch(self, items: List[dict]) -> int:
        """Embed multiple memories. items: list of {id, text}. Returns count stored."""
        if not items:
            return 0
        texts = [item["text"] for item in items]
        result = get_embeddings(texts, self._config)
        if result is None:
            return 0
        for item, vector in zip(items, result):
            self._append_vector(item["id"], vector)
        return len(items)

    def search(self, query: str, entries: List[dict], limit: int = 10) -> List[dict]:
        """Vector similarity search. Returns entries sorted by similarity."""
        query_vec_result = get_embeddings([query], self._config)
        if query_vec_result is None:
            return []
        query_vec = query_vec_result[0]

        vectors = self._load_vectors()
        scored = []
        for e in entries:
            vec = vectors.get(e["id"])
            if vec:
                sim = cosine_similarity(query_vec, vec)
                scored.append((sim, e))
        scored.sort(key=lambda x: -x[0])
        return [e for _, e in scored[:limit]]

    def delete(self, memory_id: str):
        """Remove vector for a memory."""
        vectors = self._load_vectors()
        if memory_id in vectors:
            del vectors[memory_id]
            self._save_vectors(vectors)

    def rebuild(self, entries: List[dict], batch_size: int = 100) -> int:
        """Rebuild all vectors from scratch. Returns count embedded."""
        self._save_vectors({})  # Clear
        total = 0
        for i in range(0, len(entries), batch_size):
            batch = [{"id": e["id"], "text": e["text"]} for e in entries[i:i + batch_size]]
            total += self.embed_batch(batch)
        return total
