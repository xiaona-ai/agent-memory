"""Tests for time-decay search and importance scoring (v0.4.0)."""
import json
from datetime import datetime, timezone, timedelta
from pathlib import Path

import pytest
from agent_memory.sdk import Memory


@pytest.fixture
def mem(tmp_path):
    m = Memory(str(tmp_path), config={"time_decay_lambda": 0.05})
    m.init()
    return m


def _inject(mem, text, days_ago=0, importance=3, tags=None):
    """Add a memory with a specific timestamp in the past."""
    entry = mem.add(text, tags=tags, importance=importance)
    # Rewrite timestamp
    ts = (datetime.now(timezone.utc) - timedelta(days=days_ago)).isoformat()
    entries = mem._load_all()
    for e in entries:
        if e["id"] == entry["id"]:
            e["timestamp"] = ts
    mem._save_all(entries)
    return entry


def _add_filler(mem):
    """Add a non-matching filler so TF-IDF df ratios work."""
    mem.add("completely unrelated filler content about weather and cooking")


class TestTimeDecay:
    def test_recent_ranks_higher(self, mem):
        _add_filler(mem)
        _inject(mem, "python programming tips", days_ago=100)
        _inject(mem, "python programming tips", days_ago=0)

        results = mem.search("python programming")
        assert len(results) == 2
        # Recent one should be first
        first_date = datetime.fromisoformat(results[0]["timestamp"]).date()
        recent_date = datetime.now(timezone.utc).date()
        assert abs((first_date - recent_date).days) <= 1

    def test_decay_disabled_when_lambda_zero(self, tmp_path):
        m = Memory(str(tmp_path), config={"time_decay_lambda": 0})
        m.init()
        m.add("completely unrelated filler about weather")
        _inject(m, "unique alpha keyword", days_ago=100)
        _inject(m, "unique alpha keyword", days_ago=0)
        results = m.search("unique alpha keyword")
        assert len(results) == 2


class TestImportance:
    def test_importance_stored(self, mem):
        entry = mem.add("test memory", importance=5)
        assert entry["importance"] == 5
        got = mem.get(entry["id"])
        assert got["importance"] == 5

    def test_importance_default(self, mem):
        entry = mem.add("test memory")
        assert entry["importance"] == 3

    def test_importance_clamped(self, mem):
        entry = mem.add("test", importance=10)
        assert entry["importance"] == 5
        entry2 = mem.add("test", importance=-1)
        assert entry2["importance"] == 1

    def test_high_importance_ranks_higher(self, mem):
        _add_filler(mem)
        _inject(mem, "database optimization guide", days_ago=0, importance=1)
        _inject(mem, "database optimization guide", days_ago=0, importance=5)

        results = mem.search("database optimization")
        assert len(results) == 2
        assert results[0]["importance"] == 5
        assert results[1]["importance"] == 1

    def test_backward_compat_no_importance(self, mem):
        _add_filler(mem)
        entry = {
            "id": "oldentry12345",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "text": "legacy data without importance",
            "tags": [],
            "metadata": {},
        }
        with open(mem._memories_path, "a") as f:
            f.write(json.dumps(entry) + "\n")

        results = mem.search("legacy data")
        assert len(results) == 1


class TestCombined:
    def test_importance_can_overcome_recency(self, mem):
        _add_filler(mem)
        _inject(mem, "critical security alert", days_ago=10, importance=5)
        _inject(mem, "critical security alert", days_ago=0, importance=1)

        results = mem.search("critical security alert")
        assert len(results) == 2
        assert results[0]["importance"] == 5
