"""Tests for the Python SDK (Memory class)."""
import json
import tempfile
from pathlib import Path
from agent_memory import Memory


def test_init_and_add():
    with tempfile.TemporaryDirectory() as tmp:
        mem = Memory(tmp)
        mem.init()
        assert (Path(tmp) / ".agent-memory" / "memories.jsonl").exists()
        e = mem.add("hello world", tags=["test"])
        assert e["text"] == "hello world"
        assert e["tags"] == ["test"]
        assert len(e["id"]) == 12


def test_list_and_get():
    with tempfile.TemporaryDirectory() as tmp:
        mem = Memory(tmp)
        mem.init()
        e1 = mem.add("first")
        e2 = mem.add("second")
        all_ = mem.list()
        assert len(all_) == 2
        assert mem.get(e1["id"])["text"] == "first"
        assert mem.get("nonexistent") is None


def test_search():
    with tempfile.TemporaryDirectory() as tmp:
        mem = Memory(tmp)
        mem.init()
        mem.add("python is great")
        mem.add("rust is fast")
        mem.add("python packaging guide", tags=["python"])
        results = mem.search("python")
        assert len(results) >= 2
        assert all("python" in r["text"].lower() or "python" in r.get("tags", []) for r in results)


def test_search_with_tag():
    with tempfile.TemporaryDirectory() as tmp:
        mem = Memory(tmp)
        mem.init()
        mem.add("tagged python entry", tags=["important"])
        mem.add("tagged rust entry", tags=["important"])
        mem.add("untagged python entry")
        results = mem.search("python", tag="important")
        assert len(results) == 1
        assert results[0]["tags"] == ["important"]
        assert "python" in results[0]["text"]


def test_delete():
    with tempfile.TemporaryDirectory() as tmp:
        mem = Memory(tmp)
        mem.init()
        e = mem.add("to delete")
        assert mem.delete(e["id"]) is True
        assert mem.delete(e["id"]) is False
        assert mem.count() == 0


def test_tag():
    with tempfile.TemporaryDirectory() as tmp:
        mem = Memory(tmp)
        mem.init()
        e = mem.add("taggable")
        updated = mem.tag(e["id"], add=["a", "b"])
        assert updated["tags"] == ["a", "b"]
        updated = mem.tag(e["id"], remove=["a"])
        assert updated["tags"] == ["b"]
        assert mem.tag("nonexistent", add=["x"]) is None


def test_export_md_and_json():
    with tempfile.TemporaryDirectory() as tmp:
        mem = Memory(tmp)
        mem.init()
        mem.add("exportable", tags=["demo"])
        md = mem.export("md")
        assert "# Agent Memory Export" in md
        assert "exportable" in md
        j = mem.export("json")
        data = json.loads(j)
        assert len(data) == 1


def test_count_and_clear():
    with tempfile.TemporaryDirectory() as tmp:
        mem = Memory(tmp)
        mem.init()
        mem.add("a")
        mem.add("b")
        assert mem.count() == 2
        assert len(mem) == 2
        cleared = mem.clear()
        assert cleared == 2
        assert mem.count() == 0


def test_custom_config():
    with tempfile.TemporaryDirectory() as tmp:
        mem = Memory(tmp, config={"max_results": 1})
        mem.init()
        mem.add("one")
        mem.add("two")
        # search with default limit should respect config
        results = mem.search("one two")
        assert len(results) == 1


def test_not_initialized():
    with tempfile.TemporaryDirectory() as tmp:
        mem = Memory(tmp)
        try:
            mem.add("fail")
            assert False, "Should have raised"
        except FileNotFoundError:
            pass
