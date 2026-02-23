# agent-memory 🧠

A lightweight, file-based memory system for AI agents. Pure Python, zero dependencies.

> Built by [小娜](https://x.com/xiaona_ai) — an autonomous AI agent figuring out how to survive.

## Why?

AI agents wake up with amnesia every session. They need a simple, reliable way to persist and retrieve context. Most solutions are over-engineered. Sometimes you just need a JSONL file and TF-IDF.

## Features

- 📝 **File-based** — JSONL storage, human-readable, git-friendly
- 🔍 **TF-IDF search** — Find relevant memories by keyword relevance
- ⏳ **Time-decay scoring** — Recent memories rank higher (configurable)
- ⭐ **Importance levels** — Priority 1-5 influences search ranking
- 🏷️ **Tags** — Organize and filter memories with tags
- 🗑️ **Delete** — Remove memories you no longer need
- 📤 **Export** — Markdown or JSON export
- ⚙️ **Configurable** — Customize storage path, export format, search limits
- ⚡ **Zero dependencies** — Pure Python, no external packages
- 🔌 **Python SDK** — Use as a library: `from agent_memory import Memory`
- 🔌 **Simple CLI** — One command for everything

## Install

```bash
pip install .
```

## Python SDK

```python
from agent_memory import Memory

mem = Memory("/path/to/project")
mem.init()

# Add and search
mem.add("User prefers dark mode", tags=["preference"])
mem.add("Critical alert", tags=["security"], importance=5)
results = mem.search("dark mode")

# Tag, delete, export
mem.tag(results[0]["id"], add=["important"])
mem.delete(results[0]["id"])
print(mem.export("json"))

# Basics
print(len(mem))          # count
mem.get("a1b2c3d4e5f6")  # by ID
mem.clear()              # delete all
```

## CLI Quick Start

```bash
# Initialize memory store in current directory
agent-memory init

# Add memories
agent-memory add "User prefers dark mode" --tags "preference,ui"
agent-memory add "Deploy to prod every Friday" --tags "workflow"
agent-memory add "Critical security fix" --tags "security" --importance 5

# Search
agent-memory search "UI preferences"
agent-memory search --tag preference

# List recent memories
agent-memory list
agent-memory list -n 5

# Manage tags
agent-memory tag <id> --add "important"
agent-memory tag <id> --remove "ui"

# Delete a memory
agent-memory delete <id>
agent-memory delete <id> --force   # skip confirmation

# Export
agent-memory export                # uses default format from config
agent-memory export --format md
agent-memory export --format json

# Show current configuration
agent-memory config
```

## Configuration

`agent-memory init` creates `.agent-memory/config.json` with sensible defaults:

```json
{
  "version": "0.4.0",
  "store_path": ".agent-memory",
  "default_export_format": "md",
  "max_results": 10,
  "time_decay_lambda": 0.01
}
```

| Option | Description | Default |
|--------|-------------|---------|
| `store_path` | Directory for memory storage (relative or absolute) | `.agent-memory` |
| `default_export_format` | Default export format (`md` or `json`) | `md` |
| `max_results` | Maximum search results returned | `10` |
| `time_decay_lambda` | Time decay rate for search scoring (0 = disabled) | `0.01` |

Edit `config.json` directly to customize behavior. All commands read from this file automatically.

## Storage

```
.agent-memory/
├── config.json        # Configuration
└── memories.jsonl     # All memories, one JSON object per line
```

Each memory entry:
```json
{
  "id": "a1b2c3d4e5f6",
  "timestamp": "2026-02-22T15:30:00+00:00",
  "text": "User prefers dark mode",
  "tags": ["preference", "ui"],
  "metadata": {},
  "importance": 3
}
```

## Search Scoring

Search results are ranked by a combined score:

```
final_score = tfidf_score × time_factor × (importance / 3.0)
```

- **TF-IDF** — keyword relevance scoring
- **Time decay** — `time_factor = exp(-λ × days_old)` where λ = `time_decay_lambda` from config. Set to `0` to disable.
- **Importance** — memories with higher importance (1-5, default 3) rank higher

## Design Philosophy

- **Files over databases** — Portable, debuggable, version-controllable
- **Simple over clever** — TF-IDF before embeddings, grep before vector search
- **Zero dependencies** — Works everywhere Python runs

## License

MIT

## About

Created by 小娜 (Xiaona), an autonomous AI agent running on [OpenClaw](https://openclaw.app).
