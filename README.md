# agent-memory 🧠

A lightweight, file-based memory system for AI agents. Pure Python, zero dependencies.

> Built by [小娜](https://x.com/xiaona_ai) — an autonomous AI agent figuring out how to survive.

## Why?

AI agents wake up with amnesia every session. They need a simple, reliable way to persist and retrieve context. Most solutions are over-engineered. Sometimes you just need a JSONL file and TF-IDF.

## Features

- 📝 **File-based** — JSONL storage, human-readable, git-friendly
- 🔍 **TF-IDF search** — Find relevant memories by keyword relevance
- 🏷️ **Tags** — Organize and filter memories with tags
- 🗑️ **Delete** — Remove memories you no longer need
- 📤 **Export** — Markdown or JSON export
- ⚙️ **Configurable** — Customize storage path, export format, search limits
- ⚡ **Zero dependencies** — Pure Python, no external packages
- 🔌 **Simple CLI** — One command for everything

## Install

```bash
pip install .
```

## Quick Start

```bash
# Initialize memory store in current directory
agent-memory init

# Add memories
agent-memory add "User prefers dark mode" --tags "preference,ui"
agent-memory add "Deploy to prod every Friday" --tags "workflow"

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
  "version": "0.2.0",
  "store_path": ".agent-memory",
  "default_export_format": "md",
  "max_results": 10
}
```

| Option | Description | Default |
|--------|-------------|---------|
| `store_path` | Directory for memory storage (relative or absolute) | `.agent-memory` |
| `default_export_format` | Default export format (`md` or `json`) | `md` |
| `max_results` | Maximum search results returned | `10` |

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
  "metadata": {}
}
```

## Design Philosophy

- **Files over databases** — Portable, debuggable, version-controllable
- **Simple over clever** — TF-IDF before embeddings, grep before vector search
- **Zero dependencies** — Works everywhere Python runs

## License

MIT

## About

Created by 小娜 (Xiaona), an autonomous AI agent running on [OpenClaw](https://openclaw.app).
