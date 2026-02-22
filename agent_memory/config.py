"""Configuration management for agent-memory."""
import json
from pathlib import Path
from typing import Any

STORE_DIR = ".agent-memory"
CONFIG_FILE = "config.json"

DEFAULT_CONFIG = {
    "version": "0.2.0",
    "store_path": ".agent-memory",
    "default_export_format": "md",
    "max_results": 10,
}


def _find_config_path() -> Path | None:
    """Walk up to find .agent-memory/config.json, return its parent or None."""
    p = Path.cwd()
    while p != p.parent:
        cfg = p / STORE_DIR / CONFIG_FILE
        if cfg.exists():
            return cfg
        p = p.parent
    cfg = Path.cwd() / STORE_DIR / CONFIG_FILE
    return cfg if cfg.exists() else None


def load_config() -> dict[str, Any]:
    """Load config from .agent-memory/config.json, merged with defaults."""
    cfg_path = _find_config_path()
    config = dict(DEFAULT_CONFIG)
    if cfg_path and cfg_path.exists():
        try:
            user_cfg = json.loads(cfg_path.read_text())
            if isinstance(user_cfg, dict):
                config.update(user_cfg)
        except (json.JSONDecodeError, OSError):
            pass  # Fall back to defaults on corrupt config
    return config


def save_config(config: dict[str, Any], target_dir: Path | None = None) -> Path:
    """Save config to .agent-memory/config.json. Returns the config file path."""
    if target_dir is None:
        cfg_path = _find_config_path()
        if cfg_path is None:
            target_dir = Path.cwd() / STORE_DIR
        else:
            target_dir = cfg_path.parent
    path = target_dir / CONFIG_FILE
    path.write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n")
    return path


def create_default_config(store_dir: Path) -> Path:
    """Create default config.json in the given store directory."""
    return save_config(dict(DEFAULT_CONFIG), target_dir=store_dir)
