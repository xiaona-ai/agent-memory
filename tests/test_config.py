"""Tests for configuration file support."""
import json
import os
import tempfile
import unittest
from pathlib import Path

# Ensure we can import the package
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from agent_memory.config import load_config, save_config, create_default_config, DEFAULT_CONFIG
from agent_memory import store


class TestConfig(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.orig_cwd = os.getcwd()
        os.chdir(self.tmpdir)

    def tearDown(self):
        os.chdir(self.orig_cwd)

    def test_default_config_values(self):
        """load_config returns defaults when no config file exists."""
        config = load_config()
        self.assertEqual(config["store_path"], ".agent-memory")
        self.assertEqual(config["default_export_format"], "md")
        self.assertEqual(config["max_results"], 10)

    def test_init_creates_config(self):
        """init_store creates config.json with default values."""
        store.init_store()
        cfg_path = Path(self.tmpdir) / ".agent-memory" / "config.json"
        self.assertTrue(cfg_path.exists())
        cfg = json.loads(cfg_path.read_text())
        self.assertEqual(cfg["default_export_format"], "md")
        self.assertEqual(cfg["max_results"], 10)
        self.assertEqual(cfg["store_path"], ".agent-memory")

    def test_custom_config_is_loaded(self):
        """Custom values in config.json override defaults."""
        store.init_store()
        cfg_path = Path(self.tmpdir) / ".agent-memory" / "config.json"
        cfg = json.loads(cfg_path.read_text())
        cfg["max_results"] = 5
        cfg["default_export_format"] = "json"
        cfg_path.write_text(json.dumps(cfg, indent=2))

        config = load_config()
        self.assertEqual(config["max_results"], 5)
        self.assertEqual(config["default_export_format"], "json")

    def test_corrupt_config_falls_back(self):
        """Corrupt config.json falls back to defaults gracefully."""
        store.init_store()
        cfg_path = Path(self.tmpdir) / ".agent-memory" / "config.json"
        cfg_path.write_text("NOT VALID JSON!!!")

        config = load_config()
        self.assertEqual(config["max_results"], 10)

    def test_save_config(self):
        """save_config writes valid JSON."""
        d = Path(self.tmpdir) / ".agent-memory"
        d.mkdir()
        custom = {"version": "0.2.0", "max_results": 42}
        path = save_config(custom, target_dir=d)
        loaded = json.loads(path.read_text())
        self.assertEqual(loaded["max_results"], 42)

    def test_search_uses_config_max_results(self):
        """search_memories respects max_results from config."""
        store.init_store()
        # Set max_results to 2
        cfg_path = Path(self.tmpdir) / ".agent-memory" / "config.json"
        cfg = json.loads(cfg_path.read_text())
        cfg["max_results"] = 2
        cfg_path.write_text(json.dumps(cfg))

        # Add 5 memories, some with the keyword
        store.add_memory("apple keyword match")
        store.add_memory("banana keyword match")
        store.add_memory("cherry keyword match")
        store.add_memory("unrelated stuff here")
        store.add_memory("more unrelated content")

        results = store.search_memories("keyword")
        self.assertLessEqual(len(results), 2)

    def test_export_uses_config_format(self):
        """export_memories uses default_export_format from config."""
        store.init_store()
        store.add_memory("hello world")

        # Default is md
        output = store.export_memories()
        self.assertIn("# Agent Memory Export", output)

        # Change to json
        cfg_path = Path(self.tmpdir) / ".agent-memory" / "config.json"
        cfg = json.loads(cfg_path.read_text())
        cfg["default_export_format"] = "json"
        cfg_path.write_text(json.dumps(cfg))

        output = store.export_memories()
        parsed = json.loads(output)
        self.assertIsInstance(parsed, list)

    def test_config_command(self):
        """agent-memory config shows current configuration."""
        store.init_store()
        # Just verify load_config works after init
        config = load_config()
        self.assertIn("version", config)
        self.assertIn("store_path", config)


if __name__ == "__main__":
    unittest.main()
