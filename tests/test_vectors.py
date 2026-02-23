"""Tests for vector embedding support (mocked API)."""
import json
import math
import os
import tempfile
import unittest
from unittest.mock import patch, MagicMock

from agent_memory.sdk import Memory
from agent_memory.embeddings import cosine_similarity, VectorStore, get_embeddings


class TestCosineSimilarity(unittest.TestCase):
    def test_identical(self):
        v = [1.0, 2.0, 3.0]
        self.assertAlmostEqual(cosine_similarity(v, v), 1.0)

    def test_orthogonal(self):
        self.assertAlmostEqual(cosine_similarity([1, 0], [0, 1]), 0.0)

    def test_opposite(self):
        self.assertAlmostEqual(cosine_similarity([1, 0], [-1, 0]), -1.0)

    def test_zero_vector(self):
        self.assertEqual(cosine_similarity([0, 0], [1, 2]), 0.0)


class TestVectorSearch(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.mem = Memory(self.tmpdir, config={
            "embedding": {
                "api_base": "http://fake",
                "api_key": "fake-key",
                "model": "test-model",
            }
        })
        self.mem.init()

    def _mock_embeddings(self, texts):
        """Generate simple deterministic embeddings for testing."""
        vectors = []
        for text in texts:
            words = text.lower().split()
            # Simple: hash each word to a dimension
            vec = [0.0] * 16
            for w in words:
                idx = hash(w) % 16
                vec[idx] += 1.0
            # Normalize
            norm = math.sqrt(sum(x*x for x in vec)) or 1.0
            vectors.append([x / norm for x in vec])
        return vectors

    @patch("agent_memory.embeddings.get_embeddings")
    def test_vector_search(self, mock_get_emb):
        # Make get_embeddings return deterministic vectors
        mock_get_emb.side_effect = lambda texts, config: self._mock_embeddings(texts)

        self.mem.add("the cat sat on the mat", tags=["animal"])
        self.mem.add("python programming language", tags=["code"])
        self.mem.add("the dog played in the park", tags=["animal"])

        results = self.mem.search("cat and dog animals", mode="vector")
        self.assertTrue(len(results) > 0)

    @patch("agent_memory.embeddings.get_embeddings")
    def test_rebuild_vectors(self, mock_get_emb):
        mock_get_emb.side_effect = lambda texts, config: self._mock_embeddings(texts)

        # Add with embedding disabled by removing config
        mem_no_emb = Memory(self.tmpdir, config={})
        mem_no_emb.add("memory one")
        mem_no_emb.add("memory two")

        # Now rebuild with embedding enabled
        count = self.mem.rebuild_vectors()
        self.assertEqual(count, 2)

    @patch("agent_memory.embeddings.get_embeddings")
    def test_delete_removes_vector(self, mock_get_emb):
        mock_get_emb.side_effect = lambda texts, config: self._mock_embeddings(texts)

        entry = self.mem.add("test memory")
        vectors = self.mem.vectors._load_vectors()
        self.assertIn(entry["id"], vectors)

        self.mem.delete(entry["id"])
        vectors = self.mem.vectors._load_vectors()
        self.assertNotIn(entry["id"], vectors)

    def test_no_embedding_config(self):
        mem = Memory(self.tmpdir)  # No embedding config
        self.assertFalse(mem.vectors.enabled)

    @patch("agent_memory.embeddings.get_embeddings")
    def test_hybrid_search(self, mock_get_emb):
        mock_get_emb.side_effect = lambda texts, config: self._mock_embeddings(texts)

        self.mem.add("machine learning with python")
        self.mem.add("cooking recipes for dinner")
        self.mem.add("python data science tutorial")

        results = self.mem.search("python programming", mode="hybrid")
        self.assertTrue(len(results) > 0)

    def test_keyword_fallback_without_embeddings(self):
        """Without embedding config, search falls back to keyword."""
        tmpdir2 = tempfile.mkdtemp()
        mem = Memory(tmpdir2)
        mem.init()
        mem.add("hello world test")
        results = mem.search("hello", mode=None)
        self.assertEqual(len(results), 1)


if __name__ == "__main__":
    unittest.main()
