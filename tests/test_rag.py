import unittest

import numpy as np

from analysis.rag import metrics as M
from analysis.rag.chunking import chunk_sections, split_sections, strip_math
from main import app
from rag_demo import load_rag_artifacts


class MetricTests(unittest.TestCase):
    def test_hit_at_k_and_mrr(self):
        ranked = [["a", "b", "c"], ["x", "y", "z"], ["q", "r", "s"]]
        gold = ["b", "x", "missing"]
        self.assertAlmostEqual(M.hit_at_k(ranked, gold, 1), 1 / 3)
        self.assertAlmostEqual(M.hit_at_k(ranked, gold, 2), 2 / 3)
        self.assertAlmostEqual(M.mean_reciprocal_rank(ranked, gold), (0.5 + 1.0 + 0.0) / 3)

    def test_faithfulness_threshold(self):
        self.assertEqual(M.faithfulness([0.9, 0.2, 0.5, 0.49]), 0.5)
        self.assertIsNone(M.faithfulness([]))

    def test_answer_sentences_strip_citations_and_fragments(self):
        sentences = M.answer_sentences("Mill scale is iron oxide [1]. It is removed by pickling [2, 3]. Yes.")
        self.assertEqual(sentences, ["Mill scale is iron oxide.", "It is removed by pickling."])

    def test_premise_windows(self):
        windows = M.premise_windows("One fact here. Two facts here. Three facts here.")
        self.assertIn("One fact here.", windows)
        self.assertIn("One fact here. Two facts here.", windows)
        self.assertIn("One fact here. Two facts here. Three facts here.", windows)
        self.assertEqual(len(windows), len(set(windows)))

    def test_nearest_and_overlap(self):
        points = np.array([[0.0, 0.0], [1.0, 0.0], [5.0, 5.0]])
        self.assertEqual(M.nearest_indices(points, np.array([0.9, 0.0]), 2), [1, 0])
        self.assertEqual(M.neighbor_overlap([1, 2, 3], [3, 4, 1]), 2)


class ChunkingTests(unittest.TestCase):
    EXTRACT = (
        "Mill scale is the flaky surface of hot rolled steel. " * 12 + "\n\n"
        "== Removal ==\n" + "Pickling in acid dissolves the oxide layer quickly. " * 10 + "\n\n"
        "== References ==\nSmith, J. (2001). Steel. Some Press.\n"
    )

    def test_reference_sections_are_dropped(self):
        titles = [t for t, _ in split_sections(self.EXTRACT)]
        self.assertEqual(titles, ["Introduction", "Removal"])

    def test_chunks_respect_sections_and_size(self):
        passages = chunk_sections(split_sections(self.EXTRACT), target_words=40, max_words=60, min_words=10)
        self.assertTrue(all(p.word_count <= 60 for p in passages))
        self.assertEqual({p.section for p in passages}, {"Introduction", "Removal"})
        self.assertFalse(any("Pickling" in p.text for p in passages if p.section == "Introduction"))

    def test_strip_math(self):
        self.assertEqual(strip_math("Cpk is {\\displaystyle {\\frac {a}{b}}} large"), "Cpk is large")


class RagArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = load_rag_artifacts()

    def test_samples_point_at_real_passages(self):
        ids = self.data["map"]["ids"]
        top_k = self.data["samples"]["top_k"]
        self.assertGreaterEqual(len(self.data["samples"]["samples"]), 5)
        for sample in self.data["samples"]["samples"]:
            self.assertEqual(len(sample["retrieved"]), top_k)
            sims = [r["similarity"] for r in sample["retrieved"]]
            self.assertEqual(sims, sorted(sims, reverse=True))
            for r in sample["retrieved"]:
                self.assertEqual(ids[r["index"]], r["id"])

    def test_screen_overlap_matches_published_neighbours(self):
        points = np.array([p[:5] for p in self.data["map"]["points"]])
        norm = self.data["map"]["projection"]["normalisation_3d"]
        self.assertAlmostEqual(np.abs(points[:, :3]).max(), 1.0, places=3)
        self.assertTrue(norm["scale"] > 0)
        for sample in self.data["samples"]["samples"]:
            real = [r["index"] for r in sample["retrieved"]]
            self.assertEqual(sample["overlap_3d"], M.neighbor_overlap(real, sample["nearest_in_3d"]))
            self.assertEqual(sample["overlap_2d"], M.neighbor_overlap(real, sample["nearest_in_2d"]))

    def test_metrics_are_probabilities(self):
        for method in self.data["retrieval"]["methods"]:
            for key in ("hit_at_1", "hit_at_4", "mrr_at_10", "curated_article_hit_at_5"):
                self.assertGreaterEqual(method[key], 0.0)
                self.assertLessEqual(method[key], 1.0)
        self.assertEqual(sum(m["selected"] for m in self.data["retrieval"]["methods"]), 1)
        gen = self.data["generation"]
        for block in (gen["rag"], gen["closed_book"]):
            self.assertLessEqual(block["sentence_support_rate"], 1.0)
        self.assertIn(gen["selected_config"], gen["by_config"])
        best = max(gen["by_config"].values(), key=lambda c: c["mean_faithfulness"])
        self.assertEqual(f"{best['generator']}/{best['prompt']}", gen["selected_config"])
        self.assertEqual(set(gen["closed_book_by_generator"]), set(gen["generators"]))


class RagRouteTests(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()

    def test_page(self):
        response = self.client.get("/rag")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Steel QC knowledge assistant", response.data)
        response.close()

    def test_static_assets(self):
        for asset in ("rag.js", "rag3d.js", "rag.css"):
            response = self.client.get(f"/rag/static/{asset}")
            self.assertEqual(response.status_code, 200, asset)
            response.close()

    def test_api(self):
        response = self.client.get("/api/rag")
        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        for key in ("summary", "map", "samples", "retrieval", "generation", "fidelity", "corpus"):
            self.assertIn(key, payload)


if __name__ == "__main__":
    unittest.main()
