import unittest

import hmi_page
import knowledge
import line_sim
import main
from knowledge import DEFECT_QUERIES, TfIdfIndex, passages_for_defect, tokenize

# What a quality engineer would accept as the most relevant article for each NEU-CLS class.
EXPECTED_TOP_ARTICLE = {
    "crazing": {"Crazing", "Fatigue (material)"},
    "inclusion": {"Non-metallic inclusions", "Steelmaking", "Continuous casting"},
    "patches": {"Surface finish", "Surface roughness", "Mill scale", "Rust", "Corrosion"},
    "pitted_surface": {"Pitting corrosion", "Corrosion", "Stainless steel"},
    "rolled-in_scale": {"Mill scale", "Pickling (metal)", "Hot working", "Rolling (metalworking)"},
    "scratches": {"Surface finish", "Surface roughness", "Visual inspection", "Shot peening"},
}


class TfIdfTests(unittest.TestCase):
    DOCS = ["pitting corrosion attacks steel surfaces", "the annual report of the company", "corrosion of steel and iron alloys"]

    def test_tokenize_drops_stopwords_and_punctuation(self):
        self.assertEqual(tokenize("The Mill-scale, of steel!"), ["mill", "scale", "steel"])

    def test_ranks_the_document_that_matches_the_query_best(self):
        index = TfIdfIndex(self.DOCS)
        ranked = [i for i, _ in index.search("pitting corrosion", 3)]
        self.assertEqual(ranked[0], 0)
        self.assertNotIn(1, ranked)  # shares no terms with the query

    def test_rare_terms_outweigh_common_ones(self):
        index = TfIdfIndex(self.DOCS)
        self.assertGreater(index.idf["pitting"], index.idf["steel"])

    def test_empty_or_unknown_queries_return_nothing(self):
        index = TfIdfIndex(self.DOCS)
        self.assertEqual(index.search("", 3), [])
        self.assertEqual(index.search("zzzz qqqq", 3), [])

    def test_scores_are_cosine_bounded_and_ordered(self):
        scores = [s for _, s in TfIdfIndex(self.DOCS).search("steel corrosion", 3)]
        self.assertEqual(scores, sorted(scores, reverse=True))
        self.assertTrue(all(0 < s <= 1.0001 for s in scores))


class DefectPassagesTests(unittest.TestCase):
    def test_every_neu_class_has_a_query(self):
        self.assertEqual(set(DEFECT_QUERIES), set(line_sim.DEFECT_CLASSES))

    def test_top_passage_comes_from_a_relevant_article(self):
        for defect, acceptable in EXPECTED_TOP_ARTICLE.items():
            top = passages_for_defect(defect, 3)[0]
            self.assertIn(top["article"], acceptable, f"{defect}: got {top['article']}")

    def test_returns_k_scored_passages_with_source_links(self):
        for defect in DEFECT_QUERIES:
            passages = passages_for_defect(defect, 3)
            self.assertEqual(len(passages), 3, defect)
            self.assertEqual([p["score"] for p in passages], sorted((p["score"] for p in passages), reverse=True))
            for p in passages:
                self.assertTrue(p["preview"])
                self.assertTrue(p["url"].startswith("https://en.wikipedia.org/wiki/"), p["url"])

    def test_history_and_art_sections_are_filtered_out(self):
        passages, _ = knowledge._load()
        sections = {p["section"].lower() for p in passages}
        self.assertFalse(sections & knowledge.SKIP_SECTIONS)
        self.assertGreater(len(passages), 650)  # only a handful of sections are dropped from the 728


class KnowledgeEndpointTests(unittest.TestCase):
    def setUp(self):
        self.client = main.app.test_client()

    def test_returns_passages_for_a_defect(self):
        body = self.client.get("/api/defect-knowledge?defect=pitted_surface").get_json()
        self.assertEqual(body["defect"], "pitted_surface")
        self.assertEqual(len(body["passages"]), 3)
        self.assertIn("not plant procedures", body["source"])

    def test_k_is_clamped_and_validated(self):
        self.assertEqual(len(self.client.get("/api/defect-knowledge?defect=crazing&k=99").get_json()["passages"]), 6)
        self.assertEqual(len(self.client.get("/api/defect-knowledge?defect=crazing&k=0").get_json()["passages"]), 1)
        self.assertEqual(self.client.get("/api/defect-knowledge?defect=crazing&k=abc").status_code, 400)

    def test_rejects_unknown_defects(self):
        response = self.client.get("/api/defect-knowledge?defect=<script>")
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.get_json()["classes"], line_sim.DEFECT_CLASSES)
        self.assertEqual(self.client.get("/api/defect-knowledge").status_code, 400)


class HmiWiringTests(unittest.TestCase):
    def test_review_rows_offer_background_reading_and_escape_server_text(self):
        html = hmi_page.HTML_TEMPLATE
        self.assertIn("toggleKnowledge", html)
        self.assertIn("/api/defect-knowledge", html)
        self.assertIn("const esc =", html)
        # every server-provided passage field is escaped before it reaches innerHTML
        for field in ("p.article", "p.section", "p.preview", "p.url"):
            self.assertIn(f"esc({field})", html, field)


if __name__ == "__main__":
    unittest.main()
