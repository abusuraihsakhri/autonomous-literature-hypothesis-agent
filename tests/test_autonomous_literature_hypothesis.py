#!/usr/bin/env python3
"""
Unit Test Suite for Autonomous Literature Hypothesis Agent.
Covers:
- Concept indexing, normalization, synonym resolution
- Corpus co-occurrence counting, PMI, NPMI calculations
- Jaccard similarity metrics
- Swanson Open Discovery (multi-path hypothesis generation, plausibility scoring)
- Swanson Closed Discovery (mechanistic bridging evaluation)
- Citation Network Construction & PageRank Centrality
- Source Quality Scoring (tiers, citation volume, recency)
- Deduplication Engine (exact ID collision & trigram Jaccard matching)
- Literature Gap Detection (domain coverage, missing terms)
- CLI invocation, JSON output, error handling
"""

import json
import math
import sys
import unittest
from io import StringIO
from unittest.mock import patch

from autonomous_literature_hypothesis import (
    Concept,
    Paper,
    BridgingPath,
    Hypothesis,
    LiteratureCorpus,
    SwansonDiscoveryEngine,
    CitationNetworkAnalysis,
    build_curated_benchmark_corpus,
)
import cli


class TestConceptAndCorpus(unittest.TestCase):
    def setUp(self):
        self.corpus = LiteratureCorpus()
        self.c1 = Concept("C1", "Alzheimer's Disease", "disease", ["AD", "Alzheimer dementia"])
        self.c2 = Concept("C2", "Neuroinflammation", "pathway", ["Brain inflammation"])
        self.c3 = Concept("C3", "Microglia", "cell_type", ["Microglial cells"])
        self.corpus.add_concept(self.c1)
        self.corpus.add_concept(self.c2)
        self.corpus.add_concept(self.c3)

    def test_concept_synonym_resolution(self):
        self.assertEqual(self.corpus.resolve_concept("AD"), self.c1)
        self.assertEqual(self.corpus.resolve_concept("alzheimer's disease"), self.c1)
        self.assertEqual(self.corpus.resolve_concept("Brain inflammation"), self.c2)
        self.assertIsNone(self.corpus.resolve_concept("NonExistentConceptXYZ"))

    def test_paper_concept_auto_detection(self):
        p = Paper("P1", "Targeting Neuroinflammation in Alzheimer's Disease", abstract="Microglia play a key role.")
        self.corpus.add_paper(p)
        self.assertIn("C1", p.concepts)
        self.assertIn("C2", p.concepts)
        self.assertIn("C3", p.concepts)
        self.assertEqual(self.corpus.get_direct_cooccurrence("C1", "C2"), 1)
        self.assertEqual(self.corpus.get_direct_cooccurrence("C1", "C3"), 1)

    def test_pmi_and_npmi_calculations(self):
        # Create 4 papers
        p1 = Paper("P1", "title 1", concepts=["C1", "C2"])
        p2 = Paper("P2", "title 2", concepts=["C1", "C2"])
        p3 = Paper("P3", "title 3", concepts=["C1"])
        p4 = Paper("P4", "title 4", concepts=["C2"])
        for p in [p1, p2, p3, p4]:
            self.corpus.add_paper(p)

        # N = 4, P(C1) = 3/4 = 0.75, P(C2) = 3/4 = 0.75, P(C1, C2) = 2/4 = 0.5
        # PMI = log2(0.5 / (0.75 * 0.75)) = log2(0.5 / 0.5625) = log2(0.8888) = -0.1699
        # NPMI = PMI / (-log2(0.5)) = PMI / 1.0 = -0.1699
        pmi = self.corpus.compute_pmi("C1", "C2")
        npmi = self.corpus.compute_npmi("C1", "C2")
        self.assertAlmostEqual(pmi, math.log2(0.5 / (0.75 * 0.75)), places=4)
        self.assertAlmostEqual(npmi, pmi / (-math.log2(0.5)), places=4)

    def test_pmi_zero_cooccurrence(self):
        p1 = Paper("P1", "title 1", concepts=["C1"])
        p2 = Paper("P2", "title 2", concepts=["C2"])
        self.corpus.add_paper(p1)
        self.corpus.add_paper(p2)
        self.assertEqual(self.corpus.compute_pmi("C1", "C2"), 0.0)
        self.assertEqual(self.corpus.compute_npmi("C1", "C2"), -1.0)

    def test_jaccard_similarity(self):
        p1 = Paper("P1", "title 1", concepts=["C1", "C2"])
        p2 = Paper("P2", "title 2", concepts=["C1"])
        p3 = Paper("P3", "title 3", concepts=["C2"])
        for p in [p1, p2, p3]:
            self.corpus.add_paper(p)
        # Shared papers: 1 (P1), Union: 3 (P1, P2, P3) -> 1/3
        jaccard = self.corpus.jaccard_similarity("C1", "C2")
        self.assertAlmostEqual(jaccard, 1.0 / 3.0, places=4)


class TestSwansonDiscoveryEngine(unittest.TestCase):
    def setUp(self):
        self.corpus = build_curated_benchmark_corpus()
        self.engine = SwansonDiscoveryEngine(self.corpus)

    def test_swanson_open_discovery_raynauds(self):
        # Swanson classic: Raynaud's -> Blood Viscosity / Platelet Aggregation -> Fish Oil
        hypotheses = self.engine.open_discovery("C_RAYNAUDS", filter_prior_direct=True)
        self.assertTrue(len(hypotheses) > 0)
        target_ids = [h.target_concept_id for h in hypotheses]
        self.assertIn("C_FISH_OIL", target_ids)

        fish_oil_hyp = next(h for h in hypotheses if h.target_concept_id == "C_FISH_OIL")
        self.assertEqual(fish_oil_hyp.source_concept_id, "C_RAYNAUDS")
        self.assertEqual(fish_oil_hyp.direct_prior_cooccurrences, 0)
        self.assertEqual(fish_oil_hyp.novelty_score, 1.0)
        self.assertTrue(fish_oil_hyp.plausibility_score > 0.0)
        self.assertTrue(fish_oil_hyp.overall_confidence > 50.0)

        # Check intermediate bridges
        bridge_names = [b.concept_b_name for b in fish_oil_hyp.bridging_paths]
        self.assertTrue("Blood Viscosity" in bridge_names or "Platelet Aggregation" in bridge_names)

    def test_swanson_open_discovery_migraine(self):
        # Swanson classic: Migraine -> Cortical Spreading Depression / Vasoconstriction -> Magnesium
        hypotheses = self.engine.open_discovery("C_MIGRAINE")
        self.assertTrue(len(hypotheses) > 0)
        target_ids = [h.target_concept_id for h in hypotheses]
        self.assertIn("C_MAGNESIUM", target_ids)

    def test_swanson_closed_discovery_connected_pair(self):
        hyp = self.engine.closed_discovery("C_RAYNAUDS", "C_FISH_OIL")
        self.assertEqual(hyp.discovery_mode, "closed")
        self.assertEqual(hyp.source_concept_id, "C_RAYNAUDS")
        self.assertEqual(hyp.target_concept_id, "C_FISH_OIL")
        self.assertTrue(len(hyp.bridging_paths) >= 2)  # Blood viscosity + Platelet aggregation
        self.assertTrue(hyp.overall_confidence > 40.0)
        self.assertIn("Blood Viscosity", hyp.mechanistic_rationale)

    def test_swanson_closed_discovery_unconnected_pair(self):
        # Two unrelated concepts in benchmark (e.g. Migraine and Semaglutide)
        hyp = self.engine.closed_discovery("C_MIGRAINE", "C_SEMAGLUTIDE")
        self.assertEqual(len(hyp.bridging_paths), 0)
        self.assertEqual(hyp.plausibility_score, 0.0)
        self.assertEqual(hyp.overall_confidence, 0.0)
        self.assertIn("No intermediate bridging literature found", hyp.mechanistic_rationale)

    def test_open_discovery_invalid_concept(self):
        with self.assertRaises(ValueError):
            self.engine.open_discovery("INVALID_ID_999")

    def test_closed_discovery_invalid_concept(self):
        with self.assertRaises(ValueError):
            self.engine.closed_discovery("C_RAYNAUDS", "INVALID_TARGET_999")


class TestCitationNetworkAndBibliometrics(unittest.TestCase):
    def setUp(self):
        self.corpus = build_curated_benchmark_corpus()
        self.papers = list(self.corpus.papers.values())

    def test_build_citation_graph(self):
        graph = CitationNetworkAnalysis.build_graph(self.papers)
        self.assertIn("PMID_001", graph)
        # PMID_002 cites PMID_001
        self.assertIn("PMID_001", graph["PMID_002"]["cites"])
        self.assertIn("PMID_002", graph["PMID_001"]["cited_by"])

    def test_pagerank_centrality(self):
        graph = CitationNetworkAnalysis.build_graph(self.papers)
        ranks = CitationNetworkAnalysis.pagerank(graph, damping=0.85, iterations=30)
        self.assertTrue(len(ranks) > 0)
        # PageRank sum across all nodes should equal 1.0
        self.assertAlmostEqual(sum(ranks.values()), 1.0, places=4)
        # Highly cited paper PMID_001 should have higher rank than PMID_002
        self.assertTrue(ranks["PMID_001"] >= ranks["PMID_002"])

    def test_source_quality_top_tier(self):
        p = Paper("P_TOP", "Breakthrough Oncology Trial", year=2025, journal_tier="top", citation_count=150)
        q = CitationNetworkAnalysis.source_quality(p, current_year=2026)
        self.assertEqual(q["quality_tier"], "High Impact")
        self.assertTrue(q["quality_score"] >= 80.0)

    def test_source_quality_low_tier(self):
        p = Paper("P_LOW", "Case Report", year=2000, journal_tier="low", citation_count=1)
        q = CitationNetworkAnalysis.source_quality(p, current_year=2026)
        self.assertEqual(q["quality_tier"], "Low Impact")
        self.assertTrue(q["quality_score"] <= 40.0)

    def test_duplicate_detection_by_doi_collision(self):
        p1 = Paper("P1", "Original Paper Title", doi="10.1038/nature12345")
        p2 = Paper("P2", "Duplicate Paper Title Different Words", doi="10.1038/nature12345")
        dups = CitationNetworkAnalysis.find_duplicates([p1, p2])
        self.assertEqual(len(dups), 1)
        self.assertEqual(dups[0]["original_paper_id"], "P1")
        self.assertEqual(dups[0]["duplicate_paper_id"], "P2")
        self.assertEqual(dups[0]["similarity"], 1.0)

    def test_duplicate_detection_by_title_trigrams(self):
        p1 = Paper("P1", "Effects of dietary fish oil and EPA on blood viscosity in humans")
        p2 = Paper("P2", "Effects of dietary fish oil and EPA on blood viscosity in human subjects")
        dups = CitationNetworkAnalysis.find_duplicates([p1, p2], jaccard_threshold=0.75)
        self.assertEqual(len(dups), 1)
        self.assertEqual(dups[0]["original_paper_id"], "P1")
        self.assertEqual(dups[0]["duplicate_paper_id"], "P2")

    def test_gap_detection(self):
        topics = [
            "blood viscosity digital vasospasm",  # Covered in corpus
            "crispr base editing sickling mutations in beta-globin",  # Not covered
        ]
        gaps = CitationNetworkAnalysis.detect_literature_gaps(self.corpus, topics)
        gap_topics = [g["topic"] for g in gaps]
        self.assertIn("crispr base editing sickling mutations in beta-globin", gap_topics)
        self.assertNotIn("blood viscosity digital vasospasm", gap_topics)


class TestCLIInterface(unittest.TestCase):
    def test_cli_list_concepts(self):
        with patch('sys.stdout', new=StringIO()) as fake_out:
            ret = cli.main(["--list-concepts"])
            self.assertEqual(ret, 0)
            output = fake_out.getvalue()
            self.assertIn("Raynaud's Disease", output)
            self.assertIn("Fish Oil", output)

    def test_cli_list_concepts_json(self):
        with patch('sys.stdout', new=StringIO()) as fake_out:
            ret = cli.main(["--list-concepts", "--json"])
            self.assertEqual(ret, 0)
            data = json.loads(fake_out.getvalue())
            self.assertIsInstance(data, list)
            self.assertTrue(any(c["name"] == "Raynaud's Disease" for c in data))

    def test_cli_open_discovery(self):
        with patch('sys.stdout', new=StringIO()) as fake_out:
            ret = cli.main(["--open", "Raynaud's Disease"])
            self.assertEqual(ret, 0)
            output = fake_out.getvalue()
            self.assertIn("Fish Oil", output)
            self.assertIn("HYPOTHESIS", output)

    def test_cli_open_discovery_json(self):
        with patch('sys.stdout', new=StringIO()) as fake_out:
            ret = cli.main(["--open", "Raynaud's Disease", "--json"])
            self.assertEqual(ret, 0)
            data = json.loads(fake_out.getvalue())
            self.assertIsInstance(data, list)
            self.assertTrue(len(data) > 0)
            self.assertEqual(data[0]["source_concept_id"], "C_RAYNAUDS")

    def test_cli_closed_discovery(self):
        with patch('sys.stdout', new=StringIO()) as fake_out:
            ret = cli.main(["--closed", "Migraine Disorder", "Magnesium"])
            self.assertEqual(ret, 0)
            output = fake_out.getvalue()
            self.assertIn("Cortical Spreading Depression", output)

    def test_cli_pagerank(self):
        with patch('sys.stdout', new=StringIO()) as fake_out:
            ret = cli.main(["--pagerank"])
            self.assertEqual(ret, 0)
            output = fake_out.getvalue()
            self.assertIn("PageRank Centrality", output)

    def test_cli_gaps(self):
        with patch('sys.stdout', new=StringIO()) as fake_out:
            ret = cli.main(["--gaps"])
            self.assertEqual(ret, 0)
            output = fake_out.getvalue()
            self.assertIn("Literature Gap Detection", output)

    def test_cli_duplicates(self):
        with patch('sys.stdout', new=StringIO()) as fake_out:
            ret = cli.main(["--duplicates"])
            self.assertEqual(ret, 0)
            output = fake_out.getvalue()
            self.assertIn("Duplicate Detection Results", output)

    def test_cli_source_quality(self):
        with patch('sys.stdout', new=StringIO()) as fake_out:
            ret = cli.main(["--source-quality", "PMID_001"])
            self.assertEqual(ret, 0)
            output = fake_out.getvalue()
            self.assertIn("Source Quality Assessment", output)

    def test_cli_unknown_concept_error(self):
        with patch('sys.stderr', new=StringIO()) as fake_err:
            ret = cli.main(["--open", "NonExistentConceptABC123"])
            self.assertEqual(ret, 1)

    def test_cli_batch_subcommand(self):
        import tempfile
        import os
        import csv
        with tempfile.TemporaryDirectory() as tmpdir:
            in_csv = os.path.join(tmpdir, "input.csv")
            out_csv = os.path.join(tmpdir, "output.csv")
            with open(in_csv, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["query_id", "query_type", "source_concept", "target_concept", "clinical_domain", "min_npmi", "notes"])
                writer.writerow(["Q1", "open", "Raynaud's Disease", "", "Vascular", "0.0", "Test open"])
                writer.writerow(["Q2", "closed", "Migraine Disorder", "Magnesium", "Neurology", "0.0", "Test closed"])
                writer.writerow(["Q3", "open", "NonExistentConceptXYZ", "", "General", "0.0", "Error case"])

            with patch('sys.stdout', new=StringIO()) as fake_out:
                ret = cli.main(["batch", "-i", in_csv, "-o", out_csv])
                self.assertEqual(ret, 0)
                self.assertIn("Batch processing complete", fake_out.getvalue())

            self.assertTrue(os.path.exists(out_csv))
            with open(out_csv, "r", encoding="utf-8") as f:
                reader = list(csv.DictReader(f))
                self.assertEqual(len(reader), 3)
                self.assertEqual(reader[0]["status"], "SUCCESS")
                self.assertEqual(reader[0]["top_target_concept"], "Fish Oil")
                self.assertEqual(reader[1]["status"], "SUCCESS")
                self.assertEqual(reader[1]["top_target_concept"], "Magnesium")
                self.assertTrue(reader[2]["status"].startswith("ERROR"))

    def test_cli_batch_flags(self):
        import tempfile
        import os
        import csv
        with tempfile.TemporaryDirectory() as tmpdir:
            in_csv = os.path.join(tmpdir, "input.csv")
            out_csv = os.path.join(tmpdir, "output.csv")
            with open(in_csv, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["query_id", "query_type", "source_concept", "target_concept", "clinical_domain", "min_npmi", "notes"])
                writer.writerow(["Q1", "closed", "Pancreatic Ductal Adenocarcinoma", "Curcumin", "Oncology", "0.0", "Test"])

            with patch('sys.stdout', new=StringIO()) as fake_out:
                ret = cli.main(["--batch", "--input", in_csv, "--output", out_csv])
                self.assertEqual(ret, 0)
                self.assertIn("Batch processing complete", fake_out.getvalue())

            self.assertTrue(os.path.exists(out_csv))

    def test_cli_batch_missing_args(self):
        with patch('sys.stderr', new=StringIO()) as fake_err:
            ret = cli.main(["--batch"])
            self.assertEqual(ret, 1)


if __name__ == "__main__":
    unittest.main()
