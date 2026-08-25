#!/usr/bin/env python3
"""
Autonomous Literature Hypothesis Agent (Literature-Based Discovery Engine).

Implements:
- Swanson ABC Discovery Model (Open & Closed Literature-Based Discovery)
- Biomedical Concept Co-occurrence, PMI (Pointwise Mutual Information), NPMI
- Transitive Multi-Path Hypothesis Generation & Plausibility Scoring
- Citation Graph Analysis: PageRank Centrality, H-index, Co-citation Coupling
- Source Quality Assessment (Journal Tier, Citation Velocity, Recency)
- Deduplication via Trigram Jaccard Similarity & Unique Persistent Identifiers
- Biomedical Literature Gap Detection against Knowledge Domains

Pure Python Standard Library (no external dependencies required).
"""

from __future__ import annotations
import math
import re
from collections import defaultdict, Counter
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Set, Tuple, Optional, Any


@dataclass
class Concept:
    """Biomedical concept or entity."""
    concept_id: str
    name: str
    category: str  # e.g., 'disease', 'drug_compound', 'pathway', 'gene_protein', 'phenotype'
    synonyms: List[str] = field(default_factory=list)
    description: str = ""

    def matches(self, term: str) -> bool:
        """Check if concept name or any synonym matches the given term."""
        norm_term = term.strip().lower()
        if self.name.lower() == norm_term:
            return True
        return any(syn.lower() == norm_term for syn in self.synonyms)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Paper:
    """Biomedical literature record / publication."""
    paper_id: str
    title: str
    abstract: str = ""
    year: int = 2024
    journal_tier: str = "mid"  # 'top' (e.g. NEJM, Nature), 'good', 'mid', 'low'
    citation_count: int = 0
    doi: str = ""
    pmid: str = ""
    concepts: List[str] = field(default_factory=list)  # concept_ids
    cites: List[str] = field(default_factory=list)     # paper_ids cited by this paper

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class BridgingPath:
    """An intermediate bridging path A -> B -> C."""
    concept_b_id: str
    concept_b_name: str
    concept_b_category: str
    npmi_ab: float
    npmi_bc: float
    path_score: float
    ab_cooccurrences: int
    bc_cooccurrences: int
    supporting_ab_papers: List[str]
    supporting_bc_papers: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Hypothesis:
    """Literature-Based Discovery Generated Hypothesis."""
    hypothesis_id: str
    source_concept_id: str
    source_concept_name: str
    target_concept_id: str
    target_concept_name: str
    discovery_mode: str  # 'open' or 'closed'
    bridging_paths: List[BridgingPath] = field(default_factory=list)
    plausibility_score: float = 0.0      # 0.0 - 1.0 (aggregate multi-path strength)
    novelty_score: float = 1.0           # 0.0 - 1.0 (1.0 = completely unlinked in direct literature)
    direct_prior_cooccurrences: int = 0
    overall_confidence: float = 0.0      # Composite discovery score (0.0 - 100.0)
    mechanistic_rationale: str = ""
    recommendation: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "hypothesis_id": self.hypothesis_id,
            "source_concept_id": self.source_concept_id,
            "source_concept_name": self.source_concept_name,
            "target_concept_id": self.target_concept_id,
            "target_concept_name": self.target_concept_name,
            "discovery_mode": self.discovery_mode,
            "plausibility_score": round(self.plausibility_score, 4),
            "novelty_score": round(self.novelty_score, 4),
            "direct_prior_cooccurrences": self.direct_prior_cooccurrences,
            "overall_confidence": round(self.overall_confidence, 2),
            "bridging_paths": [p.to_dict() for p in self.bridging_paths],
            "mechanistic_rationale": self.mechanistic_rationale,
            "recommendation": self.recommendation,
        }


class LiteratureCorpus:
    """Corpus of biomedical papers, indexing concepts, citations, and co-occurrences."""

    def __init__(self):
        self.papers: Dict[str, Paper] = {}
        self.concepts: Dict[str, Concept] = {}
        self.concept_name_map: Dict[str, str] = {}  # lower name/synonym -> concept_id
        # Inverted index: concept_id -> set of paper_ids
        self.concept_to_papers: Dict[str, Set[str]] = defaultdict(set)
        # Co-occurrence counts: tuple(sorted([id1, id2])) -> int
        self.cooccurrences: Dict[Tuple[str, str], int] = defaultdict(int)

    def add_concept(self, concept: Concept) -> None:
        """Register a concept into the index."""
        self.concepts[concept.concept_id] = concept
        self.concept_name_map[concept.name.strip().lower()] = concept.concept_id
        for syn in concept.synonyms:
            self.concept_name_map[syn.strip().lower()] = concept.concept_id

    def add_paper(self, paper: Paper) -> None:
        """Add a paper to the corpus and update inverted indices."""
        self.papers[paper.paper_id] = paper
        # Auto-detect concepts in text if concepts list is empty
        detected_concepts = set(paper.concepts)
        if not detected_concepts:
            text_to_search = f"{paper.title} {paper.abstract}".lower()
            for name, cid in self.concept_name_map.items():
                pattern = r'\b' + re.escape(name) + r'\b'
                if re.search(pattern, text_to_search):
                    detected_concepts.add(cid)
            paper.concepts = sorted(detected_concepts)

        # Update inverted index
        for cid in detected_concepts:
            self.concept_to_papers[cid].add(paper.paper_id)

        # Update pairwise co-occurrences
        c_list = sorted(list(detected_concepts))
        for i in range(len(c_list)):
            for j in range(i + 1, len(c_list)):
                pair = (c_list[i], c_list[j])
                self.cooccurrences[pair] += 1

    def resolve_concept(self, query: str) -> Optional[Concept]:
        """Find a concept by ID, name, or synonym."""
        query_norm = query.strip().lower()
        if query in self.concepts:
            return self.concepts[query]
        if query_norm in self.concept_name_map:
            cid = self.concept_name_map[query_norm]
            return self.concepts[cid]
        # Partial match
        for name, cid in self.concept_name_map.items():
            if query_norm in name or name in query_norm:
                return self.concepts[cid]
        return None

    def get_direct_cooccurrence(self, cid1: str, cid2: str) -> int:
        """Get direct paper co-occurrence count between two concepts."""
        pair = tuple(sorted([cid1, cid2]))
        return self.cooccurrences.get(pair, 0)

    def get_shared_papers(self, cid1: str, cid2: str) -> List[str]:
        """Return list of paper IDs mentioning both concepts."""
        p1 = self.concept_to_papers.get(cid1, set())
        p2 = self.concept_to_papers.get(cid2, set())
        return sorted(list(p1 & p2))

    def compute_pmi(self, cid1: str, cid2: str) -> float:
        """
        Pointwise Mutual Information:
        PMI(A, B) = log2( P(A, B) / (P(A) * P(B)) )
        where P(A) = count(A)/N, P(A,B) = count(A,B)/N
        """
        n = len(self.papers)
        if n == 0:
            return 0.0
        c_a = len(self.concept_to_papers.get(cid1, set()))
        c_b = len(self.concept_to_papers.get(cid2, set()))
        c_ab = self.get_direct_cooccurrence(cid1, cid2)

        if c_ab == 0 or c_a == 0 or c_b == 0:
            return 0.0

        p_a = c_a / n
        p_b = c_b / n
        p_ab = c_ab / n

        return math.log2(p_ab / (p_a * p_b))

    def compute_npmi(self, cid1: str, cid2: str) -> float:
        """
        Normalized Pointwise Mutual Information (NPMI):
        NPMI(A, B) = PMI(A, B) / ( -log2(P(A, B)) )
        Ranges from -1 (never co-occur) to +1 (perfect co-occurrence).
        """
        n = len(self.papers)
        if n == 0:
            return 0.0
        c_ab = self.get_direct_cooccurrence(cid1, cid2)
        if c_ab == 0:
            return -1.0

        p_ab = c_ab / n
        pmi = self.compute_pmi(cid1, cid2)
        denom = -math.log2(p_ab)
        if denom == 0:
            return 1.0
        npmi = pmi / denom
        return max(-1.0, min(1.0, npmi))

    def jaccard_similarity(self, cid1: str, cid2: str) -> float:
        """Jaccard similarity between paper sets of two concepts."""
        p1 = self.concept_to_papers.get(cid1, set())
        p2 = self.concept_to_papers.get(cid2, set())
        union = len(p1 | p2)
        if union == 0:
            return 0.0
        return len(p1 & p2) / union


class SwansonDiscoveryEngine:
    """
    Swanson ABC Literature-Based Discovery Engine.
    Implements Open Discovery (A -> ? -> C) and Closed Discovery (A -> B? -> C).
    """

    def __init__(self, corpus: LiteratureCorpus):
        self.corpus = corpus

    def open_discovery(
        self,
        source_concept_id: str,
        target_categories: Optional[List[str]] = None,
        intermediate_categories: Optional[List[str]] = None,
        min_npmi: float = 0.0,
        max_b_concepts: int = 10,
        max_hypotheses: int = 10,
        filter_prior_direct: bool = True
    ) -> List[Hypothesis]:
        """
        Open Swanson Discovery:
        Given source concept A, identify intermediate B concepts,
        then discover target C concepts with strong transitive support (A-B and B-C)
        and low or zero direct prior literature linking A and C.
        """
        source = self.corpus.concepts.get(source_concept_id)
        if not source:
            raise ValueError(f"Source concept '{source_concept_id}' not found in corpus.")

        # Step 1: Find candidate B concepts co-occurring with A
        b_candidates: List[Tuple[str, float, int]] = []
        for cid, concept in self.corpus.concepts.items():
            if cid == source_concept_id:
                continue
            if intermediate_categories and concept.category not in intermediate_categories:
                continue
            cooc = self.corpus.get_direct_cooccurrence(source_concept_id, cid)
            if cooc > 0:
                npmi_ab = self.corpus.compute_npmi(source_concept_id, cid)
                if npmi_ab >= min_npmi:
                    b_candidates.append((cid, npmi_ab, cooc))

        # Rank B concepts by NPMI
        b_candidates.sort(key=lambda x: x[1], reverse=True)
        b_selected = b_candidates[:max_b_concepts]

        # Step 2: From B concepts, discover candidate C concepts
        # Map: target_c_id -> List of BridgingPath
        c_paths: Dict[str, List[BridgingPath]] = defaultdict(list)

        for b_id, npmi_ab, cooc_ab in b_selected:
            b_concept = self.corpus.concepts[b_id]
            for c_id, c_concept in self.corpus.concepts.items():
                if c_id == source_concept_id or c_id == b_id:
                    continue
                if target_categories and c_concept.category not in target_categories:
                    continue
                cooc_bc = self.corpus.get_direct_cooccurrence(b_id, c_id)
                if cooc_bc > 0:
                    npmi_bc = self.corpus.compute_npmi(b_id, c_id)
                    if npmi_bc >= min_npmi:
                        # Path score: geometric mean of NPMI components
                        # NPMI scaled to [0, 1] for positive associations
                        norm_ab = max(0.0, npmi_ab)
                        norm_bc = max(0.0, npmi_bc)
                        path_score = math.sqrt(norm_ab * norm_bc)

                        ab_papers = self.corpus.get_shared_papers(source_concept_id, b_id)
                        bc_papers = self.corpus.get_shared_papers(b_id, c_id)

                        path = BridgingPath(
                            concept_b_id=b_id,
                            concept_b_name=b_concept.name,
                            concept_b_category=b_concept.category,
                            npmi_ab=round(npmi_ab, 4),
                            npmi_bc=round(npmi_bc, 4),
                            path_score=round(path_score, 4),
                            ab_cooccurrences=cooc_ab,
                            bc_cooccurrences=cooc_bc,
                            supporting_ab_papers=ab_papers,
                            supporting_bc_papers=bc_papers,
                        )
                        c_paths[c_id].append(path)

        # Step 3: Synthesize and Score Hypotheses
        hypotheses: List[Hypothesis] = []
        hypo_idx = 1

        for c_id, paths in c_paths.items():
            direct_cooc = self.corpus.get_direct_cooccurrence(source_concept_id, c_id)
            if filter_prior_direct and direct_cooc > 0:
                continue

            c_concept = self.corpus.concepts[c_id]
            # Novelty: decreases if direct co-occurrence exists
            novelty = 1.0 / (1.0 + direct_cooc)

            # Plausibility: aggregate independent paths using noisy-OR / bounded sum
            # Sum of path scores with diminishing returns
            paths.sort(key=lambda p: p.path_score, reverse=True)
            plausibility = 1.0 - math.prod(1.0 - min(0.99, p.path_score) for p in paths)

            # Overall confidence score: 0 to 100
            # Combines plausibility, novelty, and number of multi-path bridges
            path_diversity_bonus = min(1.25, 1.0 + 0.08 * (len(paths) - 1))
            raw_confidence = (plausibility * 70.0 + novelty * 30.0) * path_diversity_bonus
            confidence = min(99.9, max(5.0, raw_confidence))

            # Mechanistic rationale synthesis
            top_bridges = [f"{p.concept_b_name} ({p.concept_b_category})" for p in paths[:3]]
            rationale = (
                f"Candidate therapeutic/mechanistic connection between '{source.name}' and '{c_concept.name}' "
                f"mediated via intermediate biological pathways/phenotypes: {', '.join(top_bridges)}."
            )
            recommendation = (
                f"Validate transitive link experimentally: verify if modulating '{source.name}' affects "
                f"'{top_bridges[0]}' and downstream '{c_concept.name}'."
            )

            hypo = Hypothesis(
                hypothesis_id=f"HYP-OPEN-{hypo_idx:03d}",
                source_concept_id=source.concept_id,
                source_concept_name=source.name,
                target_concept_id=c_concept.concept_id,
                target_concept_name=c_concept.name,
                discovery_mode="open",
                bridging_paths=paths,
                plausibility_score=plausibility,
                novelty_score=novelty,
                direct_prior_cooccurrences=direct_cooc,
                overall_confidence=confidence,
                mechanistic_rationale=rationale,
                recommendation=recommendation,
            )
            hypotheses.append(hypo)
            hypo_idx += 1

        # Sort hypotheses by overall confidence
        hypotheses.sort(key=lambda h: h.overall_confidence, reverse=True)
        return hypotheses[:max_hypotheses]

    def closed_discovery(
        self,
        source_concept_id: str,
        target_concept_id: str,
        min_npmi: float = -1.0
    ) -> Hypothesis:
        """
        Closed Swanson Discovery:
        Given two concepts A and C, find all intermediate bridging concepts B,
        and evaluate the biological/mechanistic plausibility of their transitive link.
        """
        source = self.corpus.resolve_concept(source_concept_id)
        target = self.corpus.resolve_concept(target_concept_id)

        if not source:
            raise ValueError(f"Source concept '{source_concept_id}' not found.")
        if not target:
            raise ValueError(f"Target concept '{target_concept_id}' not found.")

        bridging_paths: List[BridgingPath] = []
        direct_cooc = self.corpus.get_direct_cooccurrence(source.concept_id, target.concept_id)

        for b_id, b_concept in self.corpus.concepts.items():
            if b_id == source.concept_id or b_id == target.concept_id:
                continue
            cooc_ab = self.corpus.get_direct_cooccurrence(source.concept_id, b_id)
            cooc_bc = self.corpus.get_direct_cooccurrence(b_id, target.concept_id)

            if cooc_ab > 0 and cooc_bc > 0:
                npmi_ab = self.corpus.compute_npmi(source.concept_id, b_id)
                npmi_bc = self.corpus.compute_npmi(b_id, target.concept_id)

                if npmi_ab >= min_npmi and npmi_bc >= min_npmi:
                    norm_ab = max(0.0, npmi_ab)
                    norm_bc = max(0.0, npmi_bc)
                    path_score = math.sqrt(norm_ab * norm_bc)

                    ab_papers = self.corpus.get_shared_papers(source.concept_id, b_id)
                    bc_papers = self.corpus.get_shared_papers(b_id, target.concept_id)

                    path = BridgingPath(
                        concept_b_id=b_id,
                        concept_b_name=b_concept.name,
                        concept_b_category=b_concept.category,
                        npmi_ab=round(npmi_ab, 4),
                        npmi_bc=round(npmi_bc, 4),
                        path_score=round(path_score, 4),
                        ab_cooccurrences=cooc_ab,
                        bc_cooccurrences=cooc_bc,
                        supporting_ab_papers=ab_papers,
                        supporting_bc_papers=bc_papers,
                    )
                    bridging_paths.append(path)

        bridging_paths.sort(key=lambda p: p.path_score, reverse=True)

        if bridging_paths:
            plausibility = 1.0 - math.prod(1.0 - min(0.99, p.path_score) for p in bridging_paths)
            novelty = 1.0 / (1.0 + direct_cooc)
            path_bonus = min(1.25, 1.0 + 0.08 * (len(bridging_paths) - 1))
            confidence = min(99.9, max(5.0, (plausibility * 70.0 + novelty * 30.0) * path_bonus))
            top_b = [p.concept_b_name for p in bridging_paths[:3]]
            rationale = (
                f"Closed discovery evaluated {len(bridging_paths)} bridging pathways between "
                f"'{source.name}' and '{target.name}'. Key connecting mediators: {', '.join(top_b)}. "
                f"Prior direct co-mentions: {direct_cooc}."
            )
            recommendation = (
                f"Focus translational inquiry on top intermediate mediator '{top_b[0]}' "
                f"to validate the transitive relationship."
            )
        else:
            plausibility = 0.0
            novelty = 1.0 / (1.0 + direct_cooc)
            confidence = 0.0
            rationale = f"No intermediate bridging literature found connecting '{source.name}' and '{target.name}'."
            recommendation = "Broaden literature corpus or query alternative synonyms."

        return Hypothesis(
            hypothesis_id=f"HYP-CLOSED-{source.concept_id}-{target.concept_id}",
            source_concept_id=source.concept_id,
            source_concept_name=source.name,
            target_concept_id=target.concept_id,
            target_concept_name=target.name,
            discovery_mode="closed",
            bridging_paths=bridging_paths,
            plausibility_score=plausibility,
            novelty_score=novelty,
            direct_prior_cooccurrences=direct_cooc,
            overall_confidence=confidence,
            mechanistic_rationale=rationale,
            recommendation=recommendation,
        )


class CitationNetworkAnalysis:
    """Bibliometric network analysis, PageRank, duplicate detection, and gap analysis."""

    @staticmethod
    def build_graph(papers: List[Paper]) -> Dict[str, Dict[str, List[str]]]:
        """Build directed citation adjacency list."""
        g = defaultdict(lambda: {"cites": set(), "cited_by": set()})
        for p in papers:
            g[p.paper_id]["cites"].update(p.cites)
            for target in p.cites:
                g[target]["cited_by"].add(p.paper_id)
        return {
            k: {"cites": sorted(list(v["cites"])), "cited_by": sorted(list(v["cited_by"]))}
            for k, v in g.items()
        }

    @staticmethod
    def pagerank(graph: Dict[str, Dict[str, List[str]]], damping: float = 0.85, iterations: int = 50) -> Dict[str, float]:
        """Compute PageRank centrality for papers in the citation graph."""
        nodes = list(graph.keys())
        if not nodes:
            return {}
        n = len(nodes)
        rank = {node: 1.0 / n for node in nodes}

        for _ in range(iterations):
            new_rank = {node: (1.0 - damping) / n for node in nodes}
            for src in nodes:
                out_links = [tgt for tgt in graph[src]["cites"] if tgt in graph]
                if not out_links:
                    for tgt in nodes:
                        new_rank[tgt] += damping * (rank[src] / n)
                else:
                    share = rank[src] / len(out_links)
                    for tgt in out_links:
                        new_rank[tgt] += damping * share
            rank = new_rank

        total = sum(rank.values()) or 1.0
        return {k: round(v / total, 6) for k, v in sorted(rank.items(), key=lambda kv: kv[1], reverse=True)}

    @staticmethod
    def source_quality(paper: Paper, current_year: int = 2026) -> Dict[str, Any]:
        """Evaluate paper source credibility and quality score (0 - 100)."""
        tier_scores = {"top": 1.0, "good": 0.75, "mid": 0.5, "low": 0.25}
        tier_pts = tier_scores.get(paper.journal_tier.lower(), 0.5)

        # Citation velocity / count score
        cite_pts = min(1.0, paper.citation_count / 100.0)

        # Recency score
        age = max(0, current_year - paper.year)
        if age <= 3:
            recency_pts = 1.0
        elif age <= 8:
            recency_pts = 0.75
        elif age <= 15:
            recency_pts = 0.5
        else:
            recency_pts = 0.25

        score = round(100.0 * (0.45 * tier_pts + 0.35 * cite_pts + 0.20 * recency_pts), 2)
        tier_label = "High Impact" if score >= 75 else ("Moderate Impact" if score >= 50 else "Low Impact")

        return {
            "paper_id": paper.paper_id,
            "quality_score": score,
            "quality_tier": tier_label,
            "journal_tier": paper.journal_tier,
            "citation_count": paper.citation_count,
            "age_years": age,
        }

    @staticmethod
    def _trigrams(text: str) -> Set[str]:
        norm = "".join(c.lower() if c.isalnum() else " " for c in text).split()
        joined = "".join(norm)
        if len(joined) < 3:
            return {joined}
        return {joined[i:i + 3] for i in range(len(joined) - 2)}

    @classmethod
    def find_duplicates(cls, papers: List[Paper], jaccard_threshold: float = 0.85) -> List[Dict[str, Any]]:
        """Identify duplicate papers by exact ID match (DOI/PMID) or title trigram similarity."""
        seen_ids: Dict[str, str] = {}
        seen_titles: Dict[str, Tuple[str, Set[str]]] = {}
        duplicates: List[Dict[str, Any]] = []

        for p in papers:
            # Check DOI/PMID
            keys = [k for k in [p.doi, p.pmid] if k]
            matched_id = False
            for k in keys:
                if k in seen_ids:
                    duplicates.append({
                        "original_paper_id": seen_ids[k],
                        "duplicate_paper_id": p.paper_id,
                        "reason": f"Exact persistent identifier match ({k})",
                        "similarity": 1.0,
                    })
                    matched_id = True
                    break
            if matched_id:
                continue

            # Check title similarity
            tri = cls._trigrams(p.title)
            title_matched = False
            for orig_id, (orig_title, orig_tri) in seen_titles.items():
                union = len(tri | orig_tri)
                sim = len(tri & orig_tri) / union if union else 0.0
                if sim >= jaccard_threshold:
                    duplicates.append({
                        "original_paper_id": orig_id,
                        "duplicate_paper_id": p.paper_id,
                        "reason": f"Title trigram similarity ({sim:.2f})",
                        "similarity": round(sim, 3),
                    })
                    title_matched = True
                    break

            if not title_matched:
                for k in keys:
                    seen_ids[k] = p.paper_id
                seen_titles[p.paper_id] = (p.title, tri)

        return duplicates

    @staticmethod
    def detect_literature_gaps(corpus: LiteratureCorpus, domain_topics: List[str]) -> List[Dict[str, Any]]:
        """Detect under-explored literature domains/topics."""
        stopwords = {"of", "the", "in", "and", "to", "a", "for", "with", "on", "by", "is", "at"}
        corpus_words = set()
        for p in corpus.papers.values():
            words = {w.lower().strip(".,;:()[]") for w in f"{p.title} {p.abstract}".split()}
            corpus_words |= (words - stopwords)

        gaps = []
        for topic in domain_topics:
            topic_terms = [w.lower().strip(".,;") for w in topic.split() if w.lower() not in stopwords]
            if not topic_terms:
                continue
            matched = sum(1 for t in topic_terms if t in corpus_words)
            coverage = matched / len(topic_terms)
            if coverage < 0.50:
                gaps.append({
                    "topic": topic,
                    "term_coverage": round(coverage, 2),
                    "matched_terms": [t for t in topic_terms if t in corpus_words],
                    "missing_terms": [t for t in topic_terms if t not in corpus_words],
                    "status": "High Priority Gap" if coverage < 0.25 else "Moderate Gap",
                    "recommendation": f"Expand literature indexing around '{topic}' to bridge discovery pathways.",
                })

        return gaps


def build_curated_benchmark_corpus() -> LiteratureCorpus:
    """Build the curated biomedical literature discovery benchmark knowledge base."""
    corpus = LiteratureCorpus()

    # Define Concepts across several domains
    concepts = [
        # Domain 1: Raynaud's & Fish Oil (Swanson 1986 Benchmark)
        Concept("C_RAYNAUDS", "Raynaud's Disease", "disease", ["Raynaud phenomenon", "Raynaud syndrome"]),
        Concept("C_BLOOD_VISCOSITY", "Blood Viscosity", "phenotype", ["Hyperviscosity", "Blood rheology"]),
        Concept("C_PLATELET_AGGR", "Platelet Aggregation", "pathway", ["Platelet activation", "Thrombocyte aggregation"]),
        Concept("C_FISH_OIL", "Fish Oil", "drug_compound", ["EPA", "Eicosapentaenoic acid", "Omega-3 fatty acids"]),
        Concept("C_VASOSPASM", "Digital Vasospasm", "phenotype", ["Peripheral vasospasm"]),

        # Domain 2: Migraine & Magnesium (Swanson 1988 Benchmark)
        Concept("C_MIGRAINE", "Migraine Disorder", "disease", ["Migraine headache", "Hemicrania"]),
        Concept("C_SPREADING_DEP", "Cortical Spreading Depression", "pathway", ["Spreading depression", "CSD"]),
        Concept("C_MAGNESIUM", "Magnesium", "drug_compound", ["Mg2+", "Magnesium sulfate", "Hypomagnesemia"]),
        Concept("C_VASOCONSTRICTION", "Cerebral Vasoconstriction", "phenotype", ["Cerebral vasospasm"]),

        # Domain 3: Pancreatic Cancer & Curcumin
        Concept("C_PDAC", "Pancreatic Ductal Adenocarcinoma", "disease", ["Pancreatic cancer", "PDAC"]),
        Concept("C_NFKB", "NF-kB Pathway", "pathway", ["NF-kappaB", "Nuclear factor kappa B"]),
        Concept("C_CURCUMIN", "Curcumin", "drug_compound", ["Diferuloylmethane", "Turmeric extract"]),
        Concept("C_GEMCITABINE", "Gemcitabine", "drug_compound", ["Gemzar", "dFdC"]),
        Concept("C_CHEMORESISTANCE", "Chemoresistance", "phenotype", ["Drug resistance"]),

        # Domain 4: Metformin & T-Cell Immunotherapy
        Concept("C_TCELL_EXHAUSTION", "CD8+ T-Cell Exhaustion", "phenotype", ["T-cell dysfunction", "Immune exhaustion"]),
        Concept("C_AMPK", "AMPK Pathway", "pathway", ["AMP-activated protein kinase", "PRKAA1"]),
        Concept("C_METFORMIN", "Metformin", "drug_compound", ["Glucophage", "Dimethylbiguanide"]),
        Concept("C_MELANOMA", "Malignant Melanoma", "disease", ["Cutaneous melanoma"]),

        # Domain 5: Alzheimer's & Periodontal Pathogens
        Concept("C_ALZHEIMERS", "Alzheimer's Disease", "disease", ["AD", "Senile dementia of the Alzheimer type"]),
        Concept("C_NEUROINFLAMMATION", "Neuroinflammation", "pathway", ["Microglial activation", "CNS inflammation"]),
        Concept("C_P_GINGIVALIS", "Porphyromonas gingivalis", "pathway", ["Gingipains", "P. gingivalis"]),
        Concept("C_AMYLOID_BETA", "Beta-Amyloid Plaque", "pathway", ["Abeta42", "Amyloid beta aggregation"]),

        # Domain 6: NAFLD & GLP-1
        Concept("C_NAFLD", "Non-Alcoholic Fatty Liver Disease", "disease", ["NAFLD", "MASLD", "Hepatic steatosis"]),
        Concept("C_GLP1_R", "GLP-1 Receptor Signaling", "pathway", ["GLP1R", "Incretin pathway"]),
        Concept("C_SEMAGLUTIDE", "Semaglutide", "drug_compound", ["Ozempic", "Wegovy"]),
    ]

    for c in concepts:
        corpus.add_concept(c)

    # Literature Papers
    papers = [
        # Swanson Raynaud's A-B papers
        Paper("PMID_001", "Blood viscosity and digital vasospasm in Raynaud's disease",
              abstract="Patients with Raynaud's disease demonstrate elevated blood viscosity and heightened digital vasospasm.",
              year=1982, journal_tier="top", citation_count=320,
              concepts=["C_RAYNAUDS", "C_BLOOD_VISCOSITY", "C_VASOSPASM"],
              cites=[]),
        Paper("PMID_002", "Platelet aggregation abnormalities in primary Raynaud phenomenon",
              abstract="Platelet aggregation is significantly accelerated during cold-induced digital vasospasm in Raynaud's disease.",
              year=1984, journal_tier="good", citation_count=180,
              concepts=["C_RAYNAUDS", "C_PLATELET_AGGR", "C_VASOSPASM"],
              cites=["PMID_001"]),

        # Swanson Raynaud's B-C papers (Fish Oil reduces blood viscosity & platelet aggregation)
        Paper("PMID_003", "Effects of dietary fish oil and EPA on blood viscosity in humans",
              abstract="Dietary supplementation with fish oil reduces whole blood viscosity and improves erythrocyte deformability.",
              year=1985, journal_tier="top", citation_count=450,
              concepts=["C_FISH_OIL", "C_BLOOD_VISCOSITY"],
              cites=[]),
        Paper("PMID_004", "Inhibition of platelet aggregation by eicosapentaenoic acid from fish oil",
              abstract="Eicosapentaenoic acid from fish oil markedly suppresses platelet aggregation and thromboxane release.",
              year=1985, journal_tier="top", citation_count=610,
              concepts=["C_FISH_OIL", "C_PLATELET_AGGR"],
              cites=["PMID_003"]),

        # Migraine A-B papers
        Paper("PMID_005", "Cortical spreading depression triggers cerebral vasoconstriction in migraine",
              abstract="Cortical spreading depression initiates migraine aura and subsequent cerebral vasoconstriction.",
              year=1987, journal_tier="top", citation_count=520,
              concepts=["C_MIGRAINE", "C_SPREADING_DEP", "C_VASOCONSTRICTION"],
              cites=[]),
        Paper("PMID_006", "Cerebral vasoconstriction in migraine pathogenesis",
              abstract="Focal cerebral vasoconstriction is observed during the prodromal phase of classical migraine headache.",
              year=1986, journal_tier="good", citation_count=140,
              concepts=["C_MIGRAINE", "C_VASOCONSTRICTION"],
              cites=["PMID_005"]),

        # Migraine B-C papers (Magnesium suppresses spreading depression)
        Paper("PMID_007", "Magnesium deficiency enhances susceptibility to cortical spreading depression",
              abstract="Hypomagnesemia lowers the threshold for cortical spreading depression in cerebral cortex models.",
              year=1988, journal_tier="top", citation_count=390,
              concepts=["C_MAGNESIUM", "C_SPREADING_DEP"],
              cites=[]),
        Paper("PMID_008", "Magnesium sulfate prevents cerebral vasoconstriction in experimental vascular models",
              abstract="Intravenous magnesium administration exerts potent vasodilatory effects against cerebral vasoconstriction.",
              year=1987, journal_tier="good", citation_count=210,
              concepts=["C_MAGNESIUM", "C_VASOCONSTRICTION"],
              cites=["PMID_007"]),

        # Pancreatic Cancer & Curcumin
        Paper("PMID_009", "Constitutive NF-kB activation mediates chemoresistance in pancreatic ductal adenocarcinoma",
              abstract="High NF-kB activity confers resistance to gemcitabine therapy in pancreatic ductal adenocarcinoma.",
              year=2018, journal_tier="top", citation_count=280,
              concepts=["C_PDAC", "C_NFKB", "C_CHEMORESISTANCE", "C_GEMCITABINE"],
              cites=[]),
        Paper("PMID_010", "Curcumin suppresses NF-kB pathway and sensitizes resistant tumor cells",
              abstract="Curcumin blocks IKK activation, suppressing NF-kB transcription and reversing chemoresistance.",
              year=2021, journal_tier="top", citation_count=340,
              concepts=["C_CURCUMIN", "C_NFKB", "C_CHEMORESISTANCE"],
              cites=["PMID_009"]),

        # Metformin & T-Cell Exhaustion
        Paper("PMID_011", "CD8+ T-cell exhaustion in malignant melanoma is characterized by metabolic insufficiency",
              abstract="Exhausted CD8+ T cells in melanoma show impaired glycolysis and oxidative phosphorylation.",
              year=2022, journal_tier="top", citation_count=195,
              concepts=["C_MELANOMA", "C_TCELL_EXHAUSTION"],
              cites=[]),
        Paper("PMID_012", "AMPK pathway activation by metformin rejuvenates memory CD8+ T cells",
              abstract="AMPK pathway activation using metformin alleviates CD8+ T-cell exhaustion in tumor microenvironments.",
              year=2023, journal_tier="top", citation_count=220,
              concepts=["C_METFORMIN", "C_AMPK", "C_TCELL_EXHAUSTION"],
              cites=["PMID_011"]),

        # Alzheimer's & P. gingivalis
        Paper("PMID_013", "Neuroinflammation and beta-amyloid plaque accumulation in Alzheimer's disease",
              abstract="Chronic neuroinflammation accelerates beta-amyloid plaque formation and cognitive decline in AD.",
              year=2020, journal_tier="top", citation_count=410,
              concepts=["C_ALZHEIMERS", "C_NEUROINFLAMMATION", "C_AMYLOID_BETA"],
              cites=[]),
        Paper("PMID_014", "Porphyromonas gingivalis gingipains drive neuroinflammation and amyloidogenesis",
              abstract="Periodontal pathogen Porphyromonas gingivalis infiltrates the brain, triggering neuroinflammation.",
              year=2019, journal_tier="top", citation_count=580,
              concepts=["C_P_GINGIVALIS", "C_NEUROINFLAMMATION", "C_AMYLOID_BETA"],
              cites=["PMID_013"]),

        # NAFLD & Semaglutide
        Paper("PMID_015", "GLP-1 receptor signaling reduces hepatic steatosis in non-alcoholic fatty liver disease",
              abstract="GLP-1 receptor signaling activation ameliorates lipid accumulation in hepatocytes in NAFLD.",
              year=2023, journal_tier="top", citation_count=310,
              concepts=["C_NAFLD", "C_GLP1_R"],
              cites=[]),
        Paper("PMID_016", "Semaglutide potentiation of GLP-1 receptor signaling in metabolic dysfunction",
              abstract="Semaglutide achieves sustained GLP-1 receptor activation, resolving hepatic inflammation and steatosis.",
              year=2024, journal_tier="top", citation_count=260,
              concepts=["C_SEMAGLUTIDE", "C_GLP1_R"],
              cites=["PMID_015"]),
    ]

    for p in papers:
        corpus.add_paper(p)

    return corpus
