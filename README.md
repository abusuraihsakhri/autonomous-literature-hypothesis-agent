# Autonomous Literature Hypothesis Agent (Literature-Based Discovery Engine)

> **Domain:** Biomedical Informatics, Literature-Based Discovery (LBD), Translational Knowledge Graphs  
> **Methodology:** Swanson ABC Transitive Linking, Normalized Pointwise Mutual Information (NPMI), Citation PageRank & Epistemic Confidence Scoring

<div align="center">

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-3776AB.svg?logo=python&logoColor=white)
![Build](https://img.shields.io/badge/CI-GitHub%20Actions-brightgreen.svg?logo=github-actions&logoColor=white)
![Dependencies](https://img.shields.io/badge/Dependencies-Zero%20(Pure%20Python)-brightgreen.svg)
![Coverage](https://img.shields.io/badge/Tests-31%20Passed-success.svg)

</div>

---

## 📖 Executive Overview

The **Autonomous Literature Hypothesis Agent** is a literature-based discovery (LBD) engine designed to uncover novel, non-obvious mechanistic connections across disjoint biomedical literature corpora. 

In scientific publishing, knowledge is frequently sequestered into specialized siloes (e.g., rheology literature versus clinical vasospastic literature). Seminal work by Don R. Swanson (1986, 1988) proved that transitive connections between disjoint literatures ($A \rightarrow B$ and $B \rightarrow C$) could reveal unhypothesized therapeutic interventions ($A \rightarrow C$) long before clinical trials occur—such as dietary fish oil reducing whole blood viscosity in Raynaud's syndrome, or magnesium deficiency lowering the threshold for cortical spreading depression in migraine.

This repository provides an autonomous, pure Python computational engine implementing both **Open Discovery** ($A \rightarrow B \rightarrow ?$) and **Closed Discovery** ($A \rightarrow ? \rightarrow C$), normalized statistical co-occurrence metrics (NPMI), multi-path causal plausibility scoring, citation network graph analytics (PageRank, H-index, bibliographic coupling), paper deduplication, and topic gap detection.

---

## 🔬 Literature-Based Discovery Methodology

```
Disjoint Literature Domain A                  Disjoint Literature Domain C
[ Concept A: Raynaud's Disease ]              [ Concept C: Fish Oil / EPA ]
              │                                             ▲
              │ (Documented Co-occurrence)                  │ (Documented Co-occurrence)
              ▼                                             │
      ┌──────────────────────────────────────────────────────────┐
      │          Intermediate Bridging Concepts B               │
      │   * Blood Viscosity (Hyperviscosity / Rheology)          │
      │   * Platelet Aggregation (Activation & Thrombogenesis)   │
      └──────────────────────────────────────────────────────────┘
                                  │
                                  ▼
      ===========================================================
      SYNTHESIZED NOVEL HYPOTHESIS (Swanson 1986 Paradigm):
      Dietary fish oil / EPA administration reduces Raynaud's
      symptoms by decreasing whole blood viscosity and platelet
      reactivity during digital vasospastic episodes.
      Prior direct literature co-mentions: 0 (True Novel Discovery)
      ===========================================================
```

### 1. Swanson Discovery Paradigms

* **Open Discovery ($A \rightarrow \text{intermediate } B \rightarrow \text{target } C$):**  
  Given a starting concept $A$ (e.g., an orphan disease or therapy-resistant tumor), the engine identifies all strongly co-occurring biological intermediates $B$. For each $B$, it traverses outwards to find all connected entities $C$ that have zero or minimal direct prior co-mentions with $A$.
* **Closed Discovery ($A \leftrightarrow C$ Transitive Evaluation):**  
  Given two disparate concepts $A$ and $C$ (e.g., *Migraine Disorder* and *Magnesium*), the engine discovers all mutual intermediate bridging pathways $B_1, B_2, \dots, B_k$ (e.g., *Cortical Spreading Depression*, *Cerebral Vasoconstriction*) and calculates the mechanistic plausibility of their connection.

---

## 📐 Mathematical Formulation

### 1. Pointwise Mutual Information (PMI) and Normalized PMI (NPMI)

The association strength between concept pair $(x, y)$ in a corpus of $N$ publications is calculated using Pointwise Mutual Information:

$$P(x) = \frac{\text{count}(x)}{N}, \quad P(y) = \frac{\text{count}(y)}{N}, \quad P(x, y) = \frac{\text{count}(x \cap y)}{N}$$

$$\text{PMI}(x, y) = \log_2 \left( \frac{P(x, y)}{P(x) \cdot P(y)} \right)$$

To eliminate the frequency bias of standard PMI and bound the values within $[-1, +1]$, Normalized PMI (NPMI) is applied:

$$\text{NPMI}(x, y) = \frac{\text{PMI}(x, y)}{-\log_2(P(x, y))}$$

* $\text{NPMI} = +1$: Absolute co-occurrence (whenever $x$ appears, $y$ appears).
* $\text{NPMI} = 0$: Complete statistical independence ($P(x, y) = P(x)P(y)$).
* $\text{NPMI} = -1$: Complete mutual exclusion (never co-occur in the corpus).

### 2. Transitive Bridging Path Score

For an intermediate bridging concept $B_j$ connecting source $A$ and target $C$:

$$\text{Score}(A \rightarrow B_j \rightarrow C) = \sqrt{\max(0, \text{NPMI}(A, B_j)) \times \max(0, \text{NPMI}(B_j, C))}$$

### 3. Multi-Path Plausibility Aggregation (Noisy-OR Formulation)

When multiple intermediate pathways corroborate the transitive link between $A$ and $C$, their path scores are aggregated using a bounded probabilistic Noisy-OR formulation to account for diminishing returns:

$$\text{Plausibility}(A, C) = 1.0 - \prod_{j=1}^{k} \left( 1.0 - \min(0.99, \text{Score}(A \rightarrow B_j \rightarrow C)) \right)$$

### 4. Epistemic Novelty and Composite Confidence Scoring

Novelty penalizes existing direct literature co-mentions:

$$\text{Novelty}(A, C) = \frac{1.0}{1.0 + \text{DirectCooccurrences}(A, C)}$$

A path diversity multiplier accounts for multi-mechanistic corroboration:

$$\text{DiversityBonus} = \min\left(1.25, \, 1.0 + 0.08 \cdot (k - 1)\right)$$

$$\text{Confidence}(A, C) = \min\left(99.9, \, \max\left(5.0, \, \left(70 \cdot \text{Plausibility} + 30 \cdot \text{Novelty}\right) \times \text{DiversityBonus}\right)\right)$$

### 5. Citation Network Analysis: PageRank Centrality

Citation authority is computed across the directed paper citation graph $G = (V, E)$ using the power iteration PageRank algorithm with damping factor $d = 0.85$:

$$\text{PR}(p_i) = \frac{1 - d}{|V|} + d \sum_{p_j \in M(p_i)} \frac{\text{PR}(p_j)}{L(p_j)}$$

where $M(p_i)$ is the set of papers citing $p_i$, and $L(p_j)$ is the out-degree (reference count) of $p_j$.

---

## 🏛️ Benchmark Disease-Therapy Validation Domains

The engine includes a curated benchmark knowledge base containing landmark historical and contemporary literature discovery pairs:

| Domain | Source Concept ($A$) | Intermediate Bridging Pathways ($B$) | Target Concept ($C$) | Historical Landmark / Rationale |
|:---|:---|:---|:---|:---|
| **Vascular Rheology** | Raynaud's Disease | Blood Viscosity, Platelet Aggregation | Fish Oil (EPA) | Swanson (1986) classical discovery |
| **Neurovascular** | Migraine Disorder | Cortical Spreading Depression, Vasoconstriction | Magnesium | Swanson (1988) classical discovery |
| **Oncology** | Pancreatic Adenocarcinoma | NF-kB Pathway, Chemoresistance | Curcumin | Reversal of gemcitabine resistance |
| **Immuno-oncology** | Malignant Melanoma | CD8+ T-Cell Exhaustion | AMPK Pathway / Metformin | Metabolic rejuvenation of exhausted T cells |
| **Neurodegeneration** | Alzheimer's Disease | Neuroinflammation, Beta-Amyloid Plaque | *Porphyromonas gingivalis* | Periodontal gingipains neuroinvasion hypothesis |
| **Metabolic Hepatology** | Non-Alcoholic Fatty Liver Disease | GLP-1 Receptor Signaling | Semaglutide | Incretin resolution of hepatic steatosis |

---

## 🚀 Installation & Quickstart

The library is written in pure Python 3 (Python 3.9+) with **zero mandatory external dependencies**.

```bash
# Clone the repository
git clone https://github.com/abusuraihsakhri/autonomous-literature-hypothesis-engine.git
cd autonomous-literature-hypothesis-agent

# Verify test suite
python -m pytest -p no:zarr -v
```

---

## 💻 CLI Usage Guide

The CLI interface supports individual interactive sessions, single-shot queries, subcommands, and batch CSV processing.

### 1. Batch Subcommand Processing (`batch`)

Process a batch CSV file containing open and closed discovery queries:

```bash
# Run batch discovery
python cli.py batch -i sample.csv -o batch_hypotheses_output.csv

# Or using alternate top-level flag syntax
python cli.py --batch --input sample.csv --output batch_hypotheses_output.csv
```

### 2. Open Discovery

Identify unlinked candidate therapeutic targets for a disease:

```bash
python cli.py --open "Raynaud's Disease"
python cli.py --open "Malignant Melanoma" --json
```

### 3. Closed Discovery

Evaluate the mechanistic bridging pathways connecting two concepts:

```bash
python cli.py --closed "Migraine Disorder" "Magnesium"
python cli.py --closed "Pancreatic Ductal Adenocarcinoma" "Curcumin" --json
```

### 4. Citation Graph & PageRank Centrality

Compute citation influence scores across indexed literature:

```bash
python cli.py --pagerank
```

### 5. Literature Gap & Duplicate Detection

Audit indexed literature for thematic blindspots or redundant publications:

```bash
# Detect knowledge gaps
python cli.py --gaps

# Scan for bibliographic duplicates (DOI collisions and trigram Jaccard matching)
python cli.py --duplicates

# Evaluate source quality metric for a paper
python cli.py --source-quality PMID_001
```

### 6. Interactive Shell

Launch the command-line exploration terminal:

```bash
python cli.py --interactive
```

---

## 🐍 Python API Quickstart

```python
from autonomous_literature_hypothesis import (
    build_curated_benchmark_corpus,
    SwansonDiscoveryEngine,
    CitationNetworkAnalysis,
)

# 1. Initialize curated corpus and discovery engine
corpus = build_curated_benchmark_corpus()
engine = SwansonDiscoveryEngine(corpus)

# 2. Run Open Discovery (Raynaud's Disease -> ? -> Target)
hypotheses = engine.open_discovery("C_RAYNAUDS")
for h in hypotheses:
    print(f"Discovered: {h.source_concept_name} -> {h.target_concept_name}")
    print(f"Confidence: {h.overall_confidence:.1f}% | Plausibility: {h.plausibility_score:.3f}")
    for bp in h.bridging_paths:
        print(f"  Bridge: {bp.concept_b_name} [NPMI(A,B)={bp.npmi_ab}, NPMI(B,C)={bp.npmi_bc}]")

# 3. Run Closed Discovery (Migraine -> Magnesium)
closed_hypo = engine.closed_discovery("C_MIGRAINE", "C_MAGNESIUM")
print(f"Closed Rationale: {closed_hypo.mechanistic_rationale}")

# 4. Citation Network Analysis
graph = CitationNetworkAnalysis.build_graph(list(corpus.papers.values()))
ranks = CitationNetworkAnalysis.pagerank(graph)
print("Top Cited Influence:", sorted(ranks.items(), key=lambda x: x[1], reverse=True)[:3])
```

---

## 📊 Batch CSV Input & Output Schema

### Input Schema (`sample.csv`)

| Column | Type | Required | Description |
|:---|:---|:---|:---|
| `query_id` | String | Yes | Unique inquiry identifier (e.g. `Q001`) |
| `query_type` | String | Yes | `open` (explore from source) or `closed` (verify source-target pair) |
| `source_concept` | String | Yes | Name or synonym of concept A (e.g. `Raynaud's Disease`) |
| `target_concept` | String | Conditional | Name or synonym of concept C (required for `closed` mode) |
| `clinical_domain` | String | Optional | Biomedical classification domain |
| `min_npmi` | Float | Optional | Minimum association threshold (default `0.0`) |
| `notes` | String | Optional | Translational context or clinical rationale |

### Output Schema (`batch_hypotheses_output.csv`)

| Column | Description |
|:---|:---|
| `query_id` | Identifier matching input record |
| `query_type` | Executed mode (`open` or `closed`) |
| `source_concept` | Standardized canonical source entity name |
| `target_concept` | Standardized canonical target entity name |
| `clinical_domain` | Clinical domain classification |
| `status` | `SUCCESS`, `NO_HYPOTHESES_FOUND`, or `ERROR: ...` |
| `hypotheses_count` | Number of synthesized candidate hypotheses |
| `top_target_concept`| Highest-scoring target entity |
| `top_confidence` | Composite epistemic confidence (0.0 to 100.0) |
| `top_plausibility` | Aggregated multi-path plausibility score (0.0 to 1.0) |
| `top_novelty` | Inverse direct co-mention score (1.0 = completely unlinked) |
| `top_bridges` | Ranked intermediate bridging entities with path scores |
| `top_mechanistic_rationale` | Auto-synthesized biological mechanism narrative |
| `top_recommendation`| Actionable wet-lab validation suggestion |

---

## 🧪 Testing & Verification

Run the full pytest unit and regression test suite:

```bash
python -m pytest -p no:zarr -v
```

All 31 unit tests cover:
- Concept indexing and synonym normalization
- PMI, NPMI, and Jaccard mathematical calculations
- Open and closed Swanson discovery pipelines
- Directed citation graph construction and PageRank power iteration
- Bibliographic deduplication and literature gap detection
- CLI subcommands (`batch`, `--open`, `--closed`, `--pagerank`, `--gaps`, `--duplicates`, `--source-quality`)
- CSV error handling for unmapped concepts and invalid parameters

---

## 📄 License

This project is licensed under the terms of the [MIT License](LICENSE).
