# Autonomous Literature Hypothesis Agent (Literature-Based Discovery Engine)

A high-performance biomedical Literature-Based Discovery (LBD) platform implementing the Swanson ABC discovery paradigm, Normalized Pointwise Mutual Information (NPMI) transitive inference, citation network centrality (PageRank), automated paper deduplication, and domain gap detection.

## Key Features

- **Swanson ABC Discovery Architecture**:
  - **Open Discovery ($A \rightarrow B \rightarrow ?$)**: Given a disease/target $A$, discovers intermediate mechanisms/pathways $B$, then discovers potential novel therapeutic interventions $C$ with no or low direct prior co-occurrence.
  - **Closed Discovery ($A \rightarrow ? \rightarrow C$)**: Given two disconnected entities $A$ and $C$, systematically maps and scores all intermediate connecting pathways $\{B_i\}$ to provide mechanistic plausibility.
- **Statistical Association Metrics**:
  - Pointwise Mutual Information: $\text{PMI}(A, B) = \log_2 \left( \frac{P(A, B)}{P(A) P(B)} \right)$
  - Normalized PMI: $\text{NPMI}(A, B) = \frac{\text{PMI}(A, B)}{-\log_2 P(A, B)} \in [-1, 1]$
  - Multi-path score aggregation with noisy-OR combination: $\text{Plausibility}(A \rightarrow C) = 1 - \prod_i (1 - \text{Score}(A, B_i, C))$
  - Novelty penalty: $\text{Novelty} = \frac{1}{1 + N_{\text{direct}}(A, C)}$
- **Citation Graph & Bibliometrics**:
  - Directed Citation Adjacency with PageRank Power Iteration ($\alpha = 0.85$).
  - Source Quality Evaluation incorporating Journal Impact Tier, citation velocity, and publication recency.
  - Paper deduplication via persistent identifier resolution (DOI/PMID) and trigram Jaccard similarity.
- **Literature Gap Analysis**:
  - Automated detection of unrepresented or low-coverage biomedical concepts and therapeutic hypotheses across literature corpora.
- **Pure Python Standard Library**:
  - Zero external third-party dependencies required; operates seamlessly in restricted or air-gapped environments.

---

## Benchmark Knowledge Base

The built-in benchmark dataset includes verified historical LBD breakthroughs and modern multi-omics paradigms:
1. **Raynaud's Disease $\rightarrow$ Blood Viscosity / Platelet Aggregation $\rightarrow$ Fish Oil** (Swanson 1986)
2. **Migraine Disorder $\rightarrow$ Cortical Spreading Depression / Vasoconstriction $\rightarrow$ Magnesium** (Swanson 1988)
3. **Pancreatic Cancer $\rightarrow$ NF-kB / Chemoresistance $\rightarrow$ Curcumin**
4. **Melanoma CD8+ T-Cell Exhaustion $\rightarrow$ AMPK Pathway $\rightarrow$ Metformin**
5. **Alzheimer's Disease $\rightarrow$ Neuroinflammation $\rightarrow$ Porphyromonas gingivalis**
6. **NAFLD / Hepatic Steatosis $\rightarrow$ GLP-1 Receptor Signaling $\rightarrow$ Semaglutide**

---

## CLI Usage

### 1. Swanson Open Discovery
```bash
python cli.py --open "Raynaud's Disease"
```

### 2. Swanson Closed Discovery
```bash
python cli.py --closed "Migraine Disorder" "Magnesium"
```

### 3. PageRank Citation Centrality
```bash
python cli.py --pagerank
```

### 4. Literature Gap Detection
```bash
python cli.py --gaps
```

### 5. Deduplication Analysis
```bash
python cli.py --duplicates
```

### 6. Source Quality Assessment
```bash
python cli.py --source-quality PMID_001
```

### 7. Structured JSON Output
```bash
python cli.py --open "Raynaud's Disease" --json
```

### 8. Interactive Discovery Shell
```bash
python cli.py --interactive
```

---

## Unit Testing

Run the test suite via Python's standard `unittest` framework:

```bash
python -m unittest test_autonomous_literature_hypothesis.py
```

All 28 test cases validate mathematical consistency, boundary conditions, citation metrics, deduplication thresholds, and CLI workflows.
