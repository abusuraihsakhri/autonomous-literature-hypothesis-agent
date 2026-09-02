# Autonomous Literature Hypothesis Agent

> **Domain:** Clinical Decision Support & Biomedical Computing  
> **Reference Guidelines & Standards:** `Standard Clinical Formulations & ISO/IEC Quality Frameworks`

<div align="center">

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-3776AB.svg?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688.svg?logo=fastapi&logoColor=white)
![Audit Trail](https://img.shields.io/badge/Audit-HMAC--SHA256_Tamper--Evident-brightgreen.svg)
![Zero-PHI Guard](https://img.shields.io/badge/Guard-Zero--PHI_Outbound-blue.svg)
![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg?logo=docker&logoColor=white)

</div>

---

## 📖 What It Does

**Autonomous Literature Hypothesis Agent** is an advanced analytical and computational platform implementing PubMed biomedical knowledge graph & causal abductive hypothesis generator.

---

## ⚙️ Key Capabilities & Algorithmic Modules

### 🔬 Core Algorithmic & Evaluation Engines

- **`Concept`**: Biomedical concept or entity.
- **`Paper`**: Biomedical literature record / publication.
- **`BridgingPath`**: An intermediate bridging path A -> B -> C.
- **`Hypothesis`**: Literature-Based Discovery Generated Hypothesis.
- **`LiteratureCorpus`**: Corpus of biomedical papers, indexing concepts, citations, and co-occurrences.
- **`SwansonDiscoveryEngine`**: Swanson ABC Literature-Based Discovery Engine.
Implements Open Discovery (A -> ? -> C) and Closed Discovery (A -> B? -> C).

---

## 📐 Mathematical Formulation & Logic

```text
  path_score = math.sqrt(norm_ab * norm_bc)
```

---

## 💻 CLI Quickstart & Usage

### 1. Guided Interactive Mode
```bash
python cli.py
```

### 2. Direct Parameterized Evaluation
```bash
python cli.py --open <value> --closed <value> --pagerank <value> --gaps <value>
```

### Parameter Reference
- `--open`: Specifies input measurement or parameter value.
- `--closed`: Specifies input measurement or parameter value.
- `--pagerank`: Specifies input measurement or parameter value.
- `--gaps`: Specifies input measurement or parameter value.
- `--duplicates`: Specifies input measurement or parameter value.
- `--list-concepts`: Specifies input measurement or parameter value.
- `--interactive`: Specifies input measurement or parameter value.
- `--json`: Specifies input measurement or parameter value.
- `--source-quality`: Specifies input measurement or parameter value.

### Input Data Schema

| Field | Description | Requirement |
|:------|:------------|:------------|
| `suite_name` | Parameter / observation metric | Required |
| `system_slug` | Parameter / observation metric | Required |
| `standard_reference` | Parameter / observation metric | Required |
| `test_cases` | Parameter / observation metric | Required |

---

## 🛡️ Security & Enterprise Architecture

* **Zero-PHI Outbound Interceptor:** Active AST and regex inspection blocking SSNs, MRNs, phone numbers, and patient identifiers.
* **Tamper-Evident HMAC-SHA256 Audit Trail:** Chained, cryptographically signed logs for every evaluation and state transition.
* **Air-Gapped LLM Reasoning Adapter:** Agnostic integration for local Ollama instances (`llama3`, `mistral`), Claude 3.5 Sonnet, GPT-4o, and deterministic test mocks.
* **Active Learning Bayesian Calibration:** Dynamic tracker updating worker reliability weights and monitoring Brier calibration drift.
* **FastAPI & Prometheus Telemetry:** Exposes OpenAPI 3.1 REST endpoints and operational Prometheus metrics (`/metrics`).

---

## 🧪 Testing & Verification

Run the automated test suite:

```bash
pytest -v
```

Execute high-throughput batch simulation benchmarks:

```bash
python simulator.py --tasks 1000 --concurrency 8
```

---

## 🐳 Container Deployment

```bash
docker build -t autonomous-literature-hypothesis-agent .
docker run -p 8000:8000 autonomous-literature-hypothesis-agent
```
