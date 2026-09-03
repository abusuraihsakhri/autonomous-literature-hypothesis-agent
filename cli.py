#!/usr/bin/env python3
"""
Command Line Interface for Autonomous Literature Hypothesis Agent.

Usage examples:
  python cli.py --open "Raynaud's Disease"
  python cli.py --closed "Raynaud's Disease" "Fish Oil"
  python cli.py --pagerank
  python cli.py --gaps
  python cli.py --duplicates
  python cli.py --list-concepts
  python cli.py --interactive
  python cli.py --json --open "Migraine Disorder"
"""

import argparse
import csv
import json
import sys
from typing import List, Optional

from autonomous_literature_hypothesis import (
    build_curated_benchmark_corpus,
    SwansonDiscoveryEngine,
    CitationNetworkAnalysis,
    Paper,
    Concept,
    LiteratureCorpus,
)


def format_hypothesis_box(hypo, idx: int = 1) -> str:
    lines = []
    lines.append("=" * 80)
    lines.append(f"  [HYPOTHESIS #{idx}] {hypo.source_concept_name} ===> {hypo.target_concept_name}")
    lines.append(f"  Discovery Mode: {hypo.discovery_mode.upper()} | Confidence: {hypo.overall_confidence:.1f}/100")
    lines.append(f"  Plausibility: {hypo.plausibility_score:.3f} | Novelty: {hypo.novelty_score:.3f} | Prior Co-mentions: {hypo.direct_prior_cooccurrences}")
    lines.append("-" * 80)
    lines.append(f"  Mechanistic Rationale:")
    lines.append(f"    {hypo.mechanistic_rationale}")
    lines.append(f"  Intermediate Bridging Pathways ({len(hypo.bridging_paths)} found):")
    for bp in hypo.bridging_paths:
        lines.append(f"    * Bridge (B): {bp.concept_b_name} [{bp.concept_b_category}]")
        lines.append(f"      - NPMI(A,B)={bp.npmi_ab:.3f} (cooc={bp.ab_cooccurrences}, papers: {', '.join(bp.supporting_ab_papers)})")
        lines.append(f"      - NPMI(B,C)={bp.npmi_bc:.3f} (cooc={bp.bc_cooccurrences}, papers: {', '.join(bp.supporting_bc_papers)})")
        lines.append(f"      - Path Score: {bp.path_score:.3f}")
    lines.append(f"  Actionable Recommendation:")
    lines.append(f"    {hypo.recommendation}")
    lines.append("=" * 80)
    return "\n".join(lines)


def run_interactive(corpus: LiteratureCorpus):
    engine = SwansonDiscoveryEngine(corpus)
    print("\n" + "=" * 80)
    print("  AUTONOMOUS BIOMEDICAL LITERATURE HYPOTHESIS AGENT - INTERACTIVE SHELL")
    print("=" * 80)
    print("Type 'help' for commands, 'exit' or 'quit' to exit.\n")

    while True:
        try:
            cmd_line = input("lbd-agent> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not cmd_line:
            continue

        parts = cmd_line.split(maxsplit=2)
        cmd = parts[0].lower()

        if cmd in ("exit", "quit", "q"):
            print("Goodbye.")
            break
        elif cmd == "help":
            print("""
Available Commands:
  open <concept_name_or_id>             Run Swanson Open Discovery from concept A
  closed <concept_a> | <concept_c>      Run Swanson Closed Discovery between A and C
  concepts                              List all indexed biomedical concepts
  papers                                List all literature papers in corpus
  pagerank                              Compute PageRank centrality of citation network
  gaps                                  Detect literature gap domains
  duplicates                            Find duplicate papers in corpus
  help                                  Show this help menu
  exit / quit                           Exit the interactive shell
            """)
        elif cmd == "concepts":
            print(f"\nIndexed Concepts ({len(corpus.concepts)}):")
            for c in corpus.concepts.values():
                print(f"  [{c.concept_id:18s}] {c.name:32s} ({c.category})")
            print()
        elif cmd == "papers":
            print(f"\nIndexed Papers ({len(corpus.papers)}):")
            for p in corpus.papers.values():
                print(f"  [{p.paper_id:10s}] ({p.year}) {p.title[:65]}... [cites: {len(p.cites)}]")
            print()
        elif cmd == "open":
            if len(parts) < 2:
                print("Usage: open <concept_name_or_id>")
                continue
            query = parts[1] if len(parts) == 2 else f"{parts[1]} {parts[2]}"
            resolved = corpus.resolve_concept(query)
            if not resolved:
                print(f"Concept '{query}' not found in corpus.")
                continue
            print(f"\nRunning Swanson Open Discovery for '{resolved.name}' [{resolved.concept_id}]...")
            results = engine.open_discovery(resolved.concept_id)
            if not results:
                print("No hypotheses generated.")
            for i, hyp in enumerate(results, 1):
                print(format_hypothesis_box(hyp, i))
        elif cmd == "closed":
            if "|" not in cmd_line:
                print("Usage: closed <concept_a> | <concept_c>")
                continue
            _, args_str = cmd_line.split("closed", 1)
            raw_a, raw_c = args_str.split("|", 1)
            concept_a = corpus.resolve_concept(raw_a.strip())
            concept_c = corpus.resolve_concept(raw_c.strip())
            if not concept_a:
                print(f"Concept '{raw_a.strip()}' not found.")
                continue
            if not concept_c:
                print(f"Concept '{raw_c.strip()}' not found.")
                continue
            print(f"\nEvaluating closed discovery between '{concept_a.name}' and '{concept_c.name}'...")
            hyp = engine.closed_discovery(concept_a.concept_id, concept_c.concept_id)
            print(format_hypothesis_box(hyp, 1))
        elif cmd == "pagerank":
            graph = CitationNetworkAnalysis.build_graph(list(corpus.papers.values()))
            ranks = CitationNetworkAnalysis.pagerank(graph)
            print("\nPageRank Citation Centrality:")
            for pid, rank in ranks.items():
                p = corpus.papers.get(pid)
                title = p.title if p else ""
                print(f"  {pid:10s} : {rank:.6f} | {title[:60]}")
            print()
        elif cmd == "gaps":
            sample_topics = [
                "gut microbiome immunotherapy response",
                "blood viscosity vasospasm",
                "crispr prime editing duchenne muscular dystrophy",
                "semaglutide hepatic steatosis",
            ]
            gaps = CitationNetworkAnalysis.detect_literature_gaps(corpus, sample_topics)
            print(f"\nDetected Literature Gaps ({len(gaps)}):")
            for g in gaps:
                print(f"  [{g['status']}] Topic: '{g['topic']}' | Term Coverage: {g['term_coverage']*100:.0f}%")
                print(f"    Missing: {', '.join(g['missing_terms'])}")
            print()
        elif cmd == "duplicates":
            dups = CitationNetworkAnalysis.find_duplicates(list(corpus.papers.values()))
            print(f"\nDuplicate Check: {len(dups)} duplicate(s) found.")
            for d in dups:
                print(f"  {d['duplicate_paper_id']} duplicate of {d['original_paper_id']}: {d['reason']}")
            print()
        else:
            print(f"Unknown command '{cmd}'. Type 'help' for instructions.")


def run_batch(input_path: str, output_path: str, corpus: LiteratureCorpus, engine: SwansonDiscoveryEngine) -> int:
    """
    Process batch CSV containing biomedical queries (open / closed discovery).
    Writes comprehensive hypothesis metrics and intermediate bridging pathways to output CSV.
    """
    try:
        with open(input_path, mode="r", encoding="utf-8-sig") as infile:
            reader = csv.DictReader(infile)
            fieldnames = [
                "query_id",
                "query_type",
                "source_concept",
                "target_concept",
                "clinical_domain",
                "status",
                "hypotheses_count",
                "top_target_concept",
                "top_confidence",
                "top_plausibility",
                "top_novelty",
                "top_bridges",
                "top_mechanistic_rationale",
                "top_recommendation",
            ]
            rows_out = []
            for row in reader:
                qid = row.get("query_id", "").strip()
                qtype = row.get("query_type", "open").strip().lower()
                source_raw = row.get("source_concept", "").strip()
                target_raw = row.get("target_concept", "").strip()
                domain = row.get("clinical_domain", "").strip()
                try:
                    min_npmi = float(row.get("min_npmi", 0.0) or 0.0)
                except ValueError:
                    min_npmi = 0.0

                resolved_source = corpus.resolve_concept(source_raw) if source_raw else None

                if not resolved_source:
                    rows_out.append({
                        "query_id": qid,
                        "query_type": qtype,
                        "source_concept": source_raw,
                        "target_concept": target_raw,
                        "clinical_domain": domain,
                        "status": f"ERROR: Unresolved source concept '{source_raw}'",
                        "hypotheses_count": 0,
                        "top_target_concept": "",
                        "top_confidence": 0.0,
                        "top_plausibility": 0.0,
                        "top_novelty": 0.0,
                        "top_bridges": "",
                        "top_mechanistic_rationale": "",
                        "top_recommendation": "",
                    })
                    continue

                if qtype == "closed":
                    resolved_target = corpus.resolve_concept(target_raw) if target_raw else None
                    if not resolved_target:
                        rows_out.append({
                            "query_id": qid,
                            "query_type": qtype,
                            "source_concept": source_raw,
                            "target_concept": target_raw,
                            "clinical_domain": domain,
                            "status": f"ERROR: Unresolved target concept '{target_raw}'",
                            "hypotheses_count": 0,
                            "top_target_concept": "",
                            "top_confidence": 0.0,
                            "top_plausibility": 0.0,
                            "top_novelty": 0.0,
                            "top_bridges": "",
                            "top_mechanistic_rationale": "",
                            "top_recommendation": "",
                        })
                        continue

                    hypo = engine.closed_discovery(resolved_source.concept_id, resolved_target.concept_id, min_npmi=min_npmi)
                    top_bridges = "; ".join([f"{b.concept_b_name} [{b.concept_b_category}] (score={b.path_score})" for b in hypo.bridging_paths[:3]])
                    rows_out.append({
                        "query_id": qid,
                        "query_type": qtype,
                        "source_concept": resolved_source.name,
                        "target_concept": resolved_target.name,
                        "clinical_domain": domain,
                        "status": "SUCCESS",
                        "hypotheses_count": 1 if hypo.bridging_paths else 0,
                        "top_target_concept": hypo.target_concept_name,
                        "top_confidence": hypo.overall_confidence,
                        "top_plausibility": hypo.plausibility_score,
                        "top_novelty": hypo.novelty_score,
                        "top_bridges": top_bridges,
                        "top_mechanistic_rationale": hypo.mechanistic_rationale,
                        "top_recommendation": hypo.recommendation,
                    })
                else:
                    # Open discovery
                    hypos = engine.open_discovery(resolved_source.concept_id, min_npmi=min_npmi)
                    if hypos:
                        top = hypos[0]
                        top_bridges = "; ".join([f"{b.concept_b_name} [{b.concept_b_category}] (score={b.path_score})" for b in top.bridging_paths[:3]])
                        rows_out.append({
                            "query_id": qid,
                            "query_type": qtype,
                            "source_concept": resolved_source.name,
                            "target_concept": top.target_concept_name,
                            "clinical_domain": domain,
                            "status": "SUCCESS",
                            "hypotheses_count": len(hypos),
                            "top_target_concept": top.target_concept_name,
                            "top_confidence": top.overall_confidence,
                            "top_plausibility": top.plausibility_score,
                            "top_novelty": top.novelty_score,
                            "top_bridges": top_bridges,
                            "top_mechanistic_rationale": top.mechanistic_rationale,
                            "top_recommendation": top.recommendation,
                        })
                    else:
                        rows_out.append({
                            "query_id": qid,
                            "query_type": qtype,
                            "source_concept": resolved_source.name,
                            "target_concept": "",
                            "clinical_domain": domain,
                            "status": "NO_HYPOTHESES_FOUND",
                            "hypotheses_count": 0,
                            "top_target_concept": "",
                            "top_confidence": 0.0,
                            "top_plausibility": 0.0,
                            "top_novelty": 0.0,
                            "top_bridges": "",
                            "top_mechanistic_rationale": "",
                            "top_recommendation": "",
                        })

        with open(output_path, mode="w", newline="", encoding="utf-8") as outfile:
            writer = csv.DictWriter(outfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows_out)

        print(f"Batch processing complete. Processed {len(rows_out)} queries. Results written to: {output_path}")
        return 0
    except Exception as e:
        print(f"Error during batch execution: {e}", file=sys.stderr)
        return 1


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="autonomous-literature-hypothesis-agent",
        description="Autonomous Literature-Based Hypothesis Discovery Engine (Swanson ABC Model & Citation Analysis)",
    )

    # Subparsers for commands including batch
    subparsers = parser.add_subparsers(dest="subcommand", help="Available subcommands")

    batch_parser = subparsers.add_parser("batch", help="Batch process biomedical queries from a CSV file")
    batch_parser.add_argument("--input", "-i", dest="input_file", required=True, help="Path to input CSV file")
    batch_parser.add_argument("--output", "-o", dest="output_file", required=True, help="Path to output CSV results file")

    parser.add_argument("--batch", action="store_true", help="Run in batch mode (used with --input and --output)")
    parser.add_argument("--input", "-i", dest="input_file", help="Input CSV path for batch mode")
    parser.add_argument("--output", "-o", dest="output_file", help="Output CSV path for batch mode")
    parser.add_argument("--open", dest="open_concept", help="Run Swanson open discovery from source concept A")
    parser.add_argument("--closed", "-c", dest="closed_concepts", nargs=2, metavar=("CONCEPT_A", "CONCEPT_C"),
                        help="Run Swanson closed discovery between concept A and concept C")
    parser.add_argument("--list-concepts", action="store_true", help="List all indexed concepts in the benchmark knowledge base")
    parser.add_argument("--pagerank", action="store_true", help="Compute PageRank citation centrality")
    parser.add_argument("--gaps", action="store_true", help="Detect literature gaps across biomedical topics")
    parser.add_argument("--duplicates", action="store_true", help="Scan indexed corpus for duplicate papers")
    parser.add_argument("--source-quality", dest="quality_paper_id", help="Evaluate source quality of a given paper ID")
    parser.add_argument("--interactive", action="store_true", help="Launch interactive discovery shell")
    parser.add_argument("--json", action="store_true", help="Output results in JSON format")

    args = parser.parse_args(argv)
    corpus = build_curated_benchmark_corpus()
    engine = SwansonDiscoveryEngine(corpus)

    if args.subcommand == "batch" or args.batch or (args.input_file and args.output_file):
        if not args.input_file or not args.output_file:
            print("Error: Batch mode requires both --input and --output file paths.", file=sys.stderr)
            return 1
        return run_batch(args.input_file, args.output_file, corpus, engine)

    if args.interactive:
        run_interactive(corpus)
        return 0

    if args.list_concepts:
        if args.json:
            data = [c.to_dict() for c in corpus.concepts.values()]
            print(json.dumps(data, indent=2))
        else:
            print(f"\nIndexed Biomedical Concepts ({len(corpus.concepts)}):")
            print("=" * 70)
            for c in corpus.concepts.values():
                syns = f" (Synonyms: {', '.join(c.synonyms)})" if c.synonyms else ""
                print(f"  [{c.concept_id:18s}] {c.name:32s} | Category: {c.category}{syns}")
            print("=" * 70)
        return 0

    if args.open_concept:
        resolved = corpus.resolve_concept(args.open_concept)
        if not resolved:
            err = {"error": f"Concept '{args.open_concept}' not recognized in knowledge base."}
            if args.json:
                print(json.dumps(err, indent=2))
            else:
                print(f"Error: {err['error']}", file=sys.stderr)
            return 1
        results = engine.open_discovery(resolved.concept_id)
        if args.json:
            print(json.dumps([h.to_dict() for h in results], indent=2))
        else:
            print(f"\nSwanson Open Discovery Results for '{resolved.name}' ({len(results)} found):")
            for i, hyp in enumerate(results, 1):
                print(format_hypothesis_box(hyp, i))
        return 0

    if args.closed_concepts:
        ca_raw, cc_raw = args.closed_concepts
        concept_a = corpus.resolve_concept(ca_raw)
        concept_c = corpus.resolve_concept(cc_raw)
        if not concept_a or not concept_c:
            missing = ca_raw if not concept_a else cc_raw
            err = {"error": f"Concept '{missing}' not recognized in knowledge base."}
            if args.json:
                print(json.dumps(err, indent=2))
            else:
                print(f"Error: {err['error']}", file=sys.stderr)
            return 1
        hyp = engine.closed_discovery(concept_a.concept_id, concept_c.concept_id)
        if args.json:
            print(json.dumps(hyp.to_dict(), indent=2))
        else:
            print(format_hypothesis_box(hyp, 1))
        return 0

    if args.pagerank:
        graph = CitationNetworkAnalysis.build_graph(list(corpus.papers.values()))
        ranks = CitationNetworkAnalysis.pagerank(graph)
        if args.json:
            print(json.dumps(ranks, indent=2))
        else:
            print("\nPageRank Centrality of Citation Network:")
            print("=" * 70)
            for pid, score in ranks.items():
                p = corpus.papers.get(pid)
                title = p.title if p else "Unknown"
                print(f"  {pid:12s} | Score: {score:.6f} | {title}")
            print("=" * 70)
        return 0

    if args.gaps:
        topics = [
            "gut microbiome anti-pd1 immunotherapy response colorectal cancer",
            "blood viscosity digital vasospasm",
            "crispr cas9 prime editing sickle cell disease",
            "porphyromonas gingivalis gingipains neuroinflammation amyloid",
            "nanoparticle lipid delivery targeted mrna vaccine",
        ]
        gaps = CitationNetworkAnalysis.detect_literature_gaps(corpus, topics)
        if args.json:
            print(json.dumps(gaps, indent=2))
        else:
            print("\nBiomedical Literature Gap Detection:")
            print("=" * 80)
            for g in gaps:
                print(f"  [{g['status']}] Topic: {g['topic']}")
                print(f"    Coverage: {g['term_coverage']*100:.1f}% | Missing Terms: {', '.join(g['missing_terms'])}")
                print(f"    Recommendation: {g['recommendation']}")
            print("=" * 80)
        return 0

    if args.duplicates:
        # Include a test duplicate to demonstrate detection
        papers = list(corpus.papers.values()) + [
            Paper("TEST_DUP_1", "Effects of dietary fish oil and EPA on blood viscosity in humans.", year=2024),
            Paper("TEST_DUP_2", "Distinct topic on oncology", doi="10.1126/science.aaz123"),
            Paper("TEST_DUP_3", "Another title", doi="10.1126/science.aaz123"),
        ]
        dups = CitationNetworkAnalysis.find_duplicates(papers)
        if args.json:
            print(json.dumps(dups, indent=2))
        else:
            print(f"\nDuplicate Detection Results ({len(dups)} duplicates found):")
            print("=" * 80)
            for d in dups:
                print(f"  Original: {d['original_paper_id']:12s} <--> Duplicate: {d['duplicate_paper_id']:12s}")
                print(f"    Reason: {d['reason']} | Similarity: {d['similarity']}")
            print("=" * 80)
        return 0

    if args.quality_paper_id:
        p = corpus.papers.get(args.quality_paper_id)
        if not p:
            err = {"error": f"Paper '{args.quality_paper_id}' not found."}
            if args.json:
                print(json.dumps(err, indent=2))
            else:
                print(f"Error: {err['error']}", file=sys.stderr)
            return 1
        q = CitationNetworkAnalysis.source_quality(p)
        if args.json:
            print(json.dumps(q, indent=2))
        else:
            print(f"\nSource Quality Assessment for {p.paper_id}:")
            print(f"  Title:         {p.title}")
            print(f"  Journal Tier:  {q['journal_tier']} | Citations: {q['citation_count']} | Age: {q['age_years']} yrs")
            print(f"  Quality Score: {q['quality_score']}/100 ({q['quality_tier']})")
        return 0

    # Default to open discovery on Swanson classic if no arguments provided
    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
