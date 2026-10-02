"""
CLI entry point for the SILVA reference-matching stage.

Run one stage at a time and verify its output before moving to the next --
each stage's module docstring says what to check.

Examples:
    python run_pipeline.py inspect data/SILVA_138.2_SSURef_tax_silva_trunc.fasta
    python run_pipeline.py parse
    python run_pipeline.py embed-ref
    python run_pipeline.py embed-reads
    python run_pipeline.py build-index
    python run_pipeline.py query
    
"""
import argparse
import sys

from src import config
from src.inspect_silva import inspect, print_report
from src.build_reference_table import build_reference_table
from src.embed_reference import embed_reference_table
from src.embed_experimental import embed_reads
from src.faiss_index import build_index
from src.query_index import query_index, write_query_results
from src.confidence import build_results
from src.cluster_annotation import (
    load_query_results, load_cluster_assignments, build_read_labels,
    annotate_clusters, compute_agreement_metrics, write_cluster_annotation,
    write_cluster_summary, print_cluster_summary,
)
from src.confidence import load_ref_lookup
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="eDNA-TaxAI reference-matching pipeline")
    sub = parser.add_subparsers(dest="stage", required=True)

    sub.add_parser("inspect").add_argument("fasta", nargs="?", default=config.SILVA_FASTA_PATH)
    sub.add_parser("parse")
    sub.add_parser("embed-ref")
    sub.add_parser("embed-reads")
    sub.add_parser("build-index")
    sub.add_parser("query")
    sub.add_parser("decide")
    p_annotate = sub.add_parser("annotate-clusters")
    p_annotate.add_argument("--min-similarity", type=float, default=None,
                             help="reads below this FAISS similarity are treated as 'no reference match', not dropped")
    p_annotate.add_argument("--taxonomy-depth", type=int, default=None,
                             help="overrides config.CLUSTER_LABEL_TAXONOMY_DEPTH")

    args = parser.parse_args()

    if args.stage == "inspect":
        print_report(inspect(args.fasta))

    elif args.stage == "parse":
        build_reference_table(config.SILVA_FASTA_PATH, config.REFERENCE_TABLE_CSV)

    elif args.stage == "embed-ref":
        if config.MAX_LENGTH is None:
            sys.exit("config.MAX_LENGTH is not set -- see README.md 'Region mismatch' before embedding.")
        embed_reference_table(config.REFERENCE_TABLE_CSV, config.REFERENCE_EMBEDDINGS_NPY, config.REFERENCE_IDS_CSV)

    elif args.stage == "embed-reads":
        if config.MAX_LENGTH is None:
            sys.exit("config.MAX_LENGTH is not set -- see README.md 'Region mismatch' before embedding.")
        embed_reads(config.EXPERIMENTAL_FASTQ_PATH, config.EXPERIMENTAL_EMBEDDINGS_NPY)

    elif args.stage == "build-index":
        build_index(config.REFERENCE_EMBEDDINGS_NPY, config.FAISS_INDEX_PATH)

    elif args.stage == "query":
        # expects data/reads_ids.csv: one Read_ID per line, aligned to the
        # experimental embeddings row order
        with open("data/reads_ids.csv") as f:
            read_ids = [line.strip() for line in f if line.strip()]
        read_embeddings = np.load(config.EXPERIMENTAL_EMBEDDINGS_NPY)
        sims, pos = query_index(config.FAISS_INDEX_PATH, read_ids, read_embeddings)
        write_query_results(read_ids, sims, pos, "data/query_results.csv")

    elif args.stage == "decide":
        if config.SIMILARITY_THRESHOLD is None:
            sys.exit("config.SIMILARITY_THRESHOLD is not set -- run evaluate.py's threshold sweep first.")
        build_results(
            query_results_csv="data/query_results.csv",
            ref_ids_csv=config.REFERENCE_IDS_CSV,
            ref_table_csv=config.REFERENCE_TABLE_CSV,
            threshold=config.SIMILARITY_THRESHOLD,
            min_margin=config.MIN_MARGIN,
            out_csv=config.RESULTS_CSV,
        )

    elif args.stage == "annotate-clusters":
        depth = args.taxonomy_depth or config.CLUSTER_LABEL_TAXONOMY_DEPTH
        query_results = load_query_results("data/query_results.csv")
        cluster_assignments = load_cluster_assignments(config.CLUSTER_ASSIGNMENTS_CSV)
        ref_lookup = load_ref_lookup(config.REFERENCE_IDS_CSV, config.REFERENCE_TABLE_CSV)

        read_labels = build_read_labels(query_results, ref_lookup, depth, args.min_similarity)
        summary = annotate_clusters(cluster_assignments, read_labels)
        metrics = compute_agreement_metrics(cluster_assignments, read_labels)

        write_cluster_annotation(cluster_assignments, read_labels, config.CLUSTER_ANNOTATION_CSV)
        write_cluster_summary(summary, config.CLUSTER_SUMMARY_CSV)

        print_cluster_summary(summary)
        print()
        print("=== Cluster vs. SILVA agreement metrics ===")
        for k, v in metrics.items():
            print(f"{k:28s}: {v}")


if __name__ == "__main__":
    main()
