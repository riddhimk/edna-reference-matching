"""
Orchestrates Stages 7-10 (build-index, query, decide) for a region-specific
SILVA reference set against the ALREADY-COMPUTED real experimental read
embeddings, plus prints the similarity-score distribution needed for the
16S-vs-18S-vs-full-length comparison report.

Reuses faiss_index.build_index / query_index.query_index /
confidence.build_results as-is -- this is a thin CLI wrapper, not a
reimplementation.

Usage:
    python -m src.run_region_experiment <region_name> <ref_table_csv> \
        <ref_embeddings_npy> <ref_ids_csv> <similarity_threshold> \
        [reads_ids_csv] [reads_embeddings_npy]
"""
import csv
import sys
import numpy as np

from .faiss_index import build_index
from .query_index import query_index
from .confidence import build_results


def run(region_name: str, ref_table_csv: str, ref_emb_npy: str, ref_ids_csv: str,
        threshold: float, reads_ids_csv: str = "data/reads_ids.csv",
        reads_emb_npy: str = "data/reads_embeddings.npy", out_dir: str = "data"):
    index_path = f"{out_dir}/{region_name}_faiss.index"
    build_index(ref_emb_npy, index_path)

    with open(reads_ids_csv) as f:
        read_ids = [line.strip() for line in f if line.strip()]
    read_embeddings = np.load(reads_emb_npy)

    sims, pos = query_index(index_path, read_ids, read_embeddings)
    query_out = f"{out_dir}/query_results_{region_name}.csv"
    with open(query_out, "w", newline="") as f:
        w = csv.writer(f)
        cols = ["Read_ID", "top1_index_pos", "top1_similarity", "top2_index_pos", "top2_similarity"]
        w.writerow(cols)
        for rid, s, p in zip(read_ids, sims, pos):
            w.writerow([rid, int(p[0]), float(s[0]), int(p[1]), float(s[1])])

    top1 = sims[:, 0]
    print(f"\n=== {region_name} real-read similarity distribution (n={len(top1)}) ===")
    print(f"min={top1.min():.4f} p25={np.percentile(top1,25):.4f} median={np.median(top1):.4f} "
          f"p75={np.percentile(top1,75):.4f} p95={np.percentile(top1,95):.4f} max={top1.max():.4f} "
          f"mean={top1.mean():.4f}")

    results_out = f"{out_dir}/read_taxonomy_results_{region_name}.csv"
    results = build_results(
        query_results_csv=query_out,
        ref_ids_csv=ref_ids_csv,
        ref_table_csv=ref_table_csv,
        threshold=threshold,
        min_margin=None,
        out_csv=results_out,
    )
    return top1, results


if __name__ == "__main__":
    region_name = sys.argv[1]
    ref_table_csv = sys.argv[2]
    ref_emb_npy = sys.argv[3]
    ref_ids_csv = sys.argv[4]
    threshold = float(sys.argv[5])
    reads_ids_csv = sys.argv[6] if len(sys.argv) > 6 else "data/reads_ids.csv"
    reads_emb_npy = sys.argv[7] if len(sys.argv) > 7 else "data/reads_embeddings.npy"
    out_dir = sys.argv[8] if len(sys.argv) > 8 else "data"
    run(region_name, ref_table_csv, ref_emb_npy, ref_ids_csv, threshold, reads_ids_csv, reads_emb_npy, out_dir)
