"""
Stage 8 -- Query the FAISS index with experimental read embeddings.

Retrieves top-K (config.TOP_K) hits per read so a margin-to-runner-up can be
computed downstream (Stage 9-10) -- not just the single nearest neighbor.

Usage:
    python -m src.query_index
"""
import csv
from typing import List, Tuple
import numpy as np

from . import config
from .faiss_index import l2_normalize


def query_index(index_path: str, read_ids: List[str], read_embeddings: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    import faiss

    index = faiss.read_index(index_path)
    q = l2_normalize(read_embeddings.astype("float32"))
    similarities, positions = index.search(q, config.TOP_K)  # both (N_reads, TOP_K)
    return similarities, positions


def write_query_results(read_ids, similarities, positions, out_csv: str):
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        cols = ["Read_ID"]
        for k in range(config.TOP_K):
            cols += [f"top{k+1}_index_pos", f"top{k+1}_similarity"]
        w.writerow(cols)
        for rid, sims, pos in zip(read_ids, similarities, positions):
            row = [rid]
            for k in range(config.TOP_K):
                row += [int(pos[k]), float(sims[k])]
            w.writerow(row)
    print(f"Queried {len(read_ids)} reads, top-{config.TOP_K} results written to {out_csv}")


if __name__ == "__main__":
    import sys

    ids_path = sys.argv[1] if len(sys.argv) > 1 else "data/reads_ids.csv"  # one Read_ID per line
    emb_path = sys.argv[2] if len(sys.argv) > 2 else config.EXPERIMENTAL_EMBEDDINGS_NPY
    out_csv = sys.argv[3] if len(sys.argv) > 3 else "data/query_results.csv"

    with open(ids_path) as f:
        read_ids = [line.strip() for line in f if line.strip()]
    read_embeddings = np.load(emb_path)

    sims, pos = query_index(config.FAISS_INDEX_PATH, read_ids, read_embeddings)
    write_query_results(read_ids, sims, pos, out_csv)
