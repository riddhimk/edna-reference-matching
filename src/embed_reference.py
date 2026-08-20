"""
Stage 5 -- Generate and store DNABERT-S embeddings for the SILVA reference table.

Usage:
    python -m src.embed_reference [reference_table.csv] [out_embeddings.npy] [out_ids.csv]

Verify before proceeding to Stage 7: embeddings.shape[0] must equal the
"parsed" count printed by build_reference_table.py, and shape[1] must be
config.EMBEDDING_DIM (768).
"""
import csv
import sys
import numpy as np

from . import config
from .embed_common import embed_batch


def embed_reference_table(csv_path: str, out_npy: str, out_ids_csv: str) -> np.ndarray:
    ids, seqs = [], []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            ids.append(row["Reference_ID"])
            seqs.append(row["Sequence"])

    all_vecs = []
    for i in range(0, len(seqs), config.BATCH_SIZE):
        batch = seqs[i:i + config.BATCH_SIZE]
        vecs = embed_batch(batch)
        all_vecs.append(vecs)
        print(f"embedded {i + len(batch)}/{len(seqs)}")

    embeddings = np.vstack(all_vecs).astype("float32") if all_vecs else np.zeros((0, config.EMBEDDING_DIM), dtype="float32")
    np.save(out_npy, embeddings)

    with open(out_ids_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["row_index", "Reference_ID"])
        for idx, rid in enumerate(ids):
            w.writerow([idx, rid])

    print(f"Saved {embeddings.shape} to {out_npy}")
    assert embeddings.shape[1] == config.EMBEDDING_DIM, "embedding dim mismatch -- check model/config"
    return embeddings


if __name__ == "__main__":
    csv_path = sys.argv[1] if len(sys.argv) > 1 else config.REFERENCE_TABLE_CSV
    out_npy = sys.argv[2] if len(sys.argv) > 2 else config.REFERENCE_EMBEDDINGS_NPY
    out_ids = sys.argv[3] if len(sys.argv) > 3 else config.REFERENCE_IDS_CSV
    embed_reference_table(csv_path, out_npy, out_ids)
