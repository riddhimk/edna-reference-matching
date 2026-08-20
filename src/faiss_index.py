"""
Stage 7 -- Build a FAISS index over the SILVA reference embeddings.

Cosine similarity, not raw Euclidean/L2 -- implemented via L2-normalized
vectors + inner product (IndexFlatIP), which is mathematically equivalent to
cosine similarity on unit vectors. Exact flat index is appropriate at
capstone reference-set scale; only move to an approximate index (e.g.
IndexIVFFlat) if a real speed benchmark against BLAST shows it's needed.

Usage:
    python -m src.faiss_index [ref_embeddings.npy] [out_index_path]
"""
import sys
import numpy as np

from . import config


def l2_normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0] = 1e-12  # avoid div-by-zero on a degenerate all-zero embedding
    return x / norms


def build_index(ref_embeddings_path: str, index_out_path: str):
    import faiss

    ref_emb = np.load(ref_embeddings_path).astype("float32")
    if ref_emb.shape[0] == 0:
        raise ValueError(f"No reference embeddings found in {ref_embeddings_path}")
    ref_emb = l2_normalize(ref_emb)

    index = faiss.IndexFlatIP(ref_emb.shape[1])
    index.add(ref_emb)
    faiss.write_index(index, index_out_path)

    print(f"FAISS index built: {index.ntotal} reference vectors, dim={ref_emb.shape[1]}")
    return index


if __name__ == "__main__":
    ref_npy = sys.argv[1] if len(sys.argv) > 1 else config.REFERENCE_EMBEDDINGS_NPY
    out_path = sys.argv[2] if len(sys.argv) > 2 else config.FAISS_INDEX_PATH
    build_index(ref_npy, out_path)
