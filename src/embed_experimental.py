"""
Stage 6 -- Embed experimental FASTQ reads with the SAME DNABERT-S procedure
used for the SILVA reference table (embed_common.embed_batch).

Only run this if your EXISTING 5,000-read embeddings were produced with a
DIFFERENT tokenizer/pooling configuration than the one fixed in
embed_common.py. If they already match, reuse the existing .npy directly and
do not re-run -- the original POC results must stay unchanged.

Usage:
    python -m src.embed_experimental [reads.fastq] [out_embeddings.npy]
"""
import sys
import numpy as np

from . import config
from .embed_common import embed_batch


def read_fastq(path: str):
    """Yield (Read_ID, sequence) from a 4-line-per-record FASTQ file."""
    with open(path) as f:
        while True:
            header = f.readline()
            if not header:
                break
            seq = f.readline().strip()
            f.readline()  # '+' separator line
            f.readline()  # quality line
            read_id = header.strip().lstrip("@").split()[0]
            yield read_id, seq


def embed_reads(fastq_path: str, out_npy: str):
    ids, seqs = [], []
    for rid, seq in read_fastq(fastq_path):
        ids.append(rid)
        seqs.append(seq)

    vecs = []
    for i in range(0, len(seqs), config.BATCH_SIZE):
        vecs.append(embed_batch(seqs[i:i + config.BATCH_SIZE]))
        print(f"embedded {i + len(vecs[-1])}/{len(seqs)}")

    embeddings = np.vstack(vecs).astype("float32") if vecs else np.zeros((0, config.EMBEDDING_DIM), dtype="float32")
    np.save(out_npy, embeddings)
    print(f"Saved {embeddings.shape} to {out_npy}")
    return ids, embeddings


if __name__ == "__main__":
    fastq_path = sys.argv[1] if len(sys.argv) > 1 else config.EXPERIMENTAL_FASTQ_PATH
    out_npy = sys.argv[2] if len(sys.argv) > 2 else config.EXPERIMENTAL_EMBEDDINGS_NPY
    embed_reads(fastq_path, out_npy)
