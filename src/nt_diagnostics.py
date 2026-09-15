"""
Model-vs-model comparison: Nucleotide Transformer v2 (50M, multi-species,
src/nt_embed.py) vs. the existing DNABERT-S/multimolecule pipeline, on the
IDENTICAL held-out split and synthetic fragments src/stage0_diagnostics.py
already established for DNABERT (same seed=42, holdout_frac=0.2, 118-140bp
fragment generation) -- reused via stage0_diagnostics.build_holdout_split,
UNMODIFIED, so Recall@K numbers here are directly comparable to
data/Stage0 diagnostics/recall_diagnostic_summary.txt without re-deriving
the split or cherry-picking a different, easier test set.

Ground truth ("is this candidate actually a hit") uses the same >=95%
full-sequence-identity check as every prior stage, via
nucleotide_rerank.align_pair (frozen, unmodified).

Does not modify, rerun, or recalibrate any existing stage, model, config,
or frozen Part 3/4 artifact. src/nt_embed.py and this script are the only
new files touched by this comparison.

Usage:
    python -m src.nt_diagnostics
"""
import csv
import os
import time
import numpy as np
import faiss

from .stage0_diagnostics import build_holdout_split
from .nucleotide_rerank import align_pair
from .nt_embed import embed_batch as nt_embed_batch, EMBEDDING_DIM as NT_DIM
from .faiss_index import l2_normalize

OUT_DIR = "data/NT diagnostics"
TOP_K = 10
BATCH_SIZE = 64


def _is_hit(candidate_seq: str, true_seq: str) -> bool:
    """Same >=95% full-sequence-identity ground truth as
    stage0_diagnostics.py's DNABERT diagnostic -- kept as an identical
    copy (rather than importing stage0_diagnostics' private helper) so
    this file's ground truth is self-contained and auditable."""
    if candidate_seq == true_seq:
        return True
    return align_pair(candidate_seq, true_seq)["identity_pct"] >= 95.0


def _embed_all(seqs, label):
    vecs = []
    t0 = time.time()
    for i in range(0, len(seqs), BATCH_SIZE):
        batch = nt_embed_batch(seqs[i:i + BATCH_SIZE])
        vecs.append(batch)
        if (i // BATCH_SIZE) % 20 == 0:
            print(f"  [{label}] embedded {i + len(batch)}/{len(seqs)}  ({time.time() - t0:.0f}s elapsed)")
    return np.vstack(vecs).astype("float32")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print("=== Nucleotide Transformer v2 (50M multi-species) vs DNABERT: Recall@K ===")
    print("Held-out split/fragments reused unmodified from stage0_diagnostics.build_holdout_split.\n")

    split = build_holdout_split()  # text-only reuse; this split's own DNABERT embeddings are ignored here
    train_ids, train_seqs = split["train_ids"], split["train_seqs"]
    synth_seqs, synth_true_seq = split["synth_seqs"], split["synth_true_seq"]

    print(f"Embedding {len(train_seqs)} reference (train) sequences with Nucleotide Transformer...")
    train_emb_nt = l2_normalize(_embed_all(train_seqs, "refs"))

    print(f"\nEmbedding {len(synth_seqs)} synthetic holdout fragments with Nucleotide Transformer...")
    synth_emb_nt = l2_normalize(_embed_all(synth_seqs, "frags"))

    index = faiss.IndexFlatIP(NT_DIM)
    index.add(train_emb_nt)
    _sims, pos = index.search(synth_emb_nt, TOP_K)

    earliest_hit_rank = []
    for i in range(len(synth_seqs)):
        hit_rank = None
        for k in range(TOP_K):
            if _is_hit(train_seqs[pos[i, k]], synth_true_seq[i]):
                hit_rank = k + 1
                break
        earliest_hit_rank.append(hit_rank)
        if (i + 1) % 200 == 0:
            print(f"  recall-checked {i + 1}/{len(synth_seqs)}")

    n = len(earliest_hit_rank)

    def recall_at(k):
        return sum(1 for r in earliest_hit_rank if r is not None and r <= k) / n

    n_absent = sum(1 for r in earliest_hit_rank if r is None)

    summary = {
        "n_holdout_fragments": n,
        "recall_at_1": recall_at(1),
        "recall_at_5": recall_at(5),
        "recall_at_10": recall_at(10),
        "n_absent_from_top10": n_absent,
        "pct_absent_from_top10": n_absent / n,
    }

    np.save(f"{OUT_DIR}/nt_train_embeddings.npy", train_emb_nt)
    with open(f"{OUT_DIR}/nt_train_ids.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Reference_ID"])
        for rid in train_ids:
            w.writerow([rid])

    with open(f"{OUT_DIR}/nt_recall_summary.txt", "w") as f:
        f.write("=== Nucleotide Transformer v2 (50M multi-species) Recall@K ===\n")
        f.write("(IDENTICAL held-out split/fragments as data/Stage0 diagnostics/recall_diagnostic_summary.txt)\n\n")
        for k, v in summary.items():
            f.write(f"{k:28s}: {v}\n")

    print()
    for k, v in summary.items():
        print(f"{k:28s}: {v}")
    print("\nDone. Outputs written to:", OUT_DIR)
    return summary


if __name__ == "__main__":
    main()
