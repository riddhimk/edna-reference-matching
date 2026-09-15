"""
Stage 3, Task 1 -- BM25 top-10 redundancy diagnostic.

Directly mirrors Stage 0's within-top-10 redundancy methodology
(src/stage0_diagnostics.py: _pairwise_upper_tri_cosine, _dist_summary,
_near_dup_counts, run_background_baseline, NEAR_DUP_THRESHOLDS -- all
imported unmodified, not reimplemented) so the BM25 numbers are directly
comparable to the already-reported DNABERT numbers. Stage 0 measured
redundancy on DNABERT's own top-10 candidate sets; this measures it on
BM25(k=6)'s top-10 candidate sets instead -- the retrieval arm Stage 1/2
established is the stronger one, so this is the redundancy question that
actually matters for deciding whether MMR has anything to fix.

Representation: the SAME one Stage 0 used -- pairwise cosine similarity
among the existing, frozen DNABERT reference embeddings
(data/MitoFish results/mitofish_16s_ref_embeddings.npy), accessed via
split['train_emb'] (already train-space-indexed and L2-normalized by
build_holdout_split()). No new representation is introduced, per "use
the most appropriate existing sequence representation already present in
the project."

Uses the IDENTICAL held-out split as Stage 0/1/2
(stage0_diagnostics.build_holdout_split, seed=42) and BM25 k=6 (the value
Stage 1 selected on its own held-out sweep), via
stage2_rrf_diagnostics.get_bm25_top10_lists -- reused unmodified, not
reimplemented a third time.

Also reports, as a free cross-check (no new alignments): what fraction
of near-duplicate-by-embedding pairs are ALSO exact sequence-string
duplicates -- grounding the "near-duplicate" claim in real sequence
redundancy already documented for MitoFish (Part 3), not just an
embedding-space threshold.

Does not modify Stage 0/1/2 output files, BM25, the Part 4 pipeline,
MitoFish database/primers/extracted region, the 95% identity threshold,
or the nucleotide reranker. Writes only to data/Stage3 diagnostics/.

Usage:
    python -m src.bm25_redundancy
"""
import csv
import os
import numpy as np

from .stage0_diagnostics import (
    build_holdout_split, _pairwise_upper_tri_cosine, _dist_summary,
    _near_dup_counts, run_background_baseline, NEAR_DUP_THRESHOLDS,
)
from .stage2_rrf_diagnostics import get_bm25_top10_lists, BM25_K_MER

OUT_DIR = "data/Stage3 diagnostics"


def run():
    os.makedirs(OUT_DIR, exist_ok=True)
    print("=== Task 1: BM25 top-10 redundancy diagnostic ===")
    print("Read-only reuse of Stage 0's holdout split/methodology and Stage 2's BM25 retrieval.\n")

    split = build_holdout_split()
    train_emb = split["train_emb"]  # already train-space-indexed, L2-normalized
    train_seqs = split["train_seqs"]
    n_frag = len(split["synth_seqs"])

    print(f"Retrieving BM25 (k={BM25_K_MER}) top-10 on {n_frag} holdout fragments (same as Stage 1/2)...")
    bm25_lists = get_bm25_top10_lists(split)

    pairwise_chunks = []
    rows = []
    n_exact_dup_pairs_total = 0
    n_pairs_checked_total = 0
    n_sets_too_small = 0

    for i, doc_idxs in enumerate(bm25_lists):
        if len(doc_idxs) < 2:
            n_sets_too_small += 1
            continue  # cannot form a pair (rare: only 5.4% of sets had <10 candidates at all per Stage 1)

        block = train_emb[doc_idxs]
        pw = _pairwise_upper_tri_cosine(block)
        pairwise_chunks.append(pw)

        seqs_in_set = [train_seqs[idx] for idx in doc_idxs]
        n_pairs = 0
        n_exact = 0
        for a in range(len(seqs_in_set)):
            for b in range(a + 1, len(seqs_in_set)):
                n_pairs += 1
                if seqs_in_set[a] == seqs_in_set[b]:
                    n_exact += 1
        n_exact_dup_pairs_total += n_exact
        n_pairs_checked_total += n_pairs

        row = {
            "fragment_idx": i,
            "set_size": len(doc_idxs),
            "mean_pairwise_sim": float(pw.mean()),
            "max_pairwise_sim": float(pw.max()),
            "n_exact_string_dup_pairs": n_exact,
        }
        row.update(_near_dup_counts(pw))
        rows.append(row)

    pairwise_all = np.concatenate(pairwise_chunks)
    bm25_summary = _dist_summary(pairwise_all)

    print("Computing random-pair background baseline (same as Stage 0)...")
    background_summary = run_background_baseline()

    def _set_level_rate(thresh):
        key = f"n_pairs_ge_{thresh}"
        n_sets = len(rows)
        n_with = sum(1 for r in rows if r[key] > 0)
        return n_with / n_sets if n_sets else 0.0

    set_level = {f"bm25_pct_sets_with_pair_ge_{t}": _set_level_rate(t) for t in NEAR_DUP_THRESHOLDS}
    exact_dup_rate = n_exact_dup_pairs_total / n_pairs_checked_total if n_pairs_checked_total else 0.0

    with open(f"{OUT_DIR}/bm25_redundancy_per_fragment.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    with open(f"{OUT_DIR}/bm25_redundancy_summary.txt", "w") as f:
        f.write(f"=== Task 1: BM25 (k={BM25_K_MER}) top-10 redundancy -- held-out fragments, n={n_frag} ===\n")
        f.write(f"(sets with <2 candidates, excluded from pairwise stats: {n_sets_too_small})\n\n")
        f.write("--- BM25 top-10 pairwise embedding-cosine-similarity distribution ---\n")
        for k, v in bm25_summary.items():
            f.write(f"{k:12s}: {v}\n")
        f.write("\n--- Background: random reference pairs, not conditioned on co-retrieval (same as Stage 0) ---\n")
        for k, v in background_summary.items():
            f.write(f"{k:12s}: {v}\n")
        f.write("\n--- Set-level near-duplicate rate (>=1 pair in the set exceeds threshold) ---\n")
        for k, v in set_level.items():
            f.write(f"{k:45s}: {v}\n")
        f.write("\n--- Cross-check: exact sequence-string duplicate pairs (free, no alignment) ---\n")
        f.write(f"exact_string_dup_pair_rate  : {exact_dup_rate} ({n_exact_dup_pairs_total}/{n_pairs_checked_total} pairs)\n")
        f.write("\n--- For reference: Stage 0's DNABERT top-10 numbers (already reported, not recomputed here) ---\n")
        f.write("(data/Stage0 diagnostics/redundancy_diagnostic_summary.txt -- holdout mean=0.859, real-reads mean=0.791)\n")

    print("\n--- BM25 top-10 pairwise-similarity distribution ---")
    for k, v in bm25_summary.items():
        print(f"{k:12s}: {v}")
    print("\n--- Background (random, non-co-retrieved) ---")
    for k, v in background_summary.items():
        print(f"{k:12s}: {v}")
    print("\n--- Set-level near-duplicate rates ---")
    for k, v in set_level.items():
        print(f"{k:45s}: {v}")
    print(f"\nExact-string-duplicate pair rate: {exact_dup_rate:.4f} ({n_exact_dup_pairs_total}/{n_pairs_checked_total})")

    print("\nDone. Outputs written to:", OUT_DIR)
    return bm25_summary, background_summary, set_level, exact_dup_rate


if __name__ == "__main__":
    run()
