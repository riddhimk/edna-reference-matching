"""
Stage 0 diagnostics -- prerequisite measurements from the Final Design
Review, gating whether BM25 (Stage 1) or MMR (Stage 3) are worth building
at all. This module is NOT part of the Part 4 pipeline: it only READS
frozen Part 3/4 artifacts (MitoFish reference table, embeddings, FAISS
index, real-read top-10 retrieval results) and writes its own outputs to
a separate results folder. It does not modify, rerun, or recalibrate
Part 4, and does not touch the database, primers, extracted region,
nucleotide_rerank.py, or the 95% identity threshold.

Diagnostic 1 -- DNABERT Recall@K on held-out MitoFish calibration
fragments. Reproduces the IDENTICAL held-out split / synthetic-fragment
generation as src/mitofish_rerank_calibration.py (same seed=42,
holdout_frac=0.2, n_test=1000, REAL_READ_LENGTHS=118-140bp, TOP_K=10,
train-only FAISS index, >=95% full-sequence-identity ground truth via
nucleotide_rerank.align_pair) so the result is directly comparable to the
frozen Part 4 calibration curve, not a new, non-comparable sample. This
measures whether the correct reference is anywhere in DNABERT's top-K
BEFORE nucleotide reranking -- a question Part 4's calibration never
isolated (it only reports accuracy of the rerank-chosen candidate, which
conflates "wrong candidate chosen from a correct top-10" with "correct
reference never retrieved at all").

Diagnostic 2 -- within-top-10 pairwise embedding-cosine redundancy, for
real reads (reusing the FROZEN data/MitoFish results/reads_top10_pos.npy,
no re-embedding) and for the same holdout fragments as Diagnostic 1
(reusing Diagnostic 1's own retrieval, no separate re-embedding). Also
reports a random-pair background baseline over the full reference table,
so a within-top-10 similarity threshold can be judged as elevated
relative to the general population, not against an arbitrary absolute
cutoff.

Usage:
    python -m src.stage0_diagnostics
"""
import csv
import os
import random
import numpy as np
import faiss

from . import config
from .embed_common import embed_batch
from .faiss_index import l2_normalize
from .nucleotide_rerank import align_pair

REF_TABLE_CSV = "data/MitoFish results/mitofish_16s_reference_table.csv"
REF_EMB_NPY = "data/MitoFish results/mitofish_16s_ref_embeddings.npy"
REF_IDS_CSV = "data/MitoFish results/mitofish_16s_ref_ids.csv"

REAL_READS_TOP10_POS_NPY = "data/MitoFish results/reads_top10_pos.npy"
REAL_READS_TOP10_IDS_CSV = "data/MitoFish results/reads_top10_ids.csv"

OUT_DIR = "data/Stage0 diagnostics"

REAL_READ_LENGTHS = list(range(118, 141))  # measured from ERR16922870_1_first5000.fastq -- same as mitofish_rerank_calibration.py
TOP_K = 10
HOLDOUT_FRAC = 0.2
SEED = 42
N_TEST = 1000
NEAR_DUP_THRESHOLDS = (0.999, 0.99, 0.95)


def _load_reference_table():
    ids, seqs = [], []
    with open(REF_TABLE_CSV) as f:
        for row in csv.DictReader(f):
            ids.append(row["Reference_ID"])
            seqs.append(row["Sequence"])
    return ids, seqs


def _is_hit(candidate_seq: str, true_seq: str) -> bool:
    """Same >=95% full-sequence-identity ground truth as
    mitofish_rerank_calibration.py's correctness check: exact string
    match is a hit (fast path -- MitoFish's documented accession
    redundancy makes this common), otherwise fall back to a real
    alignment via the UNMODIFIED nucleotide_rerank.align_pair."""
    if candidate_seq == true_seq:
        return True
    return align_pair(candidate_seq, true_seq)["identity_pct"] >= 95.0


def build_holdout_split(seed: int = SEED, holdout_frac: float = HOLDOUT_FRAC):
    """Reproduces the exact train/holdout split and synthetic-fragment
    generation from mitofish_rerank_calibration.py so this diagnostic is
    measured on the SAME held-out fragments Part 4's frozen calibration
    used."""
    rng = random.Random(seed)
    np_rng = np.random.RandomState(seed)

    ids, seqs = _load_reference_table()
    all_emb = np.load(REF_EMB_NPY).astype("float32")
    with open(REF_IDS_CSV) as f:
        emb_ids = [row["Reference_ID"] for row in csv.DictReader(f)]
    assert emb_ids == ids, "ref_ids.csv order does not match reference table order"

    n = len(ids)
    perm = np_rng.permutation(n)
    n_holdout = int(n * holdout_frac)
    holdout_idx = perm[:n_holdout]
    train_idx = perm[n_holdout:]
    train_ids = [ids[i] for i in train_idx]
    train_seqs = [seqs[i] for i in train_idx]

    train_emb = l2_normalize(all_emb[train_idx])
    index = faiss.IndexFlatIP(train_emb.shape[1])
    index.add(train_emb)

    holdout_idx = list(holdout_idx)[:N_TEST]
    synth_seqs, synth_true_seq = [], []
    for i in holdout_idx:
        seq = seqs[i]
        L = rng.choice(REAL_READ_LENGTHS)
        if len(seq) <= L:
            continue
        start = rng.randint(0, len(seq) - L)
        synth_seqs.append(seq[start:start + L])
        synth_true_seq.append(seq)

    return {
        "train_ids": train_ids, "train_seqs": train_seqs, "train_emb": train_emb,
        "index": index, "synth_seqs": synth_seqs, "synth_true_seq": synth_true_seq,
    }


def run_recall_diagnostic(split):
    synth_seqs = split["synth_seqs"]
    synth_true_seq = split["synth_true_seq"]
    train_seqs = split["train_seqs"]
    index = split["index"]

    print(f"Embedding {len(synth_seqs)} synthetic holdout fragments (DNABERT)...")
    synth_emb = []
    for i in range(0, len(synth_seqs), config.BATCH_SIZE):
        batch_vecs = embed_batch(synth_seqs[i:i + config.BATCH_SIZE])
        synth_emb.append(batch_vecs)
        print(f"  embedded {i + len(batch_vecs)}/{len(synth_seqs)}")
    synth_emb = l2_normalize(np.vstack(synth_emb).astype("float32"))

    # rank order here is pure DNABERT cosine-similarity order -- NOT
    # identity-reranked -- this is deliberately the pre-rerank signal.
    sims, pos = index.search(synth_emb, TOP_K)

    earliest_hit_rank = []
    per_fragment_rows = []
    for i in range(len(synth_seqs)):
        hit_rank = None
        for k in range(TOP_K):
            cand_seq = train_seqs[pos[i, k]]
            if _is_hit(cand_seq, synth_true_seq[i]):
                hit_rank = k + 1
                break
        earliest_hit_rank.append(hit_rank)
        per_fragment_rows.append({
            "fragment_idx": i,
            "fragment_len": len(synth_seqs[i]),
            "earliest_hit_rank": hit_rank if hit_rank is not None else "",
        })
        if (i + 1) % 100 == 0:
            print(f"  recall-checked {i + 1}/{len(synth_seqs)}")

    n = len(earliest_hit_rank)

    def recall_at(k):
        return sum(1 for r in earliest_hit_rank if r is not None and r <= k) / n

    n_absent = sum(1 for r in earliest_hit_rank if r is None)

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(f"{OUT_DIR}/recall_diagnostic_per_fragment.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["fragment_idx", "fragment_len", "earliest_hit_rank"])
        w.writeheader()
        w.writerows(per_fragment_rows)

    summary = {
        "n_holdout_fragments": n,
        "recall_at_1": recall_at(1),
        "recall_at_5": recall_at(5),
        "recall_at_10": recall_at(10),
        "n_absent_from_top10": n_absent,
        "pct_absent_from_top10": n_absent / n,
    }
    with open(f"{OUT_DIR}/recall_diagnostic_summary.txt", "w") as f:
        f.write("=== Stage 0 Diagnostic 1: DNABERT Recall@K on held-out MitoFish fragments ===\n")
        f.write("(ground truth: >=95% full-sequence identity -- SAME definition as Part 4 calibration)\n")
        f.write("(same holdout split/seed/fragment methodology as src/mitofish_rerank_calibration.py)\n\n")
        for k, v in summary.items():
            f.write(f"{k:28s}: {v}\n")
    print()
    for k, v in summary.items():
        print(f"{k:28s}: {v}")

    return summary, pos


def _pairwise_upper_tri_cosine(emb_block: np.ndarray) -> np.ndarray:
    """emb_block: (K, dim) L2-normalized embeddings for one candidate set.
    Returns the K*(K-1)/2 upper-triangle pairwise cosine similarities."""
    sim_matrix = emb_block @ emb_block.T
    iu = np.triu_indices(emb_block.shape[0], k=1)
    return sim_matrix[iu]


def _dist_summary(arr: np.ndarray) -> dict:
    pct = np.percentile(arr, [5, 25, 50, 75, 90, 95])
    return {
        "n_pairs": int(len(arr)),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "p5": float(pct[0]), "p25": float(pct[1]), "p50": float(pct[2]),
        "p75": float(pct[3]), "p90": float(pct[4]), "p95": float(pct[5]),
        "min": float(arr.min()), "max": float(arr.max()),
    }


def _near_dup_counts(pw: np.ndarray) -> dict:
    return {f"n_pairs_ge_{t}": int((pw >= t).sum()) for t in NEAR_DUP_THRESHOLDS}


def run_background_baseline(n_pairs: int = 50000, seed: int = SEED) -> dict:
    """Random reference-pair cosine similarity, NOT conditioned on
    co-retrieval -- context for judging whether within-top-10 similarity
    is elevated relative to the general population, rather than against
    an arbitrary absolute cutoff."""
    ref_emb_all = l2_normalize(np.load(REF_EMB_NPY).astype("float32"))
    n_refs = ref_emb_all.shape[0]
    rng = np.random.RandomState(seed)
    i_idx = rng.randint(0, n_refs, n_pairs)
    j_idx = rng.randint(0, n_refs, n_pairs)
    mask = i_idx != j_idx
    i_idx, j_idx = i_idx[mask], j_idx[mask]
    sims = np.sum(ref_emb_all[i_idx] * ref_emb_all[j_idx], axis=1)
    return _dist_summary(sims)


def run_redundancy_diagnostic(split, holdout_pos):
    os.makedirs(OUT_DIR, exist_ok=True)

    # --- Real reads: reuse FROZEN Part 4 top-10 retrieval, no re-embedding ---
    ref_emb_all = l2_normalize(np.load(REF_EMB_NPY).astype("float32"))
    real_pos = np.load(REAL_READS_TOP10_POS_NPY)
    with open(REAL_READS_TOP10_IDS_CSV) as f:
        real_read_ids = [l.strip() for l in f if l.strip()]
    assert real_pos.shape[0] == len(real_read_ids)
    assert real_pos.shape[1] == TOP_K

    real_pairwise_chunks, real_rows = [], []
    for i in range(real_pos.shape[0]):
        block = ref_emb_all[real_pos[i]]
        pw = _pairwise_upper_tri_cosine(block)
        real_pairwise_chunks.append(pw)
        row = {"Read_ID": real_read_ids[i], "mean_pairwise_sim": float(pw.mean()), "max_pairwise_sim": float(pw.max())}
        row.update(_near_dup_counts(pw))
        real_rows.append(row)
    real_pairwise = np.concatenate(real_pairwise_chunks)

    # --- Holdout fragments: reuse pos/train_emb from Diagnostic 1, no re-embedding ---
    train_emb = split["train_emb"]
    holdout_pairwise_chunks, holdout_rows = [], []
    for i in range(holdout_pos.shape[0]):
        block = train_emb[holdout_pos[i]]
        pw = _pairwise_upper_tri_cosine(block)
        holdout_pairwise_chunks.append(pw)
        row = {"fragment_idx": i, "mean_pairwise_sim": float(pw.mean()), "max_pairwise_sim": float(pw.max())}
        row.update(_near_dup_counts(pw))
        holdout_rows.append(row)
    holdout_pairwise = np.concatenate(holdout_pairwise_chunks)

    real_summary = _dist_summary(real_pairwise)
    holdout_summary = _dist_summary(holdout_pairwise)
    print("Computing random-pair background baseline...")
    background_summary = run_background_baseline()

    def _set_level_rate(rows, thresh):
        key = f"n_pairs_ge_{thresh}"
        n_sets = len(rows)
        n_with = sum(1 for r in rows if r[key] > 0)
        return n_with / n_sets if n_sets else 0.0

    set_level = {}
    for thresh in NEAR_DUP_THRESHOLDS:
        set_level[f"real_reads_pct_sets_with_pair_ge_{thresh}"] = _set_level_rate(real_rows, thresh)
    for thresh in NEAR_DUP_THRESHOLDS:
        set_level[f"holdout_pct_sets_with_pair_ge_{thresh}"] = _set_level_rate(holdout_rows, thresh)

    with open(f"{OUT_DIR}/redundancy_diagnostic_real_reads.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(real_rows[0].keys()))
        w.writeheader()
        w.writerows(real_rows)
    with open(f"{OUT_DIR}/redundancy_diagnostic_holdout.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(holdout_rows[0].keys()))
        w.writeheader()
        w.writerows(holdout_rows)

    with open(f"{OUT_DIR}/redundancy_diagnostic_summary.txt", "w") as f:
        f.write("=== Stage 0 Diagnostic 2: within-top-10 pairwise embedding-cosine redundancy ===\n\n")
        f.write("--- Real reads (n=5000, FROZEN reads_top10_pos.npy, no re-embedding) ---\n")
        for k, v in real_summary.items():
            f.write(f"{k:12s}: {v}\n")
        f.write("\n--- Holdout fragments (same set as recall diagnostic) ---\n")
        for k, v in holdout_summary.items():
            f.write(f"{k:12s}: {v}\n")
        f.write("\n--- Background: random reference PAIRS, not conditioned on co-retrieval ---\n")
        for k, v in background_summary.items():
            f.write(f"{k:12s}: {v}\n")
        f.write("\n--- Set-level near-duplicate rate (>=1 of the 45 in-set pairs exceeds threshold) ---\n")
        for k, v in set_level.items():
            f.write(f"{k:45s}: {v}\n")

    print("\n--- Real reads pairwise-similarity distribution ---")
    for k, v in real_summary.items():
        print(f"{k:12s}: {v}")
    print("\n--- Holdout pairwise-similarity distribution ---")
    for k, v in holdout_summary.items():
        print(f"{k:12s}: {v}")
    print("\n--- Background (random, non-co-retrieved) pairwise-similarity distribution ---")
    for k, v in background_summary.items():
        print(f"{k:12s}: {v}")
    print("\n--- Set-level near-duplicate rates ---")
    for k, v in set_level.items():
        print(f"{k:45s}: {v}")

    return real_summary, holdout_summary, background_summary, set_level


def main():
    print("=== Stage 0 diagnostics (read-only reuse of frozen Part 3/4 artifacts) ===")
    print("Does not modify, rerun, or recalibrate Part 4.\n")
    split = build_holdout_split()
    recall_summary, holdout_pos = run_recall_diagnostic(split)
    run_redundancy_diagnostic(split, holdout_pos)
    print("\nDone. Outputs written to:", OUT_DIR)


if __name__ == "__main__":
    main()
