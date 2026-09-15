"""
Stage 3, Tasks 2/3 -- MMR applied to a BM25 candidate pool, and a lambda
sweep.

Evaluates MMR (src/mmr.py), per held-out fragment, selecting a diverse
top-10 from BM25(k=6)'s own top-30 candidate pool -- the IDENTICAL 1,000
fragments and >=95% full-sequence-identity ground truth used in Stage
0/1/2/Task 1 (stage0_diagnostics.build_holdout_split, imported
unmodified). Per requirement, MMR operates independently within each
fragment's own BM25 pool, never across fragments; BM25, the reference
database, and the nucleotide reranker are untouched.

BM25's own top-pool (doc_idx, score) pairs are retrieved locally here
with the SAME BM25Index(k=6) Stage 1/2/Task1 already selected and used
(unmodified) -- not imported from stage2_rrf_diagnostics.get_bm25_top10_lists,
which discards scores; MMR needs the raw BM25 score as its relevance
signal.

For each lambda in {0.5, 0.6, 0.7, 0.8, 0.9}, reports:
  - Recall@1/5/10 of the MMR-selected top-10 (selected from the top-30 pool)
  - candidate-set diversity / redundancy, using the SAME pairwise-cosine-
    similarity metric as Task 1 (src/bm25_redundancy.py), so before/after
    MMR is a like-for-like comparison against Task 1's BM25-alone numbers.

Structural note, checked empirically below rather than assumed: for any
lambda > 0, MMR's first pick is always BM25's own top-relevance
candidate (the diversity term is 0 for everyone before any candidate is
selected) -- so Recall@1 is expected to be unchanged from BM25-alone at
every lambda in this sweep, and only Recall@5/10 (which depend on later,
diversity-influenced picks) can move.

IMPORTANT design correction made before running any experiment: MMR is
given a WIDER candidate pool (top-MMR_POOL_SIZE=30) than the final
K=TOP_K=10 it selects down to. Feeding MMR exactly BM25's own top-10 and
asking it to select 10 is a no-op for SET COMPOSITION -- with the pool
size equal to the output size, MMR can only reorder the same 10 items,
never trade a redundant one for a different one, so no redundancy metric
computed over the resulting SET (order-invariant, e.g. mean pairwise
similarity) could ever change. Retrieving a wider pool costs no extra
BM25 compute: BM25Index.score_query() already scores every candidate
sharing >=1 k-mer with the query regardless of k_results -- top_k() only
slices the sorted list at the end, so requesting 30 instead of 10 is
free. The "BM25 baseline (no MMR)" comparison point is still the
TRUE top-10 (the first 10 of that same sorted pool) -- identical to
Stage 1/Task 1's already-reported numbers, not recomputed differently.

Does not modify BM25, RRF, Stage 0/1/2 outputs, Task 1's output, the
Part 4 pipeline, MitoFish database/primers/extracted region, the 95%
threshold, or the nucleotide reranker. Writes only to
data/Stage3 diagnostics/.

Usage:
    python -m src.mmr_lambda_sweep
"""
import csv
import os
from typing import Dict, List, Tuple
import numpy as np

from .nucleotide_rerank import align_pair
from .bm25_retrieval import BM25Index
from .mmr import mmr_select
from .stage0_diagnostics import (
    build_holdout_split, TOP_K, _pairwise_upper_tri_cosine, _dist_summary,
    NEAR_DUP_THRESHOLDS, _near_dup_counts,
)
from .stage2_rrf_diagnostics import BM25_K_MER

OUT_DIR = "data/Stage3 diagnostics"
LAMBDA_SWEEP = (0.5, 0.6, 0.7, 0.8, 0.9)
MMR_POOL_SIZE = 30  # candidates MMR chooses FROM; must exceed TOP_K=10 or MMR degenerates to reordering only
RECALL10_TOLERANCE = 0.005  # 0.5 percentage points -- judgment call, stated explicitly
RECALL1_MAX_DROP = 0.05     # 5 percentage points -- judgment call, stated explicitly

HitCache = Dict[Tuple[int, int], bool]


def _is_hit(candidate_seq: str, true_seq: str) -> bool:
    """Identical >=95% full-sequence-identity ground truth used
    throughout Stage 0/1/2/Task 1."""
    if candidate_seq == true_seq:
        return True
    return align_pair(candidate_seq, true_seq)["identity_pct"] >= 95.0


def _earliest_hit_rank_cached(fragment_idx: int, doc_idx_list: List[int], true_seq: str,
                               train_seqs: List[str], hit_cache: HitCache, top_k: int = TOP_K):
    for k in range(min(top_k, len(doc_idx_list))):
        doc_idx = doc_idx_list[k]
        key = (fragment_idx, doc_idx)
        if key not in hit_cache:
            hit_cache[key] = _is_hit(train_seqs[doc_idx], true_seq)
        if hit_cache[key]:
            return k + 1
    return None


def _recall_at(earliest_ranks, k):
    n = len(earliest_ranks)
    return sum(1 for r in earliest_ranks if r is not None and r <= k) / n if n else 0.0


def get_bm25_pool_with_scores(split, k_mer: int = BM25_K_MER, pool_size: int = MMR_POOL_SIZE):
    """Retrieves BM25's top-pool_size (doc_idx, score) pairs per fragment.
    pool_size > TOP_K so MMR has real alternatives to trade against --
    see module docstring. This costs the same as retrieving top-10:
    BM25Index.score_query() scores every matching candidate regardless
    of how many are sliced off at the end."""
    train_ids = split["train_ids"]
    train_seqs = split["train_seqs"]
    synth_seqs = split["synth_seqs"]
    bm25 = BM25Index(k=k_mer)
    bm25.build(train_ids, train_seqs)  # TRAIN partition only -- leakage prevention
    results = []
    for i, frag in enumerate(synth_seqs):
        ranked = bm25.top_k(frag, k_results=pool_size)  # [(doc_idx, score), ...]
        results.append(ranked)
        if (i + 1) % 200 == 0:
            print(f"  [BM25 k={k_mer}] retrieved {i + 1}/{len(synth_seqs)}")
    return results


def redundancy_of_sets(doc_idx_lists, train_emb):
    pairwise_chunks = []
    rows = []
    for doc_idxs in doc_idx_lists:
        if len(doc_idxs) < 2:
            continue
        block = train_emb[doc_idxs]
        pw = _pairwise_upper_tri_cosine(block)
        pairwise_chunks.append(pw)
        row = {"mean_pairwise_sim": float(pw.mean())}
        row.update(_near_dup_counts(pw))
        rows.append(row)
    pairwise_all = np.concatenate(pairwise_chunks) if pairwise_chunks else np.array([0.0])
    dist = _dist_summary(pairwise_all)

    def _set_rate(thresh):
        key = f"n_pairs_ge_{thresh}"
        n_sets = len(rows)
        return sum(1 for r in rows if r[key] > 0) / n_sets if n_sets else 0.0

    set_level = {f"pct_sets_with_pair_ge_{t}": _set_rate(t) for t in NEAR_DUP_THRESHOLDS}
    return dist, set_level


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print("=== Tasks 2/3: MMR + lambda sweep (operating on BM25's own top-10, per fragment) ===")
    print("Read-only reuse of the Stage 0 holdout split. Does not modify BM25, RRF, or any prior stage.\n")

    split = build_holdout_split()
    train_emb = split["train_emb"]
    train_seqs = split["train_seqs"]
    synth_true_seq = split["synth_true_seq"]
    n_frag = len(split["synth_seqs"])
    hit_cache: HitCache = {}

    print(f"Retrieving BM25 (k={BM25_K_MER}) top-{MMR_POOL_SIZE} pool with scores on {n_frag} holdout fragments...")
    bm25_pool_ranked = get_bm25_pool_with_scores(split)  # list of [(doc_idx, score), ...], up to MMR_POOL_SIZE long

    # BM25 baseline (no MMR) = the TRUE top-10 -- first 10 of the same sorted pool,
    # identical to Stage 1/Task 1's already-reported numbers.
    bm25_lists = [[d for d, _s in ranked[:TOP_K]] for ranked in bm25_pool_ranked]
    bm25_earliest = [
        _earliest_hit_rank_cached(i, bm25_lists[i], synth_true_seq[i], train_seqs, hit_cache)
        for i in range(n_frag)
    ]
    bm25_recall = {
        "recall_at_1": _recall_at(bm25_earliest, 1),
        "recall_at_5": _recall_at(bm25_earliest, 5),
        "recall_at_10": _recall_at(bm25_earliest, 10),
    }
    bm25_dist, bm25_set_level = redundancy_of_sets(bm25_lists, train_emb)
    print("--- BM25 baseline (no MMR) ---")
    for k, v in bm25_recall.items():
        print(f"  {k}: {v}")
    print(f"  mean_pairwise_sim: {bm25_dist['mean']:.4f}  "
          f"pct_sets_near_dup(>=0.999): {bm25_set_level['pct_sets_with_pair_ge_0.999']:.4f}")

    # --- lambda sweep ---
    print("\nRunning MMR lambda sweep...")
    sweep_rows = []
    for lam in LAMBDA_SWEEP:
        selected_lists = []
        for i in range(n_frag):
            doc_idxs = [d for d, _s in bm25_pool_ranked[i]]  # full pool -- MMR selects DOWN to TOP_K
            scores = [s for _d, s in bm25_pool_ranked[i]]
            emb_block = train_emb[doc_idxs] if doc_idxs else np.zeros((0, train_emb.shape[1]))
            selected_lists.append(mmr_select(doc_idxs, scores, emb_block, lam=lam, k_results=TOP_K))

        earliest = [
            _earliest_hit_rank_cached(i, selected_lists[i], synth_true_seq[i], train_seqs, hit_cache)
            for i in range(n_frag)
        ]
        recall = {
            "recall_at_1": _recall_at(earliest, 1),
            "recall_at_5": _recall_at(earliest, 5),
            "recall_at_10": _recall_at(earliest, 10),
        }
        dist, set_level = redundancy_of_sets(selected_lists, train_emb)

        row = {
            "lambda": lam,
            "recall_at_1": recall["recall_at_1"],
            "recall_at_5": recall["recall_at_5"],
            "recall_at_10": recall["recall_at_10"],
            "mean_pairwise_sim": dist["mean"],
            "median_pairwise_sim": dist["median"],
            "pct_sets_with_pair_ge_0.999": set_level["pct_sets_with_pair_ge_0.999"],
            "pct_sets_with_pair_ge_0.99": set_level["pct_sets_with_pair_ge_0.99"],
            "pct_sets_with_pair_ge_0.95": set_level["pct_sets_with_pair_ge_0.95"],
        }
        sweep_rows.append(row)
        print(f"lambda={lam}: recall@1={recall['recall_at_1']:.3f} @5={recall['recall_at_5']:.3f} "
              f"@10={recall['recall_at_10']:.3f} mean_pairwise_sim={dist['mean']:.4f} "
              f"pct_near_dup(>=0.999)={set_level['pct_sets_with_pair_ge_0.999']:.3f}")

    with open(f"{OUT_DIR}/mmr_lambda_sweep_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(sweep_rows[0].keys()))
        w.writeheader()
        w.writerows(sweep_rows)

    # --- selection: preserve recall@10 (primary), improve diversity, avoid unacceptable recall@1 loss ---
    bm25_r10 = bm25_recall["recall_at_10"]
    bm25_r1 = bm25_recall["recall_at_1"]
    acceptable = [
        r for r in sweep_rows
        if r["recall_at_10"] >= bm25_r10 - RECALL10_TOLERANCE
        and r["recall_at_1"] >= bm25_r1 - RECALL1_MAX_DROP
    ]
    if acceptable:
        best_row = min(acceptable, key=lambda r: r["mean_pairwise_sim"])
        best_lambda = best_row["lambda"]
        redundancy_reduced = best_row["mean_pairwise_sim"] < bm25_dist["mean"] - 1e-9
        verdict = "accepted" if redundancy_reduced else "accepted (recall preserved, but no measurable redundancy reduction)"
    else:
        best_row = None
        best_lambda = None
        verdict = "rejected -- no lambda preserved recall within the stated tolerance"

    with open(f"{OUT_DIR}/mmr_lambda_sweep_report.txt", "w") as f:
        f.write("=== Tasks 2/3: MMR lambda sweep -- summary ===\n\n")
        f.write(f"BM25 baseline (no MMR): recall@1={bm25_r1:.3f} recall@5={bm25_recall['recall_at_5']:.3f} "
                f"recall@10={bm25_r10:.3f} mean_pairwise_sim={bm25_dist['mean']:.4f}\n\n")
        f.write("--- Sweep ---\n")
        for r in sweep_rows:
            f.write(f"lambda={r['lambda']}: recall@1={r['recall_at_1']:.3f} recall@5={r['recall_at_5']:.3f} "
                    f"recall@10={r['recall_at_10']:.3f} mean_pairwise_sim={r['mean_pairwise_sim']:.4f} "
                    f"pct_sets_near_dup(>=0.999)={r['pct_sets_with_pair_ge_0.999']:.3f}\n")
        f.write(f"\nSelection rule: recall@10 within {RECALL10_TOLERANCE} of BM25 baseline AND recall@1 within "
                f"{RECALL1_MAX_DROP} of BM25 baseline (both judgment calls, stated explicitly); "
                f"among survivors, lowest mean_pairwise_sim (most redundancy reduction).\n")
        f.write(f"Verdict: {verdict}\n")
        if best_row:
            f.write(f"Selected lambda={best_lambda}\n")

    print(f"\nVerdict: {verdict}")
    if best_row:
        print(f"Selected lambda={best_lambda}")

    print("\nDone. Outputs written to:", OUT_DIR)
    return bm25_recall, bm25_dist, sweep_rows, best_lambda, verdict


if __name__ == "__main__":
    main()
