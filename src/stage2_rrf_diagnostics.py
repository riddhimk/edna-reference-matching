"""
Stage 2 -- DNABERT + BM25 Reciprocal Rank Fusion (RRF) evaluation.

Evaluates RRF fusion of the DNABERT and BM25(k=6) retrieval arms on the
IDENTICAL 1,000 held-out MitoFish fragments used in Stage 0 and Stage 1
(src/stage0_diagnostics.py:build_holdout_split, imported unmodified --
same seed=42, holdout_frac=0.2, n_test=1000), and the SAME >=95%
full-sequence-identity ground truth (via the unmodified
nucleotide_rerank.align_pair).

Both retrieval arms' top-10 candidate lists are recomputed locally here
(same logic/parameters Stage 1 already verified reproduce Stage 0 exactly:
DNABERT via the train-only FAISS index, BM25 with k=6, the k Stage 1
selected on its own held-out sweep) rather than reading Stage 0/1's saved
files, to avoid any risk of writing into their already-approved output
folders.

Structural note stated up front, then checked empirically below (not
assumed): RRF only re-ranks the UNION of each arm's own top-10 -- it
cannot surface a candidate neither arm retrieved. Stage 1 found DNABERT's
holdout hits are a strict subset of BM25's (zero cases rescued by
DNABERT), so the theoretical ceiling for RRF here is BM25-alone's recall
(94.6%): RRF can match it, or fall short if fusion displaces a correct
BM25 candidate out of the fused top-10, but it cannot exceed it on this
specific candidate pool. This prediction is verified against the actual
sweep results below, not substituted for them.

Does not modify Stage 0/1 output files, the Part 4 pipeline, MitoFish
database/primers/extracted region, the 95% identity threshold, or the
nucleotide reranker. Writes only to data/Stage2 diagnostics/.

Usage:
    python -m src.stage2_rrf_diagnostics
"""
import csv
import os
from typing import Dict, List, Tuple
import numpy as np

from . import config
from .embed_common import embed_batch
from .faiss_index import l2_normalize
from .nucleotide_rerank import align_pair
from .bm25_retrieval import BM25Index
from .rrf_fusion import fuse_top_k
from .stage0_diagnostics import build_holdout_split, TOP_K

OUT_DIR = "data/Stage2 diagnostics"
BM25_K_MER = 6  # selected in Stage 1's held-out sweep -- not re-tuned here
RRF_K_CONST_SWEEP = (1, 5, 10, 20, 30, 40, 60, 100)


def _is_hit(candidate_seq: str, true_seq: str) -> bool:
    """Identical >=95% full-sequence-identity ground truth used in Stage
    0/1 / Part 4's calibration -- duplicated (not imported) to keep
    Stage 2 self-contained, per instruction to keep results separate.
    Both ultimately call the same, unmodified nucleotide_rerank.align_pair."""
    if candidate_seq == true_seq:
        return True
    return align_pair(candidate_seq, true_seq)["identity_pct"] >= 95.0


HitCache = Dict[Tuple[int, int], bool]


def _earliest_hit_rank_cached(fragment_idx: int, doc_idx_list: List[int], true_seq: str,
                               train_seqs: List[str], hit_cache: HitCache, top_k: int = TOP_K):
    """Same earliest-hit-rank definition as Stage 0/1, but memoizes
    _is_hit((fragment_idx, doc_idx)) -- that result depends only on WHICH
    candidate is checked, not on the rank order it's checked in, so it is
    invariant across the 8-way RRF k_const sweep below. This avoids
    redundant Smith-Waterman alignment calls for the same
    (fragment, candidate) pair across sweep values; it changes nothing
    about what is measured."""
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


def get_dnabert_top10_lists(split) -> List[List[int]]:
    synth_seqs = split["synth_seqs"]
    index = split["index"]
    synth_emb = []
    for i in range(0, len(synth_seqs), config.BATCH_SIZE):
        batch_vecs = embed_batch(synth_seqs[i:i + config.BATCH_SIZE])
        synth_emb.append(batch_vecs)
        print(f"  [DNABERT] embedded {i + len(batch_vecs)}/{len(synth_seqs)}")
    synth_emb = l2_normalize(np.vstack(synth_emb).astype("float32"))
    _sims, pos = index.search(synth_emb, TOP_K)
    return [[int(x) for x in pos[i]] for i in range(len(synth_seqs))]


def get_bm25_top10_lists(split, k_mer: int = BM25_K_MER) -> List[List[int]]:
    train_ids = split["train_ids"]
    train_seqs = split["train_seqs"]
    synth_seqs = split["synth_seqs"]

    bm25 = BM25Index(k=k_mer)
    bm25.build(train_ids, train_seqs)  # TRAIN partition only -- leakage prevention

    results = []
    for i, frag in enumerate(synth_seqs):
        ranked = bm25.top_k(frag, k_results=TOP_K)
        results.append([doc_idx for doc_idx, _score in ranked])
        if (i + 1) % 200 == 0:
            print(f"  [BM25 k={k_mer}] retrieved {i + 1}/{len(synth_seqs)}")
    return results


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print("=== Stage 2: DNABERT + BM25 RRF fusion evaluation ===")
    print("Read-only reuse of the Stage 0 holdout split. Does not modify Part 4, Stage 0, or Stage 1.\n")

    split = build_holdout_split()
    n_frag = len(split["synth_seqs"])
    train_seqs = split["train_seqs"]
    synth_true_seq = split["synth_true_seq"]
    hit_cache: HitCache = {}

    print(f"Retrieving DNABERT top-{TOP_K} on {n_frag} holdout fragments...")
    dnabert_lists = get_dnabert_top10_lists(split)

    print(f"Retrieving BM25 (k={BM25_K_MER}) top-{TOP_K}...")
    bm25_lists = get_bm25_top10_lists(split)

    dnabert_earliest = [
        _earliest_hit_rank_cached(i, dnabert_lists[i], synth_true_seq[i], train_seqs, hit_cache)
        for i in range(n_frag)
    ]
    bm25_earliest = [
        _earliest_hit_rank_cached(i, bm25_lists[i], synth_true_seq[i], train_seqs, hit_cache)
        for i in range(n_frag)
    ]

    dnabert_recall = {
        "recall_at_1": _recall_at(dnabert_earliest, 1),
        "recall_at_5": _recall_at(dnabert_earliest, 5),
        "recall_at_10": _recall_at(dnabert_earliest, 10),
    }
    bm25_recall = {
        "recall_at_1": _recall_at(bm25_earliest, 1),
        "recall_at_5": _recall_at(bm25_earliest, 5),
        "recall_at_10": _recall_at(bm25_earliest, 10),
    }

    overlap_counts = [len(set(dnabert_lists[i]) & set(bm25_lists[i])) for i in range(n_frag)]
    mean_overlap = sum(overlap_counts) / n_frag

    union_hit = [(dnabert_earliest[i] is not None) or (bm25_earliest[i] is not None) for i in range(n_frag)]
    union_recall = sum(union_hit) / n_frag

    # --- RRF k_const sensitivity sweep ---
    print("\nRunning RRF k_const sensitivity sweep...")
    sweep_rows = []
    best = None
    for k_const in RRF_K_CONST_SWEEP:
        fused_lists = [fuse_top_k([dnabert_lists[i], bm25_lists[i]], k_const=k_const, k_results=TOP_K)
                       for i in range(n_frag)]
        fused_earliest = [
            _earliest_hit_rank_cached(i, fused_lists[i], synth_true_seq[i], train_seqs, hit_cache)
            for i in range(n_frag)
        ]
        row = {
            "rrf_k_const": k_const,
            "recall_at_1": _recall_at(fused_earliest, 1),
            "recall_at_5": _recall_at(fused_earliest, 5),
            "recall_at_10": _recall_at(fused_earliest, 10),
        }
        sweep_rows.append(row)
        print(f"  RRF k_const={k_const}: recall@1={row['recall_at_1']:.3f} "
              f"@5={row['recall_at_5']:.3f} @10={row['recall_at_10']:.3f}")
        if best is None or row["recall_at_10"] > best[0]["recall_at_10"] or (
            row["recall_at_10"] == best[0]["recall_at_10"] and row["recall_at_1"] > best[0]["recall_at_1"]
        ):
            best = (row, fused_lists, fused_earliest)

    with open(f"{OUT_DIR}/rrf_kconst_sweep_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(sweep_rows[0].keys()))
        w.writeheader()
        w.writerows(sweep_rows)

    best_row, best_fused_lists, best_fused_earliest = best
    best_k_const = best_row["rrf_k_const"]
    print(f"\nSelected RRF k_const={best_k_const} (highest recall@10, tie-broken by recall@1).")

    # --- rescued / lost relative to BM25 alone ---
    n_rescued_by_rrf = 0
    n_lost_by_rrf = 0
    n_both_hit = 0
    n_both_miss = 0
    rows = []
    for i in range(n_frag):
        bm25_hit = bm25_earliest[i] is not None
        rrf_hit = best_fused_earliest[i] is not None
        rescued = (not bm25_hit) and rrf_hit
        lost = bm25_hit and (not rrf_hit)
        n_rescued_by_rrf += int(rescued)
        n_lost_by_rrf += int(lost)
        n_both_hit += int(bm25_hit and rrf_hit)
        n_both_miss += int((not bm25_hit) and (not rrf_hit))
        rows.append({
            "fragment_idx": i,
            "dnabert_earliest_hit_rank": dnabert_earliest[i] if dnabert_earliest[i] is not None else "",
            "bm25_earliest_hit_rank": bm25_earliest[i] if bm25_hit else "",
            "rrf_earliest_hit_rank": best_fused_earliest[i] if rrf_hit else "",
            "top10_overlap_dnabert_bm25": overlap_counts[i],
            "rescued_by_rrf_vs_bm25": int(rescued),
            "lost_by_rrf_vs_bm25": int(lost),
        })

    with open(f"{OUT_DIR}/rrf_vs_bm25_per_fragment.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    rrf_recall = {
        "recall_at_1": best_row["recall_at_1"],
        "recall_at_5": best_row["recall_at_5"],
        "recall_at_10": best_row["recall_at_10"],
    }

    comparison_summary = {
        "n_fragments": n_frag,
        "mean_top10_overlap_dnabert_bm25": mean_overlap,
        "dnabert_recall_at_1": dnabert_recall["recall_at_1"],
        "bm25_recall_at_1": bm25_recall["recall_at_1"],
        "rrf_recall_at_1": rrf_recall["recall_at_1"],
        "dnabert_recall_at_5": dnabert_recall["recall_at_5"],
        "bm25_recall_at_5": bm25_recall["recall_at_5"],
        "rrf_recall_at_5": rrf_recall["recall_at_5"],
        "dnabert_recall_at_10": dnabert_recall["recall_at_10"],
        "bm25_recall_at_10": bm25_recall["recall_at_10"],
        "rrf_recall_at_10": rrf_recall["recall_at_10"],
        "union_recall_at_10_plus_10": union_recall,
        "n_rescued_by_rrf_vs_bm25": n_rescued_by_rrf,
        "pct_rescued_by_rrf_vs_bm25": n_rescued_by_rrf / n_frag,
        "n_lost_by_rrf_vs_bm25": n_lost_by_rrf,
        "pct_lost_by_rrf_vs_bm25": n_lost_by_rrf / n_frag,
        "n_both_hit": n_both_hit,
        "n_both_miss": n_both_miss,
        "rrf_improves_on_bm25_alone_at_10": bool(rrf_recall["recall_at_10"] > bm25_recall["recall_at_10"]),
        "rrf_matches_bm25_alone_at_10": bool(rrf_recall["recall_at_10"] == bm25_recall["recall_at_10"]),
    }

    with open(f"{OUT_DIR}/stage2_summary.txt", "w") as f:
        f.write("=== Stage 2: DNABERT + BM25 RRF fusion -- summary ===\n\n")
        f.write("--- RRF k_const sensitivity sweep (held-out, identical split to Stage 0/1) ---\n")
        for r in sweep_rows:
            f.write(f"k_const={r['rrf_k_const']}: recall@1={r['recall_at_1']:.3f} "
                    f"recall@5={r['recall_at_5']:.3f} recall@10={r['recall_at_10']:.3f}\n")
        f.write(f"\nSelected k_const={best_k_const} (highest recall@10, tie-broken by recall@1).\n\n")
        f.write("--- DNABERT-only ---\n")
        for k, v in dnabert_recall.items():
            f.write(f"{k:20s}: {v}\n")
        f.write(f"\n--- BM25-only (k={BM25_K_MER}) ---\n")
        for k, v in bm25_recall.items():
            f.write(f"{k:20s}: {v}\n")
        f.write(f"\n--- RRF (k_const={best_k_const}) ---\n")
        for k, v in rrf_recall.items():
            f.write(f"{k:20s}: {v}\n")
        f.write("\n--- Comparison ---\n")
        for k, v in comparison_summary.items():
            f.write(f"{k:38s}: {v}\n")

    print("\n--- DNABERT-only ---")
    for k, v in dnabert_recall.items():
        print(f"{k:20s}: {v}")
    print(f"\n--- BM25-only (k={BM25_K_MER}) ---")
    for k, v in bm25_recall.items():
        print(f"{k:20s}: {v}")
    print(f"\n--- RRF (k_const={best_k_const}) ---")
    for k, v in rrf_recall.items():
        print(f"{k:20s}: {v}")
    print("\n--- Comparison ---")
    for k, v in comparison_summary.items():
        print(f"{k:38s}: {v}")

    print("\nDone. Outputs written to:", OUT_DIR)


if __name__ == "__main__":
    main()
