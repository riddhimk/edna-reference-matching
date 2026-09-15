"""
Stage 1 -- BM25 independent-retrieval evaluation.

Evaluates the new k-mer BM25 retrieval arm (src/bm25_retrieval.py) on the
IDENTICAL held-out MitoFish calibration fragments Stage 0's DNABERT
recall diagnostic used (src/stage0_diagnostics.py:build_holdout_split,
imported unmodified -- same seed=42, holdout_frac=0.2, n_test=1000,
REAL_READ_LENGTHS=118-140bp), and the SAME >=95% full-sequence-identity
ground truth (via the unmodified nucleotide_rerank.align_pair).

Per requirement, this does NOT use the real 5,000-read assignments to
tune anything: k1/b are IR-literature defaults (see bm25_retrieval.py),
and the k-mer length k is chosen only from this held-out sweep, never
from real-read outcomes. Real reads are not touched anywhere in this
module.

Leakage prevention: the BM25 index is built ONLY over the TRAIN
partition of the reference table (the same train_idx Part 4's frozen
calibration and Stage 0's DNABERT diagnostic use) -- held-out fragments'
own parent sequences are never in the index.

Does not modify src/stage0_diagnostics.py or its output folder, the Part
4 pipeline, the MitoFish database/primers/extracted region, or the 95%
identity threshold. Writes only to data/Stage1 diagnostics/. The
DNABERT top-10 retrieval step is recomputed locally here (same logic as
stage0_diagnostics.run_recall_diagnostic) rather than calling that
function directly, specifically to avoid overwriting Stage 0's already
-approved output files.

Usage:
    python -m src.stage1_bm25_diagnostics
"""
import csv
import os
import numpy as np

from . import config
from .embed_common import embed_batch
from .faiss_index import l2_normalize
from .nucleotide_rerank import align_pair
from .bm25_retrieval import BM25Index
from .stage0_diagnostics import build_holdout_split, TOP_K

OUT_DIR = "data/Stage1 diagnostics"
K_MER_SWEEP = (4, 6, 8)


def _is_hit(candidate_seq: str, true_seq: str) -> bool:
    """Identical >=95% full-sequence-identity ground truth to
    stage0_diagnostics._is_hit / mitofish_rerank_calibration.py's
    correctness check -- duplicated (not imported) so this module stays
    self-contained, per instruction to keep Stage 1 separate. Both
    ultimately call the same, unmodified nucleotide_rerank.align_pair."""
    if candidate_seq == true_seq:
        return True
    return align_pair(candidate_seq, true_seq)["identity_pct"] >= 95.0


def _earliest_hit_rank(candidate_seqs_in_rank_order, true_seq, top_k=TOP_K):
    for k in range(min(top_k, len(candidate_seqs_in_rank_order))):
        if _is_hit(candidate_seqs_in_rank_order[k], true_seq):
            return k + 1
    return None


def _recall_at(earliest_ranks, k):
    n = len(earliest_ranks)
    return sum(1 for r in earliest_ranks if r is not None and r <= k) / n if n else 0.0


def get_dnabert_top10_positions(split):
    """Locally reproduces Stage 0's DNABERT top-10 retrieval (embed +
    train-only FAISS search) without calling
    stage0_diagnostics.run_recall_diagnostic(), which writes into Stage
    0's own approved output folder. Same embedding/search logic,
    reused unmodified via imports (embed_batch, l2_normalize, the
    train-only index already built inside split['index'])."""
    synth_seqs = split["synth_seqs"]
    index = split["index"]

    synth_emb = []
    for i in range(0, len(synth_seqs), config.BATCH_SIZE):
        batch_vecs = embed_batch(synth_seqs[i:i + config.BATCH_SIZE])
        synth_emb.append(batch_vecs)
        print(f"  [DNABERT] embedded {i + len(batch_vecs)}/{len(synth_seqs)}")
    synth_emb = l2_normalize(np.vstack(synth_emb).astype("float32"))
    _sims, pos = index.search(synth_emb, TOP_K)
    return pos  # (n_fragments, TOP_K) indices into split["train_seqs"] / split["train_ids"]


def evaluate_bm25_recall(split, k_mer: int):
    train_ids = split["train_ids"]
    train_seqs = split["train_seqs"]
    synth_seqs = split["synth_seqs"]
    synth_true_seq = split["synth_true_seq"]

    bm25 = BM25Index(k=k_mer)
    bm25.build(train_ids, train_seqs)  # TRAIN partition only -- leakage prevention

    bm25_pos = np.full((len(synth_seqs), TOP_K), -1, dtype=np.int64)
    earliest_ranks = []
    for i, frag in enumerate(synth_seqs):
        ranked = bm25.top_k(frag, k_results=TOP_K)
        for j, (doc_idx, _score) in enumerate(ranked):
            bm25_pos[i, j] = doc_idx
        cand_seqs = [train_seqs[doc_idx] for doc_idx, _score in ranked]
        earliest_ranks.append(_earliest_hit_rank(cand_seqs, synth_true_seq[i]))
        if (i + 1) % 200 == 0:
            print(f"  [BM25 k={k_mer}] recall-checked {i + 1}/{len(synth_seqs)}")

    n = len(synth_seqs)
    summary = {
        "k_mer": k_mer,
        "n_holdout_fragments": n,
        "recall_at_1": _recall_at(earliest_ranks, 1),
        "recall_at_5": _recall_at(earliest_ranks, 5),
        "recall_at_10": _recall_at(earliest_ranks, 10),
        "n_absent_from_top10": sum(1 for r in earliest_ranks if r is None),
    }
    summary["pct_absent_from_top10"] = summary["n_absent_from_top10"] / n
    return summary, bm25_pos, earliest_ranks


def compare_with_dnabert(split, dnabert_pos, bm25_pos, bm25_earliest_ranks, dnabert_earliest_ranks):
    n = len(split["synth_seqs"])

    rows = []
    n_overlap_total = 0
    n_rescued_by_bm25 = 0
    n_rescued_by_dnabert = 0
    n_both_hit = 0
    n_both_miss = 0
    n_union_hit = 0

    for i in range(n):
        dbert_set = set(int(x) for x in dnabert_pos[i] if x >= 0)
        bm25_set = set(int(x) for x in bm25_pos[i] if x >= 0)
        overlap = len(dbert_set & bm25_set)
        n_overlap_total += overlap

        dbert_hit = dnabert_earliest_ranks[i] is not None
        bm25_hit = bm25_earliest_ranks[i] is not None

        rescued_by_bm25 = (not dbert_hit) and bm25_hit
        rescued_by_dnabert = dbert_hit and (not bm25_hit)
        both_hit = dbert_hit and bm25_hit
        both_miss = (not dbert_hit) and (not bm25_hit)

        n_rescued_by_bm25 += int(rescued_by_bm25)
        n_rescued_by_dnabert += int(rescued_by_dnabert)
        n_both_hit += int(both_hit)
        n_both_miss += int(both_miss)
        n_union_hit += int(dbert_hit or bm25_hit)

        rows.append({
            "fragment_idx": i,
            "dnabert_earliest_hit_rank": dnabert_earliest_ranks[i] if dbert_hit else "",
            "bm25_earliest_hit_rank": bm25_earliest_ranks[i] if bm25_hit else "",
            "top10_overlap_count": overlap,
            "rescued_by_bm25": int(rescued_by_bm25),
            "rescued_by_dnabert": int(rescued_by_dnabert),
            "both_hit": int(both_hit),
            "both_miss": int(both_miss),
        })

    summary = {
        "n_fragments": n,
        "mean_top10_overlap": n_overlap_total / n,
        "n_rescued_by_bm25": n_rescued_by_bm25,
        "pct_rescued_by_bm25": n_rescued_by_bm25 / n,
        "n_rescued_by_dnabert": n_rescued_by_dnabert,
        "pct_rescued_by_dnabert": n_rescued_by_dnabert / n,
        "n_both_hit": n_both_hit,
        "pct_both_hit": n_both_hit / n,
        "n_both_miss": n_both_miss,
        "pct_both_miss": n_both_miss / n,
        "union_recall_at_10_plus_10": n_union_hit / n,
    }
    return summary, rows


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print("=== Stage 1: BM25 independent-retrieval evaluation ===")
    print("Read-only reuse of Stage 0's holdout split. Does not modify Part 4 or Stage 0.\n")

    split = build_holdout_split()
    n_frag = len(split["synth_seqs"])

    print(f"Running DNABERT top-{TOP_K} retrieval on the same {n_frag} holdout fragments as Stage 0...")
    dnabert_pos = get_dnabert_top10_positions(split)
    dnabert_earliest_ranks = [
        _earliest_hit_rank([split["train_seqs"][idx] for idx in dnabert_pos[i]], split["synth_true_seq"][i])
        for i in range(n_frag)
    ]

    # --- k-mer sweep, held-out only -- never on real-read assignments ---
    sweep_summaries = []
    best = None
    for k_mer in K_MER_SWEEP:
        print(f"\nBuilding & evaluating BM25 index, k={k_mer} (train-only, {len(split['train_ids'])} refs)...")
        summary, bm25_pos, bm25_earliest_ranks = evaluate_bm25_recall(split, k_mer)
        sweep_summaries.append(summary)
        print(f"  k={k_mer}: recall@1={summary['recall_at_1']:.3f} "
              f"@5={summary['recall_at_5']:.3f} @10={summary['recall_at_10']:.3f}")
        if best is None or summary["recall_at_10"] > best[0]["recall_at_10"]:
            best = (summary, bm25_pos, bm25_earliest_ranks)

    with open(f"{OUT_DIR}/bm25_kmer_sweep_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(sweep_summaries[0].keys()))
        w.writeheader()
        w.writerows(sweep_summaries)

    best_summary, best_bm25_pos, best_bm25_earliest_ranks = best
    best_k = best_summary["k_mer"]
    print(f"\nSelected k={best_k} (highest recall@10 on held-out sweep) for the DNABERT comparison below.")

    comparison_summary, comparison_rows = compare_with_dnabert(
        split, dnabert_pos, best_bm25_pos, best_bm25_earliest_ranks, dnabert_earliest_ranks
    )

    with open(f"{OUT_DIR}/bm25_vs_dnabert_per_fragment.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(comparison_rows[0].keys()))
        w.writeheader()
        w.writerows(comparison_rows)

    dnabert_summary_for_report = {
        "recall_at_1": _recall_at(dnabert_earliest_ranks, 1),
        "recall_at_5": _recall_at(dnabert_earliest_ranks, 5),
        "recall_at_10": _recall_at(dnabert_earliest_ranks, 10),
        "n_absent_from_top10": sum(1 for r in dnabert_earliest_ranks if r is None),
    }

    with open(f"{OUT_DIR}/stage1_summary.txt", "w") as f:
        f.write("=== Stage 1: BM25 independent retrieval -- summary ===\n\n")
        f.write("--- k-mer sweep (held-out MitoFish fragments, identical split to Stage 0) ---\n")
        for s in sweep_summaries:
            f.write(f"k={s['k_mer']}: recall@1={s['recall_at_1']:.3f} recall@5={s['recall_at_5']:.3f} "
                    f"recall@10={s['recall_at_10']:.3f} absent={s['n_absent_from_top10']}/{s['n_holdout_fragments']}\n")
        f.write(f"\nSelected k={best_k} for BM25-vs-DNABERT comparison (highest recall@10 on the sweep).\n\n")
        f.write("--- DNABERT (recomputed here on the identical split, for direct comparison) ---\n")
        for k, v in dnabert_summary_for_report.items():
            f.write(f"{k:28s}: {v}\n")
        f.write(f"\n--- BM25 (k={best_k}) ---\n")
        for k, v in best_summary.items():
            f.write(f"{k:28s}: {v}\n")
        f.write("\n--- BM25 vs DNABERT: overlap / rescue / both-miss ---\n")
        for k, v in comparison_summary.items():
            f.write(f"{k:28s}: {v}\n")

    print("\n--- DNABERT (recomputed on identical split) ---")
    for k, v in dnabert_summary_for_report.items():
        print(f"{k:28s}: {v}")
    print(f"\n--- BM25 (k={best_k}) ---")
    for k, v in best_summary.items():
        print(f"{k:28s}: {v}")
    print("\n--- Overlap / rescue / both-miss ---")
    for k, v in comparison_summary.items():
        print(f"{k:28s}: {v}")

    print("\nDone. Outputs written to:", OUT_DIR)


if __name__ == "__main__":
    main()
