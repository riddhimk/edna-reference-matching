"""
Stage 3, Task 4 -- end-to-end evaluation of retrieval architectures
through the FROZEN nucleotide reranker.

Compares, on the IDENTICAL 1,000 held-out MitoFish fragments used by
every prior stage (stage0_diagnostics.build_holdout_split, imported
unmodified, seed=42):

    A. BM25 (k=6)                       -> nucleotide reranker
    B. BM25 (k=6) -> MMR (lambda from Task 3's sweep, if accepted)
                                         -> nucleotide reranker
    C. DNABERT                          -> nucleotide reranker
    D. RRF (DNABERT + BM25, k_const=10, Stage 2's selected value)
                                         -> nucleotide reranker

The reranking step calls nucleotide_rerank.rerank_candidates() directly,
UNMODIFIED, as a black box -- exactly the frozen function Part 4 uses.
No caching is injected into it (that would require modifying it), so
each architecture's candidates are independently realigned; this is a
deliberate compute-vs-purity tradeoff, not an oversight.

Decision rule: the FROZEN Part 4 operating threshold (identity_pct >= 95,
the same value calibrated in mitofish_rerank_calibration.py) is applied
IDENTICALLY across all four architectures -- this script does not
recalibrate or search for a new "best" threshold per architecture, which
would be tuning against this evaluation set. The full threshold sweep
(via the unmodified evaluate.sweep_thresholds) is also reported per
architecture as supplementary, informational context only -- it is not
used to pick an operating point.

Ground truth for "correct": identical >=95% full-sequence-identity check
used throughout Stage 0/1/2/Task 1/2/3 (nucleotide_rerank.align_pair,
unmodified).

Does not modify BM25, RRF, MMR, Stage 0/1/2/Task1-3 outputs, the Part 4
pipeline, MitoFish database/primers/extracted region, the 95% threshold,
or nucleotide_rerank.rerank_candidates() itself. Writes only to
data/Final diagnostics/.

Usage:
    python -m src.final_evaluation
"""
import csv
import json
import os
import time
from typing import Dict, List, Tuple
import numpy as np

from . import config
from .embed_common import embed_batch
from .faiss_index import l2_normalize
from .nucleotide_rerank import rerank_candidates, align_pair
from .bm25_retrieval import BM25Index
from .rrf_fusion import fuse_top_k
from .mmr import mmr_select
from .evaluate import sweep_thresholds
from .stage0_diagnostics import build_holdout_split, TOP_K
from .stage2_rrf_diagnostics import BM25_K_MER
from .mmr_lambda_sweep import get_bm25_pool_with_scores, redundancy_of_sets, MMR_POOL_SIZE, OUT_DIR as STAGE3_OUT_DIR

OUT_DIR = "data/Final diagnostics"
FROZEN_IDENTITY_THRESHOLD = 95.0  # Part 4's calibrated operating point -- not re-derived here
RRF_K_CONST = 10  # Stage 2's selected value -- not re-swept here


def _is_hit(candidate_seq: str, true_seq: str) -> bool:
    if candidate_seq == true_seq:
        return True
    return align_pair(candidate_seq, true_seq)["identity_pct"] >= 95.0


def _earliest_hit_rank(doc_idx_list: List[int], true_seq: str, train_seqs: List[str], top_k: int = TOP_K):
    for k in range(min(top_k, len(doc_idx_list))):
        if _is_hit(train_seqs[doc_idx_list[k]], true_seq):
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


def rerank_one_fragment(fragment_seq: str, doc_idxs: List[int], scores: Dict[int, float],
                         train_ids: List[str], train_seqs: List[str]):
    """Calls nucleotide_rerank.rerank_candidates() -- FROZEN, unmodified -- as a black box."""
    candidates = [(train_ids[idx], train_seqs[idx], scores.get(idx, 0.0)) for idx in doc_idxs]
    return rerank_candidates(fragment_seq, candidates)


def evaluate_architecture(name: str, per_fragment_lists: List[List[int]], per_fragment_scores: List[Dict[int, float]],
                           split, train_id_to_seq: Dict[str, str]):
    train_ids = split["train_ids"]
    train_seqs = split["train_seqs"]
    synth_seqs = split["synth_seqs"]
    synth_true_seq = split["synth_true_seq"]
    n = len(synth_seqs)

    t0 = time.time()

    # retrieval recall (same definition as every prior stage)
    earliest_ranks = [_earliest_hit_rank(per_fragment_lists[i], synth_true_seq[i], train_seqs) for i in range(n)]
    recall = {
        "recall_at_1": _recall_at(earliest_ranks, 1),
        "recall_at_5": _recall_at(earliest_ranks, 5),
        "recall_at_10": _recall_at(earliest_ranks, 10),
    }

    # candidate-set diversity / redundancy (same metric as Task 1/3)
    dist, set_level = redundancy_of_sets(per_fragment_lists, split["train_emb"])

    # frozen nucleotide reranker, called as a black box per fragment
    best_identity_pct = []
    best_coverage_pct = []
    is_correct = []
    chosen_ref_ids = []
    for i in range(n):
        doc_idxs = per_fragment_lists[i]
        if not doc_idxs:
            best_identity_pct.append(0.0)
            best_coverage_pct.append(0.0)
            is_correct.append(0)
            chosen_ref_ids.append("")
            continue
        reranked = rerank_one_fragment(synth_seqs[i], doc_idxs, per_fragment_scores[i], train_ids, train_seqs)
        best = reranked[0]
        best_identity_pct.append(best["identity_pct"])
        best_coverage_pct.append(best["coverage_pct"])
        chosen_ref_ids.append(best["Reference_ID"])
        chosen_seq = train_id_to_seq.get(best["Reference_ID"])
        is_correct.append(1 if (chosen_seq is not None and _is_hit(chosen_seq, synth_true_seq[i])) else 0)
        if (i + 1) % 200 == 0:
            print(f"  [{name}] reranked {i + 1}/{n}")

    best_identity_pct = np.array(best_identity_pct)
    is_correct = np.array(is_correct)
    elapsed = time.time() - t0

    # frozen threshold decision (95% identity_pct) -- applied identically across architectures
    assigned_mask = best_identity_pct >= FROZEN_IDENTITY_THRESHOLD
    n_assigned = int(assigned_mask.sum())
    accuracy_at_threshold = float(is_correct[assigned_mask].mean()) if n_assigned else None
    n_correct_at_threshold = int(is_correct[assigned_mask].sum()) if n_assigned else 0

    # supplementary-only: full sweep curve (NOT used to pick an operating point)
    y_true = [1] * n
    y_pred = list(is_correct)
    thresholds = [round(t, 1) for t in np.arange(45, 100, 5)]
    sweep_rows = sweep_thresholds(y_true, y_pred, list(best_identity_pct), thresholds)

    result = {
        "architecture": name,
        "n_fragments": n,
        "compute_seconds": round(elapsed, 1),
        "recall_at_1": recall["recall_at_1"],
        "recall_at_5": recall["recall_at_5"],
        "recall_at_10": recall["recall_at_10"],
        "mean_pairwise_sim": dist["mean"],
        "median_pairwise_sim": dist["median"],
        "pct_sets_near_dup_ge_0.999": set_level["pct_sets_with_pair_ge_0.999"],
        "n_assigned_at_frozen_threshold_95": n_assigned,
        "pct_assigned_at_frozen_threshold_95": n_assigned / n,
        "n_correct_at_frozen_threshold_95": n_correct_at_threshold,
        "accuracy_at_frozen_threshold_95": accuracy_at_threshold,
        "median_identity_pct_all": float(np.median(best_identity_pct)),
        "median_identity_pct_assigned": float(np.median(best_identity_pct[assigned_mask])) if n_assigned else None,
        "median_coverage_pct_assigned": float(np.median(np.array(best_coverage_pct)[assigned_mask])) if n_assigned else None,
    }

    per_fragment_rows = [
        {
            "fragment_idx": i,
            "earliest_retrieval_hit_rank": earliest_ranks[i] if earliest_ranks[i] is not None else "",
            "chosen_reference_id": chosen_ref_ids[i],
            "identity_pct": float(best_identity_pct[i]),
            "coverage_pct": float(best_coverage_pct[i]),
            "assigned_at_threshold_95": int(assigned_mask[i]),
            "correct": int(is_correct[i]),
        }
        for i in range(n)
    ]

    return result, sweep_rows, per_fragment_rows


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print("=== Task 4: end-to-end evaluation (A: BM25, B: BM25+MMR, C: DNABERT, D: RRF) ===")
    print("All four call nucleotide_rerank.rerank_candidates() unmodified. Frozen 95% threshold applied to all.\n")

    split = build_holdout_split()
    train_ids = split["train_ids"]
    train_seqs = split["train_seqs"]
    n_frag = len(split["synth_seqs"])
    train_id_to_seq = dict(zip(train_ids, train_seqs))

    print(f"Retrieving DNABERT top-{TOP_K} on {n_frag} holdout fragments...")
    dnabert_lists = get_dnabert_top10_lists(split)
    dnabert_scores = [{idx: 0.0 for idx in lst} for lst in dnabert_lists]  # placeholder; reranker ignores this field

    print(f"Retrieving BM25 (k={BM25_K_MER}) top-{MMR_POOL_SIZE} pool with scores...")
    bm25_pool_ranked = get_bm25_pool_with_scores(split)  # up to MMR_POOL_SIZE long, per fragment
    # Architecture A (BM25-alone) uses the TRUE top-10 -- first 10 of the same sorted pool,
    # identical to Stage 1/Task 1's already-reported numbers.
    bm25_lists = [[d for d, _s in ranked[:TOP_K]] for ranked in bm25_pool_ranked]
    bm25_scores = [{d: s for d, s in ranked} for ranked in bm25_pool_ranked]

    print(f"Fusing DNABERT + BM25 via RRF (k_const={RRF_K_CONST}, Stage 2's selected value)...")
    rrf_lists = [fuse_top_k([dnabert_lists[i], bm25_lists[i]], k_const=RRF_K_CONST, k_results=TOP_K)
                 for i in range(n_frag)]
    rrf_scores_per_frag = [dict(bm25_scores[i]) for i in range(n_frag)]  # carry BM25 score through where available

    # --- MMR: read Task 3's already-computed selected lambda from its saved report
    # (data/Stage3 diagnostics/mmr_lambda_sweep_report.txt) instead of re-running
    # that ~5-10 minute retrieval+sweep a second time here. ---
    mmr_best_lambda, mmr_verdict = None, "Task 3 report not found"
    report_path = f"{STAGE3_OUT_DIR}/mmr_lambda_sweep_report.txt"
    if os.path.exists(report_path):
        with open(report_path) as f:
            report_text = f.read()
        for line in report_text.splitlines():
            if line.startswith("Verdict:"):
                mmr_verdict = line.split("Verdict:", 1)[1].strip()
            if line.startswith("Selected lambda="):
                mmr_best_lambda = float(line.split("Selected lambda=", 1)[1].strip())

    if mmr_best_lambda is not None:
        print(f"\nUsing MMR lambda={mmr_best_lambda} (Task 3 verdict: {mmr_verdict})")
        mmr_lists = []
        for i in range(n_frag):
            doc_idxs = [d for d, _s in bm25_pool_ranked[i]]  # full pool -- MMR selects DOWN to TOP_K
            scores_list = [s for _d, s in bm25_pool_ranked[i]]
            emb_block = split["train_emb"][doc_idxs] if doc_idxs else np.zeros((0, split["train_emb"].shape[1]))
            mmr_lists.append(mmr_select(doc_idxs, scores_list, emb_block, lam=mmr_best_lambda, k_results=TOP_K))
        mmr_scores = [dict(bm25_scores[i]) for i in range(n_frag)]
    else:
        print(f"\nMMR NOT applied (Task 3 verdict: {mmr_verdict}) -- architecture B falls back to BM25-alone.")
        mmr_lists = bm25_lists
        mmr_scores = bm25_scores

    architectures = [
        ("A_BM25", bm25_lists, bm25_scores),
        ("B_BM25_MMR", mmr_lists, mmr_scores),
        ("C_DNABERT", dnabert_lists, dnabert_scores),
        ("D_RRF", rrf_lists, rrf_scores_per_frag),
    ]

    all_results = []
    all_sweeps = {}
    for name, lists_, scores_ in architectures:
        print(f"\n--- Reranking architecture {name} through the frozen nucleotide reranker ---")
        result, sweep_rows, per_fragment_rows = evaluate_architecture(name, lists_, scores_, split, train_id_to_seq)
        all_results.append(result)
        all_sweeps[name] = sweep_rows
        with open(f"{OUT_DIR}/{name}_per_fragment.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(per_fragment_rows[0].keys()))
            w.writeheader()
            w.writerows(per_fragment_rows)
        print(f"  recall@1={result['recall_at_1']:.3f} @5={result['recall_at_5']:.3f} @10={result['recall_at_10']:.3f} "
              f"assigned={result['n_assigned_at_frozen_threshold_95']}/{n_frag} "
              f"accuracy={result['accuracy_at_frozen_threshold_95']} "
              f"mean_pairwise_sim={result['mean_pairwise_sim']:.4f} time={result['compute_seconds']}s")

    with open(f"{OUT_DIR}/final_metrics.json", "w") as f:
        json.dump({
            "n_fragments": n_frag,
            "frozen_identity_threshold": FROZEN_IDENTITY_THRESHOLD,
            "rrf_k_const": RRF_K_CONST,
            "mmr_lambda_used": mmr_best_lambda,
            "mmr_verdict": mmr_verdict,
            "architectures": all_results,
        }, f, indent=2)

    with open(f"{OUT_DIR}/final_metrics_summary.txt", "w") as f:
        f.write("=== Task 4: end-to-end evaluation summary ===\n")
        f.write(f"n_fragments={n_frag}  frozen_identity_threshold={FROZEN_IDENTITY_THRESHOLD}  "
                f"rrf_k_const={RRF_K_CONST}  mmr_lambda_used={mmr_best_lambda}  mmr_verdict={mmr_verdict}\n\n")
        for r in all_results:
            f.write(f"--- {r['architecture']} ---\n")
            for k, v in r.items():
                if k == "architecture":
                    continue
                f.write(f"  {k}: {v}\n")
            f.write("\n")
        f.write("--- Supplementary threshold sweeps (NOT used to pick an operating point) ---\n")
        for name, rows in all_sweeps.items():
            f.write(f"\n{name}:\n")
            for row in rows:
                acc = f"{row['accuracy_on_assigned']:.3f}" if row["accuracy_on_assigned"] is not None else "NA"
                f.write(f"  threshold={row['threshold']:>5} pct_assigned={row['pct_assigned']*100:5.1f}% accuracy={acc}\n")

    print("\nDone. Outputs written to:", OUT_DIR)
    return all_results


if __name__ == "__main__":
    main()
