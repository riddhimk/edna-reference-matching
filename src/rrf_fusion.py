"""
Reciprocal Rank Fusion (RRF) -- combines independently-ranked candidate
lists (e.g. DNABERT cosine-similarity rank order, BM25 rank order) into a
single fused ranking, purely by rank position rather than raw score. This
avoids needing to calibrate DNABERT's ~0-1 cosine range against BM25's
unbounded score range.

RRF_score(d) = sum over lists L that contain d: 1 / (k_const + rank_L(d))

A candidate absent from a given list contributes 0 for that list (not a
penalty rank), so a candidate found by only one method is still scored,
just without that method's contribution.
"""
from typing import Dict, List, Sequence


def rrf_scores(ranked_lists: Sequence[Sequence[int]], k_const: float) -> Dict[int, float]:
    """ranked_lists: one or more rank-ordered sequences of doc_idx (best
    first, rank = position + 1). Returns {doc_idx: fused_rrf_score} for
    the union of all doc_idx appearing in ANY list."""
    scores: Dict[int, float] = {}
    for ranked in ranked_lists:
        for pos, doc_idx in enumerate(ranked):
            rank = pos + 1
            scores[doc_idx] = scores.get(doc_idx, 0.0) + 1.0 / (k_const + rank)
    return scores


def fuse_top_k(ranked_lists: Sequence[Sequence[int]], k_const: float, k_results: int = 10) -> List[int]:
    """Returns up to k_results doc_idx, sorted by fused RRF score
    descending. Ties are broken by insertion order (the order doc_idx
    first appears scanning the input lists left to right) -- a
    deterministic but otherwise arbitrary tie-break, documented here
    rather than left implicit."""
    scores = rrf_scores(ranked_lists, k_const)
    ranked = sorted(scores.items(), key=lambda kv: -kv[1])
    return [doc_idx for doc_idx, _score in ranked[:k_results]]
