"""
Maximal Marginal Relevance (MMR) -- diversifies a single read/fragment's
own top-K retrieval candidate set, trading relevance against redundancy
with already-selected candidates. Operates independently per
read/fragment, never across reads -- intra-set diversification only,
never cross-read spreading of assignments. This matters because Part 4
already established that its dominant-species concentration (tuna) is
plausible real biology (NCBI-verified), not an artifact to suppress by
forcing diversity across the whole read set -- MMR here only reshapes
which candidates get PROPOSED to the nucleotide reranker for one
read/fragment at a time.

Standard formulation (the difference form, not a ratio -- a ratio is
undefined/unstable when the diversity term is 0, which happens on every
first pick):

    MMR(d) = lambda * relevance(d) - (1 - lambda) * max_{d' in S} sim(d, d')

where S is the already-selected set. For lambda > 0, the first pick
(S empty, so the diversity term is 0 for every candidate) is always the
highest-relevance candidate -- MMR never displaces the top relevance
pick, it only shapes which candidates fill the remaining slots. At
lambda == 0 the relevance term is zeroed out entirely (by design: "pure
diversity, ignore relevance"), so the first pick is an arbitrary,
deterministic tie among all candidates.

Representation: relevance is the candidate's own upstream retrieval
score (BM25 score here), min-max normalized WITHIN this one candidate
set -- BM25 scores are not comparable across different queries/fragments
(they depend on query length and term rarity), so only the within-set
relative ordering is meaningful. Similarity between two candidates reuses
the SAME representation as src/bm25_redundancy.py and Stage 0's
redundancy diagnostic: cosine similarity between the existing, frozen
DNABERT reference embeddings -- no new representation is introduced.
"""
from typing import List, Sequence
import numpy as np


def _normalize_relevance(scores: Sequence[float]) -> List[float]:
    """Min-max normalize to [0, 1]. A degenerate all-equal input (or a
    single-element input) maps to all 1.0s rather than dividing by zero."""
    lo, hi = min(scores), max(scores)
    if hi - lo < 1e-12:
        return [1.0] * len(scores)
    return [(s - lo) / (hi - lo) for s in scores]


def mmr_select(
    doc_idxs: Sequence,
    relevance_scores: Sequence[float],
    embeddings: np.ndarray,
    lam: float,
    k_results: int,
) -> List:
    """Greedy MMR selection over a single candidate set.

    doc_idxs: candidate identifiers, in the ORIGINAL retrieval rank
        order (used only for deterministic tie-breaking -- earlier
        original rank wins ties).
    relevance_scores: the matching upstream retrieval score for each
        doc_idx (e.g. BM25 score), same order/length as doc_idxs.
    embeddings: (len(doc_idxs), dim) array -- embeddings[i] corresponds
        to doc_idxs[i]. Assumed already L2-normalized (cosine similarity
        via dot product), consistent with how this project stores
        reference embeddings elsewhere.
    lam: relevance weight in [0, 1]. lam=1 reduces to pure relevance
        ranking (no diversification). lam=0 ignores relevance entirely.
    k_results: number of candidates to select (returns fewer if the
        input set is smaller).

    Returns doc_idxs reselected/reordered by MMR, deterministic given
    the same inputs.
    """
    n = len(doc_idxs)
    if n == 0:
        return []
    k_results = min(k_results, n)
    relevance = _normalize_relevance(relevance_scores)

    remaining = list(range(n))  # positions into doc_idxs / relevance / embeddings
    selected: List[int] = []

    while remaining and len(selected) < k_results:
        best_pos = None
        best_score = None
        for pos in remaining:
            if selected:
                sims = embeddings[pos] @ embeddings[selected].T
                diversity_penalty = float(np.max(sims))
            else:
                diversity_penalty = 0.0
            score = lam * relevance[pos] - (1 - lam) * diversity_penalty
            if best_score is None or score > best_score:
                best_score = score
                best_pos = pos
        selected.append(best_pos)
        remaining.remove(best_pos)

    return [doc_idxs[p] for p in selected]
