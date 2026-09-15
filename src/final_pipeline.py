"""
Stage 3, Task 5 -- final inference pipeline.

Implements the architecture Task 4's end-to-end evaluation (see
data/Final diagnostics/final_metrics.json) empirically selected:

    BM25 (k=6) -> MMR (lambda=0.5) -> nucleotide reranker -> frozen 95%
    identity threshold

Why this architecture, from measured evidence only (no assumed numbers):
  - A (BM25 alone) and B (BM25 -> MMR) are statistically indistinguishable
    at the decision level: both assign 948/1000 held-out fragments at the
    frozen 95% threshold, both at 98.7% accuracy on the assigned subset.
  - B additionally reduces candidate-set redundancy (mean pairwise
    embedding-cosine similarity 0.822 vs A's 0.860; near-duplicate-set
    rate 90.5% vs 98.0% -- Task 3), at zero cost to Recall@1/10 and a
    negligible Recall@5 change (+0.001) -- so MMR is added because it is
    measurably free, not because it was "planned".
  - D (RRF) reached a marginally higher accuracy (98.9%, 938/948 vs
    936/948 -- a 2-fragment difference) but requires running DNABERT at
    inference time in addition to BM25, for a difference this small.
    Combined with Stage 2's already-established finding that RRF ties
    BM25 on Recall@10 while costing Recall@1, this does not meet the bar
    of "new evidence unexpectedly demonstrating a benefit" -- RRF is
    excluded from the production path per that explicit rule.
  - C (DNABERT alone) reproduces Part 4's frozen baseline exactly (598/1000
    assigned, 94.6% accuracy at threshold 95) -- confirming this harness
    is consistent with the already-validated baseline -- and is far below
    BM25's 948/1000, which is the whole basis for adopting BM25 at all.

This module is a thin composition of already-existing, tested components
-- it introduces no new retrieval, diversification, or reranking logic:
  - src/bm25_retrieval.py   (BM25Index)          -- unmodified
  - src/mmr.py               (mmr_select)         -- unmodified
  - src/nucleotide_rerank.py (rerank_candidates)  -- FROZEN, unmodified

Does not modify the reference database, primers, extracted region, or
the 95% threshold.

Usage:
    from src.final_pipeline import PipelineIndex, assign

    idx = PipelineIndex.build("data/MitoFish results/mitofish_16s_reference_table.csv",
                               "data/MitoFish results/mitofish_16s_ref_embeddings.npy",
                               "data/MitoFish results/mitofish_16s_ref_ids.csv")
    result = assign(read_seq, idx)
    # result["decision"] -> "assigned" or "unclassified"
"""
import csv
from dataclasses import dataclass
from typing import List, Optional
import numpy as np

from .bm25_retrieval import BM25Index
from .mmr import mmr_select
from .nucleotide_rerank import rerank_candidates
from .faiss_index import l2_normalize

BM25_K_MER = 6         # Stage 1's held-out-selected value
MMR_LAMBDA = 0.5        # Task 3's held-out-selected value
MMR_POOL_SIZE = 30      # candidates MMR chooses from (must exceed TOP_K)
TOP_K = 10               # Part 4's established candidate-set size
FROZEN_IDENTITY_THRESHOLD = 95.0  # Part 4's calibrated operating point -- not re-derived here
BEST_GUESS_MIN_IDENTITY = 75.0    # below this, a best_guess is relabeled low-confidence


@dataclass
class PipelineIndex:
    ref_ids: List[str]
    ref_seqs: List[str]
    ref_embeddings: np.ndarray  # L2-normalized, row-aligned with ref_ids/ref_seqs
    bm25: BM25Index

    @classmethod
    def build(cls, reference_table_csv: str, reference_embeddings_npy: str, reference_ids_csv: str) -> "PipelineIndex":
        ref_ids, ref_seqs = [], []
        with open(reference_table_csv) as f:
            for row in csv.DictReader(f):
                ref_ids.append(row["Reference_ID"])
                ref_seqs.append(row["Sequence"])

        with open(reference_ids_csv) as f:
            emb_ids = [row["Reference_ID"] for row in csv.DictReader(f)]
        assert emb_ids == ref_ids, "reference_ids_csv order must match reference_table_csv order"

        ref_embeddings = l2_normalize(np.load(reference_embeddings_npy).astype("float32"))

        bm25 = BM25Index(k=BM25_K_MER)
        bm25.build(ref_ids, ref_seqs)

        return cls(ref_ids=ref_ids, ref_seqs=ref_seqs, ref_embeddings=ref_embeddings, bm25=bm25)


def assign(read_seq: str, index: PipelineIndex) -> dict:
    """Runs one read through BM25 -> MMR -> the frozen nucleotide
    reranker -> the frozen 95% threshold decision. Returns a dict with
    the same shape as Part 4's read_taxonomy_results_mitofish_16s_reranked.csv
    columns, plus a few extra diagnostic fields.

    Confidence_Decision is one of:
      - "assigned": identity_pct >= FROZEN_IDENTITY_THRESHOLD (95%, calibrated
        in Part 4) -- same bar as before, unchanged.
      - "best_guess": identity_pct in [BEST_GUESS_MIN_IDENTITY, 95) -- a
        candidate was found and is plausible enough to report as a lead, but
        below the calibrated confidence bar.
      - "best_guess_low_confidence": identity_pct < BEST_GUESS_MIN_IDENTITY --
        a candidate was found but the match is weak; reported so "closest
        known relative" is never silently discarded, but flagged as
        low-probability rather than a lead worth trusting.
      Neither BEST_GUESS_MIN_IDENTITY nor the 95% bar is recalibrated here --
      they only change what label gets attached to reads that don't clear
      the frozen 95% bar; the nearest match is always reported when one
      exists.
      - "unclassified": no BM25 candidates at all (e.g. read shares no k-mers
        with any reference -- nothing to report a nearest match for).
    """
    pool = index.bm25.top_k(read_seq, k_results=MMR_POOL_SIZE)  # [(doc_idx, bm25_score), ...]

    if not pool:
        return {
            "Nearest_Reference_ID": "", "identity_pct": 0.0, "coverage_pct": 0.0,
            "bm25_score": 0.0, "Confidence_Decision": "unclassified",
        }

    doc_idxs = [d for d, _s in pool]
    scores = [s for _d, s in pool]
    emb_block = index.ref_embeddings[doc_idxs]
    selected = mmr_select(doc_idxs, scores, emb_block, lam=MMR_LAMBDA, k_results=TOP_K)

    score_by_idx = dict(pool)
    candidates = [(index.ref_ids[idx], index.ref_seqs[idx], score_by_idx[idx]) for idx in selected]
    reranked = rerank_candidates(read_seq, candidates)  # FROZEN, unmodified
    best = reranked[0]

    if best["identity_pct"] >= FROZEN_IDENTITY_THRESHOLD:
        decision = "assigned"
    elif best["identity_pct"] >= BEST_GUESS_MIN_IDENTITY:
        decision = "best_guess"
    else:
        decision = "best_guess_low_confidence"

    return {
        "Nearest_Reference_ID": best["Reference_ID"],
        "identity_pct": best["identity_pct"],
        "coverage_pct": best["coverage_pct"],
        "bm25_score": best["dnabert_similarity"],  # field name inherited from rerank_candidates' generic 3rd-tuple slot
        "Confidence_Decision": decision,
    }


def assign_batch(read_ids: List[str], read_seqs: List[str], index: PipelineIndex, out_csv: Optional[str] = None,
                  fetch_species_names: bool = False) -> List[dict]:
    """fetch_species_names=True adds a "Species" column, looked up once per
    unique Nearest_Reference_ID via species_lookup.fetch_organism_names
    (NCBI, cached to disk) -- MitoFish accessions carry no species name on
    their own. Off by default so tests and offline runs never touch the
    network."""
    results = []
    for rid, seq in zip(read_ids, read_seqs):
        r = assign(seq, index)
        r["Read_ID"] = rid
        results.append(r)

    if fetch_species_names:
        from .species_lookup import fetch_organism_names
        accessions = sorted({r["Nearest_Reference_ID"] for r in results if r["Nearest_Reference_ID"]})
        names = fetch_organism_names(accessions)
        for r in results:
            r["Species"] = names.get(r["Nearest_Reference_ID"], "") if r["Nearest_Reference_ID"] else ""

    if out_csv:
        fieldnames = ["Read_ID", "Nearest_Reference_ID", "identity_pct", "coverage_pct", "bm25_score", "Confidence_Decision"]
        if fetch_species_names:
            fieldnames.append("Species")
        with open(out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(results)

    return results
