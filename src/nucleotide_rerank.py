"""
Independent nucleotide-level re-ranking of DNABERT/FAISS top-K candidates.

Why: the MitoFish run (see data/MitoFish results/run_report.md) found
severe "hubness" -- a small handful of reference embeddings act as nearest
neighbor for hundreds of unrelated reads, which is a known embedding-space
artifact, not biology. DNABERT cosine similarity alone should not be
trusted as the sole evidence for a species call. This module adds a
SECOND, independent check: real local sequence alignment (Smith-Waterman,
via Biopython's Align.PairwiseAligner in "local" mode) between the raw
read and each of its top-K DNABERT candidates, and re-ranks candidates by
alignment identity instead of embedding similarity alone.

Why Smith-Waterman / local alignment (not global): reads (118-140bp) are
shorter than the reference amplicons (up to 249bp) and may not start/end
at the same position within the amplicon (unknown exact primer-trim
offset), so a LOCAL alignment that finds the best-scoring subregion is the
correct method here -- a global (end-to-end) alignment would incorrectly
penalize the length mismatch itself.

Percent identity is computed as identities / (identities + mismatches +
gaps) over the aligned region (a standard percent-identity definition),
and separately reports what fraction of the read's own length was covered
by the alignment (coverage) -- a short, spurious high-identity match to
just a few bases of the read would show high identity but very low
coverage, and should not be trusted as a genuine species match.
"""
from typing import List, Dict, Tuple
from Bio import Align

_aligner = None


def _get_aligner():
    global _aligner
    if _aligner is None:
        _aligner = Align.PairwiseAligner()
        _aligner.mode = "local"
        _aligner.match_score = 2
        _aligner.mismatch_score = -1
        _aligner.open_gap_score = -2
        _aligner.extend_gap_score = -0.5
    return _aligner


def align_pair(read_seq: str, ref_seq: str) -> Dict:
    """Local-align read_seq against ref_seq. Returns identity_pct (0-100),
    coverage_pct (0-100, fraction of READ length covered by the aligned
    region), raw alignment score, and aligned length."""
    aligner = _get_aligner()
    alns = aligner.align(read_seq.upper(), ref_seq.upper())
    best = alns[0]
    counts = best.counts()
    aligned_len = counts.aligned
    total = counts.identities + counts.mismatches + counts.gaps
    identity_pct = 100.0 * counts.identities / total if total else 0.0
    # best.aligned[0] gives the (start,end) blocks on the read side
    read_start = int(best.aligned[0][0][0])
    read_end = int(best.aligned[0][-1][-1])
    coverage_pct = 100.0 * (read_end - read_start) / len(read_seq) if read_seq else 0.0
    return {
        "score": float(best.score),
        "identity_pct": round(identity_pct, 2),
        "coverage_pct": round(coverage_pct, 2),
        "aligned_len": aligned_len,
    }


def rerank_candidates(read_seq: str, candidates: List[Tuple[str, str, float]]) -> List[Dict]:
    """candidates: list of (ref_id, ref_seq, dnabert_similarity), in DNABERT
    rank order (best first). Returns a list of dicts (one per candidate),
    each with the DNABERT rank/similarity AND the independent alignment
    identity/coverage, sorted by alignment identity_pct descending (ties
    broken by DNABERT rank, i.e. the original order)."""
    results = []
    for rank, (ref_id, ref_seq, dnabert_sim) in enumerate(candidates, start=1):
        aln = align_pair(read_seq, ref_seq)
        results.append({
            "Reference_ID": ref_id,
            "dnabert_rank": rank,
            "dnabert_similarity": dnabert_sim,
            **aln,
        })
    results.sort(key=lambda r: (-r["identity_pct"], r["dnabert_rank"]))
    return results
