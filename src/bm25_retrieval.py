"""
BM25 lexical retrieval over k-mer-tokenized DNA sequences -- an
independent, non-neural retrieval arm alongside the existing DNABERT/FAISS
retrieval (src/faiss_index.py, src/query_index.py). Built to test whether
a structurally different retrieval signal (exact/near-exact k-mer overlap)
can recover references that DNABERT's embedding-space top-10 misses --
see src/stage0_diagnostics.py, which measured DNABERT top-10 recall on
held-out MitoFish fragments capping at ~58%, well short of saturated.

Representation: overlapping k-mers, sliding window, stride 1, forward
strand only. No reverse-complement canonicalization -- consistent with
nucleotide_rerank.align_pair, which also never reverse-complements; the
real-read alignments in Part 4 already show reads sit in matching
orientation to the reference amplicons, so there is no evidence this
pipeline needs RC handling. k-mers containing any non-ACGT character
(ambiguous IUPAC codes in a reference, or an 'N' base call in a real
read) are skipped outright, not padded or substituted -- a malformed
term would either silently fail to match anything (harmless) or, worse,
coincidentally collide with an unrelated real k-mer.

Scoring: standard Okapi BM25 (Robertson / Sparck-Jones), implemented
directly over a plain Python inverted index -- no new third-party
dependency, and fast enough at this corpus scale (tens of thousands of
reference sequences, each a few hundred bp). k1/b use the standard
IR-literature defaults (1.5 / 0.75), not tuned on this project's data --
only the k-mer length k is chosen empirically (via a held-out sweep in
src/stage1_bm25_diagnostics.py), so the scoring formula itself is not an
extra hidden degree of freedom.
"""
from typing import List, Dict, Tuple
from collections import defaultdict
import math

_ACGT = frozenset("ACGT")


def kmers(seq: str, k: int) -> List[str]:
    """Overlapping k-mers, stride 1, forward strand only, uppercased.
    Skips any window containing a non-ACGT character rather than
    emitting a malformed term. Returns [] if the sequence is shorter
    than k."""
    seq = seq.upper()
    n = len(seq) - k + 1
    if n <= 0:
        return []
    out = []
    for i in range(n):
        window = seq[i:i + k]
        if _ACGT.issuperset(window):
            out.append(window)
    return out


class BM25Index:
    """Inverted index + Okapi BM25 scoring over a fixed corpus of
    (doc_id, sequence) pairs, tokenized into k-mers."""

    def __init__(self, k: int, k1: float = 1.5, b: float = 0.75):
        self.k = k
        self.k1 = k1
        self.b = b
        self.doc_ids: List[str] = []
        self.doc_len: List[int] = []
        self.avg_doc_len: float = 0.0
        self.postings: Dict[str, Dict[int, int]] = {}  # term -> {doc_idx: tf}
        self.n_docs: int = 0

    def build(self, ids: List[str], seqs: List[str]) -> None:
        """Builds the index over exactly the (ids, seqs) passed in --
        callers are responsible for excluding any held-out sequences
        before calling this (see stage1_bm25_diagnostics.py, which
        passes only the TRAIN partition of the reference table)."""
        self.doc_ids = list(ids)
        self.n_docs = len(ids)
        self.doc_len = []
        postings: Dict[str, Dict[int, int]] = defaultdict(dict)
        for doc_idx, seq in enumerate(seqs):
            terms = kmers(seq, self.k)
            self.doc_len.append(len(terms))
            tf: Dict[str, int] = defaultdict(int)
            for t in terms:
                tf[t] += 1
            for t, count in tf.items():
                postings[t][doc_idx] = count
        self.postings = dict(postings)
        self.avg_doc_len = sum(self.doc_len) / self.n_docs if self.n_docs else 0.0

    def _idf(self, term: str) -> float:
        """Robertson/Lucene-style IDF with +1 smoothing (never negative,
        even for a term present in every document)."""
        postings = self.postings.get(term)
        df = len(postings) if postings else 0
        if df == 0:
            return 0.0
        return math.log((self.n_docs - df + 0.5) / (df + 0.5) + 1.0)

    def score_query(self, query_seq: str) -> Dict[int, float]:
        """Returns {doc_idx: bm25_score} for every doc sharing >=1 k-mer
        with the query. A doc with zero shared k-mers is simply absent
        from the returned dict (never a zero-score entry)."""
        q_terms = set(kmers(query_seq, self.k))
        scores: Dict[int, float] = defaultdict(float)
        for t in q_terms:
            postings = self.postings.get(t)
            if not postings:
                continue
            idf = self._idf(t)
            if idf <= 0:
                continue
            for doc_idx, tf in postings.items():
                dl = self.doc_len[doc_idx]
                denom = tf + self.k1 * (1 - self.b + self.b * dl / self.avg_doc_len) if self.avg_doc_len else tf
                if denom:
                    scores[doc_idx] += idf * (tf * (self.k1 + 1)) / denom
        return dict(scores)

    def top_k(self, query_seq: str, k_results: int = 10) -> List[Tuple[int, float]]:
        """Returns up to k_results (doc_idx, score) pairs sorted by score
        descending. Returns FEWER than k_results if fewer documents share
        any k-mer with the query -- never pads with zero-score docs."""
        scores = self.score_query(query_seq)
        ranked = sorted(scores.items(), key=lambda kv: -kv[1])
        return ranked[:k_results]
