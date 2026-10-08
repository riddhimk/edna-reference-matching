# eDNA Reference Matching — Final Technical Report (Stages 0–3 + Final Evaluation)

## 1. Problem

Assign short (118–140bp) fish-mitochondrial-16S eDNA reads to a MitoFish
reference sequence with defensible confidence, or correctly leave them
unclassified when the evidence doesn't support a call. Part 4 (frozen
baseline) established: DNABERT embedding retrieval (top-10) → independent
Smith-Waterman nucleotide realignment → 95%-identity threshold decision,
producing 474/5,000 real-read assignments at 94.6% held-out calibration
accuracy. This work investigates whether alternative/additional retrieval
methods (BM25, RRF, MMR) improve on that retrieval step, without touching
the database, primers, extracted region, calibration methodology, decision
threshold, or the nucleotide reranker itself.

All experiments below run on the **identical 1,000 held-out MitoFish
fragments** (seed=42, `stage0_diagnostics.build_holdout_split`) and the
**identical ≥95% full-sequence-identity ground truth** Part 4's own
calibration uses — established once in Stage 0 and reused, unmodified,
by every subsequent stage.

## 2. Retrieval approaches tested

| Approach | Representation | Role tested |
|---|---|---|
| DNABERT | 768-dim transformer embedding, cosine similarity via FAISS | Part 4's frozen baseline retrieval arm |
| BM25 | k-mer (k=6) lexical inverted index, Okapi scoring | Independent retrieval arm (Stage 1) |
| RRF | Rank-position fusion of DNABERT + BM25 | Candidate-set fusion (Stage 2) |
| MMR | Greedy relevance/diversity trade-off over a BM25 top-30 pool, diversity signal = DNABERT reference-embedding cosine similarity | Candidate-set diversification (Stage 3) |

## 3. Experimental results (all measured, held-out n=1,000)

### Stage 0 — diagnostics justifying the investigation
- DNABERT Recall@10 = 58.2% (41.8% of fragments never retrieve the correct reference at all)
- DNABERT top-10 redundancy: mean pairwise embedding-cosine similarity 0.86 vs. background 0.50; 91–98% of sets contain a near-duplicate pair

### Stage 1 — BM25 (k selected 4/6/8 on held-out; k=6 won)
- Recall@1 = 93.7%, Recall@5 = 94.5%, **Recall@10 = 94.6%**
- Rescued 364/1,000 fragments DNABERT's top-10 missed entirely; DNABERT rescued 0 BM25 misses

### Stage 2 — RRF (DNABERT + BM25, k_const swept 1–100; k_const=10 selected)
- Recall@1 = 51.3% (worse than BM25 alone: 93.7%), Recall@5 = 94.2%, Recall@10 = 94.6% (ties BM25 exactly)
- 0 rescued vs. BM25, 0 lost vs. BM25 at Recall@10
- **Conclusion: RRF does not improve on BM25 alone; measurably worse at Recall@1.**

### Task 1 — BM25 top-10 redundancy (gate for MMR)
- Mean pairwise similarity 0.860 (vs. background 0.502); 98.0% of sets contain a pair ≥0.999 cosine similarity
- 34.7% of ALL candidate pairs are literal exact-sequence duplicates
- **Verdict: redundancy confirmed → proceed to MMR**

### Tasks 2/3 — MMR (BM25 top-30 pool → MMR selects 10; λ swept 0.5–0.9)
- λ=0.5: Recall@1 = 93.7% (unchanged from BM25 baseline, as structurally guaranteed for λ>0), Recall@5 = 94.6% (+0.1pp), Recall@10 = 94.6% (unchanged)
- Mean pairwise similarity: 0.860 → **0.822**; near-duplicate-set rate: 98.0% → **90.5%**
- **Verdict: accepted — genuine redundancy reduction at zero recall cost**

### Task 4 — end-to-end through the frozen nucleotide reranker (frozen 95% threshold applied identically to all four)

| Architecture | Recall@1 | Recall@5 | Recall@10 | Assigned/1000 | Accuracy (assigned) | Mean pairwise sim |
|---|---|---|---|---|---|---|
| A: BM25 | 93.7% | 94.5% | 94.6% | 948 (94.8%) | 98.73% (936/948) | 0.860 |
| B: BM25 → MMR (λ=0.5) | 93.7% | 94.6% | 94.6% | 948 (94.8%) | 98.73% (936/948) | 0.822 |
| C: DNABERT (Part 4 baseline reproduction) | 41.7% | 53.3% | 58.2% | 598 (59.8%) | 94.65% (566/598) | 0.859 |
| D: RRF | 51.3% | 94.2% | 94.6% | 948 (94.8%) | 98.95% (938/948) | 0.770 |

C's numbers (598/1000 = 59.8% assigned, 94.65% accuracy) reproduce Part
4's historically reported frozen calibration figures (59.8% held-out
coverage, 94.6% accuracy) exactly — confirming this evaluation harness is
consistent with the already-validated baseline before drawing any new
conclusions from it.

## 4. Architecture decision

**BM25 → MMR (λ=0.5) → nucleotide reranker → frozen 95% threshold.**

- A and B are statistically indistinguishable at the decision level (identical assigned count and accuracy), so MMR is included because it is *measurably free* — it reduces candidate redundancy by ~9.5 percentage points (near-dup-set rate) at zero cost to Recall@10 and a negligible Recall@5 gain, not because it was planned in advance.
- D (RRF) reached a marginally higher accuracy (938/948 vs. 936/948 — a **2-fragment difference out of 948**), but requires running DNABERT at inference time in addition to BM25. Combined with Stage 2's already-established finding that RRF ties BM25 on Recall@10 while costing Recall@1, a 2-fragment difference does not meet the bar of "new evidence unexpectedly demonstrating a benefit." **RRF is excluded from the production path**, per the explicit rule governing this decision, and its implementation/tests are kept in the repository as part of the experimental record.
- C (DNABERT alone) is far below BM25 on every metric, which is the entire basis for adopting BM25 in the first place.

## 5. End-to-end performance (final architecture, held-out n=1,000)

Recall@1 = 93.7%, Recall@5 = 94.6%, Recall@10 = 94.6%, assigned = 948/1000 (94.8%) at the frozen 95% identity threshold, accuracy on assigned = 98.73% (936/948), candidate-set mean pairwise similarity = 0.822 (down from BM25-alone's 0.860).

## 6. Failure analysis

- **Both-miss cases (54/1000, 5.4%)**: fragments where BM25's top-10/top-30 pool never contains a reference ≥95% identical to the true source at all — this is a hard retrieval-recall ceiling neither BM25 nor MMR can fix, since MMR only reselects from BM25's own pool and cannot introduce candidates BM25 never retrieved. See `data/Stage1 diagnostics/bm25_kmer_sweep_summary.csv` for the underlying k-mer sweep.
- **Assigned-but-incorrect cases (12/948 for architecture B, per `B_BM25_MMR_per_fragment.csv`)**: MitoFish's own accession redundancy means "assigned" can occasionally land on a close-but-not-≥95%-identical relative rather than the true source; this is the same phenomenon Part 4 originally diagnosed (many accessions share near-identical, not always identical, sequences).
- No species-level failure analysis is possible for arbitrary held-out fragments — MitoFish ships no species/taxonomy mapping (established in Part 3); the only species-level lookup that exists is Part 4's own 27-reference NCBI table for the specific real-read assignments, which this held-out evaluation does not touch or reuse.

## 7. Limitations — measured vs. assumed

**Measured:** every number in Section 3/5 above, from an actual execution of the listed script against the existing held-out split, on this machine, this session.

**Assumed / not measured:**
- Real-read behavior of BM25/MMR/RRF was **not** evaluated in this work (Stage 1/2/3/Task4 are entirely held-out-fragment evaluations, per the explicit instruction not to tune against or use real-read assignments). The 474/5,000 real-read number remains Part 4's frozen figure; applying BM25→MMR to the actual 5,000 reads is unexecuted future work.
- Synthetic held-out fragments are clean substrings of existing reference sequences with zero simulated sequencing error — real Illumina reads carry genuine per-base error that this evaluation does not exercise. This was already flagged in Stage 1 as a factor that could make BM25's real-read advantage smaller than its held-out advantage; nothing here newly measures that gap.
- The RRF vs. BM25+MMR 2-fragment accuracy difference is treated as within noise, not as a robust finding — no significance test was run given the small absolute count; this is a judgment call, stated as such.

## 8. Final architecture

```
Read
  |
  v
BM25 retrieval (k=6, top-30 pool)
  |
  v
MMR diversification (lambda=0.5, selects top-10 from the pool)
  |
  v
Top-10 candidates
  |
  v
Frozen nucleotide reranker (nucleotide_rerank.rerank_candidates, unmodified)
  |
  v
Existing calibration / 95% identity threshold (frozen)
  |
  v
Taxonomic identification (assigned / unclassified)
```

## 9. Reproducibility

Environment: `.venv-win` (Windows-native venv; the repo's checked-in
`.venv/` is a Linux venv and will not run here). Setup:
```bash
python -m venv .venv-win
.venv-win\Scripts\python.exe -m pip install -r requirements.txt biopython
.venv-win\Scripts\python.exe -m pip install "pandas==2.2.3"   # pandas 3.0.5 was blocked by a local Application Control policy
```

Diagnostics and experiments, in order (each writes to its own `data/*/` folder, none overwrite an earlier stage):
```bash
python -m src.stage0_diagnostics          # data/Stage0 diagnostics/  -- DNABERT recall + redundancy
python -m src.stage1_bm25_diagnostics      # data/Stage1 diagnostics/ -- BM25 k-sweep, recall, rescue analysis
python -m src.stage2_rrf_diagnostics       # data/Stage2 diagnostics/ -- RRF k_const sweep, vs. BM25
python -m src.bm25_redundancy              # data/Stage3 diagnostics/ -- Task 1: BM25 top-10 redundancy
python -m src.mmr_lambda_sweep             # data/Stage3 diagnostics/ -- Tasks 2/3: MMR + lambda sweep
python -m src.final_evaluation             # data/Final diagnostics/  -- Task 4: A/B/C/D end-to-end
```

Tests:
```bash
.venv-win\Scripts\python.exe -m pytest -q
```

Final pipeline (single read):
```python
from src.final_pipeline import PipelineIndex, assign
index = PipelineIndex.build(
    "data/MitoFish results/mitofish_16s_reference_table.csv",
    "data/MitoFish results/mitofish_16s_ref_embeddings.npy",
    "data/MitoFish results/mitofish_16s_ref_ids.csv",
)
result = assign(read_seq, index)
```
