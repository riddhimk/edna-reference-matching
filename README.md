# eDNA-TaxAI — SILVA Reference-Matching Stage

Implements: SILVA FASTA → reference table → DNABERT-S reference embeddings →
FAISS index → nearest-reference search → confidence decision → taxonomy
assignment or fall-through to the existing HDBSCAN open-set stage.

This preserves your existing architecture and does not touch or re-run your
completed 5,000-read DNABERT-S + HDBSCAN + UMAP results (18 clusters, 218
noise reads).

## Status honesty note

Nothing in this code has been run against your real files from this
conversation — I don't have your full SILVA FASTA or FASTQ on disk here.
Every stage is written to be run by you, in order, verifying the stated
output before moving to the next. The two SILVA header/sequence records
actually inspected (used to derive the schema and the findings below) are
hardcoded into `tests/test_fasta_io.py` as regression tests — run `pytest`
to confirm the parser handles them correctly before trusting it on the full
file.

## Two findings from those two real records that change your design decisions

1. **RNA alphabet.** SILVA sequences use `U`, not `T`. `fasta_io.to_dna()`
   converts a derived copy before tokenization — the stored/original
   sequence is never mutated.
2. **Region mismatch.** The two inspected references are 1,779 bp and
   1,860 bp (near full-length SSU rRNA); your experimental reads are
   118–140 bp. Do **not** set `config.MAX_LENGTH = 140` just because that
   matches your read length — that truncates references from position 1,
   which is very unlikely to be the amplified hypervariable region. Resolve
   which primer/region produced your reads first (check SRA run metadata
   for `ERR16922870`, or map a sample of reads back to full-length
   references) before setting `MAX_LENGTH` and before running
   `embed-ref` / `embed-reads`. Document this explicitly if you don't
   resolve it in time for the viva — that's a defensible limitation;
   silently truncating and not mentioning it isn't.

## Also confirmed: taxonomy depth is not fixed

Record 1 has 8 semicolon-delimited taxonomy fields; record 2 has 17. Do not
assume a fixed Domain/Kingdom/Phylum/.../Species ladder by field position —
`build_reference_table.py` only trusts `Domain` (field 0) and
`Organism_Label` (last field) positionally; everything else is kept as the
full `Taxonomy_Full` string.

## Setup

```bash
pip install -r requirements.txt
mkdir -p data
# copy SILVA_138.2_SSURef_tax_silva_trunc.fasta and ERR16922870_1.fastq into data/
```

## Run order

```bash
# Stage 1 — inspect the real file, get real counts/lengths/duplicates
python run_pipeline.py inspect data/SILVA_138.2_SSURef_tax_silva_trunc.fasta

# Stage 2 — build the reference table (check malformed-header count printed)
python run_pipeline.py parse

# Resolve region-mismatch (see above), then set config.MAX_LENGTH from the
# real percentile output of Stage 1 — do not guess it.

# Stage 5 — embed the SILVA reference table (requires model download / GPU)
python run_pipeline.py embed-ref

# Stage 6 — only if your existing read embeddings used a different
# tokenizer/pooling config than src/embed_common.py — otherwise reuse them
python run_pipeline.py embed-reads

# Stage 7 — build the FAISS cosine-similarity index
python run_pipeline.py build-index

# Stage 8 — query with experimental reads (needs data/reads_ids.csv, one
# Read_ID per line, in the same row order as your embeddings .npy)
python run_pipeline.py query

# Stage 12 — run src/evaluate.py's threshold sweep against a labeled
# holdout to pick a real SIMILARITY_THRESHOLD / MIN_MARGIN, set them in
# src/config.py, THEN:

# Stage 9-10 — attach taxonomy + make the assigned/unclassified decision
python run_pipeline.py decide
```

Stage 11 (open-set routing) is a plain function
(`src/open_set_routing.get_unclassified_read_ids`) to call from wherever
your existing HDBSCAN code lives — it returns the Read_IDs that should be
looked up against your existing embeddings and fed into the existing
clustering stage, unchanged.

## Cluster validation against SILVA (what your professor is actually asking for)

If the goal is **"do our 18 HDBSCAN clusters correspond to real taxonomic
groups, checked against a standard reference database"** — that's a
different question from Stages 9-11 above, and it's what
`src/cluster_annotation.py` answers. It does NOT re-run or change your
HDBSCAN clustering; it takes your existing cluster assignments and checks
them against SILVA.

You need `python run_pipeline.py query` to have been run on **all** your
reads first (not filtered to any subset), plus a CSV of your existing
cluster assignments:

```
data/hdbscan_cluster_assignments.csv
Read_ID,Cluster_ID
read001,0
read002,0
read003,-1      <- HDBSCAN noise reads use -1 by convention
...
```

Then:

```bash
python run_pipeline.py annotate-clusters --min-similarity 0.7 --taxonomy-depth 3
```

- `--min-similarity`: reads below this FAISS similarity are treated as "no
  confident reference match" (not silently counted as a match). Use the
  threshold from your Stage 12 sweep once you have it; omit the flag to
  label every read regardless of confidence and inspect the raw picture
  first.
- `--taxonomy-depth`: how many leading taxonomy fields to use as the
  pseudo-label (e.g. depth=3 → `"Eukaryota;Amorphea;Metazoa"`). SILVA depth
  is variable (8 vs. 18 fields seen in the two real records inspected), so
  there's no universally correct value — try 2-3 different depths and
  report what you used.

Output:
- `data/cluster_read_annotations.csv` — every read, its cluster, its
  nearest SILVA reference and taxonomy label
- `data/cluster_summary.csv` — one row per cluster: size, % with a
  confident match, dominant taxon, **purity** (what fraction of the
  matched reads in that cluster agree on the same taxon), how many
  distinct taxa appear in the cluster
- printed **agreement metrics**: Adjusted Rand Index, Normalized Mutual
  Information, Homogeneity, Completeness, V-measure — comparing your
  cluster labels against SILVA pseudo-labels, excluding noise and
  unmatched reads (this is exactly the ARI/NMI/homogeneity/completeness
  set your project already targets)

**Read this before quoting purity/ARI numbers in your viva**: SILVA's
nearest-neighbor hit is a *pseudo-label*, not verified ground truth — a
single read's top match can be wrong independent of whether your
clustering is good. High purity is evidence the clusters are taxonomically
coherent, not proof every read's species call is correct. Report it as
"agreement with SILVA nearest-reference matches," not as "% accuracy."

## Tests

```bash
pytest
```

All tests run offline (no model download, no GPU) — they check parsing
logic against the real inspected headers, normalization math, and decision
logic, not model output.

## File map

| File | Stage |
|---|---|
| `src/config.py` | central config / placeholders that must be set from real data |
| `src/fasta_io.py` | shared FASTA parsing (Stage 1/2 foundation) |
| `src/inspect_silva.py` | Stage 1 — full-file inspection |
| `src/build_reference_table.py` | Stage 2 — reference table CSV |
| `src/embed_common.py` | shared DNABERT-S loading/embedding (Stage 3-4 logic) |
| `src/embed_reference.py` | Stage 5 — reference embeddings |
| `src/embed_experimental.py` | Stage 6 — read embeddings (only if reconciling) |
| `src/faiss_index.py` | Stage 7 — FAISS index build |
| `src/query_index.py` | Stage 8 — query |
| `src/confidence.py` | Stage 9-10 — taxonomy attach + decision |
| `src/open_set_routing.py` | Stage 11 — route to existing HDBSCAN |
| `src/cluster_annotation.py` | cluster validation — purity/ARI/NMI of existing HDBSCAN clusters vs. SILVA |
| `src/evaluate.py` | Stage 12 — threshold sweep + metrics |
| `run_pipeline.py` | CLI to run each stage |

### Final architecture (post Part 4, see `Explanation.md` Parts 3–5)

| File | Role |
|---|---|
| `src/bm25_retrieval.py` | BM25 (k-mer) lexical retrieval index |
| `src/mmr.py` | MMR candidate-set diversification |
| `src/rrf_fusion.py` | RRF fusion (kept for the record; excluded from production — see Part 4/5) |
| `src/nucleotide_rerank.py` | frozen Smith-Waterman nucleotide reranker |
| `src/final_pipeline.py` | production pipeline: BM25 → MMR → nucleotide reranker → tiered confidence decision |
| `src/species_lookup.py` | NCBI accession → species name lookup, disk-cached |
| `src/final_evaluation.py` | Task 4 A/B/C/D architecture comparison (`data/Final diagnostics/`) |
| `src/nt_embed.py` | Nucleotide Transformer v2 (50M, multi-species) embedding — isolated, second model |
| `src/nt_diagnostics.py` | DNABERT vs Nucleotide Transformer Recall@K comparison (`data/NT diagnostics/`) |

## Model comparison: DNABERT-S vs Nucleotide Transformer v2 (50M)

Tested whether an open-source model swap improves retrieval (professor's
"open-source LLM models for retrieval" / "comparison of models" next
steps). Full writeup and technical-requirement analysis: **`DNABERT-S vs
Nucleotide Comparison.docx`** (project root). Short version:

| Metric (Recall@K, identical 1,000 held-out fragments) | DNABERT-S (existing) | Nucleotide Transformer v2 50M |
|---|---|---|
| Recall@1 | 41.7% | 15.3% |
| Recall@5 | 53.3% | 20.1% |
| Recall@10 | 58.2% | 22.3% |

**Result: DNABERT-S wins clearly.** Nucleotide Transformer is a general-purpose
DNA model; DNABERT-S was purpose-trained for species differentiation, which is
exactly this task. Not adopted — `final_pipeline.py` is unchanged. Reproduce
with `python -m src.nt_diagnostics` (~7-8 min on CPU, no GPU required).

## Roadmap (professor's next steps, tracked in `Explanation.md` Part 5/6)

1. Open-source LLM models for retrieval — comparison of models — which classifies DNA better — **done**, Nucleotide Transformer tested and rejected (see above / `Explanation.md` Part 6)
2. BM25 with other models — **not started** (only one alternative model tried so far, and it underperformed — see above)
3. Best model + BM25 + MMR + change DNA sequencing — **not started** (no alternative model has beaten DNABERT-S yet)
4. Extend beyond fish to other species — **not started**
5. Unknown DNA → match to most similar, with species names, tiered by confidence — **done**, see Part 5 of `Explanation.md`
