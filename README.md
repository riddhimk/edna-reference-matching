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
| `src/nt_embed.py` | Nucleotide Transformer v2 (50M, multi-species) embedding — isolated model |
| `src/nt_diagnostics.py` | DNABERT vs Nucleotide Transformer Recall@K comparison (`data/NT diagnostics/`) |
| `src/hyenadna_embed.py` | HyenaDNA-tiny embedding — isolated model |
| `src/genalm_embed.py` | GENA-LM (bert-base-t2t) embedding — isolated model |
| `src/multi_model_diagnostics.py` | DNABERT vs HyenaDNA-tiny vs GENA-LM Recall@K comparison (`data/HyenaDNA diagnostics/`, `data/GENA-LM diagnostics/`) |
| `src/other_kingdom_diagnostics.py` | Production architecture (BM25→MMR→reranker) validated on Bacteria/Archaea (16S V4) and Eukaryota (18S V9) held-out data (`data/Other Kingdoms diagnostics/`) |
| `src/extract_amplicon_region_tara.py` | Re-extracts SILVA 18S V9 with Tara Pacific's real primers (1389F/1510R) |
| `src/run_tara_18s_v9_experiment.py` | Real-read run: Tara Pacific 18S V9 (Eukaryota) vs re-extracted SILVA |
| `src/extract_amplicon_region_tara_16s.py` | Re-extracts SILVA 16S V4-V5 with Tara Oceans' real primers (515F-Y/926R) |
| `src/run_tara_16s_v4v5_experiment.py` | Real-read run: Tara Oceans 16S V4-V5 (Bacteria/Archaea) vs re-extracted SILVA |

## Model comparison: DNABERT-S vs 3 open-source alternatives

Tested whether an open-source model swap improves retrieval (professor's
"open-source LLM models for retrieval" / "comparison of models" / "which
classifies DNA better" next steps). Full writeup and technical-requirement
analysis: **`DNABERT-S vs Open-Source Model Comparison.docx`** (project
root). Short version — all four tested on the identical 1,000 held-out
fragments:

| Metric | DNABERT-S (existing) | Nucleotide Transformer v2 50M | HyenaDNA-tiny | GENA-LM |
|---|---|---|---|---|
| Recall@1 | **41.7%** | 15.3% | 0.1% | 2.3% |
| Recall@5 | **53.3%** | 20.1% | 0.7% | 7.7% |
| Recall@10 | **58.2%** | 22.3% | 1.9% | 12.9% |

**Result: DNABERT-S wins clearly against all three alternatives tested.**
It's purpose-trained for species differentiation, which is exactly this
task; the others are general-purpose (Nucleotide Transformer, GENA-LM,
trained on human/multi-species genomes broadly, not to tell species
apart) or too small for this task (HyenaDNA-tiny, 2 layers). **DNABERT-S
remains the model of choice for this project.** Not adopted —
`final_pipeline.py` is unchanged. Reproduce with `python -m
src.nt_diagnostics` and `python -m src.multi_model_diagnostics` (~20 min
combined on CPU, no GPU required).

## Architecture validation beyond fish: Bacteria/Archaea and Eukaryota

Tested whether the production architecture (BM25 → MMR → nucleotide
reranker → tiered decision) generalizes beyond fish (professor's "other
species than fish" next step), reusing the SILVA-derived Bacteria/Archaea
(16S V4) and Eukaryota (18S V9) reference sets Part 2 already built — no
new datasets needed. **Scope note: held-out synthetic-fragment
validation, not real-read classification** — this project has no real
bacterial/archaeal/eukaryotic eDNA reads (only the fish-marker
`ERR16922870` reads exist, and Part 2 already established those aren't
16S/18S at all).

| Metric (n held-out fragments) | Fish / MitoFish (Part 4/5) | Bacteria/Archaea 16S V4 | Eukaryota 18S V9 |
|---|---|---|---|
| Recall@10 | 94.6% | 82.0% | 70.5% |
| % assigned | 94.8% | 81.4% | 69.5% |
| Accuracy on assigned | 98.7% | 97.7% | 99.7% |

**Result: the architecture generalizes well.** Accuracy on assigned reads
stays ≥97.7% across all three kingdoms — the same BM25→MMR→reranker
design works without kingdom-specific tuning. Reproduce with `python -m
src.other_kingdom_diagnostics` (~2-3 min on CPU).

## Real non-fish read data

Full run-by-run log (setup, complications found and fixed, results, and
a cross-kingdom comparison table) is in **`Real eDNA Read Runs Log.docx`**
(project root). Short version below.

### Tara Pacific 18S V9 (Eukaryota)

Found and ran a **real** (not synthetic) seawater eDNA dataset — ENA run
`ERR10928327`, Tara Pacific's 18S V9 plankton metabarcoding
(`PRJEB60216`) — through the unmodified production architecture. Full
detail in `Explanation.md` Part 9; short version:

- Re-extracted SILVA's 18S region using this dataset's **actual** primers
  (`1389F`/`1510R`, confirmed via literature) rather than reusing Part 2's
  close-but-different `1391F`/`EukBr` extraction — same "don't assume,
  verify" discipline as every prior region-matching decision in this
  project. Result: 853 real-primer-verified references (smaller pool than
  Part 2's 5,000, but both-primer-confirmed rather than approximated).
- Real reads needed new handling: they still carry their PCR primer
  (unlike the fish reads), and about half are primed from the *opposite*
  end of the short amplicon — both detected and corrected before
  matching (primer trim + reverse-complement where needed).
- **Result on 4,754 usable reads:** 4.8% assigned (≥95% identity), 81.7%
  best_guess, 13.6% low-confidence. The 226 assigned reads span only 10
  references (top 2 = 82.7% of assignments) — possibly real dominance by
  a common lineage, possibly a small-reference-pool artifact; SILVA's own
  labels here are just "uncultured eukaryote," so unlike the fish/tuna
  case there's no species name to independently verify against.

Reproduce with `python -m src.run_tara_18s_v9_experiment` (~2 min on
CPU, after building the reference set per `Explanation.md` Part 9).

**Environment note:** getting DNABERT-S/`multimolecule` to load in this
Linux `.venv` required pinning `transformers==5.15.1` (matching the
working `.venv-win` combination) — the previously installed 4.56.2 (used
throughout Parts 6-7's *other* model comparisons, which never touch
`embed_common.py`) is incompatible with `multimolecule`.

### Tara Oceans 16S V4-V5 (Bacteria/Archaea)

Same treatment as above, for the other still-open kingdom — real reads
from ENA run `ERR6090675` (Tara Oceans prokaryotes, `PRJEB36282`)
against a freshly re-extracted SILVA 16S V4-V5 set matching this
dataset's actual primers (`515F-Y`/`926R`, not Part 2's V4-only
`515F`/`806R`). Full detail in `Explanation.md` Part 10.

- 1,831,825 Bacteria/Archaea SILVA records matched both primers
  completely (89.3% of those scanned) — subsampled to 5,000 references,
  same size as Part 2's other SILVA samples.
- Real reads showed the same dual-orientation mixing found in the
  Eukaryota case (Part 9) — confirmed here too, not a V9-only quirk —
  handled the same way (detect, trim, reverse-complement as needed).
- **Result on 4,767 usable reads: 66.9% assigned** (≥95% identity),
  32.9% best_guess, 0.2% low-confidence — far higher coverage than
  Eukaryota's 4.8%, because this reference pool (5,000, drawn from 1.83M
  matches) is much larger than the 18S V9 pool (853, the entire usable
  set).
- Unlike the Eukaryota case, a plausibility check **is** possible here:
  named genera appear among the top hits (*Roseivivax salinus*,
  *Rhodobacteraceae*, *Flavobacteriia*) — all well-known, genuinely
  abundant real marine bacterioplankton groups, a believable signal
  (though not independently NCBI-verified the way the fish/tuna finding
  was).

Reproduce with `python -m src.run_tara_16s_v4v5_experiment` (~a few
minutes on CPU). All outputs live under `data/SILVA results/` —
`data/MitoFish results/` (fish) remains untouched and separate.

## Roadmap (professor's next steps, tracked in `Explanation.md` Part 5/6/7/8)

1. Open-source LLM models for retrieval — comparison of models — which classifies DNA better — **done**, 3 alternatives tested and rejected, DNABERT-S confirmed best (see above / `Explanation.md` Parts 6-7)
2. BM25 with other models — **not started** (only one alternative model tried so far, and it underperformed — see above)
3. Best model + BM25 + MMR + change DNA sequencing — **not started** (no alternative model has beaten DNABERT-S yet)
4. Extend beyond fish to other species — **done**, architecture validated on Bacteria/Archaea and Eukaryota (synthetic, Part 8), and run on real Eukaryota (18S V9, Part 9) AND real Bacteria/Archaea (16S V4-V5, Part 10) seawater datasets
5. Unknown DNA → match to most similar, with species names, tiered by confidence — **done**, see Part 5 of `Explanation.md`
