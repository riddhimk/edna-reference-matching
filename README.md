# eDNA-TaxAI — Environmental DNA Reference Matching

Identifies which species a short environmental DNA (eDNA) read came from
by matching it against reference sequence databases, using a hybrid
lexical + neural retrieval pipeline with an independent nucleotide
alignment check. Validated on real water-sample sequencing data across
three kingdoms: fish, bacteria/archaea, and eukaryotic plankton.

## Overview

Given a short (~120bp) DNA read from a water sample, the pipeline finds
the closest match in a reference database and reports a confidence-tiered
decision — a confident species call, a lower-confidence best guess, or
"no match" — rather than forcing an answer the evidence doesn't support. 

```
Read
  │
  ▼
BM25 lexical retrieval (k-mer index, top-30 pool)
  │
  ▼
MMR diversification (DNABERT-S embeddings, selects top-10)
  │
  ▼
Independent nucleotide alignment (Smith-Waterman, real % identity)
  │
  ▼
Tiered confidence decision
  ├─ ≥95% identity  → assigned
  ├─ 75–95%         → best guess
  ├─ <75%            → best guess (low confidence)
  └─ no candidate    → unclassified
```

This architecture was arrived at empirically (see [Findings](#findings)
below) — BM25 alone finds far more correct candidates than the embedding
model alone, and a real nucleotide alignment step catches cases where two
sequences merely *look* similar in embedding space but aren't.

## Findings

All results below are measured, not assumed, on held-out data or real
sequencing reads; full methodology for each is in
[`Explanation.md`](Explanation.md) and the technical reports in the
project root.

### 1. Retrieval architecture: BM25 → MMR beats the embedding model alone

| Architecture | Recall@10 | Assigned (≥95% identity) | Accuracy on assigned |
|---|---|---|---|
| DNABERT-S embeddings only | 58.2% | 59.8% | 94.6% |
| **BM25 → MMR → nucleotide reranker (production)** | **94.6%** | **94.8%** | **98.7%** |

### 2. DNABERT-S vs. 3 open-source alternatives

Tested whether swapping the embedding model improves retrieval. Full
report: [`DNABERT-S vs Open-Source Model Comparison.docx`](<DNABERT-S vs Open-Source Model Comparison.docx>).

| Model | Recall@10 |
|---|---|
| **DNABERT-S (in production)** | **58.2%** |
| Nucleotide Transformer v2 (50M) | 22.3% |
| GENA-LM (bert-base-t2t) | 12.9% |
| HyenaDNA-tiny | 1.9% |

DNABERT-S wins clearly — it's purpose-trained for species
differentiation, which the alternatives are not.

### 3. Validated across three kingdoms, on real water-sample data

The architecture was built and tuned on fish data, then run unmodified
on real seawater sequencing runs for two other kingdoms. Full log:
[`Real eDNA Read Runs Log.docx`](<Real eDNA Read Runs Log.docx>).

| Kingdom | Real dataset | Reads assigned (≥95% identity) | Plausibility check |
|---|---|---|---|
| Fish | ERR16922870 (Indian Ocean eDNA) | 9.5% | ✅ Real tuna species, NCBI-verified |
| Bacteria/Archaea | ERR6090675 (Tara Oceans) | 66.9% | ✅ Named marine bacteria genera (Roseivivax, Rhodobacteraceae) |
| Eukaryota (plankton) | ERR10928327 (Tara Pacific) | 4.8% | ⚠️ SILVA labels too generic to verify |

Coverage differences track reference database size and richness, not
matching quality — the same unmodified pipeline handles all three.

### 4. Fine-tuning DNABERT-S per kingdom

Fine-tuned separately on real data from each kingdom (1,000–5,000 real
samples each) using a supervised contrastive objective. Technical
report: [`Bacteria Fine-Tuning Fix Report.docx`](<Bacteria Fine-Tuning Fix Report.docx>).

All three kingdoms have now been trained with the corrected trainer
(`finetune_dnabert_fragment.py`), which fixes the fragment/full-length
input mismatch, the coarse label granularity, and the train/test
contamination described in Part 12.

| Kingdom | Recall@10, frozen → fine-tuned | Held-out fragments |
|---|---|---|
| **Bacteria/Archaea** | 17.1% → **79.2%** (+62.1pp) | 1,000 |
| **Fish** | 58.2% → **93.4%** (+35.2pp) | 1,000 |
| Eukaryota | 62.9% → **69.7%** (+6.7pp) | 89¹ |

¹ Eukaryota's reference pool is only 853 records, so its held-out split
yields 89 test fragments — one fragment is worth 1.1pp. Treat its delta
as indicative, not precise.

Re-measuring fish and eukaryota with the corrected trainer **raised**
both numbers rather than lowering them, which is the opposite of what
the earlier train/test contamination alone would predict: fish went
+20.4pp → +35.2pp, and eukaryota reproduced its +6.7pp exactly. The
fragment-vs-full-amplicon training and instance-level labels are worth
more than the contamination was inflating.

### 5. …but the fine-tuning gain does not survive end-to-end

The numbers above measure the embedding model *in isolation*. Inside the
production pipeline, BM25 does the primary retrieval and embeddings only
feed MMR's diversity signal — so a better embedding model cannot add
candidates BM25 never found. Measured on each kingdom's held-out split,
with every arm passing through the same frozen nucleotide reranker:

**Bacteria/Archaea (n=1,000)**

| Architecture | Recall@1 | Recall@10 | Accuracy on assigned | Correct calls |
|---|---|---|---|---|
| BM25 alone | 78.8% | 83.5% | 90.0% | 782 |
| **BM25 → MMR (fine-tuned) — production** | **78.8%** | **83.7%** | 90.1% | 784 |
| BM25 → MMR (frozen embeddings) | 78.8% | 83.6% | 89.9% | 782 |
| DNABERT alone (fine-tuned) | 69.8% | 79.2% | 89.6% | 752 |
| BM25 + RRF (fine-tuned) | 75.2% | 83.3% | **91.2%** | **791** |

**Fish (n=1,000)**

| Architecture | Recall@1 | Recall@10 | Accuracy on assigned | Correct calls |
|---|---|---|---|---|
| BM25 alone | 93.7% | 94.6% | 98.7% | 936 |
| **BM25 → MMR (fine-tuned) — production** | **93.7%** | **94.6%** | 98.7% | 936 |
| BM25 → MMR (frozen embeddings) | 93.7% | 94.6% | 98.7% | 936 |
| DNABERT alone (fine-tuned) | 90.1% | 93.4% | 98.6% | 925 |
| BM25 + RRF (fine-tuned) | 92.1% | 94.6% | 98.7% | 936 |

**Eukaryota (n=89)**

| Architecture | Recall@1 | Recall@10 | Accuracy on assigned | Correct calls |
|---|---|---|---|---|
| BM25 alone | 70.8% | 70.8% | 100% | 63 |
| **BM25 → MMR (fine-tuned) — production** | **70.8%** | **70.8%** | 100% | 63 |
| BM25 → MMR (frozen embeddings) | 70.8% | 70.8% | 100% | 63 |
| DNABERT alone (fine-tuned) | 66.3% | 69.7% | 100% | 62 |
| BM25 + RRF (fine-tuned) | 68.5% | 70.8% | 100% | 63 |

The conclusion is consistent across all three kingdoms, and strongest
where the data is best. For **fish and eukaryota the production arm is
byte-identical frozen vs. fine-tuned** — same Recall@1, Recall@10,
assigned count and correct calls — despite fine-tuning improving the
embedding model by +35.2pp and +6.7pp in isolation. For bacteria it moves
**+2 correct calls per 1,000** (782 → 784) off a +62.1pp isolated gain.

BM25 alone accounts for essentially all end-to-end performance. Fine-tuning's
one measurable contribution is candidate-set diversity — exactly what MMR
exists to improve — and that holds in every kingdom:

| Kingdom | Candidate redundancy, frozen → fine-tuned |
|---|---|
| Bacteria/Archaea | 0.877 → 0.705 |
| Fish | 0.822 → 0.767 |
| Eukaryota | 0.720 → 0.556 |

**MMR vs. RRF:** on bacteria, RRF makes 7 more correct calls and is 1.1pp
more accurate but loses 3.6pp of Recall@1. On fish it ties MMR on correct
calls (936) while still losing 1.6pp of Recall@1, and on eukaryota it ties
outright. RRF also requires running the embedding model at query time,
whereas BM25 → MMR needs embeddings only on the precomputable reference
side. No kingdom shows RRF winning on both accuracy and Recall@1, so
**BM25 → MMR stays in production.**

A note on RRF worth recording: its Recall@1 weakness is an artifact of
embedding quality, not a property of RRF. With frozen embeddings it scores
12.2% (bacteria) and 51.4% (fish) Recall@1; with fine-tuned embeddings
those become 75.2% and 92.1%. The original case against RRF was correct
for the model it was measured on.

## Quick start

```bash
pip install -r requirements.txt
```

Match a single read against a reference database:

```python
from src.final_pipeline import PipelineIndex, assign

index = PipelineIndex.build(
    "data/MitoFish results/mitofish_16s_reference_table.csv",
    "data/MitoFish results/mitofish_16s_ref_embeddings.npy",
    "data/MitoFish results/mitofish_16s_ref_ids.csv",
)
result = assign(read_seq, index)
# result["Confidence_Decision"] -> "assigned" / "best_guess" / "best_guess_low_confidence" / "unclassified"
```

Run the test suite (offline, no model download required):

```bash
pytest
```

## API: upload, history, analytics, chatbot

One process serves the saved results, analysis of new uploads, and the
dashboard:

```bash
uvicorn src.api:app --host 127.0.0.1 --port 8000   # one process: the job queue is in-process
# web app: http://127.0.0.1:8000/  (opens /app/: upload & runs, project results, Assistant on every page)
# API docs: http://127.0.0.1:8000/docs
```

Only the chatbot needs a key. It uses Gemini when one is available —
`GEMINI_API_KEY`, or an `api key: ...` line in `api-key.txt` in the project
root (gitignored) — and Claude (`ANTHROPIC_API_KEY`) otherwise; force one with
`EDNA_CHAT_PROVIDER=gemini|anthropic`. Gemini defaults to `gemini-3.8-flash`
(`EDNA_GEMINI_MODEL`) and falls back to `gemini-3.5-flash`, then
`gemini-3.1-flash-lite` (`EDNA_GEMINI_FALLBACKS`) when a model is overloaded,
out of quota, or retired for the key.

| Capability | Endpoints |
|---|---|
| Upload a read file | `POST /uploads` takes a FASTQ/FASTA file (gzip OK), returns `202` and runs it in the background; the reference set is detected from the reads (pass `marker` only to override). Poll `GET /runs/{id}` for progress. `GET /markers` lists the reference sets. |
| Results for one upload | `GET /runs/{id}` (all findings + chart data), `GET /runs/{id}/reads` (filter by decision, identity, taxon), `GET /runs/{id}/download` (CSV) |
| Analytics | `GET /analytics/overview` (run history + benchmark findings), `GET /analytics/compare?run_ids=a,b` (runs side by side, taxon overlap). Charts come back as data: `{labels, series:[{name, values}]}`. |
| Upload history / no redundant uploads | `GET /runs`, `DELETE /runs/{id}`. `POST /uploads/check` with the file's SHA-256 answers "already analysed?" before uploading. `POST /uploads` itself refuses to re-run a file already analysed against the same marker, even if renamed or gzipped, and returns the existing run instead. |
| Project chatbot | `GET /chat/status`, `POST /chat/sessions` (optionally `{"run_id": ...}`), then `POST /chat/sessions/{id}/messages`. It answers from this project's documentation and from tools that read runs and benchmark results; it declines unrelated questions. |

The reference set is chosen automatically from the reads: up to 400 distinct
reads are compared with every set, scoring the share of their 12-mers found in
it (best orientation, primer-trimmed where the primers are present). Measured
on the validated real-read files, the matching set scores 0.34–0.83 and every
other pairing at most 0.025; a set is chosen when it scores at least 0.10 and
4× the runner-up. Reads that fit no set are still analysed against the closest
one, but the run is flagged "no matching set" with the evidence shown. Reads
whose primers were already removed are detected as such and used as uploaded.

Marker presets pair a reference set with the read preparation validated on
real data: `fish_16s_mitofish` (Part 4), `bacteria_16s_v4v5` (Part 10,
515F-Y/926R trim), `eukaryota_18s_v9` (Part 9, 1389F/1510R trim). Uploads
run the production pipeline (`final_pipeline.py`, unmodified), with
identical reads classified once and the work spread over a process pool. On
this machine a 5,000-read fish file takes ~40 s and a 5,000-read 16S file ~4
min; run time scales with the number of *distinct* sequences, not reads.
There is no size or read-count limit: a file is refused only if it is empty or
not FASTQ/FASTA (multi-line FASTQ, gzip, lower-case/RNA bases and IUPAC codes
are all accepted). Uploads stream to disk. `EDNA_WORKERS` sets the worker
count (default: cores − 2). Uploads, results and chat history are kept under
`data/app/` (`EDNA_APP_DIR`).

## Cloud backup of the run history

Run history (`data/app/`: each run's per-read results, one copy of each
uploaded file, and a snapshot of the history database) is copied to the
project's Google Drive folder with [rclone](https://rclone.org). Nothing else
in the project is uploaded.

One-time setup (opens a browser to sign in to Google; use an account with edit
access to the folder):

```bash
~/.local/bin/rclone config create edna-drive drive scope=drive \
    root_folder_id=1eT3ivjyrwg0JSL4eaokOWSGurkmOdLYt
```

After that, a backup runs automatically in the background whenever a run
finishes; the Runs page shows when the last one succeeded and has a
"Back up now" button. By hand:

```bash
python -m src.cloud_backup status   # connected? last backup?
python -m src.cloud_backup push     # back up now
python -m src.cloud_backup pull     # restore into data/app (stop the API first)
```

Backups only ever add files (`rclone copy`, not `sync`), so deleting a run
locally -- or losing `data/app/` -- never deletes it from Drive. Settings:
`EDNA_RCLONE_REMOTE` (default `edna-drive:`), `EDNA_RCLONE_BIN`,
`EDNA_CLOUD_BACKUP=off` to disable automatic backups.

## Reproducing the headline results

```bash
# Model comparison (DNABERT-S vs 3 alternatives), ~20 min CPU
python -m src.nt_diagnostics
python -m src.multi_model_diagnostics

# Architecture validation across kingdoms, ~2-3 min CPU
python -m src.other_kingdom_diagnostics

# Real-read runs
python -m src.run_tara_18s_v9_experiment      # Eukaryota, Tara Pacific
python -m src.run_tara_16s_v4v5_experiment    # Bacteria/Archaea, Tara Oceans

# Fine-tuning (per kingdom) on 16 CPU cores: bacteria ~75 min,
# fish ~93 min, eukaryota ~23 min (scales with each kingdom's train split)
python -m src.finetune_dnabert_fragment bacteria models/dnaberts_finetuned_bacteria_v2 2
python -m src.evaluate_finetuned bacteria models/dnaberts_finetuned_bacteria_v2

# BM25 / MMR / RRF end-to-end, frozen vs fine-tuned embeddings, ~12 min CPU
# (kingdom defaults to bacteria; pass fish or eukaryota for the others)
python -m src.finetuned_architecture_evaluation
python -m src.finetuned_architecture_evaluation fish
python -m src.finetuned_architecture_evaluation eukaryota
```

### Project structure

```
Dockerfile                    Production container build (Hugging Face Spaces / Docker)
AGENTS.md                     Workspace and agent workflow guidelines
requirements.txt              Python dependencies (CPU-optimized PyTorch)
run_pipeline.py               CLI pipeline entry point
Explanation.md                Full chronological technical narrative (Parts 1–13)

src/
  final_pipeline.py           Production pipeline (BM25 → MMR → reranker → decision)
  bm25_retrieval.py           BM25 lexical retrieval
  mmr.py                      MMR candidate diversification
  nucleotide_rerank.py        Smith-Waterman alignment reranker
  species_lookup.py           NCBI accession → species name, cached

  api.py                      FastAPI JSON API; mounts static web app at /app/
  app_routes.py               Uploads, run history, results, analytics
  run_analysis.py             Marker presets; runs uploads through final_pipeline
  upload_io.py                FASTQ/FASTA (+gzip) parsing and validation
  run_store.py                SQLite persistence of runs and chat sessions
  cloud_backup.py             rclone backup of run history
  chat.py                     Project AI assistant (Gemini/Claude + tool execution)

web/                          Web dashboard (React + vanilla JS, served at /app/)
  index.html                  Layout, design tokens, responsive styling
  app.jsx                     Shell, routing, benchmark & evaluation pages
  analysis.jsx                Upload studio, run progress, results compare, Assistant drawer
  fx.js                       Zero-latency pointer & lightweight UI motion layer
  ui.jsx                      UI helper components (CountUp, badges)

tests/                        147 offline unit tests (no GPU, network, or weights required)
data/                         Reference tables, embeddings, SQLite database (gitignored)
models/                       Model checkpoints (gitignored)
```

## Running the Web App & Deployment

### Local Development
```bash
# Start API and Web Dashboard
uvicorn src.api:app --host 127.0.0.1 --port 8000
# Then open http://127.0.0.1:8000/app/ in your browser
```

### Docker / Hugging Face Spaces
```bash
# Build container image
docker build -t edna-taxai .

# Run on port 7860
docker run -p 7860:7860 edna-taxai
```

## Documentation

- **[`Explanation.md`](Explanation.md)** — Comprehensive technical narrative covering
  Parts 1–13: benchmark results, model evaluations, ablation studies, and architectural decisions.
- **[`AGENTS.md`](AGENTS.md)** — Workspace agent guidelines for running background jobs and evaluations.

## Roadmap

| # | Item | Status |
|---|---|---|
| 1 | Open-source model comparison | ✅ Done — DNABERT-S confirmed best |
| 2 | BM25 with alternative models | ⏳ Blocked — no alternative model has beaten DNABERT-S |
| 3 | Best model + BM25 + MMR + region change | ⏳ Blocked on #2 |
| 4 | Extend beyond fish | ✅ Done — validated + run on real data, 3 kingdoms |
| 5 | Unknown-DNA best-guess matching | ✅ Done — tiered confidence decision |
| 6 | Fine-tune DNABERT-S per kingdom | ✅ Done — all three corrected: bacteria +62.1pp, fish +35.2pp, eukaryota +6.7pp (isolated) |
| 7 | BM25+MMR vs. BM25+RRF on the fine-tuned model | ✅ Done — all three kingdoms; MMR wins on cost, gain doesn't survive end-to-end anywhere |
| 8 | Re-run items 6–7 for fish and eukaryota | ✅ Done — corrected numbers went up, not down |

## Tests

147 offline unit tests (no model download, no GPU, no network, no API key required) verify parsing, retrieval logic, decision thresholds, pipeline composition, the upload → run → results flow on synthetic reference sets, duplicate detection, and the assistant tool loop.

```bash
pytest -q
```
