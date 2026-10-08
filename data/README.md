# data/ folder

Nothing in here is tracked by git (`data/` is gitignored). It holds the
raw inputs you supply plus every artifact the experiments write.

## Inputs you place here yourself

| File | Used by |
|---|---|
| `SILVA results/SILVA_138.2_SSURef_tax_silva_trunc.fasta` | SILVA reference extraction (Parts 1-2, 9-10) |
| `mitofishdb.fa.gz` | MitoFish reference extraction (Part 3) |
| `ERR16922870_1.fastq` | Real fish eDNA reads, Indian Ocean (Parts 1-5) |

The two Tara FASTQ files (`tara_16s_v4v5_first5000.fastq`,
`tara_18s_v9_first5000.fastq`) are downloaded by the real-read experiment
scripts themselves; you do not need to fetch them manually.

## Written by the pipeline, organised per experiment

| Directory | What it holds |
|---|---|
| `SILVA results/` | All SILVA work: region extractions (16S V4, 18S V9, and the Tara-primer 16S V4+V5 / 18S V9 re-extractions), reference tables, embeddings, FAISS indices, per-read results, and the bacteria/eukaryota fine-tuning logs |
| `MitoFish results/` | The 16,592-record MitoFish 16S reference table, embeddings, index, rerank artifacts, NCBI species lookups, and the fish fine-tuning log |
| `Final diagnostics/` | Production A/B/C/D architecture comparison on fish (`FINAL_REPORT.md` plus per-fragment CSVs) — the source of README Finding #1 |
| `Other Kingdoms diagnostics/` | Cross-kingdom synthetic validation, one subfolder per kingdom (Part 8) |
| `NT diagnostics/`, `HyenaDNA diagnostics/`, `GENA-LM diagnostics/` | Alternative embedding-model comparisons (Parts 6-7) |
| `Finetuned architecture evaluation/` | End-to-end BM25 / MMR / RRF results, frozen vs fine-tuned, with one subfolder per non-bacteria kingdom (Part 13) |
| `Stage0-3 diagnostics/` | Early retrieval/rerank sweeps that established the production defaults |
| `logs/` | Console logs from the corrected per-kingdom fine-tuning runs |

Loose files in this folder are shared caches and top-level run logs
(`species_name_cache.csv`, `reads_embeddings.npy`, `reads_ids.csv`, the
`finetune*_evaluation*.log` files).

See the main [`README.md`](../README.md) for the commands that produce
each of these, and [`Explanation.md`](../Explanation.md) for the full
chronological account of what each experiment measured.
