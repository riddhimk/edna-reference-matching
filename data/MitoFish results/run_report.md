# MitoFish 16S run report

## Verification performed before running anything

1. **Marker confirmed from ENA project metadata** (not read length): run
   `ERR16922870` alias = `16SFish_OcOm_2305_100_1_run`; study describes
   marker `16SFish/D`.
2. **Primers verified from a peer-reviewed source**, cross-checked against
   two independent secondary sources before use:
   - `16SF/D` forward: `GACCCTATGGAGCTTTAGAC`
   - `16S2R` reverse: `CGCTGTTATCCCTADRGTAACT`
   - Source: Berry et al. 2017, as tabulated in Zhang et al. 2020 (Table 1,
     PMC9205523). Reported amplicon ~219bp, target = mitochondrial 16S
     rRNA (fish).
3. **MitoFish FASTA structure inspected** (`data/mitofishdb.fa.gz`):
   mixed-content file -- COI-5P barcodes, 12S/16S/CYTB-tagged fragments,
   untagged accessions, and complete mitogenomes up to ~23.4kb in a
   50,001-record length sample (median 652bp). No consistent gene label in
   headers for most records, and **no companion species/taxonomy mapping
   file** was present. Conclusion: primer-based in-silico PCR is the only
   reliable extraction method here; header-based filtering was not used.
4. **In-silico PCR validated on a 20,000-record sample** before running
   the full file: 530 had the forward primer, 513/530 (96.8%) also had
   the reverse primer intact (interior length 193-227bp, mean 205bp) --
   `both_primers` mode used directly (no truncation workaround needed,
   unlike SILVA's 18S V9 case).
5. **Read compatibility checked**: amplicon (~219bp) > read length
   (118-140bp), consistent with single-end sequencing of a longer
   amplicon. 0/5000 real reads contain the raw forward primer sequence,
   consistent with standard primer-trimmed FASTQ submission (not a red
   flag).
6. **Alphabet check**: mitofishdb.fa.gz sequences confirmed DNA (`T`), not
   RNA (`U`) -- `to_dna()` conversion is a no-op here, same as reads.

## What was run

```
python -m src.extract_mitofish_16s data/mitofishdb.fa.gz "data/MitoFish results/mitofish_16s_amplicons.csv"
python -m src.subsample_reference_table "data/MitoFish results/mitofish_16s_amplicons.csv" "data/MitoFish results/mitofish_16s_reference_table.csv" 16592 42
REFERENCE_TABLE_CSV="data/MitoFish results/mitofish_16s_reference_table.csv" REFERENCE_EMBEDDINGS_NPY="data/MitoFish results/mitofish_16s_ref_embeddings.npy" REFERENCE_IDS_CSV="data/MitoFish results/mitofish_16s_ref_ids.csv" python run_pipeline.py embed-ref
REFERENCE_TABLE_CSV="data/MitoFish results/mitofish_16s_reference_table.csv" REFERENCE_EMBEDDINGS_NPY="data/MitoFish results/mitofish_16s_ref_embeddings.npy" REFERENCE_IDS_CSV="data/MitoFish results/mitofish_16s_ref_ids.csv" python -m src.mitofish_calibration
python -m src.run_region_experiment mitofish_16s "data/MitoFish results/mitofish_16s_reference_table.csv" "data/MitoFish results/mitofish_16s_ref_embeddings.npy" "data/MitoFish results/mitofish_16s_ref_ids.csv" 0.80 data/reads_ids.csv data/reads_embeddings.npy "data/MitoFish results"
```

## Extraction yield

- 624,421 total MitoFish records scanned
- 17,227 had the forward primer (2.76%)
- 16,778 had both primers intact (2.69% of total; 97.4% of forward hits)
- 186 of those excluded for ambiguous IUPAC bases (same tokenizer
  limitation documented for SILVA) -> **16,592 usable references**
- Amplicon length: min 78bp, max 249bp, mean 204.4bp (interior region,
  primers trimmed off)

## Calibration

First attempt (exact `Reference_ID` recovery) failed with 0% accuracy at
every threshold -- diagnosed as the wrong ground truth, not a real
failure: MitoFish contains many different accessions sharing an
identical 16S sequence (207/300 held-out full sequences recovered an
exact sequence-duplicate, at similarity 1.0, under a *different*
accession). Recalibrated on sequence-identity recovery instead:

| threshold | % assigned | accuracy |
|---|---|---|
| 0.10-0.55 | ~100% | 9.2% |
| 0.73 | 62.1% | 11.6% |
| 0.79 | 26.3% | 16.1% |
| 0.82 | 10.8% | 18.3% |
| 0.88 | 0.6% | 35.0% |

Real, monotonic signal (unlike SILVA full-length's flat ~9% chance
curve), though the accuracy ceiling is more modest than the SILVA V4/V9
region experiments.

**Threshold used for the real-read run: 0.80** (between the 0.79/0.82
rows above; conservative operating point).

## Real-read results

- Real-read top1 similarity: min 0.482, p25 0.612, median 0.651, p75
  0.687, p95 0.779, max 0.822, mean 0.661
- **211/5,000 reads (4.2%) assigned** at threshold 0.80; 4,789
  unclassified

## Hubness check

- Across all 5,000 reads, only 347 distinct references (2.1% of the
  16,592-reference pool) were ever selected as a nearest neighbor; top 10
  references absorbed 70.5% of all matches.
- Among the 211 *assigned* reads specifically: only 13 distinct
  references involved, and just 2 references (`AY958655`, `PX804206`)
  account for 188/211 (89%) of all assignments.
- **Conclusion: the same hubness artifact found in the SILVA experiment
  is present here, more concentrated.** The 211 "assigned" reads should
  NOT be read as 211 correct species identifications.

## Bottom line (superseded by the rerank experiment below)

MitoFish is the biologically correct database family for this dataset's
actual marker (Section 8 of Explanation.md Part 2) and the calibration
shows real, non-chance signal -- a genuine improvement over SILVA at the
gene-family level. But the real-read classification result was
dominated by a hubness artifact and was **not reliable enough to present
as validated species identifications** in this form.

**Update**: the hubness problem was directly addressed with a DNABERT
top-K + independent nucleotide-alignment rerank -- see
`Explanation.md` Part 4 and the summary below. Short version: reranking
plus a properly calibrated identity threshold (95%) produces 474/5,000
assigned reads that are, on independent NCBI lookup, 93.9% Scombridae
(tuna family) -- highly plausible for an Indian Ocean eDNA sample --
though a real, smaller hub concentration remains among the assigned
reads and is now attributed to likely genuine local tuna abundance
rather than an embedding artifact. See Part 4, Section 31 for the full
verdict and its limits.
