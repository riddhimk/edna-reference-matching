# What This Run Actually Did (Plain-Language Explanation)

This document explains, in simple terms, what happened when the SILVA
reference-matching pipeline was run on your real files, what the numbers in
the `data/*.csv` files mean, and what the honest conclusion is.

## 1. The basic idea, in one sentence

We took your 5,000 experimental DNA reads and your 2.2-million-sequence
SILVA reference database, turned both into 768-number "fingerprints"
(embeddings) using the same AI model, and asked: "for each read, which
reference sequence's fingerprint looks most similar?"

## 2. What we started with

- **Experimental reads**: `data/ERR16922870_1.fastq` actually contains
  309,155 reads, but your original POC only used the first 5,000 of them
  (we confirmed this — their length stats, min 118bp / max 140bp / average
  121.6bp, exactly match your reported results). So we used that same
  first-5,000 slice for consistency.
- **Reference database**: `data/SILVA_138.2_SSURef_tax_silva_trunc.fasta`
  has **2,224,690** sequences. We checked this by actually scanning the
  file (Stage 1 "inspect"), not guessing. Every one of them is a
  near-full-length gene sequence — between 900 and 3,718 base pairs long,
  averaging 1,427bp.

**Why that matters**: your reads are ~120bp fragments, but the references
are ~1,400bp on average. That's roughly a 10x difference in length. This
mismatch turns out to be the central finding of this run — see Section 5.

## 3. What we had to fix before anything would run

Three real problems came up, each confirmed by testing, not assumed:

1. The model named in `src/config.py` (`zhihan1996/DNABERT-S`) refused to
   load on this Windows/CPU machine because its code requires a package
   called `triton`, which doesn't have a Windows version. We switched to
   `multimolecule/dnaberts` — the alternative model your own project PDF
   mentions — which works fine on CPU and produces the same size of
   fingerprint (768 numbers).
2. A small fraction of SILVA sequences contain "ambiguous base" letters
   (like `N`, meaning "could be any base") instead of just A/C/G/U. The
   tokenizer crashed on these. We filtered them out — about 6% of
   sequences — the same way you'd throw out a smudged barcode you can't
   read.
3. Embedding **all** 2.2 million reference sequences would take an
   estimated 3+ hours on this machine with no GPU. Instead, we randomly
   sampled **5,000** references (keeping the same proportions of bacteria/
   archaea/eukaryotes as the full file) so the whole pipeline could run in
   one session. This means: **this run searched 5,000 of the 2.2 million
   available references, not all of them.**

## 4. The files this run produced, and how to read them

| File | What each row means |
|---|---|
| `data/silva_reference_table_sample.csv` | One reference sequence per row: its ID, its full DNA sequence, and its taxonomy (species classification), e.g. `HA953679.13.1872` is a 1,860bp sequence labeled `Eukaryota;...;Rattus norvegicus (Norway rat)`. |
| `data/reads_embeddings.npy` / `data/silva_ref_embeddings.npy` | The 768-number fingerprints for reads and references (not human-readable — used internally by the search step). |
| `data/query_results.csv` | For each read, the **top 5** most similar references and their similarity scores. Example: read `ERR16922870.1`'s best match is reference row `4441` with similarity `0.515`, and its *second*-best match (`137`) scored `0.499` — almost as close. |
| `data/read_taxonomy_results.csv` | The final answer per read: which reference it's closest to, what taxonomy that reference has, and whether the read was confidently **assigned** or left **unclassified**. |

A real example row from `read_taxonomy_results.csv`:

```
ERR16922870.1, nearest match = JF203696.1.1282 (a Bacteria; Paracoccus sequence),
similarity = 0.515, decision = unclassified
```

This says: "the closest reference we found for this read belongs to the
bacterium genus *Paracoccus*, but the match wasn't strong enough to trust,
so we did not assign it that identity."

## 5. The key finding: why every read came back "unclassified"

Before trusting any similarity score, we needed to know: *how similar do
two sequences need to be before we believe they're really a match?* This
is called calibrating a threshold, and the project rules said not to just
guess a number like 0.8 — it has to be measured from real data.

We tried this two ways:

**Attempt 1 — full-length vs. full-length.** We held back 1,000 full-length
references, and checked whether each one's nearest match (among the
remaining 4,000 full-length references) had the same taxonomy. This worked
great — accuracy hit 100% once similarity was above ~0.90. But this test
compares long sequences to long sequences, and your real reads are short.
It doesn't tell us what threshold to use for short reads.

**Attempt 2 — short fragment vs. full-length (the realistic test).** This
time we cut short, 118-140bp pieces (matching your real read lengths) out
of random positions of held-out references, and checked whether *those*
short pieces could find their way back to a reference with matching
taxonomy. The result: **accuracy stayed around 9%, close to random
guessing, no matter what similarity threshold we tried** (we tested
thresholds from 0.10 all the way to 0.68).

**What this means in plain terms**: cutting a random 120bp window out of a
1,400bp gene and asking "which full gene does this small window belong to"
usually doesn't work with this method — because most of a gene sequence
looks similar across many species (only certain "hypervariable" regions
are actually distinctive), and we don't know *which* small window your
sequencer actually captured. This is exactly the "region mismatch" concern
your project README already raised as a risk — we've now confirmed it
happens in practice with real numbers, not just as a theoretical worry.

Because of this, we set the final similarity threshold conservatively
(0.60 — near the highest similarity score seen in either test), which
means: **0 out of 5,000 reads were confidently assigned a species**. All
5,000 were correctly marked "unclassified" and handed off to the existing
open-set/HDBSCAN clustering stage instead of being forced into a guess.

This is not a bug or a failed run — it is the pipeline doing exactly what
Stage E and Stage F of the project design describe: when the evidence
isn't strong enough, don't force an answer.

## 6. What would need to change to get real assignments

To make reference-matching actually work for these reads, someone would
need to:

1. Figure out which specific region of the SSU rRNA gene your sequencer
   actually targeted (check the sequencing run's primer information).
2. Either trim the SILVA references down to just that same region before
   embedding them, or use a reference database that's already trimmed to
   it (this is what "amplicon-region-specific" reference databases like
   trimmed SILVA/PR2 sets are usually for).
3. Re-run Stage 5 (embed references) and Stage 12 (calibration) after that
   fix, and check whether accuracy improves — it should, because now
   short reads and references would be describing the *same* piece of DNA
   instead of different-length windows of the whole gene.

## 7. Honest summary

- The pipeline code works end-to-end: parsing, embedding, FAISS search,
  confidence decision, and open-set routing all ran successfully on real
  data.
- The reference database was only partially used (5,000 of 2.2 million
  sequences) due to compute time limits on this machine.
- The threshold was calculated from real held-out data, not guessed.
- That calculation showed the current setup cannot reliably match short
  reads to full-length references, which is why every read ended up
  "unclassified" — a correct, evidence-based outcome, not a failure of
  the code.
- Fixing this requires resolving the region-mismatch issue described in
  Section 6 before reference-matching accuracy can be meaningfully
  reported.

---

# Part 2: Following up on the region-mismatch problem

Part 1 above ended by saying: "we don't know which piece of the gene your
reads actually cover, so full-length references can't be reliably
compared to them." This section reports what we found when we actually
investigated that question properly, instead of guessing.

## 8. First: what gene are these reads actually from?

We did **not** assume this from read length (a 120bp read could plausibly
be a fragment of many different things). Instead we looked up the real
sequencing-run metadata for `ERR16922870` on ENA (the European
Nucleotide Archive — the official public database this data was
submitted to).

**What the metadata says, word for word:**
- Run name (`run_alias`): `16SFish_OcOm_2305_100_1_run`
- Study title: *"Indian Ocean Territories eDNA amplicon sequencing data
  (POE_V13_IOT)"*
- Study description: *"Metabarcoding sequencing data (MiFish-U, MiFish-E2,
  MarVer1, 16SFish/D) from filtered sea water samples..."*

So this run specifically used the **`16SFish/D`** marker. Searching the
scientific literature for what that marker actually is: **it targets the
mitochondrial 16S rRNA gene of fish** (a marker designed to detect fish
species from seawater eDNA) — not the bacterial/archaeal 16S rRNA gene,
and not the eukaryotic 18S rRNA gene.

**Why this matters a lot**: SILVA's database (the one this whole project
uses as its reference) only contains bacterial 16S, archaeal 16S, and
eukaryotic **nuclear** 18S sequences. It does not contain fish
**mitochondrial** 16S sequences — these are a biologically different gene
that happens to share the name "16S" by coincidence (both are named after
a ribosome size measurement, not because they're the same gene). This is
similar to how "Java" the programming language and "Java" the island
share a name without being related.

**Bottom line**: the real answer to "is this 16S or 18S?" is **neither,
in the sense SILVA covers**. This was only findable through the actual
sequencing metadata — read length alone would have been actively
misleading here (more on that below).

## 9. What we did anyway, and why

Even though the true marker isn't in SILVA, the project asked us to
still test the two marker families SILVA *does* cover — bacterial/archaeal
16S and eukaryotic 18S — using the actual scientifically-standard regions
for each, in case one of them turns out to be usable in practice. This is
good scientific practice: check the two available options properly with
real data, rather than giving up because the "ideal" database isn't
available.

**The two regions we used are not made up** — they come from the Earth
Microbiome Project's standard protocols, which are also what QIIME2's
official tutorials teach:

| Region | Primers used (exact sequences) | Source |
|---|---|---|
| 16S V4 | Forward `515F`: `GTGYCAGCMGCCGCGGTAA`; Reverse `806R`: `GGACTACNVGGGTWTCTAAT` | Parada et al. 2016 / Apprill et al. 2015 — the pair used in QIIME2's official "Moving Pictures" tutorial |
| 18S V9 | Forward `1391F`: `GTACACACCGCCCGTC`; Reverse `EukBr`: `TGATCCTTCTGCAGGTTCACCTAC` | Amaral-Zettler et al. 2009 / Stoeck et al. 2010 — the Earth Microbiome Project's standard short 18S region |

## 10. How the region was cut out of each SILVA sequence

This is the important part: **we did not just chop off the first N
letters of each sequence.** That would give you the right *length* but
almost certainly the *wrong piece* of the gene — like guessing a word from
a book by reading its first two letters instead of looking up the actual
page.

Instead, we searched each full-length SILVA sequence for the actual
primer sequence (the same short "landmark" a real lab machine would use
to start copying DNA), and cut out only the piece between the two
primers — this is called **in-silico PCR**, and it's the standard
approach QIIME2 itself recommends (`qiime feature-classifier
extract-reads`).

**16S V4** worked exactly as expected: found both primers in 1,829,030
sequences, and the extracted pieces averaged **253 base pairs** — which
matches the textbook V4 amplicon length almost exactly. This is in
`data/silva_16s_v4_amplicons.csv`.

**18S V9** needed one adjustment. We discovered, by testing it on real
data, that this particular SILVA file (the "trunc" version) cuts off
every eukaryotic sequence **5 bases before** the reverse primer site
finishes — every single one we checked. So instead of forcing a
reverse-primer match that will never succeed in this file, we take
"everything from the forward primer to wherever the sequence already
ends" — which *is* the real V9 region this file provides, just capped at
150bp (V9's known typical length) so we don't accidentally grab something
unrelated from an oddly long entry. This produced 140,968 usable pieces
averaging **127 base pairs** — remarkably close to the real reads' 121.6bp
average. This is in `data/silva_18s_v9_amplicons.csv`.

For both regions we then randomly kept 5,000 of the usable pieces (same
approach as Part 1, for the same time-budget reason), in
`data/silva_16s_v4_sample.csv` and `data/silva_18s_v9_sample.csv`.

*(One housekeeping check the project specifically asked about: SILVA
stores its sequences with `U` letters (RNA-style) while your FASTQ reads
use `T` letters (DNA-style). The code already converts every sequence to
the `T` form right before feeding it to the AI model, for both references
and reads alike — so this was already being handled consistently; we
verified it rather than needing to change anything.)*

## 11. Running the same calibration test on the new, region-matched references

Part 1's calibration test cut a random 120bp piece from a random
position of a *1,400bp full-length* reference — which is why it scored
almost no better than a coin flip (~9% accuracy) no matter what
similarity cutoff we tried: a random 120bp window usually isn't the
V4/V9 region at all.

This time, since the reference pieces are *already* just the V4 or V9
region, cutting a random short window from *them* is a much fairer test —
both sides are now looking at the same landmark of the gene.

**Result — real signal appeared this time:**

| Region | Accuracy at loose threshold | Accuracy at strict threshold | % of test reads kept at strict threshold |
|---|---|---|---|
| 16S V4 | 59.1% | 66.4% (threshold 0.68) | 50.6% |
| 18S V9 | 82.0% | 84.3% (threshold 0.68) | 94.5% |

Unlike Part 1's flat 9%-no-matter-what curve, accuracy here climbs as we
raise the similarity threshold — a real, usable relationship between
"how similar" and "how correct." Both are clearly better than the
full-length approach.

## 12. But: querying the *real* reads exposed a new problem

Encouraged by that, we queried the actual 5,000 real reads against both
new reference sets (files: `data/read_taxonomy_results_16s_v4.csv` and
`data/read_taxonomy_results_18s_v9.csv`). At a reasonable confidence
threshold:

- **16S V4**: 928 of 5,000 reads (18.6%) got assigned a taxonomy.
- **18S V9**: 3,219 of 5,000 reads (64.4%) got assigned a taxonomy.

That sounds like a big improvement over Part 1's 0 assigned — but
checking *which* references these reads matched against reveals a
problem. Out of the 5,000 candidate references in each pool, real reads
"chose" their nearest match from only:

- **176 distinct references** for 16S V4 (3.5% of the pool) — and just
  the top 10 of those absorbed **78%** of all 5,000 reads' matches.
- **279 distinct references** for 18S V9 (5.6% of the pool) — top 10
  absorbed **69%** of all matches.

For example, one single 16S V4 reference (`HQ739000.1.914`, an Archaea
sequence) was the "closest match" for 451 completely different real
reads — reads that (being real seawater eDNA) almost certainly come from
hundreds of different organisms, not one.

**What this means**: a handful of reference sequences are acting like
magnets, pulling in many unrelated reads regardless of whether they're
actually similar. This is a known effect in AI similarity search called
"hubness" — some points in a high-dimensional space just happen to sit
near the middle of everything and get picked as the "nearest neighbor" by
many unrelated queries. It's a mathematical/embedding-model artifact, not
a biological signal. This means the 928 and 3,219 "assigned" counts above
**should not be read as 928 or 3,219 correct species identifications** —
they're inflated by this hub effect, and we're flagging that honestly
rather than presenting the higher number as a win.

## 13. So, is region-specific matching actually better than full-length?

Yes and no, and both parts matter:

- **Yes, mechanically**: cutting out the correct amplicon region (instead
  of using the whole gene) produced a real, measurable calibration signal
  where before there was none (flat 9% chance-level). The extraction
  method itself — primer-based, not arbitrary truncation — is sound and
  reusable.
- **No, not enough to trust the numbers yet**: real-read matching is
  dominated by the hubness artifact described above, and — more
  fundamentally — **the reads aren't 16S or 18S in the first place**
  (Section 8). Even a perfectly clean 16S or 18S matching pipeline cannot
  correctly identify fish from a fish-mitochondrial-marker dataset,
  because the right reference sequences for that marker simply are not
  in SILVA.

## 14. Final recommendation

For this specific dataset (`ERR16922870`, `16SFish/D` marker), **SILVA is
the wrong reference database at the gene level**, regardless of which
region is extracted from it. The scientifically correct fix is to use a
reference database that actually contains fish mitochondrial 16S
sequences (for example, MitoFish or a GenBank/NCBI query restricted to
Chordata mitochondrial 16S) instead of SILVA SSURef.

If the POC must stay within SILVA-only reference matching as a
methodology demonstration (e.g. for a bacterial/archaeal or eukaryotic
eDNA dataset in a future run, where SILVA genuinely is the right
database), then **the region-specific extraction approach in this
section (Part 2) should be used instead of the full-length approach in
Part 1** — it produces real, measurable calibration signal, whereas the
full-length approach does not. But the hubness issue in Section 12 would
still need to be investigated and resolved (e.g. by re-ranking with a
second, stricter similarity check) before assignment counts from it
could be trusted as species identifications.

## 15. What's established vs. what's demonstrated (so nothing here gets
overstated later)

- **Established by the dataset's own metadata** (ENA, the official public
  archive — not our assumption): this run used the `16SFish/D` marker, a
  fish mitochondrial 16S gene target.
- **Established by outside authoritative sources** (Earth Microbiome
  Project protocols, used in official QIIME2 tutorials): 16S V4
  (515F/806R) and 18S V9 (1391F/EukBr) are the standard, documented
  regions for bacterial/archaeal and eukaryotic SSU rRNA amplicon work
  respectively — these were not invented for this project.
- **Demonstrated by actually running our pipeline on real data**: primer
  extraction works and produces the expected amplicon lengths; region-
  matched calibration shows real signal unlike full-length; but real-read
  matching is currently dominated by a hubness artifact, and — because of
  the metadata finding above — no SILVA-based result for this dataset
  should be reported as a validated species identification yet.
- **Still uncertain**: we could not find PCR primer sequences for
  `16SFish/D` itself in the public metadata, so we can't yet build a
  region-matched *mitochondrial* reference set the same way we did for
  16S/18S — that would need either the original study's methods section
  or a literature search specifically for "16SFish/D primer sequence."

## 16. Files and commands for this part

**New files created:**
- `src/extract_amplicon_region.py` — primer-based region extraction (in-silico PCR)
- `src/run_region_experiment.py` — builds index, queries real reads, decides, per region
- `data/silva_16s_v4_amplicons.csv`, `data/silva_18s_v9_amplicons.csv` — full extracted pools
- `data/silva_16s_v4_sample.csv`, `data/silva_18s_v9_sample.csv` — 5,000-record subsamples
- `data/read_taxonomy_results_16s_v4.csv`, `data/read_taxonomy_results_18s_v9.csv` — final per-read results

**Modified files:**
- `src/subsample_reference_table.py` — sequence-validity check now accepts both `U` and `T` alphabets (extracted amplicons are already DNA-form)

**Commands to reproduce, in order:**
```bash
# 1. Extract the two regions from the full 2.2M-record SILVA table
python -m src.extract_amplicon_region data/silva_reference_table.csv data/silva_16s_v4_amplicons.csv 16s_v4
python -m src.extract_amplicon_region data/silva_reference_table.csv data/silva_18s_v9_amplicons.csv 18s_v9

# 2. Subsample each to 5,000 records
python -m src.subsample_reference_table data/silva_16s_v4_amplicons.csv data/silva_16s_v4_sample.csv 5000 42
python -m src.subsample_reference_table data/silva_18s_v9_amplicons.csv data/silva_18s_v9_sample.csv 5000 42

# 3. Embed each reference sample (reuses the existing embed-ref stage)
REFERENCE_TABLE_CSV=data/silva_16s_v4_sample.csv REFERENCE_EMBEDDINGS_NPY=data/silva_16s_v4_ref_embeddings.npy REFERENCE_IDS_CSV=data/silva_16s_v4_ref_ids.csv python run_pipeline.py embed-ref
REFERENCE_TABLE_CSV=data/silva_18s_v9_sample.csv REFERENCE_EMBEDDINGS_NPY=data/silva_18s_v9_ref_embeddings.npy REFERENCE_IDS_CSV=data/silva_18s_v9_ref_ids.csv python run_pipeline.py embed-ref

# 4. Calibrate a threshold for each region (region-matched synthetic short reads)
REFERENCE_TABLE_CSV=data/silva_16s_v4_sample.csv REFERENCE_EMBEDDINGS_NPY=data/silva_16s_v4_ref_embeddings.npy REFERENCE_IDS_CSV=data/silva_16s_v4_ref_ids.csv python -m src.synthetic_read_calibration 3
REFERENCE_TABLE_CSV=data/silva_18s_v9_sample.csv REFERENCE_EMBEDDINGS_NPY=data/silva_18s_v9_ref_embeddings.npy REFERENCE_IDS_CSV=data/silva_18s_v9_ref_ids.csv python -m src.synthetic_read_calibration 3

# 5. Query the real 5,000 reads against each region-specific reference set and decide
python -m src.run_region_experiment 16s_v4 data/silva_16s_v4_sample.csv data/silva_16s_v4_ref_embeddings.npy data/silva_16s_v4_ref_ids.csv 0.64
python -m src.run_region_experiment 18s_v9 data/silva_18s_v9_sample.csv data/silva_18s_v9_ref_embeddings.npy data/silva_18s_v9_ref_ids.csv 0.66
```

**Housekeeping note**: all the SILVA files referenced by name above (the
`.csv`, `.npy`, `.index` files under `data/`) have since been moved into
`data/SILVA results/` to keep them separate from the MitoFish run below.
The commands above still work — just add the `data/SILVA results/` prefix
to each path, or set the env vars as shown in Part 3's reproduction
commands for the equivalent MitoFish pattern.

---

# Part 3: Trying the *actual* right database — MitoFish

Part 2 ended with a recommendation: since the real marker for this
dataset is a fish mitochondrial gene, and SILVA doesn't contain that gene
at all, the scientifically correct fix is a database that actually has
fish mitochondrial sequences. This section reports what happened when we
did that.

## 17. Verifying everything before touching any code

Following the same "verify, don't assume" approach as Part 2, four things
were checked against real evidence before writing any extraction code:

**a) Is this really the `16SFish/D` marker?** Re-confirmed from the same
official ENA metadata as Part 2 (run alias `16SFish_OcOm_2305_100_1_run`)
— yes.

**b) What are the actual `16SF/D` + `16S2R` primer sequences?** These
weren't given to us — we had to find them. A literature search turned up
the same pair, confirmed from a peer-reviewed source (Berry et al. 2017,
as tabulated in Table 1 of a 2020 methods-comparison paper on PMC/NCBI):

- `16SF/D` (forward): `GACCCTATGGAGCTTTAGAC`
- `16S2R` (reverse): `CGCTGTTATCCCTADRGTAACT`
- Expected amplicon: ~219 base pairs

**c) Is `data/mitofishdb.fa.gz` actually usable?** We opened it and
looked. It turned out to be a big mixed bag: some entries are complete
fish mitochondrial genomes (up to ~23,400 base pairs — the *entire*
mitochondrial DNA circle of a fish), some are short single-gene barcode
sequences (mostly a different gene, COI, at ~650bp), and most entries
have no label at all saying which gene they are. **There is also no
species-name file included** — each entry is just a bare accession number
like `PQ550629`.

This meant we could **not** trust the file's headers to tell us which
records contain the 16S region — the only reliable way to find it was to
actually search inside each sequence for the real primer sequences (the
same in-silico PCR approach from Part 2), which we tested on a sample of
20,000 records first: 530 had the forward primer, and 513 of those
(96.8%) had the complete reverse primer site too — unlike SILVA's 18S V9
case, MitoFish sequences are *not* systematically cut short right at the
primer site, so no special adjustment was needed here.

**d) Does this match the real reads?** The expected amplicon (~219bp) is
longer than our reads (118-140bp) — expected, since a single-end
Illumina read normally only covers part of a longer amplicon. And none
of the 5,000 real reads contain the raw primer sequence itself, which is
normal (primers are usually stripped out by standard eDNA processing
before the data is uploaded, not something to worry about).

All four checks passed, so we went ahead — see
`data/MitoFish results/run_report.md` for the full verification write-up.

## 18. Keeping this run separate from the SILVA one

Before starting, all of Part 1 and Part 2's output files were moved into
`data/SILVA results/`, and this run's outputs go into
`data/MitoFish results/` — so the two are never mixed up, and nothing
from the earlier SILVA work was deleted or changed.

## 19. Extracting the 16S region from MitoFish

Running the verified primer search on the **full** file (624,421 records,
not just the 20,000-record test sample):

- 17,227 records had the forward primer
- 16,778 of those (97.4%) also had the reverse primer, giving a complete
  amplicon — this is the same in-silico PCR method as Part 2, not a
  truncation
- 186 more were dropped for containing ambiguous "could be any base"
  letters (same rule as Parts 1-2)
- **16,592 usable fish 16S reference sequences remained**, averaging
  **204.4 base pairs** — close to the ~219bp expected from the primer
  pair's literature description (the small gap is because this length
  excludes the primers themselves, which get trimmed off during
  extraction)

Unlike SILVA (2.2 million records, needed subsampling to 5,000 for time
reasons), MitoFish's usable pool is small enough that **all 16,592**
records were used directly — no subsampling needed this time. This is in
`data/MitoFish results/mitofish_16s_reference_table.csv`.

## 20. A calibration test that failed — and what it taught us

We first tried calibrating a confidence threshold exactly the same way as
Part 2: cut a short random piece out of a held-back reference sequence,
see if its nearest match is the same one. Result: **0% accuracy at every
threshold we tried, from very loose to very strict.**

Instead of picking a different number and moving on, we stopped and
checked why (this matters — the project's own rule is not to force a
result when something doesn't check out). We took a held-out sequence's
*entire* amplicon (not a short random piece) and searched for it in the
remaining pool — if the matching code works, this should trivially find
a perfect match. It did find matches with similarity 1.0 (a perfect
score) in 207 out of 300 tries — but the record it matched almost never
had the *same accession number* as the original. Looking closer: MitoFish
contains many different accession numbers that all share the exact same
16S sequence — the same species (or a very close relative) submitted to
GenBank multiple times under different IDs. So "0% exact accession
match" didn't mean the matching was broken; it meant we were grading it
against the wrong answer key.

We fixed the test to check "did it find a sequence that reads identically
to the true one" instead of "did it find the literal same accession
number" — and got a real, usable curve this time:

| How strict the cutoff is | % of test pieces confidently matched | How often that match was correct |
|---|---|---|
| Loose | ~100% | 9.2% |
| Medium | 26.3% | 16.1% |
| Strict | 10.8% | 18.3% |
| Very strict | 0.6% | 35.0% |

This is real, climbing signal — like Part 2's 16S V4/18S V9 curves, and
unlike Part 1's flat 9%-no-matter-what curve — though the ceiling here
(35% at the strictest, least-used setting) is lower than SILVA V9's 84%.
We used **0.80** (a "strict" setting, ~18% expected accuracy on this
proxy test) as the actual cutoff for the real-read run below.

## 21. Running the real 5,000 reads against MitoFish

- Real reads' similarity to their best MitoFish match: as low as 0.48,
  as high as 0.82, averaging 0.66.
- At the 0.80 cutoff: **211 of 5,000 reads (4.2%) got assigned** a match;
  the rest stayed unclassified.

This is in `data/MitoFish results/read_taxonomy_results_mitofish_16s.csv`.

## 22. Checking for the same "hub" problem as before — and finding it, worse

Section 12 in Part 2 found that a handful of SILVA references were
soaking up matches from hundreds of unrelated reads — a known AI
similarity-search glitch called "hubness," not real biology. We checked
for the same thing here.

**It's there, and more concentrated**: out of all 5,000 reads, only 347
different references (2% of the 16,592-record pool) were ever picked as
anyone's best match. And among just the 211 reads that got "assigned":
only 13 distinct references show up at all, and **2 single references
account for 188 of the 211 assignments (89%)**.

In plain terms: two individual fish-mitochondrial sequences are being
credited with matching 188 different water samples' worth of DNA
fragments. That's not biologically plausible for a real marine eDNA
sample (which should contain many different species). This is the same
hub-artifact warning sign as Part 2, and it means **the 211 "assigned"
count should not be reported as 211 correct fish identifications.**

## 23. So — is MitoFish actually better?

**At the database level: yes, clearly.** MitoFish is a real fish
mitochondrial sequence collection, which is what this dataset's marker
(`16SFish/D`) actually targets — SILVA simply doesn't contain this gene
at all, no matter how you slice the region. The calibration curve here is
real (Section 20), same as Part 2's SILVA V4/V9 results and unlike Part
1's flat chance-level curve.

**At the "can I trust the 211 assignments" level: not yet.** The hubness
problem in Section 22 is the same failure mode found in Part 2, and it's
worse here. On top of that, MitoFish (as downloaded) has no species names
attached to its sequences, so we can't yet even double-check the 211
assignments by eye to see if they're plausible fish species for the
Indian Ocean.

## 24. What's established vs. demonstrated (same format as Part 2)

- **Established by outside sources**: `16SF/D` + `16S2R` primer sequences
  (Berry et al. 2017), confirmed from a peer-reviewed table, not guessed.
- **Established by inspecting the real file**: MitoFish is a mixed-gene,
  mixed-length collection with no reliable header labels and no species
  mapping — primer search was the only safe extraction method, and this
  was verified on a sample before running on the full 624,421-record
  file.
- **Demonstrated by running the pipeline**: extraction yields real
  amplicons at the expected length; calibration (after fixing the wrong
  ground-truth metric) shows real signal; but real-read matching is
  dominated by the same hubness artifact seen with SILVA, so the raw
  "211 assigned" number is not yet trustworthy as a set of correct fish
  identifications.
- **Still uncertain / left for future work**: no species names are
  available for MitoFish's sequences in this file, so nobody can
  currently manually check whether the 211 assigned reads make
  biological sense (e.g. "are these plausible Indian Ocean fish
  species?"). Fixing the hubness problem (e.g. by re-checking the top few
  candidate matches with a second, independent method instead of trusting
  the single nearest neighbor) is also unresolved.

## 25. Files and commands for this part

**New files created:**
- `src/extract_mitofish_16s.py` — primer-based 16S extraction from `data/mitofishdb.fa.gz`
- `src/mitofish_calibration.py` — threshold calibration using sequence-identity recovery (see Section 20 for why accession-ID recovery doesn't work here)
- `data/MitoFish results/mitofish_16s_amplicons.csv` — all 16,778 extracted amplicons
- `data/MitoFish results/mitofish_16s_reference_table.csv` — 16,592 usable references (final reference table)
- `data/MitoFish results/mitofish_16s_ref_embeddings.npy`, `mitofish_16s_ref_ids.csv` — embeddings
- `data/MitoFish results/mitofish_16s_faiss.index` — search index
- `data/MitoFish results/query_results_mitofish_16s.csv`, `read_taxonomy_results_mitofish_16s.csv` — real-read results
- `data/MitoFish results/run_report.md` — full verification + results log for this run
- `data/SILVA results/README.md` — pointer explaining the SILVA-results folder

**Modified files:**
- `src/config.py` — default paths updated to point at `data/SILVA results/` after the file reorganization (every actual command still passes paths explicitly)
- `src/run_region_experiment.py` — output directory is now a parameter (was hardcoded to `data/` with a leftover `silva_` prefix in the index filename)

**Files/folders moved (not deleted):**
- All SILVA-specific `data/*.csv`, `*.npy`, `*.index`, and the raw SILVA FASTA moved into `data/SILVA results/`

**Commands to reproduce, in order:**
```bash
# 1. Extract the 16S region from the full MitoFish file (verified primers)
python -m src.extract_mitofish_16s data/mitofishdb.fa.gz "data/MitoFish results/mitofish_16s_amplicons.csv"

# 2. Filter out ambiguous-base sequences (same rule as Parts 1-2); target
#    is set >= the clean pool size so effectively everything usable is kept
python -m src.subsample_reference_table "data/MitoFish results/mitofish_16s_amplicons.csv" "data/MitoFish results/mitofish_16s_reference_table.csv" 16592 42

# 3. Embed the reference table (reuses the existing embed-ref stage)
REFERENCE_TABLE_CSV="data/MitoFish results/mitofish_16s_reference_table.csv" REFERENCE_EMBEDDINGS_NPY="data/MitoFish results/mitofish_16s_ref_embeddings.npy" REFERENCE_IDS_CSV="data/MitoFish results/mitofish_16s_ref_ids.csv" python run_pipeline.py embed-ref

# 4. Calibrate a threshold (sequence-identity recovery -- see Section 20)
REFERENCE_TABLE_CSV="data/MitoFish results/mitofish_16s_reference_table.csv" REFERENCE_EMBEDDINGS_NPY="data/MitoFish results/mitofish_16s_ref_embeddings.npy" REFERENCE_IDS_CSV="data/MitoFish results/mitofish_16s_ref_ids.csv" python -m src.mitofish_calibration

# 5. Query the real 5,000 reads and decide
python -m src.run_region_experiment mitofish_16s "data/MitoFish results/mitofish_16s_reference_table.csv" "data/MitoFish results/mitofish_16s_ref_embeddings.npy" "data/MitoFish results/mitofish_16s_ref_ids.csv" 0.80 data/reads_ids.csv data/reads_embeddings.npy "data/MitoFish results"
```

---

# Part 4: Fixing the hubness problem with a second, independent check

Part 3 ended on an unresolved problem: even after switching to the
biologically correct MitoFish database, a handful of reference sequences
were still "winning" way more matches than makes biological sense — the
classic hubness glitch. This section reports a focused fix for exactly
that problem, without touching the database, primers, or extracted
region again (those were already verified in Part 3 and are treated as
settled).

## 26. The idea: don't trust DNABERT's ranking alone — check the actual DNA

DNABERT similarity is a single, somewhat opaque number (a "how alike do
these look to the AI" score). It has no built-in way to tell "this read
and this reference share 98% of their actual letters" apart from "this
read and this reference just happen to sit near each other in the AI's
internal math space." The fix: for each read, don't just trust DNABERT's
single best guess. Instead:

1. Ask DNABERT for its **top 10** candidate matches (not just the top 1).
2. For each of those 10, run a real, direct DNA comparison — a **local
   sequence alignment** (the same family of algorithm behind tools like
   BLAST), which lines up the read against each candidate letter-by-letter
   and reports what percent of the letters actually match ("percent
   identity") and what fraction of the read's own length was covered by
   a good alignment.
3. Re-rank the 10 candidates by this real percent-identity score instead
   of DNABERT's similarity score, and use the winner of *that* comparison
   as the actual answer.

We used "local" alignment (as opposed to a strict end-to-end
"global" alignment) because our reads (118-140bp) are shorter than the
reference amplicons (up to 249bp) and we don't know exactly where within
the amplicon the read starts — a local alignment finds the best-matching
stretch wherever it happens to sit, which is the correct tool for this
situation (this is explained further in `src/nucleotide_rerank.py`).

## 27. First result: DNABERT's own ranking is often wrong

Running this on all 5,000 real reads: only **1,071 out of 5,000 (21%)**
of the time was DNABERT's #1 pick also the winner after the real DNA
comparison. The rest of the time, a candidate DNABERT had ranked lower
(sometimes as low as #10) turned out to actually be the better DNA match.
This on its own is useful evidence that DNABERT's similarity score alone
isn't precise enough to trust blindly — which is exactly what Part 3's
hubness finding already suggested.

## 28. A calibration test that initially looked disappointing — and a real discovery underneath it

Just like Part 3 (Section 20), we calibrated a confidence cutoff using
held-out data: cut a short random piece from a reference we set aside,
run it through the same top-10-then-realign pipeline, and check if the
answer is right.

The first version of this test used the same "exact same sequence"
answer key as before, and initially the accuracy still only reached
about 33% even at the strictest settings — not obviously better than Part
3's plain-DNABERT result. But before accepting that, we checked *why* the
"wrong" answers were wrong, the same way we caught the earlier
accession-numbering issue in Part 3.

**What we found**: when the reranked pick was "wrong" by the exact-match
answer key, we compared the *chosen* reference's full sequence to the
*true* one directly. They were still **95% to 99.55% identical (median
99%)**. In other words, the method wasn't picking a wrong answer — it was
picking a different accession number that happens to carry an
essentially identical (same-species-or-closer) sequence, which is exactly
the redundancy problem already documented in Part 3.

So we recalibrated using a fairer, still strict answer key: "count it as
correct if the chosen reference's sequence is at least 95% identical to
the true one" (95% comfortably separates genuine near-duplicates, which
cluster around 99%, from an actual wrong answer). With that fix:

| How strict the identity cutoff is | % of test pieces confidently matched | How often that was actually correct |
|---|---|---|
| Loose (45-62.5%) | 100% | 57.5% |
| Medium (80%) | 84.3% | 68.2% |
| Strict (90%) | 71.7% | 80.2% |
| Very strict (95%) | 59.8% | 94.6% |
| Extremely strict (97.5%) | 51.5% | 99.4% |

This is a real, strong, climbing curve — noticeably better than
everything tried in Parts 1-3. **We used 95% identity as the cutoff**
for the real-read run below: high confidence (94.6% accuracy in
calibration) while still keeping a meaningful fraction of reads (not
picking the very tip of the curve just to look impressive).

## 29. Applying this to the real 5,000 reads

At the calibrated 95% identity cutoff:

- **474 of 5,000 reads (9.5%) got assigned** — more than double Part 3's
  211 (4.2%), and this time backed by real DNA evidence: median 98.4%
  identity, and the alignment covered essentially the entire read
  (median 100% of read length) for the assigned reads.
- The DNABERT similarity scores for these same 474 reads were, by
  themselves, unremarkable (0.53-0.82, similar to the noisy background
  Part 3 already saw) — reinforcing that DNABERT's own number wasn't
  the thing that made these trustworthy; the independent DNA check was.

This is in `data/MitoFish results/read_taxonomy_results_mitofish_16s_reranked.csv`.

## 30. Checking hubness again — smaller, but is it still a glitch?

Among the 474 assigned reads: only **27 distinct references** were
involved (down from Part 3's already-small 13, but now backed by real
evidence instead of raw similarity), and the top 2 references alone
still account for **57%** of all assignments, top 10 for **94%**.

On the surface this still looks like the same hubness problem. But this
time, instead of assuming that, we checked what these top references
actually *are* — something we couldn't do in Parts 1-3 because MitoFish
ships with no species names. We looked up the real organism names for
all 27 distinct assigned references directly from NCBI's public GenBank
database (not guessed — see
`data/MitoFish results/assigned_reference_species_lookup.csv` for the
full list with citations).

**The result was unambiguous**: 445 of 474 assigned reads (93.9%)
matched a **tuna species** (family Scombridae) — *Thunnus tonggol*
(longtail tuna), *Thunnus maccoyii* (southern bluefin), *Thunnus obesus*
(bigeye), *Thunnus albacares* (yellowfin), *Thunnus thynnus* (Atlantic
bluefin), *Thunnus orientalis* (Pacific bluefin), *Thunnus alalunga*
(albacore), *Thunnus atlanticus* (blackfin), *Katsuwonus pelamis*
(skipjack tuna), and *Auxis rochei* (bullet tuna) — plus one black marlin
(a closely related large pelagic fish) and a small number of unrelated
species (a reef fish and a deep-sea anglerfish, 5.9% combined).

**Why this matters**: the Indian Ocean is one of the world's major tuna
fishing and migration regions, and this sample was seawater collected
there. A handful of different tuna species dominating the identified
signal is exactly the ecological pattern a marine biologist would expect
— tuna are large, abundant, and shed a lot of DNA into the surrounding
water. This is a completely different situation from Part 3's hubness,
where the dominant references (checked again for comparison — see
`data/MitoFish results/run_report.md`) shared nothing in common with
these new tuna-dominated hits: the two sets of "hub" references don't
even overlap, which is further evidence the earlier hubness really was
an unrelated math artifact, not this same biological signal resurfacing
under a new name.

## 31. Verdict: is hubness solved?

**Partly, and in the way that matters most.** The raw "a few
references soak up too many matches" *symptom* is still visible in the
numbers (top 2 = 57%). But we now have independent, real-world evidence
(NCBI species lookups, not just internal statistics) that this
particular concentration reflects a genuinely dominant local species
group (tuna) rather than an AI-embedding-space glitch — which is a
fundamentally different, and much more defensible, situation than Part
3's unexplained concentration on random, unrelated-looking references.

**What would make this fully trustworthy, not just plausible:**
- Confirming with the study's collection metadata (or contacting the
  original researchers) whether tuna fishing/monitoring activity was
  actually happening at this specific sample site — the strongest
  possible confirmation.
- Checking whether the 5.9% non-tuna hits (the reef fish and anglerfish)
  are also ecologically sensible for this location, or should be treated
  more cautiously.
- Given MitoFish's own redundancy (many accessions sharing near-identical
  sequences), getting a species-level, non-redundant reference set
  would make the remaining 90.5% "unclassified" reads worth
  re-examining too, since some of them may simply have no exact database
  representative rather than truly being unidentifiable.

**Bottom-line answer to "can we present these as real fish
identifications now?"**: The 474 tuna-dominated calls are noticeably more
trustworthy than anything produced in Parts 1-3 — they're backed by a
calibrated, held-out-validated identity threshold (94.6% measured
accuracy) *and* an independent, real-world plausibility check (NCBI
species names matching known Indian Ocean tuna fisheries), not by a
single similarity number. They are reasonable to present as a
**preliminary, methodologically-supported result** — with the caveats in
this section stated alongside them — rather than as a fully validated,
publication-ready species list.

## 32. Files and commands for this part

**New files created:**
- `src/nucleotide_rerank.py` — local alignment (Smith-Waterman via Biopython) + candidate reranking logic
- `src/mitofish_rerank_experiment.py` — runs the rerank on all 5,000 real reads
- `src/mitofish_rerank_calibration.py` — calibrates the identity_pct threshold on held-out data
- `data/MitoFish results/reads_top10_sims.npy`, `reads_top10_pos.npy`, `reads_top10_ids.csv` — top-10 DNABERT candidates per read
- `data/MitoFish results/rerank_candidates.csv` — all 10 candidates x 5,000 reads, with both DNABERT similarity and alignment identity/coverage
- `data/MitoFish results/read_taxonomy_results_mitofish_16s_reranked.csv` — final per-read decision at the calibrated threshold
- `data/MitoFish results/rerank_calibration_log.txt` — calibration sweep output
- `data/MitoFish results/assigned_reference_species_lookup.csv` — NCBI-verified species names for all 27 distinct assigned references

**Modified files:** none (Part 3's reference table, embeddings, and index were reused unchanged, as instructed)

**Commands to reproduce, in order:**
```bash
# 1. Get top-10 DNABERT/FAISS candidates for all 5,000 real reads (already-computed
#    read embeddings and MitoFish index from Part 3, nothing re-embedded)
python - <<'PY'
import faiss, numpy as np
from src.faiss_index import l2_normalize
index = faiss.read_index("data/MitoFish results/mitofish_16s_faiss.index")
emb = l2_normalize(np.load("data/reads_embeddings.npy").astype("float32"))
with open("data/reads_ids.csv") as f:
    read_ids = [l.strip() for l in f if l.strip()]
sims, pos = index.search(emb, 10)
np.save("data/MitoFish results/reads_top10_sims.npy", sims)
np.save("data/MitoFish results/reads_top10_pos.npy", pos)
open("data/MitoFish results/reads_top10_ids.csv","w").write("\n".join(read_ids)+"\n")
PY

# 2. Rerank each read's top-10 candidates by real nucleotide alignment
python -m src.mitofish_rerank_experiment

# 3. Calibrate the identity_pct threshold on held-out MitoFish data
REFERENCE_TABLE_CSV="data/MitoFish results/mitofish_16s_reference_table.csv" REFERENCE_EMBEDDINGS_NPY="data/MitoFish results/mitofish_16s_ref_embeddings.npy" REFERENCE_IDS_CSV="data/MitoFish results/mitofish_16s_ref_ids.csv" python -m src.mitofish_rerank_calibration
```

---

# Part 5: Open-set handling — reporting a best guess instead of silence, with real species names

Prompted by the professor's next-steps list, specifically: *"after all that, go for unknown DNA → match to most similar."* This section covers what changed in `src/final_pipeline.py` (the production BM25 → MMR → nucleotide-reranker pipeline from Part 4/Task 4) and adds a new module, `src/species_lookup.py`. Nothing about the database, primers, extracted region, BM25/MMR parameters, or the frozen nucleotide reranker was touched — this is purely about what gets reported for a read that doesn't clear the confidence bar.

## 33. The problem this solves

Before this change, `assign()` only ever returned two outcomes: `"assigned"` (identity ≥95%, Part 4's calibrated bar) or `"unclassified"` — and `"unclassified"` reads got a **blank** nearest-reference field, even though the pipeline had already computed a nearest match and its identity score internally. A read that was 93% identical to a known sequence — genuinely close — was reported exactly the same as a read that was 52% identical to anything — essentially unrelated. That's throwing away real, differentiated evidence.

## 34. What changed: three confidence tiers instead of two

`Confidence_Decision` is now one of:

| Tier | identity_pct | Meaning |
|---|---|---|
| `assigned` | ≥ 95% | unchanged — Part 4's calibrated, trustworthy bar |
| `best_guess` | 75–95% | a real, fairly close lead — worth reporting as a candidate identification, not a confirmed one |
| `best_guess_low_confidence` | < 75% | the nearest thing found, but weak — reported so "closest known relative" is never silently discarded, explicitly flagged as unreliable |
| `unclassified` | — | no BM25 candidate at all (the read shares no k-mers with any reference) — nothing to report a nearest match for |

**Important honesty note**: the 75% split is *not* a recalibrated, held-out-validated threshold the way the 95% bar is (that one came from Part 4's calibration curve). It only controls the *label* attached to reads that already fail the 95% bar — it doesn't change which reads get assigned, and the underlying `identity_pct` number is always present in the output regardless of tier, so nothing is hidden by the label.

**Real-read check** (first 15 reads of `ERR16922870`, MitoFish 16S index): 5 of 15 landed `assigned` (97.5–100% identity, all tuna or footballfish — ecologically plausible), 1 landed `best_guess` (93.5%), and 9 landed `best_guess_low_confidence` (52–71% identity, matched to a scattered, unrelated mix of deep-sea species — exactly what a weak/noise match should look like). This distinction — strong pattern vs. scattered noise — was invisible under the old binary scheme.

## 35. Species names instead of accession codes

MitoFish ships with no species names at all — every reference is a bare accession like `AY920478` (established in Part 3, Section 17c). Part 4 solved this once, by hand, for its own 27 specific assigned references (`assigned_reference_species_lookup.csv`). That doesn't cover new accessions surfaced by the current BM25→MMR pipeline.

`src/species_lookup.fetch_organism_names()` looks up real organism names from NCBI's public Entrez `esummary` API (the same public database Part 3 used for its one-off lookup), for any list of accession IDs, and caches results to `data/species_name_cache.csv` so repeat runs never re-hit the network for an accession already seen. A network failure degrades gracefully — cached names still resolve, and unresolved accessions come back as `""` rather than crashing the run. `assign_batch(..., fetch_species_names=True)` adds a `Species` column using this; it's off by default so the core pipeline and its tests stay fully offline.

## 36. Files and commands for this part

**New file:** `src/species_lookup.py` — NCBI accession → species name, disk-cached.

**Modified:** `src/final_pipeline.py` — three-tier `Confidence_Decision`, optional `fetch_species_names` in `assign_batch`.

**New tests:** `tests/test_species_lookup.py` (cache hit/miss/network-failure, all mocked — no network in CI), plus two new cases in `tests/test_final_pipeline.py` covering the `best_guess` and `best_guess_low_confidence` tiers. Full suite: 75/75 passing.

**To reproduce the real-read check above:**
```python
from src.final_pipeline import PipelineIndex, assign_batch

index = PipelineIndex.build(
    "data/MitoFish results/mitofish_16s_reference_table.csv",
    "data/MitoFish results/mitofish_16s_ref_embeddings.npy",
    "data/MitoFish results/mitofish_16s_ref_ids.csv",
)
results = assign_batch(read_ids, read_seqs, index, fetch_species_names=True)
```

---

# Part 6: Testing a second open-source model — Nucleotide Transformer v2

Prompted by the professor's "open-source LLM models for retrieval" /
"comparison of models" / "which classifies DNA better" next steps. Goal:
test whether swapping the embedding model improves on DNABERT-S, using a
real measurement rather than assuming either model is better.

## 37. Model chosen, and why

`InstaDeepAI/nucleotide-transformer-v2-50m-multi-species` (50M
parameters). Picked over DNABERT-2 and HyenaDNA specifically for
**feasibility on this machine**: it uses a standard `transformers`
architecture with no `triton`/`flash-attn`/custom-CUDA-kernel dependency
-- the exact thing that already blocked `zhihan1996/DNABERT-S` from
loading on this Windows/CPU setup (see the README's Setup section).
Published in *Nature Methods* (de Almeida et al., 2024) with broad
benchmark validation across genomic tasks, and multi-species pretraining
fits an eDNA use case better than a human-genome-only model.

**Feasibility verified before any large-scale run**, not assumed: Python
3.10, `transformers` 4.56.2, `torch` 2.13.0 (CPU build), 16 CPU cores,
125GB RAM, 258GB free disk, reachable HuggingFace endpoint, and the
model's `modeling_esm.py` inspected directly -- confirmed plain PyTorch
only, no exotic kernel imports. A 64-sequence timing benchmark
(19.5ms/sequence) gave an estimated ~7-8 minutes for the full comparison
job (references + holdout fragments), which the actual run matched.

## 38. What was compared, and how

`src/nt_embed.py` -- a new, fully isolated embedding module (mean-pooled
final hidden layer, same pooling strategy as `embed_common.py`'s DNABERT
path, for a fair comparison). Nothing in `embed_common.py`, `config.py`,
or any frozen Part 3/4 artifact was touched.

`src/nt_diagnostics.py` reuses `stage0_diagnostics.build_holdout_split()`
UNMODIFIED -- same seed=42, same 1,000 held-out MitoFish fragments, same
>=95% full-sequence-identity ground truth (`nucleotide_rerank.align_pair`,
frozen) already used for DNABERT's own Recall@K numbers. This guarantees
the two models are compared on the *identical* test set, not a
re-sampled, potentially easier or harder one.

## 39. Result: DNABERT-S wins clearly

| Metric (n=1,000 held-out fragments) | DNABERT-S (existing, frozen) | Nucleotide Transformer v2 50M (new) |
|---|---|---|
| Recall@1 | 41.7% | 15.3% |
| Recall@5 | 53.3% | 20.1% |
| Recall@10 | 58.2% | 22.3% |
| Absent from top-10 | 41.8% | 77.7% |

**Why**: DNABERT-S was purpose-trained for species differentiation (its
own paper: "Pioneering Species Differentiation with Species-Aware DNA
Embeddings") -- exactly this task. Nucleotide Transformer was trained
more generally (masked-language-modeling across genomic sequences, not a
species-contrastive objective), so it underperforms here specifically,
not universally -- it may well be competitive on a task it was actually
built for (e.g. promoter/splicing prediction).

## 40. Decision: not adopted

`final_pipeline.py`, BM25, MMR, and the frozen nucleotide reranker are
**unchanged**. Since Nucleotide Transformer lost decisively on the exact
task this project needs (species-level retrieval), there is no basis for
carrying it into items 3/4 of the roadmap (BM25 + best-model + MMR). This
is treated as a completed, honest negative result, not a dead end: it
answers "did we check" with real evidence rather than an assumption.

## 41. Files and commands for this part

**New files:** `src/nt_embed.py`, `src/nt_diagnostics.py`,
`data/NT diagnostics/nt_recall_summary.txt`,
`data/NT diagnostics/nt_train_embeddings.npy`,
`data/NT diagnostics/nt_train_ids.csv`.

**Modified:** none of the existing pipeline, model config, or frozen
artifacts.

**Reproduce:**
```bash
python -m src.nt_diagnostics   # ~7-8 min on CPU, no GPU required
```

---

# Part 7: Two more open-source models — completing the comparison

Continuation of Part 6, testing two further open-source models so the
"comparison of models" / "which classifies DNA better" next-steps items
are answered with more than one data point.

## 42. Models chosen, and why

**DNABERT-2** (`zhihan1996/DNABERT-2-117M`) was tried first as the
literature-recommended next candidate (higher published zero-shot AUC
than Nucleotide Transformer). It loads without error, but crashes at
inference: its remote code calls a Triton flash-attention kernel, and
this environment's `torch` build ships Triton (unlike the Windows
machine that blocked DNABERT-S outright), so it imports successfully --
but the kernel itself hard-requires an actual CUDA GPU
(`assert q.is_cuda`), which this CPU-only machine doesn't have. This is
a **new, more specific finding** than Part 6's Windows/triton
observation: even where Triton *is* installed, DNABERT-2 remains
CPU-infeasible for a different reason. Confirmed by running it, not
assumed from the README. Rejected before any large-scale embedding.

Two working, architecturally distinct replacements were verified (source
code inspected for triton/flash-attn/custom-kernel imports, then
smoke-tested, before any full run -- same discipline as Part 6):

- **HyenaDNA-tiny** (`LongSafari/hyenadna-tiny-1k-seqlen-hf`) -- a
  state-space/implicit-convolution architecture (Hyena operator), not
  attention-based at all. Genuinely different design family from
  DNABERT-S/Nucleotide Transformer, chosen for architectural diversity
  in the comparison rather than three transformer variants. Very small
  (2 layers, d_model=128).
- **GENA-LM** (`AIRI-Institute/gena-lm-bert-base-t2t`) -- a standard BERT
  architecture, but trained on the human T2T reference assembly, a
  different pretraining corpus from both DNABERT-S and Nucleotide
  Transformer.

## 43. What was compared, and how

Same methodology as Part 6, generalized into one script
(`src/multi_model_diagnostics.py`) instead of duplicating it per model:
`stage0_diagnostics.build_holdout_split()` reused UNMODIFIED (identical
1,000 held-out MitoFish fragments, seed=42), identical
>=95%-full-sequence-identity ground truth
(`nucleotide_rerank.align_pair`, frozen). Both new models' embedding
logic lives in their own isolated modules (`src/hyenadna_embed.py`,
`src/genalm_embed.py`) -- `embed_common.py`, `config.py`, and every
frozen Part 3/4 artifact remain untouched.

One real implementation issue surfaced and fixed during this part:
HyenaDNA's tokenizer returns no `attention_mask` (it's a causal-LM,
char-level tokenizer, not built for masked/bidirectional pooling) and
left-pads sequences. Mean-pooling against a phantom "all tokens are
real" mask would have silently corrupted every embedding. Fixed by
deriving the mask directly from `input_ids != pad_token_id` instead of
assuming `attention_mask` exists -- caught by inspecting the tokenizer's
actual output before trusting it, not assumed to work like the other
three models' tokenizers.

## 44. Result: both underperform DNABERT-S, more severely than Part 6's model

| Metric (n=1,000 held-out fragments) | DNABERT-S (frozen) | Nucleotide Transformer 50M (Part 6) | HyenaDNA-tiny | GENA-LM |
|---|---|---|---|---|
| Recall@1 | 41.7% | 15.3% | 0.1% | 2.3% |
| Recall@5 | 53.3% | 20.1% | 0.7% | 7.7% |
| Recall@10 | 58.2% | 22.3% | 1.9% | 12.9% |
| Absent from top-10 | 41.8% | 77.7% | 98.1% | 87.1% |

**Why HyenaDNA-tiny is near-random**: at 2 layers and d_model=128 it is
the smallest model in the comparison by a wide margin, and its
architecture family (implicit long convolution) was designed for
long-range sequence modeling, not species-level discrimination between
short fragments -- a capacity/objective mismatch, not a bug.

**Why GENA-LM underperforms**: trained on the human T2T genome
specifically, not multi-species genomic data -- a domain mismatch for
fish mitochondrial 16S sequences, consistent with Part 6's finding that
training objective/corpus matters more here than raw model
sophistication.

## 45. Decision: DNABERT-S confirmed as the best model for this use case

Four models tested now (DNABERT-S, Nucleotide Transformer, HyenaDNA-tiny,
GENA-LM); DNABERT-S wins by a wide, consistent margin on every metric
against every alternative. `final_pipeline.py`, BM25, MMR, and the
frozen nucleotide reranker remain unchanged. This closes out the
"open-source LLM models for retrieval" / "comparison of models" / "which
classifies DNA better" roadmap items with a well-evidenced answer:
DNABERT-S, because it is species-contrastively trained for exactly this
task, while every alternative tested is either too general-purpose, too
small, or trained on the wrong corpus.

## 46. Files and commands for this part

**New files:** `src/hyenadna_embed.py`, `src/genalm_embed.py`,
`src/multi_model_diagnostics.py`,
`data/HyenaDNA diagnostics/recall_summary.txt`,
`data/GENA-LM diagnostics/recall_summary.txt` (+ each model's train
embeddings/ids).

**Modified:** none of the existing pipeline, model config, or frozen
artifacts.

**Rejected without a full run:** DNABERT-2 (`zhihan1996/DNABERT-2-117M`)
-- CPU-infeasible on this machine, confirmed by direct execution
(Section 42).

**Reproduce:**
```bash
python -m src.multi_model_diagnostics   # ~13 min on CPU, no GPU required
```

---

# Part 8: Validating the architecture beyond fish

Prompted by the professor's "when you get accurate results, see other
species other than fish" next-steps item. Goal: test whether the
production architecture (BM25 -> MMR -> nucleotide reranker -> tiered
confidence decision, `src/final_pipeline.py`) generalizes to kingdoms
other than fish, without sourcing brand-new datasets when suitable ones
already exist in the project.

## 47. Data source, and why no new download was needed

Part 2 (Sections 9-10) already downloaded the full SILVA database (2.2M
sequences), verified real published primer pairs (515F/806R for
bacterial/archaeal 16S V4, 1391F/EukBr for eukaryotic 18S V9) via
literature search, and extracted amplicon regions from SILVA using
primer-based in-silico PCR (not arbitrary truncation) -- producing
`data/SILVA results/silva_16s_v4_sample.csv` (5,000 Bacteria/Archaea
references) and `silva_18s_v9_sample.csv` (5,000 Eukaryota references),
plus their DNABERT embeddings. These are genuinely different kingdoms
from fish (Chordata), sitting unused since Part 2 -- Parts 3/4's
BM25/MMR/nucleotide-reranker architecture was only ever built and tested
against MitoFish (fish) afterward. Reusing this existing, already-verified
data avoided repeating Part 2/3's full dataset-sourcing effort (finding a
database, verifying primers, running in-silico PCR, checking against real
metadata) for what would otherwise be a multi-day task.

## 48. Important scope limitation, stated up front

There is **no real bacterial/archaeal or eukaryotic eDNA read data** in
this project -- only the fish-marker `ERR16922870` reads exist, and Part
2 (Section 8) already established via ENA metadata that those reads are
neither 16S nor 18S at all (they're a fish mitochondrial marker SILVA
doesn't cover). This section is therefore a **held-out synthetic-fragment
validation** of the architecture -- the same rigor Stage 0-4 used to
validate it for fish before any real read was queried -- not a
real-read classification run. Results here answer "does this
architecture generalize," not "here are real bacterial/eukaryotic
species identifications."

## 49. Method

`src/other_kingdom_diagnostics.py` generalizes
`stage0_diagnostics.build_holdout_split`'s exact methodology (seed=42,
80/20 train/holdout split, 1,000 synthetic 118-140bp fragments cut from
held-out references) to accept any reference table + embeddings instead
of MitoFish specifically, then runs the FROZEN, unmodified
`BM25Index`, `mmr_select`, and `rerank_candidates` with final_pipeline.py's
exact production hyperparameters (BM25 k=6, MMR pool=30/lambda=0.5,
top-10, 95%/75% tiered decision) against each kingdom's held-out split.

## 50. Result: the architecture generalizes well

| Metric (held-out fragments) | Fish / MitoFish (Part 4/5) | Bacteria/Archaea 16S V4 | Eukaryota 18S V9 |
|---|---|---|---|
| n_holdout_fragments | 1000 | 1000 | 505* |
| Recall@1 | 93.7% | 79.9% | 68.5% |
| Recall@10 | 94.6% | 82.0% | 70.5% |
| % assigned (>=95% identity) | 94.8% | 81.4% | 69.5% |
| Accuracy on assigned | 98.7% | 97.7% | 99.7% |
| % best_guess (75-95%) | -- | 18.4% | 27.7% |
| % best_guess_low_confidence (<75%) | -- | 0.2% | 2.8% |

*Eukaryota 18S V9 has fewer holdout fragments than requested (505 of
1,000) because many held-out 18S V9 amplicons are shorter than the
118-140bp fragment-length range (Part 2, Section 10: this SILVA file's
18S V9 extraction averages only 127bp, near the fragment-length floor),
so fewer references are long enough to cut a fragment from -- a data
characteristic, not a bug.

**Accuracy on assigned reads stays >=97.7% across all three kingdoms** --
the architecture was built and tuned entirely on fish/MitoFish data, yet
transfers to two unrelated kingdoms without any kingdom-specific
retuning of BM25's k-mer size, MMR's lambda, or the 95%/75% decision
thresholds. Recall and % assigned are lower for Bacteria/Archaea and
Eukaryota than for fish, which is expected: SILVA's V4/V9 regions are
shorter and the reference sets are broader/more diverse subsamples of a
2.2M-sequence database, versus MitoFish's more targeted single-marker
collection.

## 51. Decision and what remains open

The architecture is validated as kingdom-agnostic at the retrieval/
decision level. What's still open, honestly: no real bacterial/archaeal/
eukaryotic eDNA sample has been run through it (Section 48) -- doing that
would require sourcing a real amplicon-sequencing dataset for one of
these kingdoms, the same kind of real-world verification Part 3 did for
MitoFish. That is future work, not claimed here.

## 52. Files and commands for this part

**New file:** `src/other_kingdom_diagnostics.py`,
`data/Other Kingdoms diagnostics/Bacteria_Archaea_16S_V4/summary.txt`,
`data/Other Kingdoms diagnostics/Eukaryota_18S_V9/summary.txt` (+
per-fragment CSVs for each).

**Modified:** none of the existing pipeline, model config, or frozen
artifacts. `final_pipeline.py`'s hyperparameters are imported and reused,
not duplicated by value.

**Reproduce:**
```bash
python -m src.other_kingdom_diagnostics   # ~2-3 min on CPU
```

---

# Part 9: A real (non-fish) eDNA read, for the first time

Part 8 validated the architecture on held-out SYNTHETIC fragments only,
and stated plainly that no real bacterial/archaeal/eukaryotic eDNA read
existed in this project. This section closes that gap for Eukaryota:
finding, verifying, and running a real seawater eDNA dataset end-to-end
through the production architecture -- the same rigor Part 3 used to
find MitoFish for fish, applied here to a different kingdom.

## 53. Finding a real dataset, and a real primer mismatch caught early

Searched ENA directly (not assumed from a paper abstract) for real,
public seawater eDNA amplicon datasets. Several candidates were checked
and rejected before settling on one:

- A small Japanese 16S seawater study (`PRJDB1797`/`1803`/`1947`) used
  old 454-pyrosequencing reads (~515bp average) -- wrong platform/length
  for this project's short-read methodology.
- A larger 515F/806R V4 study (`PRJDB15800`) looked promising on primers
  alone, but its individual samples turned out to be lab-cultured
  enrichment communities, not raw environmental seawater -- checked by
  reading actual sample titles, not assumed from the project description.
- **Tara Oceans/Tara Pacific** (`PRJEB36282` for 16S, `PRJEB60216` for
  18S V9) are real, published, peer-reviewed global ocean eDNA surveys --
  genuine filtered seawater, modern Illumina sequencing. `PRJEB60216`
  (18S V9, ~151bp reads) was selected as faster to validate than the 16S
  option, since its read length already matches this project's
  established 118-140bp methodology without extra trimming decisions.

**Primer mismatch caught before downloading anything large**: Tara
Pacific's actual 18S V9 primers are `1389F`/`1510R` (confirmed via
literature, Nature Scientific Data 2017), NOT the `1391F`/`EukBr` pair
Part 2 used to build the original SILVA V9 extraction. Close, but
different -- exactly the kind of mismatch this project's own Part 1
already proved matters. Reusing Part 2's existing extraction here would
have repeated that mistake.

## 54. Re-extracting SILVA with the real primers

`src/extract_amplicon_region_tara.py` adds a new primer entry to
`extract_amplicon_region.py`'s existing `PRIMERS` dict AT IMPORT TIME
(no edit to that frozen file) and reuses its `extract_region()` function
unchanged. Run against the full 2,224,690-record SILVA table:

- 172,680 Eukaryota records scanned
- 865 had the forward primer (1389F) -- a much lower hit rate than Part
  2's 1391F extraction (140,968 usable amplicons), because 1389F's exact
  15bp window doesn't match as many real SILVA sequences as 1391F's
  slightly different window. A real, measured difference between two
  "almost the same" primers, not assumed to be equivalent.
- 864/865 (99.9%) ALSO matched the reverse primer (1510R) completely --
  unlike Part 2's EukBr, which was truncated in this SILVA file and
  needed a "forward-to-end" approximation. This extraction is a
  genuinely complete, both-primers-verified amplicon, not an
  approximation -- a real quality improvement, offset by a much smaller
  pool (864 vs. 140,968).
- After removing 11 ambiguous-base sequences (same rule as every prior
  stage): **853 usable references**, embedded with DNABERT-S
  (multimolecule/dnaberts, unchanged model).

## 55. Environment note: multimolecule/DNABERT-S needed a version fix

Embedding this new reference table was the first time this session
actually invoked `embed_common.py`'s DNABERT-S path (every model
comparison in Parts 6-7 used its own separate embedding module and never
touched it). It failed immediately: `multimolecule==0.2.1` (already
installed) could not import under `transformers==4.56.2` (also already
installed) -- a real version-incompatibility, not a code bug. Checking
`.venv-win` (the Windows venv the project's own README says has the
working combination) showed the actual working pin: `transformers==
5.15.1`. Installing that exact version fixed it immediately and the full
test suite (75/75) still passes under it. This is now the working
combination for this Linux `.venv` too.

## 56. Real reads needed new handling MitoFish's reads didn't

Downloaded the first 5,000 real reads from ERR10928327 (streamed and
truncated early via `curl | zcat | head`, avoiding a full ~100MB+
download for data only the first 5,000 records of which were needed).
Inspecting them (not assuming they'd look like `ERR16922870`) found a
real difference: **these reads still carry their PCR primer sequence**,
unlike the fish reads, which arrived primer-trimmed. Roughly half start
with the forward primer (1389F) and roughly half start with the reverse
primer (1510R) ITSELF (not its reverse complement) -- meaning this run's
R1 file mixes reads primed from both ends of the amplicon, expected
because the ~130bp insert is shorter than the 150bp read length, so a
read can run past the insert into the opposite primer's binding site.

`src/run_tara_18s_v9_experiment.py` detects which primer each read
starts with (tolerating a leading `N` low-quality basecall), trims it,
and reverse-complements reverse-primed reads so every read is compared
in the same orientation as the reference table's forward strand. Reads
matching neither primer (246/5,000, 4.9% -- likely extra sequencing
error near the read start) are excluded rather than guessed at.

## 57. Result: first real non-fish classification

4,754 of 5,000 reads had a usable, oriented primer match and were run
through the FROZEN production architecture (BM25 -> MMR -> nucleotide
reranker -> tiered decision) against the 853-reference, real-primer-
verified SILVA 18S V9 set:

| Decision | Count | % |
|---|---|---|
| assigned (>=95% identity) | 226 | 4.8% |
| best_guess (75-95%) | 3,882 | 81.7% |
| best_guess_low_confidence (<75%) | 646 | 13.6% |

The 226 assigned reads span only 10 distinct references, with the top 2
absorbing 187/226 (82.7%) -- a concentration reminiscent of Part 3's
fish "hubness" finding. Unlike that case, though, this pipeline already
includes the fix Part 4 built for exactly that problem (the independent
nucleotide-alignment reranker), so this concentration more likely
reflects the reference pool's small size (853 records, many of which are
probably near-duplicate representatives of a few commonly-sequenced
marine lineages) than an embedding-space artifact -- but this is **not
verified the way Part 4's tuna finding was** (no independent NCBI
species check was possible here: SILVA's own labels for the top hits are
literally "uncultured eukaryote" / "uncultured marine eukaryote",
offering no specific species name to cross-check against known plankton
ecology). Stated as an open question, not resolved.

## 58. What's established vs. still open

- **Established**: a real, verified, primer-matched non-fish eDNA
  dataset now exists and was run end-to-end through the unmodified
  production architecture; the low 4.8% coverage is consistent with
  Part 8's synthetic finding (69.5% assigned there used a 5,000-record
  pool -- this real run's reference pool is 853 records, ~6x smaller, so
  materially lower real-read coverage is expected, not surprising).
- **Still open**: whether the top-2-reference concentration is real
  biology or an artifact of the small reference pool is unresolved --
  unlike Part 4's tuna verification, SILVA provides no species-level
  label here to check against. Growing the reference pool (a primer
  search compatible with 1510R's un-truncated match, run against a
  larger SILVA sample) would be the natural next step to test whether
  coverage and concentration both improve with more references, mirroring
  how MitoFish's full 16,592-record pool outperformed a 5,000-record
  SILVA sample in Parts 1-3.

## 59. Files and commands for this part

**New files:** `src/extract_amplicon_region_tara.py`,
`src/run_tara_18s_v9_experiment.py`,
`data/SILVA results/silva_18s_v9_tara_amplicons.csv`,
`data/SILVA results/silva_18s_v9_tara_reference_table.csv` (853 refs),
`data/SILVA results/silva_18s_v9_tara_ref_embeddings.npy` /
`_ref_ids.csv`, `data/tara_18s_v9_first5000.fastq` (real reads from
ERR10928327), `data/SILVA results/tara_18s_v9_read_taxonomy_results.csv`.

**Modified:** none of the existing pipeline or frozen artifacts.
`extract_amplicon_region.py`'s `PRIMERS` dict gains a new entry added at
runtime by the new script, not by editing that file.

**Environment change**: `transformers` pinned to `5.15.1` in `.venv`
(was 4.56.2) to match the working `.venv-win` combination -- required
for `multimolecule`/DNABERT-S to load at all; full test suite reverified
passing (75/75) under this version.

**Reproduce:**
```bash
python -m src.extract_amplicon_region_tara "data/SILVA results/silva_reference_table.csv" "data/SILVA results/silva_18s_v9_tara_amplicons.csv"
python -m src.subsample_reference_table "data/SILVA results/silva_18s_v9_tara_amplicons.csv" "data/SILVA results/silva_18s_v9_tara_reference_table.csv" 900 42
REFERENCE_TABLE_CSV="data/SILVA results/silva_18s_v9_tara_reference_table.csv" REFERENCE_EMBEDDINGS_NPY="data/SILVA results/silva_18s_v9_tara_ref_embeddings.npy" REFERENCE_IDS_CSV="data/SILVA results/silva_18s_v9_tara_ref_ids.csv" python run_pipeline.py embed-ref
curl -s "https://ftp.sra.ebi.ac.uk/vol1/fastq/ERR109/027/ERR10928327/ERR10928327_1.fastq.gz" | zcat | head -n 20000 > "data/tara_18s_v9_first5000.fastq"
python -m src.run_tara_18s_v9_experiment
```

---

# Part 10: A real Bacteria/Archaea eDNA read, completing the pair

Part 9 closed the real-read gap for Eukaryota. This section does the
same for Bacteria/Archaea -- the other still-open item from Part 8/9 --
using Tara Oceans' real prokaryote 16S dataset. Same methodology
throughout, no code shortcuts taken because it's the "second" kingdom.

## 60. No new download needed for the reference side

Unlike sourcing a brand-new read dataset, the reference side reused data
already on disk: the full 2,224,690-record SILVA table
(`data/SILVA results/silva_reference_table.csv`, from Part 1) was
already local, so re-extracting a different region from it costs no
network time at all -- only the real Tara Oceans reads themselves needed
downloading.

## 61. Primer verification: smaller mismatch than the 18S case, but real

Tara Oceans' prokaryote protocol uses `515F-Y`/`926R` (Parada et al.
2016), confirmed via literature (Section 42/54's same discipline). `515F-Y`
is IDENTICAL to Part 2's original `515F` primer -- only the reverse
primer differs (`806R` -> `926R`), extending the amplicon from V4-only
into V4+V5. Still re-extracted properly rather than assumed close enough:
`src/extract_amplicon_region_tara_16s.py` adds a `16s_v4v5_tara` entry to
`extract_amplicon_region.py`'s `PRIMERS` dict (same additive pattern as
Part 9, no edit to that frozen file).

Run against the full local SILVA table:
- 2,052,010 Bacteria/Archaea records scanned
- 1,831,825 (89.3%) matched BOTH primers completely -- a far higher hit
  rate than the 18S V9 case, and higher even than Part 2's original V4
  extraction, consistent with 515F-Y being a very broadly-matching primer
- Amplicon length: mean 373.1bp (min 128, max 1559) -- matches the
  literature's ~400bp V4-V5 expectation well
- Subsampled to 5,000 (same size as Part 2's other SILVA samples, for
  consistency), 122 ambiguous-base records dropped -> **5,000 usable
  references**, embedded with DNABERT-S

## 62. Same real-read complication as Part 9, confirmed not a V9-only quirk

Downloaded the first 5,000 reads of ERR6090675 (Tara Oceans prokaryotes,
streamed + truncated early, same method as Part 9). Checked rather than
assumed: roughly half start with the forward primer (515F-Y) and roughly
half start with the reverse primer (926R) ITSELF -- the same dual-
orientation mixing Part 9 found for 18S V9, now confirmed to also occur
for a longer amplicon (373bp vs V9's ~130bp) where a read running past
the insert into the opposite primer is less obviously expected. This
means the mixing is a property of how this run's R1 file was assembled
generally, not simply a short-amplicon read-through effect as originally
guessed in Part 9. `src/run_tara_16s_v4v5_experiment.py` reuses the exact
same detect/trim/reverse-complement approach as Part 9's script.

- forward-primer (515F-Y) reads: 2,429
- reverse-primer (926R) reads: 2,338 (reverse-complemented before matching)
- no primer detected (excluded): 233 (4.7%)
- usable: 4,767 of 5,000

## 63. Result: much stronger coverage than Eukaryota, with a checkable plausibility signal

| Decision | Count | % |
|---|---|---|
| assigned (>=95% identity) | 3,189 | 66.9% |
| best_guess (75-95%) | 1,568 | 32.9% |
| best_guess_low_confidence (<75%) | 10 | 0.2% |

66.9% assigned vs. Part 9's 4.8% for Eukaryota -- expected, since this
reference pool (5,000, sampled from 1.83M available matches) is far
larger and better-populated than the 18S V9 pool (853, the entire usable
set). The 3,189 assigned reads span 117 distinct references (vs. Part
9's 10) -- top 5 references account for 41.8% of assignments, a real
concentration but much less extreme than Part 9's top-2 = 82.7%.

**Unlike Part 9, a plausibility check IS possible here**: SILVA's labels
for these hits include named genera, not just "uncultured" -- Roseivivax
salinus, Rhodobacteraceae bacterium, Flavobacteriia bacterium, and
"uncultured marine archaeon" all appear among the top organisms.
Rhodobacteraceae and Flavobacteriia are well-established, genuinely
abundant marine bacterioplankton lineages in real oceanography
literature -- this pattern is ecologically plausible, similar in spirit
(though not independently NCBI-verified the way Part 4's tuna finding
was) to the fish/tuna result, and a stronger signal than Part 9 could
offer.

## 64. What's established vs. still open (mirrors Part 8/9's format)

- **Established**: a real, primer-verified Bacteria/Archaea eDNA dataset
  now runs end-to-end through the unmodified production architecture,
  with materially higher coverage (66.9%) than the smaller-pool
  Eukaryota case, and a label set that's at least directionally
  consistent with known marine microbiology.
- **Still open**: no independent species-level verification (an NCBI
  lookup, as Part 4 did for the 27 tuna-dominated MitoFish references)
  was performed here -- SILVA's own taxonomy labels are trusted as-is.
  Whether the top-5-reference concentration (41.8%) reflects genuine
  ecological dominance or partial reference-pool artifact is not fully
  resolved, though it is far less extreme than Part 9's case.

## 65. Files and commands for this part

**New files:** `src/extract_amplicon_region_tara_16s.py`,
`src/run_tara_16s_v4v5_experiment.py`,
`data/SILVA results/silva_16s_v4v5_tara_amplicons.csv`,
`data/SILVA results/silva_16s_v4v5_tara_reference_table.csv` (5,000
refs), `data/SILVA results/silva_16s_v4v5_tara_ref_embeddings.npy` /
`_ref_ids.csv`, `data/tara_16s_v4v5_first5000.fastq` (real reads from
ERR6090675), `data/SILVA results/tara_16s_v4v5_read_taxonomy_results.csv`.

**Modified:** none of the existing pipeline or frozen artifacts.
`data/MitoFish results/` (fish) remains completely untouched and
separate, as it has been since Part 3.

**Reproduce:**
```bash
python -m src.extract_amplicon_region_tara_16s "data/SILVA results/silva_reference_table.csv" "data/SILVA results/silva_16s_v4v5_tara_amplicons.csv"
python -m src.subsample_reference_table "data/SILVA results/silva_16s_v4v5_tara_amplicons.csv" "data/SILVA results/silva_16s_v4v5_tara_reference_table.csv" 5000 42
REFERENCE_TABLE_CSV="data/SILVA results/silva_16s_v4v5_tara_reference_table.csv" REFERENCE_EMBEDDINGS_NPY="data/SILVA results/silva_16s_v4v5_tara_ref_embeddings.npy" REFERENCE_IDS_CSV="data/SILVA results/silva_16s_v4v5_tara_ref_ids.csv" python run_pipeline.py embed-ref
curl -s "https://ftp.sra.ebi.ac.uk/vol1/fastq/ERR609/005/ERR6090675/ERR6090675_1.fastq.gz" | zcat | head -n 20000 > "data/tara_16s_v4v5_first5000.fastq"
python -m src.run_tara_16s_v4v5_experiment
```

---

# Part 11: Fine-tuning DNABERT-S separately per kingdom

New next-step, from the professor: fine-tune DNABERT-S on each of the 3
species datasets (real data, 1,000-5,000 samples, not the whole
database), as 3 separate models. This section covers what was fine-tuned
on, how, the results, and a real evaluation bug caught and fixed along
the way.

## 66. What each kingdom was fine-tuned on, and why the labels differ

All three use REAL reference data already established as correct for
each kingdom in this project (MitoFish for fish, the Tara-primer-
verified SILVA extractions for the other two -- not Part 2's older,
primer-mismatched SILVA samples):

| Kingdom | Reference data | Records | Label source |
|---|---|---|---|
| Fish | `mitofish_16s_finetune_sample.csv` (5,000-record subsample of the full MitoFish table) | 5,000 | **Fetched from NCBI** via `species_lookup.py` (reused, unmodified) -- MitoFish itself ships with no species names at all (established in Part 3); genus (first word of the resolved name) used as the class label |
| Bacteria/Archaea | `silva_16s_v4v5_tara_reference_table.csv` (Part 10's real-primer-verified 16S V4-V5 set) | 5,000 | SILVA's own `Taxonomy_Full`, truncated to depth 4 |
| Eukaryota | `silva_18s_v9_tara_reference_table.csv` (Part 9's real-primer-verified 18S V9 set) | 853 (the entire usable pool -- naturally under 1,000) | SILVA's own `Taxonomy_Full`, truncated to depth 4 |

Sequences whose class has fewer than 2 members were dropped (no positive
pair possible) -- reported, not silently discarded: 4,654/5,000 (fish),
4,907/5,000 (bacteria), 844/853 (eukaryota) records were usable.

## 67. Method: supervised contrastive fine-tuning, not continued MLM pretraining

`src/finetune_dnabert.py` fine-tunes the FULL `multimolecule/dnaberts`
model (117M parameters) with a supervised contrastive loss (SupCon,
Khosla et al. 2020) using class-balanced batches (4 classes x 4
sequences/class per batch, so every anchor has a guaranteed in-batch
positive) -- pulling same-class sequences' mean-pooled, L2-normalized
embeddings together and pushing different-class ones apart. This
mirrors DNABERT-S's own original training philosophy (species-aware
contrastive learning), just specialized to each kingdom's real data
instead of left as the frozen general-purpose checkpoint.

Chosen over continued masked-language-model pretraining because every
downstream use of these embeddings (FAISS cosine retrieval, MMR
diversity scoring) is nearest-neighbor geometry -- directly optimizing
"same class close, different class far" is a more direct match to that
use case than optimizing token-prediction accuracy.

Each kingdom trained SEPARATELY (2 epochs, AdamW lr=1e-5, temperature
0.1) -- no weight sharing across kingdoms, three independent checkpoints
saved to `models/dnaberts_finetuned_{fish,bacteria,eukaryota}/`
(gitignored -- 447MB each, 1.3GB total, not committed). Loss decreased
in every epoch of every run (e.g. fish: 1.437 -> 1.312; bacteria: 1.489
-> 1.274; eukaryota: 2.034 -> 1.541), confirming the training loop is
genuinely learning, not stuck.

Feasibility checked before committing to the full run (same discipline
as every prior model decision in this project): a small timing benchmark
(one fwd+bwd+optimizer step, batch=16) measured ~2s/step on this CPU-only
machine, giving a real, not guessed, time estimate before the ~80-minute
total training run across all three kingdoms was launched.

## 68. A real evaluation bug caught and fixed before trusting any result

The first evaluation attempt compared each fine-tuned model against the
WRONG frozen baseline: for bacteria and eukaryota, it reused Part 8's
already-saved Recall@10 numbers (82.0% / 70.5%), which were measured on
Part 2's OLDER, primer-mismatched SILVA extractions (`silva_16s_v4_sample.csv`,
`silva_18s_v9_sample.csv`) -- NOT the same Tara-primer-verified tables
the models were actually fine-tuned on. This produced an alarming,
wrong-looking result (bacteria appearing to collapse from 82.0% to
26.4%) that turned out to be an artifact of evaluating a V4-V5-specialized
model against a V4-only test set, not a real finding about fine-tuning.

Caught by checking what `other_kingdom_diagnostics.KINGDOMS` actually
pointed to, rather than trusting the printed delta at face value.
`src/evaluate_finetuned.py` was corrected to build the held-out
split/synthetic fragments from the EXACT SAME reference tables each
model was fine-tuned on (`REF_TABLES` dict), and to recompute a fresh,
domain-matched frozen baseline in the same run for bacteria and
eukaryota (fish's existing Stage 0 baseline was legitimately reusable
as-is, since fine-tuning used a subsample of that exact same full
MitoFish table).

## 69. Result: genuinely mixed, kingdom-dependent

| Kingdom | n_holdout | Recall@10 frozen | Recall@10 fine-tuned | Delta |
|---|---|---|---|---|
| Fish | 1,000 | 58.2% | **78.6%** | **+20.4pp** |
| Bacteria/Archaea | 1,000 | 17.1% | 7.9% | -9.2pp |
| Eukaryota | 89 | 62.9% | **69.7%** | **+6.7pp** |

(Note: the bacteria/eukaryota frozen baselines here, 17.1%/62.9%, are
NOT the same numbers as Part 8's 82.0%/70.5% -- those were measured on
different, primer-mismatched reference tables, per Section 68. The
V4-V5 region is intrinsically harder for the frozen model than the
V4-only region Part 8 tested, independent of fine-tuning.)

Fish improved substantially -- expected, since real genus-level labels
(fetched from NCBI specifically for this) gave the model a strong,
clean supervised signal it never had access to before. Eukaryota
improved modestly. **Bacteria regressed.** A plausible explanation,
stated as a hypothesis, not confirmed: the bacteria fine-tuning set
spans 177 classes (the most of any kingdom) over only 2 epochs -- the
model may need more epochs or a lower learning rate to avoid disrupting
the pretrained representation before the contrastive signal fully
consolidates across that many classes, whereas fish (569 classes, but a
much stronger absolute starting-point gap of 41.8pp room-to-improve) and
eukaryota (only 29 classes) may have converged more easily in the same
2-epoch budget. This was not re-run with more epochs to confirm --
stated as open, not resolved.

## 70. What's established vs. still open

- **Established**: fine-tuning genuinely changes retrieval behavior
  (never a no-op) and helps in 2 of 3 kingdoms tested, by a real,
  measured margin, on a domain-matched held-out split -- not assumed
  to help universally, and the one regression is reported plainly
  rather than hidden.
- **Still open**: whether more epochs, a different learning rate, or
  more training classes would fix bacteria's regression; whether this
  changes end-to-end BM25->MMR->reranker pipeline results (all
  evaluation here is embedding-retrieval-only, Recall@K, the same
  scope Stage 0-3 used before the full architecture was assembled) --
  the next step (BM25+MMR vs BM25+RRF comparison, professor's next
  item) will determine whether these embedding changes matter at all
  once BM25 is the primary retriever and the fine-tuned embeddings are
  only used for MMR's diversity signal.

## 71. Files and commands for this part

**New files:** `src/finetune_dnabert.py`, `src/evaluate_finetuned.py`,
`models/dnaberts_finetuned_{fish,bacteria,eukaryota}/` (gitignored,
1.3GB total), `data/MitoFish results/mitofish_16s_finetune_sample.csv`
(5,000-record real MitoFish subsample with NCBI-resolved genus labels),
`data/MitoFish results/finetune_species_name_cache.csv`,
`data/*/finetune_metadata.json` per kingdom, `data/finetune_evaluation.log`.

**Modified:** `.gitignore` (added `models/`). No existing pipeline code,
frozen artifact, or prior model checkpoint touched.

**Reproduce:**
```bash
python -m src.finetune_dnabert "data/SILVA results/silva_18s_v9_tara_reference_table.csv" taxonomy_depth models/dnaberts_finetuned_eukaryota 4 2
python -m src.finetune_dnabert "data/SILVA results/silva_16s_v4v5_tara_reference_table.csv" taxonomy_depth models/dnaberts_finetuned_bacteria 4 2
python -m src.finetune_dnabert "data/MitoFish results/mitofish_16s_finetune_sample.csv" genus models/dnaberts_finetuned_fish 4 2
python -m src.evaluate_finetuned
```
