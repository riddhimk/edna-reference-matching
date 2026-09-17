"""
Runs the production architecture (BM25 -> MMR -> nucleotide reranker ->
tiered confidence decision, src/final_pipeline.py, FROZEN/unmodified) on
REAL seawater eDNA reads for Bacteria/Archaea -- Tara Oceans prokaryote
16S amplicon sequencing (ENA run ERR6090675, project PRJEB36282),
against the SILVA reference set extracted with this dataset's own real
primers (src/extract_amplicon_region_tara_16s.py). Companion to
src/run_tara_18s_v9_experiment.py -- same methodology, different kingdom.

Real-read preprocessing: same primer-detection/orientation issue as the
18S V9 case (Explanation.md Part 9) reproduces here too, confirmed by
inspection rather than assumed to only be a V9 quirk -- roughly half of
this run's R1 reads start with the forward primer (515F-Y) and roughly
half start with the reverse primer (926R) ITSELF, meaning R1 mixes reads
primed from both ends. Reads primed by 926R are reverse-complemented
before comparison; reads matching neither primer (~4.7%) are excluded.

Does not modify final_pipeline.py, BM25Index, mmr_select,
rerank_candidates, or any frozen artifact. Reference table/embeddings
come from the new, real-primer-verified extraction
(data/SILVA results/silva_16s_v4v5_tara_reference_table.csv), kept
entirely separate from data/MitoFish results/ (fish) and the other
SILVA/Tara outputs already in data/SILVA results/.

Usage:
    python -m src.run_tara_16s_v4v5_experiment
"""
import csv
import re

from .final_pipeline import PipelineIndex, assign_batch

READS_FASTQ = "data/tara_16s_v4v5_first5000.fastq"
REF_TABLE_CSV = "data/SILVA results/silva_16s_v4v5_tara_reference_table.csv"
REF_EMB_NPY = "data/SILVA results/silva_16s_v4v5_tara_ref_embeddings.npy"
REF_IDS_CSV = "data/SILVA results/silva_16s_v4v5_tara_ref_ids.csv"
OUT_CSV = "data/SILVA results/tara_16s_v4v5_read_taxonomy_results.csv"

FWD_PRIMER_RE = re.compile("^[NG][TC]G[CT]CAGC[AC]GCCGCGGTAA")          # 515F-Y, tolerating a leading N basecall
REV_PRIMER_RE = re.compile("^[NC]CG[CT]CAATT[CT][AC]TTT[AG]AGTTT")     # 926R, same tolerance

_COMPLEMENT = {"A": "T", "C": "G", "G": "C", "T": "A", "N": "N"}


def _revcomp(seq: str) -> str:
    return "".join(_COMPLEMENT[b] for b in reversed(seq))


def load_reads(path):
    read_ids, seqs = [], []
    with open(path) as f:
        lines = f.readlines()
    for i in range(0, len(lines), 4):
        read_ids.append(lines[i].strip().lstrip("@").split()[0])
        seqs.append(lines[i + 1].strip())
    return read_ids, seqs


def orient_and_trim(seq: str):
    m = FWD_PRIMER_RE.match(seq)
    if m:
        return seq[m.end():]
    m = REV_PRIMER_RE.match(seq)
    if m:
        return _revcomp(seq[m.end():])
    return None


def load_taxonomy_lookup(ref_table_csv):
    lookup = {}
    with open(ref_table_csv) as f:
        for row in csv.DictReader(f):
            lookup[row["Reference_ID"]] = row.get("Organism_Label", "")
    return lookup


def main():
    print("=== Real-read run: Tara Oceans 16S V4-V5 (ERR6090675) vs SILVA (real-primer-verified) ===\n")

    raw_ids, raw_seqs = load_reads(READS_FASTQ)
    print(f"Loaded {len(raw_ids)} real reads from {READS_FASTQ}")

    read_ids, read_seqs = [], []
    n_fwd = n_rev = n_no_primer = 0
    for rid, seq in zip(raw_ids, raw_seqs):
        if FWD_PRIMER_RE.match(seq):
            n_fwd += 1
        elif REV_PRIMER_RE.match(seq):
            n_rev += 1
        trimmed = orient_and_trim(seq)
        if trimmed is None:
            n_no_primer += 1
            continue
        read_ids.append(rid)
        read_seqs.append(trimmed)

    print(f"  forward-primer (515F-Y) reads: {n_fwd}")
    print(f"  reverse-primer (926R) reads: {n_rev} (reverse-complemented before comparison)")
    print(f"  no primer detected (excluded): {n_no_primer}")
    print(f"  usable reads after primer trim/orient: {len(read_ids)}\n")

    print("Building index from real-primer-verified SILVA 16S V4-V5 reference set...")
    index = PipelineIndex.build(REF_TABLE_CSV, REF_EMB_NPY, REF_IDS_CSV)
    print(f"  {len(index.ref_ids)} references loaded\n")

    print(f"Running BM25 -> MMR -> nucleotide reranker on {len(read_ids)} reads...")
    results = assign_batch(read_ids, read_seqs, index)

    taxonomy = load_taxonomy_lookup(REF_TABLE_CSV)
    for r in results:
        r["Nearest_Organism"] = taxonomy.get(r["Nearest_Reference_ID"], "")

    fieldnames = ["Read_ID", "Nearest_Reference_ID", "Nearest_Organism", "identity_pct",
                  "coverage_pct", "bm25_score", "Confidence_Decision"]
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(results)

    n = len(results)
    counts = {}
    for r in results:
        counts[r["Confidence_Decision"]] = counts.get(r["Confidence_Decision"], 0) + 1

    print("\n=== Results ===")
    print(f"n_reads_processed: {n} (of {len(raw_ids)} loaded; {n_no_primer} excluded for no primer detected)")
    for decision, count in sorted(counts.items()):
        print(f"  {decision:28s}: {count} ({100*count/n:.1f}%)")

    assigned = [r for r in results if r["Confidence_Decision"] == "assigned"]
    if assigned:
        print(f"\nSample of {min(10, len(assigned))} assigned reads (real seawater plankton eDNA -> SILVA taxonomy):")
        for r in assigned[:10]:
            print(f"  {r['Read_ID']:20s} identity={r['identity_pct']:6.2f}%  {r['Nearest_Organism']}")

    print(f"\nFull results written to: {OUT_CSV}")
    return results


if __name__ == "__main__":
    main()
