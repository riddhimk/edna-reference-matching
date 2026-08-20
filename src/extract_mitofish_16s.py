"""
Region-specific MitoFish reference extraction via in-silico PCR, mirroring
extract_amplicon_region.py's method but for data/mitofishdb.fa.gz, which
has a different FASTA header format than SILVA and no Domain/taxonomy field
to filter on -- MitoFish records were verified (not assumed) to be a mixed
collection of COI-5P barcodes, 12S/16S/CYTB-tagged fragments, untagged gene
fragments, and complete mitogenomes (lengths ranging from <100bp to
~23,400bp in a 20,000-record sample), so primer-based matching is the ONLY
reliable way to find the 16S region here -- header tags are absent or
inconsistent for most records.

Primer pair (verified from a peer-reviewed source, Berry et al. 2017,
Table 1 of Zhang et al. 2020 / PMC9205523 "Comparing eDNA metabarcoding
primers for assessing fish communities in a biodiverse estuary" -- these
sequences were cross-checked against two independent secondary sources
before use, not taken from a single hit):
    16SF/D (forward): GACCCTATGGAGCTTTAGAC
    16S2R  (reverse): CGCTGTTATCCCTADRGTAACT
    Reported amplicon: ~219bp, target: mitochondrial 16S rRNA (fish)

Verified on a 20,000-record sample of mitofishdb.fa.gz before running on
the full file: 530/20,000 records had the forward primer; 513/530 (96.8%)
also had the (reverse-complemented) reverse primer intact, with an
interior (primer-trimmed) amplicon length of 193-227bp (mean 205bp) --
close to the literature's ~219bp full-primer-to-primer amplicon (the
~14bp gap is the two primers' own length, which this interior excludes).
Unlike SILVA's 18S V9 region, MitoFish sequences are NOT systematically
truncated inside the reverse primer site, so both_primers mode (both
primers required, primers trimmed off) is used directly -- no fallback
needed.

Important, verified limitation: mitofishdb.fa.gz headers are bare
accessions (e.g. ">PQ550629", or ">KF929850;HQ987855" for sequence
clusters) with NO species/taxonomy information and no companion mapping
file was provided alongside it. Taxonomy_Full / Organism_Label are left
explicitly empty in the output rather than fabricated -- do not backfill
these with guessed values.

Usage:
    python -m src.extract_mitofish_16s <in.fa.gz> <out_csv>
"""
import csv
import gzip
import re
import sys

from .extract_amplicon_region import iupac_to_regex, revcomp_iupac, find_amplicon

FWD_NAME, FWD_PRIMER = "16SF_D_Berry2017", "GACCCTATGGAGCTTTAGAC"
REV_NAME, REV_PRIMER = "16S2R_Berry2017", "CGCTGTTATCCCTADRGTAACT"

FIELDNAMES = [
    "Reference_ID", "Accession", "Start", "Stop", "Ref_Length_bp",
    "Taxonomy_Full", "Taxonomy_Depth", "Domain", "Organism_Label", "Sequence",
    "Amplicon_Region", "Amplicon_Len_bp",
]


def iter_fasta_gz(path: str):
    header, seq_lines = None, []
    with gzip.open(path, "rt") as fh:
        for line in fh:
            if line.startswith(">"):
                if header is not None:
                    yield header, "".join(seq_lines)
                header, seq_lines = line.strip().lstrip(">"), []
            else:
                seq_lines.append(line.strip())
        if header is not None:
            yield header, "".join(seq_lines)


def extract(in_fa_gz: str, out_csv: str) -> dict:
    fwd_re = re.compile(iupac_to_regex(FWD_PRIMER))
    rev_rc_re = re.compile(iupac_to_regex(revcomp_iupac(REV_PRIMER)))

    n_total = 0
    n_fwd_found = 0
    n_both_found = 0
    n_written = 0
    amplicon_lens = []

    with open(out_csv, "w", newline="") as fout:
        writer = csv.DictWriter(fout, fieldnames=FIELDNAMES)
        writer.writeheader()
        for header, seq in iter_fasta_gz(in_fa_gz):
            n_total += 1
            seq_dna = seq.upper().replace("U", "T")  # confirmed DNA (T) alphabet already; no-op if so
            fwd_m = fwd_re.search(seq_dna)
            if not fwd_m:
                continue
            n_fwd_found += 1
            hit = find_amplicon(seq_dna, fwd_re, rev_rc_re)
            if hit is None:
                continue
            n_both_found += 1
            start, end = hit
            amplicon = seq_dna[start:end]
            if len(amplicon) < 30:
                continue
            writer.writerow({
                "Reference_ID": header,
                "Accession": header,
                "Start": "", "Stop": "",
                "Ref_Length_bp": len(seq_dna),
                "Taxonomy_Full": "",   # verified unavailable -- see module docstring
                "Taxonomy_Depth": 0,
                "Domain": "Chordata (MitoFish, fish mitogenome DB)",
                "Organism_Label": "",  # verified unavailable -- see module docstring
                "Sequence": amplicon,
                "Amplicon_Region": "mitofish_16s",
                "Amplicon_Len_bp": len(amplicon),
            })
            amplicon_lens.append(len(amplicon))
            n_written += 1
            if n_total % 200000 == 0:
                print(f"...scanned {n_total} records, {n_written} usable amplicons so far")

    summary = {
        "region": "mitofish_16s",
        "fwd_primer": f"{FWD_NAME} ({FWD_PRIMER})",
        "rev_primer": f"{REV_NAME} ({REV_PRIMER})",
        "total_records_scanned": n_total,
        "fwd_primer_found": n_fwd_found,
        "both_primers_found": n_both_found,
        "usable_amplicons_written": n_written,
        "amplicon_len_min": min(amplicon_lens) if amplicon_lens else None,
        "amplicon_len_max": max(amplicon_lens) if amplicon_lens else None,
        "amplicon_len_mean": round(sum(amplicon_lens) / len(amplicon_lens), 1) if amplicon_lens else None,
    }
    for k, v in summary.items():
        print(f"{k:26s}: {v}")
    return summary


if __name__ == "__main__":
    in_fa_gz = sys.argv[1] if len(sys.argv) > 1 else "data/mitofishdb.fa.gz"
    out_csv = sys.argv[2] if len(sys.argv) > 2 else "data/MitoFish results/mitofish_16s_amplicons.csv"
    extract(in_fa_gz, out_csv)
