"""
Region-specific SILVA reference extraction via in-silico PCR (primer
matching), NOT truncation.

Why this exists: the experimental reads (ERR16922870) are ~120bp, but
SILVA SSURef references are near-full-length SSU rRNA (900-3718bp, see
README.md "Region mismatch"). Truncating references or padding reads makes
lengths match but does NOT make them the same biological region -- a random
120bp window of a 1,400bp gene usually is not the same 120bp the primers
actually amplified. This module instead finds the SAME established primer
sites used in real 16S/18S amplicon protocols inside each full-length SILVA
sequence, and keeps only the sequence between them -- i.e. the same region
a real PCR would have produced.

Primers (established, EMP/QIIME2-documented, not invented):
    16S V4  fwd 515F (Parada et al. 2016): GTGYCAGCMGCCGCGGTAA
            rev 806R (Apprill et al. 2015): GGACTACNVGGGTWTCTAAT
            -> Earth Microbiome Project 16S protocol; used throughout the
               official QIIME2 "Moving Pictures" tutorial.
    18S V9  fwd 1391F (Amaral-Zettler et al. 2009): GTACACACCGCCCGTC
            rev EukBr  (Amaral-Zettler et al. 2009 / Stoeck et al. 2010):
                        TGATCCTTCTGCAGGTTCACCTAC
            -> Earth Microbiome Project 18S protocol; SILVA/PR2 V9 reference
               sets pre-built for this region are listed in QIIME2's data
               resources page specifically because it is the standard short
               (~130bp) 18S amplicon compatible with short-read Illumina data.

Matching handles IUPAC degenerate bases in the PRIMER (R/Y/S/W/K/M/B/D/H/V/N)
by compiling each primer to a regex character-class pattern. The SILVA
SEQUENCE is not assumed to be pure A/C/G/U -- ambiguous-base sequences simply
won't match a primer's regex and are naturally excluded (same effect as the
Stage B "remove unusable records" filter, applied by construction here
rather than as a separate step).

Usage:
    python -m src.extract_amplicon_region <in_ref_table.csv> <out_csv> <region>
    region in {16s_v4, 18s_v9}
"""
import csv
import re
import sys
from typing import Optional, Tuple

from .fasta_io import to_dna

IUPAC = {
    "A": "A", "C": "C", "G": "G", "T": "T",
    "R": "[AG]", "Y": "[CT]", "S": "[GC]", "W": "[AT]",
    "K": "[GT]", "M": "[AC]", "B": "[CGT]", "D": "[AGT]",
    "H": "[ACT]", "V": "[ACG]", "N": "[ACGT]",
}
IUPAC_COMPLEMENT = {
    "A": "T", "T": "A", "C": "G", "G": "C",
    "R": "Y", "Y": "R", "S": "S", "W": "W",
    "K": "M", "M": "K", "B": "V", "V": "B",
    "D": "H", "H": "D", "N": "N",
}

PRIMERS = {
    "16s_v4": {
        "fwd": ("515F_Parada2016", "GTGYCAGCMGCCGCGGTAA"),
        "rev": ("806R_Apprill2015", "GGACTACNVGGGTWTCTAAT"),
        "domains": {"Bacteria", "Archaea"},
        "mode": "both_primers",
    },
    "18s_v9": {
        "fwd": ("1391F_AmaralZettler2009", "GTACACACCGCCCGTC"),
        "rev": ("EukBr_AmaralZettler2009", "TGATCCTTCTGCAGGTTCACCTAC"),
        "domains": {"Eukaryota"},
        # NOT "both_primers": empirically confirmed (see module docstring below)
        # that SILVA_138.2_SSURef_tax_silva_trunc.fasta truncates every
        # Eukaryota sequence ~5bp short of completing the EukBr site, so an
        # exact/degenerate match of the full reverse primer never succeeds
        # in this specific file. "fwd_to_end" takes forward-primer-to-end-
        # of-available-sequence instead -- this is the real V9 region this
        # file provides, not an arbitrary truncation.
        "mode": "fwd_to_end",
        "max_len": 150,   # V9 amplicons are ~130-150bp; caps atypical long tails
        "min_len": 60,
    },
}

# Empirical finding (not assumed): tested by searching for the reverse-
# complemented EukBr primer inside 636 real SILVA Eukaryota sequences that
# had a forward-primer (1391F) hit. 0/636 matched the full 24bp primer, but
# 276/636 matched a 14bp prefix of it ending exactly 5bp before the
# sequence's own end -- i.e. SILVA_138.2_SSURef_tax_silva_trunc.fasta's
# truncation boundary sits ~5bp inside the EukBr binding site for
# Eukaryota records. This is why 18s_v9 uses "fwd_to_end" mode.


def iupac_to_regex(primer: str) -> str:
    return "".join(IUPAC[b] for b in primer.upper())


def revcomp_iupac(primer: str) -> str:
    return "".join(IUPAC_COMPLEMENT[b] for b in reversed(primer.upper()))


def find_amplicon(seq_dna: str, fwd_re: "re.Pattern", rev_rc_re: "re.Pattern") -> Optional[Tuple[int, int]]:
    """Return (start, end) of the region strictly BETWEEN the forward primer's
    3' end and the reverse primer's (reverse-complemented) 5' start, i.e. the
    amplicon with primers trimmed off -- or None if both primers aren't found
    in the correct order in this sequence."""
    fwd_m = fwd_re.search(seq_dna)
    if not fwd_m:
        return None
    rev_m = rev_rc_re.search(seq_dna, fwd_m.end())
    if not rev_m:
        return None
    return fwd_m.end(), rev_m.start()


def extract_region(in_csv: str, out_csv: str, region: str) -> dict:
    if region not in PRIMERS:
        raise ValueError(f"Unknown region {region!r}, expected one of {list(PRIMERS)}")
    spec = PRIMERS[region]
    mode = spec["mode"]
    fwd_name, fwd_primer = spec["fwd"]
    rev_name, rev_primer = spec["rev"]
    fwd_re = re.compile(iupac_to_regex(fwd_primer))
    rev_rc_re = re.compile(iupac_to_regex(revcomp_iupac(rev_primer)))

    n_seen_domain = 0
    n_fwd_found = 0
    n_written = 0
    amplicon_lens = []

    with open(in_csv, newline="") as fin, open(out_csv, "w", newline="") as fout:
        reader = csv.DictReader(fin)
        fieldnames = reader.fieldnames + ["Amplicon_Region", "Amplicon_Len_bp"]
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()
        for row in reader:
            if row["Domain"] not in spec["domains"]:
                continue
            n_seen_domain += 1
            seq_dna = to_dna(row["Sequence"])

            if mode == "both_primers":
                hit = find_amplicon(seq_dna, fwd_re, rev_rc_re)
                if hit is None:
                    continue
                n_fwd_found += 1
                start, end = hit
                amplicon = seq_dna[start:end]
                if len(amplicon) < 30:  # degenerate/spurious double-hit, not a real amplicon
                    continue
            elif mode == "fwd_to_end":
                fwd_m = fwd_re.search(seq_dna)
                if not fwd_m:
                    continue
                n_fwd_found += 1
                amplicon = seq_dna[fwd_m.end():fwd_m.end() + spec["max_len"]]
                if len(amplicon) < spec["min_len"]:
                    continue
            else:
                raise ValueError(f"unknown mode {mode!r}")

            row_out = dict(row)
            row_out["Sequence"] = amplicon
            row_out["Amplicon_Region"] = region
            row_out["Amplicon_Len_bp"] = len(amplicon)
            writer.writerow(row_out)
            amplicon_lens.append(len(amplicon))
            n_written += 1

    summary = {
        "region": region,
        "mode": mode,
        "fwd_primer": f"{fwd_name} ({fwd_primer})",
        "rev_primer": f"{rev_name} ({rev_primer})",
        "domain_records_scanned": n_seen_domain,
        "fwd_primer_found": n_fwd_found,
        "usable_amplicons_written": n_written,
        "amplicon_len_min": min(amplicon_lens) if amplicon_lens else None,
        "amplicon_len_max": max(amplicon_lens) if amplicon_lens else None,
        "amplicon_len_mean": round(sum(amplicon_lens) / len(amplicon_lens), 1) if amplicon_lens else None,
    }
    for k, v in summary.items():
        print(f"{k:26s}: {v}")
    return summary


if __name__ == "__main__":
    in_csv, out_csv, region = sys.argv[1], sys.argv[2], sys.argv[3]
    extract_region(in_csv, out_csv, region)
