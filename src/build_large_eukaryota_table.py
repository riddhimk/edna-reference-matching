"""
Builds a LARGER eukaryota reference table so eukaryota can support the same
held-out test sizes (1,000 / 2,000 / 5,000 fragments) as fish and bacteria.

Why this is needed. Part 9 built the eukaryota reference set from SILVA using
Tara Pacific's real 1389F/1510R primers, which matched only 865 SILVA records
(853 usable). That was the right call there -- those primers match the real
ERR10928327 reads being classified -- but it caps the held-out split at 89
test fragments, far too few to measure precision/recall/F1 with any precision
(one fragment moves the score by ~1.1pp, see Explanation.md Part 14).

Part 2's ALTERNATIVE extraction of the same gene region (18S V9) from the same
SILVA file, using the standard Earth Microbiome Project 1391F/EukBr primers,
yielded 140,968 amplicons -- 165x more. It has been sitting unused since Part 8
sampled 5,000 of it. This module draws a larger, reproducible sample from that
pool for fine-tuning and evaluation.

Two documented selection criteria, both deliberate:

  1. Minimum length 141bp. Fragments are cut at 118-140bp, and a reference
     must be strictly longer than the fragment to cut a window from it. This
     pool averages only 126.8bp (max 150bp), so without this filter ~31% of
     held-out references yield no fragment at all -- which is exactly why
     Part 8 got 505 fragments when it asked for 1,000. Filtering up front
     makes the held-out count exact and predictable instead of silently short.
     The cost, stated plainly: this samples the longer ~40% of V9 amplicons
     (55,920 of 140,968), so it is not length-representative of the full pool.

  2. No IUPAC ambiguity codes, reusing subsample_reference_table.VALID_SEQ_RE
     unchanged -- the same rule every prior stage applied, for the same reason
     (the multimolecule/dnaberts tokenizer crashes on them).

Sized at 25,000 records so that the project's standard 80/20 split
(other_kingdom_diagnostics.HOLDOUT_FRAC, unchanged) yields exactly 5,000
held-out references -- enough for 5,000 fully independent test fragments,
with no need to cut multiple correlated windows from the same reference.

This does NOT replace the Tara-primer table. Part 9's real-read results and
Part 14's eukaryota numbers both depend on that table, and it is left
untouched; this is registered as a separate kingdom key.

Usage:
    python -m src.build_large_eukaryota_table [target_n] [seed]
"""
import csv
import random
import sys

from .subsample_reference_table import VALID_SEQ_RE

csv.field_size_limit(10 ** 9)

IN_CSV = "data/SILVA results/silva_18s_v9_amplicons.csv"
OUT_CSV = "data/SILVA results/silva_18s_v9_large_reference_table.csv"
MIN_LEN = 141          # strictly greater than the longest fragment (140bp)
TARGET_N = 25_000      # 80/20 split -> exactly 5,000 held-out references
SEED = 42


def build(in_csv=IN_CSV, out_csv=OUT_CSV, target_n=TARGET_N, seed=SEED, min_len=MIN_LEN):
    rng = random.Random(seed)

    eligible = []
    n_total = n_short = n_ambiguous = 0
    with open(in_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            n_total += 1
            seq = row["Sequence"]
            if len(seq) < min_len:
                n_short += 1
                continue
            if not VALID_SEQ_RE.match(seq):
                n_ambiguous += 1
                continue
            eligible.append(row)

    print(f"scanned {n_total:,} amplicons")
    print(f"  dropped {n_short:,} shorter than {min_len}bp")
    print(f"  dropped {n_ambiguous:,} containing ambiguity codes")
    print(f"  eligible: {len(eligible):,}")

    if len(eligible) < target_n:
        raise SystemExit(f"only {len(eligible):,} eligible records, need {target_n:,}")

    sample = rng.sample(eligible, target_n)
    fieldnames = list(sample[0].keys())
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(sample)

    lens = [len(r["Sequence"]) for r in sample]
    print(f"wrote {len(sample):,} records to {out_csv}")
    print(f"  length min/mean/max: {min(lens)} / {sum(lens) / len(lens):.1f} / {max(lens)}bp")
    print(f"  at 80/20 split -> {int(target_n * 0.8):,} train / {int(target_n * 0.2):,} held out")
    return out_csv


if __name__ == "__main__":
    build(
        target_n=int(sys.argv[1]) if len(sys.argv) > 1 else TARGET_N,
        seed=int(sys.argv[2]) if len(sys.argv) > 2 else SEED,
    )
