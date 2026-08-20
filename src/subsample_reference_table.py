"""
POC scoping step (not part of the original stage list): the real SILVA
138.2 file has 2,224,690 near-full-length references (900-3718bp). Embedding
all of them with DNABERT-S on this CPU-only machine is not feasible in a
single session, so this script draws a reproducible uniform random subsample
of the already-built reference table for Stage 5 (embed-ref).

This does NOT filter or curate by taxonomy -- uniform random sampling keeps
the domain/phylum proportions of the full file, so the subsample's taxonomic
composition is representative rather than hand-picked. This is a documented
scope limitation of the POC, not a claim that the full SILVA database was
searched.

Usage:
    python -m src.subsample_reference_table <in_csv> <out_csv> <target_n> [seed]
"""
import csv
import random
import re
import sys

# multimolecule/dnaberts's tokenizer crashes (UNK-token vocab mismatch bug)
# on any sequence containing IUPAC ambiguity codes (N, R, Y, ...). Confirmed
# on the real subsample: 6.27% of records contain them. These are excluded
# here as "unusable records" (PDF Stage B, step 4) rather than worked around.
VALID_SEQ_RE = re.compile(r"^[ACGTUacgtu]+$")  # accepts both RNA (U, raw SILVA) and DNA (T, post amplicon-extraction) alphabets


def subsample(in_csv: str, out_csv: str, target_n: int, seed: int = 42, oversample: float = 1.15) -> dict:
    rng = random.Random(seed)

    with open(in_csv, newline="") as f:
        reader = csv.reader(f)
        next(reader)
        total = sum(1 for _ in f)

    keep_prob = min(1.0, (target_n * oversample) / total) if total else 0.0

    n_seen_ambiguous = 0
    n_written = 0
    with open(in_csv, newline="") as fin, open(out_csv, "w", newline="") as fout:
        reader = csv.reader(fin)
        writer = csv.writer(fout)
        header = next(reader)
        seq_idx = header.index("Sequence")
        writer.writerow(header)
        for row in reader:
            if rng.random() >= keep_prob:
                continue
            if not VALID_SEQ_RE.match(row[seq_idx]):
                n_seen_ambiguous += 1
                continue
            writer.writerow(row)
            n_written += 1
            if n_written >= target_n:
                break

    summary = {
        "total_records": total, "keep_prob": keep_prob,
        "skipped_ambiguous": n_seen_ambiguous, "written": n_written,
    }
    print(f"Total reference records:   {total}")
    print(f"Sampling probability:      {keep_prob:.6f}")
    print(f"Skipped (ambiguous bases): {n_seen_ambiguous}")
    print(f"Subsampled records:        {n_written}")
    return summary


if __name__ == "__main__":
    in_csv, out_csv, target_n = sys.argv[1], sys.argv[2], int(sys.argv[3])
    seed = int(sys.argv[4]) if len(sys.argv) > 4 else 42
    subsample(in_csv, out_csv, target_n, seed)
