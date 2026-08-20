"""
Stage 2 -- Parse the SILVA FASTA into a structured reference table (CSV).

Usage:
    python -m src.build_reference_table [in.fasta] [out.csv]

Verify before proceeding to Stage 3: if malformed_headers is more than a
tiny fraction of total_records in the printed summary, the header grammar in
fasta_io.HEADER_RE has exceptions not covered yet -- widen it rather than
silently dropping records.
"""
import csv
import sys

from . import config
from .fasta_io import iter_fasta, parse_header

FIELDNAMES = [
    "Reference_ID", "Accession", "Start", "Stop", "Ref_Length_bp",
    "Taxonomy_Full", "Taxonomy_Depth", "Domain", "Organism_Label", "Sequence",
]


def build_reference_table(fasta_path: str, out_csv: str) -> dict:
    n_total, n_parsed, n_malformed = 0, 0, 0
    with open(out_csv, "w", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=FIELDNAMES)
        writer.writeheader()
        for header, seq in iter_fasta(fasta_path):
            n_total += 1
            rec = parse_header(header)
            if rec is None:
                n_malformed += 1
                continue
            rec["Sequence"] = seq
            writer.writerow(rec)
            n_parsed += 1

    summary = {"total_records": n_total, "parsed": n_parsed, "malformed": n_malformed}
    print(f"Total FASTA records seen: {n_total}")
    print(f"Successfully parsed:      {n_parsed}")
    print(f"Malformed headers:        {n_malformed}")
    return summary


if __name__ == "__main__":
    fasta_path = sys.argv[1] if len(sys.argv) > 1 else config.SILVA_FASTA_PATH
    out_csv = sys.argv[2] if len(sys.argv) > 2 else config.REFERENCE_TABLE_CSV
    build_reference_table(fasta_path, out_csv)
