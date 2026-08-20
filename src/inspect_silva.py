"""
Stage 1 -- Inspect the full local SILVA FASTA file and print real numbers.

Run this BEFORE anything else. It answers, with real data from your file:
    - total record count
    - malformed-header count
    - sequence length distribution (min/max/mean/median/percentiles)
    - taxonomy-depth distribution (min/max)
    - duplicate sequence count
    - example headers with unusually shallow taxonomy

Usage:
    python -m src.inspect_silva [path/to/SILVA_138.2_SSURef_tax_silva_trunc.fasta]
"""
import sys
import statistics
import hashlib
from collections import Counter

from . import config
from .fasta_io import iter_fasta, parse_header


def inspect(path: str) -> dict:
    n_total = 0
    n_malformed = 0
    lengths = []
    depths = []
    seq_hashes = Counter()
    shallow_examples = []

    for header, seq in iter_fasta(path):
        n_total += 1
        rec = parse_header(header)
        if rec is None:
            n_malformed += 1
            continue
        lengths.append(len(seq))
        depths.append(rec["Taxonomy_Depth"])
        if rec["Taxonomy_Depth"] <= 3 and len(shallow_examples) < 5:
            shallow_examples.append(header)
        seq_hashes[hashlib.md5(seq.encode()).hexdigest()] += 1

    dup_count = sum(c - 1 for c in seq_hashes.values() if c > 1)
    lengths_sorted = sorted(lengths)

    def pct(p):
        if not lengths_sorted:
            return None
        idx = min(len(lengths_sorted) - 1, int(len(lengths_sorted) * p))
        return lengths_sorted[idx]

    report = {
        "total_records": n_total,
        "malformed_headers": n_malformed,
        "min_length": min(lengths) if lengths else None,
        "max_length": max(lengths) if lengths else None,
        "mean_length": round(statistics.mean(lengths), 2) if lengths else None,
        "median_length": statistics.median(lengths) if lengths else None,
        "p5_length": pct(0.05),
        "p25_length": pct(0.25),
        "p75_length": pct(0.75),
        "p95_length": pct(0.95),
        "min_taxonomy_depth": min(depths) if depths else None,
        "max_taxonomy_depth": max(depths) if depths else None,
        "duplicate_sequences": dup_count,
        "shallow_taxonomy_records": sum(1 for d in depths if d <= 3),
        "shallow_examples": shallow_examples,
    }
    return report


def print_report(report: dict) -> None:
    print("=== SILVA FASTA Inspection Report ===")
    for key in [
        "total_records", "malformed_headers", "min_length", "max_length",
        "mean_length", "median_length", "p5_length", "p25_length",
        "p75_length", "p95_length", "min_taxonomy_depth", "max_taxonomy_depth",
        "duplicate_sequences", "shallow_taxonomy_records",
    ]:
        print(f"{key:26s}: {report[key]}")
    if report["shallow_examples"]:
        print("Example shallow-taxonomy headers:")
        for h in report["shallow_examples"]:
            print("  ", h)


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else config.SILVA_FASTA_PATH
    print_report(inspect(path))
