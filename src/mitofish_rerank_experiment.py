"""
Runs the DNABERT top-K + nucleotide-alignment rerank experiment on the
real 5,000 experimental reads against the (unchanged) MitoFish 16S
reference table -- see data/MitoFish results/run_report.md for the
baseline this compares against.

Does NOT touch the reference table, primers, or extracted region --
those already passed verification in the prior run. This only adds a
second, independent nucleotide-level check on top of the existing
DNABERT/FAISS candidates.

Usage:
    python -m src.mitofish_rerank_experiment
"""
import csv
import sys
import numpy as np

from .nucleotide_rerank import rerank_candidates
from .embed_experimental import read_fastq


def load_ref_table(path: str):
    ids, seqs = [], []
    with open(path) as f:
        for row in csv.DictReader(f):
            ids.append(row["Reference_ID"])
            seqs.append(row["Sequence"])
    return ids, seqs


def run(
    fastq_path: str = "data/ERR16922870_1_first5000.fastq",
    ref_table_csv: str = "data/MitoFish results/mitofish_16s_reference_table.csv",
    top_sims_npy: str = "data/MitoFish results/reads_top10_sims.npy",
    top_pos_npy: str = "data/MitoFish results/reads_top10_pos.npy",
    top_ids_csv: str = "data/MitoFish results/reads_top10_ids.csv",
    out_csv: str = "data/MitoFish results/rerank_candidates.csv",
):
    ref_ids, ref_seqs = load_ref_table(ref_table_csv)

    with open(top_ids_csv) as f:
        query_read_ids = [line.strip() for line in f if line.strip()]
    sims = np.load(top_sims_npy)
    pos = np.load(top_pos_npy)
    assert sims.shape[0] == len(query_read_ids)

    read_seqs = {rid: seq for rid, seq in read_fastq(fastq_path)}

    fieldnames = [
        "Read_ID", "chosen_rank_by_identity", "Reference_ID", "dnabert_rank",
        "dnabert_similarity", "identity_pct", "coverage_pct", "aligned_len", "score",
    ]
    with open(out_csv, "w", newline="") as fout:
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()
        for i, read_id in enumerate(query_read_ids):
            read_seq = read_seqs.get(read_id)
            if read_seq is None:
                continue
            candidates = [
                (ref_ids[pos[i, k]], ref_seqs[pos[i, k]], float(sims[i, k]))
                for k in range(pos.shape[1])
            ]
            reranked = rerank_candidates(read_seq, candidates)
            for chosen_rank, r in enumerate(reranked, start=1):
                writer.writerow({
                    "Read_ID": read_id,
                    "chosen_rank_by_identity": chosen_rank,
                    **r,
                })
            if (i + 1) % 500 == 0:
                print(f"reranked {i + 1}/{len(query_read_ids)} reads")

    print(f"wrote {out_csv}")


if __name__ == "__main__":
    run()
