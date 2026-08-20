"""
Threshold calibration for MitoFish, adapted from synthetic_read_calibration.py.

Why this is a separate script instead of reusing synthetic_read_calibration.py
directly: that script's accuracy metric compares TAXONOMY strings (via
taxonomy_at_depth on the Taxonomy_Full column). mitofish_16s_reference_table.csv
has Taxonomy_Full = "" for every record -- verified in
src/extract_mitofish_16s.py's docstring: mitofishdb.fa.gz's headers are bare
accessions with no species mapping shipped alongside it. Running the taxonomy
based calibration unmodified would compare "" == "" for every record and
report a meaningless 100% accuracy.

First attempt (kept here as a documented dead end, not silently dropped):
calibrating on exact Reference_ID recovery gave 0% accuracy at EVERY
threshold tested. Diagnosing why (see Explanation.md Part 3) showed this
was the wrong ground truth, not a real matching failure: querying a
held-out record's FULL (unfragmented) sequence back against the training
pool recovered a sequence that is STRING-IDENTICAL to it in 207/300 cases
(69%) at similarity 1.0 -- MitoFish contains many different accessions
that share an exactly identical 16S amplicon sequence (repeat GenBank
submissions of the same/related fish), so the "correct" nearest neighbor
is very often a different accession than the literal held-out one, even
when the match is biologically perfect.

This calibrates on SEQUENCE-STRING recovery instead: cut a random
118-140bp fragment (matching the real read length distribution) from a
held-out reference, and check whether its nearest neighbor's SEQUENCE
(not accession) is identical to the fragment's true source sequence. Two
different accessions with identical sequences both count as a correct
recovery. This can still undercount true species-level accuracy (a
different-but-same-species sequence with 1-2bp of natural variation would
count as "wrong" here), so it remains a conservative lower bound, not an
inflated one.

Usage:
    python -m src.mitofish_calibration
"""
import csv
import random
import sys
import numpy as np
import faiss

from . import config
from .embed_common import embed_batch
from .faiss_index import l2_normalize
from .evaluate import sweep_thresholds, print_sweep

REAL_READ_LENGTHS = list(range(118, 141))  # measured from ERR16922870_1_first5000.fastq


def run(holdout_frac: float = 0.2, seed: int = 42):
    rng = random.Random(seed)
    np_rng = np.random.RandomState(seed)

    ids, seqs = [], []
    with open(config.REFERENCE_TABLE_CSV) as f:
        for row in csv.DictReader(f):
            ids.append(row["Reference_ID"])
            seqs.append(row["Sequence"])

    all_emb = np.load(config.REFERENCE_EMBEDDINGS_NPY).astype("float32")
    with open(config.REFERENCE_IDS_CSV) as f:
        emb_ids = [row["Reference_ID"] for row in csv.DictReader(f)]
    assert emb_ids == ids, "ref_ids.csv order does not match REFERENCE_TABLE_CSV order"
    assert all_emb.shape[0] == len(ids)

    n = len(ids)
    perm = np_rng.permutation(n)
    n_holdout = int(n * holdout_frac)
    holdout_idx = perm[:n_holdout]
    train_idx = perm[n_holdout:]
    train_ids = [ids[i] for i in train_idx]

    train_emb = l2_normalize(all_emb[train_idx])
    train_seqs = [seqs[i] for i in train_idx]
    index = faiss.IndexFlatIP(train_emb.shape[1])
    index.add(train_emb)

    synth_seqs, synth_true_seq = [], []
    for i in holdout_idx:
        seq = seqs[i]
        L = rng.choice(REAL_READ_LENGTHS)
        if len(seq) <= L:
            continue
        start = rng.randint(0, len(seq) - L)
        synth_seqs.append(seq[start:start + L])
        synth_true_seq.append(seq)  # full source sequence, for identity comparison

    synth_emb = []
    for i in range(0, len(synth_seqs), config.BATCH_SIZE):
        synth_emb.append(embed_batch(synth_seqs[i:i + config.BATCH_SIZE]))
        print(f"synthetic-read embedded {i + len(synth_emb[-1])}/{len(synth_seqs)}")
    synth_emb = l2_normalize(np.vstack(synth_emb).astype("float32"))

    sims, pos = index.search(synth_emb, 1)
    sims = sims[:, 0]
    pos = pos[:, 0]

    # Ground truth is SEQUENCE identity (see module docstring), not accession
    # identity -- collapse to small integer "same-sequence" IDs so
    # evaluate.py's label-based metrics stay cheap (thousands of distinct
    # sequences, not accessions, as the label space).
    seq_to_label = {}
    def label_of(s):
        return seq_to_label.setdefault(s, len(seq_to_label))
    y_true = [label_of(s) for s in synth_true_seq]
    y_pred = [label_of(train_seqs[p]) for p in pos]

    print("Synthetic-read similarity distribution:",
          f"min={sims.min():.4f} p25={np.percentile(sims,25):.4f}",
          f"median={np.median(sims):.4f} p75={np.percentile(sims,75):.4f} max={sims.max():.4f}")

    thresholds = [round(t, 2) for t in np.arange(0.10, 0.95, 0.03)]
    rows = sweep_thresholds(y_true, y_pred, list(sims), thresholds)
    print(f"=== MitoFish synthetic short-read threshold sweep (sequence-identity recovery), "
          f"n_train_refs={len(train_idx)}, n_synthetic_reads={len(synth_seqs)} ===")
    print_sweep(rows)
    return rows, y_true, y_pred, sims


if __name__ == "__main__":
    run()
