"""
Stage 12 driver, take 2 -- threshold calibration in the CORRECT length regime.

holdout_calibration.py compares full-length held-out references against a
full-length reference index. Running it and then comparing against the
real query_results.csv similarity distribution exposed the region-mismatch
problem the PDF explicitly warns about (README.md "Region mismatch"): real
118-140bp reads land at similarity 0.25-0.60 against full-length references,
while full-length-vs-full-length pairs only separate classes above ~0.86.
A threshold picked from the wrong regime would misclassify essentially every
real read as unclassified, not because they don't match anything, but
because mean-pooled short-fragment vectors and mean-pooled full-length
vectors are not on the same similarity scale.

This script recalibrates in the length regime that actually matters: it
holds out a fraction of the SILVA reference sample, cuts a random
118-140bp substring from each held-out full-length sequence (matching the
REAL read length distribution measured from ERR16922870_1_first5000.fastq),
embeds those short synthetic "reads" with the same DNABERT-S pipeline, and
queries them against a FAISS index built from the REMAINING full-length
references.

Ground truth for each synthetic read is exact (it's a literal substring of
its parent reference, so its true taxonomy is known) -- but the query
position within the gene is picked uniformly at random, which is NOT
verified to match the real primers/hypervariable region that produced
ERR16922870's actual reads. This calibrates "how well does a random short
fragment of known taxonomy get matched", not "how well does the specific
amplified region get matched" -- report it with that caveat, per the
project's own region-mismatch limitation note.

Usage:
    python -m src.synthetic_read_calibration
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
from .holdout_calibration import taxonomy_at_depth

REAL_READ_LENGTHS = list(range(118, 141))  # measured from ERR16922870_1_first5000.fastq


def run(depth: int = 3, holdout_frac: float = 0.2, seed: int = 42):
    rng = random.Random(seed)
    np_rng = np.random.RandomState(seed)

    ids, seqs, tax_full = [], [], []
    with open(config.REFERENCE_TABLE_CSV) as f:
        for row in csv.DictReader(f):
            ids.append(row["Reference_ID"])
            seqs.append(row["Sequence"])
            tax_full.append(row["Taxonomy_Full"])

    # Reuse the already-computed Stage 5 embeddings (data/silva_ref_embeddings.npy)
    # instead of re-embedding 4000 full-length references, which would take ~30min.
    # silva_ref_ids.csv is row-index-aligned to REFERENCE_TABLE_CSV by construction
    # (embed_reference_table iterates the CSV in file order) -- assert that here.
    all_emb = np.load(config.REFERENCE_EMBEDDINGS_NPY).astype("float32")
    with open(config.REFERENCE_IDS_CSV) as f:
        emb_ids = [row["Reference_ID"] for row in csv.DictReader(f)]
    assert emb_ids == ids, "silva_ref_ids.csv order does not match REFERENCE_TABLE_CSV order"
    assert all_emb.shape[0] == len(ids)

    n = len(ids)
    perm = np_rng.permutation(n)
    n_holdout = int(n * holdout_frac)
    holdout_idx = perm[:n_holdout]
    train_idx = perm[n_holdout:]

    train_emb = l2_normalize(all_emb[train_idx])
    index = faiss.IndexFlatIP(train_emb.shape[1])
    index.add(train_emb)

    # Cut a random real-length substring from each held-out reference.
    synth_seqs, synth_true_tax = [], []
    for i in holdout_idx:
        seq = seqs[i]
        L = rng.choice(REAL_READ_LENGTHS)
        if len(seq) <= L:
            continue
        start = rng.randint(0, len(seq) - L)
        synth_seqs.append(seq[start:start + L])
        synth_true_tax.append(taxonomy_at_depth(tax_full[i], depth))

    synth_emb = []
    for i in range(0, len(synth_seqs), config.BATCH_SIZE):
        synth_emb.append(embed_batch(synth_seqs[i:i + config.BATCH_SIZE]))
        print(f"synthetic-read embedded {i + len(synth_emb[-1])}/{len(synth_seqs)}")
    synth_emb = l2_normalize(np.vstack(synth_emb).astype("float32"))

    sims, pos = index.search(synth_emb, 1)
    sims = sims[:, 0]
    pos = pos[:, 0]

    y_true = synth_true_tax
    y_pred = [taxonomy_at_depth(tax_full[train_idx[p]], depth) for p in pos]

    print("Synthetic-read similarity distribution:",
          f"min={sims.min():.4f} p25={np.percentile(sims,25):.4f}",
          f"median={np.median(sims):.4f} p75={np.percentile(sims,75):.4f} max={sims.max():.4f}")

    thresholds = [round(t, 2) for t in np.arange(0.10, 0.70, 0.02)]
    rows = sweep_thresholds(y_true, y_pred, list(sims), thresholds)
    print(f"=== Synthetic short-read threshold sweep (n_train_refs={len(train_idx)}, "
          f"n_synthetic_reads={len(synth_seqs)}, taxonomy_depth={depth}) ===")
    print_sweep(rows)
    return rows, y_true, y_pred, sims


if __name__ == "__main__":
    depth = int(sys.argv[1]) if len(sys.argv) > 1 else config.CLUSTER_LABEL_TAXONOMY_DEPTH
    run(depth=depth)
